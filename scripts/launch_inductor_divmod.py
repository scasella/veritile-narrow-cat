#!/usr/bin/env python3
"""Stage 12: does the stage-11 fast division transfer into Inductor's own kernel?

Workload: the program from PyTorch #189940 (`nested_cat_add`, bf16), run over a sequence of
changing shapes in ONE process per configuration.

Configurations. Each runs in its own subprocess with private TORCHINDUCTOR_CACHE_DIR and
TRITON_CACHE_DIR directories; the FX-graph and AOT-autograd caches are off.
  eager          the Python function
  dyn_explicit   torch.compile(dynamic=True), last dim marked dynamic.
                 This is Inductor's kernel as emitted (variant B0).
  dyn_auto       torch.compile() default: automatic dynamic, no marks
  static_cached  torch.compile(dynamic=False): one compile per distinct shape, all cached.
                 A valid deployment only when the shape set is small and known in advance.
  dyn_N          dyn_explicit, plus the guarded rewrite with variant N (int32 narrowing of the i64 size scalars)
  dyn_F          dyn_explicit, plus variant F (proved fast divmod; other types unchanged)
  dyn_NF         dyn_explicit, plus both
  standalone_B   the stage-11 kernel (`launch_cat.cand("B")`), reference for decomposition

The rewrite (scripts/inductor_divmod.py) runs inside Inductor's codegen. It wraps
`TritonScheduling.define_kernel`, so the kernel source is transformed before Inductor hashes, writes,
compiles and autotunes it. The launch signature and the wrapper do not change. Every define_kernel
event is recorded: the returned kernel name, the before/after source hashes and sources, and whether
the rewrite fired, with the reason if it did not.

Sequence: the 14 stage-11 shapes (10 SEQUENCE plus 4 HELD_OUT) in the fixed interleaved ORDER below.
All inputs are allocated before the first call.
  pass 0 (cold)     per shape: first call and synchronize (first_call_s, compile events), then 19 more calls
  passes 1..5 warm  per shape: 20 back-to-back calls, CUDA events at shape boundaries; one synchronize per
                    pass. pass_s is the wall time of the whole pass: the complete-sequence time.
  single probe      per shape: 30 x (call; synchronize). Records host return time (host_us: guards,
                    wrapper, launch; nothing is prepared on the host for N/F/NF) and wall_us.
  profile           per shape: 5 profiled calls; summed CUDA kernel time and kernel names.
Compile events: Dynamo counters before and after every call of pass 0 and every warm pass;
torch._dynamo.utils.compile_times(); recompile log records; hook events per shape.
The full configuration set is run in 2 rounds (round 2 in reverse order).

PRE-REGISTERED SELECTION CRITERION (fixed before the GPU call; do not edit after it)
  T(c)  = warm complete-sequence time: median over warm passes 1..5 of pass_s, per round.
  S(c,x) = median over warm passes of the in-sequence per-call time at shape x
           (event span of its 20 calls / 20).
  Validity (a configuration that violates any of these is excluded):
    - bitwise equal to eager at all 14 shapes;
    - no Dynamo compile or recompile in any warm pass;
    - for dyn_N, dyn_F, dyn_NF:
        - the rewrite fired on every emitted kernel whose source contains `xindex % ks`;
        - it has the same number of Dynamo graphs and Inductor kernels as dyn_explicit.
  Transfer, for V in {dyn_N, dyn_F, dyn_NF}; ALL must hold in BOTH rounds:
    (a) T(dyn_explicit) / T(V) >= 1.10
    (b) T(V) <= T(dyn_auto) and T(V) <= T(eager)
    (c) for every shape x: S(V,x) <= S(dyn_explicit,x) / 0.95
  Selected integrated version: among the passing V, the smallest mean of T over the two rounds.
  If none passes, the outcome is "did not transfer". The explanation then comes from the isolation data:
  per-shape device time of B0/N/F/NF against standalone_B and static_cached.
  Reported but NOT part of the decision:
    - static_cached's T and total compile seconds, and its break-even call count against the selected
      version;
    - cold-sequence (pass 0) time of every configuration.

    python3 scripts/launch_inductor_divmod.py --dry     # CPU-only: config and plan, no CUDA needed
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

WIDTHS = {"issue": (2048, 256, 256), "llama8b": (4096, 1024, 1024), "mid": (3072, 512, 512),
          "odd": (1000, 120, 136), "wide": (5120, 640, 640)}
# the 10 stage-11 SEQUENCE shapes and 4 HELD_OUT shapes, interleaved so consecutive shapes differ in width
ORDER = [(2048, "issue"), (3000, "odd"), (4096, "llama8b"), (2048, "mid"), (12345, "wide"),
         (16384, "issue"), (4096, "mid"), (12345, "odd"), (2048, "llama8b"), (32768, "issue"),
         (3000, "wide"), (16384, "mid"), (4096, "issue"), (16384, "llama8b")]
CONFIGS = ["eager", "dyn_explicit", "dyn_auto", "static_cached", "dyn_N", "dyn_F", "dyn_NF", "standalone_B"]
VARIANT = {"dyn_explicit": "B0", "dyn_N": "N", "dyn_F": "F", "dyn_NF": "NF"}
CALLS, WARM_PASSES, SINGLE, PROFILED, ROUNDS = 20, 5, 30, 5, 2


def nested_cat_add(q1, k1, v1a, v1b, q2, k2, v2a, v2b):
    import torch
    v1 = v1a + v1b
    v2 = v2a + v2b
    g1 = torch.cat([q1, k1, v1], dim=-1)
    g2 = torch.cat([q2, k2, v2], dim=-1)
    return torch.cat([g1, g2], dim=-1)


def make_inputs(n, wset, seed=0):
    import torch
    wq, wk, wv = WIDTHS[wset]
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(n, w, generator=g).to(torch.bfloat16).to("cuda") for w in (wq, wk, wv, wv, wq, wk, wv, wv)]


# ---------------------------------------------------------------------------------------------
# worker (one configuration, one process)
# ---------------------------------------------------------------------------------------------

def _install_hook(variant, events):
    import torch._inductor.codegen.triton as ct
    import inductor_divmod as R
    orig = ct.TritonScheduling.define_kernel
    seen = {}

    def define_kernel(self, src_code, node_schedule, kernel):
        new_src, rec = R.rewrite(src_code, variant)
        name = orig(self, new_src, node_schedule, kernel)
        rec = {**rec, "kernel_name": name, "has_divmod": "xindex % ks" in src_code}
        for key, s in (("before", src_code), ("after", new_src)):
            h = R.sha(s)
            if h not in seen:
                seen[h] = s
        events.append(rec)
        return name

    ct.TritonScheduling.define_kernel = define_kernel
    return seen


def _counters():
    import torch._dynamo as dynamo
    c = dynamo.utils.counters
    return {"frames_total": c["frames"]["total"], "frames_ok": c["frames"]["ok"],
            "unique_graphs": c["stats"]["unique_graphs"], "calls_captured": c["stats"]["calls_captured"],
            "recompile_limit_hit": sum(v for k, v in c["frames"].items() if "limit" in k)}


def worker(config: str) -> dict:
    import logging
    import torch
    import torch._dynamo as dynamo
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(dynamo.config, attr):
            setattr(dynamo.config, attr, 64)
    recompile_log = []

    class H(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "ecompil" in msg:
                recompile_log.append(msg[:600])
    torch._logging.set_logs(recompiles=True)
    logging.getLogger("torch").addHandler(H())

    events, sources = [], {}
    if config in VARIANT:
        sources = _install_hook(VARIANT[config], events)
    if config == "eager":
        fn = nested_cat_add
    elif config in VARIANT:
        fn = torch.compile(nested_cat_add, dynamic=True)
    elif config == "dyn_auto":
        fn = torch.compile(nested_cat_add)
    elif config == "static_cached":
        fn = torch.compile(nested_cat_add, dynamic=False)
    elif config == "standalone_B":
        import launch_cat
        fn = launch_cat.cand("B", launch_cat._kernels())
    else:
        raise ValueError(config)

    shapes = [(n, w) for n, w in ORDER]
    ins = {s: make_inputs(*s) for s in shapes}
    if config in VARIANT:
        for s in shapes:
            for t in ins[s]:
                dynamo.mark_dynamic(t, t.dim() - 1)
    ref = {s: nested_cat_add(*ins[s]) for s in shapes}
    torch.cuda.synchronize()
    sync = torch.cuda.synchronize
    rec = {"config": config, "variant": VARIANT.get(config), "shapes": [list(s) for s in shapes]}

    # pass 0: cold
    cold, bitwise = [], {}
    t_pass = time.perf_counter()
    for s in shapes:
        c0, e0 = _counters(), len(events)
        t0 = time.perf_counter()
        out = fn(*ins[s])
        sync()
        first = time.perf_counter() - t0
        bitwise[f"{s[0]}x{s[1]}"] = bool(out.shape == ref[s].shape and
                                         torch.equal(out.view(torch.int16), ref[s].view(torch.int16)))
        for _ in range(CALLS - 1):
            fn(*ins[s])
        sync()
        c1 = _counters()
        cold.append({"shape": list(s), "first_call_s": first,
                     "counters_delta": {k: c1[k] - c0[k] for k in c0},
                     "kernels_defined": [{k: e[k] for k in ("kernel_name", "rewritten", "has_divmod", "before",
                                                            "after", "reason") if k in e}
                                         for e in events[e0:]]})
    rec["cold_pass_s"] = time.perf_counter() - t_pass
    rec["cold"] = cold
    rec["bitwise_equal_eager"] = bitwise

    # warm passes
    warm = []
    for p in range(WARM_PASSES):
        c0, e0 = _counters(), len(events)
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(len(shapes) + 1)]
        t0 = time.perf_counter()
        evs[0].record()
        for i, s in enumerate(shapes):
            for _ in range(CALLS):
                fn(*ins[s])
            evs[i + 1].record()
        sync()
        pass_s = time.perf_counter() - t0
        c1 = _counters()
        warm.append({"pass_s": pass_s,
                     "per_call_ms": [evs[i].elapsed_time(evs[i + 1]) / CALLS for i in range(len(shapes))],
                     "counters_delta": {k: c1[k] - c0[k] for k in c0}, "kernels_defined": len(events) - e0})
    rec["warm"] = warm

    # single-call probe: host return time and wall
    single = []
    for s in shapes:
        hs, ws = [], []
        for _ in range(SINGLE):
            t0 = time.perf_counter()
            fn(*ins[s])
            t1 = time.perf_counter()
            sync()
            t2 = time.perf_counter()
            hs.append((t1 - t0) * 1e6)
            ws.append((t2 - t0) * 1e6)
        single.append({"shape": list(s), "host_us": statistics.median(hs), "wall_us": statistics.median(ws)})
    rec["single"] = single

    # profiled device time and kernel identities
    from torch.profiler import ProfilerActivity, profile
    prof_rows = []
    for s in shapes:
        per, names = [], set()
        for _ in range(PROFILED):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn(*ins[s])
                sync()
            ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
            per.append(sum(e.device_time for e in ev))
            names |= {e.name[:90] for e in ev}
        prof_rows.append({"shape": list(s), "kernel_us": statistics.median(per), "kernels": sorted(names)})
    rec["profile"] = prof_rows

    rec["counters_final"] = _counters()
    try:
        rec["compile_times"] = str(dynamo.utils.compile_times(repr="str"))[:4000]
    except Exception:  # noqa: BLE001
        rec["compile_times"] = None
    rec["recompile_log"] = recompile_log[:40]
    rec["hook_events"] = [{k: v for k, v in e.items()} for e in events]
    rec["sources"] = sources
    return rec


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------

def _sub(config: str, timeout=1800) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"vt_{config}_") as d:
        env = {**os.environ, "TORCHINDUCTOR_CACHE_DIR": f"{d}/inductor", "TRITON_CACHE_DIR": f"{d}/triton",
               "TORCHINDUCTOR_FX_GRAPH_CACHE": "0", "TORCHINDUCTOR_AUTOGRAD_CACHE": "0"}
        env.pop("TRITON_INTERPRET", None)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, __file__, "--worker", config], capture_output=True, text=True,
                           timeout=timeout, env=env)
        wall = time.perf_counter() - t0
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not lines:
        return {"config": config, "error": (r.stderr or r.stdout)[-4000:], "process_s": wall}
    out = json.loads(lines[-1][len("RESULT "):])
    out["process_s"] = wall
    return out


def T(rec):
    return statistics.median(p["pass_s"] for p in rec["warm"])


def S(rec, i):
    return statistics.median(p["per_call_ms"][i] for p in rec["warm"])


def validity(rec, base) -> dict:
    v = {"ran": "error" not in rec}
    if not v["ran"]:
        return {**v, "valid": False}
    v["bitwise_all"] = all(rec["bitwise_equal_eager"].values())
    v["no_warm_compiles"] = all(p["counters_delta"]["frames_total"] == 0 and p["counters_delta"]["unique_graphs"] == 0
                                and p["kernels_defined"] == 0 for p in rec["warm"])
    if rec.get("variant") in ("N", "F", "NF"):
        div = [e for e in rec["hook_events"] if e["has_divmod"]]
        v["rewrite_fired_on_all_divmod_kernels"] = bool(div) and all(e["rewritten"] for e in div)
        if base is not None and "error" not in base:
            v["same_compile_count_as_B0"] = (rec["counters_final"]["unique_graphs"] == base["counters_final"]["unique_graphs"]
                                             and len(rec["hook_events"]) == len(base["hook_events"]))
        else:
            v["same_compile_count_as_B0"] = False
    v["valid"] = all(x for k, x in v.items() if k != "valid")
    return v


def decide(rounds: list) -> dict:
    res = {"per_round": [], "transfer": {}}
    for rd in rounds:
        by = {r["config"]: r for r in rd}
        base = by.get("dyn_explicit")
        val = {c: validity(r, base if c != "dyn_explicit" else None) for c, r in by.items()}
        t = {c: T(r) for c, r in by.items() if val[c]["valid"]}
        row = {"validity": val, "T_s": t, "per_variant": {}}
        for V in ("dyn_N", "dyn_F", "dyn_NF"):
            if not (val.get(V, {}).get("valid") and "dyn_explicit" in t and "dyn_auto" in t and "eager" in t):
                row["per_variant"][V] = {"pass": False, "why": "invalid or missing baseline"}
                continue
            ratio = t["dyn_explicit"] / t[V]
            worst = min(S(base, i) / S(by[V], i) for i in range(len(ORDER)))
            a, b = ratio >= 1.10, t[V] <= t["dyn_auto"] and t[V] <= t["eager"]
            c = worst >= 0.95
            row["per_variant"][V] = {"speedup_vs_B0": ratio, "worst_shape_ratio": worst, "a": a, "b": b, "c": c,
                                     "pass": a and b and c}
        res["per_round"].append(row)
    passing = [V for V in ("dyn_N", "dyn_F", "dyn_NF")
               if all(r["per_variant"][V]["pass"] for r in res["per_round"])]
    res["passing"] = passing
    if passing:
        res["selected"] = min(passing, key=lambda V: statistics.mean(r["T_s"][V] for r in res["per_round"]))
        res["outcome"] = "transferred"
    else:
        res["selected"] = None
        res["outcome"] = "did not transfer"
    return res


def env():
    import torch
    import triton
    return {"device": "cuda", "gpu_name": torch.cuda.get_device_name(), "torch": torch.__version__,
            "triton": triton.__version__, "compute_capability": list(torch.cuda.get_device_capability()),
            "cpu": subprocess.run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo; nproc"],
                                  capture_output=True, text=True).stdout.strip(),
            "TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET")}


def bench() -> dict:
    rec = {"kind": "stage 12: Inductor-integrated fast divmod (pre-registered criterion in module docstring)",
           **env(), "order": ORDER, "widths": WIDTHS, "configs": CONFIGS,
           "calls": CALLS, "warm_passes": WARM_PASSES, "rounds": []}
    for r in range(ROUNDS):
        order = CONFIGS if r % 2 == 0 else CONFIGS[::-1]
        rec["rounds"].append([_sub(c) for c in order])
    rec["decision"] = decide(rec["rounds"])
    import launch_local_check as LC
    rec["input_hashes"] = LC.input_hashes()
    return rec


def dry() -> dict:
    import inductor_divmod as R
    fx = (REPO / "bench/optimizations/inductor_divmod/fixtures/pointwise_cat_preview.py").read_text()
    return {"configs": CONFIGS, "order": ORDER, "rounds": ROUNDS,
            "preview_rewrites": {v: R.rewrite(fx, v)[1]["rewritten"] for v in R.VARIANTS},
            "numel_max": max(n * 2 * sum(WIDTHS[w]) for n, w in ORDER)}


if __name__ == "__main__":
    if "--worker" in sys.argv:
        c = sys.argv[sys.argv.index("--worker") + 1]
        try:
            print("RESULT " + json.dumps(worker(c), default=str))
        except Exception:  # noqa: BLE001
            print("RESULT " + json.dumps({"config": c, "error": traceback.format_exc()[-4000:]}))
    elif "--dry" in sys.argv:
        print(json.dumps(dry(), indent=1))
    else:
        print(json.dumps(bench(), indent=1, default=str))
