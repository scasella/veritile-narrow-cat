#!/usr/bin/env python3
"""Stage 15: time the UPSTREAM PATCH itself, flag off versus on, on the pinned revision. NOT YET RUN.

Why. Stage 14 measured N1 on torch 2.14 with a local prototype of the rule, and validated the upstream patch on pinned
main 6aa9e2fc for correctness only. This asks the question that remains: does the patch, as proposed, keep its
benefit on the revision under review, without unwanted compilation or execution costs?

Stack: torch 2.15.0.dev20260926 (cu130, git 6aa9e2fc = upstream/MANIFEST.json), its pytorch-triton, one Modal L4
(the stage-12..14 device type). The pin and the base-file SHA-256s are verified and the patch is applied with
`patch -p1 --forward` before anything runs; otherwise the run is invalid.

Variants, both compiled in the SAME patched process as distinct code objects (Dynamo caches per code object):
  OFF  torch.compile(fn, dynamic=True)                                              the emitted kernel, ks*: i64
  ON   torch.compile(fn, dynamic=True, options={"triton.narrow_proven_size_args": True})     the patch, ks*: i32

Part A, execution time. The stage-14 block design, with pairs instead of triples:
  block(v)  warm: SEQ passes until >= 2.0 s of sustained load; then one SEQ pass (20 calls/shape, CUDA events,
            T = wall time); then 50 back-to-back calls at each SMALL shape (order rotated per block),
            S(s) = us/call; NVML median SM clock and power
  pair      one OFF block and one ON block; 2 processes x 12 pairs, each process 6 OFF-first and 6 ON-first,
            shuffled with a fixed seed. Exactly this; no early stop, no extension, no retry.
Part B, compile cost. 6 fresh processes, alternating OFF/ON (OFF, ON, ON, OFF, OFF, ON), each with empty private
  Inductor/Triton caches and FX/AOT caches off. Each compiles the target program and 4 unrelated programs (3-segment
  cat, add+ReLU, row sum, softmax) and records: wall time of each first call; the time spent inside
  `proven_int32_size_args` per kernel and its verdict; Dynamo counters; guard text (hashed); and the ks signature of
  every kernel.
Artifacts, both parts: the autotuned config, registers and spills of every timed launcher (as stage 13), and the
  SHA-256 of every *.cubin in the process's private TRITON_CACHE_DIR, with the ks types found in its .ttir.

PRE-REGISTERED ANALYSIS (fixed before the GPU call; do not edit after it)
  Paired per pair p:  rT(p) = T_OFF(p) / T_ON(p);  rS(s,p) = S_OFF(s,p) / S_ON(s,p).
  Statistic: median over the 24 pairs; 95% CI: percentile bootstrap over pairs (10,000 resamples, seed 0).
  "Benefit retained"        iff median rT >= 1.03 and CI lower bound >= 1.00
  "No benefit shown"        iff CI upper bound < 1.03
  "Unresolved"              otherwise
  Each SMALL shape: "non-regression shown" iff CI lower >= 0.97; "regression shown" iff CI upper < 0.97;
  otherwise "unresolved".
  Compile cost is REPORTED, with no threshold: the median time in the eligibility check per kernel (target;
  unrelated), and the median first-call wall time OFF versus ON per program (6 processes; descriptive only).
  Validity (all required, else nothing is concluded):
    - pin and base files match; the patch applied;
    - bitwise equal to eager for OFF and ON at every SEQ and SMALL shape;
    - every OFF kernel has ks*: i64, and the ON target kernel has all ks*: i32;
    - in Part B, every unrelated kernel is rejected by the check and has the same ks types OFF and ON;
    - Dynamo unique graphs are identical OFF and ON for every program;
    - no Dynamo or Inductor compile during any measured block.
  Reported, expected equal, NOT a validity condition (a difference is a finding about the patch): normalized guard
    text OFF versus ON (object ids, source-location comments and variant names masked; the text is saved).
  Descriptive comparison, declared now: the median OFF/ON sequence ratio is reported next to stage 14's
    B0/N1 = 1.349 (torch 2.14). Absolute times are not compared across the two stacks.
  Compile cost: each compile process first compiles an untimed trivial program, so first-call times exclude
    one-time initialisation.
  Artifacts are descriptive: cubins are attributed to OFF or ON by the Triton cache entries each variant's compile
    created; `ttir_args` keeps whatever argument names TTIR prints (they may be positional, e.g. %arg3).

    python3 scripts/launch_patch_perf.py --dry        # print the design; no torch needed
    python3 scripts/launch_patch_perf.py --cpu-dry    # in the local veritile-nightly image: codegen-only checks
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UP = REPO / "bench/optimizations/inductor_narrow/upstream"
sys.path.insert(0, str(REPO / "scripts"))

import launch_narrow_confirm as C14  # noqa: E402  (SEQ, SMALL, SRC_FN, make_inputs; unchanged)

SEQ, SMALL, S12 = C14.SEQ, C14.SMALL, C14.S12
VARIANTS = ["OFF", "ON"]
PAIRS_PER_PROCESS, PROCESSES = 12, 2
SEQ_CALLS, SMALL_CALLS, WARM_S = 20, 50, 2.0
COMPILE_ORDER = ["OFF", "ON", "ON", "OFF", "OFF", "ON"]
BOOT, SEED = 10000, 0
FLAG = "triton.narrow_proven_size_args"

OTHER_SRC = {
    "cat3": "def {n}(q, k, va, vb):\n    return torch.cat([q, k, va + vb], dim=-1)\n",
    "add_relu": "def {n}(a, b):\n    return torch.relu(a + b)\n",
    "row_sum": "def {n}(a):\n    return a.float().sum(-1)\n",
    "softmax": "def {n}(a):\n    return torch.softmax(a.float(), -1)\n",
}


def make(src: str, name: str):
    import torch
    ns = {"torch": torch}
    exec(src.format(n=name, name=name), ns)
    return ns[name]


def other_inputs(prog: str, dev: str):
    import torch
    r = lambda n, w: torch.randn(n, w, device=dev, dtype=torch.bfloat16)  # noqa: E731
    return {"cat3": lambda: [r(512, w) for w in (2048, 256, 256, 256)],
            "add_relu": lambda: [r(512, 3000), r(512, 3000)],
            "row_sum": lambda: [r(512, 3000)],
            "softmax": lambda: [r(512, 3000)]}[prog]()


def compile_variant(fn, v: str):
    import torch
    return torch.compile(fn, dynamic=True, options={FLAG: v == "ON"})


# ------------------------------------------------------------------------------------------ instrumentation

class Probe:
    """Records every kernel Inductor defines (ks types) and every call of the patch's eligibility check."""

    def __init__(self):
        import torch._inductor.codegen.triton as ct
        import torch._inductor.codegen.triton_size_arg_narrowing as nar
        self.tag, self.kernels, self.checks, self.guards = None, [], [], []
        orig_def, orig_chk = ct.TritonScheduling.define_kernel, nar.proven_int32_size_args
        probe = self

        def define_kernel(sched, src, node_schedule, kernel):
            name = orig_def(sched, src, node_schedule, kernel)
            probe.kernels.append({"tag": probe.tag, "kernel": name,
                                  "ks": dict(re.findall(r"'(ks\d+)': '(i\d+)'", src))})
            return name

        def check(*a, **k):
            t0 = time.perf_counter()
            out = orig_chk(*a, **k)
            probe.checks.append({"tag": probe.tag, "us": (time.perf_counter() - t0) * 1e6,
                                 "accepted": sorted(out) if out else []})
            return out
        ct.TritonScheduling.define_kernel = define_kernel
        nar.proven_int32_size_args = check   # the hook imports it from the module at call time

        class H(logging.Handler):
            def emit(self, r):
                probe.guards.append((probe.tag, r.getMessage()))
        import torch._logging
        torch._logging.set_logs(guards=True)
        for name in ("torch._dynamo.guards.__guards", "torch.__guards"):
            lg = logging.getLogger(name)
            lg.addHandler(H())
            lg.propagate = False

    def guard_text(self, tag) -> str:
        # Normalized: source-location comments (they name the variant's function), object ids, and the variant
        # suffix of function names are dropped, so OFF and ON compare equal unless a guard itself differs.
        norm = lambda m: "\n".join(re.sub(r"\s+#.*$", "", ln) for ln in m.splitlines()  # noqa: E731
                                   if not ln.startswith("Guard eval latency"))
        # Object ids (hex or long decimals, e.g. ___check_obj_id / ___check_current_backend) differ between
        # processes and between compile wrappers, so they are masked too.
        return "\n".join(re.sub(r"0x[0-9a-f]+|\b\d{6,}\b|\b\w+_(?:OFF|ON)\b", "#", norm(m))
                         for t, m in self.guards if t == tag)

    def guard_digest(self, tag) -> str:
        text = self.guard_text(tag)
        return hashlib.sha256(text.encode()).hexdigest()[:16] if text else ""


def ttir_args(func: str) -> dict:
    """{arg name: type} from a `tt.func` line, splitting its argument list at top-level commas."""
    i, depth, cur, parts = func.find("(") + 1, 0, "", []
    for ch in func[i:]:
        if ch in "({<[":
            depth += 1
        elif ch in ")}>]":
            if depth == 0:
                break
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    out = {}
    for a in parts:
        m = re.match(r"\s*%(\w+)\s*:\s*([^\s{]+)", a)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def cache_cubins(d: str) -> list:
    out = []
    for cub in sorted(Path(d).rglob("*.cubin")):
        ttir = next(cub.parent.glob("*.ttir"), None)
        func = next((ln for ln in ttir.read_text().splitlines() if "tt.func" in ln), "") if ttir else ""
        out.append({"dir": cub.parent.name[:16], "name": cub.stem, "sha256": hashlib.sha256(cub.read_bytes()).hexdigest(),
                    "ttir_args": ttir_args(func) if ttir else None})
    return out


# ------------------------------------------------------------------------------------------ part A

def worker_time(proc: int) -> dict:
    import torch
    import torch._dynamo as dynamo
    from launch_inductor_narrow import Telemetry, summarize, artifacts
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(dynamo.config, attr):
            setattr(dynamo.config, attr, 64)
    probe = Probe()
    shapes = list(dict.fromkeys(SEQ + SMALL))
    ins = {s: S12.make_inputs(*s) for s in shapes}
    for s in shapes:
        for t in ins[s]:
            dynamo.mark_dynamic(t, t.dim() - 1)
    fns, bitwise, cache_new = {}, {}, {}
    tdir = Path(os.environ["TRITON_CACHE_DIR"])
    seen = set()
    for v in VARIANTS:
        probe.tag = v
        f = compile_variant(C14.make_fn(f"cat_{v}"), v)
        for s in shapes:
            out = f(*ins[s])
            bitwise[f"{v}:{s[0]}x{s[1]}"] = bool(torch.equal(out.view(torch.int16),
                                                             S12.nested_cat_add(*ins[s]).view(torch.int16)))
        fns[v] = f
        now = set(os.listdir(tdir)) if tdir.exists() else set()
        cache_new[v], seen = sorted(now - seen), now   # Triton cache entries created while compiling this variant
    torch.cuda.synchronize()
    probe.tag = "MEASURE"
    n_before = len(probe.kernels)
    tel = Telemetry()
    sync = torch.cuda.synchronize

    def seq_pass(f):
        for s in SEQ:
            for _ in range(SEQ_CALLS):
                f(*ins[s])

    def block(v, small_order):
        f = fns[v]
        seq_pass(f)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < WARM_S:
            seq_pass(f)
        sync()
        tel.samples.clear()
        tel.start("m")
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(len(SEQ) + 1)]
        t1 = time.perf_counter()
        evs[0].record()
        for i, s in enumerate(SEQ):
            for _ in range(SEQ_CALLS):
                f(*ins[s])
            evs[i + 1].record()
        sync()
        T = time.perf_counter() - t1
        small = {}
        for s in small_order:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(SMALL_CALLS):
                f(*ins[s])
            b.record()
            sync()
            small[f"{s[0]}x{s[1]}"] = a.elapsed_time(b) * 1e3 / SMALL_CALLS
        tel.stop()
        return {"variant": v, "T_s": T,
                "seq_us": [evs[i].elapsed_time(evs[i + 1]) * 1e3 / SEQ_CALLS for i in range(len(SEQ))],
                "small_us": small, "telemetry": summarize(tel.samples, "m")}

    orders = [["OFF", "ON"]] * (PAIRS_PER_PROCESS // 2) + [["ON", "OFF"]] * (PAIRS_PER_PROCESS // 2)
    random.Random(2000 + proc).shuffle(orders)
    pairs = []
    for k, order in enumerate(orders):
        small_order = SMALL[k % len(SMALL):] + SMALL[:k % len(SMALL)]
        pairs.append({"order": order, "blocks": {v: block(v, small_order) for v in order}})
    return {"proc": proc, "bitwise": bitwise, "kernels": probe.kernels[:n_before], "checks": probe.checks,
            "compiles_during_measurement": len(probe.kernels) - n_before, "pairs": pairs,
            "guards": {v: probe.guard_digest(v) for v in VARIANTS},
            "guard_text": {v: probe.guard_text(v) for v in VARIANTS}, "cache_entries_by_variant": cache_new,
            "unique_graphs": dynamo.utils.counters["stats"]["unique_graphs"],
            "launchers": artifacts(), "cubins": cache_cubins(os.environ["TRITON_CACHE_DIR"]),
            "telemetry_ok": tel.ok, "telemetry_error": tel.err, "telemetry_static": tel.static}


# ------------------------------------------------------------------------------------------ part B

def worker_compile(v: str, dev: str = "cuda") -> dict:
    import torch
    import torch._dynamo as dynamo
    probe = Probe()
    progs = {"target": (C14.SRC_FN, lambda: S12.make_inputs(2048, "issue"))}
    progs.update({k: (s, (lambda k=k: other_inputs(k, dev))) for k, s in OTHER_SRC.items()})
    rec = {"variant": v, "programs": {}}
    probe.tag = "warmup"   # untimed: absorbs CUDA context, Triton import and Inductor lazy setup
    try:
        compile_variant(make("def {n}(x):\n    return x + 1\n", f"warm_{v}"), v)(torch.ones(8, device=dev))
    except Exception:  # noqa: BLE001 - expected only in --cpu-dry
        pass
    for prog, (src, mk) in progs.items():
        probe.tag = prog
        ins = mk()
        for t in ins:
            dynamo.mark_dynamic(t, t.dim() - 1)
        g0 = dynamo.utils.counters["stats"]["unique_graphs"]
        f = compile_variant(make(src, f"{prog}_{v}"), v)
        t0 = time.perf_counter()
        err = None
        try:
            f(*ins)
            if dev == "cuda":
                torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 - expected only in --cpu-dry (no CPU Triton backend)
            err = repr(e)[:200]
        rec["programs"][prog] = {
            "first_call_s": time.perf_counter() - t0, "error": err,
            "graphs": dynamo.utils.counters["stats"]["unique_graphs"] - g0,
            "kernels": [k for k in probe.kernels if k["tag"] == prog],
            "checks": [c for c in probe.checks if c["tag"] == prog],
            "guards": probe.guard_digest(prog), "guard_text": probe.guard_text(prog)}
    rec["counters"] = {k: dict(dynamo.utils.counters[k]) for k in ("stats", "recompiles") if k in dynamo.utils.counters}
    if dev == "cuda":
        rec["cubins"] = cache_cubins(os.environ["TRITON_CACHE_DIR"])
    return rec


# ------------------------------------------------------------------------------------------ driver

def _sub(args: list, timeout=3000) -> dict:
    with tempfile.TemporaryDirectory(prefix="vt15_") as d:
        env = {**os.environ, "TORCHINDUCTOR_CACHE_DIR": f"{d}/inductor", "TRITON_CACHE_DIR": f"{d}/triton",
               "TORCHINDUCTOR_FX_GRAPH_CACHE": "0", "TORCHINDUCTOR_AUTOGRAD_CACHE": "0"}
        env.pop("TRITON_INTERPRET", None)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, __file__] + args, capture_output=True, text=True, timeout=timeout, env=env)
        wall = time.perf_counter() - t0
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not lines:
        return {"args": args, "error": (r.stderr or r.stdout)[-4000:], "process_s": wall}
    out = json.loads(lines[-1][len("RESULT "):])
    out["process_s"] = wall
    return out


def apply_patch() -> dict:
    import torch
    man = json.loads((UP / "MANIFEST.json").read_text())
    site = Path(torch.__file__).parent.parent
    rec = {"torch": torch.__version__, "git_version": torch.version.git_version,
           "pin_matches": torch.version.git_version == man["pytorch_commit"],
           "base_matches": {f: hashlib.sha256((site / f).read_bytes()).hexdigest() == h
                            for f, h in man["base_sha256"].items()}}
    if rec["pin_matches"] and all(rec["base_matches"].values()):
        p = subprocess.run(["patch", "-p1", "--forward", "--batch", "-d", str(site), "-i", str(UP / man["patch"])],
                           capture_output=True, text=True)
        rec["patch_rc"], rec["patch_out"] = p.returncode, (p.stdout + p.stderr)[-1500:]
    rec["applied"] = rec.get("patch_rc") == 0
    return rec


def ci(vals, rng):
    meds = sorted(statistics.median(vals[rng.randrange(len(vals))] for _ in vals) for _ in range(BOOT))
    return meds[int(0.025 * BOOT)], meds[int(0.975 * BOOT) - 1]


def analyse(rec: dict) -> dict:
    tp = [p for p in rec["time"] if "error" not in p]
    cp = [c for c in rec["compile"] if "error" not in c]
    pairs = [x for p in tp for x in p["pairs"]]
    val = {"patch_applied": rec["patch"]["applied"],
           "processes_ok": len(tp) == PROCESSES and len(cp) == len(COMPILE_ORDER),
           "pairs_complete": len(pairs) == PROCESSES * PAIRS_PER_PROCESS,
           "bitwise_all": bool(tp) and all(all(p["bitwise"].values()) for p in tp),
           "off_i64": all(set(k["ks"].values()) <= {"i64"} for p in tp for k in p["kernels"] if k["tag"] == "OFF"),
           "on_i32": all(any(k["tag"] == "ON" and k["ks"] and set(k["ks"].values()) == {"i32"} for k in p["kernels"])
                         for p in tp),
           "no_compiles_during_measurement": all(p["compiles_during_measurement"] == 0 for p in tp)}
    by = {v: [c for c in cp if c["variant"] == v] for v in VARIANTS}
    progs = ["target"] + list(OTHER_SRC)
    same = lambda f: all(len({f(c["programs"][g]) for c in cp}) == 1 for g in progs)  # noqa: E731
    val["graphs_equal_compile"] = bool(cp) and same(lambda x: x["graphs"])
    val["unrelated_rejected"] = all(not ch["accepted"] for c in by["ON"] for g in OTHER_SRC
                                    for ch in c["programs"][g]["checks"])
    val["unrelated_ks_unchanged"] = bool(cp) and all(
        len({json.dumps([k["ks"] for k in c["programs"][g]["kernels"]], sort_keys=True) for c in cp}) == 1
        for g in OTHER_SRC)
    val["valid"] = all(val.values())
    res = {"validity": val, "pairs": len(pairs),
           "reported_expected_equal": {   # findings about the patch if false; not validity conditions
               "guards_equal_time": all(p["guards"]["OFF"] == p["guards"]["ON"] for p in tp),
               "guards_equal_compile": bool(cp) and same(lambda x: x["guards"])}}
    if not pairs:
        return res
    rng = random.Random(SEED)
    rT = [x["blocks"]["OFF"]["T_s"] / x["blocks"]["ON"]["T_s"] for x in pairs]
    m, c = statistics.median(rT), ci(rT, rng)
    res["sequence"] = {"median_rT": m, "ci": c,
                       "median_T_ms": {v: statistics.median(x["blocks"][v]["T_s"] for x in pairs) * 1e3 for v in VARIANTS},
                       "verdict": "benefit retained" if (m >= 1.03 and c[0] >= 1.00)
                       else ("no benefit shown" if c[1] < 1.03 else "unresolved")}
    small = {}
    for s in SMALL:
        key = f"{s[0]}x{s[1]}"
        r = [x["blocks"]["OFF"]["small_us"][key] / x["blocks"]["ON"]["small_us"][key] for x in pairs]
        cc = ci(r, rng)
        small[key] = {"median_rS": statistics.median(r), "ci": cc,
                      "class": "non-regression shown" if cc[0] >= 0.97 else ("regression shown" if cc[1] < 0.97 else "unresolved"),
                      "median_us": {v: statistics.median(x["blocks"][v]["small_us"][key] for x in pairs) for v in VARIANTS}}
    res["small"] = small
    res["compile_cost"] = {
        "check_us_target": statistics.median(ch["us"] for c in by["ON"] for ch in c["programs"]["target"]["checks"])
        if by["ON"] else None,
        "check_us_unrelated": statistics.median([ch["us"] for c in by["ON"] for g in OTHER_SRC
                                                 for ch in c["programs"][g]["checks"]] or [0.0]),
        "first_call_s": {g: {v: statistics.median(c["programs"][g]["first_call_s"] for c in by[v]) for v in VARIANTS if by[v]}
                         for g in progs}}
    res["clock_median_mhz"] = {v: statistics.median(x["blocks"][v]["telemetry"].get("sm_median", 0) for x in pairs)
                               for v in VARIANTS}
    res["stage14_B0_over_N1_for_comparison"] = 1.349   # descriptive only (different stack); declared in advance
    res["verdict"] = res["sequence"]["verdict"] if val["valid"] else "INVALID"
    return res


def bench() -> dict:
    import launch_inductor_narrow as S13
    rec = {"kind": "stage 15: upstream patch off vs on (pre-registered analysis in module docstring)", **S13.env(),
           "seq": SEQ, "small": SMALL, "patch": apply_patch(), "time": [], "compile": []}
    if rec["patch"]["applied"]:
        for v in COMPILE_ORDER:
            rec["compile"].append(_sub(["--compile-worker", v], timeout=900))
        for p in range(PROCESSES):
            rec["time"].append(_sub(["--time-worker", str(p)]))
    rec["analysis"] = analyse(rec)
    import launch_local_check as LC
    rec["input_hashes"] = LC.input_hashes()
    return rec


def cpu_dry() -> dict:
    """Codegen-only checks on the CPU Triton backend (local veritile-nightly image, patch already applied)."""
    import torch._inductor.codegen.triton as ct
    import torch._inductor.config as ic
    import torch.utils._triton as ut
    ic.cpu_backend = "triton"
    ic.force_pointwise_cat = True
    ut.triton_hash_with_backend = lambda: "stub"
    ct.triton_hash_with_backend = lambda: "stub"
    S12.make_inputs = _cpu_inputs
    return {v: worker_compile(v, dev="cpu") for v in VARIANTS}


def _cpu_inputs(n, w):
    import torch
    ws = S12.WIDTHS[w]
    widths = [ws[0], ws[1], ws[2], ws[2], ws[0], ws[1], ws[2], ws[2]]
    return [torch.randn(n, x, dtype=torch.bfloat16) for x in widths]


if __name__ == "__main__":
    try:
        if "--time-worker" in sys.argv:
            r = worker_time(int(sys.argv[sys.argv.index("--time-worker") + 1]))
        elif "--compile-worker" in sys.argv:
            r = worker_compile(sys.argv[sys.argv.index("--compile-worker") + 1])
        elif "--cpu-dry" in sys.argv:
            r = cpu_dry()
        elif "--dry" in sys.argv:
            r = {"seq": len(SEQ), "small": SMALL, "pairs": PAIRS_PER_PROCESS * PROCESSES,
                 "compile_processes": COMPILE_ORDER, "other_programs": list(OTHER_SRC), "boot": BOOT}
        else:
            r = bench()
    except Exception:  # noqa: BLE001
        r = {"error": traceback.format_exc()[-4000:]}
    print("RESULT " + json.dumps(r, default=str))
