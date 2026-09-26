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
Part B, compile cost. 6 fresh processes, alternating OFF/ON (OFF, ON, ON, OFF, OFF, ON).
  Cache policy per process: private empty TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR (so the local autotune cache,
  which lives under the Inductor cache dir, starts empty); FX graph and AOT autograd caches off; FX graph, autotune and
  bundled-autotune REMOTE caches forced off by environment. The effective config values are recorded.
  Order in each process: an untimed warm-up program `x + 1` (a different program, so it initialises CUDA, Triton and
  Inductor but cannot pre-compile any measured program), then target, cat3, add_relu, row_sum, softmax, in that order.
  For each program: "framework-warmed, target-cold first-call latency" = wall time of its first call, which includes
  Dynamo tracing, Inductor codegen, the eligibility check, Triton compilation, autotuning and one execution. Recorded
  with it: whether a kernel was actually defined and a graph compiled during that timed call (it must be, or the number
  does not measure compilation); the time spent inside `proven_int32_size_args` per kernel and its verdict; Dynamo
  counters; guard text (raw and normalized); and the ks signature of every kernel.
Executed-artifact binding, Part A. Before and after the measured blocks, each variant is called once while
  CachingAutotuner.run is instrumented (never during timing). This records the autotuner object that actually runs for
  that callable, the ks types in the signature it was compiled with (triton_meta), its selected launcher's config,
  registers, spills and cache_hash, and the SHA-256 of the cubin in TRITON_CACHE_DIR/<cache_hash>/, i.e. the selected
  binary that was timed. All cubins in the cache (autotuning candidates included) are also hashed, as provenance.
Rule premises (guards). At kernel definition the shape environment is read: the guard `numel <= 2147483647` and the
  value-range lower bounds of numel's symbols. After compilation, the raw Dynamo guard text is kept.

PRE-REGISTERED ANALYSIS (fixed before the GPU call; do not edit after it)
  Paired per pair p:  rT(p) = T_OFF(p) / T_ON(p);  rS(s,p) = S_OFF(s,p) / S_ON(s,p).
  Statistic: median over the 24 pairs; 95% CI: percentile bootstrap over pairs (10,000 resamples, seed 0).
  Sequence (thresholds unchanged):
    "meets the predeclared benefit criterion"      iff median rT >= 1.03 and CI lower bound >= 1.00
        meaning: a point estimate of at least 1.03x with evidence of a positive effect. It does NOT establish with
        95% confidence that the improvement is at least 3%.
    "below the predeclared useful-benefit threshold" iff CI upper bound < 1.03
        meaning: any effect is smaller than the 3% the criterion asks for; a small positive effect is not excluded.
    "unresolved"                                   otherwise
  Each SMALL shape: "non-regression shown" iff CI lower >= 0.97; "regression shown" iff CI upper < 0.97;
  otherwise "unresolved".
  Overall outcome (exhaustive; the sequence result and the small-shape classes are always reported separately):
    INVALID                                             any validity condition fails
    MET, NON-REGRESSION AT ALL SMALL SHAPES             criterion met and every small shape "non-regression shown"
    MET, SMALL-SHAPE REGRESSION: <shapes>               criterion met and some shape "regression shown"
    MET, SMALL SHAPES UNRESOLVED: <shapes>              criterion met, no regression shown, some unresolved
    BELOW THRESHOLD                                     sequence below the useful-benefit threshold
    UNRESOLVED                                          sequence unresolved
  What each outcome triggers (declared now):
    MET, NON-REGRESSION   update the draft PR with the pinned-main result; continue with review and integration tests.
    MET, REGRESSION or UNRESOLVED   report both; inspect config, compile behaviour and absolute latency of the affected
                          shapes; no size-based dispatch and no rerun to change a classification.
    BELOW THRESHOLD or UNRESOLVED   keep stage 14's result on its own stack; use the captured artifacts to locate the
                          difference (config selection, backend code, surrounding execution, the patch); no new proofs.
    INVALID               save the evidence and diagnose; not a win or a loss; no automatic re-spend.
  Compile cost is REPORTED, with no threshold: the median time in the eligibility check per kernel (target;
  unrelated), and the median first-call wall time OFF versus ON per program (6 processes; descriptive only).
  Validity (all required, else nothing is concluded):
    - pin and base files match; the patch applied;
    - bitwise equal to eager for OFF and ON at every SEQ and SMALL shape;
    - every OFF kernel has ks*: i64, and the ON target kernel has all ks*: i32;
    - the EXECUTED autotuner of the OFF callable was compiled with ks*: i64 and that of the ON callable with ks*: i32,
      the same objects run before and after the measured blocks, and their selected cubins differ;
    - the rule's premises are present for the ON target: the shape environment holds `numel <= 2147483647` and every
      symbol of numel has lower bound >= 1, and the final Dynamo guard text contains 2147483647;
    - no Dynamo or Inductor compile during any measured block.
  Compile-cost validity (applies to Part B's numbers only): every process completed; each measured program defined a
  kernel and compiled a graph during its timed call; unique graphs are identical OFF and ON for every program; every
  unrelated kernel is rejected by the check and keeps the same ks types OFF and ON. (An unrelated kernel being
  ACCEPTED would be reported as a finding about the patch, prominently.)
  Reported, expected equal, NOT a validity condition (a difference is a finding about the patch): normalized guard
    text OFF versus ON. Normalization masks only hex addresses, object/type/backend ids inside ___check_*(...) calls,
    source-location comments, the guard-latency line and the variant suffix of function names; bounds, operators and
    symbolic relations (including 2147483647) are kept. The raw text is saved too. Textual equality is optional; the
    rule's premises above are not.
  Descriptive comparison, declared now: the median OFF/ON sequence ratio is reported next to stage 14's
    B0/N1 = 1.349 (torch 2.14). Absolute times are not compared across the two stacks.
  Artifacts beyond the executed binding are descriptive: cache entries are also attributed to OFF or ON by which
    variant's compile created them; `ttir_args` keeps whatever argument names TTIR prints (possibly positional).

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
                                  "ks": dict(re.findall(r"'(ks\d+)': '(i\d+)'", src)), "premises": premises(kernel)})
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

    def raw_guards(self, tag) -> str:
        return "\n".join(m for t, m in self.guards if t == tag)

    def guard_text(self, tag) -> str:
        # Normalized: source-location comments (they name the variant's function), object ids, and the variant
        # suffix of function names are dropped, so OFF and ON compare equal unless a guard itself differs.
        norm = lambda m: "\n".join(re.sub(r"\s+#.*$", "", ln) for ln in m.splitlines()  # noqa: E731
                                   if not ln.startswith("Guard eval latency"))
        # Object ids differ between processes and compile wrappers: mask hex addresses and the integer ids passed
        # to ___check_obj_id / ___check_type_id / ___check_current_backend. Bounds such as 2147483647 are kept.
        ids = r"(___check_\w+\((?:[^()]|\([^()]*\))*?)\b\d{6,}\b"
        return "\n".join(re.sub(r"0x[0-9a-f]+|\b\w+_(?:OFF|ON)\b", "#", re.sub(ids, r"\1#", norm(m)))
                         for t, m in self.guards if t == tag)

    def guard_digest(self, tag) -> str:
        text = self.guard_text(tag)
        return hashlib.sha256(text.encode()).hexdigest()[:16] if text else ""


def premises(kernel) -> dict:
    """The rule's guard premises as the shape environment holds them when `kernel` is defined."""
    try:
        import sympy
        from torch._inductor.virtualized import V
        se = V.graph.sizevars.shape_env
        numel = kernel.numels["x"] if hasattr(kernel, "numels") else None
        rec = {"numel": str(numel)}
        g32 = [g.expr for g in se.guards if "2147483647" in str(g.expr)]
        rec["int32_guards"] = [str(e) for e in g32]
        rec["numel_guarded"] = numel is not None and any(
            isinstance(e, sympy.Le) and sympy.expand(e.lhs - numel) == 0 and int(e.rhs) == 2147483647 for e in g32)
        syms = sorted(getattr(numel, "free_symbols", ()), key=str)
        lows = {str(x): se.var_to_range[x].lower for x in syms if x in se.var_to_range}
        rec["lower_bounds"] = {k: str(v) for k, v in lows.items()}
        rec["lower_ge_1"] = bool(syms) and len(lows) == len(syms) and all(int(v) >= 1 for v in lows.values())
        return rec
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:300]}


def executed(fns: dict, args: tuple, tdir: Path) -> dict:
    """Call each variant once with CachingAutotuner.run instrumented; describe what actually ran. Fails soft: an
    instrumentation error is recorded (and makes the binding check fail) instead of losing the timing data."""
    try:
        return _executed(fns, args, tdir)
    except Exception:  # noqa: BLE001
        return {"error": traceback.format_exc()[-2000:]}


def _executed(fns: dict, args: tuple, tdir: Path) -> dict:
    import torch
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    orig, seen = CachingAutotuner.run, {}
    tag = {"v": None}

    def run(self, *a, **k):
        seen.setdefault(tag["v"], {})[id(self)] = self
        return orig(self, *a, **k)
    CachingAutotuner.run = run
    try:
        for v, f in fns.items():
            tag["v"] = v
            f(*args)
            torch.cuda.synchronize()
    finally:
        CachingAutotuner.run = orig
    out = {}
    for v, objs in seen.items():
        out[v] = []
        for i, obj in objs.items():
            sig = (getattr(obj, "triton_meta", {}) or {}).get("signature", {}) or {}
            lrec = []
            for ln in obj.launchers:
                h = getattr(ln, "cache_hash", None)
                cub = sorted((tdir / h).glob("*.cubin")) if h and (tdir / h).is_dir() else []
                lrec.append({"config": {"kwargs": dict(ln.config.kwargs), "num_warps": ln.config.num_warps,
                                        "num_stages": ln.config.num_stages},
                             "n_regs": getattr(ln, "n_regs", None), "n_spills": getattr(ln, "n_spills", None),
                             "cache_hash": h, "cubin_sha256": [hashlib.sha256(c.read_bytes()).hexdigest() for c in cub]})
            out[v].append({"id": i, "kernel": (getattr(obj, "inductor_meta", {}) or {}).get("kernel_name"),
                           "ks": {str(a): str(t) for a, t in sig.items() if str(a).startswith("ks")},
                           "launchers": lrec})
    return out


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
        out.append({"dir": cub.parent.name, "name": cub.stem, "sha256": hashlib.sha256(cub.read_bytes()).hexdigest(),
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
    probe.tag = "BIND"
    bind_args = ins[SMALL[0]]
    bound_before = executed(fns, bind_args, tdir)
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
    n_after = len(probe.kernels)
    probe.tag = "BIND"
    bound_after = executed(fns, bind_args, tdir)
    return {"proc": proc, "bitwise": bitwise, "kernels": probe.kernels[:n_before], "checks": probe.checks,
            "compiles_during_measurement": n_after - n_before, "pairs": pairs,
            "executed_before": bound_before, "executed_after": bound_after,
            "raw_guards": {v: probe.raw_guards(v) for v in VARIANTS},
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
    import torch._inductor.config as ic
    rec = {"variant": v, "programs": {}, "program_order": ["warmup(x+1, untimed)"] + list(progs),
           "cache_policy": {k: os.environ.get(k) for k in CACHE_ENV} | {
               a: getattr(ic, a, None) for a in ("fx_graph_cache", "fx_graph_remote_cache", "autotune_local_cache",
                                                 "autotune_remote_cache", "bundled_autotune_remote_cache")}}
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
        k0 = len(probe.kernels)
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
            "compiled_in_timed_call": len(probe.kernels) > k0 and dynamo.utils.counters["stats"]["unique_graphs"] > g0,
            "kernels": [k for k in probe.kernels if k["tag"] == prog],
            "checks": [c for c in probe.checks if c["tag"] == prog],
            "guards": probe.guard_digest(prog), "guard_text": probe.guard_text(prog),
            "raw_guards": probe.raw_guards(prog)}
    rec["counters"] = {k: dict(dynamo.utils.counters[k]) for k in ("stats", "recompiles") if k in dynamo.utils.counters}
    if dev == "cuda":
        rec["cubins"] = cache_cubins(os.environ["TRITON_CACHE_DIR"])
    return rec


# ------------------------------------------------------------------------------------------ driver

CACHE_ENV = ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE",
             "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE", "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
             "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE"]


def _sub(args: list, timeout=3000) -> dict:
    with tempfile.TemporaryDirectory(prefix="vt15_") as d:
        env = {**os.environ, "TORCHINDUCTOR_CACHE_DIR": f"{d}/inductor", "TRITON_CACHE_DIR": f"{d}/triton",
               "TORCHINDUCTOR_FX_GRAPH_CACHE": "0", "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
               "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0", "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0",
               "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE": "0"}
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


def _ks_set(entries) -> set:
    return {t for e in entries for t in e["ks"].values()}


def binding_ok(p: dict) -> bool:
    """The executed OFF/ON autotuners have the intended ks types, are the same objects before and after the measured
    blocks, and their selected cubins differ."""
    b, a = p.get("executed_before", {}), p.get("executed_after", {})
    if not (b.get("OFF") and b.get("ON")):
        return False
    ids = lambda d, v: sorted(e["id"] for e in d.get(v, []))  # noqa: E731
    cub = lambda v: {c for e in b[v] for ln in e["launchers"] for c in ln["cubin_sha256"]}  # noqa: E731
    target = lambda v: [e for e in b[v] if e["ks"]]  # noqa: E731
    return (bool(target("OFF")) and bool(target("ON"))
            and _ks_set(target("OFF")) == {"i64"} and _ks_set(target("ON")) == {"i32"}
            and all(ids(b, v) == ids(a, v) for v in VARIANTS)
            and bool(cub("OFF")) and bool(cub("ON")) and not (cub("OFF") & cub("ON")))


def premises_ok(kernels: list, raw: str) -> bool:
    on = [k for k in kernels if k["ks"] and set(k["ks"].values()) == {"i32"}]
    return (bool(on) and all(k["premises"].get("numel_guarded") and k["premises"].get("lower_ge_1") for k in on)
            and "2147483647" in raw)


def analyse(rec: dict) -> dict:
    tp = [p for p in rec["time"] if "error" not in p]
    cp = [c for c in rec["compile"] if "error" not in c]
    pairs = [x for p in tp for x in p["pairs"]]
    on_k = lambda p: [k for k in p["kernels"] if k["tag"] == "ON"]  # noqa: E731
    val = {"patch_applied": rec["patch"]["applied"],
           "time_processes_ok": len(tp) == PROCESSES,
           "pairs_complete": len(pairs) == PROCESSES * PAIRS_PER_PROCESS,
           "bitwise_all": bool(tp) and all(all(p["bitwise"].values()) for p in tp),
           "off_i64": bool(tp) and all(set(k["ks"].values()) <= {"i64"} for p in tp for k in p["kernels"]
                                       if k["tag"] == "OFF"),
           "on_i32": bool(tp) and all(any(k["ks"] and set(k["ks"].values()) == {"i32"} for k in on_k(p)) for p in tp),
           "executed_binding": bool(tp) and all(binding_ok(p) for p in tp),
           "rule_premises_on": bool(tp) and all(premises_ok(on_k(p), p["raw_guards"]["ON"]) for p in tp),
           "no_compiles_during_measurement": bool(tp) and all(p["compiles_during_measurement"] == 0 for p in tp)}
    val["valid"] = all(val.values())
    by = {v: [c for c in cp if c["variant"] == v] for v in VARIANTS}
    progs = ["target"] + list(OTHER_SRC)
    same = lambda f: bool(cp) and all(len({f(c["programs"][g]) for c in cp}) == 1 for g in progs)  # noqa: E731
    cval = {"compile_processes_ok": len(cp) == len(COMPILE_ORDER),
            "compiled_in_timed_call": bool(cp) and all(c["programs"][g]["compiled_in_timed_call"]
                                                       for c in cp for g in progs),
            "graphs_equal_compile": same(lambda x: x["graphs"]),
            "unrelated_rejected": bool(by["ON"]) and all(not ch["accepted"] for c in by["ON"] for g in OTHER_SRC
                                                         for ch in c["programs"][g]["checks"]),
            "unrelated_ks_unchanged": bool(cp) and all(
                len({json.dumps([k["ks"] for k in c["programs"][g]["kernels"]], sort_keys=True) for c in cp}) == 1
                for g in OTHER_SRC)}
    cval["valid"] = all(cval.values())
    res = {"validity": val, "compile_validity": cval, "pairs": len(pairs),
           "findings": {   # about the patch if false; not validity conditions
               "unrelated_kernel_accepted": not cval["unrelated_rejected"] and bool(by["ON"]),
               "guards_equal_time": bool(tp) and all(p["guards"]["OFF"] == p["guards"]["ON"] for p in tp),
               "guards_equal_compile": same(lambda x: x["guards"]),
               "off_target_numel_guarded": [all(k["premises"].get("numel_guarded") for k in p["kernels"]
                                                if k["tag"] == "OFF" and k["ks"]) for p in tp]}}
    if not pairs:
        res["outcome"] = "INVALID"
        return res
    rng = random.Random(SEED)
    rT = [x["blocks"]["OFF"]["T_s"] / x["blocks"]["ON"]["T_s"] for x in pairs]
    m, c = statistics.median(rT), ci(rT, rng)
    seq = ("meets the predeclared benefit criterion" if (m >= 1.03 and c[0] >= 1.00)
           else ("below the predeclared useful-benefit threshold" if c[1] < 1.03 else "unresolved"))
    res["sequence"] = {"median_rT": m, "ci": c, "class": seq,
                       "median_T_ms": {v: statistics.median(x["blocks"][v]["T_s"] for x in pairs) * 1e3 for v in VARIANTS}}
    small = {}
    for s in SMALL:
        key = f"{s[0]}x{s[1]}"
        r = [x["blocks"]["OFF"]["small_us"][key] / x["blocks"]["ON"]["small_us"][key] for x in pairs]
        cc = ci(r, rng)
        small[key] = {"median_rS": statistics.median(r), "ci": cc,
                      "class": "non-regression shown" if cc[0] >= 0.97 else ("regression shown" if cc[1] < 0.97 else "unresolved"),
                      "median_us": {v: statistics.median(x["blocks"][v]["small_us"][key] for x in pairs) for v in VARIANTS}}
    res["small"] = small
    if cval["valid"]:
        res["compile_cost"] = {
            "label": "framework-warmed, target-cold first-call latency (s); eligibility-check time (us)",
            "check_us_target": statistics.median([ch["us"] for c in by["ON"] for ch in c["programs"]["target"]["checks"]]
                                                 or [float("nan")]),
            "check_us_unrelated": statistics.median([ch["us"] for c in by["ON"] for g in OTHER_SRC
                                                     for ch in c["programs"][g]["checks"]] or [float("nan")]),
            "first_call_s": {g: {v: statistics.median(c["programs"][g]["first_call_s"] for c in by[v])
                                 for v in VARIANTS if by[v]} for g in progs}}
    res["clock_median_mhz"] = {v: statistics.median(x["blocks"][v]["telemetry"].get("sm_median", 0) for x in pairs)
                               for v in VARIANTS}
    res["stage14_B0_over_N1_for_comparison"] = 1.349   # descriptive only (different stack); declared in advance
    reg = [k for k, v in small.items() if v["class"] == "regression shown"]
    unr = [k for k, v in small.items() if v["class"] == "unresolved"]
    if not val["valid"]:
        res["outcome"] = "INVALID"
    elif seq.startswith("meets"):
        res["outcome"] = ("MET, SMALL-SHAPE REGRESSION: " + ", ".join(reg) if reg else
                          "MET, SMALL SHAPES UNRESOLVED: " + ", ".join(unr) if unr else
                          "MET, NON-REGRESSION AT ALL SMALL SHAPES")
    elif seq.startswith("below"):
        res["outcome"] = "BELOW THRESHOLD"
    else:
        res["outcome"] = "UNRESOLVED"
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
