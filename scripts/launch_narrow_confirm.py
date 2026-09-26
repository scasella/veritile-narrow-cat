#!/usr/bin/env python3
"""Stage 14: one bounded confirmation of N1 against guarded N (and B0), focused on small-shape non-regression.

Why this run exists. Stage 13's pre-registered rule did not adopt N1: one shape (3000 x odd) was 0.951x of guarded
N in round 1 and 1.09x in round 2. That verdict stands in the record. This is a SEPARATE, predeclared experiment that
asks whether a small-shape regression is reproducible. The stage-13 threshold is not relaxed, and there are no
retries.

Variants. All three are compiled in the SAME process as distinct functions, so blocks can be paired:
  B0  torch.compile(dynamic=True), last dim marked dynamic: the emitted kernel
  N   B0 plus stage-12 guarded narrowing (scripts/inductor_divmod.py, variant "N")
  N1  B0 plus the public rule (scripts/narrow_rule.py): i32 size arguments, applied only when C1 and C2 hold
      (every premise logged)
Stack: torch 2.14.0 / Triton 3.8.0 on one Modal L4, the same as stages 12-13. Private caches; FX and AOT caches off.

Shapes, fixed before the run:
  SEQ    the 14 stage-12/13 shapes in their ORDER (the complete-sequence workload)
  SMALL  every SEQ shape whose stage-13 N1 in-sequence time was below 700 us/call, plus three neighbours of
         3000 x odd: (2048,issue) (3000,odd) (2048,mid) (4096,issue) (2048,llama8b) (12345,odd) (4096,mid)
         (2500,odd) (3500,odd) (1536,issue)

Design: independently warmed, balanced blocks; fixed size; no interim looks.
  block(v)  1. warm: one SEQ pass, then continuous SEQ passes until >= 2.0 s of sustained load (the power-capped
               regime that stage 13 measured)
            2. SEQ pass: 20 calls per shape, CUDA events at shape boundaries, one synchronize.
               T = the pass's wall time
            3. SMALL: for each small shape (order rotated per block), 50 back-to-back calls between CUDA events.
               S(s) = microseconds per call
            4. NVML median SM clock and power during steps 2-3
  triple    one block of each of B0, N and N1, in an order taken from the 6 permutations
  schedule  2 independent processes x 12 triples (each permutation twice per process) = 24 triples. Exactly this;
            no early stop, no extension.

PRE-REGISTERED ANALYSIS (fixed before the GPU call; do not edit after it)
  Paired per triple t:
    rT(t)   = T_N(t) / T_N1(t)
    rS(s,t) = S_N(s,t) / S_N1(s,t)
  Statistic: the median over the 24 triples. 95% CI: percentile bootstrap over triples (10,000 resamples, seed 0).
  Overall benefit CONFIRMED iff median rT >= 1.03 and the CI lower bound >= 1.00.
  Each SMALL shape s is:
    "non-regression shown" iff the CI lower bound of rS(s) >= 0.97
    "regression shown"     iff the CI upper bound of rS(s) <  0.97
    "unresolved"           otherwise
  Recommendation:
    ADOPT N1                  iff overall confirmed and every small shape is "non-regression shown"
    KEEP N (regression)       iff any small shape is "regression shown" (the shape is named)
    UNRESOLVED (keep N)       otherwise
  Validity (all required, else the run is reported as invalid and nothing is concluded):
    - bitwise equal to eager for every variant at every SEQ and SMALL shape;
    - the N1 rule eligible (C1 and C2 all true) on its kernel;
    - the N rewrite fired;
    - no Dynamo or Inductor compile during any measured block.
  Always reported: absolute median S (us/call) per shape per variant, the rT and rS medians with CIs, B0 ratios, and
  per-block clocks.

    python3 scripts/launch_narrow_confirm.py --dry
"""
from __future__ import annotations

import itertools
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import launch_inductor_divmod as S12  # noqa: E402  (workload, ORDER; unchanged)

SEQ = list(S12.ORDER)
S12.WIDTHS.setdefault("odd", (1000, 120, 136))
SMALL = [(2048, "issue"), (3000, "odd"), (2048, "mid"), (4096, "issue"), (2048, "llama8b"), (12345, "odd"),
         (4096, "mid"), (2500, "odd"), (3500, "odd"), (1536, "issue")]
VARIANTS = ["B0", "N", "N1"]
PERMS = list(itertools.permutations(VARIANTS))
TRIPLES_PER_PROCESS, PROCESSES = 12, 2
SEQ_CALLS, SMALL_CALLS, WARM_S = 20, 50, 2.0
BOOT, SEED = 10000, 0

SRC_FN = '''
def {name}(q1, k1, v1a, v1b, q2, k2, v2a, v2b):
    v1 = v1a + v1b
    v2 = v2a + v2b
    g1 = torch.cat([q1, k1, v1], dim=-1)
    g2 = torch.cat([q2, k2, v2], dim=-1)
    return torch.cat([g1, g2], dim=-1)
'''


def make_fn(name):
    import torch
    ns = {"torch": torch}
    exec(SRC_FN.format(name=name), ns)   # distinct code object per variant: Dynamo caches per code object
    return ns[name]


# ------------------------------------------------------------------------------------------ worker

def worker(proc: int) -> dict:
    import torch
    import torch._dynamo as dynamo
    import torch._inductor.codegen.triton as ct
    import inductor_divmod as RD
    import narrow_rule as NR
    from launch_inductor_narrow import Telemetry, summarize
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(dynamo.config, attr):
            setattr(dynamo.config, attr, 64)
    state = {"variant": None}
    events = []
    orig = ct.TritonScheduling.define_kernel

    def define_kernel(self, src, node_schedule, kernel):
        v = state["variant"]
        rec = {"variant": v, "has_divmod": "xindex % ks" in src}
        if v == "N":
            new, r = RD.rewrite(src, "N")
            rec["rewritten"] = r["rewritten"]
        elif v == "N1":
            r = NR.check(src, kernel)
            new = NR.apply(src, r)
            rec.update({"rewritten": r["eligible"], "rule": {k: r[k] for k in ("C1", "C2", "eligible")}})
        else:
            new = src
            rec["rewritten"] = False
        rec["kernel_name"] = orig(self, new, node_schedule, kernel)
        events.append(rec)
        return rec["kernel_name"]
    ct.TritonScheduling.define_kernel = define_kernel

    shapes = list(dict.fromkeys(SEQ + SMALL))
    ins = {s: S12.make_inputs(*s) for s in shapes}
    for s in shapes:
        for t in ins[s]:
            dynamo.mark_dynamic(t, t.dim() - 1)
    fns = {}
    bitwise = {}
    for v in VARIANTS:
        state["variant"] = v
        f = torch.compile(make_fn(f"cat_{v}"), dynamic=True)
        for s in shapes:
            out = f(*ins[s])
            ref = S12.nested_cat_add(*ins[s])
            bitwise[f"{v}:{s[0]}x{s[1]}"] = bool(torch.equal(out.view(torch.int16), ref.view(torch.int16)))
        fns[v] = f
    torch.cuda.synchronize()
    state["variant"] = "MEASURE"          # any compile from here on is recorded as a violation
    compiles_before = len(events)
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
        seq_us = [evs[i].elapsed_time(evs[i + 1]) * 1e3 / SEQ_CALLS for i in range(len(SEQ))]
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
        return {"variant": v, "T_s": T, "seq_us": seq_us, "small_us": small, "telemetry": summarize(tel.samples, "m")}

    rng = random.Random(1000 + proc)
    perms = [p for p in PERMS for _ in range(2)]
    rng.shuffle(perms)
    triples = []
    for k, perm in enumerate(perms[:TRIPLES_PER_PROCESS]):
        small_order = SMALL[k % len(SMALL):] + SMALL[:k % len(SMALL)]
        triples.append({"order": list(perm), "blocks": {v: block(v, small_order) for v in perm}})
    return {"proc": proc, "bitwise": bitwise, "compile_events": events[:compiles_before],
            "compiles_during_measurement": len(events) - compiles_before, "triples": triples,
            "telemetry_ok": tel.ok, "telemetry_error": tel.err, "telemetry_static": tel.static}


# ------------------------------------------------------------------------------------------ analysis

def ci(vals, rng):
    meds = []
    n = len(vals)
    for _ in range(BOOT):
        meds.append(statistics.median(vals[rng.randrange(n)] for _ in range(n)))
    meds.sort()
    return meds[int(0.025 * BOOT)], meds[int(0.975 * BOOT) - 1]


def analyse(procs: list) -> dict:
    triples = [t for p in procs if "error" not in p for t in p["triples"]]
    res = {"triples": len(triples)}
    val = {"processes_ok": all("error" not in p for p in procs) and len(procs) == PROCESSES,
           "bitwise_all": all(all(p["bitwise"].values()) for p in procs if "error" not in p),
           "no_compiles_during_measurement": all(p.get("compiles_during_measurement") == 0 for p in procs if "error" not in p),
           "N1_rule_eligible": all(any(e["variant"] == "N1" and e.get("rewritten") for e in p["compile_events"])
                                   and all(e.get("rewritten") for e in p["compile_events"] if e["variant"] == "N1" and e["has_divmod"])
                                   for p in procs if "error" not in p),
           "N_rewrite_fired": all(all(e.get("rewritten") for e in p["compile_events"] if e["variant"] == "N" and e["has_divmod"])
                                  for p in procs if "error" not in p),
           "triples_complete": len(triples) == TRIPLES_PER_PROCESS * PROCESSES}
    val["valid"] = all(val.values())
    res["validity"] = val
    if not triples:
        res["recommendation"] = "INVALID"
        return res
    rng = random.Random(SEED)
    rT = [t["blocks"]["N"]["T_s"] / t["blocks"]["N1"]["T_s"] for t in triples]
    rB = [t["blocks"]["B0"]["T_s"] / t["blocks"]["N1"]["T_s"] for t in triples]
    res["overall"] = {"median_rT": statistics.median(rT), "ci": ci(rT, rng),
                      "median_T_ms": {v: statistics.median(t["blocks"][v]["T_s"] for t in triples) * 1e3 for v in VARIANTS},
                      "median_B0_over_N1": statistics.median(rB), "ci_B0_over_N1": ci(rB, rng)}
    lo = res["overall"]["ci"][0]
    res["overall"]["confirmed"] = res["overall"]["median_rT"] >= 1.03 and lo >= 1.00
    small = {}
    for s in SMALL:
        key = f"{s[0]}x{s[1]}"
        r = [t["blocks"]["N"]["small_us"][key] / t["blocks"]["N1"]["small_us"][key] for t in triples]
        c = ci(r, rng)
        cls = "non-regression shown" if c[0] >= 0.97 else ("regression shown" if c[1] < 0.97 else "unresolved")
        small[key] = {"median_rS": statistics.median(r), "ci": c, "class": cls,
                      "median_us": {v: statistics.median(t["blocks"][v]["small_us"][key] for t in triples) for v in VARIANTS}}
    res["small"] = small
    if not val["valid"]:
        res["recommendation"] = "INVALID"
    elif any(v["class"] == "regression shown" for v in small.values()):
        res["recommendation"] = "KEEP N (regression: " + ", ".join(k for k, v in small.items() if v["class"] == "regression shown") + ")"
    elif res["overall"]["confirmed"] and all(v["class"] == "non-regression shown" for v in small.values()):
        res["recommendation"] = "ADOPT N1"
    else:
        res["recommendation"] = "UNRESOLVED (keep N)"
    res["seq_median_us"] = {v: [statistics.median(t["blocks"][v]["seq_us"][i] for t in triples) for i in range(len(SEQ))]
                            for v in VARIANTS}
    res["clock_median_mhz"] = {v: statistics.median(t["blocks"][v]["telemetry"].get("sm_median", 0) for t in triples)
                               for v in VARIANTS}
    return res


# ------------------------------------------------------------------------------------------ driver

def _sub(proc: int, timeout=3000) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"vt14_{proc}_") as d:
        env = {**os.environ, "TORCHINDUCTOR_CACHE_DIR": f"{d}/inductor", "TRITON_CACHE_DIR": f"{d}/triton",
               "TORCHINDUCTOR_FX_GRAPH_CACHE": "0", "TORCHINDUCTOR_AUTOGRAD_CACHE": "0"}
        env.pop("TRITON_INTERPRET", None)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, __file__, "--worker", str(proc)], capture_output=True, text=True,
                           timeout=timeout, env=env)
        wall = time.perf_counter() - t0
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not lines:
        return {"proc": proc, "error": (r.stderr or r.stdout)[-4000:], "process_s": wall}
    out = json.loads(lines[-1][len("RESULT "):])
    out["process_s"] = wall
    return out


def bench() -> dict:
    import launch_inductor_narrow as S13
    rec = {"kind": "stage 14: N1 confirmation (pre-registered analysis in module docstring)", **S13.env(),
           "seq": SEQ, "small": SMALL, "processes": []}
    for p in range(PROCESSES):
        rec["processes"].append(_sub(p))
    rec["analysis"] = analyse(rec["processes"])
    import launch_local_check as LC
    rec["input_hashes"] = LC.input_hashes()
    return rec


if __name__ == "__main__":
    if "--worker" in sys.argv:
        p = int(sys.argv[sys.argv.index("--worker") + 1])
        try:
            print("RESULT " + json.dumps(worker(p), default=str))
        except Exception:  # noqa: BLE001
            print("RESULT " + json.dumps({"proc": p, "error": traceback.format_exc()[-4000:]}))
    elif "--dry" in sys.argv:
        print(json.dumps({"seq": len(SEQ), "small": SMALL, "triples": TRIPLES_PER_PROCESS * PROCESSES,
                          "perms_per_process": TRIPLES_PER_PROCESS, "boot": BOOT}, indent=1))
    else:
        print(json.dumps(bench(), indent=1, default=str))
