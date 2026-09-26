#!/usr/bin/env python3
"""Dynamic-shape concatenation / QKV repack (PyTorch issue #189940): reproduce on
the pinned stack, and measure two mechanism-specific kernels.

Workload (the issue's `nested_cat_add`, bf16): per group g in {1, 2},
v_g = va_g + vb_g, then out = cat([q1, k1, v1, q2, k2, v2], dim=-1) for inputs of
shape [n, w]. The concatenation (last) dimension is dynamic
(`torch._dynamo.mark_dynamic(t, -1)`, `torch.compile(dynamic=True)`).

Candidates, each labelled valid for the dynamic workload or reference only:
  eager               the Python function (ATen: 2 adds + 3 cats)          valid
  compiled_dynamic    torch.compile(dynamic=True), last dim marked dynamic  valid
  fallback_equiv      compiled adds + eager torch.cat (what PR #190034 would do;
                      the PR is not merged)                                 valid
  compiled_static     torch.compile(dynamic=False), compiled per shape      reference only
  cand_A_tiled        Triton, 2-D grid: row = program_id(0), column tile =
                      program_id(1)*BLOCK + arange; no division             valid
  cand_B_fastdiv      Triton, 1-D flattened like Inductor's pointwise_cat,
                      but row = fast division by the runtime width using a
                      host-computed magic number (ATen IntDivider form:
                      q = (umulhi(x, m) + x) >> s); requires n*W < 2^31      valid

PREDECLARED CRITERIA (fixed before the GPU run):
  reproduce: compiled_dynamic >= 1.3x slower than eager (issue-style timing) at
             >= 6 of the 10 sequence shapes, with a symbolic divmod (`% ks` or
             `// ks`) present in its generated code.
  go:        a candidate's sequence total (issue-style timing, sum over the 10
             shapes) is >= 1.15x faster than the best valid baseline's total,
             with no shape below 0.95x of that baseline, and bitwise-equal output
             at every shape.
OUTCOME OF CALL 1 (L4, torch 2.14.0): the reproduce test above FAILED as written —
compiled_dynamic was faster than eager at all 7 measured shapes — while the symbolic
divmod was present and compiled_dynamic was ~1.8x slower than compiled_static. The
sentence that stood here ("if reproduce fails, the gap is absent") was a wrong
inference: the test compared against eager, whose cost here is dominated by its
extra memory passes, not against what a dynamic kernel can achieve.

RE-REGISTERED VALIDATION CRITERION (written after call 1, before call 2; `--validate`):
  baseline:  compiled_dynamic (the best valid baseline on this stack in call 1);
             compiled_static is reported as reference with its per-shape compile cost
  go iff, for a candidate:
    - all 10 SEQUENCE shapes measured; sequence total >= 1.15x faster than the
      baseline, and every sequence shape >= 0.95x;
    - the 4 HELD_OUT shapes (non-power-of-two and non-multiple-of-16 widths) each
      >= 0.95x, and their total >= 1.15x;
    - bitwise equal to eager at every timed shape and every EDGE case;
  EDGE cases are correctness only: n = 1, width 1, n*W just below 2^31 (B must run),
  n*W = 2^31 (B must refuse, A must run).

Timing per shape (each shape in its own process; no CUDA graphs):
  issue_ms       the issue's method: 25 warm-up, 100 back-to-back calls between
                 two CUDA events, per call
  single_us      one call + synchronize, median of 100
  kernel_us      summed CUDA kernel durations per call (torch.profiler, warm)
3 trials, candidate order rotated. The cold sequence part (one process) runs the
10 shapes in order from a cold start and records first-call (compile) times.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
WIDTHS = {"issue": (2048, 256, 256), "llama8b": (4096, 1024, 1024), "mid": (3072, 512, 512)}
SEQUENCE = [(n, w) for w in ("issue", "llama8b", "mid") for n in (2048, 4096, 16384)] + [(32768, "issue")]
WIDTHS.update({"odd": (1000, 120, 136), "wide": (5120, 640, 640), "unit": (1, 1, 1), "narrow": (48, 8, 8)})
HELD_OUT = [(3000, "odd"), (12345, "odd"), (3000, "wide"), (12345, "wide")]
EDGE = [(1, "issue"), (1000, "unit"), (7, "odd"), (2 ** 24 - 1, "narrow"), (2 ** 24, "narrow")]
TRIALS = 3


# ---------------------------------------------------------------------------
# Reference and candidates
# ---------------------------------------------------------------------------

def nested_cat_add(q1, k1, v1a, v1b, q2, k2, v2a, v2b):
    v1 = v1a + v1b
    v2 = v2a + v2b
    g1 = torch.cat([q1, k1, v1], dim=-1)
    g2 = torch.cat([q2, k2, v2], dim=-1)
    return torch.cat([g1, g2], dim=-1)


def _adds(v1a, v1b, v2a, v2b):
    return v1a + v1b, v2a + v2b


def make_inputs(n, wset, device, dtype=torch.bfloat16, seed=0):
    wq, wk, wv = WIDTHS[wset] if isinstance(wset, str) else wset
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(n, w, generator=g).to(dtype).to(device) for w in (wq, wk, wv, wv, wq, wk, wv, wv)]


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def repack_tiled(q1, k1, va1, vb1, q2, k2, va2, vb2, out, wq, wk, wv, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        G = wq + wk + wv
        inb = c < 2 * G
        g2 = c >= G
        cc = tl.where(g2, c - G, c)
        mq = inb & (cc < wq)
        mk = inb & (cc >= wq) & (cc < wq + wk)
        mv = inb & (cc >= wq + wk)
        oq = row * wq + cc
        ok = row * wk + (cc - wq)
        ov = row * wv + (cc - wq - wk)
        q_a = tl.load(q1 + oq, mask=mq & ~g2, other=0.0)
        q_b = tl.load(q2 + oq, mask=mq & g2, other=0.0)
        k_a = tl.load(k1 + ok, mask=mk & ~g2, other=0.0)
        k_b = tl.load(k2 + ok, mask=mk & g2, other=0.0)
        v_a = (tl.load(va1 + ov, mask=mv & ~g2, other=0.0).to(tl.float32)
               + tl.load(vb1 + ov, mask=mv & ~g2, other=0.0).to(tl.float32)).to(out.dtype.element_ty)
        v_b = (tl.load(va2 + ov, mask=mv & g2, other=0.0).to(tl.float32)
               + tl.load(vb2 + ov, mask=mv & g2, other=0.0).to(tl.float32)).to(out.dtype.element_ty)
        val = tl.where(mq, tl.where(g2, q_b, q_a), tl.where(mk, tl.where(g2, k_b, k_a), tl.where(g2, v_b, v_a)))
        tl.store(out + row * (2 * G) + c, val, mask=inb)

    @triton.jit(do_not_specialize=["total", "magic", "shift"])
    def repack_fastdiv(q1, k1, va1, vb1, q2, k2, va2, vb2, out, wq, wk, wv, total, magic, shift,
                       BLOCK: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)          # int32: total < 2^31 (checked)
        xm = x < total
        W = 2 * (wq + wk + wv)
        xu = x.to(tl.uint32)
        row = ((tl.umulhi(xu, magic.to(tl.uint32)) + xu) >> shift.to(tl.uint32)).to(tl.int32)
        c = x - row * W
        G = wq + wk + wv
        g2 = c >= G
        cc = tl.where(g2, c - G, c)
        mq = xm & (cc < wq)
        mk = xm & (cc >= wq) & (cc < wq + wk)
        mv = xm & (cc >= wq + wk)
        oq = row * wq + cc
        ok = row * wk + (cc - wq)
        ov = row * wv + (cc - wq - wk)
        q_a = tl.load(q1 + oq, mask=mq & ~g2, other=0.0)
        q_b = tl.load(q2 + oq, mask=mq & g2, other=0.0)
        k_a = tl.load(k1 + ok, mask=mk & ~g2, other=0.0)
        k_b = tl.load(k2 + ok, mask=mk & g2, other=0.0)
        v_a = (tl.load(va1 + ov, mask=mv & ~g2, other=0.0).to(tl.float32)
               + tl.load(vb1 + ov, mask=mv & ~g2, other=0.0).to(tl.float32)).to(out.dtype.element_ty)
        v_b = (tl.load(va2 + ov, mask=mv & g2, other=0.0).to(tl.float32)
               + tl.load(vb2 + ov, mask=mv & g2, other=0.0).to(tl.float32)).to(out.dtype.element_ty)
        val = tl.where(mq, tl.where(g2, q_b, q_a), tl.where(mk, tl.where(g2, k_b, k_a), tl.where(g2, v_b, v_a)))
        tl.store(out + x, val, mask=xm)

    return repack_tiled, repack_fastdiv


def magic_for(d: int) -> tuple[int, int]:
    """ATen IntDivider constants for 32-bit n < 2^31: (m1, shift) with
    n // d == (umulhi(n, m1) + n) >> shift."""
    shift = 0
    while (1 << shift) < d:
        shift += 1
    m1 = ((1 << 32) * ((1 << shift) - d)) // d + 1
    return m1, shift


def check_inputs(ins) -> list:
    """Obligations of both candidates (checked per call)."""
    fails = []
    q1, k1, va1, vb1, q2, k2, va2, vb2 = ins
    n = q1.shape[0]
    if any(t.dim() != 2 or not t.is_contiguous() or t.shape[0] != n for t in ins):
        fails.append("inputs: 2-D, contiguous, same row count")
    if not (q1.shape == q2.shape and k1.shape == k2.shape and va1.shape == vb1.shape == va2.shape == vb2.shape):
        fails.append("group shapes match")
    if len({t.dtype for t in ins}) != 1:
        fails.append("one dtype")
    return fails


def cand(which, kernels):
    tiled, fastdiv = kernels

    def run(*ins):
        fails = check_inputs(ins)
        if fails:
            raise ValueError(fails)
        n, (wq, wk, wv) = ins[0].shape[0], (ins[0].shape[1], ins[1].shape[1], ins[2].shape[1])
        W = 2 * (wq + wk + wv)
        if which == "B" and n * W >= 2 ** 31:
            raise ValueError("n*W < 2^31 required by the 32-bit fast division")
        out = torch.empty(n, W, dtype=ins[0].dtype, device=ins[0].device)
        if which == "A":
            B = 1024
            tiled[(n, (W + B - 1) // B)](*ins, out, wq, wk, wv, BLOCK=B)
        else:
            total = n * W
            if total >= 2 ** 31:
                raise ValueError("n*W < 2^31 required by the 32-bit fast division")
            m1, s = magic_for(W)
            B = 1024
            fastdiv[((total + B - 1) // B,)](*ins, out, wq, wk, wv, total, m1, s, BLOCK=B)
        return out
    return run


# ---------------------------------------------------------------------------
# Per-shape measurement (runs in its own process)
# ---------------------------------------------------------------------------

def sync():
    torch.cuda.synchronize()


def issue_ms(fn, ins, warmup=25, iters=100):
    for _ in range(warmup):
        fn(*ins)
    sync()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(iters):
        fn(*ins)
    e.record()
    sync()
    return s.elapsed_time(e) / iters


def single_us(fn, ins, reps=100):
    for _ in range(10):
        fn(*ins)
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(*ins); sync(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def kernel_us(fn, ins, calls=10):
    from torch.profiler import ProfilerActivity, profile
    fn(*ins); sync()
    per, names = [], set()
    for _ in range(calls):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(*ins); sync()
        ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
        per.append(sum(e.device_time for e in ev))
        names |= {e.name[:90] for e in ev}
    return {"median": statistics.median(per), "kernels": sorted(names)}


def bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def measure_shape(n, wset) -> dict:
    import torch._dynamo as dynamo
    from torch._inductor.utils import run_and_get_code
    dynamo.config.cache_size_limit = 64
    rec = {"n": n, "widths": WIDTHS[wset], "wset": wset}
    ins = make_inputs(n, wset, "cuda")
    ref = nested_cat_add(*ins)
    kernels = _kernels()
    fns = {"eager": nested_cat_add}
    dyn_ins = [t.clone() for t in ins]
    for t in dyn_ins:
        dynamo.mark_dynamic(t, t.dim() - 1)
    cdyn = torch.compile(nested_cat_add, dynamic=True)
    t0 = time.perf_counter()
    out_dyn, code = run_and_get_code(cdyn, *dyn_ins)
    rec["compile_s"] = {"compiled_dynamic": time.perf_counter() - t0}
    src = "\n".join(code)
    rec["dynamic_code"] = {"has_mod_ks": "% ks" in src, "has_div_ks": "// ks" in src,
                           "kernels": [ln.split("(")[0].replace("def ", "").strip()
                                       for ln in src.splitlines() if ln.startswith("def triton_")]}
    cstat = torch.compile(nested_cat_add, dynamic=False)
    t0 = time.perf_counter(); out_stat = cstat(*ins); sync()
    rec["compile_s"]["compiled_static"] = time.perf_counter() - t0
    cadds = torch.compile(_adds, dynamic=True)
    adds_ins = [t.clone() for t in ins]
    for t in adds_ins:
        dynamo.mark_dynamic(t, t.dim() - 1)

    def fallback_equiv(q1, k1, v1a, v1b, q2, k2, v2a, v2b):
        v1, v2 = cadds(v1a, v1b, v2a, v2b)
        return torch.cat([torch.cat([q1, k1, v1], -1), torch.cat([q2, k2, v2], -1)], -1)
    t0 = time.perf_counter(); out_fb = fallback_equiv(*adds_ins); sync()
    rec["compile_s"]["fallback_equiv"] = time.perf_counter() - t0
    A, Bk = cand("A", kernels), cand("B", kernels)
    t0 = time.perf_counter(); out_a = A(*ins); sync(); rec["compile_s"]["cand_A_tiled"] = time.perf_counter() - t0
    t0 = time.perf_counter(); out_b = Bk(*ins); sync(); rec["compile_s"]["cand_B_fastdiv"] = time.perf_counter() - t0
    fns.update({"compiled_dynamic": cdyn, "fallback_equiv": fallback_equiv, "compiled_static": cstat,
                "cand_A_tiled": A, "cand_B_fastdiv": Bk})
    args = {"eager": ins, "compiled_dynamic": dyn_ins, "fallback_equiv": adds_ins, "compiled_static": ins,
            "cand_A_tiled": ins, "cand_B_fastdiv": ins}
    rec["bitwise_equal_eager"] = {"compiled_dynamic": bits_equal(out_dyn, ref), "compiled_static": bits_equal(out_stat, ref),
                                  "fallback_equiv": bits_equal(out_fb, ref), "cand_A_tiled": bits_equal(out_a, ref),
                                  "cand_B_fastdiv": bits_equal(out_b, ref)}
    names = list(fns)
    trials = []
    for t in range(TRIALS):
        order = names[t % len(names):] + names[:t % len(names)]
        trials.append({"issue_ms": {k: issue_ms(fns[k], args[k]) for k in order},
                       "single_us": {k: single_us(fns[k], args[k]) for k in order}})
    rec["trials"] = trials
    rec["summary"] = {m: {k: statistics.median(tr[m][k] for tr in trials) for k in names}
                      for m in ("issue_ms", "single_us")}
    rec["kernel_us"] = {k: kernel_us(fns[k], args[k]) for k in names}
    return rec


def edge_case(n, wset) -> dict:
    """Correctness only (no timing)."""
    kernels = _kernels()
    ins = make_inputs(n, wset, "cuda")
    ref = nested_cat_add(*ins)
    res = {"n": n, "wset": wset, "nW": n * 2 * sum(WIDTHS[wset])}
    for name, which in (("cand_A_tiled", "A"), ("cand_B_fastdiv", "B")):
        try:
            res[name] = "bitwise_equal" if bits_equal(cand(which, kernels)(*ins), ref) else "MISMATCH"
        except ValueError as e:
            res[name] = "refused: " + str(e)
    torch.cuda.synchronize()
    return res


def cold_sequence() -> dict:
    """One process, cold: first-call time of each shape in sequence order."""
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64
    kernels = _kernels()
    cdyn = torch.compile(nested_cat_add, dynamic=True)
    A, Bk = cand("A", kernels), cand("B", kernels)
    res = []
    for n, wset in SEQUENCE:
        ins = make_inputs(n, wset, "cuda")
        dyn = [t.clone() for t in ins]
        for t in dyn:
            dynamo.mark_dynamic(t, t.dim() - 1)
        row = {"n": n, "wset": wset}
        for name, fn, a in (("compiled_dynamic", cdyn, dyn), ("cand_A_tiled", A, ins), ("cand_B_fastdiv", Bk, ins),
                            ("eager", nested_cat_add, ins)):
            t0 = time.perf_counter(); fn(*a); sync(); row[name] = time.perf_counter() - t0
        res.append(row)
    compiled_variants = {}
    for kname, kern in zip(("repack_tiled", "repack_fastdiv"), kernels):
        try:
            compiled_variants[kname] = sum(len(c[0]) for c in kern.device_caches.values())
        except Exception as e:  # noqa: BLE001
            compiled_variants[kname] = repr(e)[:100]
    return {"first_call_s": res, "triton_compiled_variants": compiled_variants,
            "dynamo_counters": {k: dict(v) for k, v in dynamo.utils.counters.items() if k in ("stats", "frames")}}


# ---------------------------------------------------------------------------
# Driver: one subprocess per shape, aggregate, decide
# ---------------------------------------------------------------------------

def env():
    import triton
    return {"device": "cuda", "gpu_name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__,
            "triton": triton.__version__, "torch_cuda": torch.version.cuda,
            "cpu": subprocess.run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo; nproc"],
                                  capture_output=True, text=True).stdout.strip(),
            "TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET")}


def _sub(args, timeout=1200) -> dict:
    r = subprocess.run([sys.executable, __file__, *args], capture_output=True, text=True, timeout=timeout)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not lines:
        return {"error": (r.stderr or r.stdout)[-3000:], "returncode": r.returncode}
    return json.loads(lines[-1][len("RESULT "):])


def bench() -> dict:
    rec = {"kind": "cat_repack (PyTorch #189940 reproduction + two candidates)", **env(),
           "sequence": SEQUENCE, "widths": WIDTHS, "rows": []}
    for n, wset in SEQUENCE:
        rec["rows"].append(_sub(["--shape", str(n), wset]))
    rec["cold_sequence"] = _sub(["--cold"], timeout=2400)
    ok = [r for r in rec["rows"] if "summary" in r]
    rep = [r for r in ok if r["summary"]["issue_ms"]["compiled_dynamic"] >= 1.3 * r["summary"]["issue_ms"]["eager"]
           and (r["dynamic_code"]["has_mod_ks"] or r["dynamic_code"]["has_div_ks"])]
    rec["reproduce"] = {"shapes": len(ok), "reproducing": len(rep), "reproduced": len(ok) == 10 and len(rep) >= 6}
    valid = ("eager", "compiled_dynamic", "fallback_equiv")
    tot = lambda k: sum(r["summary"]["issue_ms"][k] for r in ok)  # noqa: E731
    best = min(valid, key=tot)
    go = {}
    for c in ("cand_A_tiled", "cand_B_fastdiv"):
        per = [r["summary"]["issue_ms"][best] / r["summary"]["issue_ms"][c] for r in ok]
        go[c] = {"best_valid_baseline": best, "sequence_speedup": tot(best) / tot(c), "min_shape_ratio": min(per),
                 "bitwise_all": all(r["bitwise_equal_eager"][c] for r in ok),
                 "go": (len(ok) == 10 and tot(best) / tot(c) >= 1.15 and min(per) >= 0.95
                        and all(r["bitwise_equal_eager"][c] for r in ok))}
    rec["go"] = go
    import launch_local_check as LC
    rec["input_hashes"] = LC.input_hashes()
    return rec


def validate() -> dict:
    rec = {"kind": "cat_repack validation (re-registered criterion; see module docstring)", **env(),
           "sequence": SEQUENCE, "held_out": HELD_OUT, "edge": EDGE, "widths": WIDTHS,
           "rows": [], "held_out_rows": [], "edge_rows": []}
    for n, wset in SEQUENCE:
        rec["rows"].append(_sub(["--shape", str(n), wset]))
    for n, wset in HELD_OUT:
        rec["held_out_rows"].append(_sub(["--shape", str(n), wset]))
    for n, wset in EDGE:
        rec["edge_rows"].append(_sub(["--edge", str(n), wset]))
    rec["cold_sequence"] = _sub(["--cold"], timeout=2400)
    base = "compiled_dynamic"
    seq = [r for r in rec["rows"] if "summary" in r]
    ho = [r for r in rec["held_out_rows"] if "summary" in r]
    tot = lambda rows, k: sum(r["summary"]["issue_ms"][k] for r in rows)  # noqa: E731
    edge_ok = lambda c: all(r.get(c) == "bitwise_equal" or  # noqa: E731
                            (c == "cand_B_fastdiv" and r.get("nW", 0) >= 2 ** 31 and str(r.get(c)).startswith("refused"))
                            for r in rec["edge_rows"])
    go = {}
    for c in ("cand_A_tiled", "cand_B_fastdiv"):
        s_ratio = [r["summary"]["issue_ms"][base] / r["summary"]["issue_ms"][c] for r in seq]
        h_ratio = [r["summary"]["issue_ms"][base] / r["summary"]["issue_ms"][c] for r in ho]
        bit = all(r["bitwise_equal_eager"][c] for r in seq + ho)
        g = {"sequence_speedup": tot(seq, base) / tot(seq, c) if seq else None,
             "sequence_min_ratio": min(s_ratio) if s_ratio else None,
             "held_out_speedup": tot(ho, base) / tot(ho, c) if ho else None,
             "held_out_min_ratio": min(h_ratio) if h_ratio else None,
             "bitwise_timed": bit, "edge_ok": edge_ok(c)}
        g["go"] = bool(len(seq) == 10 and len(ho) == 4 and g["sequence_speedup"] >= 1.15
                       and g["sequence_min_ratio"] >= 0.95 and g["held_out_speedup"] >= 1.15
                       and g["held_out_min_ratio"] >= 0.95 and bit and g["edge_ok"])
        go[c] = g
    rec["go"] = go
    import launch_local_check as LC
    rec["input_hashes"] = LC.input_hashes()
    return rec


def dry() -> dict:
    """CPU interpreter, float32 (numpy has no bf16): indexing/plumbing only."""
    kernels = _kernels()
    out = {"note": "float32 under TRITON_INTERPRET=1; bf16 is first exercised on the GPU"}
    for n, w in ((3, (8, 4, 4)), (5, (16, 8, 8)), (7, (40, 8, 24)), (4, (1, 1, 1)), (3, (48, 8, 8))):
        ins = make_inputs(n, w, "cpu", torch.float32)
        ref = nested_cat_add(*ins)
        out[f"n{n}_w{w}"] = {"A": torch.equal(cand("A", kernels)(*ins), ref),
                             "B": torch.equal(cand("B", kernels)(*ins), ref)}
    out["magic_exhaustive_small"] = all(((((x * m) >> 32) + x) >> s) == x // d
                                        for d in (1, 2, 3, 7, 10, 640, 5120, 11264, 12288)
                                        for m, s in [magic_for(d)] for x in range(0, 200000, 37))
    out["magic_power_of_two"] = all(((((x * m) >> 32) + x) >> s) == x // d
                                     for d in (1, 2, 8192, 2 ** 20) for m, s in [magic_for(d)]
                                     for x in (0, 1, d - 1, d, 2 ** 31 - 1))
    out["magic_edge"] = all(((((x * m) >> 32) + x) >> s) == x // d for d in (5120, 11264, 12288, 2 ** 20 - 3)
                            for m, s in [magic_for(d)] for x in (2 ** 31 - 1, 2 ** 31 - 2, 2 ** 30, d - 1, d, d + 1))
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "bench"))
    if "--shape" in sys.argv:
        i = sys.argv.index("--shape")
        try:
            print("RESULT " + json.dumps(measure_shape(int(sys.argv[i + 1]), sys.argv[i + 2]), default=str))
        except Exception:  # noqa: BLE001
            print("RESULT " + json.dumps({"error": traceback.format_exc()[-3000:]}))
    elif "--edge" in sys.argv:
        i = sys.argv.index("--edge")
        try:
            print("RESULT " + json.dumps(edge_case(int(sys.argv[i + 1]), sys.argv[i + 2]), default=str))
        except Exception:  # noqa: BLE001
            print("RESULT " + json.dumps({"error": traceback.format_exc()[-3000:]}))
    elif "--validate" in sys.argv:
        print(json.dumps(validate(), indent=1, default=str))
    elif "--cold" in sys.argv:
        print("RESULT " + json.dumps(cold_sequence(), default=str))
    elif torch.cuda.is_available():
        print(json.dumps(bench(), indent=1, default=str))
    else:
        print(json.dumps(dry(), indent=1, default=str))
