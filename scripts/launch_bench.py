#!/usr/bin/env python3
"""Measured optimization experiment: (1) one bounded tuning pass of the checked
strided ReLU, (2) the fused float32 add + ReLU candidate against the best
unfused pipeline, eager PyTorch and warmed-up compiled PyTorch.

Two quantities are reported separately and never mixed:

* **device time** — `kernel_us`: sum of the GPU kernel durations of one call
  (torch.profiler, CUDA activity), median over calls; plus `event_ms`:
  `triton.testing.do_bench` event time of the launch-only callable (includes
  launch gaps; context only).
* **checked end-to-end time** — `e2e_us`: host wall time of one complete call
  (for the checked paths: metadata extraction, contract decision, output
  allocation, launch) followed by `torch.cuda.synchronize()`, median over calls.

`relu_tune(device)` / `fusion(device, relu_num_warps)` return one record each.
With `device="cpu"` and `TRITON_INTERPRET=1` only the correctness and
plumbing parts run (tiny sizes, no timings): a dry run, never evidence.
"""
from __future__ import annotations

import math
import os
import statistics
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "bench"))

import torch  # noqa: E402

import launch_fused as F  # noqa: E402
import launch_invoke as I  # noqa: E402
import launch_local_check as LC  # noqa: E402  (input_hashes)

SIZES = [2 ** 16, 2 ** 20, 2 ** 22, 2 ** 24, 2 ** 25]  # 2^25 = 65536 * 512: last one-tile ReLU size
DRY_SIZES = [1, 63, 64, 65, 1000]
WARPS = [1, 2, 4, 8]
CURRENT_WARPS = 4  # the pinned wrapper passes heuristics_for_num_warps(tile) = 4 for tile < 2048
TIE = 0.02


def env(device: str) -> dict:
    import triton
    rec = {"device": device, "triton": triton.__version__, "torch": torch.__version__,
           "python": sys.version.split()[0], "TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET")}
    if device == "cuda":
        rec["gpu_name"] = torch.cuda.get_device_name()
        rec["compute_capability"] = list(torch.cuda.get_device_capability())
        rec["torch_cuda"] = torch.version.cuda
        try:
            rec["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:  # noqa: BLE001
            rec["nvidia_smi"] = repr(e)
    return rec


def sync() -> None:
    torch.cuda.synchronize()


def event_ms(fn) -> dict:
    import triton
    q = triton.testing.do_bench(fn, warmup=25, rep=200, quantiles=[0.2, 0.5, 0.8])
    return {"p20": q[0], "median": q[1], "p80": q[2], "method": "triton.testing.do_bench(25, 200)"}


def kernel_us(fn, calls: int = 30) -> dict:
    """Median over calls of the summed CUDA kernel durations of one call."""
    from torch.profiler import ProfilerActivity, profile
    fn(); sync()
    per_call = []
    for _ in range(calls):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); sync()
        ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
        per_call.append(sum(e.device_time for e in ev))
        names = sorted({e.name for e in ev})
    return {"median": statistics.median(per_call), "min": min(per_call), "max": max(per_call),
            "kernels": names, "calls": calls, "method": "torch.profiler CUDA kernel durations, summed per call"}


def e2e_us(fn, reps: int = 300) -> dict:
    for _ in range(20):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return {"p20": ts[len(ts) // 5], "median": statistics.median(ts), "p80": ts[4 * len(ts) // 5],
            "reps": reps, "method": "perf_counter around call + torch.cuda.synchronize()"}


def bits(t: torch.Tensor) -> list:
    return [hex(struct.unpack("<I", struct.pack("<f", v))[0]) for v in t.float().cpu().tolist()]


def bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


# ---------------------------------------------------------------------------
# (1) bounded ReLU tuning pass: num_warps only (inside the proved contract)
# ---------------------------------------------------------------------------

def relu_tune(device: str) -> dict:
    dry = device == "cpu"
    rec = {"kind": "relu_tune (checked strided ReLU, store_cast_fix text; num_warps sweep)", **env(device),
           "search_space": {"num_warps": WARPS, "sizes": DRY_SIZES if dry else SIZES},
           "in_contract": "num_warps is a compile option: ignored by the recognizer and the Lean model; "
                          "tile size, grid and strides are the pinned heuristic's (StridedUnary.launch)",
           "not_searched": "tile cap (512) and CTA cap (65536): pinned wrapper text and StridedUnary.launch "
                           "(frozen contract); changing them is a contract change",
           "rows": [], "torch_relu": []}
    torch.manual_seed(0)
    for n in (DRY_SIZES if dry else SIZES):
        x = torch.randn(n, device=device)
        ref = torch.relu(x)
        for nw in WARPS:
            relu = F.CheckedReluTuned(nw, interpreter=dry)
            out = torch.empty_like(x)
            failed, c = relu.verdict(x, out)
            assert not failed, failed
            relu.launch_only(x, out, c)
            row = {"n": n, "num_warps": nw, "equal_torch_relu": bool(torch.equal(out, ref))}
            if not dry:
                row["kernel_us"] = kernel_us(lambda: relu.launch_only(x, out, c))
                row["event_ms"] = event_ms(lambda: relu.launch_only(x, out, c))
            rec["rows"].append(row)
        if not dry:
            o2 = torch.empty_like(x)
            rec["torch_relu"].append({"n": n, "kernel_us": kernel_us(lambda: torch.clamp_min(x, 0, out=o2)),
                                      "event_ms": event_ms(lambda: torch.clamp_min(x, 0, out=o2)),
                                      "op": "torch.clamp_min(x, 0, out=o) (= relu, preallocated)"})
    rec["all_correct"] = all(r["equal_torch_relu"] for r in rec["rows"])
    if not dry:
        big = [n for n in SIZES if n >= 2 ** 22]
        score = {nw: math.exp(statistics.mean(math.log(r["kernel_us"]["median"]) for r in rec["rows"]
                                              if r["num_warps"] == nw and r["n"] in big)) for nw in WARPS}
        best = min(score, key=score.get)
        sel = best if score[best] < (1 - TIE) * score[CURRENT_WARPS] else CURRENT_WARPS
        rec["selection_rule"] = (f"geometric mean of median kernel_us over n >= 2^22; a setting replaces the "
                                 f"current num_warps={CURRENT_WARPS} only if more than {TIE:.0%} faster")
        rec["score_kernel_us_geomean"] = score
        rec["selected_num_warps"] = sel
    rec["exit_code"] = 0 if rec["all_correct"] else 1
    rec["input_hashes"] = LC.input_hashes()
    return rec


# ---------------------------------------------------------------------------
# (2) fusion: fused vs best unfused vs eager vs compiled
# ---------------------------------------------------------------------------

SPECIAL = [1.5, -1.5, 0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e-45, -1e-45,
           3.4028234663852886e38, -3.4028234663852886e38, 1.0]


def fusion(device: str, relu_num_warps: int | None = None) -> dict:
    dry = device == "cpu"
    rec = {"kind": "fusion (fused float32 add+ReLU vs unfused checked pipeline, eager, torch.compile)",
           **env(device), "relu_num_warps": relu_num_warps,
           "candidates": {
               "fused_checked": "improvement/add_relu_fused.py::add_relu_wrapper via launch_fused.CheckedFusedAddRelu "
                                "(Elementwise2 contract, BLOCK_SIZE 64; AddReluFused.lean)",
               "unfused_checked": "checked add_example_block64 add_wrapper -> temporary, then checked strided ReLU "
                                  "(store_cast_fix text, selected num_warps); rank-1 view of the temporary",
               "eager": "torch.relu(x + y)",
               "compiled": "torch.compile(lambda x, y: torch.relu(x + y), dynamic=False), warmed per shape"},
           "sizes": [], "special_values": None, "frame": [], "fused_block_diagnostic": []}
    fused = F.CheckedFusedAddRelu()
    unfused = F.UnfusedPipeline(relu_num_warps, interpreter=dry)
    eager = lambda x, y: torch.relu(x + y)  # noqa: E731
    compiled = None
    if not dry:
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = 64
        compiled = torch.compile(lambda x, y: torch.relu(x + y), dynamic=False)
    torch.manual_seed(1)
    for n in (DRY_SIZES if dry else SIZES):
        x, y = torch.randn(n, device=device), torch.randn(n, device=device)
        row = {"n": n}
        a, b, e = fused(x, y), unfused(x, y), eager(x, y)
        row["fused_eq_unfused_bitwise"] = bitwise(a, b)
        row["fused_eq_eager_bitwise"] = bitwise(a, e)
        if compiled is not None:
            for _ in range(5):
                c_out = compiled(x, y)
            sync()
            row["fused_eq_compiled_bitwise"] = bitwise(a, c_out)
        if not dry:
            out = torch.empty_like(x); tmp = torch.empty_like(x); out2 = torch.empty_like(x)
            fz, fcfg = fused.verdict(x, y, out); assert not fz
            add = unfused.add
            az, acfg = add.verdict(x, y, tmp); assert not az
            rz, rcfg = unfused.relu.verdict(tmp, out2); assert not rz

            def add_launch():
                vals = {"x": x, "y": y, "out": tmp, "n": acfg["n"], "B": acfg["block"]}
                add._kernel[(acfg["grid"][0],)](*(vals[add._roles[p]] for p in add._params))

            def unfused_launch():
                add_launch(); unfused.relu.launch_only(tmp, out2, rcfg)
            dev = {"fused_checked": lambda: fused.launch_only(x, y, out, fcfg),
                   "unfused_checked": unfused_launch,
                   "eager": lambda: eager(x, y),
                   "compiled": lambda: compiled(x, y)}
            row["device"] = {k: {"kernel_us": kernel_us(f), "event_ms": event_ms(f)} for k, f in dev.items()}
            end = {"fused_checked": lambda: fused(x, y), "unfused_checked": lambda: unfused(x, y),
                   "eager": lambda: eager(x, y), "compiled": lambda: compiled(x, y)}
            row["e2e_us"] = {k: e2e_us(f) for k, f in end.items()}
            row["bytes_moved_ideal"] = {"fused": 12 * n, "unfused": 20 * n}
        rec["sizes"].append(row)
    # special values: NaN, signed zeros, infinities, denormals, extremes
    v = torch.tensor(SPECIAL, device=device)
    xs = v.repeat_interleave(len(SPECIAL)); ys = v.repeat(len(SPECIAL))
    fa, fb, fe = fused(xs, ys), unfused(xs, ys), eager(xs, ys)
    sv = {"pairs": len(xs), "fused_eq_unfused_bitwise": bitwise(fa, fb),
          "fused_eq_eager_bitwise": bitwise(fa, fe),
          "mismatch_vs_eager": [{"x": repr(float(xs[i])), "y": repr(float(ys[i])), "fused": bits(fa[i:i + 1])[0],
                                 "eager": bits(fe[i:i + 1])[0]} for i in range(len(xs))
                                if not bitwise(fa[i:i + 1], fe[i:i + 1])]}
    if compiled is not None:
        fc = compiled(xs, ys)
        sv["fused_eq_compiled_bitwise"] = bitwise(fa, fc)
        sv["mismatch_vs_compiled"] = len([i for i in range(len(xs)) if not bitwise(fa[i:i + 1], fc[i:i + 1])])
    rec["special_values"] = sv
    # frame: launch the checked fused kernel into a window of a sentinel-filled buffer
    for n in DRY_SIZES:
        x, y = torch.randn(n, device=device), torch.randn(n, device=device)
        buf = torch.full((n + 256,), 1234.5, device=device)
        out = buf[128:128 + n]
        failed, cfg = fused.verdict(x, y, out)
        fused.launch_only(x, y, out, cfg)
        rec["frame"].append({"n": n, "all_outputs_written": bitwise(out, eager(x, y)),
                             "sentinels_intact": bool((buf[:128] == 1234.5).all() and (buf[128 + n:] == 1234.5).all())})
    # diagnostic only: fused kernel device time at other block sizes (not the committed configuration)
    if not dry:
        import launch_interpret as LI
        n = 2 ** 24
        x, y = torch.randn(n, device=device), torch.randn(n, device=device)
        out = torch.empty_like(x)
        for blk in (64, 256, 1024, 4096):
            k = LI.load_defs(F.FUSED_PY)["add_relu_kernel"]
            g = (n + blk - 1) // blk
            f = lambda: k[(g,)](x, y, out, n, blk)  # noqa: E731
            f(); sync()
            rec["fused_block_diagnostic"].append({"n": n, "BLOCK_SIZE": blk, "kernel_us": kernel_us(f),
                                                  "correct": bitwise(out, eager(x, y))})
    ok = all(r["fused_eq_unfused_bitwise"] and r["fused_eq_eager_bitwise"] for r in rec["sizes"]) and \
        all(r.get("fused_eq_compiled_bitwise", True) for r in rec["sizes"]) and \
        sv["fused_eq_unfused_bitwise"] and all(f["all_outputs_written"] and f["sentinels_intact"]
                                                for f in rec["frame"])
    rec["correctness_ok"] = ok
    if not dry:
        rec["summary"] = summarize(rec)
    rec["exit_code"] = 0 if ok else 1
    rec["input_hashes"] = LC.input_hashes()
    return rec


def summarize(rec: dict) -> dict:
    out = []
    for r in rec["sizes"]:
        d = {k: v["kernel_us"]["median"] for k, v in r["device"].items()}
        e = {k: v["median"] for k, v in r["e2e_us"].items()}
        out.append({"n": r["n"], "device_kernel_us": d, "e2e_us": e,
                    "device_speedup_fused_vs": {k: d[k] / d["fused_checked"] for k in d if k != "fused_checked"},
                    "e2e_speedup_fused_vs": {k: e[k] / e["fused_checked"] for k in e if k != "fused_checked"}})
    regress = [{"n": s["n"], "metric": m, "vs": k, "ratio": v}
               for s in out for m in ("device_speedup_fused_vs", "e2e_speedup_fused_vs")
               for k, v in s[m].items() if v < 1 - TIE]
    return {"per_size": out, "regressions_fused_slower_by_more_than_2pct": regress}


def main() -> int:
    import json
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for name, fn in (("relu_tune", lambda: relu_tune(device)),):
        try:
            res[name] = fn()
        except Exception:  # noqa: BLE001
            res[f"error_{name}"] = traceback.format_exc()[-3000:]
    nw = res.get("relu_tune", {}).get("selected_num_warps")
    try:
        res["fusion_bench"] = fusion(device, nw)
    except Exception:  # noqa: BLE001
        res["error_fusion_bench"] = traceback.format_exc()[-3000:]
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
