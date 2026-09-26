#!/usr/bin/env python3
"""Two bounded follow-ups to the add + ReLU milestone (tag stage10-frozen):

(a) torch.compile integration: the existing fused kernel registered with
    `torch.library.triton_op` / `wrap_triton`, called
      op_checked        fast contract decision in Python, then the op (eager)
      compiled_checked  torch.compile of a function that runs the same check, then
                        the op; graph breaks are recorded (torch._dynamo.explain)
      compiled_unchecked torch.compile of the op alone: UNCHECKED, for reference
    against the general checked API (`CheckedFusedAddReluSelected`) and eager.
(b) the selected general API timed as a unit (the stage-9 block rule was not),
    next to its two fixed-block components, at sizes around the 2^24 threshold.

Each part runs in its own process. Host timings: single (call + synchronize,
median of 200) and repeated (100 calls, one synchronize), 3 rotated trials.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "bench"))

import torch  # noqa: E402

SIZES_A = [2 ** 12, 2 ** 16, 2 ** 20, 2 ** 22]
SIZES_B = [2 ** 12, 2 ** 20, 2 ** 24 - 1, 2 ** 24, 2 ** 25]


def sync():
    torch.cuda.synchronize()


def single_us(fn, reps=200):
    for _ in range(10):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def repeated_us(fn, k=100):
    for _ in range(10):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(k):
        fn()
    sync()
    return (time.perf_counter() - t0) / k * 1e6


def timed(fns, trials=3, big=False):
    names = list(fns)
    tr = []
    for t in range(trials):
        order = names[t % len(names):] + names[:t % len(names)]
        tr.append({"single": {k: single_us(fns[k], 50 if big else 200) for k in order},
                   "repeated": {k: repeated_us(fns[k], 20 if big else 100) for k in order}})
    return {"trials": tr, "summary": {m: {k: statistics.median(x[m][k] for x in tr) for k in names}
                                      for m in ("single", "repeated")}}


def part_a() -> dict:
    import launch_fast as FA
    import launch_select as S
    from torch.library import triton_op, wrap_triton
    sel = S.CheckedFusedAddReluSelected()
    kern = sel._kernel

    @triton_op("veritile::add_relu", mutates_args={})
    def add_relu_op(x: torch.Tensor, y: torch.Tensor, block: int) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        wrap_triton(kern)[((n + block - 1) // block,)](x, y, out, n, block)
        return out

    def op_checked(x, y):
        out = torch.empty_like(x)
        B = S.select_block(x.numel())
        ok, _, _ = FA.fast_ew2(B, FA.raw_meta(x), FA.raw_meta(y), FA.raw_meta(out))
        if not ok:
            raise ValueError("contract")
        return add_relu_op(x, y, B)

    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64
    compiled_checked = torch.compile(op_checked, dynamic=False)
    compiled_unchecked = torch.compile(lambda x, y: add_relu_op(x, y, 256), dynamic=False)
    rec = {"rows": []}
    x0, y0 = torch.randn(4096, device="cuda"), torch.randn(4096, device="cuda")
    try:
        ex = dynamo.explain(op_checked)(x0, y0)
        rec["explain_checked"] = {"graph_count": ex.graph_count, "graph_break_count": ex.graph_break_count,
                                  "break_reasons": [str(b.reason)[:300] for b in ex.break_reasons][:6]}
    except Exception:  # noqa: BLE001
        rec["explain_checked"] = {"error": traceback.format_exc()[-1200:]}
    for n in SIZES_A:
        x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
        ref = torch.relu(x + y)
        fns = {"selected_checked_api": lambda: sel(x, y), "op_checked": lambda: op_checked(x, y),
               "compiled_checked": lambda: compiled_checked(x, y),
               "compiled_unchecked": lambda: compiled_unchecked(x, y), "eager": lambda: torch.relu(x + y)}
        row = {"n": n, "equal": {}}
        for k, f in fns.items():
            try:
                o = f(); sync()
                row["equal"][k] = bool(torch.equal(o, ref))
            except Exception:  # noqa: BLE001
                row["equal"][k] = "error: " + traceback.format_exc()[-600:]
        good = {k: f for k, f in fns.items() if row["equal"][k] is True or k == "eager"}
        row.update(timed(good))
        rec["rows"].append(row)
    return rec


def part_b() -> dict:
    import launch_fast as FA
    import launch_select as S
    sel, f64, f256 = S.CheckedFusedAddReluSelected(), FA.CheckedFusedAddReluFast(block=64), \
        FA.CheckedFusedAddReluFast(block=256)
    rec = {"rows": [], "rule": "block 256 for n < 2^24 = 16,777,216, else 64"}
    for n in SIZES_B:
        x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
        eq = torch.equal(sel(x, y), f64(x, y)) and torch.equal(sel(x, y), torch.relu(x + y))
        row = {"n": n, "selected_block": S.select_block(n), "bitwise_equal": eq}
        row.update(timed({"selected": lambda: sel(x, y), "fast_B64": lambda: f64(x, y),
                          "fast_B256": lambda: f256(x, y)}, big=n >= 2 ** 22))
        rec["rows"].append(row)
    return rec


def env():
    import triton
    return {"device": "cuda", "gpu_name": torch.cuda.get_device_name(), "torch": torch.__version__,
            "triton": triton.__version__, "compute_capability": list(torch.cuda.get_device_capability()),
            "cpu": subprocess.run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo; nproc"],
                                  capture_output=True, text=True).stdout.strip(),
            "TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET")}


def _sub(part):
    r = subprocess.run([sys.executable, __file__, "--part", part], capture_output=True, text=True, timeout=1500)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not lines:
        return {"error": (r.stderr or r.stdout)[-3000:]}
    return json.loads(lines[-1][len("RESULT "):])


def bench() -> dict:
    rec = {"kind": "add_relu follow-up (triton_op integration; selected API as a unit)", **env(),
           "triton_op": _sub("a"), "selected_unit": _sub("b")}
    import launch_local_check as LC
    rec["input_hashes"] = LC.input_hashes()
    return rec


if __name__ == "__main__":
    if "--part" in sys.argv:
        p = sys.argv[sys.argv.index("--part") + 1]
        try:
            print("RESULT " + json.dumps(part_a() if p == "a" else part_b(), default=str))
        except Exception:  # noqa: BLE001
            print("RESULT " + json.dumps({"error": traceback.format_exc()[-3000:]}))
    else:
        print(json.dumps(bench(), indent=1, default=str))
