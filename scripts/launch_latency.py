#!/usr/bin/env python3
"""Stage 9: where the checked fused add + ReLU call spends its small-input time,
candidate reductions, matched baselines, and one bounded block search.

`ladder(device)` — cumulative rungs for the fused kernel, each timed with
`time.perf_counter` (no profiler in any timing):

  R0 sync            torch.cuda.synchronize() alone
  R1 compiled_launch precompiled Triton kernel handle (CompiledKernel runner), fixed args
  R2 jit_launch      kernel[grid](...)  (Triton JIT dispatch: binder, specialization key, cache)
  R3 + alloc         torch.empty_like(x) then R2
  R4 + metadata      launch_invoke.tensor_meta x3, then R3
  R5 + decision      Elementwise2 obligations + derived launch, then R4
  R6 checked_call    the general checked API, CheckedFusedAddRelu.__call__
  P1 plan_run        prepared-buffer API: revalidate snapshot + precompiled launch
  P2 plan_graph      prepared-buffer API: revalidate snapshot + CUDA graph replay
  baselines          eager relu(x + y); eager into preallocated out (add out= + relu_);
                     torch.compile default; torch.compile(mode="reduce-overhead")
  reference          unfused checked pipeline (add block 64, then strided ReLU)

Two execution modes: `single` = one call + synchronize, median over REPS;
`repeated` = K calls back to back, one synchronize, per-call mean; each
repeated over TRIALS independent trials with the candidate order rotated per
trial. Every trial's statistic is kept.

`block_search(device)` — the checked general API (`CheckedFusedAddReluB`) at
BLOCKS on search sizes and held-out sizes; judged by single-mode checked
end-to-end time; device time (summed kernel durations, torch.profiler) is kept
as secondary context only.

With device "cpu" (TRITON_INTERPRET=1) only correctness/plumbing runs.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "bench"))

import torch  # noqa: E402

import launch_fast as FA  # noqa: E402
import launch_fused as F  # noqa: E402
import launch_invoke as I  # noqa: E402

SIZES = [2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22]
HELD_OUT = [3000, 50_001, 700_001, 3_000_017]
BLOCKS = [64, 128, 256, 512, 1024, 2048, 4096]
REPS, K, TRIALS = 200, 100, 5


def env(device: str) -> dict:
    import subprocess

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
            rec["cpu"] = subprocess.run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo; nproc"],
                                        capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:  # noqa: BLE001
            rec["nvidia_smi"] = repr(e)
    return rec


def hashes() -> dict:
    import launch_local_check as LC  # only to record the input hashes
    return LC.input_hashes()


def sync() -> None:
    torch.cuda.synchronize()


def single_us(fn, reps: int = REPS) -> float:
    for _ in range(10):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def repeated_us(fn, k: int = K) -> float:
    for _ in range(10):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(k):
        fn()
    sync()
    return (time.perf_counter() - t0) / k * 1e6


def kernel_us(fn, calls: int = 20) -> float:
    """Summed CUDA kernel durations of one call (torch.profiler), median; context only."""
    from torch.profiler import ProfilerActivity, profile
    fn(); sync()
    per = []
    for _ in range(calls):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); sync()
        per.append(sum(e.device_time for e in prof.events() if e.device_type.name == "CUDA"))
    return statistics.median(per)


def bitwise(a, b) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def candidates(n: int, device: str, compiled: dict) -> tuple[dict, dict]:
    """(name -> zero-arg callable, name -> output-producing callable for correctness)."""
    x, y = torch.randn(n, device=device), torch.randn(n, device=device)
    ck = F.CheckedFusedAddRelu()
    out = torch.empty_like(x)
    failed, cfg = ck.verdict(x, y, out)
    assert not failed, failed
    k = ck._kernel
    args = (x, y, out, cfg["n"], cfg["block"])
    grid = (cfg["grid"][0],)
    errors = {}
    try:
        comp = k.warmup(*args, grid=grid)
        comp_launch = comp[(grid[0], 1, 1)]
    except Exception:  # noqa: BLE001
        errors["R1_compiled_launch"] = traceback.format_exc()[-1500:]
    obl, launch_of = ck.obligations, ck.launch_of
    plan = gplan = None
    try:
        plan = FA.prepare_add_relu(x, y, torch.empty_like(x))
    except Exception:  # noqa: BLE001
        errors["P1_plan_run"] = traceback.format_exc()[-1500:]
    try:
        gplan = FA.prepare_add_relu(x, y, torch.empty_like(x), graph=True)
    except Exception:  # noqa: BLE001
        errors["P2_plan_graph_replay"] = traceback.format_exc()[-1500:]
    pre = torch.empty_like(x)
    unfused = F.UnfusedPipeline(4)

    def r3():
        o = torch.empty_like(x); k[grid](x, y, o, n, 64)

    def r4():
        o = torch.empty_like(x)
        I.tensor_meta(x); I.tensor_meta(y); I.tensor_meta(o)
        k[grid](x, y, o, n, 64)

    def r5():
        o = torch.empty_like(x)
        mx, my, mo = I.tensor_meta(x), I.tensor_meta(y), I.tensor_meta(o)
        fl = [nm for nm, ok in obl(64, mx, my, mo) if not ok]
        c = launch_of(64, mx, my, mo)
        assert not fl
        k[(c["grid"][0],)](x, y, o, c["n"], c["block"])

    def eager_pre():
        torch.add(x, y, out=pre); pre.relu_()

    fns = {"R0_sync": lambda: None,
           "R1_compiled_launch": lambda: comp_launch(*args),
           "R2_jit_launch": lambda: k[grid](*args),
           "R3_alloc_jit": r3, "R4_meta_alloc_jit": r4, "R5_decide_meta_alloc_jit": r5,
           "R6_checked_call": lambda: ck(x, y),
           "P1_plan_run": lambda: plan.run(), "P2_plan_graph_replay": lambda: gplan.run(),
           "eager": lambda: torch.relu(x + y), "eager_preallocated": eager_pre,
           "compiled_default": lambda: compiled["default"](x, y),
           "compiled_reduce_overhead": lambda: compiled["ro"](x, y),
           "unfused_checked": lambda: unfused(x, y)}
    outs = {"R6_checked_call": lambda: ck(x, y),
            "P1_plan_run": lambda: (plan.run(), plan.tensors[2].clone())[1],
            "P2_plan_graph_replay": lambda: (gplan.run(), gplan.tensors[2].clone())[1],
            "eager": lambda: torch.relu(x + y),
            "eager_preallocated": lambda: (eager_pre(), pre.clone())[1],
            "compiled_default": lambda: compiled["default"](x, y).clone(),
            "compiled_reduce_overhead": lambda: compiled["ro"](x, y).clone(),
            "unfused_checked": lambda: unfused(x, y)}
    for name in errors:
        fns.pop(name, None)
        outs.pop(name, None)
    return fns, outs, errors


def plan_invalidation(device: str) -> dict:
    """The prepared plan must refuse to run after its buffers' metadata change."""
    res = {}
    x, y = torch.randn(4096, device=device), torch.randn(4096, device=device)
    for name, mutate in (("resize_out", lambda o: o.resize_(8192)),
                         ("set_other_storage", lambda o: o.set_(torch.empty(4096, device=device))),
                         ("contents_changed_only", lambda o: o.fill_(3.0))):
        o = torch.empty_like(x)
        try:
            plan = FA.prepare_add_relu(x, y, o)
            mutate(o)
            plan.run(); sync()
            res[name] = "ran" + (" (correct)" if bitwise(o, torch.relu(x + y)) else " (WRONG OUTPUT)")
        except FA.PlanInvalidated:
            res[name] = "refused (PlanInvalidated)"
        except Exception as e:  # noqa: BLE001
            res[name] = "error: " + repr(e)[:200]
    res["expected"] = {"resize_out": "refused", "set_other_storage": "refused", "contents_changed_only": "ran (correct)"}
    return res


def ladder(device: str) -> dict:
    rec = {"kind": "latency_ladder (fused add+ReLU, checked execution)", **env(device),
           "sizes": SIZES, "reps_single": REPS, "k_repeated": K, "trials": TRIALS,
           "modes": {"single": "one call + torch.cuda.synchronize(), median over reps (us)",
                     "repeated": "K calls back to back, one synchronize, mean per call (us)"},
           "rows": []}
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64
    f = lambda a, b: torch.relu(a + b)  # noqa: E731
    compiled = {"default": torch.compile(f, dynamic=False),
                "ro": torch.compile(f, dynamic=False, mode="reduce-overhead")}
    torch.manual_seed(0)
    for n in SIZES:
        fns, outs, errors = candidates(n, device, compiled)
        for _ in range(5):  # warm every path (compiles, cudagraph trees)
            for fn in fns.values():
                fn()
        sync()
        ref = outs["R6_checked_call"]()
        correct = {name: bitwise(o(), ref) for name, o in outs.items()}
        names = list(fns)
        trials = []
        for t in range(TRIALS):
            order = names[t % len(names):] + names[:t % len(names)]  # rotate
            trial = {"order_start": order[0], "single": {}, "repeated": {}}
            for name in order:
                trial["single"][name] = single_us(fns[name])
                trial["repeated"][name] = repeated_us(fns[name])
            trials.append(trial)
        summary = {mode: {name: {"median_of_trials": statistics.median(tr[mode][name] for tr in trials),
                                 "min": min(tr[mode][name] for tr in trials),
                                 "max": max(tr[mode][name] for tr in trials)} for name in names}
                   for mode in ("single", "repeated")}
        rec["rows"].append({"n": n, "errors": errors, "bitwise_equal_to_checked_call": correct, "trials": trials,
                            "summary": summary})
    rec["plan_invalidation"] = plan_invalidation(device)
    rec["input_hashes"] = hashes()
    rec["correctness_ok"] = all(all(v for k, v in r["bitwise_equal_to_checked_call"].items()
                                    if k != "eager" and k != "compiled_default"
                                    and k != "compiled_reduce_overhead") for r in rec["rows"])
    rec["note_correctness"] = ("bitwise equality on randn inputs (no NaN/-0.0); eager and compiled are "
                               "compared for information: they differ from the Triton ReLU only on NaN sums "
                               "and -0.0 (fusion_bench.json special values)")
    return rec


def block_search(device: str) -> dict:
    rec = {"kind": "block_search (checked general API, small/medium inputs)", **env(device),
           "blocks": BLOCKS, "search_sizes": SIZES, "held_out_sizes": HELD_OUT,
           "judged_by": "single-mode checked end-to-end time (median over reps; median of trials)",
           "rows": []}
    torch.manual_seed(1)
    for n in SIZES + HELD_OUT:
        x, y = torch.randn(n, device=device), torch.randn(n, device=device)
        ref = torch.relu(x + y)
        row = {"n": n, "held_out": n in HELD_OUT, "blocks": {}}
        cks = {b: FA.CheckedFusedAddReluB(block=b) for b in BLOCKS}
        for b, ck in cks.items():
            out = ck(x, y)
            row["blocks"][b] = {"bitwise_equal_eager": bitwise(out, ref), "e2e_single_trials": []}
        for t in range(3):
            order = BLOCKS[t % len(BLOCKS):] + BLOCKS[:t % len(BLOCKS)]
            for b in order:
                row["blocks"][b]["e2e_single_trials"].append(single_us(lambda: cks[b](x, y), reps=100))
        for b, ck in cks.items():
            e = row["blocks"][b]
            e["e2e_single_us"] = statistics.median(e["e2e_single_trials"])
            o = torch.empty_like(x)
            _, cfg = ck.verdict(x, y, o)
            e["kernel_us"] = kernel_us(lambda: ck.launch_only(x, y, o, cfg))
        rec["rows"].append(row)
    rec["correct"] = all(e["bitwise_equal_eager"] for r in rec["rows"] for e in r["blocks"].values())
    rec["input_hashes"] = hashes()
    return rec


def dry(device: str = "cpu") -> dict:
    """Interpreter plumbing check: plan revalidation, block override, correctness."""
    out = {"device": device}
    x, y = torch.randn(1000), torch.randn(1000)
    for b in (64, 256, 1024):
        o = FA.CheckedFusedAddReluB(block=b)(x, y)
        out[f"block_{b}_equal"] = bitwise(o, torch.relu(x + y))
    o = torch.empty_like(x)
    try:
        FA.prepare_add_relu(x, y, o)
        out["plan"] = "prepared (unexpected on CPU interpreter)"
    except Exception as e:  # noqa: BLE001  (warmup needs a GPU target)
        out["plan"] = "not preparable under the interpreter: " + type(e).__name__
    snap = FA.meta_snapshot(o)
    o.resize_(2000)
    out["resize_changes_snapshot"] = FA.meta_snapshot(o) != snap
    return out


def main() -> int:
    import json
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(json.dumps(dry(), indent=1, default=str))
        return 0
    res = {}
    for name, fn in (("latency_ladder", ladder), ("block_search", block_search)):
        try:
            res[name] = fn(device)
        except Exception:  # noqa: BLE001
            res[f"error_{name}"] = traceback.format_exc()[-4000:]
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
