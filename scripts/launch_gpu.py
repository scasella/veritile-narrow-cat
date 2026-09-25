#!/usr/bin/env python3
"""GPU correctness/performance runs of HANDOFF.md §B/§B2, plus extras.

`run_all(repo, device)` returns three records:

* `gpu`       — §B correctness of the pinned `add_example.py::add_kernel`
                (bitwise equality with torch `x + y`, sentinels past `n` intact);
                `exit_code` 0 only if the *pinned* kernel ran and every case passed.
* `gpu_perf`  — §B first-call and steady-state timing of the pinned kernel, and
                §B2 `zeros_like` vs `empty_like` wrapper equality and timing.
* `gpu_extras`— not part of §B: `vector_addition_custom` correctness, the W1
                rank>1 probe, the upstream module-scope tests, and (only if the
                pinned source fails to compile) a separately labelled derived
                variant with the `tl.constexpr` class annotation.

Kernel/wrapper text is the pinned file's text before its `#####` test banner
(`launch_interpret.load_defs`). With `device="cpu"` (requires
`TRITON_INTERPRET=1`) this is a dry run of the harness only: the result is
interpreter output and must never be filed as `gpu.json`. The Modal driver
(`launch_gpu_modal.py`) is the only writer of the evidence files.
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_interpret as LI  # noqa: E402  (load_defs)
import launch_local_check as LC  # noqa: E402  (input_hashes)

AE = "bench/tritonbench_g/add_example/add_example.py"
AE_IMPROVED = "bench/tritonbench_g/add_example/improvement/add_example_empty_like.py"
VAC = "bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py"
VARIANT = {'BLOCK_SIZE: "tl.constexpr"': "BLOCK_SIZE: tl.constexpr"}
SENTINEL = 1234.5


def err(e: BaseException) -> str:
    return (repr(e) + " | " + traceback.format_exc().strip().splitlines()[-1])[:400]


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    def key(t):
        v = t.contiguous().view(torch.int32).to(torch.int64)
        return torch.where(v < 0, -(v & 0x7FFFFFFF), v)
    return int((key(a) - key(b)).abs().max()) if a.numel() else 0


def compare(out: torch.Tensor, ref: torch.Tensor) -> dict:
    eq = bool(torch.equal(out, ref))
    r = {"bitwise_equal": eq}
    if not eq and out.shape == ref.shape:
        r.update(mismatches=int((out != ref).sum()), max_ulp=max_ulp(out, ref))
    return r


def bench(fn, device: str, quick: bool) -> dict:
    """Median/p20/p80 in ms. CUDA: triton.testing.do_bench (warmup 25, rep 200)."""
    if device == "cuda":
        med, p20, p80 = triton.testing.do_bench(fn, warmup=25, rep=200, quantiles=[0.5, 0.2, 0.8])
        return {"median_ms": med, "p20_ms": p20, "p80_ms": p80, "method": "triton.testing.do_bench(25, 200)"}
    ts = []
    for _ in range(3 if quick else 20):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"median_ms": ts[len(ts) // 2], "method": "perf_counter (dry run, not a GPU timing)"}


def ae_correctness(k, device: str) -> dict:
    """§B cases: manifest `valid*` n with BLOCK_SIZE 4 and cdiv grid, the
    over-provisioned grid (3,) for n = 5, and n = 2^20 + 3."""
    cases = {}
    plan = [(n, (n + 3) // 4) for n in [0, 3, 4, 5, 8, 9, 16, 32, (1 << 20) + 3]] + [(5, 3)]
    for n, grid in plan:
        name = f"n={n},grid=({grid},)"
        try:
            x, y = torch.randn(n, device=device), torch.randn(n, device=device)
            buf = torch.full((n + 8,), SENTINEL, device=device)
            k[(grid,)](x, y, buf[:n], n, 4)
            sync(device)
            cases[name] = {**compare(buf[:n], x + y),
                           "sentinels_intact": bool((buf[n:] == SENTINEL).all())}
        except Exception as e:  # noqa: BLE001
            cases[name] = {"error": err(e), "bitwise_equal": False, "sentinels_intact": False}
    return cases


def vac_correctness(ns, device: str) -> dict:
    k, w = ns["_add_kernel"], ns["custom_add"]
    res = {"wrapper": {}, "direct": {}}
    for n in [0, 1, 8, 15, 16, 17, 32, (1 << 16) + 3, (1 << 20) + 3]:
        x, y = torch.randn(n, device=device), torch.randn(n, device=device)
        res["wrapper"][n] = compare(w(x, y), x + y)
    # (n, grid, whole output expected written): cdiv, over-provisioned, short grid (P3 violated)
    for n, grid, full in [(18, 2, True), (18, 4, True), (18, 1, False)]:
        x, y = torch.randn(n, device=device), torch.randn(n, device=device)
        buf = torch.full((n + 8,), SENTINEL, device=device)
        k[(grid,)](x, y, buf[:n], n, BLOCK=16)
        sync(device)
        res["direct"][f"n={n},grid=({grid},)"] = {
            "all_outputs_correct": bool(torch.equal(buf[:n], x + y)), "expected_all_written": full,
            "sentinels_intact": bool((buf[n:] == SENTINEL).all())}
    res["passed"] = all(r["bitwise_equal"] for r in res["wrapper"].values()) and all(
        d["sentinels_intact"] and d["all_outputs_correct"] == d["expected_all_written"]
        for d in res["direct"].values())
    return res


def w1_probe(ns, device: str) -> dict:
    real = torch.empty_like
    torch.empty_like = lambda t, *a, **kw: real(t, *a, **kw).fill_(SENTINEL)
    rows = []
    try:
        for shape in [(37,), (4, 8), (3, 5, 7), (1, 40), (256, 256)]:
            x, y = torch.randn(shape, device=device), torch.randn(shape, device=device)
            out = ns["custom_add"](x, y)
            sync(device)
            ok = out.reshape(-1) == (x + y).reshape(-1)
            rows.append({"shape": list(shape), "numel": x.numel(), "launched_size": shape[0],
                         "correct_elements": int(ok.sum()),
                         "unwritten_elements": int((out.reshape(-1) == SENTINEL).sum()),
                         "w1_holds": shape[0] == x.numel()})
    finally:
        torch.empty_like = real
    confirmed = all((r["correct_elements"] == r["numel"]) if r["w1_holds"] else
                    (r["correct_elements"] == r["launched_size"]
                     and r["unwritten_elements"] == r["numel"] - r["launched_size"]) for r in rows)
    return {"cases": rows, "w1_finding_confirmed": confirmed,
            "note": "torch.empty_like patched to sentinel-fill so unwritten cells are observable"}


def module_tests(repo: Path, rel: str) -> dict:
    """Run the whole upstream file (its module-scope test) and capture its output."""
    buf = StringIO()
    try:
        with redirect_stdout(buf):
            runpy.run_path(str(repo / rel), run_name="__main__")
        return {"ok": True, "stdout_tail": buf.getvalue()[-2000:]}
    except BaseException as e:  # noqa: BLE001
        return {"ok": False, "error": err(e), "stdout_tail": buf.getvalue()[-2000:]}


def env_info(device: str) -> dict:
    info = {"device": device, "triton": triton.__version__, "torch": torch.__version__,
            "TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET"), "python": sys.version.split()[0]}
    if device == "cuda":
        info.update(gpu_name=torch.cuda.get_device_name(),
                    compute_capability=list(torch.cuda.get_device_capability()),
                    torch_cuda=torch.version.cuda)
        try:
            info["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            info["nvidia_smi"] = err(e)
    return info


def run_all(repo: Path, device: str = "cuda", quick: bool = False) -> dict:
    if device == "cuda":
        assert os.environ.get("TRITON_INTERPRET") in (None, "", "0"), "TRITON_INTERPRET set on a GPU run"
        assert torch.cuda.is_available(), "no CUDA device"
        assert torch.cuda.get_device_capability() >= (8, 0), "HANDOFF §B requires CC >= 8.0"
    torch.manual_seed(0)
    info = env_info(device)
    hashes = LC.input_hashes(repo)
    perf_n = 1 << (12 if quick else 24)
    b2_sizes = [1 << 6, 1 << 8] if quick else [1 << 10, 1 << 16, 1 << 20, 1 << 24]

    # --- pinned kernel: the very first launch doubles as the compile check and first-call timing
    ae = LI.load_defs(repo / AE)
    x, y = torch.randn(perf_n, device=device), torch.randn(perf_n, device=device)
    out = torch.empty_like(x)
    pinned_error = None
    try:
        t0 = time.perf_counter()
        ae["add_kernel"][((perf_n + 3) // 4,)](x, y, out, perf_n, 4)
        sync(device)
        first_call = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        pinned_error, first_call = err(e), None

    extras = {"kind": "gpu_extras (NOT HANDOFF §B)", **info, "input_hashes": hashes}
    if pinned_error is None:
        ae_ns, imp_ns, source = ae, LI.load_defs(repo / AE_IMPROVED), "pinned (verbatim)"
    else:
        ae_ns = LI.load_defs(repo / AE, VARIANT)
        imp_ns = LI.load_defs(repo / AE_IMPROVED, VARIANT)
        source = "DERIVED VARIANT (tl.constexpr class annotation) — NOT the pinned source"
        extras["add_example_derived_variant"] = {"variant": VARIANT,
                                                 "correctness": ae_correctness(ae_ns["add_kernel"], device)}

    # --- §B correctness (pinned only)
    gpu = {"kind": "gpu_correctness (HANDOFF §B)", **info, "kernel_source": AE + " pinned (verbatim)",
           "relation": "bitwise equality with torch x + y (fp32); sentinels past n intact",
           "input_hashes": hashes}
    if pinned_error is None:
        gpu["cases"] = ae_correctness(ae["add_kernel"], device)
        gpu["exit_code"] = 0 if all(c["bitwise_equal"] and c["sentinels_intact"]
                                    for c in gpu["cases"].values()) else 1
    else:
        gpu.update(pinned_source_error=pinned_error, exit_code=1)

    # --- §B / §B2 performance (measured on whichever source ran; labelled)
    k = ae_ns["add_kernel"]
    perf = {"kind": "gpu_performance (HANDOFF §B, §B2)", **info, "kernel_source": source,
            "input_hashes": hashes, "first_call_incl_compile_s": first_call, "n": perf_n}
    if pinned_error is not None:  # first call of the variant, for completeness
        t0 = time.perf_counter(); k[((perf_n + 3) // 4,)](x, y, out, perf_n, 4); sync(device)
        perf["variant_first_call_incl_compile_s"] = time.perf_counter() - t0
    perf["kernel_steady_state"] = bench(lambda: k[((perf_n + 3) // 4,)](x, y, out, perf_n, 4), device, quick)
    perf["kernel_effective_GBps"] = 12 * perf_n / (perf["kernel_steady_state"]["median_ms"] * 1e6)
    perf["torch_x_plus_y_reference"] = bench(lambda: torch.add(x, y, out=out), device, quick)

    eq = {}
    for n in [0, 3, 4, 5, 8, 9, 16, 32, (1 << 20) + 3]:
        a, b = torch.randn(n, device=device), torch.randn(n, device=device)
        eq[n] = bool(torch.equal(ae_ns["add_wrapper"](a, b), imp_ns["add_wrapper"](a, b)))
    rows = []
    for n in b2_sizes:
        a, b = torch.randn(n, device=device), torch.randn(n, device=device)
        rows.append({"n": n, "zeros_like_wrapper": bench(lambda: ae_ns["add_wrapper"](a, b), device, quick),
                     "empty_like_wrapper": bench(lambda: imp_ns["add_wrapper"](a, b), device, quick)})
    perf["phase5_wrapper"] = {"wrapper_outputs_equal": eq, "all_equal": all(eq.values()), "timings": rows,
                              "note": "end-to-end wrapper time (allocation + fill + launch) on this "
                                      "device only; no general speedup claim"}
    perf["exit_code"] = 0 if pinned_error is None and perf["phase5_wrapper"]["all_equal"] else 1

    # --- extras
    vac = LI.load_defs(repo / VAC)
    extras["vector_addition_custom"] = vac_correctness(vac, device)
    extras["w1_probe"] = w1_probe(vac, device)
    if device == "cuda":
        extras["upstream_module_tests"] = {AE: module_tests(repo, AE), VAC: module_tests(repo, VAC)}
    return {"gpu": gpu, "gpu_perf": perf, "gpu_extras": extras}


if __name__ == "__main__":  # dry run: TRITON_INTERPRET=1 python3 scripts/launch_gpu.py
    import json
    dev = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"
    r = run_all(Path(__file__).resolve().parents[1], dev, quick=dev == "cpu")
    print(json.dumps({k: {kk: v.get(kk) for kk in ("exit_code", "kernel_source", "pinned_source_error")}
                      for k, v in r.items()}, indent=1, default=str))
    print(json.dumps(r, indent=1, default=str)[:6000])


# ---------------------------------------------------------------------------
# BLOCK_SIZE sweep (improvement candidate; NOT HANDOFF §B)
#
# Only the wrapper constant `BLOCK_SIZE = 4` changes; the kernel text and the
# default compiler options (no num_warps override) are untouched. Selection rule,
# fixed before any measurement: choose the BLOCK minimizing the median
# end-to-end `empty_like` wrapper time at n = 2^24; any BLOCK within 2 % of that
# minimum counts as tied and the smallest tied BLOCK is chosen. A BLOCK is
# eligible only if every correctness case is bitwise equal with intact sentinels.
SWEEP_BLOCKS = [4, 16, 64, 128, 256, 512, 1024, 2048, 4096]


def block_sweep(repo: Path, device: str = "cuda", quick: bool = False) -> dict:
    if device == "cuda":
        assert os.environ.get("TRITON_INTERPRET") in (None, "", "0")
        assert torch.cuda.get_device_capability() >= (8, 0)
    torch.manual_seed(0)
    blocks = [4, 16, 64] if quick else SWEEP_BLOCKS
    sizes = [1 << 8, 1 << 10] if quick else [1 << 16, 1 << 20, 1 << 24]
    sel_n = sizes[-1]
    res = {"kind": "block_sweep (improvement candidate; NOT HANDOFF §B)", **env_info(device),
           "input_hashes": LC.input_hashes(repo),
           "text_changes": "wrapper constant `BLOCK_SIZE = 4` only" + (
               "; plus tl.constexpr class annotation (CPU dry run)" if device == "cpu" else ""),
           "selection_rule": (
               f"min median empty_like-wrapper time at n={sel_n}; within 2% = tie -> smallest BLOCK; "
               "eligible only if all correctness cases pass"), "blocks": {}}
    for blk in blocks:
        # CPU dry runs only: the interpreter rejects the pinned string annotation (REPORT finding 5).
        sub = {**(VARIANT if device == "cpu" else {}), "    BLOCK_SIZE = 4\n": f"    BLOCK_SIZE = {blk}\n"}
        z = LI.load_defs(repo / AE, sub)
        e = LI.load_defs(repo / AE_IMPROVED, sub)
        corr = {}
        for n in [0, 1, 3, blk - 1, blk, blk + 1, 3 * blk + 5, (1 << 20) + 3]:
            if n < 0:
                continue
            x, y = torch.randn(n, device=device), torch.randn(n, device=device)
            buf = torch.full((n + 8,), SENTINEL, device=device)
            e["add_kernel"][((n + blk - 1) // blk,)](x, y, buf[:n], n, blk)
            sync(device)
            wz, we = z["add_wrapper"](x, y), e["add_wrapper"](x, y)
            corr[n] = {**compare(buf[:n], x + y), "sentinels_intact": bool((buf[n:] == SENTINEL).all()),
                       "zeros_like_wrapper_equal": bool(torch.equal(wz, x + y)),
                       "empty_like_wrapper_equal": bool(torch.equal(we, x + y))}
        ok = all(c["bitwise_equal"] and c["sentinels_intact"] and c["zeros_like_wrapper_equal"]
                 and c["empty_like_wrapper_equal"] for c in corr.values())
        timings = []
        for n in sizes:
            x, y = torch.randn(n, device=device), torch.randn(n, device=device)
            out = torch.empty_like(x)
            g = ((n + blk - 1) // blk,)
            kt = bench(lambda: e["add_kernel"][g](x, y, out, n, blk), device, quick)
            timings.append({"n": n, "kernel": kt, "kernel_GBps": 12 * n / (kt["median_ms"] * 1e6),
                            "zeros_like_wrapper": bench(lambda: z["add_wrapper"](x, y), device, quick),
                            "empty_like_wrapper": bench(lambda: e["add_wrapper"](x, y), device, quick)})
        res["blocks"][blk] = {"correct": ok, "correctness": corr, "timings": timings}
    ref = []
    for n in sizes:
        x, y = torch.randn(n, device=device), torch.randn(n, device=device)
        out = torch.empty_like(x)
        ref.append({"n": n, "torch_add_out": bench(lambda: torch.add(x, y, out=out), device, quick),
                    "torch_x_plus_y_alloc": bench(lambda: x + y, device, quick)})
    res["torch_reference"] = ref
    t = {b: r["timings"][-1]["empty_like_wrapper"]["median_ms"] for b, r in res["blocks"].items() if r["correct"]}
    best = min(t.values())
    res["selected_block"] = min(b for b, v in t.items() if v <= best * 1.02)
    res["exit_code"] = 0 if all(r["correct"] for r in res["blocks"].values()) else 1
    return res


def wrapper_evidence(repo: Path) -> dict:
    """Checked invocations on CUDA tensors (whole-wrapper milestone; not HANDOFF §B)."""
    import launch_invoke as I
    assert os.environ.get("TRITON_INTERPRET") in (None, "", "0")
    assert torch.cuda.get_device_capability() >= (8, 0)
    res = I.demo("cuda")
    res["kind"] = "gpu_wrapper (checked invocations on CUDA tensors; NOT HANDOFF §B)"
    res["exit_code"] = 0 if res["all_as_expected"] else 1
    res.update(env_info("cuda"))
    return res
