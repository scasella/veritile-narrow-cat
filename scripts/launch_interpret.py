#!/usr/bin/env python3
"""Run the pinned add_example / vector_addition_custom kernels under Triton's
CPU interpreter (TRITON_INTERPRET=1). INTERPRETER EVIDENCE ONLY — not GPU
correctness, not performance.

The TritonBench-G files execute CUDA tests at module scope, so only their
import statements and top-level function definitions (the unmodified kernel
and wrapper) are executed; module-level test code is dropped.

For every `valid*` configuration of the launch manifests the wrapper output is
compared bitwise with `x + y` computed by torch (both are single IEEE fp32
additions), and an over-allocated sentinel region past `n` must be untouched.
A short-grid control (checker obligation P3 violated) must leave the tail
unwritten.

Usage (inside a Linux container with `pip install triton torch`):
    TRITON_INTERPRET=1 python3 scripts/launch_interpret.py --out interp.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

assert os.environ.get("TRITON_INTERPRET") == "1", "set TRITON_INTERPRET=1"
import torch  # noqa: E402
import triton  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import launch_local_check as LC  # noqa: E402  (input_hashes only)


def load_defs(py: Path) -> dict:
    tree = ast.parse(py.read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    mod = ast.Module(body=keep, type_ignores=[])
    ns: dict = {"__name__": py.stem}
    # Triton's JIT reads kernel source through `inspect`; register the real file.
    code = compile(mod, str(py), "exec")
    exec(code, ns)
    return ns


def run_case(wrapper, n: int, sentinel_pad: int = 8) -> dict:
    x, y = torch.randn(n), torch.randn(n)
    out = wrapper(x, y)
    ref = x + y
    return {"n": n, "bitwise_equal": bool(torch.equal(out, ref)), "shape_ok": out.shape == ref.shape}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    res = {"backend": "triton-interpreter (TRITON_INTERPRET=1, CPU)", "triton": triton.__version__,
           "torch": torch.__version__, "platform": sys.platform, "cases": {}}
    ok = True
    for rel, wrapper_name, kernel_name, block in [
            ("bench/tritonbench_g/add_example/add_example.py", "add_wrapper", "add_kernel", 4),
            ("bench/tritonbench_g/add_example/improvement/add_example_empty_like.py", "add_wrapper",
             "add_kernel", 4),
            ("bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py", "custom_add",
             "_add_kernel", 16)]:
        ns = load_defs(REPO / rel)
        man = REPO / Path(rel).parent / "launch_manifest.json"
        sizes = sorted({c.get("tensors", {}).get("x", c.get("tensors", {}).get("a", {})).get("numel")
                        for c in json.loads(man.read_text())["cases"]
                        if c["kind"].startswith("valid") and "tensors" in c and "grid" not in c} - {None})
        sizes += [(1 << 16) + 3]
        rows = [run_case(ns[wrapper_name], n) for n in sizes]
        # direct launches: sentinel region past n untouched; over-provisioned grid; short-grid control
        k = ns[kernel_name]
        direct = []
        for n, grid, expect_tail_written in [(5, 2, True), (5, 3, True), (18, 18 // block, False)]:
            x, y = torch.randn(n), torch.randn(n)
            buf = torch.full((n + 8,), 1234.5)
            k[(grid,)](x, y, buf[:n], n, block)
            full = torch.equal(buf[:n], x + y)
            direct.append({"n": n, "grid": grid, "sentinels_intact": bool((buf[n:] == 1234.5).all()),
                           "all_outputs_written_correctly": full,
                           "expected_all_written": expect_tail_written})
        good = all(r["bitwise_equal"] and r["shape_ok"] for r in rows) and all(
            d["sentinels_intact"] and d["all_outputs_written_correctly"] == d["expected_all_written"]
            for d in direct)
        ok &= good
        res["cases"][rel] = {"wrapper_runs": rows, "direct_launches": direct, "passed": good}
    res["exit_code"] = 0 if ok else 1
    res["input_hashes"] = LC.input_hashes()
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v["passed"] for k, v in res["cases"].items()}), "exit", res["exit_code"])
    return res["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
