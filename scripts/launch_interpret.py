#!/usr/bin/env python3
"""Run the pinned add_example / vector_addition_custom kernels under Triton's
CPU interpreter (TRITON_INTERPRET=1). INTERPRETER EVIDENCE ONLY — not GPU
correctness, not performance.

The TritonBench-G files execute CUDA tests at module scope, so only the text
before their `#####` test banner (imports, unmodified kernel and wrapper) is
imported. If the pinned source fails in the interpreter, that failure is
recorded and a DERIVED VARIANT with an explicit, recorded text change is run
and labelled as such.

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


def load_defs(py: Path, variant: dict | None = None) -> dict:
    """Import the file's kernel/wrapper code (text before the '#####' test banner, the
    TritonBench-G layout) as a real module from a temp dir; `variant` applies explicit,
    recorded text replacements (derived variant, NOT the pinned source)."""
    import importlib.util
    import tempfile
    src = py.read_text()
    assert "#####" in src, f"{py}: missing test banner"
    code = src.split("#####")[0]
    for old, new in (variant or {}).items():
        assert old in code, old
        code = code.replace(old, new)
    d = Path(tempfile.mkdtemp())
    f = d / (py.stem + ("_variant" if variant else "") + ".py")
    f.write_text(code)
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return vars(mod)


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
    for rel, wrapper_name, kernel_name, block, _block_param in [
            ("bench/tritonbench_g/add_example/add_example.py", "add_wrapper", "add_kernel", 4,
             "BLOCK_SIZE"),
            ("bench/tritonbench_g/add_example/improvement/add_example_empty_like.py", "add_wrapper",
             "add_kernel", 4, "BLOCK_SIZE"),
            ("bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py", "custom_add",
             "_add_kernel", 16, "BLOCK")]:
        entry = {"source": "pinned (verbatim)"}
        ns = load_defs(REPO / rel)
        try:
            run_case(ns[wrapper_name], 5)
        except Exception as e:  # noqa: BLE001
            # Recorded as a result, then continue with an explicitly derived variant.
            variant = {f'{_block_param}: "tl.constexpr"': f"{_block_param}: tl.constexpr"}
            entry = {"source": "DERIVED VARIANT (annotation spelled as the tl.constexpr class)",
                     "pinned_source_result": "interpreter error: " + repr(e)[:160],
                     "variant": variant}
            ns = load_defs(REPO / rel, variant)
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
        res["cases"][rel] = {**entry, "wrapper_runs": rows, "direct_launches": direct, "passed": good}
    res["exit_code"] = 0 if ok else 1
    res["input_hashes"] = LC.input_hashes()
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: [v["passed"], v["source"]] for k, v in res["cases"].items()}, indent=0),
          "exit", res["exit_code"])
    return res["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
