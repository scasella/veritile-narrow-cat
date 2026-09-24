#!/usr/bin/env python3
"""Regression tests for scripts/launch_check.py (the trusted source adapter).

Parser/recognizer tests are pure Python. `LeanVerdictTest` runs the real Lean
checker; it fails (never skips) when `lake` or the checker build is missing.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_check as L  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ADD = (REPO / "bench/tritonbench_g/add_example/add_example.py").read_text()
ADD_LEAN = (REPO / "bench/tritonbench_g/add_example/AddExample.lean").read_text()


def mutate(old: str, new: str, src: str = ADD) -> str:
    assert old in src, old
    return src.replace(old, new, 1)


class KernelRecognition(unittest.TestCase):
    def test_add_kernel_accepted_and_matches_lean(self):
        k = L.parse_kernel(ADD, "add_kernel")
        self.assertEqual((k.ptr_params, k.out_param, k.n_param, k.block_param),
                         (["in_ptr0", "in_ptr1"], "out_ptr", "n_elements", "BLOCK_SIZE"))
        self.assertEqual(L.lean_body_statements(ADD_LEAN, "add_kernel"), k.body_statements)

    def test_benign_rename_still_recognized(self):
        src = ADD.replace("output = x + y", "result = x + y").replace(
            "tl.store(out_ptr + offsets, output, mask=mask)",
            "tl.store(out_ptr + offsets, result, mask=mask)")
        k = L.parse_kernel(src, "add_kernel")
        self.assertEqual(k.out_param, "out_ptr")

    def test_lean_transcription_drift_detected(self):
        k = L.parse_kernel(mutate("output = x + y", "output = y + x"), "add_kernel")
        self.assertNotEqual(L.lean_body_statements(ADD_LEAN, "add_kernel"), k.body_statements)

    def _rejects(self, src, fragment):
        with self.assertRaises(L.Unsupported) as cm:
            L.parse_kernel(src, "add_kernel")
        self.assertIn(fragment, str(cm.exception))

    def test_missing_store_mask_rejected(self):
        self._rejects(mutate("tl.store(out_ptr + offsets, output, mask=mask)",
                             "tl.store(out_ptr + offsets, output)"), "tl.store must be")

    def test_off_by_one_mask_rejected(self):
        self._rejects(mutate("mask = offsets < n_elements", "mask = offsets <= n_elements"),
                      "unmodelled comparison")

    def test_other_kwarg_rejected(self):
        self._rejects(mutate("x = tl.load(in_ptr0 + offsets, mask=mask)",
                             "x = tl.load(in_ptr0 + offsets, mask=mask, other=0.0)"), "tl.load must be")

    def test_unknown_op_rejected(self):
        self._rejects(mutate("output = x + y", "output = tl.exp(x) + y"), "unmodelled call")

    def test_cast_rejected(self):
        self._rejects(mutate("output = x + y", "output = x.to(tl.float16) + y"), "unmodelled")

    def test_helper_call_rejected(self):
        self._rejects(mutate("output = x + y", "output = helper(x, y)"), "unmodelled call")

    def test_control_flow_rejected(self):
        self._rejects(mutate("    output = x + y\n", "    for i in range(2):\n        pass\n    output = x + y\n"),
                      "unmodelled statement")

    def test_second_axis_rejected(self):
        self._rejects(mutate("tl.program_id(axis=0)", "tl.program_id(axis=1)"), "program_id")

    def test_wrong_kernel_stride_rejected(self):
        self._rejects(mutate("offsets = block_start + tl.arange(0, BLOCK_SIZE)",
                             "offsets = block_start + 2 * tl.arange(0, BLOCK_SIZE)"), "unmodelled")

    def test_in_place_rejected(self):
        src = mutate("tl.store(out_ptr + offsets, output, mask=mask)",
                     "tl.store(in_ptr0 + offsets, output, mask=mask)")
        self._rejects(src, "in-place")


class WrapperRecognition(unittest.TestCase):
    def setUp(self):
        self.k = L.parse_kernel(ADD, "add_kernel")

    def test_add_wrapper(self):
        w = L.parse_launch(ADD, "add_wrapper", self.k)
        self.assertEqual((w.block, w.grid_kind, w.n_source, w.out_alloc),
                         (4, "cdiv", "x", ("out", "x")))

    def test_floor_grid_is_parsed_not_rejected(self):
        src = mutate("num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE",
                     "num_blocks = n_elements // BLOCK_SIZE")
        self.assertEqual(L.parse_launch(src, "add_wrapper", self.k).grid_kind, "floordiv")

    def test_launch_option_rejected(self):
        src = mutate("add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)",
                     "add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE, num_warps=8)")
        with self.assertRaises(L.Unsupported):
            L.parse_launch(src, "add_wrapper", self.k)

    def test_two_d_grid_rejected(self):
        src = mutate("add_kernel[(num_blocks,)]", "add_kernel[(num_blocks, 2)]")
        with self.assertRaises(L.Unsupported):
            L.parse_launch(src, "add_wrapper", self.k)

    def test_non_fresh_output_rejected(self):
        src = mutate("out = torch.zeros_like(x)", "out = x")
        with self.assertRaises(L.Unsupported):
            L.parse_launch(src, "add_wrapper", self.k)

    def test_second_family_member(self):
        src = (REPO / "bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py").read_text()
        k = L.parse_kernel(src, "_add_kernel")
        w = L.parse_launch(src, "custom_add", k)
        self.assertEqual((w.block, w.grid_kind, w.n_source), (16, "cdiv", "c#dim0"))


class Metadata(unittest.TestCase):
    def test_torch_metadata(self):
        import torch
        base = torch.randn(17)
        m = L.meta_from_tensor(base[1::2])
        self.assertEqual((m["stride"], m["capacity"], m["numel"], m["elemBytes"]), (2, 16, 8, 4))
        self.assertEqual(L.meta_from_tensor(torch.zeros(5, dtype=torch.float16))["dtype"], "float16")


class LeanVerdictTest(unittest.TestCase):
    """Uses the real Lean checker (kernel-checked verdicts)."""

    def test_verdicts(self):
        def buf(base, cap, stride=1, dt=".f32"):
            return {"base": base, "elemBytes": 4, "stride": stride, "capacity": cap, "dtype": dt}
        good = {"n": 5, "block": 4, "grid": [2], "inputs": [buf(0, 5), buf(100, 5)],
                "output": buf(200, 5)}
        short = dict(good, grid=[1])
        v = L.lean_verdicts({"good": good, "short": short})
        self.assertTrue(v["good"]["accepted"])
        self.assertEqual(v["short"]["failed_obligations"], ["P3 covers"])
        self.assertTrue(all(x["kernel_checked"] for x in v.values()))


if __name__ == "__main__":
    unittest.main()
