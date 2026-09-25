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

    def test_grid_with_other_block_constant_rejected(self):
        src = mutate("num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE",
                     "B2 = 8\n    num_blocks = (n_elements + B2 - 1) // B2")
        with self.assertRaises(L.Unsupported):
            L.parse_launch(src, "add_wrapper", self.k)

    def test_grid_with_other_count_rejected(self):
        src = mutate("num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE",
                     "m = y.numel()\n    num_blocks = (m + BLOCK_SIZE - 1) // BLOCK_SIZE")
        with self.assertRaises(L.Unsupported):
            L.parse_launch(src, "add_wrapper", self.k)

    def test_grid_with_same_count_alias_accepted(self):
        src = mutate("num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE",
                     "m = x.numel()\n    num_blocks = (m + BLOCK_SIZE - 1) // BLOCK_SIZE")
        self.assertEqual(L.parse_launch(src, "add_wrapper", self.k).grid_kind, "cdiv")

    def test_load_order_must_match_signature(self):
        src = mutate("    x = tl.load(in_ptr0 + offsets, mask=mask)\n    y = tl.load(in_ptr1 + offsets, mask=mask)",
                     "    y = tl.load(in_ptr1 + offsets, mask=mask)\n    x = tl.load(in_ptr0 + offsets, mask=mask)")
        with self.assertRaises(L.Unsupported):
            L.parse_kernel(src, "add_kernel")

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


class StridedReluRecognition(unittest.TestCase):
    RELU = REPO / "bench/tritonbench_g/relu_strided_buffer/relu_strided_buffer.py"

    def setUp(self):
        import launch_invoke
        self.I = launch_invoke
        self.src = self.RELU.read_text()

    def test_pinned_wrapper_recognized(self):
        r = self.I.recognize_relu(self.src)
        self.assertEqual(r["binding"]["in0_stride0"], "in0_strides[0]")
        self.assertEqual(r["binding"]["out0_stride0"], "out0_strides[0]")

    def test_wrong_stride_argument_rejected(self):
        src = self.src.replace("in0_strides[0], # stride for in0", "out0_strides[0], # stride for in0")
        with self.assertRaises(L.Unsupported):
            self.I.recognize_relu(src)

    def test_stride_from_other_tensor_rejected(self):
        src = self.src.replace("in0_strides = in0.stride()", "in0_strides = out0.stride()")
        with self.assertRaises(L.Unsupported):
            self.I.recognize_relu(src)

    def test_task_space_rebinding_rejected(self):
        src = self.src.replace("shape[0], # task indexing space", "num_tasks, # task indexing space")
        with self.assertRaises(L.Unsupported):
            self.I.recognize_relu(src)

    def test_grid_cap_change_rejected(self):
        src = self.src.replace("num_ctas = min(65536, num_tiles)", "num_ctas = min(1024, num_tiles)")
        with self.assertRaises(L.Unsupported):
            self.I.recognize_relu(src)

    def test_tile_heuristic_change_rejected(self):
        src = self.src.replace("tile_sizes = heuristics_for_tile_size(512, *shape)",
                               "tile_sizes = heuristics_for_tile_size(1024, *shape)")
        with self.assertRaises(L.Unsupported):
            self.I.recognize_relu(src)

    def test_kernel_body_change_rejected(self):
        src = self.src.replace("return tl.where(x > 0, x, 0)", "return tl.where(x >= 0, x, 0)")
        with self.assertRaises(L.Unsupported):
            self.I.recognize_relu(src)

    def test_store_cast_fix_recognized_only_under_its_own_label(self):
        fixed, rep = self.I.relu_source("store_cast_fix")
        self.assertEqual(rep, {self.I.RELU_STORE_CAST_FIX[0]: self.I.RELU_STORE_CAST_FIX[1]})
        self.assertEqual(self.I.recognize_relu(fixed, "store_cast_fix")["kernel_text"], "store_cast_fix")
        with self.assertRaises(L.Unsupported):  # the fixed text is not the pinned text
            self.I.recognize_relu(fixed)
        with self.assertRaises(L.Unsupported):  # the pinned text is not the fixed text
            self.I.recognize_relu(self.src, "store_cast_fix")

    def test_other_casts_not_normalized(self):
        for old, new in [(self.I.RELU_STORE_CAST_FIX[0], "out0.to(tl.float16)"),
                         (self.I.RELU_STORE_CAST_FIX[0], "out0"),
                         (".to(in0_ptr.type.element_ty)", ".to(tl.float16)")]:
            src = self.src.replace(old, new)
            for label in ("pinned", "store_cast_fix"):
                with self.assertRaises(L.Unsupported):
                    self.I.recognize_relu(src, label)

    def test_unknown_kernel_text_label_rejected(self):
        with self.assertRaises(L.Unsupported):
            self.I.relu_source("anything_else")

    def test_demo_summary_fails_on_missing_extra_or_duplicate_case(self):
        ok = [{"case": c, "as_expected": True} for c in self.I.DEMO_CASES]
        self.assertTrue(self.I.summarize(ok)["all_as_expected"])
        self.assertFalse(self.I.summarize(ok[1:])["all_as_expected"])
        self.assertEqual(self.I.summarize(ok[1:])["missing_cases"], [self.I.DEMO_CASES[0]])
        self.assertFalse(self.I.summarize(ok + [{"case": "new", "as_expected": True}])["all_as_expected"])
        self.assertFalse(self.I.summarize(ok + ok[:1])["all_as_expected"])

    def test_demo_summary_requires_the_pinned_failure_row(self):
        ok = [{"case": c, "as_expected": True} for c in self.I.DEMO_CASES
              if c != "relu_pinned_text_known_failure"]
        self.assertFalse(self.I.summarize(ok)["all_as_expected"])

    def test_add_kernel_body_change_rejected_at_binding(self):
        src = (REPO / "bench/tritonbench_g/add_example/add_example.py").read_text().replace(
            "output = x + y", "output = x - y")
        with self.assertRaises(L.Unsupported):
            self.I.add_example(src=src)

    def test_mirror_nonempty_and_negative_stride(self):
        T = self.I.TensorMeta
        x = T(4096, 4, (0,), (1,), 0, "f32")
        self.assertIn("S3 nonempty", [n for n, ok in self.I.strided_obligations(x, x) if not ok])
        y = T(4096, 4, (10,), (-1,), 10, "f32")
        self.assertEqual(self.I.strided_obligations(y, T(65536, 4, (10,), (1,), 10, "f32")),
                         [("S5 pos_strides (negative stride)", False)])


class WrapperObligations(unittest.TestCase):
    def test_empty_like_variant_recognized(self):
        src = (REPO / "bench/tritonbench_g/add_example/improvement/add_example_empty_like.py").read_text()
        k = L.parse_kernel(src, "add_kernel")
        self.assertEqual(L.lean_body_statements(ADD_LEAN, "add_kernel"), k.body_statements)
        self.assertEqual(L.parse_launch(src, "add_wrapper", k).out_alloc, ("out", "x"))

    def test_block64_variant_recognized(self):
        src = (REPO / "bench/tritonbench_g/add_example/improvement/add_example_block64.py").read_text()
        k = L.parse_kernel(src, "add_kernel")
        self.assertEqual(L.lean_body_statements(ADD_LEAN, "add_kernel"), k.body_statements)
        w = L.parse_launch(src, "add_wrapper", k)
        self.assertEqual((w.block, w.out_alloc), (64, ("out", "x")))

    def test_w1_detects_partial_output(self):
        src = (REPO / "bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py").read_text()
        k = L.parse_kernel(src, "_add_kernel")
        w = L.parse_launch(src, "custom_add", k)
        metas = L.build_tensors({"a": {"numel": 32, "dtype": "float32", "shape": [4, 8]},
                                 "b": {"numel": 32, "dtype": "float32", "shape": [4, 8]}}, w)
        cfg = L.make_config(w, k, metas)
        self.assertEqual((cfg["n"], metas["c"]["numel"]), (4, 32))


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



class EvidenceFreshness(unittest.TestCase):
    """Rerun a check when its relevant inputs change, and only then."""

    def setUp(self):
        import launch_local_check as LC
        self.LC = LC
        self.h = LC.input_hashes()

    def rec(self, name, **changes):
        deps = self.LC.evidence_deps(name)
        ih = {k: v for k, v in self.h.items() if deps is None or k in deps}
        ih.update(changes)
        return {"input_hashes": ih}

    def test_current_when_dependencies_match(self):
        for n in ("gpu_wrapper.json", "invoke_differential_strided.json", "official_comparator.json"):
            self.assertTrue(self.LC.evidence_freshness(n, self.rec(n), self.h)[0], n)

    def test_change_outside_subset_does_not_stale(self):
        h = dict(self.h)
        h["bench/tritonbench_g/relu_strided_buffer/ReluStridedBuffer.lean"] = "0" * 64
        self.assertTrue(self.LC.evidence_freshness("gpu_wrapper.json", self.rec("gpu_wrapper.json"), h)[0])

    def test_change_inside_subset_stales(self):
        h = dict(self.h)
        h["scripts/launch_invoke.py"] = "0" * 64
        ok, stale, _ = self.LC.evidence_freshness("gpu_wrapper.json", self.rec("gpu_wrapper.json"), h)
        self.assertFalse(ok)
        self.assertEqual(stale, ["scripts/launch_invoke.py"])

    def test_lean_closure_dependency_stales_differential(self):
        h = dict(self.h)
        h["VeriTile/Triton/Launch/Blocked1DConfig.lean"] = "0" * 64
        self.assertFalse(self.LC.evidence_freshness(
            "invoke_differential_strided.json", self.rec("invoke_differential_strided.json"), h)[0])

    def test_unhashed_dependency_is_not_current(self):
        r = self.rec("gpu_wrapper.json")
        del r["input_hashes"]["scripts/launch_gpu.py"]
        ok, _, unhashed = self.LC.evidence_freshness("gpu_wrapper.json", r, self.h)
        self.assertFalse(ok)
        self.assertEqual(unhashed, ["scripts/launch_gpu.py"])

    def test_whole_set_evidence_never_narrowed(self):
        self.assertIsNone(self.LC.evidence_deps("official_comparator.json"))
        h = dict(self.h)
        h["bench/tritonbench_g/relu_strided_buffer/ReluStridedBuffer.lean"] = "0" * 64
        self.assertFalse(self.LC.evidence_freshness(
            "official_comparator.json", self.rec("official_comparator.json"), h)[0])


class FusedBinding(unittest.TestCase):
    """The fused candidate's source binding (the parts that need no Triton)."""

    def setUp(self):
        try:
            import launch_fused
        except ImportError as e:  # torch absent on this host
            self.skipTest(str(e))
        self.F = launch_fused
        self.src = launch_fused.FUSED_PY.read_text()
        self.lean = L.lean_body_statements(launch_fused.FUSED_LEAN.read_text(), "add_relu_kernel")

    def test_body_equals_lean_transcription(self):
        self.assertEqual(self.F.kernel_statements(self.src, "add_relu_kernel"), self.lean)

    def test_mutants_differ_from_lean(self):
        for old, new in [("tl.where(z > 0, z, 0)", "tl.where(z >= 0, z, 0)"),
                         ("tl.where(z > 0, z, 0)", "tl.maximum(z, 0)"), ("z = x + y", "z = x - y")]:
            self.assertNotEqual(self.F.kernel_statements(self.src.replace(old, new), "add_relu_kernel"), self.lean)

    def test_projection_recognized_by_frozen_recognizer(self):
        proj = self.F.fused_projection(self.src)
        k = L.parse_kernel(proj, "add_relu_kernel")
        launch = L.parse_launch(proj, "add_relu_wrapper", k)
        self.assertEqual((launch.block, launch.grid_kind, launch.n_source), (64, "cdiv", "x"))

    def test_projection_requires_the_fused_lines(self):
        with self.assertRaises(L.Unsupported):
            self.F.fused_projection(self.src.replace("    z = x + y\n", ""))

if __name__ == "__main__":
    unittest.main()
