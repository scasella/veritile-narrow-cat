#!/usr/bin/env python3
"""Tests for scripts/narrow_rule.py (the public rule, RULE.md).

C1 tests are pure Python. The C2 tests need torch and run only where it is importable with a Triton install: the
`veritile-interp:1` image, using CPU codegen with the backend hash stubbed. Elsewhere they are skipped.

    python3 scripts/test_narrow_rule.py
    docker run --rm -v "$PWD":/w veritile-interp:1 bash -c 'pip install -q numpy >/dev/null; cd /w && python3 scripts/test_narrow_rule.py'
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import narrow_rule as R  # noqa: E402

SRC = R.PROVED_SRC


def mutate(old, new, src=SRC):
    assert old in src, old
    return src.replace(old, new, 1)


class C1(unittest.TestCase):
    def test_accepts_proved_kernel(self):
        self.assertTrue(R.c1_structural(SRC)["ok"])

    def test_rejects(self):
        cases = {
            "extra use of a size scalar": mutate("    tmp32 = tmp30 + tmp31", "    tmp32 = tmp30 + tmp31\n    tmp99 = ks1 * ks2 < ks3"),
            "changed dead code": mutate("    tmp2 = tmp0 >= tmp1", "    tmp2 = tmp0 < tmp1"),
            "changed index expression": mutate("in_ptr0 + (ks1*x1 + (x0))", "in_ptr0 + (ks1*x1 + (x0) + ks2)"),
            "changed cast": mutate("tmp41 = (ks0).to(tl.int32)", "tmp41 = (ks0).to(tl.int64)"),
            "payload dtype": mutate("'in_ptr0': '*bf16'", "'in_ptr0': '*fp32'"),
            "xnumel i64": mutate("'xnumel': 'i32'", "'xnumel': 'i64'"),
            "reduction heuristic": mutate("@triton_heuristics.pointwise(", "@triton_heuristics.reduction("),
            "2-D grid": mutate("'grid_type': 'Grid1D'", "'grid_type': 'Grid2D'"),
            "unknown construct": mutate("    tmp32 = tmp30 + tmp31", "    tmp32 = tl.math.exp(tmp30)"),
        }
        for name, src in cases.items():
            with self.subTest(name):
                self.assertFalse(R.c1_structural(src)["ok"], name)

    def test_apply_only_changes_ks_signature(self):
        rec = {"eligible": True}
        out = R.apply(SRC, rec)
        diff = [(a, b) for a, b in zip(SRC.splitlines(), out.splitlines()) if a != b]
        self.assertEqual(len(diff), 1)                      # the triton_meta line only
        self.assertEqual(R.signature(out), {**R.signature(SRC), **{k: "i32" for k in R.KS}})
        self.assertEqual(R.apply(SRC, {"eligible": False}), SRC)


def _torch_ok():
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401
        import torch._inductor.config  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_torch_ok(), "needs torch + triton (veritile-interp image)")
class C2(unittest.TestCase):
    """Real Inductor state, CPU codegen (force_pointwise_cat mirrors the CUDA lowering; see gen_preview.py)."""

    def run_program(self, fn, ins, inject=None):
        import torch
        import torch._dynamo as dynamo
        import torch._inductor.codegen.triton as ct
        import torch._inductor.config as ic
        import torch.utils._triton as ut
        ic.cpu_backend = "triton"
        ic.force_pointwise_cat = True
        ut.triton_hash_with_backend = lambda: "stub"
        ct.triton_hash_with_backend = lambda: "stub"
        dynamo.reset()
        orig = ct.TritonScheduling.define_kernel
        recs = []

        def hook(self_, src, node_schedule, kernel):
            if inject:
                inject()
            recs.append(R.check(src, kernel))
            return orig(self_, src, node_schedule, kernel)
        ct.TritonScheduling.define_kernel = hook
        try:
            for t in ins:
                dynamo.mark_dynamic(t, t.dim() - 1)
            try:
                torch.compile(fn, dynamic=True)(*ins)
            except Exception:  # noqa: BLE001 - no CPU Triton backend: compile fails after codegen
                pass
        finally:
            ct.TritonScheduling.define_kernel = orig
        return recs

    @staticmethod
    def nested(q1, k1, v1a, v1b, q2, k2, v2a, v2b):
        import torch
        v1 = v1a + v1b
        v2 = v2a + v2b
        return torch.cat([torch.cat([q1, k1, v1], -1), torch.cat([q2, k2, v2], -1)], -1)

    def inputs(self, widths):
        import torch
        return [torch.randn(64, w).bfloat16() for w in widths]

    def test_positive(self):
        recs = self.run_program(self.nested, self.inputs((2048, 256, 256, 256, 2048, 256, 256, 256)))
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0]["eligible"], recs[0])
        self.assertEqual(recs[0]["C2"]["H4_guard"], "installed")

    def test_other_program_rejected(self):
        import torch

        def three(q, k, va, vb):
            return torch.cat([q, k, va + vb], -1)
        recs = self.run_program(three, self.inputs((2048, 256, 256, 256)))
        self.assertTrue(recs and not any(r["eligible"] for r in recs))

    def test_range_lower_bound_zero_rejected(self):
        """Fault injection: pretend a width symbol may be 0. H1' must fail, and so must eligibility."""
        from torch._inductor.virtualized import V
        from torch.utils._sympy.value_ranges import ValueRanges

        def inject():
            se = V.graph.sizevars.shape_env
            for s, rng in list(se.var_to_range.items()):
                se.var_to_range[s] = ValueRanges(0, rng.upper)
        recs = self.run_program(self.nested, self.inputs((2048, 256, 256, 256, 2048, 256, 256, 256)), inject)
        self.assertTrue(recs)
        self.assertFalse(recs[0]["C2"]["H1'"])
        self.assertFalse(recs[0]["eligible"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
