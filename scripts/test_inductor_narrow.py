#!/usr/bin/env python3
"""Stage 13 tests: the extracted IR of Inductor's dynamic-cat kernel under B0 (i64 size scalars) vs N1 (i32).

Pure Python; no GPU, no Triton.

    python3 scripts/test_inductor_narrow.py

Hypotheses H, the same ones frozen in NarrowCat.lean and checked by the codegen-time eligibility test:
  H1  ks1..ks6 >= 0
  H2  ks0 = ks1 + ks2 + ks3 + ks4 + ks5 + ks6
  H3  xnumel = n * ks0 for some n >= 0
  H4  xnumel <= 2^31 - 1
Lanes: 0 <= xindex < 2^31 for every lane of the grid (grid lemma). The lane is active iff xindex < xnumel.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import inductor_ir as IR  # noqa: E402

SRC = (REPO / "bench/optimizations/inductor_divmod/fixtures/gpu_emitted_B0.py").read_text()
IRK = IR.extract(SRC)
M = 2 ** 31


def ksv_of(w):  # w = (ks1..ks6)
    return {0: sum(w), **{i + 1: x for i, x in enumerate(w)}}


def edge_lanes(w, n):
    ks0 = sum(w)
    k1, k2, k3, k4, k5, k6 = w
    cuts = [0, k1, k1 + k3, k1 + k2 + k3, k1 + k2 + k3 + k4, k1 + k2 + k3 + k4 + k5, ks0]
    x0s = sorted({c + d for c in cuts for d in (-1, 0, 1) if 0 <= c + d < ks0})
    rows = sorted({0, 1, n // 2, n - 2, n - 1} & set(range(n)))
    return [r * ks0 + c for r in rows for c in x0s]


def agree(w, n, xi):
    ksv = ksv_of(w)
    xnumel = n * ksv[0]
    return IR.observe(IRK, "B0", xi, ksv, xnumel) == IR.observe(IRK, "N1", xi, ksv, xnumel)


class Extraction(unittest.TestCase):
    def test_shape_of_ir(self):
        self.assertEqual(len(IRK["loads"]), 8)
        self.assertEqual(IRK["store"][1], ("xindex",))
        self.assertEqual(IRK["store"][3], ("xmask",))
        self.assertEqual(set(IRK["dead_kinds"].values()), {"bool"})  # dead code: comparisons only
        # every load's mask is <selector> & xmask
        for ld in IRK["loads"]:
            self.assertEqual(ld[3][0], "and")
            self.assertEqual(ld[3][2], ("xmask",))

    def test_lean_block_is_extractor_output(self):
        lean = (REPO / "bench/optimizations/inductor_narrow/NarrowCat.lean").read_text()
        a, b = lean.index(IR.BEGIN), lean.index(IR.END) + len(IR.END)
        self.assertEqual(lean[a:b], IR.lean_block(SRC))

    def test_rejects_unknown_constructs(self):
        bad = SRC.replace("tmp32 = tmp30 + tmp31", "tmp32 = tmp30 * tmp31")
        with self.assertRaises(IR.Unsupported):
            IR.extract(bad)


class Agreement(unittest.TestCase):
    """B0 and N1 make identical observations on every lane when H holds."""

    def check(self, w, n):
        for xi in edge_lanes(w, n):
            self.assertTrue(agree(w, n, xi), (w, n, xi))

    def test_issue_shapes(self):
        for w, n in [((2048, 256, 256, 2048, 256, 256), 4096), ((1000, 136, 120, 1000, 120, 136), 12345),
                     ((5120, 640, 640, 5120, 640, 640), 12345)]:
            self.check(w, n)

    def test_xnumel_at_limit(self):
        self.check((2 ** 31 - 7, 1, 1, 1, 1, 3), 1)          # n = 1, ks0 = 2^31 - 1
        self.check((1, 1, 1, 1, 1, 1), (M - 1) // 6)          # ks0 small, n large
        w = (65536, 0, 0, 0, 0, 0)
        self.check(w, (M - 1) // 65536)                       # a product ks1 * x1 near 2^31

    def test_zero_width_segments(self):
        self.check((5, 0, 3, 0, 0, 7), 11)
        self.check((0, 0, 0, 0, 0, 9), 4)

    def test_random(self):
        rng = random.Random(13)
        for _ in range(120):
            w = tuple(rng.choice([0, 1, 2, 3, rng.randrange(1, 5000), rng.randrange(1, 2 ** 20)]) for _ in range(6))
            ks0 = sum(w)
            if ks0 == 0:
                continue
            n = rng.randrange(1, max(2, (M - 1) // ks0 + 1))
            if n * ks0 > M - 1:
                continue
            for xi in edge_lanes(w, n)[:40] + [rng.randrange(n * ks0) for _ in range(10)]:
                self.assertTrue(agree(w, n, xi), (w, n, xi))

    def test_inactive_lanes_do_nothing(self):
        w, n = (2048, 256, 256, 2048, 256, 256), 4096
        ksv = ksv_of(w)
        xnumel = n * ksv[0]
        for xi in (xnumel, xnumel + 5, M - 1):
            for mode in ("B0", "N1"):
                o = IR.observe(IRK, mode, xi, ksv, xnumel)
                self.assertFalse(o["store_mask"])
                self.assertFalse(any(m for _, m, _ in o["loads"]))


class Necessity(unittest.TestCase):
    """Each hypothesis matters: dropping it admits lanes where B0 and N1 differ."""

    def test_without_H2_sum_can_overflow(self):
        # ks0 small but ks1 + ks2 + ks3 overflows int32 (H2 violated): the first-half selector differs
        ksv = {0: 8, 1: 2 ** 30, 2: 2 ** 30, 3: 2 ** 30, 4: 0, 5: 0, 6: 0}
        xnumel = 8
        self.assertTrue(any(IR.observe(IRK, "B0", xi, ksv, xnumel) != IR.observe(IRK, "N1", xi, ksv, xnumel)
                            for xi in range(8)))

    def test_H4_is_B0s_own_precondition(self):
        # xnumel is an int32 kernel argument in B0 as emitted. Beyond 2^31 - 1 the launcher wraps it, and the
        # kernel's xmask no longer means "xindex < xnumel" in EITHER mode. H4 is Inductor's guard for B0 itself,
        # not a condition that separates B0 from N1.
        w = (2 ** 31, 1, 1, 1, 1, 1)
        ksv = ksv_of(w)
        xnumel = sum(w)
        for mode in ("B0", "N1"):
            self.assertFalse(IR.eval_bool(("xmask",), mode, 7, ksv, xnumel))   # 7 < xnumel, yet masked off

    def test_scalar_bounds_alone_are_not_enough(self):
        # every ks_i < 2^31 (what an 'each symbol fits' assumption gives) but H2 fails -> disagreement
        ksv = {0: 3, 1: 2 ** 31 - 1, 2: 2 ** 31 - 1, 3: 2 ** 31 - 1, 4: 1, 5: 1, 6: 1}
        self.assertTrue(any(IR.observe(IRK, "B0", xi, ksv, 3) != IR.observe(IRK, "N1", xi, ksv, 3) for xi in range(3)))


class Grid(unittest.TestCase):
    def test_xindex_never_wraps(self):
        # xoffset = pid * XBLOCK with XBLOCK a power of two <= 2^31 and pid < cdiv(xnumel, XBLOCK):
        # the last lane is cdiv(xnumel, XBLOCK) * XBLOCK - 1 <= 2^31 - 1 whenever xnumel <= 2^31 - 1.
        for xb in [2 ** k for k in range(0, 13)]:
            for xnumel in (1, xb, M - 1, M - xb, M - xb + 1):
                last = -(-xnumel // xb) * xb - 1
                self.assertLess(last, M)


if __name__ == "__main__":
    unittest.main(verbosity=1)
