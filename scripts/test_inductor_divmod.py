#!/usr/bin/env python3
"""Unit tests for the stage-12 guarded rewrite (scripts/inductor_divmod.py). No GPU, no Triton needed.

    python3 scripts/test_inductor_divmod.py
"""
from __future__ import annotations

import random
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import inductor_divmod as R  # noqa: E402

FIXTURE = (REPO / "bench/optimizations/inductor_divmod/fixtures/pointwise_cat_preview.py").read_text()


def branches(src: str) -> tuple[list[str], list[str]]:
    """Split the rewritten kernel body into (variant branch, else branch) lines, dedented once."""
    lines = src.split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("    if ") and ln.endswith(":"))
    j = lines.index("    else:", i)
    end = len(lines)
    while lines[end - 1].strip() == "":
        end -= 1
    return [ln[4:] for ln in lines[i + 1:j]], [ln[4:] for ln in lines[j + 1:end]]


def original_body(src: str) -> list[str]:
    lines = src.split("\n")
    d = next(k for k, ln in enumerate(lines) if ln.startswith("def "))
    end = len(lines)
    while lines[end - 1].strip() == "":
        end -= 1
    return lines[d + 4:end]


class Rewrite(unittest.TestCase):
    def test_baseline_identity(self):
        s, rec = R.rewrite(FIXTURE, "B0")
        self.assertEqual(s, FIXTURE)
        self.assertFalse(rec["rewritten"])

    def test_variants_rewrite_fixture(self):
        for v in ("N", "F", "NF"):
            s, rec = R.rewrite(FIXTURE, v)
            self.assertTrue(rec["rewritten"], (v, rec))
            self.assertNotEqual(rec["before"], rec["after"])

    def test_else_branch_is_original_body(self):
        body = original_body(FIXTURE)
        for v in ("N", "F", "NF"):
            s, _ = R.rewrite(FIXTURE, v)
            _, orig = branches(s)
            self.assertEqual(orig, body, v)

    def test_prologue_and_signature_unchanged(self):
        head = FIXTURE.split("    xmask = xindex < xnumel")[0]
        for v in ("N", "F", "NF"):
            s, _ = R.rewrite(FIXTURE, v)
            # everything before the kernel body is unchanged except the inserted helper
            self.assertEqual(s.replace(R.HELPER, "\n").split("    xmask = xindex < xnumel")[0].replace("\n\n\n", "\n"),
                             head.replace("\n\n\n", "\n"))

    def test_fast_branch_has_no_runtime_division(self):
        for v in ("F", "NF"):
            s, _ = R.rewrite(FIXTURE, v)
            var, _ = branches(s)
            text = "\n".join(var)
            self.assertNotRegex(text, r"xindex\s*(%|//)")
            self.assertEqual(text.count("_vt_fast_divmod("), 1)

    def test_narrowed_branch_uses_only_int32_scalars(self):
        for v in ("N", "NF"):
            s, rec = R.rewrite(FIXTURE, v)
            var, _ = branches(s)
            body = [ln for ln in var if not re.match(r"\s*ks\d+_v = ks\d+\.to\(tl\.int32\)$", ln)]
            self.assertFalse(any(re.search(r"\bks\d+\b(?!_v)", ln) for ln in body), v)
            self.assertEqual(rec["narrowed"], ["ks0", "ks1", "ks2", "ks3", "ks4", "ks5", "ks6"])

    def test_f_keeps_i64_scalars(self):
        s, _ = R.rewrite(FIXTURE, "F")
        var, _ = branches(s)
        self.assertFalse(any("ks1_v" in ln for ln in var))
        self.assertTrue(any("x0_v = (x0_r).to(tl.int64)" in ln for ln in var))

    def test_guards(self):
        _, rn = R.rewrite(FIXTURE, "N")
        _, rf = R.rewrite(FIXTURE, "F")
        self.assertEqual(rf["guard"], "((ks0 > 0) & (ks0 < 2147483648))")
        self.assertEqual(rn["guard"].count("< 2147483648"), 7)

    def test_passthrough_on_mismatch(self):
        cases = {
            "not pointwise": FIXTURE.replace("@triton_heuristics.pointwise", "@triton_heuristics.reduction"),
            "xnumel i64": FIXTURE.replace("'xnumel': 'i32'", "'xnumel': 'i64'"),
            "prologue": FIXTURE.replace("xmask = xindex < xnumel", "xmask = xindex <= xnumel"),
            "two mods": FIXTURE.replace("    x2 = xindex\n", "    x2 = xindex\n    x9 = (xindex % ks0)\n"),
            "use between": FIXTURE.replace("    x1 = xindex // ks0\n", "    x7 = x0 + 1\n    x1 = xindex // ks0\n"),
        }
        for name, src in cases.items():
            self.assertNotEqual(src, FIXTURE, name)
            for v in ("F", "NF") if name in ("two mods", "use between") else ("N", "F", "NF"):
                out, rec = R.rewrite(src, v)
                self.assertEqual(out, src, (name, v))
                self.assertFalse(rec["rewritten"], (name, v))


def helper_model(x: int, d: int) -> tuple[int, int]:
    """Python model of _vt_fast_divmod with the kernel's integer widths."""
    s = sum(1 for j in range(32) if d > (1 << j))
    m = ((1 << 32) * ((1 << s) - d)) // d + 1
    assert 0 <= (1 << 32) * ((1 << s) - d) < 2 ** 63  # int64 product does not overflow
    assert 0 < m < 2 ** 32                              # m fits the uint32 cast
    xu = x & 0xFFFFFFFF
    hi = (xu * m) >> 32
    q = ((hi + xu) & 0xFFFFFFFF) >> s
    return q, x - q * d


class Helper(unittest.TestCase):
    def test_matches_stage11_host_constants(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cat_repack", REPO / "bench/optimizations/cat_repack/cat_repack.py")
        try:
            cat = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cat)
        except ImportError:
            self.skipTest("triton not installed")
        for d in [1, 2, 3, 7, 256, 5120, 12288, 2 ** 30 + 1, 2 ** 31 - 1]:
            m, s = cat.magic_for(d)
            self.assertEqual(sum(1 for j in range(32) if d > (1 << j)), s)
            self.assertEqual(((1 << 32) * ((1 << s) - d)) // d + 1, m)

    def test_divmod_edges_and_random(self):
        rng = random.Random(12)
        ds = [1, 2, 3, 5, 255, 256, 257, 5120, 12288, 2 ** 16, 2 ** 30, 2 ** 30 + 1, 2 ** 31 - 2, 2 ** 31 - 1]
        ds += [rng.randrange(1, 2 ** 31) for _ in range(300)]
        xs = [0, 1, 2 ** 31 - 1, 2 ** 31 - 2, 2 ** 30] + [rng.randrange(2 ** 31) for _ in range(300)]
        for d in ds:
            for x in xs + [d - 1, d, d + 1, 2 * d - 1] if d < 2 ** 30 else xs:
                if 0 <= x < 2 ** 31:
                    self.assertEqual(helper_model(x, d), divmod(x, d), (x, d))

    def test_helper_text_matches_model(self):
        h = R.HELPER
        self.assertIn("for j in tl.static_range(32):", h)
        self.assertIn("s += (d64 > (1 << j)).to(tl.int64)", h)
        self.assertIn("m = ((one << 32) * ((one << s) - d64)) // d64 + 1", h)
        self.assertIn("q = ((tl.umulhi(xu, m.to(tl.uint32)) + xu) >> s.to(tl.uint32)).to(tl.int32)", h)
        self.assertIn("r = x.to(tl.int32) - q * d.to(tl.int32)", h)


if __name__ == "__main__":
    unittest.main(verbosity=1)
