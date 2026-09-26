import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# Dynamic-width QKV repack by concatenation — fast-division kernel (stage 11)

Workload: PyTorch issue #189940's `nested_cat_add` (bf16): for groups g ∈ {1, 2},
`v_g = va_g + vb_g`, then `out = cat([cat([q1, k1, v1], -1), cat([q2, k2, v2], -1)], -1)`
for contiguous `[n, w]` inputs, with the concatenation width dynamic.

The candidate kernel (`cat_repack.py`, text identical to `scripts/launch_cat.py`'s
`repack_fastdiv`) is Inductor's flattened `pointwise_cat` indexing with the row
recovered by **fast division** by the runtime width `W` (host-computed constants,
`VeriTile.Triton.FastDiv`) instead of a per-element `//` and `%` by a runtime value.

## Transcription table (Python → Lean), each entry justified

| Python | Lean | why |
|---|---|---|
| `xu = x.to(tl.uint32)`; `umulhi(xu, magic.to(tl.uint32))`; `(... + xu) >> shift.to(tl.uint32)` | `(((x * magic) >> 32) + x) >> shift` over ℕ | `FastDiv.bitvec_quotient`: the 32-bit computation equals the ℕ one for `x < 2^31` |
| `.to(tl.int32)`, `.to(tl.float32)`, `.to(out.dtype.element_ty)` | erased | index casts are value-preserving below `2^31`; float casts are the exact-ℝ boundary |
| `W = 2 * (wq + wk + wv)`, `G = wq + wk + wv`, `wq + wk` | one antiquotation each | host constants; the DSL types a compound antiquotation as ℕ |
| `cc - wq - wk` | `cc - (wq + wk)` (one antiquotation) | equal in ℕ and ℤ; the host sum is one constant |

The row line is fully parenthesized, `(((x * magic) >> 32) + x) >> shift`, so it
does not depend on operator precedence (in Python `+` binds tighter than `>>`).
`scripts/test_launch_check.py` applies exactly these rewrites to the Python kernel
and requires the result to equal this transcription statement by statement.

## What is proved

* `segCat` is `torch.cat` along the last dimension of contiguous 2-D inputs, and
  `nestedCat_eq_flat`: the nested concatenation (with materialized intermediates
  `g1`, `g2`) equals the flat six-segment concatenation.
* `repack_lane_correct`: every active lane `x < n * W` of the kernel writes
  `catSpec (x / W) (x % W)` — the flat concatenation of the inputs as the launch
  state holds them — and inactive lanes write nothing.
* `cat_repack_launch_correctness`: the whole `(cdiv(n * W, B),)` launch writes that
  value at every output element `x < n * W` and changes no other cell.

## Modeling boundary

Exact ℝ: the bf16 add is an exact sum here. On hardware the kernel rounds once
(fp32 sum, one bf16 store), like eager; bitwise equality with eager is
established by tests (14 timed shapes and 5 edge cases), not proved.
-/

namespace VeriTile.Bench.Optimizations.CatRepack

open VeriTile.Triton VeriTile.Examples

/-- `torch.cat(..., dim=-1)` of contiguous `[n, w]` inputs, as a function of
(row, column): a list of (width, row-major source) segments. -/
def segCat : List (Nat × (Nat → ℝ)) → Nat → Nat → ℝ
  | [], _, _ => 0
  | (w, f) :: rest, r, c => if c < w then f (r * w + c) else segCat rest r (c - w)

/-- Total width of a segment list. -/
def segWidth (l : List (Nat × (Nat → ℝ))) : Nat := (l.map Prod.fst).sum

/-- A 2-D `[n, w]` tensor given by `(row, col) ↦ value`, flattened row-major. -/
def flat (w : Nat) (g : Nat → Nat → ℝ) : Nat → ℝ := fun i => g (i / w) (i % w)

/-- The six-segment flat concatenation the kernel computes. -/
def catSpec (wq wk wv : Nat) (Q1 K1 VA1 VB1 Q2 K2 VA2 VB2 : Nat → ℝ) : Nat → Nat → ℝ :=
  segCat [(wq, Q1), (wk, K1), (wv, fun i => VA1 i + VB1 i),
          (wq, Q2), (wk, K2), (wv, fun i => VA2 i + VB2 i)]

/-- The issue's program: two inner concatenations materialized as `[n, G]`
tensors, then an outer concatenation of the two. -/
def nestedCat (wq wk wv : Nat) (Q1 K1 VA1 VB1 Q2 K2 VA2 VB2 : Nat → ℝ) : Nat → Nat → ℝ :=
  let G := wq + wk + wv
  segCat [(G, flat G (segCat [(wq, Q1), (wk, K1), (wv, fun i => VA1 i + VB1 i)])),
          (G, flat G (segCat [(wq, Q2), (wk, K2), (wv, fun i => VA2 i + VB2 i)]))]

/-- **The nested program equals the flat concatenation** on every in-range
(row, column). -/
theorem nestedCat_eq_flat (wq wk wv : Nat) (Q1 K1 VA1 VB1 Q2 K2 VA2 VB2 : Nat → ℝ)
    (r c : Nat) (hc : c < 2 * (wq + wk + wv)) :
    nestedCat wq wk wv Q1 K1 VA1 VB1 Q2 K2 VA2 VB2 r c
      = catSpec wq wk wv Q1 K1 VA1 VB1 Q2 K2 VA2 VB2 r c := by
  sorry

/-- Transcription of `cat_repack.py`'s `repack_fastdiv` (see the table above). -/
def repack_fastdiv
    (q1 k1 va1 vb1 q2 k2 va2 vb2 out : RegionName)
    (wq wk wv total magic shift BLOCK : Nat) :
    ComputeKernel := triton {
  x = tl.program_id(0) * $(BLOCK) + tl.arange(0, $(BLOCK))
  xm = x < $(total)
  W = $(2 * (wq + wk + wv))
  row = (((x * $(magic)) >> $(32)) + x) >> $(shift)
  c = x - row * W
  G = $(wq + wk + wv)
  g2 = c >= G
  cc = tl.where(g2, c - G, c)
  mq = xm & (cc < $(wq))
  mk = xm & (cc >= $(wq)) & (cc < $(wq + wk))
  mv = xm & (cc >= $(wq + wk))
  oq = row * $(wq) + cc
  ok = row * $(wk) + (cc - $(wq))
  ov = row * $(wv) + (cc - $(wq + wk))
  q_a = tl.load(q1 + oq, mask=mq & ~g2, other=0.0)
  q_b = tl.load(q2 + oq, mask=mq & g2, other=0.0)
  k_a = tl.load(k1 + ok, mask=mk & ~g2, other=0.0)
  k_b = tl.load(k2 + ok, mask=mk & g2, other=0.0)
  v_a = (tl.load(va1 + ov, mask=mv & ~g2, other=0.0) + tl.load(vb1 + ov, mask=mv & ~g2, other=0.0))
  v_b = (tl.load(va2 + ov, mask=mv & g2, other=0.0) + tl.load(vb2 + ov, mask=mv & g2, other=0.0))
  val = tl.where(mq, tl.where(g2, q_b, q_a), tl.where(mk, tl.where(g2, k_b, k_a), tl.where(g2, v_b, v_a)))
  tl.store(out + x, val, mask=xm)
}

/-- **Per-lane readback.** With the host constants the wrapper passes
(`magic = FastDiv.magic W`, `shift = FastDiv.shiftFor W`, `total = n * W`) and
`n * W < 2^31`: every active lane `x = pid * BLOCK + i < total` of the output holds
the flat concatenation at `(x / W, x % W)` of the inputs as held in `s`, and an
inactive lane keeps its initial value. -/
theorem repack_lane_correct
    (q1 k1 va1 vb1 q2 k2 va2 vb2 out : RegionName) (n wq wk wv BLOCK : Nat)
    (hW : 0 < wq + wk + wv) (hfit : n * (2 * (wq + wk + wv)) < 2 ^ 31)
    (s : BlockState) (i : Fin BLOCK) :
    let W := 2 * (wq + wk + wv)
    let x := s.pid * BLOCK + i.val
    observeAt (exec (repack_fastdiv q1 k1 va1 vb1 q2 k2 va2 vb2 out wq wk wv (n * W)
        (FastDiv.magic W) (FastDiv.shiftFor W) BLOCK) s) out BLOCK s.pid i
      = some (if x < n * W then
          catSpec wq wk wv (s.readMem q1) (s.readMem k1) (s.readMem va1) (s.readMem vb1)
            (s.readMem q2) (s.readMem k2) (s.readMem va2) (s.readMem vb2) (x / W) (x % W)
        else s.readMem out x) := by
  sorry

/-- **Whole launch.** For a grid of `g` programs covering `n * W` elements
(`n * W ≤ g * BLOCK`, `0 < BLOCK`), an output region distinct from the eight
inputs, and the host constants of the wrapper: the launch writes the flat
concatenation (equivalently the nested program, `nestedCat_eq_flat`) at every
output element `x < n * W`, and every other cell is unchanged. -/
specification cat_repack_launch_correctness
    (q1 k1 va1 vb1 q2 k2 va2 vb2 out : RegionName) (n wq wk wv BLOCK g : Nat)
    (hW : 0 < wq + wk + wv) (hfit : n * (2 * (wq + wk + wv)) < 2 ^ 31)
    (hB : 0 < BLOCK) (hcov : n * (2 * (wq + wk + wv)) ≤ g * BLOCK)
    (hout : out ∉ [q1, k1, va1, vb1, q2, k2, va2, vb2]) (s : BlockState) :
    ∃ sF, Nonempty (Kernel.GridLaunchedOrdinary
        ((repack_fastdiv q1 k1 va1 vb1 q2 k2 va2 vb2 out wq wk wv (n * (2 * (wq + wk + wv)))
          (FastDiv.magic (2 * (wq + wk + wv))) (FastDiv.shiftFor (2 * (wq + wk + wv))) BLOCK).toAlgKernel)
        (Blocked1D.line g) s sF) ∧
      (∀ x, x < n * (2 * (wq + wk + wv)) →
        sF.readMem out x = nestedCat wq wk wv (s.readMem q1) (s.readMem k1) (s.readMem va1)
          (s.readMem vb1) (s.readMem q2) (s.readMem k2) (s.readMem va2) (s.readMem vb2)
          (x / (2 * (wq + wk + wv))) (x % (2 * (wq + wk + wv)))) ∧
      (∀ r o, ¬ (r = out ∧ o < n * (2 * (wq + wk + wv))) → sF.mem r o = s.mem r o) := by
  sorry

end VeriTile.Bench.Optimizations.CatRepack
