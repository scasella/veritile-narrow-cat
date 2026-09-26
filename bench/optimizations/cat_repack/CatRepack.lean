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
  have hG : ∀ c', c' < wq + wk + wv → ∀ (A B C : Nat → ℝ) (rest : List (Nat × (Nat → ℝ))),
      segCat ((wq, A) :: (wk, B) :: (wv, C) :: rest) r c' = segCat [(wq, A), (wk, B), (wv, C)] r c' := by
    intro c' hc' A B C rest
    show segCat ((wq, A) :: (wk, B) :: (wv, C) :: rest) r c' = segCat ((wq, A) :: (wk, B) :: (wv, C) :: []) r c'
    unfold segCat
    split_ifs <;> first | rfl | (unfold segCat; split_ifs <;> first | rfl | (unfold segCat; split_ifs <;> first | rfl | omega))
  have hflat : ∀ (f : Nat → Nat → ℝ) (c' : Nat), c' < wq + wk + wv →
      flat (wq + wk + wv) f (r * (wq + wk + wv) + c') = f r c' := by
    intro f c' hc'
    unfold flat
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hc', Nat.zero_add,
      Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc']
  unfold nestedCat catSpec
  by_cases h1 : c < wq + wk + wv
  · simp only [segCat, if_pos h1]
    rw [hflat _ c h1]
    have := hG c h1 Q1 K1 (fun i => VA1 i + VB1 i) [(wq, Q2), (wk, K2), (wv, fun i => VA2 i + VB2 i)]
    simp only [segCat] at this ⊢
    exact this.symm
  · have h2 : c - (wq + wk + wv) < wq + wk + wv := by omega
    simp only [segCat, if_neg h1, if_pos h2]
    rw [hflat _ _ h2]
    rw [if_neg (show ¬ c < wq by omega), if_neg (show ¬ c - wq < wk by omega),
      if_neg (show ¬ c - wq - wk < wv by omega), show c - (wq + wk + wv) = c - wq - wk - wv by omega]

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

/-- One assignment step of `stepStmts`. -/
private theorem step_assign {d : TileDType} {sh : TileShape} {nm : RegName} {e : Op d sh}
    {rest : List Stmt} {s : BlockState} {T : Tile d sh} (h : evalOp e s = some T) :
    stepStmts (.assign d sh nm e :: rest) s = stepStmts rest (s.setReg nm d sh T) := by
  simp only [stepStmts, stepStmt_assign_eq_some h]

set_option maxHeartbeats 4000000 in
/-- Frame half: the run changes no cell other than the active output lanes. -/
private theorem repack_frame
    (q1 k1 va1 vb1 q2 k2 va2 vb2 out : RegionName) (n wq wk wv BLOCK : Nat)
    (s s1 : BlockState)
    (hExec : exec (repack_fastdiv q1 k1 va1 vb1 q2 k2 va2 vb2 out wq wk wv (n * (2 * (wq + wk + wv)))
        (FastDiv.magic (2 * (wq + wk + wv))) (FastDiv.shiftFor (2 * (wq + wk + wv))) BLOCK) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : r ≠ out ∨ ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < n * (2 * (wq + wk + wv)) →
      o ≠ s.pid * BLOCK + j.val) :
    s1.mem r o = s.mem r o := by
  set W := 2 * (wq + wk + wv) with hWdef
  simp only [exec, repack_fastdiv, ComputeKernel.toAlgKernel_mk, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?_alg, bind, Except.bind, pure, Except.pure] at hExec
  have ok1 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarL (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil (VeriTile.Triton.Op.programId 0) (VeriTile.Triton.Op.constNat BLOCK)) (VeriTile.Triton.Op.arange BLOCK)) s).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true]
  set T1 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarL (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil (VeriTile.Triton.Op.programId 0) (VeriTile.Triton.Op.constNat BLOCK)) (VeriTile.Triton.Op.arange BLOCK)) s).get ok1 with hT1
  have h1 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarL (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil (VeriTile.Triton.Op.programId 0) (VeriTile.Triton.Op.constNat BLOCK)) (VeriTile.Triton.Op.arange BLOCK)) s = some T1 := (Option.some_get ok1).symm
  rw [step_assign h1] at hExec
  set S1 := s.setReg "x" TileDType.nat [BLOCK] T1 with hS1
  have ok2 : (evalOp (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (n * W))) S1).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1]
  set T2 := (evalOp (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (n * W))) S1).get ok2 with hT2
  have h2 : evalOp (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (n * W))) S1 = some T2 := (Option.some_get ok2).symm
  rw [step_assign h2] at hExec
  set S2 := S1.setReg "xm" TileDType.bool [BLOCK] T2 with hS2
  have ok3 : (evalOp (VeriTile.Triton.Op.constNat (2 * (wq + wk + wv))) S2).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2]
  set T3 := (evalOp (VeriTile.Triton.Op.constNat (2 * (wq + wk + wv))) S2).get ok3 with hT3
  have h3 : evalOp (VeriTile.Triton.Op.constNat (2 * (wq + wk + wv))) S2 = some T3 := (Option.some_get ok3).symm
  rw [step_assign h3] at hExec
  set S3 := S2.setReg "W" TileDType.nat [] T3 with hS3
  have ok4 : (evalOp (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (FastDiv.magic W))) (VeriTile.Triton.Op.constNat 32)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x")) (VeriTile.Triton.Op.constNat (FastDiv.shiftFor W))) S3).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3]
  set T4 := (evalOp (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (FastDiv.magic W))) (VeriTile.Triton.Op.constNat 32)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x")) (VeriTile.Triton.Op.constNat (FastDiv.shiftFor W))) S3).get ok4 with hT4
  have h4 : evalOp (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (FastDiv.magic W))) (VeriTile.Triton.Op.constNat 32)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x")) (VeriTile.Triton.Op.constNat (FastDiv.shiftFor W))) S3 = some T4 := (Option.some_get ok4).symm
  rw [step_assign h4] at hExec
  set S4 := S3.setReg "row" TileDType.nat [BLOCK] T4 with hS4
  have ok5 : (evalOp (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "W"))) S4).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4]
  set T5 := (evalOp (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "W"))) S4).get ok5 with hT5
  have h5 : evalOp (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "W"))) S4 = some T5 := (Option.some_get ok5).symm
  rw [step_assign h5] at hExec
  set S5 := S4.setReg "c" TileDType.nat [BLOCK] T5 with hS5
  have ok6 : (evalOp (VeriTile.Triton.Op.constNat (wq + wk + wv)) S5).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5]
  set T6 := (evalOp (VeriTile.Triton.Op.constNat (wq + wk + wv)) S5).get ok6 with hT6
  have h6 : evalOp (VeriTile.Triton.Op.constNat (wq + wk + wv)) S5 = some T6 := (Option.some_get ok6).symm
  rw [step_assign h6] at hExec
  set S6 := S5.setReg "G" TileDType.nat [] T6 with hS6
  have ok7 : (evalOp (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) S6).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6]
  set T7 := (evalOp (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) S6).get ok7 with hT7
  have h7 : evalOp (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) S6 = some T7 := (Option.some_get ok7).symm
  rw [step_assign h7] at hExec
  set S7 := S6.setReg "g2" TileDType.bool [BLOCK] T7 with hS7
  have ok8 : (evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c")) S7).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7]
  set T8 := (evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c")) S7).get ok8 with hT8
  have h8 : evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c")) S7 = some T8 := (Option.some_get ok8).symm
  rw [step_assign h8] at hExec
  set S8 := S7.setReg "cc" TileDType.nat [BLOCK] T8 with hS8
  have ok9 : (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S8).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8]
  set T9 := (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S8).get ok9 with hT9
  have h9 : evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S8 = some T9 := (Option.some_get ok9).symm
  rw [step_assign h9] at hExec
  set S9 := S8.setReg "mq" TileDType.bool [BLOCK] T9 with hS9
  have ok10 : (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S9).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9]
  set T10 := (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S9).get ok10 with hT10
  have h10 : evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S9 = some T10 := (Option.some_get ok10).symm
  rw [step_assign h10] at hExec
  set S10 := S9.setReg "mk" TileDType.bool [BLOCK] T10 with hS10
  have ok11 : (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S10).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10]
  set T11 := (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S10).get ok11 with hT11
  have h11 : evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S10 = some T11 := (Option.some_get ok11).symm
  rw [step_assign h11] at hExec
  set S11 := S10.setReg "mv" TileDType.bool [BLOCK] T11 with hS11
  have ok12 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wq)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc")) S11).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11]
  set T12 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wq)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc")) S11).get ok12 with hT12
  have h12 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wq)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc")) S11 = some T12 := (Option.some_get ok12).symm
  rw [step_assign h12] at hExec
  set S12 := S11.setReg "oq" TileDType.nat [BLOCK] T12 with hS12
  have ok13 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wk)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S12).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12]
  set T13 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wk)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S12).get ok13 with hT13
  have h13 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wk)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S12 = some T13 := (Option.some_get ok13).symm
  rw [step_assign h13] at hExec
  set S13 := S12.setReg "ok" TileDType.nat [BLOCK] T13 with hS13
  have ok14 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wv)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S13).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13]
  set T14 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wv)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S13).get ok14 with hT14
  have h14 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wv)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S13 = some T14 := (Option.some_get ok14).symm
  rw [step_assign h14] at hExec
  set S14 := S13.setReg "ov" TileDType.nat [BLOCK] T14 with hS14
  have ok15 : (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S14).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14]
  set T15 := (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S14).get ok15 with hT15
  have h15 : evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S14 = some T15 := (Option.some_get ok15).symm
  rw [step_assign h15] at hExec
  set S15 := S14.setReg "q_a" TileDType.real [BLOCK] T15 with hS15
  have ok16 : (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S15).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15]
  set T16 := (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S15).get ok16 with hT16
  have h16 : evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S15 = some T16 := (Option.some_get ok16).symm
  rw [step_assign h16] at hExec
  set S16 := S15.setReg "q_b" TileDType.real [BLOCK] T16 with hS16
  have ok17 : (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S16).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16]
  set T17 := (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S16).get ok17 with hT17
  have h17 : evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S16 = some T17 := (Option.some_get ok17).symm
  rw [step_assign h17] at hExec
  set S17 := S16.setReg "k_a" TileDType.real [BLOCK] T17 with hS17
  have ok18 : (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S17).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17]
  set T18 := (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S17).get ok18 with hT18
  have h18 : evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S17 = some T18 := (Option.some_get ok18).symm
  rw [step_assign h18] at hExec
  set S18 := S17.setReg "k_b" TileDType.real [BLOCK] T18 with hS18
  have ok19 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S18).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18]
  set T19 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S18).get ok19 with hT19
  have h19 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S18 = some T19 := (Option.some_get ok19).symm
  rw [step_assign h19] at hExec
  set S19 := S18.setReg "v_a" TileDType.real [BLOCK] T19 with hS19
  have ok20 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S19).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18, hS19]
  set T20 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S19).get ok20 with hT20
  have h20 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S19 = some T20 := (Option.some_get ok20).symm
  rw [step_assign h20] at hExec
  set S20 := S19.setReg "v_b" TileDType.real [BLOCK] T20 with hS20
  have ok21 : (evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_a")))) S20).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18, hS19, hS20]
  set T21 := (evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_a")))) S20).get ok21 with hT21
  have h21 : evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_a")))) S20 = some T21 := (Option.some_get ok21).symm
  rw [step_assign h21] at hExec
  set S21 := S20.setReg "val" TileDType.real [BLOCK] T21 with hS21
  simp +decide only [stepStmts, stepStmt, evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18, hS19, hS20, hS21] at hExec
  simp only [Option.map_some, BlockState.writeMemTyped_real, Region.cast_self, Option.some.injEq] at hExec
  have hbase : (((((((((((((((((((((s.setReg "x" TileDType.nat [BLOCK] T1).setReg "xm" TileDType.bool [BLOCK] T2).setReg "W" TileDType.nat [] T3).setReg "row" TileDType.nat [BLOCK] T4).setReg "c" TileDType.nat [BLOCK] T5).setReg "G" TileDType.nat [] T6).setReg "g2" TileDType.bool [BLOCK] T7).setReg "cc" TileDType.nat [BLOCK] T8).setReg "mq" TileDType.bool [BLOCK] T9).setReg "mk" TileDType.bool [BLOCK] T10).setReg "mv" TileDType.bool [BLOCK] T11).setReg "oq" TileDType.nat [BLOCK] T12).setReg "ok" TileDType.nat [BLOCK] T13).setReg "ov" TileDType.nat [BLOCK] T14).setReg "q_a" TileDType.real [BLOCK] T15).setReg "q_b" TileDType.real [BLOCK] T16).setReg "k_a" TileDType.real [BLOCK] T17).setReg "k_b" TileDType.real [BLOCK] T18).setReg "v_a" TileDType.real [BLOCK] T19).setReg "v_b" TileDType.real [BLOCK] T20).setReg "val" TileDType.real [BLOCK] T21) = S21 := by simp only [hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18, hS19, hS20, hS21]
  rw [hbase] at hExec
  subst hExec
  have d1 : ∀ idx, T1.data idx = s.pids 0 * BLOCK + idx.1.val := by
    intro idx; simp [hT1, evalOp, Tile.bop_data, NumericDType.add, NumericDType.mul]
  rw [foldl_store_preserve_cell (region := out) (fun k => T1.data k)
    (fun k => FloatDType.real.storeValue (T21.data k)) (fun k => T2.data k = «true») r o _ S21 ?_]
  · rfl
  · intro k _ hk ⟨hr, ho⟩
    have L2 := congrArg (fun t => t.data k) hT2
    simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1] at L2
    have hk' : T1.data k < n * W := by
      have : ComparableDType.nat.lt (T1.data k) (n * W) = «true» := by rw [← L2]; exact hk
      simpa using this
    rcases hmiss with h | h
    · exact h hr.symm
    · exact h k.1 (by rw [d1] at hk'; exact hk') (by rw [← ho]; exact d1 k)

set_option maxHeartbeats 8000000 in
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
  intro W x
  simp only [observeAt, exec, repack_fastdiv, ComputeKernel.toAlgKernel_mk, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?_alg, bind, Except.bind, pure, Except.pure]
  have ok1 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarL (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil (VeriTile.Triton.Op.programId 0) (VeriTile.Triton.Op.constNat BLOCK)) (VeriTile.Triton.Op.arange BLOCK)) s).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true]
  set T1 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarL (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil (VeriTile.Triton.Op.programId 0) (VeriTile.Triton.Op.constNat BLOCK)) (VeriTile.Triton.Op.arange BLOCK)) s).get ok1 with hT1
  have h1 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarL (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil (VeriTile.Triton.Op.programId 0) (VeriTile.Triton.Op.constNat BLOCK)) (VeriTile.Triton.Op.arange BLOCK)) s = some T1 := (Option.some_get ok1).symm
  rw [step_assign h1]
  set S1 := s.setReg "x" TileDType.nat [BLOCK] T1 with hS1
  have ok2 : (evalOp (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (n * W))) S1).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1]
  set T2 := (evalOp (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (n * W))) S1).get ok2 with hT2
  have h2 : evalOp (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (n * W))) S1 = some T2 := (Option.some_get ok2).symm
  rw [step_assign h2]
  set S2 := S1.setReg "xm" TileDType.bool [BLOCK] T2 with hS2
  have ok3 : (evalOp (VeriTile.Triton.Op.constNat (2 * (wq + wk + wv))) S2).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2]
  set T3 := (evalOp (VeriTile.Triton.Op.constNat (2 * (wq + wk + wv))) S2).get ok3 with hT3
  have h3 : evalOp (VeriTile.Triton.Op.constNat (2 * (wq + wk + wv))) S2 = some T3 := (Option.some_get ok3).symm
  rw [step_assign h3]
  set S3 := S2.setReg "W" TileDType.nat [] T3 with hS3
  have ok4 : (evalOp (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (FastDiv.magic W))) (VeriTile.Triton.Op.constNat 32)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x")) (VeriTile.Triton.Op.constNat (FastDiv.shiftFor W))) S3).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3]
  set T4 := (evalOp (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (FastDiv.magic W))) (VeriTile.Triton.Op.constNat 32)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x")) (VeriTile.Triton.Op.constNat (FastDiv.shiftFor W))) S3).get ok4 with hT4
  have h4 : evalOp (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.shiftRight VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.constNat (FastDiv.magic W))) (VeriTile.Triton.Op.constNat 32)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x")) (VeriTile.Triton.Op.constNat (FastDiv.shiftFor W))) S3 = some T4 := (Option.some_get ok4).symm
  rw [step_assign h4]
  set S4 := S3.setReg "row" TileDType.nat [BLOCK] T4 with hS4
  have ok5 : (evalOp (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "W"))) S4).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4]
  set T5 := (evalOp (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "W"))) S4).get ok5 with hT5
  have h5 : evalOp (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "x") (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "W"))) S4 = some T5 := (Option.some_get ok5).symm
  rw [step_assign h5]
  set S5 := S4.setReg "c" TileDType.nat [BLOCK] T5 with hS5
  have ok6 : (evalOp (VeriTile.Triton.Op.constNat (wq + wk + wv)) S5).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5]
  set T6 := (evalOp (VeriTile.Triton.Op.constNat (wq + wk + wv)) S5).get ok6 with hT6
  have h6 : evalOp (VeriTile.Triton.Op.constNat (wq + wk + wv)) S5 = some T6 := (Option.some_get ok6).symm
  rw [step_assign h6]
  set S6 := S5.setReg "G" TileDType.nat [] T6 with hS6
  have ok7 : (evalOp (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) S6).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6]
  set T7 := (evalOp (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) S6).get ok7 with hT7
  have h7 : evalOp (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) S6 = some T7 := (Option.some_get ok7).symm
  rw [step_assign h7]
  set S7 := S6.setReg "g2" TileDType.bool [BLOCK] T7 with hS7
  have ok8 : (evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c")) S7).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7]
  set T8 := (evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c")) S7).get ok8 with hT8
  have h8 : evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [] "G")) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "c")) S7 = some T8 := (Option.some_get ok8).symm
  rw [step_assign h8]
  set S8 := S7.setReg "cc" TileDType.nat [BLOCK] T8 with hS8
  have ok9 : (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S8).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8]
  set T9 := (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S8).get ok9 with hT9
  have h9 : evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S8 = some T9 := (Option.some_get ok9).symm
  rw [step_assign h9]
  set S9 := S8.setReg "mq" TileDType.bool [BLOCK] T9 with hS9
  have ok10 : (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S9).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9]
  set T10 := (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S9).get ok10 with hT10
  have h10 : evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) (VeriTile.Triton.Op.lt VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S9 = some T10 := (Option.some_get ok10).symm
  rw [step_assign h10]
  set S10 := S9.setReg "mk" TileDType.bool [BLOCK] T10 with hS10
  have ok11 : (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S10).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10]
  set T11 := (evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S10).get ok11 with hT11
  have h11 : evalOp (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "xm") (VeriTile.Triton.Op.ge VeriTile.Triton.ComparableDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S10 = some T11 := (Option.some_get ok11).symm
  rw [step_assign h11]
  set S11 := S10.setReg "mv" TileDType.bool [BLOCK] T11 with hS11
  have ok12 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wq)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc")) S11).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11]
  set T12 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wq)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc")) S11).get ok12 with hT12
  have h12 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wq)) (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc")) S11 = some T12 := (Option.some_get ok12).symm
  rw [step_assign h12]
  set S12 := S11.setReg "oq" TileDType.nat [BLOCK] T12 with hS12
  have ok13 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wk)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S12).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12]
  set T13 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wk)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S12).get ok13 with hT13
  have h13 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wk)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat wq))) S12 = some T13 := (Option.some_get ok13).symm
  rw [step_assign h13]
  set S13 := S12.setReg "ok" TileDType.nat [BLOCK] T13 with hS13
  have ok14 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wv)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S13).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13]
  set T14 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wv)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S13).get ok14 with hT14
  have h14 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.mul VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "row") (VeriTile.Triton.Op.constNat wv)) (VeriTile.Triton.Op.sub VeriTile.Triton.NumericDType.nat VeriTile.Triton.Broadcast.scalarR (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "cc") (VeriTile.Triton.Op.constNat (wq + wk)))) S13 = some T14 := (Option.some_get ok14).symm
  rw [step_assign h14]
  set S14 := S13.setReg "ov" TileDType.nat [BLOCK] T14 with hS14
  have ok15 : (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S14).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14]
  set T15 := (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S14).get ok15 with hT15
  have h15 : evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S14 = some T15 := (Option.some_get ok15).symm
  rw [step_assign h15]
  set S15 := S14.setReg "q_a" TileDType.real [BLOCK] T15 with hS15
  have ok16 : (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S15).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15]
  set T16 := (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S15).get ok16 with hT16
  have h16 : evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region q2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "oq")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S15 = some T16 := (Option.some_get ok16).symm
  rw [step_assign h16]
  set S16 := S15.setReg "q_b" TileDType.real [BLOCK] T16 with hS16
  have ok17 : (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S16).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16]
  set T17 := (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S16).get ok17 with hT17
  have h17 : evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S16 = some T17 := (Option.some_get ok17).symm
  rw [step_assign h17]
  set S17 := S16.setReg "k_a" TileDType.real [BLOCK] T17 with hS17
  have ok18 : (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S17).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17]
  set T18 := (evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S17).get ok18 with hT18
  have h18 : evalOp (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region k2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ok")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) S17 = some T18 := (Option.some_get ok18).symm
  rw [step_assign h18]
  set S18 := S17.setReg "k_b" TileDType.real [BLOCK] T18 with hS18
  have ok19 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S18).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18]
  set T19 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S18).get ok19 with hT19
  have h19 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb1 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").boolNot) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S18 = some T19 := (Option.some_get ok19).symm
  rw [step_assign h19]
  set S19 := S18.setReg "v_a" TileDType.real [BLOCK] T19 with hS19
  have ok20 : (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S19).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18, hS19]
  set T20 := (evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S19).get ok20 with hT20
  have h20 : evalOp (VeriTile.Triton.Op.add VeriTile.Triton.NumericDType.real VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region va2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK]))) (VeriTile.Triton.Op.load VeriTile.Triton.TileDType.real (VeriTile.Triton.MemAccess.region vb2 (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.nat [BLOCK] "ov")) (VeriTile.Triton.MaskOpt.maskOther (VeriTile.Triton.Op.boolAnd VeriTile.Triton.Broadcast.nil.consSame (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mv") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2")) ((VeriTile.Triton.Op.const 0.0).broadcast [BLOCK])))) S19 = some T20 := (Option.some_get ok20).symm
  rw [step_assign h20]
  set S20 := S19.setReg "v_b" TileDType.real [BLOCK] T20 with hS20
  have ok21 : (evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_a")))) S20).isSome := by simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.isSome_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18, hS19, hS20]
  set T21 := (evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_a")))) S20).get ok21 with hT21
  have h21 : evalOp ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mq").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "q_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "mk").where ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "k_a")) ((VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.bool [BLOCK] "g2").where (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_b") (VeriTile.Triton.Op.ref VeriTile.Triton.TileDType.real [BLOCK] "v_a")))) S20 = some T21 := (Option.some_get ok21).symm
  rw [step_assign h21]
  set S21 := S20.setReg "val" TileDType.real [BLOCK] T21 with hS21
  simp +decide only [stepStmts, stepStmt, evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18, hS19, hS20, hS21]
  simp only [Option.map_some, BlockState.writeMemTyped_real, Region.cast_self]
  have hbase : (((((((((((((((((((((s.setReg "x" TileDType.nat [BLOCK] T1).setReg "xm" TileDType.bool [BLOCK] T2).setReg "W" TileDType.nat [] T3).setReg "row" TileDType.nat [BLOCK] T4).setReg "c" TileDType.nat [BLOCK] T5).setReg "G" TileDType.nat [] T6).setReg "g2" TileDType.bool [BLOCK] T7).setReg "cc" TileDType.nat [BLOCK] T8).setReg "mq" TileDType.bool [BLOCK] T9).setReg "mk" TileDType.bool [BLOCK] T10).setReg "mv" TileDType.bool [BLOCK] T11).setReg "oq" TileDType.nat [BLOCK] T12).setReg "ok" TileDType.nat [BLOCK] T13).setReg "ov" TileDType.nat [BLOCK] T14).setReg "q_a" TileDType.real [BLOCK] T15).setReg "q_b" TileDType.real [BLOCK] T16).setReg "k_a" TileDType.real [BLOCK] T17).setReg "k_b" TileDType.real [BLOCK] T18).setReg "v_a" TileDType.real [BLOCK] T19).setReg "v_b" TileDType.real [BLOCK] T20).setReg "val" TileDType.real [BLOCK] T21) = S21 := by simp only [hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18, hS19, hS20, hS21]
  rw [hbase]
  have d1 : ∀ idx, T1.data idx = s.pids 0 * BLOCK + idx.1.val := by
    intro idx; simp [hT1, evalOp, Tile.bop_data, NumericDType.add, NumericDType.mul]
  have hinj : Function.Injective (fun k : TileIndex [BLOCK] => T1.data k) := by
    have h := injective_offset_singleton (n := BLOCK) (s.pid * BLOCK)
    intro a b hab; apply h; simpa [d1] using hab
  have hlane : s.pid * BLOCK + i.val = T1.data (i, PUnit.unit) := by rw [d1]
  have hrb := BlockState.scatter_readback_prop_masked_nd (region := out) S21 (fun k => T1.data k)
    (fun k => FloatDType.real.storeValue (T21.data k)) (fun k => T2.data k = «true») hinj (i, PUnit.unit)
  simp only at hrb
  rw [hlane, hrb]
  have hrm : ∀ o, S21.readMem out o = s.readMem out o := by
    intro o
    rw [hS21, BlockState.setReg_readMem, hS20, BlockState.setReg_readMem, hS19, BlockState.setReg_readMem,
      hS18, BlockState.setReg_readMem, hS17, BlockState.setReg_readMem, hS16, BlockState.setReg_readMem,
      hS15, BlockState.setReg_readMem, hS14, BlockState.setReg_readMem, hS13, BlockState.setReg_readMem,
      hS12, BlockState.setReg_readMem, hS11, BlockState.setReg_readMem, hS10, BlockState.setReg_readMem,
      hS9, BlockState.setReg_readMem, hS8, BlockState.setReg_readMem, hS7, BlockState.setReg_readMem,
      hS6, BlockState.setReg_readMem, hS5, BlockState.setReg_readMem, hS4, BlockState.setReg_readMem,
      hS3, BlockState.setReg_readMem, hS2, BlockState.setReg_readMem, hS1, BlockState.setReg_readMem]
  rw [hrm]
  have L2 := congrArg (fun t => t.data (i, PUnit.unit)) hT2
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1] at L2
  have L3 := congrArg (fun t => t.data PUnit.unit) hT3
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2] at L3
  have L4 := congrArg (fun t => t.data (i, PUnit.unit)) hT4
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3] at L4
  have L5 := congrArg (fun t => t.data (i, PUnit.unit)) hT5
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4] at L5
  have L6 := congrArg (fun t => t.data PUnit.unit) hT6
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5] at L6
  have L7 := congrArg (fun t => t.data (i, PUnit.unit)) hT7
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6] at L7
  have L8 := congrArg (fun t => t.data (i, PUnit.unit)) hT8
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7] at L8
  have L9 := congrArg (fun t => t.data (i, PUnit.unit)) hT9
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8] at L9
  have L10 := congrArg (fun t => t.data (i, PUnit.unit)) hT10
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9] at L10
  have L11 := congrArg (fun t => t.data (i, PUnit.unit)) hT11
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10] at L11
  have L12 := congrArg (fun t => t.data (i, PUnit.unit)) hT12
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11] at L12
  have L13 := congrArg (fun t => t.data (i, PUnit.unit)) hT13
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12] at L13
  have L14 := congrArg (fun t => t.data (i, PUnit.unit)) hT14
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13] at L14
  have L15 := congrArg (fun t => t.data (i, PUnit.unit)) hT15
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14] at L15
  have L16 := congrArg (fun t => t.data (i, PUnit.unit)) hT16
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15] at L16
  have L17 := congrArg (fun t => t.data (i, PUnit.unit)) hT17
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16] at L17
  have L18 := congrArg (fun t => t.data (i, PUnit.unit)) hT18
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17] at L18
  have L19 := congrArg (fun t => t.data (i, PUnit.unit)) hT19
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18] at L19
  have L20 := congrArg (fun t => t.data (i, PUnit.unit)) hT20
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18, hS19] at L20
  have L21 := congrArg (fun t => t.data (i, PUnit.unit)) hT21
  simp +decide only [evalOp, Option.bind_eq_bind, Option.bind, Option.pure_def, Option.get_some, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, not_false_eq_true, BlockState.setReg_readMemValue, Tile.bop_data, Tile.cop_data, Broadcast.leftIndex_nil, Broadcast.leftIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.leftIndex_consSame, Broadcast.rightIndex_nil, Broadcast.rightIndex_scalarL, Broadcast.rightIndex_scalarR, Broadcast.rightIndex_consSame, Tile.select_data, Tile.scalar_data, Tile.scalar_data_index, hS1, hS2, hS3, hS4, hS5, hS6, hS7, hS8, hS9, hS10, hS11, hS12, hS13, hS14, hS15, hS16, hS17, hS18, hS19, hS20] at L21
  have hlt : ∀ a b : Nat, ComparableDType.nat.lt a b = decide (a < b) := fun _ _ => rfl
  have hge : ∀ a b : Nat, ComparableDType.nat.ge a b = decide (a ≥ b) := fun _ _ => rfl
  simp only [NumericDType.nat_add, NumericDType.nat_sub, NumericDType.nat_mul, Nat.shiftRight_eq_div_pow,
    hlt, hge, Tile.uop_data, if_true, Region.cast_id, BlockState.readMemValue_real] at L2 L4 L5 L7 L8 L9 L10 L11 L12 L13 L14 L15 L16 L17 L18 L19 L20
  rw [L3] at L5; rw [L6] at L7 L8
  have hxX : x = T1.data (i, PUnit.unit) := hlane
  rw [hxX]
  set X := T1.data (i, PUnit.unit) with hX
  rw [L2]
  by_cases hx : X < n * W
  swap
  · simp [hx]
  have hWpos : 0 < W := by show 0 < 2 * (wq + wk + wv); omega
  have hW31 : W < 2 ^ 31 := by
    have hn : 0 < n := by rcases Nat.eq_zero_or_pos n with h | h <;> [simp [h] at hx; exact h]
    exact lt_of_le_of_lt (Nat.le_mul_of_pos_left W hn) hfit
  have hX31 : X < 2 ^ 31 := lt_trans hx hfit
  have e4 : T4.data (i, PUnit.unit) = X / W := by
    have := FastDiv.magic_quotient hWpos hW31 hX31
    unfold FastDiv.quot at this
    rw [L4]; exact this
  have e5 : T5.data (i, PUnit.unit) = X % W := by
    rw [L5, e4, Nat.mod_def, Nat.mul_comm]
  have hc : X % W < W := Nat.mod_lt _ hWpos
  simp only [L21, L15, L16, L17, L18, L19, L20, L9, L10, L11, L12, L13, L14, L8, L7, e4, e5, hx,
    decide_true, if_true, FloatDType.real_storeValue]
  have e2 : T2.data (i, PUnit.unit) = «true» := by rw [L2]; exact decide_eq_true hx
  simp only [e2, Bool.true_and]
  generalize ((X / W : TileCarrier TileDType.nat) : Nat) = r
  generalize ((X % W : TileCarrier TileDType.nat) : Nat) = c at hc ⊢
  have hcW : c < 2 * (wq + wk + wv) := hc
  clear * - hcW
  revert r c
  intro (r : Nat) (c : Nat) hcW
  replace hcW : c < 2 * (wq + wk + wv) := hcW
  unfold catSpec
  by_cases hg : wq + wk + wv ≤ c <;> by_cases ha : c - (wq + wk + wv) < wq <;>
    by_cases hb : c - (wq + wk + wv) < wq + wk <;> by_cases ha' : c < wq <;> by_cases hb' : c < wq + wk <;>
    simp (disch := omega) [segCat, Nat.sub_sub, if_pos, if_neg, NumericDType.add, hg, ha, hb, ha', hb']
  all_goals (try (exfalso; omega))
  all_goals rw [show wq + wk + wv + (wq + wk) = wq + wk + wv + wq + wk by omega]

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
  set W := 2 * (wq + wk + wv) with hWdef
  have hWpos : 0 < W := by omega
  let pre : BlockState → Prop := fun t =>
    t.readMem q1 = s.readMem q1 ∧ t.readMem k1 = s.readMem k1 ∧ t.readMem va1 = s.readMem va1 ∧
    t.readMem vb1 = s.readMem vb1 ∧ t.readMem q2 = s.readMem q2 ∧ t.readMem k2 = s.readMem k2 ∧
    t.readMem va2 = s.readMem va2 ∧ t.readMem vb2 = s.readMem vb2
  have hrun : Blocked1D.ProgramRuns
      ((repack_fastdiv q1 k1 va1 vb1 q2 k2 va2 vb2 out wq wk wv (n * W)
        (FastDiv.magic W) (FastDiv.shiftFor W) BLOCK).toAlgKernel) out (n * W) BLOCK pre
      (fun x => nestedCat wq wk wv (s.readMem q1) (s.readMem k1) (s.readMem va1)
          (s.readMem vb1) (s.readMem q2) (s.readMem k2) (s.readMem va2) (s.readMem vb2) (x / W) (x % W)) := by
    intro t ht
    obtain ⟨e1, e2, e3, e4, e5, e6, e7, e8⟩ := ht
    have hl := fun i => repack_lane_correct q1 k1 va1 vb1 q2 k2 va2 vb2 out n wq wk wv BLOCK hW hfit t i
    cases hsrc : exec ((repack_fastdiv q1 k1 va1 vb1 q2 k2 va2 vb2 out wq wk wv (n * W)
        (FastDiv.magic W) (FastDiv.shiftFor W) BLOCK).toAlgKernel) t with
    | none =>
        have h0 := hl ⟨0, hB⟩
        simp only [observeAt] at h0
        rw [show exec (repack_fastdiv q1 k1 va1 vb1 q2 k2 va2 vb2 out wq wk wv (n * W)
            (FastDiv.magic W) (FastDiv.shiftFor W) BLOCK) t = exec ((repack_fastdiv q1 k1 va1 vb1 q2 k2
            va2 vb2 out wq wk wv (n * W) (FastDiv.magic W) (FastDiv.shiftFor W) BLOCK).toAlgKernel) t
            from rfl, hsrc] at h0
        simp at h0
    | some f =>
        refine ⟨f, rfl, fun o ho hd => ?_, fun r o hno => ?_⟩
        · have hdm := Nat.div_add_mod' o BLOCK
          rw [hd] at hdm
          have h1 := hl ⟨o % BLOCK, Nat.mod_lt o hB⟩
          simp only [observeAt] at h1
          rw [show exec (repack_fastdiv q1 k1 va1 vb1 q2 k2 va2 vb2 out wq wk wv (n * W)
              (FastDiv.magic W) (FastDiv.shiftFor W) BLOCK) t = exec ((repack_fastdiv q1 k1 va1 vb1 q2 k2
              va2 vb2 out wq wk wv (n * W) (FastDiv.magic W) (FastDiv.shiftFor W) BLOCK).toAlgKernel) t
              from rfl, hsrc, Option.map_some, Option.some_inj, hdm] at h1
          rw [h1, if_pos ho, e1, e2, e3, e4, e5, e6, e7, e8]
          exact (nestedCat_eq_flat _ _ _ _ _ _ _ _ _ _ _ _ _ (Nat.mod_lt o hWpos)).symm
        · apply repack_frame q1 k1 va1 vb1 q2 k2 va2 vb2 out n wq wk wv BLOCK t f hsrc
          by_cases hr : r = out
          · refine Or.inr fun j hj ho => hno ?_
            rw [Blocked1D.blockWrites_iff hB]
            refine ⟨hr, by have hj' : t.pid * BLOCK + j.val < n * W := hj; omega, ?_⟩
            rw [ho, Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt j.isLt, Nat.zero_add]
          · exact Or.inl hr
  obtain ⟨sF, hL, hval, hframe⟩ := Blocked1D.launch_of_programRuns hB hcov s hrun
    (fun _ => ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩)
  exact ⟨sF, hL, hval, hframe⟩

end VeriTile.Bench.Optimizations.CatRepack
