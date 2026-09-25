/-
VeriTile.Triton.Launch.StridedUnaryFlat

Flat-memory placement of a checked rank-1 strided unary launch
(`StridedUnary.launch x out`), built from the tensors' own metadata, and the
facts the checked conditions give about it:

* `StridedUnary.flatAlloc` — the input and output regions sit at their data
  pointers' element addresses `base / 4`, each extending over its view's
  strided span `(s0 - 1) * stride + 1` cells.
* `Pre.flat_disjoint` — S7 (4-byte elements), S8 (aligned data pointers) and
  S9 (disjoint byte spans) discharge the bridge's `FlatAlloc.Disjoint`.
* `Pre.span_in_alloc` — S6: each modelled span lies inside its allocation.
* `Pre.addr_bytes` — the cell/byte correspondence: four times the flat cell
  address of logical element `k` is its byte address `base + 4 * offsetOf [k]`.
* `Pre.reads_outside_writes` — no input element's cell is an output element's
  cell (the flat-memory form of read/write separation).

Cells are typed and element-sized (one `MemCell` per `float32` element); the
correspondence to bytes is exactly `Pre.addr_bytes` under S7/S8, nothing more.
-/

import VeriTile.Triton.Launch.StridedUnary
import VeriTile.Triton.Launch.Blocked1DFlat

namespace VeriTile.Triton

namespace StridedUnary

open TensorMeta

/-- Cells spanned by the strided elements of a non-empty rank-1 view. -/
def spanCells (s0 stride : Nat) : Nat := (s0 - 1) * stride + 1

/-- Flat placement from the launch's own metadata: region `in0` at the input
data pointer's element address, region `out0` at the output's, each over its
view's strided span. -/
def flatAlloc (flat in0 out0 : RegionName) (c : Launch) : FlatAlloc :=
  { flat := flat
    regions := [in0, out0]
    base := fun r =>
      if r = out0 then c.output.base / c.output.elemBytes
      else if r = in0 then c.input.base / c.input.elemBytes else 0
    extent := fun r =>
      if r = out0 then spanCells c.s0 c.outStride
      else if r = in0 then spanCells c.s0 c.inStride else 0 }

/-- Regions outside the placement have extent `0` (the bridge's `hcov`). -/
theorem flatAlloc_closed (flat in0 out0 : RegionName) (c : Launch) :
    ∀ r, r ∉ (flatAlloc flat in0 out0 c).regions → (flatAlloc flat in0 out0 c).extent r = 0 := by
  intro r hr
  simp only [flatAlloc, List.mem_cons, List.mem_nil_iff, or_false, not_or] at hr ⊢
  rw [if_neg hr.2, if_neg hr.1]

/-- The logical element `k` of a rank-1 view sits at element offset `k * stride`. -/
theorem offsetOf_rank1 (t : TensorMeta) (ht : t.strides.length = 1) (k : Nat) :
    t.offsetOf [k] = k * t.strides.headD 0 := by
  obtain ⟨s, hs⟩ := List.length_eq_one_iff.1 ht
  simp [offsetOf, hs]

/-- Base and extent of the two placed regions. -/
theorem flatAlloc_in {flat in0 out0 : RegionName} (hne : in0 ≠ out0) (c : Launch) :
    (flatAlloc flat in0 out0 c).base in0 = c.input.base / c.input.elemBytes ∧
      (flatAlloc flat in0 out0 c).extent in0 = spanCells c.s0 c.inStride := by
  simp [flatAlloc, hne]

theorem flatAlloc_out (flat in0 out0 : RegionName) (c : Launch) :
    (flatAlloc flat in0 out0 c).base out0 = c.output.base / c.output.elemBytes ∧
      (flatAlloc flat in0 out0 c).extent out0 = spanCells c.s0 c.outStride := by
  simp [flatAlloc]

namespace Pre

/-- **The checked conditions discharge `FlatAlloc.Disjoint`** (S7, S8, S9). -/
theorem flat_disjoint {x out : TensorMeta} (h : Pre (launch x out))
    (flat in0 out0 : RegionName) (hne : in0 ≠ out0) :
    (flatAlloc flat in0 out0 (launch x out)).Disjoint := by
  obtain ⟨-, ebx, -, ebo⟩ := h.dtype_ok
  obtain ⟨ax, ao⟩ := h.aligned
  have hsd := h.spans_disjoint
  obtain ⟨bi, ei⟩ := flatAlloc_in (flat := flat) hne (launch x out)
  obtain ⟨bo, eo⟩ := flatAlloc_out flat in0 out0 (launch x out)
  unfold SpansDisjoint spanEnd at hsd
  unfold spanCells at ei eo
  rw [ebx] at bi hsd
  rw [ebo] at bo hsd
  generalize (launch x out).input.base = xb at *
  generalize (launch x out).output.base = ob at *
  generalize ((launch x out).s0 - 1) * (launch x out).inStride = a at *
  generalize ((launch x out).s0 - 1) * (launch x out).outStride = b at *
  intro r hr r' hr' hrr
  simp only [flatAlloc, List.mem_cons, List.mem_nil_iff, or_false] at hr hr'
  rcases hr with rfl | rfl <;> rcases hr' with rfl | rfl
  · exact absurd rfl hrr
  · rw [bi, ei, bo, eo]; omega
  · rw [bi, ei, bo, eo]; omega
  · exact absurd rfl hrr

/-- **S6: each modelled span lies inside its tensor's allocation.** -/
theorem span_in_alloc {x out : TensorMeta} (h : Pre (launch x out))
    (flat in0 out0 : RegionName) (hne : in0 ≠ out0) :
    (flatAlloc flat in0 out0 (launch x out)).extent in0 ≤ x.capacity ∧
      (flatAlloc flat in0 out0 (launch x out)).extent out0 ≤ out.capacity := by
  obtain ⟨b1, b2⟩ := h.in_bounds
  rw [(flatAlloc_in (flat := flat) hne _).2, (flatAlloc_out flat in0 out0 _).2]
  unfold spanCells
  exact ⟨b1, b2⟩

/-- **Cell/byte correspondence** (S1, S7, S8): four times the flat cell
address of logical element `k` is that element's byte address. -/
theorem addr_bytes {x out : TensorMeta} (h : Pre (launch x out))
    (flat in0 out0 : RegionName) (hne : in0 ≠ out0) (k : Nat) :
    4 * (flatAlloc flat in0 out0 (launch x out)).addr in0 (k * (launch x out).inStride)
        = x.base + x.elemBytes * x.offsetOf [k] ∧
      4 * (flatAlloc flat in0 out0 (launch x out)).addr out0 (k * (launch x out).outStride)
        = out.base + out.elemBytes * out.offsetOf [k] := by
  obtain ⟨-, ebx, -, ebo⟩ := h.dtype_ok
  obtain ⟨ax, ao⟩ := h.aligned
  obtain ⟨-, -, sx, so⟩ := h.rank1
  have hi : (launch x out).input = x := rfl
  have ho : (launch x out).output = out := rfl
  have e1 : (launch x out).inStride = x.strides.headD 0 := rfl
  have e2 : (launch x out).outStride = out.strides.headD 0 := rfl
  rw [hi] at ebx ax sx
  rw [ho] at ebo ao so
  unfold FlatAlloc.addr
  rw [(flatAlloc_in (flat := flat) hne _).1, (flatAlloc_out flat in0 out0 _).1, hi, ho, ebx, ebo,
    offsetOf_rank1 x sx, offsetOf_rank1 out so, e1, e2]
  generalize k * x.strides.headD 0 = a
  generalize k * out.strides.headD 0 = b
  constructor <;> omega

/-- **Read/write separation in flat memory**: no input logical element shares
a cell with an output logical element. -/
theorem reads_outside_writes {x out : TensorMeta} (h : Pre (launch x out))
    (flat in0 out0 : RegionName) (hne : in0 ≠ out0) :
    ∀ k j, k < (launch x out).s0 → j < (launch x out).s0 →
      (flatAlloc flat in0 out0 (launch x out)).addr in0 (k * (launch x out).inStride)
        ≠ (flatAlloc flat in0 out0 (launch x out)).addr out0 (j * (launch x out).outStride) := by
  intro k j hk hj
  have hd := h.flat_disjoint flat in0 out0 hne in0 (by simp [flatAlloc]) out0 (by simp [flatAlloc]) hne
  have hk' := Nat.mul_le_mul_right (launch x out).inStride (show k ≤ (launch x out).s0 - 1 by omega)
  have hj' := Nat.mul_le_mul_right (launch x out).outStride (show j ≤ (launch x out).s0 - 1 by omega)
  obtain ⟨bi, ei⟩ := flatAlloc_in (flat := flat) hne (launch x out)
  obtain ⟨bo, eo⟩ := flatAlloc_out flat in0 out0 (launch x out)
  unfold FlatAlloc.addr
  rw [bi, ei, bo, eo] at hd
  rw [bi, bo]
  unfold spanCells at hd
  generalize (launch x out).input.base / (launch x out).input.elemBytes = p at *
  generalize (launch x out).output.base / (launch x out).output.elemBytes = q at *
  generalize k * (launch x out).inStride = a at *
  generalize j * (launch x out).outStride = b at *
  generalize ((launch x out).s0 - 1) * (launch x out).inStride = A at *
  generalize ((launch x out).s0 - 1) * (launch x out).outStride = B at *
  omega

end Pre

end StridedUnary

end VeriTile.Triton
