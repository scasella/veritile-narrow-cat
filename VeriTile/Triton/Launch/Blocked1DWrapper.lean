/-
VeriTile.Triton.Launch.Blocked1DWrapper

Caller-level (wrapper) contract for two-input elementwise wrappers built on a
1-D blocked masked kernel, e.g. `add_example.add_wrapper`:

    out = torch.zeros_like(x); n = x.numel(); grid = ((n + B - 1) // B,)
    kernel[grid](x, y, out, n, B); return out

`Blocked1DLaunch.Pre` constrains one launch. The caller, however, receives a
whole tensor `out`. This module adds the tensor-level metadata the caller
holds (`TensorMeta`: data pointer, shape, strides, capacity, dtype), computes
the launch the wrapper derives from it (`Elementwise2.launch`), and states the
wrapper preconditions (`Elementwise2.Pre`): the supported API (equal shapes,
contiguous tensors), two layout conditions the flat-memory correspondence
needs (element-aligned data pointers; inputs identical or disjoint), and the
launch preconditions P1–P10 of the derived launch. From these, the two
wrapper obligations are *derived*, not assumed:

* W1 `output_covered` — the launch covers every element of the returned
  tensor (`n = out.numel`);
* W2 `inputs_cover` — every input tensor holds the `n` elements read
  (`n ≤ t.numel`).

Deliberately core-only (no Mathlib) so that concrete configurations can be
decided by the Lean kernel.
-/

import VeriTile.Triton.Launch.Blocked1DConfig

namespace VeriTile.Triton

/-- Host metadata of one tensor argument, as the caller holds it: `base` is
the byte address of element 0 (`data_ptr()`), `strides` are element strides
(one per dimension), and `capacity` is the number of elements addressable
from `base` inside the tensor's allocation. -/
structure TensorMeta where
  base : Nat
  elemBytes : Nat
  shape : List Nat
  strides : List Nat
  capacity : Nat
  dtype : ElemDType
  deriving DecidableEq, Repr

namespace TensorMeta

/-- Number of elements (`numel()`). -/
def numel (t : TensorMeta) : Nat := t.shape.foldr (· * ·) 1

/-- Row-major (C-contiguous) strides of a shape. -/
def rowMajor : List Nat → List Nat
  | [] => []
  | _ :: rest => rest.foldr (· * ·) 1 :: rowMajor rest

/-- `is_contiguous()` for the row-major memory format: every dimension of size
greater than one carries its row-major stride (size-0/1 dimensions are free,
as in torch). Conservative for empty tensors with unusual strides. -/
def Contiguous (t : TensorMeta) : Prop :=
  t.strides.length = t.shape.length ∧
    ∀ i, i < t.shape.length → 1 < t.shape.getD i 0 →
      t.strides.getD i 0 = (rowMajor t.shape).getD i 0

def contiguousB (t : TensorMeta) : Bool :=
  t.strides.length == t.shape.length &&
    (List.range t.shape.length).all fun i =>
      !(decide (1 < t.shape.getD i 0)) || t.strides.getD i 0 == (rowMajor t.shape).getD i 0

/-- Element offset (from `base`) of the logical element at multi-index `idx`:
`Σ idx[k] * strides[k]`. -/
def offsetOf (t : TensorMeta) (idx : List Nat) : Nat :=
  (idx.zip t.strides).foldr (fun p acc => p.1 * p.2 + acc) 0

/-- `idx` is an in-bounds multi-index of `shape`. -/
def InShape (shape idx : List Nat) : Prop :=
  idx.length = shape.length ∧ ∀ k, k < shape.length → idx.getD k 0 < shape.getD k 0

/-- `base` is a multiple of the element size (element-addressable). -/
def Aligned (t : TensorMeta) : Prop := t.base % t.elemBytes = 0

/-- The flat 1-D buffer view the kernel addresses (`ptr + offsets`): unit
stride exactly when the tensor is contiguous. -/
def toBuf (t : TensorMeta) : BufMeta :=
  { base := t.base, elemBytes := t.elemBytes,
    stride := if t.contiguousB then 1 else 0,
    capacity := t.capacity, dtype := t.dtype }

end TensorMeta

namespace Elementwise2

open TensorMeta

/-- The launch a two-input elementwise wrapper derives from its tensors with
constexpr block `B`: `n = x.numel()`, grid `((n + B - 1) // B,)`, kernel
arguments `(x, y, out, n, B)`. -/
def launch (B : Nat) (x y out : TensorMeta) : Blocked1DLaunch :=
  { n := x.numel, block := B, grid := [(x.numel + B - 1) / B],
    inputs := [x.toBuf, y.toBuf], output := out.toBuf }

/-- WA: supported API — equal shapes (the output is `*_like(x)`). -/
def SameShape (x y out : TensorMeta) : Prop := y.shape = x.shape ∧ out.shape = x.shape

/-- WC: every tensor is contiguous. -/
def AllContiguous (x y out : TensorMeta) : Prop :=
  x.Contiguous ∧ y.Contiguous ∧ out.Contiguous

/-- P12: every data pointer is element-aligned. -/
def AllAligned (x y out : TensorMeta) : Prop := x.Aligned ∧ y.Aligned ∧ out.Aligned

/-- P11: the two inputs are the same buffer, or their accessed byte ranges
are disjoint (partial overlap of inputs is not representable by distinct
flat regions). -/
def InputsSeparate (x y : TensorMeta) : Prop :=
  x.base = y.base ∨ Blocked1DLaunch.RangesDisjoint x.numel x.toBuf y.toBuf

/-- Wrapper preconditions of a two-input elementwise wrapper with block `B`. -/
structure Pre (B : Nat) (x y out : TensorMeta) : Prop where
  same_shape : SameShape x y out
  contiguous : AllContiguous x y out
  aligned : AllAligned x y out
  inputs_separate : InputsSeparate x y
  launch : Blocked1DLaunch.Pre (launch B x y out)

/-- W1: the launch covers every element of the returned tensor. -/
def OutputCovered (c : Blocked1DLaunch) (out : TensorMeta) : Prop := c.n = out.numel

/-- W2: every input tensor holds the elements the launch reads. -/
def InputsCover (c : Blocked1DLaunch) (ins : List TensorMeta) : Prop :=
  ∀ t ∈ ins, c.n ≤ t.numel

def check (B : Nat) (x y out : TensorMeta) : Bool :=
  (y.shape == x.shape && out.shape == x.shape) &&
  (x.contiguousB && y.contiguousB && out.contiguousB) &&
  (x.base % x.elemBytes == 0 && y.base % y.elemBytes == 0 && out.base % out.elemBytes == 0) &&
  (x.base == y.base || Blocked1DLaunch.disjointB x.numel x.toBuf y.toBuf) &&
  Blocked1DLaunch.check (launch B x y out)

theorem contiguousB_iff (t : TensorMeta) : t.contiguousB = true ↔ t.Contiguous := by
  simp only [contiguousB, Contiguous, Bool.and_eq_true, beq_iff_eq, List.all_eq_true,
    List.mem_range, Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not]
  constructor
  · rintro ⟨hl, h⟩
    exact ⟨hl, fun i hi h1 => (h i hi).resolve_left (by omega)⟩
  · rintro ⟨hl, h⟩
    refine ⟨hl, fun i hi => ?_⟩
    by_cases h1 : 1 < t.shape.getD i 0
    · exact Or.inr (h i hi h1)
    · exact Or.inl h1

theorem check_ok (B : Nat) (x y out : TensorMeta) (h : check B x y out = true) :
    Pre B x y out := by
  simp only [check, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨hs1, hs2⟩, ⟨⟨hc1, hc2⟩, hc3⟩⟩, ⟨⟨ha1, ha2⟩, ha3⟩⟩, hsep⟩, hl⟩ := h
  refine ⟨⟨hs1, hs2⟩, ⟨(contiguousB_iff x).1 hc1, (contiguousB_iff y).1 hc2,
    (contiguousB_iff out).1 hc3⟩, ⟨ha1, ha2, ha3⟩, ?_, Blocked1DLaunch.check_ok _ hl⟩
  rcases hsep with h | h
  · exact Or.inl h
  · exact Or.inr ((Blocked1DLaunch.disjointB_iff _ _ _).1 h)

theorem check_complete (B : Nat) (x y out : TensorMeta) (h : Pre B x y out) :
    check B x y out = true := by
  simp only [check, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq]
  refine ⟨⟨⟨⟨h.same_shape, ⟨⟨(contiguousB_iff x).2 h.contiguous.1,
    (contiguousB_iff y).2 h.contiguous.2.1⟩, (contiguousB_iff out).2 h.contiguous.2.2⟩⟩,
    ⟨⟨h.aligned.1, h.aligned.2.1⟩, h.aligned.2.2⟩⟩, ?_⟩,
    Blocked1DLaunch.check_complete _ h.launch⟩
  rcases h.inputs_separate with hs | hs
  · exact Or.inl hs
  · exact Or.inr ((Blocked1DLaunch.disjointB_iff _ _ _).2 hs)

/-- W1 is derived: the wrapper's `n = x.numel()` and `out = *_like(x)`. -/
theorem Pre.output_covered {B : Nat} {x y out : TensorMeta} (h : Pre B x y out) :
    OutputCovered (Elementwise2.launch B x y out) out := by
  simp [OutputCovered, Elementwise2.launch, TensorMeta.numel, h.same_shape.2]

/-- W2 is derived from equal shapes. -/
theorem Pre.inputs_cover {B : Nat} {x y out : TensorMeta} (h : Pre B x y out) :
    InputsCover (Elementwise2.launch B x y out) [x, y] := by
  intro t ht
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at ht
  rcases ht with rfl | rfl
  · exact Nat.le_refl _
  · simp [Elementwise2.launch, TensorMeta.numel, h.same_shape.1]

end Elementwise2

end VeriTile.Triton
