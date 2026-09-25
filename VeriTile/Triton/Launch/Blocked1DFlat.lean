/-
VeriTile.Triton.Launch.Blocked1DFlat

Connects checked host layout conditions to VeriTile's flat-memory machinery
(`FlatAlloc`, `MemoryKernelIO.Implements` / `⊨`) and generalizes whole-grid
composition to an injective lane-address map.

* `Blocked1D.launch_of_frames_addr` — `launch_of_frames` for footprints
  `{addr t : t active}` in one region, for any injective `addr` (unit-stride
  output windows, strided outputs `t * stride`, flat addresses `base + t`).
* `Elementwise2.flatAlloc` — the flat placement of a two-input elementwise
  launch built from the tensors' own metadata: each region sits at its
  tensor's element address `base / elemBytes` with extent `n`.
* `Elementwise2.Pre.flat_disjoint` — the checked conditions (P9 dtype/width,
  P10 output disjointness, P11 input separation, P12 alignment) discharge the
  bridge's `FlatAlloc.Disjoint`; `Pre.windows_in_alloc` (P3 + P4) places each
  flat window inside its tensor's allocation.
* `TensorMeta.offsetOf_eq_linear` — the logical view: for a contiguous tensor
  the element at an in-shape multi-index sits at its row-major linear offset.
-/

import VeriTile.Triton.Launch.Blocked1DWrapper
import VeriTile.Triton.Launch.Blocked1D
import VeriTile.Triton.Memory.Flatten

namespace VeriTile.Triton

namespace Blocked1D

/-- Per-program footprint in region `R` under lane-address map `addr`: the
active lanes `t = pid * B + j < n`, written at `addr t`. -/
def addrWrites (R : RegionName) (addr : Nat → Nat) (n B pid : Nat) : WriteFootprint :=
  WriteFootprint.activeTileImage R
    (fun i : TileIndex [B] => addr (pid * B + i.1.val))
    (fun i : TileIndex [B] => pid * B + i.1.val < n)

/-- **Whole-grid composition for an injective lane-address map.** As
`launch_of_frames`, with lane `t` written at `addr t`. -/
theorem launch_of_frames_addr {k : Kernel} {g n B : Nat} (hB : 0 < B)
    (hcov : n ≤ g * B) {s : BlockState} {R : RegionName} (addr : Nat → Nat)
    (hinj : ∀ a b, addr a = addr b → a = b)
    (frames : Kernel.GridFrames k (line g) s)
    (hwrites : ∀ idx, (frames idx).writes = addrWrites R addr n B (pidOf idx))
    (expected : Nat → ℝ)
    (hval : ∀ idx j, j < B → pidOf idx * B + j < n →
      (frames idx).final.readMem R (addr (pidOf idx * B + j))
        = expected (pidOf idx * B + j)) :
    ∃ L : Kernel.GridLaunchedOrdinary k (line g) s (Kernel.mergeFrames (line g) s frames),
      L.frames = frames ∧
      (∀ i, i < n → (Kernel.mergeFrames (line g) s frames).readMem R (addr i) = expected i) ∧
      (∀ r o, ¬ (r = R ∧ ∃ i, i < n ∧ addr i = o) →
        (Kernel.mergeFrames (line g) s frames).mem r o = s.mem r o) := by
  sorry

end Blocked1D

namespace TensorMeta

/-- Row-major linear index of a multi-index. -/
def linear : List Nat → List Nat → Nat
  | _ :: rest, i :: is => i * rest.foldr (· * ·) 1 + linear rest is
  | _, _ => 0

/-- The logical view of a contiguous tensor: the element at an in-shape
multi-index sits at its row-major linear offset, below `numel`. -/
theorem offsetOf_eq_linear (t : TensorMeta) (hc : t.Contiguous) (idx : List Nat)
    (hi : InShape t.shape idx) :
    t.offsetOf idx = linear t.shape idx ∧ linear t.shape idx < t.numel := by
  sorry

end TensorMeta

namespace Elementwise2

open TensorMeta

/-- Flat placement of a two-input elementwise launch from the tensors' own
metadata: region `r` sits at its tensor's element address `base / elemBytes`
and extends over the `n = x.numel` accessed elements. -/
def flatAlloc (flat in0 in1 outR : RegionName) (x y out : TensorMeta) : FlatAlloc :=
  { flat := flat
    regions := [in0, in1, outR]
    base := fun r =>
      if r = outR then out.base / out.elemBytes
      else if r = in0 then x.base / x.elemBytes
      else if r = in1 then y.base / y.elemBytes else 0
    extent := fun r => if r = outR ∨ r = in0 ∨ r = in1 then x.numel else 0 }

/-- Regions outside the placement have extent `0` (the bridge's `hcov`). -/
theorem flatAlloc_closed (flat in0 in1 outR : RegionName) (x y out : TensorMeta) :
    ∀ r, r ∉ (flatAlloc flat in0 in1 outR x y out).regions →
      (flatAlloc flat in0 in1 outR x y out).extent r = 0 := by
  sorry

/-- **The checked layout conditions discharge `FlatAlloc.Disjoint`.** Region
names follow the allocations: the output is its own region, and the two
inputs share a region exactly when their data pointers coincide. -/
theorem Pre.flat_disjoint {B : Nat} {x y out : TensorMeta} (h : Pre B x y out)
    (flat in0 in1 outR : RegionName) (hout0 : outR ≠ in0) (hout1 : outR ≠ in1)
    (hnames : in0 = in1 ↔ x.base = y.base) :
    (flatAlloc flat in0 in1 outR x y out).Disjoint := by
  sorry

/-- Each flat window lies inside its tensor's allocation (P3 + P4). -/
theorem Pre.windows_in_alloc {B : Nat} {x y out : TensorMeta} (h : Pre B x y out) :
    x.numel ≤ x.capacity ∧ x.numel ≤ y.capacity ∧ x.numel ≤ out.capacity := by
  sorry

end Elementwise2

end VeriTile.Triton
