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

theorem addrWrites_iff {R : RegionName} {addr : Nat → Nat} {n B pid : Nat} (hB : 0 < B)
    (r : RegionName) (o : Nat) :
    addrWrites R addr n B pid (r, o) ↔ r = R ∧ ∃ t, t < n ∧ t / B = pid ∧ addr t = o := by
  unfold addrWrites WriteFootprint.activeTileImage
  constructor
  · rintro ⟨hr, ⟨i, _⟩, ha, ho⟩
    refine ⟨hr, pid * B + i.val, ha, ?_, ho⟩
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt i.isLt, Nat.zero_add]
  · rintro ⟨hr, t, ht, hd, ho⟩
    have hdm := Nat.div_add_mod' t B
    rw [hd] at hdm
    exact ⟨hr, (⟨t % B, Nat.mod_lt t hB⟩, PUnit.unit), by simp [hdm, ht], by simp [hdm, ho]⟩

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
  have hdisj : Kernel.GridWritesDisjoint frames := by
    intro i₁ i₂ hne ⟨r, o⟩ h₁ h₂
    rw [hwrites, addrWrites_iff hB] at h₁ h₂
    obtain ⟨-, t₁, -, hd₁, ha₁⟩ := h₁
    obtain ⟨-, t₂, -, hd₂, ha₂⟩ := h₂
    have ht : t₁ = t₂ := hinj _ _ (ha₁.trans ha₂.symm)
    exact hne (idx_ext (by rw [← hd₁, ← hd₂, ht]))
  refine ⟨⟨frames, hdisj, rfl⟩, rfl, ?_, ?_⟩
  · intro i hi
    have hown : i / B < g := (Nat.div_lt_iff_lt_mul hB).2 (by omega)
    have hdm := Nat.div_add_mod' i B
    have hw : (frames (indexOf (i / B) hown)).writes (R, addr i) := by
      rw [hwrites, addrWrites_iff hB, pidOf_indexOf]
      exact ⟨rfl, i, hi, rfl, rfl⟩
    have hm := Kernel.mergeFrames_mem_written hdisj _ R (addr i) hw
    have hv := hval (indexOf (i / B) hown) (i % B) (Nat.mod_lt i hB)
      (by rw [pidOf_indexOf, hdm]; exact hi)
    rw [pidOf_indexOf, hdm] at hv
    rw [← hv]
    unfold BlockState.readMem
    rw [hm]
  · intro r o hno
    apply Kernel.mergeFrames_mem_eq_of_not_written
    rintro ⟨idx, hw⟩
    rw [hwrites, addrWrites_iff hB] at hw
    obtain ⟨hr, t, ht, -, ho⟩ := hw
    exact hno ⟨hr, t, ht, ho⟩

end Blocked1D

/-- Setting a program's grid index commutes with flattening (both only touch
`pids`/`numPids`, which flattening preserves). -/
theorem FlatAlloc.flattenState_withGridIndex (A : FlatAlloc) {g : Grid} (s : BlockState)
    (idx : GridIndex g) :
    (A.flattenState s).withGridIndex idx = A.flattenState (s.withGridIndex idx) := rfl

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
  obtain ⟨hlen, hstr⟩ := hc
  obtain ⟨hil, hib⟩ := hi
  unfold offsetOf numel
  generalize t.strides = strides at hlen hstr
  generalize t.shape = shape at hlen hstr hil hib
  induction shape generalizing strides idx with
  | nil =>
    cases idx with
    | nil => simp [linear]
    | cons _ _ => simp at hil
  | cons d rest ih =>
    cases idx with
    | nil => simp at hil
    | cons i is =>
      cases strides with
      | nil => simp at hlen
      | cons s0 ss =>
        simp only [List.length_cons, Nat.add_right_cancel_iff] at hlen hil
        have h0 : i < d := by simpa using hib 0 (by simp)
        have hs0 : 1 < d → s0 = rest.foldr (· * ·) 1 := by
          intro h1; simpa [rowMajor] using hstr 0 (by simp) (by simpa using h1)
        obtain ⟨ihe, ihl⟩ := ih is ss hlen
          (fun k hk h1 => by simpa [rowMajor] using hstr (k + 1) (by simp; omega) (by simpa using h1))
          hil (fun k hk => by simpa using hib (k + 1) (by simp; omega))
        simp only [List.zip_cons_cons, List.foldr_cons, linear]
        refine ⟨?_, ?_⟩
        · rw [ihe]
          by_cases h1 : 1 < d
          · rw [hs0 h1]
          · have : i = 0 := by omega
            simp [this]
        · have hle : (i + 1) * rest.foldr (· * ·) 1 ≤ d * rest.foldr (· * ·) 1 :=
            Nat.mul_le_mul_right _ (by omega)
          calc i * rest.foldr (· * ·) 1 + linear rest is
              < i * rest.foldr (· * ·) 1 + rest.foldr (· * ·) 1 := by omega
            _ = (i + 1) * rest.foldr (· * ·) 1 := by ring
            _ ≤ d * rest.foldr (· * ·) 1 := hle

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
  intro r hr
  simp only [flatAlloc, List.mem_cons, List.mem_nil_iff, or_false, not_or] at hr ⊢
  rw [if_neg (by tauto)]

/-- **The checked layout conditions discharge `FlatAlloc.Disjoint`.** Region
names follow the allocations: the output is its own region, and the two
inputs share a region exactly when their data pointers coincide. -/
theorem Pre.flat_disjoint {B : Nat} {x y out : TensorMeta} (h : Pre B x y out)
    (flat in0 in1 outR : RegionName) (hout0 : outR ≠ in0) (hout1 : outR ≠ in1)
    (hnames : in0 = in1 ↔ x.base = y.base) :
    (flatAlloc flat in0 in1 outR x y out).Disjoint := by
  have hp := h.launch
  have hdt := hp.dtype_ok
  have ebx : x.elemBytes = 4 := (hdt x.toBuf (by simp [Blocked1DLaunch.bufs, Elementwise2.launch])).2
  have eby : y.elemBytes = 4 := (hdt y.toBuf (by simp [Blocked1DLaunch.bufs, Elementwise2.launch])).2
  have ebo : out.elemBytes = 4 := (hdt out.toBuf (by simp [Blocked1DLaunch.bufs, Elementwise2.launch])).2
  obtain ⟨ax, ay, ao⟩ := h.aligned
  unfold TensorMeta.Aligned at ax ay ao
  rw [ebx] at ax; rw [eby] at ay; rw [ebo] at ao
  -- byte-range disjointness, as arithmetic
  have hbytes : ∀ a b : TensorMeta, a.elemBytes = 4 → b.elemBytes = 4 →
      a.base % 4 = 0 → b.base % 4 = 0 →
      Blocked1DLaunch.RangesDisjoint x.numel a.toBuf b.toBuf →
      a.base / 4 + x.numel ≤ b.base / 4 ∨ b.base / 4 + x.numel ≤ a.base / 4 := by
    intro a b ea eb ha hb hd
    have := (Blocked1DLaunch.disjointB_iff _ _ _).2 hd
    simp only [Blocked1DLaunch.disjointB, TensorMeta.toBuf, ea, eb, Bool.or_eq_true,
      decide_eq_true_eq] at this
    omega
  have hxo := hbytes out x ebo ebx ao ax (hp.out_disjoint x.toBuf (by simp [Elementwise2.launch]))
  have hyo := hbytes out y ebo eby ao ay (hp.out_disjoint y.toBuf (by simp [Elementwise2.launch]))
  have hxy : in0 ≠ in1 → x.base / 4 + x.numel ≤ y.base / 4 ∨ y.base / 4 + x.numel ≤ x.base / 4 := by
    intro hne
    rcases h.inputs_separate with he | hd
    · exact absurd (hnames.2 he) hne
    · exact hbytes x y ebx eby ax ay hd
  set A := flatAlloc flat in0 in1 outR x y out with hA
  have bo : A.base outR = out.base / 4 := by simp [hA, flatAlloc, ebo]
  have b0 : A.base in0 = x.base / 4 := by simp [hA, flatAlloc, Ne.symm hout0, ebx]
  have b1 : A.base in1 = if in1 = in0 then x.base / 4 else y.base / 4 := by
    by_cases h10 : in1 = in0
    · subst h10; simp [hA, flatAlloc, Ne.symm hout0, ebx]
    · simp [hA, flatAlloc, Ne.symm hout1, h10, eby]
  have ext : ∀ r, r = in0 ∨ r = in1 ∨ r = outR → A.extent r = x.numel := by
    intro r hr; simp only [hA, flatAlloc]; rw [if_pos (by tauto)]
  intro r hr r' hr' hne
  have hr0 : r = in0 ∨ r = in1 ∨ r = outR := by simpa [hA, flatAlloc] using hr
  have hr0' : r' = in0 ∨ r' = in1 ∨ r' = outR := by simpa [hA, flatAlloc] using hr'
  rw [ext r hr0, ext r' hr0']
  by_cases h10 : in1 = in0
  · rw [if_pos h10] at b1
    rcases hr0 with rfl | rfl | rfl <;> rcases hr0' with rfl | rfl | rfl <;>
      first | exact absurd rfl hne | (subst h10; exact absurd rfl hne) |
        (simp only [bo, b0, b1]; omega)
  · rw [if_neg h10] at b1
    have hxy' := hxy (Ne.symm h10)
    rcases hr0 with rfl | rfl | rfl <;> rcases hr0' with rfl | rfl | rfl <;>
      first | exact absurd rfl hne | (simp only [bo, b0, b1]; omega)

/-- Each flat window lies inside its tensor's allocation (P3 + P4). -/
theorem Pre.windows_in_alloc {B : Nat} {x y out : TensorMeta} (h : Pre B x y out) :
    x.numel ≤ x.capacity ∧ x.numel ≤ y.capacity ∧ x.numel ≤ out.capacity := by
  have hp := h.launch
  have hB := hp.block_pos
  have key : ∀ b ∈ (Elementwise2.launch B x y out).bufs, x.numel ≤ b.capacity := by
    intro b hb
    rcases Nat.eq_zero_or_pos x.numel with h0 | hpos
    · rw [h0]; exact Nat.zero_le _
    have hcov := hp.covers (x.numel - 1) (by simp [Elementwise2.launch]; omega)
    have hdm := Nat.div_add_mod' (x.numel - 1) B
    have := hp.lanes_in_bounds b hb ((x.numel - 1) / B) hcov ((x.numel - 1) % B)
      (Nat.mod_lt _ hB) (by simp only [Elementwise2.launch] at hdm ⊢; omega)
    simp only [Elementwise2.launch] at hdm this
    omega
  refine ⟨key x.toBuf ?_, key y.toBuf ?_, key out.toBuf ?_⟩ <;>
    simp [Blocked1DLaunch.bufs, Elementwise2.launch]

end Elementwise2

end VeriTile.Triton
