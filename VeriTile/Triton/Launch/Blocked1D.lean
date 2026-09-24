/-
VeriTile.Triton.Launch.Blocked1D

Whole-grid composition for one-dimensional blocked masked kernels over a
host-supplied grid `(g,)`: program `pid` owns the lanes `pid * B + j`
(`j < B`), active when `< n`. Lifts the per-file helpers of the Adam grid
showcase (`adamGrid`, `adamOwner`, `block_offset_inj`, …) to one reusable
statement, `Blocked1D.launch_of_frames`, parameterized by the kernel and by
the grid size instead of hard-wiring `cdiv`.

Coverage (`n ≤ g * B`) is a hypothesis here; `Blocked1DLaunch.check_ok`
discharges it (and the remaining host obligations) from a checked launch
configuration.
-/

import VeriTile.Triton.Launch.Composition
import VeriTile.Triton.Launch.Blocked1DConfig

namespace VeriTile.Triton

namespace Blocked1D

/-- The one-dimensional launch grid `(g,)`. -/
abbrev line (g : Nat) : Grid := { dims := [g] }

/-- The axis-0 program id of a `line g` grid index. -/
def pidOf {g : Nat} (idx : GridIndex (line g)) : Nat :=
  (idx ⟨0, by simp [Grid.rank]⟩).val

theorem pidOf_lt {g : Nat} (idx : GridIndex (line g)) : pidOf idx < g := by
  simpa [pidOf, line, Grid.dim] using (idx ⟨0, by simp [Grid.rank]⟩).isLt

@[simp] theorem withGridIndex_pid_line {g : Nat} (s : BlockState)
    (idx : GridIndex (line g)) : (s.withGridIndex idx).pid = pidOf idx :=
  BlockState.withGridIndex_pid s idx (by simp [Grid.rank])

/-- A `line g` grid index is determined by its program id. -/
theorem idx_ext {g : Nat} {idx₁ idx₂ : GridIndex (line g)}
    (h : pidOf idx₁ = pidOf idx₂) : idx₁ = idx₂ := by
  funext a
  have ha : a = ⟨0, by simp [Grid.rank]⟩ := by
    apply Fin.ext
    have hlt := a.isLt
    have hr : (line g).rank = 1 := rfl
    omega
  subst ha
  exact Fin.ext h

/-- The grid index of program `p < g`. -/
def indexOf {g : Nat} (p : Nat) (hp : p < g) : GridIndex (line g) :=
  fun ax => ⟨p, by
    have h0 : ax.val = 0 := by
      have h := ax.isLt
      have hr : (line g).rank = 1 := rfl
      omega
    simpa [line, Grid.dim, h0] using hp⟩

@[simp] theorem pidOf_indexOf {g p : Nat} (hp : p < g) :
    pidOf (indexOf p hp) = p := rfl

/-- Per-program write footprint: the active lanes of the output block. -/
def blockWrites (out : RegionName) (n B pid : Nat) : WriteFootprint :=
  WriteFootprint.activeTileImage out
    (fun i : TileIndex [B] => pid * B + i.1.val)
    (fun i : TileIndex [B] => pid * B + i.1.val < n)

theorem blockWrites_iff {out : RegionName} {n B pid : Nat} (hB : 0 < B)
    (r : RegionName) (o : Nat) :
    blockWrites out n B pid (r, o) ↔ r = out ∧ o < n ∧ o / B = pid := by
  unfold blockWrites WriteFootprint.activeTileImage
  constructor
  · rintro ⟨hr, ⟨i, _⟩, ha, ho⟩
    simp only at hr ha ho
    subst ho
    refine ⟨hr, ha, ?_⟩
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt i.isLt, Nat.zero_add]
  · rintro ⟨hr, hn, hd⟩
    have hdm := Nat.div_add_mod' o B
    rw [hd] at hdm
    exact ⟨hr, (⟨o % B, Nat.mod_lt o hB⟩, PUnit.unit), by simp [hdm, hn], by simp [hdm]⟩

theorem blockWrites_disjoint {out : RegionName} {n B p₁ p₂ : Nat} (hB : 0 < B)
    (hne : p₁ ≠ p₂) :
    WriteFootprint.disjoint (blockWrites out n B p₁) (blockWrites out n B p₂) := by
  rintro ⟨r, o⟩ h₁ h₂
  rw [blockWrites_iff hB] at h₁ h₂
  exact hne (h₁.2.2.symm.trans h₂.2.2)

/-- **Whole-grid composition for 1-D blocked masked kernels.** Given one
successful framed execution per program whose writes are exactly its active
output lanes, a per-program value fact on those lanes, and grid coverage, the
merged launch is an ordinary disjoint launch whose final memory holds
`expected i` at every output index `i < n` and agrees with the initial state
everywhere else. -/
theorem launch_of_frames {k : Kernel} {g n B : Nat} (hB : 0 < B)
    (hcov : n ≤ g * B) {s : BlockState} {out : RegionName}
    (frames : Kernel.GridFrames k (line g) s)
    (hwrites : ∀ idx, (frames idx).writes = blockWrites out n B (pidOf idx))
    (expected : Nat → ℝ)
    (hval : ∀ idx j, j < B → pidOf idx * B + j < n →
      (frames idx).final.readMem out (pidOf idx * B + j)
        = expected (pidOf idx * B + j)) :
    ∃ L : Kernel.GridLaunchedOrdinary k (line g) s (Kernel.mergeFrames (line g) s frames),
      L.frames = frames ∧
      (∀ i, i < n → (Kernel.mergeFrames (line g) s frames).readMem out i = expected i) ∧
      (∀ r o, ¬ (r = out ∧ o < n) →
        (Kernel.mergeFrames (line g) s frames).mem r o = s.mem r o) := by
  have hdisj : Kernel.GridWritesDisjoint frames := by
    intro i₁ i₂ hne
    rw [hwrites i₁, hwrites i₂]
    exact blockWrites_disjoint hB fun h => hne (idx_ext h)
  refine ⟨⟨frames, hdisj, rfl⟩, rfl, ?_, ?_⟩
  · intro i hi
    have hown : i / B < g := (Nat.div_lt_iff_lt_mul hB).2 (by omega)
    have hdm := Nat.div_add_mod' i B
    have hw : (frames (indexOf (i / B) hown)).writes (out, i) := by
      rw [hwrites, blockWrites_iff hB, pidOf_indexOf]
      exact ⟨rfl, hi, rfl⟩
    have hm := Kernel.mergeFrames_mem_written hdisj _ out i hw
    have hv := hval (indexOf (i / B) hown) (i % B) (Nat.mod_lt i hB)
      (by rw [pidOf_indexOf, hdm]; exact hi)
    rw [pidOf_indexOf, hdm] at hv
    rw [← hv]
    unfold BlockState.readMem
    rw [hm]
  · intro r o hno
    apply Kernel.mergeFrames_mem_eq_of_not_written
    rintro ⟨idx, hw⟩
    rw [hwrites, blockWrites_iff hB] at hw
    exact hno ⟨hw.1, hw.2.1⟩

end Blocked1D

end VeriTile.Triton
