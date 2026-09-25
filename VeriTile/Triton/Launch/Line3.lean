/-
VeriTile.Triton.Launch.Line3

Launch grids `(g, 1, 1)` versus `(g,)`. Triton wrappers often pass a padded
grid tuple; the programs of `(g, 1, 1)` see exactly the program ids and
program counts of `(g,)` (axes 1 and 2 have one program, and out-of-rank axes
default to one program and id `0`). This module transfers per-program frames,
their merge, serial robustness, and `Kernel.LaunchCorrectFramed` from the
1-D grid `Blocked1D.line g` to `Blocked1D.line3 g`.
-/

import VeriTile.Triton.Launch.Serial
import VeriTile.Triton.Launch.Blocked1D

namespace VeriTile.Triton

namespace Blocked1D

/-- The launch grid `(g, 1, 1)`. -/
abbrev line3 (g : Nat) : Grid := { dims := [g, 1, 1] }

/-- The `(g,)` program with the same axis-0 id. -/
def toLine {g : Nat} (idx : GridIndex (line3 g)) : GridIndex (line g) :=
  indexOf (idx ⟨0, by simp [Grid.rank]⟩).val (by
    simpa [Grid.dim] using (idx ⟨0, by simp [Grid.rank]⟩).isLt)

theorem line3_dim_lt {g : Nat} (idx : GridIndex (line3 g)) (a : Fin (line3 g).rank)
    (ha : a.val ≠ 0) : (idx a).val = 0 := by
  have h := (idx a).isLt
  have hr : a.val < 3 := a.isLt
  have hd : (line3 g).dim a.val = 1 := by
    rcases a with ⟨k, hk⟩
    simp only at ha hr ⊢
    interval_cases k <;> simp_all [Grid.dim]
  omega

theorem toLine_bijective {g : Nat} : Function.Bijective (@toLine g) := by
  constructor
  · intro a b h
    have h0 : (a ⟨0, by simp [Grid.rank]⟩).val = (b ⟨0, by simp [Grid.rank]⟩).val := by
      have := congrArg pidOf h
      simpa [toLine, pidOf_indexOf] using this
    funext ax
    apply Fin.ext
    by_cases hax : ax.val = 0
    · have : ax = ⟨0, by simp [Grid.rank]⟩ := Fin.ext hax
      subst this
      exact h0
    · rw [line3_dim_lt a ax hax, line3_dim_lt b ax hax]
  · intro i
    refine ⟨fun ax => if h : ax.val = 0 then ⟨pidOf i, by
        have := pidOf_lt i
        simp [Grid.dim, h]
        exact this⟩ else ⟨0, by
        have hr : ax.val < 3 := ax.isLt
        rcases ax with ⟨k, hk⟩
        simp only at h hr ⊢
        interval_cases k <;> simp_all [Grid.dim]⟩, ?_⟩
    apply idx_ext
    simp [toLine, pidOf_indexOf]

/-- A `(g, 1, 1)` program starts from the same state as its `(g,)` program. -/
theorem withGridIndex_toLine {g : Nat} (s : BlockState) (idx : GridIndex (line3 g)) :
    s.withGridIndex idx = s.withGridIndex (toLine idx) := by
  have hp : idx.toPids = (toLine idx).toPids := by
    funext a
    rcases Nat.lt_or_ge a 3 with ha | ha
    · by_cases h0 : a = 0
      · subst h0
        simp [GridIndex.toPids, Grid.rank, toLine, indexOf]
      · have := line3_dim_lt idx ⟨a, by simpa [Grid.rank] using ha⟩ h0
        simp only [GridIndex.toPids, Grid.rank, List.length_cons, List.length_nil]
        rw [dif_pos (by omega), dif_neg (by omega)]
        exact this
    · simp only [GridIndex.toPids, Grid.rank, List.length_cons, List.length_nil]
      rw [dif_neg (by omega), dif_neg (by omega)]
  have hn : (line3 g).toNumPids = (line g).toNumPids := by
    funext a
    simp only [Grid.toNumPids, Grid.rank, Grid.dim, List.length_cons, List.length_nil]
    rcases a with _ | _ | _ | a <;> simp
  simp only [BlockState.withGridIndex, BlockState.withPids, BlockState.withNumPids, hp, hn]

/-- Frames of the `(g,)` launch, re-indexed by the `(g, 1, 1)` grid. -/
def liftFrames {k : Kernel} {g : Nat} {s : BlockState}
    (frames : Kernel.GridFrames k (line g) s) : Kernel.GridFrames k (line3 g) s :=
  fun idx =>
    { final := (frames (toLine idx)).final
      writes := (frames (toLine idx)).writes
      h_exec := by rw [withGridIndex_toLine]; exact (frames (toLine idx)).h_exec
      h_writeWithin := by
        rw [withGridIndex_toLine]; exact (frames (toLine idx)).h_writeWithin }

theorem liftFrames_disjoint {k : Kernel} {g : Nat} {s : BlockState}
    {frames : Kernel.GridFrames k (line g) s} (h : Kernel.GridWritesDisjoint frames) :
    Kernel.GridWritesDisjoint (liftFrames frames) := by
  intro i₁ i₂ hne
  exact h (toLine i₁) (toLine i₂) fun e => hne (toLine_bijective.1 e)

theorem mergeFrames_liftFrames {k : Kernel} {g : Nat} {s : BlockState}
    {frames : Kernel.GridFrames k (line g) s} (h : Kernel.GridWritesDisjoint frames) :
    Kernel.mergeFrames (line3 g) s (liftFrames frames) = Kernel.mergeFrames (line g) s frames := by
  classical
  have hmem : (Kernel.mergeFrames (line3 g) s (liftFrames frames)).mem
      = (Kernel.mergeFrames (line g) s frames).mem := by
    funext r o
    by_cases hw : ∃ idx, (frames idx).writes (r, o)
    · obtain ⟨idx, hidx⟩ := hw
      obtain ⟨idx3, h3⟩ := toLine_bijective.2 idx
      have hw3 : (liftFrames frames idx3).writes (r, o) := by
        show (frames (toLine idx3)).writes (r, o)
        rw [h3]; exact hidx
      rw [Kernel.mergeFrames_mem_written (liftFrames_disjoint h) idx3 r o hw3,
        Kernel.mergeFrames_mem_written h idx r o hidx]
      show (frames (toLine idx3)).final.mem r o = _
      rw [h3]
    · rw [Kernel.mergeFrames_mem_unwritten r o (fun ⟨idx3, h3⟩ => hw ⟨toLine idx3, h3⟩),
        Kernel.mergeFrames_mem_unwritten r o (fun ⟨idx, hidx⟩ => hw ⟨idx, hidx⟩)]
  have e1 : Kernel.mergeFrames (line3 g) s (liftFrames frames)
      = { s with mem := (Kernel.mergeFrames (line3 g) s (liftFrames frames)).mem } := rfl
  have e2 : Kernel.mergeFrames (line g) s frames
      = { s with mem := (Kernel.mergeFrames (line g) s frames).mem } := rfl
  rw [e1, e2, hmem]

theorem liftFrames_robust {k : Kernel} {g : Nat} {s : BlockState}
    {frames : Kernel.GridFrames k (line g) s} (h : Kernel.FrameRobust frames) :
    Kernel.FrameRobust (liftFrames frames) := by
  intro idx3 m hm
  have hm' : ∀ r o, (¬ ∃ idx', idx' ≠ toLine idx3 ∧ (frames idx').writes (r, o)) →
      m r o = s.mem r o := by
    intro r o hno
    apply hm
    rintro ⟨idx3', hne, hw⟩
    exact hno ⟨toLine idx3', fun e => hne (toLine_bijective.1 e), hw⟩
  obtain ⟨f, hf, hv, hfr⟩ := h (toLine idx3) m hm'
  refine ⟨f, ?_, hv, hfr⟩
  rw [withGridIndex_toLine]
  exact hf

end Blocked1D

end VeriTile.Triton
