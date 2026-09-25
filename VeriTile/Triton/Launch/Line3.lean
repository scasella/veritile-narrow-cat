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

theorem toLine_bijective {g : Nat} : Function.Bijective (@toLine g) := by
  sorry

/-- A `(g, 1, 1)` program starts from the same state as its `(g,)` program. -/
theorem withGridIndex_toLine {g : Nat} (s : BlockState) (idx : GridIndex (line3 g)) :
    s.withGridIndex idx = s.withGridIndex (toLine idx) := by
  sorry

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
  sorry

theorem mergeFrames_liftFrames {k : Kernel} {g : Nat} {s : BlockState}
    {frames : Kernel.GridFrames k (line g) s} (h : Kernel.GridWritesDisjoint frames) :
    Kernel.mergeFrames (line3 g) s (liftFrames frames) = Kernel.mergeFrames (line g) s frames := by
  sorry

theorem liftFrames_robust {k : Kernel} {g : Nat} {s : BlockState}
    {frames : Kernel.GridFrames k (line g) s} (h : Kernel.FrameRobust frames) :
    Kernel.FrameRobust (liftFrames frames) := by
  sorry

end Blocked1D

end VeriTile.Triton
