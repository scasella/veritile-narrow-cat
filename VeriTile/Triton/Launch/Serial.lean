/-
VeriTile.Triton.Launch.Serial

Serial execution of a grid's programs, and its agreement with the relational
merge `Kernel.mergeFrames`.

`GridLaunchedOrdinary` composes a launch by running every program from the
*same* initial state and merging the written cells. That presumes no program
observes another program's writes. This module makes the presumption a
checkable hypothesis: `Kernel.FrameRobust` says each program, started from
any memory that differs from the launch state only on cells *other* programs
write, still produces its frame's values on its own footprint and leaves every
other cell as it found it. Under that hypothesis and pairwise-disjoint
footprints, `runSerial_agrees_merge` shows that running the programs one after
another, in **any** duplicate-free complete order, yields the merged result:
the same values on every written cell and the same memory everywhere else.

Scope: this relates whole-program serial orders to the merge. It does not
model interleavings of instructions from concurrently running programs.
-/

import VeriTile.Triton.Launch.Composition

namespace VeriTile.Triton

namespace Kernel

/-- Run the listed programs of a launch one after another. Each program
starts from the launch state `s` (registers, program ids, `undef`) carrying
the memory left by the previous program; only memory flows between programs.
`none` if some program fails. -/
noncomputable def runSerial (k : Kernel) {g : Grid} (s : BlockState) :
    List (GridIndex g) → (RegionName → Nat → MemCell) →
      Option (RegionName → Nat → MemCell)
  | [], m => some m
  | idx :: rest, m =>
      match exec k (({ s with mem := m } : BlockState).withGridIndex idx) with
      | none => none
      | some f => runSerial k s rest f.mem

/-- The real value read from a raw memory (as `BlockState.readMem`). -/
def memReal (m : RegionName → Nat → MemCell) (r : RegionName) (o : Nat) : ℝ :=
  match (m r o).readAs .real with
  | some (some value) => value
  | _ => 0

/-- Each program is insensitive to the other programs' writes: from any
memory that agrees with the launch state outside the cells written by the
*other* programs, program `idx` succeeds, produces its frame's values on its
own footprint, and leaves every cell outside its footprint unchanged. -/
def FrameRobust {k : Kernel} {g : Grid} {s : BlockState}
    (frames : Kernel.GridFrames k g s) : Prop :=
  ∀ (idx : GridIndex g) (m : RegionName → Nat → MemCell),
    (∀ r o, (¬ ∃ idx', idx' ≠ idx ∧ (frames idx').writes (r, o)) → m r o = s.mem r o) →
    ∃ f, exec k (({ s with mem := m } : BlockState).withGridIndex idx) = some f ∧
      (∀ r o, (frames idx).writes (r, o) → f.readMem r o = (frames idx).final.readMem r o) ∧
      (∀ r o, ¬ (frames idx).writes (r, o) → f.mem r o = m r o)

/-- **Serial orders agree with the merge.** For robust frames with pairwise
disjoint footprints, running all programs serially in any duplicate-free
complete order succeeds and matches `mergeFrames`: equal values on every
written cell, equal memory on every other cell. -/
theorem runSerial_agrees_merge {k : Kernel} {g : Grid} {s : BlockState}
    (frames : Kernel.GridFrames k g s)
    (hdisj : Kernel.GridWritesDisjoint frames) (hrob : Kernel.FrameRobust frames)
    (L : List (GridIndex g)) (hnd : L.Nodup) (hall : ∀ idx, idx ∈ L) :
    ∃ m, runSerial k s L s.mem = some m ∧
      (∀ r o, Kernel.GridWriteFootprint frames (r, o) →
        memReal m r o = (Kernel.mergeFrames g s frames).readMem r o) ∧
      (∀ r o, ¬ Kernel.GridWriteFootprint frames (r, o) →
        m r o = (Kernel.mergeFrames g s frames).mem r o) := by
  sorry

end Kernel

end VeriTile.Triton
