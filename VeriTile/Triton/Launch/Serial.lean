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

theorem memReal_mem (s : BlockState) (r : RegionName) (o : Nat) :
    memReal s.mem r o = s.readMem r o := rfl

/-- Invariant of a serial prefix `P`: every cell written by a finished program
holds that program's value; every cell no finished program writes is as in
the launch state. -/
def SerialInv {k : Kernel} {g : Grid} {s : BlockState}
    (frames : Kernel.GridFrames k g s) (P : List (GridIndex g))
    (m : RegionName → Nat → MemCell) : Prop :=
  (∀ idx ∈ P, ∀ r o, (frames idx).writes (r, o) →
      memReal m r o = (frames idx).final.readMem r o) ∧
  (∀ r o, (¬ ∃ idx ∈ P, (frames idx).writes (r, o)) → m r o = s.mem r o)

theorem runSerial_inv {k : Kernel} {g : Grid} {s : BlockState}
    (frames : Kernel.GridFrames k g s)
    (hdisj : Kernel.GridWritesDisjoint frames) (hrob : Kernel.FrameRobust frames) :
    ∀ (L P : List (GridIndex g)) (m : RegionName → Nat → MemCell),
      (P ++ L).Nodup → SerialInv frames P m →
      ∃ m', runSerial k s L m = some m' ∧ SerialInv frames (P ++ L) m' := by
  intro L
  induction L with
  | nil => intro P m _ hinv; exact ⟨m, rfl, by simpa using hinv⟩
  | cons idx rest ih =>
    intro P m hnd hinv
    have hnotin : idx ∉ P := by
      intro hmem
      have := (List.nodup_append.mp hnd).2.2
      exact this idx hmem idx (by simp) rfl
    have hpre : ∀ r o, (¬ ∃ idx', idx' ≠ idx ∧ (frames idx').writes (r, o)) →
        m r o = s.mem r o := by
      intro r o hno
      apply hinv.2
      rintro ⟨p, hp, hw⟩
      exact hno ⟨p, fun h => hnotin (h ▸ hp), hw⟩
    obtain ⟨f, hf, hval, hframe⟩ := hrob idx m hpre
    have hstep : runSerial k s (idx :: rest) m = runSerial k s rest f.mem := by
      simp [runSerial, hf]
    have hinv' : SerialInv frames (P ++ [idx]) f.mem := by
      refine ⟨fun p hp r o hw => ?_, fun r o hno => ?_⟩
      · rcases List.mem_append.mp hp with hp | hp
        · have hne : p ≠ idx := fun h => hnotin (h ▸ hp)
          have hnw : ¬ (frames idx).writes (r, o) := fun h => hdisj p idx hne _ hw h
          rw [show memReal f.mem r o = memReal m r o by simp [memReal, hframe r o hnw]]
          exact hinv.1 p hp r o hw
        · simp only [List.mem_singleton] at hp
          subst hp
          rw [memReal_mem]
          exact hval r o hw
      · have hnw : ¬ (frames idx).writes (r, o) :=
          fun h => hno ⟨idx, by simp, h⟩
        rw [hframe r o hnw]
        exact hinv.2 r o fun ⟨p, hp, hw⟩ => hno ⟨p, List.mem_append_left _ hp, hw⟩
    obtain ⟨m', hm', hinv''⟩ := ih (P ++ [idx]) f.mem (by simpa using hnd) hinv'
    exact ⟨m', hstep ▸ hm', by simpa using hinv''⟩

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
  classical
  obtain ⟨m, hm, hinv⟩ := runSerial_inv frames hdisj hrob L [] s.mem (by simpa using hnd)
    ⟨fun _ h => absurd h (by simp), fun _ _ _ => rfl⟩
  simp only [List.nil_append] at hinv
  refine ⟨m, hm, fun r o hw => ?_, fun r o hnw => ?_⟩
  · obtain ⟨idx, hidx⟩ := hw
    rw [hinv.1 idx (hall idx) r o hidx]
    unfold BlockState.readMem
    rw [Kernel.mergeFrames_mem_written hdisj idx r o hidx]
  · rw [Kernel.mergeFrames_mem_unwritten r o hnw]
    exact hinv.2 r o fun ⟨idx, _, hw⟩ => hnw ⟨idx, hw⟩

end Kernel

end VeriTile.Triton
