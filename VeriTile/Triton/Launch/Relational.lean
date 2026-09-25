/-
VeriTile.Triton.Launch.Relational

Kernel-agnostic relational results for 1-D blocked launches:

* `Blocked1D.ProgramRuns` — the per-program shape every blocked elementwise
  kernel proof provides: from any state satisfying a precondition, the program
  terminates, writes `F o` at every active output lane `o` of its block, and
  changes no other cell.
* `Blocked1D.launch_of_programRuns` — such programs compose into a whole
  `(g,)` launch (`GridLaunchedOrdinary`) writing `F o` for every `o < n` and
  leaving every other cell unchanged.
* `Blocked1D.fused_agrees_two_launch` — **fusion as a relation between
  executions**: a launch `kA` writing a private temporary `T` with `F`, followed
  by a launch `kB` reading `T` and writing `O` with `G ∘ F`, and a single fused
  launch `kF` writing `O` with `G ∘ F`, end in states that agree on every output
  element `O[o]`, `o < n` (as observed by `readMem`), and on every cell outside
  the temporary's window `T[0, n)`. The temporary is the only difference.

Launch semantics are the model's (`GridLaunchedOrdinary`: every program starts
from the launch state; disjoint write sets are merged). No floating-point claim.
-/

import VeriTile.Triton.Launch.Blocked1D

namespace VeriTile.Triton

namespace Blocked1D

/-- Per-program run of a blocked kernel writing region `R` with `F` on the
active lanes `o < n` of its block, from any state satisfying `pre`. -/
def ProgramRuns (k : Kernel) (R : RegionName) (n B : Nat) (pre : BlockState → Prop)
    (F : Nat → ℝ) : Prop :=
  ∀ t, pre t → ∃ f, exec k t = some f ∧
    (∀ o, o < n → o / B = t.pid → f.readMem R o = F o) ∧
    (∀ r o, ¬ blockWrites R n B t.pid (r, o) → f.mem r o = t.mem r o)

/-- Programs with `ProgramRuns` compose into a whole `(g,)` launch. -/
theorem launch_of_programRuns {k : Kernel} {R : RegionName} {n B g : Nat}
    {pre : BlockState → Prop} {F : Nat → ℝ} (hB : 0 < B) (hcov : n ≤ g * B)
    (s : BlockState) (hrun : ProgramRuns k R n B pre F)
    (hpre : ∀ idx : GridIndex (line g), pre (s.withGridIndex idx)) :
    ∃ sF, Nonempty (Kernel.GridLaunchedOrdinary k (line g) s sF) ∧
      (∀ o, o < n → sF.readMem R o = F o) ∧
      (∀ r o, ¬ (r = R ∧ o < n) → sF.mem r o = s.mem r o) := by
  classical
  have hex : ∀ idx : GridIndex (line g), ∃ f, exec k (s.withGridIndex idx) = some f ∧
      (∀ o, o < n → o / B = pidOf idx → f.readMem R o = F o) ∧
      (∀ r o, ¬ blockWrites R n B (pidOf idx) (r, o) →
        f.mem r o = (s.withGridIndex idx).mem r o) := by
    intro idx
    obtain ⟨f, h1, h2, h3⟩ := hrun _ (hpre idx)
    rw [withGridIndex_pid_line] at h2 h3
    exact ⟨f, h1, h2, h3⟩
  let frames : Kernel.GridFrames k (line g) s := fun idx =>
    { final := Classical.choose (hex idx)
      writes := blockWrites R n B (pidOf idx)
      h_exec := (Classical.choose_spec (hex idx)).1
      h_writeWithin := fun r o hno => ((Classical.choose_spec (hex idx)).2.2 r o hno).symm }
  obtain ⟨L, -, hout, hframe⟩ := launch_of_frames hB hcov frames (fun _ => rfl) F
    (fun idx j hj hn => (Classical.choose_spec (hex idx)).2.1 _ hn (by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hj, Nat.zero_add]))
  exact ⟨_, ⟨L⟩, hout, hframe⟩

/-- **Fused vs two-launch pipeline.** Under the per-program runs of the three
kernels (a temporary region `T`, an output region `O`), the two-launch
pipeline (`kA` then `kB`) and the fused launch `kF`, all over the same `(g,)`
grid, end in states that agree on every output element `o < n` and on every
cell outside `T[0, n)`. -/
theorem fused_agrees_two_launch {kA kB kF : Kernel} {T O : RegionName}
    {n B g : Nat} (hB : 0 < B) (hcov : n ≤ g * B)
    {preA preF : BlockState → Prop} {F : Nat → ℝ} {G : ℝ → ℝ} (s : BlockState)
    (hA : ProgramRuns kA T n B preA F)
    (hpA : ∀ idx : GridIndex (line g), preA (s.withGridIndex idx))
    (hBk : ProgramRuns kB O n B (fun t => ∀ o, o < n → t.readMem T o = F o) (fun o => G (F o)))
    (hF : ProgramRuns kF O n B preF (fun o => G (F o)))
    (hpF : ∀ idx : GridIndex (line g), preF (s.withGridIndex idx)) :
    ∃ sT sU sF,
      Nonempty (Kernel.GridLaunchedOrdinary kA (line g) s sT) ∧
      Nonempty (Kernel.GridLaunchedOrdinary kB (line g) sT sU) ∧
      Nonempty (Kernel.GridLaunchedOrdinary kF (line g) s sF) ∧
      (∀ o, o < n → sF.readMem O o = sU.readMem O o) ∧
      (∀ o, o < n → sF.readMem O o = G (F o)) ∧
      (∀ r o, ¬ (r = T ∧ o < n) → ¬ (r = O ∧ o < n) → sF.mem r o = sU.mem r o) := by
  obtain ⟨sT, hLT, hTv, hTf⟩ := launch_of_programRuns hB hcov s hA hpA
  obtain ⟨sU, hLU, hUv, hUf⟩ := launch_of_programRuns (pre := fun t => ∀ o, o < n → t.readMem T o = F o)
    hB hcov sT hBk (fun idx o ho => by
      simp only [BlockState.readMem, BlockState.withGridIndex_mem]
      exact hTv o ho)
  obtain ⟨sF, hLF, hFv, hFf⟩ := launch_of_programRuns hB hcov s hF hpF
  refine ⟨sT, sU, sF, hLT, hLU, hLF, fun o ho => by rw [hFv o ho, hUv o ho], hFv, ?_⟩
  intro r o hT hO
  rw [hFf r o hO, hUf r o hO, hTf r o hT]

end Blocked1D

end VeriTile.Triton
