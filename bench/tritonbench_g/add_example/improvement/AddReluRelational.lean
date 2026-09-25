import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# Fused add + ReLU versus the two-launch pipeline — relational theorem

The fused kernel `add_relu_kernel` (text identical to `AddReluFused.lean`),
and the unfused pipeline it replaces: `add_kernel` (text identical to
`AddExample.lean`, i.e. `add_example.py` / `add_example_block64.py`) writing a
private temporary `T`, followed by the masked-pointer `relu_kernel`
(`relu_masked.py`, the ReLU spelled as `relu_strided_buffer`'s
`relu_forward`) reading `T` and writing the output `O`. Bench files never
import one another, so the kernels are restated here; unit tests
(`scripts/test_launch_check.py`) require the three transcriptions to equal
their sources statement for statement.

`add_relu_fusion_relational`: for a checked launch configuration, from any
state whose inputs hold `xs`, `ys` on `[0, n)`, the pipeline and the fused
launch end in states that agree on every output element and on every cell
outside the temporary's window; every output element is
`relu (xs o + ys o)`. Per-program runs are proved here by the same unrolling
as the source files; composition is the kernel-agnostic
`Blocked1D.fused_agrees_two_launch`.

Not covered: the strided block-pointer ReLU pipeline benchmarked in stage 8
(related only at the specification level); floating point (exact-ℝ model).
-/

namespace VeriTile.Bench.TritonBenchG.AddReluRelational

open VeriTile.Triton VeriTile.Examples

/-- Transcription of `add_relu_fused.py`'s `add_relu_kernel`. -/
def add_relu_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  z = x + y
  output = tl.where(z > 0, z, 0)
  tl.store(out_ptr + offsets, output, mask=mask)
}

/-- Transcription of `add_example.py`'s `add_kernel`. -/
def add_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
}

/-- Transcription of `relu_masked.py`'s `relu_kernel`. -/
def relu_kernel
    (in_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr + offsets, mask=mask)
  output = tl.where(x > 0, x, 0)
  tl.store(out_ptr + offsets, output, mask=mask)
}

/-- Algorithm-layer lane readback: active lanes hold `relu (xs i + ys i)`,
inactive lanes keep their initial value. -/
theorem add_relu_kernel_correct
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs)
    (h_y : InputLoadedAt s in_ptr1 BLOCK_SIZE ys) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE) s)
                out_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then TiledActivation.relu (xs i + ys i)
                else s.readMem out_ptr addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, add_relu_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x h_y
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x, h_y]
    split
    · rename_i hc
      have h : (0 : ℝ) < xs i + ys i := WithBot.coe_lt_coe.mp hc
      simp [TiledActivation.relu, max_eq_right h.le]
    · rename_i hc
      have h : ¬ (0 : ℝ) < xs i + ys i := fun h0 => hc (WithBot.coe_lt_coe.mpr h0)
      simp [TiledActivation.relu, max_eq_left (not_lt.mp h)]
  · simp [hi]


set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `out_ptr`, and the *inactive* lanes of the output window itself. -/
private theorem add_relu_kernel_frame
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(out_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, add_relu_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- Region-model masked Hoare triple (the `hrun` obligation). -/
theorem add_relu_kernel_region_run
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem in_ptr0 (s₀.pid * BLOCK_SIZE + j.val) = xs j)
    (hy : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem in_ptr1 (s₀.pid * BLOCK_SIZE + j.val) = ys j) :
    ∃ s1, exec ((add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem out_ptr (s₀.pid * BLOCK_SIZE + j.val)
            = TiledActivation.relu (xs j + ys j))
      ∧ (∀ r o,
          (r ≠ out_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := add_relu_kernel_correct in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
    s₀ (fun j => s₀.readMem in_ptr0 (s₀.pid * BLOCK_SIZE + j.val))
    (fun j => s₀.readMem in_ptr1 (s₀.pid * BLOCK_SIZE + j.val))
    (fun _ => rfl) (fun _ => rfl)
  rw [show exec (add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE) s₀
      = exec ((add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements
          BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements
      BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, add_relu_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj, hy j hj]
      · refine add_relu_kernel_frame in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm


/-- Algorithm-layer correctness for `add_kernel`.

For each lane `i ∈ Fin BLOCK_SIZE`:
* In-bounds (`pid * BLOCK_SIZE + i < n_elements`): the output region holds
  `xs i + ys i`.
* Out-of-bounds: the value at `pid * BLOCK_SIZE + i` is preserved from the
  initial state (mask=false → no store).

No region-disjointness hypothesis: the kernel reads both inputs into local
registers BEFORE the scatter to `out_ptr`, so the result is correct even when
`out_ptr` aliases `in_ptr0` or `in_ptr1`. -/
theorem add_kernel_correct
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs)
    (h_y : InputLoadedAt s in_ptr1 BLOCK_SIZE ys) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE) s)
                out_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then xs i + ys i
                else s.readMem out_ptr addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, add_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x h_y
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x, h_y]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `out_ptr`, and the *inactive* lanes of the output window itself. -/
private theorem add_kernel_frame
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(out_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, add_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input windows are loaded at the **active lanes only**. This is the `hrun`
obligation of `MaskedKernelIO₂.Implements.intro`; the value half reuses
`add_kernel_correct` (instantiated at the tiles the state actually holds). -/
theorem add_kernel_region_run
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem in_ptr0 (s₀.pid * BLOCK_SIZE + j.val) = xs j)
    (hy : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem in_ptr1 (s₀.pid * BLOCK_SIZE + j.val) = ys j) :
    ∃ s1, exec ((add_kernel in_ptr0 in_ptr1 out_ptr n_elements
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem out_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j + ys j)
      ∧ (∀ r o,
          (r ≠ out_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := add_kernel_correct in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
    s₀ (fun j => s₀.readMem in_ptr0 (s₀.pid * BLOCK_SIZE + j.val))
    (fun j => s₀.readMem in_ptr1 (s₀.pid * BLOCK_SIZE + j.val))
    (fun _ => rfl) (fun _ => rfl)
  rw [show exec (add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE) s₀
      = exec ((add_kernel in_ptr0 in_ptr1 out_ptr n_elements
          BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((add_kernel in_ptr0 in_ptr1 out_ptr n_elements
      BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, add_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj, hy j hj]
      · refine add_kernel_frame in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

/-- Algorithm-layer lane readback: active lanes hold `relu (xs i)`,
inactive lanes keep their initial value. -/
theorem relu_kernel_correct
    (in_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (relu_kernel in_ptr out_ptr n_elements BLOCK_SIZE) s)
                out_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then TiledActivation.relu (xs i)
                else s.readMem out_ptr addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, relu_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
    split
    · rename_i hc
      have h : (0 : ℝ) < xs i := WithBot.coe_lt_coe.mp hc
      simp [TiledActivation.relu, max_eq_right h.le]
    · rename_i hc
      have h : ¬ (0 : ℝ) < xs i := fun h0 => hc (WithBot.coe_lt_coe.mpr h0)
      simp [TiledActivation.relu, max_eq_left (not_lt.mp h)]
  · simp [hi]


set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `out_ptr`, and the *inactive* lanes of the output window itself. -/
private theorem relu_kernel_frame
    (in_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((relu_kernel in_ptr out_ptr n_elements BLOCK_SIZE
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(out_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, relu_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- Region-model masked Hoare triple (the `hrun` obligation). -/
theorem relu_kernel_region_run
    (in_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem in_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j) :
    ∃ s1, exec ((relu_kernel in_ptr out_ptr n_elements
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem out_ptr (s₀.pid * BLOCK_SIZE + j.val)
            = TiledActivation.relu (xs j))
      ∧ (∀ r o,
          (r ≠ out_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := relu_kernel_correct in_ptr out_ptr n_elements BLOCK_SIZE
    s₀ (fun j => s₀.readMem in_ptr (s₀.pid * BLOCK_SIZE + j.val))
    (fun _ => rfl)
  rw [show exec (relu_kernel in_ptr out_ptr n_elements BLOCK_SIZE) s₀
      = exec ((relu_kernel in_ptr out_ptr n_elements
          BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((relu_kernel in_ptr out_ptr n_elements
      BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, relu_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj]
      · refine relu_kernel_frame in_ptr out_ptr n_elements BLOCK_SIZE
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm


/-- Per-program run of `add_kernel`: writes `xs o + ys o` into `T`. -/
theorem add_kernel_programRuns (x y T : RegionName) (n B : Nat) (hB : 0 < B) (xs ys : Nat → ℝ) :
    Blocked1D.ProgramRuns ((add_kernel x y T n B).toAlgKernel) T n B
      (fun t => ∀ o, o < n → t.readMem x o = xs o ∧ t.readMem y o = ys o)
      (fun o => xs o + ys o) := by
  intro t ht
  obtain ⟨f, hexec, hvals, hframe⟩ := add_kernel_region_run x y T n B t
    (fun j => xs (t.pid * B + j.val)) (fun j => ys (t.pid * B + j.val))
    (fun j hj => (ht _ hj).1) (fun j hj => (ht _ hj).2)
  refine ⟨f, hexec, fun o ho hd => ?_, fun r o hno => ?_⟩
  · have hdm := Nat.div_add_mod' o B
    rw [hd] at hdm
    have := hvals ⟨o % B, Nat.mod_lt o hB⟩ (show t.pid * B + o % B < n by omega)
    simp only at this
    rw [hdm] at this
    exact this
  · apply hframe
    by_cases hr : r = T
    · refine Or.inr fun j hj ho => hno ?_
      rw [Blocked1D.blockWrites_iff hB]
      refine ⟨hr, by omega, ?_⟩
      rw [ho, Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt j.isLt, Nat.zero_add]
    · exact Or.inl hr

/-- Per-program run of `relu_kernel`: reads `T`, writes `relu` into `O`. -/
theorem relu_kernel_programRuns (T O : RegionName) (n B : Nat) (hB : 0 < B) (F : Nat → ℝ) :
    Blocked1D.ProgramRuns ((relu_kernel T O n B).toAlgKernel) O n B
      (fun t => ∀ o, o < n → t.readMem T o = F o)
      (fun o => TiledActivation.relu (F o)) := by
  intro t ht
  obtain ⟨f, hexec, hvals, hframe⟩ := relu_kernel_region_run T O n B t
    (fun j => F (t.pid * B + j.val))
    (fun j hj => ht _ hj)
  refine ⟨f, hexec, fun o ho hd => ?_, fun r o hno => ?_⟩
  · have hdm := Nat.div_add_mod' o B
    rw [hd] at hdm
    have := hvals ⟨o % B, Nat.mod_lt o hB⟩ (show t.pid * B + o % B < n by omega)
    simp only at this
    rw [hdm] at this
    exact this
  · apply hframe
    by_cases hr : r = O
    · refine Or.inr fun j hj ho => hno ?_
      rw [Blocked1D.blockWrites_iff hB]
      refine ⟨hr, by omega, ?_⟩
      rw [ho, Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt j.isLt, Nat.zero_add]
    · exact Or.inl hr

/-- Per-program run of the fused kernel. -/
theorem add_relu_kernel_programRuns (x y O : RegionName) (n B : Nat) (hB : 0 < B) (xs ys : Nat → ℝ) :
    Blocked1D.ProgramRuns ((add_relu_kernel x y O n B).toAlgKernel) O n B
      (fun t => ∀ o, o < n → t.readMem x o = xs o ∧ t.readMem y o = ys o)
      (fun o => TiledActivation.relu (xs o + ys o)) := by
  intro t ht
  obtain ⟨f, hexec, hvals, hframe⟩ := add_relu_kernel_region_run x y O n B t
    (fun j => xs (t.pid * B + j.val)) (fun j => ys (t.pid * B + j.val))
    (fun j hj => (ht _ hj).1) (fun j hj => (ht _ hj).2)
  refine ⟨f, hexec, fun o ho hd => ?_, fun r o hno => ?_⟩
  · have hdm := Nat.div_add_mod' o B
    rw [hd] at hdm
    have := hvals ⟨o % B, Nat.mod_lt o hB⟩ (show t.pid * B + o % B < n by omega)
    simp only at this
    rw [hdm] at this
    exact this
  · apply hframe
    by_cases hr : r = O
    · refine Or.inr fun j hj ho => hno ?_
      rw [Blocked1D.blockWrites_iff hB]
      refine ⟨hr, by omega, ?_⟩
      rw [ho, Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt j.isLt, Nat.zero_add]
    · exact Or.inl hr

/-- **Fused versus two-launch pipeline.** For a launch configuration accepted
by `Blocked1DLaunch.check` (the same `n`, block and grid for all three
launches), any temporary `T` and output `O`, and a launch state
whose inputs hold `xs`, `ys` on `[0, n)`: the pipeline `add_kernel` (into `T`)
then `relu_kernel` (from `T` into `O`), and the fused `add_relu_kernel` (into
`O`), each launched over `c.grid`, end in states `sU` and `sF` with

1. `sF.readMem O o = sU.readMem O o` for every `o < n`;
2. `sF.readMem O o = relu (xs o + ys o)` for every `o < n`;
3. `sF.mem r o = sU.mem r o` for every cell outside `T[0, n)` and `O[0, n)`
   (the temporary's window is the only place the two may differ). -/
specification add_relu_fusion_relational
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (x y T O : RegionName)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ o, o < c.n → s.readMem x o = xs o) (hy : ∀ o, o < c.n → s.readMem y o = ys o) :
    ∃ sT sU sF,
      Nonempty (Kernel.GridLaunchedOrdinary ((add_kernel x y T c.n c.block).toAlgKernel)
        { dims := c.grid } s sT) ∧
      Nonempty (Kernel.GridLaunchedOrdinary ((relu_kernel T O c.n c.block).toAlgKernel)
        { dims := c.grid } sT sU) ∧
      Nonempty (Kernel.GridLaunchedOrdinary ((add_relu_kernel x y O c.n c.block).toAlgKernel)
        { dims := c.grid } s sF) ∧
      (∀ o, o < c.n → sF.readMem O o = sU.readMem O o) ∧
      (∀ o, o < c.n → sF.readMem O o = TiledActivation.relu (xs o + ys o)) ∧
      (∀ r o, ¬ (r = T ∧ o < c.n) → ¬ (r = O ∧ o < c.n) → sF.mem r o = sU.mem r o) := by
  have hpre := Blocked1DLaunch.check_ok c hc
  have hB := hpre.block_pos
  have hcov : c.n ≤ c.gridX * c.block :=
    of_decide_eq_true ((Blocked1DLaunch.coversB_iff c hB).2 hpre.covers)
  rw [hpre.grid_eq]
  have hin : ∀ idx : GridIndex (Blocked1D.line c.gridX), ∀ o, o < c.n →
      (s.withGridIndex idx).readMem x o = xs o ∧ (s.withGridIndex idx).readMem y o = ys o :=
    fun idx o ho => ⟨by simpa [BlockState.readMem] using hx o ho,
      by simpa [BlockState.readMem] using hy o ho⟩
  exact Blocked1D.fused_agrees_two_launch (G := TiledActivation.relu) hB hcov s
    (add_kernel_programRuns x y T c.n c.block hB xs ys) hin
    (relu_kernel_programRuns T O c.n c.block hB _)
    (add_relu_kernel_programRuns x y O c.n c.block hB xs ys) hin

end VeriTile.Bench.TritonBenchG.AddReluRelational
