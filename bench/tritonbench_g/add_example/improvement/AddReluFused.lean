import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# Fused float32 add + ReLU — candidate kernel (not a corpus port)

`improvement/add_relu_fused.py`'s `add_relu_kernel` is `add_example.py`'s
`add_kernel` with one inserted line: the sum `z = x + y` is passed through the
ReLU spelled exactly as `relu_strided_buffer.py`'s `relu_forward`
(`tl.where(z > 0, z, 0)`) before the masked store. `add_relu_wrapper` has
`add_example`'s launch arithmetic (`BLOCK_SIZE = 64`, `cdiv` grid over
`x.numel()`) with an `empty_like` output.

## What is proved

* per program, on the flat-memory IO surface (`⊨`): every active output lane
  holds `relu (xs i + ys i)` and every other flat cell is unchanged
  (`add_relu_kernel_correctness`);
* for the wrapper, from the tensors' own metadata, under the **unchanged**
  `Elementwise2` contract (`Elementwise2.check 64`): complete output coverage,
  the logical view, trace safety within each tensor's extent, serial-order
  independence, and the flat placement at the tensors' element addresses
  (`add_relu_wrapper_correctness`).

The whole-wrapper proof reuses the library machinery (`Elementwise2.*`,
`Blocked1D.launch_of_frames`, `launch_of_frames_addr`,
`Kernel.runSerial_agrees_merge`, `FlatAlloc`); the kernel-specific glue follows
`AddExample.lean` line for line (bench files do not import each other).

## Transformation

The fused spec `relu (xs i + ys i)` is the composition of the two proven
specs: `add_wrapper_correctness` (`AddExample.lean`: element `i` of the sum is
`xs i + ys i`) and `relu_wrapper_one_tile_correctness`
(`ReluStridedBuffer.lean`: element `k` of the output is `relu` of element `k`
of its input). So on every logical element the fused wrapper returns the value
the unfused pipeline (add into a temporary, then ReLU) returns under those
theorems' hypotheses, and its write footprint is the output alone (the unfused
pipeline also writes the temporary). The memory-level sequential composition of
the two unfused launches is not formalized here.

## Modeling boundary

Exact-ℝ (`⊨`, not `⊨[R]`). On hardware both pipelines round once, at the f32
add (`fl(x + y)`); the unfused pipeline stores that f32 value and reloads it
unchanged, and the ReLU is a select, so the two are expected to be bitwise
equal — checked on tensors, not proved. NaN is outside the model:
`tl.where(z > 0, z, 0)` yields `+0.0` for NaN.
-/

namespace VeriTile.Bench.TritonBenchG.AddReluFused

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Transcription of `add_relu_fused.py`'s `add_relu_kernel` (the Python
`BLOCK_SIZE: tl.constexpr` annotation becomes a `Nat` parameter). -/
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


/-- Per-execution safety walk (lane-wise bounds contract). -/
theorem add_relu_kernel_traceSafe
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds in_ptr0)
    (hy : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds in_ptr1)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds out_ptr) :
    Kernel.TraceSafe bounds
      ((add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all eight statements, discharging every
  -- load-free `SafeAt` and reducing the three memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [add_relu_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hy a ha, fun a ha => hout a ha⟩


/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem add_relu_kernel_flattenOk
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ((add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [add_relu_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]


/-- Masked IO signature: `add_example`'s windows and mask. -/
def addReluIO (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
  in1 := in_ptr0
  in2 := in_ptr1
  out := out_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements

/-- **Per-program headline (flat memory):** every active output lane holds
`relu (xs i + ys i)`; every other flat cell is unchanged. -/
specification add_relu_kernel_correctness
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    addReluIO in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs ys i => TiledActivation.relu (xs i + ys i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact add_relu_kernel_flattenOk in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
  · intro bounds s h1 h2 h3 _
    exact add_relu_kernel_traceSafe in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ := add_relu_kernel_region_run
      in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE s₀ xs ys hx hy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩


/-- Progress: `add_relu_kernel` always executes to a defined state. -/
theorem add_relu_kernel_exec_isSome
    (in_ptr0 in_ptr1 out_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) :
    (exec ((add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ).toAlgKernel) s).isSome := by
  simp [exec, add_relu_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt]

/-- The framed execution of program `idx` of a `(g,)` launch. -/
noncomputable def addReluLaunchFrame
    (in_ptr0 in_ptr1 out_ptr : RegionName) (n_elements BLOCK_SIZE g : Nat)
    (s : BlockState) (idx : GridIndex (Blocked1D.line g)) :
    Kernel.ExecFrame ((add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements
      BLOCK_SIZE).toAlgKernel) (s.withGridIndex idx) where
  final := (exec ((add_relu_kernel in_ptr0 in_ptr1 out_ptr n_elements
      BLOCK_SIZE).toAlgKernel) (s.withGridIndex idx)).get
    (add_relu_kernel_exec_isSome _ _ _ _ _ _)
  writes := Blocked1D.blockWrites out_ptr n_elements BLOCK_SIZE (Blocked1D.pidOf idx)
  h_exec := (Option.some_get _).symm
  h_writeWithin := by
    intro r o hno
    refine (add_relu_kernel_frame in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      (s.withGridIndex idx) _ (Option.some_get _).symm r o ?_).symm
    intro i hi hc
    apply hno
    rw [Blocked1D.withGridIndex_pid_line] at hi hc
    unfold Blocked1D.blockWrites WriteFootprint.activeTileImage
    exact ⟨hc.1.symm, (i, PUnit.unit), hi, hc.2⟩

/-- The framed whole-grid launch of a checked configuration (the first
conjunct of `add_relu_kernel_launch_correctness`; needs no input-buffer binding). -/
theorem add_relu_kernel_launch_framed
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < c.n → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < c.n → s.mem in_ptr1 i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((add_relu_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
        { dims := c.grid } s
        (fun i : Nat => if i < c.n then some (out_ptr, i) else none)
        (fun i => TiledActivation.relu (xs i + ys i)) := by
  have hpre := Blocked1DLaunch.check_ok c hc
  have hB := hpre.block_pos
  have hcov : c.n ≤ c.gridX * c.block :=
    of_decide_eq_true ((Blocked1DLaunch.coversB_iff c hB).2 hpre.covers)
  rw [hpre.grid_eq]
  let frames : Kernel.GridFrames
      ((add_relu_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
      (Blocked1D.line c.gridX) s :=
    fun idx => addReluLaunchFrame in_ptr0 in_ptr1 out_ptr c.n c.block c.gridX s idx
  have hval : ∀ idx j, j < c.block → Blocked1D.pidOf idx * c.block + j < c.n →
      (frames idx).final.readMem out_ptr (Blocked1D.pidOf idx * c.block + j)
        = TiledActivation.relu (xs (Blocked1D.pidOf idx * c.block + j)
          + ys (Blocked1D.pidOf idx * c.block + j)) := by
    intro idx j hj hn
    have hread : ∀ (r : RegionName) (v : Nat → ℝ),
        (∀ i, i < c.n → s.mem r i = MemCell.real (v i)) →
        ∀ l : Fin c.block,
          (s.withGridIndex idx).pid * c.block + l.val < c.n →
          (s.withGridIndex idx).readMem r ((s.withGridIndex idx).pid * c.block + l.val)
            = v (Blocked1D.pidOf idx * c.block + l.val) := by
      intro r v hv l hl
      rw [Blocked1D.withGridIndex_pid_line] at hl ⊢
      simp [BlockState.readMem, hv _ hl]
    obtain ⟨s1, hexec, hvals, -⟩ := add_relu_kernel_region_run in_ptr0 in_ptr1 out_ptr
      c.n c.block (s.withGridIndex idx)
      (fun l => xs (Blocked1D.pidOf idx * c.block + l.val))
      (fun l => ys (Blocked1D.pidOf idx * c.block + l.val))
      (hread in_ptr0 xs hx) (hread in_ptr1 ys hy)
    have hfin : (frames idx).final = s1 := by
      have h1 := (frames idx).h_exec
      rw [hexec] at h1
      exact (Option.some.inj h1).symm
    have := hvals ⟨j, hj⟩ (by rw [Blocked1D.withGridIndex_pid_line]; exact hn)
    rw [Blocked1D.withGridIndex_pid_line] at this
    rw [hfin]
    exact this
  obtain ⟨L, -, hout, hframe⟩ := Blocked1D.launch_of_frames hB hcov frames
    (fun _ => rfl) (fun i => TiledActivation.relu (xs i + ys i)) hval
  refine ⟨_, L, ?_, ?_⟩
  · intro i addr hw
    by_cases hi : i < c.n
    · simp only [hi, if_true, Option.some.injEq] at hw
      subst hw
      exact hout i hi
    · simp [hi] at hw
  · rintro ⟨r, o⟩ hno
    apply hframe
    rintro ⟨hr, ho⟩
    simp only at hr ho
    subst hr
    exact hno o (by simp [ho])

/-- One program of `add_relu_kernel`, from any state whose input cells below `n`
hold typed real values: it succeeds, writes `TiledActivation.relu (xs o + ys o)` at every active
output lane of its block, and changes no other cell. -/
theorem add_relu_kernel_program_run
    (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) (hB : 0 < B)
    (t : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < n → t.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < n → t.mem in_ptr1 i = MemCell.real (ys i)) :
    ∃ f, exec ((add_relu_kernel in_ptr0 in_ptr1 out_ptr n B).toAlgKernel) t = some f ∧
      (∀ o, o < n → o / B = t.pid → f.readMem out_ptr o = TiledActivation.relu (xs o + ys o)) ∧
      (∀ r o, ¬ Blocked1D.blockWrites out_ptr n B t.pid (r, o) → f.mem r o = t.mem r o) := by
  obtain ⟨f, hexec, hvals, hframe⟩ := add_relu_kernel_region_run in_ptr0 in_ptr1 out_ptr n B t
    (fun l => xs (t.pid * B + l.val)) (fun l => ys (t.pid * B + l.val))
    (fun l hl => by simp [BlockState.readMem, hx _ hl])
    (fun l hl => by simp [BlockState.readMem, hy _ hl])
  refine ⟨f, hexec, fun o ho hd => ?_, fun r o hno => ?_⟩
  · have hdm := Nat.div_add_mod' o B
    rw [hd] at hdm
    have := hvals ⟨o % B, Nat.mod_lt o hB⟩ (show t.pid * B + o % B < n by omega)
    simp only at this
    rw [hdm] at this
    exact this
  · apply hframe
    by_cases hr : r = out_ptr
    · refine Or.inr fun j hj ho => hno ?_
      rw [Blocked1D.blockWrites_iff hB]
      refine ⟨hr, by omega, ?_⟩
      rw [ho, Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt j.isLt, Nat.zero_add]
    · exact Or.inl hr

/-- **Serial orders.** Running the programs of a checked launch one after
another, in any complete duplicate-free order, writes `TiledActivation.relu (xs i + ys i)` at every
output index `i < n` and leaves every other cell unchanged — the same result
as the merge (`Kernel.runSerial_agrees_merge`), because no program reads a
cell another program writes. -/
theorem add_relu_kernel_launch_serial
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (in_ptr0 in_ptr1 out_ptr : RegionName) (h0 : out_ptr ≠ in_ptr0) (h1 : out_ptr ≠ in_ptr1)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < c.n → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < c.n → s.mem in_ptr1 i = MemCell.real (ys i)) :
    ∀ L : List (GridIndex { dims := c.grid }), L.Nodup → (∀ idx, idx ∈ L) →
      ∃ m, Kernel.runSerial ((add_relu_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
            s L s.mem = some m ∧
        (∀ i, i < c.n → Kernel.memReal m out_ptr i = TiledActivation.relu (xs i + ys i)) ∧
        (∀ r o, ¬ (r = out_ptr ∧ o < c.n) → m r o = s.mem r o) := by
  have hpre := Blocked1DLaunch.check_ok c hc
  have hB := hpre.block_pos
  have hcov : c.n ≤ c.gridX * c.block :=
    of_decide_eq_true ((Blocked1DLaunch.coversB_iff c hB).2 hpre.covers)
  rw [hpre.grid_eq]
  intro L hnd hall
  let frames : Kernel.GridFrames ((add_relu_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
      (Blocked1D.line c.gridX) s :=
    fun idx => addReluLaunchFrame in_ptr0 in_ptr1 out_ptr c.n c.block c.gridX s idx
  have hwr : ∀ idx, (frames idx).writes
      = Blocked1D.blockWrites out_ptr c.n c.block (Blocked1D.pidOf idx) := fun _ => rfl
  have hdisj : Kernel.GridWritesDisjoint frames := by
    intro i₁ i₂ hne
    rw [hwr i₁, hwr i₂]
    exact Blocked1D.blockWrites_disjoint hB fun h => hne (Blocked1D.idx_ext h)
  have hfinal : ∀ idx o, o < c.n → o / c.block = Blocked1D.pidOf idx →
      (frames idx).final.readMem out_ptr o = TiledActivation.relu (xs o + ys o) := by
    intro idx o ho hd
    obtain ⟨f, hf, hv, -⟩ := add_relu_kernel_program_run in_ptr0 in_ptr1 out_ptr c.n c.block hB
      (s.withGridIndex idx) xs ys (by simpa using hx) (by simpa using hy)
    have hfe : (frames idx).final = f := by
      have h1 := (frames idx).h_exec
      rw [hf] at h1
      exact (Option.some.inj h1).symm
    rw [hfe]
    exact hv o ho (by rw [Blocked1D.withGridIndex_pid_line]; exact hd)
  have hrob : Kernel.FrameRobust frames := by
    intro idx m hm
    have hin : ∀ r, r ≠ out_ptr → ∀ i, m r i = s.mem r i := by
      intro r hr i
      apply hm
      rintro ⟨idx', -, hw⟩
      rw [hwr, Blocked1D.blockWrites_iff hB] at hw
      exact hr hw.1
    obtain ⟨f, hf, hv, hfr⟩ := add_relu_kernel_program_run in_ptr0 in_ptr1 out_ptr c.n c.block hB
      (({ s with mem := m } : BlockState).withGridIndex idx) xs ys
      (fun i hi => by
        rw [BlockState.withGridIndex_mem]
        show m in_ptr0 i = _
        rw [hin _ (Ne.symm h0)]
        exact hx i hi)
      (fun i hi => by
        rw [BlockState.withGridIndex_mem]
        show m in_ptr1 i = _
        rw [hin _ (Ne.symm h1)]
        exact hy i hi)
    refine ⟨f, hf, fun r o hw => ?_, fun r o hnw => ?_⟩
    · rw [hwr, Blocked1D.blockWrites_iff hB] at hw
      obtain ⟨rfl, ho, hd⟩ := hw
      rw [hv o ho (by rw [Blocked1D.withGridIndex_pid_line]; exact hd), hfinal idx o ho hd]
    · rw [hwr] at hnw
      rw [hfr r o (by rw [Blocked1D.withGridIndex_pid_line]; exact hnw)]
      rfl
  obtain ⟨m, hm, hw, hnw⟩ := Kernel.runSerial_agrees_merge frames hdisj hrob L hnd hall
  obtain ⟨-, -, hout, hframe⟩ := Blocked1D.launch_of_frames hB hcov frames hwr
    (fun i => TiledActivation.relu (xs i + ys i))
    (fun idx j hj hn => hfinal idx _ hn (by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hj, Nat.zero_add]))
  refine ⟨m, hm, fun i hi => ?_, fun r o hno => ?_⟩
  · have hown : i / c.block < c.gridX := (Nat.div_lt_iff_lt_mul hB).2 (by omega)
    rw [hw out_ptr i ⟨Blocked1D.indexOf _ hown, by
      rw [hwr, Blocked1D.blockWrites_iff hB, Blocked1D.pidOf_indexOf]; exact ⟨rfl, hi, rfl⟩⟩]
    exact hout i hi
  · have hnot : ¬ Kernel.GridWriteFootprint frames (r, o) := by
      rintro ⟨idx, h⟩
      rw [hwr, Blocked1D.blockWrites_iff hB] at h
      exact hno ⟨h.1, h.2.1⟩
    rw [hnw r o hnot]
    exact hframe r o hno

/-- **Flat-memory whole launch.** For any flat placement of the three buffers
that satisfies the bridge's hypotheses (disjoint, closed, extents covering
the `n` accessed elements), the flattened launch of a checked configuration
writes `TiledActivation.relu (xs i + ys i)` at flat address `A.base out_ptr + i` for every `i < n`
and leaves every other flat cell unchanged. Per program this is the upstream
flat headline `add_relu_kernel_correctness` (`⊨`); `launch_of_frames_addr`
composes the programs. -/
theorem add_relu_kernel_launch_flat
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (in_ptr0 in_ptr1 out_ptr : RegionName) (A : FlatAlloc)
    (hd : A.Disjoint) (hreg : A.regions = [in_ptr0, in_ptr1, out_ptr])
    (hcl : ∀ r, r ∉ A.regions → A.extent r = 0)
    (he0 : c.n ≤ A.extent in_ptr0) (he1 : c.n ≤ A.extent in_ptr1)
    (heo : c.n ≤ A.extent out_ptr)
    (s : BlockState) (hu : s.undef = (fun _ _ => 0)) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < c.n → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < c.n → s.mem in_ptr1 i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
      (A.flattenKernel ((add_relu_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel))
      { dims := c.grid } (A.flattenState s)
      (fun i : Nat => if i < c.n then some (A.flat, A.base out_ptr + i) else none)
      (fun i => TiledActivation.relu (xs i + ys i)) := by
  have hpre := Blocked1DLaunch.check_ok c hc
  have hB := hpre.block_pos
  have hcov : c.n ≤ c.gridX * c.block :=
    of_decide_eq_true ((Blocked1DLaunch.coversB_iff c hB).2 hpre.covers)
  rw [hpre.grid_eq]
  have hI := add_relu_kernel_correctness in_ptr0 in_ptr1 out_ptr c.n c.block
  have hprog : ∀ idx : GridIndex (Blocked1D.line c.gridX), ∃ s',
      exec (A.flattenKernel ((add_relu_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel))
        ((A.flattenState s).withGridIndex idx) = some s' ∧
      (∀ j, j < c.block → Blocked1D.pidOf idx * c.block + j < c.n →
        s'.readMem A.flat (A.base out_ptr + (Blocked1D.pidOf idx * c.block + j))
          = TiledActivation.relu (xs (Blocked1D.pidOf idx * c.block + j) + ys (Blocked1D.pidOf idx * c.block + j))) ∧
      (∀ r o, ¬ Blocked1D.addrWrites A.flat (fun t => A.base out_ptr + t) c.n c.block
          (Blocked1D.pidOf idx) (r, o) →
        ((A.flattenState s).withGridIndex idx).mem r o = s'.mem r o) := by
    intro idx
    obtain ⟨s', hex, hval, hfr⟩ := hI A hd (by simp [addReluIO, hreg]) hcl (Blocked1D.pidOf idx)
      (fun j hj => by simp only [addReluIO] at hj ⊢; omega)
      (fun j hj => by simp only [addReluIO] at hj ⊢; omega)
      (fun j hj => by simp only [addReluIO] at hj ⊢; omega)
      (fun p hp => by simp [addReluIO] at hp)
      (fun j => xs (Blocked1D.pidOf idx * c.block + j.val))
      (fun j => ys (Blocked1D.pidOf idx * c.block + j.val))
      (s.withGridIndex idx) (Blocked1D.withGridIndex_pid_line s idx) (by simp [hu])
      (fun j hj => by
        simp only [addReluIO] at hj ⊢
        simp [BlockState.readMem, hx _ hj])
      (fun j hj => by
        simp only [addReluIO] at hj ⊢
        simp [BlockState.readMem, hy _ hj])
    rw [FlatAlloc.flattenState_withGridIndex]
    refine ⟨s', hex, fun j hj hn => ?_, fun r o hno => ?_⟩
    · have := hval ⟨j, hj⟩ hn
      simpa [addReluIO, FlatAlloc.addr] using this
    · refine (hfr r o ?_).symm
      by_cases hr : r = A.flat
      · refine Or.inr ⟨fun j hj ho => hno ?_, fun p hp => by simp [addReluIO] at hp⟩
        rw [Blocked1D.addrWrites_iff hB]
        have hj' : j.val < c.block := j.isLt
        refine ⟨hr, Blocked1D.pidOf idx * c.block + j.val, hj, ?_, ?_⟩
        · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hj', Nat.zero_add]
        · simp only [addReluIO, FlatAlloc.addr] at ho; exact ho.symm
      · exact Or.inl hr
  classical
  let frames : Kernel.GridFrames
      (A.flattenKernel ((add_relu_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel))
      (Blocked1D.line c.gridX) (A.flattenState s) := fun idx =>
    { final := Classical.choose (hprog idx)
      writes := Blocked1D.addrWrites A.flat (fun t => A.base out_ptr + t) c.n c.block
        (Blocked1D.pidOf idx)
      h_exec := (Classical.choose_spec (hprog idx)).1
      h_writeWithin := fun r o hno => (Classical.choose_spec (hprog idx)).2.2 r o hno }
  obtain ⟨L, -, hout, hframe⟩ := Blocked1D.launch_of_frames_addr hB hcov
    (fun t => A.base out_ptr + t) (fun a b h => by simp only at h; omega) frames (fun _ => rfl)
    (fun i => TiledActivation.relu (xs i + ys i)) (fun idx j hj hn => (Classical.choose_spec (hprog idx)).2.1 j hj hn)
  refine ⟨_, L, ?_, ?_⟩
  · intro i addr hw
    by_cases hi : i < c.n
    · simp only [hi, if_true, Option.some.injEq] at hw
      subst hw
      exact hout i hi
    · simp [hi] at hw
  · rintro ⟨r, o⟩ hno
    apply hframe
    rintro ⟨hr, i, hi, ho⟩
    simp only at hr ho
    subst hr
    exact hno i (by simp [hi, ho])

/-- **Whole-wrapper headline for every block size.** As
`add_relu_wrapper_correctness`, for any `BLOCK_SIZE = B` the unchanged
`Elementwise2` contract accepts (`Elementwise2.check B x y out = true`; the
contract's P2/P5–P7 constrain `B`). A checked API may therefore launch the fused
kernel with any block it decides this check for. -/
specification add_relu_wrapper_correctness_block
    (B : Nat)
    (x y out : TensorMeta) (hc : Elementwise2.check B x y out = Bool.true)
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (h0 : out_ptr ≠ in_ptr0) (h1 : out_ptr ≠ in_ptr1)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < x.numel → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < y.numel → s.mem in_ptr1 i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((add_relu_kernel in_ptr0 in_ptr1 out_ptr x.numel B).toAlgKernel)
        { dims := (Elementwise2.launch B x y out).grid } s
        (fun i : Nat => if i < out.numel then some (out_ptr, i) else none)
        (fun i => TiledActivation.relu (xs i + ys i)) ∧
      (∀ idx, TensorMeta.InShape out.shape idx →
        out.offsetOf idx = x.offsetOf idx ∧ y.offsetOf idx = x.offsetOf idx ∧
          x.offsetOf idx < out.numel) ∧
      (∀ bounds : RegionBounds,
        x.numel ≤ bounds in_ptr0 → y.numel ≤ bounds in_ptr1 → out.numel ≤ bounds out_ptr →
        ∀ idx : GridIndex { dims := (Elementwise2.launch B x y out).grid },
          Kernel.TraceSafe bounds
            ((add_relu_kernel in_ptr0 in_ptr1 out_ptr x.numel B).toAlgKernel)
            (s.withGridIndex idx)) ∧
      (∀ L : List (GridIndex { dims := (Elementwise2.launch B x y out).grid }),
        L.Nodup → (∀ idx, idx ∈ L) →
        ∃ m, Kernel.runSerial ((add_relu_kernel in_ptr0 in_ptr1 out_ptr x.numel B).toAlgKernel)
              s L s.mem = some m ∧
          (∀ i, i < out.numel → Kernel.memReal m out_ptr i = TiledActivation.relu (xs i + ys i)) ∧
          (∀ r o, ¬ (r = out_ptr ∧ o < out.numel) → m r o = s.mem r o)) ∧
      (∀ flat : RegionName, (in_ptr0 = in_ptr1 ↔ x.base = y.base) →
        s.undef = (fun _ _ => 0) →
        (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).Disjoint ∧
        (∀ r, r ∉ (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).regions →
          (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).extent r = 0) ∧
        Kernel.LaunchCorrectFramed
          ((Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).flattenKernel
            ((add_relu_kernel in_ptr0 in_ptr1 out_ptr x.numel B).toAlgKernel))
          { dims := (Elementwise2.launch B x y out).grid }
          ((Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).flattenState s)
          (fun i : Nat => if i < out.numel then
            some (flat, out.base / out.elemBytes + i) else none)
          (fun i => TiledActivation.relu (xs i + ys i))) := by
  have hpre := Elementwise2.check_ok B x y out hc
  have hL := hpre.launch
  have hc' : Blocked1DLaunch.check (Elementwise2.launch B x y out) = Bool.true :=
    Blocked1DLaunch.check_complete _ hL
  have hW1 : x.numel = out.numel := hpre.output_covered
  have hyn : x.numel = y.numel := by simp [TensorMeta.numel, hpre.same_shape.1]
  have hx' : ∀ i, i < (Elementwise2.launch B x y out).n →
      s.mem in_ptr0 i = MemCell.real (xs i) := hx
  have hy' : ∀ i, i < (Elementwise2.launch B x y out).n →
      s.mem in_ptr1 i = MemCell.real (ys i) := fun i hi =>
    hy i (by change i < x.numel at hi; omega)
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [← hW1]
    exact add_relu_kernel_launch_framed _ hc' in_ptr0 in_ptr1 out_ptr s xs ys hx' hy'
  · intro idx hi
    obtain ⟨ss1, ss2⟩ := hpre.same_shape
    obtain ⟨cx, cy, co⟩ := hpre.contiguous
    have ho := TensorMeta.offsetOf_eq_linear out co idx hi
    have hix : TensorMeta.InShape x.shape idx := by rw [← ss2]; exact hi
    have hiy : TensorMeta.InShape y.shape idx := by rw [ss1, ← ss2]; exact hi
    have hxl := TensorMeta.offsetOf_eq_linear x cx idx hix
    have hyl := TensorMeta.offsetOf_eq_linear y cy idx hiy
    rw [ss2] at ho
    rw [ss1] at hyl
    refine ⟨ho.1.trans hxl.1.symm, hyl.1.trans hxl.1.symm, ?_⟩
    rw [hxl.1, ← hW1]
    exact hxl.2
  · intro bounds b0 b1 b2
    rw [hL.grid_eq]
    intro idx
    apply add_relu_kernel_traceSafe <;> intro j hj <;>
      rw [Blocked1D.withGridIndex_pid_line] at hj ⊢ <;> omega
  · rw [← hW1]
    exact add_relu_kernel_launch_serial _ hc' in_ptr0 in_ptr1 out_ptr h0 h1 s xs ys hx' hy'
  · intro flat hnames hu
    refine ⟨hpre.flat_disjoint flat in_ptr0 in_ptr1 out_ptr h0 h1 hnames,
      Elementwise2.flatAlloc_closed flat in_ptr0 in_ptr1 out_ptr x y out, ?_⟩
    have hbase : (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).base out_ptr
        = out.base / out.elemBytes := by simp [Elementwise2.flatAlloc]
    have hflat := add_relu_kernel_launch_flat _ hc' in_ptr0 in_ptr1 out_ptr
      (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out)
      (hpre.flat_disjoint flat in_ptr0 in_ptr1 out_ptr h0 h1 hnames) rfl
      (Elementwise2.flatAlloc_closed flat in_ptr0 in_ptr1 out_ptr x y out)
      (by simp [Elementwise2.flatAlloc, Elementwise2.launch])
      (by simp [Elementwise2.flatAlloc, Elementwise2.launch])
      (by simp [Elementwise2.flatAlloc, Elementwise2.launch])
      s hu xs ys hx' hy'
    rw [← hW1, ← hbase]
    exact hflat


/-- **Whole-wrapper headline (fused add + ReLU).** For tensors accepted by the
unchanged `Elementwise2` contract at `BLOCK_SIZE = 64`, region names that follow
the allocations, and inputs holding typed real values:

1. every element `i < out.numel` of the returned tensor holds
   `relu (xs i + ys i)`, and every other cell is unchanged;
2. the three tensors address the same row-major element for every in-shape
   multi-index (the logical view);
3. every program is trace-safe for bounds equal to the tensors' element counts;
4. any complete serial order of the programs gives the same result;
5. in the flat memory placed at the tensors' element addresses, the checked
   conditions discharge the bridge's hypotheses and the flattened launch writes
   `relu (xs i + ys i)` at `out.base / out.elemBytes + i` for every
   `i < out.numel`, leaving every other flat cell unchanged. -/
specification add_relu_wrapper_correctness
    (x y out : TensorMeta) (hc : Elementwise2.check 64 x y out = Bool.true)
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (h0 : out_ptr ≠ in_ptr0) (h1 : out_ptr ≠ in_ptr1)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < x.numel → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < y.numel → s.mem in_ptr1 i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((add_relu_kernel in_ptr0 in_ptr1 out_ptr x.numel 64).toAlgKernel)
        { dims := (Elementwise2.launch 64 x y out).grid } s
        (fun i : Nat => if i < out.numel then some (out_ptr, i) else none)
        (fun i => TiledActivation.relu (xs i + ys i)) ∧
      (∀ idx, TensorMeta.InShape out.shape idx →
        out.offsetOf idx = x.offsetOf idx ∧ y.offsetOf idx = x.offsetOf idx ∧
          x.offsetOf idx < out.numel) ∧
      (∀ bounds : RegionBounds,
        x.numel ≤ bounds in_ptr0 → y.numel ≤ bounds in_ptr1 → out.numel ≤ bounds out_ptr →
        ∀ idx : GridIndex { dims := (Elementwise2.launch 64 x y out).grid },
          Kernel.TraceSafe bounds
            ((add_relu_kernel in_ptr0 in_ptr1 out_ptr x.numel 64).toAlgKernel)
            (s.withGridIndex idx)) ∧
      (∀ L : List (GridIndex { dims := (Elementwise2.launch 64 x y out).grid }),
        L.Nodup → (∀ idx, idx ∈ L) →
        ∃ m, Kernel.runSerial ((add_relu_kernel in_ptr0 in_ptr1 out_ptr x.numel 64).toAlgKernel)
              s L s.mem = some m ∧
          (∀ i, i < out.numel → Kernel.memReal m out_ptr i = TiledActivation.relu (xs i + ys i)) ∧
          (∀ r o, ¬ (r = out_ptr ∧ o < out.numel) → m r o = s.mem r o)) ∧
      (∀ flat : RegionName, (in_ptr0 = in_ptr1 ↔ x.base = y.base) →
        s.undef = (fun _ _ => 0) →
        (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).Disjoint ∧
        (∀ r, r ∉ (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).regions →
          (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).extent r = 0) ∧
        Kernel.LaunchCorrectFramed
          ((Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).flattenKernel
            ((add_relu_kernel in_ptr0 in_ptr1 out_ptr x.numel 64).toAlgKernel))
          { dims := (Elementwise2.launch 64 x y out).grid }
          ((Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).flattenState s)
          (fun i : Nat => if i < out.numel then
            some (flat, out.base / out.elemBytes + i) else none)
          (fun i => TiledActivation.relu (xs i + ys i))) := by
  exact add_relu_wrapper_correctness_block 64 x y out hc in_ptr0 in_ptr1 out_ptr h0 h1 s xs ys hx hy

end VeriTile.Bench.TritonBenchG.AddReluFused
