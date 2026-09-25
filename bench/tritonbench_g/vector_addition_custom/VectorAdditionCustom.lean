import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `vector_addition_custom` — strict per-kernel correctness

`_add_kernel` is an elementwise add: program `prog_id` loads block
`[prog_id·BLOCK, (prog_id+1)·BLOCK)` of inputs `A` and `B`, adds them lane-wise,
and stores to `C`, masked by `offs < size`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_add_kernel[grid](...)`, the grid size
`cdiv(size, BLOCK)`, the host-side `BLOCK` choice, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because `prog_id` is universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
add_kernel_correctness                        ← TOP THEOREM (addCustomIO ⊨ pointwise add)
  ├─ add_kernel_flattenOk                     bridge fragment membership
  ├─ add_kernel_traceSafe                     per-execution lane-wise safety walk
  └─ add_kernel_region_run                    region-model masked Hoare triple
       ├─ add_kernel_correct                  algorithm-layer readback per lane
       └─ add_kernel_frame                    masked scatter-store cell frame
```

The headline is stated on the kernel's masked **IO signature** `addCustomIO`
(`MaskedKernelIO₂`): which buffer is which argument, where program `prog_id`
reads/writes its `BLOCK`-lane window, and the active-lane predicate
`prog_id * BLOCK + j < size`. `⊨` is the audit-once masked Hoare-triple
combinator (`MaskedKernelIO₂.Implements`): for **every** disjoint placement of
the three buffers in flat memory, **every** program id all of whose *active*
lanes are in bounds (partial blocks may overhang the buffer on their inactive
lanes), and **every** launch state whose input windows hold `as`/`bs` at the
active lanes, the translated pointer kernel terminates, every active output
lane holds `as i + bs i`, and every other memory cell is unchanged.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). No output/input
disjointness is assumed at the region layer: both inputs are read into
registers before the scatter, so the result is correct even if `C` aliases `A`
or `B`.
-/

namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Faithful 1:1 transcription of `vector_addition_custom.py`'s `_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK: tl.constexpr` → Lean `Nat` parameter. -/
def _add_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, a + b, mask=offs < $(size))
}

/-- Algorithm-layer correctness for `_add_kernel`.

Each active lane writes `A + B`; inactive tail lanes are preserved. -/
theorem add_kernel_correct
    (A B C : RegionName)
    (size BLOCK : Nat)
    (s : BlockState) (as bs : Fin BLOCK → ℝ)
    (h_a : InputLoadedAt s A BLOCK as)
    (h_b : InputLoadedAt s B BLOCK bs) :
    ∀ i : Fin BLOCK,
      let addr := s.pid * BLOCK + i.val
      observeAt (exec (_add_kernel A B C size BLOCK) s) C BLOCK s.pid i
        = some (if addr < size then as i + bs i else s.readMem C addr) := by
  intro i
  simp [observeAt, exec, _add_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_a h_b
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK + i.val < size
  · simp [hi, h_a, h_b]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `C`, and the *inactive* lanes of the output window itself. -/
private theorem add_kernel_frame
    (A B C : RegionName)
    (size BLOCK : Nat) (s s1 : BlockState)
    (hExec : exec ((_add_kernel A B C size BLOCK).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK, s.pid * BLOCK + i.val < size →
      ¬(C = r ∧ s.pid * BLOCK + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, _add_kernel, ComputeKernel.toAlgKernel,
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
    (A B C : RegionName)
    (size BLOCK : Nat)
    (s₀ : BlockState) (as bs : Fin BLOCK → ℝ)
    (ha : ∀ j : Fin BLOCK, s₀.pid * BLOCK + j.val < size →
      s₀.readMem A (s₀.pid * BLOCK + j.val) = as j)
    (hb : ∀ j : Fin BLOCK, s₀.pid * BLOCK + j.val < size →
      s₀.readMem B (s₀.pid * BLOCK + j.val) = bs j) :
    ∃ s1, exec ((_add_kernel A B C size BLOCK).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK, s₀.pid * BLOCK + j.val < size →
          s1.readMem C (s₀.pid * BLOCK + j.val) = as j + bs j)
      ∧ (∀ r o,
          (r ≠ C ∨ ∀ j : Fin BLOCK,
            s₀.pid * BLOCK + j.val < size →
              o ≠ s₀.pid * BLOCK + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := add_kernel_correct A B C size BLOCK
    s₀ (fun j => s₀.readMem A (s₀.pid * BLOCK + j.val))
    (fun j => s₀.readMem B (s₀.pid * BLOCK + j.val))
    (fun _ => rfl) (fun _ => rfl)
  rw [show exec (_add_kernel A B C size BLOCK) s₀
      = exec ((_add_kernel A B C size BLOCK).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((_add_kernel A B C size BLOCK).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, _add_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, ha j hj, hb j hj]
      · refine add_kernel_frame A B C size BLOCK
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: both masked loads and the masked store address
the same window `prog_id * BLOCK + j`, active only when `< size`, so the
bounds contract is **lane-wise** — every *active* lane's address is below the
region bound of the buffer it touches. -/
theorem add_kernel_traceSafe
    (A B C : RegionName)
    (size BLOCK : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (ha : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size →
      s.pid * BLOCK + j.val < bounds A)
    (hb : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size →
      s.pid * BLOCK + j.val < bounds B)
    (hout : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size →
      s.pid * BLOCK + j.val < bounds C) :
    Kernel.TraceSafe bounds
      ((_add_kernel A B C size BLOCK).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all five statements, discharging every
  -- load-free `SafeAt` and reducing the three memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [_add_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha' => ha a ha', fun a hb' => hb a hb', fun a ho' => hout a ho'⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem add_kernel_flattenOk
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ((_add_kernel A B C size BLOCK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [_add_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- `_add_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline: which buffer is which argument (the wiring), where
program `prog_id` reads its input tiles / writes its output tile, and the
active-lane predicate `prog_id * BLOCK + j < size`. The windows and mask are
declared, not parsed from the kernel: they formalize the host-side launch
convention (`offs = prog_id * BLOCK + arange; mask = offs < size`), and the
headline **proves** the kernel's actual addressing and masking match them.
Buffer sizes are not signature content: the headline quantifies over every
allocation whose extents cover the active lanes. -/
def addCustomIO (A B C : RegionName)
    (size BLOCK : Nat) : MaskedKernelIO₂ where
  kernel := _add_kernel A B C size BLOCK
  in1 := A
  in2 := B
  out := C
  B := BLOCK
  read1 := fun pid => pid * BLOCK
  read2 := fun pid => pid * BLOCK
  write := fun pid => pid * BLOCK
  mask := fun pid j => pid * BLOCK + j.val < size

/-- **The headline**: `_add_kernel` implements pointwise addition on its masked
IO signature — for every disjoint flat placement of the three buffers, every
program id whose active lanes are in bounds, and every launch state whose
input windows hold `as`/`bs` at the active lanes, the translated pointer
kernel terminates, every active output lane holds `as i + bs i`, and every
other memory cell is unchanged. Proof: `MaskedKernelIO₂.Implements.intro`
assembles the region-model masked triple with the flat-memory bridge side
conditions. -/
specification add_kernel_correctness
    (A B C : RegionName)
    (size BLOCK : Nat) :
    addCustomIO A B C size BLOCK
      ⊨ fun as bs i => as i + bs i := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact add_kernel_flattenOk A B C size BLOCK
  · intro bounds s h1 h2 h3 _
    exact add_kernel_traceSafe A B C size BLOCK bounds s h1 h2 h3
  · intro s₀ as bs ha hb
    obtain ⟨s1, hexec, hval, hframe⟩ := add_kernel_region_run
      A B C size BLOCK s₀ as bs ha hb
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-! ## Host launch: checked configuration → whole-grid contract

The per-program headline above leaves the host launch as hypotheses: which
programs run, and that each active lane is inside its allocation. This
section discharges them from a host launch configuration accepted by the
proved checker `Blocked1DLaunch.check` (`VeriTile/Triton/Launch/Blocked1DConfig.lean`) — the same checker and
composition as `add_example` —
for the grid the host actually passes (`c.grid`, not a hard-wired `cdiv`).
The scalar argument `size` and the constexpr `BLOCK` of the
launched kernel are the configuration's `c.n` and `c.block`; the kernel's
three pointer arguments are bound to `c.inputs = [bx, by]` and `c.output`. -/

/-- Progress: `_add_kernel` always executes to a defined state. -/
theorem add_kernel_exec_isSome
    (A B C : RegionName) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) :
    (exec ((_add_kernel A B C n_elements BLOCK_SIZE
      ).toAlgKernel) s).isSome := by
  simp [exec, _add_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt]

/-- The framed execution of program `idx` of a `(g,)` launch. -/
noncomputable def addCustomLaunchFrame
    (A B C : RegionName) (n_elements BLOCK_SIZE g : Nat)
    (s : BlockState) (idx : GridIndex (Blocked1D.line g)) :
    Kernel.ExecFrame ((_add_kernel A B C n_elements
      BLOCK_SIZE).toAlgKernel) (s.withGridIndex idx) where
  final := (exec ((_add_kernel A B C n_elements
      BLOCK_SIZE).toAlgKernel) (s.withGridIndex idx)).get
    (add_kernel_exec_isSome _ _ _ _ _ _)
  writes := Blocked1D.blockWrites C n_elements BLOCK_SIZE (Blocked1D.pidOf idx)
  h_exec := (Option.some_get _).symm
  h_writeWithin := by
    intro r o hno
    refine (add_kernel_frame A B C n_elements BLOCK_SIZE
      (s.withGridIndex idx) _ (Option.some_get _).symm r o ?_).symm
    intro i hi hc
    apply hno
    rw [Blocked1D.withGridIndex_pid_line] at hi hc
    unfold Blocked1D.blockWrites WriteFootprint.activeTileImage
    exact ⟨hc.1.symm, (i, PUnit.unit), hi, hc.2⟩

/-- Every launched program's active lanes lie inside the checked capacities,
so every program of the launch is trace-safe for any region bounds that
cover those capacities. -/
theorem add_kernel_launch_traceSafe
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (bx by_ : BufMeta) (hin : c.inputs = [bx, by_])
    (A B C : RegionName) (bounds : RegionBounds)
    (hbx : bx.capacity ≤ bounds A) (hby : by_.capacity ≤ bounds B)
    (hbo : c.output.capacity ≤ bounds C) (s : BlockState) :
    ∀ idx : GridIndex { dims := c.grid },
      Kernel.TraceSafe bounds
        ((_add_kernel A B C c.n c.block).toAlgKernel)
        (s.withGridIndex idx) := by
  have hpre := Blocked1DLaunch.check_ok c hc
  have hbx' : bx ∈ c.bufs := by simp [Blocked1DLaunch.bufs, hin]
  have hby' : by_ ∈ c.bufs := by simp [Blocked1DLaunch.bufs, hin]
  have hbo' : c.output ∈ c.bufs := by simp [Blocked1DLaunch.bufs]
  rw [hpre.grid_eq]
  intro idx
  have hp := Blocked1D.pidOf_lt idx
  apply add_kernel_traceSafe <;> intro j hj <;>
    rw [Blocked1D.withGridIndex_pid_line] at hj ⊢
  · have := hpre.lanes_in_bounds bx hbx' _ hp j.val j.isLt hj; omega
  · have := hpre.lanes_in_bounds by_ hby' _ hp j.val j.isLt hj; omega
  · have := hpre.lanes_in_bounds c.output hbo' _ hp j.val j.isLt hj; omega

/-- **Applicability of the per-program headline.** For every program the
checked launch actually runs, the lane-wise bound hypotheses of
`add_kernel_correctness` hold in every flat allocation whose extents cover
the checked capacities. -/
theorem add_kernel_launch_applicable
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (bx by_ : BufMeta) (hin : c.inputs = [bx, by_])
    (A B C : RegionName) (Al : FlatAlloc)
    (hbx : bx.capacity ≤ Al.extent A) (hby : by_.capacity ≤ Al.extent B)
    (hbo : c.output.capacity ≤ Al.extent C) :
    ∀ pid, pid < c.gridX →
      let io := addCustomIO A B C c.n c.block
      (∀ j : Fin io.B, io.mask pid j → io.read1 pid + j.val < Al.extent io.in1) ∧
      (∀ j : Fin io.B, io.mask pid j → io.read2 pid + j.val < Al.extent io.in2) ∧
      (∀ j : Fin io.B, io.mask pid j → io.write pid + j.val < Al.extent io.out) := by
  intro pid hpid
  have hpre := Blocked1DLaunch.check_ok c hc
  have hbx' : bx ∈ c.bufs := by simp [Blocked1DLaunch.bufs, hin]
  have hby' : by_ ∈ c.bufs := by simp [Blocked1DLaunch.bufs, hin]
  have hbo' : c.output ∈ c.bufs := by simp [Blocked1DLaunch.bufs]
  refine ⟨fun j hj => ?_, fun j hj => ?_, fun j hj => ?_⟩
  · change pid * c.block + j.val < c.n at hj
    show pid * c.block + j.val < Al.extent A
    have := hpre.lanes_in_bounds bx hbx' pid hpid j.val j.isLt hj; omega
  · change pid * c.block + j.val < c.n at hj
    show pid * c.block + j.val < Al.extent B
    have := hpre.lanes_in_bounds by_ hby' pid hpid j.val j.isLt hj; omega
  · change pid * c.block + j.val < c.n at hj
    show pid * c.block + j.val < Al.extent C
    have := hpre.lanes_in_bounds c.output hbo' pid hpid j.val j.isLt hj; omega

/-- The framed whole-grid launch of a checked configuration (the first
conjunct of `add_kernel_launch_correctness`; needs no input-buffer binding). -/
theorem add_kernel_launch_framed
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (A B C : RegionName)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < c.n → s.mem A i = MemCell.real (xs i))
    (hy : ∀ i, i < c.n → s.mem B i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((_add_kernel A B C c.n c.block).toAlgKernel)
        { dims := c.grid } s
        (fun i : Nat => if i < c.n then some (C, i) else none)
        (fun i => xs i + ys i) := by
  have hpre := Blocked1DLaunch.check_ok c hc
  have hB := hpre.block_pos
  have hcov : c.n ≤ c.gridX * c.block :=
    of_decide_eq_true ((Blocked1DLaunch.coversB_iff c hB).2 hpre.covers)
  rw [hpre.grid_eq]
  let frames : Kernel.GridFrames
      ((_add_kernel A B C c.n c.block).toAlgKernel)
      (Blocked1D.line c.gridX) s :=
    fun idx => addCustomLaunchFrame A B C c.n c.block c.gridX s idx
  have hval : ∀ idx j, j < c.block → Blocked1D.pidOf idx * c.block + j < c.n →
      (frames idx).final.readMem C (Blocked1D.pidOf idx * c.block + j)
        = xs (Blocked1D.pidOf idx * c.block + j)
          + ys (Blocked1D.pidOf idx * c.block + j) := by
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
    obtain ⟨s1, hexec, hvals, -⟩ := add_kernel_region_run A B C
      c.n c.block (s.withGridIndex idx)
      (fun l => xs (Blocked1D.pidOf idx * c.block + l.val))
      (fun l => ys (Blocked1D.pidOf idx * c.block + l.val))
      (hread A xs hx) (hread B ys hy)
    have hfin : (frames idx).final = s1 := by
      have h1 := (frames idx).h_exec
      rw [hexec] at h1
      exact (Option.some.inj h1).symm
    have := hvals ⟨j, hj⟩ (by rw [Blocked1D.withGridIndex_pid_line]; exact hn)
    rw [Blocked1D.withGridIndex_pid_line] at this
    rw [hfin]
    exact this
  obtain ⟨L, -, hout, hframe⟩ := Blocked1D.launch_of_frames hB hcov frames
    (fun _ => rfl) (fun i => xs i + ys i) hval
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

/-- **Whole-launch headline.** For a host configuration accepted by the proved
launch checker, launching `_add_kernel` over the host's grid `c.grid` from any
state whose input cells `i < n` hold typed real values `xs i`, `ys i`:
every program terminates and is trace-safe for bounds covering the checked
capacities; the programs' write sets are pairwise disjoint and compose into
one final memory (`GridLaunchedOrdinary`) in which `C[i] = xs i + ys i`
for every `i < n`, while every other cell is unchanged; and every lane offset
and mask the launch computes in Triton's `i32` arithmetic equals its ℕ
counterpart used by the model. -/
specification add_kernel_launch_correctness
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (bx by_ : BufMeta) (hin : c.inputs = [bx, by_])
    (A B C : RegionName)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < c.n → s.mem A i = MemCell.real (xs i))
    (hy : ∀ i, i < c.n → s.mem B i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((_add_kernel A B C c.n c.block).toAlgKernel)
        { dims := c.grid } s
        (fun i : Nat => if i < c.n then some (C, i) else none)
        (fun i => xs i + ys i) ∧
      (∀ bounds : RegionBounds,
        bx.capacity ≤ bounds A → by_.capacity ≤ bounds B →
        c.output.capacity ≤ bounds C →
        ∀ idx : GridIndex { dims := c.grid },
          Kernel.TraceSafe bounds
            ((_add_kernel A B C c.n c.block).toAlgKernel)
            (s.withGridIndex idx)) ∧
      ∀ pid j, pid < c.gridX → j < c.block →
        (BitVec.ofNat 32 pid * BitVec.ofNat 32 c.block + BitVec.ofNat 32 j).toInt
            = ((pid * c.block + j : Nat) : Int) ∧
        BitVec.slt (BitVec.ofNat 32 pid * BitVec.ofNat 32 c.block + BitVec.ofNat 32 j)
            (BitVec.ofNat 32 c.n) = decide (pid * c.block + j < c.n) := by
  have hpre := Blocked1DLaunch.check_ok c hc
  exact ⟨add_kernel_launch_framed c hc A B C s xs ys hx hy,
    fun bounds h1 h2 h3 => add_kernel_launch_traceSafe c hc bx by_ hin
      A B C bounds h1 h2 h3 s,
    fun pid j hp hj => ⟨hpre.i32_offset_toInt hp hj, hpre.i32_mask_eq hp hj⟩⟩


/-! ## Whole-wrapper contract (`custom_add`, rank 1)

`custom_add(a, b)` allocates `c = torch.empty_like(a)` and launches over
`size = c.size(0)`. Its supported API is one-dimensional tensors
(`Elementwise2.checkRank1`): there `size(0) = numel` and every element of the
returned tensor is written. On higher-rank inputs the launch covers only the
first `size(0)` elements (`Elementwise2.launchDim0_not_covered`), so such
inputs are rejected by the wrapper contract rather than accepted with a
partially uninitialized result. -/

/-- **Whole-wrapper headline (`custom_add`).** For rank-1 tensors accepted by
the wrapper checker and input cells holding typed real values, every element
`i < c.numel` of the returned tensor is written with `xs i + ys i`, every
other cell is unchanged, and every program is trace-safe for bounds equal to
the tensors' own element counts. -/
specification custom_add_correctness
    (a b c : TensorMeta) (hc : Elementwise2.checkRank1 16 a b c = Bool.true)
    (A B C : RegionName) (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < a.numel → s.mem A i = MemCell.real (xs i))
    (hy : ∀ i, i < b.numel → s.mem B i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((_add_kernel A B C (c.shape.headD 0) 16).toAlgKernel)
        { dims := (Elementwise2.launchDim0 16 a b c).grid } s
        (fun i : Nat => if i < c.numel then some (C, i) else none)
        (fun i => xs i + ys i) ∧
      (∀ bounds : RegionBounds,
        a.numel ≤ bounds A → b.numel ≤ bounds B → c.numel ≤ bounds C →
        ∀ idx : GridIndex { dims := (Elementwise2.launchDim0 16 a b c).grid },
          Kernel.TraceSafe bounds
            ((_add_kernel A B C (c.shape.headD 0) 16).toAlgKernel)
            (s.withGridIndex idx)) := by
  sorry

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom
