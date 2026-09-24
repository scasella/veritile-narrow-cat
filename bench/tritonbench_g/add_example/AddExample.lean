import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `add_example` — strict per-kernel correctness

`add_kernel` is the canonical elementwise add: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of two inputs, adds them lane-wise, and
stores to `out_ptr`, masked by `offsets < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`add_kernel[(num_blocks,)](...)`, the grid size
`cdiv(n_elements, BLOCK_SIZE)`, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`pid` is universally quantified, the per-program statement covers every program
of the grid.

## Proof architecture

```
add_kernel_correctness                        ← TOP THEOREM (addIO ⊨ pointwise add)
  ├─ add_kernel_flattenOk                     bridge fragment membership
  ├─ add_kernel_traceSafe                     per-execution lane-wise safety walk
  └─ add_kernel_region_run                    region-model masked Hoare triple
       ├─ add_kernel_correct                  algorithm-layer readback per lane
       └─ add_kernel_frame                    masked scatter-store cell frame
```

The headline is stated on the kernel's masked **IO signature** `addIO`
(`MaskedKernelIO₂`): which buffer is which argument, where program `pid`
reads/writes its `BLOCK_SIZE`-lane window, and the active-lane predicate
`pid * BLOCK_SIZE + j < n_elements`. `⊨` is the audit-once masked
Hoare-triple combinator (`MaskedKernelIO₂.Implements`): for **every** disjoint
placement of the three buffers in flat memory, **every** program id all of
whose *active* lanes are in bounds (partial blocks may overhang the buffer on
their inactive lanes), and **every** launch state whose input windows hold
`xs`/`ys` at the active lanes, the translated pointer kernel terminates, every
active output lane holds `xs i + ys i`, and every other memory cell is
unchanged.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). No output/input
disjointness is assumed at the region layer: both inputs are read into
registers before the scatter, so the result is correct even if `out_ptr`
aliases an input.
-/

namespace VeriTile.Bench.TritonBenchG.AddExample

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Faithful 1:1 transcription of `add_example.py`'s `add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` annotation → Lean `Nat` parameter
  (the `tl.constexpr` is implicit in Lean params).

Everything else is verbatim from the upstream kernel. -/
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

/-! ## Correctness -/

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

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: both masked loads and the masked store address
the same window `pid * BLOCK_SIZE + j`, active only when `< n_elements`, so the
bounds contract is **lane-wise** — every *active* lane's address is below the
region bound of the buffer it touches. -/
theorem add_kernel_traceSafe
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
      ((add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all eight statements, discharging every
  -- load-free `SafeAt` and reducing the three memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [add_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hy a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem add_kernel_flattenOk
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ((add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [add_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- `add_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline: which buffer is which argument (the wiring), where
program `pid` reads its input tiles / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < n_elements`. The windows and
mask are declared, not parsed from the kernel: they formalize the host-side
launch convention (`offsets = pid * BLOCK_SIZE + arange;
mask = offsets < n_elements`), and the headline **proves** the kernel's actual
addressing and masking match them. Buffer sizes are not signature content: the
headline quantifies over every allocation whose extents cover the active
lanes. -/
def addIO (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
  in1 := in_ptr0
  in2 := in_ptr1
  out := out_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements

/-- **The headline**: `add_kernel` implements pointwise addition on its masked
IO signature — for every disjoint flat placement of the three buffers, every
program id whose active lanes are in bounds, and every launch state whose
input windows hold `xs`/`ys` at the active lanes, the translated pointer
kernel terminates, every active output lane holds `xs i + ys i`, and every
other memory cell is unchanged. Proof: `MaskedKernelIO₂.Implements.intro`
assembles the region-model masked triple with the flat-memory bridge side
conditions. -/
specification add_kernel_correctness
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    addIO in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs ys i => xs i + ys i := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact add_kernel_flattenOk in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
  · intro bounds s h1 h2 h3 _
    exact add_kernel_traceSafe in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ := add_kernel_region_run
      in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE s₀ xs ys hx hy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-! ## Host launch: checked configuration → whole-grid contract

The per-program headline above leaves the host launch as hypotheses: which
programs run, and that each active lane is inside its allocation. This
section discharges them from a host launch configuration accepted by the
proved checker `Blocked1DLaunch.check` (`VeriTile/Triton/Launch/Blocked1DConfig.lean`),
for the grid the host actually passes (`c.grid`, not a hard-wired `cdiv`).
The scalar argument `n_elements` and the constexpr `BLOCK_SIZE` of the
launched kernel are the configuration's `c.n` and `c.block`; the kernel's
three pointer arguments are bound to `c.inputs = [bx, by]` and `c.output`. -/

/-- Progress: `add_kernel` always executes to a defined state. -/
theorem add_kernel_exec_isSome
    (in_ptr0 in_ptr1 out_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) :
    (exec ((add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ).toAlgKernel) s).isSome := by
  simp [exec, add_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt]

/-- The framed execution of program `idx` of a `(g,)` launch. -/
noncomputable def addLaunchFrame
    (in_ptr0 in_ptr1 out_ptr : RegionName) (n_elements BLOCK_SIZE g : Nat)
    (s : BlockState) (idx : GridIndex (Blocked1D.line g)) :
    Kernel.ExecFrame ((add_kernel in_ptr0 in_ptr1 out_ptr n_elements
      BLOCK_SIZE).toAlgKernel) (s.withGridIndex idx) where
  final := (exec ((add_kernel in_ptr0 in_ptr1 out_ptr n_elements
      BLOCK_SIZE).toAlgKernel) (s.withGridIndex idx)).get
    (add_kernel_exec_isSome _ _ _ _ _ _)
  writes := Blocked1D.blockWrites out_ptr n_elements BLOCK_SIZE (Blocked1D.pidOf idx)
  h_exec := (Option.some_get _).symm
  h_writeWithin := by
    intro r o hno
    refine (add_kernel_frame in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
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
    (in_ptr0 in_ptr1 out_ptr : RegionName) (bounds : RegionBounds)
    (hbx : bx.capacity ≤ bounds in_ptr0) (hby : by_.capacity ≤ bounds in_ptr1)
    (hbo : c.output.capacity ≤ bounds out_ptr) (s : BlockState) :
    ∀ idx : GridIndex { dims := c.grid },
      Kernel.TraceSafe bounds
        ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
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
    (in_ptr0 in_ptr1 out_ptr : RegionName) (A : FlatAlloc)
    (hbx : bx.capacity ≤ A.extent in_ptr0) (hby : by_.capacity ≤ A.extent in_ptr1)
    (hbo : c.output.capacity ≤ A.extent out_ptr) :
    ∀ pid, pid < c.gridX →
      let io := addIO in_ptr0 in_ptr1 out_ptr c.n c.block
      (∀ j : Fin io.B, io.mask pid j → io.read1 pid + j.val < A.extent io.in1) ∧
      (∀ j : Fin io.B, io.mask pid j → io.read2 pid + j.val < A.extent io.in2) ∧
      (∀ j : Fin io.B, io.mask pid j → io.write pid + j.val < A.extent io.out) := by
  intro pid hpid
  have hpre := Blocked1DLaunch.check_ok c hc
  have hbx' : bx ∈ c.bufs := by simp [Blocked1DLaunch.bufs, hin]
  have hby' : by_ ∈ c.bufs := by simp [Blocked1DLaunch.bufs, hin]
  have hbo' : c.output ∈ c.bufs := by simp [Blocked1DLaunch.bufs]
  refine ⟨fun j hj => ?_, fun j hj => ?_, fun j hj => ?_⟩
  · change pid * c.block + j.val < c.n at hj
    show pid * c.block + j.val < A.extent in_ptr0
    have := hpre.lanes_in_bounds bx hbx' pid hpid j.val j.isLt hj; omega
  · change pid * c.block + j.val < c.n at hj
    show pid * c.block + j.val < A.extent in_ptr1
    have := hpre.lanes_in_bounds by_ hby' pid hpid j.val j.isLt hj; omega
  · change pid * c.block + j.val < c.n at hj
    show pid * c.block + j.val < A.extent out_ptr
    have := hpre.lanes_in_bounds c.output hbo' pid hpid j.val j.isLt hj; omega

/-- The framed whole-grid launch of a checked configuration (the first
conjunct of `add_kernel_launch_correctness`; needs no input-buffer binding). -/
theorem add_kernel_launch_framed
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < c.n → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < c.n → s.mem in_ptr1 i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
        { dims := c.grid } s
        (fun i : Nat => if i < c.n then some (out_ptr, i) else none)
        (fun i => xs i + ys i) := by
  have hpre := Blocked1DLaunch.check_ok c hc
  have hB := hpre.block_pos
  have hcov : c.n ≤ c.gridX * c.block :=
    of_decide_eq_true ((Blocked1DLaunch.coversB_iff c hB).2 hpre.covers)
  rw [hpre.grid_eq]
  let frames : Kernel.GridFrames
      ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
      (Blocked1D.line c.gridX) s :=
    fun idx => addLaunchFrame in_ptr0 in_ptr1 out_ptr c.n c.block c.gridX s idx
  have hval : ∀ idx j, j < c.block → Blocked1D.pidOf idx * c.block + j < c.n →
      (frames idx).final.readMem out_ptr (Blocked1D.pidOf idx * c.block + j)
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
    obtain ⟨s1, hexec, hvals, -⟩ := add_kernel_region_run in_ptr0 in_ptr1 out_ptr
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

/-- **The output's initial contents are dead** — justification for allocating
`out` with `torch.empty_like(x)` instead of `torch.zeros_like(x)` in
`add_wrapper`. For a checked configuration and output region distinct from
both input regions (P10 + TA-region), launching from `s` or from `s` with the
output buffer overwritten by arbitrary cells `init` (e.g. zeros) yields, in
both cases, a framed whole-grid launch realizing the same values
`xs i + ys i` at every output cell `i < n`. -/
theorem add_kernel_launch_initial_output_irrelevant
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (h0 : out_ptr ≠ in_ptr0) (h1 : out_ptr ≠ in_ptr1)
    (s : BlockState) (init : Nat → MemCell) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < c.n → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < c.n → s.mem in_ptr1 i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
        { dims := c.grid } s
        (fun i : Nat => if i < c.n then some (out_ptr, i) else none)
        (fun i => xs i + ys i) ∧
      Kernel.LaunchCorrectFramed
        ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
        { dims := c.grid }
        { s with mem := fun r o => if r = out_ptr then init o else s.mem r o }
        (fun i : Nat => if i < c.n then some (out_ptr, i) else none)
        (fun i => xs i + ys i) := by
  have hx' : ∀ i, i < c.n →
      ({ s with mem := fun r o => if r = out_ptr then init o else s.mem r o } : BlockState).mem
        in_ptr0 i = MemCell.real (xs i) := by
    intro i hi; simp [Ne.symm h0, hx i hi]
  have hy' : ∀ i, i < c.n →
      ({ s with mem := fun r o => if r = out_ptr then init o else s.mem r o } : BlockState).mem
        in_ptr1 i = MemCell.real (ys i) := by
    intro i hi; simp [Ne.symm h1, hy i hi]
  exact ⟨add_kernel_launch_framed c hc in_ptr0 in_ptr1 out_ptr s xs ys hx hy,
    add_kernel_launch_framed c hc in_ptr0 in_ptr1 out_ptr _ xs ys hx' hy'⟩

/-- **Whole-launch headline.** For a host configuration accepted by the proved
launch checker, launching `add_kernel` over the host's grid `c.grid` from any
state whose input cells `i < n` hold typed real values `xs i`, `ys i`:
every program terminates and is trace-safe for bounds covering the checked
capacities; the programs' write sets are pairwise disjoint and compose into
one final memory (`GridLaunchedOrdinary`) in which `out_ptr[i] = xs i + ys i`
for every `i < n`, while every other cell is unchanged; and every lane offset
and mask the launch computes in Triton's `i32` arithmetic equals its ℕ
counterpart used by the model. -/
specification add_kernel_launch_correctness
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (bx by_ : BufMeta) (hin : c.inputs = [bx, by_])
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < c.n → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < c.n → s.mem in_ptr1 i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
        { dims := c.grid } s
        (fun i : Nat => if i < c.n then some (out_ptr, i) else none)
        (fun i => xs i + ys i) ∧
      (∀ bounds : RegionBounds,
        bx.capacity ≤ bounds in_ptr0 → by_.capacity ≤ bounds in_ptr1 →
        c.output.capacity ≤ bounds out_ptr →
        ∀ idx : GridIndex { dims := c.grid },
          Kernel.TraceSafe bounds
            ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
            (s.withGridIndex idx)) ∧
      ∀ pid j, pid < c.gridX → j < c.block →
        (BitVec.ofNat 32 pid * BitVec.ofNat 32 c.block + BitVec.ofNat 32 j).toInt
            = ((pid * c.block + j : Nat) : Int) ∧
        BitVec.slt (BitVec.ofNat 32 pid * BitVec.ofNat 32 c.block + BitVec.ofNat 32 j)
            (BitVec.ofNat 32 c.n) = decide (pid * c.block + j < c.n) := by
  have hpre := Blocked1DLaunch.check_ok c hc
  exact ⟨add_kernel_launch_framed c hc in_ptr0 in_ptr1 out_ptr s xs ys hx hy,
    fun bounds h1 h2 h3 => add_kernel_launch_traceSafe c hc bx by_ hin
      in_ptr0 in_ptr1 out_ptr bounds h1 h2 h3 s,
    fun pid j hp hj => ⟨hpre.i32_offset_toInt hp hj, hpre.i32_mask_eq hp hj⟩⟩

end VeriTile.Bench.TritonBenchG.AddExample
