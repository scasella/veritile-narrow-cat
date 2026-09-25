import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `add_example` — strict per-kernel correctness

`add_kernel` is the canonical elementwise add: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of two inputs, adds them lane-wise, and
stores to `out_ptr`, masked by `offsets < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. Because `pid` is universally quantified, the per-program statement
covers every program of the grid. The host launch (`add_kernel[(num_blocks,)](...)`,
the grid, the allocation sizes, strides, `i32` widths, aliasing, and the
composition of per-program writes into one buffer) is discharged in the
final section from a configuration accepted by the proved checker
`Blocked1DLaunch.check` (see `CONTRACT.md`, `SOURCE_LINK.md`).

## Proof architecture

```
add_kernel_correctness                        ← TOP THEOREM (addIO ⊨ pointwise add)
  ├─ add_kernel_flattenOk                     bridge fragment membership
  ├─ add_kernel_traceSafe                     per-execution lane-wise safety walk
  └─ add_kernel_region_run                    region-model masked Hoare triple
       ├─ add_kernel_correct                  algorithm-layer readback per lane
       └─ add_kernel_frame                    masked scatter-store cell frame

add_kernel_launch_correctness                 ← LAUNCH HEADLINE (checked config c)
  ├─ add_kernel_launch_framed                 LaunchCorrectFramed over c.grid
  │    ├─ Blocked1DLaunch.check_ok            host obligations P1–P10
  │    ├─ Blocked1D.launch_of_frames          disjoint whole-grid composition
  │    ├─ addLaunchFrame                      per-program ExecFrame
  │    │    (add_kernel_exec_isSome, add_kernel_frame)
  │    └─ add_kernel_region_run               per-program values
  ├─ add_kernel_launch_traceSafe              every program trace-safe
  └─ Blocked1DLaunch.Pre.i32_offset_toInt / i32_mask_eq
add_kernel_launch_applicable                  lane hypotheses of `⊨`, per launched pid
add_kernel_launch_initial_output_irrelevant   zero-fill of `out` is dead
```

This file carries two `specification`s: the upstream per-program headline
`add_kernel_correctness`, kept verbatim (this contribution only appends to
upstream files), and the launch headline `add_kernel_launch_correctness`,
appended last. The launch headline does not restate `addIO ⊨ …`;
`add_kernel_launch_applicable` connects the two by discharging the per-lane
hypotheses of `⊨` for every launched program.

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
`xs i + ys i` at every output cell `i < n` (it applies
`add_kernel_launch_framed` to both states). That the *returned tensor* is then
fully determined also needs wrapper obligation W1 (`n == out.numel()`), which
the adapter checks outside Lean. -/
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
for every `i < n`, while every other cell is unchanged; and, as arithmetic
facts about `c`, every lane offset and mask evaluated in two's-complement
`i32` equals its ℕ counterpart used by the model (the model itself computes
over ℕ; that Triton evaluates these expressions in `i32` is the trusted
assumption TA-i32). Obligations used: P1–P3 (first conjunct), P4 (second),
P5–P7 (third); P8–P10 justify the translation assumptions TA-region and
TA-compose (`CONTRACT.md` §2). -/
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


/-! ## Whole-wrapper contract (`add_wrapper`)

The launch headline above speaks about one launch configuration `c`. The
caller of `add_wrapper(x, y)` receives the whole tensor `out`. This section
states what the caller receives, from the tensors' own metadata
(`TensorMeta`): the wrapper derives `n = x.numel()`, `BLOCK_SIZE = 4` and the
grid (`Elementwise2.launch`), and `Elementwise2.check` decides the wrapper
preconditions (`Elementwise2.Pre`: equal shapes, contiguity, element
alignment, input separation, and P1–P10 of the derived launch). W1 (every
element of `out` is written) and W2 (every input tensor holds the elements
read) are consequences, not hypotheses. -/

/-- One program of `add_kernel`, from any state whose input cells below `n`
hold typed real values: it succeeds, writes `xs o + ys o` at every active
output lane of its block, and changes no other cell. -/
theorem add_kernel_program_run
    (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) (hB : 0 < B)
    (t : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < n → t.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < n → t.mem in_ptr1 i = MemCell.real (ys i)) :
    ∃ f, exec ((add_kernel in_ptr0 in_ptr1 out_ptr n B).toAlgKernel) t = some f ∧
      (∀ o, o < n → o / B = t.pid → f.readMem out_ptr o = xs o + ys o) ∧
      (∀ r o, ¬ Blocked1D.blockWrites out_ptr n B t.pid (r, o) → f.mem r o = t.mem r o) := by
  obtain ⟨f, hexec, hvals, hframe⟩ := add_kernel_region_run in_ptr0 in_ptr1 out_ptr n B t
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
another, in any complete duplicate-free order, writes `xs i + ys i` at every
output index `i < n` and leaves every other cell unchanged — the same result
as the merge (`Kernel.runSerial_agrees_merge`), because no program reads a
cell another program writes. -/
theorem add_kernel_launch_serial
    (c : Blocked1DLaunch) (hc : Blocked1DLaunch.check c = Bool.true)
    (in_ptr0 in_ptr1 out_ptr : RegionName) (h0 : out_ptr ≠ in_ptr0) (h1 : out_ptr ≠ in_ptr1)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < c.n → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < c.n → s.mem in_ptr1 i = MemCell.real (ys i)) :
    ∀ L : List (GridIndex { dims := c.grid }), L.Nodup → (∀ idx, idx ∈ L) →
      ∃ m, Kernel.runSerial ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
            s L s.mem = some m ∧
        (∀ i, i < c.n → Kernel.memReal m out_ptr i = xs i + ys i) ∧
        (∀ r o, ¬ (r = out_ptr ∧ o < c.n) → m r o = s.mem r o) := by
  have hpre := Blocked1DLaunch.check_ok c hc
  have hB := hpre.block_pos
  have hcov : c.n ≤ c.gridX * c.block :=
    of_decide_eq_true ((Blocked1DLaunch.coversB_iff c hB).2 hpre.covers)
  rw [hpre.grid_eq]
  intro L hnd hall
  let frames : Kernel.GridFrames ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel)
      (Blocked1D.line c.gridX) s :=
    fun idx => addLaunchFrame in_ptr0 in_ptr1 out_ptr c.n c.block c.gridX s idx
  have hwr : ∀ idx, (frames idx).writes
      = Blocked1D.blockWrites out_ptr c.n c.block (Blocked1D.pidOf idx) := fun _ => rfl
  have hdisj : Kernel.GridWritesDisjoint frames := by
    intro i₁ i₂ hne
    rw [hwr i₁, hwr i₂]
    exact Blocked1D.blockWrites_disjoint hB fun h => hne (Blocked1D.idx_ext h)
  have hfinal : ∀ idx o, o < c.n → o / c.block = Blocked1D.pidOf idx →
      (frames idx).final.readMem out_ptr o = xs o + ys o := by
    intro idx o ho hd
    obtain ⟨f, hf, hv, -⟩ := add_kernel_program_run in_ptr0 in_ptr1 out_ptr c.n c.block hB
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
    obtain ⟨f, hf, hv, hfr⟩ := add_kernel_program_run in_ptr0 in_ptr1 out_ptr c.n c.block hB
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
    (fun i => xs i + ys i)
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
writes `xs i + ys i` at flat address `A.base out_ptr + i` for every `i < n`
and leaves every other flat cell unchanged. Per program this is the upstream
flat headline `add_kernel_correctness` (`⊨`); `launch_of_frames_addr`
composes the programs. -/
theorem add_kernel_launch_flat
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
      (A.flattenKernel ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel))
      { dims := c.grid } (A.flattenState s)
      (fun i : Nat => if i < c.n then some (A.flat, A.base out_ptr + i) else none)
      (fun i => xs i + ys i) := by
  have hpre := Blocked1DLaunch.check_ok c hc
  have hB := hpre.block_pos
  have hcov : c.n ≤ c.gridX * c.block :=
    of_decide_eq_true ((Blocked1DLaunch.coversB_iff c hB).2 hpre.covers)
  rw [hpre.grid_eq]
  have hI := add_kernel_correctness in_ptr0 in_ptr1 out_ptr c.n c.block
  have hprog : ∀ idx : GridIndex (Blocked1D.line c.gridX), ∃ s',
      exec (A.flattenKernel ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel))
        ((A.flattenState s).withGridIndex idx) = some s' ∧
      (∀ j, j < c.block → Blocked1D.pidOf idx * c.block + j < c.n →
        s'.readMem A.flat (A.base out_ptr + (Blocked1D.pidOf idx * c.block + j))
          = xs (Blocked1D.pidOf idx * c.block + j) + ys (Blocked1D.pidOf idx * c.block + j)) ∧
      (∀ r o, ¬ Blocked1D.addrWrites A.flat (fun t => A.base out_ptr + t) c.n c.block
          (Blocked1D.pidOf idx) (r, o) →
        ((A.flattenState s).withGridIndex idx).mem r o = s'.mem r o) := by
    intro idx
    obtain ⟨s', hex, hval, hfr⟩ := hI A hd (by simp [addIO, hreg]) hcl (Blocked1D.pidOf idx)
      (fun j hj => by simp only [addIO] at hj ⊢; omega)
      (fun j hj => by simp only [addIO] at hj ⊢; omega)
      (fun j hj => by simp only [addIO] at hj ⊢; omega)
      (fun p hp => by simp [addIO] at hp)
      (fun j => xs (Blocked1D.pidOf idx * c.block + j.val))
      (fun j => ys (Blocked1D.pidOf idx * c.block + j.val))
      (s.withGridIndex idx) (Blocked1D.withGridIndex_pid_line s idx) (by simp [hu])
      (fun j hj => by
        simp only [addIO] at hj ⊢
        simp [BlockState.readMem, hx _ hj])
      (fun j hj => by
        simp only [addIO] at hj ⊢
        simp [BlockState.readMem, hy _ hj])
    rw [FlatAlloc.flattenState_withGridIndex]
    refine ⟨s', hex, fun j hj hn => ?_, fun r o hno => ?_⟩
    · have := hval ⟨j, hj⟩ hn
      simpa [addIO, FlatAlloc.addr] using this
    · refine (hfr r o ?_).symm
      by_cases hr : r = A.flat
      · refine Or.inr ⟨fun j hj ho => hno ?_, fun p hp => by simp [addIO] at hp⟩
        rw [Blocked1D.addrWrites_iff hB]
        have hj' : j.val < c.block := j.isLt
        refine ⟨hr, Blocked1D.pidOf idx * c.block + j.val, hj, ?_, ?_⟩
        · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hj', Nat.zero_add]
        · simp only [addIO, FlatAlloc.addr] at ho; exact ho.symm
      · exact Or.inl hr
  classical
  let frames : Kernel.GridFrames
      (A.flattenKernel ((add_kernel in_ptr0 in_ptr1 out_ptr c.n c.block).toAlgKernel))
      (Blocked1D.line c.gridX) (A.flattenState s) := fun idx =>
    { final := Classical.choose (hprog idx)
      writes := Blocked1D.addrWrites A.flat (fun t => A.base out_ptr + t) c.n c.block
        (Blocked1D.pidOf idx)
      h_exec := (Classical.choose_spec (hprog idx)).1
      h_writeWithin := fun r o hno => (Classical.choose_spec (hprog idx)).2.2 r o hno }
  obtain ⟨L, -, hout, hframe⟩ := Blocked1D.launch_of_frames_addr hB hcov
    (fun t => A.base out_ptr + t) (fun a b h => by simp only at h; omega) frames (fun _ => rfl)
    (fun i => xs i + ys i) (fun idx j hj hn => (Classical.choose_spec (hprog idx)).2.1 j hj hn)
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

/-- **Whole-wrapper headline.** For tensors accepted by the wrapper checker
and region names that follow the allocations, from any state whose input
tensors hold typed real values:

1. every element `i < out.numel` of the returned tensor is written with
   `xs i + ys i`, and every other cell is unchanged (framed whole-grid launch);
2. for every in-shape multi-index, the three tensors address the same
   row-major element, below `out.numel` (the logical view);
3. every program is trace-safe for bounds equal to the tensors' own element
   counts — no access leaves an input's logical extent;
4. running the programs one after another in **any** complete order gives the
   same result (so the merge semantics does not rely on a program never
   reading another's writes);
5. in the flat memory placed at the tensors' element addresses
   (`base / elemBytes`), the checked conditions discharge the bridge's
   disjointness and closure hypotheses and the flattened launch writes
   `xs i + ys i` at element address `out.base / out.elemBytes + i` for every
   `i < out.numel`, leaving every other flat cell unchanged. -/
specification add_wrapper_correctness
    (x y out : TensorMeta) (hc : Elementwise2.check 4 x y out = Bool.true)
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (h0 : out_ptr ≠ in_ptr0) (h1 : out_ptr ≠ in_ptr1)
    (s : BlockState) (xs ys : Nat → ℝ)
    (hx : ∀ i, i < x.numel → s.mem in_ptr0 i = MemCell.real (xs i))
    (hy : ∀ i, i < y.numel → s.mem in_ptr1 i = MemCell.real (ys i)) :
    Kernel.LaunchCorrectFramed
        ((add_kernel in_ptr0 in_ptr1 out_ptr x.numel 4).toAlgKernel)
        { dims := (Elementwise2.launch 4 x y out).grid } s
        (fun i : Nat => if i < out.numel then some (out_ptr, i) else none)
        (fun i => xs i + ys i) ∧
      (∀ idx, TensorMeta.InShape out.shape idx →
        out.offsetOf idx = x.offsetOf idx ∧ y.offsetOf idx = x.offsetOf idx ∧
          x.offsetOf idx < out.numel) ∧
      (∀ bounds : RegionBounds,
        x.numel ≤ bounds in_ptr0 → y.numel ≤ bounds in_ptr1 → out.numel ≤ bounds out_ptr →
        ∀ idx : GridIndex { dims := (Elementwise2.launch 4 x y out).grid },
          Kernel.TraceSafe bounds
            ((add_kernel in_ptr0 in_ptr1 out_ptr x.numel 4).toAlgKernel)
            (s.withGridIndex idx)) ∧
      (∀ L : List (GridIndex { dims := (Elementwise2.launch 4 x y out).grid }),
        L.Nodup → (∀ idx, idx ∈ L) →
        ∃ m, Kernel.runSerial ((add_kernel in_ptr0 in_ptr1 out_ptr x.numel 4).toAlgKernel)
              s L s.mem = some m ∧
          (∀ i, i < out.numel → Kernel.memReal m out_ptr i = xs i + ys i) ∧
          (∀ r o, ¬ (r = out_ptr ∧ o < out.numel) → m r o = s.mem r o)) ∧
      (∀ flat : RegionName, (in_ptr0 = in_ptr1 ↔ x.base = y.base) →
        s.undef = (fun _ _ => 0) →
        (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).Disjoint ∧
        (∀ r, r ∉ (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).regions →
          (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).extent r = 0) ∧
        Kernel.LaunchCorrectFramed
          ((Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).flattenKernel
            ((add_kernel in_ptr0 in_ptr1 out_ptr x.numel 4).toAlgKernel))
          { dims := (Elementwise2.launch 4 x y out).grid }
          ((Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).flattenState s)
          (fun i : Nat => if i < out.numel then
            some (flat, out.base / out.elemBytes + i) else none)
          (fun i => xs i + ys i)) := by
  have hpre := Elementwise2.check_ok 4 x y out hc
  have hL := hpre.launch
  have hc' : Blocked1DLaunch.check (Elementwise2.launch 4 x y out) = Bool.true :=
    Blocked1DLaunch.check_complete _ hL
  have hW1 : x.numel = out.numel := hpre.output_covered
  have hyn : x.numel = y.numel := by simp [TensorMeta.numel, hpre.same_shape.1]
  have hx' : ∀ i, i < (Elementwise2.launch 4 x y out).n →
      s.mem in_ptr0 i = MemCell.real (xs i) := hx
  have hy' : ∀ i, i < (Elementwise2.launch 4 x y out).n →
      s.mem in_ptr1 i = MemCell.real (ys i) := fun i hi =>
    hy i (by change i < x.numel at hi; omega)
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [← hW1]
    exact add_kernel_launch_framed _ hc' in_ptr0 in_ptr1 out_ptr s xs ys hx' hy'
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
    apply add_kernel_traceSafe <;> intro j hj <;>
      rw [Blocked1D.withGridIndex_pid_line] at hj ⊢ <;> omega
  · rw [← hW1]
    exact add_kernel_launch_serial _ hc' in_ptr0 in_ptr1 out_ptr h0 h1 s xs ys hx' hy'
  · intro flat hnames hu
    refine ⟨hpre.flat_disjoint flat in_ptr0 in_ptr1 out_ptr h0 h1 hnames,
      Elementwise2.flatAlloc_closed flat in_ptr0 in_ptr1 out_ptr x y out, ?_⟩
    have hbase : (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out).base out_ptr
        = out.base / out.elemBytes := by simp [Elementwise2.flatAlloc]
    have hflat := add_kernel_launch_flat _ hc' in_ptr0 in_ptr1 out_ptr
      (Elementwise2.flatAlloc flat in_ptr0 in_ptr1 out_ptr x y out)
      (hpre.flat_disjoint flat in_ptr0 in_ptr1 out_ptr h0 h1 hnames) rfl
      (Elementwise2.flatAlloc_closed flat in_ptr0 in_ptr1 out_ptr x y out)
      (by simp [Elementwise2.flatAlloc, Elementwise2.launch])
      (by simp [Elementwise2.flatAlloc, Elementwise2.launch])
      (by simp [Elementwise2.flatAlloc, Elementwise2.launch])
      s hu xs ys hx' hy'
    rw [← hW1, ← hbase]
    exact hflat

end VeriTile.Bench.TritonBenchG.AddExample
