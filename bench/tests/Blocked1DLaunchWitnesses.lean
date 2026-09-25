/-
bench/tests/Blocked1DLaunchWitnesses

Concrete, kernel-checked witnesses for the `add_example` host-launch contract
(`bench/tritonbench_g/add_example/CONTRACT.md`):

* positive controls — the checker accepts the file's own test configurations
  (non-empty and empty), so the supported domain is nonempty;
* i32 witnesses — configurations rejected by P5/P6 really do break the ℕ
  model under Triton's two's-complement `i32` offset arithmetic;
* kernel-mutant witnesses — dropping the store mask, or weakening the mask to
  `offsets <= n_elements`, writes an output cell at index `≥ n`, violating the
  contract's frame clause (cells outside `out[0, n)` are unchanged).
* wrapper-contract cases (`Elementwise2`) — accepted configurations, and the
  allocation-safe but tensor-invalid (W2) and partially-written (W1)
  configurations that the launch checker alone accepts but the wrapper
  contract rejects;
* strided unary (`relu_strided_buffer`, one-tile branch) layout cases decided
  by `StridedUnary.check`.

The mutant kernels are local copies; the upstream transcription in
`AddExample.lean` is not modified.
-/

import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Tests.Blocked1DLaunchWitnesses

open VeriTile.Triton VeriTile.Examples VeriTile.Meta

/-! ## Positive controls -/

/-- `test_case_1`: `n = 16`, `BLOCK_SIZE = 4`, grid `(4,)`, fresh contiguous
f32 buffers at disjoint addresses. -/
def testCase1 : Blocked1DLaunch :=
  { n := 16, block := 4, grid := [4],
    inputs := [⟨4096, 4, 1, 16, .f32⟩, ⟨8192, 4, 1, 16, .f32⟩],
    output := ⟨12288, 4, 1, 16, .f32⟩ }

theorem testCase1_pre : Blocked1DLaunch.Pre testCase1 :=
  Blocked1DLaunch.check_ok _ (by decide)

/-- `test_case_4`: the empty launch `n = 0`, grid `(0,)`. -/
def emptyCase : Blocked1DLaunch :=
  { n := 0, block := 4, grid := [0],
    inputs := [⟨4096, 4, 1, 0, .f32⟩, ⟨4096, 4, 1, 0, .f32⟩],
    output := ⟨4096, 4, 1, 0, .f32⟩ }

theorem emptyCase_pre : Blocked1DLaunch.Pre emptyCase :=
  Blocked1DLaunch.check_ok _ (by decide)

/-! ## i32 witnesses -/

/-- M4: `n_elements = 3·2^30` passed as `tl.int32` wraps to `-2^30`, so lane 0
of program 0 (ℕ offset `0 < n`) is masked off and never written. -/
theorem i32_n_truncated :
    (BitVec.ofNat 32 (3 * 2 ^ 30)).toInt = -(2 ^ 30) ∧
    BitVec.slt (BitVec.ofNat 32 0 * BitVec.ofNat 32 1024 + BitVec.ofNat 32 0)
      (BitVec.ofNat 32 (3 * 2 ^ 30)) = Bool.false := by decide

/-- M5: grid `(2^29 + 1,)`, `BLOCK_SIZE = 4`, `n = 16`: program `2^29`, lane 0
has ℕ offset `2^31 ≥ 16` (inactive in the model), but its `i32` offset wraps
to `-2^31` and passes the mask — an active lane addressing before the buffer. -/
theorem i32_offset_wraps :
    (BitVec.ofNat 32 (2 ^ 29) * BitVec.ofNat 32 4 + BitVec.ofNat 32 0).toInt = -(2 ^ 31) ∧
    BitVec.slt (BitVec.ofNat 32 (2 ^ 29) * BitVec.ofNat 32 4 + BitVec.ofNat 32 0)
      (BitVec.ofNat 32 16) = Bool.true ∧ ¬ (2 ^ 29 * 4 + 0 < 16) := by decide

/-! ## Kernel mutants -/

/-- `add_kernel` with the store mask removed. -/
def addNoStoreMask (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output)
}

set_option maxHeartbeats 1600000 in
/-- K1: with `n = 5`, `BLOCK_SIZE = 4`, program 1 of the unmasked-store mutant
overwrites `out[7]` (index `≥ n`), which the contract requires unchanged. -/
theorem unmasked_store_frame_violation (a b o : RegionName) (s : BlockState)
    (hu : s.undef = fun _ _ => 0) (h7 : s.mem o 7 = MemCell.real 1) :
    ∃ s', exec ((addNoStoreMask a b o 5 4).toAlgKernel) { s with pids := fun _ => 1 }
        = some s' ∧ s'.readMem o 7 ≠ s.readMem o 7 := by
  have hinit : s.readMem o 7 = 1 := by simp [BlockState.readMem, h7]
  simp [exec, addNoStoreMask, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, hu, hinit]
  rw [show (7 : Nat) = 4 + ((⟨3, by decide⟩ : Fin 4) : Nat) from rfl]
  erw [BlockState.scatter_readback_nd (region := o) (shape := [4]) _
    (fun k : TileIndex [4] => 4 + k.1.val) _
    (injective_offset_singleton 4) ((⟨3, by decide⟩ : Fin 4), PUnit.unit)]
  simp

/-- `add_kernel` with the off-by-one mask `offsets <= n_elements`. -/
def addOffByOneMask (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets <= $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
}

set_option maxHeartbeats 1600000 in
/-- K2: with `n = 5`, `BLOCK_SIZE = 4`, program 1 of the off-by-one mutant
writes `out[5] = x[5] + y[5]` (index `= n`), which the contract requires
unchanged. -/
theorem offbyone_mask_frame_violation (a b o : RegionName) (s : BlockState)
    (ha : s.mem a 5 = MemCell.real 0) (hb : s.mem b 5 = MemCell.real 0)
    (h5 : s.mem o 5 = MemCell.real 1) :
    ∃ s', exec ((addOffByOneMask a b o 5 4).toAlgKernel) { s with pids := fun _ => 1 }
        = some s' ∧ s'.readMem o 5 ≠ s.readMem o 5 := by
  have hinit : s.readMem o 5 = 1 := by simp [BlockState.readMem, h5]
  simp [exec, addOffByOneMask, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.le, hinit]
  rw [show (5 : Nat) = 4 + ((⟨1, by decide⟩ : Fin 4) : Nat) from rfl]
  erw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
    (injective_offset_singleton 4) ((⟨1, by decide⟩ : Fin 4), PUnit.unit)]
  simp [BlockState.readMem, ha, hb]

/-! ## Wrapper-contract witnesses

Two configurations that the launch checker alone accepts — every access is
inside its allocation — but that do not give the caller the intended tensor.
The wrapper contract (`Elementwise2`) rejects both. -/

/-- Accepted: equal-shape contiguous `(4, 8)` f32 tensors at disjoint aligned
addresses (the wrapper contract of `add_wrapper`). -/
theorem add_wrapper_accepts_2d :
    Elementwise2.check 4 ⟨4096, 4, [4, 8], [8, 1], 32, .f32⟩ ⟨8192, 4, [4, 8], [8, 1], 32, .f32⟩
      ⟨16384, 4, [4, 8], [8, 1], 32, .f32⟩ = Bool.true := by decide

/-- Accepted: `x` and `y` the same tensor (read-only aliasing). -/
theorem add_wrapper_accepts_aliased_inputs :
    Elementwise2.check 4 ⟨4096, 4, [16], [1], 16, .f32⟩ ⟨4096, 4, [16], [1], 16, .f32⟩
      ⟨8192, 4, [16], [1], 16, .f32⟩ = Bool.true := by decide

/-- Accepted: `custom_add` on rank-1 `n = 37`. -/
theorem custom_add_accepts_rank1 :
    Elementwise2.checkRank1 16 ⟨4096, 4, [37], [1], 37, .f32⟩ ⟨8192, 4, [37], [1], 37, .f32⟩
      ⟨16384, 4, [37], [1], 37, .f32⟩ = Bool.true := by decide

/-- W2: `y` is a 12-element view of 16-element storage while `x` has 16
elements. Every address stays inside `y`'s allocation (capacity 16), so the
launch checker accepts; but the kernel reads 4 elements past the tensor `y`. -/
def w2X : TensorMeta := ⟨4096, 4, [16], [1], 16, .f32⟩
def w2Y : TensorMeta := ⟨8192, 4, [12], [1], 16, .f32⟩
def w2Out : TensorMeta := ⟨12288, 4, [16], [1], 16, .f32⟩

theorem w2_launch_accepted :
    Blocked1DLaunch.check (Elementwise2.launch 4 w2X w2Y w2Out) = Bool.true := by decide

theorem w2_inputs_not_covered :
    ¬ Elementwise2.InputsCover (Elementwise2.launch 4 w2X w2Y w2Out) [w2X, w2Y] := by
  intro h
  have : (Elementwise2.launch 4 w2X w2Y w2Out).n ≤ w2Y.numel := h w2Y (by simp)
  revert this
  decide

theorem w2_wrapper_rejected : Elementwise2.check 4 w2X w2Y w2Out = Bool.false := by decide

/-- W1: `custom_add` on `(4, 8)` tensors launches over `size(0) = 4`. The
launch is accepted (memory-safe, correct on `out[0, 4)`), but 28 of the 32
returned elements are never written; the rank-1 wrapper contract rejects it. -/
def w1A : TensorMeta := ⟨4096, 4, [4, 8], [8, 1], 32, .f32⟩
def w1B : TensorMeta := ⟨8192, 4, [4, 8], [8, 1], 32, .f32⟩
def w1C : TensorMeta := ⟨16384, 4, [4, 8], [8, 1], 32, .f32⟩

theorem w1_launch_accepted :
    Blocked1DLaunch.check (Elementwise2.launchDim0 16 w1A w1B w1C) = Bool.true := by decide

theorem w1_output_not_covered :
    ¬ Elementwise2.OutputCovered (Elementwise2.launchDim0 16 w1A w1B w1C) w1C :=
  Elementwise2.launchDim0_not_covered 16 w1A w1B w1C 4 [8] rfl (by decide) (by decide)

theorem w1_wrapper_rejected : Elementwise2.checkRank1 16 w1A w1B w1C = Bool.false := by decide

/-! ## Strided unary (ReLU one-tile) layout cases

Configurations of `relu_forward_wrapper_rank_1` decided by
`StridedUnary.check` in the kernel. Element strides and capacities are
measured from each view's data pointer. -/

/-- Contiguous `n = 1025`: tile 512, 3 programs. -/
theorem relu_valid_contiguous :
    StridedUnary.check (StridedUnary.launch ⟨4096, 4, [1025], [1], 1025, .f32⟩
      ⟨65536, 4, [1025], [1], 1025, .f32⟩) = Bool.true := by decide

/-- Input stride 2 (every other element of its storage), output stride 3 (two
gap cells between outputs). -/
theorem relu_valid_strided :
    StridedUnary.check (StridedUnary.launch ⟨4096, 4, [10], [2], 20, .f32⟩
      ⟨65536, 4, [10], [3], 30, .f32⟩) = Bool.true := by decide

/-- Empty input: the pinned wrapper divides by zero; rejected. -/
theorem relu_empty_rejected :
    StridedUnary.check (StridedUnary.launch ⟨4096, 4, [0], [1], 0, .f32⟩
      ⟨65536, 4, [0], [1], 0, .f32⟩) = Bool.false := by decide

/-- `n = 65536 * 512 + 1` needs 65537 tiles: the wrapper takes the
grid-stride-loop branch, which this contract does not cover; rejected. -/
theorem relu_loop_branch_rejected :
    StridedUnary.check (StridedUnary.launch ⟨0, 4, [33554433], [1], 33554433, .f32⟩
      ⟨2 ^ 40, 4, [33554433], [1], 33554433, .f32⟩) = Bool.false := by decide

/-- Stride 2 over a view with capacity 18 from its data pointer: element 9
sits at offset 18, outside the allocation; rejected. -/
theorem relu_stride_out_of_alloc_rejected :
    StridedUnary.check (StridedUnary.launch ⟨4096, 4, [10], [2], 18, .f32⟩
      ⟨65536, 4, [10], [1], 10, .f32⟩) = Bool.false := by decide

/-- Output starts inside the input's strided span; rejected. -/
theorem relu_overlap_rejected :
    StridedUnary.check (StridedUnary.launch ⟨4096, 4, [10], [2], 20, .f32⟩
      ⟨4100, 4, [10], [1], 10, .f32⟩) = Bool.false := by decide

/-- float16 is outside the supported dtype domain; rejected. -/
theorem relu_f16_rejected :
    StridedUnary.check (StridedUnary.launch ⟨4096, 2, [16], [1], 16, .f16⟩
      ⟨65536, 2, [16], [1], 16, .f16⟩) = Bool.false := by decide

#axiomsClean testCase1_pre
#axiomsClean emptyCase_pre
#axiomsClean i32_n_truncated
#axiomsClean i32_offset_wraps
#axiomsClean unmasked_store_frame_violation
#axiomsClean offbyone_mask_frame_violation
#axiomsClean add_wrapper_accepts_2d
#axiomsClean add_wrapper_accepts_aliased_inputs
#axiomsClean custom_add_accepts_rank1
#axiomsClean w2_launch_accepted
#axiomsClean w2_inputs_not_covered
#axiomsClean w2_wrapper_rejected
#axiomsClean w1_launch_accepted
#axiomsClean w1_output_not_covered
#axiomsClean w1_wrapper_rejected
#axiomsClean relu_valid_contiguous
#axiomsClean relu_valid_strided
#axiomsClean relu_empty_rejected
#axiomsClean relu_loop_branch_rejected
#axiomsClean relu_stride_out_of_alloc_rejected
#axiomsClean relu_overlap_rejected
#axiomsClean relu_f16_rejected

end VeriTile.Bench.Tests.Blocked1DLaunchWitnesses
