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

#axiomsClean testCase1_pre
#axiomsClean emptyCase_pre
#axiomsClean i32_n_truncated
#axiomsClean i32_offset_wraps
#axiomsClean unmasked_store_frame_violation
#axiomsClean offbyone_mask_frame_violation

end VeriTile.Bench.Tests.Blocked1DLaunchWitnesses
