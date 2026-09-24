/-
VeriTile.Triton.Launch.Blocked1DConfig

Host-launch configuration checker for one-dimensional, blocked, masked
elementwise kernels — the family

    pid = tl.program_id(0); offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n; tl.load(p + offsets, mask=mask) …; tl.store(out + offsets, …, mask=mask)

launched as `kernel[(g,)](…, n, BLOCK)`. The per-program theorems of such
kernels (e.g. `MaskedKernelIO₂.Implements`) take the active-lane bounds as
hypotheses and leave the grid, allocation sizes, strides, integer widths and
aliasing to the host. This module states those host obligations as
independent propositions (`Blocked1DLaunch.Pre`), gives one executable Bool
checker per obligation, and proves each Bool helper equivalent to its
proposition. `check_ok` / `check_complete` package the conjunction, following
the local checker convention of `documents/SemanticCaveats.md` (one Bool
helper, one Prop contract, one `_ok` bridge).

The integer width is fixed to Triton's `i32` offset arithmetic; the
`i32_offset_toInt` / `i32_mask_eq` lemmas show that, under the checked bounds,
two's-complement 32-bit evaluation of `pid * BLOCK + j` and of the mask
comparison agrees with the ℕ arithmetic of the kernel model.

Deliberately core-only (no Mathlib import) so that concrete configurations
can be decided by the Lean kernel cheaply.
-/

namespace VeriTile.Triton

/-- Element dtype of a launch buffer, as reported by the host tensor. -/
inductive ElemDType where
  | f32 | f16 | bf16 | other
  deriving DecidableEq, Repr

/-- Host metadata of one buffer argument, viewed as a flat 1-D tensor.
`base` is the byte address of element 0, `stride` the element stride of the
flattened view, and `capacity` the number of elements addressable from
`base` inside its allocation. -/
structure BufMeta where
  base : Nat
  elemBytes : Nat
  stride : Nat
  capacity : Nat
  dtype : ElemDType
  deriving DecidableEq, Repr

/-- One host launch of a 1-D blocked masked elementwise kernel. -/
structure Blocked1DLaunch where
  /-- The `n_elements` scalar argument. -/
  n : Nat
  /-- The `BLOCK_SIZE` constexpr. -/
  block : Nat
  /-- The launch grid tuple as passed by the host. -/
  grid : List Nat
  /-- Input buffers, in kernel-argument order. -/
  inputs : List BufMeta
  /-- The output buffer. -/
  output : BufMeta
  deriving DecidableEq, Repr

namespace Blocked1DLaunch

/-- Program count along axis 0. -/
def gridX (c : Blocked1DLaunch) : Nat := c.grid.headD 0

/-- Every buffer touched by the kernel. -/
def bufs (c : Blocked1DLaunch) : List BufMeta := c.output :: c.inputs

/-- Exclusive upper bound of non-negative `i32` values. -/
def i32Limit : Nat := 2 ^ 31

/-- Byte ranges `[base, base + n * elemBytes)` of two buffers do not meet. -/
def RangesDisjoint (n : Nat) (a b : BufMeta) : Prop :=
  ∀ addr : Nat, ¬ (a.base ≤ addr ∧ addr < a.base + n * a.elemBytes ∧
    b.base ≤ addr ∧ addr < b.base + n * b.elemBytes)

/-! ## Obligations (independent propositions) -/

/-- P1: the grid is one-dimensional. -/
def GridRank (c : Blocked1DLaunch) : Prop := c.grid.length = 1

/-- P2: `tl.arange(0, BLOCK)` is legal: a power of two not exceeding
`TRITON_MAX_TENSOR_NUMEL = 2^20`. -/
def BlockOk (c : Blocked1DLaunch) : Prop := ∃ k, k ≤ 20 ∧ c.block = 2 ^ k

/-- P3: the program owning each output index is launched. -/
def Covers (c : Blocked1DLaunch) : Prop :=
  ∀ i, i < c.n → i / c.block < c.gridX

/-- P4: every active lane of every launched program addresses inside every
buffer's allocation. -/
def LanesInBounds (c : Blocked1DLaunch) : Prop :=
  ∀ b ∈ c.bufs, ∀ pid, pid < c.gridX → ∀ j, j < c.block →
    pid * c.block + j < c.n → pid * c.block + j < b.capacity

/-- P5: every lane offset computed by a launched program is a non-negative
`i32` (no wrap-around in `pid * BLOCK + arange`). -/
def OffsetsFit (c : Blocked1DLaunch) : Prop :=
  ∀ pid, pid < c.gridX → ∀ j, j < c.block → pid * c.block + j < i32Limit

/-- P6: the `n_elements: tl.int32` argument is representable. -/
def NFits (c : Blocked1DLaunch) : Prop := c.n < i32Limit

/-- P7: the grid dimension is representable as the launcher's C `int`. -/
def GridFits (c : Blocked1DLaunch) : Prop := c.gridX < i32Limit

/-- P8: the flat `ptr + offsets` addressing matches every buffer's layout. -/
def UnitStride (c : Blocked1DLaunch) : Prop :=
  ∀ b ∈ c.bufs, c.n ≤ 1 ∨ b.stride = 1

/-- P9: supported dtype domain (v1: `float32`). -/
def DTypeOk (c : Blocked1DLaunch) : Prop :=
  ∀ b ∈ c.bufs, b.dtype = .f32 ∧ b.elemBytes = 4

/-- P10: the output's accessed byte range meets no input's accessed range. -/
def OutDisjoint (c : Blocked1DLaunch) : Prop :=
  ∀ b ∈ c.inputs, RangesDisjoint c.n c.output b

/-- The host-launch preconditions of a 1-D blocked masked elementwise kernel. -/
structure Pre (c : Blocked1DLaunch) : Prop where
  grid_rank : GridRank c
  block_ok : BlockOk c
  covers : Covers c
  lanes_in_bounds : LanesInBounds c
  offsets_fit : OffsetsFit c
  n_fits : NFits c
  grid_fits : GridFits c
  unit_stride : UnitStride c
  dtype_ok : DTypeOk c
  out_disjoint : OutDisjoint c

/-! ## Executable checker -/

def gridRankB (c : Blocked1DLaunch) : Bool := c.grid.length == 1

def blockOkB (c : Blocked1DLaunch) : Bool :=
  (List.range 21).any fun k => c.block == 2 ^ k

def coversB (c : Blocked1DLaunch) : Bool := decide (c.n ≤ c.gridX * c.block)

def lanesB (c : Blocked1DLaunch) : Bool :=
  c.bufs.all fun b => decide (min c.n (c.gridX * c.block) ≤ b.capacity)

def offsetsB (c : Blocked1DLaunch) : Bool := decide (c.gridX * c.block ≤ i32Limit)

def nFitsB (c : Blocked1DLaunch) : Bool := decide (c.n < i32Limit)

def gridFitsB (c : Blocked1DLaunch) : Bool := decide (c.gridX < i32Limit)

def strideB (c : Blocked1DLaunch) : Bool :=
  c.bufs.all fun b => decide (c.n ≤ 1) || b.stride == 1

def dtypeB (c : Blocked1DLaunch) : Bool :=
  c.bufs.all fun b => b.dtype == .f32 && b.elemBytes == 4

def disjointB (n : Nat) (a b : BufMeta) : Bool :=
  decide (n * a.elemBytes = 0) || decide (n * b.elemBytes = 0) ||
    decide (a.base + n * a.elemBytes ≤ b.base) ||
    decide (b.base + n * b.elemBytes ≤ a.base)

def outDisjointB (c : Blocked1DLaunch) : Bool :=
  c.inputs.all fun b => disjointB c.n c.output b

/-- The launch checker: conjunction of the per-obligation helpers. -/
def check (c : Blocked1DLaunch) : Bool :=
  gridRankB c && blockOkB c && coversB c && lanesB c && offsetsB c &&
    nFitsB c && gridFitsB c && strideB c && dtypeB c && outDisjointB c

/-- Names of the failed obligations (diagnostics only; the verdict is `check`). -/
def failures (c : Blocked1DLaunch) : List String :=
  [("P1 grid_rank", gridRankB c), ("P2 block_ok", blockOkB c),
   ("P3 covers", coversB c), ("P4 lanes_in_bounds", lanesB c),
   ("P5 offsets_fit", offsetsB c), ("P6 n_fits", nFitsB c),
   ("P7 grid_fits", gridFitsB c), ("P8 unit_stride", strideB c),
   ("P9 dtype_ok", dtypeB c), ("P10 out_disjoint", outDisjointB c)].filterMap
    fun (name, ok) => if ok then none else some name

/-! ## Bool ↔ Prop bridges -/

theorem gridRankB_iff (c : Blocked1DLaunch) : gridRankB c = true ↔ GridRank c := by
  simp [gridRankB, GridRank]

theorem blockOkB_iff (c : Blocked1DLaunch) : blockOkB c = true ↔ BlockOk c := by
  simp only [blockOkB, BlockOk, List.any_eq_true, List.mem_range, beq_iff_eq]
  constructor
  · rintro ⟨k, hk, h⟩; exact ⟨k, by omega, h⟩
  · rintro ⟨k, hk, h⟩; exact ⟨k, by omega, h⟩

theorem coversB_iff (c : Blocked1DLaunch) (hB : 0 < c.block) :
    coversB c = true ↔ Covers c := by
  simp only [coversB, Covers, decide_eq_true_eq]
  constructor
  · intro h i hi
    exact (Nat.div_lt_iff_lt_mul hB).2 (by omega)
  · intro h
    refine Nat.le_of_not_lt fun hlt => ?_
    have := h (c.gridX * c.block) hlt
    rw [Nat.mul_div_cancel _ hB] at this
    exact Nat.lt_irrefl _ this

/-- Pointwise core of `lanesB_iff`: the active lanes of a `g`-program grid are
exactly the offsets below `min n (g * B)`. -/
theorem min_le_iff_lanes (n g B cap : Nat) :
    min n (g * B) ≤ cap ↔
      ∀ pid, pid < g → ∀ j, j < B → pid * B + j < n → pid * B + j < cap := by
  constructor
  · intro h pid hpid j hj hn
    have h1 : (pid + 1) * B ≤ g * B := Nat.mul_le_mul_right B hpid
    rw [Nat.succ_mul] at h1
    omega
  · intro h
    refine Nat.le_of_not_lt fun hlt => ?_
    have hB : 0 < B := by
      rcases Nat.eq_zero_or_pos B with h0 | h0
      · subst h0; simp at hlt
      · exact h0
    have hgB : cap < g * B := by omega
    have hpid : cap / B < g := (Nat.div_lt_iff_lt_mul hB).2 hgB
    have hdm := Nat.div_add_mod' cap B
    have := h (cap / B) hpid (cap % B) (Nat.mod_lt cap hB) (by omega)
    omega

theorem lanesB_iff (c : Blocked1DLaunch) : lanesB c = true ↔ LanesInBounds c := by
  simp only [lanesB, LanesInBounds, List.all_eq_true, decide_eq_true_eq]
  exact forall_congr' fun b => imp_congr_right fun _ =>
    min_le_iff_lanes c.n c.gridX c.block b.capacity

theorem offsetsB_iff (c : Blocked1DLaunch) : offsetsB c = true ↔ OffsetsFit c := by
  simp only [offsetsB, OffsetsFit, decide_eq_true_eq]
  constructor
  · intro h pid hpid j hj
    have h1 : (pid + 1) * c.block ≤ c.gridX * c.block := Nat.mul_le_mul_right _ hpid
    rw [Nat.succ_mul] at h1
    omega
  · intro h
    refine Nat.le_of_not_lt fun hlt => ?_
    obtain ⟨g', hg⟩ : ∃ g', c.gridX = g' + 1 := by
      rcases Nat.exists_eq_succ_of_ne_zero (n := c.gridX) (by
        intro h0; rw [h0, Nat.zero_mul] at hlt; exact Nat.not_lt_zero _ hlt) with ⟨g', hg'⟩
      exact ⟨g', hg'⟩
    obtain ⟨B', hB'⟩ : ∃ B', c.block = B' + 1 := by
      rcases Nat.exists_eq_succ_of_ne_zero (n := c.block) (by
        intro h0; rw [h0, Nat.mul_zero] at hlt; exact Nat.not_lt_zero _ hlt) with ⟨B', hB''⟩
      exact ⟨B', hB''⟩
    have := h g' (by omega) B' (by omega)
    rw [hg, Nat.succ_mul] at hlt
    rw [hB'] at this hlt
    omega

theorem nFitsB_iff (c : Blocked1DLaunch) : nFitsB c = true ↔ NFits c := by
  simp [nFitsB, NFits]

theorem gridFitsB_iff (c : Blocked1DLaunch) : gridFitsB c = true ↔ GridFits c := by
  simp [gridFitsB, GridFits]

theorem strideB_iff (c : Blocked1DLaunch) : strideB c = true ↔ UnitStride c := by
  simp [strideB, UnitStride]

theorem dtypeB_iff (c : Blocked1DLaunch) : dtypeB c = true ↔ DTypeOk c := by
  simp [dtypeB, DTypeOk]

theorem disjointB_iff (n : Nat) (a b : BufMeta) :
    disjointB n a b = true ↔ RangesDisjoint n a b := by
  simp only [disjointB, RangesDisjoint, Bool.or_eq_true, decide_eq_true_eq]
  generalize n * a.elemBytes = A
  generalize n * b.elemBytes = Bb
  constructor
  · rintro (((h | h) | h) | h) addr ⟨h1, h2, h3, h4⟩ <;> omega
  · intro h
    refine Classical.byContradiction fun hn => ?_
    by_cases hab : a.base ≤ b.base
    · exact h b.base ⟨hab, by omega, Nat.le_refl _, by omega⟩
    · exact h a.base ⟨Nat.le_refl _, by omega, by omega, by omega⟩

theorem outDisjointB_iff (c : Blocked1DLaunch) :
    outDisjointB c = true ↔ OutDisjoint c := by
  simp only [outDisjointB, OutDisjoint, List.all_eq_true, disjointB_iff]

theorem blockOk_pos {c : Blocked1DLaunch} (h : BlockOk c) : 0 < c.block := by
  obtain ⟨k, _, hk⟩ := h
  rw [hk]; exact Nat.two_pow_pos k

/-- **Checker soundness.** An accepted configuration satisfies every host
obligation. -/
theorem check_ok (c : Blocked1DLaunch) (h : check c = true) : Pre c := by
  simp only [check, Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩, h8⟩, h9⟩, h10⟩ := h
  have hB := blockOk_pos ((blockOkB_iff c).1 h2)
  exact ⟨(gridRankB_iff c).1 h1, (blockOkB_iff c).1 h2, (coversB_iff c hB).1 h3,
    (lanesB_iff c).1 h4, (offsetsB_iff c).1 h5, (nFitsB_iff c).1 h6,
    (gridFitsB_iff c).1 h7, (strideB_iff c).1 h8, (dtypeB_iff c).1 h9,
    (outDisjointB_iff c).1 h10⟩

/-- **Checker completeness.** The checker accepts every configuration meeting
the obligations, so a rejection is a failed named obligation. -/
theorem check_complete (c : Blocked1DLaunch) (h : Pre c) : check c = true := by
  have hB := blockOk_pos h.block_ok
  simp only [check, Bool.and_eq_true]
  exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨(gridRankB_iff c).2 h.grid_rank, (blockOkB_iff c).2 h.block_ok⟩,
    (coversB_iff c hB).2 h.covers⟩, (lanesB_iff c).2 h.lanes_in_bounds⟩,
    (offsetsB_iff c).2 h.offsets_fit⟩, (nFitsB_iff c).2 h.n_fits⟩,
    (gridFitsB_iff c).2 h.grid_fits⟩, (strideB_iff c).2 h.unit_stride⟩,
    (dtypeB_iff c).2 h.dtype_ok⟩, (outDisjointB_iff c).2 h.out_disjoint⟩

/-! ## Consequences used by kernel proofs -/

theorem Pre.block_pos {c : Blocked1DLaunch} (h : Pre c) : 0 < c.block :=
  blockOk_pos h.block_ok

theorem Pre.grid_eq {c : Blocked1DLaunch} (h : Pre c) : c.grid = [c.gridX] := by
  have hl := h.grid_rank
  unfold GridRank at hl
  unfold gridX
  match hg : c.grid, hl with
  | [x], _ => rfl

/-- Distinct (program, lane) pairs address distinct offsets. -/
theorem Pre.offset_injective {c : Blocked1DLaunch} (h : Pre c)
    {p₁ p₂ j₁ j₂ : Nat} (hj₁ : j₁ < c.block) (hj₂ : j₂ < c.block)
    (heq : p₁ * c.block + j₁ = p₂ * c.block + j₂) : p₁ = p₂ ∧ j₁ = j₂ := by
  have hB := h.block_pos
  have hdiv : ∀ p j, j < c.block → (p * c.block + j) / c.block = p := by
    intro p j hj
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hj, Nat.zero_add]
  have hp : p₁ = p₂ := by
    rw [← hdiv p₁ j₁ hj₁, ← hdiv p₂ j₂ hj₂, heq]
  subst hp
  exact ⟨rfl, Nat.add_left_cancel heq⟩

/-- Under the checked bounds, the i32 bit-vector offset has the ℕ offset as
its unsigned value. -/
theorem Pre.i32_offset_toNat {c : Blocked1DLaunch} (h : Pre c)
    {pid j : Nat} (hpid : pid < c.gridX) (hj : j < c.block) :
    (BitVec.ofNat 32 pid * BitVec.ofNat 32 c.block + BitVec.ofNat 32 j).toNat
      = pid * c.block + j := by
  have hfit := h.offsets_fit pid hpid j hj
  have hgrid := h.grid_fits
  have hB := h.block_pos
  unfold i32Limit at hfit
  unfold GridFits i32Limit at hgrid
  have hblk : c.block ≤ 2 ^ 20 := by
    obtain ⟨k, hk, hb⟩ := h.block_ok
    rw [hb]; exact Nat.pow_le_pow_right (by decide) hk
  have hpB : pid * c.block < 2 ^ 31 := by omega
  simp only [BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_ofNat]
  have e1 : pid % 2 ^ 32 = pid := Nat.mod_eq_of_lt (by omega)
  have e2 : c.block % 2 ^ 32 = c.block := Nat.mod_eq_of_lt (by omega)
  have e3 : j % 2 ^ 32 = j := Nat.mod_eq_of_lt (by omega)
  rw [e1, e2, e3, Nat.mod_eq_of_lt (a := pid * c.block) (by omega),
    Nat.mod_eq_of_lt (by omega)]

/-- Two's-complement `i32` evaluation of `pid * BLOCK + j` (as Triton lowers
`pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`) equals the ℕ offset. -/
theorem Pre.i32_offset_toInt {c : Blocked1DLaunch} (h : Pre c)
    {pid j : Nat} (hpid : pid < c.gridX) (hj : j < c.block) :
    (BitVec.ofNat 32 pid * BitVec.ofNat 32 c.block + BitVec.ofNat 32 j).toInt
      = ((pid * c.block + j : Nat) : Int) := by
  have hfit := h.offsets_fit pid hpid j hj
  unfold i32Limit at hfit
  rw [BitVec.toInt_eq_toNat_cond, h.i32_offset_toNat hpid hj]
  simp only [show 2 * (pid * c.block + j) < 2 ^ 32 by omega, if_true]

/-- The `i32` signed mask comparison `offsets < n_elements` agrees with ℕ. -/
theorem Pre.i32_mask_eq {c : Blocked1DLaunch} (h : Pre c)
    {pid j : Nat} (hpid : pid < c.gridX) (hj : j < c.block) :
    BitVec.slt (BitVec.ofNat 32 pid * BitVec.ofNat 32 c.block + BitVec.ofNat 32 j)
        (BitVec.ofNat 32 c.n)
      = decide (pid * c.block + j < c.n) := by
  have hn := h.n_fits
  unfold NFits i32Limit at hn
  have hnInt : (BitVec.ofNat 32 c.n).toInt = (c.n : Int) := by
    rw [BitVec.toInt_eq_toNat_cond, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]
    simp only [show 2 * c.n < 2 ^ 32 by omega, if_true]
  rw [BitVec.slt, h.i32_offset_toInt hpid hj, hnInt]
  simp only [decide_eq_decide]
  omega

end Blocked1DLaunch

end VeriTile.Triton
