/-
VeriTile.Triton.Launch.StridedUnary

Caller-level contract for a rank-1 **strided** unary wrapper with the launch
heuristics of `relu_strided_buffer.relu_forward_wrapper_rank_1`:

    tile   = min(512, next_power_of_2(s0))          (heuristics_for_tile_size)
    tiles  = cdiv(s0, tile); ctas = min(65536, tiles); tiles_per_cta = cdiv(tiles, ctas)
    kernel[(ctas, 1, 1)](in0, out0, in0.stride(0), 0, out0.stride(0), 0, s0, out0.numel(),
                         tiles_per_cta=…, tile_size0=tile, one_tile_per_cta=(tiles_per_cta == 1))

`StridedUnary.launch` computes that launch from the tensors' own metadata;
`StridedUnary.Pre` states the supported domain: rank-1 equal-shape tensors,
non-empty (`s0 = 0` makes the pinned wrapper divide by zero), the one-tile
branch, positive element strides, every strided element inside its
allocation (`(s0 - 1) * stride < capacity`, measured from the view's data
pointer), f32, element-aligned data pointers, disjoint input/output byte
spans, and the integer widths Triton 3.8.0 uses (block-pointer offsets i32;
shape/strides i64). Negative strides, dtype reinterpretation, overlapping
output and the grid-stride-loop branch are rejected.

Deliberately core-only (no Mathlib) so that concrete configurations can be
decided by the Lean kernel.
-/

import VeriTile.Triton.Launch.Blocked1DWrapper

namespace VeriTile.Triton

namespace StridedUnary

open TensorMeta

/-- Doubling search for the least power of two `≥ n`, with fuel (structural,
so concrete values reduce in the kernel). -/
def nextPow2Aux : Nat → Nat → Nat → Nat
  | 0, p, _ => p
  | fuel + 1, p, n => if n ≤ p then p else nextPow2Aux fuel (2 * p) n

/-- `triton.next_power_of_2` (`0 ↦ 0`, `1 ↦ 1`, else the least power of two
`≥ n`; exact for `n ≤ 2^64`). -/
def nextPow2 (n : Nat) : Nat := if n ≤ 1 then n else nextPow2Aux 64 1 n

/-- `triton.cdiv` on naturals (`a / 0 = 0` in `Nat`; the Python raises). -/
def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- The launch the wrapper computes. -/
structure Launch where
  s0 : Nat
  numTasks : Nat
  tile : Nat
  numTiles : Nat
  numCtas : Nat
  tilesPerCta : Nat
  grid : List Nat
  inStride : Nat
  outStride : Nat
  input : TensorMeta
  output : TensorMeta
  deriving DecidableEq, Repr

/-- The launch `relu_forward_wrapper_rank_1(in0, out0=out0)` derives. -/
def launch (x out : TensorMeta) : Launch :=
  let s0 := out.shape.headD 0
  let tile := min 512 (nextPow2 s0)
  let tiles := cdiv s0 tile
  let ctas := min 65536 tiles
  { s0 := s0, numTasks := out.numel, tile := tile, numTiles := tiles, numCtas := ctas,
    tilesPerCta := cdiv tiles ctas, grid := [ctas, 1, 1],
    inStride := x.strides.headD 0, outStride := out.strides.headD 0,
    input := x, output := out }

/-- Byte span `[base, base + ((s0 - 1) * stride + 1) * elemBytes)` of the
strided elements of a non-empty rank-1 view. -/
def spanEnd (t : TensorMeta) (s0 stride : Nat) : Nat :=
  t.base + ((s0 - 1) * stride + 1) * t.elemBytes

def i32Limit : Nat := 2 ^ 31
def i64Limit : Nat := 2 ^ 63

/-- S1: rank-1 tensors with one stride each. -/
def Rank1 (c : Launch) : Prop :=
  c.input.shape.length = 1 ∧ c.output.shape.length = 1 ∧
    c.input.strides.length = 1 ∧ c.output.strides.length = 1
/-- S2: the wrapper's shape assertion. -/
def SameShape (c : Launch) : Prop := c.input.shape = c.output.shape
/-- S3: non-empty (the pinned wrapper raises `ZeroDivisionError` at `s0 = 0`). -/
def NonEmpty (c : Launch) : Prop := 0 < c.s0
/-- S4: the one-tile-per-program branch is selected. -/
def OneTile (c : Launch) : Prop := c.numTiles ≤ 65536
/-- S5: positive element strides. -/
def PosStrides (c : Launch) : Prop := 0 < c.inStride ∧ 0 < c.outStride
/-- S6: every strided element lies inside its allocation. -/
def InBounds (c : Launch) : Prop :=
  (c.s0 - 1) * c.inStride < c.input.capacity ∧ (c.s0 - 1) * c.outStride < c.output.capacity
/-- S7: supported dtype (`float32`, 4 bytes) for both tensors. -/
def DTypeOk (c : Launch) : Prop :=
  c.input.dtype = .f32 ∧ c.input.elemBytes = 4 ∧ c.output.dtype = .f32 ∧ c.output.elemBytes = 4
/-- S8: element-aligned data pointers. -/
def Aligned (c : Launch) : Prop := c.input.base % 4 = 0 ∧ c.output.base % 4 = 0
/-- S9: the input and output byte spans do not meet. -/
def SpansDisjoint (c : Launch) : Prop :=
  spanEnd c.output c.s0 c.outStride ≤ c.input.base ∨ spanEnd c.input c.s0 c.inStride ≤ c.output.base
/-- S10: block-pointer offsets `pid * tile + i` are non-negative `i32`. -/
def OffsetsFit (c : Launch) : Prop := c.numCtas * c.tile ≤ i32Limit
/-- S11: strided element offsets fit the `i64` block-pointer arithmetic. -/
def AddressesFit (c : Launch) : Prop :=
  (c.s0 - 1) * c.inStride * 4 < i64Limit ∧ (c.s0 - 1) * c.outStride * 4 < i64Limit

structure Pre (c : Launch) : Prop where
  rank1 : Rank1 c
  same_shape : SameShape c
  nonempty : NonEmpty c
  one_tile : OneTile c
  pos_strides : PosStrides c
  in_bounds : InBounds c
  dtype_ok : DTypeOk c
  aligned : Aligned c
  spans_disjoint : SpansDisjoint c
  offsets_fit : OffsetsFit c
  addresses_fit : AddressesFit c

def check (c : Launch) : Bool :=
  (c.input.shape.length == 1 && c.output.shape.length == 1 &&
    c.input.strides.length == 1 && c.output.strides.length == 1) &&
  c.input.shape == c.output.shape &&
  decide (0 < c.s0) &&
  decide (c.numTiles ≤ 65536) &&
  (decide (0 < c.inStride) && decide (0 < c.outStride)) &&
  (decide ((c.s0 - 1) * c.inStride < c.input.capacity) &&
    decide ((c.s0 - 1) * c.outStride < c.output.capacity)) &&
  (c.input.dtype == .f32 && c.input.elemBytes == 4 &&
    c.output.dtype == .f32 && c.output.elemBytes == 4) &&
  (c.input.base % 4 == 0 && c.output.base % 4 == 0) &&
  (decide (spanEnd c.output c.s0 c.outStride ≤ c.input.base) ||
    decide (spanEnd c.input c.s0 c.inStride ≤ c.output.base)) &&
  decide (c.numCtas * c.tile ≤ i32Limit) &&
  (decide ((c.s0 - 1) * c.inStride * 4 < i64Limit) &&
    decide ((c.s0 - 1) * c.outStride * 4 < i64Limit))

theorem check_ok (c : Launch) (h : check c = true) : Pre c := by
  simp only [check, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨a1, a2⟩, a3⟩, a4⟩, sh⟩, ne⟩, one⟩, ⟨p1, p2⟩⟩, ⟨b1, b2⟩⟩, ⟨⟨⟨d1, d2⟩, d3⟩, d4⟩⟩, ⟨l1, l2⟩⟩, sd⟩, of⟩, ⟨f1, f2⟩⟩ := h
  exact ⟨⟨a1, a2, a3, a4⟩, sh, ne, one, ⟨p1, p2⟩, ⟨b1, b2⟩, ⟨d1, d2, d3, d4⟩, ⟨l1, l2⟩, sd,
    of, ⟨f1, f2⟩⟩

theorem check_complete (c : Launch) (h : Pre c) : check c = true := by
  obtain ⟨⟨a1, a2, a3, a4⟩, sh, ne, one, ⟨p1, p2⟩, ⟨b1, b2⟩, ⟨d1, d2, d3, d4⟩, ⟨l1, l2⟩, sd,
    of, ⟨f1, f2⟩⟩ := h
  simp only [check, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq]
  exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨a1, a2⟩, a3⟩, a4⟩, sh⟩, ne⟩, one⟩, ⟨p1, p2⟩⟩, ⟨b1, b2⟩⟩, ⟨⟨⟨d1, d2⟩, d3⟩, d4⟩⟩, ⟨l1, l2⟩⟩, sd⟩, of⟩, ⟨f1, f2⟩⟩

theorem nextPow2Aux_pos : ∀ (fuel p n : Nat), 0 < p → 0 < nextPow2Aux fuel p n
  | 0, p, _, hp => hp
  | fuel + 1, p, n, hp => by
    unfold nextPow2Aux
    split
    · exact hp
    · exact nextPow2Aux_pos fuel (2 * p) n (by omega)

theorem nextPow2_pos {n : Nat} (hn : 0 < n) : 0 < nextPow2 n := by
  unfold nextPow2
  split
  · exact hn
  · exact nextPow2Aux_pos 64 1 n (by decide)

theorem cdiv_mul_ge (a b : Nat) (hb : 0 < b) : a ≤ cdiv a b * b := by
  unfold cdiv
  have := Nat.lt_div_mul_add (a := a + b - 1) hb
  omega

theorem cdiv_pos {a b : Nat} (ha : 0 < a) (hb : 0 < b) : 0 < cdiv a b := by
  unfold cdiv
  exact Nat.div_pos (by omega) hb

theorem cdiv_self {a : Nat} (ha : 0 < a) : cdiv a a = 1 := by
  unfold cdiv
  exact Nat.div_eq_of_lt_le (by omega) (by omega)

/-- Facts the composition needs, derived from a checked launch of `launch x out`. -/
theorem Pre.derived {x out : TensorMeta} (h : Pre (launch x out)) :
    let c := launch x out
    0 < c.tile ∧ c.numCtas = c.numTiles ∧ c.tilesPerCta = 1 ∧ c.s0 ≤ c.numCtas * c.tile ∧
      c.numTasks = c.s0 ∧ x.numel = c.s0 := by
  obtain ⟨⟨hrx, hro, -, -⟩, hsh, hne, hone, -, -, -, -, -, -, -⟩ := h
  simp only [launch, NonEmpty, OneTile, SameShape] at hrx hro hsh hne hone ⊢
  have hs : out.shape = [out.shape.headD 0] := by
    match hm : out.shape, hro with
    | [d], _ => rfl
  have htile : 0 < min 512 (nextPow2 (out.shape.headD 0)) :=
    Nat.lt_min.2 ⟨by decide, nextPow2_pos hne⟩
  have htiles := cdiv_pos hne htile
  refine ⟨htile, Nat.min_eq_right hone, ?_, ?_, ?_, ?_⟩
  · rw [Nat.min_eq_right hone]; exact cdiv_self htiles
  · rw [Nat.min_eq_right hone]; exact cdiv_mul_ge _ _ htile
  · simp only [TensorMeta.numel]; rw [hs]; simp
  · simp only [TensorMeta.numel]; rw [hsh, hs]; simp

/-- Every launched lane offset `pid * tile + i` evaluates in two's-complement
`i32` to its ℕ value (block-pointer offsets are `i32` in Triton 3.8.0). -/
theorem Pre.i32_offset_toInt {x out : TensorMeta} (h : Pre (launch x out))
    {pid i : Nat} (hp : pid < (launch x out).numCtas) (hi : i < (launch x out).tile) :
    (BitVec.ofNat 32 pid * BitVec.ofNat 32 (launch x out).tile + BitVec.ofNat 32 i).toInt
      = ((pid * (launch x out).tile + i : Nat) : Int) := by
  have hfit := h.offsets_fit
  unfold OffsetsFit i32Limit at hfit
  have hlt : pid * (launch x out).tile + i < 2 ^ 31 := by
    have : (pid + 1) * (launch x out).tile ≤ (launch x out).numCtas * (launch x out).tile :=
      Nat.mul_le_mul_right _ hp
    rw [Nat.succ_mul] at this
    omega
  have htile : (launch x out).tile < 2 ^ 32 := by
    have : (launch x out).tile ≤ 512 := Nat.min_le_left _ _
    omega
  have hpid : pid < 2 ^ 32 := by
    have := h.offsets_fit
    unfold OffsetsFit i32Limit at this
    have h1 : 0 < (launch x out).tile := by omega
    have h2 := Nat.le_mul_of_pos_right (launch x out).numCtas h1
    omega
  have hn : (BitVec.ofNat 32 pid * BitVec.ofNat 32 (launch x out).tile
      + BitVec.ofNat 32 i).toNat = pid * (launch x out).tile + i := by
    simp only [BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_ofNat]
    rw [Nat.mod_eq_of_lt hpid, Nat.mod_eq_of_lt htile, Nat.mod_eq_of_lt (a := i) (by omega),
      Nat.mod_eq_of_lt (a := pid * (launch x out).tile) (by omega), Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_cond, hn]
  simp only [show 2 * (pid * (launch x out).tile + i) < 2 ^ 32 by omega, if_true]

end StridedUnary

end VeriTile.Triton
