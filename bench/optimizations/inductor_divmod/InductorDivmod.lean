import VeriTile.Triton

/-!
# In-kernel fast divmod for Inductor's dynamic `cat` kernel (stage 12)

`scripts/inductor_divmod.py` rewrites Inductor's pointwise kernel so that, under the
scalar guard `0 < ks < 2^31`, the two emitted index lines

    x0 = (xindex % ks)
    x1 = xindex // ks

are replaced by one call to `_vt_fast_divmod(xindex, ks)`. That helper computes, once
per program and from the runtime divisor `d = ks` alone:

    s = 0; for j in static_range(32): s += (d > (1 << j))      -- int64
    m = ((1 << 32) * ((1 << s) - d)) // d + 1                    -- int64
    q = ((umulhi(xu, m.to(uint32)) + xu) >> s.to(uint32)).to(int32)   -- xu = x.to(uint32)
    r = x.to(int32) - q * d.to(int32)

The kernel's launch signature and wrapper do not change: no constants are
computed on the host. Outside the guard, the kernel runs its original body verbatim.

This file proves that the helper's arithmetic is the stage-11 `FastDiv` computation
and returns `(x / d, x % d)` for every active lane (`0 ≤ x < 2^31`), with every
intermediate in range for the machine type it is computed in.

**Scope.** Only the replaced expression is proved: that `(q, r)` equals what the two
original lines compute. That the rest of Inductor's kernel is unaffected is argued,
not proved: the rest consumes only `x0` and `x1`, and the rewrite leaves it textually
unchanged. The rewrite is also tested (`scripts/test_inductor_divmod.py`,
`local_checks.json`). Masked-off lanes, where `xindex ≥ xnumel`, may compute other
values; the kernel's masks discard them.
-/

namespace VeriTile.Bench.Optimizations.InductorDivmod

open VeriTile.Triton

/-- The kernel's shift: `s = #{ j < 32 : 2^j < d }`. -/
def shiftCount (d : Nat) : Nat := ((List.range 32).filter (fun j => 2 ^ j < d)).length

/-- The kernel's multiplier, `((1 << 32) * ((1 << s) - d)) // d + 1`. -/
def kMagic (d : Nat) : Nat := 2 ^ 32 * (2 ^ shiftCount d - d) / d + 1

/-- The kernel's 32-bit quotient: `(umulhi(x, m) + x) >> s` on `uint32`. -/
def kQuot (d x : Nat) : Nat :=
  ((FastDiv.umulhi32 (BitVec.ofNat 32 x) (BitVec.ofNat 32 (kMagic d)) + BitVec.ofNat 32 x)
    >>> shiftCount d).toNat

/-- The counted shift is `FastDiv.shiftFor`, the least `s` with `d ≤ 2^s`. -/
theorem shiftCount_eq_shiftFor {d : Nat} (hd : 0 < d) (hd' : d ≤ 2 ^ 31) :
    shiftCount d = FastDiv.shiftFor d := by
  have hd32 : d ≤ 2 ^ 32 := le_trans hd' (Nat.pow_le_pow_right (by norm_num) (by norm_num))
  obtain ⟨hle, hmin⟩ := FastDiv.shiftFor_spec hd hd32
  set t := FastDiv.shiftFor d
  have key : ∀ j, 2 ^ j < d ↔ j < t := by
    intro j
    constructor
    · intro h
      by_contra hj
      have : 2 ^ t ≤ 2 ^ j := Nat.pow_le_pow_right (by norm_num) (by omega)
      omega
    · intro hj
      rcases hmin with h0 | h1
      · omega
      · exact lt_of_le_of_lt (Nat.pow_le_pow_right (by norm_num) (by omega)) h1
  have ht : t ≤ 31 := by
    rcases hmin with h0 | h1
    · omega
    · have : 2 ^ (t - 1) < 2 ^ 31 := lt_of_lt_of_le h1 hd'
      have := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp this
      omega
  have count : ∀ n, ((List.range n).filter (fun j => decide (j < t))).length = min n t := by
    intro n
    induction n with
    | zero => simp
    | succ m ih =>
      rw [List.range_succ, List.filter_append, List.length_append, ih]
      by_cases hm : m < t
      · simp [hm]; omega
      · simp [hm]; omega
  unfold shiftCount
  rw [List.filter_congr (p := fun j => decide (2 ^ j < d)) (q := fun j => decide (j < t))
    (fun j _ => by simp [key j]), count 32]
  omega

/-- The shift is at most 31, so `>> s` on `uint32` is a defined shift. -/
theorem shiftCount_le {d : Nat} (hd : 0 < d) (hd' : d ≤ 2 ^ 31) : shiftCount d ≤ 31 := by
  have hd32 : d ≤ 2 ^ 32 := le_trans hd' (Nat.pow_le_pow_right (by norm_num) (by norm_num))
  rw [shiftCount_eq_shiftFor hd hd']
  obtain ⟨-, hmin⟩ := FastDiv.shiftFor_spec hd hd32
  rcases hmin with h0 | h1
  · omega
  · have : 2 ^ (FastDiv.shiftFor d - 1) < 2 ^ 31 := lt_of_lt_of_le h1 hd'
    have := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp this
    omega

/-- The int64 intermediates are in range. `(1 << s) - d` does not underflow, and
`(1 << 32) * ((1 << s) - d)` is below `2^63`. -/
theorem int64_in_range {d : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) :
    d ≤ 2 ^ shiftCount d ∧ 2 ^ 32 * (2 ^ shiftCount d - d) < 2 ^ 63 := by
  have hd32 : d ≤ 2 ^ 32 := by omega
  rw [shiftCount_eq_shiftFor hd hd'.le]
  obtain ⟨hle, -⟩ := FastDiv.shiftFor_spec hd hd32
  have hlt := FastDiv.pow_shift_lt hd hd32
  refine ⟨hle, ?_⟩
  have : 2 ^ FastDiv.shiftFor d - d < 2 ^ 31 := by omega
  calc 2 ^ 32 * (2 ^ FastDiv.shiftFor d - d) < 2 ^ 32 * 2 ^ 31 := Nat.mul_lt_mul_of_pos_left this (by positivity)
    _ = 2 ^ 63 := by norm_num

/-- The in-kernel multiplier is `FastDiv.magic`, and it fits the `uint32` cast. -/
theorem kMagic_eq {d : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) :
    kMagic d = FastDiv.magic d ∧ kMagic d < 2 ^ 32 := by
  have h : kMagic d = FastDiv.magic d := by
    unfold kMagic FastDiv.magic; rw [shiftCount_eq_shiftFor hd hd'.le]
  exact ⟨h, h ▸ FastDiv.magic_lt hd hd'⟩

/-- **The helper returns `(x / d, x % d)`** for `0 < d < 2^31` and `x < 2^31`. The quotient
also fits the `int32` cast, and `q * d ≤ x`, so the `int32` remainder `x - q * d` neither
overflows nor goes negative. -/
specification kernel_divmod {d x : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) (hx : x < 2 ^ 31) :
    kQuot d x = x / d ∧ x - kQuot d x * d = x % d ∧ kQuot d x < 2 ^ 31 ∧ kQuot d x * d ≤ x := by
  have hq : kQuot d x = x / d := by
    unfold kQuot
    rw [(kMagic_eq hd hd').1, shiftCount_eq_shiftFor hd hd'.le]
    exact FastDiv.bitvec_quotient hd hd' hx
  rw [hq]
  refine ⟨rfl, ?_, lt_of_le_of_lt (Nat.div_le_self x d) hx, Nat.div_mul_le_self x d⟩
  rw [Nat.mod_eq_sub_mul_div, Nat.mul_comm]

end VeriTile.Bench.Optimizations.InductorDivmod
