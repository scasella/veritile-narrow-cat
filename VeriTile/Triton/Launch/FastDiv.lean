/-
VeriTile.Triton.Launch.FastDiv

Division by a runtime divisor with host-computed constants (the ATen
`IntDivider` form, also used by the stage-11 concatenation kernel): for a
divisor `d` with `0 < d < 2^31`, the host computes

  `s = ⌈log₂ d⌉` (the least `s` with `d ≤ 2^s`) and
  `m = ⌊2^32 · (2^s − d) / d⌋ + 1`,

and the kernel computes the quotient of any `x < 2^31` as
`(umulhi₃₂(x, m) + x) >> s`, where `umulhi₃₂(x, m) = ⌊x · m / 2^32⌋`.

* `FastDiv.magic_quotient` — in ℕ: `((x * m) / 2^32 + x) / 2^s = x / d`.
* `FastDiv.magic_lt` / `FastDiv.sum_lt` — no 32-bit overflow: `m < 2^32` and
  `(x * m) / 2^32 + x < 2^32`.
* `FastDiv.bitvec_quotient` — the same computation in 32-bit unsigned
  arithmetic (`BitVec 32`): the high word of the 64-bit product, a wrapping
  32-bit add, and a logical right shift give `x / d`.
-/

import Mathlib

namespace VeriTile.Triton

namespace FastDiv

/-- Least `s` with `d ≤ 2^s`, searched up to `2^32` (structural, so concrete
values reduce in the kernel). -/
def shiftFor (d : Nat) : Nat := go 0 33
where
  go (s : Nat) : Nat → Nat
    | 0 => s
    | fuel + 1 => if d ≤ 2 ^ s then s else go (s + 1) fuel

/-- The multiplier `⌊2^32 · (2^s − d) / d⌋ + 1`. -/
def magic (d : Nat) : Nat := 2 ^ 32 * (2 ^ shiftFor d - d) / d + 1

/-- The kernel's quotient, in ℕ. -/
def quot (d x : Nat) : Nat := ((x * magic d) / 2 ^ 32 + x) / 2 ^ shiftFor d

/-- `shiftFor d` is the least `s` with `d ≤ 2^s`, for `0 < d ≤ 2^32`. -/
theorem shiftFor_spec {d : Nat} (hd : 0 < d) (hd' : d ≤ 2 ^ 32) :
    d ≤ 2 ^ shiftFor d ∧ (shiftFor d = 0 ∨ 2 ^ (shiftFor d - 1) < d) := by
  have key : ∀ fuel s, (s = 0 ∨ 2 ^ (s - 1) < d) → d ≤ 2 ^ (s + fuel) →
      d ≤ 2 ^ shiftFor.go d s fuel ∧ (shiftFor.go d s fuel = 0 ∨ 2 ^ (shiftFor.go d s fuel - 1) < d) := by
    intro fuel
    induction fuel with
    | zero => intro s h1 h2; simpa [shiftFor.go] using ⟨h2, h1⟩
    | succ f ih =>
      intro s h1 h2
      simp only [shiftFor.go]
      split
      · rename_i h; exact ⟨h, h1⟩
      · rename_i h
        apply ih (s + 1) (Or.inr (by simpa using Nat.lt_of_not_le h))
        rw [show s + 1 + f = s + (f + 1) by omega]; exact h2
  exact key 33 0 (Or.inl rfl) (le_trans hd' (Nat.pow_le_pow_right (by norm_num) (by norm_num)))

/-- `2^s < 2 * d` for the chosen shift. -/
theorem pow_shift_lt {d : Nat} (hd : 0 < d) (hd' : d ≤ 2 ^ 32) : 2 ^ shiftFor d < 2 * d := by
  obtain ⟨hle, h⟩ := shiftFor_spec hd hd'
  rcases h with h | h
  · rw [h]; omega
  · have hne : shiftFor d ≠ 0 := by
      intro h0; rw [h0] at hle h; simp at hle h; omega
    have : 2 ^ shiftFor d = 2 * 2 ^ (shiftFor d - 1) := by
      rw [← pow_succ']; congr 1; omega
    omega

/-- The quotient without the 32-bit split: `quot d x = x * (⌊2^32·2^s/d⌋ + 1) / (2^32·2^s)`. -/
theorem quot_eq {d x : Nat} (hd : 0 < d) (hd' : d ≤ 2 ^ 32) :
    quot d x = x * (2 ^ 32 * 2 ^ shiftFor d / d + 1) / (2 ^ 32 * 2 ^ shiftFor d) := by
  obtain ⟨hle, -⟩ := shiftFor_spec hd hd'
  unfold quot magic
  set P := 2 ^ shiftFor d with hP
  have hM : 2 ^ 32 * (P - d) / d + 1 + 2 ^ 32 = 2 ^ 32 * P / d + 1 := by
    have h1 : 2 ^ 32 * P = 2 ^ 32 * (P - d) + 2 ^ 32 * d := by
      rw [← Nat.mul_add, Nat.sub_add_cancel hle]
    rw [h1, Nat.add_mul_div_right _ _ hd]; ring
  have h2 : x * (2 ^ 32 * (P - d) / d + 1) / 2 ^ 32 + x
      = x * (2 ^ 32 * (P - d) / d + 1 + 2 ^ 32) / 2 ^ 32 := by
    rw [Nat.mul_add x _ (2 ^ 32), Nat.add_mul_div_right _ _ (by positivity)]
  rw [h2, hM, Nat.div_div_eq_div_mul]

/-- **Fast division is exact** for `0 < d < 2^31` and `x < 2^31`. -/
theorem magic_quotient {d x : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) (hx : x < 2 ^ 31) :
    quot d x = x / d := by
  have hd32 : d ≤ 2 ^ 32 := by omega
  obtain ⟨hle, -⟩ := shiftFor_spec hd hd32
  rw [quot_eq hd hd32]
  set P := 2 ^ shiftFor d
  set N := 2 ^ 32 * P with hN
  set K := N / d with hK
  have hNpos : 0 < N := by positivity
  have hKd : d * K ≤ N := Nat.mul_div_le N d
  have hKd' : N < d * (K + 1) := by
    have := Nat.lt_mul_div_succ N hd; rw [Nat.mul_comm] at this; simpa [hK, Nat.mul_comm] using this
  have hdx : d * x < N := by
    have : d * x ≤ P * x := Nat.mul_le_mul_right x hle
    have : P * x < P * 2 ^ 32 := Nat.mul_lt_mul_of_pos_left (by omega) (by positivity)
    rw [hN, Nat.mul_comm (2 ^ 32) P]; omega
  set q := x / d with hq
  have hqd : q * d ≤ x := Nat.div_mul_le_self x d
  have hxq : x < (q + 1) * d := by
    have := Nat.lt_mul_div_succ x hd; rw [Nat.mul_comm] at this; simpa [hq, Nat.mul_comm] using this
  apply Nat.div_eq_of_lt_le
  · -- q * N ≤ x * (K + 1)
    calc q * N ≤ q * (d * (K + 1)) := Nat.mul_le_mul_left q hKd'.le
      _ = (q * d) * (K + 1) := by ring
      _ ≤ x * (K + 1) := Nat.mul_le_mul_right _ hqd
  · -- x * (K + 1) < (q + 1) * N, multiplied through by d
    have h1 : d * (x * (K + 1)) ≤ x * N + d * x := by
      calc d * (x * (K + 1)) = x * (d * K) + d * x := by ring
        _ ≤ x * N + d * x := by have := Nat.mul_le_mul_left x hKd; omega
    have h2 : x * N + d * x < (x + 1) * N := by nlinarith
    have h3 : (x + 1) * N ≤ d * ((q + 1) * N) := by
      have : x + 1 ≤ (q + 1) * d := hxq
      calc (x + 1) * N ≤ ((q + 1) * d) * N := Nat.mul_le_mul_right N this
        _ = d * ((q + 1) * N) := by ring
    exact Nat.lt_of_mul_lt_mul_left (a := d) (by omega)

/-- The multiplier fits in 32 bits. -/
theorem magic_lt {d : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) : magic d < 2 ^ 32 := by
  have hd32 : d ≤ 2 ^ 32 := by omega
  obtain ⟨hle, -⟩ := shiftFor_spec hd hd32
  have hlt := pow_shift_lt hd hd32
  unfold magic
  set e := 2 ^ shiftFor d - d with he
  have hed : e + 1 ≤ d := by omega
  have hQ : 2 ^ 32 * e / d < 2 ^ 32 - 1 := by
    rw [Nat.div_lt_iff_lt_mul hd]
    have : 2 ^ 32 * e + 2 ^ 32 ≤ 2 ^ 32 * d := by rw [← Nat.mul_add_one]; exact Nat.mul_le_mul_left _ hed
    have : d < 2 ^ 32 := by omega
    have : (2 ^ 32 - 1) * d = 2 ^ 32 * d - d := by rw [Nat.sub_mul, one_mul]
    omega
  omega

/-- The 32-bit add `umulhi(x, m) + x` does not wrap. -/
theorem sum_lt {d x : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) (hx : x < 2 ^ 31) :
    (x * magic d) / 2 ^ 32 + x < 2 ^ 32 := by
  have hm := magic_lt hd hd'
  have : x * magic d / 2 ^ 32 ≤ x := by
    rw [Nat.div_le_iff_le_mul_add_pred (by positivity)]
    have := Nat.mul_le_mul_left x hm.le
    omega
  omega

/-- High 32 bits of the 64-bit unsigned product (`tl.umulhi` on `uint32`). -/
def umulhi32 (a b : BitVec 32) : BitVec 32 := BitVec.ofNat 32 ((a.toNat * b.toNat) / 2 ^ 32)

/-- **The 32-bit computation the kernel performs** equals `x / d`. -/
theorem bitvec_quotient {d x : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) (hx : x < 2 ^ 31) :
    ((umulhi32 (BitVec.ofNat 32 x) (BitVec.ofNat 32 (magic d)) + BitVec.ofNat 32 x)
        >>> shiftFor d).toNat = x / d := by
  have hm := magic_lt hd hd'
  have hs := sum_lt hd hd' hx
  have hq := magic_quotient hd hd' hx
  unfold quot at hq
  have hx32 : x < 2 ^ 32 := by omega
  simp only [umulhi32, BitVec.toNat_ushiftRight, BitVec.toNat_add, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt hx32, Nat.mod_eq_of_lt hm, Nat.shiftRight_eq_div_pow]
  have h1 : x * magic d / 2 ^ 32 < 2 ^ 32 := by omega
  rw [Nat.mod_eq_of_lt h1, Nat.mod_eq_of_lt hs]
  exact hq

end FastDiv

end VeriTile.Triton
