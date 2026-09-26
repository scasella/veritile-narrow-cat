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
  sorry

/-- **Fast division is exact** for `0 < d < 2^31` and `x < 2^31`. -/
theorem magic_quotient {d x : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) (hx : x < 2 ^ 31) :
    quot d x = x / d := by
  sorry

/-- The multiplier fits in 32 bits. -/
theorem magic_lt {d : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) : magic d < 2 ^ 32 := by
  sorry

/-- The 32-bit add `umulhi(x, m) + x` does not wrap. -/
theorem sum_lt {d x : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) (hx : x < 2 ^ 31) :
    (x * magic d) / 2 ^ 32 + x < 2 ^ 32 := by
  sorry

/-- High 32 bits of the 64-bit unsigned product (`tl.umulhi` on `uint32`). -/
def umulhi32 (a b : BitVec 32) : BitVec 32 := BitVec.ofNat 32 ((a.toNat * b.toNat) / 2 ^ 32)

/-- **The 32-bit computation the kernel performs** equals `x / d`. -/
theorem bitvec_quotient {d x : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) (hx : x < 2 ^ 31) :
    ((umulhi32 (BitVec.ofNat 32 x) (BitVec.ofNat 32 (magic d)) + BitVec.ofNat 32 x)
        >>> shiftFor d).toNat = x / d := by
  sorry

end FastDiv

end VeriTile.Triton
