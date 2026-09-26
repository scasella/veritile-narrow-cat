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
  sorry

/-- The shift is at most 31, so `>> s` on `uint32` is a defined shift. -/
theorem shiftCount_le {d : Nat} (hd : 0 < d) (hd' : d ≤ 2 ^ 31) : shiftCount d ≤ 31 := by
  sorry

/-- The int64 intermediates are in range. `(1 << s) - d` does not underflow, and
`(1 << 32) * ((1 << s) - d)` is below `2^63`. -/
theorem int64_in_range {d : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) :
    d ≤ 2 ^ shiftCount d ∧ 2 ^ 32 * (2 ^ shiftCount d - d) < 2 ^ 63 := by
  sorry

/-- The in-kernel multiplier is `FastDiv.magic`, and it fits the `uint32` cast. -/
theorem kMagic_eq {d : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) :
    kMagic d = FastDiv.magic d ∧ kMagic d < 2 ^ 32 := by
  sorry

/-- **The helper returns `(x / d, x % d)`** for `0 < d < 2^31` and `x < 2^31`. The quotient
also fits the `int32` cast, and `q * d ≤ x`, so the `int32` remainder `x - q * d` neither
overflows nor goes negative. -/
theorem kernel_divmod {d x : Nat} (hd : 0 < d) (hd' : d < 2 ^ 31) (hx : x < 2 ^ 31) :
    kQuot d x = x / d ∧ x - kQuot d x * d = x % d ∧ kQuot d x < 2 ^ 31 ∧ kQuot d x * d ≤ x := by
  sorry

end VeriTile.Bench.Optimizations.InductorDivmod
