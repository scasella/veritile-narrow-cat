import VeriTile.Triton

/-!
# Narrowing Inductor's dynamic-`cat` kernel to 32-bit size scalars (stage 13)

**Kernel.** For PyTorch #189940, `torch.compile(dynamic=True)` emits one pointwise kernel,
`triton_poi_fused_add_cat_0`. It takes seven size scalars `ks0..ks6`. `signature_to_meta` declares them
`i64` on purpose: a product of symbols may not fit in 32 bits even when each symbol does. That kernel is
**B0**. **N1** is the same source with those seven arguments declared `i32`. This is what current PyTorch
`main` emits under `assume_32bit_indexing`, and it is the variant that stage 12 measured as a speed-up.

**What is modelled.** The kernel's integer indexing, its masks, and the dataflow of its payload are written
here as a small typed expression language: `IE` for integers, `BE` for masks, `FE` for payload. The term
`catValue` is **generated** from the emitted source by `scripts/inductor_ir.py`, between the markers below.
`scripts/test_inductor_narrow.py` checks that it is the extractor's output for
`fixtures/gpu_emitted_B0.py`. Temporaries are inlined. The six dead comparisons in the source reach no
observable operation; the extractor reports them and checks that they are side-effect free.

**Integer semantics (Triton 3.8, as lowered to arith/LLVM).**
* The types are `int32` and `int64`. A binary operation on `int32` and `int64` sign-extends to `int64`.
* A Python integer literal takes the other operand's type. `tl.full(…, tl.int64)` is a strong `int64`.
* Every `+` and `*` wraps to its result type (two's complement). `.to(t)` wraps, or sign-extends, to `t`.
* `%` and `//` truncate toward zero (`arith.remsi` / `arith.divsi`), which is `Int.tmod` / `Int.tdiv`.
* A size scalar declared `i32` is passed by the launcher as its value wrapped to 32 bits.

**Payload.** The payload is symbolic (`Val`). A load denotes the memory cell it reads, `+` and `where` stay
as constructors, and the conversion to bf16 at the store is the same in both variants. So equal `Val`s give
equal stored values under **any** memory contents and **any** floating-point semantics. Floating point is
not modelled, and no floating-point claim is made.

**Hypotheses.** These are exactly what the codegen-time eligibility check reads from Inductor's own
symbolic state; see `scripts/inductor_divmod.py`.
* H1 `0 ≤ ks1 … ks6`: the size symbols' value ranges.
* H2 `ks0 = ks1 + ks2 + ks3 + ks4 + ks5 + ks6`: the precomputed replacement for `ks0`.
* H3 `xnumel = n · ks0`: the kernel's `numel` expression.
* H4 `xnumel ≤ 2^31 − 1`: the guard Inductor installs when it chooses int32 indexing. Dynamo re-checks it
  on every call, before the kernel is entered.

H4 is B0's own precondition, because `xnumel` is already an `i32` argument in B0. H1 and H3 are symbolic
identities. None of the four is a new runtime check.

**Assumptions not modelled.**
* A masked-off lane's address is never dereferenced, whatever its value. B0 already relies on this for
  past-the-end addresses. N1's inactive lanes may also compute wrapped addresses.
* The Triton → PTX → SASS toolchain is trusted, as elsewhere in VeriTile.
-/

namespace VeriTile.Bench.Optimizations.NarrowCat

/-! ## Semantics (hand-written) -/

inductive Ty | i32 | i64
  deriving DecidableEq, Repr

def Ty.bits : Ty → Nat
  | .i32 => 32
  | .i64 => 64

/-- Two's-complement wrap to a type; `none` is an untyped Python literal, which is left unwrapped. -/
def wrapT : Option Ty → Int → Int
  | none, v => v
  | some t, v => Int.bmod v (2 ^ t.bits)

/-- Result type of a binary operation. -/
def join : Option Ty → Option Ty → Option Ty
  | none, b => b
  | a, none => a
  | some .i64, _ => some .i64
  | _, some .i64 => some .i64
  | _, _ => some .i32

inductive IE
  | xindex
  | ks (i : Nat)
  | lit (v : Int)
  | slit (v : Int) (t : Ty)
  | cast (e : IE) (t : Ty)
  | add (a b : IE)
  | mul (a b : IE)
  | tmod (a b : IE)
  | tdiv (a b : IE)

inductive BE
  | xmask
  | lt (a b : IE)
  | ge (a b : IE)
  | and (a b : BE)

inductive FE
  | zero
  | load (p : Nat) (off : IE) (m : BE)
  | add (a b : FE)
  | where (c : BE) (a b : FE)

/-- One lane: its `xindex`, the size-scalar values the host computes, and `xnumel`. -/
structure Lane where
  xi : Int
  ks : Nat → Int
  xnumel : Int

/-- Integer evaluation when the size scalars are declared `ksTy`. -/
def evI (ksTy : Ty) (L : Lane) : IE → Int × Option Ty
  | .xindex => (wrapT (some .i32) L.xi, some .i32)
  | .ks i => (wrapT (some ksTy) (L.ks i), some ksTy)
  | .lit v => (v, none)
  | .slit v t => (wrapT (some t) v, some t)
  | .cast e t => (wrapT (some t) (evI ksTy L e).1, some t)
  | .add a b => let t := join (evI ksTy L a).2 (evI ksTy L b).2
      (wrapT t ((evI ksTy L a).1 + (evI ksTy L b).1), t)
  | .mul a b => let t := join (evI ksTy L a).2 (evI ksTy L b).2
      (wrapT t ((evI ksTy L a).1 * (evI ksTy L b).1), t)
  | .tmod a b => let t := join (evI ksTy L a).2 (evI ksTy L b).2
      (wrapT t (Int.tmod (evI ksTy L a).1 (evI ksTy L b).1), t)
  | .tdiv a b => let t := join (evI ksTy L a).2 (evI ksTy L b).2
      (wrapT t (Int.tdiv (evI ksTy L a).1 (evI ksTy L b).1), t)

def evB (ksTy : Ty) (L : Lane) : BE → Bool
  | .xmask => decide (wrapT (some .i32) L.xi < wrapT (some .i32) L.xnumel)
  | .lt a b => decide ((evI ksTy L a).1 < (evI ksTy L b).1)
  | .ge a b => decide ((evI ksTy L b).1 ≤ (evI ksTy L a).1)
  | .and a b => evB ksTy L a && evB ksTy L b

/-- Symbolic payload values. -/
inductive Val
  | zero
  | mem (p : Nat) (off : Int)
  | add (a b : Val)
  deriving DecidableEq

def evF (ksTy : Ty) (L : Lane) : FE → Val
  | .zero => .zero
  | .load p off m => if evB ksTy L m then .mem p (evI ksTy L off).1 else .zero
  | .add a b => .add (evF ksTy L a) (evF ksTy L b)
  | .where c a b => if evB ksTy L c then evF ksTy L a else evF ksTy L b

/-- Every memory read the lane performs, whether or not `where` later selects it. `tl.where` evaluates both
arms, so every enabled load in the tree executes. -/
def reads (ksTy : Ty) (L : Lane) : FE → List (Nat × Int)
  | .zero => []
  | .load p off m => if evB ksTy L m then [(p, (evI ksTy L off).1)] else []
  | .add a b => reads ksTy L a ++ reads ksTy L b
  | .where _ a b => reads ksTy L a ++ reads ksTy L b

/-! ## The emitted kernel (generated; do not edit) -/

-- BEGIN GENERATED (scripts/inductor_ir.py, fixtures/gpu_emitted_B0.py)
/-- `tmp74`, the stored value, with every temporary inlined (8 loads). -/
def catValue : FE :=
  (FE.where (BE.lt (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i64) Ty.i64)) (FE.where (BE.lt (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i64) Ty.i64)) (FE.where (BE.lt (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.ks 1) Ty.i64) Ty.i64)) (FE.load 0 (IE.add (IE.mul (IE.ks 1) (IE.tdiv IE.xindex (IE.ks 0))) (IE.tmod IE.xindex (IE.ks 0))) (BE.and (BE.and (BE.lt (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.ks 1) Ty.i64) Ty.i64)) (BE.lt (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i64) Ty.i64))) BE.xmask)) (FE.where (BE.and (BE.ge (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.ks 1) Ty.i64) Ty.i64)) (BE.lt (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.add (IE.ks 1) (IE.ks 3)) Ty.i64) Ty.i64))) (FE.load 1 (IE.add (IE.mul (IE.ks 3) (IE.tdiv IE.xindex (IE.ks 0))) (IE.add (IE.mul (IE.lit (-1)) (IE.ks 1)) (IE.tmod IE.xindex (IE.ks 0)))) (BE.and (BE.and (BE.and (BE.ge (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.ks 1) Ty.i64) Ty.i64)) (BE.lt (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.add (IE.ks 1) (IE.ks 3)) Ty.i64) Ty.i64))) (BE.lt (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i64) Ty.i64))) BE.xmask)) (FE.where (BE.and (BE.ge (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i32) (IE.cast (IE.add (IE.ks 1) (IE.ks 3)) Ty.i32)) (BE.lt (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i64) Ty.i64))) (FE.add (FE.load 2 (IE.add (IE.mul (IE.ks 2) (IE.tdiv IE.xindex (IE.ks 0))) (IE.add (IE.add (IE.mul (IE.lit (-1)) (IE.ks 1)) (IE.mul (IE.lit (-1)) (IE.ks 3))) (IE.tmod IE.xindex (IE.ks 0)))) (BE.and (BE.and (BE.ge (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i32) (IE.cast (IE.add (IE.ks 1) (IE.ks 3)) Ty.i32)) (BE.lt (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i64) Ty.i64))) BE.xmask)) (FE.load 3 (IE.add (IE.mul (IE.ks 2) (IE.tdiv IE.xindex (IE.ks 0))) (IE.add (IE.add (IE.mul (IE.lit (-1)) (IE.ks 1)) (IE.mul (IE.lit (-1)) (IE.ks 3))) (IE.tmod IE.xindex (IE.ks 0)))) (BE.and (BE.and (BE.ge (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i32) (IE.cast (IE.add (IE.ks 1) (IE.ks 3)) Ty.i32)) (BE.lt (IE.cast (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i64) Ty.i64))) BE.xmask))) FE.zero))) FE.zero) (FE.where (BE.ge (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i32) (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i32)) (FE.where (BE.lt (IE.cast (IE.cast (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.ks 4) Ty.i64) Ty.i64)) (FE.load 4 (IE.add (IE.mul (IE.ks 4) (IE.tdiv IE.xindex (IE.ks 0))) (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3)))) (BE.and (BE.and (BE.lt (IE.cast (IE.cast (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.ks 4) Ty.i64) Ty.i64)) (BE.ge (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i32) (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i32))) BE.xmask)) (FE.where (BE.and (BE.ge (IE.cast (IE.cast (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.ks 4) Ty.i64) Ty.i64)) (BE.lt (IE.cast (IE.cast (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.add (IE.ks 4) (IE.ks 5)) Ty.i64) Ty.i64))) (FE.load 5 (IE.add (IE.mul (IE.ks 5) (IE.tdiv IE.xindex (IE.ks 0))) (IE.add (IE.mul (IE.lit (-1)) (IE.ks 4)) (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))))) (BE.and (BE.and (BE.and (BE.ge (IE.cast (IE.cast (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.ks 4) Ty.i64) Ty.i64)) (BE.lt (IE.cast (IE.cast (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))) Ty.i64) Ty.i64) (IE.cast (IE.cast (IE.add (IE.ks 4) (IE.ks 5)) Ty.i64) Ty.i64))) (BE.ge (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i32) (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i32))) BE.xmask)) (FE.where (BE.and (BE.ge (IE.cast (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))) Ty.i32) (IE.cast (IE.add (IE.ks 4) (IE.ks 5)) Ty.i32)) (BE.ge (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i32) (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i32))) (FE.add (FE.load 6 (IE.add (IE.mul (IE.ks 6) (IE.tdiv IE.xindex (IE.ks 0))) (IE.add (IE.add (IE.mul (IE.lit (-1)) (IE.ks 4)) (IE.mul (IE.lit (-1)) (IE.ks 5))) (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))))) (BE.and (BE.and (BE.ge (IE.cast (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))) Ty.i32) (IE.cast (IE.add (IE.ks 4) (IE.ks 5)) Ty.i32)) (BE.ge (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i32) (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i32))) BE.xmask)) (FE.load 7 (IE.add (IE.mul (IE.ks 6) (IE.tdiv IE.xindex (IE.ks 0))) (IE.add (IE.add (IE.mul (IE.lit (-1)) (IE.ks 4)) (IE.mul (IE.lit (-1)) (IE.ks 5))) (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))))) (BE.and (BE.and (BE.ge (IE.cast (IE.add (IE.add (IE.add (IE.tmod IE.xindex (IE.ks 0)) (IE.mul (IE.lit (-1)) (IE.ks 1))) (IE.mul (IE.lit (-1)) (IE.ks 2))) (IE.mul (IE.lit (-1)) (IE.ks 3))) Ty.i32) (IE.cast (IE.add (IE.ks 4) (IE.ks 5)) Ty.i32)) (BE.ge (IE.cast (IE.tmod IE.xindex (IE.ks 0)) Ty.i32) (IE.cast (IE.add (IE.add (IE.ks 1) (IE.ks 2)) (IE.ks 3)) Ty.i32))) BE.xmask))) FE.zero))) FE.zero))
-- END GENERATED

/-- The store: `tl.store(out_ptr0 + (x2), tmp74, xmask)` with `x2 = xindex`. -/
def catStoreOffset : IE := IE.xindex
def catStoreMask : BE := BE.xmask

/-! ## Results -/

/-- The launch grid never wraps `xindex`. `xoffset = pid · XBLOCK` with `XBLOCK = 2^k ≤ 2^31` and
`pid < ⌈xnumel / XBLOCK⌉`, so every lane satisfies `0 ≤ xindex < 2^31` when `xnumel ≤ 2^31 − 1`. -/
theorem grid_xindex_lt {xnumel k pid j : Nat} (hk : k ≤ 31) (hx : xnumel ≤ 2 ^ 31 - 1)
    (hpid : pid < (xnumel + 2 ^ k - 1) / 2 ^ k) (hj : j < 2 ^ k) : pid * 2 ^ k + j < 2 ^ 31 := by
  sorry

/-- **Narrowing preserves every observable of a lane.** For every lane of the grid (`0 ≤ xindex < 2^31`)
under H1–H4, B0 (`i64` size scalars) and N1 (`i32`) agree on:
* the store mask;
* when it is set, the store address and the stored payload;
* the full list of memory reads the lane performs.
Active lanes agree on every observable. Inactive lanes perform no memory operation in either variant. -/
specification narrow_cat_equiv (L : Lane) (n : Nat)
    (H1 : ∀ i, 1 ≤ i → i ≤ 6 → 0 ≤ L.ks i)
    (H2 : L.ks 0 = L.ks 1 + L.ks 2 + L.ks 3 + L.ks 4 + L.ks 5 + L.ks 6)
    (H3 : L.xnumel = n * L.ks 0) (H4 : L.xnumel ≤ 2 ^ 31 - 1)
    (hxi : 0 ≤ L.xi) (hxi' : L.xi < 2 ^ 31) :
    evB .i64 L catStoreMask = evB .i32 L catStoreMask ∧
    (evB .i64 L catStoreMask = Bool.true →
      (evI .i64 L catStoreOffset).1 = (evI .i32 L catStoreOffset).1 ∧
      evF .i64 L catValue = evF .i32 L catValue) ∧
    reads .i64 L catValue = reads .i32 L catValue := by
  sorry

/-- No new undefined behaviour. Whenever the launch has any lane to do (`xnumel > 0`), N1's divisor for `%`
and `//` (`ks0` passed as `i32`) equals B0's, and it is positive. So neither variant divides by zero, and
`INT_MIN / −1` cannot occur. -/
theorem narrow_cat_divisor (L : Lane) (n : Nat)
    (H1 : ∀ i, 1 ≤ i → i ≤ 6 → 0 ≤ L.ks i)
    (H2 : L.ks 0 = L.ks 1 + L.ks 2 + L.ks 3 + L.ks 4 + L.ks 5 + L.ks 6)
    (H3 : L.xnumel = n * L.ks 0) (H4 : L.xnumel ≤ 2 ^ 31 - 1) (hxn : 0 < L.xnumel) :
    (evI .i32 L (.ks 0)).1 = (evI .i64 L (.ks 0)).1 ∧ 0 < (evI .i32 L (.ks 0)).1 := by
  sorry

/-- **The stage-12 in-kernel guard is always true** in the compiled graph's domain. When the kernel has any
lane to do (`xnumel > 0`), every size scalar is in `[0, 2^31)`, so guarded N's `else` branch (the original
body) is unreachable. -/
theorem guarded_N_else_unreachable (L : Lane) (n : Nat)
    (H1 : ∀ i, 1 ≤ i → i ≤ 6 → 0 ≤ L.ks i)
    (H2 : L.ks 0 = L.ks 1 + L.ks 2 + L.ks 3 + L.ks 4 + L.ks 5 + L.ks 6)
    (H3 : L.xnumel = n * L.ks 0) (H4 : L.xnumel ≤ 2 ^ 31 - 1) (hxn : 0 < L.xnumel) :
    ∀ i, i ≤ 6 → 0 ≤ L.ks i ∧ L.ks i < 2 ^ 31 := by
  sorry

end VeriTile.Bench.Optimizations.NarrowCat
