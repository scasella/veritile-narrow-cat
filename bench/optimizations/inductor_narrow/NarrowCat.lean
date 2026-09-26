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

/-! ## Proof infrastructure -/

/-- A value representable in `int32`. -/
def R32 (v : Int) : Prop := -2147483648 ≤ v ∧ v < 2147483648

theorem wrap_R32 {v : Int} (h : R32 v) (o : Option Ty) : wrapT o v = v := by
  rcases o with _ | t
  · rfl
  · cases t <;> simp only [wrapT, Ty.bits] <;>
      exact Int.bmod_eq_of_le_mul_two (by unfold R32 at h; norm_num; omega) (by unfold R32 at h; norm_num; omega)

/-- Unbounded (exact) value of an integer expression. -/
def exI (L : Lane) : IE → Int
  | .xindex => L.xi
  | .ks i => L.ks i
  | .lit v => v
  | .slit v _ => v
  | .cast e _ => exI L e
  | .add a b => exI L a + exI L b
  | .mul a b => exI L a * exI L b
  | .tmod a b => Int.tmod (exI L a) (exI L b)
  | .tdiv a b => Int.tdiv (exI L a) (exI L b)

/-- Every subterm's exact value is representable in `int32`. -/
def FitsI (L : Lane) : IE → Prop
  | .xindex => R32 L.xi
  | .ks i => R32 (L.ks i)
  | .lit v => R32 v
  | .slit v _ => R32 v
  | .cast e _ => FitsI L e
  | .add a b => FitsI L a ∧ FitsI L b ∧ R32 (exI L a + exI L b)
  | .mul a b => FitsI L a ∧ FitsI L b ∧ R32 (exI L a * exI L b)
  | .tmod a b => FitsI L a ∧ FitsI L b ∧ R32 (Int.tmod (exI L a) (exI L b))
  | .tdiv a b => FitsI L a ∧ FitsI L b ∧ R32 (Int.tdiv (exI L a) (exI L b))

theorem exI_R32 (L : Lane) : ∀ e, FitsI L e → R32 (exI L e)
  | .xindex, h => h
  | .ks _, h => h
  | .lit _, h => h
  | .slit _ _, h => h
  | .cast e _, h => exI_R32 L e h
  | .add _ _, ⟨_, _, h⟩ => h
  | .mul _ _, ⟨_, _, h⟩ => h
  | .tmod _ _, ⟨_, _, h⟩ => h
  | .tdiv _ _, ⟨_, _, h⟩ => h

theorem evI_fits (t : Ty) (L : Lane) : ∀ e, FitsI L e → (evI t L e).1 = exI L e
  | .xindex, h => by simp only [evI, exI]; exact wrap_R32 h _
  | .ks _, h => by simp only [evI, exI]; exact wrap_R32 h _
  | .lit _, _ => rfl
  | .slit _ _, h => by simp only [evI, exI]; exact wrap_R32 h _
  | .cast e _, h => by simp only [evI, exI]; rw [evI_fits t L e h]; exact wrap_R32 (exI_R32 L e h) _
  | .add a b, ⟨ha, hb, h⟩ => by simp only [evI, exI]; rw [evI_fits t L a ha, evI_fits t L b hb]; exact wrap_R32 h _
  | .mul a b, ⟨ha, hb, h⟩ => by simp only [evI, exI]; rw [evI_fits t L a ha, evI_fits t L b hb]; exact wrap_R32 h _
  | .tmod a b, ⟨ha, hb, h⟩ => by simp only [evI, exI]; rw [evI_fits t L a ha, evI_fits t L b hb]; exact wrap_R32 h _
  | .tdiv a b, ⟨ha, hb, h⟩ => by simp only [evI, exI]; rw [evI_fits t L a ha, evI_fits t L b hb]; exact wrap_R32 h _

def FitsB (L : Lane) : BE → Prop
  | .xmask => True
  | .lt a b => FitsI L a ∧ FitsI L b
  | .ge a b => FitsI L a ∧ FitsI L b
  | .and a b => FitsB L a ∧ FitsB L b

def FitsF (L : Lane) : FE → Prop
  | .zero => True
  | .load _ off m => FitsI L off ∧ FitsB L m
  | .add a b => FitsF L a ∧ FitsF L b
  | .where c a b => FitsB L c ∧ FitsF L a ∧ FitsF L b

theorem evB_fits (L : Lane) : ∀ b, FitsB L b → evB .i64 L b = evB .i32 L b
  | .xmask, _ => rfl
  | .lt a b, ⟨ha, hb⟩ => by simp only [evB]; rw [evI_fits _ L a ha, evI_fits _ L b hb, evI_fits _ L a ha, evI_fits _ L b hb]
  | .ge a b, ⟨ha, hb⟩ => by simp only [evB]; rw [evI_fits _ L a ha, evI_fits _ L b hb, evI_fits _ L a ha, evI_fits _ L b hb]
  | .and a b, ⟨ha, hb⟩ => by simp only [evB]; rw [evB_fits L a ha, evB_fits L b hb]

theorem evF_fits (L : Lane) : ∀ f, FitsF L f → evF .i64 L f = evF .i32 L f ∧ reads .i64 L f = reads .i32 L f
  | .zero, _ => ⟨rfl, rfl⟩
  | .load p off m, ⟨ho, hm⟩ => by
      simp only [evF, reads]; rw [evB_fits L m hm, evI_fits _ L off ho, evI_fits _ L off ho]; exact ⟨rfl, rfl⟩
  | .add a b, ⟨ha, hb⟩ => by
      obtain ⟨a1, a2⟩ := evF_fits L a ha; obtain ⟨b1, b2⟩ := evF_fits L b hb
      simp only [evF, reads]; rw [a1, a2, b1, b2]; exact ⟨rfl, rfl⟩
  | .where c a b, ⟨hc, ha, hb⟩ => by
      obtain ⟨a1, a2⟩ := evF_fits L a ha; obtain ⟨b1, b2⟩ := evF_fits L b hb
      simp only [evF, reads]; rw [evB_fits L c hc, a1, a2, b1, b2]; exact ⟨rfl, rfl⟩

/-- Every load's mask is `… & xmask`. -/
def maskedX : FE → Bool
  | .zero => Bool.true
  | .load _ _ (.and _ .xmask) => Bool.true
  | .load _ _ _ => Bool.false
  | .add a b => maskedX a && maskedX b
  | .where _ a b => maskedX a && maskedX b

theorem reads_inactive (t : Ty) (L : Lane) (hx : evB t L .xmask = Bool.false) :
    ∀ f, maskedX f = Bool.true → reads t L f = []
  | .zero, _ => rfl
  | .load _ _ (.and _ .xmask), _ => by simp only [reads, evB] at hx ⊢; simp [hx]
  | .load _ _ .xmask, h => by simp [maskedX] at h
  | .load _ _ (.lt _ _), h => by simp [maskedX] at h
  | .load _ _ (.ge _ _), h => by simp [maskedX] at h
  | .load _ _ (.and _ (.lt _ _)), h => by simp [maskedX] at h
  | .load _ _ (.and _ (.ge _ _)), h => by simp [maskedX] at h
  | .load _ _ (.and _ (.and _ _)), h => by simp [maskedX] at h
  | .add a b, h => by
      simp only [maskedX, Bool.and_eq_true] at h
      simp only [reads]; rw [reads_inactive t L hx a h.1, reads_inactive t L hx b h.2]; rfl
  | .where _ a b, h => by
      simp only [maskedX, Bool.and_eq_true] at h
      simp only [reads]; rw [reads_inactive t L hx a h.1, reads_inactive t L hx b h.2]; rfl

theorem catValue_maskedX : maskedX catValue = Bool.true := by decide

set_option maxHeartbeats 4000000 in
/-- On an active lane under H1–H4, every integer subterm of the kernel fits `int32`. The key facts are
`ks_i ≤ ks0` (H1, H2) and `ks0 · (x1 + 1) ≤ n · ks0 = xnumel ≤ 2^31 − 1` (H3, H4, the lane is active). -/
theorem cat_fits (L : Lane) (n : Nat)
    (H1 : ∀ i, 1 ≤ i → i ≤ 6 → 0 ≤ L.ks i)
    (H2 : L.ks 0 = L.ks 1 + L.ks 2 + L.ks 3 + L.ks 4 + L.ks 5 + L.ks 6)
    (H3 : L.xnumel = n * L.ks 0) (H4 : L.xnumel ≤ 2 ^ 31 - 1)
    (hxi : 0 ≤ L.xi) (hact : L.xi < L.xnumel) : FitsF L catValue := by
  have k1 := H1 1 (by omega) (by omega); have k2 := H1 2 (by omega) (by omega)
  have k3 := H1 3 (by omega) (by omega); have k4 := H1 4 (by omega) (by omega)
  have k5 := H1 5 (by omega) (by omega); have k6 := H1 6 (by omega) (by omega)
  have H4' : L.xnumel ≤ 2147483647 := by norm_num at H4; exact H4
  have hk0 : 0 < L.ks 0 := by
    rcases (show L.ks 0 = 0 ∨ 0 < L.ks 0 by omega) with h | h
    · rw [H3, h, mul_zero] at hact; omega
    · exact h
  have hqr : Int.tmod L.xi (L.ks 0) + L.ks 0 * Int.tdiv L.xi (L.ks 0) = L.xi := Int.tmod_add_mul_tdiv _ _
  have hr0 : 0 ≤ Int.tmod L.xi (L.ks 0) := Int.tmod_nonneg _ hxi
  have hr1 : Int.tmod L.xi (L.ks 0) < L.ks 0 := Int.tmod_lt_of_pos _ hk0
  have hq0 : 0 ≤ Int.tdiv L.xi (L.ks 0) := Int.tdiv_nonneg hxi hk0.le
  simp only [catValue, FitsF, FitsB, FitsI, exI, R32]
  generalize Int.tdiv L.xi (L.ks 0) = q at hqr hq0 ⊢
  generalize Int.tmod L.xi (L.ks 0) = r at hqr hr0 hr1 ⊢
  have hqn : q + 1 ≤ (n : Int) := by
    by_contra hc
    have : (n : Int) * L.ks 0 ≤ q * L.ks 0 := Int.mul_le_mul_of_nonneg_right (by omega) hk0.le
    have : L.ks 0 * q = q * L.ks 0 := Int.mul_comm _ _
    omega
  have hrow : L.ks 0 * q + L.ks 0 ≤ L.xnumel := by
    have : (q + 1) * L.ks 0 ≤ (n : Int) * L.ks 0 := Int.mul_le_mul_of_nonneg_right hqn hk0.le
    have e : (q + 1) * L.ks 0 = L.ks 0 * q + L.ks 0 := by ring
    omega
  have p : ∀ i, 1 ≤ i → i ≤ 6 → 0 ≤ L.ks i * q ∧ L.ks i * q ≤ L.ks 0 * q := by
    intro i hi1 hi6
    have hle : L.ks i ≤ L.ks 0 := by
      interval_cases i <;> omega
    exact ⟨Int.mul_nonneg (H1 i hi1 hi6) hq0, Int.mul_le_mul_of_nonneg_right hle hq0⟩
  have p1 := p 1 (by omega) (by omega); have p2 := p 2 (by omega) (by omega)
  have p3 := p 3 (by omega) (by omega); have p4 := p 4 (by omega) (by omega)
  have p5 := p 5 (by omega) (by omega); have p6 := p 6 (by omega) (by omega)
  have hqle : q ≤ L.ks 0 * q := le_mul_of_one_le_left hq0 (by omega)
  and_intros <;> first | trivial | omega

/-- The store: `tl.store(out_ptr0 + (x2), tmp74, xmask)` with `x2 = xindex`. -/
def catStoreOffset : IE := IE.xindex
def catStoreMask : BE := BE.xmask

/-! ## Results -/

/-- The launch grid never wraps `xindex`. `xoffset = pid · XBLOCK` with `XBLOCK = 2^k ≤ 2^31` and
`pid < ⌈xnumel / XBLOCK⌉`, so every lane satisfies `0 ≤ xindex < 2^31` when `xnumel ≤ 2^31 − 1`. -/
theorem grid_xindex_lt {xnumel k pid j : Nat} (hk : k ≤ 31) (hx : xnumel ≤ 2 ^ 31 - 1)
    (hpid : pid < (xnumel + 2 ^ k - 1) / 2 ^ k) (hj : j < 2 ^ k) : pid * 2 ^ k + j < 2 ^ 31 := by
  have hpos : 0 < 2 ^ k := by positivity
  have hPQ : 2 ^ (31 - k) * 2 ^ k = 2147483648 := by rw [← pow_add, Nat.sub_add_cancel hk]; norm_num
  have hQP : 2 ^ k * 2 ^ (31 - k) = 2147483648 := by rw [Nat.mul_comm]; exact hPQ
  have hx' : xnumel ≤ 2147483647 := by norm_num at hx; exact hx
  have hdiv : (xnumel + 2 ^ k - 1) / 2 ^ k ≤ 2 ^ (31 - k) := by
    rw [Nat.div_le_iff_le_mul_add_pred hpos]
    omega
  have h1 : pid + 1 ≤ 2 ^ (31 - k) := by omega
  have h2 : (pid + 1) * 2 ^ k ≤ 2 ^ (31 - k) * 2 ^ k := Nat.mul_le_mul_right _ h1
  have h3 : (pid + 1) * 2 ^ k = pid * 2 ^ k + 2 ^ k := by ring
  norm_num
  omega

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
  have hxn0 : 0 ≤ L.xnumel := by
    rw [H3]; have := H1 1 (by omega) (by omega); exact Int.mul_nonneg (by positivity) (by
      rw [H2]; have := H1 2 (by omega) (by omega); have := H1 3 (by omega) (by omega)
      have := H1 4 (by omega) (by omega); have := H1 5 (by omega) (by omega)
      have := H1 6 (by omega) (by omega); omega)
  have hmask : evB .i64 L catStoreMask = evB .i32 L catStoreMask := rfl
  have hoff : (evI .i64 L catStoreOffset).1 = (evI .i32 L catStoreOffset).1 := rfl
  by_cases hact : L.xi < L.xnumel
  · obtain ⟨e1, e2⟩ := evF_fits L catValue (cat_fits L n H1 H2 H3 H4 hxi hact)
    exact ⟨hmask, fun _ => ⟨hoff, e1⟩, e2⟩
  · have hw1 : wrapT (some .i32) L.xi = L.xi := wrap_R32 ⟨by omega, by omega⟩ _
    have hw2 : wrapT (some .i32) L.xnumel = L.xnumel := wrap_R32 ⟨by omega, by norm_num at H4; omega⟩ _
    have hx : ∀ t, evB t L .xmask = Bool.false := by
      intro t; simp only [evB, hw1, hw2]; simp; omega
    refine ⟨hmask, fun h => ?_, ?_⟩
    · have := hx .i64; simp only [catStoreMask] at h; rw [this] at h; exact absurd h (by decide)
    · rw [reads_inactive .i64 L (hx .i64) catValue catValue_maskedX,
        reads_inactive .i32 L (hx .i32) catValue catValue_maskedX]

/-- No new undefined behaviour. Whenever the launch has any lane to do (`xnumel > 0`), N1's divisor for `%`
and `//` (`ks0` passed as `i32`) equals B0's, and it is positive. So neither variant divides by zero, and
`INT_MIN / −1` cannot occur. -/
theorem narrow_cat_divisor (L : Lane) (n : Nat)
    (H1 : ∀ i, 1 ≤ i → i ≤ 6 → 0 ≤ L.ks i)
    (H2 : L.ks 0 = L.ks 1 + L.ks 2 + L.ks 3 + L.ks 4 + L.ks 5 + L.ks 6)
    (H3 : L.xnumel = n * L.ks 0) (H4 : L.xnumel ≤ 2 ^ 31 - 1) (hxn : 0 < L.xnumel) :
    (evI .i32 L (.ks 0)).1 = (evI .i64 L (.ks 0)).1 ∧ 0 < (evI .i32 L (.ks 0)).1 := by
  have hs : 0 ≤ L.ks 0 := by
    rw [H2]; have := H1 1 (by omega) (by omega); have := H1 2 (by omega) (by omega)
    have := H1 3 (by omega) (by omega); have := H1 4 (by omega) (by omega)
    have := H1 5 (by omega) (by omega); have := H1 6 (by omega) (by omega); omega
  have hpos : 0 < L.ks 0 := by
    rcases (show L.ks 0 = 0 ∨ 0 < L.ks 0 by omega) with h | h
    · rw [H3, h, mul_zero] at hxn; omega
    · exact h
  have hn : (1 : Int) ≤ n := by
    rcases (show (n : Int) = 0 ∨ 1 ≤ (n : Int) by omega) with h | h
    · rw [H3, h, zero_mul] at hxn; omega
    · exact h
  have hle : L.ks 0 ≤ L.xnumel := by
    rw [H3]; have := Int.mul_le_mul_of_nonneg_right hn hs; simpa using this
  have hR : R32 (L.ks 0) := ⟨by omega, by norm_num at H4; omega⟩
  simp only [evI]
  rw [wrap_R32 hR, wrap_R32 hR]
  exact ⟨rfl, hpos⟩

/-- **The stage-12 in-kernel guard is always true** in the compiled graph's domain. When the kernel has any
lane to do (`xnumel > 0`), every size scalar is in `[0, 2^31)`, so guarded N's `else` branch (the original
body) is unreachable. -/
theorem guarded_N_else_unreachable (L : Lane) (n : Nat)
    (H1 : ∀ i, 1 ≤ i → i ≤ 6 → 0 ≤ L.ks i)
    (H2 : L.ks 0 = L.ks 1 + L.ks 2 + L.ks 3 + L.ks 4 + L.ks 5 + L.ks 6)
    (H3 : L.xnumel = n * L.ks 0) (H4 : L.xnumel ≤ 2 ^ 31 - 1) (hxn : 0 < L.xnumel) :
    ∀ i, i ≤ 6 → 0 ≤ L.ks i ∧ L.ks i < 2 ^ 31 := by
  have k1 := H1 1 (by omega) (by omega); have k2 := H1 2 (by omega) (by omega)
  have k3 := H1 3 (by omega) (by omega); have k4 := H1 4 (by omega) (by omega)
  have k5 := H1 5 (by omega) (by omega); have k6 := H1 6 (by omega) (by omega)
  have hs : 0 ≤ L.ks 0 := by rw [H2]; omega
  have hn : (1 : Int) ≤ n := by
    rcases (show (n : Int) = 0 ∨ 1 ≤ (n : Int) by omega) with h | h
    · rw [H3, h, zero_mul] at hxn; omega
    · exact h
  have hle : L.ks 0 ≤ L.xnumel := by
    rw [H3]; have := Int.mul_le_mul_of_nonneg_right hn hs; simpa using this
  have H4' : L.xnumel ≤ 2147483647 := by norm_num at H4; exact H4
  intro i hi
  interval_cases i <;> constructor <;> norm_num <;> omega

/-- **Nonempty launches and exact `i32` packing** (the rule's premises H1' and H3'). When every width and the row
count are at least 1, as Inductor's value ranges establish, the launch is nonempty and `0 < ks0 ≤ xnumel`.
Every size argument lies in `[0, 2^31)`, so passing it as `i32` is exact: the launcher's value equals the host's. -/
theorem narrow_cat_args_fit (L : Lane) (n : Nat)
    (H1' : ∀ i, 1 ≤ i → i ≤ 6 → 1 ≤ L.ks i)
    (H2 : L.ks 0 = L.ks 1 + L.ks 2 + L.ks 3 + L.ks 4 + L.ks 5 + L.ks 6)
    (H3 : L.xnumel = n * L.ks 0) (H4 : L.xnumel ≤ 2 ^ 31 - 1) (hn : 1 ≤ n) :
    0 < L.xnumel ∧ 0 < L.ks 0 ∧ L.ks 0 ≤ L.xnumel ∧
    ∀ i, i ≤ 6 → 0 ≤ L.ks i ∧ L.ks i < 2 ^ 31 ∧ (evI .i32 L (.ks i)).1 = L.ks i := by
  have k := fun i (h1 : 1 ≤ i) (h6 : i ≤ 6) => H1' i h1 h6
  have k1 := k 1 (by omega) (by omega); have k2 := k 2 (by omega) (by omega)
  have k3 := k 3 (by omega) (by omega); have k4 := k 4 (by omega) (by omega)
  have k5 := k 5 (by omega) (by omega); have k6 := k 6 (by omega) (by omega)
  have hs : 0 < L.ks 0 := by rw [H2]; omega
  have hn' : (1 : Int) ≤ n := by exact_mod_cast hn
  have hle : L.ks 0 ≤ L.xnumel := by
    rw [H3]; have := Int.mul_le_mul_of_nonneg_right hn' hs.le; simpa using this
  have H4' : L.xnumel ≤ 2147483647 := by norm_num at H4; exact H4
  refine ⟨by omega, hs, hle, fun i hi => ?_⟩
  have hb : 0 ≤ L.ks i ∧ L.ks i < 2 ^ 31 := by
    interval_cases i <;> constructor <;> norm_num <;> omega
  refine ⟨hb.1, hb.2, ?_⟩
  simp only [evI]
  exact wrap_R32 ⟨by omega, by have := hb.2; norm_num at this; omega⟩ _

end VeriTile.Bench.Optimizations.NarrowCat
