# Selective 32-bit size arguments: the rule and what it rests on

This is the public statement of the narrowing rule (N1). Each premise names **who establishes it**:
- **the checker:** the eligibility check run at codegen;
- **the compiler:** existing Inductor or Dynamo behaviour;
- **the proof:** `NarrowCat.lean`;
- **trusted:** stated, not checked.

## The rule

For an Inductor pointwise kernel `K`, declare its seven size arguments `ks0 … ks6` as `i32` instead of `i64`
when all of the following hold. Otherwise emit `K` unchanged.

### C1. Expression class (checker, structural)
- **The body.** `K`'s body is the proved kernel: every observable operation and every dead statement is equal, as
  extracted IR, to the kernel `NarrowCat.catValue` is generated from. The observable operations are the store
  address, the store mask, the stored value and every load's address and mask. The IR compared includes all types,
  casts, promotions and evaluation order.
- **The signature.** `K`'s argument list and signature are the proved ones: eight pointers in, one out, `ks0..ks6`,
  `xnumel: i32`, and `XBLOCK: constexpr`.
- **The kind of kernel.** `K` is a plain pointwise kernel. It is not a reduction, template, cooperative or
  fixed-config kernel.

Any difference means `K` is not in the class. The rule is deliberately **one kernel shape**: the dynamic
6-segment concatenation of pytorch/pytorch#189940. Extending it needs a new extraction and a new proof run.

### C2. Symbolic facts (checker, reading Inductor's own state; never size hints or example values)
- **H1'** Each of `ks1 … ks6` is a size symbol whose `shape_env.var_to_range` lower bound is ≥ 1.
- **H2** `ks0`'s definition in `sizevars.inv_precomputed_replacements` expands to `ks1 + … + ks6`.
- **H3'** The kernel's `numel` expands to `s · ks0` for a size symbol `s`, whose range lower bound is ≥ 1.
- **H4** The kernel's index dtype is `int32`, and `numel ≤ 2147483647` either is among the installed shape
  guards or is statically known true.

H1' and H3' are the **nonempty-launch premises**. Dynamo specializes sizes 0 and 1, so the symbols' ranges start
at 2 in practice. The checker verifies the bounds instead of assuming them. Under H1'–H4 every call that reaches
`K` has `xnumel ≥ 1`, which gives the following (Lean: `narrow_cat_args_fit`):
- `0 < ks0 ≤ xnumel ≤ 2^31 − 1`;
- every `ks_i` is in `[0, 2^31)`, so packing each size argument as `i32` is exact.

**Empty inputs.** They never reach `K` under these facts. A zero-row or zero-width input fails H1' or H3', which
Dynamo enforces through specialization and guards, and is compiled separately, without this rule.

### C3. Launch (compiler)
- The grid is 1-D: `grid_type == "Grid1D"`, one program per `XBLOCK` chunk, `pid < ⌈xnumel / XBLOCK⌉`.
- `XBLOCK = 2^k` with `k ≤ 31`. Triton requires power-of-two `tl.arange` extents, and Inductor's configs use
  them.

With `xnumel ≤ 2^31 − 1`, these give `0 ≤ xindex < 2^31` on every lane (Lean: `grid_xindex_lt`). The checker
checks `grid_type`; the rest is compiler behaviour.

### C4. Guard maintenance (compiler)
The H4 guard is re-evaluated by Dynamo on every call before the graph runs. If it fails (`numel ≥ 2^31`), Dynamo
recompiles, and Inductor chooses `int64` indexing (`xnumel: i64`). That kernel is not in the C1 class, so the
original wide kernel is used. The stage-13 L4 run exercised this at `numel = 2^31`.

## What the proof gives, given C1–C4

`NarrowCat.narrow_cat_equiv`: for every lane with `0 ≤ xindex < 2^31`, the `i64` and `i32` kernels agree on:
- the store mask;
- when it is set, the store address and the stored value;
- the full list of memory reads.

Companion lemmas:
- `narrow_cat_divisor`: the divisor for `%` and `//` is equal and positive, so there is no new undefined
  behaviour. It needs `xnumel > 0`, which C2 provides.
- `guarded_N_else_unreachable`: the in-kernel fallback of the earlier guarded variant never runs.
- `narrow_cat_args_fit`: the argument-packing corollary above.

## Trusted, not checked

- **Symbolic state.** SymPy's `expand` and Inductor's `sizevars` / `shape_env` report the facts correctly.
- **Extractor.** The source-to-IR extractor (`scripts/inductor_ir.py`) reads the emitted source correctly. It is
  binding-tested against the fixture and rejects unknown constructs, but it is not itself verified.
- **Triton semantics.** Triton's integer semantics are as modelled:
  - `int32` and `int64` with promotion;
  - two's-complement wraparound;
  - truncating `%` and `//`;
  - `i32` arguments packed from Python ints.
- **Masked-off lanes.** Addresses on masked-off lanes are never dereferenced, even when N1's `int32` arithmetic
  wraps them. The emitted kernel already relies on this for past-the-end addresses.
- **Toolchain.** The Triton → PTX → SASS toolchain.
- **Payload.** The payload operations are identical in both kernels, and the proof treats them symbolically. No
  floating-point claim is made.

## What the rule is not

- It is **not** `assume_32bit_indexing`. That flag is a global user promise that every `ks*` symbol fits. As
  `signature_to_meta` warns, a product of fitting symbols can still overflow, and the boundary tests show the two
  kernels disagreeing when H2 fails even though every `ks_i < 2^31`.
- On torch 2.14, `assume_32bit_indexing` does not narrow `ks*` at all. On `main` it does, globally.
