# Stage 12: the standalone gain inside Inductor's own kernel

**Question.** Stage 11's standalone fast-division kernel reached compiled-static speed with dynamic shapes. Does
that gain transfer into the kernel `torch.compile(dynamic=True)` itself emits for the #189940 program?

**Method.** We rewrite Inductor's generated kernel at codegen time and change nothing else: payload arithmetic,
masks, loads and stores, launch signature, grid and wrapper all stay as emitted. The two mechanisms are isolated
in a 2×2 design.

| variant | integer width | divmod | guard (scalar, in-kernel) |
|---|---|---|---|
| B0 | as emitted (seven `i64` size scalars; int64 index math) | `xindex % ks0`, `xindex // ks0` | none |
| N | size scalars cast to int32 | unchanged | `0 ≤ ksK < 2^31` for all K |
| F | as emitted | proved fast divmod (`_vt_fast_divmod`) | `0 < ks0 < 2^31` |
| NF | int32 | proved fast divmod | both |

Under the guard's `else` branch, N, F and NF keep Inductor's original body verbatim. The rewrite is
`scripts/inductor_divmod.py`. It runs inside `TritonScheduling.define_kernel`, so Inductor hashes, compiles and
autotunes the rewritten source. The kernel the L4 emitted (`fixtures/gpu_emitted_B0.py`) is byte-identical in
body to the CPU preview used for development.

## Outcome (pre-registered criterion, one L4 call, `RESULTS.md`)

**Decision: transferred; selected N (int32 narrowing). The proved fast divmod (F) did not transfer.**

| | N | F | NF |
|---|---|---|---|
| complete-sequence speed-up vs B0 (round 1 / round 2) | **1.28× / 1.21×** | 0.94× / 0.95× | 1.10× / 1.11× |
| worst shape vs B0 | 1.25× / 1.16× | 0.90× / 0.89× | 1.02× / 1.03× |
| device time vs B0 (profiler, 14 shapes) | 1.11–1.25× | 0.83–0.96× | — |
| passes the pre-registered criterion | yes (both rounds) | no | yes (both rounds) |

N is also 1.21–1.29× faster than automatic dynamic compilation, and 1.49–1.59× faster than eager.

**How much of the standalone gap N captures:**
- **Device time:** all of it at the larger shapes. At 32768 × issue: B0 3478 µs, N 2952 µs, standalone 2957 µs,
  static 3022 µs.
- **Complete-sequence time:** about half. T is 587 → 471 ms for B0 → N (mean of the two rounds), against 350 ms
  for the standalone kernel. The other half is the sustained-versus-device gap listed under Open, which affects
  only Inductor's dynamic kernels.

Every configuration was valid in both rounds:
- bitwise equal to eager at all 14 shapes;
- no Dynamo or Inductor compile in any warm pass;
- the rewrite fired on the one divmod kernel;
- one Dynamo graph and one kernel, the same as B0.

**Against cached static compilation,** static is still faster at sustained load: 1.29× / 1.37× over N. Static
compilation is a different cost profile, not a mode limited to a known shape set: it specializes and caches a
new kernel as each new shape arrives. On this 14-shape sequence its cold pass costs 25.2 s against about 9–10 s
for N (14 compiles against 1). The extra 15–16 s pays back after roughly 33,000–44,000 calls. That break-even
is specific to this workload: a small, repeated shape set favours static, and a stream that keeps presenting
new shapes favours the reusable dynamic kernel.

*Correction (stage 13):* an earlier version of this paragraph called static compilation "valid only when the
set of shapes is known in advance". That was wrong.

## Why the proved fast division did not transfer, and what did

SASS (machine code) for sm_89, XBLOCK=1024, 4 warps, 8 elements per thread (`sass_counts.json`; static counts).
"Branch" means the guarded branch compiled alone, with the guard forced true. "Full kernel" is the kernel as it
ran on the L4: both branches, since the `else` keeps the original body.

| | branch instructions | IMAD | I2F/F2I/MUFU | CALL | registers, branch | **registers, full kernel** |
|---|---|---|---|---|---|---|
| B0 | 1864 | 614 | 30 | 16 | 90 | 90 |
| N | 1192 | 393 | 3 | 0 | 56 | **90** |
| F | 1696 | 508 | 3 | 1 | 92 | **128** |
| NF | 1336 | 428 | 3 | 1 | 64 | **128** |

1. **B0 pays for width, not for a missing division trick.**
   - The divisor arrives as `i64`, so every `%` and `//` is a 64-bit operation. The compiler's 64-to-32-bit bypass
     then rebuilds the float-reciprocal division sequence for each element: 30 I2F/F2I/MUFU instructions, and 16
     calls to the 64-bit slow path.
   - All address arithmetic is int64: 614 IMADs and 90 registers.
2. **Narrowing alone (N) removes most of that.**
   - With an int32 divisor, the compiler computes the divisor-dependent part once and reuses it across the
     thread's elements: 3 MUFU/I2F in total and no calls.
   - The address arithmetic becomes 32-bit: 393 IMADs.
   - In effect, the compiler already performs a fast division by a loop-invariant divisor once the types allow it.
   - The branch's lower register count (56) does not carry over to the kernel. The guarded kernel still allocates
     90 registers, the same as B0, because of the retained original body. So N's gain is instruction count, not
     occupancy.
3. **The proved IntDivider replacement (F) adds work that is not amortized.**
   - Computing the constants in-kernel costs, per thread, a 32-step shift count plus one true 64-bit division.
     That division is the remaining `CALL`: the numerator `2^32·(2^s − d)` exceeds 32 bits, so the bypass
     cannot apply.
   - With only 8 elements per thread, this costs more than the per-element divisions it replaces.
   - F without N also leaves every other int64 operation in place. Compared branch to branch, NF is still larger
     than N: 144 more instructions and 8 more registers.
   - **Confound from the guard design.** As built, the F and NF kernels allocate 128 registers against 90 for B0
     and N. The helper's branch together with the retained original body raises the kernel's register high-water
     mark, which lowers occupancy. Part of F's 0.94× against B0 may therefore come from this guard layout rather
     than from the divmod arithmetic.
   - **Untested alternative.** A leaner guard around only the two divmod lines, instead of the whole body, might
     avoid the register increase. That was not measured. The branch-to-branch comparison (NF against N) does not
     depend on the guard layout, and it still favours N.
4. **Where the standalone kernel's advantage came from.** It precomputes the constants on the host and runs all
   index math in int32. Inside Inductor, the width part reaches the same device time: at the 12 larger shapes, N's
   device time is 0.94–1.06× that of static and 0.97–1.06× that of the standalone kernel. The division part is either already done by the
   compiler (N) or too costly to compute in the kernel (F). Passing host-computed constants would change
   Inductor's launch signature and wrapper, which this stage deliberately did not do.

## Proof conditions of the selected version (N), and what is proved

**N is not the Lean-proved transformation.** Its correctness rests on the following argument, and on tests.
- **Guard (checked in the kernel on every launch):** every size scalar is in `[0, 2^31)`. Otherwise the original
  int64 body runs.
- **Inductor's own int32-indexing guards,** installed when it chose `xnumel: i32` (`SIMDScheduling.
  can_use_32bit_indexing`): the output numel and every buffer's storage size are at most `2^31 − 1`.
- **From these two,** on every active lane with a true load mask, every intermediate of the index arithmetic is
  bounded by a buffer offset or a size below `2^31`, so int32 and int64 compute the same values. Masked-off lanes
  may wrap, but their loads and stores are masked.
- **Tested:**
  - bitwise equality with eager at the 14 timed shapes in both rounds;
  - interpreter equivalence with B0 on 6 shapes × 2 XBLOCK values (`local_checks.json`);
  - 12 rewrite unit tests.
- **Not proved in Lean.** The model would have to include Inductor's generated index expressions.

**What is proved** (`InductorDivmod.lean`, standard axioms): the F/NF helper computes exactly `FastDiv`'s
constants and returns `(x / d, x % d)` with every intermediate in range, for `0 < d < 2^31` and
`0 ≤ x < 2^31`. That result holds, and the equivalence tests pass, but on this hardware the helper does not
improve the kernel.

## Open

**Sustained time exceeds device time for Inductor's dynamic kernels.** In-sequence time per call is 1.24–1.83×
the profiler device time for B0, N, F and NF (1.24–1.54× at the 12 larger shapes). For eager, static and the
stage-11 kernel the two agree within 1.00–1.07× at those shapes; the two small shapes are host-bound. Stage 11
showed the same pattern.
- **Hypothesis, not tested:** the L4 is capped at 72 W. Its clocks drop under sustained load for the more
  ALU-heavy dynamic kernels, while profiled calls are separated by idle gaps.
- **Discriminating measurement:**
  - record SM clocks with NVML during a warm pass;
  - profile a whole warm pass instead of isolated calls.
- Either measurement needs another GPU call. The narrowing gain is measured at both levels, so the decision does
  not depend on this.

**Not measured:**
- other GPUs, including the H100 from the issue;
- cold-L2 timing;
- other programs with symbolic non-leading `cat` dimensions;
- the Inductor autotuner's chosen config (not recorded; the SASS table assumes XBLOCK=1024, 4 warps);
- **the guard's `else` branch was compiled but never executed.** Every tested shape had all size scalars below
  2^31, both in the interpreter and on the GPU. That is the intended domain, but the fallback path itself has only
  been compiled, not run on a GPU;
- the leaner guard layout for F/NF described above.
