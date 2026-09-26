# Stage 13: proving the narrowing that won (N), and N1 without an in-kernel fallback

**Stage 12's result.** The optimization that helped Inductor's dynamic `cat` kernel was narrowing its seven
`int64` size scalars to `int32` (variant N), not a different division algorithm. This stage:
1. proves that narrowing safe from facts Inductor already holds;
2. builds **N1**, which establishes eligibility at compile time and emits a narrow-only kernel with no fallback
   body;
3. measures N1 against guarded N, with the stage-12 sustained-time discrepancy diagnosed in the same campaign.

## 1. What is proved (`NarrowCat.lean`, standard axioms; statements frozen before proof)

- **The model.** The kernel's integer indexing, masks and payload dataflow are **extracted from the emitted
  source** (`scripts/inductor_ir.py`) into a typed expression language. Its semantics is Triton's:
  - `int32` or `int64` with promotion;
  - wraparound at every operation;
  - truncating `%` and `//`;
  - the launcher's conversion of `i32` arguments.
  The generated term `catValue` is binding-tested against the source. The payload is symbolic, so no
  floating-point claim is made.
- **`narrow_cat_equiv` (headline).** For every lane of the grid, B0 (`ks*: i64`, as emitted) and N1 (`ks*: i32`)
  agree on:
  - the store mask;
  - when it is set, the store address and the stored value;
  - the complete list of memory reads.
  It holds under H1–H4:
  - H1: `ks1..ks6 ≥ 0`;
  - H2: `ks0 = ks1 + … + ks6`;
  - H3: `xnumel = n · ks0`;
  - H4: `xnumel ≤ 2^31 − 1`.
- **`narrow_cat_divisor`.** Same positive divisor in both, so no new undefined behaviour.
- **`guarded_N_else_unreachable`.** In the graph's domain, the stage-12 in-kernel fallback can never run.
- **`grid_xindex_lt`.** The launch grid never wraps `xindex`.
- **Why the relations are needed.** The boundary sweep (`scripts/test_inductor_narrow.py`) shows that every
  `ks_i < 2^31` alone is **not** enough: without H2, `ks1+ks2+ks3` overflows and the variants disagree. H4 is
  B0's own precondition, because `xnumel` is already `i32` in B0.
- **Assumptions not modelled.**
  - Masked-off lanes' addresses are never dereferenced, even when N1's int32 arithmetic wraps them. B0 relies on
    the same property for past-the-end addresses.
  - The Triton → PTX → SASS toolchain is trusted.

**Eligibility is established before the kernel runs** (`scripts/inductor_narrow.py`). At codegen, N1 reads H1–H4
from Inductor's own state:
- the value ranges of the `ks` symbols;
- `ks0`'s precomputed definition;
- the numel expression;
- the installed guard `numel ≤ 2147483647`, which Dynamo re-checks on every call before the graph runs.

It also requires the kernel's extracted IR to equal the proved one. Only then does it change the seven signature
entries to `i32`; the body is unchanged. This is the kernel current PyTorch `main` emits under
`assume_32bit_indexing`, applied only where it is shown safe.

## 2. Measurement (one L4 call, 781 s; `RESULTS.md`; criterion fixed before the run)

**Pre-registered decision: "keep guarded N; N1 proved but not adopted."**

| | round 1 | round 2 |
|---|---|---|
| (a) T(B0)/T(N1) ≥ 1.10 | 1.42× ✓ | 1.43× ✓ |
| (b) T(N)/T(N1) ≥ 0.99 **and every shape ≥ 0.97×** | 1.09×, **worst shape 0.951× (3000 × odd) ✗** | 1.11×, worst 1.08× ✓ |
| (c) T(N)/T(N1) ≥ 1.03 | ✓ | ✓ |

- **What failed.** N1 is faster than guarded N over the whole sequence in both rounds. In round 1 it is slower at
  one shape, the smallest one (about 175 µs per call). In round 2 that same shape is 1.09× faster. The rule
  required (b) in both rounds, so N1 is not adopted.
- **No re-registration.** We do not change the rule after seeing the data. A confirmatory run would need a new
  criterion and a new allowance.

**Complete-sequence time T** (ms, median of 5 warm passes):

| | autotuned r1 / r2 | one fixed config (XBLOCK 1024, 4 warps) r1 / r2 |
|---|---|---|
| B0 (emitted) | 576 / 571 | 836 / 833 |
| guarded N | 441 / 444 | 466 / 458 |
| **N1** | **404 / 399** | **408 / 395** |
| static (per-shape compile) | 361 / 357 | 362 / 359 |

- **Across stages.** N1 is 1.42–1.43× faster than B0. That is a larger gain than stage 12's guarded N (1.21–1.28×,
  and 1.29–1.30× in this run).
- **At a matched config,** N1 against N is 1.14× / 1.16×.
- **Remaining gap to static.** Static stays 1.12× faster than N1 at sustained load. The gap was 1.29–1.37× for
  guarded N in stage 12.

**Timed artifacts** (recorded on the L4; SASS recompiled locally at the same config, with matching register
counts):

| | autotuned config | registers | SASS per thread | elements per thread | SASS per element | 64-bit division slow-path calls |
|---|---|---|---|---|---|---|
| B0 | XBLOCK 512, 8 warps | 40 | 520 | 2 | 260 | 2 |
| guarded N | XBLOCK 512, 8 warps | 40 | 848, of which about 520 is the fallback body (proved never to run) | 2 | about 164 executed (estimate: (848 − 520) / 2) | 2 (in the fallback) |
| N1 | XBLOCK 1024, 4 warps | 48 | 1080 | 8 | 135 | 0 |

- **Correction to stage 12.** Stage 12's SASS table assumed 1024/4. Autotuning actually chose 512/8 for B0 and N.
  The stage-12 README now carries a correction block.
- **Recompilation is not guaranteed byte-identical.** The three autotuned kernels' register counts match the L4
  record exactly. Fixed-mode guarded N recorded 92 registers on the L4 against 90 locally.
- **Not captured: cubins.** The L4 cubins were not captured, because compile results are empty in the parent
  process after asynchronous compilation. The Triton cache hashes were recorded. A future run should read
  `$TRITON_CACHE_DIR/<launcher.cache_hash>/*.cubin` before its temporary cache directory is removed.

**Beyond the guard** (numel = 2^31, about 17 GB peak). In B0, N and N1 alike:
- Dynamo recompiled, and Inductor emitted an `xnumel: i64` kernel. The evidence is that new kernel definition: the
  guard-failure (recompile-reason) text was **not captured**, because the log handler recorded no recompile
  message.
- No rewrite fired. N1 declined at the structural check, because the int64 kernel's prologue differs.
- The output was bitwise equal to eager.

This exercises the retained original for real.

## 3. The sustained-time discrepancy: resolved as power capping

Telemetry comes from NVML samples every 10 ms (round 1). The idle state is the probe with a 50 ms host sleep
before each call.

| | median SM clock, sustained / idle | software power cap active (sustained) | power | in-sequence / isolated kernel time | gaps in the profiled pass | pre-registered verdict |
|---|---|---|---|---|---|---|
| B0 | 1170 / 2040 MHz (0.57) | 100% of samples | ~71 W of 72 W | 1.39 | 0.3% | **D1: device state** |
| guarded N | 1155 / 2040 (0.57) | 100% | ~71 W | 1.26 | 0.9% | **D1** |
| N1 | 1065 / 2040 (0.52) | 100% | ~70 W | 1.12 | 1.2% | unassigned (below the 1.15 threshold) |
| static | 1620 / 2040 (0.79) | 100% | ~71 W | 1.01 | 1.6% | unassigned |

- **The mechanism.** Under sustained load every configuration hits the L4's 72 W software power cap. The SM clock
  falls to about 52–57% of idle for the dynamic kernels and to 79% for static.
  - The index arithmetic in B0 and N is clock-sensitive, so their kernels run 1.26–1.39× longer inside the
    sequence than in isolation.
  - N1 has much less integer work and is least affected (1.12×).
  - Static's kernel is memory-bound (1.01×).
- **Not dispatch.** Gaps between kernels are at most 1.6% of the pass.
- **Consequence.** Isolated-call device time overstates what the dynamic kernels achieve at sustained load on this
  platform. Narrowing helps partly because it makes the kernel less clock-sensitive. The sustained numbers above
  are the deployment numbers; they include the power cap and are not corrected for it.

## 4. Where this leaves the optimization

- **Guarded N stays the recommendation under the pre-registered rule.** It is 1.29–1.31× faster than B0 in this
  run.
  - `guarded_N_else_unreachable` proves its fallback can never run.
  - Its fast branch casts each `ks` to `int32` at entry, which gives the same values as N1's `i32` arguments. So
    `narrow_cat_equiv` covers that branch **by argument**. The branch has the same IR, but it was not extracted
    and bound separately.
- **N1 has the stronger measured case and the simpler implementation.**
  - Measured: 1.42× over B0 and 1.09–1.11× over guarded N, failing only the per-shape floor at one small shape
    in one round.
  - Implementation: a signature change only, no fallback body, and eligibility proved at compile time.
- **Adopting N1 needs either** a confirmatory run under a new pre-registered criterion, or the owner's explicit
  decision to accept the one-shape miss.

Not measured: other GPUs (including the issue's H100), other programs, or cold L2.

## 5. Stage 14 update: confirmation and upstream port (`CONFIRM.md`, `RULE.md`, `upstream/`)

- **The stage-13 verdict stands in the record.** Its pre-registered rule did not adopt N1.
- **A separate, pre-registered confirmation recommends ADOPT N1.**
  - Design: 24 paired triples of B0, N and N1, independently warmed with at least 2 s of sustained load, balanced
    over all six orders, with a fixed stopping point.
  - N/N1 on the complete sequence: 1.090 (95% CI 1.082–1.094). B0/N1: 1.349.
  - All 10 small shapes show "non-regression shown" (every CI lower bound ≥ 1.023). At 3000 × odd, N1 takes
    155 µs against 161 µs for N.
  - **N1 is now the recommended implementation.**
- **The rule is stated in full in `RULE.md`,** with all four premise groups: C1 expression class, C2 symbolic facts,
  C3 launch, C4 guard maintenance.
  - The symbolic facts now require value-range lower bounds ≥ 1, which makes launches nonempty.
  - The new Lean corollary `narrow_cat_args_fit` proves that exact `i32` packing follows.
- **Ported to pinned PyTorch `main` 6aa9e2fc** (nightly 2.15.0.dev20260926).
  - That revision emits the proved kernel byte-identically.
  - The patch adds the opt-in `config.triton.narrow_proven_size_args`, a recognizer pinned by the canonical-IR
    SHA-256, and the symbolic checks.
  - On the pinned CUDA nightly it applies, passes 4/4 upstream tests on an L4, and is bitwise at 14 shapes.
  - Binding tests connect the patch's recognizer, the extractor, and the Lean `catValue`.

## 6. Stage 15: review package (no new measurement, no proof change)

- **N1 is adopted for its supported opt-in domain.** Guarded N is now a historical comparator. The proof target is
  frozen until reviewers have given direction.
- **Two bodies of evidence, kept separate.** The performance numbers (§2, §5) were measured on torch 2.14 / Triton 3.8
  with a local prototype of the rule. The upstream patch was correctness-tested on pinned `main` 6aa9e2fc; it has
  **not** been timed there.
- **`REPORT.md`** is the full report for reviewers. It keeps the negative results (fast division, the stage-13
  selection) and asks three review questions: the recognition layer, the symbolic-premise connection, and rollout.
- **`SENSITIVITY.md`** (non-gating; same data) slices the stage-14 confirmation by process and by variant order.
  The sequence ratio N/N1 stays 1.084–1.096 in every slice. Single blocks at the smallest shapes vary widely
  (0.82–1.26 at 2500 × odd), so only the paired medians and CIs are informative there. CONFIRM.md now explains why
  the paired-ratio median differs from the ratio of displayed medians.
- **Operating conditions.** The confirmation used a different sustained-load protocol from stage 13, so its absolute
  times are reported separately. The difference is consistent with changed operating conditions; its causes were not
  separately isolated.
- **Designed, not run: `scripts/launch_patch_perf.py`** (+ `_modal`). It times the patch itself, flag off versus on,
  at 6aa9e2fc on one L4. It is paired and pre-registered, and records compile cost (time in the eligibility check
  for the target and for four unrelated kernels, first-call time, graphs, guards), autotuned configs, and cubin
  hashes from a private Triton cache. A CPU codegen dry run in the local nightly image showed:
  - OFF gives `i64` and ON gives `i32` on the target;
  - the four unrelated kernels are rejected and keep their signatures;
  - normalized guard text is equal OFF and ON.
  Running it needs a new explicit allowance.
