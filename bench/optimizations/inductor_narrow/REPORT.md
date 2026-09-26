# Proven int32 size arguments for Inductor's dynamic-concat kernel (pytorch/pytorch#189940): full report

*Companion report to a draft PyTorch PR. Status of each claim is labelled **measured**, **tested**, or **proved**.*

## 0. What was measured where

Two separate bodies of evidence support this contribution. They used different software stacks and answer
different questions. Do not combine them.

| | Stack | Question it answers | Result |
|---|---|---|---|
| **Performance** (measured) | torch **2.14.0**, Triton 3.8.0, one NVIDIA L4 | Is the N1 kernel (the emitted kernel with `ks*: i32`) faster? | Yes on this device and workload: 1.35× over the emitted kernel and 1.09× over our earlier guarded variant, paired and pre-registered (§3). The kernel was produced by a local prototype of the rule, not by the upstream patch. |
| **Port** (tested) | torch **2.15.0.dev20260926** (pinned `main` 6aa9e2fc), cu130, one L4 | Does the upstream patch apply, fire only where intended, and keep output exact? | Yes: 4/4 new tests pass; the 14-shape sequence is bitwise equal to eager with all seven `ks` as `i32` (§5). **No timing was taken on this stack.** |

The pinned revision emits the same Triton source for this program, byte for byte, as the measured stack did. That is
encouraging, but it does not establish identical backend output, autotuning selection, or runtime overhead. An
off-versus-on timing of the patch itself at 6aa9e2fc is designed but not yet run (§8).

## 1. Program

```python
def nested_cat_add(q1, k1, v1a, v1b, q2, k2, v2a, v2b):
    v1 = v1a + v1b; v2 = v2a + v2b
    return torch.cat([torch.cat([q1, k1, v1], -1), torch.cat([q2, k2, v2], -1)], -1)
# inputs [n, w] bf16, torch._dynamo.mark_dynamic(t, -1); torch.compile(nested_cat_add, dynamic=True)
```

Inductor emits one pointwise kernel indexing with `xindex % ks0` and `xindex // ks0`. `signature_to_meta` declares
every `ks*` argument `int64`, because a product or sum of symbols can overflow even when each fits. That makes every
index operation in this kernel 64-bit.

**About the issue's headline claim.** On L4 we did *not* reproduce "compiled dynamic slower than eager" (0 of 7
completed shapes; dynamic was faster than eager at each). We did reproduce the mechanism: compiled dynamic was
1.52–2.38× slower than compiled static across 14 shapes. The issue's H100 was not available to us.

## 2. The change

**N1:** declare `ks0..ks6` as `i32` in the kernel signature when a rule establishes that this is exact (§4). The
kernel body is unchanged, and there is no in-kernel fallback. When the rule's premises fail, the kernel keeps `i64`.

## 3. Performance (measured; torch 2.14 / Triton 3.8 / one L4)

Setup: Modal L4 (72 W power limit), bf16, warm L2. Every configuration was bitwise equal to eager at every shape,
with no recompiles in warm passes. Selection criteria were fixed in the harness before each run.

### 3a. First experiment (two rounds)

| warm complete-sequence time, ms (round 1 / round 2), 14 shapes × 20 calls | autotuned | one fixed config (XBLOCK 1024, 4 warps) |
|---|---|---|
| emitted kernel (`ks*: i64`) | 576 / 571 | 836 / 833 |
| guarded N (size scalars cast to int32 at kernel entry; original body kept as fallback) | 441 / 444 | 466 / 458 |
| **N1** (`ks*: i32` in the signature; no fallback) | **404 / 399** | **408 / 395** |
| `dynamic=False` (specializes per shape) | 361 / 357 | 362 / 359 |

- N1 is about 1.43× faster than the emitted kernel and 1.09–1.11× faster than guarded N (1.14–1.16× at a matched
  config).
- Static compilation stays about 12% faster at warm steady state and costs about 20 s more cold compilation over
  this sequence (28.95–29.55 s against 8.66–9.05 s), paying back after roughly 131,000–137,000 calls on this
  sequence. That is a workload-specific trade-off, not a verdict.
- Beyond the int32 guard (numel = 2^31), Dynamo recompiled to an `xnumel: i64` kernel, the rule declined, and the
  output was bitwise equal to eager.

**Selection outcome of this experiment: N, not N1.** Its pre-registered rule required N1 ≥ 0.97× of guarded N at
*every* shape in *both* rounds. N1 was 0.951× at one small shape (3000 × (1000, 120, 136)) in round 1 and 1.09× in
round 2. That verdict stands in the record.

**Autotuned kernels.** The emitted kernel ran at XBLOCK 512 / 8 warps (40 registers); N1 at 1024 / 4 warps
(48 registers). Recompiled SASS is about 260 instructions per element for the emitted kernel, with 64-bit division
slow-path calls, and about 135 for N1, with none. Configuration and work per thread differ, so we do not attribute
the gain to a single factor; the matched-config timing supports the benefit without isolating its cause.

### 3b. Separate confirmation (pre-registered; answers the small-shape question)

This experiment was designed **after** the first experiment's miss, to test whether that small-shape regression is
reproducible. It is not an independent replication of the first protocol.

- **Design.** B0 (emitted), N and N1 compiled in one process as distinct functions. Each block: at least 2 s of
  sustained warm-up, one pass over the 14-shape sequence, then 50 back-to-back calls at each of 10 small shapes.
  24 triples in 2 processes, all 6 variant orders equally often; fixed stopping point; no interim looks.
- **Statistics.** Per-triple paired ratios; median; percentile-bootstrap 95% CI over triples (10,000 resamples).
- **Rules.** Overall benefit confirmed iff median N/N1 ≥ 1.03 and CI lower bound ≥ 1.00. Each small shape:
  "non-regression shown" iff CI lower bound ≥ 0.97. Adopt N1 iff both hold everywhere.
- **Validity:** all checks passed (bitwise everywhere, rule eligible, no compiles during measurement).
- **Result: ADOPT N1.**

| | median | 95% CI |
|---|---|---|
| sequence time N / N1 | 1.090 | 1.082 – 1.094 |
| sequence time emitted / N1 | 1.349 | 1.344 – 1.358 |

Median sequence time: emitted 600.7 ms, N 482.8 ms, N1 443.9 ms.

| shape (n × widths) | emitted µs | N µs | N1 µs | paired N/N1 (95% CI) |
|---|---|---|---|---|
| 1536 × (2048, 256, 256) | 220.9 | 181.1 | 171.5 | 1.059 (1.023–1.095) |
| 2048 × (2048, 256, 256) | 277.0 | 226.9 | 202.5 | 1.101 (1.086–1.121) |
| 2500 × (1000, 120, 136) | 199.1 | 161.7 | 155.3 | 1.041 (1.025–1.053) |
| 3000 × (1000, 120, 136) | 194.2 | 160.8 | 155.0 | 1.102 (1.077–1.145) |
| 3500 × (1000, 120, 136) | 264.2 | 215.6 | 198.5 | 1.056 (1.030–1.109) |
| 2048 × (3072, 512, 512) | 415.8 | 357.7 | 301.9 | 1.175 (1.128–1.197) |
| 4096 × (2048, 256, 256) | 579.9 | 465.9 | 386.9 | 1.183 (1.174–1.207) |
| 2048 × (4096, 1024, 1024) | 750.9 | 601.1 | 506.3 | 1.174 (1.147–1.219) |
| 12345 × (1000, 120, 136) | 957.3 | 795.5 | 709.2 | 1.119 (1.096–1.141) |
| 4096 × (3072, 512, 512) | 1042.6 | 828.5 | 793.8 | 1.046 (1.036–1.052) |

**Reading the table.** The µs columns are each variant's median time over triples. The last column is the median of
the per-triple paired ratio, which is the pre-registered statistic. These are different statistics, so dividing the
displayed medians need not reproduce the paired median: at 3000 × (1000, 120, 136), 160.8 / 155.0 ≈ 1.04, while the
paired median is 1.10.

**Sensitivity (non-gating; same data).** Sliced by process, and by whether N ran before or after N1, the sequence
ratio N/N1 stays between 1.086 and 1.093 (per-permutation medians 1.084–1.096). Individual blocks at the smallest
shapes vary widely: single-triple ratios range from 0.82 to 1.26 at 2500 × (1000, 120, 136), so a single paired
measurement at ~150–200 µs is not informative on its own. Full tables: `SENSITIVITY.md`.

**Scope of the CIs.** They describe variability within this one experiment (one L4, one day, one stack), not across
machines, GPU families, days, or compiler revisions.

### 3c. Operating conditions on L4

- During sustained passes, NVML reports the software power cap active in 100% of samples for every configuration.
- In the first experiment the SM clock fell to 52–57% of idle for the dynamic kernels, and to 79% for static.
  Kernels ran 1.39× (emitted), 1.26× (N), 1.12× (N1) and 1.01× (static) longer inside the sequence than in
  isolation; inter-kernel gaps were at most 1.6% of the pass.
- In the confirmation, N1's median SM clock during measured blocks was lower than N's (945 against 1166 MHz), yet
  N1 was faster. This is consistent with N1 doing less work; we do not interpret it further.
- The confirmation used a different sustained-load protocol from the first experiment, so its absolute times are
  reported separately (e.g. emitted 601 ms against 571–576 ms). The difference is consistent with changed operating
  conditions; its causes were not separately isolated.

## 4. The eligibility rule (patch: `config.triton.narrow_proven_size_args`, default off)

Declare `ks0..ks6` as `i32` for a kernel `K` only when all of the following hold; otherwise `K` is unchanged.

**C1 — expression class.** `K` is a pointwise kernel on a 1-D grid with no fixed config; it has the proved argument
list and signature; and its body (indexing, masks, payload dataflow, dead statements) equals the proved kernel's, by
SHA-256 of a canonical IR. The recognizer rejects any construct it does not model.

**C2 — symbolic facts** from Inductor's state, never from size hints:
- H1′: `ks1..ks6` are size symbols with a value-range lower bound ≥ 1.
- H2: `ks0`'s precomputed definition equals `ks1 + … + ks6`.
- H3′: `numel = s · ks0` for a size symbol `s` with lower bound ≥ 1.
- H4: the index dtype is int32, and `numel ≤ 2^31 − 1` is guarded (Inductor installs this guard and Dynamo re-checks
  it on every call) or statically known.

H1′ and H3′ make every launch that reaches `K` nonempty; empty inputs fail them and are compiled separately.

**C3 — launch.** 1-D grid of `⌈xnumel / XBLOCK⌉` programs with `XBLOCK = 2^k ≤ 2^31`. This is compiler behaviour; the
rule checks the grid type.

**C4 — premise maintenance.** When H4's guard fails on a later call, Dynamo recompiles; Inductor then picks int64
indexing and the new kernel is not in the class. Exercised at numel = 2^31 (§3a).

**Why "each symbol fits in int32" is not enough.** Our boundary tests include inputs with every `ks_i < 2^31` and H2
false; there `ks1 + ks2 + ks3` overflows and the two kernels disagree. That is the hazard the `signature_to_meta`
comment warns about. The rule answers it with relations Inductor already knows. Current `main`'s
`assume_32bit_indexing` narrows `ks*` globally on a user's promise; on 2.14 it does not narrow `ks*` at all.

## 5. Port to pinned `main` (tested; correctness only)

- **CPU codegen, pinned nightly:** with the flag on, the proved kernel's `ks*` become `i32` and its body is
  byte-identical; with the flag off, and for a 3-segment kernel, nothing changes; fault-injected symbol ranges with
  lower bound 0 are refused; body mutations (extra size use, changed index, changed cast, changed dead code, unknown
  op) are rejected.
- **CUDA, one L4, torch 2.15.0.dev20260926+cu130 at 6aa9e2fc:** base files match the pin and the patch applies;
  4/4 tests in `test/inductor/test_triton_size_arg_narrowing.py` pass; with the flag on the 14-shape sequence is
  bitwise equal to eager at every shape, with one graph, one kernel, and all seven `ks` as `i32`.
- The patch also applies (dry run) to a later `main`, 6f8b3cdb.
- **Not established:** timing on this stack; PyTorch-wide CI. Four targeted tests are not a substitute for the
  project's integration testing.

## 6. Proof (proved; Lean 4, standard axioms only)

`NarrowCat.lean` models the kernel's actual integer semantics: int32/int64 promotion, weak Python literals, strong
`tl.full` constants, two's-complement wrap at every operation, truncating `%` and `//`, and `i32` argument packing.

- `narrow_cat_equiv`: under C1–C3, for every lane of the grid, the `i64` and `i32` kernels agree on the store mask,
  the store address and value, and every memory read. The payload is symbolic, so no floating-point claim is made.
- `narrow_cat_args_fit`: with positive widths and a positive row count, every `ks_i` is in `[0, 2^31)` and `i32`
  packing is exact.
- `narrow_cat_divisor`: equal, positive divisors, so no new undefined behaviour.
- `grid_xindex_lt`: no `xindex` wraparound on the launch grid.

Structure: `evI_fits` / `evF_fits` relate integer evaluation under range conditions to agreement of payload and
reads; the kernel-specific `cat_fits` establishes those conditions for this expression graph. The IR the theorem is
stated over is extracted from the emitted source, and the patch's recognizer pins the same IR; both links are
checked by tests.

**Trusted, not checked:** SymPy and Inductor's `sizevars` / `shape_env` facts; the recognizer's parse of the generated
source (binding-tested, not verified) and the extractor; Triton's integer semantics as modelled; masked-off lanes'
addresses (which may wrap in int32) are never dereferenced, as the emitted kernel already assumes for past-the-end
addresses; the Triton → PTX → SASS toolchain.

## 7. Limitations and history

- **One kernel class, by digest.** The digest is a compact identity check, not an explanation of why a class of
  programs is eligible. A three-segment concat is rejected; each new class would currently need its own extraction
  and proof. The final upstream form may need to be a structural or range-based rule at an earlier representation
  rather than a whitelist of generated kernels. We are asking for guidance before building that.
- **Maintenance.** Recognition keys on generated-source structure, including argument naming and temporary
  numbering. A benign codegen change makes the rule silently stop firing (the kernel keeps `i64`); it cannot make it
  misfire, provided the recognizer and extractor are correct (they are trusted, §6).
- **The flag is a staging mechanism.** Default off during evaluation. The intended end state is an automatic
  compiler decision when the facts hold, not a user-facing knob.
- **One device, one program, warm L2.** The issue's H100 is untested.
- **Not captured:** L4 cubins (only Triton cache hashes); the recompile-reason text for the beyond-guard shape.
- **Negative result — fast division.** Replacing `%` and `//` with magic-number division (constants computed in the
  kernel) was slower on L4 (0.94×). As built, its guard kept both bodies in the kernel; at XBLOCK 1024 / 4 warps that
  raised registers to 128 (40 at 512 / 8 warps; its autotuned config was not recorded). A branch-only SASS comparison
  shows the fast-division branch doing more work than the narrowing branch; that excludes the guard layout and does
  not settle the end-to-end comparison.
- **Selection history.** Guarded N was selected by the first experiment; N1 was adopted after the separate
  confirmation (§3b). Guarded N is now a historical comparator.

## 8. Next measurement (designed, not run)

The patch itself, flag off versus on, on the pinned revision 6aa9e2fc and the same L4 type: paired, balanced,
pre-registered sequence and small-shape timing; compile-time cost of the eligibility check on the target kernel and
on unrelated kernels (rejection cost); compile count, graph count, guards and recompiles; autotuning selections; and
hashes of the timed cubins. A second device (ideally the issue's H100) would then address portability. Independent
reproduction by a reviewer would be especially valuable.

## 9. Questions for reviewers

1. Is generated-source recognition acceptable as an initial implementation, or should the rule operate earlier, on
   typed indexing expressions?
2. Does the symbolic-fact extraction (C2) establish exactly the theorem's premises, including their maintenance across
   later invocations (C4)?
3. What evidence would justify making this an automatic decision for eligible kernels rather than an opt-in flag?
