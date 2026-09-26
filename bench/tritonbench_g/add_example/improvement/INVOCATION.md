# Checked execution of the fused add + ReLU — invocation-cost pass (stage 9)

One device (NVIDIA L4, CC 8.9), Triton 3.8.0, torch 2.14.0+cu130, two Modal calls
(apps in `work/logs/91_modal_latency.log`, `92_modal_probe.log`). Raw evidence:
`launch_evidence/latency_ladder.json` (run 1), `block_search.json` (run 1),
`validation.json` (run 2, final code), `workload_probe.json` (run 2),
`fast_check_differential.json` (host).

## Harness (all in the repository)

| file | role |
|---|---|
| `scripts/launch_fused.py` | general checked API (stage 8) and the unfused checked pipeline |
| `scripts/launch_fast.py` | `fast_ew2` (fast `Elementwise2.check`), `CheckedFusedAddReluFast`, block override, prepared-buffer API (`prepare_add_relu`, `PreparedAddRelu`), fast-path differential test |
| `scripts/launch_select.py` | the selected general API: block 256 for n < 2^24, 64 for n ≥ 2^24 |
| `scripts/launch_latency.py` | latency ladder, prepared-plan invalidation test, block search |
| `scripts/launch_probe.py` | validation of the final code, workload probe |
| `scripts/launch_latency_modal.py` | Modal driver (upload set = `launch_local_check.input_files()`) |

Timing method:
- Every timing is `time.perf_counter` on the GPU host; no profiler is active in any
  timing.
- **single** = one call + `torch.cuda.synchronize()`, median of 200 (50 for n ≥ 2^24).
- **repeated** = K calls, then one synchronize, mean per call.
- 5 independent trials (3 for n ≥ 2^24), candidate order rotated each trial; the
  median of trial medians is reported, and every trial is kept in the JSON.
- Device time is summed kernel durations (torch.profiler, warm cache). It is context
  only and never mixed with host timings.

## 1. Where the small-input time went (run 1, single mode, 2^16, µs)

| rung | cumulative | step |
|---|---|---|
| synchronize only | 9.2 | 9.2 |
| launch via precompiled kernel handle | 23.5 | 14.3 |
| launch via Triton JIT dispatch (`kernel[grid]`) | 34.6 | 11.1 |
| + output allocation (`empty_like`) | 46.3 | 11.7 |
| + metadata of three tensors (`tensor_meta`) | 69.6 | 23.3 |
| + contract decision (full mirror) | 112.7 | 43.1 |
| full checked call | 118.2 | 5.5 |

**Host caveat:** the Modal container reported a single vCPU of unidentified model
(`cpu` field in the JSON). Absolute host latencies are therefore specific to this host.
The decision cost 43 µs here against about 6 µs on an M4 Pro. What transfers is the
ranking and the ratios between candidates measured on the same host.

The contract check (metadata extraction and decision) was **56%** of the call. On this
host the decision alone costs 43 µs; on the Mac it was about 6 µs. Triton's JIT
dispatch costs 11 µs over a precompiled launch.

## 2. Changes

**General checked API** (`CheckedFusedAddReluSelected`):
- The decision is `fast_ew2`, a direct evaluation of the Boolean `Elementwise2.check`.
  It gives the same verdict and the same launch as the full mirror, in:
  - 20,000 random cases against the mirror, 0 mismatches;
  - 3,000 cases against Lean `#eval`, 0 mismatches;
  - real CPU and CUDA tensor layouts, all agreeing.
- On rejection it falls back to the full mirror, so the error still names the failed
  obligations.
- JIT dispatch is kept. A cached compiled kernel is specialised on pointer alignment
  and `n % 16`, and reusing it for other arguments would be silently wrong.
- Block rule: 256 below 2^24 elements, 64 at or above.
- Observable behaviour is unchanged: it returns a fresh output and rejects the same
  inputs.

**Prepared-buffer API** (`prepare_add_relu(x, y, out, block, graph)`), a separate API:
- The caller owns `out`, which is overwritten on every `run()`.
- `prepare` decides the full contract once and compiles the kernel for exactly these
  arguments.
- `run` revalidates a metadata snapshot of the three buffers (data pointer, shape,
  strides, storage offset, dtype, device, storage address and size), then launches the
  precompiled kernel or replays a CUDA graph of that launch.
- Why metadata alone is enough:
  - The contract's verdict is a function of metadata only.
  - `add_relu_wrapper_correctness_block` quantifies over all input values, so contents
    may change between runs.
  - With pointers and `n` unchanged, Triton's specialisation is unchanged.
- On the GPU: a resize or a storage swap is refused (`PlanInvalidated`); changing only
  the contents still runs correctly.
- Not thread-safe.

## 3. Results (run 2, final code; median of trials, µs)

Single call + synchronize:

| n | checked (original) | **checked (selected)** | prepared | prepared + graph | eager | eager, prealloc. | eager, graph | compile default | compile reduce-overhead | unfused checked |
|---|---|---|---|---|---|---|---|---|---|---|
| 2^12 | 95.4 | **54.9** | 23.6 | 17.1 | 28.1 | 22.3 | 10.9 | 85.8 | 148.5 | 158.4 |
| 2^16 | 105.5 | **63.4** | 28.9 | 22.2 | 37.6 | 30.2 | 14.6 | 94.9 | 162.9 | 173.5 |
| 2^20 | 104.9 | **63.7** | 31.4 | 22.1 | 37.6 | 30.6 | 21.6 | 96.7 | 163.6 | 177.0 |
| 2^22 | 173.0 | **131.0** | 80.0 | 99.8 | 238.5 | 76.3 | 73.0 | 144.3 | 518.0 | 296.4 |
| 700,001 | 104.7 | **64.2** | 30.2 | 21.9 | 37.3 | 30.0 | 18.5 | 93.2 | 157.6 | 174.0 |
| 3,000,017 | 117.9 | **64.0** | 36.8 | 30.4 | 48.1 | 43.0 | 39.2 | 94.8 | 308.7 | 176.0 |
| 2^24 | 919.6 | **877.4**¹ | 854.1 | 855.3 | 1460.7 | 1431.8 | 1428.6 | 901.0 | 2113.9 | 1533.1 |
| 2^25 | 1741.5 | **1701.6**¹ | 1689.0 | 1689.9 | 2892.5 | 2848.7 | 2845.0 | 1752.6 | 4111.9 | 2871.5 |

Repeated (per call):

| n | checked (original) | **checked (selected)** | prepared | prepared + graph | eager | eager, prealloc. | eager, graph | compile default |
|---|---|---|---|---|---|---|---|---|
| 2^12 | 78.6 | **36.3** | 13.1 | 8.0 | 13.1 | 10.5 | 3.8 | 66.4 |
| 2^16 | 85.7 | **43.9** | 15.2 | 9.8 | 19.3 | 14.2 | 5.4 | 67.9 |
| 2^20 | 84.2 | **43.2** | 17.1 | 10.5 | 19.7 | 14.9 | 12.6 | 70.0 |
| 3,000,017 | 86.0 | **43.7** | 15.6 | 15.5 | 31.0 | 29.3 | 28.8 | 68.0 |

¹ Block 64 (the rule's choice at n ≥ 2^24), measured as `checked_fast_B64`. The rule
itself was not timed as a unit. Its per-size timings are those of the configuration it
selects; every element is computed independently, so outputs are bitwise identical
across block sizes.

**Against the reviewer's target** (at least a 2× reduction of small-input checked-call
latency): **met back to back** (1.95–2.17×), **not met for single calls** (1.63–1.84×).
The remaining floor is synchronize + JIT dispatch + allocation.

**Gains:**
- **General checked API:** 1.63–1.84× faster for single calls at n ≤ 3M (1.32× at 2^22),
  and 1.95–2.17× faster back to back. Large inputs are unchanged (1.02–1.05×).
  It is now 1.5× faster than default `torch.compile` for small calls.
- **Prepared API:** 3.2–4.0× faster than the original checked call for single calls
  at n ≤ 3M, and 3.9–5.6× with graph replay.
  - Without the graph it matches eager into a preallocated output (e.g. 31.4 vs
    30.6 µs at 2^20; 36.8 vs 43.0 µs at 3,000,017).
  - With graph replay it is 6–8 µs slower than the eager CUDA graph at 2^12–2^16. That
    is the cost of the metadata revalidation, which the eager graph does not do. It is
    at parity from 2^20.
  - At 2^24–2^25 it is 1.7× faster than both eager baselines.
- **Device time with block 256** (warm cache): 5.8 vs 12.2 µs at 2^20, and 15.4 vs
  33.3 µs at 3,000,017. The stage-8 warm-cache regression against compiled at 2^20 is
  gone (5.8 vs 7.1 µs).

**Block search (run 1):**
- Checked end-to-end time differs by at most 2% across blocks 64–4096 at 2^12–2^20 and
  on the held-out sizes (the host dominates).
- The tie is broken by device time: 256 is lowest on both the search and held-out sizes.
- Block 64 stays 1–3% faster at 2^24–2^25 in every measurement, which gives the rule.

## 4. Regressions and anomalies

1. **General checked API vs eager at small n:** 54.9–64.2 µs vs 28.1–37.6 µs for single
   calls (about 1.7× slower), and about 2.2× slower back to back.
   - What remains is synchronize (9), Triton JIT dispatch (≈25 incl. launch), allocation
     (≈12) and the fast check (≈17).
   - Removing JIT dispatch safely needs a specialisation-keyed cache (not done).
   - **Eager is still faster for small inputs through the general API.** The prepared API
     matches it.
2. **Graph replay at 2^22** (99.8 µs) was slower than the precompiled launch (80.0);
   not investigated.
3. **`torch.compile(mode="reduce-overhead")` was slower than default compile at every
   size** (148–518 µs single), also with `cudagraph_mark_step_begin()`. This was
   measured, not diagnosed. Possible causes are cudagraph-tree bookkeeping on this
   host's slow CPU, or recording per shape. It should not be read as a general
   property of that mode.
4. **Block 256 at 2^25:** 1750.9 vs 1701.6 µs for block 64 (3%). The rule avoids it.

## 5. Proofs

- **`add_relu_wrapper_correctness_block (B)`** (`AddReluFused.lean`, frozen from
  `sorry`, FREEZE stage 9): the whole-wrapper headline for every block the unchanged
  `Elementwise2` contract accepts. The stage-8 64 headline is now its one-line
  instance, with the statement unchanged. The proof was generalised, not cloned.
- **`Blocked1D.fused_agrees_two_launch`** (new library file `Launch/Relational.lean`;
  FREEZE stage 10, kernel-agnostic):
  - Setup: a launch writing a temporary `T` with `F`, then a launch reading `T` and
    writing `O` with `G ∘ F`, versus one fused launch writing `O` with `G ∘ F`, all
    under per-program runs (`ProgramRuns`).
  - Result: the final states agree on every output element and on every cell outside
    `T[0, n)`.
  - The stated hypothesis `T ≠ O` was unused and was removed before proving (recorded).
- **`add_relu_fusion_relational`** (`AddReluRelational.lean`) instantiates that result
  for a checked configuration:
  - The unfused pipeline is `add_kernel` (text equal to `add_example.py`) into the
    temporary, then the masked `relu_kernel` (`relu_masked.py`, the ReLU spelled as
    `relu_forward`).
  - The fused kernel is `add_relu_kernel`.
  - Unit tests require all three transcriptions to equal their sources.
  - In the CPU interpreter the two-launch pipeline equals the fused kernel bitwise
    (`work/logs/93_relational_interp.log`).
  - Not covered: the strided block-pointer ReLU pipeline from stage 8, which is related
    only at the specification level.
- **Numerical contract unchanged:** exact ℝ; NaN and the sign of zero are outside it.
  No eager fallback is used anywhere, because eager differs on NaN sums and −0.0.

## 6. Not attempted, and a procedural slip

- Integrating the kernel into `torch.compile` (user-defined Triton kernel or
  `triton_op`) as an overhead-reduction route was not attempted.
- `latency_ladder.json` and `block_search.json` (run 1) are marked stale by the
  freshness rule. `AddReluFused.lean`, one of their hashed dependencies, was edited
  while run 1 was in flight (to add the block-general theorem). The kernel text they
  read is unchanged. They are superseded by `validation.json` (run 2, final code).
  Lesson: don't edit a hashed dependency during a run.
- The relational instance restates three kernels and re-derives their per-program runs
  (about 200 lines of the same unrolling), because bench files cannot import each other.
  The reusable part is the kernel-agnostic `Launch/Relational.lean`. The instance's
  input hypotheses are `readMem`-level (observations), deliberately weaker than the
  typed-cell wrapper headlines.

## 7. Remaining assumptions

- Metadata extraction (`raw_meta` / `tensor_meta`) is trusted.
- `fast_ew2` and the mirror are tested against Lean, not proved equal to it.
- The kernel-source binding: the body equals the Lean text, and roles are read by the
  unchanged recognizer from a projection.
- Triton's compiler and its specialisation rules are trusted. The prepared plan relies
  on unchanged arguments implying an unchanged specialisation.
- CUDA-graph replay reproduces the captured launch. The plan owns its buffers; nothing
  else may reallocate them between runs.
- Model level: typed element-sized cells, exact-ℝ arithmetic, and merge semantics for
  programs (TA-sched / TA-compose).
