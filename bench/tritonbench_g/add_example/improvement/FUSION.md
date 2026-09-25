# Fused float32 add + ReLU — measured optimization experiment

One device (NVIDIA L4, CC 8.9, driver 580.95.05), Triton 3.8.0, torch 2.14.0+cu130,
one Modal call. Raw results: `launch_evidence/relu_tune.json`,
`launch_evidence/fusion_bench.json` (every timing sample summary, kernel names,
environment, input hashes); raw Modal output `work/logs/bench_raw.json`.

## Result

**Workload:** `relu(x + y)` on contiguous float32 tensors of 2^24–2^25 elements.

| n | device time (kernel sum), fused vs | | | end-to-end (checked), fused vs | | |
|---|---|---|---|---|---|---|
| | best unfused | eager | compiled | best unfused | eager | compiled |
| 2^24 | **1.71×** | 1.78× | 1.05× | **1.68×** | 1.61× | 0.98× |
| 2^25 | **1.71×** | 1.78× | 1.02× | **1.66×** | 1.66× | 1.00× |

At these sizes all four candidates run at 236–253 GB/s (L4 DRAM peak ≈ 300 GB/s).
The gain is the predicted one: the fused kernel moves 12 bytes per element, and the
unfused pipeline and eager PyTorch move 20 (ceiling ≈ 1.67×). `torch.compile` fuses the
same way and is at **parity**; the fused kernel's advantage is that it carries a proof,
not that it is faster than the compiler.

**Selected implementation:** `add_relu_fused.py::add_relu_wrapper` (BLOCK_SIZE 64,
Triton's default `num_warps`), called through `launch_fused.CheckedFusedAddRelu`. ReLU
tuning: **no change** (below).

## Times (medians)

`kernel` = summed CUDA kernel durations of one call (torch.profiler; no L2 flush between
calls, so for n ≤ 2^22, where the working set fits the L4's 48 MB L2, it is a warm-cache
figure). `event` = `triton.testing.do_bench` (L2 flushed before each run; includes launch
gaps). `e2e` = wall time of one complete call plus `synchronize` (checked paths: metadata,
contract decision, allocation, launch). Microseconds.

| n | candidate | kernel | event | e2e |
|---|---|---|---|---|
| 2^16 | fused checked | 1.9 | 8.2 | 99.0 |
| | unfused checked | 3.2 | 10.8 | 164.5 |
| | eager | 2.9 | 11.0 | 35.0 |
| | compiled | 1.5 | 8.3 | 86.4 |
| 2^20 | fused checked | 12.2 | 65.6 | 100.6 |
| | unfused checked | 17.2 | 79.9 | 165.7 |
| | eager | 12.2 | 76.4 | 33.1 |
| | compiled | 7.0 | 64.1 | 86.2 |
| 2^22 | fused checked | 73.5 | 233.6 | 170.1 |
| | unfused checked | 202.3 | 289.9 | 288.6 |
| | eager | 214.8 | 290.0 | 235.3 |
| | compiled | 86.7 | 225.5 | 132.3 |
| 2^24 | fused checked | 797.1 | 828.8 | 906.5 |
| | unfused checked | 1360.8 | 1453.6 | 1522.5 |
| | eager | 1419.7 | 1455.0 | 1458.8 |
| | compiled | 835.1 | 879.1 | 888.7 |
| 2^25 | fused checked | 1611.8 | 1645.2 | 1731.4 |
| | unfused checked | 2758.8 | 2897.5 | 2865.9 |
| | eager | 2867.5 | 2903.9 | 2871.1 |
| | compiled | 1639.9 | 1741.5 | 1739.7 |

The best unfused implementation is the checked `add_example_block64` add (proved by
`add_kernel_launch_correctness`) into a temporary, then the checked strided ReLU
(`store_cast_fix` text, `num_warps` 4). 2^25 is the largest size the ReLU's proved
one-tile branch accepts; larger inputs take its unsupported grid-stride branch, so the
comparison stops there.

## Regressions (fused slower by more than 2%)

1. **Small tensors, end-to-end, vs eager:** 99–101 µs vs 33–35 µs at 2^16–2^20 (0.33–0.35×),
   and 0.78–0.87× vs `torch.compile` up to 2^22. The checked path's host cost dominates.
   On the Mac, the contract check alone measured ≈ 8 µs (`invoke_overhead.json`); the rest
   is Triton's Python launch path. This run did not separate the two on the GPU host.
   For small inputs, eager PyTorch is the better choice.
2. **Warm-cache kernel time vs compiled at 2^16–2^20** (0.58–0.77×): the compiled kernel
   is faster when the data is L2-resident; with the cache flushed (`event`) the two are
   equal (65.6 vs 64.1 µs at 2^20). Likely cause: BLOCK_SIZE 64 launches many tiny
   programs; the block diagnostic was only run at 2^24 (64: 798 µs, 256: 811, 1024: 845,
   4096: 833). Not pursued (it would be further tuning).

## ReLU tuning pass (bounded)

Searched: `num_warps` ∈ {1, 2, 4, 8} for the checked strided ReLU at five sizes. This is
the only knob inside the proved contract: the recognizer and the Lean model ignore it.
Not searched: tile cap 512 and CTA cap 65536, which are fixed by the pinned wrapper text
and `StridedUnary.launch` (the frozen contract). Result: all settings are within about 2%
(geometric mean over n ≥ 2^22: 1: 222.5, 2: 223.4, 4: 220.4, 8: 218.6 µs). The rule keeps
the current 4. The checked Triton ReLU matches `torch.relu` (e.g. 1144 vs 1139 µs at 2^25).
**No worthwhile gain.**

## Proof and assumptions

- `AddReluFused.lean` (official comparator: 39 theorems accepted; frozen from its `sorry`
  state, FREEZE stage 8):
  - `add_relu_kernel_correctness` is the per-program flat-memory triple: every active
    lane gets `relu (xs i + ys i)` and every other cell is unchanged.
  - `add_relu_wrapper_correctness`, under the unchanged `Elementwise2` contract at block
    64: complete output coverage, the logical view, trace safety within each tensor's
    extent, serial-order independence, and the flat placement at the tensors' element
    addresses with every other flat cell unchanged.
- **Transformation:**
  - The fused spec is the composition of the two proven specs: `add_wrapper_correctness`
    (`xs i + ys i`) and `relu_wrapper_one_tile_correctness` (`relu` per element). On
    every logical element the fused wrapper returns what the unfused pipeline returns
    under those theorems.
  - Its write footprint is the output alone. The unfused pipeline also writes a temporary.
  - The memory-level composition of the two unfused launches is **not** formalized.
- **Rounding points:** the model is exact-ℝ. On the hardware both pipelines round once,
  at the f32 add, and the ReLU is a select. The fused result was bitwise equal to the
  unfused pipeline for:
  - every random input at every size;
  - all 144 special-value pairs;
  - `torch.compile` on random inputs.
- **Numerical scope:** NaN and the sign of zero are outside the contract.
  - Where `x + y` is NaN, the fused and unfused Triton paths return +0.0 and eager
    returns NaN (25 of 144 pairs).
  - For (−0.0) + (−0.0), eager CUDA `torch.relu` returns −0.0; the Triton paths return
    +0.0.
  - `torch.compile` differs from the fused kernel on 25 of the 144 pairs. The harness
    recorded only that count, which equals the number of NaN-sum pairs; which pairs they
    are was not recorded.
- **Frame on hardware:** launched into a window of a sentinel-filled buffer (n = 1, 63, 64,
  65, 1000), every output element was written and every sentinel kept.
- **Trusted:**
  - metadata extraction and the Python mirror of `Elementwise2.check` (differential-tested);
  - the kernel-body binding (statement-for-statement equality with the Lean text) and
    role extraction from a projection read by the unchanged recognizer;
  - Triton's compilation of the kernel;
  - typed element-sized cells standing for bytes.

## Reproduce

```bash
python3 scripts/launch_bench_modal.py --list          # upload set (42 files)
modal run scripts/launch_bench_modal.py               # one L4 call, ~2 minutes
# on a CUDA host directly:
python3 scripts/launch_bench.py > bench.json
# CPU interpreter dry run (correctness and plumbing only, no timings):
TRITON_INTERPRET=1 python3 scripts/launch_bench.py
```
