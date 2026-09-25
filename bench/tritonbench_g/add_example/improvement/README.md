# add_example — contract-preserving improvement candidates

Both candidates change only the host wrapper. The `@triton.jit` kernel text is
byte-identical to the pinned `../add_example.py`, and every manifest
configuration is accepted by the proved checker (verdicts kernel-checked), so
`add_kernel_launch_correctness` applies to those launches unchanged, under the
same translation assumptions (TA-*, CONTRACT.md §4). Wrapper obligations W1
(`n == out.numel()`, needed for `empty_like` to return only written cells) and
W2 hold for every manifest configuration.

| file | change vs pinned | Lean / checker basis |
|---|---|---|
| `add_example_empty_like.py` | `zeros_like` → `empty_like` | `add_kernel_launch_initial_output_irrelevant` (initial output contents are dead); manifest `launch_manifest.json` |
| `add_example_block64.py` | as above, plus `BLOCK_SIZE = 4` → `64` | checker obligation P2 admits any power of two ≤ 2^20; all 10 manifest configs accepted with kernel-checked verdicts, incl. n = 2^31 − 1 (grid 2^25); `../launch_evidence/adapter_block64.json` (manifest `launch_manifest_block64.json`). The adapter's kernel summary is identical to the pinned file's; the launch summary differs only in `block` |

## GPU evidence (NVIDIA L4 via Modal, Triton 3.8.0, torch 2.14.0+cu130 — this device only)

`../launch_evidence/block_sweep.json` (`modal run scripts/launch_gpu_modal.py --mode sweep`)
sweeps BLOCK ∈ {4, 16, …, 4096} over the `empty_like` wrapper text. It was run three
times on separate L4 instances (`block_sweep_run1.json` at `104ab827`, `block_sweep_run2.json` at
`d3cdb079`, `block_sweep.json` at `174c4d55`); the table below is run 1. Selection rule, fixed in
`scripts/launch_gpu.py` before the run: smallest BLOCK within 2 % of the minimum median
`empty_like`-wrapper time at n = 2^24, among BLOCKs whose correctness cases all pass. All nine
BLOCKs were bitwise equal to torch `x + y` with intact sentinels; the rule selected **64**. The
sweep's text for BLOCK 64 equals `add_example_block64.py`'s kernel/wrapper text byte-for-byte.

End-to-end wrapper medians (`triton.testing.do_bench`, warmup 25, rep 200):

| n | pinned (4, zeros_like) | empty_like only | block64 (64, empty_like) | torch `x + y` |
|---|---|---|---|---|
| 2^16 | 25.6 µs | 24.6 µs | 8.2 µs | 9.2 µs |
| 2^20 | 260 µs | 247 µs | 63.5 µs | 63.5 µs |
| 2^24 | 3.81 ms | 3.55 ms | 0.890 ms | 0.886 ms |

Runs 2 and 3 replicated it: all BLOCKs correct, the rule selected 64 both times; at 2^24
pinned 3.83 / 3.81 ms, empty_like only 3.60 / 3.56 ms, block64 0.891 / 0.891 ms, torch
0.884 / 0.886 ms.

On this L4 the block64 wrapper matches torch's own `x + y` (≈ 225 GB/s effective at 2^24);
every BLOCK from 64 to 4096 lands within 2 % of it at 2^24, so the gain comes from leaving the
BLOCK = 4 regime, not from fine tuning. Other GPUs, drivers or Triton versions were not
measured; no general speedup claim is made. Run-to-run variation between the HANDOFF §B run and
the sweep, same configuration: bare kernel ≈ 1.5 %, BLOCK 4 wrappers ≈ 3–4 %. The `empty_like`
gain at BLOCK 4 replicated in both runs in direction and size (5.9 % and 6.7 % at 2^24), so it is
not noise even though it is close to that spread.
