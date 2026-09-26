# Dynamic-width concatenation repack (PyTorch #189940): stage 11

The program is the issue's nested concatenation. Two inner `torch.cat([q, k, va + vb], -1)` of `[n, w]` inputs are
concatenated again along the last dimension. Under `torch.compile(dynamic=True)`, Inductor lowers it to one kernel
that indexes with symbolic `% ks` / `// ks`. `repack_fastdiv` (`cat_repack.py`, candidate B in
`scripts/launch_cat.py`) replaces that runtime division with host-computed magic-number division (the ATen
`IntDivider` form).

## Measured on that host

One NVIDIA L4 (Modal, single vCPU), torch 2.14.0+cu130, Triton 3.8.0. Evidence:
`bench/tritonbench_g/add_example/launch_evidence/cat_repack.json` (call 1) and `cat_validation.json` (call 2).

- **The issue's claim did not reproduce here.** Its claim is relative to eager, and compiled dynamic was faster than
  eager on this host.
- **The mechanism did reproduce.** Compiled dynamic is about 1.8× slower than compiled static.
- **Criterion change.** The call-1 reproduce criterion failed. With the user's approval, the criterion was
  re-registered in the harness docstring before call 2.
- **Validation (call 2), candidate B against compiled dynamic:**
  - sequence (10 shapes): 1.78×;
  - held-out (4 shapes): 1.91×.
- **Against the best dynamic-shape baseline per shape** (eager, compiled dynamic, or the ATen-fallback formulation):
  - range 1.03× to 2.33× (the minimum is 3000 rows at odd widths);
  - candidate A (tiled, runtime division) drops to 0.67× at that shape.
- **Against compiled static, which recompiles for every shape:** 0.93× to 1.48×. This is parity, not a win.
- **Bitwise equality with eager:** every timed shape.
- **Edge cases:** correct. B refuses when `n * W ≥ 2^31`.

## Proved (exact ℝ, region memory model)

| theorem | statement |
|---|---|
| `FastDiv.magic_quotient` | `((x·m)/2^32 + x)/2^s = x / d` for `0 < d < 2^31`, `x < 2^31` |
| `FastDiv.magic_lt`, `FastDiv.sum_lt` | the multiplier and the 32-bit add do not overflow |
| `FastDiv.bitvec_quotient` | the same computation in `BitVec 32` (umulhi, wrapping add, shift) equals `x / d` |
| `nestedCat_eq_flat` | the nested program equals the six-segment flat concatenation at every in-range (row, col) |
| `repack_lane_correct` | the transcribed kernel with the wrapper's constants (`total = n·W`, `magic W`, `shiftFor W`), `n·W < 2^31`: each active lane holds the flat concatenation at `(x / W, x % W)`, and each inactive lane is unchanged |
| `cat_repack_launch_correctness` | a `(g,)` launch covering `n·W` with `out` distinct from the eight inputs writes `nestedCat` at every `x < n·W` and changes no other cell |

- **Axioms:** `propext`, `Classical.choice`, `Quot.sound` only. There is no `sorry`, `admit` or `native_decide`.
- **Frozen statements:** they were frozen while their proofs were still `sorry`, and are unchanged
  (`work/frozen/FREEZE.md`, stage 11).

## Not claimed

- **Floating point.** Values are exact ℝ. The kernel's `va + vb` is one fp add in both eager and the kernel, and
  bitwise agreement was tested, not proved.
- **Host code.** The Python wrapper's constant computation (`magic_for`) is tested against Lean `shiftFor` / `magic`
  on 313 divisors × 7 values. It is not proved. Its guard (`n * W < 2^31`) is the theorem's hypothesis.
- **The transcription.** The transcription binding is a syntax-tree comparison against the pinned kernel through a
  declared table (`scripts/test_launch_check.py`). It is not a verified front end.
- **Other hardware.** Nothing is claimed about performance on any other GPU or host.
