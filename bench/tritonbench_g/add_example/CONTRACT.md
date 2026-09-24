# `add_example` host-launch contract (frozen v1, 2026-09-24)

Scope: one launch `add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE)` of the
source-pinned kernel in `add_example.py` (TritonBench-G v1, VeriTile
@95a01f5). This contract adds the **host launch** obligations that the existing
per-program headline `add_kernel_correctness` (`addIO ⊨ xs + ys`) leaves as
hypotheses or as a trusted boundary.

## 1. Semantic contract (implementation-independent)

Inputs `x`, `y`; output `out`; `n = n_elements`.

- **Reference.** `ref(x, y)[i] = x[i] + y[i]` for `0 ≤ i < n`, over ℝ. The
  reference mentions neither the kernel, its denotation, nor the launch
  metadata (Lean: the literal expression `xs i + ys i`; Python test oracle:
  `ref_add`).
- **Numerical relation.** Exact-ℝ (algorithm layer). Relation to hardware is
  the named external assumption *IEEE-single-add*: each lane performs one
  correctly rounded fp32 addition of the loaded fp32 values, so hardware output
  = `round_fp32(x[i] + y[i])`. Not proved here; recorded as external.
- **Output correctness.** After the launch, for every `i < n`,
  `out[i] = x[i] + y[i]`.
- **Preservation.** Every memory cell other than `out[0..n)` is unchanged
  (including `out[n..capacity)`, the inputs, and every unrelated region).
- **Termination.** Every launched program terminates successfully.
- **Memory bounds.** Every active-lane load/store address lies inside the
  allocation of the buffer it touches.
- **Disjoint writes.** Distinct (program, lane) pairs with active masks write
  distinct cells; each output cell `i < n` is written by exactly one program
  (`pid = i / BLOCK_SIZE`, lane `i % BLOCK_SIZE`).
- **Empty input (`n = 0`).** Accepted. With `grid = cdiv(0, B) = (0,)` the
  NVIDIA/AMD launchers skip the launch (TRITON_FACTS §4); with an
  over-provisioned grid every lane is masked off. Either way no cell changes.

## 2. Implementation-specific launch preconditions (checked)

Metadata `c : Blocked1DLaunch`: `n`, `block`, `grid : List Nat`, and for
each buffer (`x`, `y`, `out`): `base`
(byte address), `elemBytes`, `stride` (elements, dim 0 of the flattened
view), `capacity` (elements addressable from `base` inside its allocation),
`dtype`. The integer width is fixed to Triton's i32 (`i32Limit = 2^31`).

`Blocked1DLaunch.Pre c` (Lean, `VeriTile/Triton/Launch/Blocked1DConfig.lean`):

| # | Obligation | Meaning |
|---|---|---|
| P1 `grid_rank` | `grid = [g]` | 1-D launch only |
| P2 `block_ok` | `∃ k ≤ 20, block = 2^k` | `tl.arange` power-of-two, ≤ TRITON_MAX_TENSOR_NUMEL (TRITON_FACTS §3) |
| P3 `covers` | `∀ i < n, i / block < g` | the owner program of every output index is launched |
| P4 `lanes_in_bounds` | `∀ b, ∀ pid < g, ∀ j < block, pid*block+j < n → pid*block+j < b.capacity` | active-lane addresses in every allocation |
| P5 `offsets_fit` | `∀ pid < g, ∀ j < block, pid*block+j < 2^31` | i32 offset arithmetic does not wrap |
| P6 `n_fits` | `n < 2^31` | `n_elements: tl.int32` is not truncated |
| P7 `grid_fits` | `g < 2^31` | grid dim fits the launcher's C `int` |
| P8 `unit_stride` | `∀ b, n ≤ 1 ∨ b.stride = 1` | kernel's `ptr + offsets` addressing matches the tensor layout |
| P9 `dtype_ok` | every buffer `dtype = f32 ∧ elemBytes = 4` | supported dtype domain v1 |
| P10 `out_disjoint` | the byte ranges `[base, base + n*elemBytes)` of `out` and of each input do not intersect | non-overlapping output storage |

Soundness theorem shape: `Blocked1DLaunch.check c = true → Blocked1DLaunch.Pre c`
(`check_ok`), plus completeness `Pre c → check c = true` (`check_complete`), so
a rejection is a checked failure of a named obligation, and the checker is not
vacuously strict.

Aliasing: `x` and `y` may alias each other (read-only). `out` must not overlap
either input, even exactly in place (conservative; in-place is a non-goal).

## 3. Kernel-level whole-launch theorem (region model)

`add_kernel_launch_correctness` (in `AddExample.lean`): from `check c = true`,
region names `in_ptr0 in_ptr1 out_ptr`, and an initial state whose input cells
`k < n` are **typed real cells** (`s.mem in_ptr0 k = MemCell.real (xs k)`; no
reliance on totalized reads), the grid `[g]` launch composes (disjoint frames,
`GridLaunchedOrdinary`) into a final memory with output correctness and
preservation as in §1, and every program is `TraceSafe` for any region bounds
at least the checked capacities. Separately, `add_kernel_launch_applicable`
discharges the lane-wise bound hypotheses of the existing flat-memory headline
`add_kernel_correctness` for every launched program.

## 4. External / translation assumptions (not checked by the checker)

- TA-meta: the supplied metadata truthfully describes the tensors (adapter
  reads real torch tensors when given; a dict is trusted as supplied).
- TA-region: disjoint byte ranges are modeled as distinct VeriTile regions,
  element `i` of a unit-stride buffer as cell `(region, i)`.
- TA-i32: Triton computes `pid`, `pid*BLOCK_SIZE`, `+ arange`, and the mask
  compare in two's-complement i32 (TRITON_FACTS §1–3); the Lean BitVec lemma
  proves agreement with ℕ under P5–P7, the Triton typing itself is trusted.
- TA-compose: whole-grid composition by `mergeFrames` from the initial state
  presumes no program reads a cell another program writes; P10 + distinct
  regions make the kernel's read set (inputs) disjoint from all write sets.
- TA-transcription: the Lean `add_kernel` corresponds to the Python body
  (existing upstream structural scan + this project's AST matcher; trusted,
  tested, not proved).
- IEEE-single-add (numerics), device/driver behavior, and compiler correctness.
- Wrapper obligation W1 (`n == out.numel()`: the returned tensor is exactly
  the cells this contract covers) is outside this contract; it is checked by
  the adapter and recorded separately (SOURCE_LINK.md).

## 5. Unsupported / non-goals

Grids of rank ≠ 1; non-power-of-two blocks; strided or non-contiguous inputs;
dtypes other than f32; in-place or partially overlapping output; `n ≥ 2^31`;
autotuning; multi-stream concurrency with other kernels; IEEE NaN/overflow;
performance.

## 6. Concrete valid configurations (nonempty domain)

`n ∈ {16, 8, 32, 0}`, `block = 4`, `grid = [cdiv(n,4)]` (the file's own test
cases), fresh contiguous f32 tensors; also `n = 5, block = 4, grid = [3]`
(over-provisioned) and `vector_addition_custom`'s `block = 16`.
