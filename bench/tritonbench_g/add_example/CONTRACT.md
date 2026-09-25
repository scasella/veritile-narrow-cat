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

Which obligations the Lean conclusions use (§3): P1–P3 the framed
correctness conjunct, P4 the trace-safety conjunct, P5–P7 the i32 conjunct.
P8–P10 appear in no Lean conclusion; they are checked because they justify the
translation assumptions of §4 — P8 and P9 that element `i` of each buffer is
the cell `(region, i)` holding an f32 (TA-region), P10 that the output is
disjoint from the inputs (TA-region, TA-compose). `capacity` bounds addresses
by the *allocation* (memory safety); that each input *tensor* holds the `n`
elements read is wrapper obligation W2 (§4).

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
at least the checked capacities, and the i32 offset/mask values equal their ℕ
counterparts as arithmetic facts about `c` (TA-i32 is what ties them to the
kernel's execution). Separately, `add_kernel_launch_applicable`
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
- Wrapper obligations W1 (`n == out.numel()`: the returned tensor is exactly
  the cells this contract covers) and W2 (`n ≤ t.numel()` for every input
  tensor `t`: the cells read are the tensor's own elements, not merely inside
  its allocation) are outside this *launch-level* contract (§1–§3), where the
  adapter checks them (SOURCE_LINK.md). The wrapper-level contract of §7
  derives both as theorems.

## 5. Unsupported / non-goals

Grids of rank ≠ 1; non-power-of-two blocks; strided or non-contiguous inputs;
dtypes other than f32; in-place or partially overlapping output; `n ≥ 2^31`;
autotuning; multi-stream concurrency with other kernels; IEEE NaN/overflow;
performance; device placement (all tensors on one CUDA device is assumed, not
checked — the adapter reads metadata from CPU tensors built from the manifest).

## 6. Concrete valid configurations (nonempty domain)

`n ∈ {16, 8, 32, 0}`, `block = 4`, `grid = [cdiv(n,4)]` (the file's own test
cases), fresh contiguous f32 tensors; also `n = 5, block = 4, grid = [3]`
(over-provisioned) and `vector_addition_custom`'s `block = 16`.

## 7. Whole-wrapper contract (revision 3)

What the caller of `add_wrapper(x, y)` receives, stated from the tensors'
own metadata (`VeriTile/Triton/Launch/Blocked1DWrapper.lean`):

- `TensorMeta`: data pointer (bytes), element size, shape, element strides,
  capacity (elements addressable from the data pointer inside the
  allocation), dtype. `Elementwise2.launch 4 x y out` is the launch the
  wrapper derives (`n = x.numel()`, `BLOCK_SIZE = 4`, `cdiv` grid).
- `Elementwise2.Pre` (decided by `Elementwise2.check`, with `check_ok` /
  `check_complete`): WA equal shapes (the supported API), WC all tensors
  contiguous, P12 element-aligned data pointers, P11 inputs identical or
  byte-disjoint, and P1–P10 of the derived launch.
- **W1 and W2 are theorems**, not adapter checks: `Pre.output_covered`
  (`n = out.numel`) and `Pre.inputs_cover` (`n ≤ t.numel` for both inputs).
- Headline `add_wrapper_correctness` (`AddExample.lean`): (1) every element
  `i < out.numel` of the returned tensor holds `xs i + ys i`, all other cells
  unchanged; (2) for every in-shape multi-index the three tensors address the
  same row-major element below `out.numel`; (3) every program is trace-safe for
  bounds equal to the tensors' own element counts; (4) running the programs
  serially in **any** complete order gives the same result
  (`Kernel.runSerial_agrees_merge`); (5) in the flat memory placed at the
  tensors' element addresses `base / elemBytes`, the checked conditions
  discharge `FlatAlloc.Disjoint` and closure and the flattened launch writes
  `xs i + ys i` at `out.base / out.elemBytes + i`, every other flat cell
  unchanged (per program this is the upstream `⊨` headline).

How the layout obligations now do formal work (for `add_example`): P9 + P12
give the element addressing `base / 4`; P10 + P11 give `FlatAlloc.Disjoint`
(`Elementwise2.Pre.flat_disjoint`); P3 + P4 place every flat window inside its
allocation (`Pre.windows_in_alloc`, not used by the headline). The logical
view comes from WC and `TensorMeta.offsetOf_eq_linear`; P8 itself is still
used by no Lean conclusion.

Scope of conjunct (5): it is stated for the flat image `flattenState s` of a
region state, in which every flat cell outside the three windows is `0`. It
shows that the checked metadata satisfies the flat bridge's hypotheses and
what the flattened launch does from such an image; it does not relate the
windows to arbitrary surrounding memory, so TA-region is narrowed, not
discharged.

Remaining assumptions after revision 3: TA-transcription (the Lean kernel is
the Python kernel); TA-region (tensor elements are the region / flat cells,
see the scope of (5)); metadata extraction from live tensors and the
invocation-time Python mirror of the checker (differentially tested against
the Lean definitions, not proved); the flat model's typed element-sized cells
(not byte-level memory); hardware executing the programs
equivalently to some whole-program serial order (TA-sched: instruction-level
interleaving is not modelled — the serial theorem removes the stronger
merge-from-initial-state presumption, and non-interference is proved for this
kernel); TA-i32 typing; IEEE-single-add; device placement.

`vector_addition_custom.custom_add` has a narrower wrapper contract on its
supported rank-1 API (`Elementwise2.checkRank1`, headline
`custom_add_correctness`): conjuncts (1) and (3) only — every element of the
returned tensor written, logical extents respected — with no logical-view,
serial or flat conjunct, so it still relies on TA-compose. Beyond rank 1 its
`size(0)` launch provably leaves outputs unwritten
(`Elementwise2.launchDim0_not_covered`) and is rejected.

## Revisions

- Doc revision 2 (2026-09-25, after a separate-model review): obligation-use
  mapping in §2, i32 conjunct described as arithmetic under TA-i32 (§3),
  wrapper obligation W2 (§4), device placement listed as a non-goal (§5).
  No Lean statement changed.
- Doc revision 3 (2026-09-25, whole-wrapper milestone): §7 added. It
  describes new theorems; no obligation, reference or relation of §1–§6
  changed. Review: same-agent, then a separate-model review (see REPORT).
