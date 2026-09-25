# `relu_strided_buffer` — whole-wrapper contract (one-tile branch)

Scope: `relu_forward_wrapper_rank_1(in0, out0=out0)` of the source-pinned
`relu_strided_buffer.py`, for rank-1 **strided** tensors, when the wrapper
selects the one-tile-per-program branch. The per-program kernel proofs of
`ReluStridedBuffer.lean` (`relu_one_tile_region_run`,
`relu_one_tile_traceSafe`) are reused unchanged; this contract adds host
applicability and whole-grid composition.

## Launch (computed from metadata)

`StridedUnary.launch x out` (`VeriTile/Triton/Launch/StridedUnary.lean`)
mirrors the wrapper: `s0 = out.shape[0]`, `tile = min(512, next_power_of_2(s0))`,
`tiles = cdiv(s0, tile)`, `ctas = min(65536, tiles)`,
`tiles_per_cta = cdiv(tiles, ctas)`, grid `(ctas, 1, 1)`, kernel stride
arguments `in0.stride(0)` / `out0.stride(0)`. Element strides and capacities
are measured from each view's **data pointer** (`data_ptr()`, which for a
`StridedBuffer` already includes its element offset), so the allocation bound
reads `(s0 − 1)·stride < capacity`.

## Obligations (`StridedUnary.Pre`, decided by `StridedUnary.check`)

| # | Obligation | Why |
|---|---|---|
| S1 `rank1` | both tensors rank 1, one stride each | supported API |
| S2 `same_shape` | `in0.shape = out0.shape` | the wrapper's assertion |
| S3 `nonempty` | `0 < s0` | the pinned wrapper raises `ZeroDivisionError` at `s0 = 0` (`next_power_of_2(0) = 0`, then `cdiv(0, 0)`) |
| S4 `one_tile` | `tiles ≤ 65536` | selects the one-tile branch; the grid-stride branch is not covered |
| S5 `pos_strides` | both element strides `> 0` | negative strides (possible for `StridedBuffer`) are outside ℕ; zero strides alias outputs |
| S6 `in_bounds` | `(s0−1)·stride < capacity` for both | every strided element inside its allocation |
| S7 `dtype_ok` | both f32, 4 bytes | supported dtype; `.to()` casts are erased in Lean; `StridedBuffer` dtype reinterpretation is rejected |
| S8 `aligned` | data pointers multiples of 4 | element addressing |
| S9 `spans_disjoint` | input and output byte spans `[ptr, ptr + ((s0−1)·stride+1)·4)` disjoint | no program reads a cell another writes (conservative: interleaved disjoint strided sets are rejected) |
| S10 `offsets_fit` | `ctas · tile ≤ 2^31` | block-pointer offsets are `i32` (Triton 3.8.0) |
| S11 `addresses_fit` | `(s0−1)·stride·4 < 2^63` | block-pointer shape/strides are `i64` |

`check_ok` / `check_complete` make every rejection a named failed obligation.

## Headline `relu_wrapper_one_tile_correctness` (`ReluStridedBuffer.lean`)

For checked tensors, input and output in distinct regions, and input logical
element `k` (element offset `k·in_stride` from the data pointer) reading `xs k`:

1. every logical output element `k < out.numel`, at `k·out_stride`, holds
   `relu (xs k)`, and **every other cell is unchanged** — including the gap
   cells between strided outputs and the whole input;
2. every program is trace-safe for bounds covering each tensor's own strided
   extent;
3. running the programs serially in any complete order gives the same result;
4. every block-pointer offset `pid·tile + i` agrees between `i32` and ℕ.

The `(ctas, 1, 1)` grid is handled by `Blocked1D.liftFrames` /
`mergeFrames_liftFrames` (programs of `(g,1,1)` see the same ids and counts as
`(g,)`); composition over the strided footprint is
`Blocked1D.launch_of_frames_addr`.

## Wrong stride inside the allocation

`relu_wrong_stride_reads_wrong_element` (Lean): launching with input stride 1
over a view whose stride is 2 stays inside the allocation but writes
`relu(cell 1)` where the view's element 1 is cell 2. A bounds check cannot
see this. What excludes it is the stride-argument correspondence: the
invocation recognizer (`scripts/launch_invoke.py: recognize_relu`) requires
the kernel's stride arguments to be `in0_strides[0]` / `out0_strides[0]` with
`in0_strides = in0.stride()` / `out0_strides = out0.stride()`; a source with
the arguments swapped is `unsupported` (unit tests + interpreter demo).

## Evidence

- Lean, kernel-checked layout cases (`bench/tests/Blocked1DLaunchWitnesses.lean`):
  accepted contiguous `n = 1025` and strided (in stride 2, out stride 3);
  rejected empty, grid-stride branch (`n = 2^25 + 1`), stride outside the
  allocation, overlapping spans, float16.
- Differential test of the Python mirror against the Lean definitions: 2000
  randomized cases, 0 mismatches in verdict or derived launch
  (`launch_evidence/invoke_differential_strided.json`).
- Triton 3.8.0 CPU interpreter (`launch_evidence/invoke_interpreter.json`,
  interpreter evidence only): contiguous, input-strided and output-strided
  (gap cells intact) invocations equal `torch.relu`; empty, grid-stride,
  overlap, float16, negative-stride and dtype-reinterpreting `StridedBuffer`
  inputs rejected before launch. The interpreter needs a derived text
  (device guard → `nullcontext`; store cast to the pointer's element type)
  and cannot receive `StridedBuffer` arguments (verdict only).
- No GPU run of this consumer yet.

## Not covered / trusted

Grid-stride-loop branch; negative strides; dtype reinterpretation; empty
input; flat-memory placement for this kernel (the flat bridge is instantiated
for `add_example` only); byte-level memory (cells are typed and element-sized);
instruction-level interleaving (TA-sched); the recognizer (a pinned-pattern
AST match, trusted and tested) and metadata extraction (trusted).
