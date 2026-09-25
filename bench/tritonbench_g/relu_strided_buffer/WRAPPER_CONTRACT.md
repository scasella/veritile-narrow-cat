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
| S3 `nonempty` | `0 < s0` | the pinned wrapper raises `ZeroDivisionError` at `s0 = 0`: `next_power_of_2(0) = 0`, then `max_tile_size // tile_size` in `heuristics_for_tile_size` (line 14), before the device guard and the launch |
| S4 `one_tile` | `tiles ≤ 65536` | selects the one-tile branch; the grid-stride branch is not covered |
| S5 `pos_strides` | both element strides `> 0` | negative strides (possible for `StridedBuffer`) are outside ℕ; zero strides alias outputs |
| S6 `in_bounds` | `(s0−1)·stride < capacity` for both | every strided element inside its allocation |
| S7 `dtype_ok` | both f32, 4 bytes | supported dtype; `.to()` casts are erased in Lean; `StridedBuffer` dtype reinterpretation is rejected |
| S8 `aligned` | data pointers multiples of 4 | element addressing |
| S9 `spans_disjoint` | input and output byte spans `[ptr, ptr + ((s0−1)·stride+1)·4)` disjoint | justifies modelling input and output as distinct regions (the headline's `in0_ptr ≠ out0_ptr`, from which non-interference is proved); conservative: interleaved disjoint strided sets are rejected |
| S10 `offsets_fit` | `ctas · tile ≤ 2^31` | block-pointer offsets are `i32` (Triton 3.8.0) |
| S11 `addresses_fit` | `(s0−1)·stride·4 < 2^63` | block-pointer shape/strides are `i64` |

`check_ok` / `check_complete` make every rejection a named failed obligation.
S6–S9 are consumed by the flat-memory headline below (S7/S8/S9 →
`FlatAlloc.Disjoint`, S6 → span inside the allocation, S1/S7/S8 → cell/byte
correspondence). S11 still has no Lean consumer (the model has no `i64`
arithmetic); it is checked for the real-memory correspondence only.

## Headline `relu_wrapper_one_tile_correctness` (`ReluStridedBuffer.lean`)

For checked tensors, input and output in distinct regions, and input logical
element `k` (element offset `k·in_stride` from the data pointer) holding the
typed real cell `xs k`:

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

## Headline in flat memory `relu_wrapper_one_tile_flat_correctness` (`ReluStridedBuffer.lean`)

The region-model headline above assumes input and output are distinct
regions. This one places both in **one flat memory**, from the tensors' own
metadata: `StridedUnary.flatAlloc` (`VeriTile/Triton/Launch/StridedUnaryFlat.lean`)
puts region `in0` at cell `x.base / 4` and region `out0` at cell `out.base / 4`,
each over its view's strided span `(s0 − 1)·stride + 1` cells. For checked
tensors, any two distinct region names, and typed real input cells `xs k`:

1. the translated `(num_ctas, 1, 1)` launch writes `relu (xs k)` at the flat
   cell of every logical output element `k < out.numel`, and **every other
   flat cell is unchanged** — output gap cells, the whole input span, and
   cells outside both spans;
2. the placement satisfies the bridge's `FlatAlloc.Disjoint` (from S7, S8, S9);
3. `4 · (flat cell of logical element k) = base + elemBytes · offsetOf [k]`
   for both tensors (S1, S7, S8) — the stated cell/byte correspondence;
4. each placed span lies inside its tensor's allocation (S6);
5. no input logical element shares a cell with an output logical element.

The per-program step is upstream's `relu_strided_buffer_one_tile_io_correctness`
(`⊨`, flat placement for every disjoint placement), reused unchanged; the new
proof discharges its placement, window and state hypotheses from the checked
metadata and composes programs with `Blocked1D.launch_of_frames_addr` over the
flat footprint `base_out + k · out_stride`.

Limits of this statement: cells are typed and element-sized, so (3) is the
whole byte-level claim — nothing about sub-element aliasing, caching or
hardware memory ordering. Serial-order agreement (conjunct 3 of the region
headline) is not restated here: upstream's `⊨` starts from `flattenState` of a
region state, not an arbitrary flat memory, so per-program robustness does not
lift through the bridge; conjunct (5) is the flat-memory separation fact.

## Wrong stride inside the allocation

`relu_wrong_stride_reads_wrong_element` (Lean): launching with input stride 1
over a view whose stride is 2 stays inside the allocation but writes
`relu(cell 1)` where the view's element 1 is cell 2. A bounds check cannot
see this. What excludes it is the stride-argument correspondence: the
invocation recognizer (`scripts/launch_invoke.py: recognize_relu`) requires
the kernel's stride arguments to be `in0_strides[0]` / `out0_strides[0]` with
`in0_strides = in0.stride()` / `out0_strides = out0.stride()`; a source with
the arguments swapped is `unsupported` (unit tests + interpreter demo).

## Wrong output stride writes a gap cell

`relu_wrong_out_stride_writes_gap_cell` (Lean): launching with output stride
`1` over a view whose stride is `2` overwrites storage cell `1` — a gap cell
no logical element owns (sentinel `7` becomes `relu 5`). The same
stride-argument binding in the recognizer excludes it.

## The compilable kernel text and its cast

The pinned kernel text does not compile with Triton 3.8.0 (both branches):
`tl.make_block_ptr` returns a `_block_ptr` aggregate whose `type` has no
`element_ty`, so `out0.to(out0_bptr.type.element_ty)` raises during
AST→TTIR lowering. The one accepted derived text (`store_cast_fix`,
recognized by its own hash in `scripts/launch_invoke.py`; no other text is
accepted) changes only the store cast's destination to
`out0_ptr.type.element_ty` — the same fix the pinned source already applies to
its load, with a comment explaining why.

The model erasing casts is **not** the argument that this edit is harmless.
The argument is: within the checked domain (S7: both tensors `float32`), the
destination `out0_ptr.type.element_ty` is `fp32`; the value `out0 =
tl.where(in0 > 0, in0, 0)` is `fp32` (the load is cast to `in0_ptr`'s `fp32`
and the literal `0` is promoted); so the cast is `fp32 → fp32`, the identity.
The Triton 3.8.0 frontend output confirms it: the TTIR store path is
`tt.load` (f32) → `arith.cmpf ogt` → `arith.select` → `tt.store` (f32), with no
float conversion (only `arith.sitofp` of the literal `0`)
(`work/submissions/relu_compat/frontend_variant.log`). Outside S7 (e.g. a
different output dtype) the two casts could differ; such inputs are rejected.

## Numerical scope

The Lean contract is over exact reals: `xs k : ℝ`, `relu = max 0`. It says
nothing about NaN. Characterization on the Triton 3.8.0 CPU interpreter
(`work/submissions/relu_compat/characterize_values.json`, interpreter only,
not GPU): the kernel returns `+0.0` for NaN inputs (`arith.cmpf ogt` is false
for NaN) where `torch.relu` returns NaN; `-0.0`, `±inf`, denormals and
`±FLT_MAX` are bitwise equal to `torch.relu`. "Equals `torch.relu`" in the
evidence below means on finite test inputs; NaN is outside the contract.

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
- Lean flat-placement cases: the `StridedBuffer` regression layout (offset 5,
  stride 3) is accepted and its metadata placement is disjoint
  (`relu_offset5_stride3_flat_placement`); interleaved views of one storage
  are rejected by S9 (conservative).
- NVIDIA L4 via Modal, Triton 3.8.0 (`launch_evidence/gpu_wrapper.json`, one device):
  **the pinned kernel text does not compile** — `out0.to(out0_bptr.type.element_ty)`
  fails because a block-pointer type has no `element_ty` in Triton 3.8.0. With a
  labelled single-change variant (store cast to `out0_ptr.type.element_ty`;
  device guard kept), contiguous, input-strided, output-strided (gap cells
  intact) and `StridedBuffer` (offset 5, stride 3) invocations equal
  `torch.relu`, and the pinned wrapper function (same variant) agrees with the
  checked path; all rejection cases rejected before launch.

## Not covered / trusted

Grid-stride-loop branch; negative strides; dtype reinterpretation; empty
input; interleaved-but-disjoint views (S9 is conservative); NaN and any
bit-level floating-point claim; serial-order agreement in flat memory;
byte-level memory beyond the stated cell/byte correspondence (cells are typed
and element-sized);
instruction-level interleaving (TA-sched); the recognizer (a pinned-pattern
AST match, trusted and tested) and metadata extraction (trusted; a view
pointer outside its storage gets capacity 0 and is rejected by S6);
TA-transcription and TA-region.
