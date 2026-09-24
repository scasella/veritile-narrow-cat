# add_example: source ↔ proof link

Machine-readable counterpart: `launch_manifest.json` (cases) and
`launch_evidence/ledger.json` (hashes, statuses). Correspondence level:
**tested-but-trusted structural translation**, not a proved translation.

## Artifacts

| Role | Path | Identity |
|---|---|---|
| Original kernel + host wrapper | `add_example.py` (TritonBench-G v1, via VeriTile@95a01f5, MIT) | sha256 in ledger `input_hashes` |
| Helper dependencies | none: the kernel calls only `tl.program_id`, `tl.arange`, `tl.load`, `tl.store`; the wrapper only `torch.zeros_like`, `x.numel()` | — |
| Formal kernel | `AddExample.lean` · `add_kernel` (`triton { }` DSL transcription, upstream) | `#print add_kernel` in frozen surface |
| Per-program theorem | `add_kernel_correctness : addIO ⊨ xs + ys` (upstream) | frozen surface |
| Launch checker | `VeriTile/Triton/Launch/Blocked1DConfig.lean` · `Blocked1DLaunch.check`, `check_ok`, `check_complete`, `Pre.i32_*` | frozen surface |
| Grid composition | `VeriTile/Triton/Launch/Blocked1D.lean` · `launch_of_frames`; `Launch/Composition.lean` · `LaunchCorrectFramed` | frozen surface |
| Whole-launch theorem | `AddExample.lean` · `add_kernel_launch_correctness` (+ `_applicable`, `_traceSafe`) | frozen surface |
| Host adapter | `scripts/launch_check.py` (+ tests `scripts/test_launch_check.py`) | sha256 in ledger |
| Reference | `xs i + ys i` over ℝ in the headline; `launch_emulate.reference_add` (test oracle) | — |

## What establishes correspondence

1. **Kernel body ↔ Lean transcription** — `launch_check.parse_kernel` walks
   the Python AST (never imported/executed) and accepts only the modelled
   grammar; `lean_body_statements` extracts the Lean `triton { }` body with the
   upstream helper `bench/audit_source.lean_first_triton_body`, unwraps `$(x)`
   antiquotations, and the normalized statement lists must be identical.
   Trust: the recognizer, the normalizer, and the upstream `triton { }` macro
   (DSL → AST) are trusted. Evidence: 20 unit tests incl. drift and rejection
   cases; statement match recorded per run.
2. **Wrapper ↔ `Blocked1DLaunch`** — `parse_launch` binds kernel parameters to
   wrapper expressions: `BLOCK_SIZE` must be an integer literal, `n_elements`
   `x.numel()` (or `c.size(0)`), the grid a rank-1 tuple of `cdiv`
   (`(n+B-1)//B`, `triton.cdiv`), floor-div, or literal; the output a fresh
   `torch.zeros_like/empty_like`. Launch keywords (`num_warps`, …), other grid
   ranks, extra statements are **rejected**, not dropped.
3. **Tensor metadata ↔ `BufMeta`** — from real CPU `torch.Tensor`s created by
   the adapter (`data_ptr`, `element_size`, `stride`, `storage_offset`,
   `untyped_storage().nbytes()`), or supplied metadata (only for sizes that
   cannot be allocated here, e.g. `n = 2^31-1`, labelled
   `supplied-metadata (trusted as supplied)`).
4. **Verdict** — Lean evaluates `check`/`failures` (diagnostics only), then
   the Lean kernel re-establishes each verdict with `decide` and produces
   `Pre c` (via `check_ok`) or `¬ Pre c` (via `check_complete`); axioms of each
   certificate are checked. A disagreement between passes is an
   infrastructure failure.

Hashes establish identity only; they do not establish semantic equivalence.

## Translation assumptions

| ID | Assumption | Evidence |
|---|---|---|
| TA-dsl | The upstream `triton { }` macro gives the Lean AST the meaning the Python statements have in Triton (at VeriTile's semantic abstraction). | Upstream design; not re-verified here. |
| TA-i32 | Triton types `pid`, `pid*BLOCK_SIZE`, `arange`, `n_elements: tl.int32` and the mask compare as two's-complement i32. | TRITON_FACTS §1–3 (source-cited); the agreement *under* P5–P7 is proved (`Pre.i32_offset_toInt`, `Pre.i32_mask_eq`). |
| TA-region | Disjoint byte ranges ↔ distinct VeriTile regions; unit-stride element `i` ↔ cell `(region, i)`. | Modelling convention (upstream `FlatAlloc`); P8/P10 checked. |
| TA-compose | Programs run as a disjoint-frame merge from the initial state (no program reads another's writes). | Input regions are never written (P10 + the kernel stores only to `out_ptr`). |
| TA-meta | Supplied metadata describes the real tensors. | Real tensors for all allocatable cases. |
| TA-empty | Grid `(0,)` launches nothing on NVIDIA/AMD. | TRITON_FACTS §4. |
| IEEE-single-add | Hardware lane result = correctly rounded fp32 sum. | External; not proved. |

## Supported / rejected source constructs

Supported: `tl.program_id(axis=0)`, `pid * BLOCK`, `+ tl.arange(0, BLOCK)`,
`offsets < n` masks, `tl.load(p + offsets, mask=m)`, `+` on loaded values,
exactly one `tl.store(out + offsets, v, mask=m)`, docstrings.
Rejected (unsupported): other axes, `other=`, extra load/store kwargs,
comparisons other than `<`, any other `tl.*`/helper call, casts, control
flow, in-place stores, multiple stores, launch options, grids of rank ≠ 1,
non-literal block sizes, non-fresh outputs.

## Reproduction

    python3 scripts/launch_check.py bench/tritonbench_g/add_example/launch_manifest.json
    python3 scripts/launch_local_check.py            # all local gates, prints LOCAL_CHECKS_PASSED

A change to `add_example.py`, `AddExample.lean`, the checker, the adapter,
the manifest, `CONTRACT.md`, the toolchain pin or `lake-manifest.json`
changes the ledger `input_hashes`; external evidence carrying other hashes is
reported `STALE` and fails any `--require-*` invocation. Changes to protected
definitions or headline statements fail the frozen-contract step.
