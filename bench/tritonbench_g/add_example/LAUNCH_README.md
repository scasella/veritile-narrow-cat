# add_example — checked host launch (local reproduction)

Adds a proved launch-configuration checker and a whole-grid theorem for
`add_example.py::add_kernel`. Contract: `CONTRACT.md`. Source link and trust
boundary: `SOURCE_LINK.md`. External (Linux comparator, GPU) gates:
`HANDOFF.md` — both **not run**.

## Prerequisites
- elan; the repo's `lean-toolchain` (`leanprover/lean4:v4.29.0`) is fetched automatically.
- `lake exe cache get` (pinned Mathlib `8a17838…` oleans).
- Python ≥ 3.11 with `torch` (CPU is enough) for real tensor metadata.
- ~6 GB free RAM; builds are run with `LEAN_NUM_THREADS=2`.

## Commands

```bash
lake exe cache get
LEAN_NUM_THREADS=2 lake build VeriTile.Triton VeriTile.Meta.StatementAudit VeriTile.Examples.Common
python3 scripts/launch_local_check.py            # all local gates; prints LOCAL_CHECKS_PASSED
python3 scripts/launch_local_check.py --fresh    # same, project modules rebuilt from source in a temp workspace
python3 scripts/launch_mutation_suite.py         # adversarial suite -> launch_evidence/mutation_results.json
python3 scripts/launch_check.py bench/tritonbench_g/add_example/launch_manifest.json   # adapter only
```

`--require-official` / `--require-gpu` additionally demand fresh external
evidence and exit non-zero (2/3) while it is missing or stale.

## Outputs
- `launch_evidence/ledger.json` — per-gate status/scope/assumptions/evidence and input hashes
  (model proof, checker soundness, source correspondence, local trust checks, frozen contract,
  official comparator, GPU correctness, GPU performance).
- `launch_evidence/adapter_results.json` — each configuration, its metadata provenance and
  kernel-checked verdict with failed obligations.
- `launch_evidence/mutation_results.json` — outcome class per adversarial case.
- `launch_evidence/frozen/` — frozen protected surface (elaborated statements/definitions).
