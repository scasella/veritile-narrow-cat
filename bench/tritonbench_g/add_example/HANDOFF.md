# add_example launch checker — external evidence handoff

Both gates below are **NOT RUN**. The development host is macOS arm64
(no Landlock/systemd; no Triton wheels; no NVIDIA/AMD GPU). Nothing in this
directory claims either result. `python3 scripts/launch_local_check.py
--require-official` / `--require-gpu` exit non-zero until a result file with
matching `input_hashes` is placed in `launch_evidence/`.

## A. Official comparator audit (Linux)

Requirements (upstream `scripts/README.md`): Linux with Landlock, a working
systemd user service, Go ≥ 1.24, Lean `v4.29.0` via elan. Comparator pin
`2a00b30df5e9173e70c4e4ec669fdf03da3163b9`, landrun pin
`5283024a2f49b28046c3b4a06d7d775c058d4d80` (installed by the upstream script).

```bash
git clone https://github.com/Lizn-zn/VeriTile && cd VeriTile
git fetch <this-branch-bundle> veritile-launch-checker && git checkout veritile-launch-checker
scripts/setup-comparator.sh /tmp/veritile-proof-tools
export PATH="/tmp/veritile-proof-tools/bin:$PATH"
lake exe cache get
lake build VeriTile
# the changed library modules (manifest `proven` rows are unchanged, so also run the file gate):
python3 scripts/check_comparator.py --library
python3 scripts/check_comparator.py --file bench/tritonbench_g/add_example/AddExample.lean --trust
python3 scripts/check_comparator.py --file bench/tests/Blocked1DLaunchWitnesses.lean --trust
```

Record the result (only if every command exited 0; otherwise record the
failing exit code) — hashes must come from the audited checkout:

```bash
python3 - <<'EOF'
import json, sys; sys.path.insert(0, "scripts")
import launch_local_check as LC
json.dump({"exit_code": 0, "commands": ["check_comparator.py --library",
  "check_comparator.py --file AddExample.lean --trust",
  "check_comparator.py --file Blocked1DLaunchWitnesses.lean --trust"],
  "comparator_log_dirs": ["Logs/comparator-check-..."],
  "input_hashes": LC.input_hashes()},
  open("bench/tritonbench_g/add_example/launch_evidence/official_comparator.json", "w"), indent=1)
EOF
python3 scripts/launch_local_check.py --require-official
```

Also run the unmodified upstream release gates; this project did not run or
alter them: `scripts/check-artifact.sh`, `bench/audit_tritonbench_g.sh`.

## B. GPU correctness and performance (Linux + NVIDIA CC ≥ 8.0 or AMD ROCm ≥ 6.2)

Kernel: `add_example.py::add_kernel` exactly as pinned (sha256 in
`launch_evidence/ledger.json`). Reference: `x + y` computed by torch on the
same device in float32. Relation: **bitwise equality** is expected on IEEE
fp32 hardware (one correctly rounded add per lane, *IEEE-single-add*); report
any mismatch with max-ULP distance rather than loosening the relation.
Record `triton.__version__`, `torch.__version__`, driver, GPU name, compiler
options (defaults; no `num_warps` override — the wrapper passes none).

Configurations: every `valid*` case of `launch_manifest.json` (n ∈ {0, 3, 4,
5, 8, 9, 16, 32}, BLOCK_SIZE = 4; over-provisioned grid (3,) for n = 5), plus
n = 2^20 + 3. Boundary checks: `out[n:]` of an over-allocated output buffer
must remain untouched (allocate `out` with 8 extra sentinel elements and
compare sentinels); n = 0 must not launch (grid (0,)).

```python
import json, time, torch, triton, importlib.util, sys
spec = importlib.util.spec_from_file_location("ae", "bench/tritonbench_g/add_example/add_example.py")
# NOTE: importing runs the file's CUDA test at module scope — acceptable on the GPU host.
ae = importlib.util.module_from_spec(spec); spec.loader.exec_module(ae)
dev = "cuda"
res = {"device": torch.cuda.get_device_name(), "triton": triton.__version__, "torch": torch.__version__, "cases": {}}
for n in [0, 3, 4, 5, 8, 9, 16, 32, (1 << 20) + 3]:
    x, y = torch.randn(n, device=dev), torch.randn(n, device=dev)
    buf = torch.full((n + 8,), 1234.5, device=dev); out = buf[:n]
    grid = ((n + 3) // 4,)
    ae.add_kernel[grid](x, y, out, n, 4); torch.cuda.synchronize()
    res["cases"][n] = {"bitwise_equal": bool(torch.equal(out, x + y)),
                       "sentinels_intact": bool((buf[n:] == 1234.5).all())}
# performance (separate from compilation): warmup, synchronize, repeat
n = 1 << 24; x, y = torch.randn(n, device=dev), torch.randn(n, device=dev); out = torch.empty_like(x)
t0 = time.perf_counter(); ae.add_kernel[((n + 3) // 4,)](x, y, out, n, 4); torch.cuda.synchronize()
res["first_call_incl_compile_s"] = time.perf_counter() - t0
ms = triton.testing.do_bench(lambda: ae.add_kernel[((n + 3) // 4,)](x, y, out, n, 4), warmup=25, rep=200)
res["steady_state_ms_median"] = ms
print(json.dumps(res, indent=1))
```

Write `launch_evidence/gpu.json` (`exit_code` 0 only if every case is
bitwise equal with intact sentinels) and `launch_evidence/gpu_perf.json`
(timing + hardware identification), each with `input_hashes` from
`launch_local_check.input_hashes()` on the tested checkout. A CPU interpreter
run (`TRITON_INTERPRET=1`) is interpreter evidence only and must not be filed
as `gpu.json`.

No performance claim is made for this change: the contribution adds host
checks and proofs, not a kernel transformation.
