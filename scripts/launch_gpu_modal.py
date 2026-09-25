#!/usr/bin/env python3
"""Run `launch_gpu.run_all` on a Modal GPU (HANDOFF.md §B/§B2) and write the
evidence files. Uploads only UPLOAD (the files `launch_local_check.input_hashes`
names plus the harness and its imports) — no `.git`, `.lake`, or other files.

    modal run scripts/launch_gpu_modal.py            # §B/§B2 on L4 (CC 8.9); timeout 900 s
    modal run scripts/launch_gpu_modal.py --mode sweep   # BLOCK_SIZE sweep -> block_sweep.json
    python3 scripts/launch_gpu_modal.py --list       # print the upload set, no Modal call

Evidence (`launch_evidence/gpu.json`, `gpu_perf.json`, `gpu_extras.json`) is
written only if the remote device is CUDA with CC >= 8.0, `TRITON_INTERPRET` was
unset, and the remote `input_hashes` equal the local ones key-for-key.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

GPU = "L4"
REMOTE = "/repo"
HARNESS = ["bench/audit_source.py", "scripts/launch_interpret.py", "scripts/launch_gpu.py"]

if modal.is_local():
    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "scripts"))
    import launch_local_check as LC  # noqa: E402
    LOCAL_HASHES = LC.input_hashes(REPO)
    assert len(LOCAL_HASHES) == len(LC.input_files()), sorted(LOCAL_HASHES)  # all exist locally
    UPLOAD = sorted(set(LOCAL_HASHES) | set(HARNESS))
    image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")
    for f in UPLOAD:
        image = image.add_local_file(REPO / f, f"{REMOTE}/{f}")
else:
    image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")

app = modal.App("veritile-launch-gpu", image=image)


@app.function(gpu=GPU, timeout=900)
def run() -> dict:
    sys.path.insert(0, f"{REMOTE}/scripts")
    import launch_gpu
    return launch_gpu.run_all(Path(REMOTE), "cuda")


@app.function(gpu=GPU, timeout=900)
def sweep() -> dict:
    sys.path.insert(0, f"{REMOTE}/scripts")
    import launch_gpu
    return {"block_sweep": launch_gpu.block_sweep(Path(REMOTE), "cuda")}


@app.local_entrypoint()
def main(out_dir: str = "", mode: str = "handoff") -> None:
    res = {"handoff": run, "sweep": sweep}[mode].remote()
    evid = LC.EVID if not out_dir else Path(out_dir)
    raw = Path(out_dir or "/Users/scasella/Downloads/kernel-claude/work/logs") / f"gpu_raw_{mode}.json"
    raw.write_text(json.dumps(res, indent=1, default=str) + "\n")
    for key, rec in res.items():
        assert rec.get("input_hashes") == LOCAL_HASHES, f"{key}: remote hashes differ from local"
        assert rec.get("device") == "cuda" and not rec.get("TRITON_INTERPRET"), key
        assert tuple(rec.get("compute_capability", (0, 0))) >= (8, 0), key
    for key, rec in res.items():
        (evid / f"{key}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: [v.get("exit_code"), v.get("gpu_name"), v.get("kernel_source")]
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__" and "--list" in sys.argv:
    print("\n".join(UPLOAD))
