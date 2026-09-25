#!/usr/bin/env python3
"""Run the optimization experiment (`launch_bench.py`) on one Modal GPU and
write `launch_evidence/relu_tune.json` and `fusion_bench.json`.

    python3 scripts/launch_bench_modal.py --list   # print the upload set; no Modal call
    modal run scripts/launch_bench_modal.py        # one L4 call (needs an explicit spending allowance)

Uploads only UPLOAD: the files `launch_local_check.input_hashes` names (sources,
Lean files, harness scripts; no `.git`, `.lake`, credentials or other files).
The two parts run in isolation: a failure in one is recorded and does not lose
the other. Evidence is written only if the remote device is CUDA with CC >= 8.0,
`TRITON_INTERPRET` was unset, and the remote `input_hashes` equal the local ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

GPU = "L4"
REMOTE = "/repo"
LOG_DIR = Path("/Users/scasella/Downloads/kernel-claude/work/logs")


def _image():
    # torch.compile (inductor) needs a C compiler on the host side.
    return (modal.Image.debian_slim(python_version="3.12").apt_install("gcc", "g++")
            .pip_install("torch", "numpy"))


if modal.is_local():
    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "scripts"))
    import launch_local_check as LC  # noqa: E402
    LOCAL_HASHES = LC.input_hashes(REPO)
    assert len(LOCAL_HASHES) == len(LC.input_files()), sorted(set(LC.input_files()) - set(LOCAL_HASHES))
    UPLOAD = sorted(LOCAL_HASHES)
    image = _image()
    for f in UPLOAD:
        image = image.add_local_file(REPO / f, f"{REMOTE}/{f}")
else:
    image = _image()

app = modal.App("veritile-launch-bench", image=image)


@app.function(gpu=GPU, timeout=1800)
def bench() -> dict:
    import traceback
    sys.path.insert(0, f"{REMOTE}/scripts")
    import launch_bench as B
    out = {}
    try:
        out["relu_tune"] = B.relu_tune("cuda")
    except Exception:  # noqa: BLE001
        out["error_relu_tune"] = {"traceback": traceback.format_exc()[-4000:]}
    nw = out.get("relu_tune", {}).get("selected_num_warps")
    try:
        out["fusion_bench"] = B.fusion("cuda", nw)
    except Exception:  # noqa: BLE001
        out["error_fusion_bench"] = {"traceback": traceback.format_exc()[-4000:]}
    return out


@app.local_entrypoint()
def main() -> None:
    res = bench.remote()
    (LOG_DIR / "bench_raw.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    errors = {k: v for k, v in res.items() if k.startswith("error_")}
    if errors:
        print(json.dumps(errors, indent=1)[:6000])
    for key, rec in res.items():
        if key.startswith("error_"):
            continue
        assert rec.get("input_hashes") == LOCAL_HASHES, f"{key}: remote hashes differ from local"
        assert rec.get("device") == "cuda" and not rec.get("TRITON_INTERPRET"), key
        assert tuple(rec.get("compute_capability", (0, 0))) >= (8, 0), key
        (LC.EVID / f"{key}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: v.get("exit_code") for k, v in res.items() if not k.startswith("error_")}))


if __name__ == "__main__" and "--list" in sys.argv:
    print("\n".join(UPLOAD))
    print(f"# {len(UPLOAD)} files")
