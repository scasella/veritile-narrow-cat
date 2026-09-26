#!/usr/bin/env python3
"""Stage 15 on Modal: one `modal run`, one L4 function, launch_patch_perf.py on the pinned nightly. NOT YET RUN.

    python3 scripts/launch_patch_perf_modal.py --list   # print the upload set; no Modal call
    modal run scripts/launch_patch_perf_modal.py        # needs an explicit spending allowance

Image: torch==2.15.0.dev20260926 (cu130; git 6aa9e2fc) with its pytorch-triton, plus `patch`, numpy and nvidia-ml-py.
The harness verifies the pin and the base-file SHA-256s and applies the patch itself.

Upload set: exactly the files `launch_local_check.input_hashes` names, plus the two stage-15 harness files (EXTRA).
It excludes .git, .lake, credentials and every other file. The record carries the remote hashes of both; evidence is
written to launch_evidence/patch_perf.json only if they equal the local ones. The raw result is always saved to
work/logs/stage15_patch_perf_raw.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import modal

GPU = "L4"
REMOTE = "/repo"
LOG_DIR = Path("/Users/scasella/Downloads/kernel-claude/work/logs")
EXTRA = ["scripts/launch_patch_perf.py", "scripts/launch_patch_perf_modal.py"]


def _img():
    return (modal.Image.debian_slim(python_version="3.12").apt_install("gcc", "g++", "patch")
            .pip_install("torch==2.15.0.dev20260926", pre=True,
                         index_url="https://download.pytorch.org/whl/nightly/cu130",
                         extra_index_url="https://pypi.org/simple")
            .pip_install("numpy", "nvidia-ml-py"))


def extra_hashes(root: Path) -> dict:
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in EXTRA}


if modal.is_local():
    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "scripts"))
    import launch_local_check as LC  # noqa: E402
    LOCAL_HASHES = LC.input_hashes(REPO)
    assert len(LOCAL_HASHES) == len(LC.input_files()), sorted(set(LC.input_files()) - set(LOCAL_HASHES))
    LOCAL_EXTRA = extra_hashes(REPO)
    UPLOAD = sorted(set(LOCAL_HASHES) | set(EXTRA))
    img = _img()
    for f in UPLOAD:
        img = img.add_local_file(REPO / f, f"{REMOTE}/{f}")
else:
    img = _img()

app = modal.App("veritile-stage15")


@app.function(gpu=GPU, timeout=3600, image=img)
def patch_perf() -> dict:
    import subprocess
    r = subprocess.run([sys.executable, f"{REMOTE}/scripts/launch_patch_perf.py"], capture_output=True, text=True,
                       timeout=3500, cwd="/tmp")
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
        return {"error_patch_perf": {"stderr": r.stderr[-4000:], "stdout": r.stdout[-2000:]}}
    rec = json.loads(lines[-1][7:])
    rec["extra_hashes"] = extra_hashes(Path(REMOTE))
    return {"patch_perf": rec}


@app.local_entrypoint()
def main() -> None:
    try:
        res = patch_perf.remote()
    except Exception as e:  # noqa: BLE001
        res = {"error_patch_perf": {"exception": repr(e)[:2000]}}
    (LOG_DIR / "stage15_patch_perf_raw.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    rec = res.get("patch_perf")
    if rec is None or "error" in rec:
        print("ERROR", json.dumps(res)[:3000])
        return
    if rec.get("input_hashes") != LOCAL_HASHES or rec.get("extra_hashes") != LOCAL_EXTRA:
        print("remote hashes differ from local; evidence not written")
    else:
        (LC.EVID / "patch_perf.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps(rec.get("analysis", {}), indent=1, default=str)[:5000])


if __name__ == "__main__" and "--list" in sys.argv:
    print("\n".join(UPLOAD))
    print(f"# {len(UPLOAD)} files")
