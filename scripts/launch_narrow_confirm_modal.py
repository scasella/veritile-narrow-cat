#!/usr/bin/env python3
"""Stage 14 on Modal: one `modal run`, two L4 functions.

  confirm      launch_narrow_confirm.py: N1 confirmation, pre-registered, on torch 2.14.0 / Triton 3.8.0
               -> launch_evidence/narrow_confirm.json
  patch_check  launch_patch_check.py: the upstream patch on the pinned nightly torch 2.15.0.dev20260926 (cu130)
               -> launch_evidence/patch_check.json (correctness only)

    python3 scripts/launch_narrow_confirm_modal.py --list   # print the upload set; no Modal call
    modal run scripts/launch_narrow_confirm_modal.py        # needs an explicit spending allowance

The upload set is exactly the files `launch_local_check.input_hashes` names. It excludes .git, .lake, credentials
and every other file. Evidence is written only if the remote input hashes equal the local ones. Raw results are
always saved to work/logs/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

GPU = "L4"
REMOTE = "/repo"
LOG_DIR = Path("/Users/scasella/Downloads/kernel-claude/work/logs")


def _img214():
    return (modal.Image.debian_slim(python_version="3.12").apt_install("gcc", "g++")
            .pip_install("torch==2.14.0", "numpy", "nvidia-ml-py"))


def _imgnightly():
    return (modal.Image.debian_slim(python_version="3.12").apt_install("gcc", "g++", "patch")
            .pip_install("torch==2.15.0.dev20260926", pre=True,
                         index_url="https://download.pytorch.org/whl/nightly/cu130",
                         extra_index_url="https://pypi.org/simple")
            .pip_install("numpy", "expecttest"))


if modal.is_local():
    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "scripts"))
    import launch_local_check as LC  # noqa: E402
    LOCAL_HASHES = LC.input_hashes(REPO)
    assert len(LOCAL_HASHES) == len(LC.input_files()), sorted(set(LC.input_files()) - set(LOCAL_HASHES))
    UPLOAD = sorted(LOCAL_HASHES)
    img214, imgn = _img214(), _imgnightly()
    for f in UPLOAD:
        img214 = img214.add_local_file(REPO / f, f"{REMOTE}/{f}")
        imgn = imgn.add_local_file(REPO / f, f"{REMOTE}/{f}")
else:
    img214, imgn = _img214(), _imgnightly()

app = modal.App("veritile-stage14")


@app.function(gpu=GPU, timeout=3600, image=img214)
def confirm() -> dict:
    import traceback
    sys.path.insert(0, f"{REMOTE}/scripts")
    try:
        import launch_narrow_confirm
        return {"narrow_confirm": launch_narrow_confirm.bench()}
    except Exception:  # noqa: BLE001
        return {"error_narrow_confirm": {"traceback": traceback.format_exc()[-4000:]}}


@app.function(gpu=GPU, timeout=2400, image=imgn)
def patch_check() -> dict:
    import subprocess
    r = subprocess.run([sys.executable, f"{REMOTE}/scripts/launch_patch_check.py"], capture_output=True, text=True,
                       timeout=2300)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
        return {"error_patch_check": {"stderr": r.stderr[-4000:], "stdout": r.stdout[-2000:]}}
    rec = json.loads(lines[-1][7:])
    sys.path.insert(0, f"{REMOTE}/scripts")
    import launch_local_check as LCr
    rec["input_hashes"] = LCr.input_hashes()
    return {"patch_check": rec}


@app.local_entrypoint()
def main() -> None:
    res = {}
    for f in (confirm, patch_check):
        try:
            res.update(f.remote())
        except Exception as e:  # noqa: BLE001 - one part failing must not lose the other
            res[f"error_{f.__name__}"] = {"exception": repr(e)[:2000]}
    (LOG_DIR / "stage14_raw.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    for key in ("narrow_confirm", "patch_check"):
        rec = res.get(key)
        if rec is None:
            print(key, "ERROR", json.dumps(res.get(f"error_{key}"))[:2000])
            continue
        if rec.get("input_hashes") != LOCAL_HASHES:
            print(key, "remote hashes differ from local; evidence not written")
            continue
        (LC.EVID / f"{key}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    if "narrow_confirm" in res:
        print(json.dumps(res["narrow_confirm"].get("analysis", {}), indent=1, default=str)[:4000])
    if "patch_check" in res:
        pc = res["patch_check"]
        print(json.dumps({k: pc.get(k) for k in ("git_version", "pin_matches", "base_matches", "applied",
                                                  "upstream_tests_rc")}, indent=1))


if __name__ == "__main__" and "--list" in sys.argv:
    print("\n".join(UPLOAD))
    print(f"# {len(UPLOAD)} files")
