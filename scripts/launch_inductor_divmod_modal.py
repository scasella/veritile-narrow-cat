#!/usr/bin/env python3
"""Run stage 12 (`launch_inductor_divmod.py`) on one Modal L4 and write
`launch_evidence/inductor_divmod.json`. torch is pinned to 2.14.0 (Triton 3.8.0), the same stack as stage 11.

    python3 scripts/launch_inductor_divmod_modal.py --list   # print the upload set; no Modal call
    modal run scripts/launch_inductor_divmod_modal.py        # one L4 call (needs an explicit spending allowance)

The upload set is exactly the files `launch_local_check.input_hashes` names: sources, Lean files and harness
scripts. It excludes `.git`, `.lake`, credentials and every other file. Evidence is written only when:
- the remote device is CUDA with compute capability >= 8.0;
- `TRITON_INTERPRET` was unset;
- the remote `input_hashes` equal the local ones.
The raw result is always saved to work/logs/inductor_divmod_raw.json.
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
    return (modal.Image.debian_slim(python_version="3.12").apt_install("gcc", "g++")
            .pip_install("torch==2.14.0", "numpy"))


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

app = modal.App("veritile-inductor-divmod", image=image)


@app.function(gpu=GPU, timeout=3000)
def bench() -> dict:
    import traceback
    sys.path.insert(0, f"{REMOTE}/scripts")
    try:
        import launch_inductor_divmod
        return {"inductor_divmod": launch_inductor_divmod.bench()}
    except Exception:  # noqa: BLE001
        return {"error_inductor_divmod": {"traceback": traceback.format_exc()[-4000:]}}


@app.local_entrypoint()
def main() -> None:
    res = bench.remote()
    (LOG_DIR / "inductor_divmod_raw.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    if "error_inductor_divmod" in res:
        print(json.dumps(res, indent=1)[:6000])
        return
    rec = res["inductor_divmod"]
    assert rec.get("input_hashes") == LOCAL_HASHES, "remote hashes differ from local"
    assert rec.get("device") == "cuda" and not rec.get("TRITON_INTERPRET")
    assert tuple(rec.get("compute_capability", (0, 0))) >= (8, 0)
    (LC.EVID / "inductor_divmod.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps(rec["decision"], indent=1, default=str)[:4000])


if __name__ == "__main__" and "--list" in sys.argv:
    print("\n".join(UPLOAD))
    print(f"# {len(UPLOAD)} files")
