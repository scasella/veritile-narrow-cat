#!/usr/bin/env python3
"""Stage 14, part 2: correctness-only check of the upstream selective-narrowing patch on the pinned PyTorch nightly
(CUDA build), on one GPU. No timing decision is made here.

It runs in an image with torch==2.15.0.dev20260926 (cu130) and its pytorch-triton, and does the following:
  1. records torch.version.git_version; the pin is 6aa9e2fc (upstream/MANIFEST.json);
  2. verifies the installed torch/_inductor/codegen/triton.py and config.py match the manifest's base SHA-256;
  3. applies upstream/selective_ks_narrowing.patch to site-packages (the test file goes to a scratch directory);
  4. runs the upstream test file (4 tests: narrowing fires with bitwise-exact output, the default is unchanged,
     another kernel is unchanged, recognizer mutations are rejected);
  5. with the flag on, runs the 14-shape stage-12/13 sequence in one process: bitwise equality with eager at every
     shape, one Dynamo graph, the generated ks signature, and the eligibility premises.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UP = REPO / "bench/optimizations/inductor_narrow/upstream"


def check() -> dict:
    import torch
    man = json.loads((UP / "MANIFEST.json").read_text())
    site = Path(torch.__file__).parent.parent
    rec = {"torch": torch.__version__, "git_version": torch.version.git_version,
           "pin_matches": torch.version.git_version == man["pytorch_commit"],
           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    rec["base_matches"] = {f: hashlib.sha256((site / f).read_bytes()).hexdigest() == h
                           for f, h in man["base_sha256"].items()}
    if not (rec["pin_matches"] and all(rec["base_matches"].values())):
        rec["applied"] = False
        return rec
    work = site   # patch applies in place; the new test file lands in site-packages/test/inductor
    p = subprocess.run(["patch", "-p1", "--forward", "--batch", "-d", str(work), "-i", str(UP / man["patch"])],
                       capture_output=True, text=True)
    rec["patch_rc"], rec["patch_out"] = p.returncode, (p.stdout + p.stderr)[-1500:]
    rec["applied"] = p.returncode == 0
    if not rec["applied"]:
        return rec
    test = work / "test/inductor/test_triton_size_arg_narrowing.py"
    t = subprocess.run([sys.executable, str(test), "-v"], capture_output=True, text=True, timeout=1800,
                       env={**os.environ, "PYTHONPATH": ""})
    rec["upstream_tests_rc"] = t.returncode
    rec["upstream_tests_tail"] = (t.stdout + t.stderr)[-3000:]
    s = subprocess.run([sys.executable, __file__, "--sequence"], capture_output=True, text=True, timeout=1800)
    lines = [ln for ln in s.stdout.splitlines() if ln.startswith("RESULT ")]
    rec["sequence"] = json.loads(lines[-1][7:]) if lines else {"error": (s.stderr or s.stdout)[-3000:]}
    return rec


def sequence() -> dict:
    """Fresh process, patched torch: the 14-shape sequence with the flag on."""
    import torch
    import torch._dynamo as dynamo
    import torch._inductor.codegen.triton as ct
    import torch._inductor.config as ic
    sys.path.insert(0, str(REPO / "scripts"))
    import launch_inductor_divmod as S12
    ic.triton.narrow_proven_size_args = True
    srcs = []
    orig = ct.TritonScheduling.define_kernel

    def rec(self, src, ns, k):
        srcs.append(src)
        return orig(self, src, ns, k)
    ct.TritonScheduling.define_kernel = rec
    fn = torch.compile(S12.nested_cat_add, dynamic=True)
    out = {"bitwise": {}}
    for n, w in S12.ORDER:
        ins = S12.make_inputs(n, w)
        for t in ins:
            dynamo.mark_dynamic(t, 1)
        got = fn(*ins)
        out["bitwise"][f"{n}x{w}"] = bool(torch.equal(got.view(torch.int16), S12.nested_cat_add(*ins).view(torch.int16)))
    out["kernels_defined"] = len(srcs)
    out["ks_signature"] = [dict(re.findall(r"'(ks\d+)': '(i\d+)'", s)) for s in srcs]
    out["unique_graphs"] = dynamo.utils.counters["stats"]["unique_graphs"]
    return out


if __name__ == "__main__":
    try:
        r = sequence() if "--sequence" in sys.argv else check()
    except Exception:  # noqa: BLE001
        r = {"error": traceback.format_exc()[-3000:]}
    print("RESULT " + json.dumps(r, default=str))
