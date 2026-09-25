#!/usr/bin/env python3
"""Probe wrapper obligation W1 (`n == out.numel()`) for the pinned
`vector_addition_custom.py::custom_add` under Triton's CPU interpreter
(TRITON_INTERPRET=1). INTERPRETER EVIDENCE ONLY.

`custom_add` launches with `size = c.size(0)`. For rank > 1 inputs this is the
leading dimension, not `numel`, so the static finding predicts that only the
first `size(0)` flat elements of the `empty_like` output are written. To make
unwritten cells observable, `torch.empty_like` is patched for the duration of
the probe to return a sentinel-filled tensor. The pinned kernel/wrapper text is
imported unmodified (text before the `#####` test banner).

Usage (Linux container with triton + torch):
    TRITON_INTERPRET=1 python3 scripts/launch_w1_probe.py --out w1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

assert os.environ.get("TRITON_INTERPRET") == "1", "set TRITON_INTERPRET=1"
import torch  # noqa: E402
import triton  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from launch_interpret import load_defs  # noqa: E402

SENTINEL = 1234.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    ns = load_defs(REPO / "bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py")
    real_empty_like = torch.empty_like
    ns["torch"].empty_like = lambda t, *k, **kw: real_empty_like(t, *k, **kw).fill_(SENTINEL)
    rows = []
    try:
        for shape in [(37,), (4, 8), (3, 5, 7), (1, 40)]:
            x, y = torch.randn(shape), torch.randn(shape)
            out = ns["custom_add"](x, y)
            flat_ok = (out.reshape(-1) == (x + y).reshape(-1))
            unwritten = int((out.reshape(-1) == SENTINEL).sum())
            rows.append({"shape": list(shape), "numel": x.numel(), "launched_size": shape[0],
                         "correct_elements": int(flat_ok.sum()), "unwritten_elements": unwritten,
                         "output_equals_x_plus_y": bool(flat_ok.all()),
                         "w1_holds": shape[0] == x.numel()})
    finally:
        ns["torch"].empty_like = real_empty_like
    # Finding confirmed iff: every W1-satisfying case is fully correct, and every W1-violating
    # case writes exactly `size(0)` elements and leaves the rest unwritten.
    confirmed = all(r["output_equals_x_plus_y"] if r["w1_holds"] else
                    (r["correct_elements"] == r["launched_size"]
                     and r["unwritten_elements"] == r["numel"] - r["launched_size"]) for r in rows)
    res = {"backend": "triton-interpreter (TRITON_INTERPRET=1, CPU)", "triton": triton.__version__,
           "torch": torch.__version__, "source": "pinned (verbatim); torch.empty_like patched to "
           "sentinel-fill so unwritten cells are observable", "cases": rows,
           "w1_finding_confirmed": confirmed}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(rows, indent=0), "\nconfirmed", confirmed)
    return 0 if confirmed else 1


if __name__ == "__main__":
    sys.exit(main())
