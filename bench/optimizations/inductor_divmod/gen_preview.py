#!/usr/bin/env python3
"""Regenerate fixtures/pointwise_cat_preview.py without a GPU (development preview only).

Run inside the local `veritile-interp:1` image (torch 2.14.0+cpu, Triton 3.8.0):

    docker run --rm -v "$PWD":/w veritile-interp:1 python3 /w/bench/optimizations/inductor_divmod/gen_preview.py

It compiles the issue's program with `torch.compile(dynamic=True)` on the CPU Triton backend and saves the
kernel source Inductor emits. The later Triton compile fails because there is no CPU Triton backend; that is
expected, since only the source is kept.

Two settings differ from the GPU path:
- `force_pointwise_cat = True`. On CPU, Inductor otherwise lowers `cat` to per-input copy kernels (see
  `lowering.cat`). On CUDA it chose one pointwise kernel (stage 11 evidence: `triton_poi_fused_add_cat_0`).
- The backend hash is stubbed, because no Triton driver is active.

The GPU harness applies the rewrite to whatever the GPU compile actually emits, and records the before and
after source. This fixture is only used to test the transform.
"""
import os
import traceback
from pathlib import Path

import torch
import torch._dynamo as dynamo
import torch._inductor.codegen.triton as ct
import torch._inductor.config as ic
import torch.utils._triton as ut

ic.cpu_backend = "triton"
ic.force_pointwise_cat = True
ut.triton_hash_with_backend = lambda: "stub"
ct.triton_hash_with_backend = lambda: "stub"
SRCS = []
_orig = ct.TritonScheduling.define_kernel


def _record(self, src_code, node_schedule, kernel):
    SRCS.append(src_code)
    return _orig(self, src_code, node_schedule, kernel)


ct.TritonScheduling.define_kernel = _record


def nested_cat_add(q1, k1, v1a, v1b, q2, k2, v2a, v2b):
    v1 = v1a + v1b
    v2 = v2a + v2b
    return torch.cat([torch.cat([q1, k1, v1], -1), torch.cat([q2, k2, v2], -1)], -1)


if __name__ == "__main__":
    ins = [torch.randn(64, w).bfloat16() for w in (2048, 256, 256, 256, 2048, 256, 256, 256)]
    for t in ins:
        dynamo.mark_dynamic(t, 1)
    try:
        torch.compile(nested_cat_add, dynamic=True)(*ins)
    except Exception:  # noqa: BLE001 - expected: no CPU Triton backend to compile with
        traceback.print_exc(limit=1)
    assert len(SRCS) == 1, f"expected one pointwise kernel, got {len(SRCS)}"
    out = Path(os.path.dirname(os.path.abspath(__file__))) / "fixtures" / "pointwise_cat_preview.py"
    out.write_text(SRCS[0])
    print(f"wrote {out}")
