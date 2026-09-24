#!/usr/bin/env python3
"""Untrusted flat-memory emulation of the 1-D blocked masked add kernel.

TEST ORACLE ONLY — not a proof, not Triton, not a GPU. It re-states the
kernel's addressing (`data_ptr + offsets`, ignoring tensor strides, as the
Triton kernel does) over the real storage of CPU torch tensors, with i32
wrap-around for offsets and `n_elements`, so configuration faults rejected by
the Lean checker can be accompanied by a concrete wrong output, an
out-of-allocation access, or an order-dependent (racy) result.
"""
from __future__ import annotations

import torch


def _i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _storage_view(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Whole-storage 1-D view of `t` and the element index of `t[0]`."""
    full = torch.empty(0, dtype=t.dtype).set_(t.untyped_storage())
    return full, int(t.storage_offset())


def emulate_add(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor,
                n_elements: int, block: int, grid: int, order: str = "asc") -> dict:
    """Run programs in `order` ('asc' | 'desc'); returns events and final out."""
    xs, xo = _storage_view(x)
    ys, yo = _storage_view(y)
    os_, oo = _storage_view(out)
    n32 = _i32(n_elements)
    events: list[str] = []
    pids = range(grid) if order == "asc" else reversed(range(grid))
    for pid in pids:
        offs = [_i32(_i32(pid * block) + j) for j in range(block)]
        mask = [o < n32 for o in offs]
        vals = []
        for (buf, base, name) in ((xs, xo, "x"), (ys, yo, "y")):
            lane = []
            for o, m in zip(offs, mask):
                if not m:
                    lane.append(None)
                    continue
                idx = base + o
                if idx < 0 or idx >= buf.numel():
                    events.append(f"OOB read {name}[{o}] (pid {pid})")
                    lane.append(float("nan"))
                else:
                    lane.append(float(buf[idx]))
            vals.append(lane)
        for j, (o, m) in enumerate(zip(offs, mask)):
            if not m:
                continue
            idx = oo + o
            if idx < 0 or idx >= os_.numel():
                events.append(f"OOB write out[{o}] (pid {pid})")
                continue
            os_[idx] = vals[0][j] + vals[1][j]
    return {"events": events, "out": out.clone()}


def reference_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Independent reference: elementwise sum of the logical tensor values."""
    return x.reshape(-1) + y.reshape(-1)
