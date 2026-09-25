#!/usr/bin/env python3
"""Lower-overhead checked execution of the fused add + ReLU (stage 9).

* `CheckedFusedAddReluB` — the general checked API (`launch_fused.CheckedFusedAddRelu`)
  with an explicit block size. Safe for any block the contract accepts:
  `add_relu_wrapper_correctness_block` (AddReluFused.lean) is stated for every
  `B` with `Elementwise2.check B x y out = true`, and the mirror decides exactly
  that check with the block in use. Returns a fresh output, like the source wrapper.
* `PreparedAddRelu` / `prepare_add_relu` — a separately labelled prepared-buffer
  API (caller-owned output, overwritten per run).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_fused as F  # noqa: E402
import launch_invoke as I  # noqa: E402


class CheckedFusedAddReluB(F.CheckedFusedAddRelu):
    """The general checked API with an explicit block size."""

    def __init__(self, block: int | None = None, num_warps: int | None = None):
        super().__init__(num_warps=num_warps)
        if block is not None:
            self.block = block


# ---------------------------------------------------------------------------
# Prepared-buffer API (separate from the general checked API)
# ---------------------------------------------------------------------------

class PlanInvalidated(RuntimeError):
    """A prepared buffer's metadata changed since `prepare_add_relu`."""


def meta_snapshot(t) -> tuple:
    """Everything the contract and Triton's specialization read from a tensor:
    data pointer (alignment), shape, strides, storage offset, dtype, device, and
    the backing storage's address and size (catches `set_`/`resize_`)."""
    st = t.untyped_storage()
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.storage_offset(), t.dtype, t.device,
            st.data_ptr(), st.nbytes())


class PreparedAddRelu:
    """`out <- relu(x + y)` on three FIXED buffers.

    Observable behaviour differs from the general API: the caller supplies the
    output buffer, which is overwritten on every `run()`; no tensor is returned
    fresh. `prepare` decides the full `Elementwise2` contract once and compiles
    (without launching) the kernel for exactly these arguments. `run` then only
    revalidates the metadata snapshot of the three buffers — the contract's
    verdict is a function of metadata alone, and the correctness theorem
    quantifies over all input VALUES, so contents may change between runs — and
    launches the precompiled kernel (or replays a CUDA graph of that launch).
    Because the pointers and `n` are identical to the ones compiled for, Triton's
    argument specialization (pointer alignment, `n % 16`) is unchanged.
    Not thread-safe; one plan per buffer triple and stream.
    """

    def __init__(self, checked: "F.CheckedFusedAddRelu", x, y, out, graph: bool = False):
        import torch
        failed, cfg = checked.verdict(x, y, out)
        if failed:
            raise I.ContractViolation(failed)
        self.tensors = (x, y, out)
        self.snapshot = tuple(meta_snapshot(t) for t in self.tensors)
        vals = {"x": x, "y": y, "out": out, "n": cfg["n"], "B": cfg["block"]}
        self.args = tuple(vals[checked._roles[p]] for p in checked._params)
        self.grid = (cfg["grid"][0], 1, 1)
        kw = {"num_warps": checked.num_warps} if checked.num_warps else {}
        self.compiled = checked._kernel.warmup(*self.args, grid=self.grid, **kw)
        self._launch = self.compiled[self.grid]
        self.graph = None
        if graph:
            self._launch(*self.args)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._launch(*self.args)
            self.graph = g

    def revalidate(self) -> None:
        if tuple(meta_snapshot(t) for t in self.tensors) != self.snapshot:
            raise PlanInvalidated("prepared buffer metadata changed; prepare a new plan")

    def run(self) -> None:
        self.revalidate()
        if self.graph is not None:
            self.graph.replay()
        else:
            self._launch(*self.args)


def prepare_add_relu(x, y, out, block: int | None = None, graph: bool = False) -> PreparedAddRelu:
    return PreparedAddRelu(CheckedFusedAddReluB(block=block), x, y, out, graph=graph)
