#!/usr/bin/env python3
"""Checked invocation of the fused float32 add + ReLU candidate and of the
unfused pipeline it is compared with. New module; the frozen recognizer and
invocation harness (`launch_check.py`, `launch_invoke.py`) are used, not changed.

* `CheckedFusedAddRelu` — `improvement/add_relu_fused.py::add_relu_wrapper`,
  checked by the unchanged `Elementwise2` contract (the Python mirror in
  `launch_invoke.py`, differential-tested against Lean). Source binding:
  1. the kernel body must equal, statement for statement, its Lean
     transcription `add_relu_kernel` in `improvement/AddReluFused.lean` (the
     kernel `add_relu_wrapper_correctness` is about);
  2. argument roles (pointers, element count, block) and the wrapper launch are
     read by the frozen recognizer (`parse_kernel` / `parse_launch`) from a
     *projection* of the source in which the two fused lines
     `z = x + y` / `output = tl.where(z > 0, z, 0)` are collapsed to
     `output = x + y`. The projection is used for role extraction only; the
     computed value is fixed by step 1.
* `CheckedReluTuned` — the checked strided ReLU (`store_cast_fix` text) with a
  `num_warps` launch option. `num_warps` is a compile option: the recognizer
  ignores it and nothing in the Lean model depends on it.
* `CheckedAdd64` — checked `improvement/add_example_block64.py::add_wrapper`
  (proved by `add_kernel_launch_correctness`; block 64, `empty_like`).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "bench"))
import launch_check as LC  # noqa: E402
import launch_invoke as I  # noqa: E402

IMPROVEMENT = REPO / "bench/tritonbench_g/add_example/improvement"
FUSED_PY = IMPROVEMENT / "add_relu_fused.py"
FUSED_LEAN = IMPROVEMENT / "AddReluFused.lean"
BLOCK64_PY = IMPROVEMENT / "add_example_block64.py"
ADD_LEAN = REPO / "bench/tritonbench_g/add_example/AddExample.lean"
FUSED_LINES = ("    z = x + y\n    output = tl.where(z > 0, z, 0)\n", "    output = x + y\n")
# add_example_block64.py spells the annotation as a string; the CPU interpreter needs the class.
ADD_INTERP_VARIANT = {'BLOCK_SIZE: "tl.constexpr"': "BLOCK_SIZE: tl.constexpr"}


def fused_projection(src: str) -> str:
    if src.count(FUSED_LINES[0]) != 1:
        raise LC.Unsupported("fused ReLU lines not found exactly once")
    return src.replace(*FUSED_LINES)


def kernel_statements(src: str, name: str) -> list[str]:
    fn = LC.find_function(ast.parse(src), name)
    body = [st for st in fn.body if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))]
    return [LC._normalize(ast.unparse(st)) for st in body]


class CheckedFusedAddRelu:
    """`add_relu_wrapper` on actual tensors, checked by `Elementwise2` at its block size."""

    def __init__(self, src: str | None = None, variant: dict | None = None, num_warps: int | None = None):
        src = src if src is not None else FUSED_PY.read_text()
        if kernel_statements(src, "add_relu_kernel") != \
                LC.lean_body_statements(FUSED_LEAN.read_text(), "add_relu_kernel"):
            raise LC.Unsupported("fused kernel body differs from its Lean transcription")
        proj = fused_projection(src)
        k = LC.parse_kernel(proj, "add_relu_kernel")
        L = LC.parse_launch(proj, "add_relu_wrapper", k)
        x_var, y_var = L.wrapper_params[:2]
        out_var, like = L.out_alloc
        if L.grid_kind != "cdiv" or like != x_var or L.n_source != x_var:
            raise LC.Unsupported("launch is not the modelled cdiv grid over x.numel() with *_like(x)")
        if [L.arg_binding[p] for p in k.ptr_params] != [x_var, y_var] or L.arg_binding[k.out_param] != out_var:
            raise LC.Unsupported("kernel pointer arguments are not (x, y, out)")
        self.launch, self.block = L, L.block
        self.alloc = I._alloc_fn(src, "add_relu_wrapper", out_var)
        self.obligations, self.launch_of, self.lean_name = I.CONTRACTS["numel"]
        self._params = k.params
        self._roles = {k.ptr_params[0]: "x", k.ptr_params[1]: "y", k.out_param: "out",
                       k.n_param: "n", k.block_param: "B"}
        import launch_interpret as LI  # noqa: E402
        self.namespace = LI.load_defs(FUSED_PY, variant)
        self._kernel = self.namespace["add_relu_kernel"]
        self.num_warps = num_warps
        self.stats = {"calls": 0}

    def verdict(self, x, y, out) -> tuple:
        mx, my, mo = I.tensor_meta(x), I.tensor_meta(y), I.tensor_meta(out)
        obl = self.obligations(self.block, mx, my, mo)
        return [n for n, ok in obl if not ok], self.launch_of(self.block, mx, my, mo)

    def launch_only(self, x, y, out, cfg) -> None:
        vals = {"x": x, "y": y, "out": out, "n": cfg["n"], "B": cfg["block"]}
        kw = {"num_warps": self.num_warps} if self.num_warps else {}
        self._kernel[(cfg["grid"][0],)](*(vals[self._roles[p]] for p in self._params), **kw)

    def __call__(self, x, y):
        import torch
        self.stats["calls"] += 1
        out = getattr(torch, self.alloc)(x)
        failed, cfg = self.verdict(x, y, out)
        if failed:
            raise I.ContractViolation(failed)
        self.launch_only(x, y, out, cfg)
        return out


class CheckedReluTuned(I.CheckedStridedRelu):
    """The checked strided ReLU with an optional `num_warps` launch option."""

    def __init__(self, num_warps: int | None = None, interpreter: bool = False):
        super().__init__("store_cast_fix", interpreter=interpreter)
        self.num_warps = num_warps

    def launch_only(self, in0, out0, c) -> None:
        kw = {"num_warps": self.num_warps} if self.num_warps else {}
        self._kernel[(c["numCtas"], 1, 1)](
            in0, out0, c["inStride"], 0, c["outStride"], 0, c["s0"], c["numTasks"],
            tiles_per_cta=c["tilesPerCta"], tile_size0=c["tile"], one_tile_per_cta=True, **kw)

    def __call__(self, in0, out0, launch: bool = True):
        self.stats["calls"] += 1
        failed, c = self.verdict(in0, out0)
        if failed:
            raise I.ContractViolation(failed)
        if launch:
            self.launch_only(in0, out0, c)
        return out0


def checked_add64(interpreter: bool = False) -> I.CheckedWrapper:
    return I.CheckedWrapper(BLOCK64_PY, "add_wrapper", "add_kernel", "numel",
                            ADD_INTERP_VARIANT if interpreter else None, ADD_LEAN)


class UnfusedPipeline:
    """Checked add (block 64) into a temporary, then the checked ReLU into `out`."""

    def __init__(self, relu_num_warps: int | None = None, interpreter: bool = False):
        self.add = checked_add64(interpreter)
        self.relu = CheckedReluTuned(relu_num_warps, interpreter)

    def __call__(self, x, y):
        import torch
        tmp = self.add(x, y)
        out = torch.empty_like(tmp)
        self.relu(tmp.view(-1), out.view(-1))
        return out
