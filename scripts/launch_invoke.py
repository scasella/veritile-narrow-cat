#!/usr/bin/env python3
"""Checked invocation of a two-input elementwise wrapper on the caller's actual
tensors.

`CheckedWrapper(pyfile, wrapper, kernel, contract)` binds to the pinned source:
the AST recognizer of `launch_check.py` reads the wrapper's launch (block
constant, element-count rule, grid formula, output allocation, kernel argument
binding) and refuses sources it cannot map to the Lean contract. A call then

1. allocates the output exactly as the wrapper does (`zeros_like`/`empty_like`),
2. reads `TensorMeta` from the live tensors (`tensor_meta`: data pointer, shape,
   strides, capacity, dtype — **trusted** extraction, not verified),
3. decides the wrapper contract with `ew2_obligations` / `rank1_obligations`,
   a Python transliteration of the Lean `Elementwise2.check` /
   `Elementwise2.checkRank1` (**trusted by differential testing** against the
   Lean definitions: `--differential`),
4. on acceptance launches the kernel with the checked configuration (the same
   `n`, `BLOCK` and grid the model uses); on rejection raises
   `ContractViolation` naming the failed obligations.

Verdicts are cached under the complete metadata tuple (every field of every
tensor, including data pointers, plus the block and the contract), never
under shapes alone: aliasing and alignment depend on addresses.

    python3 scripts/launch_invoke.py --differential [--cases N]   # Python mirror vs Lean
    python3 scripts/launch_invoke.py --overhead                   # host-side cost (CPU tensors)
    TRITON_INTERPRET=1 python3 scripts/launch_invoke.py --demo    # interpreter run (Linux)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from dataclasses import astuple, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "bench"))
import launch_check as LC  # noqa: E402

I32 = 2 ** 31
DTYPE_NAMES = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}


@dataclass(frozen=True)
class TensorMeta:
    """Mirror of the Lean `TensorMeta`."""
    base: int
    elemBytes: int
    shape: tuple
    strides: tuple
    capacity: int
    dtype: str  # "f32" | "f16" | "bf16" | "other"


def tensor_meta(t) -> TensorMeta:
    """Metadata of a live torch tensor (trusted extraction)."""
    es = t.element_size()
    cap = t.untyped_storage().nbytes() // es - int(t.storage_offset())
    return TensorMeta(int(t.data_ptr()), es, tuple(int(d) for d in t.shape),
                      tuple(int(s) for s in t.stride()), cap,
                      DTYPE_NAMES.get(str(t.dtype).replace("torch.", ""), "other"))


# ---------------------------------------------------------------------------
# Transliteration of the Lean checker (Blocked1DConfig + Blocked1DWrapper).
# Nat semantics: truncated subtraction, `a / 0 = 0`, `a % 0 = a`.
# ---------------------------------------------------------------------------

def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def _div(a: int, b: int) -> int:
    return a // b if b else 0


def _mod(a: int, b: int) -> int:
    return a % b if b else a


def numel(t: TensorMeta) -> int:
    return _prod(t.shape)


def row_major(shape) -> list:
    return [_prod(shape[i + 1:]) for i in range(len(shape))]


def contiguous(t: TensorMeta) -> bool:
    rm = row_major(t.shape)
    return len(t.strides) == len(t.shape) and all(
        not (1 < t.shape[i]) or t.strides[i] == rm[i] for i in range(len(t.shape)))


def to_buf(t: TensorMeta) -> dict:
    return {"base": t.base, "elemBytes": t.elemBytes, "stride": 1 if contiguous(t) else 0,
            "capacity": t.capacity, "dtype": t.dtype}


def disjoint(n: int, a: dict, b: dict) -> bool:
    return (n * a["elemBytes"] == 0 or n * b["elemBytes"] == 0
            or a["base"] + n * a["elemBytes"] <= b["base"]
            or b["base"] + n * b["elemBytes"] <= a["base"])


def launch_obligations(c: dict) -> list:
    """`Blocked1DLaunch.check`, obligation by obligation (P1–P10)."""
    g = c["grid"][0] if c["grid"] else 0
    n, B = c["n"], c["block"]
    bufs = [c["output"], *c["inputs"]]
    return [("P1 grid_rank", len(c["grid"]) == 1),
            ("P2 block_ok", any(B == 2 ** k for k in range(21))),
            ("P3 covers", n <= g * B),
            ("P4 lanes_in_bounds", all(min(n, g * B) <= b["capacity"] for b in bufs)),
            ("P5 offsets_fit", g * B <= I32),
            ("P6 n_fits", n < I32),
            ("P7 grid_fits", g < I32),
            ("P8 unit_stride", all(n <= 1 or b["stride"] == 1 for b in bufs)),
            ("P9 dtype_ok", all(b["dtype"] == "f32" and b["elemBytes"] == 4 for b in bufs)),
            ("P10 out_disjoint", all(disjoint(n, c["output"], b) for b in c["inputs"]))]


def _launch(n: int, B: int, x, y, out) -> dict:
    return {"n": n, "block": B, "grid": [_div(max(n + B - 1, 0), B)],
            "inputs": [to_buf(x), to_buf(y)], "output": to_buf(out)}


def ew2_launch(B, x, y, out) -> dict:
    """`Elementwise2.launch`: n = x.numel()."""
    return _launch(numel(x), B, x, y, out)


def dim0_launch(B, x, y, out) -> dict:
    """`Elementwise2.launchDim0`: n = out.shape[0]."""
    return _launch(out.shape[0] if out.shape else 0, B, x, y, out)


def ew2_obligations(B, x, y, out) -> list:
    """`Elementwise2.check`, obligation by obligation."""
    return [("WA same_shape", y.shape == x.shape and out.shape == x.shape),
            ("WC contiguous", contiguous(x) and contiguous(y) and contiguous(out)),
            ("P12 aligned", all(_mod(t.base, t.elemBytes) == 0 for t in (x, y, out))),
            ("P11 inputs_separate", x.base == y.base or disjoint(numel(x), to_buf(x), to_buf(y))),
            *launch_obligations(ew2_launch(B, x, y, out))]


def rank1_obligations(B, x, y, out) -> list:
    """`Elementwise2.checkRank1`."""
    return [("WR rank1", len(x.shape) == 1 and len(y.shape) == 1 and len(out.shape) == 1),
            *ew2_obligations(B, x, y, out)]


CONTRACTS = {"numel": (ew2_obligations, ew2_launch, "Elementwise2.check"),
             "dim0_rank1": (rank1_obligations, dim0_launch, "Elementwise2.checkRank1")}


# ---------------------------------------------------------------------------
# Source-bound checked wrapper
# ---------------------------------------------------------------------------

class ContractViolation(Exception):
    def __init__(self, failures):
        super().__init__("wrapper contract violated: " + ", ".join(failures))
        self.failures = failures


def _alloc_fn(src: str, wrapper: str, out_var: str) -> str:
    fn = LC.find_function(ast.parse(src), wrapper)
    for st in fn.body:
        if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name) \
                and st.targets[0].id == out_var:
            return st.value.func.attr
    raise LC.Unsupported("output allocation not found")


class CheckedWrapper:
    """A wrapper bound to its pinned source and checked on every call."""

    def __init__(self, pyfile: Path, wrapper: str, kernel: str, contract: str,
                 variant: dict | None = None):
        src = pyfile.read_text()
        self.kernel_summary = LC.parse_kernel(src, kernel)
        self.launch = LC.parse_launch(src, wrapper, self.kernel_summary)
        L = self.launch
        x_var, y_var = L.wrapper_params[:2]
        out_var, like = L.out_alloc
        # Source correspondence with the Lean model of the chosen contract.
        if L.grid_kind != "cdiv" or like != x_var:
            raise LC.Unsupported("launch is not the modelled cdiv grid over *_like(x)")
        expect_n = {"numel": x_var, "dim0_rank1": f"{out_var}#dim0"}[contract]
        if L.n_source != expect_n:
            raise LC.Unsupported(f"element count {L.n_source!r} does not match contract {contract!r}")
        k = self.kernel_summary
        if [L.arg_binding[p] for p in k.ptr_params] != [x_var, y_var] or \
                L.arg_binding[k.out_param] != out_var:
            raise LC.Unsupported("kernel pointer arguments are not (x, y, out)")
        self.alloc = _alloc_fn(src, wrapper, out_var)
        self.block, self.contract = L.block, contract
        self.obligations, self.launch_of, self.lean_name = CONTRACTS[contract]
        self._params = k.params
        self._roles = {k.ptr_params[0]: "x", k.ptr_params[1]: "y", k.out_param: "out",
                       k.n_param: "n", k.block_param: "B"}
        import launch_interpret as LI  # noqa: E402  (text before the test banner)
        self._kernel = LI.load_defs(pyfile, variant)[kernel]
        self.cache: dict = {}
        self.stats = {"calls": 0, "cache_hits": 0}

    def verdict(self, x, y, out) -> tuple:
        mx, my, mo = tensor_meta(x), tensor_meta(y), tensor_meta(out)
        key = (self.contract, self.block, astuple(mx), astuple(my), astuple(mo))
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return hit
        obl = self.obligations(self.block, mx, my, mo)
        failed = [name for name, ok in obl if not ok]
        cfg = self.launch_of(self.block, mx, my, mo)
        self.cache[key] = (failed, cfg)
        return failed, cfg

    def __call__(self, x, y):
        import torch
        self.stats["calls"] += 1
        out = getattr(torch, self.alloc)(x)
        failed, cfg = self.verdict(x, y, out)
        if failed:
            raise ContractViolation(failed)
        vals = {"x": x, "y": y, "out": out, "n": cfg["n"], "B": cfg["block"]}
        self._kernel[(cfg["grid"][0],)](*(vals[self._roles[p]] for p in self._params))
        return out


def add_example(variant=None) -> CheckedWrapper:
    return CheckedWrapper(REPO / "bench/tritonbench_g/add_example/add_example.py",
                          "add_wrapper", "add_kernel", "numel", variant)


def custom_add(variant=None) -> CheckedWrapper:
    return CheckedWrapper(REPO / "bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py",
                          "custom_add", "_add_kernel", "dim0_rank1", variant)


# ---------------------------------------------------------------------------
# Strided unary wrapper (relu_strided_buffer, one-tile branch)
# ---------------------------------------------------------------------------

I64 = 2 ** 63


def storage_meta(t) -> TensorMeta:
    """Metadata of a torch tensor or a `StridedBuffer` (trusted extraction):
    capacity counts the elements from the view's data pointer to the end of
    the underlying storage, and is 0 when the pointer lies outside the storage
    (e.g. a `StridedBuffer` with a negative offset), so S6 rejects it.
    Negative strides are reported as-is and rejected."""
    base_t = t.unwrap() if hasattr(t, "unwrap") else t
    st = base_t.untyped_storage()
    es = t.element_size()
    ptr, lo, hi = int(t.data_ptr()), st.data_ptr(), st.data_ptr() + st.nbytes()
    cap = (hi - ptr) // es if lo <= ptr <= hi else 0
    dt = DTYPE_NAMES.get(str(t.dtype).replace("torch.", ""), "other")
    if hasattr(t, "unwrap") and t.dtype != base_t.dtype:
        dt = "other"  # dtype reinterpretation of the base storage is unsupported
    return TensorMeta(int(t.data_ptr()), es, tuple(int(d) for d in t.shape),
                      tuple(int(s) for s in t.stride()), max(cap, 0), dt)


def next_pow2(n: int) -> int:
    """Mirror of `StridedUnary.nextPow2`."""
    if n <= 1:
        return n
    p = 1
    for _ in range(64):
        if n <= p:
            return p
        p *= 2
    return p


def strided_launch(x: TensorMeta, out: TensorMeta) -> dict:
    """Mirror of `StridedUnary.launch`."""
    s0 = out.shape[0] if out.shape else 0
    tile = min(512, next_pow2(s0))
    cd = lambda a, b: _div(max(a + b - 1, 0), b)
    tiles = cd(s0, tile)
    ctas = min(65536, tiles)
    return {"s0": s0, "numTasks": numel(out), "tile": tile, "numTiles": tiles, "numCtas": ctas,
            "tilesPerCta": cd(tiles, ctas), "grid": [ctas, 1, 1],
            "inStride": x.strides[0] if x.strides else 0,
            "outStride": out.strides[0] if out.strides else 0}


def strided_obligations(x: TensorMeta, out: TensorMeta) -> list:
    """Mirror of `StridedUnary.check (launch x out)`, obligation by obligation.
    Negative strides (possible for `StridedBuffer`) are outside ℕ and rejected first."""
    if any(s < 0 for s in (*x.strides, *out.strides)):
        return [("S5 pos_strides (negative stride)", False)]
    c = strided_launch(x, out)
    s0, si, so = c["s0"], c["inStride"], c["outStride"]
    span = lambda tm, s: tm.base + (max(s0 - 1, 0) * s + 1) * tm.elemBytes
    return [("S1 rank1", len(x.shape) == 1 and len(out.shape) == 1 and len(x.strides) == 1
             and len(out.strides) == 1),
            ("S2 same_shape", x.shape == out.shape),
            ("S3 nonempty", 0 < s0),
            ("S4 one_tile", c["numTiles"] <= 65536),
            ("S5 pos_strides", 0 < si and 0 < so),
            ("S6 in_bounds", max(s0 - 1, 0) * si < x.capacity and max(s0 - 1, 0) * so < out.capacity),
            ("S7 dtype_ok", x.dtype == "f32" and x.elemBytes == 4 and out.dtype == "f32"
             and out.elemBytes == 4),
            ("S8 aligned", x.base % 4 == 0 and out.base % 4 == 0),
            ("S9 spans_disjoint", span(out, so) <= x.base or span(x, si) <= out.base),
            ("S10 offsets_fit", c["numCtas"] * c["tile"] <= I32),
            ("S11 addresses_fit", max(s0 - 1, 0) * si * 4 < I64 and max(s0 - 1, 0) * so * 4 < I64)]


RELU_PY = REPO / "bench/tritonbench_g/relu_strided_buffer/relu_strided_buffer.py"
RELU_WRAPPER_STMTS = [  # the modelled wrapper, statement by statement (normalized)
    "assert in0.shape == out0.shape, 'operand shapes mismatch'",
    "shape = out0.shape", "num_tasks = out0.numel()",
    "tile_sizes = heuristics_for_tile_size(512, *shape)", "tile_size = math.prod(tile_sizes)",
    "num_tiles = math.prod((triton.cdiv(size, tile_size) for size, tile_size in zip(shape, tile_sizes)))",
    "num_ctas = min(65536, num_tiles)", "tiles_per_cta = triton.cdiv(num_tiles, num_ctas)",
    "num_warps = heuristics_for_num_warps(tile_size)", "one_tile_per_cta = tiles_per_cta == 1",
    "grid = (num_ctas, 1, 1)", "in0_strides = in0.stride()", "in0_stride_order = (0,)",
    "out0_strides = out0.stride()", "out0_stride_order = (0,)"]
RELU_BINDING = {  # kernel parameter -> the wrapper expression the model requires
    "in0_ptr": "in0", "out0_ptr": "out0", "in0_stride0": "in0_strides[0]",
    "in0_stride_order0": "in0_stride_order[0]", "out0_stride0": "out0_strides[0]",
    "out0_stride_order0": "out0_stride_order[0]", "s0": "shape[0]", "num_tasks": "num_tasks",
    "tiles_per_cta": "tiles_per_cta", "tile_size0": "tile_sizes[0]",
    "one_tile_per_cta": "one_tile_per_cta"}
HEURISTIC_TILE = ("def heuristics_for_tile_size(max_tile_size, *sizes):\n    ndim = len(sizes)\n"
                  "    tile_sizes = [0 for _ in range(ndim)]\n    for i in range(ndim):\n"
                  "        size = sizes[ndim - 1 - i]\n"
                  "        tile_size = min(max_tile_size, triton.next_power_of_2(size))\n"
                  "        tile_sizes[ndim - 1 - i] = tile_size\n"
                  "        max_tile_size = max(1, max_tile_size // tile_size)\n"
                  "    return tuple(tile_sizes)")


def recognize_relu(src: str) -> dict:
    """Bind `relu_forward_wrapper_rank_1` to `StridedUnary.launch`: every modelled
    statement present, the tile heuristic's text as pinned, and every kernel
    argument bound to the expression the model requires — in particular the
    stride arguments to `in0.stride()[0]` / `out0.stride()[0]` of the same
    tensors. Anything else raises `Unsupported`."""
    tree = ast.parse(src)
    heur = LC.find_function(tree, "heuristics_for_tile_size")
    if ast.unparse(heur) != HEURISTIC_TILE:
        raise LC.Unsupported("heuristics_for_tile_size differs from the modelled text")
    fn = LC.find_function(tree, "relu_forward_wrapper_rank_1")
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    stmts = [ast.unparse(s) for s in body if not isinstance(s, (ast.With, ast.Return))]
    if stmts != RELU_WRAPPER_STMTS:
        missing = [s for s in RELU_WRAPPER_STMTS if s not in stmts]
        extra = [s for s in stmts if s not in RELU_WRAPPER_STMTS]
        raise LC.Unsupported(f"wrapper statements differ: missing {missing}, extra {extra}")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)]
    if len(calls) != 1 or ast.unparse(calls[0].func) != "relu_forward_kernel_rank_1[grid]":
        raise LC.Unsupported("expected exactly one launch relu_forward_kernel_rank_1[grid](...)")
    kfn = LC.find_function(tree, "relu_forward_kernel_rank_1")
    params = [a.arg for a in kfn.args.args]
    binding = dict(zip(params, (ast.unparse(a) for a in calls[0].args)))
    for kw in calls[0].keywords:
        if kw.arg == "num_warps":
            continue  # compile option; no effect on the modelled semantics
        binding[kw.arg] = ast.unparse(kw.value)
    for p_, want in RELU_BINDING.items():
        if binding.get(p_) != want:
            raise LC.Unsupported(f"kernel argument {p_} bound to {binding.get(p_)!r}, model requires {want!r}")
    return {"params": params, "binding": binding}


class CheckedStridedRelu:
    """`relu_forward_wrapper_rank_1` on actual tensors, checked by the strided contract."""

    def __init__(self, variant: dict | None = None, src: str | None = None):
        self.recognized = recognize_relu(src if src is not None else RELU_PY.read_text())
        import launch_interpret as LI  # noqa: E402
        self._kernel = LI.load_defs(RELU_PY, variant)["relu_forward_kernel_rank_1"]
        self.stats = {"calls": 0}

    def verdict(self, in0, out0) -> tuple:
        mx, mo = storage_meta(in0), storage_meta(out0)
        obl = strided_obligations(mx, mo)
        return [n for n, ok in obl if not ok], strided_launch(mx, mo)

    def __call__(self, in0, out0, launch: bool = True):
        self.stats["calls"] += 1
        failed, c = self.verdict(in0, out0)
        if failed:
            raise ContractViolation(failed)
        if launch:
            self._kernel[(c["numCtas"], 1, 1)](
                in0, out0, c["inStride"], 0, c["outStride"], 0, c["s0"], c["numTasks"],
                tiles_per_cta=c["tilesPerCta"], tile_size0=c["tile"], one_tile_per_cta=True)
        return out0


# ---------------------------------------------------------------------------
# Differential test: Python mirror vs the Lean definitions
# ---------------------------------------------------------------------------

LEAN_DT = {"f32": ".f32", "f16": ".f16", "bf16": ".bf16", "other": ".other"}


def lean_tm(t: TensorMeta) -> str:
    return (f"⟨{t.base}, {t.elemBytes}, [{', '.join(map(str, t.shape))}], "
            f"[{', '.join(map(str, t.strides))}], {t.capacity}, {LEAN_DT[t.dtype]}⟩")


def near_valid_case(rnd: random.Random) -> tuple:
    """A valid configuration, then (half the time) exactly one field perturbed."""
    rank = rnd.choice([0, 1, 1, 2, 3])
    shape = tuple(rnd.choice([1, 2, 3, 4, 5, 8, 16, 33]) for _ in range(rank))
    if rnd.random() < 0.05:
        shape = (rnd.choice([2 ** 31 - 1, 2 ** 31 - 4]),)
    m = _prod(shape)
    B = rnd.choice([1, 2, 4, 16, 64, 1024, 2 ** 20])
    gap = 4 * m + 4 * rnd.choice([0, 1, 64])
    bx = 4 * rnd.randrange(1, 2 ** 20)
    by = rnd.choice([bx, bx + gap])
    bo = max(bx, by) + gap
    mk = lambda b: TensorMeta(b, 4, shape, tuple(row_major(shape)), m, "f32")
    x, y, o = mk(bx), mk(by), mk(bo)
    if rnd.random() < 0.5:
        which = rnd.randrange(3)
        t = [x, y, o][which]
        field = rnd.choice(["base", "shape", "strides", "capacity", "dtype", "elemBytes", "B"])
        if field == "B":
            B = rnd.choice([0, 3, 2 ** 21])
        else:
            val = {"base": t.base + rnd.choice([1, 2, -4 * max(m, 1) // 2, 4]),
                   "shape": tuple(reversed(shape)) + (1,) if rnd.random() < 0.5 else shape[:-1],
                   "strides": tuple(s + 1 for s in t.strides),
                   "capacity": max(0, t.capacity - 1), "dtype": rnd.choice(["f16", "other"]),
                   "elemBytes": 2}[field]
            t = TensorMeta(**{**t.__dict__, field: max(0, val) if field == "base" else val})
            x, y, o = [t if i == which else v for i, v in enumerate((x, y, o))]
    return (B, x, y, o)


def random_cases(n: int, seed: int = 0) -> list:
    rnd = random.Random(seed)
    cases = [near_valid_case(rnd) for _ in range(n // 2)]
    for i in range(n - n // 2):
        rank = rnd.choice([0, 1, 1, 1, 2, 2, 3])
        shape = tuple(rnd.choice([0, 1, 2, 3, 5, 8, 16]) for _ in range(rank))
        if rnd.random() < 0.05:
            shape = (rnd.choice([2 ** 31 - 1, 2 ** 31, 3 * 2 ** 30]),)

        def strides_for(sh):
            rm = row_major(sh)
            if rnd.random() < 0.75:
                return tuple(rm)
            return tuple(rnd.choice([0, 1, 2, r, r + 1]) for r in rm)

        def meta(sh, base):
            eb = rnd.choice([4, 4, 4, 4, 2, 0])
            m = _prod(sh)
            return TensorMeta(base, eb, sh, strides_for(sh),
                              max(0, m + rnd.choice([0, 0, 0, -1, 1, 7])),
                              rnd.choice(["f32"] * 6 + ["f16", "bf16", "other"]))
        bx = rnd.choice([0, 4096, 4098, 2 ** 40])
        by = rnd.choice([bx, bx + 4, bx + 64, 8192, 8190, 2 ** 40 + 2 ** 20])
        bo = rnd.choice([bx, bx + 8, 16384, 16386, 2 ** 41])
        ysh = shape if rnd.random() < 0.8 else tuple(rnd.choice([1, 2, 12]) for _ in range(rank))
        osh = shape if rnd.random() < 0.9 else tuple(reversed(shape))
        B = rnd.choice([0, 1, 3, 4, 4, 16, 64, 1024, 2 ** 20, 2 ** 21])
        cases.append((B, meta(shape, bx), meta(ysh, by), meta(osh, bo)))
    return cases


def differential(n_cases: int) -> dict:
    cases = random_cases(n_cases)
    head = "import VeriTile.Triton.Launch.Blocked1DWrapper\nopen VeriTile.Triton\n\n"
    body = []
    for i, (B, x, y, o) in enumerate(cases):
        args = f"{B} {lean_tm(x)} {lean_tm(y)} {lean_tm(o)}"
        body.append(f'#eval IO.println s!"R|{{Elementwise2.check {args}}}|'
                    f'{{Elementwise2.checkRank1 {args}}}|'
                    f'{{String.intercalate "," (Blocked1DLaunch.failures (Elementwise2.launch {args}))}}|'
                    f'{{String.intercalate "," (Blocked1DLaunch.failures (Elementwise2.launchDim0 {args}))}}"')
    r = LC.run_lean(head + "\n".join(body) + "\n", timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    rows = [ln.split("|")[1:] for ln in r.stdout.splitlines() if ln.startswith("R|")]
    assert len(rows) == len(cases), (len(rows), len(cases))
    split = lambda s: [a for a in s.split(",") if a]
    mismatches, acc = [], {"numel": 0, "dim0_rank1": 0}
    for (B, x, y, o), (c1, c2, f1, f2) in zip(cases, rows):
        py = (all(ok for _, ok in ew2_obligations(B, x, y, o)),
              all(ok for _, ok in rank1_obligations(B, x, y, o)),
              [nm for nm, ok in launch_obligations(ew2_launch(B, x, y, o)) if not ok],
              [nm for nm, ok in launch_obligations(dim0_launch(B, x, y, o)) if not ok])
        lean = (c1 == "true", c2 == "true", split(f1), split(f2))
        acc["numel"] += lean[0]
        acc["dim0_rank1"] += lean[1]
        if py != lean:
            mismatches.append({"case": [B, astuple(x), astuple(y), astuple(o)],
                               "python": py, "lean": lean})
    return {"cases": len(cases), "lean_accepted": acc, "mismatches": mismatches,
            "input_hashes": __import__("launch_local_check").input_hashes(),
            "reference": "Lean #eval of Elementwise2.check / checkRank1 and Blocked1DLaunch.failures "
                         "of the derived launches (compiled Lean evaluation, not `decide`)"}


def strided_cases(n: int, seed: int = 1) -> list:
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        s0 = rnd.choice([1, 2, 3, 7, 100, 511, 512, 513, 1025, 4096, 10000, 2 ** 25,
                         2 ** 25 + 1, 33554432, 0])
        si, so = rnd.choice([1, 1, 2, 3, 7]), rnd.choice([1, 1, 2, 3])
        ci, co = (max(s0 - 1, 0) * si + 1), (max(s0 - 1, 0) * so + 1)
        bx = 4 * rnd.randrange(1, 2 ** 20)
        bo = bx + 4 * ci + 4 * rnd.choice([0, 0, 1, 1000])
        x = TensorMeta(bx, 4, (s0,), (si,), ci, "f32")
        o = TensorMeta(bo, 4, (s0,), (so,), co, "f32")
        if rnd.random() < 0.6:
            which = rnd.randrange(2)
            tm = [x, o][which]
            field = rnd.choice(["base", "shape", "strides", "capacity", "dtype", "elemBytes", "overlap"])
            if field == "overlap":
                o = TensorMeta(x.base + 4 * rnd.randrange(0, max(ci, 1)), 4, o.shape, o.strides,
                               o.capacity, o.dtype)
            else:
                val = {"base": tm.base + rnd.choice([1, 2, 3]),
                       "shape": rnd.choice([(s0 + 1,), (s0, 2), ()]),
                       "strides": rnd.choice([(0,), (si + 1,), (1, 1), ()]),
                       "capacity": max(0, tm.capacity - rnd.choice([1, 2])),
                       "dtype": rnd.choice(["f16", "bf16", "other"]), "elemBytes": 2}[field]
                tm = TensorMeta(**{**tm.__dict__, field: val})
                x, o = (tm, o) if which == 0 else (x, tm)
        out.append((x, o))
    return out


def differential_strided(n_cases: int) -> dict:
    cases = strided_cases(n_cases)
    head = "import VeriTile.Triton.Launch.StridedUnary\nopen VeriTile.Triton\n\n"
    body = []
    for x, o in cases:
        L = f"(StridedUnary.launch {lean_tm(x)} {lean_tm(o)})"
        body.append(f'#eval IO.println s!"R|{{StridedUnary.check {L}}}|{{{L}.s0}}|{{{L}.tile}}|'
                    f'{{{L}.numTiles}}|{{{L}.numCtas}}|{{{L}.tilesPerCta}}|{{{L}.numTasks}}|'
                    f'{{{L}.inStride}}|{{{L}.outStride}}|{{{L}.grid}}"')
    r = LC.run_lean(head + "\n".join(body) + "\n", timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    rows = [ln.split("|")[1:] for ln in r.stdout.splitlines() if ln.startswith("R|")]
    assert len(rows) == len(cases), (len(rows), len(cases))
    mism, acc = [], 0
    for (x, o), row in zip(cases, rows):
        c = strided_launch(x, o)
        py = [str(all(ok for _, ok in strided_obligations(x, o))).lower(), str(c["s0"]),
              str(c["tile"]), str(c["numTiles"]), str(c["numCtas"]), str(c["tilesPerCta"]),
              str(c["numTasks"]), str(c["inStride"]), str(c["outStride"]),
              "[" + ", ".join(map(str, c["grid"])) + "]"]
        acc += row[0] == "true"
        if py != row:
            mism.append({"case": [astuple(x), astuple(o)], "python": py, "lean": row})
    return {"cases": len(cases), "lean_accepted": acc, "mismatches": mism,
            "compared": "verdict of StridedUnary.check and every derived launch value passed to the "
                        "kernel (s0, tile, numTiles, numCtas, tilesPerCta, numTasks, inStride, "
                        "outStride, grid) — Lean #eval vs Python mirror",
            "input_hashes": __import__("launch_local_check").input_hashes()}


# ---------------------------------------------------------------------------
# Host-side overhead
# ---------------------------------------------------------------------------

def overhead(reps: int = 2000) -> dict:
    import torch
    res = {"host": os.uname().machine + " " + sys.platform, "torch": torch.__version__,
           "note": "host-side cost of the checked path only (metadata extraction, contract "
                   "decision, cache lookup), CPU tensors; kernel launch not included"}

    def t_us(fn, k=reps):
        fn()
        t0 = time.perf_counter()
        for _ in range(k):
            fn()
        return (time.perf_counter() - t0) / k * 1e6
    rows = []
    for shape in [(1024,), (1 << 20,), (256, 256)]:
        x, y = torch.randn(shape), torch.randn(shape)
        out = torch.zeros_like(x)
        mx, my, mo = tensor_meta(x), tensor_meta(y), tensor_meta(out)
        cache = {}
        key = ("numel", 4, astuple(mx), astuple(my), astuple(mo))
        cache[key] = ([], None)
        rows.append({
            "shape": list(shape),
            "tensor_meta_x3_us": t_us(lambda: (tensor_meta(x), tensor_meta(y), tensor_meta(out))),
            "contract_decision_us": t_us(lambda: ew2_obligations(4, mx, my, mo)),
            "cache_key_and_lookup_us": t_us(
                lambda: cache.get(("numel", 4, astuple(tensor_meta(x)), astuple(tensor_meta(y)),
                                   astuple(tensor_meta(out))))),
            "zeros_like_alloc_us_reference": t_us(lambda: torch.zeros_like(x), max(50, reps // 20))})
    xr, orr = torch.randn(40)[::2], torch.empty(20)
    mxr, mor = storage_meta(xr), storage_meta(orr)
    res["strided_relu"] = {
        "storage_meta_x2_us": t_us(lambda: (storage_meta(xr), storage_meta(orr))),
        "contract_decision_us": t_us(lambda: strided_obligations(mxr, mor))}
    res["rows"] = rows
    res["input_hashes"] = __import__("launch_local_check").input_hashes()
    # allocator address reuse decides the cache hit rate for freshly allocated outputs
    x, y = torch.randn(1 << 16), torch.randn(1 << 16)
    ptrs = [torch.zeros_like(x).data_ptr() for _ in range(200)]
    res["fresh_output_distinct_ptrs_in_200_allocs"] = len(set(ptrs))
    return res


# ---------------------------------------------------------------------------
# Interpreter demo: valid and invalid invocations on actual tensors
# ---------------------------------------------------------------------------

def demo() -> dict:
    import torch
    assert os.environ.get("TRITON_INTERPRET") == "1"
    import triton
    torch.manual_seed(0)
    variant = {'BLOCK_SIZE: "tl.constexpr"': "BLOCK_SIZE: tl.constexpr"}  # interpreter only
    ae, vac = add_example(variant), custom_add()
    base = torch.randn(64)
    x16 = torch.randn(16)
    cases = [
        ("ae_1d_n16", ae, lambda: (torch.randn(16), torch.randn(16)), None),
        ("ae_1d_n5_tail", ae, lambda: (torch.randn(5), torch.randn(5)), None),
        ("ae_empty", ae, lambda: (torch.randn(0), torch.randn(0)), None),
        ("ae_2d_contiguous", ae, lambda: (torch.randn(4, 8), torch.randn(4, 8)), None),
        ("ae_3d_contiguous", ae, lambda: (torch.randn(2, 3, 5), torch.randn(2, 3, 5)), None),
        ("ae_x_y_same_tensor", ae, lambda: (x16, x16), None),
        ("ae_transposed_input", ae, lambda: (torch.randn(8, 4).t(), torch.randn(4, 8)), ["WC contiguous"]),
        ("ae_shape_mismatch", ae, lambda: (torch.randn(16), torch.randn(12)), ["WA same_shape"]),
        ("ae_y_short_view_of_big_storage", ae, lambda: (torch.randn(16), torch.randn(16)[:12]),
         ["WA same_shape"]),
        ("ae_inputs_partially_overlap", ae, lambda: (base[0:16], base[8:24]), ["P11 inputs_separate"]),
        ("ae_float16", ae, lambda: (torch.randn(16).half(), torch.randn(16).half()), ["P9 dtype_ok"]),
        ("vac_1d_n37", vac, lambda: (torch.randn(37), torch.randn(37)), None),
        ("vac_2d_rejected", vac, lambda: (torch.randn(4, 8), torch.randn(4, 8)), ["WR rank1"]),
    ]
    rows = []
    for name, w, mk, expect in cases:
        x, y = mk()
        try:
            out = w(x, y)
            ok = bool(torch.equal(out, x + y))
            rows.append({"case": name, "outcome": "accepted", "output_equals_x_plus_y": ok,
                         "output_shape_matches": list(out.shape) == list(x.shape),
                         "expected": "accepted", "as_expected": expect is None and ok})
        except ContractViolation as e:
            rows.append({"case": name, "outcome": "rejected", "failed": e.failures,
                         "expected": expect, "as_expected": expect is not None
                         and set(expect) <= set(e.failures)})
    # strided ReLU (one-tile branch)
    rv = {"torch.cuda._DeviceGuard(in0.device.index)": "__import__('contextlib').nullcontext()",
          "out0.to(out0_bptr.type.element_ty)": "out0.to(out0_ptr.type.element_ty)"}
    relu = CheckedStridedRelu(rv)
    SB = __import__("launch_interpret").load_defs(RELU_PY, rv)["StridedBuffer"]
    rb = torch.randn(64)

    def relu_case(name, mk, expect, launch=True, check_gaps=None):
        x, o = mk()
        try:
            relu(x, o, launch=launch)
            if launch:
                ok = bool(torch.equal(o, torch.relu(x)))
                gaps = check_gaps() if check_gaps else None
                rows.append({"case": name, "outcome": "accepted", "output_equals_relu": ok,
                             "gap_cells_intact": gaps, "expected": "accepted",
                             "as_expected": expect is None and ok and gaps is not False})
            else:
                rows.append({"case": name, "outcome": "accepted (verdict only; not launched)",
                             "expected": "accepted", "as_expected": expect is None})
        except ContractViolation as e:
            rows.append({"case": name, "outcome": "rejected", "failed": e.failures, "expected": expect,
                         "as_expected": expect is not None and set(expect) <= set(e.failures)})
    ob = torch.full((30,), 7.5)
    relu_case("relu_contiguous_n1025", lambda: (torch.randn(1025), torch.empty(1025)), None)
    relu_case("relu_in_stride2", lambda: (torch.randn(40)[::2], torch.empty(20)), None)
    relu_case("relu_out_stride3_gaps", lambda: (torch.randn(10), ob[::3]), None,
              check_gaps=lambda: bool((ob[[i for i in range(30) if i % 3]] == 7.5).all()))
    relu_case("relu_empty", lambda: (torch.randn(0), torch.empty(0)), ["S3 nonempty"])
    relu_case("relu_grid_stride_branch_n_2p25_plus_1",
              lambda: (torch.empty(2 ** 25 + 1), torch.empty(2 ** 25 + 1)), ["S4 one_tile"], launch=False)
    relu_case("relu_out_overlaps_in", lambda: (rb[0:10], rb[5:15]), ["S9 spans_disjoint"])
    relu_case("relu_float16", lambda: (torch.randn(16).half(), torch.empty(16).half()), ["S7 dtype_ok"])
    relu_case("relu_stridedbuffer_offset5_stride3",
              lambda: (SB(torch.randn(50), shape=(10,), strides=(3,), offset=5), torch.empty(10)),
              None, launch=False)
    relu_case("relu_stridedbuffer_negative_stride",
              lambda: (SB(torch.randn(50), shape=(10,), strides=(-1,), offset=9), torch.empty(10)),
              ["S5 pos_strides (negative stride)"])
    relu_case("relu_stridedbuffer_negative_offset_before_storage",
              lambda: (SB(torch.randn(50), shape=(10,), strides=(1,), offset=-5), torch.empty(10)),
              ["S6 in_bounds"], launch=False)
    relu_case("relu_stridedbuffer_dtype_reinterpret",
              lambda: (SB(torch.randn(16), dtype=torch.int32), torch.empty(16)), ["S7 dtype_ok"])
    mutant = RELU_PY.read_text().replace("in0_strides[0], # stride for in0",
                                         "out0_strides[0], # stride for in0")
    try:
        CheckedStridedRelu(rv, src=mutant)
        rows.append({"case": "relu_source_mutant_wrong_stride_arg", "outcome": "ACCEPTED",
                     "as_expected": False})
    except LC.Unsupported as e:
        rows.append({"case": "relu_source_mutant_wrong_stride_arg", "outcome": "unsupported_input",
                     "reason": str(e), "as_expected": True})
    np2 = all(next_pow2(k) == triton.next_power_of_2(k) for k in range(0, 5001))
    rows.append({"case": "next_pow2_mirror_vs_triton_0_to_5000", "outcome": "equal" if np2 else "DIFFER",
                 "as_expected": np2})
    return {"backend": "triton-interpreter (TRITON_INTERPRET=1, CPU)", "triton": triton.__version__,
            "relu_source": "DERIVED VARIANT for the interpreter (device guard -> nullcontext; store cast "
                           "to the pointer element type) — the pinned text needs CUDA; StridedBuffer "
                           "arguments cannot be passed to the interpreter (verdict only)",
            "torch": torch.__version__,
            "add_example_source": "DERIVED VARIANT for the interpreter (class tl.constexpr annotation; "
                                  "REPORT finding 4) — the pinned text runs on GPU",
            "vector_addition_custom_source": "pinned (verbatim)",
            "cases": rows, "all_as_expected": all(r["as_expected"] for r in rows),
            "input_hashes": __import__("launch_local_check").input_hashes()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--differential", action="store_true")
    ap.add_argument("--differential-strided", action="store_true")
    ap.add_argument("--cases", type=int, default=1500)
    ap.add_argument("--overhead", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = (differential(a.cases) if a.differential
           else differential_strided(a.cases) if a.differential_strided
           else overhead() if a.overhead else demo())
    text = json.dumps(res, indent=1, default=str) + "\n"
    (a.out.write_text(text) if a.out else print(text[:4000]))
    if a.differential or a.differential_strided:
        return 1 if res["mismatches"] else 0
    if a.demo:
        return 0 if res["all_as_expected"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
