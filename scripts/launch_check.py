#!/usr/bin/env python3
"""Host-launch adapter for the proved 1-D blocked launch checker.

Binds one source-pinned Triton kernel + host wrapper to
`VeriTile.Triton.Blocked1DLaunch.check` (Lean, proved sound and complete in
`VeriTile/Triton/Launch/Blocked1DConfig.lean`):

1. Parses the kernel and wrapper source with `ast` — never imports or executes
   it (TritonBench-G files run CUDA tests at import time).
2. Accepts only the supported kernel shape (1-D program id, block offsets,
   `offsets < n` mask on every load/store, `+`-only lane arithmetic) and the
   supported wrapper shape (literal constexpr block, `numel()`/`size(0)`
   element count, `cdiv`/floor-div/literal 1-D grid, `*_like` output
   allocation). Anything else is rejected as `unsupported`.
3. Compares the kernel body statement-by-statement with the Lean `triton { }`
   transcription (structural match — trusted and tested, not proved).
4. Builds `Blocked1DLaunch` values from real tensor metadata (CPU torch
   tensors built by this adapter) or from explicitly supplied metadata.
5. Emits a Lean file, lets Lean evaluate the checker for diagnostics, then
   re-checks every verdict with `decide` in the Lean kernel (no
   `native_decide`); accepted cases also produce `Pre` via `check_ok`, and
   rejected cases a proof of `¬ Pre` via `check_complete`. The reported
   verdict is the kernel-checked one.

This file is part of the trusted computing base for source correspondence
(steps 1–4); see bench/tritonbench_g/add_example/SOURCE_LINK.md.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))
import audit_source  # noqa: E402  (shared upstream source helpers)

CHECKER_MODULE = "VeriTile.Triton.Launch.Blocked1DConfig"


class Unsupported(Exception):
    """Source or configuration outside the supported, modelled subset."""


# --------------------------------------------------------------------------
# Kernel source recognition
# --------------------------------------------------------------------------

@dataclass
class KernelSummary:
    name: str
    params: list[str]
    ptr_params: list[str]            # pointer params loaded, in load order
    out_param: str
    n_param: str
    block_param: str
    body_statements: list[str]       # normalized, for the Lean comparison


def _is_tl_call(node: ast.AST, fn: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == fn and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl")


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise Unsupported(f"function {name!r} not found at module top level")


def parse_kernel(src: str, name: str) -> KernelSummary:
    """Symbolically evaluate the kernel body over the supported grammar.

    Values: ('pid',), ('const', param), ('blk', start_param) meaning
    `start * BLOCK + arange(0, BLOCK)`, ('mask', offs_sym, n_param),
    ('val',) for loaded/combined lane values."""
    tree = ast.parse(src)
    fn = find_function(tree, name)
    if not any(audit_source.is_triton_jit(d) for d in fn.decorator_list):
        raise Unsupported(f"{name} is not @triton.jit")
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.defaults:
        raise Unsupported("kernel signature with defaults/varargs is not modelled")
    params = [a.arg for a in fn.args.args]
    env: dict[str, tuple] = {}
    loads: list[str] = []
    stores: list[tuple[str, tuple, tuple]] = []
    block_param: str | None = None
    n_param: str | None = None
    stmts: list[str] = []

    def ev(e: ast.AST) -> tuple:
        nonlocal block_param, n_param
        if isinstance(e, ast.Name):
            if e.id in env:
                return env[e.id]
            if e.id in params:
                return ("param", e.id)
            raise Unsupported(f"unknown name {e.id!r}")
        if _is_tl_call(e, "program_id"):
            args = list(e.args) + [k.value for k in e.keywords if k.arg == "axis"]
            if len(args) != 1 or not (isinstance(args[0], ast.Constant) and args[0].value == 0):
                raise Unsupported("only tl.program_id(axis=0) is modelled")
            if any(k.arg not in ("axis",) for k in e.keywords):
                raise Unsupported("unexpected tl.program_id keyword")
            return ("pid",)
        if _is_tl_call(e, "arange"):
            if e.keywords or len(e.args) != 2 or not (
                    isinstance(e.args[0], ast.Constant) and e.args[0].value == 0
                    and isinstance(e.args[1], ast.Name) and e.args[1].id in params):
                raise Unsupported("only tl.arange(0, BLOCK_PARAM) is modelled")
            b = e.args[1].id
            if block_param not in (None, b):
                raise Unsupported("inconsistent block parameter")
            block_param = b
            return ("arange", b)
        if isinstance(e, ast.BinOp):
            l, r = ev(e.left), ev(e.right)
            if isinstance(e.op, ast.Mult) and l == ("pid",) and r[0] == "param":
                if block_param not in (None, r[1]):
                    raise Unsupported("inconsistent block parameter")
                block_param = r[1]
                return ("start", r[1])
            if isinstance(e.op, ast.Add) and l[0] == "start" and r[0] == "arange" and l[1] == r[1]:
                return ("offs", r[1])
            if isinstance(e.op, ast.Add) and l[0] == "param" and r[0] == "offs":
                return ("ptr", l[1])
            if isinstance(e.op, ast.Add) and l == ("val",) and r == ("val",):
                return ("val",)
            raise Unsupported(f"unmodelled arithmetic: {ast.unparse(e)}")
        if isinstance(e, ast.Compare):
            if (len(e.ops) == 1 and isinstance(e.ops[0], ast.Lt)):
                l, r = ev(e.left), ev(e.comparators[0])
                if l[0] == "offs" and r[0] == "param":
                    if n_param not in (None, r[1]):
                        raise Unsupported("inconsistent element-count parameter")
                    n_param = r[1]
                    return ("mask", r[1])
            raise Unsupported(f"unmodelled comparison: {ast.unparse(e)}")
        if _is_tl_call(e, "load"):
            if len(e.args) != 1 or [k.arg for k in e.keywords] != ["mask"]:
                raise Unsupported("tl.load must be tl.load(ptr + offsets, mask=mask) (no other=, no extra kwargs)")
            p = ev(e.args[0])
            m = ev(e.keywords[0].value)
            if p[0] != "ptr" or m[0] != "mask":
                raise Unsupported("tl.load address/mask not of the blocked form")
            loads.append(p[1])
            return ("val",)
        if isinstance(e, ast.Call):
            raise Unsupported(f"unmodelled call: {ast.unparse(e.func)}")
        raise Unsupported(f"unmodelled expression: {ast.unparse(e)}")

    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # docstring
    for st in body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            env[st.targets[0].id] = ev(st.value)
        elif isinstance(st, ast.Expr) and _is_tl_call(st.value, "store"):
            c = st.value
            if len(c.args) != 2 or [k.arg for k in c.keywords] != ["mask"]:
                raise Unsupported("tl.store must be tl.store(ptr + offsets, value, mask=mask)")
            p, v, m = ev(c.args[0]), ev(c.args[1]), ev(c.keywords[0].value)
            if p[0] != "ptr" or v != ("val",) or m[0] != "mask":
                raise Unsupported("tl.store address/value/mask not of the blocked form")
            stores.append((p[1], v, m))
        else:
            raise Unsupported(f"unmodelled statement: {ast.unparse(st).splitlines()[0]}")
        stmts.append(_normalize(ast.unparse(st)))
    if len(stores) != 1:
        raise Unsupported(f"exactly one masked store is modelled (found {len(stores)})")
    if block_param is None or n_param is None:
        raise Unsupported("kernel lacks the blocked offset/mask pattern")
    out = stores[0][0]
    if out in loads:
        raise Unsupported("in-place update (output pointer also loaded) is not modelled")
    # Config inputs are listed in load order and the Lean theorems bind them positionally to the
    # kernel's pointer parameters in signature order; require the two orders to agree.
    if loads != [p for p in params if p in loads]:
        raise Unsupported(f"pointer loads {loads} are not in signature order")
    return KernelSummary(name, params, loads, out, n_param, block_param, stmts)


def _normalize(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    return s.replace("'", '"')


def lean_body_statements(lean_text: str, kernel_name: str) -> list[str]:
    """Statements of the Lean `triton { }` transcription of `kernel_name`,
    with antiquotations `$(x)` unwrapped to `x`."""
    code = audit_source.strip_lean_comments(lean_text)
    m = re.search(r"\bdef\s+" + re.escape(kernel_name) + r"\b", code)
    if not m:
        raise Unsupported(f"Lean transcription `{kernel_name}` not found")
    body = audit_source.lean_first_triton_body(code[m.start():])
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"\$\(\s*([A-Za-z_][A-Za-z_0-9]*)\s*\)", r"\1", line)
        out.append(_normalize(line))
    return out


# --------------------------------------------------------------------------
# Host wrapper recognition
# --------------------------------------------------------------------------

@dataclass
class LaunchSummary:
    wrapper: str
    wrapper_params: list[str]
    arg_binding: dict[str, str]          # kernel param -> wrapper variable / literal
    block: int
    n_source: str                        # wrapper param whose numel is n
    grid_kind: str                       # 'cdiv' | 'floordiv' | 'literal'
    grid_literal: int | None
    out_alloc: tuple[str, str]           # (out var, like-source var)
    notes: list[str] = field(default_factory=list)


def parse_launch(src: str, wrapper: str, kernel: KernelSummary) -> LaunchSummary:
    tree = ast.parse(src)
    fn = find_function(tree, wrapper)
    wparams = [a.arg for a in fn.args.args]
    consts: dict[str, int] = {}
    nvars: dict[str, str] = {}
    grids: dict[str, tuple[str, int | None, tuple[str, int] | None]] = {}
    allocs: dict[str, str] = {}
    launch: ast.Call | None = None

    def n_of(e: ast.AST) -> str | None:
        # x.numel()  |  x.size(0)  |  x.shape[0]  |  name bound to one of these
        if isinstance(e, ast.Name) and e.id in nvars:
            return nvars[e.id]
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and \
                isinstance(e.func.value, ast.Name):
            v, a = e.func.value.id, e.func.attr
            if a == "numel" and not e.args:
                return v
            if a == "size" and len(e.args) == 1 and isinstance(e.args[0], ast.Constant) \
                    and e.args[0].value == 0:
                return v + "#dim0"
        return None

    def int_of(e: ast.AST) -> int | None:
        if isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool):
            return e.value
        if isinstance(e, ast.Name) and e.id in consts:
            return consts[e.id]
        return None

    def grid_of(e: ast.AST) -> tuple[str, int | None, tuple[str, int] | None]:
        if isinstance(e, ast.Name) and e.id in grids:
            return grids[e.id]
        if isinstance(e, ast.Tuple):
            if len(e.elts) != 1:
                raise Unsupported(f"grid of rank {len(e.elts)} is not modelled")
            return grid_axis(e.elts[0])
        raise Unsupported(f"unrecognized grid expression: {ast.unparse(e)}")

    def grid_axis(e: ast.AST) -> tuple[str, int | None, tuple[str, int] | None]:
        # Returns (kind, literal, (count source, block value) used by the formula); the
        # caller checks the formula's count source and block value against the launch.
        if isinstance(e, ast.Name) and e.id in grids:
            return grids[e.id]
        lit = int_of(e)
        if lit is not None:
            return ("literal", lit, None)
        u = _normalize(ast.unparse(e))
        for nv, src_ in nvars.items():
            for bname, bval in consts.items():
                if u in (f"({nv}+{bname}-1)//{bname}", f"triton.cdiv({nv},{bname})"):
                    return ("cdiv", None, (src_, bval))
                if u == f"{nv}//{bname}":
                    return ("floordiv", None, (src_, bval))
        raise Unsupported(f"unrecognized grid axis formula: {ast.unparse(e)}")

    wbody = list(fn.body)
    if wbody and isinstance(wbody[0], ast.Expr) and isinstance(wbody[0].value, ast.Constant) \
            and isinstance(wbody[0].value.value, str):
        wbody = wbody[1:]  # docstring
    for st in wbody:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            t, v = st.targets[0].id, st.value
            if int_of(v) is not None:
                consts[t] = int_of(v)
            elif n_of(v) is not None:
                nvars[t] = n_of(v)
            elif isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and \
                    isinstance(v.func.value, ast.Name) and v.func.value.id == "torch" and \
                    v.func.attr in ("zeros_like", "empty_like") and len(v.args) == 1 and \
                    isinstance(v.args[0], ast.Name) and not v.keywords:
                allocs[t] = v.args[0].id
            else:
                try:
                    grids[t] = grid_of(v) if isinstance(v, ast.Tuple) else grid_axis(v)
                except Unsupported:
                    raise Unsupported(f"unmodelled wrapper statement: {ast.unparse(st)}")
        elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and \
                isinstance(st.value.func, ast.Subscript) and \
                isinstance(st.value.func.value, ast.Name) and st.value.func.value.id == kernel.name:
            if launch is not None:
                raise Unsupported("more than one launch in the wrapper")
            launch = st.value
        elif isinstance(st, ast.Return):
            pass
        else:
            raise Unsupported(f"unmodelled wrapper statement: {ast.unparse(st).splitlines()[0]}")
    if launch is None:
        raise Unsupported(f"no launch of {kernel.name} in {wrapper}")
    grid_kind, grid_lit, grid_terms = grid_of(launch.func.slice)
    binding: dict[str, str] = {}
    if len(launch.args) > len(kernel.params):
        raise Unsupported("too many launch arguments")
    for p, a in zip(kernel.params, launch.args):
        binding[p] = ast.unparse(a)
    for k in launch.keywords:
        if k.arg not in kernel.params:
            raise Unsupported(f"launch keyword {k.arg!r} is not a kernel parameter (launch options are not modelled)")
        binding[k.arg] = ast.unparse(k.value)
    missing = [p for p in kernel.params if p not in binding]
    if missing:
        raise Unsupported(f"kernel parameters not bound at launch: {missing}")
    bexpr = binding[kernel.block_param]
    block = consts.get(bexpr, int(bexpr) if bexpr.isdigit() else None)
    if block is None:
        raise Unsupported(f"BLOCK constexpr bound to non-literal {bexpr!r}")
    nexpr = binding[kernel.n_param]
    if nexpr not in nvars:
        raise Unsupported(f"element count bound to unrecognized {nexpr!r}")
    n_source = nvars[nexpr]
    if grid_terms is not None and grid_terms != (n_source, block):
        raise Unsupported(f"grid formula uses count {grid_terms[0]!r} / block {grid_terms[1]} but the "
                          f"kernel is launched with count {n_source!r} / block {block}")
    out_var = binding[kernel.out_param]
    if out_var not in allocs:
        raise Unsupported(f"output {out_var!r} is not a fresh *_like allocation")
    for p in kernel.ptr_params + [kernel.out_param]:
        v = binding[p]
        if v not in wparams and v not in allocs:
            raise Unsupported(f"pointer argument {p} bound to unrecognized {v!r}")
    return LaunchSummary(wrapper, wparams, binding, block, n_source, grid_kind, grid_lit,
                         (out_var, allocs[out_var]))


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

DTYPES = {"float32": (".f32", 4), "float16": (".f16", 2), "bfloat16": (".bf16", 2)}


def meta_from_tensor(t) -> dict:
    """Read layout metadata from a real torch tensor (flattened 1-D view)."""
    dt = str(t.dtype).replace("torch.", "")
    es = t.element_size()
    storage_elems = t.untyped_storage().nbytes() // es
    if t.is_contiguous():
        stride = 1
    elif t.dim() == 1:
        stride = int(t.stride(0))
    else:
        stride = 0  # non-contiguous multi-dim: no single flat stride
    return {"base": int(t.data_ptr()), "elemBytes": es, "stride": stride,
            "capacity": storage_elems - int(t.storage_offset()), "dtype": dt,
            "numel": int(t.numel()), "dim0": int(t.shape[0]) if t.dim() else 1,
            "source": "torch"}


def build_tensors(spec: dict, launch: LaunchSummary) -> dict[str, dict]:
    """Construct CPU tensors from a small declarative spec (no eval) and
    apply the wrapper's recognized output allocation."""
    import torch
    tens = {}
    keep = []
    for name, s in spec.items():
        if "alias_of" in s:
            continue
        dtype = getattr(torch, s.get("dtype", "float32"))
        step, off, numel = s.get("step", 1), s.get("offset", 0), s["numel"]
        size = (off + max(numel - 1, 0) * step + 1 if numel else off) + s.get("pad", 0)
        base = torch.randn(size, dtype=torch.float32).to(dtype)
        t = base[off:][::step][:numel] if numel else base[off:off]
        if "shape" in s:
            t = t.reshape(s["shape"])
        tens[name] = t
        keep.append(base)
    for name, s in spec.items():
        if "alias_of" in s:
            src = tens[s["alias_of"]]
            base = src._base if src._base is not None else src
            off = s.get("offset", 0)
            tens[name] = base.view(-1)[off: off + s["numel"]]
    out_var, like = launch.out_alloc
    if out_var not in tens:
        tens[out_var] = torch.zeros_like(tens[like])  # the wrapper's own allocation rule
    return {k: meta_from_tensor(v) for k, v in tens.items()}


def make_config(launch: LaunchSummary, kernel: KernelSummary, metas: dict[str, dict],
                grid_override: list[int] | None = None) -> dict:
    src, _, sel = launch.n_source.partition("#")
    n = metas[src]["dim0"] if sel == "dim0" else metas[src]["numel"]
    B = launch.block
    if grid_override is not None:
        grid = grid_override
    elif launch.grid_kind == "cdiv":
        grid = [(n + B - 1) // B]
    elif launch.grid_kind == "floordiv":
        grid = [n // B]
    else:
        grid = [launch.grid_literal]
    def buf(var):
        m = metas[var]
        if m["dtype"] not in DTYPES:
            dt = ".other"
        else:
            dt = DTYPES[m["dtype"]][0]
        return {k: m[k] for k in ("base", "elemBytes", "stride", "capacity")} | {"dtype": dt}
    return {"n": n, "block": B, "grid": grid,
            "inputs": [buf(launch.arg_binding[p]) for p in kernel.ptr_params],
            "output": buf(launch.arg_binding[kernel.out_param])}


# --------------------------------------------------------------------------
# Lean evaluation (checked implementation, kernel-checked verdicts)
# --------------------------------------------------------------------------

def lean_buf(b: dict) -> str:
    return f"⟨{b['base']}, {b['elemBytes']}, {b['stride']}, {b['capacity']}, {b['dtype']}⟩"


def lean_config(c: dict) -> str:
    ins = ", ".join(lean_buf(b) for b in c["inputs"])
    grid = ", ".join(str(g) for g in c["grid"])
    return (f"{{ n := {c['n']}, block := {c['block']}, grid := [{grid}],\n"
            f"    inputs := [{ins}],\n    output := {lean_buf(c['output'])} }}")


def run_lean(text: str, timeout: int = 600) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, dir=None) as f:
        f.write(text)
        path = f.name
    try:
        root = os.environ.get("VERITILE_LEAN_ROOT", str(REPO))  # fresh-workspace override
        return subprocess.run(["lake", "env", "lean", path], cwd=root, capture_output=True,
                              text=True, timeout=timeout)
    finally:
        os.unlink(path)


def lean_verdicts(configs: dict[str, dict]) -> dict[str, dict]:
    """Pass 1: Lean evaluates `check`/`failures` (diagnostics). Pass 2: every
    verdict is re-established by `decide` in the kernel, with `Pre` / `¬ Pre`."""
    names = list(configs)
    head = f"import {CHECKER_MODULE}\nopen VeriTile.Triton\n\n"
    defs = "".join(f"def cfg_{i} : Blocked1DLaunch :=\n  {lean_config(configs[n])}\n\n"
                   for i, n in enumerate(names))
    evals = "".join(f"#eval (Blocked1DLaunch.check cfg_{i}, Blocked1DLaunch.failures cfg_{i})\n"
                    for i in range(len(names)))
    r1 = run_lean(head + defs + evals)
    if r1.returncode != 0:
        raise RuntimeError("Lean diagnostic pass failed:\n" + r1.stdout + r1.stderr)
    pat = re.compile(r"^\((true|false), \[(.*)\]\)$", re.M)
    found = pat.findall(r1.stdout)
    if len(found) != len(names):
        raise RuntimeError(f"expected {len(names)} diagnostic lines, got {len(found)}:\n{r1.stdout}")
    diag = {n: (v == "true", [x.strip().strip('"') for x in f.split(",") if x.strip()])
            for n, (v, f) in zip(names, found)}
    thms = []
    for i, n in enumerate(names):
        ok = diag[n][0]
        thms.append(f"theorem cfg_{i}_verdict : Blocked1DLaunch.check cfg_{i} = {str(ok).lower()} := by decide\n")
        if ok:
            thms.append(f"theorem cfg_{i}_pre : Blocked1DLaunch.Pre cfg_{i} :=\n"
                        f"  Blocked1DLaunch.check_ok _ cfg_{i}_verdict\n")
        else:
            thms.append(f"theorem cfg_{i}_not_pre : ¬ Blocked1DLaunch.Pre cfg_{i} := fun h => by\n"
                        f"  have := Blocked1DLaunch.check_complete _ h\n"
                        f"  rw [cfg_{i}_verdict] at this; exact Bool.false_ne_true this\n")
        thms.append(f"#print axioms cfg_{i}_{'pre' if ok else 'not_pre'}\n")
    r2 = run_lean(head + defs + "".join(thms))
    if r2.returncode != 0:
        raise RuntimeError("Lean kernel re-check failed (verdict disagreement or infra):\n"
                           + r2.stdout + r2.stderr)
    for line in r2.stdout.splitlines():
        if "depends on axioms" in line:
            axs = set(re.findall(r"\[(.*)\]", line)[0].replace(" ", "").split(","))
            if not axs <= {"propext", "Classical.choice", "Quot.sound", ""}:
                raise RuntimeError(f"unapproved axiom in case certificate: {line}")
        elif "does not depend on any axioms" in line:
            pass
    return {n: {"accepted": diag[n][0], "failed_obligations": diag[n][1],
                "kernel_checked": True} for n in names}


# --------------------------------------------------------------------------
# Manifest-driven run
# --------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(manifest_path: Path) -> dict:
    man = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    py = (root / man["source"]).resolve()
    lean = (root / man["lean"]).resolve()
    src = py.read_text()
    result: dict = {"manifest": str(manifest_path.relative_to(REPO)), "cases": {}}
    kern = parse_kernel(src, man["kernel"])
    lstmts = lean_body_statements(lean.read_text(), man["lean_kernel"])
    if lstmts != kern.body_statements:
        raise Unsupported("Lean transcription differs from the Python kernel body:\n"
                          f"  python: {kern.body_statements}\n  lean:   {lstmts}")
    launch = parse_launch(src, man["wrapper"], kern)
    result["kernel"] = {"name": kern.name, "ptr_params": kern.ptr_params, "out": kern.out_param,
                        "n_param": kern.n_param, "block_param": kern.block_param,
                        "statements": kern.body_statements, "lean_match": True}
    result["launch"] = {"block": launch.block, "grid_kind": launch.grid_kind,
                        "binding": launch.arg_binding, "out_alloc": list(launch.out_alloc)}
    configs, provenance, wrapper_ob = {}, {}, {}
    for case in man["cases"]:
        if "tensors" in case:
            metas = build_tensors(case["tensors"], launch)
            provenance[case["name"]] = "torch-cpu-tensors"
        else:
            metas = case["meta"]
            provenance[case["name"]] = "supplied-metadata (trusted as supplied)"
        configs[case["name"]] = make_config(launch, kern, metas, case.get("grid"))
        out_meta = metas[launch.arg_binding[kern.out_param]]
        wrapper_ob[case["name"]] = {
            # W1 (adapter-checked, NOT part of the Lean contract): the wrapper returns the
            # whole output tensor, so every element of it must be one the kernel writes.
            "W1 output_fully_written": configs[case["name"]]["n"] == out_meta["numel"],
            # W2 (adapter-checked, NOT part of the Lean contract): P4 bounds addresses by the
            # allocation (memory safety); for the result to be `x + y` of the *tensors*, every
            # input tensor must itself hold the n elements read.
            "W2 inputs_cover_n": all(configs[case["name"]]["n"] <= metas[launch.arg_binding[p]]["numel"]
                                     for p in kern.ptr_params)}
    verdicts = lean_verdicts(configs)
    for case in man["cases"]:
        n = case["name"]
        v = verdicts[n]
        exp = case.get("expect")
        wfail = sorted(k for k, ok in wrapper_ob[n].items() if not ok)
        result["cases"][n] = {"config": configs[n], "metadata": provenance[n], **v,
                              "wrapper_obligations": wrapper_ob[n],
                              "wrapper_failures": wfail,
                              "expected": exp,
                              "matches_expectation": (exp is None or exp == ("accept" if v["accepted"] else "reject"))
                              and set(case.get("expect_failures", v["failed_obligations"]))
                              == set(v["failed_obligations"])
                              and set(case.get("expect_wrapper_failures", [])) == set(wfail)}
    result["hashes"] = {str(p.relative_to(REPO)): sha256(p) for p in
                        [py, lean, REPO / "VeriTile/Triton/Launch/Blocked1DConfig.lean",
                         Path(__file__).resolve(), manifest_path.resolve()]}
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, help="write the JSON result here")
    a = ap.parse_args(argv)
    try:
        res = analyze(a.manifest.resolve())
        status = "ok" if all(c["matches_expectation"] for c in res["cases"].values()) else "mismatch"
    except Unsupported as e:
        res, status = {"unsupported": str(e)}, "unsupported"
    except (RuntimeError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        res, status = {"infrastructure_failure": str(e)}, "infrastructure_failure"
    res["status"] = status
    text = json.dumps(res, indent=2, sort_keys=True)
    if a.out:
        a.out.write_text(text + "\n")
    print(text if not a.out else f"{status}: wrote {a.out}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
