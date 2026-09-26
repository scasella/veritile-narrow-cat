#!/usr/bin/env python3
"""Extract the indexing/mask/payload IR of an Inductor pointwise kernel (stage 13).

`extract(src)` parses the kernel body that Inductor emitted (Python AST, no execution) into a small IR. Temporaries
are inlined from the observable operations: the one store and the loads its value reaches. Anything the parser does
not recognise raises `Unsupported`, so a caller can refuse to rewrite. The same IR feeds:
  * `eval_int` / `eval_bool` / `eval_float`: a Python model of Triton's integer semantics, per width mode;
  * `to_lean`: the Lean terms in bench/optimizations/inductor_narrow/NarrowIR.lean (generated; binding-tested).

Integer semantics modelled (Triton 3.8, as lowered to arith/LLVM):
  * types int32 / int64. A binary op on int32 and int64 sign-extends to int64. A Python int literal is a
    32-bit constant that takes the other operand's type (it fits both). `tl.full(.., tl.int64)` is a strong int64.
  * every add/mul result wraps to its type's width (two's complement); `.to(t)` wraps or sign-extends to t.
  * `//` and `%` are truncating (arith.divsi / arith.remsi), i.e. Lean `Int.tdiv` / `Int.tmod`.
  * comparisons compare the promoted values; `&` on masks is boolean and.
  * size scalars ks_i: in mode "B0" (as emitted) they are int64 arguments; in mode "N1" they are int32 arguments,
    so the launcher passes the wrapped value.
Payload (float) values are symbolic: a load returns ("mem", pointer, offset) when enabled and ("zero",) otherwise;
`+` and `where` are kept as constructors. So two modes agree on the payload iff they agree on every enabled load's
pointer and offset and on every selector.
"""
from __future__ import annotations

import ast
import re
import sys

I32, I64 = "i32", "i64"


class Unsupported(Exception):
    pass


# ---------------------------------------------------------------------------- IR constructors (tuples)
# int:   ("xindex",) ("ks", i) ("lit", v) ("slit", v, ty) ("cast", e, ty) ("add", a, b) ("mul", a, b)
#        ("tmod", a, b) ("tdiv", a, b)
# bool:  ("lt", a, b) ("ge", a, b) ("and", a, b) ("xmask",)
# float: ("load", ptr_index, off, mask) ("fadd", a, b) ("where", c, a, b) ("fzero",)


def _dtype(node) -> str:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "tl":
        if node.attr in ("int32", "int64"):
            return {"int32": I32, "int64": I64}[node.attr]
    raise Unsupported(f"dtype {ast.dump(node)[:80]}")


def extract(src: str) -> dict:
    """Return {"store": (ptr, off, val, mask), "loads": [...], "dead": [...], "kernel_args": [...]}."""
    tree = ast.parse(re.sub(r"def Placeholder\.KERNEL_NAME", "def KERNEL", src))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "_vt_fast_divmod"]
    if len(fns) != 1:
        raise Unsupported("expected one kernel function")
    fn = fns[0]
    args = [a.arg for a in fn.args.args]
    ptrs = [a for a in args if re.fullmatch(r"(in|out)_ptr\d+", a)]
    ks = [a for a in args if re.fullmatch(r"ks\d+", a)]
    stmts = fn.body
    want = ["xoffset = tl.program_id(0) * XBLOCK", "xindex = xoffset + tl.arange(0, XBLOCK)[:]",
            "xmask = xindex < xnumel"]
    if [ast.unparse(s) for s in stmts[:3]] != want:
        raise Unsupported("prologue")
    env: dict = {"xindex": ("xindex",), "xmask": ("xmask",)}
    kinds: dict = {"xindex": "int", "xmask": "bool"}
    used: set = set()
    store = None
    loads: list = []

    def ie(n):  # integer expression
        if isinstance(n, ast.Name):
            if n.id in ks:
                return ("ks", int(n.id[2:]))
            if kinds.get(n.id) != "int":
                raise Unsupported(f"int name {n.id}")
            used.add(n.id)
            return env[n.id]
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            return ("lit", n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant):
            return ("lit", -n.operand.value)
        if isinstance(n, ast.BinOp):
            op = {ast.Add: "add", ast.Mult: "mul", ast.Mod: "tmod", ast.FloorDiv: "tdiv"}.get(type(n.op))
            if op is None:
                raise Unsupported(f"binop {type(n.op).__name__}")
            return (op, ie(n.left), ie(n.right))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            f = n.func
            if f.attr == "to" and len(n.args) == 1:
                return ("cast", ie(f.value), _dtype(n.args[0]))
            if ast.unparse(f) == "tl.broadcast_to" and ast.unparse(n.args[1]) == "[XBLOCK]":
                return ie(n.args[0])
            if ast.unparse(f) == "tl.full" and ast.unparse(n.args[0]) == "[1]":
                v = n.args[1]
                if isinstance(v, ast.Constant) and isinstance(v.value, int):
                    return ("slit", v.value, _dtype(n.args[2]))
        raise Unsupported(f"int expr {ast.unparse(n)[:80]}")

    def be(n):  # boolean expression
        if isinstance(n, ast.Name):
            if kinds.get(n.id) != "bool":
                raise Unsupported(f"bool name {n.id}")
            used.add(n.id)
            return env[n.id]
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            op = {ast.Lt: "lt", ast.GtE: "ge"}.get(type(n.ops[0]))
            if op is None:
                raise Unsupported("compare op")
            return (op, ie(n.left), ie(n.comparators[0]))
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd):
            return ("and", be(n.left), be(n.right))
        raise Unsupported(f"bool expr {ast.unparse(n)[:80]}")

    def fe(n):  # float expression
        if isinstance(n, ast.Name):
            if kinds.get(n.id) != "float":
                raise Unsupported(f"float name {n.id}")
            used.add(n.id)
            return env[n.id]
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return ("fadd", fe(n.left), fe(n.right))
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.where":
            return ("where", be(n.args[0]), fe(n.args[1]), fe(n.args[2]))
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.full":
            if (isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, float) and n.args[1].value == 0.0
                    and ast.unparse(n.args[0]) != "[1]"):
                return ("fzero",)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "to":
            inner = n.func.value
            if isinstance(inner, ast.Call) and ast.unparse(inner.func) == "tl.load":
                return load(inner)
        raise Unsupported(f"float expr {ast.unparse(n)[:80]}")

    def load(c: ast.Call):
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        if kw.get("other") != "0.0" or set(kw) - {"other", "eviction_policy"}:
            raise Unsupported("load keywords")
        a0 = c.args[0]
        if not (isinstance(a0, ast.BinOp) and isinstance(a0.op, ast.Add) and isinstance(a0.left, ast.Name)
                and a0.left.id in ptrs):
            raise Unsupported("load address form")
        node = ("load", ptrs.index(a0.left.id), ie(a0.right), be(c.args[1]))
        loads.append(node)
        return node

    def classify(n) -> str:
        for kind, f in (("float", fe), ("bool", be), ("int", ie)):
            saved, used_before = len(loads), set(used)
            try:
                f(n)
                return kind
            except (Unsupported, KeyError):
                pass
            finally:
                del loads[saved:]
                used.clear()
                used.update(used_before)
        raise Unsupported(f"statement {ast.unparse(n)[:100]}")

    for s in stmts[3:]:
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
            name = s.targets[0].id
            kind = classify(s.value)
            kinds[name] = kind
            env[name] = {"int": ie, "bool": be, "float": fe}[kind](s.value)
            continue
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and ast.unparse(s.value.func) == "tl.store":
            if store is not None:
                raise Unsupported("more than one store")
            c = s.value
            a0 = c.args[0]
            if not (isinstance(a0, ast.BinOp) and isinstance(a0.op, ast.Add) and a0.left.id in ptrs):
                raise Unsupported("store address")
            store = (ptrs.index(a0.left.id), ie(a0.right), fe(c.args[1]), be(c.args[2]))
            continue
        raise Unsupported(f"statement {ast.unparse(s)[:100]}")
    if store is None:
        raise Unsupported("no store")
    # which loads does the stored value reach
    reach: list = []

    def walk(n):
        if n[0] == "load":
            reach.append(n)
        for c in n[1:]:
            if isinstance(c, tuple):
                walk(c)
    walk(store[2])
    dead = [k for k in env if k not in used and k not in ("xindex", "xmask")]
    return {"store": store, "loads": reach, "dead": dead, "kernel_args": args, "pointers": ptrs, "ks": ks,
            "dead_kinds": {k: kinds[k] for k in dead}, "dead_ir": {k: env[k] for k in dead}}


# ---------------------------------------------------------------------------- Python semantics


def wrap(v: int, ty) -> int:
    if ty is None:
        return v
    m = 32 if ty == I32 else 64
    v &= (1 << m) - 1
    return v - (1 << m) if v >= 1 << (m - 1) else v


def join(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return I64 if I64 in (a, b) else I32


def tdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def tmod(a, b):
    return a - b * tdiv(a, b)


def eval_int(n, mode: str, xi: int, ksv: dict) -> tuple:
    """(value, type) in width mode 'B0' (ks int64) or 'N1' (ks int32)."""
    t = n[0]
    if t == "xindex":
        return wrap(xi, I32), I32
    if t == "ks":
        ty = I64 if mode == "B0" else I32
        return wrap(ksv[n[1]], ty), ty
    if t == "lit":
        return n[1], None
    if t == "slit":
        return wrap(n[1], n[2]), n[2]
    if t == "cast":
        v, _ = eval_int(n[1], mode, xi, ksv)
        return wrap(v, n[2]), n[2]
    a, ta = eval_int(n[1], mode, xi, ksv)
    b, tb = eval_int(n[2], mode, xi, ksv)
    ty = join(ta, tb)
    if t == "add":
        return wrap(a + b, ty), ty
    if t == "mul":
        return wrap(a * b, ty), ty
    if t == "tdiv":
        return wrap(tdiv(a, b), ty), ty
    if t == "tmod":
        return wrap(tmod(a, b), ty), ty
    raise Unsupported(t)


def exact(n, xi: int, ksv: dict) -> int:
    """Unbounded integer value (no wrapping), with ks given exactly."""
    t = n[0]
    if t == "xindex":
        return xi
    if t == "ks":
        return ksv[n[1]]
    if t in ("lit",):
        return n[1]
    if t == "slit":
        return n[1]
    if t == "cast":
        return exact(n[1], xi, ksv)
    a, b = exact(n[1], xi, ksv), exact(n[2], xi, ksv)
    return {"add": a + b, "mul": a * b, "tdiv": tdiv(a, b) if b else 0, "tmod": tmod(a, b) if b else a}[t]


def eval_bool(n, mode, xi, ksv, xnumel) -> bool:
    t = n[0]
    if t == "xmask":
        return wrap(xi, I32) < wrap(xnumel, I32)
    if t == "and":
        return eval_bool(n[1], mode, xi, ksv, xnumel) and eval_bool(n[2], mode, xi, ksv, xnumel)
    a, _ = eval_int(n[1], mode, xi, ksv)
    b, _ = eval_int(n[2], mode, xi, ksv)
    return a < b if t == "lt" else a >= b


def eval_float(n, mode, xi, ksv, xnumel):
    t = n[0]
    if t == "fzero":
        return ("zero",)
    if t == "load":
        if eval_bool(n[3], mode, xi, ksv, xnumel):
            return ("mem", n[1], eval_int(n[2], mode, xi, ksv)[0])
        return ("zero",)
    if t == "fadd":
        return ("add", eval_float(n[1], mode, xi, ksv, xnumel), eval_float(n[2], mode, xi, ksv, xnumel))
    if t == "where":
        c = eval_bool(n[1], mode, xi, ksv, xnumel)
        return eval_float(n[2] if c else n[3], mode, xi, ksv, xnumel)
    raise Unsupported(t)


def observe(ir, mode, xi, ksv, xnumel) -> dict:
    """Everything a lane makes observable: store mask/address/value and every load's mask and enabled address."""
    ptr, off, val, mask = ir["store"]
    loads = [(n[1], eval_bool(n[3], mode, xi, ksv, xnumel)) for n in ir["loads"]]
    return {"store_mask": eval_bool(mask, mode, xi, ksv, xnumel),
            "store_addr": eval_int(off, mode, xi, ksv)[0],
            "store_val": eval_float(val, mode, xi, ksv, xnumel),
            "loads": [(p, m, eval_int(n[2], mode, xi, ksv)[0] if m else None)
                      for (p, m), n in zip(loads, ir["loads"])]}


# ---------------------------------------------------------------------------- Lean emission


def to_lean(n) -> str:
    t = n[0]
    if t == "xindex":
        return "IE.xindex"
    if t == "ks":
        return f"(IE.ks {n[1]})"
    if t == "lit":
        return f"(IE.lit ({n[1]}))"
    if t == "slit":
        return f"(IE.slit ({n[1]}) Ty.{n[2]})"
    if t == "cast":
        return f"(IE.cast {to_lean(n[1])} Ty.{n[2]})"
    if t in ("add", "mul", "tmod", "tdiv"):
        return f"(IE.{t} {to_lean(n[1])} {to_lean(n[2])})"
    if t == "xmask":
        return "BE.xmask"
    if t in ("lt", "ge"):
        return f"(BE.{t} {to_lean(n[1])} {to_lean(n[2])})"
    if t == "and":
        return f"(BE.and {to_lean(n[1])} {to_lean(n[2])})"
    if t == "fzero":
        return "FE.zero"
    if t == "load":
        return f"(FE.load {n[1]} {to_lean(n[2])} {to_lean(n[3])})"
    if t == "fadd":
        return f"(FE.add {to_lean(n[1])} {to_lean(n[2])})"
    if t == "where":
        return f"(FE.where {to_lean(n[1])} {to_lean(n[2])} {to_lean(n[3])})"
    raise Unsupported(t)


BEGIN = "-- BEGIN GENERATED (scripts/inductor_ir.py, fixtures/gpu_emitted_B0.py)"
END = "-- END GENERATED"


def lean_block(src: str) -> str:
    """The generated section of NarrowCat.lean: the stored payload tree of the emitted kernel."""
    ir = extract(src)
    if ir["store"][1] != ("xindex",) or ir["store"][3] != ("xmask",):
        raise Unsupported("store is not out_ptr + xindex under xmask")
    return (f"{BEGIN}\n/-- `tmp74`, the stored value, with every temporary inlined (8 loads). -/\n"
            f"def catValue : FE :=\n  {to_lean(ir['store'][2])}\n{END}")


def splice(lean_text: str, block: str) -> str:
    a, b = lean_text.index(BEGIN), lean_text.index(END) + len(END)
    return lean_text[:a] + block + lean_text[b:]


if __name__ == "__main__" and "--write-lean" in sys.argv[1:2]:
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    src = (root / "bench/optimizations/inductor_divmod/fixtures/gpu_emitted_B0.py").read_text()
    lf = root / "bench/optimizations/inductor_narrow/NarrowCat.lean"
    lf.write_text(splice(lf.read_text(), lean_block(src)))
    print(f"wrote generated block into {lf}")
elif __name__ == "__main__":
    import sys
    ir = extract(open(sys.argv[1]).read())
    print("loads reached:", len(ir["loads"]), "dead:", ir["dead"], ir["dead_kinds"])
    print("store:", to_lean(ir["store"][1]), "mask", to_lean(ir["store"][3]))
