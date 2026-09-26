#!/usr/bin/env python3
"""N1: provably-safe 32-bit size scalars for Inductor's dynamic-cat kernel (stage 13).

Inductor 2.14 declares every `ks*` size argument `i64` (`signature_to_meta`). The reason given in its source is
that a product of symbols may not fit in 32 bits even when each symbol does. N1 re-declares the seven `ks*`
arguments of one kernel class `i32`. It changes only those seven `triton_meta` signature entries:
- the kernel body is byte-identical;
- the grid is unchanged;
- the wrapper still passes the same Python integers, and the launcher packs them as int32.
This is what current PyTorch `main` emits under `config.assume_32bit_indexing`. The difference is that N1
applies it only where it has been shown safe for this kernel, instead of relying on a user promise.

N1 applies only when BOTH of these hold:
 (a) Structural: the kernel's extracted IR (`inductor_ir.extract`) equals the IR of
     fixtures/gpu_emitted_B0.py, which is the IR `NarrowCat.lean` is proved about (binding-tested by
     scripts/test_inductor_narrow.py). Its signature must declare exactly ks0..ks6 as i64 and xnumel as i32.
 (b) Eligibility: Inductor's own symbolic state at codegen time establishes the theorem's hypotheses.
       H1  every ks1..ks6 expression is a size symbol whose value range has lower bound >= 0
       H2  ks0's expression expands to ks1 + ... + ks6
       H3  the kernel's numel expands to s * ks0 with s a size symbol (lower bound >= 0)
       H4  index dtype is int32, and the guard `numel <= 2147483647` is in the shape environment's guards.
           Inductor installs it (`can_use_32bit_indexing`), and Dynamo re-checks it on every call before the
           graph runs. When it fails, Dynamo recompiles, Inductor emits an int64 kernel, and (a) no longer
           holds, so the original is retained.
Then `NarrowCat.narrow_cat_equiv` gives lane-by-lane equality of every observable. No check is added at run
time, and the kernel carries no fallback body.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import inductor_ir as IR  # noqa: E402

PROVED_SRC = (REPO / "bench/optimizations/inductor_divmod/fixtures/gpu_emitted_B0.py").read_text()
PROVED_IR = IR.extract(PROVED_SRC)
KS = [f"ks{i}" for i in range(7)]
INT_MAX = 2 ** 31 - 1


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def structural(src: str) -> tuple[bool, str]:
    try:
        ir = IR.extract(src)
    except IR.Unsupported as e:
        return False, f"extract: {e}"
    if ir["store"] != PROVED_IR["store"]:
        return False, "IR differs from the proved kernel"
    m = re.search(r"'signature': \{([^}]*)\}", src)
    sig = dict(re.findall(r"'(\w+)': '([^']+)'", m.group(1))) if m else {}
    if [k for k in sig if k.startswith("ks")] != KS or any(sig[k] != "i64" for k in KS):
        return False, "size-scalar signature is not ks0..ks6 : i64"
    if sig.get("xnumel") != "i32":
        return False, "xnumel is not i32"
    return True, "IR equals the proved kernel"


def eligibility(kernel) -> dict:
    """Read H1-H4 from Inductor's symbolic state at codegen time (called inside define_kernel)."""
    import sympy
    from torch._inductor.virtualized import V
    sv = V.graph.sizevars
    se = sv.shape_env
    rec = {"checks": {}}
    try:
        name_of = {v: k for k, v in kernel.args.sizevars.items()}          # "ks0" -> sympy expr
        exprs = {k: name_of[k] for k in KS}
        inv = getattr(sv, "inv_precomputed_replacements", {})
        full = {k: inv.get(e, e) for k, e in exprs.items()}                  # ps0 -> its definition
        rec["ks_exprs"] = {k: str(v) for k, v in full.items()}

        def lower_ok(e):
            if not isinstance(e, sympy.Symbol):
                return False
            r = se.var_to_range.get(e)
            return r is not None and r.lower >= 0
        rec["checks"]["H1"] = all(lower_ok(full[k]) for k in KS[1:])
        rec["checks"]["H2"] = sympy.expand(full["ks0"] - sum(full[k] for k in KS[1:])) == 0
        numel = sympy.expand(kernel.numels["x"])
        rec["numel"] = str(numel)
        q = sympy.simplify(numel / full["ks0"])
        rec["checks"]["H3"] = bool(isinstance(q, sympy.Symbol) and lower_ok(q)
                                   and sympy.expand(q * full["ks0"] - numel) == 0)
        rec["n_symbol"] = str(q)
        guards = [str(g.expr) for g in se.guards]
        want = str(sympy.Le(numel, INT_MAX))
        rec["guard_text"] = want
        rec["checks"]["H4"] = bool(str(kernel.index_dtype) == "tl.int32" and want in guards)
    except Exception as e:  # noqa: BLE001 - any failure means "not shown eligible"
        rec["error"] = repr(e)[:400]
        rec["checks"] = {k: rec["checks"].get(k, False) for k in ("H1", "H2", "H3", "H4")}
    rec["eligible"] = all(rec["checks"].get(k) for k in ("H1", "H2", "H3", "H4"))
    return rec


def rewrite_n1(src: str, elig: dict | None) -> tuple[str, dict]:
    rec = {"variant": "N1", "before": sha(src), "rewritten": False}
    ok, why = structural(src)
    rec["structural"] = why
    if not ok:
        rec["reason"] = why
        return src, rec
    if not (elig and elig.get("eligible")):
        rec["reason"] = f"eligibility not established: {elig and elig.get('checks')}"
        return src, rec
    m = re.search(r"'signature': \{([^}]*)\}", src)
    sig_old = m.group(0)
    sig_new = sig_old
    for k in KS:
        sig_new = sig_new.replace(f"'{k}': 'i64'", f"'{k}': 'i32'")
    new = src.replace(sig_old, sig_new, 1)
    rec.update({"rewritten": True, "after": sha(new), "eligibility": elig})
    return new, rec
