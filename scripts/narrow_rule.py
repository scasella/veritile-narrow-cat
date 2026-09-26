#!/usr/bin/env python3
"""Reference implementation of the selective 32-bit size-argument rule (bench/optimizations/inductor_narrow/RULE.md).

`check(src, kernel=None)` returns a record with every premise C1-C4 evaluated separately, and `eligible` only if all
hold. `apply(src, record)` re-declares ks0..ks6 as i32 in the kernel's triton_meta signature. The kernel body is left
byte-identical.

This is stricter than stage 13's `inductor_narrow.py`:
- C1 compares the dead statements too, the exact argument list, the full signature, the heuristic and the grid type.
- C2 requires range lower bounds >= 1 (nonempty launches), and accepts H4 as an installed guard or a statically
  known fact.
The C2 facts come only from Inductor's symbolic state (var_to_range, precomputed replacements, numel, guards), never
from size hints or example values.
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
PROVED = IR.extract(PROVED_SRC)
KS = [f"ks{i}" for i in range(7)]
INT_MAX = 2 ** 31 - 1
_SIG = re.compile(r"'signature': \{([^}]*)\}")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def signature(src: str) -> dict:
    m = _SIG.search(src)
    return dict(re.findall(r"'(\w+)': '([^']+)'", m.group(1))) if m else {}


def c1_structural(src: str) -> dict:
    r = {}
    r["pointwise_heuristic"] = "@triton_heuristics.pointwise(" in src and "fixed_config" not in src
    r["grid_1d"] = "'grid_type': 'Grid1D'" in src
    try:
        ir = IR.extract(src)
    except IR.Unsupported as e:
        return {**r, "extract": f"unsupported: {e}", "ok": False}
    r["observables_equal"] = ir["store"] == PROVED["store"]
    r["dead_code_equal"] = ir["dead_ir"] == PROVED["dead_ir"]
    r["argument_list_equal"] = ir["kernel_args"] == PROVED["kernel_args"]
    r["signature_equal"] = signature(src) == signature(PROVED_SRC)
    r["ok"] = all(v is True for v in r.values())
    return r


def c2_symbolic(kernel) -> dict:
    """Read H1', H2, H3', H4 from Inductor's state inside define_kernel/codegen (needs torch)."""
    import sympy
    from torch._inductor.virtualized import V
    sv = V.graph.sizevars
    se = sv.shape_env
    r = {}
    try:
        name_of = {v: k for k, v in kernel.args.sizevars.items()}
        full = {k: sv.inv_precomputed_replacements.get(name_of[k], name_of[k]) for k in KS}
        r["ks_exprs"] = {k: str(v) for k, v in full.items()}

        def lower_ge1(e):
            if not isinstance(e, sympy.Symbol):
                return False
            rng = se.var_to_range.get(e)
            return rng is not None and rng.lower >= 1
        r["H1'"] = all(lower_ge1(full[k]) for k in KS[1:])
        r["H2"] = sympy.expand(full["ks0"] - sum(full[k] for k in KS[1:])) == 0
        numel = sympy.expand(kernel.numels["x"])
        q = sympy.simplify(numel / full["ks0"])
        r["numel"], r["n_symbol"] = str(numel), str(q)
        r["H3'"] = bool(lower_ge1(q) and sympy.expand(q * full["ks0"] - numel) == 0)
        guard = sympy.Le(numel, INT_MAX)
        installed = str(guard) in {str(g.expr) for g in se.guards}
        static = bool(sv.statically_known_true(guard))
        r["H4_guard"] = "installed" if installed else ("statically known" if static else "absent")
        r["H4"] = str(kernel.index_dtype) == "tl.int32" and (installed or static)
    except Exception as e:  # noqa: BLE001 - any failure is "not established"
        r["error"] = repr(e)[:300]
    r["ok"] = all(r.get(h) is True for h in ("H1'", "H2", "H3'", "H4"))
    return r


def check(src: str, kernel=None) -> dict:
    rec = {"before": sha(src), "C1": c1_structural(src)}
    rec["C2"] = c2_symbolic(kernel) if kernel is not None else {"ok": False, "why": "no kernel state"}
    rec["eligible"] = rec["C1"]["ok"] and rec["C2"]["ok"]
    return rec


def apply(src: str, rec: dict) -> str:
    if not rec.get("eligible"):
        return src
    m = _SIG.search(src)
    new_sig = m.group(0)
    for k in KS:
        new_sig = new_sig.replace(f"'{k}': 'i64'", f"'{k}': 'i32'")
    out = src.replace(m.group(0), new_sig, 1)
    rec["after"] = sha(out)
    return out
