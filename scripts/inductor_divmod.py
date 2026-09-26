#!/usr/bin/env python3
"""Guarded source rewrite of Inductor's pointwise Triton kernels (stage 12).

Inductor lowers `cat` along a dynamic last dimension to one pointwise kernel whose
index prologue is

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    ...
    x0 = (xindex % ks0)           # runtime divisor, passed as i64
    x1 = xindex // ks0

`rewrite(src, variant)` returns the kernel source with the body duplicated under a
scalar guard:

    <prologue, unchanged>
    if <guard>:
        <variant body>            # every local renamed with suffix _v
    else:
        <original body>           # byte-for-byte the emitted body

Variants (the 2x2 isolation):
  "B0"  no rewrite (the source is returned unchanged)
  "N"   integer-width narrowing: every i64 size scalar ksK is cast to int32 in the
        variant body; the division stays `%` / `//`. Guard: 0 <= ksK < 2^31 for all K.
  "F"   fast divmod: the two divmod lines become `_vt_fast_divmod(xindex, ks)`
        (FastDiv, IntDivider form, constants computed in-kernel once per program), with
        results cast back to the original dtype; all other types unchanged.
        Guard: 0 < ks < 2^31.
  "NF"  both. Guard: the conjunction.

Payload arithmetic, masks, loads/stores, eviction policies, launch signature and
grid are untouched. Anything that does not match the expected shape exactly is
returned unchanged, and the returned record says why.

In-kernel constants (proof: VeriTile bench/optimizations/inductor_divmod/InductorDivmod.lean):
  s = #{ j < 32 : 2^j < d }                     (= FastDiv.shiftFor d for 0 < d <= 2^31)
  m = (2^32 * (2^s - d)) // d + 1               (= FastDiv.magic d; int64, no overflow for d < 2^31)
  q = (umulhi(x, m) + x) >> s                   (= x / d for 0 <= x < 2^31, FastDiv.bitvec_quotient)
  r = x - q * d                                 (= x % d)
Proved for active lanes (0 <= xindex < xnumel <= 2^31 - 1); masked-off lanes compute
unused values.
"""
from __future__ import annotations

import hashlib
import re

VARIANTS = ("B0", "N", "F", "NF")

HELPER = '''
@triton.jit
def _vt_fast_divmod(x, d):
    # FastDiv (IntDivider form). Caller guarantees 0 < d < 2^31; exact for 0 <= x < 2^31.
    d64 = d.to(tl.int64)
    s = tl.full([], 0, tl.int64)
    for j in tl.static_range(32):
        s += (d64 > (1 << j)).to(tl.int64)
    one = tl.full([], 1, tl.int64)
    m = ((one << 32) * ((one << s) - d64)) // d64 + 1
    xu = x.to(tl.uint32)
    q = ((tl.umulhi(xu, m.to(tl.uint32)) + xu) >> s.to(tl.uint32)).to(tl.int32)
    r = x.to(tl.int32) - q * d.to(tl.int32)
    return q, r

'''

_SIG = re.compile(r"'signature': \{([^}]*)\}")
_DEF = re.compile(r"^def (\S+)\((.*)\):\s*$")
_MOD = re.compile(r"^    (\w+) = \(xindex % (ks\d+)\)$")
_DIV = re.compile(r"^    (\w+) = xindex // (ks\d+)$")
_LOCAL = re.compile(r"\b((?:tmp|x)\d+|ks\d+)\b")
PROLOGUE = ["    xoffset = tl.program_id(0) * XBLOCK",
            "    xindex = xoffset + tl.arange(0, XBLOCK)[:]",
            "    xmask = xindex < xnumel"]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _signature(src: str) -> dict:
    m = _SIG.search(src)
    if not m:
        return {}
    return dict(re.findall(r"'(\w+)': '([^']+)'", m.group(1)))


def rewrite(src: str, variant: str) -> tuple[str, dict]:
    """Return (new_source, record). record['rewritten'] is False on any mismatch."""
    rec = {"variant": variant, "before": sha(src), "rewritten": False}
    if variant == "B0":
        rec["reason"] = "baseline"
        return src, rec
    if variant not in VARIANTS:
        raise ValueError(variant)
    sig = _signature(src)
    if "@triton_heuristics.pointwise" not in src or sig.get("xnumel") != "i32":
        rec["reason"] = "not a pointwise kernel with i32 xnumel"
        return src, rec
    lines = src.split("\n")
    try:
        d = next(i for i, ln in enumerate(lines) if _DEF.match(ln))
    except StopIteration:
        rec["reason"] = "no kernel def"
        return src, rec
    if lines[d + 1:d + 4] != PROLOGUE:
        rec["reason"] = "prologue differs"
        return src, rec
    end = len(lines)
    while end > d and lines[end - 1].strip() == "":
        end -= 1
    body = lines[d + 4:end]
    if not body or any(ln and not ln.startswith("    ") for ln in body):
        rec["reason"] = "body not a flat indented block"
        return src, rec
    ks_params = sorted(k for k, t in sig.items() if re.fullmatch(r"ks\d+", k) and t == "i64")
    mods = [(i, _MOD.match(ln)) for i, ln in enumerate(body) if _MOD.match(ln)]
    divs = [(i, _DIV.match(ln)) for i, ln in enumerate(body) if _DIV.match(ln)]
    want_f = variant in ("F", "NF")
    if want_f:
        if len(mods) != 1 or len(divs) != 1 or mods[0][1].group(2) != divs[0][1].group(2):
            rec["reason"] = f"divmod pattern not unique ({len(mods)} mod, {len(divs)} div)"
            return src, rec
        (im, mm), (idv, md) = mods[0], divs[0]
        a, b, ks = mm.group(1), md.group(1), mm.group(2)
        lo, hi = sorted((im, idv))
        between = "\n".join(body[lo + 1:hi])
        if re.search(rf"\b({a}|{b})\b", between):
            rec["reason"] = "divmod results used between the two lines"
            return src, rec
    if variant in ("N", "NF") and not ks_params:
        rec["reason"] = "no i64 size scalars to narrow"
        return src, rec

    var = []
    guards = []
    if variant in ("N", "NF"):
        for k in ks_params:
            var.append(f"        {k}_v = {k}.to(tl.int32)")
        guards += [f"({k} >= 0) & ({k} < 2147483648)" for k in ks_params]
    if want_f:
        guards.append(f"({ks} > 0) & ({ks} < 2147483648)")
    for i, ln in enumerate(body):
        if want_f and i == min(im, idv):
            dsrc = f"{ks}_v" if variant == "NF" else ks
            if variant == "F":
                # xindex (int32) % ks (i64) is int64 in the original; keep that dtype downstream
                dt = "tl.int64" if sig.get(ks) == "i64" else "tl.int32"
                var.append(f"        {b}_q, {a}_r = _vt_fast_divmod(xindex, {ks})")
                var.append(f"        {a}_v = ({a}_r).to({dt})")
                var.append(f"        {b}_v = ({b}_q).to({dt})")
            else:
                var.append(f"        {b}_v, {a}_v = _vt_fast_divmod(xindex, {dsrc})")
            continue
        if want_f and i == max(im, idv):
            continue
        new = ln
        if variant == "F":
            # rename locals only; ks scalars keep their original (i64) values
            new = re.sub(r"\b((?:tmp|x)\d+)\b", r"\1_v", ln)
        else:
            new = _LOCAL.sub(r"\1_v", ln)
        var.append("    " + new)
    guard = " & ".join(f"({g})" for g in guards)
    out = lines[:d + 4] + [f"    if {guard}:"] + var + ["    else:"] + ["    " + ln for ln in body] + lines[end:]
    # helper goes right before the decorator that precedes the kernel def
    dec = max(i for i in range(d) if lines[i].startswith("@triton_heuristics."))
    out = out[:dec] + HELPER.split("\n") + out[dec:]
    new_src = "\n".join(out)
    rec.update({"rewritten": True, "after": sha(new_src), "guard": guard,
                "divmod": {"mod": a, "div": b, "divisor": ks} if want_f else None,
                "narrowed": ks_params if variant in ("N", "NF") else [],
                "variant_lines": len(var)})
    return new_src, rec


if __name__ == "__main__":
    import sys
    s = open(sys.argv[1]).read()
    for v in VARIANTS:
        ns, r = rewrite(s, v)
        print(v, r)
        if len(sys.argv) > 2 and v == sys.argv[2]:
            print(ns)
