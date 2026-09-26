#!/usr/bin/env python3
"""GPU-free checks of the stage-12 rewrite on the preview fixture; writes local_checks.json.

    docker run --rm -v "$PWD":/w veritile-interp:1 bash -c \
      'pip install -q numpy && python3 /w/bench/optimizations/inductor_divmod/local_checks.py'

What it checks:
1. `triton.compile` of every variant to PTX for sm_89 (the L4's architecture) with num_warps=4. It records
   the integer-division and mul.hi opcode counts. The else branch keeps the original, so B0's opcodes appear
   in every variant.
2. Under `TRITON_INTERPRET=1`: `_vt_fast_divmod` against Python `divmod` on edge divisors and edge values.
3. Under `TRITON_INTERPRET=1`: each variant's output against B0's output, bitwise, for several shapes and two
   XBLOCK values. The oracle is B0 and not torch, because the interpreter's fp32 -> bf16 store differs from
   torch's rounding in the add segments (1 ulp). That difference is identical for all variants.

No GPU is used, so nothing here is a performance measurement.
"""
import json
import os
import random
import re
import sys
import importlib.util
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

_HELPER = None
from triton.backends.compiler import GPUTarget  # noqa: E402

import inductor_divmod as R  # noqa: E402

SRC = (HERE / "fixtures/pointwise_cat_preview.py").read_text()
OPS = ("div.s64", "rem.s64", "div.u64", "rem.u64", "div.s32", "rem.s32", "div.u32", "rem.u32", "mul.hi.u32")


def standalone(src: str, tag: str):
    s = src.replace("triton_helpers.set_driver_to_cpu()", "")
    s = re.sub(r"@triton_heuristics\.pointwise\((?:.|\n)*?\n\)\n", "", s)
    s = s.replace("Placeholder.KERNEL_NAME", "kern")
    p = Path(f"/tmp/vt_{tag}.py")
    p.write_text(s)
    spec = importlib.util.spec_from_file_location(f"vt_{tag}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ptx_counts(mods) -> dict:
    sig = dict(re.findall(r"'(\w+)': '([^']+)'", re.search(r"'signature': \{([^}]*)\}", SRC).group(1)))
    out = {}
    for v, m in mods.items():
        try:
            k = triton.compile(triton.compiler.ASTSource(m.kern, sig, constexprs={"XBLOCK": 1024}),
                               target=GPUTarget("cuda", 89, 32), options={"num_warps": 4})
            ptx = k.asm["ptx"]
            out[v] = {"compiled": True, "ptx_lines": ptx.count("\n"),
                      "ops": {o: len(re.findall(rf"\b{re.escape(o)}\b", ptx)) for o in OPS}}
        except Exception as e:  # noqa: BLE001
            out[v] = {"compiled": False, "error": repr(e)[:2000]}
    return out


def helper_sweep(helper) -> dict:
    global _HELPER
    _HELPER = helper

    @triton.jit
    def probe(xp, dp, qp, rp, N: tl.constexpr):
        i = tl.arange(0, N)
        q, r = _HELPER(tl.load(xp + i), tl.load(dp))
        tl.store(qp + i, q)
        tl.store(rp + i, r)

    ds = [1, 2, 3, 5, 7, 16, 255, 256, 257, 1000, 4096, 5120, 12288, 65535, 65536, 2 ** 20 + 1,
          2 ** 30, 2 ** 30 + 1, 2 ** 31 - 2, 2 ** 31 - 1]
    rng = random.Random(0)
    xs = sorted({0, 1, 2, 3, 2 ** 31 - 1, 2 ** 31 - 2, 2 ** 30, 2 ** 16, 2 ** 16 - 1} |
                {rng.randrange(2 ** 31) for _ in range(55)})[:64]
    xs += [0] * (64 - len(xs))
    bad = []
    for d in ds:
        q = torch.empty(64, dtype=torch.int32)
        r = torch.empty(64, dtype=torch.int32)
        probe[(1,)](torch.tensor(xs, dtype=torch.int32), torch.tensor([d], dtype=torch.int64), q, r, 64)
        bad += [(x, d) for x, qq, rr in zip(xs, q.tolist(), r.tolist()) if (qq, rr) != divmod(x, d)]
    return {"divisors": len(ds), "values": 64, "mismatches": len(bad), "examples": bad[:5]}


def equivalence(mods) -> dict:
    cases = [(3, (5, 3, 2), (4, 1, 6)), (7, (16, 8, 8), (16, 8, 8)), (2, (1, 1, 1), (1, 1, 1)),
             (5, (64, 32, 32), (64, 32, 32)), (4, (100, 12, 36), (7, 9, 11)), (9, (2048, 256, 256), (2048, 256, 256))]
    rows = []
    for n, (a1, b1, c1), (a2, b2, c2) in cases:
        g = torch.Generator().manual_seed(n)
        ws = (a1, b1, c1, c1, a2, b2, c2, c2)
        ins = [torch.randn(n, w, generator=g).to(torch.bfloat16) for w in ws]
        W = a1 + b1 + c1 + a2 + b2 + c2
        ks = [W, a1, c1, b1, a2, b2, c2]  # ks mapping read off the emitted index expressions
        row = {"n": n, "widths": ws, "W": W}
        for XB in (64, 1024):
            outs = {}
            for v, m in mods.items():
                o = torch.full((n, W), float("nan"), dtype=torch.bfloat16)
                m.kern[(triton.cdiv(n * W, XB),)](*ins, o, *ks, n * W, XBLOCK=XB)
                outs[v] = o.view(torch.int16)
            row[f"XBLOCK={XB}"] = {v: bool(torch.equal(o, outs["B0"])) for v, o in outs.items() if v != "B0"}
            ref = torch.cat([torch.cat([ins[0], ins[1], ins[2] + ins[3]], -1),
                             torch.cat([ins[4], ins[5], ins[6] + ins[7]], -1)], -1).view(torch.int16)
            copy = torch.ones(W, dtype=torch.bool)
            copy[a1 + b1:a1 + b1 + c1] = False
            copy[W - c2:] = False
            row[f"XBLOCK={XB}"]["B0_copy_segments_equal_torch"] = bool(torch.equal(outs["B0"][:, copy], ref[:, copy]))
        rows.append(row)
    return {"rows": rows, "all_equal": all(all(r[k].values()) for r in rows for k in r if k.startswith("XBLOCK"))}


if __name__ == "__main__":
    compiled = {v: standalone(R.rewrite(SRC, v)[0], "ptx_" + v) for v in R.VARIANTS}
    ptx = ptx_counts(compiled)
    os.environ["TRITON_INTERPRET"] = "1"   # read by @triton.jit at decoration time
    mods = {v: standalone(R.rewrite(SRC, v)[0], "interp_" + v) for v in R.VARIANTS}
    rec = {"torch": torch.__version__, "triton": triton.__version__,
           "fixture_sha": R.sha(SRC), "rewrites": {v: R.rewrite(SRC, v)[1] for v in R.VARIANTS},
           "helper_sweep": helper_sweep(mods["F"]._vt_fast_divmod),
           "equivalence": equivalence(mods)}
    rec["ptx_sm89"] = ptx
    (HERE / "local_checks.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("helper_sweep", "ptx_sm89")}, indent=1))
    print("equivalence all_equal:", rec["equivalence"]["all_equal"])
