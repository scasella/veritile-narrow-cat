#!/usr/bin/env python3
"""GPU-free: compile B0, guarded N (stage 12) and N1 for sm_89 and record full-kernel SASS and registers.
Writes local_checks.json. Analysis only; the timed artifacts are recorded on the GPU host by the harness.

    docker run --rm -v "$PWD":/w veritile-interp:1 python3 /w/bench/optimizations/inductor_narrow/local_checks.py
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402

import inductor_divmod as R  # noqa: E402
import inductor_narrow as NR  # noqa: E402

SRC = NR.PROVED_SRC
BIN = Path(triton.__file__).parent / "backends/nvidia/bin"


def load(src, tag):
    s = src.replace("triton_helpers.set_driver_to_cpu()", "").replace("triton_helpers.set_driver_to_gpu()", "")
    s = re.sub(r"@triton_heuristics\.pointwise\((?:.|\n)*?\n\)\n", "", s).replace("Placeholder.KERNEL_NAME", "kern")
    p = Path(tempfile.gettempdir()) / f"n1_{tag}.py"
    p.write_text(s)
    spec = importlib.util.spec_from_file_location(f"n1_{tag}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stats(src, tag, xblock, warps):
    sig = dict(re.findall(r"'(\w+)': '([^']+)'", re.search(r"'signature': \{([^}]*)\}", src).group(1)))
    k = triton.compile(triton.compiler.ASTSource(load(src, tag).kern, sig, constexprs={"XBLOCK": xblock}),
                       target=GPUTarget("cuda", 89, 32), options={"num_warps": warps})
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(k.asm["cubin"])
    sass = subprocess.run([str(BIN / "nvdisasm"), "-c", f.name], capture_output=True, text=True).stdout
    ops = re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", sass)
    res = subprocess.run([str(BIN / "cuobjdump"), "-res-usage", f.name], capture_output=True, text=True).stdout
    return {"instructions": len(ops), "IMAD": sum(o.startswith("IMAD") for o in ops),
            "I2F_F2I_MUFU": sum(o.split(".")[0] in ("I2F", "F2I", "MUFU") for o in ops),
            "CALL": sum(o.startswith("CALL") for o in ops),
            "regs": int(re.search(r"REG:(\d+)", res).group(1)), "ptx_ks_params": re.findall(r"\.param \.(\w+) kern_param_(?:9|1[0-5])\b", k.asm["ptx"])[:7]}


if __name__ == "__main__":
    n1_src, n1_rec = NR.rewrite_n1(SRC, {"eligible": True, "checks": {"H1": True, "H2": True, "H3": True, "H4": True}})
    variants = {"B0": SRC, "N_guarded": R.rewrite(SRC, "N")[0], "N1": n1_src}
    out = {"n1_rewrite": {k: v for k, v in n1_rec.items() if k != "eligibility"}, "sm_89": {}}
    for name, src in variants.items():
        out["sm_89"][name] = {f"XBLOCK={xb},warps={w}": stats(src, name, xb, w) for xb, w in ((1024, 4), (512, 4), (256, 4))}
    (HERE / "local_checks.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, r in out["sm_89"].items():
        print(name, {k: (v["instructions"], v["regs"], v["CALL"]) for k, v in r.items()}, r["XBLOCK=1024,warps=4"]["ptx_ks_params"])
