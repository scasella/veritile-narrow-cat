#!/usr/bin/env python3
"""Analysis only (no GPU): SASS instruction counts per variant branch, sm_89. Writes sass_counts.json.

    docker run --rm -v "$PWD":/w veritile-interp:1 python3 /w/bench/optimizations/inductor_divmod/sass_counts.py

Each variant's guard is replaced by `True`, so the compiler drops the else branch and the counts are for the
branch that runs under the guard. B0 is the emitted kernel. The source is fixtures/gpu_emitted_B0.py, which the
L4 run emitted and which is byte-identical in body to the preview. Configurations: XBLOCK in {1024, 512} with
num_warps = 4, the usual 1-D pointwise configs. The config Inductor autotuned to on the L4 was not recorded.
"""
import collections
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

SRC = (HERE / "fixtures/gpu_emitted_B0.py").read_text()
NVDISASM = Path(triton.__file__).parent / "backends/nvidia/bin/nvdisasm"


def load(src, tag):
    s = src.replace("triton_helpers.set_driver_to_cpu()", "").replace("triton_helpers.set_driver_to_gpu()", "")
    s = re.sub(r"@triton_heuristics\.pointwise\((?:.|\n)*?\n\)\n", "", s)
    s = s.replace("Placeholder.KERNEL_NAME", "kern")
    s = re.sub(r"^    if \(.*\):$", "    if True:", s, flags=re.M)  # analysis: keep only the guarded branch
    p = Path(tempfile.gettempdir()) / f"sass_{tag}.py"
    p.write_text(s)
    spec = importlib.util.spec_from_file_location(f"sass_{tag}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sass(m, xblock):
    sig = dict(re.findall(r"'(\w+)': '([^']+)'", re.search(r"'signature': \{([^}]*)\}", SRC).group(1)))
    k = triton.compile(triton.compiler.ASTSource(m.kern, sig, constexprs={"XBLOCK": xblock}),
                       target=GPUTarget("cuda", 89, 32), options={"num_warps": 4})
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(k.asm["cubin"])
    out = subprocess.run([str(NVDISASM), "-c", f.name], capture_output=True, text=True).stdout
    ops = re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", out)
    base = collections.Counter(o.split(".")[0] for o in ops)
    return {"instructions": len(ops), "IMAD": base["IMAD"], "IMAD.HI/WIDE": sum(1 for o in ops if o.startswith("IMAD.HI") or o.startswith("IMAD.WIDE")),
            "I2F/F2I/MUFU (division sequences)": base["I2F"] + base["F2I"] + base["MUFU"],
            "CALL": base["CALL"], "BRA": base["BRA"], "LDG": base["LDG"], "STG": base["STG"],
            "resource_usage": subprocess.run([str(NVDISASM.parent / "cuobjdump"), "-res-usage", f.name],
                                             capture_output=True, text=True).stdout.strip().splitlines()[-1].strip()}


if __name__ == "__main__":
    res = {}
    for v in R.VARIANTS:
        src = R.rewrite(SRC, v)[0]
        m = load(src, v)
        res[v] = {f"XBLOCK={xb}": sass(m, xb) for xb in (1024, 512)}
    (HERE / "sass_counts.json").write_text(json.dumps(res, indent=1) + "\n")
    for v, r in res.items():
        print(v, r["XBLOCK=1024"])
