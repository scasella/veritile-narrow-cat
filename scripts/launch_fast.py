#!/usr/bin/env python3
"""Lower-overhead checked execution of the fused add + ReLU (stage 9).

* `CheckedFusedAddReluB` — the general checked API (`launch_fused.CheckedFusedAddRelu`)
  with an explicit block size. Safe for any block the contract accepts:
  `add_relu_wrapper_correctness_block` (AddReluFused.lean) is stated for every
  `B` with `Elementwise2.check B x y out = true`, and the mirror decides exactly
  that check with the block in use. Returns a fresh output, like the source wrapper.
* `CheckedFusedAddReluFast` — the same general checked API with the contract
  decided by `fast_ew2`, a direct evaluation of the Boolean `Elementwise2.check`
  (same verdict and same launch as the full mirror `launch_invoke.ew2_obligations`
  / `ew2_launch`; differential-tested against both the mirror and Lean `#eval`).
  On rejection it falls back to the full mirror to name the failed obligations.
* `PreparedAddRelu` / `prepare_add_relu` — a separately labelled prepared-buffer
  API (caller-owned output, overwritten per run).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_fused as F  # noqa: E402
import launch_invoke as I  # noqa: E402


class CheckedFusedAddReluB(F.CheckedFusedAddRelu):
    """The general checked API with an explicit block size."""

    def __init__(self, block: int | None = None, num_warps: int | None = None):
        super().__init__(num_warps=num_warps)
        if block is not None:
            self.block = block


# ---------------------------------------------------------------------------
# Fast evaluation of `Elementwise2.check` (same Boolean as the full mirror)
# ---------------------------------------------------------------------------

I32 = 2 ** 31
POW2 = frozenset(2 ** k for k in range(21))


def raw_meta(t) -> tuple:
    """(base, elemBytes, shape, strides, capacity, dtype-name) of a live tensor;
    the same fields `launch_invoke.tensor_meta` extracts."""
    es = t.element_size()
    return (t.data_ptr(), es, tuple(t.shape), t.stride(),
            t.untyped_storage().nbytes() // es - t.storage_offset(), _DT.get(t.dtype, "other"))


def _dtypes() -> dict:
    try:
        import torch
        return {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}
    except ImportError:
        return {}


_DT = _dtypes()


def _contig(shape, strides) -> bool:
    if len(strides) != len(shape):
        return False
    expect = 1
    for d, st in zip(reversed(shape), reversed(strides)):
        if d > 1 and st != expect:
            return False
        expect *= d
    return True


def _aligned(base, es) -> bool:
    return base % es == 0 if es else base == 0  # Nat: a % 0 = a


def _disj(n, a0, ae, b0, be) -> bool:
    return n * ae == 0 or n * be == 0 or a0 + n * ae <= b0 or b0 + n * be <= a0


def fast_ew2(B: int, x: tuple, y: tuple, o: tuple):
    """`Elementwise2.check B x y out` as one Boolean; returns (ok, n, grid)."""
    xb, xe, xs, xt, xc, xd = x
    yb, ye, ys, yt, yc, yd = y
    ob, oe, os_, ot, oc, od = o
    n = 1
    for d in xs:
        n *= d
    g = (n + B - 1) // B if B else 0
    cx, cy, co = _contig(xs, xt), _contig(ys, yt), _contig(os_, ot)
    m = min(n, g * B)
    ok = (ys == xs and os_ == xs and cx and cy and co
          and _aligned(xb, xe) and _aligned(yb, ye) and _aligned(ob, oe)
          and (xb == yb or _disj(n, xb, xe, yb, ye))
          and B in POW2 and n <= g * B and m <= oc and m <= xc and m <= yc
          and g * B <= I32 and n < I32 and g < I32
          and (n <= 1 or (cx and cy and co))
          and xd == "f32" and yd == "f32" and od == "f32" and xe == 4 and ye == 4 and oe == 4
          and _disj(n, ob, oe, xb, xe) and _disj(n, ob, oe, yb, ye))
    return ok, n, g


class CheckedFusedAddReluFast(CheckedFusedAddReluB):
    """General checked API, fast contract evaluation. Same observable behaviour
    (fresh output; ContractViolation naming the failed obligations)."""

    def __init__(self, block: int | None = None, num_warps: int | None = None):
        super().__init__(block, num_warps)
        order = [self._roles[p] for p in self._params]
        if order != ["x", "y", "out", "n", "B"]:
            raise I.LC.Unsupported(f"fast launch assumes (x, y, out, n, B) parameter order, got {order}")

    def __call__(self, x, y):
        import torch
        self.stats["calls"] += 1
        out = torch.empty_like(x)
        B = self.block
        ok, n, g = fast_ew2(B, raw_meta(x), raw_meta(y), raw_meta(out))
        if not ok:
            failed, _ = self.verdict(x, y, out)
            raise I.ContractViolation(failed or ["fast/mirror disagreement"])
        self._kernel[(g,)](x, y, out, n, B, **({"num_warps": self.num_warps} if self.num_warps else {}))
        return out


# ---------------------------------------------------------------------------
# Prepared-buffer API (separate from the general checked API)
# ---------------------------------------------------------------------------

class PlanInvalidated(RuntimeError):
    """A prepared buffer's metadata changed since `prepare_add_relu`."""


def meta_snapshot(t) -> tuple:
    """Everything the contract and Triton's specialization read from a tensor:
    data pointer (alignment), shape, strides, storage offset, dtype, device, and
    the backing storage's address and size (catches `set_`/`resize_`)."""
    st = t.untyped_storage()
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.storage_offset(), t.dtype, t.device,
            st.data_ptr(), st.nbytes())


class PreparedAddRelu:
    """`out <- relu(x + y)` on three FIXED buffers.

    Observable behaviour differs from the general API: the caller supplies the
    output buffer, which is overwritten on every `run()`; no tensor is returned
    fresh. `prepare` decides the full `Elementwise2` contract once and compiles
    (without launching) the kernel for exactly these arguments. `run` then only
    revalidates the metadata snapshot of the three buffers — the contract's
    verdict is a function of metadata alone, and the correctness theorem
    quantifies over all input VALUES, so contents may change between runs — and
    launches the precompiled kernel (or replays a CUDA graph of that launch).
    Because the pointers and `n` are identical to the ones compiled for, Triton's
    argument specialization (pointer alignment, `n % 16`) is unchanged.
    Not thread-safe; one plan per buffer triple and stream.
    """

    def __init__(self, checked: "F.CheckedFusedAddRelu", x, y, out, graph: bool = False):
        import torch
        failed, cfg = checked.verdict(x, y, out)
        if failed:
            raise I.ContractViolation(failed)
        self.tensors = (x, y, out)
        self.snapshot = tuple(meta_snapshot(t) for t in self.tensors)
        vals = {"x": x, "y": y, "out": out, "n": cfg["n"], "B": cfg["block"]}
        self.args = tuple(vals[checked._roles[p]] for p in checked._params)
        self.grid = (cfg["grid"][0], 1, 1)
        kw = {"num_warps": checked.num_warps} if checked.num_warps else {}
        self.compiled = checked._kernel.warmup(*self.args, grid=self.grid, **kw)
        self._launch = self.compiled[self.grid]
        self.graph = None
        if graph:
            self._launch(*self.args)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._launch(*self.args)
            self.graph = g

    def revalidate(self) -> None:
        if tuple(meta_snapshot(t) for t in self.tensors) != self.snapshot:
            raise PlanInvalidated("prepared buffer metadata changed; prepare a new plan")

    def run(self) -> None:
        self.revalidate()
        if self.graph is not None:
            self.graph.replay()
        else:
            self._launch(*self.args)


def prepare_add_relu(x, y, out, block: int | None = None, graph: bool = False) -> PreparedAddRelu:
    return PreparedAddRelu(CheckedFusedAddReluB(block=block), x, y, out, graph=graph)


# ---------------------------------------------------------------------------
# Differential test: fast_ew2 vs the full mirror and vs Lean `#eval`
# ---------------------------------------------------------------------------

def _as_raw(t) -> tuple:
    return (t.base, t.elemBytes, tuple(t.shape), tuple(t.strides), t.capacity, t.dtype)


def differential_fast(n_mirror: int = 20000, n_lean: int = 3000, seed: int = 7) -> dict:
    """Verdict and launch of `fast_ew2` against `launch_invoke.ew2_obligations`/`ew2_launch`
    on `n_mirror` random near-valid/adversarial cases, and verdict against Lean's
    `Elementwise2.check` on the first `n_lean` of them."""
    cases = I.random_cases(n_mirror, seed)
    mism = []
    for i, (B, x, y, o) in enumerate(cases):
        ok_m = all(v for _, v in I.ew2_obligations(B, x, y, o))
        L = I.ew2_launch(B, x, y, o)
        ok_f, n_f, g_f = fast_ew2(B, _as_raw(x), _as_raw(y), _as_raw(o))
        if ok_m != ok_f or (ok_m and (L["n"], L["grid"][0]) != (n_f, g_f)):
            mism.append(i)
    head = "import VeriTile.Triton.Launch.Blocked1DWrapper\nopen VeriTile.Triton\n\n"
    body = [f'#eval IO.println s!"R|{{Elementwise2.check {B} {I.lean_tm(x)} {I.lean_tm(y)} {I.lean_tm(o)}}}"'
            for B, x, y, o in cases[:n_lean]]
    r = I.LC.run_lean(head + "\n".join(body) + "\n", timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    rows = [ln.split("|")[1] for ln in r.stdout.splitlines() if ln.startswith("R|")]
    assert len(rows) == n_lean, (len(rows), n_lean)
    lean_mism = [i for i, ((B, x, y, o), v) in enumerate(zip(cases[:n_lean], rows))
                 if fast_ew2(B, _as_raw(x), _as_raw(y), _as_raw(o))[0] != (v == "true")]
    return {"cases_vs_mirror": n_mirror, "mismatches_vs_mirror": mism,
            "cases_vs_lean": n_lean, "mismatches_vs_lean": lean_mism,
            "accepted_by_lean": sum(v == "true" for v in rows), "seed": seed}


if __name__ == "__main__" and "--differential" in sys.argv:
    import json
    res = differential_fast()
    import launch_local_check as LCK
    res["input_hashes"] = LCK.input_hashes()
    out = Path(__file__).resolve().parents[1] / "bench/tritonbench_g/add_example/launch_evidence/fast_check_differential.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in res.items() if k != "input_hashes"}))
