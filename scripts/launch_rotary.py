#!/usr/bin/env python3
"""Go/no-go benchmark for the next workload: one decoding step of rotary
embedding (Q, K) + paged KV-cache write (`fused_rotary_embedding`, old cache
layout only: `use_new_kcache_layout=False`).

Candidates (all on identical fresh inputs; the kernel indexes `cos`/`sin` by
token, so every candidate receives the same pre-gathered per-token rows):
  eager               PyTorch reference `ref` (below)
  compiled            torch.compile(ref, dynamic=False)
  compiled_graph      CUDA graph of the warmed compiled function (matched fixed-buffer baseline)
  corpus_wrapper      the corpus wrapper `decoding_fused_rotary_embedding` (Python stride plumbing + JIT launch)
  corpus_jit          the corpus kernel launched directly via JIT dispatch with precomputed arguments
  corpus_compiled     the corpus kernel via its precompiled handle (fixed arguments)
  corpus_graph        CUDA graph replay of that launch
  corpus_verify_jit   host copy of kv_lengths/block_tables, check of the value-dependent safety
                      conditions (below), then corpus_jit — the cost of establishing them per step

Value-dependent conditions (contents, not metadata; a metadata check or graph
replay cannot see them): every kv_length >= 1; the block-table column
(len-1)//block_size is in range and its block id < num_blocks; the slot map
token -> (block id, (len-1) % block_size) is injective over tokens.

Timings: perf_counter single (call + synchronize, median) and repeated (K calls,
one synchronize), TRIALS trials with rotated order; device time warm
(torch.profiler kernel sum, no flush) and flushed (triton.testing.do_bench,
L2 flushed per run; event time, includes launch gaps). Correctness: max ULP of
q, k_cache, v_cache against eager on one fresh step.
"""
from __future__ import annotations

import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "bench"))

import torch  # noqa: E402

TOKENS = [1, 8, 32, 64, 256]
HEAD_DIMS = [64, 128]
QH, KH, BS, BLOCKS_PER_SEQ = 32, 8, 16, 64
REPS, K, TRIALS = 200, 100, 5
ROTARY_PY = REPO / "bench/tritonbench_g/fused_rotary_embedding/fused_rotary_embedding.py"


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def single_us(fn, device, reps=REPS):
    for _ in range(10):
        fn()
    sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(device); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def repeated_us(fn, device, k=K):
    for _ in range(10):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(k):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / k * 1e6


def kernels(fn, calls=10):
    from torch.profiler import ProfilerActivity, profile
    fn(); sync("cuda")
    per, names = [], {}
    for _ in range(calls):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); sync("cuda")
        ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
        per.append(sum(e.device_time for e in ev))
        for e in ev:
            names.setdefault(e.name[:120], []).append(e.device_time)
    return {"kernel_us_warm": statistics.median(per),
            "per_kernel_us": {n: statistics.median(v) for n, v in names.items()},
            "kernels_per_call": round(sum(len(v) for v in names.values()) / calls, 2)}


def flushed_us(fn):
    import triton
    q = triton.testing.do_bench(fn, warmup=10, rep=100, quantiles=[0.2, 0.5, 0.8])
    return {"p20": q[0] * 1e3, "median": q[1] * 1e3, "p80": q[2] * 1e3}


def max_ulp(a, b) -> int:
    if a.shape != b.shape:
        return -1
    ka = a.contiguous().view(torch.int32).to(torch.int64)
    kb = b.contiguous().view(torch.int32).to(torch.int64)
    ka = torch.where(ka < 0, -(ka & 0x7FFFFFFF), ka)
    kb = torch.where(kb < 0, -(kb & 0x7FFFFFFF), kb)
    return int((ka - kb).abs().max()) if a.numel() else 0


def make(T, D, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    NB = T * BLOCKS_PER_SEQ
    q = torch.randn(T, QH, D, generator=g).to(device)
    k = torch.randn(T, KH, D, generator=g).to(device)
    v = torch.randn(T, KH, D, generator=g).to(device)
    pos = torch.randint(1, BLOCKS_PER_SEQ * BS, (T,), generator=g)
    inv = 1.0 / (10000 ** (torch.arange(0, D // 2).float() / (D // 2)))
    ang = pos.float()[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1).to(device)   # per-token rows, [T, D]
    sin = torch.cat([ang.sin(), ang.sin()], -1).to(device)
    kc = torch.zeros(NB, KH, BS, D, device=device)
    vc = torch.zeros(NB, KH, BS, D, device=device)
    bt = torch.randperm(NB, generator=g).to(torch.int32).view(T, BLOCKS_PER_SEQ).to(device)
    lens = pos.to(torch.int32).to(device)
    return dict(q=q, k=k, v=v, cos=cos, sin=sin, kc=kc, vc=vc, bt=bt, lens=lens, T=T, D=D)


def ref(q, k, v, cos, sin, kc, vc, bt, lens):
    T, D = q.shape[0], q.shape[2]
    h = D // 2
    c, s = cos[:T, :h][:, None, :], sin[:T, :h][:, None, :]
    q0, q1 = q[..., :h], q[..., h:]
    q.copy_(torch.cat([q0 * c - q1 * s, q0 * s + q1 * c], -1))
    k0, k1 = k[..., :h], k[..., h:]
    kr = torch.cat([k0 * c - k1 * s, k0 * s + k1 * c], -1)
    p = (lens - 1).long()
    blk = bt[torch.arange(T, device=q.device), p // kc.shape[2]].long()
    off = p % kc.shape[2]
    kc[blk, :, off, :] = kr
    vc[blk, :, off, :] = v


def value_conditions(lens_h, bt_h, block_size, num_blocks) -> list:
    """The value-dependent obligations, checked on host copies; returns failures."""
    fails = []
    T = lens_h.shape[0]
    slots = set()
    for t in range(T):
        L = int(lens_h[t])
        if L < 1:
            fails.append(f"kv_length[{t}] < 1"); continue
        col = (L - 1) // block_size
        if col >= bt_h.shape[1]:
            fails.append(f"block-table column {col} out of range for token {t}"); continue
        b = int(bt_h[t, col])
        if not (0 <= b < num_blocks):
            fails.append(f"block id {b} out of range for token {t}"); continue
        slot = (b, (L - 1) % block_size)
        if slot in slots:
            fails.append(f"slot {slot} written by two tokens")
        slots.add(slot)
    return fails


def corpus(device):
    import launch_interpret as LI
    ns = LI.load_defs(ROTARY_PY)
    return ns["decoding_fused_rotary_embedding"], ns["decoding_fused_rotary_embedding_kernel"]


def kernel_args(d):
    q, k, v, cos, sin, kc, vc, bt, lens = (d[n] for n in ("q", "k", "v", "cos", "sin", "kc", "vc", "bt", "lens"))
    D = d["D"]
    args = (q, k, v, cos, sin, kc, vc, bt, lens, D, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            q.stride(2), cos.stride(0), cos.stride(1), kc.stride(0), kc.stride(1), 0, kc.stride(2),
            kc.stride(3), vc.stride(0), vc.stride(1), vc.stride(2), vc.stride(3), bt.stride(0), bt.stride(1),
            kc.size(-2))
    kw = {"KV_GROUP_NUM": QH // KH, "HEAD_DIM": D, "num_warps": 8 if D >= 256 else 4}
    return args, kw, (QH, d["T"])


def candidates(T, D, device, compiled_fn, wrapper, kern):
    d = make(T, D, device)
    names = ("q", "k", "v", "cos", "sin", "kc", "vc", "bt", "lens")
    tens = [d[n] for n in names]
    fns, errors = {}, {}
    fns["eager"] = lambda: ref(*tens)
    fns["compiled"] = lambda: compiled_fn(*tens)
    fns["corpus_wrapper"] = lambda: wrapper(q=d["q"], k=d["k"], v=d["v"], cos=d["cos"], sin=d["sin"],
                                             k_cache=d["kc"], v_cache=d["vc"], block_tables=d["bt"],
                                             kv_lengths=d["lens"])
    args, kw, grid = kernel_args(d)
    fns["corpus_jit"] = lambda: kern[grid](*args, **kw)
    NB = d["kc"].shape[0]

    def verify():
        fails = value_conditions(d["lens"].cpu(), d["bt"].cpu(), BS, NB)
        if fails:
            raise RuntimeError(fails)
        kern[grid](*args, **kw)
    fns["corpus_verify_jit"] = verify
    if device == "cuda":
        try:
            ck = kern.warmup(*args, grid=grid, **kw)
            launch = ck[(grid[0], grid[1], 1)]
            full = tuple(args) + (QH // KH, D)
            fns["corpus_compiled"] = lambda: launch(*full)
            fns["corpus_compiled"](); sync(device)
            g1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g1):
                launch(*full)
            fns["corpus_graph"] = g1.replay
        except Exception:  # noqa: BLE001
            errors["corpus_compiled/graph"] = traceback.format_exc()[-1500:]
        try:
            for _ in range(3):
                compiled_fn(*tens)
            sync(device)
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2):
                compiled_fn(*tens)
            fns["compiled_graph"] = g2.replay
        except Exception:  # noqa: BLE001
            errors["compiled_graph"] = traceback.format_exc()[-1500:]
    return fns, errors


def correctness(T, D, device, compiled_fn, wrapper, kern):
    base = make(T, D, device, seed=11)
    out = {}
    def fresh():
        return {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in base.items()}
    e = fresh(); ref(*(e[n] for n in ("q", "k", "v", "cos", "sin", "kc", "vc", "bt", "lens")))
    runs = {"corpus_wrapper": lambda f: wrapper(q=f["q"], k=f["k"], v=f["v"], cos=f["cos"], sin=f["sin"],
                                                  k_cache=f["kc"], v_cache=f["vc"], block_tables=f["bt"],
                                                  kv_lengths=f["lens"])}
    if compiled_fn is not None:
        runs["compiled"] = lambda f: compiled_fn(*(f[n] for n in ("q", "k", "v", "cos", "sin", "kc", "vc", "bt", "lens")))
    for name, run in runs.items():
        f = fresh(); run(f); sync(device)
        out[name] = {n: max_ulp(f[key], e[key]) for n, key in (("q", "q"), ("k_cache", "kc"), ("v_cache", "vc"))}
    return out


def env(device):
    import subprocess

    import triton
    rec = {"device": device, "triton": triton.__version__, "torch": torch.__version__,
           "TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET")}
    if device == "cuda":
        rec["gpu_name"] = torch.cuda.get_device_name()
        rec["compute_capability"] = list(torch.cuda.get_device_capability())
        rec["cpu"] = subprocess.run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo; nproc"],
                                    capture_output=True, text=True).stdout.strip()
    return rec


def bench(device="cuda") -> dict:
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64
    wrapper, kern = corpus(device)
    compiled_fn = torch.compile(ref, dynamic=False)
    rec = {"kind": "rotary_decode_bench (go/no-go)", **env(device),
           "config": {"q_heads": QH, "kv_heads": KH, "block_size": BS, "blocks_per_seq": BLOCKS_PER_SEQ,
                      "cache_layout": "old (use_new_kcache_layout=False)", "dtype": "float32"},
           "tokens": TOKENS, "head_dims": HEAD_DIMS, "rows": []}
    for D in HEAD_DIMS:
        for T in TOKENS:
            row = {"T": T, "D": D}
            try:
                row["max_ulp_vs_eager"] = correctness(T, D, device, compiled_fn, wrapper, kern)
                fns, errors = candidates(T, D, device, compiled_fn, wrapper, kern)
                row["errors"] = errors
                for fn in fns.values():
                    fn()
                sync(device)
                names = list(fns)
                trials = []
                for t in range(TRIALS):
                    order = names[t % len(names):] + names[:t % len(names)]
                    trials.append({"single": {n: single_us(fns[n], device) for n in order},
                                   "repeated": {n: repeated_us(fns[n], device) for n in order}})
                row["trials"] = trials
                row["summary"] = {m: {n: statistics.median(tr[m][n] for tr in trials) for n in names}
                                  for m in ("single", "repeated")}
                row["device_warm"] = {n: kernels(fns[n]) for n in names if n != "corpus_verify_jit"}
                row["device_flushed"] = {n: flushed_us(fns[n]) for n in
                                         ("eager", "compiled", "compiled_graph", "corpus_jit", "corpus_graph")
                                         if n in fns}
            except Exception:  # noqa: BLE001
                row["error"] = traceback.format_exc()[-2500:]
            rec["rows"].append(row)
    import launch_local_check as LC
    rec["input_hashes"] = LC.input_hashes()
    return rec


def dry() -> dict:
    """CPU interpreter: corpus wrapper vs eager reference, value-condition checker."""
    wrapper, kern = corpus("cpu")
    res = {"ulp": {}}
    for D in (64,):
        for T in (1, 4):
            res["ulp"][f"T{T}_D{D}"] = correctness(T, D, "cpu", None, wrapper, kern)
    d = make(4, 64, "cpu")
    NB = d["kc"].shape[0]
    res["conditions_ok"] = value_conditions(d["lens"], d["bt"], BS, NB)
    bad = d["bt"].clone(); bad[1] = bad[0]
    lens2 = d["lens"].clone(); lens2[1] = lens2[0]
    res["conditions_duplicate_slot"] = value_conditions(lens2, bad, BS, NB)
    lens3 = d["lens"].clone(); lens3[2] = 0
    res["conditions_zero_length"] = value_conditions(lens3, d["bt"], BS, NB)
    return res


if __name__ == "__main__":
    import json
    if torch.cuda.is_available():
        print(json.dumps(bench("cuda"), indent=1, default=str))
    else:
        print(json.dumps(dry(), indent=1, default=str))
