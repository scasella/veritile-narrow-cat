#!/usr/bin/env python3
"""Stage 9, run 2: (1) validation of the selected checked-execution changes on
the final code, (2) a bounded workload probe for the next optimization target.

validation(device) — single (call + synchronize) and repeated (K calls, one
synchronize) host timings, TRIALS trials with rotated candidate order:
  checked_orig_B64     launch_fused.CheckedFusedAddRelu (mirror decision, block 64)
  checked_fast_B64     launch_fast.CheckedFusedAddReluFast(block=64)
  checked_fast_B256    launch_fast.CheckedFusedAddReluFast(block=256)   <- selected general API
  plan_B256            prepared-buffer API, precompiled launch          <- prepared API
  plan_graph_B256      prepared-buffer API, CUDA graph replay
  eager                torch.relu(x + y)
  eager_preallocated   torch.add(x, y, out=o); o.relu_()
  eager_graph          CUDA graph of eager_preallocated (matched fixed-buffer baseline)
  compiled_default     torch.compile(f, dynamic=False)
  compiled_reduce_overhead            torch.compile(f, dynamic=False, mode="reduce-overhead")
  compiled_reduce_overhead_markstep   same, torch.compiler.cudagraph_mark_step_begin() before each call
  unfused_checked      checked add (block 64) then checked strided ReLU
plus device time (summed kernel durations, torch.profiler; context only), fast-vs-mirror
verdict agreement on real CUDA tensors, and bitwise agreement of outputs.

probe(device) — for three real workloads: eager, torch.compile (default), and the
corpus Triton kernel where one exists: kernel count, kernel names, summed kernel
time, logical bytes, and the time at 250 GB/s for those bytes. The purpose is
selection evidence only.
"""
from __future__ import annotations

import statistics
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "bench"))

import torch  # noqa: E402

import launch_fast as FA  # noqa: E402
import launch_fused as F  # noqa: E402
import launch_invoke as I  # noqa: E402
import launch_latency as LL  # noqa: E402

VAL_SIZES = [2 ** 12, 2 ** 16, 2 ** 20, 2 ** 22, 700_001, 3_000_017, 2 ** 24, 2 ** 25]
SELECTED_BLOCK = 256
BW = 250e9  # bytes/s: the effective bandwidth measured for these kernels on the L4 (stage 8)


def profile_call(fn, calls: int = 10) -> dict:
    from torch.profiler import ProfilerActivity, profile
    fn(); LL.sync()
    per, names, counts = [], set(), []
    for _ in range(calls):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); LL.sync()
        ev = [e for e in prof.events() if e.device_type.name == "CUDA" and "Memcpy" not in e.name
              and "Memset" not in e.name]
        per.append(sum(e.device_time for e in ev))
        counts.append(len(ev))
        names |= {e.name for e in ev}
    return {"kernel_us": statistics.median(per), "kernels_per_call": statistics.median(counts),
            "kernel_names": sorted(names)[:12]}


def validation(device: str) -> dict:
    rec = {"kind": "validation (checked execution after stage-9 changes)", **LL.env(device),
           "sizes": VAL_SIZES, "selected_block": SELECTED_BLOCK, "rows": []}
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64
    f = lambda a, b: torch.relu(a + b)  # noqa: E731
    cd = torch.compile(f, dynamic=False)
    ro = torch.compile(f, dynamic=False, mode="reduce-overhead")
    torch.manual_seed(3)
    for n in VAL_SIZES:
        x, y = torch.randn(n, device=device), torch.randn(n, device=device)
        orig = F.CheckedFusedAddRelu()
        f64 = FA.CheckedFusedAddReluFast(block=64)
        f256 = FA.CheckedFusedAddReluFast(block=SELECTED_BLOCK)
        plan = FA.prepare_add_relu(x, y, torch.empty_like(x), block=SELECTED_BLOCK)
        gplan = FA.prepare_add_relu(x, y, torch.empty_like(x), block=SELECTED_BLOCK, graph=True)
        pre = torch.empty_like(x)

        def eager_pre():
            torch.add(x, y, out=pre); pre.relu_()
        eager_pre(); LL.sync()
        eg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(eg):
            eager_pre()
        unfused = F.UnfusedPipeline(4) if n <= 2 ** 25 else None

        def ro_mark():
            torch.compiler.cudagraph_mark_step_begin()
            return ro(x, y)
        fns = {"checked_orig_B64": lambda: orig(x, y), "checked_fast_B64": lambda: f64(x, y),
               "checked_fast_B256": lambda: f256(x, y), "plan_B256": plan.run,
               "plan_graph_B256": gplan.run, "eager": lambda: torch.relu(x + y),
               "eager_preallocated": eager_pre, "eager_graph": eg.replay,
               "compiled_default": lambda: cd(x, y), "compiled_reduce_overhead": lambda: ro(x, y),
               "compiled_reduce_overhead_markstep": ro_mark}
        if unfused is not None:
            fns["unfused_checked"] = lambda: unfused(x, y)
        for _ in range(5):
            for fn in fns.values():
                fn()
        LL.sync()
        ref = orig(x, y)
        outs = {"checked_fast_B64": f64(x, y), "checked_fast_B256": f256(x, y),
                "plan_B256": (plan.run(), plan.tensors[2].clone())[1],
                "plan_graph_B256": (gplan.run(), gplan.tensors[2].clone())[1],
                "eager_graph": (eg.replay(), pre.clone())[1],
                "compiled_reduce_overhead_markstep": ro_mark().clone()}
        LL.sync()
        big = n > 2 ** 22
        reps, trials = (50, 3) if big else (LL.REPS, LL.TRIALS)
        names = list(fns)
        tr = []
        for t in range(trials):
            order = names[t % len(names):] + names[:t % len(names)]
            tr.append({"single": {nm: LL.single_us(fns[nm], reps) for nm in order},
                       "repeated": {nm: LL.repeated_us(fns[nm], 20 if big else LL.K) for nm in order}})
        summary = {m: {nm: {"median_of_trials": statistics.median(t_[m][nm] for t_ in tr),
                            "min": min(t_[m][nm] for t_ in tr), "max": max(t_[m][nm] for t_ in tr)}
                       for nm in names} for m in ("single", "repeated")}
        o = torch.empty_like(x)
        _, c64 = orig.verdict(x, y, o)
        _, c256 = f256.verdict(x, y, o)
        device_us = {"fused_B64": LL.kernel_us(lambda: orig.launch_only(x, y, o, c64), 10),
                     "fused_B256": LL.kernel_us(lambda: f256.launch_only(x, y, o, c256), 10),
                     "eager": LL.kernel_us(lambda: torch.relu(x + y), 10),
                     "compiled_default": LL.kernel_us(lambda: cd(x, y), 10)}
        rec["rows"].append({"n": n, "bitwise_equal_to_orig": {k: LL.bitwise(v, ref) for k, v in outs.items()},
                            "trials": tr, "summary": summary, "device_kernel_us": device_us})
    # fast decision vs full mirror on real CUDA tensors (layouts that must be rejected included)
    base = torch.randn(4096, device=device)
    layouts = [(torch.randn(16, device=device),) * 2, (torch.randn(8, 4, device=device).t(),
               torch.randn(4, 8, device=device)), (base[0:100], base[50:150]), (base[1:101], base[200:300]),
               (base[::2][:50], base[1::2][:50]), (torch.randn(0, device=device),) * 2,
               (torch.randn(16, device=device).half(), torch.randn(16, device=device).half())]
    agree = []
    for B in (64, 256):
        for x, y in layouts:
            o = torch.empty_like(x)
            mx, my, mo = I.tensor_meta(x), I.tensor_meta(y), I.tensor_meta(o)
            m = all(v for _, v in I.ew2_obligations(B, mx, my, mo))
            fz = FA.fast_ew2(B, FA.raw_meta(x), FA.raw_meta(y), FA.raw_meta(o))[0]
            agree.append({"B": B, "shape": list(x.shape), "stride": list(x.stride()), "mirror": m, "fast": fz})
    rec["fast_vs_mirror_real_tensors"] = agree
    rec["fast_vs_mirror_all_agree"] = all(a["mirror"] == a["fast"] for a in agree)
    rec["correct"] = rec["fast_vs_mirror_all_agree"] and all(all(r["bitwise_equal_to_orig"].values())
                                                            for r in rec["rows"])
    rec["input_hashes"] = LL.hashes()
    return rec


# ---------------------------------------------------------------------------
# Workload probe
# ---------------------------------------------------------------------------

def _rmsnorm_ref(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _load(rel, name):
    import launch_interpret as LI
    return LI.load_defs(REPO / rel)[name]


def probe_rmsnorm(device: str, residual: bool) -> dict:
    M, N = 4096, 4096
    x, r = torch.randn(M, N, device=device), torch.randn(M, N, device=device)
    w = torch.randn(N, device=device)
    if residual:
        ref = lambda: (lambda h: (_rmsnorm_ref(h, w), h))(x + r)  # noqa: E731
        ideal = 4 * (4 * M * N + N)   # read x, r, w; write h, y (one pass over h)
    else:
        ref = lambda: _rmsnorm_ref(x, w)  # noqa: E731
        ideal = 4 * (2 * M * N + N)   # read x, write y
    comp = torch.compile(ref, dynamic=False)
    for _ in range(3):
        comp()
    res = {"workload": "Llama RMSNorm" + (" with residual add (returns y and h = x + r)" if residual else ""),
           "shape": [M, N], "dtype": "float32", "logical_bytes": ideal,
           "time_at_250GBps_us": ideal / BW * 1e6,
           "eager": profile_call(ref), "compiled": profile_call(comp)}
    if not residual:
        k = _load("bench/tritonbench_g/rmsnorm_fused/rmsnorm_fused.py", "rms_norm_fwd_fused")
        y = torch.empty_like(x)
        corpus = lambda: k[(M,)](x, y, w, x.stride(0), N, 1e-6, BLOCK_SIZE=4096, num_warps=8)  # noqa: E731
        corpus()
        res["corpus_triton"] = profile_call(corpus)
        res["corpus_max_abs_diff_vs_eager"] = float((y - ref()).abs().max())
    return res


def probe_rotary(device: str) -> dict:
    T, QH, KH, D, BS = 64, 32, 8, 128, 16
    blocks_per_seq = 8
    NB = T * blocks_per_seq
    q = torch.randn(T, QH, D, device=device); k = torch.randn(T, KH, D, device=device)
    v = torch.randn(T, KH, D, device=device)
    cos = torch.randn(4096, D, device=device); sin = torch.randn(4096, D, device=device)
    kc = torch.zeros(NB, KH, BS, D, device=device); vc = torch.zeros(NB, KH, BS, D, device=device)
    bt = torch.arange(NB, device=device, dtype=torch.int32).view(T, blocks_per_seq)  # disjoint blocks per sequence
    lens = torch.randint(1, blocks_per_seq * BS, (T,), device=device, dtype=torch.int32)
    h = D // 2

    def ref(q, k, v, kc, vc):
        c, s = cos[:T, :h][:, None, :], sin[:T, :h][:, None, :]
        q0, q1 = q[..., :h], q[..., h:]
        q.copy_(torch.cat([q0 * c - q1 * s, q0 * s + q1 * c], -1))
        k0, k1 = k[..., :h], k[..., h:]
        kr = torch.cat([k0 * c - k1 * s, k0 * s + k1 * c], -1)
        pos = (lens - 1).long()
        blk = bt[torch.arange(T, device=q.device), pos // BS].long()
        off = pos % BS
        kc[blk, :, off, :] = kr
        vc[blk, :, off, :] = v

    comp = torch.compile(ref, dynamic=False)
    fresh = lambda: (q.clone(), k.clone(), v.clone(), kc.clone(), vc.clone())  # noqa: E731
    for _ in range(3):
        comp(*fresh())
    a = fresh(); ref(*a)
    kern = _load("bench/tritonbench_g/fused_rotary_embedding/fused_rotary_embedding.py",
                 "decoding_fused_rotary_embedding")
    b = fresh()
    kern(q=b[0], k=b[1], v=b[2], cos=cos, sin=sin, k_cache=b[3], v_cache=b[4], block_tables=bt, kv_lengths=lens)
    agree = {nm: float((a[i] - b[i]).abs().max()) for i, nm in enumerate(["q", "k", "v", "k_cache", "v_cache"])}
    ideal = 4 * (2 * T * QH * D + T * KH * D * 2 + T * KH * D * 2 + 2 * T * h * 2) + 4 * 2 * T
    args_e, args_c, args_k = fresh(), fresh(), fresh()
    return {"workload": "decoding rotary embedding (Q, K) + paged KV-cache write",
            "shape": {"tokens": T, "q_heads": QH, "kv_heads": KH, "head_dim": D, "block_size": BS},
            "logical_bytes": ideal, "time_at_250GBps_us": ideal / BW * 1e6,
            "corpus_vs_reference_max_abs_diff": agree,
            "eager": profile_call(lambda: ref(*args_e)), "compiled": profile_call(lambda: comp(*args_c)),
            "corpus_triton": profile_call(lambda: kern(q=args_k[0], k=args_k[1], v=args_k[2], cos=cos, sin=sin,
                                                       k_cache=args_k[3], v_cache=args_k[4], block_tables=bt,
                                                       kv_lengths=lens))}


def probe(device: str) -> dict:
    rec = {"kind": "workload_probe (selection evidence only)", **LL.env(device), "workloads": {},
           "errors": {}}
    for name, fn in (("rmsnorm", lambda: probe_rmsnorm(device, False)),
                     ("rmsnorm_residual", lambda: probe_rmsnorm(device, True)),
                     ("rotary_kvcache", lambda: probe_rotary(device))):
        try:
            rec["workloads"][name] = fn()
        except Exception:  # noqa: BLE001
            rec["errors"][name] = traceback.format_exc()[-2500:]
    rec["input_hashes"] = LL.hashes()
    return rec


PARTS = [("validation", lambda: validation("cuda")), ("workload_probe", lambda: probe("cuda"))]
