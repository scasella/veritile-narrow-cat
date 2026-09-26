#!/usr/bin/env python3
"""Stage 13: N1 against stage-12 guarded N on Inductor's dynamic-cat kernel, with the sustained-time
discrepancy diagnosed in the same campaign.

Workload, shapes, ORDER, CALLS and WARM_PASSES are the same as stage 12 (`launch_inductor_divmod.py`,
imported unchanged).

Configurations. Each runs in its own subprocess with private Inductor and Triton caches; the FX-graph and
AOT-autograd caches are off.
  B0      torch.compile(dynamic=True), last dim marked dynamic; the kernel as Inductor emits it
  N       B0 plus stage-12 guarded narrowing: scalars cast to int32 at kernel entry; the original body is
          kept under an in-kernel `else`
  N1      B0 plus the signature-level narrowing of `scripts/inductor_narrow.py`: the seven `ks*` arguments are
          declared i32 and the body is unchanged. It applies only if the kernel's extracted IR equals the proved
          IR and Inductor's own symbolic state establishes H1-H4 (logged per kernel).
          Outside that domain Dynamo recompiles and the original kernel is kept.
  static  torch.compile(dynamic=False): specializes and caches a kernel per shape (context only)
Each configuration runs in two modes:
  auto    Inductor's normal pointwise autotuning. This is the deployment number, and the decision uses it.
  fixed   `triton.autotune_pointwise = False`: one config, identical across variants. Used for mechanism
          only.
The full set is run in 2 rounds; round 2 runs the configurations in reverse order.

Per process, beyond the stage-12 measurements (cold pass, 5 warm passes, single-call probe, isolated
profile):
  artifacts   for every compiled Triton kernel, the chosen launcher config (kwargs, num_warps, num_stages),
              n_regs, n_spills, shared memory, cache hash, cubin sha256, and SASS instruction counts
              (Triton's bundled nvdisasm, run on the actual cubin)
  telemetry   NVML sampled every ~10 ms during the warm passes and the idle probe: SM and memory clocks,
              power, temperature, clock-event (throttle) reason bitmask; plus the enforced power limit
  full_pass   one extra warm pass under torch.profiler: every kernel's start/end in the sequence, the kernel
              sum, the span, and the gaps
  idle_probe  10 calls per shape, each after a 50 ms host sleep (GPU idle), with CUDA-event time per call
  fallback    (auto mode, round 1, dynamic configurations only) one shape with numel = 2^31, beyond the
              int32 guard: records the recompile, whether any rewrite fired (it must not), and bitwise
              equality with eager

PRE-REGISTERED CRITERION (fixed before the GPU call; do not edit after it)
  T, S and validity are defined exactly as in stage 12, on the `auto` mode. In addition:
    - N1 must fire on the cat kernel with H1-H4 all true and the structural check passed;
    - N1 must not fire in the fallback shape, which must still be bitwise equal to eager.
  Decision (auto mode; must hold in BOTH rounds):
    (a) N1 keeps the stage-12 gain: T(B0)/T(N1) >= 1.10
    (b) adopt N1 in place of N: T(N)/T(N1) >= 0.99, and for every shape x, S(N,x)/S(N1,x) >= 0.97
    (c) report "N1 improves on N" only if T(N)/T(N1) >= 1.03
  If (a) or (b) fails, guarded N stays the recommended implementation, and N1 is reported as proved but
  not adopted.
  Diagnosis of sustained time against device time, per configuration (auto mode, round 1). Here k_seq is
  the median in-sequence kernel duration from full_pass, k_iso the isolated profile, and clk the NVML SM
  clock:
    D1 device state: k_seq >= 1.15 * k_iso, and (median clk during warm passes <= 0.90 * median clk during
       the idle probe, or a power/thermal reason bit set in >= 50% of warm-pass samples)
    D2 dispatch gaps: k_seq within 10% of k_iso, and gaps >= 15% of the full-pass span
    D3 comparison artifact: the timed and the profiled calls ran different configs or cubins
    otherwise "unassigned". The rule outcomes are reported with their inputs.

    python3 scripts/launch_inductor_narrow.py --dry
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import launch_inductor_divmod as S12  # noqa: E402  (workload, ORDER, timing constants; unchanged)

ORDER, CALLS, WARM_PASSES = S12.ORDER, S12.CALLS, S12.WARM_PASSES
CONFIGS = ["B0", "N", "N1", "static"]
MODES = ["auto", "fixed"]
ROUNDS = 2
IDLE_CALLS, IDLE_SLEEP = 10, 0.05
FALLBACK_SHAPE = (2 ** 24, (48, 8, 8))   # numel = 2^24 * 2 * (48 + 8 + 8 + 8) ... see fallback()


# ------------------------------------------------------------------------------------------ telemetry

class Telemetry:
    """NVML sampler in a background thread. Read-only."""

    def __init__(self):
        self.ok, self.err, self.samples, self._stop = False, None, [], threading.Event()
        try:
            import pynvml as N
            N.nvmlInit()
            self.N, self.h = N, N.nvmlDeviceGetHandleByIndex(0)
            self.ok = True
            self.static = {"enforced_power_limit_mw": self._try(N.nvmlDeviceGetEnforcedPowerLimit),
                           "max_sm_clock": self._try(lambda h: N.nvmlDeviceGetMaxClockInfo(h, N.NVML_CLOCK_SM)),
                           "driver": self._try(lambda h: N.nvmlSystemGetDriverVersion())}
        except Exception as e:  # noqa: BLE001
            self.err = repr(e)[:300]
            self.static = {}

    def _try(self, f):
        try:
            v = f(self.h)
            return v.decode() if isinstance(v, bytes) else v
        except Exception as e:  # noqa: BLE001
            return f"n/a: {e!r}"[:80]

    def _reasons(self):
        N = self.N
        for name in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            if hasattr(N, name):
                return getattr(N, name)(self.h)
        return None

    def _loop(self, tag):
        N = self.N
        while not self._stop.is_set():
            try:
                self.samples.append({"t": time.perf_counter(), "tag": tag,
                                     "sm": N.nvmlDeviceGetClockInfo(self.h, N.NVML_CLOCK_SM),
                                     "mem": N.nvmlDeviceGetClockInfo(self.h, N.NVML_CLOCK_MEM),
                                     "power_mw": N.nvmlDeviceGetPowerUsage(self.h),
                                     "temp": N.nvmlDeviceGetTemperature(self.h, N.NVML_TEMPERATURE_GPU),
                                     "reasons": self._reasons()})
            except Exception as e:  # noqa: BLE001
                self.samples.append({"t": time.perf_counter(), "tag": tag, "error": repr(e)[:120]})
            time.sleep(0.01)

    def start(self, tag):
        if not self.ok:
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, args=(tag,), daemon=True)
        self._t.start()

    def stop(self):
        if self.ok:
            self._stop.set()
            self._t.join()


# power cap = 0x4 (SW power cap), 0x20 (SW thermal slowdown), 0x40 (HW thermal), 0x80 (HW power brake),
# 0x8 (HW slowdown); see NVML nvmlClocksEventReasons
LIMIT_BITS = {"sw_power_cap": 0x4, "hw_slowdown": 0x8, "sw_thermal": 0x20, "hw_thermal": 0x40, "hw_power_brake": 0x80}


def summarize(samples, tag):
    xs = [s for s in samples if s.get("tag") == tag and "sm" in s]
    if not xs:
        return {"n": 0}
    out = {"n": len(xs), "sm_median": statistics.median(s["sm"] for s in xs), "sm_min": min(s["sm"] for s in xs),
           "mem_median": statistics.median(s["mem"] for s in xs),
           "power_w_median": statistics.median(s["power_mw"] for s in xs) / 1000,
           "power_w_max": max(s["power_mw"] for s in xs) / 1000, "temp_max": max(s["temp"] for s in xs)}
    rs = [s["reasons"] for s in xs if isinstance(s.get("reasons"), int)]
    if rs:
        out["reason_fraction"] = {k: sum(1 for r in rs if r & b) / len(rs) for k, b in LIMIT_BITS.items()}
    return out


# ------------------------------------------------------------------------------------------ artifacts

def artifacts() -> list:
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    import triton
    nvdisasm = Path(triton.__file__).parent / "backends/nvidia/bin/nvdisasm"
    out = []
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, CachingAutotuner):
                continue
        except ReferenceError:
            continue
        name = getattr(obj, "inductor_meta", {}).get("kernel_name") or getattr(obj.fn, "__name__", "?")
        for ln in obj.launchers:
            rec = {"kernel": name, "config": {"kwargs": dict(ln.config.kwargs), "num_warps": ln.config.num_warps,
                                              "num_stages": ln.config.num_stages},
                   "n_regs": getattr(ln, "n_regs", None), "n_spills": getattr(ln, "n_spills", None),
                   "shared": getattr(ln, "shared", None), "cache_hash": getattr(ln, "cache_hash", None),
                   "n_launchers": len(obj.launchers)}
            try:
                cr = [r for r in obj.compile_results if r.config is ln.config or r.config == ln.config]
                binary = cr[0].kernel if cr else None
                cubin = None
                if binary is not None and hasattr(binary, "asm"):
                    cubin = binary.asm.get("cubin")
                elif binary is not None and getattr(binary, "cubin_path", None):
                    cubin = Path(binary.cubin_path).read_bytes()
                    rec["launcher_kind"] = "static"
                if cubin:
                    rec["cubin_sha256"] = hashlib.sha256(cubin).hexdigest()
                    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
                        f.write(cubin)
                    sass = subprocess.run([str(nvdisasm), "-c", f.name], capture_output=True, text=True).stdout
                    ops = re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", sass)
                    rec["sass"] = {"instructions": len(ops),
                                   "IMAD": sum(1 for o in ops if o.startswith("IMAD")),
                                   "I2F_F2I_MUFU": sum(1 for o in ops if o.split(".")[0] in ("I2F", "F2I", "MUFU")),
                                   "CALL": sum(1 for o in ops if o.startswith("CALL")),
                                   "LDG": sum(1 for o in ops if o.startswith("LDG"))}
                sig = getattr(binary, "src", None)
                rec["signature_ks"] = {k: v for k, v in (getattr(getattr(binary, "src", None), "signature", {}) or {}).items()
                                       if str(k).startswith("ks")} if sig is not None else None
            except Exception as e:  # noqa: BLE001
                rec["artifact_error"] = repr(e)[:300]
            out.append(rec)
    return out


# ------------------------------------------------------------------------------------------ worker

def _hook(config, events):
    import torch._inductor.codegen.triton as ct
    import inductor_divmod as R
    import inductor_narrow as NR
    orig = ct.TritonScheduling.define_kernel
    sources = {}

    def define_kernel(self, src_code, node_schedule, kernel):
        if config == "N":
            new, rec = R.rewrite(src_code, "N")
        elif config == "N1":
            elig = NR.eligibility(kernel)
            new, rec = NR.rewrite_n1(src_code, elig)
            rec["eligibility"] = elig
        else:
            new, rec = src_code, {"variant": "B0", "before": R.sha(src_code), "rewritten": False, "reason": "baseline"}
        name = orig(self, new, node_schedule, kernel)
        sig = re.search(r"'signature': \{([^}]*)\}", new)
        rec.update({"kernel_name": name, "has_divmod": "xindex % ks" in src_code,
                    "xnumel_type": (dict(re.findall(r"'(\w+)': '([^']+)'", sig.group(1))).get("xnumel") if sig else None)})
        for s in (src_code, new):
            sources.setdefault(R.sha(s), s)
        events.append(rec)
        return name

    ct.TritonScheduling.define_kernel = define_kernel
    return sources


def worker(config: str, mode: str, do_fallback: bool) -> dict:
    import logging
    import torch
    import torch._dynamo as dynamo
    import torch._inductor.config as ic
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(dynamo.config, attr):
            setattr(dynamo.config, attr, 64)
    if mode == "fixed":
        ic.triton.autotune_pointwise = False
    recompile_log = []

    class H(logging.Handler):
        def emit(self, record):
            m = record.getMessage()
            if "ecompil" in m:
                recompile_log.append(m[:600])
    torch._logging.set_logs(recompiles=True)
    logging.getLogger("torch").addHandler(H())

    events = []
    sources = _hook(config, events) if config != "static" else {}
    fn = torch.compile(S12.nested_cat_add, dynamic=(config != "static"))
    shapes = list(ORDER)
    ins = {s: S12.make_inputs(*s) for s in shapes}
    if config != "static":
        for s in shapes:
            for t in ins[s]:
                dynamo.mark_dynamic(t, t.dim() - 1)
    ref = {s: S12.nested_cat_add(*ins[s]) for s in shapes}
    torch.cuda.synchronize()
    sync = torch.cuda.synchronize
    tel = Telemetry()
    rec = {"config": config, "mode": mode, "telemetry_ok": tel.ok, "telemetry_error": tel.err,
           "telemetry_static": tel.static}

    cold, bitwise = [], {}
    t_pass = time.perf_counter()
    for s in shapes:
        c0, e0 = S12._counters(), len(events)
        t0 = time.perf_counter()
        out = fn(*ins[s])
        sync()
        first = time.perf_counter() - t0
        bitwise[f"{s[0]}x{s[1]}"] = bool(out.shape == ref[s].shape and torch.equal(out.view(torch.int16),
                                                                                  ref[s].view(torch.int16)))
        for _ in range(CALLS - 1):
            fn(*ins[s])
        sync()
        c1 = S12._counters()
        cold.append({"shape": list(s), "first_call_s": first, "counters_delta": {k: c1[k] - c0[k] for k in c0},
                     "kernels_defined": len(events) - e0})
    rec.update({"cold_pass_s": time.perf_counter() - t_pass, "cold": cold, "bitwise_equal_eager": bitwise})

    warm = []
    tel.start("warm")
    for _ in range(WARM_PASSES):
        c0, e0 = S12._counters(), len(events)
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(len(shapes) + 1)]
        t0 = time.perf_counter()
        evs[0].record()
        for i, s in enumerate(shapes):
            for _ in range(CALLS):
                fn(*ins[s])
            evs[i + 1].record()
        sync()
        pass_s = time.perf_counter() - t0
        c1 = S12._counters()
        warm.append({"pass_s": pass_s,
                     "per_call_ms": [evs[i].elapsed_time(evs[i + 1]) / CALLS for i in range(len(shapes))],
                     "counters_delta": {k: c1[k] - c0[k] for k in c0}, "kernels_defined": len(events) - e0})
    tel.stop()
    rec["warm"] = warm

    # idle probe: GPU idle before each call
    idle = []
    tel.start("idle")
    for s in shapes:
        ts = []
        for _ in range(IDLE_CALLS):
            time.sleep(IDLE_SLEEP)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn(*ins[s])
            b.record()
            sync()
            ts.append(a.elapsed_time(b) * 1e3)
        idle.append({"shape": list(s), "event_us": statistics.median(ts)})
    tel.stop()
    rec["idle_probe"] = idle
    rec["telemetry"] = {"warm": summarize(tel.samples, "warm"), "idle": summarize(tel.samples, "idle")}

    # isolated profile (as stage 12) and one full warm pass under the profiler
    from torch.profiler import ProfilerActivity, profile
    iso = []
    for s in shapes:
        per = []
        for _ in range(5):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn(*ins[s])
                sync()
            per.append(sum(e.device_time for e in prof.events() if e.device_type.name == "CUDA"))
        iso.append({"shape": list(s), "kernel_us": statistics.median(per)})
    rec["profile_isolated"] = iso
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for s in shapes:
            for _ in range(CALLS):
                fn(*ins[s])
        sync()
        prof_pass_s = time.perf_counter() - t0
    allk = [(e.time_range.start, e.time_range.end, e.name[:60]) for e in prof.events() if e.device_type.name == "CUDA"]
    kev = sorted([k for k in allk if k[2].startswith(("triton_", "repack"))], key=lambda x: x[0])
    per_shape = []
    for i in range(len(shapes)):
        chunk = kev[i * CALLS:(i + 1) * CALLS] if len(kev) == len(shapes) * CALLS else []
        per_shape.append(statistics.median(e - b for b, e, _ in chunk) if chunk else None)
    span = (kev[-1][1] - kev[0][0]) if kev else None
    ksum = sum(e - b for b, e, _ in kev)
    rec["full_pass"] = {"pass_s": prof_pass_s, "kernels": len(kev), "other_device_events": len(allk) - len(kev),
                        "kernel_sum_us": ksum, "span_us": span,
                        "gap_fraction": (1 - ksum / span) if span else None,
                        "per_shape_median_kernel_us": per_shape,
                        "kernel_names": sorted({n for _, _, n in kev})}

    rec["artifacts"] = artifacts()
    rec["counters_final"] = S12._counters()
    rec["recompile_log_seq"] = list(recompile_log)
    rec["hook_events"] = list(events)        # snapshot: the fallback shape appends its own events later
    rec["sources"] = dict(sources)

    if do_fallback and config != "static":
        del ins, ref
        gc.collect()
        torch.cuda.empty_cache()
        rec["fallback"] = fallback(fn, events, recompile_log)
    return rec


def fallback(fn, events, recompile_log) -> dict:
    """One shape beyond the int32 guard: numel = 2^31 exactly."""
    import torch
    import torch._dynamo as dynamo
    n, (wq, wk, wv) = FALLBACK_SHAPE
    W = 2 * (wq + wk + wv)
    out = {"n": n, "widths": [wq, wk, wv], "numel": n * W}
    try:
        g = torch.Generator(device="cuda").manual_seed(1)
        ins = [torch.randn(n, w, device="cuda", generator=g, dtype=torch.float32).to(torch.bfloat16)
               for w in (wq, wk, wv, wv, wq, wk, wv, wv)]
        for t in ins:
            dynamo.mark_dynamic(t, t.dim() - 1)
        e0, r0 = len(events), len(recompile_log)
        got = fn(*ins)
        torch.cuda.synchronize()
        new_events = [{k: v for k, v in e.items() if k != "eligibility"} for e in events[e0:]]
        ref = S12.nested_cat_add(*ins)
        out.update({"bitwise_equal_eager": bool(torch.equal(got.view(torch.int16), ref.view(torch.int16))),
                    "kernels_defined": new_events, "recompile_log": recompile_log[r0:][:5],
                    "any_rewrite_fired": any(e.get("rewritten") for e in new_events),
                    "max_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
    except Exception:  # noqa: BLE001
        out["error"] = traceback.format_exc()[-2000:]
    return out


# ------------------------------------------------------------------------------------------ driver

def _sub(config, mode, do_fallback, timeout=2400) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"vt13_{config}_{mode}_") as d:
        env = {**os.environ, "TORCHINDUCTOR_CACHE_DIR": f"{d}/inductor", "TRITON_CACHE_DIR": f"{d}/triton",
               "TORCHINDUCTOR_FX_GRAPH_CACHE": "0", "TORCHINDUCTOR_AUTOGRAD_CACHE": "0"}
        env.pop("TRITON_INTERPRET", None)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, __file__, "--worker", config, mode, str(int(do_fallback))],
                           capture_output=True, text=True, timeout=timeout, env=env)
        wall = time.perf_counter() - t0
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not lines:
        return {"config": config, "mode": mode, "error": (r.stderr or r.stdout)[-4000:], "process_s": wall}
    out = json.loads(lines[-1][len("RESULT "):])
    out["process_s"] = wall
    return out


def T(r):
    return statistics.median(p["pass_s"] for p in r["warm"])


def S(r, i):
    return statistics.median(p["per_call_ms"][i] for p in r["warm"])


def valid(r, base) -> dict:
    v = {"ran": "error" not in r}
    if not v["ran"]:
        return {**v, "valid": False}
    v["bitwise_all"] = all(r["bitwise_equal_eager"].values())
    v["no_warm_compiles"] = all(p["counters_delta"]["frames_total"] == 0 and p["counters_delta"]["unique_graphs"] == 0
                                and p["kernels_defined"] == 0 for p in r["warm"])
    if r["config"] in ("N", "N1"):
        div = [e for e in r["hook_events"] if e["has_divmod"]]
        v["rewrite_fired"] = bool(div) and all(e["rewritten"] for e in div)
        v["same_compile_count_as_B0"] = bool(base) and "error" not in base and (
            r["counters_final"]["unique_graphs"] == base["counters_final"]["unique_graphs"]
            and len(r["hook_events"]) == len(base["hook_events"]))
    if r["config"] == "N1":
        v["eligibility_H1_H4"] = all(all(e.get("eligibility", {}).get("checks", {}).get(h) for h in ("H1", "H2", "H3", "H4"))
                                     for e in r["hook_events"] if e["has_divmod"])
    if "fallback" in r:
        fb = r["fallback"]
        v["fallback_ok"] = ("error" not in fb and fb.get("bitwise_equal_eager") is True
                            and not fb.get("any_rewrite_fired"))
    v["valid"] = all(x for k, x in v.items() if k != "valid")
    return v


def decide(rounds) -> dict:
    res = {"per_round": []}
    for rd in rounds:
        by = {r["config"]: r for r in rd if r.get("mode") == "auto"}
        val = {c: valid(r, by.get("B0")) for c, r in by.items()}
        ok = all(val.get(c, {}).get("valid") for c in ("B0", "N", "N1"))
        row = {"validity": val}
        if ok:
            b0, n, n1 = by["B0"], by["N"], by["N1"]
            row.update({"T_s": {c: T(by[c]) for c in by if val[c]["valid"]},
                        "a_T_B0_over_N1": T(b0) / T(n1), "b_T_N_over_N1": T(n) / T(n1),
                        "b_worst_shape": min(S(n, i) / S(n1, i) for i in range(len(ORDER)))})
            row["a"] = row["a_T_B0_over_N1"] >= 1.10
            row["b"] = row["b_T_N_over_N1"] >= 0.99 and row["b_worst_shape"] >= 0.97
            row["c"] = row["b_T_N_over_N1"] >= 1.03
        else:
            row.update({"a": False, "b": False, "c": False, "why": "a configuration was invalid or missing"})
        res["per_round"].append(row)
    a = all(r["a"] for r in res["per_round"])
    b = all(r["b"] for r in res["per_round"])
    c = all(r["c"] for r in res["per_round"])
    res["adopt_N1"] = a and b
    res["N1_improves_on_N"] = a and b and c
    res["outcome"] = ("adopt N1; improves on N" if res["N1_improves_on_N"] else
                      "adopt N1 (not slower than N)" if res["adopt_N1"] else
                      "keep guarded N; N1 proved but not adopted")
    res["diagnosis"] = diagnose(rounds[0])
    return res


def diagnose(rd) -> dict:
    out = {}
    for r in rd:
        if r.get("mode") != "auto" or "error" in r:
            continue
        fp, iso = r["full_pass"], r["profile_isolated"]
        ratios = [k / i["kernel_us"] for k, i in zip(fp["per_shape_median_kernel_us"], iso) if k and i["kernel_us"]]
        kr = statistics.median(ratios) if ratios else None
        tw, ti = r["telemetry"]["warm"], r["telemetry"]["idle"]
        clk = (tw.get("sm_median") / ti["sm_median"]) if tw.get("n") and ti.get("n") else None
        lim = max(((tw.get("reason_fraction") or {}).get(k, 0) for k in ("sw_power_cap", "hw_power_brake",
                                                                        "sw_thermal", "hw_thermal", "hw_slowdown")),
                  default=0)
        d1 = kr is not None and kr >= 1.15 and ((clk is not None and clk <= 0.90) or lim >= 0.5)
        d2 = kr is not None and abs(kr - 1) <= 0.10 and (fp["gap_fraction"] or 0) >= 0.15
        out[r["config"]] = {"k_seq_over_k_iso_median": kr, "sm_clock_warm_over_idle": clk,
                            "limit_reason_max_fraction": lim, "gap_fraction": fp["gap_fraction"],
                            "D1_device_state": d1, "D2_dispatch_gaps": d2,
                            "verdict": "D1" if d1 else "D2" if d2 else "unassigned"}
    return out


def env():
    import torch
    import triton
    return {"device": "cuda", "gpu_name": torch.cuda.get_device_name(), "torch": torch.__version__,
            "triton": triton.__version__, "compute_capability": list(torch.cuda.get_device_capability()),
            "cpu": subprocess.run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo; nproc"],
                                  capture_output=True, text=True).stdout.strip(),
            "nvidia_smi": subprocess.run(["bash", "-c", "nvidia-smi -q -d POWER,CLOCK,PERFORMANCE 2>&1 | head -80"],
                                         capture_output=True, text=True).stdout[-4000:],
            "TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET")}


def bench() -> dict:
    rec = {"kind": "stage 13: N1 vs guarded N, sustained-time diagnosis (criterion in module docstring)",
           **env(), "order": ORDER, "configs": CONFIGS, "modes": MODES, "rounds": []}
    for rnd in range(ROUNDS):
        order = CONFIGS if rnd == 0 else CONFIGS[::-1]
        rows = []
        for mode in MODES:
            for c in order:
                rows.append(_sub(c, mode, do_fallback=(rnd == 0 and mode == "auto")))
        rec["rounds"].append(rows)
    rec["decision"] = decide(rec["rounds"])
    import launch_local_check as LC
    rec["input_hashes"] = LC.input_hashes()
    return rec


def dry() -> dict:
    import inductor_narrow as NR
    fx = (REPO / "bench/optimizations/inductor_divmod/fixtures/gpu_emitted_B0.py").read_text()
    n, (wq, wk, wv) = FALLBACK_SHAPE
    return {"configs": CONFIGS, "modes": MODES, "rounds": ROUNDS, "processes": len(CONFIGS) * len(MODES) * ROUNDS,
            "structural_on_fixture": NR.structural(fx), "fallback_numel": n * 2 * (wq + wk + wv),
            "fallback_exceeds_int32_guard": n * 2 * (wq + wk + wv) > 2 ** 31 - 1}


if __name__ == "__main__":
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        c, m, fb = sys.argv[i + 1], sys.argv[i + 2], sys.argv[i + 3] == "1"
        try:
            print("RESULT " + json.dumps(worker(c, m, fb), default=str))
        except Exception:  # noqa: BLE001
            print("RESULT " + json.dumps({"config": c, "mode": m, "error": traceback.format_exc()[-4000:]}))
    elif "--dry" in sys.argv:
        print(json.dumps(dry(), indent=1))
    else:
        print(json.dumps(bench(), indent=1, default=str))
