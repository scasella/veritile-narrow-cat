#!/usr/bin/env python3
"""Adversarial suite for the add_example launch contract and evidence pipeline.

Each case records WHAT happened, using exactly one outcome class:

  violated_obligation   kernel-checked checker rejection naming P1–P10
  concrete_witness      a concrete counterexample (Lean-checked or, where
                        labelled, the untrusted flat-memory emulator)
  evidence_rejection    rejected by a statement / axiom / placeholder /
                        source-integrity / freshness gate
  unsupported_input     rejected by the adapter as outside the modelled subset
  infrastructure_failure  a required tool is unavailable
  accepted              valid or benign case accepted (controls)

Checker rejections, syntax rejections, stale evidence and infrastructure
failures are NOT counted as discovered semantic bugs; only
`concrete_witness` entries demonstrate wrong behavior, and each states whether
the witness is Lean-checked or emulator-only.

Usage: python3 scripts/launch_mutation_suite.py [--out FILE]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import launch_check as L  # noqa: E402
import launch_emulate as E  # noqa: E402
import launch_local_check as LC  # noqa: E402

KDIR = REPO / "bench/tritonbench_g/add_example"
PY = KDIR / "add_example.py"
MAN = KDIR / "launch_manifest.json"
ELAN = str(Path.home() / ".elan/bin")
os.environ["PATH"] = f"{ELAN}:{os.environ['PATH']}"
os.environ.setdefault("LEAN_NUM_THREADS", "2")

results: list[dict] = []


def record(cid, category, mutation, outcome, detail, semantic=None):
    r = {"id": cid, "category": category, "mutation": mutation, "outcome": outcome,
         "detail": detail}
    if semantic is not None:
        r["semantic_witness"] = semantic
    results.append(r)
    print(f"{cid:<34} {outcome:<22} {detail if isinstance(detail, str) else ''}"[:160], flush=True)


def lean(text: str, timeout=900) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
        f.write(text)
    try:
        r = subprocess.run(["lake", "env", "lean", f.name], cwd=REPO, capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    finally:
        os.unlink(f.name)


def with_manifest(src_text: str, cases: list[dict]) -> dict:
    """Run the adapter on a temporary copy of the kernel directory."""
    d = Path(tempfile.mkdtemp(prefix="launch-mut-"))
    try:
        (d / "add_example.py").write_text(src_text)
        shutil.copy2(KDIR / "AddExample.lean", d / "AddExample.lean")
        man = json.loads(MAN.read_text())
        man["cases"] = cases
        (d / "launch_manifest.json").write_text(json.dumps(man))
        # analyze() reports paths relative to REPO; place the temp dir inside it
        inside = REPO / "bench/tritonbench_g" / f".mut_{d.name}"
        shutil.copytree(d, inside)
        try:
            return L.analyze(inside / "launch_manifest.json")
        finally:
            shutil.rmtree(inside, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def mutate(old, new, src=None):
    src = PY.read_text() if src is None else src
    assert old in src, old
    return src.replace(old, new, 1)


# --------------------------------------------------------------------------
# 1. Configuration faults: checker verdicts + concrete witnesses
# --------------------------------------------------------------------------

def config_faults():
    import torch
    res = L.analyze(MAN)
    for name, c in res["cases"].items():
        if c["accepted"]:
            record(name, "valid-control", "none", "accepted",
                   f"n={c['config']['n']} grid={c['config']['grid']}; Pre proved via check_ok")
            continue
        if not c["matches_expectation"]:
            record(name, "config-fault", name, "infrastructure_failure",
                   f"unexpected verdict {c['failed_obligations']}")
            continue
        record(name, "config-fault", name, "violated_obligation",
               f"¬Pre kernel-checked; failed {c['failed_obligations']}")
    # concrete witnesses for selected faults (emulator = untrusted test oracle)
    torch.manual_seed(0)
    b = torch.randn(15); x = b[::2]; y = torch.randn(8); o = torch.zeros_like(x)
    r = E.emulate_add(x, y, o, 8, 4, 2)
    bad = int((r["out"] != E.reference_add(x, y)).sum())
    record("W_M1_stride", "witness", "x = base[::2]", "concrete_witness",
           f"emulator: {bad}/8 outputs differ from reference (kernel ignores stride)",
           semantic="emulator-only")
    x = torch.randn(16); y = torch.randn(12); o = torch.zeros_like(x)
    r = E.emulate_add(x, y, o, 16, 4, 4)
    record("W_M2_short_y", "witness", "y.numel()=12 < n=16", "concrete_witness",
           f"emulator: {len(r['events'])} out-of-allocation reads: {r['events'][:2]}",
           semantic="emulator-only")
    x = torch.randn(18); y = torch.randn(18); o = torch.zeros_like(x)
    r = E.emulate_add(x, y, o, 18, 4, 4)
    record("W_M3_short_grid", "witness", "grid=(18//4,)", "concrete_witness",
           f"emulator: out[16:18]={r['out'][16:].tolist()} ≠ ref "
           f"{E.reference_add(x, y)[16:].tolist()}", semantic="emulator-only")
    base = torch.randn(9); keep = base.clone(); y = torch.randn(8)
    r1 = E.emulate_add(base[:8], y, base[1:9], 8, 4, 2, "asc")["out"]
    base.copy_(keep)
    r2 = E.emulate_add(base[:8], y, base[1:9], 8, 4, 2, "desc")["out"]
    record("W_M6_overlap_race", "witness", "out = x shifted by 1 element", "concrete_witness",
           f"emulator: result depends on program order (asc≠desc: {not bool((r1 == r2).all())})",
           semantic="emulator-only")


# --------------------------------------------------------------------------
# 2. Lean-checked witnesses (i32 overflow, kernel mutants)
# --------------------------------------------------------------------------

def lean_witnesses():
    rc, out = lean((REPO / "bench/tests/Blocked1DLaunchWitnesses.lean").read_text())
    if rc != 0:
        record("W_lean", "witness", "bench/tests/Blocked1DLaunchWitnesses.lean",
               "infrastructure_failure", out[-800:])
        return
    for tag, desc in [("i32_n_truncated", "M4: n=3·2^30 as i32 is negative ⇒ every lane masked off"),
                      ("i32_offset_wraps", "M5: pid=2^29, BLOCK=4 ⇒ offset wraps to -2^31 and passes mask"),
                      ("unmasked_store_frame_violation",
                       "K1: store without mask writes out[7] with n=5 (frame clause false)"),
                      ("offbyone_mask_frame_violation",
                       "K2: mask `offsets <= n` writes out[n] (frame clause false)")]:
        ok = re.search(rf"{tag}.*axiom footprint ⊆ standard base ✓", out)
        record(f"W_{tag}", "witness", desc, "concrete_witness" if ok else "infrastructure_failure",
               "Lean-checked theorem, axiom-clean" if ok else "theorem missing from output",
               semantic="lean-checked")


# --------------------------------------------------------------------------
# 3. Source mutants of add_example.py
# --------------------------------------------------------------------------

def source_mutants():
    f32 = lambda n: {"numel": n, "dtype": "float32"}
    cases = [{"name": "n16", "tensors": {"x": f32(16), "y": f32(16)}},
             {"name": "n18", "tensors": {"x": f32(18), "y": f32(18)}}]
    muts = {
        "S1_floor_grid": ("num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE",
                          "num_blocks = n_elements // BLOCK_SIZE"),
        "S2_block_6": ("BLOCK_SIZE = 4", "BLOCK_SIZE = 6"),
        "S3_missing_store_mask": ("tl.store(out_ptr + offsets, output, mask=mask)",
                                  "tl.store(out_ptr + offsets, output)"),
        "S4_offbyone_mask": ("mask = offsets < n_elements", "mask = offsets <= n_elements"),
        "S5_load_other_kwarg": ("x = tl.load(in_ptr0 + offsets, mask=mask)",
                                "x = tl.load(in_ptr0 + offsets, mask=mask, other=0.0)"),
        "S6_helper_call": ("output = x + y", "output = my_add(x, y)"),
        "S7_launch_option": ("add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)",
                             "add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE, num_warps=8)"),
    }
    for cid, (old, new) in muts.items():
        try:
            res = with_manifest(mutate(old, new), cases)
            v = {k: (c["accepted"], c["failed_obligations"]) for k, c in res["cases"].items()}
            if all(a for a, _ in v.values()):
                record(cid, "source-mutant", new, "accepted", f"UNDETECTED: {v}")
            else:
                record(cid, "source-mutant", new, "violated_obligation", f"verdicts {v}")
        except L.Unsupported as e:
            record(cid, "source-mutant", new, "unsupported_input", str(e).splitlines()[0])
    # benign edits
    src = PY.read_text()
    res = with_manifest(src.replace("# Calculate the number of blocks needed",
                                    "# number of programs (cdiv)"), cases)
    acc = all(c["accepted"] for c in res["cases"].values())
    record("B1_comment_only_edit", "benign", "comment edit in wrapper",
           "accepted" if acc else "infrastructure_failure",
           f"verdicts unchanged; add_example.py hash changes → ledger/external evidence STALE")
    try:
        with_manifest(src.replace("output = x + y", "result = x + y").replace(
            "tl.store(out_ptr + offsets, output, mask=mask)",
            "tl.store(out_ptr + offsets, result, mask=mask)"), cases)
        record("B2_python_only_rename", "benign", "rename in Python only", "accepted",
               "UNDETECTED transcription drift")
    except L.Unsupported as e:
        record("B2_python_only_rename", "benign", "rename in Python only (Lean not updated)",
               "evidence_rejection", "transcription drift detected: " + str(e).splitlines()[0])


# --------------------------------------------------------------------------
# 4. Evidence-pipeline mutants
# --------------------------------------------------------------------------

def pipeline_mutants(scratch: Path):
    cfg = (REPO / "VeriTile/Triton/Launch/Blocked1DConfig.lean").read_text()
    olean_dir = scratch / "ol/VeriTile/Triton/Launch"
    olean_dir.mkdir(parents=True, exist_ok=True)
    frozen_cfg = (KDIR / "launch_evidence/frozen/config_surface.txt").read_text()
    printer = (KDIR / "launch_evidence/frozen/print_config_surface.lean").read_text()

    def build_cfg(text):
        f = scratch / "Blocked1DConfig.lean"
        f.write_text(text)
        r = subprocess.run(["lean", str(f), "-o", str(olean_dir / "Blocked1DConfig.olean")],
                           capture_output=True, text=True, cwd=scratch)
        return r.returncode, r.stdout + r.stderr

    def surface():
        p = scratch / "print.lean"
        p.write_text(printer)
        r = subprocess.run(["lean", str(p)], capture_output=True, text=True, cwd=scratch,
                           env={**os.environ, "LEAN_PATH": str(scratch / "ol")})
        return r.stdout

    # E1 weakened contract: coverage obligation weakened (≤ instead of <)
    weak = cfg.replace("∀ i, i < c.n → i / c.block < c.gridX",
                       "∀ i, i < c.n → i / c.block ≤ c.gridX")
    rc, out = build_cfg(weak)
    changed = surface() != frozen_cfg
    record("E1_weakened_contract", "pipeline", "Covers: `<` → `≤` (proofs still compile: "
           f"{'yes' if rc == 0 else 'no'})",
           "evidence_rejection" if changed else "accepted",
           "frozen-contract diff: CONTRACT_CHANGED" if changed else "UNDETECTED")
    # E2 unapproved axiom
    ax = cfg.replace("theorem check_ok (c : Blocked1DLaunch) (h : check c = true) : Pre c := by",
                     "axiom cheat_pre : ∀ c, Blocked1DLaunch.Pre c\n\n"
                     "theorem check_ok (c : Blocked1DLaunch) (h : check c = true) : Pre c := cheat_pre c\n"
                     "theorem check_ok_orig (c : Blocked1DLaunch) (h : check c = true) : Pre c := by")
    rc, out = build_cfg(ax + "\nopen VeriTile.Triton.Blocked1DLaunch in\n#print axioms check_ok\n")
    hit = "cheat_pre" in out
    record("E2_unapproved_axiom", "pipeline", "check_ok proved from a new axiom",
           "evidence_rejection" if hit else "accepted",
           "axiom footprint contains cheat_pre (placeholder scan also flags `axiom`)"
           if hit else "UNDETECTED")
    # E3 sorry
    so = re.sub(r"(theorem check_complete[^\n]*\n)(?:.*\n)*?(?=\n/-!)",
                r"\1  sorry\n", cfg, count=1)
    rc, out = build_cfg(so + "\nopen VeriTile.Triton.Blocked1DLaunch in\n#print axioms check_complete\n")
    hit = "sorryAx" in out
    record("E3_sorry", "pipeline", "check_complete := sorry",
           "evidence_rejection" if hit else "accepted",
           "sorryAx in axioms; placeholder scan flags `sorry`" if hit else "UNDETECTED")
    build_cfg(cfg)  # restore scratch olean
    # E4 omitted target: drop the launch headline from AddExample.lean
    ex = (KDIR / "AddExample.lean").read_text()
    cut = ex.index("/-- **Whole-launch headline.**")
    end = ex.index("end VeriTile.Bench.TritonBenchG.AddExample")
    gate = ("\nopen VeriTile.Meta\n"
            "#axiomsClean VeriTile.Bench.TritonBenchG.AddExample.add_kernel_launch_correctness\n")
    rc0, _ = lean(LC.AUDIT_IMPORT + ex + gate)            # control: intact file passes the gate
    rc, out = lean(LC.AUDIT_IMPORT + ex[:cut] + ex[end:] + gate)
    detected = rc0 == 0 and rc != 0 and "unknown" in out.lower()
    record("E4_omitted_target", "pipeline", "delete add_kernel_launch_correctness",
           "evidence_rejection" if detected else ("accepted" if rc == 0 else "infrastructure_failure"),
           f"control rc={rc0}; mutant rc={rc}: unknown constant at the axiom/inventory gate"
           if detected else f"UNDETECTED (control rc={rc0}, mutant rc={rc})")
    # E5 stale source hash vs external evidence
    ev = scratch / "evid"; ev.mkdir(exist_ok=True)
    h = LC.input_hashes()
    stale = dict(h); stale["bench/tritonbench_g/add_example/add_example.py"] = "0" * 64
    (ev / "official_comparator.json").write_text(json.dumps({"exit_code": 0, "input_hashes": stale}))
    r = subprocess.run([sys.executable, str(REPO / "scripts/launch_local_check.py"), "--external-only",
                        "--require-official", "--evidence-dir", str(ev)], capture_output=True, text=True)
    record("E5_stale_source_hash", "pipeline", "official evidence recorded for other add_example.py",
           "evidence_rejection" if r.returncode != 0 and "STALE" in r.stdout else "accepted",
           f"--require-official rc={r.returncode}: " + r.stdout.strip().splitlines()[-1])
    # E6 fresh successful evidence is accepted (control for E5)
    (ev / "official_comparator.json").write_text(json.dumps({"exit_code": 0, "input_hashes": h}))
    r = subprocess.run([sys.executable, str(REPO / "scripts/launch_local_check.py"), "--external-only",
                        "--require-official", "--evidence-dir", str(ev)], capture_output=True, text=True)
    record("E6_fresh_evidence_control", "pipeline", "synthetic fresh evidence file (test only)",
           "accepted" if r.returncode == 0 else "infrastructure_failure", f"rc={r.returncode}")
    # E7 missing result
    (ev / "official_comparator.json").unlink()
    r = subprocess.run([sys.executable, str(REPO / "scripts/launch_local_check.py"), "--external-only",
                        "--require-official", "--require-gpu", "--evidence-dir", str(ev)],
                       capture_output=True, text=True)
    record("E7_missing_result", "pipeline", "no official/GPU result files",
           "evidence_rejection" if r.returncode != 0 else "accepted",
           f"rc={r.returncode}: {' | '.join(l for l in r.stdout.splitlines() if 'REQUIRED' in l)}")
    # E9 edit inside pinned upstream code (additive-only gate), on a scratch copy
    pin_root = scratch / "pin"
    for f in LC.ADDITIVE:
        (pin_root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, pin_root / f)
    comp = pin_root / "VeriTile/Triton/Launch/Composition.lean"
    comp.write_text(comp.read_text().replace("h_disjoint : Kernel.GridWritesDisjoint frames",
                                             "h_disjoint : True", 1))
    probs = LC.pin_integrity(pin_root)
    record("E9_upstream_pin_edit", "pipeline", "weaken GridLaunchedOrdinary.h_disjoint in pinned code",
           "evidence_rejection" if probs else "accepted",
           "; ".join(probs) if probs else "UNDETECTED")
    # E8 unavailable required tool: adapter without lake on PATH
    env = {k: v for k, v in os.environ.items()}
    env["PATH"] = "/usr/bin:/bin"
    r = subprocess.run([sys.executable, str(REPO / "scripts/launch_check.py"), str(MAN)],
                       capture_output=True, text=True, env=env)
    status = json.loads(r.stdout).get("status") if r.stdout.strip().startswith("{") else r.stdout[-200:]
    record("E8_tool_unavailable", "pipeline", "PATH without lake/lean",
           "infrastructure_failure" if status == "infrastructure_failure" else "accepted",
           f"adapter status={status}, rc={r.returncode} (no verdict produced)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=KDIR / "launch_evidence/mutation_results.json")
    a = ap.parse_args(argv)
    t = time.time()
    scratch = Path(tempfile.mkdtemp(prefix="launch-suite-"))
    try:
        config_faults()
        lean_witnesses()
        source_mutants()
        pipeline_mutants(scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    undetected = [r["id"] for r in results if "UNDETECTED" in str(r["detail"])]
    summary = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "elapsed_s": round(time.time() - t, 1), "counts": counts,
               "undetected": undetected, "input_hashes": LC.input_hashes(), "results": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(counts), "undetected:", undetected)
    unexpected_infra = [r["id"] for r in results if r["outcome"] == "infrastructure_failure"
                        and r["id"] != "E8_tool_unavailable"]
    return 1 if undetected or unexpected_infra else 0


if __name__ == "__main__":
    sys.exit(main())
