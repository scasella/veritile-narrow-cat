#!/usr/bin/env python3
"""Local verification for the host-launch checker and its add_example case.

Separately named from the upstream release gates (`scripts/check-artifact.sh`,
`bench/audit_trust.sh`), which require the official comparator on Linux and
are neither weakened nor replaced here.

    python3 scripts/launch_local_check.py                 # local gates -> LOCAL_CHECKS_PASSED
    python3 scripts/launch_local_check.py --require-official   # also demand a fresh official comparator result
    python3 scripts/launch_local_check.py --require-gpu        # also demand fresh GPU correctness+perf results

Every step is recorded in the ledger (`--ledger`, default
bench/tritonbench_g/add_example/launch_evidence/ledger.json) with status,
scope, assumptions, evidence and input hashes. Official/GPU evidence is read
from `launch_evidence/official_comparator.json` and `launch_evidence/gpu.json`;
missing, stale (hash mismatch) or unsuccessful evidence is `not_run`/`failed`
and makes the corresponding `--require-*` invocation fail.
"""
from __future__ import annotations

import argparse
import hashlib
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
sys.path.insert(0, str(REPO / "bench"))
import audit_source  # noqa: E402
import launch_check  # noqa: E402

KDIR = REPO / "bench/tritonbench_g/add_example"
EVID = KDIR / "launch_evidence"
FROZEN = EVID / "frozen"
NEW_LEAN = ["VeriTile/Triton/Launch/Blocked1DConfig.lean",
            "VeriTile/Triton/Launch/Blocked1D.lean",
            "VeriTile/Triton/Launch/Composition.lean",
            "bench/tritonbench_g/add_example/AddExample.lean"]
PINNED_TOOLCHAIN = "leanprover/lean4:v4.29.0"
PINNED_MATHLIB = "8a178386ffc0f5fef0b77738bb5449d50efeea95"
HEADLINES = ["VeriTile.Bench.TritonBenchG.AddExample.add_kernel_launch_correctness",
             "VeriTile.Bench.TritonBenchG.AddExample.add_kernel_launch_applicable",
             "VeriTile.Bench.TritonBenchG.AddExample.add_kernel_launch_traceSafe",
             "VeriTile.Bench.TritonBenchG.AddExample.add_kernel_correctness",
             "VeriTile.Triton.Blocked1D.launch_of_frames",
             "VeriTile.Triton.Kernel.LaunchCorrectFramed.toLaunchCorrect",
             "VeriTile.Triton.Blocked1DLaunch.check_ok",
             "VeriTile.Triton.Blocked1DLaunch.check_complete",
             "VeriTile.Triton.Blocked1DLaunch.Pre.i32_offset_toInt",
             "VeriTile.Triton.Blocked1DLaunch.Pre.i32_mask_eq",
             "VeriTile.Triton.Blocked1DLaunch.Pre.offset_injective"]

# Protected surface: printed from the elaborated environment and compared
# byte-for-byte with the frozen snapshot.
SURFACE_PRINTS = [
    "#print VeriTile.Triton.Kernel.LaunchCorrectFramed",
    "#print VeriTile.Triton.Kernel.LaunchCorrect",
    "#print VeriTile.Triton.Kernel.GridLaunchedOrdinary",
    "#print VeriTile.Triton.Kernel.mergeFrames",
    "#print VeriTile.Triton.Blocked1D.line",
    "#print VeriTile.Triton.Blocked1D.pidOf",
    "#print VeriTile.Triton.Blocked1D.blockWrites",
    "#print VeriTile.Triton.MaskedKernelIO₂.Implements",
    "#print VeriTile.Bench.TritonBenchG.AddExample.add_kernel",
    "#print VeriTile.Bench.TritonBenchG.AddExample.addIO",
] + [f"#check @{h}" for h in HEADLINES]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def input_hashes(root: Path = REPO) -> dict:
    files = NEW_LEAN + ["bench/tritonbench_g/add_example/add_example.py",
                        "bench/tritonbench_g/add_example/CONTRACT.md",
                        "bench/tritonbench_g/add_example/launch_manifest.json",
                        "scripts/launch_check.py", "scripts/launch_local_check.py",
                        "lean-toolchain", "lake-manifest.json", "lakefile.toml"]
    return {f: sha(root / f) for f in files if (root / f).exists()}


class Step:
    def __init__(self, key, scope, assumptions=""):
        self.key, self.scope, self.assumptions = key, scope, assumptions
        self.status, self.evidence, self.reason, self.detail = "not_run", "", "", {}
        self.t0 = time.time()

    def done(self, status, evidence="", reason="", **detail):
        self.status, self.evidence, self.reason, self.detail = status, evidence, reason, detail
        self.elapsed = round(time.time() - self.t0, 1)
        mark = {"passed": "ok", "failed": "FAIL"}.get(status, status.upper())
        print(f"[{mark}] {self.key}: {reason or evidence}", flush=True)
        return self

    def json(self):
        return {"status": self.status, "scope": self.scope, "assumptions": self.assumptions,
                "evidence": self.evidence, "reason": self.reason,
                "elapsed_s": getattr(self, "elapsed", None), **self.detail}


def run(cmd, cwd, timeout, log: Path | None = None):
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           start_new_session=True)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = f"TIMEOUT after {timeout}s\n{e.stdout or ''}", 124
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"$ {' '.join(map(str, cmd))}\n# cwd={cwd} rc={rc} "
                       f"elapsed={time.time()-t:.1f}s\n{out}")
    return rc, out


def fresh_workspace(logs: Path) -> Path:
    """Copy tracked + new project sources (no .lake/build) into a temp dir and
    link only the pinned dependency packages, so project modules are rebuilt
    from source rather than trusted from the working tree's build artifacts."""
    ws = Path(tempfile.mkdtemp(prefix="veritile-launch-fresh-"))
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.split()
    for f in files:
        src = REPO / f
        if src.is_file():
            dst = ws / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (ws / ".lake").mkdir()
    os.symlink(REPO / ".lake/packages", ws / ".lake/packages")
    return ws


def placeholder_scan(root: Path) -> list[str]:
    bad = []
    for f in NEW_LEAN:
        code = audit_source.strip_lean_comments((root / f).read_text())
        for m in re.finditer(r"\b(sorry|admit|native_decide|ofReduceBool)\b|^\s*axiom\s", code, re.M):
            bad.append(f"{f}: {m.group(0).strip()}")
    return bad


AUDIT_IMPORT = "import VeriTile.Meta.StatementAudit\n"
MARK = "SURFACE-BEGIN-7f3a"


def audit_file(root: Path) -> str:
    """AddExample.lean + appended audit commands (a temp copy; the source is untouched)."""
    body = (root / "bench/tritonbench_g/add_example/AddExample.lean").read_text()
    lines = ["", "open VeriTile.Meta"]
    lines += [f"#axiomsClean {h}" for h in HEADLINES]
    lines += ["namespace VeriTile.Bench.TritonBenchG.AddExample", "#auditModuleAxioms",
              "end VeriTile.Bench.TritonBenchG.AddExample"]
    return AUDIT_IMPORT + body + "\n".join(lines) + "\n"


def surface_file(root: Path) -> str:
    body = (root / "bench/tritonbench_g/add_example/AddExample.lean").read_text()
    lines = ["", "set_option pp.proofs false", f'#eval IO.println "{MARK}"'] + SURFACE_PRINTS
    return body + "\n".join(lines) + "\n"


def print_surface(root: Path, logs: Path, timeout: int) -> tuple[int, str]:
    tmp = root / "bench/tritonbench_g/add_example/.launch_surface_tmp.lean"
    tmp.write_text(surface_file(root))
    try:
        rc, out = run(["lake", "env", "lean", str(tmp)], root, timeout, logs / "surface.log")
    finally:
        tmp.unlink(missing_ok=True)
    if rc != 0 or MARK not in out:
        return rc or 1, ""
    return rc, out.split(MARK, 1)[1].strip("\n") + "\n"


def external_steps(hashes: dict) -> dict:
    """Official comparator / GPU: evidence files only, never inferred."""
    def external(key, fname, scope):
        st = Step(key, scope)
        f = EVID / fname
        if not f.exists():
            return st.done("not_run", "", f"no {fname}: requires a compatible external environment "
                           f"(see bench/tritonbench_g/add_example/HANDOFF.md)")
        ev = json.loads(f.read_text())
        stale = {k: v for k, v in ev.get("input_hashes", {}).items() if hashes.get(k) != v}
        if stale or not ev.get("input_hashes"):
            return st.done("failed", str(f), f"STALE evidence: {sorted(stale) or 'no hashes'}")
        return st.done("passed" if ev.get("exit_code") == 0 else "failed", str(f),
                       f"exit_code={ev.get('exit_code')}")
    return {
        "official_comparator": external(
            "official_comparator", "official_comparator.json",
            "upstream comparator (Linux, Landlock, systemd) on the changed library + AddExample.lean"),
        "gpu_correctness": external("gpu_correctness", "gpu.json",
                                    "real Triton launch on a supported NVIDIA/AMD GPU"),
        "gpu_performance": external("gpu_performance", "gpu_perf.json",
                                    "steady-state timing on a supported GPU"),
    }


def freeze(a) -> int:
    """Write the frozen protected-surface snapshot. Run only on the reviewed
    statement (sorry) state before proof search, or after a recorded review."""
    logs = EVID / "logs"
    rc, surface = print_surface(REPO, logs, a.timeout)
    cfg_rc, cfg_out = run(["lake", "env", "lean", str(FROZEN / "print_config_surface.lean")],
                          REPO, 600)
    if rc != 0 or cfg_rc != 0:
        print(f"FREEZE_FAILED rc={rc} cfg_rc={cfg_rc}")
        return 1
    FROZEN.mkdir(parents=True, exist_ok=True)
    (FROZEN / "launch_surface.txt").write_text(surface)
    (FROZEN / "config_surface.txt").write_text(cfg_out)
    (FROZEN / "contract.sha256").write_text(sha(KDIR / "CONTRACT.md") + "\n")
    print(f"FROZEN: {len(surface.splitlines())} surface lines, contract {sha(KDIR / 'CONTRACT.md')[:12]}")
    return 0


def external_only(a) -> int:
    steps = external_steps(input_hashes())
    rc = 0
    if a.require_official and steps["official_comparator"].status != "passed":
        print(f"OFFICIAL_AUDIT_REQUIRED_BUT_{steps['official_comparator'].status.upper()}")
        rc = 2
    if a.require_gpu and not all(steps[k].status == "passed"
                                 for k in ("gpu_correctness", "gpu_performance")):
        print("GPU_EVIDENCE_REQUIRED_BUT_MISSING_OR_FAILED")
        rc = rc or 3
    return rc


def main(argv=None) -> int:
    global EVID
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="rebuild project modules from source in a fresh workspace")
    ap.add_argument("--require-official", action="store_true")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--ledger", type=Path, default=EVID / "ledger.json")
    ap.add_argument("--freeze", action="store_true",
                    help="write the frozen surface snapshot (review-gated; records who/why)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--external-only", action="store_true",
                    help="only evaluate official/GPU evidence against current hashes")
    ap.add_argument("--evidence-dir", type=Path, default=None,
                    help="read external evidence from here (tests)")
    a = ap.parse_args(argv)
    if a.evidence_dir is not None:
        EVID = a.evidence_dir
    env_path = Path.home() / ".elan/bin"
    os.environ["PATH"] = f"{env_path}:{os.environ['PATH']}"
    os.environ.setdefault("LEAN_NUM_THREADS", "2")  # resource cap (see README)
    logs = EVID / "logs"
    steps: dict[str, Step] = {}
    root = REPO

    if a.external_only:
        return external_only(a)
    if a.freeze:
        return freeze(a)

    # L0 pins -------------------------------------------------------------
    s = Step("toolchain_pins", "lean-toolchain, Lean binary, Mathlib revision")
    tc = (REPO / "lean-toolchain").read_text().strip()
    man = json.loads((REPO / "lake-manifest.json").read_text())
    mrev = next(p["rev"] for p in man["packages"] if p["name"] == "mathlib")
    rc, out = run(["lake", "env", "lean", "--version"], REPO, 120)
    ok = tc == PINNED_TOOLCHAIN and mrev == PINNED_MATHLIB and rc == 0 and "4.29.0" in out
    steps[s.key] = s.done("passed" if ok else "failed", out.strip()[:120],
                          "" if ok else f"toolchain={tc} mathlib={mrev} rc={rc}")

    # L1 build ------------------------------------------------------------
    s = Step("build", "project modules rebuilt from source" if a.fresh else
             "lake build in working tree (incremental; use --fresh for final evidence)")
    if a.fresh:
        root = fresh_workspace(logs)
    rc, out = run(["lake", "build", "VeriTile.Triton.Launch.Blocked1D", "VeriTile.Triton",
                   "VeriTile.Meta.StatementAudit", "VeriTile.Examples.Common"],
                  root, a.timeout, logs / "build.log")
    steps[s.key] = s.done("passed" if rc == 0 else "failed", str(logs / "build.log"),
                          f"rc={rc}" + (f" workspace={root}" if a.fresh else ""))

    # L2 placeholder scan ---------------------------------------------------
    s = Step("placeholder_scan", "comment-stripped changed Lean files")
    bad = placeholder_scan(root)
    steps[s.key] = s.done("failed" if bad else "passed", "; ".join(bad) or "none",
                          f"{len(bad)} hits" if bad else "no sorry/admit/native_decide/axiom")

    # L3 kernel file + axioms + inventory + surface ----------------------
    s = Step("kernel_file_axioms", "AddExample.lean elaborates; every listed theorem's axioms "
             "⊆ {propext, Classical.choice, Quot.sound}; headline inventory")
    tmp = root / "bench/tritonbench_g/add_example/.launch_audit_tmp.lean"
    tmp.write_text(audit_file(root))
    try:
        rc, out = run(["lake", "env", "lean", str(tmp)], root, a.timeout, logs / "audit.log")
    finally:
        tmp.unlink(missing_ok=True)
    clean = sorted(set(re.findall(r"(\S+): axiom footprint ⊆ standard base ✓", out)))
    inv = re.search(r"Axiom audit: headlines=(\d+)\nheadlines: \[(.*?)\]", out, re.S)
    missing = [h for h in HEADLINES if h not in clean]
    inv_names = inv.group(2) if inv else ""
    inv_ok = inv is not None and all(n in inv_names for n in
                                     ["add_kernel_correctness", "add_kernel_launch_correctness"])
    ok = rc == 0 and not missing and inv_ok
    steps[s.key] = s.done("passed" if ok else "failed", str(logs / "audit.log"),
                          f"rc={rc}, axiom-clean {len(clean)}/{len(HEADLINES)}, "
                          f"inventory={'ok' if inv_ok else 'MISSING'}"
                          + (f", missing={missing}" if missing else ""),
                          axiom_clean=clean, inventory=inv_names)

    # L4 frozen contract ---------------------------------------------------
    s = Step("frozen_contract", "elaborated protected definitions + headline statements "
             "vs frozen snapshot; CONTRACT.md and checker surface hashes")
    srf_rc, surface = print_surface(root, logs, a.timeout)
    cfg_rc, cfg_out = run(["lake", "env", "lean", str(FROZEN / "print_config_surface.lean")],
                          root, 600)
    diffs = []
    if srf_rc != 0 or surface != (FROZEN / "launch_surface.txt").read_text():
        diffs.append("launch_surface")
    if cfg_rc != 0 or cfg_out != (FROZEN / "config_surface.txt").read_text():
        diffs.append("config_surface")
    if sha(KDIR / "CONTRACT.md") != (FROZEN / "contract.sha256").read_text().strip():
        diffs.append("CONTRACT.md")
    steps[s.key] = s.done("failed" if diffs else "passed", str(FROZEN),
                          f"CONTRACT_CHANGED: {diffs} (requires recorded review + refreeze)"
                          if diffs else "surface identical to frozen snapshot")
    if diffs:
        (logs / "surface.current.txt").write_text(surface)

    # L5 adapter on manifests -------------------------------------------
    s = Step("source_link_adapter", "actual source + wrapper + Lean transcription + real CPU "
             "tensor metadata -> kernel-checked checker verdicts",
             "adapter recognition and torch metadata are trusted (tested, not proved)")
    results = {}
    manifests = sorted(REPO.glob("bench/tritonbench_g/*/launch_manifest.json"))
    all_ok = True
    for m in manifests:
        try:
            res = launch_check.analyze(m)
            res["status"] = "ok" if all(c["matches_expectation"] for c in res["cases"].values()) \
                else "mismatch"
        except launch_check.Unsupported as e:
            res = {"status": "unsupported", "unsupported": str(e)}
        except Exception as e:  # infrastructure
            res = {"status": "infrastructure_failure", "error": repr(e)}
        results[str(m.relative_to(REPO))] = res
        all_ok &= res["status"] == "ok"
    (EVID / "adapter_results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    n_cases = sum(len(r.get("cases", {})) for r in results.values())
    n_acc = sum(c["accepted"] for r in results.values() for c in r.get("cases", {}).values())
    steps[s.key] = s.done("passed" if all_ok and manifests else "failed",
                          str(EVID / "adapter_results.json"),
                          f"{len(manifests)} manifests, {n_cases} cases, {n_acc} accepted, "
                          f"all verdicts kernel-checked" if all_ok else
                          f"statuses={[r['status'] for r in results.values()]}")

    # L6 adapter unit tests -------------------------------------------------
    s = Step("adapter_tests", "scripts/test_launch_check.py (parser/metadata/Lean-verdict regressions)")
    rc, out = run([sys.executable, "-m", "unittest", "-q", "test_launch_check"],
                  REPO / "scripts", a.timeout, logs / "adapter_tests.log")
    steps[s.key] = s.done("passed" if rc == 0 else "failed", str(logs / "adapter_tests.log"),
                          out.strip().splitlines()[-1] if out.strip() else f"rc={rc}")

    local_ok = all(st.status == "passed" for st in steps.values())
    hashes = input_hashes()

    steps.update(external_steps(hashes))

    ledger = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                   text=True).stdout.strip(),
        "input_hashes": hashes,
        "summary": {
            "model_proof": {"status": steps["kernel_file_axioms"].status,
                            "scope": "add_kernel whole-grid launch in VeriTile's exact-ℝ region model",
                            "evidence": steps["kernel_file_axioms"].evidence},
            "checker_soundness": {"status": steps["kernel_file_axioms"].status,
                                  "scope": "check_ok / check_complete / i32 lemmas (Lean)",
                                  "evidence": steps["kernel_file_axioms"].evidence},
            "source_correspondence": {"status": steps["source_link_adapter"].status,
                                      "level": "tested-but-trusted (AST recognizer + statement "
                                               "match); not proved",
                                      "evidence": steps["source_link_adapter"].evidence},
            "local_trust_checks": {"status": "passed" if all(
                steps[k].status == "passed" for k in ("placeholder_scan", "kernel_file_axioms"))
                else "failed", "scope": "placeholder scan, #axiomsClean, headline inventory"},
            "frozen_contract": {"status": steps["frozen_contract"].status,
                                "evidence": steps["frozen_contract"].evidence},
            "official_comparator": steps["official_comparator"].json(),
            "gpu_correctness": steps["gpu_correctness"].json(),
            "gpu_performance": steps["gpu_performance"].json(),
        },
        "steps": {k: v.json() for k, v in steps.items()},
        "fresh_workspace": a.fresh,
    }
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    a.ledger.write_text(json.dumps(ledger, indent=1, ensure_ascii=False) + "\n")
    if a.fresh and root != REPO:
        shutil.rmtree(root, ignore_errors=True)

    rc = 0
    if local_ok:
        print("LOCAL_CHECKS_PASSED")
    else:
        print("LOCAL_CHECKS_FAILED")
        rc = 1
    if a.require_official and steps["official_comparator"].status != "passed":
        print(f"OFFICIAL_AUDIT_REQUIRED_BUT_{steps['official_comparator'].status.upper()}")
        rc = rc or 2
    if a.require_gpu and not all(steps[k].status == "passed"
                                 for k in ("gpu_correctness", "gpu_performance")):
        print("GPU_EVIDENCE_REQUIRED_BUT_MISSING_OR_FAILED")
        rc = rc or 3
    return rc


if __name__ == "__main__":
    sys.exit(main())
