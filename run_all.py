"""
One command to reproduce the whole package, end to end, in the scikit-learn 1.3.2
reference environment:

    python run_all.py                        # full official run (~10-15 min)
    python run_all.py --fast                 # smoke test in a sandbox; data/ is not touched
    python run_all.py --allow-nonreference   # run even if package versions differ

Steps (official run)
--------------------
evaluate.py -> permutation_test.py (primary/legacy group-count-preserving null) ->
permutation_sensitivity.py (corrected immutable-block sensitivity, reported as-is) ->
missingness_audit.py (sidecar), all into a private staging dir; then, after the
0.703 gate + atomic promotion, robustness/run_robustness.py (recomputes the
robustness perturbations from the frozen cones and renders the forest) and the
figure scripts, a rebuild of SHA256SUMS.txt, and final consistency checks.

Notes
-----
* Python 3.11.9 + the four pinned packages are the reference environment; a mismatch
  (including the Python minor version) stops the official run unless you pass
  --allow-nonreference. No network, ever.
* The inputs are checked for presence and for the 169/30/139 event counts before
  anything runs. File checksums are in SHA256SUMS.txt (`sha256sum -c SHA256SUMS.txt`).
* The derived results are transactional: the compute steps (evaluate /
  permutation / sensitivity / sidecar) write to a private staging directory, the
  run is gated on the 0.703 macro AUROC, and data/ is overwritten only then, one
  atomic move at a time. If the permutation fails, data/ is left untouched.
* Robustness and the figures are deterministic, idempotent re-renders of the
  committed results / frozen inputs; a failure there leaves data/ correct and is
  fixed by re-running; the transactional boundary is exactly the promote step above.
* --fast runs a 100-shuffle permutation in the sandbox and never promotes it, so a
  smoke test can never leave non-publishable numbers in data/, figures/, robustness/,
  or SHA256SUMS.txt.
"""

import csv
import hashlib
import io
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FAST = "--fast" in sys.argv
ALLOW_NONREF = "--allow-nonreference" in sys.argv

ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONIOENCODING="utf-8")

REFERENCE = {"numpy": "1.26.4", "scipy": "1.17.1", "scikit-learn": "1.3.2", "matplotlib": "3.8.2"}

# The derived result files the compute steps write (staged, then promoted).
RESULT_FILES = ["main_results.csv", "per_crew_results.csv", "per_event_predictions.csv",
                "permutation_null.csv", "permutation_summary.csv", "permutation_legacy_summary.csv",
                "permutation_config.json",
                "permutation_sensitivity_null.csv", "permutation_sensitivity_summary.csv",
                "permutation_sensitivity_config.json",
                "missingness_audit_metrics.csv", "missingness_audit_increments.csv"]

# The frozen inputs every step reads.
FROZEN_INPUTS = [
    "data/upstream_features.csv",
    "data/entrance_events.csv",
    "data/dataset_funnel.csv",
    "data/evidence_gate_decisions.csv",
    "data/missingness_by_crew.csv",
    "data/outflow_recovery.csv",
    "data/lfi_q25_features.csv",
    "data/sidecar/sidecar_features_before_recovery.csv",
    "data/sidecar/sidecar_features_after_recovery.csv",
    "data/sidecar/recovered_out_edges.csv",
    "data/sidecar/outedge_recovery_status.csv",
    "data/sidecar/source_log.jsonl",
    "data/sidecar/_real_ledgers.json",
    "data/sidecar/_bean_ledgers.json",
    "data/sidecar/_audmonk_ledgers.json",
    "data/sidecar/A17_cone_ledger_v2.json",
    "data/sidecar/l17_cone_edges.csv",
    "data/sidecar/l20_step0_feature_table.csv",
    "archive/frozen_extraction/phi.py",
    "archive/frozen_extraction/pipeline.py",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(script, cwd, extra=(), env=None):
    args = [sys.executable, script] + list(extra)
    print(f"\n{'='*70}\n# {os.path.relpath(os.path.join(cwd, script), HERE)} {' '.join(extra)}\n{'='*70}")
    subprocess.run(args, cwd=cwd, env=(env or ENV), check=True)


def check_environment():
    print(f"{'='*70}\n# environment\n{'='*70}")
    import importlib.metadata as md
    mismatch = []
    # Python itself is part of the reference (3.11.9). pip cannot pin the interpreter,
    # so this is checked here and hard-fails the official run like any other mismatch.
    pyver = sys.version.split()[0]
    py_mm = sys.version_info[:2]
    if py_mm != (3, 11):
        mismatch.append(f"python {pyver} != 3.11.x")
        print(f"  {'python':<14} {pyver:<10} (reference 3.11.9)   <- not the reference minor version")
    elif pyver != "3.11.9":
        print(f"  {'python':<14} {pyver:<10} (reference 3.11.9)   note: 3.11.x patch differs (acceptable)")
    else:
        print(f"  {'python':<14} {pyver:<10} (reference 3.11.9)")
    for pkg, want in REFERENCE.items():
        try:
            got = md.version(pkg)
        except Exception:
            got = "MISSING"
        if got != want:
            mismatch.append(f"{pkg} {got} != {want}")
        print(f"  {pkg:<14} {got:<10} (reference {want}){'' if got == want else '   <- not the reference version'}")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        print(f"  {var}={ENV[var]}")
    if mismatch and not FAST and not ALLOW_NONREF:
        print("\nREFUSING TO RUN: not the reference environment (" + "; ".join(mismatch) + ").")
        print("The reported 0.703 is defined in the reference environment (Python 3.11.9 +")
        print("scikit-learn 1.3.2). Re-run in it, or pass --allow-nonreference to proceed anyway")
        print("(the full-precision value may differ).")
        sys.exit(1)
    if mismatch:
        print("  NOTE: not the reference environment; proceeding (--allow-nonreference / --fast).")


def check_inputs():
    print(f"\n{'='*70}\n# inputs\n{'='*70}")
    missing = [rel for rel in FROZEN_INPUTS if not os.path.exists(os.path.join(HERE, rel))]
    if missing:
        print("missing input files:")
        for rel in missing:
            print("  ", rel)
        sys.exit(1)
    # structural check on the main input
    events = list(csv.DictReader(io.open(os.path.join(DATA, "upstream_features.csv"), encoding="utf-8-sig")))
    n_illicit = sum(int(e["is_illicit"]) for e in events)
    assert len(events) == 169 and n_illicit == 30 and len(events) - n_illicit == 139, "169/30/139 broken"
    assert len({e["crew"] for e in events if int(e["is_illicit"])}) == 17, "expected 17 crews"
    print(f"  {len(FROZEN_INPUTS)} input files present")
    print(f"  upstream_features.csv: 169 events (30 illicit / 139 background / 17 crews)")


def rebuild_checksums():
    rows = []
    for root, _, files in os.walk(HERE):
        if "__pycache__" in root or os.sep + ".git" in root or os.sep + ".staging_results" in root:
            continue
        for fn in sorted(files):
            if fn == "SHA256SUMS.txt" or fn == ".forest_ckpt.json":
                continue
            p = os.path.join(root, fn)
            rows.append((sha256(p), os.path.relpath(p, HERE).replace(os.sep, "/")))
    rows.sort(key=lambda r: r[1])
    # LF endings, or `sha256sum -c` on Linux/macOS reads every name with a trailing \r
    with io.open(os.path.join(HERE, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(f"{h}  {r}" for h, r in rows) + "\n")
    print(f"  SHA256SUMS.txt: {len(rows)} files")


def staged_macro(staging):
    """The 17-crew AUROC as written to the staging dir (for the pre-promotion gate)."""
    with io.open(os.path.join(staging, "main_results.csv"), encoding="utf-8-sig") as fh:
        row = next(r for r in csv.DictReader(fh) if r["crew_set"] == "17_crews")
    return float(row["auroc"])


def final_checks():
    main = {r["crew_set"]: r for r in csv.DictReader(
        io.open(os.path.join(DATA, "main_results.csv"), encoding="utf-8-sig"))}
    a17 = float(main["17_crews"]["auroc"])
    assert round(a17, 3) == 0.703, f"17-crew AUROC {a17} does not round to 0.703"
    assert abs(float(main["17_crews"]["permutation_observed"]) - a17) < 1e-9, "observed != AUROC"
    assert "18_crews" in main, "18-crew sensitivity row missing"
    pc = list(csv.DictReader(io.open(os.path.join(DATA, "per_crew_results.csv"), encoding="utf-8-sig")))
    assert "auroc_boosting" not in pc[0], "auroc_boosting column still present"
    pep = list(csv.DictReader(io.open(os.path.join(DATA, "per_event_predictions.csv"), encoding="utf-8-sig")))
    assert len(pep) == 169 and {r["model"] for r in pep} == {"logistic"}, "per_event not 169 logistic rows"
    mm = list(csv.DictReader(io.open(os.path.join(DATA, "missingness_audit_metrics.csv"), encoding="utf-8-sig")))
    phi = [r for r in mm if r["feature_set"].startswith("provenance")][0]
    assert round(float(phi["auroc"]), 3) == 0.703, "sidecar Phi baseline is not 0.703"
    # corrected permutation SENSITIVITY: present, both variants, p reported AS-IS (never
    # gated on significance -- we only assert the file exists and carries both nulls).
    ps = list(csv.DictReader(io.open(os.path.join(DATA, "permutation_sensitivity_summary.csv"), encoding="utf-8-sig")))
    variants = {r["null_variant"] for r in ps}
    assert {"corrected_154block", "singleton_conditional_sizematched"} <= variants, "sensitivity variants missing"
    leg = list(csv.DictReader(io.open(os.path.join(DATA, "permutation_legacy_summary.csv"), encoding="utf-8-sig")))
    assert leg and leg[0]["null_name"] == "legacy_group_count_preserving_null", "legacy summary label missing"
    print(f"  17-crew AUROC {a17:.6f} -> 0.703, observed==AUROC, primary(legacy) p={main['17_crews']['p_value'][:8]}")
    print(f"  18-crew AUROC {float(main['18_crews']['auroc']):.6f}, primary(legacy) p={main['18_crews']['p_value'][:8]}")
    for r in ps:
        print(f"  sensitivity[{r['null_variant']}] observed {float(r['observed_auroc']):.4f} "
              f"p={r['p_value'][:8]} (reported as-is)")
    print(f"  per_crew: no boosting column; per_event: 169 logistic full-precision")
    print(f"  sidecar Phi baseline 0.703 (no splicing)")


def main():
    check_environment()
    check_inputs()

    ev_dir = os.path.join(HERE, "code", "02_evaluate_cross_actor")
    fig_dir = os.path.join(HERE, "code", "03_make_figures")
    perm_extra = ["--fast"] if FAST else []

    # ---- compute, transactionally, into a private staging dir ----
    staging = os.path.join(HERE, ".staging_results")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    stage_env = dict(ENV, CANON_RESULTS_DIR=staging)
    print(f"\n{'='*70}\n# compute (staging: .staging_results{' , --fast' if FAST else ''})\n{'='*70}")
    try:
        run("evaluate.py", ev_dir, env=stage_env)
        run("permutation_test.py", ev_dir, perm_extra, env=stage_env)          # primary (legacy) null
        run("permutation_sensitivity.py", ev_dir, perm_extra, env=stage_env)   # corrected sensitivity
        run("missingness_audit.py", ev_dir, env=stage_env)
        macro = staged_macro(staging)
        if not FAST:
            assert round(macro, 3) == 0.703, f"staged 17-crew AUROC {macro} != 0.703 -- not promoting"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if FAST:
        print(f"\n{'='*70}\n# SMOKE TEST\n{'='*70}")
        print(f"  staged 17-crew AUROC {macro:.6f} (100-shuffle permutation p is not publishable)")
        shutil.rmtree(staging, ignore_errors=True)
        print("  --fast complete. data/, figures/ and SHA256SUMS.txt were NOT modified.")
        return

    # ---- promote staging -> data/ atomically, only now that everything passed ----
    # Everything after this promote is a re-render of promoted results / frozen
    # inputs, so a failure below leaves data/ correct and re-running fixes it.
    print(f"\n{'='*70}\n# promote results to data/\n{'='*70}")
    for f in RESULT_FILES:
        os.replace(os.path.join(staging, f), os.path.join(DATA, f))
    shutil.rmtree(staging, ignore_errors=True)
    print(f"  promoted {len(RESULT_FILES)} result files")

    try:
        run("run_robustness.py", os.path.join(HERE, "robustness"))     # recompute from cones + render
        for fig in ("fig3_signal.py", "fig4_audit.py", "figC1_heatmap.py",
                    "fig5_forest.py", "fig6_timeline.py", "fig7_dualcontrol.py"):
            run(fig, fig_dir)
    except BaseException:
        print("\nRESULTS were promoted to data/ and are correct, but robustness/figure rendering")
        print("failed. Re-run `python run_all.py` to finish -- rendering is idempotent and reads")
        print("only the committed results/inputs; nothing is left partial in data/.")
        raise

    print(f"\n{'='*70}\n# checksums + final checks\n{'='*70}")
    assert os.path.exists(os.path.join(HERE, "figures", "fig1_dataset.png")), "figures/fig1_dataset.png missing"
    assert os.path.exists(os.path.join(HERE, "figures", "fig2.png")), "figures/fig2.png missing"
    rebuild_checksums()
    final_checks()
    print(f"\n{'='*70}\nDONE. Headline: data/main_results.csv (17-crew AUROC 0.703). "
          f"Figures in figures/.\n{'='*70}")


if __name__ == "__main__":
    main()
