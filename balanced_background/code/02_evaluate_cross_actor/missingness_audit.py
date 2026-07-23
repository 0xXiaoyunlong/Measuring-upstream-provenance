"""
The missingness audit (sidecar).

The "factory-pattern" feature (funder fan-out, batches of fresh wallets) is nearly
perfect within a single crew, and adding it to the cross-crew model first looked
like a large improvement. The feature depends on where a source address sent its
money, and those outflow records were missing from public nodes for the twelve
newly gated crews but present for the older crews and the background. So "is the
outflow record missing?" tracked the label. After re-fetching the real outflow
records where we could and re-running the same evaluation, the gain disappeared.

This recomputes the sidecar from the underlying factory tables in `data/sidecar/`
under the reference environment (scikit-learn 1.3.2), so the provenance (Phi)
baseline here is the same 0.703 the main analysis reports. Three feature blocks --
provenance (Phi), factory-pattern, and both -- at two stages: before recovery (the
factory features as first extracted, with the new crews' out-edges missing) and
after recovery (out-edges re-fetched where possible). It overwrites
data/missingness_audit_metrics.csv and data/missingness_audit_increments.csv.

    python missingness_audit.py

The factory features themselves are rebuildable from the frozen cones by
sidecar/build_sidecar_features.py; sidecar/evaluate_sidecar.py holds the block
evaluation. Figure 4's conclusion follows the after-recovery result.
"""

import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sidecar"))
import evaluate as ev
import evaluate_sidecar as es


def main():
    events = ev._load_events("upstream_features.csv")
    addr_of = ev._address_map()
    addr_to_event = {a.lower(): e for e, a in addr_of.items()}
    before = es.load_factory_table("sidecar_features_before_recovery.csv", addr_to_event)
    after = es.load_factory_table("sidecar_features_after_recovery.csv", addr_to_event)

    LABEL = {"phi": "provenance (Phi)", "factory": "factory-pattern",
             "phi+factory": "provenance + factory"}
    metrics_rows, per_fold = [], {}
    for stage, factory_of in (("before_recovery", before), ("after_recovery", after)):
        for block in ("phi", "factory", "phi+factory"):
            pc = es.per_crew_auroc(events, block, factory_of)
            per_fold[(stage, block)] = pc
            vals = list(pc.values())
            macro = statistics.mean(vals)
            lo, hi = ev.group_bootstrap_ci(vals)
            metrics_rows.append([stage, LABEL[block], repr(macro), repr(lo), repr(hi)])

    # sanity gate: the provenance-only sidecar baseline must equal this run's main
    # AUROC (read from the same run's main_results.csv, staged or promoted) -- never a
    # hard-coded constant. This works in any environment; if the underlying sidecar
    # tables cannot be joined to the 169 events / folds, the Phi-only block will not
    # match the main AUROC and the assert fails instead of passing silently.
    phi_macro = statistics.mean(per_fold[("before_recovery", "phi")].values())
    _main = ev._load_results("main_results.csv")
    main_auroc = float(next(r for r in _main if r["crew_set"] == "17_crews")["auroc"])
    assert abs(phi_macro - main_auroc) < 5e-4, (
        f"provenance-only sidecar baseline {phi_macro} != this run's main AUROC {main_auroc} -- "
        "the sidecar tables do not connect to the 169 events/folds of this run; STOP")

    # paired per-fold increments over provenance, after recovery
    inc_rows = []
    base = per_fold[("after_recovery", "phi")]
    for block, name in (("factory", "factory-pattern minus provenance"),
                        ("phi+factory", "(provenance + factory) minus provenance")):
        other = per_fold[("after_recovery", block)]
        common = sorted(set(base) & set(other))
        deltas = [other[f] - base[f] for f in common]
        md = statistics.mean(deltas)
        lo, hi = ev.group_bootstrap_ci(deltas)
        inc_rows.append([name, repr(md), repr(lo), repr(hi)])

    ev._atomic_write("missingness_audit_metrics.csv",
                     ["stage", "feature_set", "auroc", "ci_low", "ci_high"], metrics_rows)
    ev._atomic_write("missingness_audit_increments.csv",
                     ["comparison", "auroc_change", "ci_low", "ci_high"], inc_rows)

    print("Sidecar recomputed under scikit-learn 1.3.2 from data/sidecar/ (provenance baseline = the main 0.703):")
    for r in metrics_rows:
        print(f"  {r[0]:<16} {r[1]:<22} AUROC {float(r[2]):.4f}")
    print("Increments after recovery:")
    for r in inc_rows:
        print(f"  {r[0]:<42} {float(r[1]):+.4f}  [{float(r[2]):+.4f}, {float(r[3]):+.4f}]")
    print("wrote data/missingness_audit_metrics.csv, data/missingness_audit_increments.csv")


if __name__ == "__main__":
    main()
