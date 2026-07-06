"""
The missingness audit.

The "factory-pattern" feature (funder fan-out, batches of fresh wallets) is nearly
perfect within a single crew, and adding it to the cross-crew model first looked
like a large improvement. The feature depends on where a source address sent its
money, and those outflow records were missing from public nodes for the twelve
newly gated crews but present for the older crews and the background. So "is the
outflow record missing?" was correlated with the label, and the model was picking
up that missingness rather than laundering structure.

To check this we re-fetched the real pre-deposit outflow records where we could
(see ../01_collect_and_build_features and ../../data/outflow_recovery.csv), left
the rest marked missing, and re-ran the same evaluation. The gain disappeared.

This script prints the before/after comparison from the archived metrics.

    python missingness_audit.py
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")


def load_csv(name):
    with open(os.path.join(DATA, name), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main():
    metrics = load_csv("missingness_audit_metrics.csv")
    increments = load_csv("missingness_audit_increments.csv")
    recovery = load_csv("outflow_recovery.csv")

    # --- how much of the missing outflow data we could recover ---
    from collections import Counter
    tally = Counter(r["recovery_status"] for r in recovery)
    print("Outflow records for the 12 newly gated crews:")
    for status, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>2}  {status}")
    print()

    # --- AUROC before vs after recovering the missing data ---
    def auroc(stage, feature_set):
        for r in metrics:
            if r["stage"] == stage and r["feature_set"] == feature_set:
                return r["auroc"], r["ci_low"], r["ci_high"]
        return ("-", "", "")

    print(f"{'feature set':<24}{'before recovery':<20}{'after recovery'}")
    for fs in ["provenance (Phi)", "factory-pattern", "provenance + factory"]:
        b, _, _ = auroc("before_recovery", fs)
        a, _, _ = auroc("after_recovery", fs)
        print(f"  {fs:<22}{b:<20}{a}")
    print()

    # --- marginal AUROC of factory-pattern over provenance, after recovery ---
    print("Marginal AUROC of factory-pattern over provenance, after recovery:")
    for r in increments:
        change = float(r["auroc_change"])
        spans_zero = float(r["ci_low"]) <= 0 <= float(r["ci_high"])
        note = "interval spans zero -> no robust gain" if spans_zero else "interval excludes zero"
        print(f"  {r['comparison']:<42} {change:+.4f}  "
              f"[{r['ci_low']}, {r['ci_high']}]  ({note})")

    print("\nThe combined model's advantage collapses once the missing outflow records are")
    print("recovered: the increment is +0.004 with a confidence interval spanning zero.")
    print("Only the data changed between the two runs (same folds, same model), so the")
    print("earlier gain came from the missingness pattern, not from laundering structure.")


if __name__ == "__main__":
    main()
