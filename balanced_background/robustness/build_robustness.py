"""
Turns a robustness_results_sklearn_<ver>.json into the result tables
(robustness_results.csv, robustness_deltas.csv). The robustness figures are the
forest-and-table set
(build_forest_tables.py) plus the combined overview (build_robustness_coef.py),
rendered separately from robustness_forest_data.json.

The JSON itself is produced by run_robustness.py, which recomputes the perturbations
(Phi extraction parameters, feature-family ablations, leave-one-crew-out) offline
from the frozen cones in inputs/tier3_assembled_cones.json under the scikit-learn
1.3.2 reference environment (baseline 0.7029, CI [0.5446, 0.8469], identical to the
main analysis). run_all.py calls run_robustness.py, which renders via this module.

Run this module directly only to re-render the tables/figure from an existing JSON
without recomputing:

    python build_robustness.py
"""

import csv
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "robustness_results_sklearn_1.3.2.json")
FIGDIR = os.path.join(HERE, "figures")


def render(src=SRC):
    os.makedirs(FIGDIR, exist_ok=True)
    fig = json.load(io.open(src, encoding="utf-8"))
    base = fig["baseline"][0]

    rows = [("baseline", "Canonical (7d / 2hop / 10 ETH, 17 crews)", *fig["baseline"])]
    for name, q in fig["params"].items():
        rows.append(("phi_parameter", name, *q))
    rows.append(("phi_parameter", "Window 14 d", *fig["Window 14d"]))
    rows.append(("phi_parameter", "Depth 3 hops", *fig["Depth 3 hops"]))
    for name, q in fig["feats"].items():
        q = list(q) + [17][: max(0, 4 - len(q))]           # feats store [auroc,lo,hi]
        rows.append(("feature_ablation", name, q[0], q[1], q[2], 17))
    for name, q in fig["loo"].items():
        rows.append(("leave_one_crew_out", name.replace("− ", "drop "), *q))

    with io.open(os.path.join(HERE, "robustness_results.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "variant", "auroc", "ci_low", "ci_high", "n_estimable_folds", "delta_vs_canonical"])
        for fam, name, a, lo, hi, n in rows:
            w.writerow([fam, name, a, lo, hi, n, round(a - base, 4)])

    with io.open(os.path.join(HERE, "robustness_deltas.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "variant", "auroc", "delta_vs_canonical", "canonical_baseline"])
        for fam, name, a, lo, hi, n in rows:
            if fam == "baseline":
                continue
            w.writerow([fam, name, a, round(a - base, 4), round(base, 4)])

    # The robustness figures proper are the forest-and-table set
    # (build_forest_tables.py) plus the combined overview (build_robustness_coef.py),
    # rendered from robustness_forest_data.json. This renderer produces only the data
    # tables; it no longer draws the old single-panel robustness_forest figure.
    npert = sum(1 for r in rows if r[0] in ("phi_parameter", "feature_ablation"))
    print(f"robustness: {npert} perturbations vs canonical baseline {base:.4f}")
    print(f"  wrote robustness_results.csv, robustness_deltas.csv")


def main():
    render(SRC)


if __name__ == "__main__":
    main()
