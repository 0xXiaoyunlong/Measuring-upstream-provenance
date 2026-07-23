# -*- coding: utf-8 -*-
"""Spot checks for a few numbers the paper quotes outside the main
evaluation: the AUPRC row of Table 2a (ledger T8), the sign test on the
per-cohort AUROCs (T6), Box 1's analytic positive-unlabeled correction (T7),
and the Appendix E question of how often the 25-node extraction cap actually
binds.

Everything is computed from the archived artifacts in data/, never from a
fresh fit. Refitting would pull sklearn's solver back into the picture, and
the single-event cohorts are known to flip there; the archived out-of-fold
scores give the same numbers on any environment.

    python ledger_checks.py
"""
import csv
import io
import json
import math
import os

import numpy as np
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))        # robustness/appendix_checks
PKG = os.path.dirname(os.path.dirname(HERE))             # package root
DATA = os.path.join(PKG, "data")
INPUTS = os.path.join(os.path.dirname(HERE), "inputs")


def rc(p):
    return list(csv.DictReader(io.open(p, encoding="utf-8-sig")))


out = {}

# The AUPRC row of Table 2a. Scores are pooled per held-out fold and the fold
# AUPRCs macro-averaged, same unit as the AUROC everywhere else in the paper.
pred = rc(os.path.join(DATA, "per_event_predictions.csv"))
folds = {}
for r in pred:
    if r["model"] == "logistic":
        folds.setdefault(int(r["held_out_fold"]), []).append(
            (int(r["is_illicit"]), float(r["score"])))
aps, prevs = [], []
for k in sorted(folds):
    y = [a for a, _ in folds[k]]
    s = [b for _, b in folds[k]]
    aps.append(average_precision_score(y, s))
    prevs.append(sum(y) / len(y))
aps = np.array(aps)
rng = np.random.RandomState(0)
means = [np.mean(rng.choice(aps, size=len(aps), replace=True)) for _ in range(2000)]
out["auprc"] = {"macro": float(aps.mean()),
                "ci_lo": float(np.percentile(means, 2.5)),
                "ci_hi": float(np.percentile(means, 97.5)),
                "macro_fold_prevalence": float(np.mean(prevs)),
                "global_prevalence": 30 / 169}
print(f"AUPRC macro {aps.mean():.4f}  CI [{np.percentile(means, 2.5):.3f}, "
      f"{np.percentile(means, 97.5):.3f}]  macro prevalence {np.mean(prevs):.4f}  "
      f"global 30/169 = {30/169:.4f}")

# T6 is a sign test because the single-event cohorts polarise to 0 or 1; only
# which side of the null each cohort lands on carries information here.
crew = rc(os.path.join(DATA, "per_crew_results.csv"))
vals = [float(r["auroc_logistic"]) for r in crew]


def sign_test(vs, null):
    above = sum(1 for v in vs if v > null)
    below = sum(1 for v in vs if v < null)
    n, k = above + below, max(above, below)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    return {"above": above, "below": below, "ties": len(vs) - n, "p": p}


out["sign_test"] = {"vs_0.53_empirical_null": sign_test(vals, 0.53),
                    "vs_0.5": sign_test(vals, 0.5)}
s53 = out["sign_test"]["vs_0.53_empirical_null"]
print(f"sign test vs 0.53: {s53['above']} up / {s53['below']} down  p = {s53['p']:.4f}")

# Box 1's positive-unlabeled correction is closed form, so no model comes into
# it: assume a share alpha of the background is actually illicit and unwind the
# dilution of the observed macro AUROC.
obs = sum(vals) / len(vals)
out["pu_correction"] = {"observed_macro": obs,
                        "corrected": {str(a): (obs - 0.5 * a) / (1 - a)
                                      for a in (0.05, 0.10, 0.20)}}
print("PU corrected:", "  ".join(f"alpha={a:.0%} -> {(obs-0.5*a)/(1-a):.3f}"
                                 for a in (0.05, 0.10, 0.20)))

# Appendix E asks how often the 25-node extraction cap and the coverage stop
# actually bind, counted straight off the archived cones and feature table.
cones = json.load(io.open(os.path.join(INPUTS, "tier3_assembled_cones.json"), encoding="utf-8"))
cap25 = sum(1 for v in cones.values() if len(v.get("cone", {})) >= 25)
uf = rc(os.path.join(DATA, "upstream_features.csv"))
covstop = sum(1 for r in uf if r.get("boundary_type") == "coverage-stop"
              or r.get("censoring_reason") == "coverage-stop")   # column name varies by vintage
out["binding"] = {"cone_cap_25_hit": cap25, "of_cones": len(cones),
                  "coverage_stop_events": covstop, "of_events": len(uf)}
print(f"cone cap 25 hit by {cap25}/{len(cones)} cones; "
      f"coverage stop ends {covstop}/{len(uf)} events")

json.dump(out, io.open(os.path.join(HERE, "ledger_checks.json"), "w", encoding="utf-8"),
          indent=2)
print("saved -> ledger_checks.json")
