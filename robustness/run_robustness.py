# -*- coding: utf-8 -*-
"""
Robustness -- the official recompute from underlying inputs (in-package, relative paths).

This recomputes every robustness perturbation from the frozen cones and feature
tables under this package, then renders the tables and the forest figure. It is the
script run_all.py calls in the official flow; build_robustness.py is only the
renderer (JSON -> tables/figure).

What is recomputed from the frozen cones (inputs/tier3_assembled_cones.json + phi.py):
  - baseline (7 d / 2 hop / 10 ETH, 17 crews)
  - leave-one-crew-out (17 crews dropped one at a time)
  - Phi-extraction parameters: Window 3 d / 5 d, Depth 1 hop, Value floor 15/20 ETH,
    Share floor 0.2 V
  - feature-family ablations: - path timing, Geometry only, - source context,
    - all observability
Group-bootstrap 95% CIs are recomputed too, all of it from the cones.

Two widen perturbations (Window 14 d, Depth 3 hops) need edges beyond the frozen
2-hop / 7-day cone that had to be fetched from a public node; their pre-extracted
features are frozen in inputs/tier3_widen_features.json (+ inputs/tier3_band_scan.json
for the 14-day active set). Those two are recomputed from the frozen feature tables
and are labelled FROZEN_INTERMEDIATE. If a required input is missing the affected
entries are marked FROZEN_INTERMEDIATE_ONLY and taken from the shipped JSON.

Reference environment (scikit-learn 1.3.2): the recomputed baseline is 0.7029
(reported 0.703) and this script asserts that 3-decimal anchor. On another
scikit-learn the value may drift; the script still runs and prints a notice.

    python run_robustness.py            # recompute + render
"""
import io
import json
import os
import statistics
import sys

import numpy as np
import sklearn

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "inputs")
sys.path.insert(0, os.path.join(HERE, "appendix_checks"))
import tier3_lib as T                       # relative-path cone library (vendored in appendix_checks/)
import build_robustness as BR               # renderer

VER = sklearn.__version__
NUM = T.ev.NUMERIC_FEATURES
BK = T.ev.BOUNDARY_KINDS
CONES = os.path.join(INPUTS, "tier3_assembled_cones.json")
WIDEN = os.path.join(INPUTS, "tier3_widen_features.json")
BAND = os.path.join(INPUTS, "tier3_band_scan.json")

# feature families (same partition as the paper)
TIMING = ["path_elapsed_time", "dwell_mean_h", "dwell_max_h", "wallet_age_min_h"]
SRC = ["node_context_out_degree", "node_context_in_degree", "node_context_resid_Rrange_max",
       "node_context_resid_nin_mean", "node_context_resid_nout_mean"]
GEOM = ["observed_hops", "coverage", "attribution_ambiguous"]
OC = ["observation_confidence"]


def _to_num(raw, f):
    return T.ev.to_number(raw, f)


def _design(rows, train, keep, flags, bd):
    med = {}
    for f in keep:
        s = [_to_num(e[f], f) for e in train]
        s = [v for v in s if not np.isnan(v)]
        med[f] = float(np.median(s)) if s else 0.0
    M = []
    for e in rows:
        row = []
        for f in keep:
            v = _to_num(e[f], f)
            if np.isnan(v):
                row.append(med[f])
                if flags: row.append(0.0)
            else:
                row.append(v)
                if flags: row.append(1.0)
        if bd:
            row.extend(1.0 if e["boundary_type"] == k else 0.0 for k in BK)
        M.append(row)
    return np.array(M, dtype=float)


def _auroc_cfg(te, tr, keep, flags, bd):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    yq = np.array([int(e["is_illicit"]) for e in tr])
    yt = np.array([int(e["is_illicit"]) for e in te])
    Xtr = _design(tr, tr, keep, flags, bd)
    Xte = _design(te, tr, keep, flags, bd)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), yq)
    return roc_auc_score(yt, m.predict_proba(sc.transform(Xte))[:, 1])


def _perfold(rows, keep=NUM, flags=True, bd=True):
    d = {}
    for f, te, tr in T.ev.leave_one_crew_out(rows):
        if len(set(int(x["is_illicit"]) for x in te)) < 2:
            continue
        d[f] = _auroc_cfg(te, tr, keep, flags, bd)
    return d


def _AC(d):
    v = list(d.values())
    lo, hi = T.ev.group_bootstrap_ci(v)
    return [round(statistics.mean(v), 4), round(lo, 4), round(hi, 4), len(v)]


def _frozen_row(eid):
    er = T.EBID[eid]; fr = T.uf[eid]
    row = {"crew": er["crew"], "is_illicit": er["is_illicit"], "held_out_fold": er["held_out_fold"]}
    for ft in T.FEATS: row[ft] = fr.get(ft, "")
    return row


def _feat_row(eid, fdict):
    er = T.EBID[eid]
    row = {"crew": er["crew"], "is_illicit": er["is_illicit"], "held_out_fold": er["held_out_fold"]}
    for ft in T.FEATS: row[ft] = fdict.get(ft, "")
    return row


def recompute():
    missing = [p for p in (CONES,) if not os.path.exists(p)]
    if missing:
        print(f"FROZEN_INTERMEDIATE_ONLY: {missing} absent -- cannot recompute from cones; "
              f"falling back to the shipped JSON via build_robustness.render().")
        return None

    out = {"sklearn": VER}
    BASE = T.build_rows(7, 2, 10, 0.1)[0]
    FOLD2CREW = {int(r["held_out_fold"]): r["crew"] for r in T.ent if r["is_illicit"] == "1"}

    base = _perfold(BASE)
    out["baseline"] = _AC(base)

    # leave-one-crew-out
    loo = {}
    for f in sorted(base):
        crew = FOLD2CREW.get(f, str(f))
        kept = [r for r in BASE if r["crew"] != crew]
        loo["− " + crew] = _AC(_perfold(kept))
    out["loo"] = loo

    # Phi-extraction parameters (recomputed from cones)
    def pr(W, dp, vt, mvs):
        return _AC(_perfold(T.build_rows(W, dp, vt, mvs)[0]))
    out["params"] = {"Window 3d": pr(3, 2, 10, 0.1), "Window 5d": pr(5, 2, 10, 0.1),
                     "Depth 1 hop": pr(7, 1, 10, 0.1), "Value floor 15 ETH": pr(7, 2, 15, 0.1),
                     "Value floor 20 ETH": pr(7, 2, 20, 0.1), "Share floor 0.2V": pr(7, 2, 10, 0.2)}

    # widen (14 d / 3 hop): from frozen pre-extracted features (network was needed)
    ship = json.load(io.open(os.path.join(HERE, "robustness_results_sklearn_1.3.2.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(HERE, "robustness_results_sklearn_1.3.2.json")) else {}
    if os.path.exists(WIDEN) and os.path.exists(BAND):
        band = json.load(io.open(BAND, encoding="utf-8"))
        active14 = set(e for e, v in band.items() if v["band_edges"] > 0)
        depthA = set(r["event_id"] for r in T.uf.values()
                     if r.get("censoring_reason", "") == "depth-limit" or r.get("boundary_type", "") == "depth-limit")
        wf = json.load(io.open(WIDEN, encoding="utf-8"))

        def widen_rows(featdict, active):
            rows = []
            for r in T.ent:
                eid = r["event_id"]
                if eid in featdict:
                    rows.append(_feat_row(eid, featdict[eid]))
                elif eid not in active:
                    rows.append(_frozen_row(eid))
            return rows
        out["Window 14d"] = _AC(_perfold(widen_rows(wf["14d"], active14)))
        out["Depth 3 hops"] = _AC(_perfold(widen_rows(wf["3hop"], depthA)))
        out["_widen_provenance"] = "FROZEN_INTERMEDIATE (recomputed from inputs/tier3_widen_features.json)"
    else:
        out["Window 14d"] = ship.get("Window 14d", [None, None, None, 0])
        out["Depth 3 hops"] = ship.get("Depth 3 hops", [None, None, None, 0])
        out["_widen_provenance"] = "FROZEN_INTERMEDIATE_ONLY (widen inputs absent; taken from shipped JSON)"

    # feature-family ablations (recomputed from cones)
    out["feats"] = {
        "− path timing": _AC(_perfold(BASE, [f for f in NUM if f not in TIMING])),
        "Geometry only": _AC(_perfold(BASE, GEOM, flags=False, bd=False)),
        "− source context": _AC(_perfold(BASE, [f for f in NUM if f not in SRC])),
        "− all observability": _AC(_perfold(BASE, [f for f in NUM if f not in OC], flags=False, bd=True)),
    }
    return out


def main():
    print(f"Robustness recompute from frozen cones -- scikit-learn {VER}\n")
    out = recompute()
    if out is None:
        BR.render(BR.SRC)
        return

    base_auroc = out["baseline"][0]
    is_reference = VER == "1.3.2"
    if is_reference:
        assert round(base_auroc, 3) == 0.703, (
            f"recomputed robustness baseline {base_auroc} != canonical 0.703 in the reference "
            f"environment (scikit-learn 1.3.2); STOP")
        print(f"baseline recomputed = {base_auroc:.4f}  (anchors to canonical 0.703)")
    else:
        print(f"baseline recomputed = {base_auroc:.4f}  (scikit-learn {VER}, not the 1.3.2 reference; "
              f"the 3-decimal value may drift)")

    dst = os.path.join(HERE, f"robustness_results_sklearn_{VER}.json")
    json.dump(out, io.open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {os.path.basename(dst)} (recomputed)")

    # render the data tables (CSVs) from the freshly recomputed JSON
    BR.render(dst)

    # render the robustness figures: the forest-and-table set + the combined overview.
    # These read robustness_forest_data.json -- the frozen canonical per-variant
    # permutation recompute produced by robustness_permutation.py (N=1000, main p =
    # 0.026973). Regenerating that JSON is a multi-hour job, so the one-command flow
    # renders from the shipped JSON; run robustness_permutation.py to recompute it.
    forest_json = os.path.join(HERE, "robustness_forest_data.json")
    if os.path.exists(forest_json):
        sys.path.insert(0, HERE)   # the package figure renderers must win any name clash
        import build_forest_tables as FT
        import build_robustness_coef as CF
        FT.main()
        CF.main()
    else:
        print("FROZEN_INTERMEDIATE_ONLY: robustness_forest_data.json absent -- skipping the "
              "forest-and-table figures; run robustness_permutation.py to produce it (N=1000, "
              "the canonical per-variant permutation p-values).")
    print("robustness recompute complete.")


if __name__ == "__main__":
    main()
