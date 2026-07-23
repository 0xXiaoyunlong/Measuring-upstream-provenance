"""
Leave-one-crew-out evaluation of the upstream-provenance features.

Measures whether the pre-deposit money trail lets the model rank a crew it has
never seen above the background depositors. One whole crew is held out at a time
(all its deposits together), the model trains on the rest, and the held-out crew
is scored against the background deposits in that fold. Each crew yields one
AUROC; the reported figure is the plain average over the 17 crews, so the largest
crew cannot dominate it.

Running it recomputes the logistic results from the feature table under the
reference environment (Python 3.11.9, scikit-learn 1.3.2) and overwrites the
derived result files it produced -- per_crew_results.csv, per_event_predictions.csv
and the logistic rows of main_results.csv -- so those files always match this run:

    python evaluate.py

Reference value: the macro leave-one-crew-out AUROC is about 0.702949, reported as
0.703. A few single-event crews are near ranking ties, so on a different BLAS/CPU
the full-precision value can move slightly; there is no bit-for-bit claim.
"""

import csv
import hashlib
import os
import statistics

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")               # frozen inputs (read-only)
# Derived result files are written to RESULTS. run_all.py points this at a staging
# directory so the whole run is transactional -- data/ is only overwritten after
# every step has succeeded (and never at all under --fast). Defaults to data/.
RESULTS = os.environ.get("CANON_RESULTS_DIR", DATA)

# The thirteen numeric provenance features the model actually learns from.
# (boundary_type is folded in separately as one-hot flags; censoring_reason is
#  kept in the data for transparency but is not fed to the model.)
NUMERIC_FEATURES = [
    "observed_hops", "path_elapsed_time", "dwell_mean_h", "dwell_max_h",
    "coverage", "attribution_ambiguous", "node_context_out_degree",
    "node_context_in_degree", "wallet_age_min_h",
    "node_context_resid_Rrange_max", "node_context_resid_nin_mean",
    "node_context_resid_nout_mean", "observation_confidence",
]

# How a backward walk can end. Each event gets a one-hot flag for its ending.
BOUNDARY_KINDS = ["eoa-source", "asset-conversion", "contract", "depth-limit",
                  "coverage-stop", "unresolved", "origin-reached"]


def to_number(raw, feature_name):
    """Turn a raw cell into a float, or NaN when the value was never observed."""
    if raw in ("", None):
        return np.nan
    if feature_name == "attribution_ambiguous":       # stored as True / False
        return 1.0 if str(raw) in ("True", "1", "1.0") else 0.0
    try:
        return float(raw)
    except ValueError:
        return np.nan


def build_design_matrix(events, training_events):
    """Turn a list of events into the numeric matrix the model sees.

    For every numeric feature we (a) fill a missing value with the median of the
    training events (not a zero -- see build_upstream_features.py) and (b) add a
    companion flag for whether the value was observed. The ending of the backward
    walk is added as one-hot flags. Fitting the median on the training events only
    keeps the held-out crew from leaking into its own test.
    """
    train_median = {}
    for feature in NUMERIC_FEATURES:
        seen = [to_number(e[feature], feature) for e in training_events]
        seen = [v for v in seen if not np.isnan(v)]
        train_median[feature] = float(np.median(seen)) if seen else 0.0

    matrix = []
    for event in events:
        row = []
        for feature in NUMERIC_FEATURES:
            value = to_number(event[feature], feature)
            if np.isnan(value):
                row.append(train_median[feature])   # imputed value
                row.append(0.0)                      # ...and flagged as unobserved
            else:
                row.append(value)
                row.append(1.0)
        ending = event["boundary_type"]
        row.extend(1.0 if ending == kind else 0.0 for kind in BOUNDARY_KINDS)
        matrix.append(row)
    return np.array(matrix, dtype=float)


def leave_one_crew_out(events, fold_column="held_out_fold"):
    """Yield (held-out fold id, test events, training events) for each crew."""
    fold_ids = sorted({int(e[fold_column]) for e in events})
    for fold in fold_ids:
        test = [e for e in events if int(e[fold_column]) == fold]
        train = [e for e in events if int(e[fold_column]) != fold]
        yield fold, test, train


def check_background_fold_rule():
    """Background deposits were assigned to folds by hashing the address:
    sha256 of the lowercase 0x-prefixed address string, as an integer, mod 17.
    The folds ship precomputed in the data; this re-derives them from the rule
    so nobody has to take the column on faith."""
    with open(os.path.join(DATA, "entrance_events.csv"), encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["crew"] == "background"]
    bad = [r["event_id"] for r in rows
           if int(hashlib.sha256(r["deposit_address"].lower().encode()).hexdigest(), 16) % 17
           != int(r["held_out_fold"])]
    assert not bad, f"held_out_fold disagrees with the hash rule for: {bad}"
    print(f"background fold assignment: {len(rows)}/139 match sha256(address) % 17")


def crew_auroc(test_events, train_events):
    """Train logistic regression on the other crews, score the held-out crew.
    Returns the AUROC, the AUPRC, and the full-precision test scores (in test order)."""
    y_train = np.array([int(e["is_illicit"]) for e in train_events])
    y_test = np.array([int(e["is_illicit"]) for e in test_events])

    x_train = build_design_matrix(train_events, train_events)
    x_test = build_design_matrix(test_events, train_events)

    scaler = StandardScaler().fit(x_train)
    # A plain, untuned logistic regression. We deliberately do not tune it:
    # with only 17 crews, any tuning would leak the test set into the model.
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(x_train), y_train)
    scores = model.predict_proba(scaler.transform(x_test))[:, 1]

    return roc_auc_score(y_test, scores), average_precision_score(y_test, scores), scores


def group_bootstrap_ci(per_crew_values, n_resamples=2000, seed=0):
    """95% interval by resampling whole crews (crews are the independent unit)."""
    rng = np.random.RandomState(seed)
    means = [float(np.mean(rng.choice(per_crew_values, len(per_crew_values), replace=True)))
             for _ in range(n_resamples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --- running the evaluation and writing the derived result files ---
def _load_events(name):
    with open(os.path.join(DATA, name), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _load_results(name):
    """Read a derived result file from RESULTS (the staging dir during a run,
    otherwise data/). Falls back to data/ if it is not yet in the staging dir."""
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        path = os.path.join(DATA, name)
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _address_map():
    return {r["event_id"]: r["deposit_address"]
            for r in _load_events("entrance_events.csv")}


def build_18crew_events():
    """The 18-crew sensitivity set: the 169 events plus the one cross-chain crew
    (LFI / Q25, from data/lfi_q25_features.csv). The 17 original crews keep their
    fold; the added crew is fold 17; the backgrounds are re-bucketed into 18 folds
    by sha256(address) mod 18. Returns None if the LFI input is not present."""
    lfi_path = os.path.join(DATA, "lfi_q25_features.csv")
    if not os.path.exists(lfi_path):
        return None
    lfi = _load_events("lfi_q25_features.csv")[0]
    addr = _address_map()
    events = []
    for e in _load_events("upstream_features.csv"):
        row = dict(e)
        if int(e["is_illicit"]) == 1:
            row["held_out_fold_18"] = e["held_out_fold"]
        else:
            row["held_out_fold_18"] = str(int(hashlib.sha256(
                addr[e["event_id"]].lower().encode()).hexdigest(), 16) % 18)
        events.append(row)
    lfi_event = {"event_id": "NEW-013", "crew": "N13", "is_illicit": "1",
                 "held_out_fold": "17", "held_out_fold_18": "17"}
    for f in NUMERIC_FEATURES + ["boundary_type", "censoring_reason", "observation_confidence"]:
        lfi_event[f] = lfi.get(f, "")
    events.append(lfi_event)
    return events


def _atomic_write(name, header, rows):
    """Write to a temp file in RESULTS, then replace, so a crash never leaves half
    a file (and, under a staging RESULTS dir, data/ is untouched until promotion)."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def run_and_write():
    check_background_fold_rule()
    events = _load_events("upstream_features.csv")
    n_illicit = sum(int(e["is_illicit"]) for e in events)
    crews = sorted({e["crew"] for e in events if int(e["is_illicit"])})
    print(f"Loaded {len(events)} events: {n_illicit} illicit across {len(crews)} crews, "
          f"{len(events) - n_illicit} background.")

    crew_type = {r["crew"]: r["crew_type"] for r in _load_events("per_crew_results.csv")}
    fold_to_crew = {int(e["held_out_fold"]): e["crew"] for e in events if int(e["is_illicit"])}

    # --- 17-crew primary: per-crew AUROC + full-precision out-of-fold scores ---
    per_crew_rows, prediction_rows, aurocs = [], [], []
    for fold, test, train in leave_one_crew_out(events):
        auroc, _auprc, scores = crew_auroc(test, train)
        aurocs.append(auroc)
        crew = fold_to_crew.get(fold, f"fold {fold}")
        n_pos = sum(int(e["is_illicit"]) for e in test)
        per_crew_rows.append([crew, crew_type.get(crew, ""), n_pos, len(test) - n_pos, repr(auroc)])
        order = np.argsort(np.argsort(scores)) / max(1, len(scores) - 1)
        for e, s, rk in zip(test, scores, order):
            prediction_rows.append([e["event_id"], e["crew"], e["is_illicit"],
                                    e["held_out_fold"], "logistic", repr(float(s)), round(float(rk), 4)])

    macro = statistics.mean(aurocs)
    lo, hi = group_bootstrap_ci(aurocs)

    # --- 18-crew sensitivity, computed from the same protocol (if LFI present) ---
    events18 = build_18crew_events()
    if events18 is not None:
        a18 = [crew_auroc(te, tr)[0] for _f, te, tr in leave_one_crew_out(events18, "held_out_fold_18")]
        macro18 = statistics.mean(a18)
        lo18, hi18 = group_bootstrap_ci(a18)
    else:
        macro18 = lo18 = hi18 = None

    # --- overwrite the derived result files (logistic only; no boosting here) ---
    _atomic_write("per_crew_results.csv",
                  ["crew", "crew_type", "n_events", "n_background_in_cohort", "auroc_logistic"],
                  per_crew_rows)
    _atomic_write("per_event_predictions.csv",
                  ["event_id", "crew", "is_illicit", "held_out_fold", "model", "score", "rank_percentile"],
                  prediction_rows)

    main_rows = [["17_crews", "primary", "logistic", repr(macro), repr(lo), repr(hi), "", ""]]
    if macro18 is not None:
        main_rows.append(["18_crews", "sensitivity", "logistic", repr(macro18), repr(lo18), repr(hi18), "", ""])
    _atomic_write("main_results.csv",
                  ["crew_set", "analysis", "model", "auroc", "ci_low", "ci_high",
                   "permutation_observed", "p_value"],
                  main_rows)

    print(f"\n17-crew macro AUROC = {macro:.6f}   (report {macro:.3f})   95% CI [{lo:.4f}, {hi:.4f}]")
    if macro18 is not None:
        print(f"18-crew macro AUROC = {macro18:.6f}   (report {macro18:.3f})   95% CI [{lo18:.4f}, {hi18:.4f}]")
    print("wrote data/per_crew_results.csv (logistic, boosting removed), "
          "data/per_event_predictions.csv (169 full-precision OOF scores), "
          "data/main_results.csv (permutation columns filled by permutation_test.py).")


def main():
    run_and_write()


if __name__ == "__main__":
    main()
