# -*- coding: utf-8 -*-
"""
The sidecar evaluation: leave-one-crew-out logistic AUROC for three feature
blocks -- provenance (Phi) only, factory-pattern only, and both -- reusing the
main evaluation pipeline (same folds, same preprocessing, same model) from
evaluate.py. missingness_audit.py drives this over the before/after factory
tables in data/sidecar/.
"""
import csv
import io
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # code/02_evaluate_cross_actor
import evaluate as ev

SIDECAR_DIR = os.path.join(ev.DATA, "sidecar")
FACTORY_FEATURES = ["fac_funder_present", "fac_hub_present", "fac_funder_fanout",
                    "fac_hub_fanout", "fac_funder_out_unif", "fac_funder_out_quant",
                    "fac_fund_quant", "fac_n_fresh_siblings"]


def _num(v):
    if v in ("", None):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def load_factory_table(name, addr_to_event):
    """Read a data/sidecar/ factory table (keyed by address) and re-key it to the
    upstream event_id via the address map (the sidecar tables and upstream_features
    use different event_id strings for the newly gated crews)."""
    out = {}
    with io.open(os.path.join(SIDECAR_DIR, name), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            eid = addr_to_event.get(r["address"].lower())
            if eid is not None:
                out[eid] = {f: r.get(f, "") for f in FACTORY_FEATURES}
    return out


def block_matrix(events, train, block, factory_of):
    """PHI uses evaluate's 13 numerics + boundary one-hot; FACTORY uses the eight
    fac_* features (median-imputed + observed-indicator)."""
    parts = []
    if block in ("phi", "phi+factory"):
        parts.append(ev.build_design_matrix(events, train))
    if block in ("factory", "phi+factory"):
        med = {}
        for f in FACTORY_FEATURES:
            seen = [_num(factory_of.get(e["event_id"], {}).get(f)) for e in train]
            seen = [v for v in seen if not np.isnan(v)]
            med[f] = float(np.median(seen)) if seen else 0.0
        rows = []
        for e in events:
            row = []
            for f in FACTORY_FEATURES:
                v = _num(factory_of.get(e["event_id"], {}).get(f))
                row += ([med[f], 0.0] if np.isnan(v) else [v, 1.0])
            rows.append(row)
        parts.append(np.array(rows, dtype=float))
    return np.hstack(parts)


def per_crew_auroc(events, block, factory_of):
    """One AUROC per held-out crew, in fold order."""
    out = {}
    for fold, test, train in ev.leave_one_crew_out(events):
        y_train = np.array([int(e["is_illicit"]) for e in train])
        y_test = np.array([int(e["is_illicit"]) for e in test])
        x_train = block_matrix(train, train, block, factory_of)
        x_test = block_matrix(test, train, block, factory_of)
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(x_train), y_train)
        out[fold] = roc_auc_score(y_test, model.predict_proba(scaler.transform(x_test))[:, 1])
    return out
