"""
Significance of the cross-crew signal by a group-preserving permutation test
(Ojala & Garriga, 2010) -- the group-count (legacy) harness null, reported
alongside the corrected null (see the two-nulls section of the root README).

A held-out AUROC only means something if a random labelling would not score that
high as easily. This test shuffles which *whole crews* are illicit (a crew's
deposits move together, never one deposit at a time), re-runs the entire
leave-one-crew-out evaluation, and records the shuffled AUROC; 1000 shuffles give
a null distribution. The p-value is the share of shuffles reaching the observed
score: p = (1 + #{shuffled >= observed}) / (1 + N).

This recomputes both the observed statistic and the null from the same feature
table, sample order and model as evaluate.py -- the observed statistic equals the
macro AUROC that evaluate.py reports (about 0.703). It then overwrites
permutation_null.csv, permutation_summary.csv, permutation_legacy_summary.csv and
permutation_config.json and fills the permutation columns of main_results.csv. It
does not read any archived null.

    python permutation_test.py            # 1000 shuffles per crew set (slow)

Null definition -- legacy_group_count_preserving_null (state this in the paper).
The permutation blocks are the 17 illicit crews (each block holds all of that
crew's deposits) plus one singleton block per background deposit, i.e. 17 + 139 =
156 blocks. Each shuffle picks 17 of the 156 blocks to be "illicit". This keeps the
*number of positive blocks* fixed at 17 but not the number of positive *events*,
because the crews differ in size (Harmony has 14 deposits, the others one each): a
selected real crew contributes its 14 or 1 positives, a selected background block
contributes 1, so the null mixes over positive-event counts. It is a group-count-
preserving null (Ojala & Garriga) with a known repeat-address splitting artifact
(2 of the 156 blocks; see below) that the sensitivity corrects, and it is the null
behind the reported p.

This is called the *legacy* null because its background blocks are one-per-deposit:
two background addresses deposit twice, so a shuffle could put one of those two
deposits in the positive set and leave its twin in the background, splitting one
real address across the labels. The audit-requested sensitivity analysis in
`permutation_sensitivity.py` corrects that (de-duplicates background by address into
137 immutable blocks -> 154 total) and additionally reports a block-size-matched
conditional null; both are reported as-is and neither overwrites the files here.
A null that also fixes the positive-event count / block-size distribution is not
identifiable on this data (no background block has 14 deposits to match Harmony);
see `permutation_sensitivity.py`.
"""

import hashlib
import json
import os
import statistics

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import sys

import evaluate as ev

DATA = ev.DATA
SEED = 20260629
N_PERM = 100 if "--fast" in sys.argv else 1000   # --fast is a smoke test only, not a publishable run


def _bg_fold(address, k):
    return int(hashlib.sha256(address.lower().encode()).hexdigest(), 16) % k


def _fold_auroc(test_events, train_events, y_test, y_train):
    """Same pipeline as evaluate.crew_auroc, but with explicit (permuted) labels."""
    x_train = ev.build_design_matrix(train_events, train_events)
    x_test = ev.build_design_matrix(test_events, train_events)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(x_train), np.asarray(y_train))
    return roc_auc_score(np.asarray(y_test), model.predict_proba(scaler.transform(x_test))[:, 1])


def loago_by_group(events, is_pos, group_of, group_order, addrs):
    """Leave-one-actor-group-out built from group membership, so it also works for
    the permuted labels. Backgrounds are bucketed by sha256(address) mod K; group
    gi is tested against bucket gi. With the real labels and group_order sorted by
    the held-out fold this reproduces evaluate.py's macro AUROC exactly."""
    k = len(group_order)
    idx = list(range(len(events)))
    aurocs = []
    for gi, g in enumerate(group_order):
        test = [i for i in idx if (is_pos[i] and group_of[i] == g)
                or (not is_pos[i] and _bg_fold(addrs[i], k) == gi)]
        if not test:
            continue
        train = [i for i in idx if i not in test]
        yte = [is_pos[i] for i in test]
        ytr = [is_pos[i] for i in train]
        if sum(yte) in (0, len(yte)) or sum(ytr) == 0:
            continue
        aurocs.append(_fold_auroc([events[i] for i in test], [events[i] for i in train], yte, ytr))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def make_blocks(events):
    """One block per illicit crew, one singleton block per background deposit."""
    crews = sorted({e["crew"] for e in events if int(e["is_illicit"]) == 1})
    blocks = [[i for i, e in enumerate(events) if int(e["is_illicit"]) == 1 and e["crew"] == c]
              for c in crews]
    blocks += [[i] for i, e in enumerate(events) if int(e["is_illicit"]) == 0]
    return blocks, len(crews)


def run(events, fold_column, observed, addrs):
    # sanity: the by-group machinery on the real labels must match the observed macro
    crews = sorted({e["crew"] for e in events if int(e["is_illicit"]) == 1},
                   key=lambda c: min(int(e[fold_column]) for e in events if e["crew"] == c))
    is_pos = [int(e["is_illicit"]) for e in events]
    group_of = [e["crew"] if int(e["is_illicit"]) else "" for e in events]
    real = loago_by_group(events, is_pos, group_of, crews, addrs)
    assert abs(real - observed) < 1e-9, f"permutation observed {real} != evaluate macro {observed}"

    blocks, n_pos = make_blocks(events)
    rng = np.random.RandomState(SEED)
    null = []
    for _ in range(N_PERM):
        chosen = sorted(rng.choice(len(blocks), n_pos, replace=False).tolist())
        pos = [0] * len(events)
        grp = [""] * len(events)
        order = []
        for bi in chosen:
            for i in blocks[bi]:
                pos[i] = 1
                grp[i] = f"PG_{bi}"
            order.append(f"PG_{bi}")
        null.append(loago_by_group(events, pos, grp, order, addrs))
    null = [v for v in null if not np.isnan(v)]
    ge = sum(1 for v in null if v >= observed)
    p = (1 + ge) / (1 + len(null))
    return null, p


def main():
    print(f"Crew-preserving permutation test (Ojala & Garriga 2010), N={N_PERM}, seed={SEED}\n")
    addr = ev._address_map()

    events17 = ev._load_events("upstream_features.csv")
    obs17 = statistics.mean(ev.crew_auroc(te, tr)[0] for _f, te, tr in ev.leave_one_crew_out(events17))
    addrs17 = [addr[e["event_id"]] for e in events17]
    null17, p17 = run(events17, "held_out_fold", obs17, addrs17)
    print(f"17 crews: observed {obs17:.4f}  p = {p17:.5f}  null mean {statistics.mean(null17):.3f}")

    rows_null = [["17_crews", "logistic", repr(v)] for v in null17]
    summary = [["17_crews", "logistic", repr(obs17), len(null17), repr(p17),
                repr(statistics.mean(null17))]]

    events18 = ev.build_18crew_events()
    obs18 = p18 = None
    if events18 is not None:
        lfi_addr = ev._load_events("lfi_q25_features.csv")[0]["a0"]
        obs18 = statistics.mean(ev.crew_auroc(te, tr)[0]
                                for _f, te, tr in ev.leave_one_crew_out(events18, "held_out_fold_18"))
        addrs18 = [addr.get(e["event_id"], lfi_addr) for e in events18]
        null18, p18 = run(events18, "held_out_fold_18", obs18, addrs18)
        print(f"18 crews: observed {obs18:.4f}  p = {p18:.5f}  null mean {statistics.mean(null18):.3f}")
        rows_null += [["18_crews", "logistic", repr(v)] for v in null18]
        summary.append(["18_crews", "logistic", repr(obs18), len(null18), repr(p18),
                        repr(statistics.mean(null18))])

    ev._atomic_write("permutation_null.csv", ["crew_set", "model", "shuffled_auroc"], rows_null)
    ev._atomic_write("permutation_summary.csv",
                     ["crew_set", "model", "observed_auroc", "n_permutations", "p_value", "null_mean"],
                     summary)
    # An explicitly-labelled copy of the summary so downstream readers cannot mistake
    # this (the legacy group-count-preserving null) for the corrected sensitivity.
    ev._atomic_write("permutation_legacy_summary.csv",
                     ["null_name", "crew_set", "model", "observed_auroc", "n_permutations",
                      "p_value", "null_mean"],
                     [["legacy_group_count_preserving_null"] + r for r in summary])
    json.dump({"reference": "Ojala & Garriga (2010)", "seed": SEED, "n_permutations": N_PERM,
               "null_name": "legacy_group_count_preserving_null",
               "permutation_unit": "actor_group_block", "model": "logistic",
               "split": "LOAGO_leave_one_actor_group_out", "metric": "AUROC",
               "observed_equals_point_estimate": True,
               "empirical_p_formula": "p = (1 + #{permuted_AUROC >= observed}) / (1 + N)",
               "nan_handling": ("permutations with a NaN macro AUROC (a degenerate fold) are dropped "
                                   "from both numerator and denominator; the effective N is the "
                                   "n_permutations column of permutation_summary.csv and can fall just "
                                   "below the nominal 1000."),
               "null_definition": ("17 illicit-crew blocks + 139 background singleton blocks (156 total); "
                                   "each shuffle selects 17 blocks as illicit. Fixes the number of positive "
                                   "actor-groups (17), not the number of positive events (crews differ in size; "
                                   "Harmony has 14 deposits). Background blocks are one-per-deposit."),
               "role": ("PRIMARY null whose p-value is reported and written into main_results.csv."),
               "corrected_sensitivity": ("permutation_sensitivity.py de-duplicates background by address "
                                   "into 137 immutable blocks (154 total) and adds a block-size-matched "
                                   "conditional null; those p-values are reported as-is and never overwrite "
                                   "these files. A positive-event-count / block-size-matched null over the full "
                                   "set is NOT_IDENTIFIABLE_WITH_AVAILABLE_BACKGROUND_BLOCKS (no 14-deposit "
                                   "background block to match Harmony).")},
              open(os.path.join(ev.RESULTS, "permutation_config.json"), "w", encoding="utf-8"), indent=2)

    main_rows = ev._load_results("main_results.csv")
    fill = {"17_crews": (obs17, p17), "18_crews": (obs18, p18)}
    out = []
    for r in main_rows:
        obs, p = fill.get(r["crew_set"], (None, None))
        out.append([r["crew_set"], r["analysis"], r["model"], r["auroc"], r["ci_low"], r["ci_high"],
                    ("" if obs is None else repr(obs)), ("" if p is None else repr(p))])
    ev._atomic_write("main_results.csv",
                     ["crew_set", "analysis", "model", "auroc", "ci_low", "ci_high",
                      "permutation_observed", "p_value"], out)
    print("\nwrote data/permutation_null.csv, permutation_summary.csv, permutation_config.json; "
          "filled permutation columns of main_results.csv.")


if __name__ == "__main__":
    main()
