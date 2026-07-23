# -*- coding: utf-8 -*-
"""Fold-reassignment check for the background deposits (ledger T10), plus a
tie-aware reading of the headline AUROC.

The tie-aware read asks whether 0.70 survives an explicit near-tie rule:
score pairs closer than 1e-7 count as half a win, the Mann-Whitney statistic
is recomputed per fold, and the folds are averaged as usual.

The reassignment part is the actual T10 check. The unique background
addresses are re-bucketed into the 17 folds uniformly at random -- all events
of an address move together, and the illicit folds stay where they are --
then the full leave-one-crew-out evaluation is re-run. Over 100 seeds this
gives a distribution of macro AUROCs, and the script reports its range plus
where the prespecified sha256 assignment lands in it.

    python fold_reassignment.py

Unlike the ledger_checks script this one refits the model, so the exact
decimals depend on the sklearn build (see NUMERICAL_SENSITIVITY in the root
README). The archived fold_reassignment_output.txt is from the run cited in
the paper.
"""
import collections
import os
import random
import statistics
import sys

# the loaders and the model live in the main evaluation script; reuse them
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PKG, "code", "02_evaluate_cross_actor"))
import evaluate as ev  # noqa: E402

events = ev._load_events("upstream_features.csv")
addr = ev._address_map()
EPS = 1e-7


def tie_auroc(y, s):
    pos = [v for v, yy in zip(s, y) if yy == 1]
    neg = [v for v, yy in zip(s, y) if yy == 0]
    tot = 0.0
    for a in pos:
        for b in neg:
            d = a - b
            tot += 0.5 if abs(d) < EPS else (1.0 if d > 0 else 0.0)
    return tot / (len(pos) * len(neg))


strict, tiea = [], []
for fold, test, train in ev.leave_one_crew_out(events):
    au, _ap, scores = ev.crew_auroc(test, train)
    y = [int(e["is_illicit"]) for e in test]
    strict.append(au)
    tiea.append(tie_auroc(y, list(scores)))
print(f"(a) macro AUROC strict {statistics.mean(strict):.4f}   "
      f"tie-aware(eps=1e-7) {statistics.mean(tiea):.4f}")

ill = [e for e in events if int(e["is_illicit"]) == 1]
bg = [e for e in events if int(e["is_illicit"]) == 0]
by_addr = collections.defaultdict(list)
for e in bg:
    by_addr[addr[e["event_id"]].lower()].append(e)
addrs = sorted(by_addr)
print(f"    background: {len(bg)} events, {len(addrs)} unique addresses")

vals = []
for s in range(100):
    rng = random.Random(1000 + s)
    # one draw per address; its events all follow it, illicit folds untouched
    fold_of_addr = {a: rng.randrange(17) for a in addrs}
    evs = ill + [dict(e, held_out_fold=str(fold_of_addr[addr[e['event_id']].lower()]))
                 for e in bg]
    m = statistics.mean(ev.crew_auroc(te, tr)[0] for _f, te, tr in ev.leave_one_crew_out(evs))
    vals.append(m)
    if (s + 1) % 25 == 0:
        print(f"    addr-block reassignment {s+1}/100", flush=True)
obs = statistics.mean(strict)
pct = sum(1 for v in vals if v <= obs) / len(vals) * 100   # share of random buckets below ours
print(f"(b) address-block reassignment: range [{min(vals):.3f}, {max(vals):.3f}]  "
      f"median {statistics.median(vals):.3f}  prespecified {obs:.3f} at {pct:.0f}th pct")
