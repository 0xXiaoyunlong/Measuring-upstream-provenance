# 2. Evaluating the cross-crew signal

*The write-up below is copied unchanged from the root package, so its worked
examples still quote the original-background numbers (0.703, p = 0.027, 156
blocks). Run the same scripts on the data in this folder and you get 0.802 /
p = 0.003 (corrected 0.002); see `../../README.md`.*

This is the analysis. Every script here reads only from `../../data/` — **no API
key, no internet, no re-crawling.** Together they reproduce the headline results of
the paper. Each script recomputes and overwrites the derived result CSVs it owns
(under `run_all.py` these writes go to a staging dir and are promoted only on success).

```bash
pip install -r ../../requirements-exact.txt
python evaluate.py                 # AUROC, per-crew, 18-crew sensitivity
python permutation_test.py         # group-count-preserving (legacy) harness null
python permutation_sensitivity.py  # corrected immutable-block null
python missingness_audit.py        # sidecar / factory-pattern audit
```

| file | question it answers | headline output |
|---|---|---|
| `evaluate.py` | Can we rank an unseen crew ahead of the background using only pre-deposit provenance? | macro leave-one-crew-out AUROC **0.702949, reported 0.703** (17 crews); 18-crew sensitivity **0.631** |
| `permutation_test.py` | significance against the group-count-preserving harness null | observed **0.702949**, legacy **p = 0.027** (17 crews); **p = 0.146** (18 crews) |
| `permutation_sensitivity.py` | corrected immutable-block null + a block-size-matched conditional null | recomputed p-values in `permutation_sensitivity_summary.csv`, reported **as-is** |
| `missingness_audit.py` | marginal AUROC of factory-pattern after outflow recovery | **-0.006**, 95% CI **[-0.069, 0.056]** spanning zero |

One note on the two p columns above: `main_results.csv` and the battery rows
carry the group-count p, while the paper quotes the corrected null for the two
main models (the two-nulls section of the root README explains the split).

## How the evaluation works (evaluate.py)

We never let the model see the crew it is being tested on. We hold out one whole
crew at a time — all of its deposits move together — train an untuned logistic
regression on the remaining crews, and score the held-out crew against the
background deposits assigned to that fold. Each crew yields one AUROC; we report
the plain average over the 17 crews, so the one big crew (Harmony, 14 of the 30
deposits) cannot dominate the number.

Two design choices affect the numbers:

- **We do not tune the model.** With only 17 crews, any tuning would leak the test
  set into the model. The canonical evaluation covers the **logistic model only**;
  the earlier gradient-boosting rows were removed because no executable boosting
  pipeline ships with this package.
- **Missing values are flagged, not zeroed.** Each feature enters the model with a
  companion "was this observed?" flag, and unseen values are imputed with the
  training median, not a zero (`missingness_audit.py` shows why that matters).

`evaluate.py` recomputes the 17- and 18-crew results and the 169 full-precision
out-of-fold scores. A few single-event crews are near ranking ties, so the last
reported decimal can move on a newer scikit-learn/BLAS (see the top-level README).

## The permutation tests (permutation_test.py + permutation_sensitivity.py)

A held-out AUROC only means something if a random labelling would not score that
high just as easily. We shuffle which *whole crews* are illicit (never one deposit
at a time — that would break the within-crew dependence and manufacture
significance), re-run the entire evaluation, and repeat 1000 times to get a null
distribution. The p-value is the share of shuffles that did at least as well:
p = (1 + #{shuffled ≥ observed}) / (1 + N).

`permutation_test.py` **recomputes** both the observed statistic and the whole
1000-shuffle null on every run (it does *not* read an archived null) and writes
`permutation_null.csv`, `permutation_summary.csv`, `permutation_legacy_summary.csv`
and `permutation_config.json`, and fills the permutation columns of
`main_results.csv`. This is the **legacy group-count-preserving harness null**
(17 illicit-crew blocks + 139 background singleton blocks = 156; each shuffle picks
17). It fixes the number of positive *blocks*, not positive *events*, and — because
background blocks are one-per-deposit — two repeat-deposit addresses could in
principle be split across the labels.

`permutation_sensitivity.py` is the audit-requested correction. It de-duplicates the
background by deposit address into 137 immutable blocks (154 total), so a real
address can no longer be split, and it additionally reports a block-size-matched
**conditional** null over the 16 single-deposit crews. An unconditional null that
also fixes the positive-event count / block-size distribution is not identifiable
here (Harmony is the unique 14-deposit block); this is documented in
`permutation_sensitivity_config.json`. Both sensitivity p-values are reported as-is;
the two-nulls section of the root README states which null is primary and why.

## The missingness audit (missingness_audit.py)

Inside a single crew, a "factory-pattern" feature (funder fan-out, batches of fresh
wallets) looks almost perfect. Added to the cross-crew model it first looked like a
big improvement — but that feature depends on outflow records that were missing for
the twelve newly gated crews and present for the older crews and the background. So
"is the record missing?" lined up with the label. After we re-fetched the real
records where we could and re-ran the *exact same* evaluation, the gain vanished:
on its own the factory block drops from 0.75 to 0.51, and the increment over
provenance is **−0.006, 95% CI [−0.069, 0.056]**. Only the data changed between the
two runs — same folds, same model. `missingness_audit.py` recomputes the sidecar
from the underlying tables in `../../data/sidecar/` and checks its provenance-only
baseline equals *this run's* main AUROC (0.703), never a hard-coded constant.
