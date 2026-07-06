# 2. Evaluating the cross-crew signal

This is the analysis. Every script here reads only from `../../data/` — **no API
key, no internet, no re-crawling.** Together they reproduce the three headline
results of the paper.

```bash
pip install -r ../../requirements.txt
python evaluate.py
python permutation_test.py
python missingness_audit.py
```

| file | question it answers | headline output |
|---|---|---|
| `evaluate.py` | Can we rank an unseen crew ahead of the background using only pre-deposit provenance? | average cross-crew AUROC **≈ 0.68** |
| `permutation_test.py` | significance against a crew-preserving null | observed 0.687, p = 0.044 (17 crews); p = 0.166 (18 crews) |
| `missingness_audit.py` | marginal AUROC of factory-pattern after outflow recovery | +0.004, confidence interval spanning zero |

## How the evaluation works (evaluate.py)

We never let the model see the crew it is being tested on. We hold out one whole
crew at a time — all of its deposits move together — train an untuned logistic
regression on the remaining crews, and score the held-out crew against the
background deposits assigned to that fold. Each crew yields one AUROC; we report
the plain average over the 17 crews, so the one big crew (Harmony, 14 of the 30
deposits) cannot dominate the number.

Two design choices affect the numbers:

- **We do not tune the model.** With only 17 crews, any tuning would leak the test
  set into the model, so both baselines (logistic regression, gradient boosting)
  are fixed.
- **Missing values are flagged, not zeroed.** Each feature enters the model with a
  companion "was this observed?" flag, and unseen values are imputed with the
  training median, not a zero (`missingness_audit.py` shows why that matters).

`evaluate.py` prints its re-run next to the archived per-crew values and explains
the two single-event crews that can flip on a newer scikit-learn (see the top-level
README for the full note).

## The permutation test (permutation_test.py)

A held-out AUROC of 0.68 only means something if a random labelling would not score
that high just as easily. We shuffle which *whole crews* are illicit (never one
deposit at a time — that would break the within-crew dependence and manufacture
significance), re-run the entire evaluation, and repeat 1000 times to get a null
distribution. The p-value is the share of shuffles that did at least as well.
Because each shuffle re-runs the whole evaluation, the 1000 shuffled scores are
shipped in `../../data/permutation_null.csv` rather than regenerated here.

## The missingness audit (missingness_audit.py)

Inside a single crew, a "factory-pattern" feature (funder fan-out, batches of fresh
wallets) looks almost perfect. Added to the cross-crew model it first looked like a
big improvement — but that feature depends on outflow records that were missing for
the twelve newly gated crews and present for the older crews and the background. So
"is the record missing?" lined up with the label. After we re-fetched the real
records where we could and re-ran the *exact same* evaluation, the gain vanished
(+0.004, interval spanning zero). Only the data changed between the two runs — same
folds, same model. `missingness_audit.py` prints the before/after comparison from
the archived metrics.
