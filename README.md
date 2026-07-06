# Measuring Upstream Provenance at Cryptocurrency Mixer Entrances

Data and code behind our paper, *Measuring Upstream Provenance at Cryptocurrency
Mixer Entrances: A Cross-Actor Forensic Study.*

A mixer cuts the money trail at the deposit. But the money that fed the deposit is
already sitting on-chain before it happens — who funded it, how many hops back, how
long it sat at each stop. The question we set out to answer was simple to state and
annoying to test: does that pre-deposit trail look similar across different
laundering crews, enough to flag a crew we've never seen? This repo is everything
needed to reproduce the answer.

Short version of what we found:

- Across crews the signal is real but weak. Average AUROC 0.68, permutation
  p = 0.044 — and it stops being significant (p = 0.166) the moment you add one
  odd crew to the mix. So: detectable, not robust.
- A "factory-pattern" feature (funder fan-out, batches of fresh wallets) looks
  almost perfect within one crew, ~0.99. Across crews it adds basically nothing
  once we go back and recover the outflow records it depends on: +0.004, and the
  confidence interval straddles zero. On its own it actually drops from 0.77 to
  0.52. The "0.99" was the model reading which records happened to be missing, not
  laundering structure. That result is really the point of the paper.

Bottom line, it's a triage cue at best. You can't attribute a deposit to a crew
with this, and we're careful in the paper not to claim otherwise.

## Layout

```
Measuring upstream provenance/
├── requirements.txt          numpy, scikit-learn, matplotlib
├── SHA256SUMS.txt            fingerprints of everything in data/ and archive/
├── data/                     169 deposits + the archived results (see data/README.md)
├── code/
│   ├── 01_collect_and_build_features/   pulling the data off the chain
│   ├── 02_evaluate_cross_actor/         the analysis
│   └── 03_make_figures/                 the figures
├── archive/frozen_extraction/           the frozen phi.py/pipeline.py, exact bytes
└── figures/
```

Each `code/` folder has its own README.

The paper's Data availability statement promises five things; here is where each
one lives, so nobody has to hunt:

- evidence-gate decisions → `data/evidence_gate_decisions.csv` (per candidate) and
  `data/dataset_funnel.csv` (the counts)
- frozen extraction code with recorded SHA-256 fingerprints → `archive/frozen_extraction/`
- fold assignments → the `held_out_fold` column in `data/entrance_events.csv`.
  Background deposits go to folds by `sha256(address) % 17` (lowercase address
  string); `evaluate.py` re-derives all 139 from that rule on every run, so the
  column isn't taken on faith
- per-cohort predictions → `data/per_event_predictions.csv`
- bootstrap and permutation configurations → `data/permutation_config.json`, and
  the bootstrap note at the end of `data/README.md`

And the sixth, implicit one: every row of `data/entrance_events.csv` now carries
its (address, time, tx hash), so any deposit can be re-derived from a public
Ethereum node without trusting us.

## Reproducing the results

Python 3.8+ and three packages. Nothing here needs an API key or a network
connection — it all reads from `data/`.

```bash
pip install -r requirements.txt

cd code/02_evaluate_cross_actor
python evaluate.py            # main result: average cross-crew AUROC ~0.68
python permutation_test.py    # significance against a null: observed 0.687, p = 0.044
python missingness_audit.py   # factory-pattern adds +0.004 across crews (CI spans 0)

cd ../03_make_figures
python figure1_dataset_funnel.py
python figure2_provenance_cone.py
python figure3_cross_actor_signal.py
python figure4_missingness_audit.py
python figureC1_missingness_heatmap.py
```

`evaluate.py` re-runs the whole leave-one-crew-out evaluation from the feature
table and prints it alongside the archived numbers. The figure scripts pull their
numbers out of the CSVs rather than having them hard-coded, so re-run them if you
touch the data. Want to rebuild the features off the chain instead of using the
shipped ones? That's `code/01_collect_and_build_features`.

### One caveat about exact numbers

The permutation test, the audit, and the figures reproduce exactly. The learned
model is 15/17 exact and two crews wobble, which needs a word of explanation.

Run `evaluate.py` and you'll get 15 of the 17 crews matching the archived AUROC to
the last digit. Two of them (N09, N12) come out a bit different — they flip by one
rank on a newer scikit-learn — and that nudges the average from the archived 0.6824
up to ~0.698. Both are the same weak ~0.68 signal and both sit well inside the CI
[0.5311, 0.8179], so nothing changes, but I know a mismatch looks alarming in a
repro package so here's why: those two crews have a single event each, and a
single-event AUROC is decided by one rank — flip it and you get 1.0 instead of
0.83. It's fragile by construction, which is exactly the Figure 3B story. The
archived per-crew values in `data/per_crew_results.csv` are the reference, and
`requirements-exact.txt` pins the sklearn that hits them exactly if you want that.

## The data

169 Tornado Cash entrance deposits on Ethereum. 30 of them are illicit, from 17
laundering crews we could attribute (each one either has an official attribution or
two independent forensic write-ups pointing at the same address — the four evidence
gates in `data/dataset_funnel.csv` show how 86 candidates got whittled down to 12
new crews). The other 139 are background: ordinary people depositing into the same
pools in the same window. Note they're unlabeled, not known-clean — some could be undetected bad
actors, which only makes our numbers conservative.

Five of the crews are ones everyone already knows, so we use their real names
(Harmony, Beanstalk, Audius, Monkey, Convergence). The twelve new ones are N01–N12,
same as the paper. Every row carries the deposit address, time, and (for 145 of
169) the transaction hash — all public on-chain facts — so you can go re-fetch the
raw records yourself and check the features against them. `data/README.md` has the
full column-by-column rundown.

## How a deposit becomes 15 numbers

Walk backward from the deposit along whoever funded it, and describe the shape of
that trail: hop count, how long it took, dwell time, how much of the deposit we
could actually trace, how concentrated the source was, and so on. The walk is kept
deliberately narrow and never touches identities or labels — only transfers ≥ 10
ETH, only the 7 days before the deposit, at most 2 hops, following the biggest
funder at each step. The one subtlety worth reading the code for is how it treats
"censored" (trail hit a contract, real source cut off) versus "missing" (couldn't
fetch the record) — that distinction is the whole missingness-audit story, and it
lives in `code/01_collect_and_build_features/build_upstream_features.py`.

## Citation & license

Cite the paper if you use this. `CITATION.cff`, the license, and the DOI go in on
publication (they're TODO placeholders right now — we're still in review). The
on-chain records themselves are public data.
