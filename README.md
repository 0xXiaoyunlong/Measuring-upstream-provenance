# Measuring Upstream Provenance at Cryptocurrency Mixer Entrances

Data and code behind our paper, *Measuring Upstream Provenance at Cryptocurrency
Mixer Entrances: A Cross-Actor Forensic Study.*

A mixer cuts the money trail at the deposit. But the money that fed the deposit is
already sitting on-chain before it happens — who funded it, how many hops back, how
long it sat at each stop. The question: does that pre-deposit trail look similar
across laundering crews, enough to flag a crew we've never seen? This repo is
everything needed to reproduce the answer.

Short version of what we found:

- Across crews the signal is real but weak. The macro leave-one-crew-out AUROC is
  **0.702949, reported 0.703**, 95% CI **[0.545, 0.847]**, corrected-null
  permutation **p = 0.031** (the group-count harness null agrees at p = 0.027) —
  and it stops being significant (0.631, p = 0.146) the moment you add one odd
  cross-chain crew (the 18-crew sensitivity).
- A "factory-pattern" feature (funder fan-out, batches of fresh wallets) looks
  almost perfect within one crew. Across crews it adds nothing once we recover the
  outflow records it depends on: on its own it drops from 0.75 to 0.51, and the
  increment over provenance is **-0.006 with a CI [-0.069, 0.056] that straddles
  zero**. The high within-crew number was the model reading which records happened
  to be missing, not laundering structure. That result is really the point of the
  paper.
- The original background is 2022-only, while the illicit events run to 2026. So
  we re-ran the frozen pipeline on a **temporally balanced background** of 140
  deposits spread over 2022–2026. A signal that was really a timing artifact
  should shrink under that swap; this one got stronger, 0.802 [0.671, 0.910],
  p = 0.002, with 25 of the 30 battery variants significant versus 6 of 30 on
  the original background. The whole control lives in `balanced_background/`.
  The headline numbers stay the ones from the original, pre-registered
  background.

Bottom line, it's a triage cue at best. You can't attribute a deposit to a crew
with this, and we're careful in the paper not to claim otherwise.

### The reported number, and the reference environment

The macro AUROC is a report value, not a bit-for-bit constant. Sixteen of the
seventeen crews have a single illicit event, so their AUROC is the rank of that one
event, and a few are near ranking ties — the last reported decimal can move with
the BLAS/CPU. In the frozen **reference environment (Python 3.11.9,
scikit-learn 1.3.2, single-thread)** the canonical pipeline yields **0.702949,
reported 0.703**. We do **not** claim it reproduces bit-for-bit on any machine.
The existing row order of `data/upstream_features.csv` is part of the canonical
input; it is not re-sorted. (Historical note: an earlier internal run reported
0.6824 on the same data via a different execution path; that is not the value the
current public code produces.)

## Reproducing the results

Python 3.11.9 and four packages. No API key or network — it all reads from `data/`.
One command runs the whole chain:

```bash
pip install -r requirements-exact.txt      # the scikit-learn 1.3.2 reference environment
python run_all.py                          # ~10-15 min (the permutation is the slow part)
```

`run_all.py` checks the environment (including the Python 3.11.9 minor version) and
the frozen input, then runs, in order: `evaluate.py` (recomputes and overwrites
`per_crew_results.csv`, `per_event_predictions.csv`, `main_results.csv` — 17- and
18-crew), `permutation_test.py` (the group-count harness null — recomputes the
1000-shuffle null and the observed statistic, which equals the macro AUROC),
`permutation_sensitivity.py` (the corrected immutable-block null the paper quotes
for the main models + a block-size-matched conditional null, reported as-is),
`missingness_audit.py` (recomputes the sidecar from the underlying factory tables
in `data/sidecar/`). These derived results are staged and promoted transactionally;
then `robustness/run_robustness.py` (recomputes the robustness perturbations
from the frozen cones and renders the data tables + battery figures), the
**figure scripts** (`fig3_signal.py`, `fig4_audit.py`, `figC1_heatmap.py`, and
the two-background `fig5_forest.py`, `fig6_timeline.py`, `fig7_dualcontrol.py`),
a rebuild of `SHA256SUMS.txt`, and final consistency checks.
Every step uses single-thread BLAS and fails the whole run on any error; nothing
falls back to an old CSV or the network. (`--fast` runs a 100-shuffle permutation
for a smoke test — not a publishable run, and it never touches `data/`, `figures/`,
`robustness/`, or `SHA256SUMS.txt`.)

**Figures.** The statistical figures (Fig 3-7 and C1) are regenerated as PNGs
from the archived files by the scripts in `code/03_make_figures/`; the numbers
in them are read out of the CSVs / JSONs rather than hard-coded.
`python code/03_make_figures/figdata.py` starts with an oracle that asserts the
headline numbers of both backgrounds straight from the archived files, so one
command re-checks every headline number against what is on disk. The two
overview graphics have no scripts: Fig 1 is the dataset waterfall, hand-finished
from the counts in `data/dataset_funnel.csv`, and Fig 2 is the hand-polished
schematic of the backward funding cone with no experimental values in it.

**Sidecar.** The factory-pattern (missingness) audit is recomputed under 1.3.2 from
the underlying tables in `data/sidecar/`; the *before-recovery* factory features are
rebuildable byte-equal from the frozen funding cones by
`code/02_evaluate_cross_actor/sidecar/build_sidecar_features.py`. The *after-recovery*
table is shipped as a frozen artifact (the L22 out-edge recovery required fetching
from a public node; see `data/sidecar/README.md`). Figure 4 follows the
after-recovery result, and the provenance (Phi) baseline is the same 0.703 as the
main analysis, not the 0.6824 of the historical note above.

### The two nulls

Each background comes with two permutation nulls, and we fixed which one the
paper quotes where before computing any p-values. The **group-count harness
null** (`permutation_test.py`) holds the number of illicit actor-group blocks
fixed (17 of 156 singleton-background blocks) but lets the number of positive
events float; its p goes into `main_results.csv`, and it is the p carried by
every row of the 30-variant battery. The **corrected 154-block null**
(`permutation_sensitivity.py`) additionally de-duplicates the background by
deposit address (139 events → 137 blocks), so an actor who reused a deposit
address can never end up split across the labels. That is the better-specified
design, and it is what the paper quotes for the two main models: **0.031** on
the original background, **0.002** on the balanced one (the harness values are
0.027 / 0.003; the same address-dedup rule on the balanced data gives 157
blocks). A null that also fixed the positive-event count is not identifiable
here (Harmony is the unique 14-deposit block); the identifiable size-matched
*conditional* variant over the 16 single-deposit crews is reported as a further
sensitivity, p = 0.001 on both backgrounds. The nulls and their configs all sit
in `data/`, reported as-is.

**Robustness.** The one-command flow recomputes the perturbations from the
frozen funding cones in `robustness/inputs/` under scikit-learn 1.3.2
(`robustness/run_robustness.py`, baseline anchored to 0.703) and renders the data
tables plus the battery figures. Each forest row carries the per-variant AUROC,
95% CI, and the **canonical N=1000 group-aware permutation p** (the main row
equals the headline 0.027, not the 1.9.0-era value); those p-values are the
multi-hour `robustness/robustness_permutation.py` recompute, shipped frozen as
`robustness_forest_data.json`. The signal holds up under the value and window
choices; what moves it is cone depth and the source-context / observability
features (`robustness/README.md`). The smaller appendix numbers (the AUPRC row, the sign
test, the analytic PU correction, the paired reduced-fold baselines, the T10
fold-reassignment check) each have their own script and archived output in
`robustness/appendix_checks/`.

## Layout

```
├── run_all.py                                   one-command end-to-end reproduction
├── requirements-exact.txt / requirements.txt   the reference environment
├── SHA256SUMS.txt                               checksums (`sha256sum -c SHA256SUMS.txt`)
├── data/                                        169 deposits, the folds, and the derived result CSVs
│   └── sidecar/                                 underlying factory tables + frozen cones for the audit
├── code/
│   ├── 01_collect_and_build_features/           pulling the data off the chain
│   ├── 02_evaluate_cross_actor/                 evaluate / permutation_test / permutation_sensitivity / missingness_audit
│   │   └── sidecar/                             build_sidecar_features + evaluate_sidecar
│   └── 03_make_figures/                         the six figure scripts + figdata.py oracle
├── archive/frozen_extraction/                   the frozen phi.py/pipeline.py, exact bytes
├── figures/                                     fig1-fig7 + figC1 as published (+ static fig2.png)
├── robustness/                                  the 30-variant battery (1.3.2, anchored to 0.703)
│   └── appendix_checks/                         scripts + outputs behind the appendix numbers
└── balanced_background/                         the temporally balanced control (anchor 0.802), with its own README
```

Each `code/` folder has its own README; `robustness/` has `README.md`,
`run_robustness.py` (the recompute run by `run_all.py`), `build_robustness.py` (the
renderer), `inputs/` (frozen cones), and `appendix_checks/`.

Data availability: evidence-gate decisions in `data/evidence_gate_decisions.csv` /
`data/dataset_funnel.csv`; frozen extraction code in
`archive/frozen_extraction/`; fold assignments in the `held_out_fold` column of
`data/entrance_events.csv` (backgrounds by `sha256(address) % 17`, re-derived on
every run); per-cohort predictions in `data/per_event_predictions.csv` (169
full-precision logistic out-of-fold scores); the permutation config in
`data/permutation_config.json`. Every row of `data/entrance_events.csv` carries its
(address, time, tx hash) so any deposit can be re-derived from a public node. The
balanced-background control has the same set of files under
`balanced_background/data/`.

## The data

169 Tornado Cash entrance deposits on Ethereum. 30 are illicit, from 17 crews we
could attribute; the other 139 are background — ordinary depositors into the same
pools in the same window, **unlabeled, not known-clean**, which only makes the
numbers conservative. **A blank feature is "never observed", not a real 0**: it is
imputed with the training-fold median and flagged by its observed-indicator. Five
crews are public (Harmony, Beanstalk, Audius, Monkey, Convergence); the twelve new
ones are N01–N12; the 18-crew sensitivity adds N13 (the LFI / Q25 cross-chain
group, `data/lfi_q25_features.csv`). `data/README.md` has the column rundown. The
temporally balanced control replaces the 139 background deposits with 140 spread
over 2022–2026 and changes nothing else (see `balanced_background/README.md`).

The canonical evaluation covers the **logistic model only**; the gradient-boosting
rows were removed because no executable boosting pipeline ships with this package.

## How a deposit becomes 15 numbers

Walk backward from the deposit along whoever funded it, and describe the shape of
that trail: hop count, elapsed time, dwell, how much we could trace, how
concentrated the source was. The walk is deliberately narrow — transfers ≥ 10 ETH,
the 7 days before the deposit, at most 2 hops, biggest funder at each step — and
never touches identities or labels. The one subtlety worth reading the code for is
"censored" (trail hit a contract) versus "missing" (couldn't fetch the record) —
that distinction is the whole missingness-audit story, in
`code/01_collect_and_build_features/build_upstream_features.py`.

## Citation & license

Cite the paper if you use this (`CITATION.cff`; the DOI goes in on publication). The
**code** is licensed MIT (`LICENSE`); the **data and figures** are licensed CC BY 4.0
(`LICENSE-CC-BY-4.0.md`). The on-chain records themselves are public data and are not
covered by either license.
