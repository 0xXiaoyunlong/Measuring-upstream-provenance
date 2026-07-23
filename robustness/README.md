# Robustness

All figures here are recomputed under the reference environment (scikit-learn
1.3.2) and anchored to this package's canonical 17-crew baseline of **0.702949
(reported 0.703)**, the same number the main analysis
(`../code/02_evaluate_cross_actor/evaluate.py`) reports.

## How to regenerate

```bash
python run_robustness.py            # RECOMPUTE from the frozen cones + render tables/forest
# or, to only re-render from an existing JSON without recomputing:
python build_robustness.py
```

`run_robustness.py` (the script `run_all.py` calls) **recomputes** the perturbations
— baseline, leave-one-crew-out, Φ-extraction parameters, feature-family ablations,
and their group-bootstrap CIs — offline from the frozen extraction cones in
`inputs/tier3_assembled_cones.json` under scikit-learn 1.3.2, asserts the recomputed
baseline anchors to **0.7029 (reported 0.703)**, writes
`robustness_results_sklearn_1.3.2.json`, and renders the tables + forest via
`build_robustness.py`. The recompute reproduces the shipped JSON values exactly
(verified: 0 numeric difference over 30 keys). The Window-14d / Depth-3hop
perturbations are recomputed from frozen pre-extracted features (a public node was
needed once) and labelled `FROZEN_INTERMEDIATE`. Every delta below is measured
against 0.703; `build_robustness.py` on its own is only the renderer.

## What we perturbed

Each check re-extracts Φ under a different parameter, or drops a feature family,
and re-runs the identical leave-one-crew-out logistic evaluation. Full numbers and
95% CIs are in `robustness_results.csv` / `robustness_deltas.csv`; the picture is
`figures/robustness_forest.*`.

**Φ extraction parameters** (observation window, cone depth, value / share floor):
- Window 3 d → 0.672, Window 5 d → 0.698, Window 14 d → 0.722.
- Depth 1 hop → 0.489 (a single hop loses the trail), Depth 3 hops → 0.675.
- Value floor 15 / 20 ETH → 0.690 / 0.661; Share floor 0.2 V → 0.585.

**Feature-family ablations** (drop a family, keep the rest):
- − path timing → 0.692, Geometry only → 0.678, − source context → 0.579,
  − all observability → 0.580.

## What it means

Most local perturbations sit within a couple of hundredths of the 0.703 baseline,
but the signal is **sensitive to a few choices**: it collapses to 0.489 at a single
hop (Depth 1), and drops to about 0.58 when the source-context or the observability
features are removed. In absolute terms it is weak throughout, consistent with the main finding
that this is a triage cue, not a detector. The 95% CI on the baseline ([0.545,
0.847]) is far wider than the spread of the perturbations, so sampling uncertainty
dominates.

## Figures

The robustness figures are **forest-and-table** panels plus a
combined overview, all in the canonical Nature style (`../code/03_make_figures/natstyle.py`):

- `figures/forest_table_groups.png` — A · leave-one-actor-out (17 crews)
- `figures/forest_table_params.png` — B · Φ-extraction parameters
- `figures/forest_table_features.png` — C · feature-family ablation
- `figures/robustness_coef.png` — all perturbations on one combined coefficient axis

Each row shows AUROC (with significance stars), 95% CI, the **group-aware permutation
p**, and the estimable folds; filled markers mark p < 0.05, the diamond is the
unperturbed reference. Every p is the **canonical 1.3.2, N=1000** value — the main
row reproduces `permutation_test.py`'s p = 0.026973 exactly (never the old 1.9.0 p).

## Files

- `run_robustness.py` — the recompute run by `run_all.py`: recomputes AUROC/CI from the
  frozen cones in `inputs/`, writes the data tables, and renders the figures above.
- `build_robustness.py` — the data-table renderer (JSON → the two CSVs).
- `build_forest_tables.py` / `build_robustness_coef.py` — the figure renderers; they read
  `robustness_forest_data.json` and hard-code nothing.
- `robustness_permutation.py` — the **canonical per-variant permutation** (N=1000, seed
  20260629) that produces `robustness_forest_data.json`. It asserts the main variant
  reproduces `permutation_test.py` (p = 27/1001 = 0.026973) before computing the rest.
  This is a multi-hour job; the shipped `robustness_forest_data.json` is its frozen
  output, so the one-command flow renders the figures directly and you only re-run this
  script for a full from-scratch recompute of the per-variant p-values.
- `inputs/` — the frozen extraction cones + widen feature tables the recompute reads.
- `robustness_results_sklearn_1.3.2.json` — the AUROC/CI recompute output.
- `robustness_forest_data.json` — the canonical per-variant AUROC/CI/**p** for the figures.
- `robustness_results.csv` / `robustness_deltas.csv` — the tables, deltas vs 0.703.
- `appendix_checks/` — small scripts behind the paper numbers the main pipeline
  does not produce: the AUPRC row, the sign test, the analytic PU correction, the
  paired reduced-fold baselines of Table F2, and the T10 fold reassignment. Each
  script's output is archived next to it; see the README in that folder. The folder
  carries its own copy of `tier3_lib.py`, which resolves the cones and CSVs
  relative to the package root.
