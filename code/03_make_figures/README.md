# 3. Regenerating the figures

Each script rebuilds one figure and writes it to `../../figures/`. Every number a
figure shows is **read from the CSVs in `../../data/`** — nothing is typed into the
plotting code. Regenerate them after any change to the data. Each script prints
which files and values it used.

```bash
pip install -r ../../requirements.txt
python figure1_dataset_funnel.py
python figure2_provenance_cone.py
python figure3_cross_actor_signal.py
python figure4_missingness_audit.py
python figureC1_missingness_heatmap.py
```

| script | figure | reads from |
|---|---|---|
| `figure1_dataset_funnel.py` | evidence-gate funnel + dataset composition | `dataset_funnel.csv`, `entrance_events.csv` |
| `figure2_provenance_cone.py` | schematic of the backward funding cone | (a diagram; its parameters are imported from `build_upstream_features.py`) |
| `figure3_cross_actor_signal.py` | primary vs sensitivity, per-crew scatter, permutation null | `main_results.csv`, `per_crew_results.csv`, `permutation_null.csv`, `permutation_summary.csv` |
| `figure4_missingness_audit.py` | AUROC before/after recovery + increment forest | `missingness_audit_metrics.csv`, `missingness_audit_increments.csv` |
| `figureC1_missingness_heatmap.py` | per-crew missingness heatmap | `missingness_by_crew.csv` |

Figure 2 is a drawing rather than a plot of numbers. The parameters printed on it
(7-day window, 10 ETH threshold, depth 2, and so on) are imported from the
extraction code.

Colours are the colourblind-safe Okabe–Ito set (blue `#0072B2`, orange `#E69F00`,
grey `#7F7F7F`); the three panels of Figures 3 and 4 are forced square so their
proportions match the published version.
