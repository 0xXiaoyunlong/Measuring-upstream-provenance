# 3. Regenerating the figures

Each script rebuilds one figure and writes it to `../../figures/`. Every number a
figure shows is read at plot time from the archived CSVs / JSONs. Regenerate the
figures after any change to the data.

```bash
pip install -r ../../requirements-exact.txt
python figdata.py        # oracle: asserts the paper's headline numbers first
python fig3_signal.py
python fig4_audit.py
python fig5_forest.py
python fig6_timeline.py
python fig7_dualcontrol.py
python figC1_heatmap.py
```

| script | figure in the paper | reads from |
|---|---|---|
| `fig3_signal.py` | Fig. 3 — primary vs sensitivity, per-crew scatter, permutation null | `main_results.csv`, `per_crew_results.csv`, `permutation_null.csv`, `permutation_summary.csv` |
| `fig4_audit.py` | Fig. 4 — AUROC before/after recovery + increment forest | `missingness_audit_metrics.csv`, `missingness_audit_increments.csv` |
| `fig5_forest.py` | Fig. 5 — paired robustness forest, both backgrounds | `robustness_forest_data.json` of the root package **and** of `balanced_background/` (via `figdata.py`) |
| `fig6_timeline.py` | Fig. 6 — deposit timeline, balanced vs original background | `entrance_events.csv` of both packages |
| `fig7_dualcontrol.py` | Fig. 7 — dual-control AUROC comparison, 4 rows | `main_results.csv` + `permutation_null.csv` of both packages |
| `figC1_heatmap.py` | Fig. C1 — per-crew missingness heatmap | `missingness_by_crew.csv` |

Three of the files are support modules and make no figure of their own.
`natstyle.py` is the design system for fig3/4/C1 (palette, type scale, 600 dpi
PNG export). `research_style.py` does the same job for fig5, with the Okabe-Ito
palette and 450 dpi export. `figdata.py` is the shared read-only data layer for the
two-background figures; run `python figdata.py` by itself and it acts as an oracle,
asserting the headline numbers of both packages (anchor 0.703 / 0.802, the
p-values, the funnel, the audit deltas, the forest ranges), all read directly
from the archived tables.

Two figures have no script. `fig1_dataset.png` is the dataset waterfall,
hand-finished from the funnel counts in `data/dataset_funnel.csv` (the counts
themselves are asserted by the figdata oracle), and `fig2.png` is a hand-drawn
schematic of the funding cone with no experimental values in it. Both are
checked against the SHA-256 values recorded in `run_all.py`.

If you compare against the PDF, Fig. C1 will look smaller there than
`figC1_missingness.png` does here. The manuscript embeds this figure's 150 dpi
preview export; the content is identical.

Both design systems stick to colourblind-safe colours: Φ/primary blue `#0072B2`,
newly-gated sky `#56B4E9`, factory orange `#E69F00`, neutral grey `#8C8C8C`. fig5
uses its own pair, dark `#454545` for the original background and `#fda501` for
the balanced one.
