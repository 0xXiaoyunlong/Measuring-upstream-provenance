# The temporally balanced background

The main dataset has a known timing asymmetry: all 139 background deposits are
from 2022, while the illicit events run from 2022 to 2026. A classifier that
picked up deposit timing rather than funding structure would benefit from that
asymmetry. This folder holds the control run. Only the background changes: the
139 original deposits are replaced by 140 deposits spread across 2022–2026
(30/30/30/20/30 by year, one event per address), passed through the same
evidence gates and Φ extraction. The 30 illicit events and the 17 crews are
untouched.

On this background the 17-group macro AUROC is **0.802** [0.671, 0.910],
corrected-null **p = 0.002** (group-count null p = 0.003, size-matched
p = 0.001), and the empirical null mean rises to 0.54, from 0.53 on the original
background. The 18-group sensitivity run comes out at 0.772 [0.629, 0.892] with
p = 0.006, where the original background gave 0.631 at p = 0.146 and was not
significant. The 30-variant battery gives 25 of 30 significant variants (6 of 30 on the
original background); 28 of 30 are above the null mean.

The paper treats this as *indirect* evidence. Its reported numbers remain those
of the original, pre-registered background; this run only rules out the timing
explanation for them. Figures 5–7 of the paper draw on both packages side by
side (rendered by `../code/03_make_figures/`, which reads this folder directly).

## Layout

```
├── data/         same schema as ../data — 170 events (30 illicit + 140 background),
│   │             per-event predictions, both permutation nulls, audit outputs
│   └── sidecar/  factory tables incl. _bgswap_ledgers.json (cones of the 140)
├── code/
│   ├── 01_collect_and_build_features/   byte-identical to the root package
│   └── 02_evaluate_cross_actor/         byte-identical except build_sidecar_features.py,
│                                        which additionally loads _bgswap_ledgers.json
│                                        (the change is commented in the file)
├── archive/frozen_extraction/           the frozen phi.py/pipeline.py (the battery
│                                        rebuilds variant features through them)
└── robustness/   the 30-variant battery on this background: results under both
                  scikit-learn 1.3.2 and 1.9.0, the forest JSON with per-variant
                  permutation p, and the battery figures (tier3_lib.py is in
                  its appendix_checks/, same spot as in the root package)
```

We left `run_all.py` out of this folder on purpose. The root `run_all.py` is
anchored to the original background, so its 0.703 gate would reject this data.
To regenerate, run `code/02_evaluate_cross_actor/evaluate.py`, then
`permutation_test.py` and `permutation_sensitivity.py` (they fill the permutation
columns), then `robustness/run_robustness.py`. Before release we re-checked that
`evaluate.py` reproduces `per_crew_results.csv` and `per_event_predictions.csv`
byte-identically from the frozen inputs (macro 0.802177), and that
`main_results.csv` differs only in the permutation columns until the permutation
scripts have run.

The pipeline code still mentions the original 139-event background in a few
comments and progress prints. We left those as written, since the control only
means something if the code stays byte-identical; every computed number comes from the data files
in this folder. The corrected null's variant label `corrected_154block` and the
block-count prose in `permutation_sensitivity_config.json` and
`permutation_config.json` are inherited the same way. The block construction itself is dynamic (same address-dedup rule),
and on this background (140 events / 140 unique addresses, against the
original's 139 / 137) it actually builds 17 + 140 = 157 blocks. Only the label
and description still say 154.
