# Data

*The column documentation below is carried over verbatim from the root package,
so its worked-example numbers still describe the original background: 169
events, 139 background, 0.703 / p = 0.027. The CSVs actually sitting in this
folder are the balanced-background control — 170 events (30 illicit + 140
balanced background), anchor 0.802. That run's own numbers are in
`../README.md`.*

All CSVs, UTF-8 with a header row. Everything in here is either a raw on-chain fact
or a number computed from one. Crew naming: the five well-known crews use their real
names, the twelve new ones are N01–N12 (same as the paper), and background deposits
are just `background`.

## The dataset

### `entrance_events.csv` — the list of deposits (169 rows)
One row per mixer-entrance deposit.

| column | meaning |
|---|---|
| `event_id` | stable id for the deposit (`OLD-###` for pre-existing samples, `NEW-###` for newly gated crews, `CONV-…` for Convergence) |
| `crew` | which laundering crew, or `background` |
| `crew_type` | `known`, `newly_gated`, or `background` |
| `is_illicit` | 1 for a known illicit deposit, 0 for background |
| `deposit_address` | the depositing address (public on-chain) |
| `deposit_time_utc` | deposit time, filled for all 169 rows (backfilled from the frozen ledgers for the earlier benchmark samples). Together with the address this is the (a0, t0) the paper says everything is re-derivable from. |
| `deposit_tx` | the deposit transaction hash. 145 of 169; for 24 background deposits in the Audius/Monkey windows the archive kept only (address, time, block), so the hash is blank — any node will give it to you from the other two. |
| `held_out_fold` | which leave-one-crew-out fold this row is tested in (0–16). Ships explicitly so the evaluation reproduces exactly. |

### `upstream_features.csv` — the 15 provenance features (169 rows)
The model's actual input: for each deposit, the 15 features plus a companion
`observed_*` flag for each (1 = we saw a value, 0 = it could not be observed).
A blank feature cell means the feature was not observed (not a zero; see
`code/01_collect_and_build_features/build_upstream_features.py`).

Key columns: `event_id`, `crew`, `is_illicit`, `held_out_fold`, then the 15
features below, then `observed_<feature>` for each.

Feature dictionary (see the paper's Appendix D):

| feature | plain reading |
|---|---|
| `observed_hops` | length of the traced funding path |
| `path_elapsed_time` | hours from the earliest traced transfer to the deposit |
| `dwell_mean_h`, `dwell_max_h` | how long intermediaries held the money before passing it on |
| `coverage` | share of the deposit traced back to a source |
| `attribution_ambiguous` | whether a competing funder existed at some hop |
| `node_context_out_degree`, `node_context_in_degree` | distinct counterparties of the intermediaries (context) |
| `wallet_age_min_h` | youngest observed intermediary age (a lower bound — left-censored) |
| `node_context_resid_Rrange_max` | largest balance swing at a path node |
| `node_context_resid_nin_mean`, `node_context_resid_nout_mean` | mean in/out transfer counts at path nodes |
| `boundary_type` | how the backward walk ended (e.g. `eoa-source`, `contract`, `asset-conversion`) |
| `censoring_reason` | why observation was cut off (kept for transparency; not fed to the model) |
| `observation_confidence` | traced coverage, reused as a data-quality flag |

### `dataset_funnel.csv` — how 86 candidates became 17 crews
The evidence-gate funnel: `stage`, `n_candidates`, and a plain-English
`description` of each gate (census → gate-eligible → 4 gates → 12 new crews).

### `evidence_gate_decisions.csv` — the same funnel, per candidate (86 rows)
One row per candidate address, with the archived status it received at each gate:
`gate0_census`, `gate1_chain_verification`, `gate2_second_source`,
`gate3_independence`, `gate4_frozen_extraction`, and `admitted_as` (N01–N12 for
the twelve that made it, blank otherwise). Statuses are the frozen adjudication
codes, verbatim — count the passes yourself and you get the funnel numbers.
(One quirk: the 20 professional-scale candidates set aside at the census were
still run through gate 1 for completeness, so 68 rows carry a gate-1 code even
though only 48 were gate-eligible. The pass counts are what feed the funnel.)
What's deliberately NOT here: incident names and the free-text reasoning, because
they would undo the paper's pseudonymisation of N01–N12. The full dossiers
(attribution write-ups, URLs, per-gate memos) stay in the versioned archive and
can be provided to editors on request.

## The results (recomputed by the scripts under scikit-learn 1.3.2)

These CSVs are **overwritten by the scripts** in `code/02_evaluate_cross_actor/`,
so they always match the run. All AUROC/CI/observed/p values come from the same
reference-environment run (scikit-learn 1.3.2). The **logistic model only** — the
gradient-boosting rows were removed (no executable boosting pipeline ships here).

### `per_crew_results.csv` — per-crew AUROC (17 rows)
The per-crew table out of `evaluate.py`. Columns: `crew`, `crew_type`, `n_events`,
`n_background_in_cohort`, `auroc_logistic`. The plain average of `auroc_logistic`
is the headline **0.702949, reported 0.703**.

### `per_event_predictions.csv` — held-out scores (169 rows)
Every one of the 169 deposits gets one held-out logistic
prediction from the fold it was tested in. Columns: `event_id`, `crew`,
`is_illicit`, `held_out_fold`, `model` (`logistic`), `score`, `rank_percentile`.
The scores are **full precision** (not rounded), so recomputing any crew's AUROC
from its cohort's scores reproduces `per_crew_results.csv` exactly.

### `main_results.csv` — headline AUROCs (2 rows)
Written by `evaluate.py` (AUROC, CI) and `permutation_test.py` (the permutation
columns). The primary (17 crews) and sensitivity (18 crews) results: `auroc`, 95%
interval (`ci_low`, `ci_high`), `permutation_observed` (equals `auroc`), and
`p_value` — the `p_value` here is the **(legacy) group-count-preserving harness**
p-value (0.027 for 17 crews, 0.146 for 18).

### `permutation_null.csv` — the null distribution (group-count harness)
Written by `permutation_test.py`. Columns: `crew_set` (`17_crews` / `18_crews`),
`model` (`logistic`), `shuffled_auroc`; 1000 shuffles per crew set.

### `permutation_summary.csv` — observed values and p-values (group-count harness)
Per (`crew_set`, `model`): `observed_auroc`,
`n_permutations`, `p_value`, `null_mean` (≈ 0.53, not 0.5 — see the paper). The
observed value equals the macro AUROC that `evaluate.py` reports; it is recomputed,
not archived. NaN permutations (degenerate folds) are dropped from both numerator
and denominator, so `n_permutations` can be just under 1000.

### `permutation_legacy_summary.csv` — the same, labelled `legacy_group_count_preserving_null`
Identical to `permutation_summary.csv` with an
explicit `null_name` column, so the harness null cannot be mistaken for the
corrected sensitivity below.

### `permutation_config.json` — the harness test configuration
Written by `permutation_test.py`: `null_name` = `legacy_group_count_preserving_null`,
seed 20260629, 1000 permutations, shuffling whole actor-group blocks (17 crew + 139
background singleton = 156, pick 17), p-value formula
`p = (1 + #{permuted ≥ observed}) / (1 + N)`, and a pointer to the corrected sensitivity.

### `permutation_sensitivity_null.csv` — the corrected + conditional nulls
One row per shuffle from `permutation_sensitivity.py`: `null_variant`
(`corrected_154block` / `singleton_conditional_sizematched`), `shuffled_auroc`,
`n_positive_groups`, `n_positive_events`, `max_block_size` — so the block-size
distribution the null actually draws is auditable.

### `permutation_sensitivity_summary.csv` — the corrected + conditional p-values
Written by `permutation_sensitivity.py`. Per `null_variant`: `observed_auroc`,
`n_permutations`, `p_value` (**reported as-is**, never tuned for significance),
`null_mean`, `preservation_note`. The `corrected_154block` null de-duplicates
background by address (154 immutable blocks); the `singleton_conditional_sizematched`
null is a different estimand over the 16 single-deposit crews (Harmony excluded).

### `permutation_sensitivity_config.json` — the sensitivity null definitions
Written by `permutation_sensitivity.py`: block construction, what each null does and
does not preserve, the `NOT_IDENTIFIABLE_WITH_AVAILABLE_BACKGROUND_BLOCKS` note, NaN
handling, and the relationship to the primary null (see the two-nulls section of the root README).

While I'm at it, the bootstrap behind every confidence interval in these files:
resampling unit is the **crew** (never individual deposits — pre-registered that
way), 2000 resamples, percentile 2.5/97.5, numpy `RandomState(0)`. Recorded in
the frozen analysis scripts; stated here so nobody has to dig for it.

## The missingness audit

### `outflow_recovery.csv` — recovering the missing outflow records (12 rows)
For each newly gated crew: `source_address`, how many source nodes we
`nodes_queried` / `nodes_recovered`, and the `recovery_status` (fully / partially
recovered, nothing to recover, or unrecoverable). Tally: 1 fully, 3 partially,
4 nothing-to-recover, 4 unrecoverable.

### `missingness_audit_metrics.csv` — AUROC before vs after recovery
Recomputed by `missingness_audit.py` under scikit-learn 1.3.2, so the provenance
(Phi) baseline here is the same **0.703** the main analysis reports. Per `stage`
(`before_recovery` / `after_recovery`) and `feature_set` (provenance /
factory-pattern / combined): `auroc`, `ci_low`, `ci_high`. Factory-pattern alone
drops from 0.75 (before) to 0.51 (after recovery).

### `missingness_audit_increments.csv` — marginal AUROC of factory-pattern
Written by `missingness_audit.py`. The AUROC change of factory-pattern (and
combined) over provenance alone, after recovery: `auroc_change`, `ci_low`,
`ci_high`. The combined increment is -0.006 with a CI that spans zero.

### factory / sensitivity inputs
`sidecar/sidecar_features_before_recovery.csv` and
`sidecar/sidecar_features_after_recovery.csv` are the factory (`fac_*`) feature
tables (before and after out-edge recovery), read by `missingness_audit.py` (both
live in the `sidecar/` subfolder, described in `sidecar/README.md`). `lfi_q25_features.csv`
is the one extra cross-chain crew (N13) used by the 18-crew sensitivity in
`evaluate.py`. These are frozen inputs, keyed to the deposits by address.

### `missingness_by_crew.csv` — the missingness heatmap (18 rows)
For each crew (and the background pool): the share of that crew's events for which
each feature was not observed. Columns: `crew`, `n_events`, then one column per
feature. This is the source for Figure C1.

## A note on the raw transactions

The full per-transaction records are not shipped as files (a complete re-crawl is
tens of gigabytes and better re-fetched fresh). They are re-derivable from public
Ethereum nodes given each deposit's address and time; `code/01_collect_and_build_features`
is the reference implementation that does exactly that.
