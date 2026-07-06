# Data

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

## The results (archived, cite these)

### `per_crew_results.csv` — per-crew AUROC (17 rows)
The canonical per-crew results (the paper's Table B1). Columns: `crew`,
`crew_type`, `n_events`, `n_background_in_cohort`, `auroc_logistic`,
`auroc_boosting`. The plain average of `auroc_logistic` is the paper's headline
**0.6824**.

### `per_event_predictions.csv` — the archived held-out scores (338 rows)
Every one of the 169 deposits gets a held-out prediction from the fold it was
tested in, for both models (169 × 2). Columns: `event_id`, `crew`, `is_illicit`,
`held_out_fold`, `model`, `score`, `rank_percentile`. These are the frozen
out-of-fold scores the per-crew AUROCs were computed from — recompute any crew's
AUROC from its cohort's scores and you land on `per_crew_results.csv`, with one
footnote: scores are archived to 5 decimal places, and in the N12 cohort that
rounding creates a tie the full-precision scores didn't have. `rank_percentile`
keeps the exact within-fold order, which resolves it (archived 0.8333). Ties in
the boosting scores are real ties (identical model outputs) and count as
half-wins, as usual for AUROC.

### `main_results.csv` — headline AUROCs (2 rows)
The primary (17 crews) and sensitivity (18 crews) results: `auroc`, 95% interval
(`ci_low`, `ci_high`), the permutation `permutation_observed` value, and `p_value`.

### `permutation_null.csv` — the null distribution
The shuffled AUROCs from the crew-preserving permutation test. Columns:
`permutation_id`, `crew_set` (`17_crews` / `18_crews`), `model`
(`logistic` / `boosting`), `shuffled_auroc`. 1000 logistic + 300 boosting for
17 crews, 1000 logistic for 18 crews.

### `permutation_summary.csv` — observed values and p-values
Per (`crew_set`, `model`): `observed_auroc`, `n_permutations`, `p_value`,
`null_mean` (≈ 0.53, not 0.5 — see the paper).

### `permutation_config.json` — the frozen test configuration
The permutation run's config file, verbatim from the archive: seed 20260629,
1000 permutations, shuffling whole actor-group blocks (17 positive groups kept),
and the p-value formula `p = (1 + #{permuted ≥ observed}) / (1 + N)`.
Field names inside are the archive's internal ones (e.g. the split is called
`LOAGO_leave_one_actor_group_out` — that's the paper's leave-one-crew-out).

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
Per `stage` (`before_recovery` / `after_recovery`) and `feature_set`
(provenance / factory-pattern / combined): `auroc`, `ci_low`, `ci_high`.

### `missingness_audit_increments.csv` — marginal AUROC of factory-pattern
The AUROC change of factory-pattern (and combined) over provenance alone, after
recovery: `auroc_change`, `ci_low`, `ci_high`. Both intervals span zero.

### `missingness_by_crew.csv` — the missingness heatmap (18 rows)
For each crew (and the background pool): the share of that crew's events for which
each feature was not observed. Columns: `crew`, `n_events`, then one column per
feature. This is the source for Figure C1.

## A note on the raw transactions

The full per-transaction records are not shipped as files (a complete re-crawl is
tens of gigabytes and better re-fetched fresh). They are re-derivable from public
Ethereum nodes given each deposit's address and time; `code/01_collect_and_build_features`
is the reference implementation that does exactly that.
