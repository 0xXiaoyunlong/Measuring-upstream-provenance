"""
permutation_sensitivity.py -- corrected, immutable-block permutation sensitivity.

Companion to permutation_test.py. That script runs the group-count-preserving
null of Ojala & Garriga (2010) and writes its p into main_results.csv. This one
is the corrected null the audit asked for: the blocks are rebuilt with two
fixes, described below, and the 154-block p that comes out is what the paper
quotes for the two main models. The older group-count p is still reported next
to it, and that is the p the battery rows carry; the root README has a
two-nulls section on how the two sit together. Nothing here touches the other
null's files, and nothing was tuned to keep a p under a threshold.

The blocks change in two ways. First the background is de-duplicated by deposit
address: the 139 background events resolve to 137 unique lowercase addresses
(two addresses each deposit twice). The legacy null made 139 singleton blocks,
so a shuffle could put one deposit of a repeat address into the positive set
while its twin stayed in the background, splitting one real actor across the
labels; here the two deposits of a repeat address form one immutable block and
always move together, which gives 17 illicit-crew blocks + 137 background
address-blocks = 154. Second, block integrity holds across the split: every
block has a fold that does not depend on the permuted labels, and a block's
events are always trained and tested together.

Each shuffle labels exactly 17 of the 154 blocks illicit, blocks intact. What
the shuffle cannot also hold fixed is the positive-event count and the
block-size spread: the crews range from 1 to 14 deposits while the largest
background address-block has 2, and Harmony is the unique size-14 block, so a
free permutation matched on event counts or block sizes is simply not
identifiable on this data (the config records this as
NOT_IDENTIFIABLE_WITH_AVAILABLE_BACKGROUND_BLOCKS). A stratified null that
pins Harmony always-illicit would be identifiable, but then Harmony's own
label is never tested, so the size-matched check reported here (variant B)
drops Harmony instead. With only the 16 single-deposit crews left, every
illicit block has size 1 and a null drawn from size-1 blocks is matched
exactly. That is a *different estimand* -- the macro AUROC over the 16
singleton crews with the background re-bucketed into 16 folds -- and it is
labelled as such.

Writes permutation_sensitivity_null.csv (per-shuffle AUROC and block stats),
permutation_sensitivity_summary.csv (observed, N, p, null mean per variant)
and permutation_sensitivity_config.json. The primary permutation files and
main_results.csv are not touched.

    python permutation_sensitivity.py            # 1000 shuffles (slow)
    python permutation_sensitivity.py --fast     # 100 shuffles, smoke test only
"""

import json
import os
import statistics
import sys

import numpy as np

import evaluate as ev
import permutation_test as pt      # reuse the exact LOAGO-by-group machinery

SEED = 20260629
N_PERM = 100 if "--fast" in sys.argv else 1000


def _addrs(events):
    amap = ev._address_map()
    # the LFI/Q25 cross-chain crew is not in entrance_events; fall back to its a0
    return [amap.get(e["event_id"], e.get("event_id")).lower() for e in events]


def build_immutable_blocks(events, addrs):
    """154 immutable blocks: 17 illicit-crew blocks + 137 background address-blocks.
    Background events are de-duplicated by lowercase deposit address, so the two
    repeat-deposit addresses each form a single size-2 block instead of two
    singletons. Returns (blocks, meta): blocks[i] is a list of event indices,
    meta[i] = {"kind", "key", "size"}."""
    pos_crews = sorted({e["crew"] for e in events if int(e["is_illicit"]) == 1})
    blocks, meta = [], []
    for c in pos_crews:
        idx = [i for i, e in enumerate(events) if int(e["is_illicit"]) == 1 and e["crew"] == c]
        blocks.append(idx)
        meta.append({"kind": "illicit_crew", "key": c, "size": len(idx)})
    bg_by_addr = {}
    for i, e in enumerate(events):
        if int(e["is_illicit"]) == 0:
            bg_by_addr.setdefault(addrs[i], []).append(i)
    for a in sorted(bg_by_addr):
        blocks.append(bg_by_addr[a])
        meta.append({"kind": "background_address", "key": a, "size": len(bg_by_addr[a])})
    return blocks, meta


def _labels_from_choice(events, blocks, chosen):
    """Turn a set of chosen block indices into (is_pos, group_of, group_order) for
    pt.loago_by_group. Every event of a chosen block becomes positive with that
    block's group id; unchosen blocks stay background. Blocks are never split."""
    is_pos = [0] * len(events)
    grp = [""] * len(events)
    order = []
    for bi in chosen:
        for i in blocks[bi]:
            is_pos[i] = 1
            grp[i] = f"PB_{bi}"
        order.append(f"PB_{bi}")
    return is_pos, grp, order


def corrected_null(events, addrs, observed, size1_only=False, k=None):
    """Immutable-block group-count-preserving null.

    Picks `n_groups` blocks to be illicit each shuffle, keeping every block intact.
    With size1_only=True the draw is restricted to size-1 blocks (block-size-matched
    conditional variant). Returns (null_aurocs, p, per_shuffle_stats)."""
    blocks, meta = build_immutable_blocks(events, addrs)
    pool = [i for i, m in enumerate(meta) if (not size1_only) or m["size"] == 1]
    n_groups = sum(1 for m in meta if m["kind"] == "illicit_crew"
                   and ((not size1_only) or m["size"] == 1))
    k = k if k is not None else len({e["crew"] for e in events if int(e["is_illicit"]) == 1})

    rng = np.random.RandomState(SEED)
    null, stats = [], []
    for _ in range(N_PERM):
        chosen = sorted(rng.choice(pool, n_groups, replace=False).tolist())
        is_pos, grp, order = _labels_from_choice(events, blocks, chosen)
        auroc = pt.loago_by_group(events, is_pos, grp, order, addrs)
        sizes = [meta[bi]["size"] for bi in chosen]
        null.append(auroc)
        stats.append((auroc, len(chosen), sum(sizes), max(sizes)))
    valid = [(a, g, e, s) for (a, g, e, s) in stats if not np.isnan(a)]
    aurocs = [a for (a, _g, _e, _s) in valid]
    ge = sum(1 for a in aurocs if a >= observed)
    p = (1 + ge) / (1 + len(aurocs))
    return aurocs, p, valid


def main():
    print(f"Corrected immutable-block permutation SENSITIVITY, N={N_PERM}, seed={SEED}\n")
    events = ev._load_events("upstream_features.csv")
    addrs = _addrs(events)

    # --- observed statistics (recomputed here, not read from disk) ---------------
    is_pos = [int(e["is_illicit"]) for e in events]
    crews = sorted({e["crew"] for e in events if int(e["is_illicit"]) == 1},
                   key=lambda c: min(int(e["held_out_fold"]) for e in events if e["crew"] == c))
    group_of = [e["crew"] if int(e["is_illicit"]) else "" for e in events]
    obs_17 = pt.loago_by_group(events, is_pos, group_of, crews, addrs)
    main17 = float(next(r for r in ev._load_results("main_results.csv")
                        if r["crew_set"] == "17_crews")["auroc"])
    assert abs(obs_17 - main17) < 1e-9, f"sensitivity observed {obs_17} != main {main17}"

    null_rows, summary_rows = [], []

    # (A) corrected group-count-preserving null over all 154 immutable blocks -----
    nullA, pA, statsA = corrected_null(events, addrs, obs_17)
    print(f"(A) corrected 154-block null   observed {obs_17:.6f}  p = {pA:.5f}  "
          f"null mean {statistics.mean(nullA):.3f}")
    for (a, g, e, s) in statsA:
        null_rows.append(["corrected_154block", repr(a), g, e, s])
    summary_rows.append(["corrected_154block", repr(obs_17), len(nullA), repr(pA),
                         repr(statistics.mean(nullA)),
                         "positive_block_count_preserved=17; event_count/block_size NOT preserved "
                         "(unconditional size-matched null NOT_IDENTIFIABLE_WITH_AVAILABLE_BACKGROUND_BLOCKS; "
                         "only a Harmony-pinned conditional/stratified null is size-matched)"])

    # (B) block-size-matched CONDITIONAL null over the 16 singleton crews ----------
    #     Drop Harmony (the only multi-deposit crew); K = 16 folds.
    harmony = "Harmony"
    sub = [e for e in events if not (int(e["is_illicit"]) == 1 and e["crew"] == harmony)]
    addrs_sub = _addrs(sub)
    crews16 = sorted({e["crew"] for e in sub if int(e["is_illicit"]) == 1},
                     key=lambda c: min(int(e["held_out_fold"]) for e in sub if e["crew"] == c))
    is_pos_sub = [int(e["is_illicit"]) for e in sub]
    grp_sub = [e["crew"] if int(e["is_illicit"]) else "" for e in sub]
    obs_16 = pt.loago_by_group(sub, is_pos_sub, grp_sub, crews16, addrs_sub)
    nullB, pB, statsB = corrected_null(sub, addrs_sub, obs_16, size1_only=True, k=16)
    print(f"(B) size-matched 16-singleton  observed {obs_16:.6f}  p = {pB:.5f}  "
          f"null mean {statistics.mean(nullB):.3f}   (Harmony excluded; different estimand)")
    for (a, g, e, s) in statsB:
        null_rows.append(["singleton_conditional_sizematched", repr(a), g, e, s])
    summary_rows.append(["singleton_conditional_sizematched", repr(obs_16), len(nullB), repr(pB),
                         repr(statistics.mean(nullB)),
                         "block_size matched exactly (all size 1); Harmony excluded; "
                         "estimand = macro AUROC over 16 singleton crews, K=16"])

    ev._atomic_write("permutation_sensitivity_null.csv",
                     ["null_variant", "shuffled_auroc", "n_positive_groups",
                      "n_positive_events", "max_block_size"], null_rows)
    ev._atomic_write("permutation_sensitivity_summary.csv",
                     ["null_variant", "observed_auroc", "n_permutations", "p_value",
                      "null_mean", "preservation_note"], summary_rows)

    config = {
        "reference": "Ojala & Garriga (2010), immutable-block variant",
        "seed": SEED, "n_permutations": N_PERM, "model": "logistic", "metric": "AUROC",
        "split": "LOAGO_leave_one_actor_group_out",
        "empirical_p_formula": "p = (1 + #{permuted_AUROC >= observed}) / (1 + N)",
        "nan_handling": ("permutations whose macro AUROC is NaN (a degenerate fold with no "
            "estimable positive) are dropped from BOTH numerator and denominator, so the "
            "effective N can fall just below the nominal 1000; the effective N is the "
            "n_permutations column of permutation_sensitivity_summary.csv."),
        "relationship_to_primary": ("The corrected 154-block null computed here is the one "
            "the paper quotes for the two main models (p = 0.031 original background, "
            "0.002 balanced); the group-count-preserving null from permutation_test.py is "
            "reported alongside (p = 0.027 / 0.003) and is the p carried by the 30-variant "
            "battery rows, where the corrected blocks are not re-derived. This file never "
            "overwrites the other null's outputs; which null is quoted where was "
            "pre-committed, independent of which p-value is smaller -- both ship in "
            "data/. See the two-nulls section of the root README."),
        "nulls": {
            "corrected_154block": {
                "blocks": "17 illicit-crew blocks + 137 background address-blocks = 154",
                "background_dedup": "background events grouped by lowercase deposit address "
                                    "(139 events -> 137 addresses; two addresses deposit twice)",
                "draw": "each shuffle selects 17 of the 154 blocks as illicit, blocks kept intact",
                "preserved": "number of positive blocks (17); full block integrity",
                "not_preserved": "number of positive events; block-size distribution",
                "identifiability": ("unconditional size-matched null is "
                                   "NOT_IDENTIFIABLE_WITH_AVAILABLE_BACKGROUND_BLOCKS (no background "
                                   "block has 14 deposits to match Harmony, the unique size-14 block); "
                                   "a Harmony-pinned conditional/stratified null IS size-matched but "
                                   "cannot test Harmony's own label, so variant B (Harmony-excluded) "
                                   "is reported instead")},
            "singleton_conditional_sizematched": {
                "blocks": "16 singleton illicit-crew blocks + 137 background address-blocks",
                "draw": "each shuffle selects 16 size-1 blocks as illicit (block-size matched)",
                "preserved": "number of positive groups (16) AND block size (all 1)",
                "estimand": "macro AUROC over the 16 single-deposit crews, K=16 folds, "
                            "Harmony excluded -- a different estimand from the 0.703 primary",
                "identifiability": "IDENTIFIABLE (conditional on excluding the one multi-event crew)"}}}
    json.dump(config, open(os.path.join(ev.RESULTS, "permutation_sensitivity_config.json"),
                           "w", encoding="utf-8"), indent=2)

    print("\nwrote permutation_sensitivity_null.csv, permutation_sensitivity_summary.csv, "
          "permutation_sensitivity_config.json (primary permutation files untouched).")


if __name__ == "__main__":
    main()
