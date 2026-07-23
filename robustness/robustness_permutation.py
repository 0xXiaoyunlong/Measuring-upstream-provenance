# -*- coding: utf-8 -*-
"""
Canonical per-variant group-aware permutation p-values for the robustness forest.

For every robustness perturbation (the main model, each leave-one-crew-out variant,
each Phi-extraction-parameter variant, and each feature-family ablation) this computes
the observed macro LOAGO AUROC, a group-bootstrap 95% CI, and a group-aware permutation
p-value under the reference environment (scikit-learn 1.3.2), N=1000, seed 20260629 --
the same null construction as code/02's permutation_test.py (crew blocks + one
background singleton per deposit; each shuffle picks the same number of crew blocks).

Correctness gate: the main variant (17 crews, full features) must reproduce the
package's canonical permutation exactly -- observed 0.702949 and p = 0.026973. The
script asserts this before doing anything else, so the per-variant p-values cannot
silently drift from the headline analysis.

It writes robustness_forest_data.json (schema: none/loo/params/feats rows carrying
auroc/lo/hi/p/folds, plus main), which build_forest_tables.py renders. All p-values
are the canonical 1.3.2 values -- never the historical 1.9.0-layer numbers.

    python robustness_permutation.py            # N=1000 (slow, hours)
    python robustness_permutation.py --fast     # N=100 smoke test, not publishable
"""
import hashlib
import io
import json
import os
import statistics
import sys

import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "appendix_checks"))
import tier3_lib as T                      # relative-path cone library (gives T.ev, T.build_rows, T.ent)

SEED = 20260629
N_PERM = 100 if "--fast" in sys.argv else 1000
# The 1000 permutations are independent and the p-value depends only on the SET of
# null AUROCs (order-independent), so they parallelise exactly: we draw the shuffles
# sequentially with RandomState(SEED) -- identical to permutation_test.py, which the
# main-variant gate verifies (p = 27/1001) -- then compute their AUROCs across workers.
_NWORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))
_POOL = None                               # set in main(); None -> sequential fallback
NUM = T.ev.NUMERIC_FEATURES
BK = T.ev.BOUNDARY_KINDS

# feature families (same partition as run_robustness.py / the paper)
TIMING = ["path_elapsed_time", "dwell_mean_h", "dwell_max_h", "wallet_age_min_h"]
SRC = ["node_context_out_degree", "node_context_in_degree", "node_context_resid_Rrange_max",
       "node_context_resid_nin_mean", "node_context_resid_nout_mean"]
GEOM = ["observed_hops", "coverage", "attribution_ambiguous"]
OC = ["observation_confidence"]

ADDR = {r["event_id"]: r["deposit_address"] for r in T.ent}


def _bg_fold(address, k):
    return int(hashlib.sha256(address.lower().encode()).hexdigest(), 16) % k


def _design(rows, train, keep, flags, bd):
    med = {}
    for f in keep:
        s = [T.ev.to_number(e[f], f) for e in train]
        s = [v for v in s if not np.isnan(v)]
        med[f] = float(np.median(s)) if s else 0.0
    M = []
    for e in rows:
        row = []
        for f in keep:
            v = T.ev.to_number(e[f], f)
            if np.isnan(v):
                row.append(med[f])
                if flags: row.append(0.0)
            else:
                row.append(v)
                if flags: row.append(1.0)
        if bd:
            row.extend(1.0 if e["boundary_type"] == kk else 0.0 for kk in BK)
        M.append(row)
    return np.array(M, dtype=float)


def _auroc_cfg(te, tr, yte, ytr, keep, flags, bd):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    Xtr = _design(tr, tr, keep, flags, bd)
    Xte = _design(te, tr, keep, flags, bd)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), np.asarray(ytr))
    return roc_auc_score(np.asarray(yte), m.predict_proba(sc.transform(Xte))[:, 1])


def _loago(rows, is_pos, group_of, group_order, addrs, keep, flags, bd):
    """Macro leave-one-actor-group-out AUROC for an (optionally permuted) labelling,
    with an explicit feature subset. Backgrounds bucket by sha256(address) mod K; group
    gi is tested against bucket gi. With real labels + full features this reproduces
    the canonical evaluate.py macro AUROC."""
    k = len(group_order)
    idx = list(range(len(rows)))
    aurocs = []
    for gi, g in enumerate(group_order):
        test = [i for i in idx if (is_pos[i] and group_of[i] == g)
                or (not is_pos[i] and _bg_fold(addrs[i], k) == gi)]
        if not test:
            continue
        train = [i for i in idx if i not in test]
        yte = [is_pos[i] for i in test]
        ytr = [is_pos[i] for i in train]
        if sum(yte) in (0, len(yte)) or sum(ytr) == 0:
            continue
        aurocs.append(_auroc_cfg([rows[i] for i in test], [rows[i] for i in train],
                                 yte, ytr, keep, flags, bd))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def _make_blocks(rows):
    """One block per illicit crew (all its deposits) + one singleton per background --
    the canonical legacy group-count-preserving null construction."""
    crews = sorted({e["crew"] for e in rows if int(e["is_illicit"]) == 1})
    blocks = [[i for i, e in enumerate(rows) if int(e["is_illicit"]) == 1 and e["crew"] == c]
              for c in crews]
    blocks += [[i] for i, e in enumerate(rows) if int(e["is_illicit"]) == 0]
    return blocks, len(crews)


def _null_batch(payload):
    """Worker: compute the null AUROC for a batch of chosen block-sets. Runs in a
    separate process (single-thread BLAS); the parent generated the block-sets, so the
    result set is identical to the sequential computation."""
    batch, rows, addrs, blocks, keep, flags, bd = payload
    out = []
    for chosen in batch:
        pos = [0] * len(rows); grp = [""] * len(rows); order = []
        for bi in chosen:
            for i in blocks[bi]:
                pos[i] = 1; grp[i] = f"PG_{bi}"
            order.append(f"PG_{bi}")
        out.append(_loago(rows, pos, grp, order, addrs, keep, flags, bd))
    return out


def variant(rows, eids, keep=NUM, flags=True, bd=True):
    """Observed AUROC, group-bootstrap CI, estimable folds, and the group-aware
    permutation p (N_PERM, SEED) for one variant's feature rows."""
    addrs = [ADDR[e] for e in eids]
    is_pos = [int(e["is_illicit"]) for e in rows]
    crews = sorted({e["crew"] for e in rows if int(e["is_illicit"]) == 1},
                   key=lambda c: min(int(e["held_out_fold"]) for e in rows if e["crew"] == c))
    group_of = [e["crew"] if int(e["is_illicit"]) else "" for e in rows]

    # observed per-fold AUROCs (for point estimate + CI + fold count)
    per = {}
    for gi, g in enumerate(crews):
        test = [i for i in range(len(rows)) if (is_pos[i] and group_of[i] == g)
                or (not is_pos[i] and _bg_fold(addrs[i], len(crews)) == gi)]
        if not test:
            continue
        train = [i for i in range(len(rows)) if i not in test]
        yte = [is_pos[i] for i in test]; ytr = [is_pos[i] for i in train]
        if sum(yte) in (0, len(yte)) or sum(ytr) == 0:
            continue
        per[g] = _auroc_cfg([rows[i] for i in test], [rows[i] for i in train], yte, ytr, keep, flags, bd)
    vals = list(per.values())
    obs = statistics.mean(vals)
    lo, hi = T.ev.group_bootstrap_ci(vals)

    # group-aware permutation null: draw all shuffles sequentially (exact, matches
    # permutation_test.py), then compute their AUROCs -- in parallel if a pool is set.
    blocks, n_pos = _make_blocks(rows)
    rng = np.random.RandomState(SEED)
    chosen_list = [sorted(rng.choice(len(blocks), n_pos, replace=False).tolist())
                   for _ in range(N_PERM)]
    if _POOL is not None:
        nb = _NWORKERS
        batches = [chosen_list[i::nb] for i in range(nb)]
        payloads = [(b, rows, addrs, blocks, keep, flags, bd) for b in batches if b]
        nulls = [a for sub in _POOL.map(_null_batch, payloads) for a in sub]
    else:
        nulls = _null_batch((chosen_list, rows, addrs, blocks, keep, flags, bd))
    null = [a for a in nulls if not np.isnan(a)]
    ge = sum(1 for a in null if a >= obs)
    p = (1 + ge) / (1 + len(null))
    return round(obs, 3), round(lo, 3), round(hi, 3), len(vals), p


def _widen_rows(featdict, active):
    ebid = {r["event_id"]: r for r in T.ent}

    def frozen_row(eid):
        er = ebid[eid]; fr = T.uf[eid]
        row = {"crew": er["crew"], "is_illicit": er["is_illicit"], "held_out_fold": er["held_out_fold"]}
        for ft in T.FEATS: row[ft] = fr.get(ft, "")
        return row

    def feat_row(eid, fd):
        er = ebid[eid]
        row = {"crew": er["crew"], "is_illicit": er["is_illicit"], "held_out_fold": er["held_out_fold"]}
        for ft in T.FEATS: row[ft] = fd.get(ft, "")
        return row
    rows, eids = [], []
    for r in T.ent:
        eid = r["event_id"]
        if eid in featdict:
            rows.append(feat_row(eid, featdict[eid])); eids.append(eid)
        elif eid not in active:
            rows.append(frozen_row(eid)); eids.append(eid)
    return rows, eids


CKPT = os.path.join(HERE, ".forest_ckpt.json")


def _load_ckpt():
    return json.load(io.open(CKPT, encoding="utf-8")) if os.path.exists(CKPT) else {}


def _save_ckpt(ck):
    tmp = CKPT + ".tmp"
    json.dump(ck, io.open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, CKPT)


def main():
    import time
    global _POOL
    print(f"Canonical per-variant robustness permutation, N={N_PERM}, seed={SEED} "
          f"({_NWORKERS} workers)", flush=True)
    if _NWORKERS > 1 and N_PERM > 1:
        _POOL = mp.Pool(_NWORKERS)
    BASE, base_eids, _ = T.build_rows(7, 2, 10, 0.1)
    crews = sorted({r["crew"] for r in T.ent if r["is_illicit"] == "1"})

    band = json.load(io.open(os.path.join(HERE, "inputs", "tier3_band_scan.json"), encoding="utf-8"))
    active14 = set(e for e, v in band.items() if v["band_edges"] > 0)
    depthA = set(r["event_id"] for r in T.uf.values()
                 if r.get("censoring_reason", "") == "depth-limit" or r.get("boundary_type", "") == "depth-limit")
    wf = json.load(io.open(os.path.join(HERE, "inputs", "tier3_widen_features.json"), encoding="utf-8"))
    w14_rows, w14_eids = _widen_rows(wf["14d"], active14)
    w3_rows, w3_eids = _widen_rows(wf["3hop"], depthA)

    def loo_rows(c):
        rows = [BASE[i] for i in range(len(BASE)) if BASE[i]["crew"] != c]
        eids = [base_eids[i] for i in range(len(BASE)) if BASE[i]["crew"] != c]
        return rows, eids

    # ordered compute specs: (key, thunk -> (auroc, lo, hi, folds, p))
    specs = [("main", lambda: variant(BASE, base_eids))]
    for c in crews:
        specs.append((f"loo:{c}", (lambda c=c: variant(*loo_rows(c)))))
    for name, wv in [("Window 3 d", (3, 2, 10, 0.1)), ("Window 5 d", (5, 2, 10, 0.1)),
                     ("Depth 1 hop", (7, 1, 10, 0.1)), ("Value floor 15 ETH", (7, 2, 15, 0.1)),
                     ("Value floor 20 ETH", (7, 2, 20, 0.1)), ("Share floor 0.2 V", (7, 2, 10, 0.2))]:
        specs.append((f"param:{name}", (lambda wv=wv: variant(*T.build_rows(*wv)[:2]))))
    specs.append(("param:Window 14 d", lambda: variant(w14_rows, w14_eids)))
    specs.append(("param:Depth 3 hops", lambda: variant(w3_rows, w3_eids)))
    for name, keep, flags, bd in [("− path-timing family", [f for f in NUM if f not in TIMING], True, True),
                                  ("Path-geometry only (3)", GEOM, False, False),
                                  ("− source-context family", [f for f in NUM if f not in SRC], True, True),
                                  ("− all observability flags", [f for f in NUM if f not in OC], False, True)]:
        specs.append((f"feat:{name}", (lambda keep=keep, flags=flags, bd=bd: variant(BASE, base_eids, keep, flags, bd))))

    # resume from the checkpoint; compute only what is missing (foreground-chunk safe:
    # a timeout kills at most the in-progress variant, everything before it is saved).
    ck = _load_ckpt()
    print(f"checkpoint: {len(ck)}/{len(specs)} variants already done", flush=True)
    t0 = time.time()
    for key, thunk in specs:
        if key in ck:
            continue
        a, lo, hi, nf, p = thunk()
        ck[key] = dict(auroc=a, lo=lo, hi=hi, p=p, folds=nf)
        _save_ckpt(ck)
        if key == "main" and N_PERM == 1000:
            assert abs(a - 0.703) < 1e-3 and abs(p - 27.0 / 1001.0) < 1e-12, (
                f"[gate] main observed {a} / p {p} != canonical 0.703 / 27/1001 -- engine inconsistent")
            print("  [gate] PASS: main p = 27/1001 = 0.026973 (matches permutation_test.py)", flush=True)
        print(f"  [{len(ck)}/{len(specs)}] {key}: {a}  p={p:.4f}  ({nf})  [+{time.time()-t0:.0f}s]", flush=True)

    if _POOL is not None:
        _POOL.close(); _POOL.join()

    missing = [k for k, _ in specs if k not in ck]
    if missing:
        print(f"still {len(missing)} variants to go: {missing[:3]}... re-run to continue.", flush=True)
        return

    # all done -> assemble the forest data JSON
    def row(label, key):
        return dict(label=label, **{k: ck[key][k] for k in ("auroc", "lo", "hi", "p", "folds")})

    none_row = row("None (all 17)", "main")
    loo = [row("− " + c, f"loo:{c}") for c in crews]
    params = [row("Main (7d/2hop/10 ETH)", "main"),
              row("Window 3 d", "param:Window 3 d"), row("Window 5 d", "param:Window 5 d"),
              row("Window 14 d", "param:Window 14 d"), row("Depth 1 hop", "param:Depth 1 hop"),
              row("Depth 3 hops", "param:Depth 3 hops"), row("Value floor 15 ETH", "param:Value floor 15 ETH"),
              row("Value floor 20 ETH", "param:Value floor 20 ETH"), row("Share floor 0.2 V", "param:Share floor 0.2 V")]
    feats = [row("Full Φ", "main"),
             row("− path-timing family", "feat:− path-timing family"),
             row("Path-geometry only (3)", "feat:Path-geometry only (3)"),
             row("− source-context family", "feat:− source-context family"),
             row("− all observability flags", "feat:− all observability flags")]

    main_p = ck["main"]["p"]
    if N_PERM == 1000:
        assert abs(main_p - 27.0 / 1001.0) < 1e-12, f"main p {main_p} != 27/1001"
    data = dict(none=none_row, loo=loo, params=params, feats=feats, main=ck["main"]["auroc"],
                meta=dict(sklearn="1.3.2", n_perm=N_PERM, seed=SEED,
                          null="group-count-preserving (crew blocks + background singletons)",
                          note="all p-values are canonical 1.3.2 values; main matches permutation_test.py (0.026973)"))
    out = os.path.join(HERE, "robustness_forest_data.json")
    json.dump(data, io.open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if os.path.exists(CKPT):
        os.remove(CKPT)          # clean the resume checkpoint on a full, successful run
    print(f"\nALL {len(specs)} DONE. wrote {os.path.basename(out)} (main p={main_p:.6f}, canonical)", flush=True)


if __name__ == "__main__":
    main()
