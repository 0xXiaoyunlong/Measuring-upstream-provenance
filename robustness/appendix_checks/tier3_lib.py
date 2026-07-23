# -*- coding: utf-8 -*-
"""Tier-3 robustness shared library (in-package, relative paths only).

Reads the frozen cone set robustness/inputs/tier3_assembled_cones.json and provides:
- build_rows(W_days, depth, vthresh_eth, mvs): faithful cones are re-filtered by
  (window / depth / value-floor / min_value_share) and Phi is recomputed exactly;
  non-faithful cones are kept from the frozen feature table only when the requested
  perturbation provably removes/changes no edge (else the event is dropped).
- macro(rows) / macro_folds(rows): macro-averaged AUROC + the set of estimable folds.

All paths are derived from this file's location; there are NO absolute paths. The
underlying inputs are robustness/inputs/tier3_assembled_cones.json plus the package's
frozen data/ and archive/frozen_extraction/phi.py. This module is provenance for the
recompute in ../run_robustness.py; run_all.py does not import it directly.

NOTE ON THE OLD SELF-CHECK CONSTANT. An earlier version gated import on
build_rows(7,2,10,0.1) == 0.6980, a cross-version (scikit-learn 1.9.0) value. That
constant is stale and is NOT used to gate anything here -- the canonical baseline in
the reference environment (scikit-learn 1.3.2) is 0.7029, and the authoritative
anchor lives in ../run_robustness.py. _selfcheck() below only prints, under __main__.
"""
import sys, os, io, json, csv, statistics

HERE = os.path.dirname(os.path.abspath(__file__))        # robustness/appendix_checks
ROB = os.path.dirname(HERE)                              # robustness
PKG = os.path.dirname(ROB)                               # package root
DATA = os.path.join(PKG, "data")
FROZEN = os.path.join(PKG, "archive", "frozen_extraction")
EVALDIR = os.path.join(PKG, "code", "02_evaluate_cross_actor")
INPUTS = os.path.join(ROB, "inputs")
OUT = ROB                                                # recompute outputs land in robustness/

sys.path.insert(0, FROZEN); sys.path.insert(0, EVALDIR)
import phi as PHI
import importlib.util
spec = importlib.util.spec_from_file_location("ev", os.path.join(EVALDIR, "evaluate.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
FEATS = PHI.PHI_FEATURES

def rc(p): return list(csv.DictReader(io.open(p, encoding="utf-8-sig")))
ent = rc(os.path.join(DATA, "entrance_events.csv"))
uf = {r["event_id"]: r for r in rc(os.path.join(DATA, "upstream_features.csv"))}
REC = json.load(io.open(os.path.join(INPUTS, "tier3_assembled_cones.json"), encoding="utf-8"))
ADDR2REC = {v["addr"]: v for v in REC.values()}
EBID = {r["event_id"]: r for r in ent}

def approx(a, b, t=1e-3):
    if a in ("", None) and b is None: return True
    if a in ("", None) or b is None: return False
    try: return abs(float(a) - float(b)) <= t
    except: return str(a).lower() == str(b).lower()

def filter_cone(cone, anchor, vthresh):
    """Filter in/out edges to |v|>=vthresh (ETH), then prune nodes unreachable from
    the anchor along in-edges. vthresh<=10 returns the original cone unchanged."""
    if vthresh <= 10: return cone
    f = {}
    for nd, c in cone.items():
        f[nd] = {"boundary": c.get("boundary", False),
                 "in": [t for t in c.get("in", []) if abs(t[1]) >= vthresh],
                 "out": [t for t in c.get("out", []) if abs(t[1]) >= vthresh],
                 "init": c.get("init", 0.0)}
    keep = set([anchor]); frontier = [anchor]
    while frontier:
        x = frontier.pop(0)
        for (frm, v, ts) in f.get(x, {}).get("in", []):
            if frm in f and frm not in keep: keep.add(frm); frontier.append(frm)
    return {nd: c for nd, c in f.items() if nd in keep}

def removed_any_edge(cone, vthresh):
    if vthresh <= 10: return False
    for nd, c in cone.items():
        for t in c.get("in", []) + c.get("out", []):
            if abs(t[1]) < vthresh: return True
    return False

def phi_row(rec, W_days, depth, vthresh, mvs):
    W = W_days * 24 * 3600; P = dict(PHI.PARAMS)
    P["W_seconds"] = W; P["max_depth"] = depth; P["min_value_share"] = mvs
    cone = filter_cone(rec["cone"], rec["addr"], vthresh)
    ph = PHI.compute_phi(rec["addr"], rec["ts"], rec["V"], cone, P)
    er = EBID[rec["eid"]]
    row = {"crew": er["crew"], "is_illicit": er["is_illicit"], "held_out_fold": er["held_out_fold"]}
    for ft in FEATS:
        v = ph[ft]["value"]; row[ft] = "" if (not ph[ft]["is_observed"] or v is None) else str(v)
    return row

def frozen_row(eid):
    er = EBID[eid]; fr = uf[eid]
    row = {"crew": er["crew"], "is_illicit": er["is_illicit"], "held_out_fold": er["held_out_fold"]}
    for ft in FEATS: row[ft] = fr.get(ft, "")
    return row

def build_rows(W_days, depth, vthresh=10, mvs=0.1, exclude=None):
    """Return (rows, eids, dropped). Iterate the 169 events (cone looked up by
    address); rows[i] corresponds to eids[i]. `exclude` forces extra event drops
    (for paired-fold comparisons -- the baseline drops the same events as a variant)."""
    exclude = exclude or set()
    rows = []; eids = []; dropped = []
    for er in ent:
        eid = er["event_id"]; a = er["deposit_address"].lower(); rec = ADDR2REC.get(a)
        if eid in exclude: dropped.append(eid); continue
        if rec is None: dropped.append(eid); continue
        rec = dict(rec); rec["eid"] = eid
        if rec["faithful"]:
            rows.append(phi_row(rec, W_days, depth, vthresh, mvs)); eids.append(eid)
        else:
            t0 = rec["ts"]; ets = rec["edge_ts"]; W = W_days * 24 * 3600
            win_inv = all(ts >= t0 - W for ts in ets)
            val_inv = not removed_any_edge(rec["cone"], vthresh)
            dep_inv = (depth >= 2)
            mvs_inv = (mvs == 0.1)
            if win_inv and val_inv and dep_inv and mvs_inv:
                rows.append(frozen_row(eid)); eids.append(eid)
            else:
                dropped.append(eid)
    return rows, eids, dropped

def macro_folds(rows):
    aur = {}
    for fold, test, train in ev.leave_one_crew_out(rows):
        if not any(int(x["is_illicit"]) for x in test): continue
        au, _ap, _sc = ev.crew_auroc(test, train)      # package evaluate returns (auroc, auprc, scores)
        aur[fold] = au
    return aur

def macro(rows):
    a = macro_folds(rows); return (statistics.mean(a.values()) if a else 0.0), len(a)

# ---- informational self-check (prints only; does NOT gate import) ----
def _selfcheck():
    r, e, d = build_rows(7, 2, 10, 0.1); m, nf = macro(r)
    print(f"[tier3_lib] baseline 7d/2hop/10ETH = {m:.4f} ({nf} folds, dropped {len(d)}); "
          f"reference (sklearn 1.3.2) canonical baseline is 0.7029 -- authoritative anchor in run_robustness.py")
    return m, nf, len(d)
if __name__ == "__main__": _selfcheck()
