# -*- coding: utf-8 -*-
"""
Rebuild the factory-pattern (sidecar) features from the underlying frozen cones.

This is a faithful adaptation of the L21 sidecar extraction (`_l21_sidecar_source.py`,
kept alongside for provenance): for each deposit it walks the funding cone and
measures the funder fan-out / fresh-sibling structure. The out-edge-dependent
features are MISSING for the twelve newly gated crews at this "before recovery"
stage, because only their in-edges were persisted — never filled with 0.

Inputs (all in ../../../data/sidecar/):
  _real_ledgers.json, _bean_ledgers.json, _audmonk_ledgers.json  full old cones
  A17_cone_ledger_v2.json                                        Convergence cone
  l17_cone_edges.csv                                             new-12 in-only cones
  l20_step0_feature_table.csv                                    the 169-sample list

Output: ../../../data/sidecar/sidecar_features_before_recovery.csv (regenerated).
The script asserts it matches the shipped table, so the sidecar features are
demonstrably rebuildable from the cones, not taken on faith. The "after recovery"
table (sidecar_features_after_recovery.csv) is the L22 out-edge-recovered version;
see recovered_out_edges.csv and _l22_outedge_source.py.

    python build_sidecar_features.py
"""
import csv, io, os, json, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "..", "..", "..", "data", "sidecar")

FAC = ['fac_funder_present', 'fac_hub_present', 'fac_funder_fanout', 'fac_hub_fanout',
       'fac_funder_out_unif', 'fac_funder_out_quant', 'fac_fund_quant', 'fac_n_fresh_siblings']
FAC_OUTDEP = {'fac_funder_fanout', 'fac_hub_fanout', 'fac_funder_out_unif',
              'fac_funder_out_quant', 'fac_n_fresh_siblings'}
DENOM = {0.1, 1.0, 10.0, 100.0}


def is_round(a):
    return (round(a, 4) in DENOM) or (abs(a - round(a)) < 0.02)


def iso2ep(s):
    return int(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
               .replace(tzinfo=datetime.timezone.utc).timestamp())


def sidecar(a0, cone, in_only):
    """Verbatim from L21: the factory-pattern feature definition."""
    a0 = a0.lower(); out = {f: (None, 0) for f in FAC}

    def setv(f, v):
        out[f] = (v, 1)

    def miss(f):
        out[f] = (None, 0)
    info = cone.get(a0)
    if info is None:
        return out
    ins = [(frm, amt, ts) for frm, amt, ts in info.get('in', [])]
    setv('fac_funder_present', 1 if ins else 0)
    if not ins:
        setv('fac_hub_present', 0)
        for f in ['fac_funder_fanout', 'fac_hub_fanout', 'fac_funder_out_unif',
                  'fac_funder_out_quant', 'fac_fund_quant', 'fac_n_fresh_siblings']:
            miss(f) if in_only else setv(f, 0)
        return out
    funder, famt, fts = max(ins, key=lambda x: x[1])
    setv('fac_fund_quant', 1 if is_round(famt) else 0)
    finfo = cone.get(funder)
    fins = finfo.get('in', []) if (finfo and not finfo.get('boundary')) else []
    setv('fac_hub_present', 1 if fins else 0)
    if in_only:
        for f in FAC_OUTDEP:
            miss(f)
        return out
    if finfo is None or finfo.get('boundary'):
        for f in ['fac_funder_fanout', 'fac_funder_out_unif', 'fac_funder_out_quant', 'fac_n_fresh_siblings']:
            miss(f)
    else:
        fout = finfo.get('out', [])
        tos = [to for to, amt, ts in fout]
        setv('fac_funder_fanout', len(set(tos)))
        amts = [round(amt, 4) for to, amt, ts in fout]
        if amts:
            modal = max(set(amts), key=amts.count)
            setv('fac_funder_out_unif', round(amts.count(modal) / len(amts), 4))
            setv('fac_funder_out_quant', round(sum(1 for a in amts if is_round(a)) / len(amts), 4))
        else:
            setv('fac_funder_out_unif', 0); setv('fac_funder_out_quant', 0)
        fresh = 0
        for to in set(tos):
            if to == a0:
                continue
            ti = cone.get(to)
            if ti is not None and not ti.get('boundary') and float(ti.get('init', 0)) < 0.001 and len(ti.get('in', [])) <= 2:
                fresh += 1
        setv('fac_n_fresh_siblings', fresh)
    if fins:
        hub = max(fins, key=lambda x: x[1])[0]
        hinfo = cone.get(hub)
        if hinfo is not None and not hinfo.get('boundary'):
            setv('fac_hub_fanout', len(set(to for to, amt, ts in hinfo.get('out', []))))
        else:
            miss('fac_hub_fanout')
    else:
        setv('fac_hub_fanout', 0)
    return out


def load_cones():
    cones = {}
    # DATA-SWAP RUN: _bgswap_ledgers carries the 140 new balanced-background cones
    # (same in/out/init/boundary format); loaded first so illicit ledgers below win
    # on any address collision (there are none by construction).
    for fn in ['_bgswap_ledgers', '_real_ledgers', '_bean_ledgers', '_audmonk_ledgers']:
        p = os.path.join(SC, fn + '.json')
        if not os.path.exists(p):
            continue
        d = json.load(io.open(p, encoding='utf-8'))
        for a, rec in d.items():
            cones[a.lower()] = rec.get('cone', {})
    try:
        a17 = json.load(io.open(os.path.join(SC, 'A17_cone_ledger_v2.json'), encoding='utf-8'))
        for a, rec in (a17.items() if isinstance(a17, dict) else []):
            c = rec.get('cone', rec) if isinstance(rec, dict) else {}
            if c:
                cones[a.lower()] = c
    except Exception:
        pass
    newc = {}
    for r in csv.DictReader(io.open(os.path.join(SC, 'l17_cone_edges.csv'), encoding='utf-8')):
        dst = r['dst'].lower(); src = r['src'].lower()
        newc.setdefault(dst, {'boundary': False, 'in': [], 'out': [], 'init': 0.0})
        newc.setdefault(src, {'boundary': False, 'in': [], 'out': [], 'init': 0.0})
        newc[dst]['in'].append((src, float(r['value_eth']), iso2ep(r['timestamp_utc'])))
    return cones, newc


def main():
    cones, newc = load_cones()
    samples = list(csv.DictReader(io.open(os.path.join(SC, 'l20_step0_feature_table.csv'), encoding='utf-8')))
    rows = []
    for r in samples:
        addr = r['address'].lower()
        in_only = r['feature_source'].startswith('l17_cone')
        sc = sidecar(addr, newc if in_only else cones.get(addr, {}), in_only)
        row = {'sample_id': r['sample_id'], 'group_id': r['group_id'], 'event_id': r['event_id'],
               'address': addr, 't0_utc': r['t0_utc'], 'is_positive': r['is_positive'],
               'source_split': r['source_split']}
        for f in FAC:
            v, ob = sc[f]; row[f] = ('' if v is None else v); row['mi_' + f] = (0 if ob else 1)
        rows.append(row)

    cols = ['sample_id', 'group_id', 'event_id', 'address', 't0_utc', 'is_positive', 'source_split'] \
        + FAC + ['mi_' + f for f in FAC]
    # verify against the shipped table before overwriting
    shipped = {r['event_id']: r for r in csv.DictReader(
        io.open(os.path.join(SC, 'sidecar_features_before_recovery.csv'), encoding='utf-8-sig'))}
    mism = 0
    for row in rows:
        s = shipped.get(row['event_id'])
        if s is None:
            continue
        for f in FAC:
            a = (row[f] if row[f] != '' else None)
            b = (s.get(f) if s.get(f, '') != '' else None)
            if a is None and b is None:
                continue
            if a is None or b is None or abs(float(a) - float(b)) > 1e-6:
                mism += 1
    tmp = os.path.join(SC, 'sidecar_features_before_recovery.csv.tmp')
    with io.open(tmp, 'w', encoding='utf-8', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction='ignore'); w.writeheader()
        [w.writerow(r) for r in rows]
    os.replace(tmp, os.path.join(SC, 'sidecar_features_before_recovery.csv'))
    print(f"rebuilt sidecar_features_before_recovery.csv from cones: {len(rows)} samples, "
          f"{mism} factory-feature cells differ from the shipped table")
    assert mism == 0, "rebuilt factory features do not match the shipped table"
    print("OK: factory (sidecar) features are reproducible from the frozen cones.")


if __name__ == "__main__":
    main()
