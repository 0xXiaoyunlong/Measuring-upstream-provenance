# -*- coding: utf-8 -*-
import csv, io, os, json, hashlib, datetime
import numpy as np
ROOT = r"C:\Users\XYL\Desktop\区块链论文\研究工程_结构化"
L20 = os.path.join(ROOT, r"02_当前研究\05_Phi数据扩展\26_L20_A18预注册评估执行\A0_merge")
L17 = os.path.join(ROOT, r"02_当前研究\05_Phi数据扩展\23_L17_Phi预检查与冻结提取\A0_merge")
F = os.path.join(ROOT, r"01_冻结档案\04_FACTORY_sidecar")
A17C = os.path.join(ROOT, r"02_当前研究\05_Phi数据扩展\04_Phi提取\A17_cone_ledger_v2.json")
OUTD = os.path.join(ROOT, r"02_当前研究\05_Phi数据扩展\27_L21_Sidecar增量评估\A0_merge"); os.makedirs(OUTD, exist_ok=True)

PHI15 = ['observed_hops', 'path_elapsed_time', 'dwell_mean_h', 'dwell_max_h', 'coverage', 'attribution_ambiguous',
         'node_context_out_degree', 'node_context_in_degree', 'wallet_age_min_h',
         'node_context_resid_Rrange_max', 'node_context_resid_nin_mean', 'node_context_resid_nout_mean',
         'boundary_type', 'censoring_reason', 'observation_confidence']
FAC = ['fac_funder_present', 'fac_hub_present', 'fac_funder_fanout', 'fac_hub_fanout',
       'fac_funder_out_unif', 'fac_funder_out_quant', 'fac_fund_quant', 'fac_n_fresh_siblings']
FAC_OUTDEP = {'fac_funder_fanout', 'fac_hub_fanout', 'fac_funder_out_unif', 'fac_funder_out_quant', 'fac_n_fresh_siblings'}
DENOM = {0.1, 1.0, 10.0, 100.0}

def is_round(a):
    return (round(a, 4) in DENOM) or (abs(a - round(a)) < 0.02)

# ---- load merged old cones ----
cones = {}
for fn in ['_real_ledgers', '_bean_ledgers', '_audmonk_ledgers']:
    d = json.load(io.open(os.path.join(F, fn + '.json'), encoding='utf-8'))
    for a, rec in d.items():
        cones[a.lower()] = (rec.get('cone', {}), 'full')
# Convergence
try:
    a17 = json.load(io.open(A17C, encoding='utf-8'))
    for a, rec in (a17.items() if isinstance(a17, dict) else []):
        c = rec.get('cone', rec) if isinstance(rec, dict) else {}
        if c:
            cones[a.lower()] = (c, 'full')
except Exception:
    pass
# new-12: in-only cone from l17_cone_edges
def iso2ep(s):
    return int(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp())
newc = {}
for r in csv.DictReader(io.open(os.path.join(L17, 'l17_cone_edges.csv'), encoding='utf-8')):
    dst = r['dst'].lower(); src = r['src'].lower()
    newc.setdefault(dst, {'boundary': False, 'in': [], 'out': [], 'init': 0.0})
    newc.setdefault(src, {'boundary': False, 'in': [], 'out': [], 'init': 0.0})
    newc[dst]['in'].append((src, float(r['value_eth']), iso2ep(r['timestamp_utc'])))

def sidecar(a0, cone, in_only):
    a0 = a0.lower(); out = {f: (None, 0) for f in FAC}  # (value, observed_flag)

    def setv(f, v):
        out[f] = (v, 1)
    def miss(f):
        out[f] = (None, 0)
    info = cone.get(a0)
    if info is None:
        return out  # all missing
    ins = [(frm, amt, ts) for frm, amt, ts in info.get('in', [])]
    setv('fac_funder_present', 1 if ins else 0)
    if not ins:
        setv('fac_hub_present', 0)
        for f in ['fac_funder_fanout', 'fac_hub_fanout', 'fac_funder_out_unif', 'fac_funder_out_quant', 'fac_fund_quant', 'fac_n_fresh_siblings']:
            # no funder -> structurally 0 for fanout/siblings IF cone fully observed; else missing
            if in_only:
                miss(f)
            else:
                setv(f, 0)
        return out
    funder, famt, fts = max(ins, key=lambda x: x[1])
    setv('fac_fund_quant', 1 if is_round(famt) else 0)
    finfo = cone.get(funder)
    # hub present = funder has in-edge
    fins = finfo.get('in', []) if (finfo and not finfo.get('boundary')) else []
    setv('fac_hub_present', 1 if fins else 0)
    if in_only:
        for f in FAC_OUTDEP:
            miss(f)  # out edges not persisted for new-12
        return out
    # full cone: out-dependent features
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
    # hub fanout
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

# ---- load L20 samples ----
s0 = list(csv.DictReader(io.open(os.path.join(L20, 'l20_step0_feature_table.csv'), encoding='utf-8')))
samples = []
for r in s0:
    addr = r['address'].lower()
    in_only = r['feature_source'].startswith('l17_cone')
    cone = newc if in_only else cones.get(addr, ({}, 'none'))[0]
    if in_only:
        sc = sidecar(addr, newc, True); status = 'IN_ONLY_OUT_MISSING'
    elif addr in cones:
        sc = sidecar(addr, cones[addr][0], False); status = 'FULL_CONE'
    else:
        sc = {f: (None, 0) for f in FAC}; status = 'CONE_NOT_FOUND'
    phi = {f: r.get(f, '') for f in PHI15}
    phimi = {f: r.get('mi_' + f, '0') for f in PHI15}
    samples.append(dict(sample_id=r['sample_id'], group_id=r['group_id'], event_id=r['event_id'], address=addr,
                        t0_utc=r['t0_utc'], is_positive=int(r['is_positive']), source_split=r['source_split'],
                        sidecar_source_status=status, sc=sc, phi=phi, phimi=phimi))

pos = sum(s['is_positive'] for s in samples); bg = len(samples) - pos
groups = sorted(set(s['group_id'] for s in samples if s['is_positive']))
print('samples', len(samples), 'pos', pos, 'bg', bg, 'groups', len(groups))
from collections import Counter
print('sidecar status:', dict(Counter(s['sidecar_source_status'] for s in samples)))

# ---- feature tables ----
def wd(f, rows, cols):
    with io.open(os.path.join(OUTD, f), 'w', encoding='utf-8', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction='ignore'); w.writeheader()
        for r in rows:
            w.writerow(r)

scols = ['sample_id', 'group_id', 'event_id', 'address', 't0_utc', 'is_positive', 'source_split', 'sidecar_source_status'] + FAC + ['mi_' + f for f in FAC]
srows = []
for s in samples:
    row = dict(sample_id=s['sample_id'], group_id=s['group_id'], event_id=s['event_id'], address=s['address'], t0_utc=s['t0_utc'],
               is_positive=s['is_positive'], source_split=s['source_split'], sidecar_source_status=s['sidecar_source_status'])
    for f in FAC:
        v, ob = s['sc'][f]; row[f] = ('' if v is None else v); row['mi_' + f] = (0 if ob else 1)
    srows.append(row)
wd('l21_sidecar_feature_table.csv', srows, scols)

ccols = ['sample_id', 'group_id', 'event_id', 'address', 't0_utc', 'is_positive', 'source_split', 'feature_block'] + PHI15 + ['mi_' + f for f in PHI15] + FAC + ['mi_' + f for f in FAC]
crows = []
for s in samples:
    row = dict(sample_id=s['sample_id'], group_id=s['group_id'], event_id=s['event_id'], address=s['address'], t0_utc=s['t0_utc'],
               is_positive=s['is_positive'], source_split=s['source_split'], feature_block='PHI_PLUS_SIDECAR')
    for f in PHI15:
        row[f] = s['phi'][f]; row['mi_' + f] = s['phimi'][f]
    for f in FAC:
        v, ob = s['sc'][f]; row[f] = ('' if v is None else v); row['mi_' + f] = (0 if ob else 1)
    crows.append(row)
wd('l21_combined_feature_table.csv', crows, ccols)

# ---- design matrices ----
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
BOUND = ['eoa-source', 'asset-conversion', 'contract', 'depth-limit', 'coverage-stop', 'unresolved', 'origin-reached']

def num(v):
    try:
        if v in ('', None):
            return np.nan
        if v in ('True', True):
            return 1.0
        if v in ('False', False):
            return 0.0
        return float(v)
    except Exception:
        return np.nan

def block_matrix(rows, train, block):
    feats_num = []
    if block in ('PHI_ONLY', 'PHI_PLUS_SIDECAR'):
        feats_num += [f for f in PHI15 if f not in ('boundary_type', 'censoring_reason')]
    if block in ('SIDECAR_ONLY', 'PHI_PLUS_SIDECAR'):
        feats_num += FAC
    med = {}
    for f in feats_num:
        vs = [num(s['phi'].get(f) if f in PHI15 else s['sc'][f][0]) for s in train]
        vs = [v for v in vs if not np.isnan(v)]
        med[f] = float(np.median(vs)) if vs else 0.0
    X = []
    for s in rows:
        v = []
        for f in feats_num:
            raw = num(s['phi'].get(f) if f in PHI15 else s['sc'][f][0])
            v.append(med[f] if np.isnan(raw) else raw); v.append(0.0 if np.isnan(raw) else 1.0)
        if block in ('PHI_ONLY', 'PHI_PLUS_SIDECAR'):
            b = str(s['phi'].get('boundary_type') or '')
            v += [1.0 if b == c else 0.0 for c in BOUND]
        X.append(v)
    return np.array(X, dtype=float)

y = np.array([s['is_positive'] for s in samples])
idx = list(range(len(samples)))
posidx = [i for i in idx if samples[i]['is_positive']]
bgidx = [i for i in idx if not samples[i]['is_positive']]
K = len(groups)
def bgfold(s, k):
    return int(hashlib.sha256(s['address'].encode()).hexdigest(), 16) % k
# LOAGO folds (identical to L20)
loago = []
for gi, g in enumerate(groups):
    te = [i for i in posidx if samples[i]['group_id'] == g] + [i for i in bgidx if bgfold(samples[i], K) == gi]
    loago.append((gi, g, te, [i for i in idx if i not in te]))
from sklearn.model_selection import GroupKFold
allg = [samples[i]['group_id'] if samples[i]['is_positive'] else 'BG_%d' % bgfold(samples[i], 5) for i in idx]
gkf = [(fid, '', list(te), list(tr)) for fid, (tr, te) in enumerate(GroupKFold(5).split(idx, y, groups=allg))]

BLOCKS = ['PHI_ONLY', 'SIDECAR_ONLY', 'PHI_PLUS_SIDECAR']
MODELS = ['logreg', 'xgboost', 'baseline_sidecar_rank']
fold_metrics = []
per = {}  # (split,block,model,metric)->[fold vals]

def evaluate(split_name, folds):
    for fid, hg, te, tr in folds:
        yte = y[te]; ytr = y[tr]
        for block in BLOCKS:
            for mid in MODELS:
                key = (split_name, block, mid)
                if mid == 'baseline_sidecar_rank' and block != 'SIDECAR_ONLY':
                    continue
                if mid != 'baseline_sidecar_rank' and False:
                    pass
                if yte.sum() == 0 or yte.sum() == len(yte):
                    fold_metrics.append(dict(split_name=split_name, fold_id=fid, feature_block=block, model_id=mid, metric_name='auroc', value='', status='NOT_ESTIMABLE_FOR_THIS_FOLD', notes='single-class test')); continue
                try:
                    if mid == 'baseline_sidecar_rank':
                        def bscore(s):
                            comp = []
                            for f in ['fac_hub_fanout', 'fac_funder_fanout', 'fac_n_fresh_siblings']:
                                v = s['sc'][f][0]
                                comp.append(np.nan if v is None else float(v))
                            return np.nanmean(comp) if not all(np.isnan(c) for c in comp) else np.nan
                        raw = np.array([bscore(samples[i]) for i in te])
                        if np.all(np.isnan(raw)):
                            fold_metrics.append(dict(split_name=split_name, fold_id=fid, feature_block=block, model_id=mid, metric_name='auroc', value='', status='NOT_ESTIMABLE_FOR_THIS_FOLD', notes='all sidecar core missing')); continue
                        sco = np.where(np.isnan(raw), np.nanmin(raw) - 1, raw)
                    else:
                        Xtr = block_matrix([samples[i] for i in tr], [samples[i] for i in tr], block)
                        Xte = block_matrix([samples[i] for i in te], [samples[i] for i in tr], block)
                        if mid == 'logreg':
                            scl = StandardScaler().fit(Xtr)
                            m = LogisticRegression(max_iter=2000, class_weight='balanced').fit(scl.transform(Xtr), ytr)
                            sco = m.predict_proba(scl.transform(Xte))[:, 1]
                        else:
                            m = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss', verbosity=0)
                            m.fit(Xtr, ytr); sco = m.predict_proba(Xte)[:, 1]
                    for met, fn in [('auroc', roc_auc_score), ('auprc', average_precision_score)]:
                        val = fn(yte, sco)
                        fold_metrics.append(dict(split_name=split_name, fold_id=fid, feature_block=block, model_id=mid, metric_name=met, value=round(val, 4), status='ESTIMABLE', notes=''))
                        per.setdefault((split_name, block, mid, met), []).append((fid, val))
                except Exception as e:
                    fold_metrics.append(dict(split_name=split_name, fold_id=fid, feature_block=block, model_id=mid, metric_name='auroc', value='', status='MODEL_ERROR', notes=str(e)[:40]))

evaluate('LOAGO_leave_one_actor_group_out', loago)
evaluate('GroupKFold_by_group_id', gkf)
wd('l21_fold_metrics.csv', fold_metrics, ['split_name', 'fold_id', 'feature_block', 'model_id', 'metric_name', 'value', 'status', 'notes'])

rng = np.random.RandomState(0)
agg = []
aggmap = {}
for (split, block, mid, met), vals in per.items():
    v = [x[1] for x in vals]
    mean = float(np.mean(v)); boots = [float(np.mean(rng.choice(v, len(v), replace=True))) for _ in range(2000)]
    agg.append(dict(split_name=split, feature_block=block, model_id=mid, metric_name=met, value=round(mean, 4),
                    ci_low=round(float(np.percentile(boots, 2.5)), 4), ci_high=round(float(np.percentile(boots, 97.5)), 4),
                    n_estimable_folds=len(v), status='ESTIMABLE', notes=''))
    aggmap[(split, block, mid, met)] = dict(vals)
wd('l21_aggregate_metrics.csv', agg, ['split_name', 'feature_block', 'model_id', 'metric_name', 'value', 'ci_low', 'ci_high', 'n_estimable_folds', 'status', 'notes'])

# incremental: paired per-fold deltas
inc = []
for split in ['LOAGO_leave_one_actor_group_out', 'GroupKFold_by_group_id']:
    for mid in ['logreg', 'xgboost']:
        for met in ['auroc', 'auprc']:
            base = aggmap.get((split, 'PHI_ONLY', mid, met), {})
            for cmpb, label in [('SIDECAR_ONLY', 'SIDECAR_ONLY_minus_PHI_ONLY'), ('PHI_PLUS_SIDECAR', 'PHI_PLUS_SIDECAR_minus_PHI_ONLY')]:
                other = aggmap.get((split, cmpb, mid, met), {})
                common = sorted(set(base) & set(other))
                if not common:
                    inc.append(dict(split_name=split, model_id=mid, metric_name=met, comparison=label, value_delta='', ci_low='', ci_high='', status='NOT_ESTIMABLE', interpretation_guardrail='no common estimable folds')); continue
                deltas = [other[f] - base[f] for f in common]
                md = float(np.mean(deltas)); boots = [float(np.mean(rng.choice(deltas, len(deltas), replace=True))) for _ in range(2000)]
                inc.append(dict(split_name=split, model_id=mid, metric_name=met, comparison=label, value_delta=round(md, 4),
                                ci_low=round(float(np.percentile(boots, 2.5)), 4), ci_high=round(float(np.percentile(boots, 97.5)), 4),
                                status='ESTIMABLE', interpretation_guardrail='descriptive paired delta on common folds; NOT comparable to historical within-Harmony 0.990'))
wd('l21_incremental_comparison.csv', inc, ['split_name', 'model_id', 'metric_name', 'comparison', 'value_delta', 'ci_low', 'ci_high', 'status', 'interpretation_guardrail'])
wd('l21_model_specs.csv', [
 dict(model_id='logreg', feature_block='all', preprocessing='train-median impute + observed-indicator + StandardScaler', parameters='max_iter=2000,class_weight=balanced', tuning_allowed='no', notes='fixed'),
 dict(model_id='xgboost', feature_block='all', preprocessing='train-median impute + indicator', parameters='n_estimators=100,max_depth=3,lr=0.1', tuning_allowed='no', notes='fixed'),
 dict(model_id='baseline_sidecar_rank', feature_block='SIDECAR_ONLY', preprocessing='none', parameters='mean(z? no) of hub_fanout+funder_fanout+n_fresh_siblings; missing->lowest', tuning_allowed='no', notes='non-learning; missing core -> NOT_ESTIMABLE'),
], ['model_id', 'feature_block', 'preprocessing', 'parameters', 'tuning_allowed', 'notes'])

phi_h = hashlib.sha256(open(os.path.join(ROOT, r"01_冻结档案\01_Phi初期实验\phi.py"), 'rb').read()).hexdigest()
pipe_h = hashlib.sha256(open(os.path.join(ROOT, r"01_冻结档案\01_Phi初期实验\pipeline.py"), 'rb').read()).hexdigest()
def g(split, block, mid, met):
    a = [x for x in agg if x['split_name'] == split and x['feature_block'] == block and x['model_id'] == mid and x['metric_name'] == met]
    return a[0] if a else None
po = g('LOAGO_leave_one_actor_group_out', 'PHI_ONLY', 'logreg', 'auroc')
so = g('LOAGO_leave_one_actor_group_out', 'SIDECAR_ONLY', 'logreg', 'auroc')
ps = g('LOAGO_leave_one_actor_group_out', 'PHI_PLUS_SIDECAR', 'logreg', 'auroc')
newmiss = sum(1 for s in samples if s['sidecar_source_status'] == 'IN_ONLY_OUT_MISSING')
rs = "LOAGO logreg AUROC: PHI=%s SIDECAR=%s PHI+SIDECAR=%s (new-12 out-dep sidecar MISSING=%d)" % (po and po['value'], so and so['value'], ps and ps['value'], newmiss)
status = 'L21_PARTIAL_EXECUTED_WITH_BLOCKED_SIDECAR' if newmiss > 0 else 'L21_EXECUTED'
json.dump(dict(round='L21', status=status, input_files_sha256={'l20_step0': 'f69a210b', '_real_ledgers': '666d0b81'},
               n_samples=len(samples), n_positive=int(pos), n_unknown_background=int(bg), n_groups=len(groups),
               feature_blocks_run=BLOCKS, models_run=MODELS, splits_run=['LOAGO_leave_one_actor_group_out', 'GroupKFold_by_group_id'],
               sidecar_missing_policy='out-edge-dependent fac features MISSING(mi=1) for new-12 (l17 saved in-edges only); never filled 0',
               auc_used=True, training_used=True, attribution_interval_used=False, main_manuscript_changed=False,
               frozen_hashes_match=(phi_h.startswith('18c2db') and pipe_h.startswith('773fe3c3')), result_summary=rs, next_gate='L21_SIDECAR_INCREMENT_REVIEW_GATE'),
          io.open(os.path.join(OUTD, 'l21_execution_manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('STATUS:', status)
print(rs)
for x in agg:
    if x['metric_name'] == 'auroc' and x['split_name'].startswith('LOAGO'):
        print('  LOAGO %-18s %-22s AUROC=%s CI[%s,%s] folds=%s' % (x['feature_block'], x['model_id'], x['value'], x['ci_low'], x['ci_high'], x['n_estimable_folds']))
print('--- increments (LOAGO logreg auroc) ---')
for x in inc:
    if x['split_name'].startswith('LOAGO') and x['model_id'] == 'logreg' and x['metric_name'] == 'auroc':
        print('  ', x['comparison'], 'delta=', x['value_delta'], 'CI[%s,%s]' % (x['ci_low'], x['ci_high']))
