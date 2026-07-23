# -*- coding: utf-8 -*-
import csv, io, os, json, hashlib, datetime, urllib.request, time
import numpy as np
ROOT = r"C:\Users\XYL\Desktop\区块链论文\研究工程_结构化"
L21 = os.path.join(ROOT, r"02_当前研究\05_Phi数据扩展\27_L21_Sidecar增量评估\A0_merge")
L17 = os.path.join(ROOT, r"02_当前研究\05_Phi数据扩展\23_L17_Phi预检查与冻结提取\A0_merge")
L20 = os.path.join(ROOT, r"02_当前研究\05_Phi数据扩展\26_L20_A18预注册评估执行\A0_merge")
OUTD = os.path.join(ROOT, r"02_当前研究\05_Phi数据扩展\28_L22_新12组Out边补全与Sidecar重测\A0_merge"); os.makedirs(OUTD, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}; W = 7 * 24 * 3600; THRESH = 10.0; DENOM = {0.1, 1.0, 10.0, 100.0}
PHI15 = ['observed_hops', 'path_elapsed_time', 'dwell_mean_h', 'dwell_max_h', 'coverage', 'attribution_ambiguous', 'node_context_out_degree', 'node_context_in_degree', 'wallet_age_min_h', 'node_context_resid_Rrange_max', 'node_context_resid_nin_mean', 'node_context_resid_nout_mean', 'boundary_type', 'censoring_reason', 'observation_confidence']
FAC = ['fac_funder_present', 'fac_hub_present', 'fac_funder_fanout', 'fac_hub_fanout', 'fac_funder_out_unif', 'fac_funder_out_quant', 'fac_fund_quant', 'fac_n_fresh_siblings']
def is_round(a): return (round(a, 4) in DENOM) or (abs(a - round(a)) < 0.02)
def iso2ep(s): return int(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp())
src_log = []
def jget(url, sid='', node='', rtype='', tries=4):
    for i in range(tries):
        try:
            d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=35).read().decode('utf-8', 'ignore'))
            src_log.append(dict(sample_id=sid, node_address=node, source='blockscout', request_type=rtype, page=1, status_code=200, ok=True, error='', n_items=len(d.get('result') or []) if isinstance(d.get('result'), list) else 0, timestamp_utc='UNKNOWN_NOT_RECORDED'))
            return d
        except Exception as e:
            if i == tries - 1:
                src_log.append(dict(sample_id=sid, node_address=node, source='blockscout', request_type=rtype, page=1, status_code=0, ok=False, error=str(e)[:50], n_items=0, timestamp_utc='UNKNOWN_NOT_RECORDED')); return None
            time.sleep(0.7)

def out_edges(node, t0, dep_block, sid):
    sb = max(0, dep_block - 50400); outs = []; ok = False
    for act in ['txlist', 'txlistinternal']:
        j = jget("https://eth.blockscout.com/api?module=account&action=%s&address=%s&startblock=%d&endblock=%d&sort=asc" % (act, node, sb, dep_block), sid, node, act)
        if j is None: return outs, False
        res = j.get('result')
        if isinstance(res, list):
            ok = True
            for t in res:
                if not isinstance(t, dict): continue
                try: v = int(t.get('value', '0')) / 1e18; ts = int(t.get('timeStamp', '0'))
                except: continue
                if v < THRESH or ts >= t0 or ts < t0 - W: continue
                if (t.get('from') or '').lower() == node:
                    outs.append((( t.get('to') or '').lower(), v, ts, t.get('hash', ''), int(t.get('blockNumber', '0')), act))
    return outs, ok
def bal(node, blk, sid):
    j = jget("https://eth.blockscout.com/api?module=account&action=balance&address=%s&block=%d" % (node, blk), sid, node, 'balance')
    try: return int(j.get('result', '0')) / 1e18
    except: return None

# targets
targets = [r for r in csv.DictReader(io.open(os.path.join(L21, 'l21_sidecar_feature_table.csv'), encoding='utf-8')) if r['is_positive'] == '1' and r['sidecar_source_status'] == 'IN_ONLY_OUT_MISSING']
assert len(targets) == 12, "L22_BLOCKED_TARGET_MISMATCH"
pre = {r['candidate_id']: r for r in csv.DictReader(io.open(os.path.join(L17, 'l17_phi_precheck.csv'), encoding='utf-8'))}
edges = {}
for r in csv.DictReader(io.open(os.path.join(L17, 'l17_cone_edges.csv'), encoding='utf-8')):
    edges.setdefault(r['candidate_id'], []).append((r['src'].lower(), r['dst'].lower(), float(r['value_eth']), iso2ep(r['timestamp_utc'])))

def sidecar_full(a0, in_cone, recovered_out, status_partial):
    # in_cone: {addr:{'in':[(frm,amt,ts)]}}; recovered_out: {node:[(to,amt,ts)]}; bals: dict
    a0 = a0.lower(); o = {f: (None, 0) for f in FAC}
    def S(f, v): o[f] = (v, 1)
    ins = in_cone.get(a0, {}).get('in', [])
    S('fac_funder_present', 1 if ins else 0)
    if not ins:
        S('fac_hub_present', 0)
        for f in ['fac_funder_fanout', 'fac_hub_fanout', 'fac_funder_out_unif', 'fac_funder_out_quant', 'fac_fund_quant', 'fac_n_fresh_siblings']: S(f, 0)
        return o  # eoa-source: real 0, OUT_NOT_NEEDED
    funder, famt, _ = max(ins, key=lambda x: x[1])
    S('fac_fund_quant', 1 if is_round(famt) else 0)
    fins = in_cone.get(funder, {}).get('in', [])
    S('fac_hub_present', 1 if fins else 0)
    fout = recovered_out.get(funder)
    if fout is None:  # funder out not recovered -> MISSING
        for f in ['fac_funder_fanout', 'fac_funder_out_unif', 'fac_funder_out_quant', 'fac_n_fresh_siblings']: o[f] = (None, 0)
    else:
        tos = [to for to, amt, ts, *_ in fout]; amts = [round(amt, 4) for to, amt, ts, *_ in fout]
        S('fac_funder_fanout', len(set(tos)))
        if amts:
            modal = max(set(amts), key=amts.count); S('fac_funder_out_unif', round(amts.count(modal) / len(amts), 4)); S('fac_funder_out_quant', round(sum(1 for a in amts if is_round(a)) / len(amts), 4))
        else: S('fac_funder_out_unif', 0); S('fac_funder_out_quant', 0)
        S('fac_n_fresh_siblings', len(set(t for t in tos if t != a0)))  # distinct non-a0 recipients (freshness probe optional)
    hout = recovered_out.get((fins and max(fins, key=lambda x: x[1])[0]) or None)
    if fins:
        if hout is None: o['fac_hub_fanout'] = (None, 0)
        else: S('fac_hub_fanout', len(set(to for to, amt, ts, *_ in hout)))
    else: S('fac_hub_fanout', 0)
    return o

rec_status = []; rec_edges = []; new_sc = {}
for r in targets:
    sid = r['sample_id']; cid = sid.replace('NEW-', ''); a0 = r['address'].lower()
    p = pre[cid]; t0 = iso2ep(p['t0_utc']); dep_block = int(p['block_number'])
    in_cone = {}
    for s, d, v, ts in edges.get(cid, []):
        in_cone.setdefault(d, {'in': []}); in_cone.setdefault(s, {'in': []}); in_cone[d]['in'].append((s, v, ts))
    ins = in_cone.get(a0, {}).get('in', [])
    nodes_to_query = []
    if ins:
        funder = max(ins, key=lambda x: x[1])[0]; nodes_to_query.append(funder)
        fins = in_cone.get(funder, {}).get('in', [])
        if fins: nodes_to_query.append(max(fins, key=lambda x: x[1])[0])
    recovered = {}; nblock = 0; nsucc = 0; nraw = 0; nstruct = 0
    for node in nodes_to_query[:25]:
        oe, ok = out_edges(node, t0, dep_block, sid)
        if ok: nsucc += 1; recovered[node] = oe; nraw += len(oe); nstruct += len(oe)
        else: nblock += 1
    if not nodes_to_query:
        status = 'OUT_EDGE_NOT_NEEDED'  # eoa-source, no funder
    elif nblock == 0:
        status = 'OUT_EDGE_RECOVERED_FULL'
    elif nsucc > 0:
        status = 'OUT_EDGE_RECOVERED_PARTIAL'
    else:
        status = 'OUT_EDGE_BLOCKED_SOURCE_ACCESS'
    sc = sidecar_full(a0, in_cone, recovered, status)
    new_sc[sid] = (sc, status)
    rec_status.append(dict(sample_id=sid, group_id=r['group_id'], event_id=r['event_id'], address=a0,
                           n_cone_nodes=len(in_cone), n_nodes_attempted=len(nodes_to_query), n_nodes_success=nsucc,
                           n_nodes_blocked=nblock, n_raw_edges=nraw, n_structured_edges=nstruct, status=status, notes=''))
    for node, oe in recovered.items():
        for to, v, ts, h, bn, et in oe:
            rec_edges.append(dict(sample_id=sid, group_id=r['group_id'], event_id=r['event_id'], node_address=node, edge_from=node, edge_to=to, value_eth=round(v, 4), timestamp_utc=datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%SZ'), block_number=bn, tx_hash=h, edge_type=et, source='blockscout', source_status='ok'))
    time.sleep(0.15)

def wd(f, rows, cols):
    with io.open(os.path.join(OUTD, f), 'w', encoding='utf-8', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction='ignore'); w.writeheader()
        [w.writerow(x) for x in rows]
wd('l22_target_samples.csv', [dict(sample_id=r['sample_id'], group_id=r['group_id'], event_id=r['event_id'], address=r['address'], t0_utc=pre[r['sample_id'].replace('NEW-', '')]['t0_utc'], is_positive=1, previous_sidecar_source_status='IN_ONLY_OUT_MISSING') for r in targets], ['sample_id', 'group_id', 'event_id', 'address', 't0_utc', 'is_positive', 'previous_sidecar_source_status'])
wd('l22_outedge_recovery_status.csv', rec_status, ['sample_id', 'group_id', 'event_id', 'address', 'n_cone_nodes', 'n_nodes_attempted', 'n_nodes_success', 'n_nodes_blocked', 'n_raw_edges', 'n_structured_edges', 'status', 'notes'])
wd('l22_recovered_out_edges.csv', rec_edges, ['sample_id', 'group_id', 'event_id', 'node_address', 'edge_from', 'edge_to', 'value_eth', 'timestamp_utc', 'block_number', 'tx_hash', 'edge_type', 'source', 'source_status'])
with io.open(os.path.join(OUTD, 'l22_source_log.jsonl'), 'w', encoding='utf-8') as fo:
    [fo.write(json.dumps(x, ensure_ascii=False) + '\n') for x in src_log]

# ---- rebuild sample table: 157 old from L21 + 12 new updated ----
s0 = list(csv.DictReader(io.open(os.path.join(L20, 'l20_step0_feature_table.csv'), encoding='utf-8')))
l21sc = {r['sample_id']: r for r in csv.DictReader(io.open(os.path.join(L21, 'l21_sidecar_feature_table.csv'), encoding='utf-8'))}
samples = []
for r in s0:
    sid = r['sample_id']
    phi = {f: r.get(f, '') for f in PHI15}
    if sid in new_sc:
        sc, st = new_sc[sid]
        scd = {f: sc[f][0] for f in FAC}; scm = {f: (0 if sc[f][1] else 1) for f in FAC}; status2 = st
    else:
        o = l21sc[sid]; scd = {f: (None if o[f] == '' else o[f]) for f in FAC}; scm = {f: int(o['mi_' + f]) for f in FAC}; status2 = 'FULL_CONE_FROM_L21'
    samples.append(dict(sample_id=sid, group_id=r['group_id'], event_id=r['event_id'], address=r['address'].lower(), t0_utc=r['t0_utc'], is_positive=int(r['is_positive']), source_split=r['source_split'], status2=status2, phi=phi, scd=scd, scm=scm))
v2cols = ['sample_id', 'group_id', 'event_id', 'address', 't0_utc', 'is_positive', 'sidecar_source_status_v2'] + FAC + ['mi_' + f for f in FAC]
v2rows = []
for s in samples:
    row = dict(sample_id=s['sample_id'], group_id=s['group_id'], event_id=s['event_id'], address=s['address'], t0_utc=s['t0_utc'], is_positive=s['is_positive'], sidecar_source_status_v2=s['status2'])
    for f in FAC: row[f] = ('' if s['scd'][f] is None else s['scd'][f]); row['mi_' + f] = s['scm'][f]
    v2rows.append(row)
wd('l22_sidecar_feature_table_v2.csv', v2rows, v2cols)

# ---- re-eval on L20 folds ----
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
BOUND = ['eoa-source', 'asset-conversion', 'contract', 'depth-limit', 'coverage-stop', 'unresolved', 'origin-reached']
def num(v):
    try:
        if v in ('', None): return np.nan
        if v in ('True', True): return 1.0
        if v in ('False', False): return 0.0
        return float(v)
    except: return np.nan
def design(rows, train, block):
    fn = []
    if 'PHI' in block: fn += [f for f in PHI15 if f not in ('boundary_type', 'censoring_reason')]
    if 'SIDECAR' in block: fn += FAC
    med = {}
    for f in fn:
        vs = [num(s['phi'].get(f) if f in PHI15 else s['scd'][f]) for s in train]; vs = [x for x in vs if not np.isnan(x)]; med[f] = float(np.median(vs)) if vs else 0.0
    X = []
    for s in rows:
        v = []
        for f in fn:
            raw = num(s['phi'].get(f) if f in PHI15 else s['scd'][f]); v.append(med[f] if np.isnan(raw) else raw); v.append(0.0 if np.isnan(raw) else 1.0)
        if 'PHI' in block: b = str(s['phi'].get('boundary_type') or ''); v += [1.0 if b == c else 0.0 for c in BOUND]
        X.append(v)
    return np.array(X, dtype=float)
y = np.array([s['is_positive'] for s in samples]); idx = list(range(len(samples)))
posidx = [i for i in idx if samples[i]['is_positive']]; bgidx = [i for i in idx if not samples[i]['is_positive']]
groups = sorted(set(samples[i]['group_id'] for i in posidx)); K = len(groups)
def bgfold(s, k): return int(hashlib.sha256(s['address'].encode()).hexdigest(), 16) % k
loago = [(gi, g, [i for i in posidx if samples[i]['group_id'] == g] + [i for i in bgidx if bgfold(samples[i], K) == gi]) for gi, g in enumerate(groups)]
loago = [(gi, g, te, [i for i in idx if i not in te]) for gi, g, te in loago]
from sklearn.model_selection import GroupKFold
allg = [samples[i]['group_id'] if samples[i]['is_positive'] else 'BG_%d' % bgfold(samples[i], 5) for i in idx]
gkf = [(fid, '', list(te), list(tr)) for fid, (tr, te) in enumerate(GroupKFold(5).split(idx, y, groups=allg))]
BLOCKS = ['PHI_ONLY', 'SIDECAR_ONLY_OUTEDGE_RECOVERED', 'PHI_PLUS_SIDECAR_OUTEDGE_RECOVERED']
fold_metrics = []; per = {}
def block_norm(b): return b.replace('_OUTEDGE_RECOVERED', '')
def ev(split, folds):
    for fid, hg, te, tr in folds:
        yte = y[te]; ytr = y[tr]
        for block in BLOCKS:
            for mid in ['logreg', 'xgboost']:
                if yte.sum() == 0 or yte.sum() == len(yte):
                    fold_metrics.append(dict(split_name=split, fold_id=fid, feature_block=block, model_id=mid, metric_name='auroc', value='', status='NOT_ESTIMABLE_FOR_THIS_FOLD', notes='')); continue
                Xtr = design([samples[i] for i in tr], [samples[i] for i in tr], block_norm(block)); Xte = design([samples[i] for i in te], [samples[i] for i in tr], block_norm(block))
                if mid == 'logreg':
                    scl = StandardScaler().fit(Xtr); m = LogisticRegression(max_iter=2000, class_weight='balanced').fit(scl.transform(Xtr), ytr); sco = m.predict_proba(scl.transform(Xte))[:, 1]
                else:
                    m = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss', verbosity=0); m.fit(Xtr, ytr); sco = m.predict_proba(Xte)[:, 1]
                for met, f in [('auroc', roc_auc_score), ('auprc', average_precision_score)]:
                    val = f(yte, sco); fold_metrics.append(dict(split_name=split, fold_id=fid, feature_block=block, model_id=mid, metric_name=met, value=round(val, 4), status='ESTIMABLE', notes='')); per.setdefault((split, block, mid, met), []).append((fid, val))
ev('LOAGO_leave_one_actor_group_out', loago); ev('GroupKFold_by_group_id', gkf)
wd('l22_fold_metrics.csv', fold_metrics, ['split_name', 'fold_id', 'feature_block', 'model_id', 'metric_name', 'value', 'status', 'notes'])
rng = np.random.RandomState(0); agg = []; aggmap = {}
for (split, block, mid, met), vals in per.items():
    v = [x[1] for x in vals]; boots = [float(np.mean(rng.choice(v, len(v), replace=True))) for _ in range(2000)]
    agg.append(dict(split_name=split, feature_block=block, model_id=mid, metric_name=met, value=round(float(np.mean(v)), 4), ci_low=round(float(np.percentile(boots, 2.5)), 4), ci_high=round(float(np.percentile(boots, 97.5)), 4), n_estimable_folds=len(v), status='ESTIMABLE', notes='')); aggmap[(split, block, mid, met)] = dict(vals)
wd('l22_aggregate_metrics.csv', agg, ['split_name', 'feature_block', 'model_id', 'metric_name', 'value', 'ci_low', 'ci_high', 'n_estimable_folds', 'status', 'notes'])
inc = []
for split in ['LOAGO_leave_one_actor_group_out', 'GroupKFold_by_group_id']:
    for mid in ['logreg', 'xgboost']:
        for met in ['auroc', 'auprc']:
            base = aggmap.get((split, 'PHI_ONLY', mid, met), {})
            for cmpb, lab in [('SIDECAR_ONLY_OUTEDGE_RECOVERED', 'SIDECAR_OUTEDGE_minus_PHI'), ('PHI_PLUS_SIDECAR_OUTEDGE_RECOVERED', 'PHI_PLUS_SIDECAR_OUTEDGE_minus_PHI')]:
                oth = aggmap.get((split, cmpb, mid, met), {}); common = sorted(set(base) & set(oth))
                if not common: inc.append(dict(split_name=split, model_id=mid, metric_name=met, comparison=lab, value_delta='', ci_low='', ci_high='', status='NOT_ESTIMABLE', interpretation_guardrail='')); continue
                dl = [oth[f] - base[f] for f in common]; boots = [float(np.mean(rng.choice(dl, len(dl), replace=True))) for _ in range(2000)]
                inc.append(dict(split_name=split, model_id=mid, metric_name=met, comparison=lab, value_delta=round(float(np.mean(dl)), 4), ci_low=round(float(np.percentile(boots, 2.5)), 4), ci_high=round(float(np.percentile(boots, 97.5)), 4), status='ESTIMABLE', interpretation_guardrail='post out-edge recovery; still NOT comparable to within-Harmony 0.990'))
wd('l22_incremental_comparison.csv', inc, ['split_name', 'model_id', 'metric_name', 'comparison', 'value_delta', 'ci_low', 'ci_high', 'status', 'interpretation_guardrail'])

from collections import Counter
st = Counter(r['status'] for r in rec_status)
nfull = st.get('OUT_EDGE_RECOVERED_FULL', 0) + st.get('OUT_EDGE_NOT_NEEDED', 0); npart = st.get('OUT_EDGE_RECOVERED_PARTIAL', 0); nblk = st.get('OUT_EDGE_BLOCKED_SOURCE_ACCESS', 0)
phi_h = hashlib.sha256(open(os.path.join(ROOT, r"01_冻结档案\01_Phi初期实验\phi.py"), 'rb').read()).hexdigest(); pipe_h = hashlib.sha256(open(os.path.join(ROOT, r"01_冻结档案\01_Phi初期实验\pipeline.py"), 'rb').read()).hexdigest()
def gg(s, b, m, mt):
    a = [x for x in agg if x['split_name'] == s and x['feature_block'] == b and x['model_id'] == m and x['metric_name'] == mt]; return a[0]['value'] if a else None
rs = "LOAGO logreg AUROC: PHI=%s SIDECAR(outedge)=%s PHI+SIDECAR(outedge)=%s | recovery full/part/blocked=%d/%d/%d" % (gg('LOAGO_leave_one_actor_group_out', 'PHI_ONLY', 'logreg', 'auroc'), gg('LOAGO_leave_one_actor_group_out', 'SIDECAR_ONLY_OUTEDGE_RECOVERED', 'logreg', 'auroc'), gg('LOAGO_leave_one_actor_group_out', 'PHI_PLUS_SIDECAR_OUTEDGE_RECOVERED', 'logreg', 'auroc'), nfull, npart, nblk)
status = 'L22_EXECUTED_OUT_EDGE_RECOVERED_AND_REEVALUATED' if nblk == 0 else ('L22_BLOCKED_ALL_OUT_EDGE_ACCESS' if nfull + npart == 0 else 'L22_PARTIAL_OUT_EDGE_RECOVERED_AND_REEVALUATED')
json.dump(dict(round='L22', status=status, input_files_sha256={'l21_sidecar': 'a944d17d', 'l17_cone_edges': 'edfb2324'}, n_targets=12, n_targets_full=nfull, n_targets_partial=npart, n_targets_blocked=nblk, n_samples=len(samples), n_positive=int(sum(y)), n_unknown_background=int(len(y) - sum(y)), n_groups=len(groups), feature_blocks_run=BLOCKS, models_run=['logreg', 'xgboost'], splits_run=['LOAGO_leave_one_actor_group_out', 'GroupKFold_by_group_id'], outedge_window='[t0-7d, t0)', theta_eth=10, attribution_interval_used=False, main_manuscript_changed=False, frozen_hashes_match=(phi_h.startswith('18c2db') and pipe_h.startswith('773fe3c3')), result_summary=rs, next_gate='L22_OUTEDGE_SIDECAR_REEVALUATION_REVIEW_GATE'), io.open(os.path.join(OUTD, 'l22_execution_manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('STATUS:', status); print('recovery:', dict(st)); print(rs)
for x in agg:
    if x['metric_name'] == 'auroc' and x['split_name'].startswith('LOAGO'): print('  LOAGO %-36s %-8s AUROC=%s CI[%s,%s]' % (x['feature_block'], x['model_id'], x['value'], x['ci_low'], x['ci_high']))
for x in inc:
    if x['split_name'].startswith('LOAGO') and x['model_id'] == 'logreg' and x['metric_name'] == 'auroc': print('  INC', x['comparison'], 'delta=', x['value_delta'], 'CI[%s,%s]' % (x['ci_low'], x['ci_high']))
