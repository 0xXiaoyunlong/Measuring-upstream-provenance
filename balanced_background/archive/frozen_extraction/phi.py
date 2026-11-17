# Phi v2: censoring-aware upstream-process representation (frozen; fixes the v1 spec/code mismatch)
# fixes: (1)strict window t0-W<=ts<t0; (2)drop campaign_gap(upstream pipeline's); (3)dwell uses current path edges;
#        (4)split/consol/residual marked node_context(attribution uncertain); meaning and feature set unchanged.
SWAP={'0x1111111254eeb25477b68fb85ed929f73a960582','0x1111111254fb6c44bac0bed2854e76f90643097d',
 '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2','0x7a250d5630b4cf539739df2c5dacb4c659f2488d',
 '0xe592427a0aece92de3edee1f18e0157c05861564','0xdef1c0ded9bec7f1a1670819833240f027b25eff',
 '0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45','0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f'}
TORNADO={'0xd90e2f925da726b50c4ed8d0fb90ad053324f31b','0x722122df12d4e14e13ac3b6895a86e84145b6967',
 '0xa160cdab225685da1d56aa342ad8841c3b53f291','0x910cbd523d972eb0a6f4cae4618ad62622b39dbf',
 '0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936','0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc'}

# ===== pre-registered params (from data availability/compute cost/operational window, not separation) =====
# note: campaign merging belongs to the upstream pipeline(pipeline.py), not to Phi.
PARAMS = {
 'W_seconds': 7*24*3600,      # observation window: t0-W<=ts<t0
 'max_depth': 2,
 'coverage_threshold': 0.3,
 'min_value_share': 0.1,      # relative to V
}

def F(value, observed=True, censored=False, reason=None, scope=None):
    return {'value':value,'is_observed':observed,'is_censored':censored,'missing_reason':reason,'scope':scope}

def _inwin(ts, t0, W): return (t0 - W) <= ts < t0     # strict window

def reverse_bounded_trace(a, cone, t0, V, P=PARAMS):
    # single dominant value path; each hop follows the largest-share funder(ties: earliest, then address order); label-free; strict window
    W=P['W_seconds']; path=[a]; edges=[]; ident=float(V); cur=a; ambiguous=False
    for depth in range(P['max_depth']):
        info=cone.get(cur)
        if info is None: return path,cur,'unresolved','unresolved',ident,edges,ambiguous
        ins=[(frm,amt,ts) for frm,amt,ts in info.get('in',[]) if _inwin(ts,t0,W) and amt>=P['min_value_share']*V]
        if not ins:
            bt='contract' if info.get('boundary') else 'eoa-source'
            return path,cur,bt,('contract-opacity' if bt=='contract' else 'origin-reached'),ident,edges,ambiguous
        frm,amt,ts = max(ins, key=lambda x:(x[1], -x[2], x[0]))
        if amt < ident: ambiguous=True
        ident=min(ident,amt)
        # edge: (downstream=cur received from upstream=frm, amt, ts)
        e=(cur,frm,amt,ts)
        if frm in SWAP: return path+[frm],frm,'asset-conversion','cross-asset',min(ident,V),edges+[e],ambiguous
        fi=cone.get(frm)
        if fi is None: return path+[frm],frm,'unresolved','unresolved',min(ident,V),edges+[e],ambiguous
        if fi.get('boundary'): return path+[frm],frm,'contract','contract-opacity',min(ident,V),edges+[e],ambiguous
        if min(amt,V)/V < P['coverage_threshold']: return path+[frm],frm,'coverage-stop','coverage-stop',min(ident,V),edges+[e],ambiguous
        path.append(frm); edges.append(e); cur=frm
    return path,cur,'depth-limit','depth-limit',min(ident,V),edges,ambiguous

def _node_context_resid(info, t0, W):
    ev=[(ts,amt) for frm,amt,ts in info.get('in',[]) if _inwin(ts,t0,W)]+\
       [(ts,-amt) for to,amt,ts in info.get('out',[]) if _inwin(ts,t0,W) and to not in TORNADO]
    ev.sort()
    if not ev: return None
    R=0; Rs=[]
    for ts,d in ev: R+=d; Rs.append(R)
    nin=sum(1 for frm,amt,ts in info.get('in',[]) if _inwin(ts,t0,W))
    nout=sum(1 for to,amt,ts in info.get('out',[]) if _inwin(ts,t0,W) and to not in TORNADO)
    return {'Rrange':max(Rs)-min(Rs),'n_in':nin,'n_out':nout}

def compute_phi(a, t0, V, cone, P=PARAMS):
    a=a.lower(); W=P['W_seconds']
    path,bnode,btype,creason,ident,edges,ambiguous = reverse_bounded_trace(a,cone,t0,V,P)
    inter = path[1:-1]    # intermediaries(excludes deposit address a and the boundary node)
    phi={}
    phi['observed_hops']=F(len(path)-1)
    # path_elapsed_time: t0 - earliest path edge ts(value starts moving at the boundary); boundary not a natural origin -> censored
    if edges:
        t_enter=min(e[3] for e in edges)
        cen = btype in ('contract','depth-limit','coverage-stop','unresolved')
        phi['path_elapsed_time']=F((t0-t_enter)/3600.0, censored=cen, reason=(creason if cen else None))
    else:
        phi['path_elapsed_time']=F(None,observed=False,reason='no_edge')
    # dwell: current path edges -- path[i] receives the value(edge i) and passes it on(edge i-1)
    dwells=[]; anom=False
    for i in range(1,len(path)-1):
        ts_arrive = edges[i][3]      # path[i] received from path[i+1] (edge i)
        ts_depart = edges[i-1][3]    # path[i-1] received from path[i] = path[i] sent on (edge i-1)
        d=(ts_depart-ts_arrive)/3600.0
        if d<0: anom=True
        dwells.append(d)
    if dwells:
        phi['dwell_mean_h']=F(sum(dwells)/len(dwells), censored=anom, reason=('negative_dwell_anomaly' if anom else None))
        phi['dwell_max_h']=F(max(dwells), censored=anom, reason=('negative_dwell_anomaly' if anom else None))
    else:
        phi['dwell_mean_h']=F(None,observed=False,reason='no_intermediary'); phi['dwell_max_h']=F(None,observed=False,reason='no_intermediary')
    # coverage
    cov=min(max(ident,0.0),V)/V
    phi['coverage']=F(cov, censored=ambiguous, reason=('attribution_ambiguous' if ambiguous else None))
    phi['attribution_ambiguous']=F(bool(ambiguous))
    # node_context: node window stats, attribution uncertain(not proven to be this value) -> scope='node_context'
    if inter:
        outd=[len(set(to for to,amt,ts in cone.get(m,{}).get('out',[]) if _inwin(ts,t0,W) and to not in TORNADO)) for m in inter]
        ind=[len(set(frm for frm,amt,ts in cone.get(m,{}).get('in',[]) if _inwin(ts,t0,W))) for m in inter]
        phi['node_context_out_degree']=F(sum(outd)/len(outd), reason='node_context_not_path_isolated', scope='node_context')
        phi['node_context_in_degree']=F(sum(ind)/len(ind), reason='node_context_not_path_isolated', scope='node_context')
    else:
        phi['node_context_out_degree']=F(None,observed=False,reason='no_intermediary'); phi['node_context_in_degree']=F(None,observed=False,reason='no_intermediary')
    # wallet age: cone has in-window tx only -> true age left-censored(lower bound)
    ages=[]
    for m in inter:
        tss=[ts for frm,amt,ts in cone.get(m,{}).get('in',[]) if _inwin(ts,t0,W)]+[ts for to,amt,ts in cone.get(m,{}).get('out',[]) if _inwin(ts,t0,W)]
        if tss: ages.append((t0-min(tss))/3600.0)
    phi['wallet_age_min_h']=F(min(ages),censored=True,reason='left_censored_window_only') if ages else F(None,observed=False,reason='no_intermediary')
    # node_context residual dynamics (path nodes, in-window, attribution uncertain)
    rds=[_node_context_resid(cone.get(m,{}),t0,W) for m in (inter if inter else path[1:])]
    rds=[r for r in rds if r]
    if rds:
        phi['node_context_resid_Rrange_max']=F(max(r['Rrange'] for r in rds),reason='node_context_not_path_isolated',scope='node_context')
        phi['node_context_resid_nin_mean']=F(sum(r['n_in'] for r in rds)/len(rds),scope='node_context')
        phi['node_context_resid_nout_mean']=F(sum(r['n_out'] for r in rds)/len(rds),scope='node_context')
    else:
        for k in ['node_context_resid_Rrange_max','node_context_resid_nin_mean','node_context_resid_nout_mean']:
            phi[k]=F(None,observed=False,reason='no_path_node')
    phi['boundary_type']=F(btype); phi['censoring_reason']=F(creason); phi['observation_confidence']=F(cov)
    return phi

PHI_FEATURES=['observed_hops','path_elapsed_time','dwell_mean_h','dwell_max_h','coverage','attribution_ambiguous',
 'node_context_out_degree','node_context_in_degree','wallet_age_min_h',
 'node_context_resid_Rrange_max','node_context_resid_nin_mean','node_context_resid_nout_mean',
 'boundary_type','censoring_reason','observation_confidence']
