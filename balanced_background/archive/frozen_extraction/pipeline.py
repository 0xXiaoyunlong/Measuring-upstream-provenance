# 统一数据管线: 原始链上事件 -> campaign -> cone -> Phi (同一入口)
# Harmony/Beanstalk/第三案件 必须调用完全相同代码。不读 label/incident。
import urllib.request, json, time, os
from phi import compute_phi, PARAMS, PHI_FEATURES
KEY=os.environ.get('ETHERSCAN_KEY','')   # 从环境变量读, 不硬编码
DEP_TOPIC='0xa945e51eec50ab98c161376f0db4cf2aeba3ec92755fe2fcd388bdbbb80ff196'
THRESH=10*10**18; DEPTH=PARAMS['max_depth']; CONE_CAP=25
W=PARAMS['W_seconds']; CAMPAIGN_GAP=48*3600   # campaign 合并间隔(数据管线职责, 非 Phi 参数)
SWAP_SRC={'0x1111111254eeb25477b68fb85ed929f73a960582','0x1111111254fb6c44bac0bed2854e76f90643097d',
 '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2','0x7a250d5630b4cf539739df2c5dacb4c659f2488d',
 '0xe592427a0aece92de3edee1f18e0157c05861564','0xdef1c0ded9bec7f1a1670819833240f027b25eff',
 '0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45','0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f'}
def _get(p):
    u='https://api.etherscan.io/v2/api?chainid=1&'+p+'&apikey='+KEY
    for _ in range(5):
        try:
            with urllib.request.urlopen(u,timeout=45) as r: return json.load(r)
        except Exception: time.sleep(0.7)
    return {}
_code={}
def _is_contract(a, block):
    # 时间一致: 在 t0 对应 block 查代码, 禁用 t0 后链上状态
    k=(a,block)
    if k in _code: return _code[k]
    r=_get('module=proxy&action=eth_getCode&address=%s&tag=0x%x'%(a,block)).get('result','0x')
    c=(r is not None and r!='0x' and len(r)>3); _code[k]=c; return c
def _bal_at(a,blk):
    r=_get('module=proxy&action=eth_getBalance&address=%s&tag=0x%x'%(a,blk)).get('result','0x0')
    try: return int(r,16)/1e18
    except: return 0.0
def _win_txs(a,sb,eb,t0):
    ins=[]; outs=[]
    for act in ['txlist','txlistinternal']:
        j=_get('module=account&action=%s&address=%s&startblock=%d&endblock=%d&sort=asc'%(act,a,sb,eb))
        for t in (j.get('result') or []):
            if not isinstance(t,dict): continue
            v=int(t.get('value','0')); ts=int(t['timeStamp'])
            if v<THRESH or ts>=t0 or ts<t0-W: continue       # 严格窗口 [t0-W, t0)
            if t.get('to','').lower()==a: ins.append((t.get('from','').lower(),v/1e18,ts))
            if t.get('from','').lower()==a: outs.append((t.get('to','').lower(),v/1e18,ts))
    return ins,outs
# === 共享 cone 构造 (Harmony/Beanstalk/第三案件 同一函数) ===
def build_cone(sender,t0,dep_block):
    sb=max(0,dep_block-50400); cone={}; frontier=[(sender,0)]
    while frontier and len(cone)<CONE_CAP:
        addr,d=frontier.pop(0)
        if addr in cone: continue
        if d>0 and (addr in SWAP_SRC or _is_contract(addr, dep_block)):
            cone[addr]={'boundary':True,'in':[],'out':[],'init':0.0}; continue
        ins,outs=_win_txs(addr,sb,dep_block,t0); init=_bal_at(addr,sb)
        cone[addr]={'boundary':False,'in':ins,'out':outs,'init':init}
        if d<DEPTH:
            for (frm,amt,ts) in ins:
                if frm and frm not in cone: frontier.append((frm,d+1))
    return cone
# === campaign 构造 (数据管线职责) ===
def build_campaigns(deposits):
    # deposits: [{'from','ts','block',...}]; 同址 gap<CAMPAIGN_GAP 合并; t0=首存
    by={}
    for o in deposits: by.setdefault(o['from'],[]).append(o)
    camps=[]
    for s,lst in by.items():
        lst=sorted(lst,key=lambda x:x['ts']); cur=[lst[0]]
        for o in lst[1:]:
            if o['ts']-cur[-1]['ts']<CAMPAIGN_GAP: cur.append(o)
            else: camps.append(cur); cur=[o]
        camps.append(cur)
    return [{'address':c[0]['from'],'t0':c[0]['ts'],'block':c[0]['block'],'n_events':len(c)} for c in camps]
# === 端到端: 一个案件配置 -> Phi 结果 (无标签) ===
def get_pool_deposits(pool, sb, eb):
    lg=_get('module=logs&action=getLogs&address=%s&topic0=%s&fromBlock=%d&toBlock=%d'%(pool,DEP_TOPIC,sb,eb))
    out=[]; seen=set()
    for L in (lg.get('result') or []):
        h=L['transactionHash']
        if h in seen: continue
        seen.add(h)
        r=_get('module=proxy&action=eth_getTransactionByHash&txhash='+h).get('result'); time.sleep(0.12)
        if r and r.get('from'): out.append({'from':r['from'].lower(),'ts':int(L['timeStamp'],16),'block':int(r['blockNumber'],16)})
    return out
def process_case(config, V=100.0, on_each=None):
    # config={'pool','start_block','end_block'} —— 仅结构, 无 label/incident
    deps=get_pool_deposits(config['pool'],config['start_block'],config['end_block'])
    camps=build_campaigns(deps); results=[]
    for c in camps:
        cone=build_cone(c['address'],c['t0'],c['block'])
        try: ph=compute_phi(c['address'],c['t0'],V,cone); results.append({'address':c['address'],'phi':ph})
        except Exception as e: results.append({'address':c['address'],'phi':None,'error':str(e)})
        if on_each: on_each()
    return results
# 复用已存 cone 计算 Phi (dev 验证; cone 由本 build_cone 等价逻辑生成)
def phi_from_ledger(ledger_path, V=100.0):
    led=json.load(open(ledger_path)); res=[]
    for s,rec in led.items():
        try: res.append({'address':s,'phi':compute_phi(s,rec['deposit']['ts'],V,rec['cone'])})
        except Exception as e: res.append({'address':s,'phi':None,'error':str(e)})
    return res
