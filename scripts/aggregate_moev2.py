import os, csv, numpy as np
from scipy import stats
SD = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext"
BB = {"llmesr_sasrec":"SASRec","llmesr_gru4rec":"GRU4Rec","llmesr_bert4rec":"Bert4Rec"}
combos = [(p[0],p[1],p[2]) for p in (l.strip().split("|") for l in open(SD+"/jobs/cfg_moeall.txt")) if len(p)>=3]
CFGS = ["item_hard_a0","item_soft_lin","item_soft_log","user_gate_a0","user_gate_a50","ts_x0.5_a0","ts_x2.0_a0","ts_x4.0_a0"]
print("dHR@10 vs full+max baseline (paired, 5 seeds). *=p<.05")
print("%-9s %-8s | %s | BEST"%("backbone","dataset"," | ".join("%-11s"%c.replace('_a0','').replace('item_','i_').replace('user_gate','u') for c in CFGS)))
for tag, model, ds in combos:
    data={c:[] for c in CFGS}; base=[]
    for s in range(42,47):
        f="%s/saved/%s/%s/%s%d/moe_route_v2.csv"%(SD,ds,model,tag,s)
        if not os.path.exists(f): continue
        d={r["config"]:float(r["HR@10"]) for r in csv.DictReader(open(f))}
        base.append(d["baseline_fullmax"])
        for c in CFGS: data[c].append(d.get(c,np.nan))
    if not base: print("%-9s %-8s | (no data)"%(BB.get(model,model),ds)); continue
    base=np.array(base); cells=[]
    best=("baseline",0.0)
    for c in CFGS:
        a=np.array(data[c]); d=a.mean()-base.mean()
        p=stats.ttest_rel(a,base)[1] if len(a)>=3 else 1.0
        cells.append("%+.4f%s"%(d,"*" if p<0.05 else " "))
        if d>best[1]: best=(c,d)
    print("%-9s %-8s | %s | %s %+.4f"%(BB.get(model,model),ds," | ".join("%-11s"%x for x in cells),best[0],best[1]))
