import os, csv, numpy as np
from scipy import stats
SD="/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext"
BB={"llmesr_sasrec":"SASRec","llmesr_gru4rec":"GRU4Rec","llmesr_bert4rec":"Bert4Rec"}
DSN={"yelp":"Yelp","fashion":"Fashion","beauty":"Beauty"}
combos=[(p[0],p[1],p[2]) for p in (l.strip().split("|") for l in open(SD+"/jobs/cfg_moeall.txt")) if len(p)>=3]
COLS=["HR@10","NDCG@10","TailItem_HR","TailItem_N","HeadItem_HR","HeadItem_N","TailUser_HR","TailUser_N","HeadUser_HR","HeadUser_N"]
out=[]
for ds in ["yelp","fashion","beauty"]:
    out.append("\n**%s**"%DSN[ds]); out.append("| backbone | | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |"); out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag,model,dd in combos:
        if dd!=ds: continue
        B={c:[] for c in COLS}; R={c:[] for c in COLS}
        for s in range(42,47):
            f="%s/saved/%s/%s/%s%d/moe_route_full.csv"%(SD,ds,model,tag,s)
            if not os.path.exists(f): continue
            d={r["config"]:r for r in csv.DictReader(open(f))}
            for c in COLS: B[c].append(float(d["full+max"][c])); R[c].append(float(d["router_a0"][c]))
        if not B["HR@10"]: continue
        pv=stats.ttest_rel(np.array(R["HR@10"]),np.array(B["HR@10"]))[1]
        sig="*" if pv<0.05 else ""
        out.append("| %s | full+max | %s |"%(BB[model]," | ".join("%.4f"%np.mean(B[c]) for c in COLS)))
        out.append("| %s | **+router**%s | %s |"%(BB[model],sig," | ".join("%.4f(%+.4f)"%(np.mean(R[c]),np.mean(R[c])-np.mean(B[c])) for c in COLS)))
open(SD+"/log/yelp/_moe_router_blocks.md","w").write("\n".join(out)+"\n")
print("wrote _moe_router_blocks.md"); print("\n".join(out))
