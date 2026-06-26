import os, csv, numpy as np
from scipy import stats
SD = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext"
BB = {"llmesr_sasrec":"SASRec","llmesr_gru4rec":"GRU4Rec","llmesr_bert4rec":"Bert4Rec"}
combos = [(p[0],p[1],p[2]) for p in (l.strip().split("|") for l in open(SD+"/jobs/cfg_moeall.txt")) if len(p)>=3]
EN=["collab","semantic","full","full+max"]; CELLS=["TailU-TailI","TailU-HeadI","HeadU-TailI","HeadU-HeadI"]
print("Stage-1 diagnostic, ALL 9 cells (5 seeds). cold-cold = TailUser x TailItem.")
print("%-9s %-8s | cold-cold: semantic vs full+max (d, p) | best expert per cell (TT/TH/HT/HH)"%("backbone","dataset"))
for tag,model,ds in combos:
    per=[]
    for s in range(42,47):
        f="%s/saved/%s/%s/%s%d/moe_diag.csv"%(SD,ds,model,tag,s)
        if not os.path.exists(f): continue
        d={r[0]:{EN[i]:float(r[2+i]) for i in range(4)} for r in csv.reader(open(f)) if r[0] in CELLS}
        if d: per.append(d)
    if not per: print("%-9s %-8s | (no data)"%(BB.get(model,model),ds)); continue
    a=np.array([p["TailU-TailI"]["semantic"] for p in per]); b=np.array([p["TailU-TailI"]["full+max"] for p in per])
    pv=stats.ttest_rel(a,b)[1] if len(a)>=3 else float('nan')
    best=[]
    for c in CELLS:
        m={e:np.mean([p[c][e] for p in per]) for e in EN}; best.append(max(m,key=m.get).replace("full+max","f+m").replace("semantic","sem").replace("collab","col").replace("full","full"))
    print("%-9s %-8s | %.4f vs %.4f (%+.4f, p=%.3f)%s | %s"%(BB.get(model,model),ds,a.mean(),b.mean(),a.mean()-b.mean(),pv,"*" if pv<0.05 else " "," / ".join(best)))
