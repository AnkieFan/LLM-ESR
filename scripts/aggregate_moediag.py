"""Stage-1 two-specialist diagnostic: per dataset (SASRec, 5 seeds), the 2x2 user x item grid with each
expert's HR@10; best expert per cell; oracle-routed vs best-single overall (paired t-test); cold-cold
semantic-vs-(full+max) paired test. Also sanity-checks full-expert HR == model's standard logged HR."""
import os, csv, glob, numpy as np
from scipy import stats
SD = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext"
DS = [("Yelp","yelp"),("Fashion","fashion"),("Beauty","beauty")]
TAG = {"yelp":"d6_sas_yelp","fashion":"d6_sas_fashion","beauty":"d6_sas_beauty"}
EN = ["collab","semantic","full","full+max"]
CELLS = ["TailU-TailI","TailU-HeadI","HeadU-TailI","HeadU-HeadI"]
for name, ds in DS:
    print("="*88); print(name, "(SASRec, 5 seeds)"); print("="*88)
    # collect per-seed cell tables
    perseed=[]
    for s in range(42,47):
        f="%s/saved/%s/llmesr_sasrec/%s%d/moe_diag.csv"%(SD,ds,TAG[ds],s)
        if not os.path.exists(f): continue
        d={}
        for r in csv.reader(open(f)):
            if r[0] in CELLS or r[0]=="OVERALL":
                d[r[0]]={"N":int(r[1]),**{EN[i]:float(r[2+i]) for i in range(4)}}
                if r[0]=="OVERALL":
                    d["routed"]=float(r[6].split("=")[1]); d["gain"]=float(r[7].split("[")[1].rstrip("]"))
        perseed.append(d)
    if not perseed: print("  (no data)"); continue
    print("  %-12s %-7s %-8s %-8s %-8s %-9s  best"%("cell","N","collab","semantic","full","full+max"))
    for c in CELLS:
        n=int(np.mean([p[c]["N"] for p in perseed]))
        m={e:np.mean([p[c][e] for p in perseed]) for e in EN}
        best=max(m,key=m.get)
        star=""
        if best!="full+max":  # is the winner sig better than full+max on this cell?
            a=np.array([p[c][best] for p in perseed]); b=np.array([p[c]["full+max"] for p in perseed])
            if len(a)>=3 and stats.ttest_rel(a,b)[1]<0.05: star="*"
        print("  %-12s %-7d %.4f   %.4f   %.4f   %.4f    %s%s"%(c,n,m["collab"],m["semantic"],m["full"],m["full+max"],best,star))
    o={e:np.mean([p["OVERALL"][e] for p in perseed]) for e in EN}
    routed=np.array([p["routed"] for p in perseed]); fm=np.array([p["OVERALL"]["full+max"] for p in perseed])
    pg=stats.ttest_rel(routed,fm)[1] if len(routed)>=3 else float('nan')
    print("  OVERALL      full+max=%.4f  routed(oracle)=%.4f  gain=%+.4f (p=%.3f)"%(o["full+max"],routed.mean(),routed.mean()-fm.mean(),pg))
    # cold-cold semantic vs full+max
    a=np.array([p["TailU-TailI"]["semantic"] for p in perseed]); b=np.array([p["TailU-TailI"]["full+max"] for p in perseed])
    print("  cold-cold: semantic=%.4f vs full+max=%.4f  d=%+.4f (p=%.3f)"%(a.mean(),b.mean(),a.mean()-b.mean(),stats.ttest_rel(a,b)[1] if len(a)>=3 else float('nan')))
    # sanity: full overall HR vs standard logged HR
    chk=[]
    for s in range(42,47):
        cf=glob.glob("%s/log/%s/llmesr_sasrec_%s%d.csv"%(SD,ds,TAG[ds],s))
        if cf:
            try: chk.append(abs(float(list(csv.DictReader(open(cf[0])))[-1]["HR@10"])-perseed[s-42]["OVERALL"]["full"]))
            except: pass
    if chk: print("  [sanity] max |full_expert - standard HR| = %.2e (should be ~0)"%max(chk))
