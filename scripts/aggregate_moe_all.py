"""Stage-2 router across all 9 backbone x dataset cells (5 seeds): alpha_tail sweep of the popularity-gated
collaborative score; best alpha vs full+max baseline (paired t-test) + tail-item gain. Also the cold-cold
diagnostic d (semantic vs full+max) and the full==standard-HR sanity check."""
import os, csv, glob, numpy as np
from scipy import stats
SD = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext"
BB = {"llmesr_sasrec":"SASRec","llmesr_gru4rec":"GRU4Rec","llmesr_bert4rec":"Bert4Rec"}
combos = [(p[0],p[1],p[2]) for p in (l.strip().split("|") for l in open(SD+"/jobs/cfg_moeall.txt")) if len(p)>=3]
ALPHAS = ["1.00","0.50","0.25","0.00"]
print("%-9s %-8s | best a_tail | overall dHR (p) | tail-item dHR | cold-cold sem-(full+max) | sanity"%("backbone","dataset"))
for tag, model, ds in combos:
    rt = {a:[] for a in ALPHAS}; ti = {a:[] for a in ALPHAS}; cc=[]; ccb=[]; san=[]
    for s in range(42,47):
        rf = "%s/saved/%s/%s/%s%d/moe_route.csv"%(SD,ds,model,tag,s)
        df = "%s/saved/%s/%s/%s%d/moe_diag.csv"%(SD,ds,model,tag,s)
        if os.path.exists(rf):
            for r in csv.DictReader(open(rf)):
                a="%.2f"%float(r["alpha_tail"]); rt[a].append(float(r["HR@10"])); ti[a].append(float(r["tailitem_HR"]))
        if os.path.exists(df):
            d={r[0]:r for r in csv.reader(open(df))}
            if "TailU-TailI" in d: cc.append(float(d["TailU-TailI"][3])); ccb.append(float(d["TailU-TailI"][5]))  # semantic, full+max
            # sanity: full expert OVERALL vs standard logged HR
            cf=glob.glob("%s/log/%s/%s_%s%d.csv"%(SD,ds,model,tag,s))
            if "OVERALL" in d and cf:
                try: san.append(abs(float(d["OVERALL"][4])-float(list(csv.DictReader(open(cf[0])))[-1]["HR@10"])))
                except: pass
    if not rt["1.00"]: print("%-9s %-8s | (no data)"%(BB.get(model,model),ds)); continue
    base=np.array(rt["1.00"]); best=max(ALPHAS,key=lambda a:np.mean(rt[a])); hb=np.array(rt[best])
    p=stats.ttest_rel(hb,base)[1] if best!="1.00" else 1.0
    dti=np.mean(ti[best])-np.mean(ti["1.00"])
    ccd=(np.mean(cc)-np.mean(ccb)) if cc else float('nan')
    s=("%.0e"%max(san)) if san else "n/a"
    star="*" if p<0.05 else ""
    print("%-9s %-8s |    %s     | %+.4f (%.3f)%s | %+.4f | %+.4f | %s"%(
        BB.get(model,model),ds,best,np.mean(hb)-np.mean(base),p,star,dti,ccd,s))
