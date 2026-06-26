"""Stage-2 realizable router: per dataset (SASRec, 5 seeds), alpha_tail sweep of the popularity-gated
collaborative score. Reports overall HR/NDCG + tail-item HR per alpha; best alpha vs full+max (alpha=1)
with paired t-test."""
import os, csv, numpy as np
from scipy import stats
SD = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext"
DS = [("Yelp","yelp","d6_sas_yelp"),("Fashion","fashion","d6_sas_fashion"),("Beauty","beauty","d6_sas_beauty")]
ALPHAS = ["1.00","0.50","0.25","0.00"]
for name, ds, tag in DS:
    per = {a: {"HR":[],"NDCG":[],"cc":[],"ti":[],"hi":[]} for a in ALPHAS}
    ns = 0
    for s in range(42,47):
        f = "%s/saved/%s/llmesr_sasrec/%s%d/moe_route.csv"%(SD,ds,tag,s)
        if not os.path.exists(f): continue
        ns += 1
        for r in csv.DictReader(open(f)):
            a = "%.2f"%float(r["alpha_tail"])
            per[a]["HR"].append(float(r["HR@10"])); per[a]["NDCG"].append(float(r["NDCG@10"]))
            per[a]["cc"].append(float(r["coldcold_HR"])); per[a]["ti"].append(float(r["tailitem_HR"])); per[a]["hi"].append(float(r["headitem_HR"]))
    if ns == 0: print(name, "(no data)"); continue
    print("="*78); print("%s (SASRec, %d seeds)  [alpha_tail=1.00 = full+max baseline]"%(name,ns)); print("="*78)
    print("  a_tail  HR@10            NDCG@10   tailitem_HR   headitem_HR")
    base = np.array(per["1.00"]["HR"])
    bestA, bestHR = "1.00", np.mean(base)
    for a in ALPHAS:
        hr = np.array(per[a]["HR"]); d = hr.mean()-base.mean()
        p = stats.ttest_rel(hr, base)[1] if (a!="1.00" and len(hr)>=3) else float('nan')
        tag2 = "" if a=="1.00" else " (d%+.4f, p=%.3f)"%(d,p)
        print("  %s   %.5f%s   %.4f    %.4f       %.4f"%(a, hr.mean(), tag2, np.mean(per[a]["NDCG"]), np.mean(per[a]["ti"]), np.mean(per[a]["hi"])))
        if hr.mean() > bestHR: bestA, bestHR = a, hr.mean()
    hb = np.array(per[bestA]["HR"])
    print("  -> best alpha_tail=%s: overall HR %+.4f vs baseline (p=%.3f); tail-item %+.4f"%(
        bestA, hb.mean()-base.mean(), stats.ttest_rel(hb,base)[1] if bestA!="1.00" else float('nan'),
        np.mean(per[bestA]["ti"])-np.mean(per["1.00"]["ti"])))
