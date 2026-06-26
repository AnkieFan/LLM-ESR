import os, sys
import numpy as np, pandas as pd
from scipy import stats
WT = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext/log/yelp"
MAIN = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR/log/yelp"
SEEDS = list(range(42, 52))
def series(base, tag, col):
    v = {}
    for s in SEEDS:
        p = os.path.join(base, "llmesr_sasrec_%s%d.csv" % (tag, s))
        if os.path.exists(p):
            v[s] = float(pd.read_csv(p).iloc[-1][col])
    return v
def stat(d):
    a = np.array(list(d.values())); return (a.mean(), a.std(), len(a)) if len(a) else (np.nan, np.nan, 0)
def paired(new, base):
    common = sorted(set(new) & set(base))
    if len(common) < 3: return np.nan, np.nan
    a = np.array([new[s] for s in common]); b = np.array([base[s] for s in common])
    return float((a-b).mean()), float(stats.ttest_rel(a, b)[1])
base_o = series(MAIN, "paper", "HR@10"); base_n = series(MAIN, "paper", "NDCG@10")
base_t = series(MAIN, "paper", "Tail HR@10")
bo, _, _ = stat(base_o); bn,_,_ = stat(base_n); bt,_,_ = stat(base_t)
print("baseline (full 2-layer, main paper runs): Overall H@10 %.4f  N@10 %.4f  Tail H@10 %.4f\n" % (bo, bn, bt))
print("%-10s %-18s %-18s %-18s %-14s" % ("config", "Overall H@10", "Overall N@10", "TailItem H@10", "dOverallH (p)"))
for tag in sys.argv[1:]:
    o = series(WT, tag, "HR@10"); n = series(WT, tag, "NDCG@10"); t = series(WT, tag, "Tail HR@10")
    (om, osd, k) = stat(o); (nm, nsd, _) = stat(n); (tm, tsd, _) = stat(t)
    d, p = paired(o, base_o)
    sig = "" if np.isnan(p) else (" *" if p < 0.05 else "")
    print("%-10s %.4f+/-%.4f     %.4f+/-%.4f     %.4f+/-%.4f     %+.4f (%.3g)%s  [%d seeds]" %
          (tag, om, osd, nm, nsd, tm, tsd, d, p, sig, k))
