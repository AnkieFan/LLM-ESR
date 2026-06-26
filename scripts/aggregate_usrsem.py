"""Aggregate per-seed usr_sem_sweep.csv -> mean per gamma + PAIRED d vs gamma=0 (same model) with
paired t-test. gamma=0 is the in-run baseline; positive d on HR/NDCG = a Pareto (overall) gain."""
import os, csv, numpy as np
from scipy import stats
SD = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext/saved/yelp/llmesr_sasrec"
SEEDS = range(42, 52)
GAMMAS = [0.0, 4.0, 8.0, 16.0, 32.0, 64.0]
cols = ["HR@10", "NDCG@10", "Tail_HR@10", "Tail_NDCG@10"]
val = {g: {c: [] for c in cols} for g in GAMMAS}
nseed = 0
for s in SEEDS:
    p = os.path.join(SD, "debias%d" % s, "usr_sem_sweep.csv")
    if not os.path.exists(p):
        continue
    nseed += 1
    rows = {float(r["gamma"]): r for r in csv.DictReader(open(p))}
    for g in GAMMAS:
        if g in rows:
            for c in cols:
                val[g][c].append(float(rows[g][c]))
print("user-semantic scoring sweep - %d seeds (gamma=0 = in-run baseline)" % nseed)
print("%-6s %-17s %-17s %-17s %-17s" % ("gamma", *cols))
for g in GAMMAS:
    if not val[g]["HR@10"]:
        print("%-6.1f (no data)" % g); continue
    cells = ["%.4f+/-%.4f" % (np.mean(val[g][c]), np.std(val[g][c])) for c in cols]
    print("%-6.1f %-17s %-17s %-17s %-17s" % (g, *cells))
print("\nPaired d vs gamma=0 (per-seed) + paired t-test p  (* = p<0.05):")
for g in GAMMAS:
    if g == 0:
        continue
    parts = []
    for c in cols:
        a, b = np.array(val[g][c]), np.array(val[0.0][c])
        if len(a) == len(b) and len(a) >= 3:
            d = (a - b).mean(); p = stats.ttest_rel(a, b)[1]
            parts.append("%s %+.4f(p=%.2g%s)" % (c, d, p, "*" if p < 0.05 else ""))
    print("  gamma=%-4.0f " % g + "  ".join(parts))
