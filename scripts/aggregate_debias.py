"""Aggregate the per-seed debias_sweep.csv files across seeds -> mean+/-std per alpha.
alpha=0 row is the (no-debias) baseline within these runs."""
import os, csv, numpy as np
SD = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext/saved/yelp/llmesr_sasrec"
SEEDS = range(42, 52)
ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
cols = ["HR@10", "NDCG@10", "Tail_HR@10", "Tail_NDCG@10"]
data = {a: {c: [] for c in cols} for a in ALPHAS}
nseed = 0
for s in SEEDS:
    p = os.path.join(SD, "debias%d" % s, "debias_sweep.csv")
    if not os.path.exists(p):
        continue
    nseed += 1
    for r in csv.DictReader(open(p)):
        a = float(r["alpha"])
        if a in data:
            for c in cols:
                data[a][c].append(float(r[c]))
print("debias popularity sweep - %d seeds (alpha=0 is the in-run baseline)" % nseed)
print("%-6s %-16s %-16s %-16s %-16s" % ("alpha", *cols))
base = {c: np.mean(data[0.0][c]) if data[0.0][c] else float("nan") for c in cols}
for a in ALPHAS:
    d = data[a]
    if not d["HR@10"]:
        print("%-6.2f (no data)" % a); continue
    cells = []
    for c in cols:
        v = np.array(d[c]); cells.append("%.4f+/-%.4f" % (v.mean(), v.std()))
    print("%-6.2f %-16s %-16s %-16s %-16s" % (a, *cells))
# best alpha per metric + delta vs in-run baseline
print("\nbest non-zero alpha per metric (d vs alpha=0 baseline):")
for c in cols:
    best_a, best_v = None, -1
    for a in ALPHAS:
        if a == 0 or not data[a][c]:
            continue
        m = np.mean(data[a][c])
        if m > best_v:
            best_v, best_a = m, a
    if best_a is not None:
        print("  %-14s alpha=%.2f  %.4f  (d %+.4f vs %.4f)" % (c, best_a, best_v, best_v - base[c], base[c]))
