"""D7+MoE: per combo, compare user-profile experts (mean/recency/max) + MoE ensemble at the gamma=16
operating point. Paired d vs gamma=0 across 5 seeds, with significance. Flags the best profile."""
import os, csv, numpy as np
from scipy import stats
WT = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext"
SD = WT + "/saved"
SEEDS = range(42, 47)
cols = ["HR@10", "NDCG@10", "Tail_HR@10"]
PROFILES = [("mean", "usr_sem_sweep.csv"), ("recency", "usr_sem_sweep_recency.csv"),
            ("max", "usr_sem_sweep_max.csv"), ("ens(MoE)", "usr_sem_sweep_ens.csv")]
GAMMAS = [16.0, 32.0]
combos = []
for line in open(WT + "/jobs/cfg_d6all.txt"):
    p = line.strip().split("|")
    if len(p) >= 3:
        combos.append((p[0], p[1], p[2]))

def stats_at(ds, model, tag, pfile, g):
    v, v0 = {c: [] for c in cols}, {c: [] for c in cols}
    for s in SEEDS:
        f = "%s/%s/%s/%s%d/%s" % (SD, ds, model, tag, s, pfile)
        if not os.path.exists(f):
            continue
        rows = {float(r["gamma"]): r for r in csv.DictReader(open(f))}
        if g in rows and 0.0 in rows:
            for c in cols:
                v[c].append(float(rows[g][c])); v0[c].append(float(rows[0.0][c]))
    out = {}
    for c in cols:
        a, b = np.array(v[c]), np.array(v0[c])
        if len(a) >= 3:
            out[c] = ((a - b).mean(), stats.ttest_rel(a, b)[1])
        else:
            out[c] = (float("nan"), float("nan"))
    return out, len(v["HR@10"])

print("=" * 100)
print("D7 profiles + MoE ensemble - paired d vs gamma=0 (HR / NDCG / Tail_HR)")
print("=" * 100)
for g in GAMMAS:
    print("\n########## gamma = %.0f ##########" % g)
    for tag, model, ds in combos:
        print("\n### %-16s %-18s %-9s" % (tag, model, ds))
        best, bestv = None, -9
        for pname, pfile in PROFILES:
            o, n = stats_at(ds, model, tag, pfile, g)
            if n == 0:
                print("   %-9s (no data)" % pname); continue
            cells = []
            for c in cols:
                d, p = o[c]
                cells.append("%-7s %+.4f%s" % (c.replace("@10", ""), d, "*" if (p == p and p < 0.05) else ""))
            mark = ""
            if o["HR@10"][0] == o["HR@10"][0] and o["HR@10"][0] > bestv:
                bestv, best = o["HR@10"][0], pname
            print("   %-9s %s   %s   %s   [%d]" % (pname, cells[0], cells[1], cells[2], n))
        print("   -> best overall-HR profile: %s" % best)
print("\n" + "=" * 100)
