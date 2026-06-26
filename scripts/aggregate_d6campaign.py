"""Aggregate the D6 cross-backbone/dataset campaign: for each combo, read per-seed usr_sem_sweep.csv,
report per-gamma PAIRED d vs gamma=0 (same model) + paired t-test. gamma=0 is that combo's baseline."""
import os, csv, numpy as np
from scipy import stats
WT = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext"
SD = WT + "/saved"
SEEDS = range(42, 47)  # 5 seeds
GAMMAS = [0.0, 4.0, 8.0, 16.0, 32.0, 64.0]
cols = ["HR@10", "NDCG@10", "Tail_HR@10", "Tail_NDCG@10"]

combos = []
for line in open(WT + "/jobs/cfg_d6all.txt"):
    p = line.strip().split("|")
    if len(p) >= 3:
        combos.append((p[0], p[1], p[2]))

def pe(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    return float((a - b).mean()), (float(stats.ttest_rel(a, b)[1]) if len(a) >= 3 else float("nan"))

print("=" * 90)
print("D6 user-semantic term - validation across backbones x datasets (paired d vs gamma=0)")
print("=" * 90)
for tag, model, ds in combos:
    val = {g: {c: [] for c in cols} for g in GAMMAS}
    n = 0
    for s in SEEDS:
        f = "%s/%s/%s/%s%d/usr_sem_sweep.csv" % (SD, ds, model, tag, s)
        if not os.path.exists(f):
            continue
        n += 1
        rows = {float(r["gamma"]): r for r in csv.DictReader(open(f))}
        for g in GAMMAS:
            if g in rows:
                for c in cols:
                    val[g][c].append(float(rows[g][c]))
    print("\n### %-16s %-18s %-9s  [%d seeds]" % (tag, model, ds, n))
    if n == 0:
        print("   (no data yet)"); continue
    b = {c: np.mean(val[0.0][c]) for c in cols}
    print("   baseline (gamma=0): HR %.4f  NDCG %.4f  Tail_HR %.4f  Tail_NDCG %.4f" %
          (b["HR@10"], b["NDCG@10"], b["Tail_HR@10"], b["Tail_NDCG@10"]))
    print("   %-5s %-16s %-16s %-16s %-16s" % ("gamma", "dHR@10(p)", "dNDCG@10(p)", "dTail_HR(p)", "dTail_NDCG(p)"))
    for g in GAMMAS:
        if g == 0 or not val[g]["HR@10"]:
            continue
        cells = []
        for c in cols:
            d, p = pe(val[g][c], val[0.0][c])
            star = "*" if (p == p and p < 0.05) else ""
            cells.append("%+.4f(%.2g%s)" % (d, p, star))
        print("   %-5.0f %-16s %-16s %-16s %-16s" % (g, *cells))
print("\n" + "=" * 90)
