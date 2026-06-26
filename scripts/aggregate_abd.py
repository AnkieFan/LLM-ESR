"""A (cold-item strata) + B (lin3 x max-pool D6 compounding) + D (full-catalog ranking).
All on Yelp/SASRec, 5 seeds (42..46): base d6_sas_yelp vs lin3 (3-layer linear adapter)."""
import os, csv, numpy as np
from scipy import stats
SD = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext/saved/yelp/llmesr_sasrec"
SEEDS = range(42, 47)


def read_sweep(tag, pfile):
    D = {}
    for s in SEEDS:
        p = f"{SD}/{tag}{s}/{pfile}"
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            g = float(r["gamma"]); D.setdefault(g, {"HR": [], "NDCG": [], "Tail": []})
            D[g]["HR"].append(float(r["HR@10"])); D[g]["NDCG"].append(float(r["NDCG@10"]))
            if "Tail_HR@10" in r:
                D[g]["Tail"].append(float(r["Tail_HR@10"]))
    return D


print("=" * 74); print("B) COMPOUNDING: lin3 adapter (train-time) x max-pool D6 (eval-time)"); print("=" * 74)
base = read_sweep("d6_sas_yelp", "usr_sem_sweep_max.csv")
lin3 = read_sweep("lin3", "usr_sem_sweep_max.csv")
if base and lin3 and 16.0 in base and 16.0 in lin3:
    b0, b16 = np.array(base[0.0]["HR"]), np.array(base[16.0]["HR"])
    l0, l16 = np.array(lin3[0.0]["HR"]), np.array(lin3[16.0]["HR"])
    print(f"base  g0 HR {b0.mean():.4f}  +maxD6(g16) {b16.mean():.4f}  (dD6 {b16.mean()-b0.mean():+.4f})")
    print(f"lin3  g0 HR {l0.mean():.4f}  +maxD6(g16) {l16.mean():.4f}  (dD6 {l16.mean()-l0.mean():+.4f})")
    print(f"lin3 standalone (g0 vs base g0): {l0.mean()-b0.mean():+.4f} (p={stats.ttest_ind(l0,b0)[1]:.3f})")
    print(f"lin3+maxD6 vs base+maxD6 (g16) : {l16.mean()-b16.mean():+.4f} (p={stats.ttest_ind(l16,b16)[1]:.3f})")
    print(f"BEST config lin3+maxD6 vs baseline(base g0): {l16.mean()-b0.mean():+.4f}")
    add = (l0.mean() - b0.mean()) + (b16.mean() - b0.mean())
    act = l16.mean() - b0.mean()
    print(f"additive-predicted {add:+.4f} vs actual {act:+.4f}  ->  "
          f"{'~additive (levers stack)' if abs(add-act) < 0.002 else 'sub-additive (overlap)' if act < add else 'super-additive'}")
else:
    print("(waiting for data)")

print("\n" + "=" * 74); print("A) COLD-ITEM STRATA (max-pool D6, HR@10 by target-item training freq)"); print("=" * 74)
for tag in ["d6_sas_yelp", "lin3"]:
    bins = {}
    for s in SEEDS:
        p = f"{SD}/{tag}{s}/usr_sem_cold.csv"
        if not os.path.exists(p):
            continue
        rows = list(csv.reader(open(p))); hdr = rows[0]
        for r in rows[1:]:
            lab = r[0].split("(")[0]
            bins.setdefault(lab, {gi: [] for gi in range(1, len(hdr))})
            for gi in range(1, len(hdr)):
                bins[lab][gi].append(float(r[gi]))
    if not bins:
        print(f"[{tag}] (no data)"); continue
    print(f"\n[{tag}]  freq-bin       HR g0     g8      g16     g32     (dg16   dg32)")
    for lab in ["0-2", "3-5", "6-20", "21-100", "101-1000000000"]:
        if lab not in bins:
            continue
        m = [np.mean(bins[lab][gi]) for gi in sorted(bins[lab])]
        nm = lab.replace("-1000000000", "+")
        print(f"  {nm:13s} {m[0]:.4f}  {m[1]:.4f}  {m[2]:.4f}  {m[3]:.4f}   ({m[2]-m[0]:+.4f}  {m[3]-m[0]:+.4f})")

print("\n" + "=" * 74); print("D) FULL-CATALOG RANKING (rank target vs ALL items; max-pool D6 swept)"); print("=" * 74)
for tag in ["d6_sas_yelp", "lin3"]:
    D = {}
    for s in SEEDS:
        p = f"{SD}/{tag}{s}/fullrank_sweep.csv"
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            g = float(r["gamma"]); D.setdefault(g, {"HR": [], "NDCG": [], "MRR": []})
            D[g]["HR"].append(float(r["HR@10"])); D[g]["NDCG"].append(float(r["NDCG@10"])); D[g]["MRR"].append(float(r["MRR"]))
    if not D:
        print(f"[{tag}] (no data)"); continue
    print(f"\n[{tag}]  gamma   HR@10    NDCG@10   MRR")
    g0 = np.array(D[0.0]["HR"])
    for g in sorted(D):
        hr = np.array(D[g]["HR"])
        dp = "" if g == 0 else f"   (dHR {hr.mean()-g0.mean():+.4f}, p={stats.ttest_rel(hr,g0)[1]:.3f})"
        print(f"  {g:5.0f}  {hr.mean():.5f}  {np.mean(D[g]['NDCG']):.5f}  {np.mean(D[g]['MRR']):.5f}{dp}")
print("\n" + "=" * 74)
