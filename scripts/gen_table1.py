"""Build paper-Table-1-style breakdowns (Overall / Tail-Item / Head-Item / Tail-User / Head-User, H@10 & N@10)
for the positive D6 extensions, baseline (g=0) -> extension (g=16), 5 seeds, all 9 backbone x dataset cells.
Emits markdown to log/yelp/SUMMARY_table1.md."""
import os, csv, numpy as np
from scipy import stats
WT = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext"
SD = WT + "/saved"
SEEDS = range(42, 47)
GROUPS = [("Overall", "HR@10", "NDCG@10"), ("Tail-Item", "Tail_HR@10", "Tail_NDCG@10"),
          ("Head-Item", "HeadItem_HR@10", "HeadItem_NDCG@10"), ("Tail-User", "TailUser_HR@10", "TailUser_NDCG@10"),
          ("Head-User", "HeadUser_HR@10", "HeadUser_NDCG@10")]
combos = [(p[0], p[1], p[2]) for p in (l.strip().split("|") for l in open(WT + "/jobs/cfg_d6all.txt")) if len(p) >= 3]
DSNAME = {"yelp": "Yelp", "fashion": "Fashion", "beauty": "Beauty"}
BB = {"llmesr_sasrec": "SASRec", "llmesr_gru4rec": "GRU4Rec", "llmesr_bert4rec": "Bert4Rec"}


def load(tag, model, ds, pfile, gcol="gamma"):
    """return {gamma: {col: [vals over seeds]}}"""
    D = {}
    for s in SEEDS:
        f = "%s/%s/%s/%s%d/%s" % (SD, ds, model, tag, s, pfile)
        if not os.path.exists(f):
            continue
        rows = list(csv.DictReader(open(f)))
        for r in rows:
            g = float(r[gcol])
            D.setdefault(g, {})
            for k, v in r.items():
                if k != gcol:
                    D[g].setdefault(k, []).append(float(v))
    return D


CH = "| backbone | | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |"


def emit(title, pfile, gcol, g_ext, extlabel, note):
    out = ["", "### %s" % title, "", note, ""]
    for ds in ["yelp", "fashion", "beauty"]:
        out.append("**%s**" % DSNAME[ds])
        out.append(CH)
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for tag, model, dd in combos:
            if dd != ds:
                continue
            D = load(tag, model, ds, pfile, gcol)
            if not D or 0.0 not in D or g_ext not in D:
                continue   # skip cells with no data
            a = np.array(D[g_ext]["HR@10"]); b = np.array(D[0.0]["HR@10"])
            sig = "*" if len(a) >= 3 and stats.ttest_rel(a, b)[1] < 0.05 else ""
            for lab, grow in [("base", 0.0), (extlabel + sig, g_ext)]:
                cells = []
                for _, hk, nk in GROUPS:
                    hv = np.mean(D[grow][hk]); nv = np.mean(D[grow][nk])
                    if grow == g_ext:   # extension row: show absolute and (d vs baseline)
                        cells.append("%.4f(%+.4f) | %.4f(%+.4f)" %
                                     (hv, hv - np.mean(D[0.0][hk]), nv, nv - np.mean(D[0.0][nk])))
                    else:
                        cells.append("%.4f | %.4f" % (hv, nv))
                out.append("| %s | %s | %s |" % (BB.get(model, model), lab, " | ".join(cells)))
        out.append("")
    return "\n".join(out)


doc = ["# Paper-Table-1-style group breakdown - positive D6 extensions",
       "", "Groups follow the paper: Overall, Tail-Item / Head-Item (by item train-freq @ ts_item),",
       "Tail-User / Head-User (by sequence length @ ts_user). 5 seeds; H@10 cells show **value**(d vs baseline)."]
doc.append(emit("D7 max-pool (the headline extension)", "usr_sem_sweep_max.csv", "gamma", 16.0, "**+max**",
                "Max-pool user profile, score += gamma*max_t<e_hist_t, e_cand> at gamma=16 (eval-time, frozen model)."))
doc.append(emit("D6 mean-pool (base version)", "usr_sem_sweep.csv", "gamma", 16.0, "**+mean**",
                "Mean-centroid user profile at gamma=16 (superseded by max-pool above; shown for completeness)."))
doc.append(emit("Per-segment gamma (Tail-User-only routing)", "usr_sem_seg_sweep.csv", "gamma_tail", 16.0, "**+seg**",
                "Max-pool term applied only to Tail Users (seq_len < ts_user); Head-User columns stay unchanged."))
open(WT + "/log/yelp/SUMMARY_table1.md", "w").write("\n".join(doc) + "\n")
print("wrote log/yelp/SUMMARY_table1.md")
print("\n".join(doc))
