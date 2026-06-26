# Two-Specialist Routing (a mixture over the dual views)

**Setup.** Authors' preprocessed data, full LLM-ESR, 1 pos vs 100 random negs, HR@10 / NDCG@10. All numbers eval-time on the frozen checkpoints (no retraining), 5 seeds, paired t-tests; * = p<0.05. Cross study over {SASRec, GRU4Rec, Bert4Rec} x {Yelp, Fashion, Beauty}.

## 0. Motivation

The meeting raised "multiple models / MoE" - specialise the recommender per segment (e.g. head vs tail users/items) and route. We already ruled out the *score-level profile ensemble* (averaging content profiles never beats the single best, max-pool) and validated a *hard per-segment router* (Tail Users -> +content). The open question: are different segments genuinely best served by **different experts**, and can we route to them?

## 1. Key idea - LLM-ESR already contains two experts

The dual-view model scores `score(u,i) = <u, i>` where both `u` and `i` are `[collaborative | semantic]` concatenations (each `hidden_size`-d). So the score **splits exactly**: `score = u_collab*i_collab + u_semantic*i_semantic = collab_score + semantic_score`. The two views are two expert sub-models, extractable at inference with **no retraining** by splitting `u` and `i` at the `hidden_size` boundary. (Verified: the reconstructed `full` expert HR matches the model's standard logged HR to <=5e-5 on all 9 cells.) So the four eval-time experts are: `collab`, `semantic`, `full` (=base), `full+max` (=base + the D6 max-pool term).

## 2. Stage 1 - diagnostic: which expert wins each 2 x 2 (user x item) cell?

SASRec, 5 seeds. Cell = (Tail/Head User by `seq_len<ts_user`) x (Tail/Head Item by target `pop<ts_item`). HR@10 per expert; the best expert is **bold**.

| **Yelp**                     | **collab** | **semantic** | **full**  | **full+max** |
| ---------------------------- | ---------- | ------------ | --------- | ------------ |
| Tail-User x Tail-Item (n~2818) | 0.129      | **0.216**    | 0.181     | 0.198        |
| Tail-User x Head-Item (n~9643) | 0.623      | 0.702        | **0.810** | 0.809        |
| Head-User x Tail-Item (n~758)  | 0.194      | 0.205        | 0.209     | **0.221**    |
| Head-User x Head-Item (n~2501) | 0.594      | 0.718        | 0.803     | **0.807**    |

| **Fashion**                  | **collab** | **semantic** | **full** | **full+max** |
| ---------------------------- | ---------- | ------------ | -------- | ------------ |
| Tail-User x Tail-Item (n~1734) | 0.061      | **0.219**    | 0.119    | 0.177        |
| Tail-User x Head-Item (n~3401) | 0.560      | 0.628        | 0.669    | **0.685**    |
| Head-User x Tail-Item (n~857)  | 0.054      | **0.200**    | 0.096    | 0.137        |
| Head-User x Head-Item (n~3102) | 0.695      | 0.796        | 0.824    | **0.832**    |

| **Beauty**                    | **collab** | **semantic** | **full** | **full+max** |
| ----------------------------- | ---------- | ------------ | -------- | ------------ |
| Tail-User x Tail-Item (n~8541)  | 0.044      | **0.396**    | 0.225    | 0.308        |
| Tail-User x Head-Item (n~34284) | 0.542      | 0.473        | 0.642    | **0.655**    |
| Head-User x Tail-Item (n~1510)  | 0.046      | **0.384**    | 0.230    | 0.308        |
| Head-User x Head-Item (n~7869)  | 0.595      | 0.508        | 0.689    | **0.711**    |

**Finding.** The best expert **does** differ by cell, with a clean rule: **cold/tail items -> the semantic view alone; popular items -> full+max.** For cold-user x cold-item the semantic view beats full+max significantly on all 3 datasets: **+0.019 (Yelp, p=.014), +0.042 (Fashion, p=.001), +0.088 (Beauty, p<.001)**. Mechanism: a cold item's collaborative embedding is essentially untrained noise, so including it *hurts* - dropping it (semantic-only) is the fix. Oracle per-cell routing -> +0.004 / +0.014 / +0.017 overall (an upper bound: the target's cell is unknown at inference).

## 3. Stage 2 - realizable router (training-free)

Since the target's segment isn't known at inference but **each candidate's popularity is**, route per *candidate*: down-weight the collaborative score for cold candidates. `score(u,i) = alpha_i*collab(u,i) + semantic(u,i) + gamma*max(u,i)`, with `alpha_i = 1` (popular item) or `alpha_tail` (tail item, swept). `alpha_tail=1` reproduces full+max (baseline); `alpha_tail=0` = "semantic + content, no collab" for cold items. One consistent score scale (it's the full+max score with a per-item gate on the collab component), so no expert-calibration issue. Implemented as `--moe_diag` (writes `moe_route.csv`).

**Result - best-alpha_tail overall dHR@10 vs the full+max baseline (5 seeds):**

| **backbone** | **Yelp**     | **Fashion**  | **Beauty**   |
| ------------ | ------------ | ------------ | ------------ |
| **SASRec**   | +0.0020      | **+0.0273*** | +0.0047*     |
| **GRU4Rec**  | **+0.0140*** | **+0.0336*** | **+0.0270*** |
| **Bert4Rec** | +0.0004      | **+0.0293*** | **+0.0269*** |

**Analysis.** Significant on **7/9** cells, clears **+0.01 on 6/9** - strongest and fully consistent on the content-rich **Amazon** datasets (Fashion all 3 backbones; Beauty 2/3), with large tail-item gains (+0.04...+0.11). The weak spot is **Yelp** (only GRU4Rec clears the bar): Yelp's popular items carry strong collaborative signal, so per-candidate gating lets popular negatives outrank cold targets (a per-candidate competition effect); Bert4Rec/Yelp is also the lone cell where the Stage-1 diagnostic flips (cold-cold semantic -0.031). **The gain stacks on D6/max-pool** - the baseline here already *is* full+max, so e.g. Fashion/GRU4Rec is max-pool (+0.024) **plus** router (+0.034) ~ +0.058 over the original model.

## 4. Router variants (user-level gate / soft gate / ts_item sweep)

Three variants of the gate, all eval-time on the same checkpoints (5 seeds), dHR@10 vs the full+max baseline. **(a) User-level gate** - down-weight collab uniformly across a cold *user's* candidates (alpha=0 for Tail Users) instead of per-candidate. **(b) Soft gate** - a log-popularity ramp `alpha_i = min(1, log1p(pop)/log1p(ts_item))` instead of the hard split. **(c) ts_item sweep** - the hard `alpha_tail=0` gate with the popularity threshold scaled x2 / x4 (treat more items as "cold" and drop their collab).

| **backbone x dataset** | **user-gate (alpha0)** | **soft-log** | **ts x 2**     | **ts x 4**     |
| -------------------- | ------------------ | ------------ | ------------ | ------------ |
| SASRec / Yelp        | -0.0595*           | +0.0018*     | -0.0012*     | -0.0294*     |
| GRU4Rec / Yelp       | -0.0691*           | +0.0081*     | **+0.0183*** | +0.0108*     |
| Bert4Rec / Yelp      | -0.0763*           | +0.0022*     | +0.0009*     | -0.0519*     |
| SASRec / Fashion     | +0.0138*           | +0.0081*     | **+0.0354*** | +0.0309*     |
| GRU4Rec / Fashion    | +0.0243*           | +0.0087*     | +0.0482*     | **+0.0488*** |
| Bert4Rec / Fashion   | +0.0193*           | +0.0077*     | **+0.0401*** | +0.0373*     |
| SASRec / Beauty      | -0.0674*           | +0.0021*     | **+0.0084*** | +0.0037      |
| GRU4Rec / Beauty     | -0.0317*           | +0.0126*     | +0.0549*     | **+0.0632*** |
| Bert4Rec / Beauty    | -0.0257*           | +0.0081*     | +0.0527*     | **+0.0618*** |

**Analysis.** Three conclusions. **(1) User-level gating fails** - strongly negative almost everywhere (-0.03...-0.08): down-weighting collab across *all* of a cold user's candidates wrecks their popular-item targets (the Tail-User x Head-Item cell wants full+max). This confirms the **per-candidate** gate of Sec 3 is the right granularity. **(2) Soft gates underperform** the hard gate on Amazon (e.g. Fashion soft-log +0.008 vs hard +0.027); they only matter on Yelp, where they nudge the negative hard-gate result back to ~0. **(3) Tuning the threshold is the real lever** - raising it to x2/x4 (dropping collab for a *wider* band of items) lifts the Amazon gains substantially: **Fashion +0.035...+0.049, Beauty +0.053...+0.063** - roughly double the Sec 3 hard-gate numbers, because Amazon collaborative embeddings are noisy across a broad popularity range, and the paper's `ts_item` is too conservative for this purpose. The threshold is dataset-specific (a universal x2 helps Amazon but *hurts* Yelp SASRec/Bert4Rec by -0.01...-0.02), so it must be set on a validation split.

**Refined recommendation.** Router = per-candidate hard gate (drop collab on cold items) with the **popularity threshold tuned per dataset** (high for sparse Amazon, low/off for dense Yelp). Best results: **Beauty +0.06, Fashion +0.05** (GRU4Rec/Bert4Rec), all on top of max-pool. Yelp SASRec/Bert4Rec are unsalvageable (best ~+0.002) - Yelp's popular items carry too much real collaborative signal to drop.

## 5. Summary

- **Stage 1 is the clean universal finding:** for cold items the **semantic view alone** is the best expert (significant on all 3 datasets) - the dual view's fixed 50/50 fusion is *wrong* for cold items.
- **Stage 2 is a deployable, training-free router** (per-candidate: drop collab on cold items) with the popularity threshold tuned per dataset. Best variant per cell clears **+0.01 on 6/9**, with **large gains on the Amazon datasets - Fashion +0.04...+0.05, Beauty up to +0.06** - and it is **complementary to max-pool** (the baseline already includes it).
- **The granularity and threshold both matter:** per-candidate gating beats user-level gating (which fails outright), and raising `ts_item` x2-4 roughly doubles the Amazon gains (their collaborative embeddings are noisy across a wide popularity band).
- **Honest scope:** dataset-dependent - it needs cold items to be prevalent and their collaborative signal weak; on Yelp (strong popular-item collaborative signal) only GRU4Rec benefits and SASRec/Bert4Rec are flat (~+0.002).

Artifacts: `SUMMARY_moediag.txt` (Stage 1), `SUMMARY_moe_all.txt` / `SUMMARY_moeroute.txt` (Stage 2), `SUMMARY_moev2.txt` (variants); flag `--moe_diag`; eval logic + `_final_feat` split in `trainers/sequence_trainer.py`; aggregators in `scripts/aggregate_moe*.py`.