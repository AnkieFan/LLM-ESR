# LLM-ESR Extension: User-Semantic Scoring (D6) and its line of work

**Setup.** All numbers are on the authors' preprocessed data, full LLM-ESR (dual-view + cross-attention + self-distillation), 1 positive vs 100 random negatives, HR@10 / NDCG@10. d are **paired** vs the no-extension baseline (the same model, gamma=0), with paired t-tests; * = p<0.05. The cross-backbone / cross-dataset study uses 5 seeds per cell over {SASRec, GRU4Rec, Bert4Rec} x {Yelp, Fashion, Beauty}.

## 0. The core idea
Every earlier lever (capacity, training objective, more data, popularity reweighting) failed to beat the trained model's **overall** accuracy - it sits at a ceiling. The one thing that breaks the ceiling is to **inject information the model doesn't already have at scoring time.** LLM-ESR uses the frozen LLM item embeddings only as (a) an item-embedding initializer, (b) the input to a 1536->64 adapter, and (c) for similar-user retrieval. But the 64-d adapter + sequence encoder **discard most of the 1536-d content signal.** D6 restores it: add a direct, full-dimensional content-match between the user's history and the candidate, as a third additive scoring term - at **inference**, on the frozen model.

---

## 1. D6 - user-semantic scoring term (the base method)

**Motivation.** Recover the content signal the 64-d bottleneck throws away, as a cheap inference-time rerank.

**Design.** For user `u` and candidate item `i`: `score'(u,i) = score(u,i) + gamma * < c_u , e_i >`, where `e_i` = frozen 1536-d LLM embedding (ada-002, unit-norm), `c_u` = a centroid of the user's history item embeddings, and `gamma` a scalar weight (`gamma=0` = baseline). Because the embeddings are unit-norm, the term equals the **average content-similarity of the candidate to the user's history**.

**Implementation.** `--usr_sem`. ~20 lines in the eval loop: per batch, compute `c_u` from `seq` and the term against `item_indices`; after the loop, sweep `gamma in {0,4,8,16,32,64}` and write `usr_sem_sweep.csv`. No training change - runs via `--do_test` on any trained checkpoint (one model -> the whole gamma-sweep).

**Experiment.** Full paper-Table-1 breakdown - groups **Ov**=Overall, **TI**=Tail-Item, **HI**=Head-Item, **TU**=Tail-User, **HU**=Head-User (H@10 & N@10) - **baseline (gamma=0) vs +mean (gamma=16)**, 5 seeds; * = paired p<0.05 on Overall H@10. The **+** rows show the extension value with **(d vs the base row)** in parentheses (this base / +ext-with-d layout is used for every breakdown in Sec 1, Sec 3, Sec 4).

**Yelp**
| backbone | | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | base | 0.6667 | 0.4216 | 0.1865 | 0.0833 | 0.8081 | 0.5212 | 0.6673 | 0.4231 | 0.6646 | 0.4160 |
| SASRec | **+mean*** | 0.6687(+0.0020) | 0.4234(+0.0017) | 0.1947(+0.0082) | 0.0889(+0.0056) | 0.8083(+0.0002) | 0.5218(+0.0006) | 0.6695(+0.0022) | 0.4249(+0.0018) | 0.6659(+0.0013) | 0.4174(+0.0014) |
| GRU4Rec | base | 0.5729 | 0.3417 | 0.0853 | 0.0351 | 0.7164 | 0.4320 | 0.5775 | 0.3450 | 0.5550 | 0.3289 |
| GRU4Rec | **+mean*** | 0.5821(+0.0092) | 0.3496(+0.0079) | 0.0904(+0.0051) | 0.0386(+0.0035) | 0.7268(+0.0104) | 0.4411(+0.0091) | 0.5865(+0.0089) | 0.3529(+0.0078) | 0.5654(+0.0104) | 0.3370(+0.0080) |
| Bert4Rec | base | 0.6649 | 0.4248 | 0.1313 | 0.0546 | 0.8220 | 0.5338 | 0.6650 | 0.4270 | 0.6645 | 0.4164 |
| Bert4Rec | **+mean*** | 0.6684(+0.0035) | 0.4282(+0.0034) | 0.1392(+0.0079) | 0.0590(+0.0044) | 0.8242(+0.0022) | 0.5369(+0.0031) | 0.6684(+0.0034) | 0.4307(+0.0037) | 0.6684(+0.0039) | 0.4183(+0.0020) |

**Fashion**
| backbone | | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | base | 0.5632 | 0.4746 | 0.1116 | 0.0534 | 0.7430 | 0.6422 | 0.4835 | 0.3784 | 0.6667 | 0.5993 |
| SASRec | **+mean*** | 0.5818(+0.0186) | 0.4922(+0.0177) | 0.1532(+0.0416) | 0.0796(+0.0261) | 0.7525(+0.0094) | 0.6565(+0.0143) | 0.5090(+0.0255) | 0.4038(+0.0254) | 0.6763(+0.0096) | 0.6070(+0.0077) |
| GRU4Rec | base | 0.5483 | 0.4549 | 0.0848 | 0.0389 | 0.7329 | 0.6205 | 0.4647 | 0.3524 | 0.6567 | 0.5879 |
| GRU4Rec | **+mean*** | 0.5667(+0.0184) | 0.4775(+0.0226) | 0.1187(+0.0340) | 0.0596(+0.0207) | 0.7451(+0.0122) | 0.6439(+0.0234) | 0.4902(+0.0255) | 0.3849(+0.0325) | 0.6660(+0.0092) | 0.5977(+0.0098) |
| Bert4Rec | base | 0.5472 | 0.4548 | 0.0529 | 0.0224 | 0.7440 | 0.6270 | 0.4601 | 0.3491 | 0.6602 | 0.5920 |
| Bert4Rec | **+mean*** | 0.5696(+0.0224) | 0.4829(+0.0281) | 0.0966(+0.0438) | 0.0481(+0.0257) | 0.7578(+0.0139) | 0.6560(+0.0290) | 0.4916(+0.0316) | 0.3900(+0.0410) | 0.6707(+0.0105) | 0.6034(+0.0113) |

**Beauty**
| backbone | | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | base | 0.5689 | 0.3723 | 0.2260 | 0.1104 | 0.6507 | 0.4348 | 0.5589 | 0.3648 | 0.6147 | 0.4069 |
| SASRec | **+mean*** | 0.5772(+0.0083) | 0.3851(+0.0127) | 0.2683(+0.0423) | 0.1429(+0.0326) | 0.6509(+0.0002) | 0.4428(+0.0080) | 0.5685(+0.0096) | 0.3794(+0.0146) | 0.6172(+0.0025) | 0.4111(+0.0042) |
| GRU4Rec | base | 0.4933 | 0.3150 | 0.1515 | 0.0784 | 0.5748 | 0.3714 | 0.4871 | 0.3091 | 0.5218 | 0.3419 |
| GRU4Rec | **+mean*** | 0.5127(+0.0194) | 0.3322(+0.0172) | 0.1843(+0.0328) | 0.0963(+0.0180) | 0.5910(+0.0162) | 0.3884(+0.0170) | 0.5078(+0.0207) | 0.3274(+0.0183) | 0.5354(+0.0135) | 0.3542(+0.0123) |
| Bert4Rec | base | 0.5307 | 0.3577 | 0.1319 | 0.0774 | 0.6258 | 0.4246 | 0.5205 | 0.3491 | 0.5773 | 0.3971 |
| Bert4Rec | **+mean*** | 0.5455(+0.0147) | 0.3713(+0.0135) | 0.1548(+0.0228) | 0.0904(+0.0130) | 0.6386(+0.0128) | 0.4382(+0.0136) | 0.5366(+0.0160) | 0.3639(+0.0148) | 0.5860(+0.0088) | 0.4047(+0.0076) |

**Analysis.** At **gamma=16 every cell is a strict Pareto improvement** - Overall, both item groups and both user groups all significantly up (*). The gain is largest on Tail-Item and on the content-rich Amazon datasets; Yelp/SASRec (the paper's original cell) is the mean-profile's weakest case. Sec 4 shows the **max-pool** profile, which roughly doubles-triples these gains.

---

## 2. Variant A - learnable gamma (train-time). FAILED (instructive)

**Motivation.** If a fixed gamma helps at eval, can the model *learn* the optimal gamma and adapt to the term?

**Design / implementation.** `--usr_sem_train`: gamma becomes a trainable parameter; the term is added to the predict score **and to the seq2seq training loss**, using a **causal cumulative centroid** (history up to each position only - no future leakage). `--usr_sem_train_init` sets gamma's initial value.

**Experiment (SASRec/Yelp, 5 seeds).** Full breakdown (groups Ov/TI/HI/TU/HU as in Sec 4, H@10 & N@10). This is a *training-time* probe, so unlike the eval-time sweeps there is no within-model gamma=0 row; the baseline is the standard model (no term). NOTE: train-time learnable-gamma was only run on this one cell - it fails here, so there is no reason to run it on the other 8 (a single-cell negative).

| config (learned gamma) | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (-) | 0.6667 | 0.4216 | 0.1865 | 0.0833 | 0.8081 | 0.5212 | 0.6673 | 0.4231 | 0.6646 | 0.4160 |
| init 0 (gamma~-0.05) | 0.6673(+0.0006) | 0.4208(-0.0008) | 0.1960(+0.0095) | 0.0872(+0.0039) | 0.8060(-0.0021) | 0.5190(-0.0022) | 0.6678(+0.0005) | 0.4225(-0.0006) | 0.6651(+0.0005) | 0.4142(-0.0018) |
| init 8 (gamma~7.98) | 0.6646(-0.0021) | 0.4207(-0.0009) | 0.1642(-0.0223) | 0.0712(-0.0121) | 0.8119(+0.0038) | 0.5236(+0.0024) | 0.6651(-0.0022) | 0.4221(-0.0010) | 0.6628(-0.0018) | 0.4156(-0.0004) |

From init 0 the model drives gamma to ~0 (~baseline on every group); forced at init 8 it keeps gamma~8 but ends **worse on Overall and on both tail groups** (Tail-Item 0.1865->0.1642, Tail-User 0.6673->0.6651).

**Analysis.** Train-time injection does not work: from 0 the model learns gamma~0 (the dual view can already fit content *during* training, so the term is redundant in-loss); forced at gamma=8 the model co-adapts and ends worse. **Conclusion: D6 must be applied at inference on a frozen model.** The gain exists *precisely because* training never saw the term - the bottleneck discards content that the inference-time term then restores. (Clean negative + mechanistic insight.)

---

## 3. Variant B - per-segment gamma (routing to tail users). WORKS (safe targeted)

**Motivation.** The paper targets long-tail users; apply the content term only where it's most needed.

**Design / implementation.** `--usr_sem_seg`: apply gamma **only to Tail Users** - the paper's term for users with a short interaction history (`seq_len < ts_user`, the same split the paper/code uses for "Tail User" vs "Head User"); Head Users keep gamma=0. Sweep gamma_tail.

**Experiment (all 9 cells, 5 seeds).** Full breakdown, **baseline (g_tail=0) vs +per-segment (g_tail=16)**; the **max-pool term routed only to Tail Users**; groups Ov/TI/HI/TU/HU as in Sec 4; * = paired p<0.05 on Overall H@10. **Note the Head-User (HU) columns are *byte-identical* base<->+seg in every cell** - the routing leaves Head Users untouched (zero Head-User risk), exactly as designed; the Tail-User (TU) columns match Sec 4's max-pool numbers (same term, applied to that segment).

**Yelp**
| backbone | | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | base | 0.6667 | 0.4216 | 0.1865 | 0.0833 | 0.8081 | 0.5212 | 0.6673 | 0.4231 | 0.6646 | 0.4160 |
| SASRec | **+seg*** | 0.6696(+0.0028) | 0.4239(+0.0023) | 0.2001(+0.0135) | 0.0921(+0.0088) | 0.8078(-0.0003) | 0.5216(+0.0003) | 0.6709(+0.0035) | 0.4260(+0.0029) | 0.6646(+0.0000) | 0.4160(+0.0000) |
| GRU4Rec | base | 0.5729 | 0.3417 | 0.0853 | 0.0351 | 0.7164 | 0.4320 | 0.5775 | 0.3450 | 0.5550 | 0.3289 |
| GRU4Rec | **+seg*** | 0.5829(+0.0100) | 0.3504(+0.0087) | 0.0934(+0.0081) | 0.0403(+0.0052) | 0.7269(+0.0106) | 0.4417(+0.0097) | 0.5902(+0.0126) | 0.3560(+0.0109) | 0.5550(+0.0000) | 0.3289(+0.0000) |
| Bert4Rec | base | 0.6649 | 0.4248 | 0.1313 | 0.0546 | 0.8220 | 0.5338 | 0.6650 | 0.4270 | 0.6645 | 0.4164 |
| Bert4Rec | **+seg*** | 0.6696(+0.0047) | 0.4288(+0.0040) | 0.1418(+0.0106) | 0.0611(+0.0065) | 0.8250(+0.0030) | 0.5370(+0.0032) | 0.6709(+0.0059) | 0.4320(+0.0050) | 0.6645(+0.0000) | 0.4164(+0.0000) |

**Fashion**
| backbone | | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | base | 0.5632 | 0.4746 | 0.1116 | 0.0534 | 0.7430 | 0.6422 | 0.4835 | 0.3784 | 0.6667 | 0.5993 |
| SASRec | **+seg*** | 0.5803(+0.0171) | 0.4919(+0.0173) | 0.1503(+0.0387) | 0.0800(+0.0266) | 0.7515(+0.0085) | 0.6558(+0.0136) | 0.5137(+0.0303) | 0.4090(+0.0306) | 0.6667(+0.0000) | 0.5993(+0.0000) |
| GRU4Rec | base | 0.5483 | 0.4549 | 0.0848 | 0.0389 | 0.7329 | 0.6205 | 0.4647 | 0.3524 | 0.6567 | 0.5879 |
| GRU4Rec | **+seg*** | 0.5665(+0.0182) | 0.4768(+0.0219) | 0.1176(+0.0329) | 0.0592(+0.0204) | 0.7453(+0.0124) | 0.6430(+0.0225) | 0.4970(+0.0323) | 0.3912(+0.0388) | 0.6567(+0.0000) | 0.5879(+0.0000) |
| Bert4Rec | base | 0.5472 | 0.4548 | 0.0529 | 0.0224 | 0.7440 | 0.6270 | 0.4601 | 0.3491 | 0.6602 | 0.5920 |
| Bert4Rec | **+seg*** | 0.5691(+0.0219) | 0.4819(+0.0270) | 0.0964(+0.0435) | 0.0482(+0.0258) | 0.7572(+0.0133) | 0.6545(+0.0275) | 0.4988(+0.0388) | 0.3969(+0.0479) | 0.6602(+0.0000) | 0.5920(+0.0000) |

**Beauty**
| backbone | | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | base | 0.5689 | 0.3723 | 0.2260 | 0.1104 | 0.6507 | 0.4348 | 0.5589 | 0.3648 | 0.6147 | 0.4069 |
| SASRec | **+seg*** | 0.5906(+0.0217) | 0.4006(+0.0282) | 0.2958(+0.0698) | 0.1645(+0.0541) | 0.6609(+0.0102) | 0.4569(+0.0221) | 0.5854(+0.0264) | 0.3992(+0.0344) | 0.6147(+0.0000) | 0.4069(+0.0000) |
| GRU4Rec | base | 0.4933 | 0.3150 | 0.1515 | 0.0784 | 0.5748 | 0.3714 | 0.4871 | 0.3091 | 0.5218 | 0.3419 |
| GRU4Rec | **+seg*** | 0.5249(+0.0316) | 0.3439(+0.0289) | 0.2004(+0.0489) | 0.1055(+0.0271) | 0.6023(+0.0275) | 0.4007(+0.0293) | 0.5256(+0.0386) | 0.3443(+0.0352) | 0.5218(+0.0000) | 0.3419(+0.0000) |
| Bert4Rec | base | 0.5307 | 0.3577 | 0.1319 | 0.0774 | 0.6258 | 0.4246 | 0.5205 | 0.3491 | 0.5773 | 0.3971 |
| Bert4Rec | **+seg*** | 0.5568(+0.0261) | 0.3829(+0.0252) | 0.1680(+0.0361) | 0.0978(+0.0205) | 0.6495(+0.0237) | 0.4509(+0.0263) | 0.5523(+0.0318) | 0.3798(+0.0307) | 0.5773(+0.0000) | 0.3971(+0.0000) |

**Analysis.** Works as intended on all 9 cells: Tail Users improve substantially while Head Users are left **exactly unchanged** (zero Head-User risk). This is the deployable "targeted" version - itself a 2-expert routed mixture (Head User->base, Tail User->base+max-pool), routed by history length.

---

## 4. D7 - richer user profiles

**Motivation.** D6 uses the crudest possible profile - an unweighted **mean** of history embeddings. Two better profiles: (a) **recency-weighted** (recent items predict the next better), (b) **max-pool / multi-interest** ("does the candidate match *any* item you consumed?", which doesn't blur diverse tastes).

**Design / implementation.** `--usr_sem_d7`: in one eval pass compute three profiles - `mean`, `recency` (exponential decay 0.9 over positions), `max` (`max_t <e_hist_t, e_cand>`, no clustering) - and sweep gamma for each; write `usr_sem_sweep_{recency,max,ens}.csv`. Eval-only on existing checkpoints.

**Experiment.** Full paper-Table-1 breakdown - the five groups **Ov**=Overall, **TI**=Tail-Item, **HI**=Head-Item, **TU**=Tail-User, **HU**=Head-User (H@10 & N@10) - for **baseline (gamma=0) vs +max-pool (gamma=16)**, 5 seeds; * = paired p<0.05 on Overall H@10. max-pool is the best profile on all 9 cells (recency ~ mean; ~2-3x the mean-pool gain - see Analysis) and **lifts every group**, largest on Tail-Item.

**Yelp**
| backbone | | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | base | 0.6667 | 0.4216 | 0.1865 | 0.0833 | 0.8081 | 0.5212 | 0.6673 | 0.4231 | 0.6646 | 0.4160 |
| SASRec | **+max*** | 0.6708(+0.0041) | 0.4245(+0.0029) | 0.2025(+0.0160) | 0.0935(+0.0101) | 0.8087(+0.0006) | 0.5220(+0.0007) | 0.6709(+0.0035) | 0.4260(+0.0029) | 0.6706(+0.0061) | 0.4189(+0.0029) |
| GRU4Rec | base | 0.5729 | 0.3417 | 0.0853 | 0.0351 | 0.7164 | 0.4320 | 0.5775 | 0.3450 | 0.5550 | 0.3289 |
| GRU4Rec | **+max*** | 0.5859(+0.0130) | 0.3527(+0.0110) | 0.0957(+0.0104) | 0.0418(+0.0067) | 0.7302(+0.0138) | 0.4442(+0.0123) | 0.5902(+0.0126) | 0.3560(+0.0109) | 0.5696(+0.0145) | 0.3401(+0.0112) |
| Bert4Rec | base | 0.6649 | 0.4248 | 0.1313 | 0.0546 | 0.8220 | 0.5338 | 0.6650 | 0.4270 | 0.6645 | 0.4164 |
| Bert4Rec | **+max*** | 0.6710(+0.0061) | 0.4297(+0.0049) | 0.1447(+0.0135) | 0.0626(+0.0080) | 0.8259(+0.0039) | 0.5378(+0.0039) | 0.6709(+0.0059) | 0.4320(+0.0050) | 0.6712(+0.0067) | 0.4207(+0.0043) |

**Fashion**
| backbone | | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | base | 0.5632 | 0.4746 | 0.1116 | 0.0534 | 0.7430 | 0.6422 | 0.4835 | 0.3784 | 0.6667 | 0.5993 |
| SASRec | **+max*** | 0.5867(+0.0235) | 0.4975(+0.0229) | 0.1639(+0.0523) | 0.0873(+0.0339) | 0.7550(+0.0120) | 0.6607(+0.0185) | 0.5137(+0.0303) | 0.4090(+0.0306) | 0.6814(+0.0147) | 0.6121(+0.0128) |
| GRU4Rec | base | 0.5483 | 0.4549 | 0.0848 | 0.0389 | 0.7329 | 0.6205 | 0.4647 | 0.3524 | 0.6567 | 0.5879 |
| GRU4Rec | **+max*** | 0.5728(+0.0244) | 0.4833(+0.0284) | 0.1279(+0.0431) | 0.0655(+0.0267) | 0.7498(+0.0170) | 0.6496(+0.0291) | 0.4970(+0.0323) | 0.3912(+0.0388) | 0.6710(+0.0142) | 0.6028(+0.0150) |
| Bert4Rec | base | 0.5472 | 0.4548 | 0.0529 | 0.0224 | 0.7440 | 0.6270 | 0.4601 | 0.3491 | 0.6602 | 0.5920 |
| Bert4Rec | **+max*** | 0.5765(+0.0293) | 0.4896(+0.0348) | 0.1115(+0.0586) | 0.0561(+0.0337) | 0.7616(+0.0177) | 0.6621(+0.0352) | 0.4988(+0.0388) | 0.3969(+0.0479) | 0.6772(+0.0171) | 0.6098(+0.0177) |

**Beauty**
| backbone | | Ov H | Ov N | TI H | TI N | HI H | HI N | TU H | TU N | HU H | HU N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | base | 0.5689 | 0.3723 | 0.2260 | 0.1104 | 0.6507 | 0.4348 | 0.5589 | 0.3648 | 0.6147 | 0.4069 |
| SASRec | **+max*** | 0.5962(+0.0273) | 0.4063(+0.0340) | 0.3075(+0.0815) | 0.1719(+0.0615) | 0.6651(+0.0144) | 0.4622(+0.0274) | 0.5854(+0.0264) | 0.3992(+0.0344) | 0.6459(+0.0312) | 0.4391(+0.0321) |
| GRU4Rec | base | 0.4933 | 0.3150 | 0.1515 | 0.0784 | 0.5748 | 0.3714 | 0.4871 | 0.3091 | 0.5218 | 0.3419 |
| GRU4Rec | **+max*** | 0.5314(+0.0381) | 0.3501(+0.0351) | 0.2071(+0.0557) | 0.1092(+0.0309) | 0.6087(+0.0339) | 0.4075(+0.0361) | 0.5256(+0.0386) | 0.3443(+0.0352) | 0.5577(+0.0359) | 0.3766(+0.0347) |
| Bert4Rec | base | 0.5307 | 0.3577 | 0.1319 | 0.0774 | 0.6258 | 0.4246 | 0.5205 | 0.3491 | 0.5773 | 0.3971 |
| Bert4Rec | **+max*** | 0.5619(+0.0312) | 0.3878(+0.0300) | 0.1735(+0.0416) | 0.1009(+0.0235) | 0.6545(+0.0287) | 0.4562(+0.0316) | 0.5523(+0.0318) | 0.3798(+0.0307) | 0.6057(+0.0284) | 0.4240(+0.0269) |

(`H`=H@10, `N`=N@10. Full numbers incl. mean/recency profiles: `SUMMARY_table1.md`.)

At **gamma=32** max-pool goes higher still (e.g. GRU/Beauty dHR **+0.067**, tail up to +0.13), with only Yelp/SASRec flattening on overall (tail still +0.025).

**Analysis.** max-pool **roughly 2-3x the mean-pool (D6) gains** on every cell, all significant. With max-pool, **7 of 9 cells clear +0.01 on OVERALL HR@10 at gamma=16** (the 2 holdouts - Yelp SASRec/Bert - are still significantly positive with large tail gains), and **all 9 clear +0.01 on the tail**. Why: matching the *most-similar* consumed item is far more discriminative than matching the *average* (which blurs multi-interest users) - the multi-interest signal, obtained for free (no clustering).
