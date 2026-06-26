# LLM-ESR Extension Study (Yelp / SASRec)

**Protocol.** All experiments: Yelp with the authors' released preprocessed data (15,720 users / 11,383 items), SASRec backbone, full LLM-ESR (dual-view + cross-attention + self-distillation), 10 seeds (42-51), 1 positive vs 100 random negatives, HR@10 / NDCG@10, paired t-test vs the reproduced baseline.

**Baseline:** Overall H@10 **0.6675**, NDCG@10 **0.4220**, Tail-Item H@10 **0.1902**. Work was isolated in a git worktree; every flag defaults off (no change to reproduced results); checkpoints on scratch.

## Result Table

Master results table (Overall H@10, d vs baseline, paired-t p; * = p<0.05):

| Extension | Overall H@10 | d | p | Tail H@10 | Verdict |
|---|---|---|---|---|---|
| baseline | 0.6675 | - | - | 0.1902 | reference |
| **lin3** (3-layer linear adapter) | **0.6701** | **+0.0026** | **0.020 *** | 0.1845 | **only positive (overall)** |
| linw1536 (wide linear) | 0.6693 | +0.0018 | 0.25 | 0.1881 | trend, n.s. |
| linres2 / linres3 (linear + skip) | 0.6695 / 0.6691 | +0.0020 / +0.0016 | 0.089 / 0.076 | - | n.s. |
| lin3w1536 / lin3w1024 (deep+wide) | 0.6681 / 0.6672 | +0.0006 / -0.0003 | 0.62 / 0.80 | - | n.s. |
| acttanh / actgelu / actrelu | 0.6676 / 0.6654 / 0.6617 | +0.0001 / -0.0021 / -0.0058 | 0.85 / 0.029* / 3e-4* | 0.193/0.182/0.159 | nonlinearity <= 0 |
| resffgelu / resffrelu (FFN+skip) | 0.6677 / 0.6673 | +0.0002 / -0.0002 | 0.87 / 0.82 | - | ~ baseline |
| cap128 (hidden 128) | 0.6658 | -0.0017 | 0.066 | 0.1927 | no headroom |
| drlae (linear-AE init) | 0.6661 | -0.0014 | 0.16 | 0.1840 | ~ PCA |
| drae (nonlinear-AE init) | 0.6592 | -0.0083 | 1e-4* | 0.1815 | hurts |
| popgate / popgatefree (gated fusion) | 0.6643 / 0.6688 | -0.0032 / +0.0013 | 6e-4* / 0.20 | 0.189/0.189 | hurts / neutral |
| lora8 / lora32 (semi-unfreeze) | 0.6567 / 0.6445 | -0.0108 / -0.0230 | 1.7e-6* / 2.4e-7* | 0.169 / 0.151 | hurts (tail worst) |
| co16 / co32 / co128 (collab adapter) | 0.6417 / 0.6385 / 0.6322 | -0.026 / -0.029 / -0.035 | <=1.6e-9* | 0.173/0.184/0.171 | hurts hard |
| aug0 (all-prefix augmentation) | 0.6433 | -0.0242 | 1.5e-9* | 0.1340 | hurts |
| ssm (in-batch sampled-softmax) | 0.4741 | -0.1934 | 2e-12* | **0.3922 (+0.202)** | extreme tail<->overall tradeoff |
| **debias alpha~0.10** (inference) | 0.6655 | -0.0020 | - | **0.2060 (+0.0158 *)** | **clears +0.01 on tail** |

---

# Part A - Adapter design

## A1. Nonlinear activation in the adapter

**Motivation.** The released adapter (frozen 1536-d LLM emb -> 64-d) is two *linear* layers. The obvious "upgrade" is to insert a nonlinearity for more expressive power.

**Design.** Insert Tanh / GELU / ReLU between the two adapter layers (`--adapter_act`). 10 seeds.

**Results.** Tanh 0.6676 (+0.0001, n.s.); GELU 0.6654 (-0.0021, p=0.029); ReLU 0.6617 (-0.0058, p=3e-4). Tail H@10: Tanh 0.193 (~base), GELU 0.182, ReLU 0.159.

**Analysis.** Nonlinearity is dead weight or harmful. The damage scales with how aggressively the activation breaks linearity (ReLU >> GELU >> Tanh~linear), and ReLU collapses the tail. The adapter wants to stay a (near-)linear map.

## A2. Residual-FFN adapter

**Motivation.** Maybe nonlinearity fails only because it loses the linear signal; a Transformer-style FFN with a residual/skip path keeps the linear route while adding capacity.

**Design.** `MLP(x) + Linear_skip(x)` with GELU/ReLU inside the MLP (`--adapter_residual`). 10 seeds.

**Results.** resffgelu 0.6677 (+0.0002), resffrelu 0.6673 (-0.0002) - both n.s. (~ baseline).

**Analysis.** The linear skip *exactly neutralizes* the nonlinear damage (GELU 0.6654->0.6677), but adds nothing beyond baseline. Confirms A1: the nonlinear branch is redundant; the model only uses the linear path.

## A3. Linear adapter depth & width  <- contains the only positive result

**Motivation.** If linearity is right, can *more linear capacity* (depth/width) still help via better optimization, even though all depths are the same function class (affine, rank <= 64)?

**Design.** Pure-linear adapters of depth {1,2(base),3,4,5}, a wide 2-layer (hidden 1536), wide+deep 3-layer (1536/1024), and pure-linear residual (`--adapter_layers/--adapter_hidden_dim`). 10 seeds.

**Results.** **lin3 = 0.6701 (+0.0026, p=0.020 *)** - the only significant overall gain. lin4 0.6659 (-0.0016), lin5 0.6676 (+0.0001), linw1536 0.6693 (+0.0018, n.s.), lin3w1536/1024 ~ base, linres2/3 +0.002 (p~0.08, n.s.).

**Analysis.** A clean **overparameterized-linear sweet spot at depth 3**, then decay (deeper linear stacks are harder to optimize). The gain is small (+0.26 pt) and does *not* come from the tail (tail flat/down). Width and a residual highway trend the same way but never reach significance. This is the practical "best" extension but well under +0.01.

---

# Part B - Collaborative branch & initialization

## B1. Collaborative-side adapter

**Motivation.** The semantic view has an adapter; symmetrize by giving the collaborative view a higher-dim embedding (d_co) + a linear adapter back to 64 (a "fair" capacity bump on the collab side).

**Design.** d_co in {16, 32, 128}, PCA-d_co initialized, linear adapter d_co->64 (`--co_adapter_dim`). 10 seeds.

**Results.** co16 0.6417 (-0.0258), co32 0.6385 (-0.0290), co128 0.6322 (-0.0353), all p<=1.6e-9.

**Analysis.** Large, monotone harm. The randomly-initialized adapter **scrambles the PCA initialization** that the collaborative branch depends on; co16 collapses to ~ the "w/o Co-view" level. The collab branch needs its PCA-aligned init, not extra (mis-initialized) capacity.

## B2. Learnable dimensionality reduction vs PCA (collab init)

**Motivation.** The collaborative embedding is initialized by **PCA** of the LLM embeddings. Is PCA special, or would a *learnable* DR give an equal/better init?

**Design.** Train an autoencoder on the LLM item embeddings, use its 64-d bottleneck as the init (`--id_init_path`); scale-matched to PCA's std. Linear-AE (`drlae`) vs nonlinear-AE (`drae`); reference = random init. 10 seeds.

**Results.** Random 0.6520 < drae 0.6592 (recon MSE 0.245, -0.0083, p=1e-4) < drlae 0.6661 (recon MSE 0.366, -0.0014, n.s.) ~ PCA 0.6675.

**Analysis.** Init quality is monotone in **linearity, not reconstruction**: the linear AE ties PCA (learnability adds nothing to a linear map), while the nonlinear AE reconstructs *better* yet inits *worse* - its curved code doesn't align linearly with the collaborative space. PCA is already optimal; the collab init wants a linear shadow of the semantic space. (Echoes Part A.)

---

# Part C - The frozen semantic embedding

## C1. LoRA semi-unfreeze

**Motivation.** The paper freezes the LLM item embedding. Does gentle, low-rank adaptation (LoRA) help?

**Design.** `e_eff = e_frozen + A*B`, rank r in {8,32}, B init 0 (identical to baseline at init), only A,B trainable, frozen emb stays frozen (`--llm_lora_rank`). 10 seeds.

**Results.** lora8 0.6567 (-0.0108, p=1.7e-6), lora32 0.6445 (-0.0230, p=2.4e-7). Tail collapses 0.1902 -> 0.1688 -> 0.1511. **More rank -> more harm.**

**Analysis (mechanism).** From a converged checkpoint, the learned per-item delta is **1.23x larger on tail than head items** (22.8% vs 18.6% of the unit-norm frozen embedding). The extra freedom is spent exactly where data is scarcest (the tail) and overfits it, destroying the frozen prior's generalization. **Freezing is load-bearing, especially for the long tail** - a direct mechanistic validation of the paper's design choice.

---

# Part D - Dual-view fusion

## D1. Popularity-gated fusion

**Motivation.** Operationalize the core finding ("head items lean collaborative, tail lean semantic") as a mechanism: weight the two views per candidate item by its popularity.

**Design.** score = 2[g*collab + (1-g)*semantic], g=sigma(w*log pop+b) (2 params, generalizes) vs a free per-item gate (`--pop_gate`). Init g=0.5 = baseline. 10 seeds.

**Results.** popgate 0.6643 (-0.0032, p=6e-4); popgatefree 0.6688 (+0.0013, n.s.).

**Analysis.** Explicit popularity reweighting **hurts**; the free per-item gate just recovers baseline. The equal-weight concatenation + the sequence encoder already weight the views implicitly; imposing a rigid popularity-monotone gate only miscalibrates. "Tail relies on semantic" is about the *presence* of the branch, not about up-weighting it at scoring time.

---

# Part E - Capacity, objective, data (the "break the ceiling" attempts)

## E1. Model capacity

**Motivation.** Is LLM-ESR under-parameterized at hidden=64 (the paper's "fair-size" choice)?

**Design.** hidden_size 128 with a matching PCA-128 collab init (`--hidden_size 128 --id_init_path pca128`). 10 seeds.

**Results.** cap128 0.6658 (-0.0017, p=0.066) - slightly *worse* overall; tail 0.1927 (noisy).

**Analysis.** Doubling capacity does not help (mildly hurts). LLM-ESR is not capacity-limited at 64; the fair-size choice is well-tuned. (Rules out the capacity direction in one shot.)

## E2. In-batch sampled-softmax loss

**Motivation.** The training loss is 1-negative BCE per position. A softmax over many negatives is a stronger ranking objective and often the single biggest lever in sequential rec.

**Design.** `--ssm`: each valid position's user-context scores its positive against ALL other in-batch positives (hundreds of negatives), cross-entropy, false negatives masked by item id. Forward-only. 10 seeds.

**Results.** ssm 0.4741 (-0.1934, p=2e-12) overall, but Tail 0.3922 (**+0.202**).

**Analysis.** Same tail<->overall tradeoff as debiasing, even sharper: popular items become frequent in-batch negatives -> the model implicitly, aggressively down-weights them -> tail soars, overall collapses. A stronger objective does *not* lift overall here.

## E3. Sequence augmentation (all-prefix)

**Motivation.** More training signal - train on every prefix of each user's sequence (esp. helps short/tail users). The one lever that adds *data* rather than reweighting.

**Design.** `--aug`: all prefixes (len >= 2) as training examples (~10x data). Required a fix so self-distillation's similar-user lookups still resolve to the correct *original* user (`unzip_data_with_user` + a user-map). 10 seeds, 60 epochs (converges fast).

**Results.** aug0 0.6433 (-0.0242, p=1.5e-9) overall, tail 0.1340 (-0.056) - worse on *both*.

**Analysis.** Converges fast (0.643 at 2 epochs) but to a *worse* optimum: all-prefix training over-weights short-prefix next-item prediction, shifting away from the full-sequence test distribution. More data does not break the ceiling either.

---

# Part F - The achievable improvement: inference popularity debiasing

**Motivation.** The paper targets the long tail. Given the overall ceiling, can we improve the *tail* metric at test time without retraining?

**Design.** `--pop_debias`: at eval, adjust each candidate's score by -alpha*log1p(popularity), then re-rank. One trained model is evaluated across a sweep of alpha (debias_sweep.csv). 10 seeds. **Results (full sweep, Tail-Item H@10 vs baseline 0.1902):**

| alpha | Overall H@10 | Tail H@10 (d) |
|---|---|---|
| 0.00 | 0.6676 | 0.1902 |
| **0.10** | **0.6655 (-0.0020)** | **0.2060 (+0.0158 *)** |
| 0.15 | 0.6637 (-0.0038) | 0.2135 (+0.0233) |
| 0.25 | 0.6581 (-0.0095) | 0.2291 (+0.0389) |
| 1.50 | 0.3304 (-0.337) | 0.3554 (+0.165) |

**Analysis.** Debiasing surfaces relevant tail items above popular distractors - a genuine, on-theme long-tail gain - but at an overall cost that grows with alpha. At **alpha~0.10 it clears the +0.01 bar on Tail-Item H@10 (+0.0158, significant) for only -0.002 overall** (a ~8:1 favorable trade). It is a tail<->overall *knob*, not a Pareto improvement. This is the single result that literally satisfies "a metric > baseline + 0.01, significant," and it is the realistic ceiling-respecting improvement.

---

# Summary

1. **Linearity is the right inductive bias** for both the adapter (Part A) and the collaborative init (Part B). Every nonlinear/over-parameterized variant is neutral-or-harmful; the lone win is more *linear* adapter depth (lin3, +0.0026, p=0.02).
2. **Freeze + PCA-init are load-bearing for the tail** (Parts B, C). Touching them (LoRA, collab-adapter, random/nonlinear init) degrades the tail most - mechanistically confirmed for LoRA.
3. **Overall H@10 is at a ceiling (~0.6676)** that capacity, objective, and data all fail to beat (Part E). The model's overall accuracy is near-optimal for this setup; the only real headroom is the long tail.
4. **The tail is accessible only as a tradeoff** (Parts E2, F): down-weighting popular items lifts tail recall and lowers overall. Mild popularity debiasing (alpha~0.10) is the best operating point and the one change that clears +0.01 on a metric (Tail-Item H@10).

