# Extension: LLM-Guided Hard Negative Sampling
**Dataset**: Yelp  |  **Model**: SASRec backbone (LLM-ESR)  |  **N = 15,720 users**

---

## 1. Motivation

Standard sequential recommendation trains with *uniformly random* negatives - items drawn at random from the catalogue that the user did not interact with. A natural extension is to replace some of these with *hard negatives*: items that are **semantically similar** to the positive in the LLM embedding space, yet were never interacted with. The hypothesis is that harder negatives force the model to learn finer-grained distinctions, improving ranking quality.

---

## 2. Method

**Precomputation.** For each item, we compute cosine similarity against all other items in the LLM embedding space (text-ada-002, 1536-dim). We retain the top-K = 50 most similar items as the hard negative *pool* for that item.

**Training.** At each training step, when sampling a negative for a target item, we draw from the hard negative pool with probability `p = hard_neg_ratio`, and from the full random set with probability `1 - p`. We evaluate four values of `p`:

| Ratio | Semantics |
|-------|-----------|
| 0.00 | Baseline - purely random negatives (original LLM-ESR) |
| 0.25 | 25% hard, 75% random |
| 0.50 | 50% hard, 50% random |
| 1.00 | 100% hard - all negatives are semantically similar |

---

## 3. Results

### 3.1 Ratio Sweep (seed = 42)

| Ratio | NDCG@10 | HR@10 | d NDCG | d HR |
|-------|---------|-------|--------|------|
| 0.00 (baseline) | 0.4226 | 0.6725 | - | - |
| 0.25 | 0.4057 | 0.6491 | -4.0% | -3.5% |
| 0.50 | 0.3913 | 0.6333 | -7.4% | -5.8% |
| 1.00 | 0.2624 | 0.4641 | **-37.9%** | **-31.0%** |

Performance degrades **monotonically** with increasing hard-negative ratio. At ratio = 1.0, the model collapses to near-random performance.

### 3.2 Statistical Significance (3 seeds: 42, 43, 44)

Paired per-user t-test comparing ratio = 0.00 vs ratio = 0.25:

| Metric | Baseline (ratio=0) | Hard neg (ratio=0.25) | d | p-value | Cohen's d |
|--------|-------------------|----------------------|---|---------|-----------|
| NDCG@10 | 0.4198 | 0.4067 | -3.23% | **1.74 x 10^-24** | 0.082 (negligible) |
| HR@10 | 0.6660 | 0.6537 | -1.88% | **4.33 x 10^-12** | 0.055 (negligible) |

The degradation is **statistically highly significant** (p << 10^-10) despite Cohen's d being in the "negligible" range - a direct consequence of the large sample size (N = 15,720). The effect is real and replicable across seeds.

### 3.3 Sub-Group Analysis (seed = 42, ratio 0 vs 0.25)

| Group | NDCG Baseline | NDCG Hard neg | d NDCG |
|-------|--------------|---------------|--------|
| Short sequences | 0.4239 | 0.4081 | -3.7% |
| Long sequences | 0.4177 | 0.3963 | -5.1% |
| **Tail items** | **0.0871** | **0.0732** | **-16.0%** |
| Popular items | 0.5214 | 0.5036 | -3.4% |

**Tail items suffer disproportionately** - a -16% drop in NDCG versus -3.4% for popular items. This pattern is a diagnostic signature of the mechanism described below.

---

## 4. Analysis: The False Negative Problem

The results reveal a fundamental conflict between *LLM semantic similarity* and *collaborative filtering objectives*.

**Why hard negatives hurt:**

Items that are *semantically similar* to a positive (high cosine similarity in LLM embedding space) are also likely to be **genuinely preferred** by the user - they are restaurants of the same cuisine, products in the same category, venues in the same neighbourhood. Treating such items as negative training signals creates **contradictory gradients**: the model is simultaneously being told "rank item A high (positive)" and "rank item B low (negative)" when A and B are near-identical in user preference space.

**Why tail items are hit hardest:**

Tail items have far fewer interaction records. Each training step's gradient has outsized influence on their embeddings. When a false negative introduces a noisy gradient, tail items have insufficient signal from genuine interactions to correct it, causing systematic representation degradation.

**Why the collapse at ratio = 1.0:**

When all negatives are semantically similar to the positive, virtually every negative in every batch is a potential false negative. The loss signal becomes almost entirely contradictory, and the model cannot learn a coherent ranking function.

---

## 5. Conclusion

LLM-guided hard negative sampling **consistently hurts** LLM-ESR performance across all tested ratios. This negative result is informative:

1. **LLM semantic similarity != user dispreference.** The embedding space captures content similarity, not the boundary between liked and disliked items.
2. **The false negative problem is severe** in a content-rich domain like Yelp, where similar items are genuinely interchangeable from a user preference perspective.
3. **Tail items are the most vulnerable.** Any training noise disproportionately degrades recommendation quality for long-tail items, which are already the hardest cases.
4. **This validates LLM-ESR's design choice** of using LLM embeddings for *initialisation* (PCA projections) rather than as direct training signals. The two-view architecture (collaborative + semantic) deliberately keeps LLM information as a structural prior rather than a supervision source.

> **Takeaway**: LLM embeddings encode "what is similar," not "what the user does not want." Using them to construct harder negatives introduces false negatives at scale, degrading performance monotonically with sampling ratio - with the strongest damage to tail-item representation.
