# LLM-ESR Reproduction and Extension Inventory

This file is the canonical presentation inventory for the material currently in this
folder. It separates:

1. faithful reproduction work;
2. methodological extensions;
3. supporting ablations and negative results;
4. results that are promising but not yet strong enough for a headline claim.

## Source-of-truth rules

- Paper-comparable claims use the authors' released preprocessed datasets:
  Yelp (15,720 users / 11,383 items), Fashion, and Beauty.
- The 10-seed results in [10-SEED UPDATE.md](10-SEED%20UPDATE.md) supersede the
  earlier 3-seed reproduction and SASRec ablation tables.
- For adapter interpretation, later significance analysis supersedes conclusions
  based only on the best 3-seed mean.
- Results on the separately preprocessed 441,393-user Yelp dataset are kept as a
  separate experimental setting and must not be numerically compared with the
  paper's tables.
- The folder contains result summaries, not the underlying checkpoints, logs, or
  most referenced CSV files. Claims below therefore organize the supplied evidence;
  they do not independently recompute it from raw runs.

## A. Reproduction work

### R1. Raw-data preprocessing investigation

- Re-running preprocessing on the current Yelp dump produced about 441,000 users
  instead of the paper's 15,720 and much higher sampled-ranking scores
  (SASRec HR@10 around 0.90).
- This identified an important reproducibility dependency: the paper's released
  preprocessed files are required for numerical comparability.
- Status: important reproducibility caveat, not an extension.
- Source: [Reproduction 1.md](Reproduction%201.md).

### R2. Faithful three-backbone reproduction

- Re-ran LLM-ESR with SASRec, BERT4Rec, and GRU4Rec using the authors' data.
- The final 10-seed overall results reproduce the paper within roughly 0.003 HR@10:
  SASRec 0.6676, BERT4Rec 0.6648, GRU4Rec 0.5710.
- Status: successful reproduction.
- Sources: [Reproduction 2.md](Reproduction%202.md) and
  [10-SEED UPDATE.md](10-SEED%20UPDATE.md).

### R3. Table-2 component ablation reproduction

- Reproduced removal of Co-view, Se-view, self-distillation, parameter sharing,
  cross-attention, the second adapter layer, and the PCA-based initialization.
- The component directions reproduce, with Co-view and Se-view remaining the
  largest contributors.
- The paper's dramatic one-layer-adapter drop does not reproduce: the observed
  drop is about 1.1%, not 8.5%.
- The supplied report adds the analytical explanation: without an activation,
  two affine adapter layers have the same function class as one affine layer, so
  only an optimization difference should remain.
- Self-distillation is significant before correction (p=0.016 by seed) but not
  after a seven-comparison Bonferroni threshold.
- Status: successful reproduction with two substantive qualifications.
- Sources: [Ablation Study 1.md](Ablation%20Study%201.md),
  [10-SEED UPDATE.md](10-SEED%20UPDATE.md), and
  [TTEST_ANALYSIS_EN.md](TTEST_ANALYSIS_EN.md).

### R4. Stronger statistical protocol

- Extended major reproduction and SASRec ablation conditions from 3 to 10 seeds.
- Added paired tests, variance reporting, and multiple-comparison reasoning.
- Status: experimental-rigor contribution.
- Source: [10-SEED UPDATE.md](10-SEED%20UPDATE.md).

## B. Extension family 1: adapter architecture and inductive bias

### E1. Activation-function sweep

- Tested linear/no activation, Tanh, GELU, and ReLU.
- SASRec: Tanh is effectively a no-op; GELU and ReLU hurt, especially on tail
  items.
- GRU4Rec: linear is clearly best; GELU and ReLU significantly hurt.
- BERT4Rec: GELU/Tanh show small NDCG trends, but no nonlinear choice delivers a
  consistent, practically meaningful win across HR and NDCG. The conservative
  conclusion is that linear remains the reliable default.
- Main insight: stronger nonlinearities cause larger tail-item damage; ReLU is the
  least safe choice.
- Status: strong architecture finding, mostly negative for nonlinear adapters.
- Sources: [ADAPTER_ANALYSIS_EN.md](ADAPTER_ANALYSIS_EN.md),
  [TTEST_ANALYSIS_EN.md](TTEST_ANALYSIS_EN.md), and
  [10-SEED UPDATE.md](10-SEED%20UPDATE.md).

### E2. Adapter width sweep

- Tested intermediate widths 64, 128, 256, 512, and 768.
- SASRec improves toward the original width 768.
- GRU4Rec saturates around 512-768.
- BERT4Rec's best overall result is near 768, while width 128 can be slightly
  better for tail-item NDCG.
- Status: mixed supporting result; the original width is already close to optimal.
- Source: [ADAPTER_ANALYSIS_EN.md](ADAPTER_ANALYSIS_EN.md).

### E3. Adapter depth sweep

- Tested one to five pure-linear layers and deeper/wider combinations.
- SASRec has a three-layer linear sweet spot:
  HR@10 0.6676 to about 0.6701, delta +0.0026, p=0.020 over 10 seeds.
- The gain is an overall optimization benefit, not a long-tail gain; tail HR is
  slightly lower.
- BERT4Rec is best around two layers; GRU4Rec becomes unstable with three layers.
- Status: small but statistically positive overall result for SASRec.
- Sources: [extension-log-0618.md](extension-log-0618.md) and
  [ADAPTER_ANALYSIS_EN.md](ADAPTER_ANALYSIS_EN.md).

### E4. Residual and skip adapters

- Tested a direct linear skip around the adapter and nonlinear FFN-plus-skip
  variants.
- GRU4Rec gains significantly from a linear residual connection
  (about +1.18% NDCG, p<0.001).
- SASRec shows a small positive trend, but the later 10-seed result is not
  significant.
- BERT4Rec receives no reliable benefit.
- Nonlinear FFN-plus-skip variants on SASRec return to roughly baseline but do not
  improve it.
- Status: backbone-specific positive result for GRU4Rec.
- Sources: [ADAPTER_ANALYSIS_EN.md](ADAPTER_ANALYSIS_EN.md),
  [TTEST_ANALYSIS_EN.md](TTEST_ANALYSIS_EN.md), and
  [extension-log-0618.md](extension-log-0618.md).

## C. Extension family 2: initialization, capacity, and view design

### E5. Collaborative-side adapter

- Added a higher-dimensional collaborative embedding followed by an adapter back
  to 64 dimensions.
- Dimensions 16, 32, and 128 all hurt badly and monotonically.
- Interpretation: the extra adapter disrupts the PCA-aligned initialization.
- Status: negative result.
- Source: [extension-log-0618.md](extension-log-0618.md).

### E6. Learned dimensionality reduction instead of PCA

- Replaced PCA initialization with linear and nonlinear autoencoder bottlenecks.
- Linear autoencoder nearly ties PCA; nonlinear autoencoder reconstructs better
  but recommends worse; random initialization is worst.
- Insight: linear alignment matters more than reconstruction error, and PCA is
  already a strong choice.
- Status: negative result that validates the original PCA design.
- Source: [extension-log-0618.md](extension-log-0618.md).

### E7. LoRA semi-unfreezing of semantic embeddings

- Added low-rank trainable deltas with ranks 8 and 32 to the frozen item
  embeddings.
- Both hurt; higher rank hurts more, with the largest damage on tail items.
- The learned update is reported as 1.23 times larger on tail than head items,
  indicating overfitting where interaction data are scarcest.
- Status: strong negative result validating frozen embeddings.
- Source: [extension-log-0618.md](extension-log-0618.md).

### E8. Popularity-gated dual-view fusion

- Tested a learned monotonic popularity gate and a free per-item gate between
  collaborative and semantic views.
- The monotonic gate hurts; the free gate is approximately neutral.
- Interpretation: a rigid training-time popularity rule miscalibrates the views.
- Status: negative/mixed result.
- Source: [extension-log-0618.md](extension-log-0618.md).

### E9. Larger recommender hidden size

- Increased hidden size from 64 to 128 with matching PCA initialization.
- Overall HR@10 slightly decreases; no capacity headroom is found.
- Status: negative result.
- Source: [extension-log-0618.md](extension-log-0618.md).

## D. Extension family 3: self-distillation and teacher retrieval

### E10. Similarity-weighted teacher pooling

- Replaced mean pooling of retrieved teachers with similarity-weighted pooling.
- Performance decreases.
- Status: negative result.
- Source: [self-distillation.md](self-distillation.md).

### E11. Stochastic teacher sampling

- Replaced fixed top-10 teachers with 10 random teachers sampled from the top-50
  semantic-neighbor pool.
- Results are effectively neutral and do not improve tail users.
- Status: neutral result.
- Source: [self-distillation.md](self-distillation.md).

### E12. Teacher-count sweep

- Tested top-k and stochastic retrieval with k in {5, 10, 20, 30}.
- More teachers generally hurt; top-k 10 remains the most reliable setting.
- Stochastic k=5 gives the highest mean overall HR in the small 3-seed table, but
  does not improve the target tail-user NDCG and is not established as significant.
- Status: no confirmed improvement.
- Source: [sd_teacher_alter.md](sd_teacher_alter.md).

### E13. Hybrid semantic/collaborative teacher retrieval

- Added collaborative re-ranking to teacher retrieval.
- Overall metrics are nearly unchanged and tail-user NDCG is slightly lower.
- Status: neutral result.
- Source: [sd_hybrid_retrieval.md](sd_hybrid_retrieval.md).

## E. Extension family 4: training data, objective, and negative sampling

### E14. LLM-guided hard-negative sampling

- Sampled negatives from the top-50 nearest items in LLM embedding space with
  hard-negative ratios 0.25, 0.50, and 1.00.
- Performance degrades monotonically; ratio 1.00 nearly collapses the model.
- Tail items are hurt most.
- Main insight: semantic similarity indicates plausible user preference, not
  dispreference, so these "hard negatives" are often false negatives.
- Status: strong negative result with a clear mechanism.
- Source: [HARDNEG_EXTENSION_EN.md](HARDNEG_EXTENSION_EN.md).

### E15. In-batch sampled-softmax objective

- Replaced the original one-negative ranking loss with an in-batch softmax over
  many negatives.
- Overall HR collapses from about 0.668 to 0.474, while tail-item HR rises by about
  0.202.
- Status: extreme tail-vs-overall trade-off, not a deployable overall improvement.
- Source: [extension-log-0618.md](extension-log-0618.md).

### E16. All-prefix sequence augmentation

- Trained on every prefix of each user sequence, increasing the training set by
  roughly ten times.
- Overall and tail performance both worsen.
- Interpretation: the augmented training distribution no longer matches the
  full-sequence test distribution.
- Status: negative result.
- Source: [extension-log-0618.md](extension-log-0618.md).

## F. Extension family 5: inference-time reranking and content recovery

### E17. Popularity-debiased inference

- Subtracted an item-popularity term from candidate scores at inference.
- At alpha around 0.10, tail-item HR@10 rises by about +0.0158 for only about
  -0.002 overall HR.
- Stronger debiasing creates larger tail gains and larger overall losses.
- Status: useful controllable tail-vs-overall trade-off, not a Pareto improvement.
- Source: [extension-log-0618.md](extension-log-0618.md).

### E18. D6 mean-profile user-semantic scoring

- Added a direct full-dimensional content-match term between a candidate item and
  the mean LLM embedding of the user's history.
- It is applied only at inference to a frozen checkpoint; no retraining is needed.
- At gamma=16 it improves all reported groups across all nine
  backbone-by-dataset cells, with the largest gains on tail items and Amazon.
- Status: primary positive extension.
- Source: [LLM-ESR_D6_summary.md](LLM-ESR_D6_summary.md).

### E19. Train-time learnable semantic-score weight

- Made the D6 weight trainable and inserted the term into training.
- From a zero initialization the model learns a near-zero weight; from a forced
  positive initialization it co-adapts and performs worse.
- Insight: the content term works because it restores information after training,
  not because it should be absorbed by the training loss.
- Status: instructive negative variant.
- Source: [LLM-ESR_D6_summary.md](LLM-ESR_D6_summary.md).

### E20. Tail-user-only semantic-score routing

- Applied the max-pool content term only to users with short histories.
- Tail-user performance improves across all nine cells while head-user outputs are
  exactly unchanged.
- Status: primary positive targeted/fairness extension with zero head-user risk.
- Source: [LLM-ESR_D6_summary.md](LLM-ESR_D6_summary.md).

### E21. D7 richer user profiles

- Compared mean pooling, recency-weighted pooling, and max-pool/multi-interest
  matching.
- Max-pool is best in all nine cells and roughly doubles or triples the mean-profile
  gain.
- At gamma=16, 7/9 cells gain more than +0.01 overall HR@10 and all nine gain more
  than +0.01 on tail-item HR@10.
- Status: strongest universal positive extension in the folder.
- Source: [LLM-ESR_D6_summary.md](LLM-ESR_D6_summary.md).

## G. Extension family 6: two-specialist routing over the dual views

### E22. Expert decomposition diagnostic

- Decomposed the original score exactly into collaborative and semantic expert
  scores, plus the full model and full+max variants.
- Across user-by-item head/tail cells, tail items are generally best served by the
  semantic expert alone, while popular items favor full+max.
- Mechanism: collaborative embeddings for cold items behave like weakly trained
  noise.
- Status: strong diagnostic result that challenges fixed equal fusion.
- Source:
  [LLM-ESR Extension Two-Specialist Routing (a mixture over the dual views).md](LLM-ESR%20Extension%20Two-Specialist%20Routing%20%28a%20mixture%20over%20the%20dual%20views%29.md).

### E23. Per-candidate hard expert router

- Used each candidate's popularity to drop or down-weight the collaborative expert
  on cold items, on top of D7 max-pool.
- Significant on 7/9 cells and above +0.01 overall HR on 6/9.
- The strongest tuned gains are around +0.05 on Fashion and +0.06 on Beauty.
- Yelp is the weak case; only GRU4Rec gains materially.
- Status: primary positive extension, but dataset-dependent.
- Source:
  [LLM-ESR Extension Two-Specialist Routing (a mixture over the dual views).md](LLM-ESR%20Extension%20Two-Specialist%20Routing%20%28a%20mixture%20over%20the%20dual%20views%29.md).

### E24. Router variants

- User-level gating fails strongly because it also suppresses collaboration for a
  tail user's popular-item candidates.
- Soft popularity gates are safer on Yelp but weaker than hard routing on Amazon.
- Raising the cold-item threshold by 2x or 4x roughly doubles Amazon gains, but the
  best threshold is dataset-specific.
- Status: supporting ablation that establishes per-candidate granularity and
  validation-tuned thresholds.
- Source:
  [LLM-ESR Extension Two-Specialist Routing (a mixture over the dual views).md](LLM-ESR%20Extension%20Two-Specialist%20Routing%20%28a%20mixture%20over%20the%20dual%20views%29.md).

## H. Extension family 7: embedding-model ablation

### E25. Item-embedding encoder replacement

- Replaced the semantic item encoder with:
  word2vec, text-embedding-ada-002, text-embedding-3-small,
  text-embedding-3-large, and gemini-embedding-001.
- The model, data, retrieval set, and user-side semantic signal are otherwise held
  fixed.
- This is an item-only ablation on the separately preprocessed 441,393-user Yelp
  dataset, using seed 42.
- Tail-item HR@10 is the correct target metric:
  word2vec 0.6039, ada 0.5988, small 0.6247, large 0.6570, Gemini 0.7799.
- Main insight: "LLM versus non-LLM" is not the useful distinction. The original
  ada embedding does not beat word2vec; modern embedding quality matters, and
  Gemini is much stronger on tail items.
- Status: promising and presentation-worthy as exploratory evidence, but not yet a
  paper-comparable or multi-seed headline result.
- Source: [embedding.md](embedding.md).