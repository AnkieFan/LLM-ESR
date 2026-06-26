#!/usr/bin/env bash
# Extension family 4: training data / objective / negative sampling (E14-E16). Trains models.
cd "$(dirname "$0")/.." && source experiments/common.sh

# E14: LLM-guided hard-negative sampling (negatives from the top-K LLM-nearest items). Monotonic harm; tail worst.
#   prerequisite: a hard-neg pool at data/<ds>/handled/hard_neg_k50_pool.pkl
for R in 0.25 0.5 1.0; do run "hardneg_r${R}_s${SEED}" --hard_neg_ratio "$R" --hard_neg_k 50; done

# E15: in-batch sampled-softmax objective instead of 1-negative BCE (extreme tail<->overall trade-off).
run "ssm_s${SEED}" --ssm

# E16: all-prefix sequence augmentation (~10x training data; both overall & tail worsen).
#   --aug builds every-prefix expansion and trains on it (distillation alignment is preserved via user_map).
run "aug_s${SEED}" --aug
