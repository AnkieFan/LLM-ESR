#!/usr/bin/env bash
# Extension family 3: self-distillation / teacher retrieval (E10-E13). Trains models. (sd-extension code.)
cd "$(dirname "$0")/.." && source experiments/common.sh

# baseline = mean pooling + top-n retrieval (the defaults: --sim_pool mean --sampling_method topn).
run "sd_baseline_s${SEED}"

# E10: similarity-weighted teacher pooling instead of mean (decreases performance).
run "sd_pool_weighted_s${SEED}" --sim_pool weighted

# E11: stochastic teacher sampling: 10 random from the top-50 pool (~neutral).
run "sd_stochastic_s${SEED}" --sampling_method stochastic

# E12: teacher-count sweep, top-n AND stochastic, k in {5,10,20,30} (more teachers generally hurt).
for K in 5 10 20 30; do
  run "sd_topn_k${K}_s${SEED}" --sim_user_num "$K" --sampling_method topn
  run "sd_samp_k${K}_s${SEED}" --sim_user_num "$K" --sampling_method stochastic
done

# E13: hybrid semantic/collaborative teacher retrieval (~neutral).
#   prerequisite (once): build the hybrid neighbour file ->  $PY scripts/retrieval_hybrid_yelp.py
#   (writes data/<ds>/handled/sim_user_hybrid_100.pkl)
run "sd_hybrid_s${SEED}" --sim_user_file sim_user_hybrid_100.pkl
