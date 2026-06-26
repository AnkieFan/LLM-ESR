#!/usr/bin/env bash
# Extension family 2: initialization, capacity, view design (E5-E9). Trains models.
cd "$(dirname "$0")/.." && source experiments/common.sh
HANDLED="data/$DATASET/handled"

# E5: collaborative-side adapter (higher-dim collab embedding -> adapter back to 64). All dims hurt.
for D in 16 32 128; do run "co_adapter_${D}_s${SEED}" --co_adapter_dim "$D"; done

# E6: learned dimensionality reduction vs PCA (linear AE ~ties PCA; nonlinear AE worse; random worst).
#   prerequisite (once): build the learnable-DR init files ->  $PY scripts/make_learnable_dr.py --dataset $DATASET
run "dr_pca64_s${SEED}"                                                  # PCA baseline (default)
run "dr_lae64_s${SEED}"  --id_init_path "data/$DATASET/handled_dr/lae64_itm_emb_np.pkl"  # linear autoencoder
run "dr_ae64_s${SEED}"   --id_init_path "data/$DATASET/handled_dr/ae64_itm_emb_np.pkl"   # nonlinear autoencoder
run "dr_random_s${SEED}" --id_random_init                               # random control

# E7: LoRA semi-unfreezing of the frozen LLM item embeddings (both ranks hurt, worse on tail).
for R in 8 32; do run "lora_r${R}_s${SEED}" --llm_lora_rank "$R"; done

# E8: popularity-gated dual-view fusion (monotonic gate hurts; free per-item gate ~neutral).
run "popgate_pop_s${SEED}"  --pop_gate pop    # monotonic 2-param gate g=sigmoid(w*logpop+b)
run "popgate_item_s${SEED}" --pop_gate item   # free per-item learnable gate

# E9: larger recommender hidden size (no capacity headroom; slightly worse).
#   prerequisite: a matching PCA-128 init at $HANDLED/pca128_itm_emb_np.pkl
run "hidden128_s${SEED}" --hidden_size 128 --id_init_path "$HANDLED/pca128_itm_emb_np.pkl"
