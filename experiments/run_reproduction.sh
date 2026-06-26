#!/usr/bin/env bash
# R2 faithful reproduction + R3 Table-2 component ablation.  (Trains models; run per MODEL/DATASET/SEED.)
cd "$(dirname "$0")/.." && source experiments/common.sh

# R2: faithful reproduction. Sweep MODEL in {llmesr_sasrec,llmesr_gru4rec,llmesr_bert4rec} and SEED in 42..51.
run "repro_${MODEL}_s${SEED}"

# R3: Table-2 component ablations (each removes ONE component from the full model).
run "abl_woCo_s${SEED}"      --se_view              # remove collaborative view  (semantic-only)
run "abl_woSe_s${SEED}"      --co_view              # remove semantic view       (collaborative-only)
run "abl_split_s${SEED}"     --split_backbone       # separate backbone per view (no parameter sharing)
run "abl_randInit_s${SEED}"  --id_random_init       # random collab init instead of PCA-of-LLM
run "abl_1adapter_s${SEED}"  --adapter_layers 1     # single adapter layer (vs the default 2)
# remove cross-attention: drop --use_cross_att from BASE (handled explicitly since BASE includes it)
echo ">>> TRAIN [abl_woCross_s${SEED}] (no cross-attention)"
$PY main.py ${BASE/--use_cross_att/} --check_path "abl_woCross_s${SEED}"

# R4: stronger protocol is just R2/R3 repeated over seeds 42..51 (10 seeds) + paired tests in scripts/.
