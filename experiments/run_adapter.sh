#!/usr/bin/env bash
# Extension family 1: adapter architecture & inductive bias (E1-E4). Trains models.
cd "$(dirname "$0")/.." && source experiments/common.sh

# E1: activation-function sweep (linear is the reliable default; nonlinearity hurts, worst on tail).
for ACT in none tanh gelu relu; do run "adp_act_${ACT}_s${SEED}" --adapter_act "$ACT"; done

# E2: adapter intermediate-width sweep (original ~768 is near-optimal).
for W in 64 128 256 512 768; do run "adp_width_${W}_s${SEED}" --adapter_hidden_dim "$W"; done

# E3: adapter depth sweep (pure-linear layers; SASRec has a 3-layer sweet spot, +0.0026 p=.02).
for L in 1 2 3 4 5; do run "adp_depth_${L}_s${SEED}" --adapter_layers "$L" --adapter_act none; done

# E4: residual / linear-skip adapter (significant only for GRU4Rec).
run "adp_residual_s${SEED}" --adapter_residual
