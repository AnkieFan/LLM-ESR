#!/usr/bin/env bash
# Extension family 5: inference-time reranking & content recovery (E17-E21).
# E17/E18/E20/E21 are EVAL-ONLY (--do_test) on a trained checkpoint; E19 retrains.
# Requires a trained baseline first, e.g.:  bash experiments/run_reproduction.sh   ->  repro_${MODEL}_s${SEED}
cd "$(dirname "$0")/.." && source experiments/common.sh
CKPT="${CKPT:-repro_${MODEL}_s${SEED}}"   # the trained checkpoint to rerank

# E17: popularity-debiased inference (sweeps alpha; writes debias_sweep.csv). Tail<->overall knob.
evaltest "$CKPT" --pop_debias

# E18: D6 mean-profile user-semantic scoring (sweeps gamma; writes usr_sem_sweep.csv). Primary positive.
evaltest "$CKPT" --usr_sem

# E20: tail-user-only routing of the content term (writes usr_sem_seg_sweep.csv). Zero head-user risk.
evaltest "$CKPT" --usr_sem --usr_sem_seg

# E21: D7 richer user profiles incl. max-pool/multi-interest (writes usr_sem_sweep_{recency,max,ens}.csv).
#       Strongest universal positive; the max-pool column is the headline result.
evaltest "$CKPT" --usr_sem --usr_sem_d7

# E19: train-time learnable semantic-score weight (RETRAINS; instructive negative). init 0 and forced 8.
run "d6_learnable_init0_s${SEED}" --usr_sem_train --usr_sem_train_init 0
run "d6_learnable_init8_s${SEED}" --usr_sem_train --usr_sem_train_init 8
