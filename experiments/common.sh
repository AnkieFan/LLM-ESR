#!/usr/bin/env bash
# Shared setup for all LLM-ESR extension experiments.
# Override via env, e.g.:  DATASET=beauty MODEL=llmesr_gru4rec SEED=43 PY=/path/to/python  bash experiments/run_routing.sh
#   DATASET : yelp | fashion | beauty
#   MODEL   : llmesr_sasrec | llmesr_gru4rec | llmesr_bert4rec
#   SEED    : integer (paper uses 42..51 for 10 seeds)
#   PY      : python interpreter (default: python). On Snellius: PY=/home/<user>/.conda/envs/llm/bin/python
# All commands below assume the authors' preprocessed data is in data/<dataset>/handled/.
set -u
PY=${PY:-python}
DATASET=${DATASET:-yelp}
MODEL=${MODEL:-llmesr_sasrec}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-200}
# dataset-specific Tail/Head thresholds used by the paper's group split (ts_user=seq len, ts_item=item pop)
case "$DATASET" in
  yelp)    TSU=12; TSI=13 ;;
  fashion) TSU=3;  TSI=4  ;;
  beauty)  TSU=9;  TSI=4  ;;
  *) echo "unknown DATASET=$DATASET (use yelp|fashion|beauty)"; exit 1 ;;
esac
# canonical LLM-ESR training/eval config (matches the reproduction); BERT4Rec additionally needs --mask_prob
BASE="--dataset $DATASET --model_name $MODEL --hidden_size 64 --train_batch_size 128 --max_len 200 \
 --gpu_id 0 --num_workers 8 --num_train_epochs $EPOCHS --seed $SEED --patience 20 \
 --ts_user $TSU --ts_item $TSI --freeze --log --user_sim_func kd --alpha 0.1 --use_cross_att"
[ "$MODEL" = "llmesr_bert4rec" ] && BASE="$BASE --mask_prob 0.6"

# run <check_path-tag> [extra training flags...]   -- TRAINS a model under saved/<dataset>/<model>/<tag>/
run () { local tag="$1"; shift; echo ">>> TRAIN [$tag] extra: $*"; $PY main.py $BASE --check_path "$tag" "$@"; }

# evaltest <existing-check_path> [extra eval flags...]  -- EVAL-ONLY (--do_test) on an already-trained checkpoint
#   used by the inference-time extensions (debias / D6 / D7 / routing); requires a trained baseline first.
evaltest () { local tag="$1"; shift; echo ">>> EVAL  [$tag] extra: $*"; $PY main.py $BASE --do_test --check_path "$tag" "$@"; }

echo "[common] DATASET=$DATASET MODEL=$MODEL SEED=$SEED ts_user=$TSU ts_item=$TSI PY=$PY"
