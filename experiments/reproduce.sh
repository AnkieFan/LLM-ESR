#!/usr/bin/env bash
# Minimal end-to-end reproduction: train one LLM-ESR baseline, then run the two headline inference-time
# extensions on it (D6/D7 max-pool user-semantic scoring + two-specialist routing) and print the results.
# Requires the preprocessed data in data/yelp/handled/ (see README -> "Data setup").
#   override via env:  PY=<python interpreter>   EPOCHS=<n>   (default 200; set EPOCHS=20 for a fast demo)
set -e
cd "$(dirname "$0")/.."
export DATASET=yelp MODEL=llmesr_sasrec SEED=42
source experiments/common.sh
CKPT=repro_demo

echo "### [1/3] train baseline LLM-ESR ($MODEL / $DATASET, seed $SEED, $EPOCHS epochs) ..."
run "$CKPT"

echo "### [2/3] D6/D7 user-semantic scoring (eval-only on the frozen checkpoint) ..."
evaltest "$CKPT" --usr_sem --usr_sem_d7

echo "### [3/3] two-specialist routing (eval-only) ..."
evaltest "$CKPT" --moe_diag

OUT="saved/$DATASET/$MODEL/$CKPT"
echo ""
echo "===================== RESULTS ====================="
echo "--- D7 max-pool user-semantic scoring: HR@10 by gamma  (usr_sem_sweep_max.csv) ---"
column -t -s, "$OUT/usr_sem_sweep_max.csv" 2>/dev/null | cut -c1-78
echo ""
echo "--- two-specialist router: HR@10 by alpha_tail  (moe_route.csv; alpha_tail=1.00 == baseline) ---"
column -t -s, "$OUT/moe_route.csv" 2>/dev/null
echo "==================================================="
echo "All 25 extensions + flag map: experiments/RUN_GUIDE.md   |   write-ups: docs/"
