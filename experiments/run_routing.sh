#!/usr/bin/env bash
# Extension family 6: two-specialist routing over the dual views (E22-E24). EVAL-ONLY on a trained checkpoint.
# A single --moe_diag pass writes all three result files:
#   moe_diag.csv     (E22) expert decomposition diagnostic over the 2x2 user x item grid
#   moe_route.csv    (E23) per-candidate hard router: alpha_tail sweep (drop collab on cold items)
#   moe_route_v2.csv (E24) router variants: user-level gate / soft gates / ts_item x2 x4 sweep
# Requires a trained baseline first, e.g.:  bash experiments/run_reproduction.sh   ->  repro_${MODEL}_s${SEED}
cd "$(dirname "$0")/.." && source experiments/common.sh
CKPT="${CKPT:-repro_${MODEL}_s${SEED}}"
evaltest "$CKPT" --moe_diag
# aggregate across seeds:  $PY scripts/aggregate_moe_all.py   (E22/E23)   and   $PY scripts/aggregate_moev2.py  (E24)
