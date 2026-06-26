#!/usr/bin/env bash
# Build every derived prerequisite file the extension experiments need, so all of R1-R4 / E1-E25 are
# run-ready. Idempotent (skips existing files). Run after the base data is in place (setup_data.sh).
# Needs a few GB of RAM for the hard-negative kNN.   override:  PY=<python>  DATASET=<yelp|fashion|beauty>
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
DATASET=${DATASET:-yelp}
H="data/$DATASET/handled"
[ -f "$H/itm_emb_np.pkl" ] || { echo "ERROR: $H/itm_emb_np.pkl missing - run experiments/setup_data.sh first"; exit 1; }

# E9: PCA-128 collaborative init (matches pca64; unpadded (item_num, 128)).
if [ ! -f "$H/pca128_itm_emb_np.pkl" ]; then
  $PY - "$H" <<'PYEOF'
import sys, pickle, numpy as np
from sklearn.decomposition import PCA
H = sys.argv[1]
X = np.asarray(pickle.load(open(H + "/itm_emb_np.pkl", "rb")), dtype=np.float32)
Z = PCA(n_components=128, random_state=42).fit_transform(X).astype(np.float32)
pickle.dump(Z, open(H + "/pca128_itm_emb_np.pkl", "wb")); print("  built pca128", Z.shape)
PYEOF
fi

# E14: top-50 LLM-nearest hard-negative pool, padded to (item_num+2, 50) to match the model's item indexing.
if [ ! -f "$H/hard_neg_k50_pool.pkl" ]; then
  $PY - "$H" <<'PYEOF'
import sys, pickle, numpy as np
H = sys.argv[1]
X = np.asarray(pickle.load(open(H + "/itm_emb_np.pkl", "rb")), dtype=np.float32)
X = np.concatenate([np.zeros((1, X.shape[1]), np.float32), X, np.zeros((1, X.shape[1]), np.float32)], 0)  # pad
Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
pool = np.zeros((X.shape[0], 50), np.int32)
for s in range(0, X.shape[0], 2000):
    sims = Xn[s:s + 2000] @ Xn.T
    for j in range(sims.shape[0]):
        sims[j, s + j] = -2.0  # exclude self
    pool[s:s + 2000] = np.argsort(-sims, axis=1)[:, :50].astype(np.int32)
pool[0] = 0; pool[-1] = 0      # padding rows
pickle.dump(pool, open(H + "/hard_neg_k50_pool.pkl", "wb")); print("  built hard_neg_k50_pool", pool.shape)
PYEOF
fi

# E6: learnable-DR inits (lae64/ae64); E13: hybrid teacher neighbours. (Generators are currently yelp-specific.)
if [ "$DATASET" = "yelp" ]; then
  [ -f data/yelp/handled_dr/lae64_itm_emb_np.pkl ] || { echo "  building learnable-DR inits (E6) ..."; $PY scripts/make_learnable_dr.py; }
  [ -f "$H/sim_user_hybrid_100.pkl" ]             || { echo "  building hybrid neighbours (E13) ..."; $PY scripts/retrieval_hybrid_yelp.py; }
fi

echo "Prereqs ready for $DATASET."
