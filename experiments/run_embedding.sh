#!/usr/bin/env bash
# Extension E25: embedding-encoder ablation: swap the frozen LLM item embedding (the semantic view).
# Each encoder's vectors live in a directory holding item.npy (+ a run-local pca64.npy for the collaborative
# init); --emb_path selects the directory and --d_llm asserts its dimension.
#
# NOTE: the embedding files are large (gitignored under embeddings/) and are tied to the dataset they were
# encoded for, so run on the matching dataset. The published ablation (results/embedding.md) was on the
# 441k-user yelp setting, which the inventory marks as NOT numerically comparable to the paper's tables.
# Build the embeddings with:  OPENAI_API_KEY=... OPENAI_BASE_URL=... $PY tools/gen_embeddings.py
cd "$(dirname "$0")/.." && source experiments/common.sh
EMB_ROOT="${EMB_ROOT:-embeddings/$DATASET}"   # dir holding <encoder>/item.npy (+ pca64.npy)

# word2vec baseline (300-d, local, no LLM): encodes the SAME item text the LLM runs use (isolates "LLM vs averaged word vectors").
run "emb_word2vec_s${SEED}" --emb_path "$EMB_ROOT/word2vec" --d_llm 300
# OpenAI / Gemini encoders (via the OpenAI-compatible gateway):
run "emb_ada_s${SEED}"      --emb_path "$EMB_ROOT/ada"      --d_llm 1536   # text-embedding-ada-002
run "emb_small_s${SEED}"    --emb_path "$EMB_ROOT/small"    --d_llm 1536   # text-embedding-3-small
run "emb_large_s${SEED}"    --emb_path "$EMB_ROOT/large"    --d_llm 3072   # text-embedding-3-large
run "emb_gemini_s${SEED}"   --emb_path "$EMB_ROOT/gemini"   --d_llm 3072   # gemini-embedding-001 (best on Tail-Item)
