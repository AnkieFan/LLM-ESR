# LLM-ESR: Reproduction & Extensions (Group 5)

This repository reproduces and extends **LLM-ESR** - *"Large Language Models Enhanced Sequential Recommendation for Long-tail User and Item"* ([arXiv:2405.20646](https://arxiv.org/abs/2405.20646)) - for the UvA Recommender Systems course. On top of a faithful reproduction (with a stronger 10-seed statistical protocol) it adds **25 extension experiments** across the adapter, initialization/capacity, self-distillation, training objective, inference-time reranking, two-specialist routing, and the embedding encoder - all reachable by toggling command-line flags, with one run script per extension family.

## Highlights

- **Faithful reproduction** on Yelp / Fashion / Beauty with SASRec / GRU4Rec / Bert4Rec, within ~0.003 HR@10 of the paper (10 seeds, paired t-tests).
- **D6 / D7 user-semantic scoring** (headline positive): an inference-time, frozen-model content-match term; the **max-pool / multi-interest** profile gives a significant Pareto gain on all 9 backbone x dataset cells.
- **Two-specialist routing** (E22-E24): decompose the score into collaborative + semantic experts and route by item popularity - cold items are best served by the semantic view alone. Significant on 7/9 cells, training-free.
- **Embedding-encoder ablation** (E25): swap the frozen LLM embedding (word2vec -> ada-002 -> text-embedding-3 -> gemini); a stronger encoder sharply lifts tail-item recall.
- **A documented wall of clean negatives** (capacity, nonlinear adapters, LoRA, hard-negatives, sampled-softmax, augmentation, popularity gating, teacher pooling/sampling...), each with a mechanism.

A few headline numbers (full tables in [`docs/`](docs/)):

| experiment | result |
|---|---|
| Reproduction (R2) | within ~0.003 HR@10 of the paper - SASRec 0.6676 / BERT4Rec 0.6648 / GRU4Rec 0.5710 (10 seeds) |
| D7 max-pool user-semantic (E21) | significant Pareto gain on **all 9** backbone x dataset cells; 7/9 clear +0.01 overall HR@10, all 9 clear +0.01 tail-item - training-free |
| Two-specialist routing (E23) | tuned router adds **+0.05 (Fashion) / +0.06 (Beauty)** overall HR@10 on top of max-pool - training-free |
| Embedding-encoder swap (E25) | a stronger encoder lifts Tail-Item HR@10 from **0.60 (ada-002) -> 0.78 (gemini)** |

Per-experiment write-ups are in [`docs/`](docs/); start with the canonical index [`docs/EXTENSIONS_INVENTORY.md`](docs/EXTENSIONS_INVENTORY.md).

## Repository structure

```
main.py            entry point (train / --do_test) exposing every extension flag
models/            LLM-ESR + dual-view backbones (SASRec/GRU4Rec/Bert4Rec) and extension code
trainers/          training + evaluation, incl. the inference-time extension sweeps
generators/        datasets / loaders, incl. self-distillation, hard-neg and augmentation
experiments/       RUN_GUIDE.md (flag map) + one run_*.sh per extension family + reproduce.sh
scripts/           result aggregators (per-seed CSV -> report tables)
tools/             gen_embeddings.py (re-embed items for the encoder ablation, E25)
docs/              all experiment write-ups + the two results notebooks
data/              datasets (preprocessed files go in data/<ds>/handled/ - see Data setup)
jobs/              SLURM array runners (optional, for cluster batches)
```

Where each extension lives in the code (file -> function -> flag): [`docs/CODE_OVERVIEW.md`](docs/CODE_OVERVIEW.md).

## Setup

```bash
conda env create -f environment.yml      # or:  pip install -r requirements.txt
```

Reference environment: Python 3.9, PyTorch 1.12 (CUDA). A single GPU suffices; training is ~minutes per epoch. Run the extension-helper unit tests (no GPU/data needed) with `python tests/test_extensions.py` (or `pytest tests/`).

## Data setup

Paper-comparable runs use the authors' preprocessed files under `data/<dataset>/handled/`. Two commands set everything up:

```bash
bash experiments/setup_data.sh       # download the authors' preprocessed bundle into data/<ds>/handled/
bash experiments/setup_prereqs.sh    # build the derived files that E6/E9/E13/E14 need (pca128, hard-neg pool, ...)
```

To preprocess from raw instead, see [Preprocessing from raw](#preprocessing-from-raw-original-instructions). Each dataset needs: `inter.txt`, `itm_emb_np.pkl`, `usr_emb_np.pkl`, `pca64_itm_emb_np.pkl`, `sim_user_100.pkl` (the bundle download link is also in the [original README](https://drive.google.com/file/d/1MpBUjCDLiFIEODTnopSCzDAnS8RzO9aV/view?usp=sharing)).

## Running experiments

**[`experiments/RUN_GUIDE.md`](experiments/RUN_GUIDE.md) is the single source of truth** - a table mapping every experiment (R1-R4, E1-E25) to its exact flags and output file. Each extension family has a runnable script:

```bash
export PY=python DATASET=yelp MODEL=llmesr_sasrec SEED=42   # or beauty/fashion, gru4rec/bert4rec, 42..51
bash experiments/run_reproduction.sh     # R2 reproduction + R3 Table-2 ablation
bash experiments/run_adapter.sh          # E1-E4   adapter activation / width / depth / residual
bash experiments/run_init_capacity.sh    # E5-E9   co-adapter / learned-DR / LoRA / pop-gate / hidden-128
bash experiments/run_selfdistill.sh      # E10-E13 teacher pooling / sampling / count / hybrid
bash experiments/run_data_objective.sh   # E14-E16 hard-neg / sampled-softmax / augmentation
bash experiments/run_inference_rerank.sh # E17-E21 debias / D6 / learnable-gamma / seg / D7
bash experiments/run_routing.sh          # E22-E24 two-specialist routing (diagnostic + router + variants)
bash experiments/run_embedding.sh        # E25     embedding-encoder swap
```

`experiments/common.sh` defines the canonical config + per-dataset Tail/Head thresholds and provides `run` (trains a checkpoint) and `evaltest` (eval-only `--do_test`, for the inference-time families). The inference extensions (E17/E18/E20-E25) rerank an already-trained checkpoint, so train a baseline first. The original full-reproduction scripts `experiments/{yelp,fashion,beauty}.bash` are also kept. Per-seed CSVs are reduced to the report tables by `scripts/aggregate_*.py`.

## Minimal reproduction

```bash
bash experiments/reproduce.sh
```

Trains a SASRec/Yelp baseline, runs the headline inference-time extensions on it (D6 max-pool + two-specialist routing), and prints the result tables via the aggregators - a small end-to-end path once the data is in place.

## Results & write-ups

[`docs/`](docs/) holds the write-up for every experiment plus two notebooks ([Headline](docs/LLM_ESR_Headline_Extensions.ipynb), [Supporting](docs/LLM_ESR_Supporting_Extensions.ipynb)). [`docs/EXTENSIONS_INVENTORY.md`](docs/EXTENSIONS_INVENTORY.md) indexes each result with its source, status, and the source-of-truth rules (e.g. the separately-preprocessed 441k-user Yelp setting is *not* numerically comparable to the paper's tables).

## Credits

Built on the original LLM-ESR implementation (Liu et al., 2024). Reproduction and extensions by Group 5, UvA Recommender Systems.

```bibtex
@article{liu2024large,
  title={Large Language Models Enhanced Sequential Recommendation for Long-tail User and Item},
  author={Liu, Qidong and Wu, Xian and Zhao, Xiangyu and Wang, Yejing and Zhang, Zijian and Tian, Feng and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2405.20646},
  year={2024}
}
```

## Poster
![poster](poster.png)

---

### Preprocessing from raw (original instructions)

1. Put the raw dataset in `data/<yelp|fashion|beauty>/raw/`. Yelp: [yelp.com/dataset](https://www.yelp.com/dataset); Fashion/Beauty: [Amazon reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews).
2. Run `data/data_process.py` to filter cold-start users/items -> `id_map.json`, `inter_seq.txt`.
3. Convert interactions with `data/convert_inter.ipynb`.
4. Encode items/users with `data/<ds>/get_item_embedding.ipynb` and `get_user_embedding.ipynb` -> `itm_emb_np.pkl`, `usr_emb_np.pkl`.
5. Run `data/pca.ipynb` -> `pca64_itm_emb_np.pkl` (collaborative-view init).
6. Run `data/retrieval_users.ipynb` -> `sim_user_100.pkl` (self-distillation teachers).

For the encoder ablation (E25), re-embed item text with `tools/gen_embeddings.py` (needs `OPENAI_API_KEY` / `OPENAI_BASE_URL`).
