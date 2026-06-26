# LLM-ESR - Reproduction & Extension Run Guide (`final` branch)

This branch consolidates all code needed to reproduce every result in `result_inventory` (items R1-R4 and E1-E25). It merges four sources: the reproduction/ablation base (`master`), the extension code (`claude/adapter-ext`: adapter/init/capacity sweeps, hard-neg, SSM, augmentation, popularity-debias, D6/D7 user-semantic scoring, two-specialist routing), the self-distillation variants (`sd-extension`: teacher pooling/sampling/count/hybrid), and the embedding-encoder ablation (`--emb_path`, E25: word2vec/ada/small/large/gemini). Every experiment is reachable by toggling command-line flags; one runnable script per extension family lives in `experiments/`.

## Setup

All paper-comparable runs use the authors' preprocessed data in `data/<dataset>/handled/`. Configure a run with environment variables, then call a family script:

```bash
export PY=/home/<user>/.conda/envs/llm/bin/python   # python with the project env (default: python)
export DATASET=yelp        # yelp | fashion | beauty
export MODEL=llmesr_sasrec # llmesr_sasrec | llmesr_gru4rec | llmesr_bert4rec
export SEED=42             # paper uses 42..51 for the 10-seed protocol (R4)
bash experiments/run_routing.sh
```

`experiments/common.sh` defines the canonical config (`BASE`), the per-dataset Tail/Head thresholds (`ts_user`/`ts_item`), a `run` helper (trains a checkpoint), and an `evaltest` helper (eval-only `--do_test` on a trained checkpoint). On SLURM, wrap any command in your usual `sbatch`; the legacy array runners in `jobs/` (`array_run_*.slurm`, `sd_*.slurm`) are also kept.

## Family scripts

| script | inventory items | trains or eval-only |
|---|---|---|
| `experiments/run_reproduction.sh` | R2 reproduction, R3 Table-2 ablation, R4 (seeds) | train |
| `experiments/run_adapter.sh` | E1 activation, E2 width, E3 depth, E4 residual | train |
| `experiments/run_init_capacity.sh` | E5 co-adapter, E6 learned-DR, E7 LoRA, E8 pop-gate, E9 hidden-128 | train |
| `experiments/run_selfdistill.sh` | E10 weighted pool, E11 stochastic, E12 teacher-count, E13 hybrid | train |
| `experiments/run_data_objective.sh` | E14 hard-neg, E15 SSM, E16 augmentation | train |
| `experiments/run_inference_rerank.sh` | E17 debias, E18 D6-mean, E19 learnable-gamma, E20 seg, E21 D7 | mostly eval-only (E19 trains) |
| `experiments/run_routing.sh` | E22 diagnostic, E23 router, E24 variants | eval-only |
| `experiments/run_embedding.sh` | E25 embedding-encoder ablation (word2vec / ada / small / large / gemini) | train |

## Flag map (the single source of truth)

| ID | name | key flags (added to `BASE`) | result file |
|---|---|---|---|
| R2 | reproduction | *(none)* - sweep `MODEL`, `SEED` 42..51 | standard logged CSV |
| R3 | Table-2 ablation | `--se_view` / `--co_view` / `--split_backbone` / `--id_random_init` / `--adapter_layers 1` / drop `--use_cross_att` | standard logged CSV |
| E1 | adapter activation | `--adapter_act {none,tanh,gelu,relu}` | standard logged CSV |
| E2 | adapter width | `--adapter_hidden_dim {64,128,256,512,768}` | standard logged CSV |
| E3 | adapter depth | `--adapter_layers {1..5} --adapter_act none` | standard logged CSV |
| E4 | residual adapter | `--adapter_residual` | standard logged CSV |
| E5 | collab-side adapter | `--co_adapter_dim {16,32,128}` | standard logged CSV |
| E6 | learned DR vs PCA | `--id_init_path <{lae64,ae64}...pkl>` / `--id_random_init` | standard logged CSV |
| E7 | LoRA semi-unfreeze | `--llm_lora_rank {8,32}` | standard logged CSV |
| E8 | popularity-gated fusion | `--pop_gate {pop,item}` | standard logged CSV |
| E9 | larger hidden size | `--hidden_size 128 --id_init_path ...pca128...pkl` | standard logged CSV |
| E10 | weighted teacher pool | `--sim_pool weighted` | standard logged CSV |
| E11 | stochastic sampling | `--sampling_method stochastic` | standard logged CSV |
| E12 | teacher-count sweep | `--sim_user_num {5,10,20,30} --sampling_method {topn,stochastic}` | standard logged CSV |
| E13 | hybrid retrieval | `--sim_user_file sim_user_hybrid_100.pkl` | standard logged CSV |
| E14 | hard negatives | `--hard_neg_ratio {0.25,0.5,1.0} --hard_neg_k 50` | standard logged CSV |
| E15 | sampled-softmax | `--ssm` | standard logged CSV |
| E16 | all-prefix augmentation | `--aug` | standard logged CSV |
| E17 | popularity debias | `--do_test --pop_debias` | `debias_sweep.csv` |
| E18 | D6 mean user-semantic | `--do_test --usr_sem` | `usr_sem_sweep.csv` |
| E19 | learnable-gamma (train) | `--usr_sem_train --usr_sem_train_init {0,8}` | standard logged CSV |
| E20 | tail-user-only routing | `--do_test --usr_sem --usr_sem_seg` | `usr_sem_seg_sweep.csv` |
| E21 | D7 max-pool profiles | `--do_test --usr_sem --usr_sem_d7` | `usr_sem_sweep_{recency,max,ens}.csv` |
| E22 | expert decomposition | `--do_test --moe_diag` | `moe_diag.csv` |
| E23 | per-candidate router | `--do_test --moe_diag` | `moe_route.csv` |
| E24 | router variants | `--do_test --moe_diag` | `moe_route_v2.csv` |
| E25 | embedding encoder swap | `--emb_path <encoder_dir> --d_llm {300,1536,3072}` | standard logged CSV |

The inference-time families (E17, E18, E20-E24) all run `--do_test` on an already-trained checkpoint (default `repro_${MODEL}_s${SEED}`, override with `CKPT=...`), so a baseline must be trained first.

## One-time prerequisites (only for the experiments that need a pre-built file)

**Build them all at once:** `bash experiments/setup_prereqs.sh` (after `setup_data.sh`). It generates pca128, the hard-neg pool, the learnable-DR inits, and the hybrid neighbours, skipping any that already exist. The individual sources:

- E6: `$PY scripts/make_learnable_dr.py` (currently yelp-specific) -> writes `lae64`/`ae64` into `data/yelp/handled_dr/` (because `handled/` is a read-only symlink in this worktree).
- E9: a `pca128_itm_emb_np.pkl` matching `--hidden_size 128`.
- E13: `$PY scripts/retrieval_hybrid_yelp.py` -> writes `sim_user_hybrid_100.pkl`.
- E14: a `hard_neg_k50_pool.pkl` (top-50 LLM-nearest pool per item).
- E25: per-encoder embedding dirs `embeddings/<dataset>/<encoder>/` (each with `item.npy` + run-local `pca64.npy`), built by `OPENAI_API_KEY=... OPENAI_BASE_URL=... $PY tools/gen_embeddings.py`. Gitignored (large) and tied to the dataset they were encoded for; the published study (`results/embedding.md`) is on the non-comparable 441k-user yelp setting.
- R1 (preprocessing investigation): `data/data_process.py` on the raw dump - kept as a reproducibility caveat, not a comparable result.

## Aggregation (results -> tables)

The per-seed CSVs are reduced to the report tables by `scripts/aggregate_*.py`, e.g. `aggregate_moe_all.py` (E22/E23), `aggregate_moev2.py` (E24), `aggregate_d7.py` (E21), `aggregate_usrsem.py` (E18/E20), `aggregate_debias.py` (E17), `aggregate_ext.py` (adapter/init families), and `gen_table1.py` / `gen_moe_tables.py` (paper-style group breakdowns).
