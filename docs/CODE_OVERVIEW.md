# Code overview - where each extension lives

This maps every reproduction/extension to the code that implements it, so the codebase is navigable. Design principles: **defaults reproduce the original LLM-ESR exactly**; every extension is **behind a CLI flag** (off by default); inference-time extensions are **eval-only sweeps** that rerank a frozen checkpoint and write a `*.csv` next to it (one training run -> a whole gamma/alpha sweep for free). The flag map with exact commands is in [`../experiments/RUN_GUIDE.md`](../experiments/RUN_GUIDE.md).

## Architecture in one paragraph

`main.py` parses flags and builds an `LLMESR_{SASRec,GRU4Rec,Bert4Rec}` model (`models/LLMESR.py`) on top of a dual-view backbone (`models/DualLLMSRS.py`): a **collaborative** view (PCA-initialised id embedding) and a **semantic** view (frozen LLM item embedding -> adapter), fused and scored so that `score = collab_score + semantic_score`. `trainers/sequence_trainer.py::SeqTrainer` runs training and evaluation; the inference-time extensions live in its `eval()` as gated post-loop sweeps. `generators/` builds the datasets (including the self-distillation teacher loading and the hard-neg / augmentation paths). `scripts/aggregate_*.py` reduce the per-seed CSVs to the report tables.

## Map

| Inventory | Flag(s) | Code |
|---|---|---|
| R3 Table-2 ablation | `--se_view`/`--co_view`/`--split_backbone`/`--id_random_init` | `models/DualLLMSRS.py` (view/backbone/init branches) |
| E1-E4 adapter (act/width/depth/residual) | `--adapter_act`/`--adapter_hidden_dim`/`--adapter_layers`/`--adapter_residual` | `models/DualLLMSRS.py::_build_adapter` |
| E5 collab-side adapter | `--co_adapter_dim` | `models/DualLLMSRS.py` (DualLLMSASRec id-init, `co_adapter`) |
| E6 learned-DR init | `--id_init_path` | `scripts/make_learnable_dr.py` (build) + id-init load |
| E7 LoRA semi-unfreeze | `--llm_lora_rank` | `models/DualLLMSRS.py` (`lora_A`/`lora_B`, `_llm_emb`) |
| E8 popularity-gated fusion | `--pop_gate {pop,item}` | `models/DualLLMSRS.py` (gate params + scoring) |
| E9 hidden size | `--hidden_size` (+ matching `--id_init_path`) | `models/*` (`hidden_size`) |
| E10-E13 self-distillation | `--sim_pool`/`--sampling_method`/`--sim_user_file` | `models/LLMESR.py::pool_sim`, `generators/data.py::sample_sim` |
| E14 hard negatives | `--hard_neg_ratio`/`--hard_neg_k` | `generators/data.py` (Seq2Seq neg sampling) |
| E15 sampled-softmax | `--ssm` | `models/SASRec.py::SASRec_seq.forward` |
| E16 augmentation | `--aug` | `generators/data.py` + `generators/generator.py` |
| E17 popularity debias | `--pop_debias` | `trainers/sequence_trainer.py::eval` (debias sweep) |
| E18 D6 mean / E20 seg / E21 D7 | `--usr_sem`/`--usr_sem_seg`/`--usr_sem_d7` | `trainers/sequence_trainer.py::eval` (`_usr`/`_d7`/seg blocks) |
| E19 learnable-gamma (train-time) | `--usr_sem_train`/`--usr_sem_train_init` | `models/SASRec.py` (`usr_gamma`, `_usr_sem_score`) |
| E22-E24 two-specialist routing | `--moe_diag` | `trainers/sequence_trainer.py::eval` (`_moe` block) + `_final_feat` |
| E25 embedding encoder | `--emb_path`/`--d_llm` | `models/DualLLMSRS.py::load_llm_item_emb`/`load_id_item_emb`; build: `tools/gen_embeddings.py` |
| full-catalog ranking (D analysis) | `--full_rank` | `trainers/sequence_trainer.py::_eval_fullrank` |

## The eval-time sweep pattern (E17, E18, E20-E25)

In `SeqTrainer.eval()` the per-batch loop accumulates the base scores plus the extension-specific tensors (the user-semantic content term, the per-candidate collaborative/semantic split, etc.) only when the corresponding flag is on. After the loop, each gated block sweeps its hyper-parameter (gamma for the content term, alpha for the router) by re-ranking the accumulated scores and writing one CSV row per value - so a single forward pass over the test set yields the whole sweep. `scripts/aggregate_*.py` then average those CSVs across seeds (paired t-tests) into the tables in [the write-ups](EXTENSIONS_INVENTORY.md).
