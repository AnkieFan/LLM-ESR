# Documentation index

Write-ups for every reproduction and extension experiment. The canonical, status-annotated index is **[EXTENSIONS_INVENTORY.md](EXTENSIONS_INVENTORY.md)**; **[CODE_OVERVIEW.md](CODE_OVERVIEW.md)** maps each experiment to the code that implements it. Run commands are in [`../experiments/RUN_GUIDE.md`](../experiments/RUN_GUIDE.md).

## Notebooks (tables + plots)
- [LLM_ESR_Headline_Extensions.ipynb](LLM_ESR_Headline_Extensions.ipynb) - the headline positive extensions.
- [LLM_ESR_Supporting_Extensions.ipynb](LLM_ESR_Supporting_Extensions.ipynb) - supporting ablations and negative results.

## Write-ups by topic
- **Reproduction & protocol** (R1-R4): [Reproduction 1](Reproduction%201.md), [Reproduction 2](Reproduction%202.md), [10-SEED UPDATE](10-SEED%20UPDATE.md), [TTEST_ANALYSIS_EN](TTEST_ANALYSIS_EN.md)
- **Adapter / init / capacity** (E1-E9): [ADAPTER_ANALYSIS_EN](ADAPTER_ANALYSIS_EN.md), [Ablation Study 1](Ablation%20Study%201.md), [extension-log-0618](extension-log-0618.md)
- **Self-distillation** (E10-E13): [self-distillation](self-distillation.md), [sd_teacher_alter](sd_teacher_alter.md), [sd_hybrid_retrieval](sd_hybrid_retrieval.md)
- **Data / objective / negatives** (E14-E16): [HARDNEG_EXTENSION_EN](HARDNEG_EXTENSION_EN.md), [extension-log-0618](extension-log-0618.md)
- **Inference reranking - D6/D7** (E17-E21): [LLM-ESR_D6_summary](LLM-ESR_D6_summary.md)
- **Two-specialist routing** (E22-E24): [Two-Specialist Routing](LLM-ESR%20Extension%20Two-Specialist%20Routing%20%28a%20mixture%20over%20the%20dual%20views%29.md)
- **Embedding encoder** (E25): [embedding](embedding.md)

> Source-of-truth note (from the inventory): paper-comparable claims use the authors' preprocessed data; results on the separately-preprocessed 441k-user Yelp setting (e.g. the embedding ablation) are a *separate* experimental setting and are not numerically comparable to the paper's tables.
