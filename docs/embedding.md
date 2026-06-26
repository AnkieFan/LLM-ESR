## Runs

| run | encoder | dim | LLM? | API |
|-----|---------|-----|------|-----|
| word2vec | GoogleNews-300, mean-pooled over the item text | 300 | no (baseline) | none (local) |
| ada | `text-embedding-ada-002` | 1536 | yes (2022) | OpenAI-compat gateway |
| small | `text-embedding-3-small` | 1536 | yes | OpenAI-compat gateway |
| large | `text-embedding-3-large` (native, not truncated) | 3072 | yes | OpenAI-compat gateway |
| gemini | `gemini-embedding-001` | 3072 | yes (newest) | OpenAI-compat gateway |

The word2vec baseline encodes the **identical** item text the LLM runs use; only
the encoder differs, so any gap is "LLM understanding vs. averaged word vectors,"
not different input.

## Results (seed 42, test set)

H@10 / N@10 per group; **bold** = best in row. Rows are ordered by relevance to
this item-only ablation: **Tail Item first (the headline)**, then the do-no-harm
checks, with Tail/Head User flagged as controls (user side held fixed).

| Group | word2vec (300) | ada (1536) | small (1536) | large (3072) | gemini (3072) |
|-------|----------------|------------|--------------|--------------|---------------|
| **Tail Item** * | 0.6039 / 0.3285 | 0.5988 / 0.3226 | 0.6247 / 0.3321 | 0.6570 / 0.3449 | **0.7799 / 0.4057** |
| Head Item *(no-harm)* | 0.9440 / 0.7549 | 0.9405 / 0.7510 | 0.9450 / **0.7555** | 0.9418 / 0.7443 | **0.9469** / 0.7448 |
| Overall *(no-harm)* | 0.8949 / 0.6934 | 0.8912 / 0.6892 | 0.8988 / **0.6944** | 0.9007 / 0.6867 | **0.9228** / 0.6959 |
| Tail User *(control)* | 0.8953 / 0.6969 | 0.8906 / 0.6924 | 0.8982 / **0.6979** | 0.9000 / 0.6902 | **0.9210** / 0.6969 |
| Head User *(control)* | 0.8930 / 0.6743 | 0.8949 / 0.6711 | 0.9021 / 0.6751 | 0.9045 / 0.6674 | **0.9326 / 0.6905** |
| *best val epoch* | 114 | 43 | 29 | 18 | 0 |