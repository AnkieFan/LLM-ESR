Hey so for the distillation extension I checked if changing from mean to weighted pooling helped, same for the sampling, so instead of gathering the top k [:10] I gathered random 10 teachers from the top 50 pool. Here are the results on Yelp, 3 seeds (42, 43 and 44) on the SASRec one. Sadly I do not see major improvements, and changing to weighted pooling even seems to hurt the performance. I wanted to try to check if going from 10 teachers to 5 or 20 would help, in both cases; getting top-k and sampling randomly.


# Self-Distillation Extension (Yelp, SASRec)

SASRec / Yelp, 3 seeds (42/43/44), Baseline = standard LLM-ESR (mean pool + top-N, sim_user_num=10)

## Baseline

| Backbone   | Group     | H@10 paper |     H@10 ours | N@10 paper |     N@10 ours |
| :--------- | :-------- | ---------: | ------------: | ---------: | ------------: |
| **SASRec** | Overall   |     0.6673 | 0.6666 +/-.0051 |     0.4208 | 0.4198 +/-.0035 |
|            | Tail Item |     0.1893 | 0.1922 +/-.0178 |     0.0845 | 0.0843 +/-.0096 |
|            | Head Item |     0.8080 | 0.8062 +/-.0021 |     0.5199 | 0.5186 +/-.0036 |
|            | Tail User |     0.6685 | 0.6679 +/-.0051 |     0.4229 | 0.4222 +/-.0039 |
|            | Head User |     0.6627 | 0.6614 +/-.0049 |     0.4128 | 0.4106 +/-.0018 |

## Changing Pooling/Sampling 

| Variant              | Overall H@10   | Overall N@10   | Tail-User N@10 |
| :------------------- | -------------: | -------------: | -------------: |
| baseline (mean+topN) | 0.6666 +/-.0051  | 0.4198 +/-.0035  | 0.4222 +/-.0039  |
| weighted pooling     | 0.6617 +/-.0032  | 0.4163 +/-.0032  | 0.4171 +/-.0031  |
| stochastic sampling  | 0.6659 +/-.0008  | 0.4200 +/-.0029  | 0.4216 +/-.0030  |