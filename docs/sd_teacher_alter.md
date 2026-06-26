Hey so I also ran it with different number of teachers for both top k and stochastic method and found that more teachers hurt performance, while less so k=5 was close to k=10 performance, but seems like k=10 (baseline) is still the best with top-k. Nothing to improve here.

# Teacher Count Alteration

## 3-seed mean+/-std (k=10, top-k is the baseline)

| Method         | Overall H@10   | Overall N@10   | Tail-User N@10 |
| :-------------- | :------------- | :------------- | :------------- |
| top-k k=5       | 0.6660 +/-.0042  | 0.4194 +/-.0033  | 0.4221 +/-.0034  |
| top-k k=10 (bl) | 0.6666 +/-.0051  | 0.4198 +/-.0035  | 0.4222 +/-.0039  |
| top-k k=20      | 0.6656 +/-.0018  | 0.4198 +/-.0024  | 0.4210 +/-.0026  |
| top-k k=30      | 0.6654 +/-.0046  | 0.4192 +/-.0040  | 0.4208 +/-.0032  |
| stoch k=5       | 0.6680 +/-.0021  | 0.4203 +/-.0026  | 0.4208 +/-.0024  |
| stoch k=10      | 0.6659 +/-.0008  | 0.4200 +/-.0029  | 0.4216 +/-.0030  |
| stoch k=20      | 0.6628 +/-.0033  | 0.4184 +/-.0042  | 0.4198 +/-.0032  |
| stoch k=30      | 0.6670 +/-.0004  | 0.4194 +/-.0026  | 0.4211 +/-.0026  |