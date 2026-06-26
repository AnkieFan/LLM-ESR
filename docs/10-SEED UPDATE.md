# 10-SEED UPDATE

All configurations above were extended from 3 seeds (42-44) to 10 seeds (42-51); 91 additional runs, all completed. Sections above are kept for history; the tables below supersede them. Significance: paired t-test by seed vs full LLM-ESR, n=10, two-sided (`log/yelp/STATS_10seeds.txt`).

## Main reproduction (3 backbones, Overall)

| Backbone | H@10 paper / ours        | N@10 paper / ours        |
| :------- | :----------------------- | :----------------------- |
| SASRec   | 0.6673 / 0.6676 +/- 0.0019 | 0.4208 / 0.4217 +/- 0.0023 |
| Bert4Rec | 0.6623 / 0.6648 +/- 0.0024 | 0.4222 / 0.4239 +/- 0.0022 |
| GRU4Rec  | 0.5724 / 0.5710 +/- 0.0042 | 0.3413 / 0.3403 +/- 0.0032 |

## Table 2 ablation (Overall; ours = 10-seed mean+/-std)

| Variant         | H@10 paper / ours       | N@10 paper / ours       | Overall vs full (p) |
| :-------------- | :---------------------- | :---------------------- | :------------------ |
| LLM-ESR (full)  | 0.6673 / 0.6676 +/- .0019 | 0.4208 / 0.4217 +/- .0023 | -                   |
| w/o Co-view     | 0.6320 / 0.6417 +/- .0029 | 0.3816 / 0.3894 +/- .0023 | worse, p=4e-10      |
| w/o Se-view     | 0.6468 / 0.6384 +/- .0021 | 0.4038 / 0.3944 +/- .0016 | worse, p=4e-12      |
| w/o SD          | 0.6572 / 0.6634 +/- .0045 | 0.4121 / 0.4175 +/- .0048 | worse, p=0.016      |
| w/o Share       | 0.6595 / 0.6626 +/- .0024 | 0.4158 / 0.4175 +/- .0017 | worse, p=3e-4       |
| w/o CA          | 0.6644 / 0.6616 +/- .0023 | 0.4160 / 0.4157 +/- .0016 | worse, p=5e-4       |
| 1-layer Adapter | 0.6108 / 0.6604 +/- .0023 | 0.3713 / 0.4153 +/- .0019 | worse, p=8e-5       |
| Random Init     | 0.6440 / 0.6519 +/- .0032 | 0.3984 / 0.4067 +/- .0022 | worse, p=3e-8       |

## Adapter-activation extension (10 seeds + paired tests)

| Adapter       |   Overall H@10 |   Overall N@10 |  TailItem H@10 | verdict vs baseline                        |
| :------------ | -------------: | -------------: | -------------: | :----------------------------------------- |
| Linear (orig) | 0.6676 +/- .0019 | 0.4217 +/- .0023 | 0.1902 +/- .0060 | baseline                                   |
| + Tanh        | 0.6676 +/- .0025 | 0.4206 +/- .0031 | 0.1929 +/- .0076 | NO-OP (all p>0.13)                         |
| + GELU        | 0.6654 +/- .0026 | 0.4187 +/- .0027 | 0.1823 +/- .0100 | worse: H p=0.027, N p=1e-4; tail N p=0.019 |
| + ReLU        | 0.6617 +/- .0037 | 0.4159 +/- .0035 | 0.1587 +/- .0115 | worse: all p<3e-4; tail H -0.032 (p=2e-5)  |
| 1-layer (ref) | 0.6604 +/- .0023 | 0.4153 +/- .0019 | 0.1530 +/- .0104 | -                                          |

