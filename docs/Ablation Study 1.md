just finished running reproduction for ablation study; all results aligns well with the original paper except for the one about 1 layer adapter - changing the adapter to 1 layer only results in slight drop on performance

# Ablation Study 1

| Variant         | H@10 paper | H@10 ours (mean+/-std) | N@10 paper | N@10 ours (mean+/-std) |
| :-------------- | ---------: | -------------------: | ---------: | -------------------: |
| LLM-ESR (full)  |     0.6673 |      0.6670 +/- 0.0021 |     0.4208 |      0.4192 +/- 0.0028 |
| w/o Co-view     |     0.6320 |      0.6387 +/- 0.0020 |     0.3816 |      0.3872 +/- 0.0016 |
| w/o Se-view     |     0.6468 |      0.6392 +/- 0.0023 |     0.4038 |      0.3945 +/- 0.0018 |
| w/o SD          |     0.6572 |      0.6611 +/- 0.0066 |     0.4121 |      0.4145 +/- 0.0058 |
| w/o Share       |     0.6595 |      0.6630 +/- 0.0009 |     0.4158 |      0.4172 +/- 0.0011 |
| w/o CA          |     0.6644 |      0.6619 +/- 0.0019 |     0.4160 |      0.4156 +/- 0.0009 |
| 1-layer Adapter |     0.6108 |      0.6609 +/- 0.0026 |     0.3713 |      0.4152 +/- 0.0015 |
| Random Init     |     0.6440 |      0.6520 +/- 0.0016 |     0.3984 |      0.4062 +/- 0.0008 |