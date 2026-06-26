after rerunning the experiments with the authors' released preprocessed data, we now get a faithful reproduction of the paper's results. all 3 backbones (SASRec / Bert4Rec / GRU4Rec) match the paper within +/-0.002 on overall H@10/N@10 and within +/-0.008 on every long-tail subgroup (tail/head item & user).

# Reproduction 2

| Backbone     | Group     | H@10 paper |     H@10 ours | N@10 paper |     N@10 ours |
| :----------- | :-------- | ---------: | ------------: | ---------: | ------------: |
| **SASRec**   | Overall   |     0.6673 | 0.6670 +/-.0021 |     0.4208 | 0.4192 +/-.0028 |
|              | Tail Item |     0.1893 | 0.1928 +/-.0003 |     0.0845 | 0.0849 +/-.0009 |
|              | Head Item |     0.8080 | 0.8065 +/-.0026 |     0.5199 | 0.5177 +/-.0037 |
|              | Tail User |     0.6685 | 0.6679 +/-.0019 |     0.4229 | 0.4209 +/-.0028 |
|              | Head User |     0.6627 | 0.6634 +/-.0029 |     0.4128 | 0.4127 +/-.0032 |
| **Bert4Rec** | Overall   |     0.6623 | 0.6639 +/-.0022 |     0.4222 | 0.4222 +/-.0019 |
|              | Tail Item |     0.1227 | 0.1304 +/-.0019 |     0.0500 | 0.0532 +/-.0011 |
|              | Head Item |     0.8212 | 0.8210 +/-.0034 |     0.5318 | 0.5309 +/-.0025 |
|              | Tail User |     0.6637 | 0.6641 +/-.0024 |     0.4247 | 0.4241 +/-.0027 |
|              | Head User |     0.6571 | 0.6633 +/-.0042 |     0.4127 | 0.4152 +/-.0013 |
| **GRU4Rec**  | Overall   |     0.5724 | 0.5726 +/-.0035 |     0.3413 | 0.3413 +/-.0010 |
|              | Tail Item |     0.0763 | 0.0846 +/-.0021 |     0.0318 | 0.0361 +/-.0013 |
|              | Head Item |     0.7184 | 0.7162 +/-.0045 |     0.4324 | 0.4311 +/-.0017 |
|              | Tail User |     0.5782 | 0.5786 +/-.0032 |     0.3456 | 0.3450 +/-.0019 |
|              | Head User |     0.5501 | 0.5497 +/-.0080 |     0.3247 | 0.3271 +/-.0028 |