just finished the first reproduction runs. our numbers come out much higher than the original paper's - i think it's because our dataset is actually different from theirs: re-preprocessing the current Yelp dump gives ~28x more users (441k vs 15.7k) and ~12x more items than the paper reports. right now i'm trying to run with the authors' released preprocessed files to match their exact split.

# Reproduction 1

| Backbone     | Group     | H@10 (Paper) |       H@10 (Ours) | N@10 (Paper) |       N@10 (Ours) |
| :----------- | :-------- | -----------: | ----------------: | -----------: | ----------------: |
| **SASRec**   | Overall   |       0.6673 | **0.9002** +/-.0044 |       0.4208 | **0.6947** +/-.0017 |
|              | Tail Item |       0.1893 |     0.6457 +/-.0291 |       0.0845 |     0.3505 +/-.0118 |
|              | Head Item |       0.8080 |     0.9431 +/-.0013 |       0.5199 |     0.7528 +/-.0039 |
|              | Tail User |       0.6685 |     0.8994 +/-.0039 |       0.4229 |     0.6971 +/-.0027 |
|              | Head User |       0.6627 |     0.9041 +/-.0074 |       0.4128 |     0.6818 +/-.0049 |
| **Bert4Rec** | Overall   |       0.6623 | **0.8912** +/-.0034 |       0.4222 | **0.6668** +/-.0033 |
|              | Tail Item |       0.1227 |     0.5616 +/-.0149 |       0.0500 |     0.2501 +/-.0071 |
|              | Head Item |       0.8212 |     0.9468 +/-.0014 |       0.5318 |     0.7371 +/-.0029 |
|              | Tail User |       0.6637 |     0.8920 +/-.0032 |       0.4247 |     0.6718 +/-.0031 |
|              | Head User |       0.6571 |     0.8869 +/-.0046 |       0.4127 |     0.6391 +/-.0043 |
| **GRU4Rec**  | Overall   |       0.5724 | **0.8831** +/-.0008 |       0.3413 | **0.6418** +/-.0013 |
|              | Tail Item |       0.0763 |     0.6047 +/-.0048 |       0.0318 |     0.2826 +/-.0044 |
|              | Head Item |       0.7184 |     0.9301 +/-.0012 |       0.4324 |     0.7024 +/-.0020 |
|              | Tail User |       0.5782 |     0.8822 +/-.0010 |       0.3456 |     0.6442 +/-.0012 |
|              | Head User |       0.5501 |     0.8883 +/-.0008 |       0.3247 |     0.6287 +/-.0026 |

