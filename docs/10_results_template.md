# 10 - Results template (numbers must be pasted from results/tables/; leave blank until real)

## Table A. Classification (test months; thresholds chosen on validation month)
| Model | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@5%FPR | Precision | Recall | F1 | MCC | Threshold rule |
|---|---|---|---|---|---|---|---|---|---|
| RF scikit-learn | | | | | | | | | FPR<=5% on val |
| RF Smile | | | | | | | | | FPR<=5% on val |
| RF scikit-learn | | | | | | | | | max-F1 on val |
| RF Smile | | | | | | | | | max-F1 on val |
Majority-class accuracy on test: ____ ; test rows: ____ ; test fraud: ____

Confusion matrices (TP / FP / FN / TN) for each row above: ____

## Table B. Latency (µs per transaction, batch = 1, n = 5,000 per arm, session S__)
| Arm | median [95% CI] | P95 | P99 [95% CI] | P99.9 | mean | SD | CV |
|---|---|---|---|---|---|---|---|
| py_sklearn_njobs1 | | | | | | | |
| py_sklearn_njobsall | | | | | | | |
| py_onnxruntime | | | | | | | |
| rest_flaskdev_nokeepalive_njobsall | | | | | | | |
| rest_waitress_nokeepalive_njobs1 | | | | | | | |
| rest_waitress_keepalive_njobs1 | | | | | | | |
| rest_waitress_keepalive_onnx | | | | | | | |
| jvm_rest_client_waitress | | | | | | | |
| jvm_smile_inprocess | | | | | | | |
| jvm_onnx_inprocess | | | | | | | |

## Table C. Pairwise comparisons (reference: ________)
| Arm | ratio of medians [95% CI] | Mann-Whitney p | Cliff's delta |
|---|---|---|---|

## Decomposition (fill from Table B)
- Model-only floor (in-process, same runtime): ____ µs
- + serialization/transport (REST, keep-alive): + ____ µs
- + connection setup (no keep-alive): + ____ µs
- + old-baseline effects (dev server, n_jobs=-1): + ____ µs
- JVM in-process (ONNX / Smile): ____ / ____ µs

## Session agreement
Between-session difference in medians per arm: ____ (compare with CI width)
