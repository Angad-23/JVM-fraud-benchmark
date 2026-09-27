# 02 - Methodology (write the paper's Section III from this)

## 1. Dataset
Bank Account Fraud (BAF) suite, **Base** variant (Jesus et al., NeurIPS 2022, Datasets and Benchmarks). 1,000,000 rows,
30 features, `month`, `fraud_bool`; 11,029 frauds (1.10%); synthetic data derived from a real bank account-opening fraud dataset
with noise addition, feature encoding and a CTGAN. Missing values are coded `-1`.

Scope note: this is account-opening fraud, so the paper should say "fraud risk scoring for banking applications/transactions",
not "payment authorization". Latency conclusions transfer to any tabular forest of comparable size; classification conclusions
apply only to this dataset.

## 2. Split (temporal, fixed in `python/config.yaml`)
Train on early months, validate on the next month, test on the final months. Default: train {0-4}, validation {5}, test {6,7}.
Confirm with `s01_inspect_data.py` that each part has enough fraud cases (aim for several hundred in test; state the counts).
The validation month is used **only** to select operating thresholds; the test months are touched once for final metrics.
No random shuffling, so the evaluation respects the direction of time.

## 3. Preprocessing
- Categorical columns (text columns such as `payment_type`, `employment_status`, `housing_status`, `source`, `device_os`) are ordinal-encoded with a
  mapping **fitted on the training months only**; unseen levels map to `-1`. Ordinal codes are acceptable for tree ensembles but impose an arbitrary
  order; state this as a limitation (alternatives: one-hot, target encoding).
- Constant columns in the training data are dropped (e.g., `device_fraud_count` is expected to be constant).
- `-1` missing markers are kept as a distinct value. Trees can split on it; nothing is replaced with 0.
- All arms receive the same float32 feature vector (`bench_rows.csv`), so preprocessing cost is identical and excluded from the timed path;
  state this explicitly (in production the encoding step also costs time, in both architectures).

## 4. Models
| Model | Library | Notes |
|---|---|---|
| RF-sklearn | scikit-learn `RandomForestClassifier`: 100 trees, `max_depth=20`, `max_leaf_nodes=500`, `min_samples_leaf=5`, `max_features="sqrt"`, `class_weight={0:1,1:90}`, `random_state=42` | Reference model; also exported to ONNX |
| RF-ONNX | The exact RF-sklearn model run by ONNX Runtime | Predictions match RF-sklearn to ~1e-7 (`onnx_parity.json`); latency arms only |
| RF-Smile | Smile 3.1.1 `RandomForest`: 100 trees, `maxDepth=20`, `maxNodes=500`, `nodeSize=5`, `mtry=floor(sqrt(p))`, Gini, class weights {1,90} | **Different implementation.** Parameters are analogous, not guaranteed identical: e.g., Smile's `nodeSize` and scikit-learn's `min_samples_leaf` are not the same constraint, and class weights are applied by each library's own rule. Report Smile's accuracy as measured. Never claim "identical models" |

The class weight 90 is the inverse prevalence in BAF (988,971 / 11,029 = 89.7). It is a fixed design choice, not tuned; a tuned
comparison (same tuning budget for both libraries) is future work.

## 5. Classification evaluation
On the test months, for each model: ROC-AUC, PR-AUC (average precision), TPR at 1% FPR and 5% FPR (threshold-free operating points),
and precision, recall, F1, MCC at two thresholds chosen on the validation month: (a) highest recall with FPR <= 5%, (b) maximum F1.
Also report accuracy alongside the majority-class accuracy (~98.9%) to show why accuracy is uninformative. Confusion matrices for **both** libraries.
All models pass through the same code (`common.evaluate`, `s08_evaluate_scores.py`).
Optional: bootstrap CIs on AUC/PR-AUC by resampling test rows.

## 6. Latency evaluation
Unit of measurement: **one transaction scored, batch size 1, closed-loop** (next request issued after the previous returns).

| Arm | Client | Path timed |
|---|---|---|
| `py_sklearn_njobs1`, `py_sklearn_njobsall` | Python | `predict_proba` on one row |
| `py_onnxruntime` | Python | ONNX Runtime `run` on one row |
| `rest_*` | Python `requests` | JSON encode -> HTTP -> server -> model -> JSON decode; keep-alive on or off as labeled |
| `jvm_rest_client_waitress` | Java `HttpClient` | same, from the JVM (the actual "bridge" scenario) |
| `jvm_smile_inprocess` | Java | `Tuple` creation + `predict` |
| `jvm_onnx_inprocess` | Java | float[] packing + tensor creation + `run` |

Design points: identical row sequence for all arms; warm-up calls discarded (20,000 for JVM in-process arms; 500 for REST and Python);
`System.nanoTime()` / `time.perf_counter_ns()`; IPv4 address `127.0.0.1` rather than `localhost`; 5,000 measured requests per arm;
each arm repeated in at least two separate sessions; raw per-request values stored.

## 7. Statistics
Report per arm: n, mean, SD, median, P95, P99, P99.9, min, max, coefficient of variation. Bootstrap 95% CIs (2,000 resamples, moving
blocks of 20 because consecutive timings are autocorrelated) for the median and P99. Pairwise vs a reference arm: ratio of medians with CI,
Mann-Whitney U (two-sided) and Cliff's delta. With thousands of samples p-values will be tiny; **effect sizes and CIs carry the message**.
Say "P95/P99 from N = 5,000 samples", never P95 from 30 samples.

## 8. What is deliberately out of scope (say so in the paper)
Concurrent load and throughput; batch scoring; feature-engineering cost; network hops between machines; gRPC; GPU inference; deep models;
gradient-boosted trees (recommended as a follow-up baseline for accuracy: LightGBM/XGBoost).
