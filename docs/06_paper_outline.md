# 06 - Paper outline (write only after results exist)

## Working titles (pick the one your results support)
1. *Where Does the Time Go? Decomposing Cross-Language Model-Serving Latency in JVM Fraud-Scoring Stacks*
2. *In-Process or Over REST? A Statistically Grounded Latency Comparison of Python, ONNX Runtime and Smile for Fraud Scoring on the JVM*
Avoid "revolutionary" wording and any multiplier in the title unless the data and CIs support it.

## Abstract template (fill from results; 150-250 words)
Problem (JVM stacks, Python-trained models, REST bridge) -> gap (prior comparisons conflate model cost and transport cost) -> what you did
(five arms, identical rows, BAF Base, temporal split) -> key results with CIs (median and P99 per arm; ratio of medians for ONNX in-process vs tuned REST bridge)
-> accuracy result (AUC/PR-AUC for both libraries) -> limitation in one clause (single machine, no concurrency) -> availability (code, raw timings, DOI).
Write the **numbers last**, copied from `results/tables/`.

## Sections
1. **Introduction**: problem; why a decomposition is needed; RQ1-RQ3; contributions (3 bullets, only what you actually deliver).
2. **Related work** (recent!): model-serving systems and latency; ONNX/PMML deployment; JVM ML libraries; fraud detection on tabular data with tree ensembles; benchmarking methodology (Georges et al.). State the gap precisely, no "no prior study exists" unless you did a systematic search (record the search terms).
3. **Method**: dataset and temporal split; preprocessing; models (with the Smile/scikit-learn equivalence caveat); latency arms table; measurement protocol; statistics. Reuse `02_methodology.md`.
4. **Experimental setup**: hardware/OS/JDK/Python/library versions; power and affinity settings; sessions.
5. **Results**: (a) latency table with medians, P95, P99, CIs; (b) ECDF figure; (c) decomposition: in-process floor -> serialization/HTTP -> server; (d) ONNX-vs-Smile in-JVM comparison; (e) classification table with AUC, PR-AUC, TPR@FPR, thresholds; confusion matrices for both libraries.
6. **Discussion**: what dominates the cost (data-driven); when REST is acceptable and when in-process wins; practical guidance (ONNX export as the low-risk path; native library when you must train in the JVM); the precision-recall differences discussed as *operating-point* differences unless you have evidence otherwise.
7. **Threats to validity and limitations**: from `04_threats_to_validity.md`, edited to match results.
8. **Conclusion and future work**: modest; concurrency, gRPC, GBDT models, larger models, real network, second dataset.
9. **Availability**: GitHub URL + Zenodo DOI; data source and license; how to reproduce.

## Claims you may make only if the data shows them
- "X% of the REST bridge latency is model computation / serialization / connection setup" (needs the decomposition arms).
- "ONNX Runtime in the JVM matches scikit-learn predictions (max difference 1e-7) at Y times lower median latency" (needs `onnx_parity.json` and the ratio with CI).
- "Smile achieves lower/higher AUC than scikit-learn under these settings" (needs `metrics_smile.json`; do not attribute causes you did not test).

## Figures and tables (all generated from files, none hand-typed)
Fig 1 architecture of arms (redraw cleanly). Fig 2 ECDF (`latency_ecdf.png`). Fig 3 box/violin (`latency_box.png`). Table I arms. Table II latency summary.
Table III pairwise comparisons. Table IV classification metrics. Optional: latency over request index for JIT/GC behavior.
