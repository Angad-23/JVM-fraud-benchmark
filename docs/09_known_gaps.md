# 09 - Known gaps and things to verify (read before running)

## Not tested when this scaffold was written
1. **`TrainSmile.java`, `SmileArm.java`, `OnnxArm.java`, `pom.xml`** were never compiled here because the build environment could not reach Maven Central. The Python code and the dependency-free Java classes *were* run.
   Expect to fix small API mismatches on first `mvn package`. Points most likely to need attention (Smile 3.1.1):
   - `RandomForest.fit(Formula, DataFrame, ntrees, mtry, SplitRule, maxDepth, maxNodes, nodeSize, subsample, int[] classWeight, LongStream seeds)` - argument order/types.
   - `DataFrame.of(double[][], String...)` and `DataFrame.merge(IntVector)`.
   - `Tuple.of(double[], StructType)` and `RandomForest.predict(Tuple, double[] posteriori)`.
   - `new StructField(name, DataTypes.DoubleType)` constructor.
   - Serialization of the trained model with `ObjectOutputStream`.
   If Smile 3.1.1 does not match, either adjust to the documentation of the version you installed or pin another version. If Smile 4.x needs Java 21+, that is fine with JDK 21. **Report the exact Smile version in the paper.**
   In ONNX Runtime Java: the output index (`r.get(1)` = probabilities) depends on the exporter; the Python parity test confirms the model, but check the Java result equals `predict_proba` for a few rows.
2. **Model equivalence.** Smile's `nodeSize` and scikit-learn's `min_samples_leaf` are different constraints; class weighting is implemented differently; sampling details differ. The scaffold does not try to force equality. Report both models as separate implementations.
3. **ONNX conversion** was validated on synthetic data only (max difference 1.7e-7 there). Re-check `onnx_parity.json` on the real model.
4. **Timing numbers.** Any numbers seen while building this scaffold came from a synthetic dataset on a different machine and mean nothing for your paper.
5. **Flask development server** replies HTTP/1.0 with `Connection: close`; keep-alive cannot be enabled there (verified). Use waitress for keep-alive comparisons. Gunicorn does not run natively on Windows; use WSL/Docker if you want it.
6. **BAF column details** used by the synthetic generator (names beyond those visible on the Kaggle page) come from memory of the dataset. The real pipeline discovers columns from the file, so it does not depend on this, but confirm with `s01_inspect_data.py`.
7. **No concurrency test** is implemented. If you claim anything about throughput or behavior under load, you must add and document one.
8. **Ordinal encoding** of categorical columns is a simplification (see 02).
9. **Only one dataset** (BAF Base). IEEE-CIS is optional and would need its own loader.

## Improvement ideas if time remains (in priority order)
1. A JMH cross-check of the in-process JVM arms (JMH `SampleTime` mode).
2. Concurrency test (1/4/16 clients) for REST vs in-process.
3. gRPC or Unix-socket bridge arm.
4. LightGBM/XGBoost baseline for accuracy (with ONNX export for latency).
5. Second dataset (IEEE-CIS with a temporal split).
6. Bootstrap CIs on AUC/PR-AUC; multiple seeds for model training.
