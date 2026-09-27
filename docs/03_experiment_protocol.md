# 03 - Experiment protocol (run in this order; log everything in 08_lab_notebook.md)

## A. Prepare the machine (do this before EVERY session, and write what you did in the notebook)
Your laptop has a hybrid CPU (2 performance + 8 efficient cores), which makes timings noisy. Reduce the noise:
1. Plug in the charger. Windows: Settings -> System -> Power -> **Best performance**.
2. Close browsers, chat apps, cloud-sync clients, IDEs. Pause Windows Update. Keep the laptop cool and on a hard surface.
3. Wait a few minutes after boot so start-up tasks finish. Note the room/laptop state if it was already hot.
4. Optional but recommended: pin the benchmark processes to the performance cores so the OS scheduler cannot move them to efficient cores
   mid-run. In Windows, use Task Manager -> Details -> right-click process -> Set affinity, or `start /affinity F cmd` (mask `F` = logical CPUs 0-3,
   which on the i5-1235U are the two P-cores with hyper-threading; verify in Task Manager). Whatever you choose, use the **same** setting for all arms
   and state it in the paper.
5. Record versions: `python --version`, `pip freeze > results/tables/pip_freeze.txt`, `java -version`, `mvn -v`, Windows build, CPU model.
6. Run **each arm in at least two separate sessions** (different hour or day). If sessions disagree by more than the CI width, treat the noise as a finding and report it.

## B. Data and models (once)
```powershell
.\scripts\run_python_steps.ps1
# inspect results\tables\data_inspection.txt; edit python\config.yaml months if needed; re-run
cd java; mvn -q package
java -Xmx8g -XX:+UseG1GC -jar target\bench.jar train
cd ..\python\src; python s08_evaluate_scores.py --name smile
```
Checks that must pass before benchmarking:
- `results/tables/onnx_parity.json` -> `max_abs_diff` below 1e-4 (expected ~1e-7).
- `s08` did not print a label-mismatch error (Java and Python saw identical splits).
- `train.csv` row count equals `n_train` in `data/processed/meta.json`.

## C. Python in-process arms
```powershell
cd python\src
python s05_bench_inprocess.py            # py_sklearn_njobs1, py_sklearn_njobsall, py_onnxruntime
```

## D. REST arms (server in terminal 1, client in terminal 2; restart the server between configurations)
```powershell
# D1. replica of the OLD baseline (development server, n_jobs=-1, new TCP connection per request)
python s04_serve.py --server flask-dev --n-jobs -1
python s06_bench_rest.py --label rest_flaskdev_nokeepalive_njobsall --no-keepalive --server-desc "flask dev server, n_jobs=-1"

# D2. (No "flask-dev + keep-alive" arm: Flask's development server answers with HTTP/1.0 and "Connection: close",
#      so it cannot reuse connections. Verified while building this scaffold. Say so in the paper: the old baseline
#      necessarily paid a new TCP handshake per request.)

# D3. tuned bridge: waitress, n_jobs=1, keep-alive
python s04_serve.py --server waitress --n-jobs 1
python s06_bench_rest.py --label rest_waitress_keepalive_njobs1 --server-desc "waitress 4 threads, sklearn n_jobs=1"
# same server, new connection per request (isolates the connection-setup cost)
python s06_bench_rest.py --label rest_waitress_nokeepalive_njobs1 --no-keepalive --server-desc "waitress 4 threads, sklearn n_jobs=1"
java -jar ..\..\java\target\bench.jar latency --arm rest --label jvm_rest_client_waitress --rows ..\..\data\processed\bench_rows.csv --out ..\..\results

# D4. tuned bridge with ONNX backend
python s04_serve.py --server waitress --backend onnx --n-jobs 1
python s06_bench_rest.py --label rest_waitress_keepalive_onnx --server-desc "waitress, onnxruntime, 1 thread"
```
Optional (WSL/Linux/Docker only, Gunicorn does not run on Windows):
`gunicorn -w 4 -b 127.0.0.1:5000 "s04_serve:create_app('rf_sklearn.joblib', 1)"`.

Run the Java REST client against the same service configuration you name in the label. Keep server and client on the same machine
(and say so: this measures software overhead, not network latency).

## E. JVM in-process arms
```powershell
cd java
java -Xmx8g -XX:+UseG1GC -jar target\bench.jar latency --arm smile --label jvm_smile_inprocess
java -Xmx8g -XX:+UseG1GC -jar target\bench.jar latency --arm onnx  --label jvm_onnx_inprocess
```
Defaults: 20,000 warm-up calls, 5,000 measured. Keep the JVM flags identical across arms and report them. Optional: add `-Xlog:gc:file=results\raw\gc_smile.log`
to see whether GC pauses explain the tail.

## F. Statistics and figures
```powershell
cd python\src
python s07_stats.py --reference jvm_onnx_inprocess     # main comparison: identical model, different runtime
python s07_stats.py --reference jvm_smile_inprocess    # secondary
```
Outputs: `results/tables/latency_summary.*`, `latency_comparisons.*`, `results/figures/latency_ecdf.png`, `latency_box.png`.

## G. Sanity checks before you trust any number
- Median of `py_sklearn_njobs1` should be **lower** than `py_sklearn_njobsall` for single rows. If not, look at thread oversubscription.
- Every REST arm should be at least as slow as its in-process counterpart.
- Java `sink=` printed at the end must be a plausible number; if it prints 0.0 the model returned zeros: stop and debug.
- Compare `jvm_onnx_inprocess` and `py_onnxruntime`: same runtime, same model, both single-thread, so they should be in the same order of magnitude. A big gap means something is wrong in one harness.
- Re-run one arm and compare: medians within the bootstrap CIs.

## H. Optional extension if time allows (strengthens the paper a lot)
Concurrency: run 1, 4, 16 concurrent clients against waitress and against the in-process JVM (multi-threaded) and report throughput and P99.
Not implemented in this scaffold; if you do it, add a script and document it here. If you do not, list it as a limitation.
