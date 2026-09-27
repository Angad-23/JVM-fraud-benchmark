"""In-process single-row latency (no network): the model-cost floor for Python.

Arms:
  py_sklearn_njobs1     RandomForest.predict_proba, n_jobs=1
  py_sklearn_njobsall   same model, n_jobs=-1 (what the old paper accidentally used)
  py_onnxruntime        the ONNX copy of the same model via ONNX Runtime

One timing = one call scoring one transaction (batch size 1). Each arm gets warm-up
calls that are discarded, then N measured calls timed with time.perf_counter_ns().
"""
import argparse
import time
import joblib
import numpy as np
from common import load_config, resolve, sample_rows, write_raw


def time_calls(fn, rows, warmup):
    for i in range(warmup):
        fn(rows[i % len(rows)][None, :])
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        x = r[None, :]
        t0 = time.perf_counter_ns()
        fn(x)
        out[i] = (time.perf_counter_ns() - t0) / 1000.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    b = cfg["benchmark"]
    n = a.n or b["measured_requests"]
    rows = sample_rows(cfg, n)
    mdir = resolve(cfg, cfg["paths"]["results_dir"]) / "models"
    extra = {"batch_size": 1, "warmup": b["warmup_requests"], "seed": b["seed"]}

    clf = joblib.load(mdir / "rf_sklearn.joblib")
    for label, nj in (("py_sklearn_njobs1", 1), ("py_sklearn_njobsall", -1)):
        clf.set_params(n_jobs=nj)
        lat = time_calls(lambda x: clf.predict_proba(x), rows, b["warmup_requests"])
        write_raw(cfg, label, lat, {**extra, "n_jobs": nj})

    onnx_path = mdir / "rf_sklearn.onnx"
    if onnx_path.exists():
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        lat = time_calls(lambda x: sess.run(None, {"input": x}), rows, b["warmup_requests"])
        write_raw(cfg, "py_onnxruntime", lat, {**extra, "intra_op_threads": 1})


if __name__ == "__main__":
    main()
