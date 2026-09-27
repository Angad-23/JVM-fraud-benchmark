"""REST inference service (the "Python bridge").

Modes (pick one; report which you used in the paper):
  --server flask-dev   Flask development server (reproduces the old paper's baseline; NOT for production)
  --server waitress    multi-threaded production WSGI server (works on Windows)
  (Linux/WSL/Docker) gunicorn:  gunicorn -w 4 -b 127.0.0.1:5000 "s04_serve:create_app('rf_sklearn.joblib', 1)"

--n-jobs sets the model's INFERENCE thread count. n_jobs=-1 on a single row spawns
threads with large overhead; n_jobs=1 is the fair setting for single-transaction scoring.
--backend onnx serves the ONNX copy through ONNX Runtime instead of scikit-learn.

Endpoint: POST /predict   {"features": [f1, ..., fD]}   ->   {"prob": 0.0123}

Server-side instrumentation: the time spent inside the model call alone is recorded per request.
Every 1000 requests the server prints the median model-only time for the last 1000, so REST
latency can be decomposed into (model time) + (HTTP / framework / scheduling overhead).
"""
import argparse
import time
import joblib
import numpy as np
from flask import Flask, jsonify, request
from common import load_config, resolve


def create_app(model_file="rf_sklearn.joblib", n_jobs=1, backend="sklearn", config=None):
    cfg = load_config(config)
    mdir = resolve(cfg, cfg["paths"]["results_dir"]) / "models"
    app = Flask(__name__)

    if backend == "onnx":
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(n_jobs) if int(n_jobs) > 0 else 0
        sess = ort.InferenceSession(str(mdir / "rf_sklearn.onnx"), so, providers=["CPUExecutionProvider"])

        def score(x):
            return float(sess.run(None, {"input": x})[1][0, 1])
    else:
        clf = joblib.load(mdir / model_file)
        clf.set_params(n_jobs=int(n_jobs))

        def score(x):
            return float(clf.predict_proba(x)[0, 1])

    model_us = []

    @app.post("/predict")
    def predict():
        x = np.asarray(request.get_json(force=True)["features"], dtype=np.float32).reshape(1, -1)
        t0 = time.perf_counter()
        p = score(x)
        model_us.append((time.perf_counter() - t0) * 1e6)
        if len(model_us) % 1000 == 0:
            print(f"[server] requests={len(model_us)} model-only median over last 1000 = "
                  f"{np.median(model_us[-1000:]):.1f}us", flush=True)
        return jsonify({"prob": p})

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", choices=["flask-dev", "waitress"], default="waitress")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--backend", choices=["sklearn", "onnx"], default="sklearn")
    ap.add_argument("--threads", type=int, default=4, help="waitress worker threads")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    host = a.host or cfg["benchmark"]["service_host"]
    port = a.port or cfg["benchmark"]["service_port"]
    app = create_app("rf_sklearn.joblib", a.n_jobs, a.backend, a.config)
    print(f"serving backend={a.backend} n_jobs={a.n_jobs} server={a.server} on {host}:{port}")
    if a.server == "flask-dev":
        app.run(host=host, port=port, threaded=False)
    else:
        from waitress import serve
        serve(app, host=host, port=port, threads=a.threads)


if __name__ == "__main__":
    main()