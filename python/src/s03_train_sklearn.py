"""Train the scikit-learn Random Forest, evaluate, and export to ONNX.

Thresholds are chosen on the VALIDATION month only, then applied to test.
Two operating points are reported: (a) FPR<=5% and (b) best F1 on validation.
Also writes an ONNX copy of the SAME model, so a JVM can run the identical
trees (no accuracy difference to explain away) via ONNX Runtime.
"""
import argparse
import json
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from common import (load_config, resolve, load_processed, evaluate,
                    threshold_at_fpr, threshold_best_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-onnx", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    m = cfg["model"]
    npz, meta = load_processed(cfg)
    res_dir = resolve(cfg, cfg["paths"]["results_dir"])
    (res_dir / "models").mkdir(parents=True, exist_ok=True)
    (res_dir / "tables").mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xva, yva, Xte, yte = (npz[k] for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"))
    clf = RandomForestClassifier(
        n_estimators=m["n_estimators"], max_depth=m["max_depth"],
        max_leaf_nodes=m["max_leaf_nodes"], min_samples_leaf=m["min_samples_leaf"],
        max_features=m["max_features"], class_weight={0: 1, 1: m["fraud_class_weight"]},
        random_state=m["random_state"], n_jobs=-1)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    train_s = time.time() - t0

    pva = clf.predict_proba(Xva)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    thr_fpr = threshold_at_fpr(yva, pva, 0.05)
    thr_f1 = threshold_best_f1(yva, pva)
    out = {
        "model": "sklearn RandomForest", "params": m, "train_seconds": train_s,
        "n_features": meta["n_features"],
        "test_at_val_fpr5_threshold": evaluate(yte, pte, thr_fpr),
        "test_at_val_bestf1_threshold": evaluate(yte, pte, thr_f1),
        "test_at_default_0.5": evaluate(yte, pte, 0.5),
    }
    (res_dir / "tables" / "metrics_sklearn.json").write_text(json.dumps(out, indent=2))
    import pandas as pd
    pd.DataFrame({"label": yva, "score": pva}).to_csv(res_dir / "tables" / "scores_sklearn_val.csv", index=False)
    pd.DataFrame({"label": yte, "score": pte}).to_csv(res_dir / "tables" / "scores_sklearn_test.csv", index=False)
    joblib.dump(clf, res_dir / "models" / "rf_sklearn.joblib")
    json.dump({"fpr5": thr_fpr, "bestf1": thr_f1}, open(res_dir / "models" / "thresholds_sklearn.json", "w"))
    for k in ("test_at_val_fpr5_threshold", "test_at_val_bestf1_threshold"):
        r = out[k]
        print(f"{k}: AUC={r['roc_auc']:.4f} PR-AUC={r['pr_auc']:.4f} P={r['precision']:.3f} "
              f"R={r['recall']:.3f} MCC={r['mcc']:.4f}")

    if not a.skip_onnx:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnxruntime as ort
        onx = convert_sklearn(clf, initial_types=[("input", FloatTensorType([None, Xtr.shape[1]]))],
                              options={id(clf): {"zipmap": False}}, target_opset=15)
        path = res_dir / "models" / "rf_sklearn.onnx"
        path.write_bytes(onx.SerializeToString())
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        sub = Xte[:5000]
        p_onnx = sess.run(None, {"input": sub})[1][:, 1]
        diff = float(np.max(np.abs(p_onnx - clf.predict_proba(sub)[:, 1])))
        print(f"ONNX parity: max |p_onnx - p_sklearn| on {len(sub)} rows = {diff:.2e}")
        (res_dir / "tables" / "onnx_parity.json").write_text(json.dumps({"max_abs_diff": diff, "rows": len(sub)}))
        if diff > 1e-4:
            print("WARNING: parity above 1e-4; investigate before using the ONNX arm in the paper.")


if __name__ == "__main__":
    main()
