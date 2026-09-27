"""Shared helpers: config, data loading, temporal split, encoding, metrics.

Design rules (see docs/02_methodology.md):
  * temporal split by month (no random shuffle)
  * encoders and constant-column removal are fitted on TRAIN only
  * the validation month is used only to choose thresholds
  * BAF encodes missing values as -1; trees can use that directly, so we keep it
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

SRC_DIR = Path(__file__).resolve().parent
PY_DIR = SRC_DIR.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PY_DIR / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = path.parent
    return cfg


def resolve(cfg: dict, rel: str) -> Path:
    return (cfg["_root"] / rel).resolve()


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, cfg: dict):
    d = cfg["data"]
    t = df[d["time_col"]]
    tr = df[t.isin(d["train_months"])]
    va = df[t.isin(d["val_months"])]
    te = df[t.isin(d["test_months"])]
    used = set(d["train_months"]) | set(d["val_months"]) | set(d["test_months"])
    overlap = (set(d["train_months"]) & set(d["val_months"])) | \
              (set(d["train_months"]) & set(d["test_months"])) | \
              (set(d["val_months"]) & set(d["test_months"]))
    if overlap:
        raise ValueError(f"Month sets overlap: {sorted(overlap)}")
    missing = used - set(t.unique())
    if missing:
        raise ValueError(f"Months not present in data: {sorted(missing)}")
    return tr.copy(), va.copy(), te.copy()


def fit_encoders(train: pd.DataFrame, label: str, time_col: str):
    """Fit on TRAIN only. Returns (feature_names, categorical_maps)."""
    feats = [c for c in train.columns if c not in (label, time_col)]
    cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(train[c])]
    cat_maps = {}
    for c in cat_cols:
        levels = sorted(train[c].dropna().astype(str).unique().tolist())
        cat_maps[c] = {lvl: i for i, lvl in enumerate(levels)}
    # drop constant columns (after encoding, on train)
    keep = []
    for c in feats:
        s = train[c].map(cat_maps[c]) if c in cat_maps else train[c]
        if s.nunique(dropna=False) > 1:
            keep.append(c)
    dropped = [c for c in feats if c not in keep]
    cat_maps = {c: m for c, m in cat_maps.items() if c in keep}
    return keep, cat_maps, dropped


def apply_encoders(df: pd.DataFrame, feats, cat_maps) -> np.ndarray:
    out = pd.DataFrame(index=df.index)
    for c in feats:
        if c in cat_maps:
            out[c] = df[c].astype(str).map(cat_maps[c]).fillna(-1)  # unseen level -> -1
        else:
            out[c] = df[c]
    return out.to_numpy(dtype=np.float32)


def load_processed(cfg: dict):
    """Load the arrays written by 02_prepare_data.py."""
    pdir = resolve(cfg, cfg["data"]["processed_dir"])
    npz = np.load(pdir / "arrays.npz")
    meta = json.loads((pdir / "meta.json").read_text())
    return npz, meta


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def threshold_at_fpr(y, p, target_fpr: float) -> float:
    """Highest-recall threshold whose FPR <= target_fpr (chosen on validation)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(fpr <= target_fpr)[0]
    return float(thr[ok[-1]]) if len(ok) else 1.0


def threshold_best_f1(y, p, grid: int = 400) -> float:
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, grid)))
    best_t, best_f = 0.5, -1.0
    for t in qs:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def evaluate(y, p, threshold: float) -> dict:
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, p)
    def tpr_at(f):
        idx = np.where(fpr <= f)[0]
        return float(tpr[idx[-1]]) if len(idx) else 0.0
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "tpr_at_1pct_fpr": tpr_at(0.01),
        "tpr_at_5pct_fpr": tpr_at(0.05),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yhat)),
        "accuracy": float((yhat == y).mean()),
        "majority_class_accuracy": float(1 - np.mean(y)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y)), "n_fraud": int(np.sum(y)),
    }


# ----------------------------------------------------------------------------
# Benchmark helpers
# ----------------------------------------------------------------------------
def env_info() -> dict:
    import platform, os, sys
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count_logical": os.cpu_count(),
            "python": sys.version.split()[0]}
    for pkg in ("numpy", "sklearn", "onnxruntime", "flask", "waitress", "requests"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "?")
        except Exception:
            info[pkg] = None
    return info


def write_raw(cfg: dict, label: str, latencies_us, extra: dict):
    """Write raw per-request latencies (one column, microseconds) + a JSON sidecar."""
    d = resolve(cfg, cfg["paths"]["results_dir"]) / "raw"
    d.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(latencies_us, dtype=np.float64)
    np.savetxt(d / f"{label}.csv", arr, header="latency_us", comments="", fmt="%.3f")
    meta = {"label": label, "n": int(len(arr)), "env": env_info(), **extra}
    (d / f"{label}.json").write_text(json.dumps(meta, indent=2))
    q = np.percentile(arr, [50, 95, 99])
    print(f"{label}: n={len(arr)} median={q[0]:.1f}us p95={q[1]:.1f}us p99={q[2]:.1f}us mean={arr.mean():.1f}us")


BENCH_POOL = 20000   # size of the fixed benchmark row sequence shared by all arms (Python and Java)


def bench_matrix(Xte, seed: int):
    """Fixed-seed sequence of test rows. Written to data/processed/bench_rows.csv so that
    Java arms replay EXACTLY the same rows in the same order as the Python arms."""
    rng = np.random.default_rng(seed)
    return Xte[rng.integers(0, len(Xte), size=BENCH_POOL)]


def sample_rows(cfg: dict, n: int):
    """First n rows of the shared benchmark sequence (float32)."""
    if n > BENCH_POOL:
        raise ValueError(f"n={n} exceeds the shared pool of {BENCH_POOL} rows")
    npz, _ = load_processed(cfg)
    return bench_matrix(npz["X_test"], cfg["benchmark"]["seed"])[:n]
