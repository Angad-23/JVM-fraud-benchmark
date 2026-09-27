"""Print the facts the methodology depends on. Paste the output into docs/lab_notebook.md.

Usage:  python s01_inspect_data.py [--csv path]
"""
import argparse
import pandas as pd
from common import load_config, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    path = a.csv or resolve(cfg, cfg["data"]["baf_csv"])
    df = pd.read_csv(path)
    d = cfg["data"]
    print("shape:", df.shape)
    print("fraud count:", int(df[d["label"]].sum()), " prevalence: %.4f%%" % (100 * df[d["label"]].mean()))
    print("\nrows / fraud / prevalence per month")
    print(df.groupby(d["time_col"])[d["label"]].agg(["size", "sum", "mean"]).to_string())
    print("\ndtypes")
    print(df.dtypes.to_string())
    print("\nconstant columns:", [c for c in df.columns if df[c].nunique() <= 1])
    print("\nmissing (NaN) counts:", df.isna().sum()[df.isna().sum() > 0].to_dict())
    neg1 = {c: int((df[c] == -1).sum()) for c in df.select_dtypes("number").columns if (df[c] == -1).any()}
    print("count of -1 (BAF missing marker) per column:", neg1)


if __name__ == "__main__":
    main()
