"""Create a SMALL SYNTHETIC file with the same column layout as BAF Base.csv.

Purpose: smoke-test the whole pipeline before the real 213 MB file is used.
The numbers it produces are meaningless. Never report them.

Usage:  python s00_make_synthetic_baf.py --rows 60000 --out ../../data/raw/Base_synthetic.csv
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--out", default="../../data/raw/Base_synthetic.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = a.rows
    fraud = (rng.random(n) < 0.011).astype(int)
    sig = fraud * 1.0

    def miss(x, frac):  # BAF encodes missing as -1
        x = x.copy()
        x[rng.random(n) < frac] = -1
        return x

    df = pd.DataFrame({
        "fraud_bool": fraud,
        "income": np.round(rng.choice(np.arange(1, 10) / 10, n) + 0.05 * sig, 1).clip(0.1, 0.9),
        "name_email_similarity": rng.random(n) - 0.15 * sig,
        "prev_address_months_count": miss(rng.integers(0, 380, n), 0.7).astype(float),
        "current_address_months_count": miss(rng.integers(0, 400, n), 0.02).astype(float),
        "customer_age": rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90], n, p=[.02, .25, .3, .23, .13, .05, .01, .005, .005]) + (10 * sig).astype(int) * 0,
        "days_since_request": rng.exponential(0.5, n),
        "intended_balcon_amount": miss(rng.normal(10, 30, n) + 20 * sig, 0.74),
        "payment_type": rng.choice(["AA", "AB", "AC", "AD", "AE"], n, p=[.26, .37, .2, .15, .02]),
        "zip_count_4w": rng.integers(1, 5000, n),
        "velocity_6h": rng.normal(5000, 1500, n) + 300 * sig,
        "velocity_24h": rng.normal(4500, 1000, n),
        "velocity_4w": rng.normal(4800, 800, n),
        "bank_branch_count_8w": rng.integers(0, 2500, n),
        "date_of_birth_distinct_emails_4w": rng.integers(0, 40, n),
        "employment_status": rng.choice(["CA", "CB", "CC", "CD", "CE", "CF", "CG"], n),
        "credit_risk_score": rng.normal(130, 60, n) + 25 * sig,
        "email_is_free": (rng.random(n) < 0.5 + 0.2 * sig).astype(int),
        "housing_status": rng.choice(["BA", "BB", "BC", "BD", "BE", "BF", "BG"], n),
        "phone_home_valid": rng.integers(0, 2, n),
        "phone_mobile_valid": rng.integers(0, 2, n),
        "bank_months_count": miss(rng.integers(1, 30, n), 0.25).astype(float),
        "has_other_cards": rng.integers(0, 2, n),
        "proposed_credit_limit": rng.choice([200, 500, 1000, 1500, 2000], n),
        "foreign_request": (rng.random(n) < 0.02).astype(int),
        "source": rng.choice(["INTERNET", "TELEAPP"], n, p=[.99, .01]),
        "session_length_in_minutes": miss(rng.exponential(6, n), 0.02),
        "device_os": rng.choice(["linux", "windows", "macintosh", "other", "x11"], n),
        "keep_alive_session": rng.integers(0, 2, n),
        "device_distinct_emails_8w": rng.integers(0, 3, n),
        "device_fraud_count": np.zeros(n, dtype=int),   # constant column, as in real BAF
        "month": rng.integers(0, 8, n),
    })
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {df.shape}, fraud={df.fraud_bool.sum()} ({df.fraud_bool.mean():.4f})")


if __name__ == "__main__":
    main()
