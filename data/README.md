# Data

Data files are **not** included in this repository.

## Primary dataset: Bank Account Fraud (BAF) suite, Base variant
- Source: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- File to download: **Base.csv** (213.43 MB). Do NOT download the full 1.36 GB suite unless you need the variants.
- Paper to cite: Jesus, Pombal, Alves, Cruz, Saleiro, Ribeiro, Gama, Bizarro,
  "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation", NeurIPS 2022
  (Datasets and Benchmarks). arXiv:2211.13358.
- License shown on the Kaggle page when this project was set up: **CC BY-NC-SA 4.0**. Re-check it on the day you
  publish, and do not redistribute the data or modified copies without honoring that license.
- Nature of the data: synthetic, generated from an anonymized real bank account-opening fraud dataset
  (noise addition, feature encoding and a CTGAN). Describe it as synthetic in the paper; do not call it "real transactions".
- Facts from the Kaggle page: 1,000,000 rows, 30 features + `month` + `fraud_bool`, 11,029 fraud (1.10%).
  Missing values are encoded as **-1**.

Place the file at `data/raw/Base.csv`.

## Optional second dataset: IEEE-CIS Fraud Detection (Vesta, Kaggle 2019)
Real e-commerce payment fraud. Use it only as a second dataset for the accuracy half; latency does not depend on the dataset.
If you add it, write a separate loader that: uses a temporal split on `TransactionDT`, keeps categorical columns
(encoded on train only), and does not replace missing values with 0.0.

## Synthetic smoke-test data
`python python/src/s00_make_synthetic_baf.py` creates a small fake file with the BAF column layout. Use it only to
check the code runs. Never report results from it.
