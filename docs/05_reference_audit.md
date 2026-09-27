# 05 - Reference audit

IEEE Access's rejection letter stressed that references must be "relevant and accurate" and that "unidentifiable sources" undermine a paper.
**I cannot open journal sites from the build environment, so every entry below is marked VERIFY.** For each one: open the DOI/publisher page,
confirm authors, title, venue, volume, pages, year, and confirm the cited paper actually supports the sentence you use it for.
Delete anything you cannot verify. Do not keep a citation "because it sounds right".

Legend: OK-meta = bibliographic details look correct from memory but must still be checked; CLAIM = the sentence it supported in the old paper may not be supported by the source.

## A. References in the rejected manuscript

| # | Old reference | Issue to check | Action |
|---|---|---|---|
| 1 | Amazon Web Services, DJL, djl.ai, 2024 | Software documentation; fine for a tool citation | Keep only if DJL stays in the paper. Add access date |
| 2 | Carcillo, Le Borgne, Caelen, Mazzer(?), Bontempi, "Combining unsupervised and supervised learning in credit card fraud detection", Inf. Sci. 557 (2021) 317-331 | Author list looked wrong (co-authors include Kessaci and Oble, I believe). **CLAIM:** the old text said this paper's SCARFF framework kept Python scoring behind JVM bridges. SCARFF is a different paper (Carcillo et al., Information Fusion, 2018, Spark-based) | Verify authors. Cite the SCARFF paper for SCARFF. Remove the "bridge latency unmeasured" claim unless you read it in the source |
| 3 | Bhattacharyya, Jha, Tharakunnel, Westland, "Data mining for credit card fraud: a comparative study", Decis. Support Syst. 50(3) 602-613, 2011 | OK-meta. **CLAIM:** it compares logistic regression, SVM and random forests (not "decision tree vs random forest") and is not a source for a 100-300 ms latency requirement | Use only for what it studies |
| 4 | Bolton & Hand, "Statistical fraud detection: a review", Stat. Sci. 17(3) 235-255, 2002 | OK-meta. **CLAIM:** a 2002 review cannot support "$32 billion annual losses" | Find a current, citable loss statistic (e.g., an industry report) or drop the number |
| 5 | Chawla, Bowyer, Hall, Kegelmeyer, "SMOTE", JAIR 16, 321-357, 2002 | OK-meta | Keep if you discuss resampling |
| 6 | Dal Pozzolo, Caelen, Johnson, Bontempi, "Calibrating probability with undersampling for unbalanced classification", IEEE SSCI 2015, 159-166 | OK-meta. **CLAIM:** the paper is about calibration after **undersampling**, not "SMOTE distortion above 500,000 records" | Rewrite the sentence or drop |
| 7 | Breiman, "Random forests", Mach. Learn. 45(1) 5-32, 2001 | OK-meta | Keep |
| 8 | He & Garcia, "Learning from imbalanced data", IEEE TKDE 21(9) 1263-1284, 2009 | OK-meta. **CLAIM:** check that it discusses cost-sensitive/class-weight methods in the way you describe; "streaming contexts" is doubtful | Keep for general imbalance background |
| 9 | IEEE-CIS Fraud Detection, Kaggle, 2019 | A dataset page, cannot support "published feature-importance analyses" | Use only as dataset citation, if you use IEEE-CIS |
| 10 | Chicco & Jurman, MCC vs F1 and accuracy, BMC Genomics 21:6, 2020 | OK-meta | Keep for MCC justification |
| 11 | Oracle, "Java SE 21 LTS: Platform architecture and multithreading", 2025 | **Unidentifiable.** It was used for the GIL, JIT-benchmarking method and JDK claims | Remove. Replace as below |
| 12 | Sculley et al., "Hidden technical debt in machine learning systems", NeurIPS 28, 2015 | OK-meta | Keep, but do not overstate: it discusses system-level debt broadly, not "cross-language REST bridges" specifically |

## B. Candidate additions (VERIFY every detail before citing)

| Purpose | Candidate |
|---|---|
| Dataset (**required**) | Jesus, Pombal, Alves, Cruz, Saleiro, Ribeiro, Gama, Bizarro, "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation", NeurIPS 2022 (Datasets and Benchmarks); arXiv:2211.13358. BibTeX is on the Kaggle page |
| Benchmarking methodology (**replaces #11**) | Georges, Buytaert, Eeckhout, "Statistically rigorous Java performance evaluation", OOPSLA 2007 |
| JMH (if you add a cross-check) | OpenJDK Code Tools, Java Microbenchmark Harness (project page) |
| Tail latency | Dean & Barroso, "The Tail at Scale", Communications of the ACM, 2013 |
| Library citations | Pedregosa et al., "Scikit-learn: Machine learning in Python", JMLR 12 (2011); Smile (Haifeng Li), GitHub project/documentation; ONNX Runtime and ONNX project pages (with version and access date) |
| Tabular ML context | Grinsztajn, Oyallon, Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?", NeurIPS 2022 (Datasets and Benchmarks) |
| Fraud-detection practice | Le Borgne, Siblini, Lebichot, Bontempi, "Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook", 2022 (open online book) |
| Optional accuracy baselines | Chen & Guestrin, XGBoost, KDD 2016; Ke et al., LightGBM, NeurIPS 2017 |
| Model serving / JVM ML (search yourself) | 2019-2026 papers on model serving latency, ONNX/JPMML deployment, JVM ML libraries. Related work in the old paper leaned on 2002-2011 |

## C. Rules for the new manuscript
1. Every number and factual claim in the intro (losses, latency budgets) gets a verifiable source or is rephrased as an assumption.
2. Cite software with name, version, URL, access date.
3. No citation for a claim the source does not make. When unsure, cite less.
4. Keep a `paper/references.bib` and check each key against its DOI at the end (Day 6).
