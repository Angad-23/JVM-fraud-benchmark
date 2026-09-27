# 01 - Project plan and what changed from the rejected manuscript

Both venues (IEEE Access, Applied Sciences) desk-rejected the earlier version without reviewer comments, so the reasons below are
informed inference, not quotes. The design fixes every one of them.

## Problems in the old version -> how this project addresses them

| Old problem | Fix in this project | Where |
|---|---|---|
| JVM in-process call compared with a full Flask HTTP round trip; REST cost never separated from model cost | Latency **decomposition** with five arms (in-process Python, REST variants, JVM REST client, JVM Smile, JVM ONNX) | 02, 03 |
| Baseline was Flask's development server, `n_jobs=-1` on single rows, new TCP connection each call | Tuned bridge (waitress, `n_jobs=1`, keep-alive) **plus** the old configuration as a documented replica | `s04`, `s06` |
| No ONNX/JPMML path (the standard industry answer) | ONNX Runtime arm running the identical forest | `s03`, `OnnxArm` |
| "Statistically rigorous" with only mean/SD from 30 samples | 5,000 samples per arm, raw data saved, block-bootstrap CIs, Mann-Whitney, Cliff's delta, median/P95/P99/P99.9 | `s07` |
| 5 warm-up runs | 20,000 discarded warm-up calls for JVM arms | `LatencyBench` |
| Random shuffle split; 20 hand-picked features; missing -> 0.0 | Temporal split by month, all usable features, `-1` kept | `s02`, `common.py` |
| Accuracy as headline; single threshold; "identical hyperparameters" claim | AUC/PR-AUC/TPR@FPR, validation-chosen thresholds, alignment caveats stated | `s03`, `s08`, 04 |
| Smile vs scikit-learn differences explained by speculation | Not explained away; reported as measured, and the ONNX arm removes the confound for latency | 02, 04 |
| Unverifiable references, claims not supported by citations | Reference audit with per-claim status | 05 |
| Template leftovers, vague AI disclosure, no code link | Checklist and code/DOI plan | 07 |
| Simultaneous submission to two venues | Single-venue rule | 07 |

## Research questions (state these in the paper)
- **RQ1.** For single-transaction scoring on one machine, how is end-to-end latency of a Python-trained forest split between
  model computation, serialization/transport, and server overhead?
- **RQ2.** How much of that cost can a JVM stack remove by running the model in-process, via a JVM-native library (Smile) or
  via ONNX Runtime executing the same trained model?
- **RQ3.** Does a JVM-native library change classification quality relative to scikit-learn under matched settings on a
  temporally split, imbalanced fraud dataset?

Do **not** promise a headline multiplier in advance. Report what the data shows, including if REST overhead turns out small.

## Seven-day schedule (adjust; do not skip the stopping rules)

| Day | Work | Done when |
|---|---|---|
| 1 | Set up Python/JDK/Maven. Download `Base.csv`. Run `scripts/run_python_steps`. Record data facts in the notebook. Start the **reference audit** (05). | `results/tables/data_inspection.txt` exists; month range confirmed; config finalized |
| 2 | Build the Java jar (`mvn package`). Fix any Smile/ONNX API compile errors (see 09). Run `java ... train`, then `s08 --name smile`. | Smile metrics JSON exists; labels matched Python split |
| 3 | Python latency arms (`s05`, `s04` + `s06`). Machine prepared per protocol. | Raw CSVs for all Python arms |
| 4 | JVM latency arms (`smile`, `onnx`, `rest`). Repeat everything in a second session on another day/hour if possible. | Raw CSVs for all JVM arms x 2 sessions |
| 5 | `s07_stats.py`; threshold/metrics tables; concurrency test if time allows (see 04/06). Fill `10_results_template.md`. | All tables and figures generated from files |
| 6 | Write the paper from `06_paper_outline.md`. Finish the reference audit; delete anything unverified. | Full draft; every citation checked against its DOI/URL |
| 7 | Proofread, template, code archive (GitHub + Zenodo DOI), AI-use statement, choose ONE venue, submit. | Submission checklist (07) fully ticked |

**Stopping rules.** If Day 2 ends with the Smile jar still not compiling, run everything without the Smile arm and reduce the paper's
claim to "ONNX Runtime in-process vs REST bridge" (still publishable as an engineering benchmark). If Day 5 ends with unstable timings
(large drift between sessions), fix the machine setup before writing anything.
