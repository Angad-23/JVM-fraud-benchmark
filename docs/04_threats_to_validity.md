# 04 - Threats to validity (Section VI of the paper; edit as results come in)

**Internal validity (are the measurements right?)**
- *Single machine, hybrid CPU, laptop thermal limits.* Mitigation: power plan, P-core pinning, repeated sessions, raw data released. Residual noise remains: quantify with CV and between-session differences.
- *Closed-loop measurement.* The next request waits for the previous one, so queueing delays under load are invisible (a form of coordinated omission). Latencies here are service-time under no contention, not tail latency in production.
- *Harness differences between languages.* Java `nanoTime` vs Python `perf_counter_ns`, different clients. Mitigation: the Python and JVM ONNX arms should agree; report the comparison.
- *JIT and GC.* Warm-up of 20,000 calls is a chosen heuristic, not proof of steady state. Plot latency over the request index once to show stability.
- *Feature preparation excluded.* Encoding cost is outside the timed path for all arms.

**Construct validity (does it measure what the claim says?)**
- *localhost REST is not a production network.* It isolates software overhead; real deployments add network hops, TLS, load balancers, which strengthen the in-process advantage but are not measured.
- *Model size.* Random forest with 100 trees of <= 500 leaves is small; conclusions on latency do not extend to large ensembles or deep models.
- *Smile vs scikit-learn hyperparameter equivalence is approximate* (see 02, section 4). Any accuracy difference may be an implementation or parameter-semantics difference, not a "JVM vs Python" difference.

**External validity**
- One dataset, synthetic, account-opening fraud. Classification conclusions are not general. Fraud patterns drift, so a two-month test window is a snapshot.
- One hardware/OS/JDK/Python version combination. Report all versions.

**Statistical conclusion validity**
- Timings are autocorrelated: block bootstrap mitigates, does not remove. With n = 5,000, extreme tails (P99.9) rest on ~5 observations.
- Large n makes p-values trivially small: rely on effect sizes and CIs.

**Things the paper must NOT claim**
- That REST overhead is "irreducible" or that Python is inherently slow: the data will show which parts dominate.
- Production readiness, or "sub-millisecond" as a general property: report the measured median and P99 for the measured configuration.
- That the two forests are "identical". Only the ONNX copy is identical to scikit-learn.
- Any industry latency threshold (e.g., "100-300 ms", "50 ms SLA") without a verifiable source. Otherwise phrase it as "for illustration, a budget of X ms".
