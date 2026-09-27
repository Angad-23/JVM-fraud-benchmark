package bench;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Closed-loop single-request latency harness.
 *  - replays the SAME row sequence as the Python arms (bench_rows.csv, written by Python)
 *  - warm-up requests are executed and discarded (default 20,000 for in-process arms so the JIT reaches steady state)
 *  - each measured call = one transaction, timed with System.nanoTime(); raw microseconds are written, one per line
 *  - a running sink of scores is kept and printed so the JIT cannot eliminate the call
 */
public final class LatencyBench {
    static Map<String, String> parse(String[] a) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i < a.length - 1; i += 2) m.put(a[i].replaceFirst("^--", ""), a[i + 1]);
        return m;
    }

    public static void run(String[] args) throws Exception {
        Map<String, String> o = parse(args);
        String armName = o.getOrDefault("arm", "");
        String label = o.getOrDefault("label", "jvm_" + armName);
        int n = Integer.parseInt(o.getOrDefault("n", "5000"));
        int warmup = Integer.parseInt(o.getOrDefault("warmup", armName.equals("rest") ? "500" : "20000"));
        Path rowsPath = Path.of(o.getOrDefault("rows", "../data/processed/bench_rows.csv"));
        Path outDir = Path.of(o.getOrDefault("out", "../results"));

        Csv csv = Csv.read(rowsPath);
        double[][] rows = csv.features();
        if (n > rows.length) throw new IllegalArgumentException("n exceeds bench_rows.csv length " + rows.length);

        Arm arm;
        switch (armName) {
            case "rest" -> arm = new RestArm(o.getOrDefault("url", "http://127.0.0.1:5000/predict"));
            case "smile" -> arm = load("bench.SmileArm", o.getOrDefault("model", "../results/models/rf_smile.ser"), csv.featureNames());
            case "onnx" -> arm = load("bench.OnnxArm", o.getOrDefault("model", "../results/models/rf_sklearn.onnx"), null);
            default -> throw new IllegalArgumentException("--arm must be smile|onnx|rest");
        }

        double sink = 0;
        try (arm) {
            double warmSink = 0;
            for (int i = 0; i < warmup; i++) warmSink += arm.score(rows[i % rows.length]);
            if (warmSink == 12345.6789) System.out.print("");            double[] us = new double[n];
            double[] sc = new double[n];
            for (int i = 0; i < n; i++) {
                double[] r = rows[i];
                long t0 = System.nanoTime();
                double s = arm.score(r);
                long t1 = System.nanoTime();
                sink += s;
                sc[i] = s;
                us[i] = (t1 - t0) / 1000.0;
            }
            write(outDir, label, us, armName, warmup, o);
            StringBuilder sb = new StringBuilder("score\n");
            double mn = Double.MAX_VALUE, mx = -Double.MAX_VALUE;
            for (double v : sc) { sb.append(v).append('\n'); mn = Math.min(mn, v); mx = Math.max(mx, v); }
            Files.writeString(outDir.resolve("raw").resolve(label + "_scores.csv"), sb.toString());
            System.out.printf(java.util.Locale.ROOT, "scores: min=%.6f max=%.6f first3=%.6f, %.6f, %.6f%n", mn, mx, sc[0], sc[1], sc[2]);
        }
        System.out.println("score_sum_measured=" + sink + "  (sum of scores over the measured rows only; use for cross-arm parity)");
    }

    private static Arm load(String cls, String model, String[] names) throws Exception {
        try {
            return (Arm) Class.forName(cls).getConstructor(String.class, String[].class).newInstance(model, names);
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException(cls + " not on classpath: build with Maven (mvn -q package) and run the shaded jar.", e);
        }
    }

    private static void write(Path outDir, String label, double[] us, String arm, int warmup, Map<String, String> o) throws IOException {
        Path raw = outDir.resolve("raw");
        Files.createDirectories(raw);
        StringBuilder sb = new StringBuilder("latency_us\n");
        for (double v : us) sb.append(String.format(java.util.Locale.ROOT, "%.3f%n", v));
        Files.writeString(raw.resolve(label + ".csv"), sb.toString());
        String json = String.format(java.util.Locale.ROOT,
                "{%n  \"label\": \"%s\",%n  \"arm\": \"%s\",%n  \"n\": %d,%n  \"warmup\": %d,%n  \"batch_size\": 1,%n"
                        + "  \"client\": \"java\",%n  \"java\": \"%s\",%n  \"jvm\": \"%s\",%n  \"os\": \"%s\",%n  \"cpu_count_logical\": %d,%n  \"max_heap_mb\": %d%n}%n",
                label, arm, us.length, warmup, System.getProperty("java.version"), System.getProperty("java.vm.name"),
                System.getProperty("os.name") + " " + System.getProperty("os.version"),
                Runtime.getRuntime().availableProcessors(), Runtime.getRuntime().maxMemory() / (1024 * 1024));
        Files.writeString(raw.resolve(label + ".json"), json);
        double[] s = us.clone();
        java.util.Arrays.sort(s);
        System.out.printf(java.util.Locale.ROOT, "%s: n=%d median=%.1fus p95=%.1fus p99=%.1fus max=%.1fus%n",
                label, s.length, s[s.length / 2], s[(int) (0.95 * (s.length - 1))], s[(int) (0.99 * (s.length - 1))], s[s.length - 1]);
    }
}
