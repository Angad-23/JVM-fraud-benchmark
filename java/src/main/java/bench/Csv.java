package bench;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Minimal numeric CSV reader (no external dependency). Header required. */
public final class Csv {
    public final String[] header;
    public final double[][] rows;

    private Csv(String[] header, double[][] rows) { this.header = header; this.rows = rows; }

    public static Csv read(Path path) throws IOException {
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String[] header = br.readLine().split(",");
            List<double[]> out = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) continue;
                String[] p = line.split(",");
                double[] r = new double[p.length];
                for (int i = 0; i < p.length; i++) r[i] = Double.parseDouble(p[i]);
                out.add(r);
            }
            return new Csv(header, out.toArray(new double[0][]));
        }
    }

    /** Feature names = all columns except a trailing "label" column, if present. */
    public String[] featureNames() {
        int n = header[header.length - 1].equals("label") ? header.length - 1 : header.length;
        String[] f = new String[n];
        System.arraycopy(header, 0, f, 0, n);
        return f;
    }

    public double[][] features() {
        int d = featureNames().length;
        double[][] x = new double[rows.length][d];
        for (int i = 0; i < rows.length; i++) System.arraycopy(rows[i], 0, x[i], 0, d);
        return x;
    }

    public int[] labels() {
        int[] y = new int[rows.length];
        int c = header.length - 1;
        for (int i = 0; i < rows.length; i++) y[i] = (int) rows[i][c];
        return y;
    }
}
