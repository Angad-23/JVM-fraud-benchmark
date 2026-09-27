package bench;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/** The real "bridge": a Java client calling the Python REST service. Uses HTTP/1.1 keep-alive (JDK HttpClient pools connections). */
public final class RestArm implements Arm {
    private final HttpClient client = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(10)).build();
    private final URI uri;

    public RestArm(String url) { this.uri = URI.create(url); }

    @Override public double score(double[] f) throws Exception {
        StringBuilder sb = new StringBuilder(f.length * 12 + 20).append("{\"features\":[");
        for (int i = 0; i < f.length; i++) { if (i > 0) sb.append(','); sb.append((float) f[i]); }
        sb.append("]}");
        HttpRequest req = HttpRequest.newBuilder(uri).timeout(Duration.ofSeconds(30))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(sb.toString())).build();
        HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200) throw new IllegalStateException("HTTP " + resp.statusCode());
        String b = resp.body();
        int k = b.indexOf("\"prob\":");
        int s = k + 7, e = s;
        while (e < b.length() && "0123456789.eE+-".indexOf(b.charAt(e)) >= 0) e++;
        return Double.parseDouble(b.substring(s, e));
    }
}
