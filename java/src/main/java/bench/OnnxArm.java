package bench;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import java.util.Map;

/**
 * In-process ONNX Runtime running the SAME scikit-learn forest exported by s03_train_sklearn.py.
 * Same trees => identical predictions, so any latency difference is not an accuracy difference.
 * Per-request work timed: float[][] packing + tensor creation + run + read probability.
 */
public final class OnnxArm implements Arm {
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;

    public OnnxArm(String onnxPath, String[] unused) throws Exception {
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        so.setIntraOpNumThreads(1);
        this.session = env.createSession(onnxPath, so);
    }

    @Override public double score(double[] f) throws Exception {
        float[][] x = new float[1][f.length];
        for (int i = 0; i < f.length; i++) x[0][i] = (float) f[i];
        try (OnnxTensor t = OnnxTensor.createTensor(env, x);
             OrtSession.Result r = session.run(Map.of("input", t))) {
            float[][] p = (float[][]) r.get(1).getValue();   // output 0 = label, output 1 = probabilities
            return p[0][1];
        }
    }

    @Override public void close() throws Exception { session.close(); }
}
