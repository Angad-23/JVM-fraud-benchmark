package bench;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import smile.classification.RandomForest;
import smile.data.Tuple;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.type.StructType;

/**
 * In-process Smile Random Forest. UNTESTED against a live Smile install when this scaffold was written
 * (see docs/09_known_gaps.md). If your Smile version changes an API, edit only this class,
 * TrainSmile.java and pom.xml.
 * Per-request work timed: build a Tuple from the double[] + predict(Tuple, posteriori).
 */
public final class SmileArm implements Arm {
    private final RandomForest model;
    private final StructType schema;
    private final double[] posteriori = new double[2];

    public SmileArm(String modelPath, String[] featureNames) throws Exception {
        try (ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(modelPath)))) {
            this.model = (RandomForest) in.readObject();
        }
        StructField[] fields = new StructField[featureNames.length];
        for (int i = 0; i < fields.length; i++) fields[i] = new StructField(featureNames[i], DataTypes.DoubleType);
        this.schema = new StructType(fields);
    }

    @Override public double score(double[] f) {
        Tuple t = Tuple.of(f, schema);
        model.predict(t, posteriori);
        return posteriori[1];
    }
}
