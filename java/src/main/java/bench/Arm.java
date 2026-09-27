package bench;

/** One inference path under test. score() must do ALL per-request work (feature packing, call, result read). */
public interface Arm extends AutoCloseable {
    double score(double[] features) throws Exception;
    @Override default void close() throws Exception {}
}
