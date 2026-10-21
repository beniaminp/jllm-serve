package org.robomiri.jllm.vision;

/**
 * Runs the vision transformer and the projector of a {@link ClipModel} on one device. An encoder is bound to one
 * projector file; {@link #encode} may be called from one thread at a time.
 */
public interface VisionEncoder extends AutoCloseable {

    /**
     * What one preprocessed image becomes.
     *
     * @param embd   {@code tokens x dim} floats: one embedding per image token, in the text model's embedding space
     * @param tokens number of image tokens
     * @param nx     tokens per row of the output grid (Qwen-VL: the merged patch grid; 0 when the tokens are not a grid)
     * @param ny     rows of the output grid
     */
    record Encoded(float[] embd, int tokens, int nx, int ny) {
    }

    /** Short description for logs, e.g. {@code cpu} or {@code metal (Apple M1 Pro)}. */
    String name();

    Encoded encode(Preprocess.Pixels image);

    @Override
    void close();
}
