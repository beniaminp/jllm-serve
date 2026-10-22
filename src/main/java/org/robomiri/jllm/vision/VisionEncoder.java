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
     * @param deepstack Qwen3-VL: features from intermediate layers of the vision transformer, each {@code tokens x dim}
     *                  like {@code embd}, which the text model adds to its hidden states after its first layers; null otherwise
     */
    record Encoded(float[] embd, int tokens, int nx, int ny, float[][] deepstack) {
        public Encoded(float[] embd, int tokens, int nx, int ny) {
            this(embd, tokens, nx, ny, null);
        }
    }

    /** Short description for logs, e.g. {@code cpu} or {@code metal (Apple M1 Pro)}. */
    String name();

    Encoded encode(Preprocess.Pixels image);

    @Override
    void close();
}
