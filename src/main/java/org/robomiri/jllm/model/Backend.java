package org.robomiri.jllm.model;

/** A device that runs the transformer forward pass and owns the KV cache. */
public interface Backend extends AutoCloseable {

    String name();

    /** Maximum sequence length (KV cache capacity). */
    int contextLength();

    /** Maximum number of tokens per {@link #forward} call (prefill chunk size). */
    int maxBatch();

    /**
     * Runs {@code n} tokens {@code tokens[off .. off+n)} occupying sequence positions {@code [pos, pos+n)},
     * appending their keys/values to the cache. Positions after {@code pos+n} are considered invalid afterwards.
     *
     * @param logits if non-null, receives the logits of the last token
     */
    void forward(int[] tokens, int off, int n, int pos, float[] logits);

    /**
     * Whether the state after tokens {@code [0, N)} can be rewound to an earlier position. A KV cache can (positions
     * past {@code pos} are simply overwritten); a recurrent state (Mamba) cannot, so the engine then reuses the
     * cache only when the new prompt extends everything that was fed before, and otherwise restarts at position 0.
     */
    default boolean supportsPrefixReuse() {
        return true;
    }

    @Override
    void close();
}
