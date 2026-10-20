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
     * <p><b>Prefill and decode are not bit-identical.</b> A backend may compute an {@code n > 1} call with
     * different kernels than an {@code n == 1} call (the CPU backend dequantizes four weight rows and tiles over
     * tokens for {@code n > 1}, streams quantized rows - or int8 products - for {@code n == 1}; Metal uses
     * different threadgroup shapes), so the same token's keys, values and logits differ in the last bits
     * depending on how many tokens its call carried. Consequently a result depends on history: when the engine
     * reuses cached positions, those were produced either by an earlier prefill or by earlier decode steps, and
     * the logits of the new tokens can differ (and after a near-tie, the greedy continuation can diverge) from
     * running the same prompt from position 0 in one go. Only the sequence of {@code forward} calls, not the
     * token sequence alone, determines the output. {@code -Djllm.cpu.deterministic=true} makes the CPU backend
     * run every call token by token through its single-token kernels for comparisons (docs/07 §7.14).
     *
     * @param logits if non-null, receives the logits of the last token
     */
    void forward(int[] tokens, int off, int n, int pos, float[] logits);

    /**
     * Like {@link #forward} but returns the logits of <em>every</em> token of the chunk: row {@code i} (the
     * distribution after {@code tokens[off + i]}) goes to {@code logitsOut[i * stride ..]}, {@code min(head rows,
     * stride)} values each. The K/V of all {@code n} positions are appended, exactly as for {@code forward}; a
     * caller that accepts only a prefix simply continues from the shorter position (later calls overwrite the
     * rest). Used to verify speculative drafts in one pass. The default runs {@code n} single-token steps.
     *
     * @param stride row stride of {@code logitsOut} (normally the model's vocabulary size)
     */
    default void forwardAll(int[] tokens, int off, int n, int pos, float[] logitsOut, int stride) {
        float[] row = new float[stride];
        for (int i = 0; i < n; i++) {
            forward(tokens, off + i, 1, pos + i, row);
            System.arraycopy(row, 0, logitsOut, i * stride, stride);
        }
    }

    /**
     * Whether the state after tokens {@code [0, N)} can be rewound to an earlier position. A KV cache can (positions
     * past {@code pos} are simply overwritten); a recurrent state (Mamba) cannot, so the engine then reuses the
     * cache only when the new prompt extends everything that was fed before, and otherwise restarts at position 0.
     */
    default boolean supportsPrefixReuse() {
        return true;
    }

    /**
     * The layout of {@link #saveState}'s output (cache type and shape), or null when this backend cannot save its
     * state. A saved state is only loaded into a backend reporting the same format.
     */
    default String stateFormat() {
        return null;
    }

    /** Writes the state after positions {@code [0, n)} (the KV cache). Only when {@link #stateFormat} is not null. */
    default void saveState(int n, java.nio.channels.WritableByteChannel out) throws java.io.IOException {
        throw new UnsupportedOperationException(name() + " cannot save its cache");
    }

    /** Replaces the state of positions {@code [0, n)} with what {@link #saveState} wrote. */
    default void loadState(int n, java.nio.channels.ReadableByteChannel in) throws java.io.IOException {
        throw new UnsupportedOperationException(name() + " cannot load a saved cache");
    }

    /**
     * Replaces the state of positions {@code [0, n)} with that of {@code other}, a backend of the same model with the
     * same {@link #stateFormat}: copying a finished prefix costs a memory copy instead of a prefill. Both backends
     * must be idle (the engines hold their locks).
     */
    default void copyStateFrom(Backend other, int n) {
        throw new UnsupportedOperationException(name() + " cannot copy a cache");
    }

    /**
     * Pipelined greedy decoding: each step feeds one token and leaves the argmax of its logits on the device, where the
     * next step can read it, so the next step is queued before the CPU has looked at the previous token. Used by the
     * engine for greedy requests without penalties; while a pipe has steps in flight, no other method of the backend
     * may be called (the engine drains it first).
     */
    interface GreedyPipe {
        /** Queues the step that feeds {@code token} at {@code pos}, or with {@code token < 0} the argmax of the step before. */
        void submit(int token, int pos);

        /** The argmax token of the oldest step not read yet (waits for that step only). */
        int next();

        /** Steps submitted and not read yet. */
        int pending();

        /** Waits for every step in flight and drops their unread results (their cache rows are simply overwritten later). */
        void drain();
    }

    /** A {@link GreedyPipe} over the token ids {@code [0, vocab)}, or null when this backend has none. */
    default GreedyPipe greedyPipe(int vocab) {
        return null;
    }

    @Override
    void close();
}
