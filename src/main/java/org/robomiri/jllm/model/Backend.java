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
