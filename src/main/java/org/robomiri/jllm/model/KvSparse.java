package org.robomiri.jllm.model;

/**
 * Sparse attention while decoding (docs/27): a decode step attends exactly over the blocks of the paged KV cache that
 * a cheap upper bound of {@code q·k} ranks highest, plus the first blocks (attention sinks) and the most recent ones.
 * Set once from the settings by the model loader ({@code backend.kv-sparse*}), read when a backend or a block pool is
 * built, like {@link KvDtype#q8Rotate()}; off by default, and off means nothing below is ever consulted.
 *
 * @param enabled      {@code backend.kv-sparse}
 * @param blocks       {@code backend.kv-sparse-blocks}: the blocks chosen by their bound, besides the forced ones
 * @param sinkTokens   {@code backend.kv-sparse-sink}: leading positions always attended (rounded up to whole blocks)
 * @param windowTokens {@code backend.kv-sparse-window}: trailing positions always attended (whole blocks, at least one)
 * @param denseLayers  {@code backend.kv-sparse-dense-layers}: the first layers keep dense attention
 */
public record KvSparse(boolean enabled, int blocks, int sinkTokens, int windowTokens, int denseLayers) {

    public static final KvSparse OFF = new KvSparse(false, 64, 16, 256, 2);

    private static volatile KvSparse current = OFF;

    public KvSparse {
        if (blocks < 1) {
            throw new IllegalArgumentException("backend.kv-sparse-blocks must be at least 1, got " + blocks);
        }
        if (sinkTokens < 0 || windowTokens < 0 || denseLayers < 0) {
            throw new IllegalArgumentException("backend.kv-sparse-sink, -window and -dense-layers must not be negative");
        }
    }

    /** The configuration backends built from now on use. */
    public static KvSparse get() {
        return current;
    }

    public static void set(KvSparse s) {
        current = s;
    }

    /** Blocks of the sink: {@code ceil(sinkTokens / B)}. */
    public int sinkBlocks(int blockTokens) {
        return (sinkTokens + blockTokens - 1) / blockTokens;
    }

    /** Blocks of the recent window, the one holding the current position included: {@code max(1, ceil(windowTokens / B))}. */
    public int windowBlocks(int blockTokens) {
        return Math.max(1, (windowTokens + blockTokens - 1) / blockTokens);
    }

    /** Blocks one decode step attends when the context is longer than this: sinks + window + {@link #blocks}. */
    public int budget(int blockTokens) {
        return sinkBlocks(blockTokens) + windowBlocks(blockTokens) + blocks;
    }

    /** Whether attention layer {@code layer} (0-based among the attention layers) selects blocks. */
    public boolean layer(int layer) {
        return enabled && layer >= denseLayers;
    }

    /**
     * A step of fewer than this many tokens (one decode step, or a speculative verification chunk, whose rows are
     * attended one by one) selects blocks; a chunk of this many or more is prefill and attends densely.
     */
    public static final int MAX_ROWS = 16;
}
