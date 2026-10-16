package org.robomiri.jllm.model;

/** The attention of one decoder layer. */
public sealed interface Attention permits Attention.Gqa, Attention.Mla {

    int heads();

    /** Per-head query / key width: {@code head_dim}, or {@code nope + rope} for latent attention. */
    int headDim();

    /** Width of the projected queries of one token, {@code heads * headDim}. */
    int qDim();

    /** Width of the attention output of one token (the input of {@code o_proj}). */
    int outDim();

    /** Index into {@link Decoder#ropeTables()} of the table this layer rotates with. */
    int ropeTable();

    /** True if {@code o} is the same variant with the same head shape (what a backend sizes its caches from). */
    boolean sameShape(Attention o);

    /**
     * Grouped-query attention over a per-layer key/value cache.
     *
     * @param kvHeads   key/value heads ({@code heads} for MHA)
     * @param window    attention window in tokens, 0 = the full causal context
     * @param ropeTable the rotary table (Gemma 3 sliding layers use a second one)
     * @param qkNorm    per-head RMSNorm on queries and keys before RoPE (Qwen3, Gemma 3)
     * @param bias      q/k/v projections have a bias (Qwen2, GLM)
     * @param scale     multiplier of {@code q.k} before the softmax
     * @param softcap   {@code tanh(s / cap) * cap} on the scaled score, 0 = off (Gemma 2)
     */
    record Gqa(int heads, int kvHeads, int headDim, int window, int ropeTable, boolean qkNorm, boolean bias,
               float scale, float softcap) implements Attention {

        @Override
        public int qDim() {
            return heads * headDim;
        }

        public int kvDim() {
            return kvHeads * headDim;
        }

        @Override
        public int outDim() {
            return qDim();
        }

        @Override
        public boolean sameShape(Attention o) {
            return o instanceof Gqa g && g.heads == heads && g.kvHeads == kvHeads && g.headDim == headDim;
        }
    }

    /**
     * DeepSeek Multi-head Latent Attention (docs/14.11): the key/value of every head is expanded from one cached
     * {@code [c | k_pe]} latent row of {@code kvLoraRank + rope} values per token.
     *
     * @param qLoraRank rank of the query's low-rank projection, 0 when the query is a single {@code q_proj}
     * @param kvLoraRank width of the compressed key/value latent
     * @param nope      per-head key/query width that carries no position
     * @param rope      per-head query width (and the shared key width) that is rotated
     * @param vDim      per-head value width
     * @param scale     softmax scale: {@code (nope + rope)^-0.5}, times {@code mscale^2} for YaRN
     */
    record Mla(int heads, int qLoraRank, int kvLoraRank, int nope, int rope, int vDim, int ropeTable, float scale)
            implements Attention {

        /** Per-head query / key width. */
        public int qkDim() {
            return nope + rope;
        }

        @Override
        public int headDim() {
            return qkDim();
        }

        @Override
        public int qDim() {
            return heads * qkDim();
        }

        @Override
        public int outDim() {
            return heads * vDim;
        }

        /** Width of one cached latent row, {@code kvLoraRank + rope}. */
        public int latentDim() {
            return kvLoraRank + rope;
        }

        @Override
        public boolean sameShape(Attention o) {
            return o instanceof Mla m && m.heads == heads && m.qLoraRank == qLoraRank && m.kvLoraRank == kvLoraRank
                    && m.nope == nope && m.rope == rope && m.vDim == vDim;
        }
    }
}
