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

    /** True if {@code o_proj} has a bias ({@code DecoderLayer.bo}), added to the projection before the residual. */
    boolean outBias();

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
     * @param outBias   {@code o_proj} has a bias too ({@code attention_bias} of Llama, Qwen3, Gemma)
     * @param scale     multiplier of {@code q.k} before the softmax
     * @param softcap   {@code tanh(s / cap) * cap} on the scaled score, 0 = off (Gemma 2)
     * @param sinks     one learned logit per query head joins the softmax as an extra "key" with a zero value, so a head
     *                  can attend to nothing (gpt-oss: {@code self_attn.sinks})
     * @param chunk     chunked attention (Llama 4): a query sees only the keys of its own chunk of {@code chunk}
     *                  positions, {@code [p / chunk * chunk, p]}; 0 = no chunking. Combined with a window, the stricter wins
     * @param l2Norm    parameterless RMS norm of every q / k head <em>after</em> the rotation (Llama 4's {@code use_qk_norm})
     * @param tempFloor Llama 4's attention temperature tuning of the NoPE layers: the query of position {@code p} is
     *                  multiplied by {@code 1 + tempScale * ln(1 + floor((p + 1) / tempFloor))}; 0 = off
     * @param tempScale see {@code tempFloor}
     */
    record Gqa(int heads, int kvHeads, int headDim, int window, int ropeTable, boolean qkNorm, boolean bias,
               boolean outBias, float scale, float softcap, boolean sinks, int chunk, boolean l2Norm, int tempFloor,
               float tempScale) implements Attention {

        /** The attention every family but gpt-oss and Llama 4 has: no sinks, no chunks, no L2 norm, no temperature. */
        public Gqa(int heads, int kvHeads, int headDim, int window, int ropeTable, boolean qkNorm, boolean bias,
                   boolean outBias, float scale, float softcap) {
            this(heads, kvHeads, headDim, window, ropeTable, qkNorm, bias, outBias, scale, softcap, false, 0, false, 0, 0f);
        }

        /** The keys a query at position {@code p} sees start at this position. */
        public int firstKey(int p) {
            int first = window > 0 ? Math.max(0, p + 1 - window) : 0;
            return chunk > 0 ? Math.max(first, p / chunk * chunk) : first;
        }

        /** The multiplier of a query at position {@code p} (1 unless {@link #tempFloor} is set). */
        public float tempFactor(int p) {
            return tempFloor <= 0 ? 1f : (float) (Math.log1p(Math.floor((p + 1.0) / tempFloor)) * tempScale + 1.0);
        }

        /** True if the layer needs more than the plain causal / windowed softmax. */
        public boolean special() {
            return sinks || chunk > 0 || tempFloor > 0;
        }

        /**
         * This layer as a backend of context length {@code ctx} sees it: a chunk or a temperature threshold that no position
         * below {@code ctx} reaches changes nothing, so it is switched off and the layer keeps the fast kernels.
         */
        public Gqa forContext(int ctx) {
            int c = chunk >= ctx ? 0 : chunk;
            int t = tempFloor > ctx ? 0 : tempFloor;
            if (c == chunk && t == tempFloor) {
                return this;
            }
            return new Gqa(heads, kvHeads, headDim, window, ropeTable, qkNorm, bias, outBias, scale, softcap, sinks, c, l2Norm,
                    t, tempScale);
        }

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

        /** DeepSeek's {@code attention_bias} would also bias q_a / kv_a; the family rejects it, so never. */
        @Override
        public boolean outBias() {
            return false;
        }

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
