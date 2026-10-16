package org.robomiri.jllm.model;

/**
 * One decoder layer: {@code x += attn(norm(x))}, then {@code x += mlp(norm(x))}, where a {@link Norms#post()}
 * family additionally normalizes each sub-layer's output before the residual add.
 *
 * @param attn  grouped-query or latent attention, with this layer's window, rotary table, scale and soft-cap
 * @param mlp   a dense gated MLP or a mixture of experts
 * @param norms which norms the layer has and the checkpoint names they are stored under
 */
public record LayerSpec(Attention attn, Mlp mlp, Norms norms) {

    /**
     * The RMSNorms of a layer and their tensor names. Every layer has {@code input_layernorm} before the
     * attention and a norm before the MLP; the sandwich families add a norm on each sub-layer's output.
     */
    public enum Norms {
        /** Llama and most others: {@code input_layernorm}, {@code post_attention_layernorm} (before the MLP). */
        PRE("post_attention_layernorm", null, null),
        /** GLM-4: the two above plus {@code post_self_attn_layernorm} / {@code post_mlp_layernorm} on the outputs. */
        SANDWICH_GLM4("post_attention_layernorm", "post_self_attn_layernorm", "post_mlp_layernorm"),
        /** Gemma: the MLP's input norm is {@code pre_feedforward_layernorm}; {@code post_attention_layernorm} is the attention output norm. */
        SANDWICH_GEMMA("pre_feedforward_layernorm", "post_attention_layernorm", "post_feedforward_layernorm");

        private final String ffnNorm;
        private final String postAttnNorm;
        private final String postMlpNorm;

        Norms(String ffnNorm, String postAttnNorm, String postMlpNorm) {
            this.ffnNorm = ffnNorm;
            this.postAttnNorm = postAttnNorm;
            this.postMlpNorm = postMlpNorm;
        }

        /** Tensor name (without the layer prefix and {@code .weight}) of the norm before the MLP. */
        public String ffnNorm() {
            return ffnNorm;
        }

        /** Tensor name of the norm on the attention output, null when the family has none. */
        public String postAttnNorm() {
            return postAttnNorm;
        }

        /** Tensor name of the norm on the MLP output, null when the family has none. */
        public String postMlpNorm() {
            return postMlpNorm;
        }

        /** True if the sub-layer outputs are normalized before they join the residual stream. */
        public boolean post() {
            return postAttnNorm != null;
        }
    }
}
