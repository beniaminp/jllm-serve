package org.robomiri.jllm.model;

/** How a MoE layer turns the router logits of a token into its {@code topK} experts and their weights. */
public sealed interface Router permits Router.Softmax, Router.DeepSeek {

    /** Softmax over the logits, the top-k probabilities (transformers' {@code Qwen3MoeSparseMoeBlock}). */
    Softmax SOFTMAX = new Softmax();

    record Softmax() implements Router {
    }

    /** {@code topk_method} of the DeepSeek router. */
    enum TopK {
        GREEDY,
        GROUP_LIMITED_GREEDY,
        NOAUX_TC;

        /** The config.json spelling. */
        public String key() {
            return name().toLowerCase(java.util.Locale.ROOT);
        }
    }

    /**
     * The DeepSeek-V2 / V3 rule ({@link DeepSeekRouter}).
     *
     * @param sigmoid     score with a sigmoid (V3) instead of a softmax (V2)
     * @param nGroup      expert groups for the group-limited rules
     * @param topkGroup   groups kept by the group-limited rules
     * @param routedScale {@code routed_scaling_factor}
     * @param scaleAlways V3 applies {@code routedScale} also after renormalizing; V2 only when not renormalized
     */
    record DeepSeek(boolean sigmoid, TopK topK, int nGroup, int topkGroup, double routedScale, boolean scaleAlways)
            implements Router {

        public boolean hasGroups() {
            return topK != TopK.GREEDY;
        }

        /** {@code noaux_tc} selects on scores plus a learned {@code e_score_correction_bias}. */
        public boolean hasBias() {
            return topK == TopK.NOAUX_TC;
        }
    }
}
