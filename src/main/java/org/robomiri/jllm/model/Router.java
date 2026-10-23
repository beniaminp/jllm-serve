package org.robomiri.jllm.model;

/** How a MoE layer turns the router logits of a token into its {@code topK} experts and their weights. */
public sealed interface Router permits Router.Softmax, Router.DeepSeek, Router.TopLogits {

    /** Softmax over the logits, the top-k probabilities (transformers' {@code Qwen3MoeSparseMoeBlock}). */
    Softmax SOFTMAX = new Softmax();

    record Softmax() implements Router {
    }

    /**
     * Top-k on the raw router logits (plus the learned {@code bias}, a linear bias of the router), then a function of the
     * selected logits only: gpt-oss softmaxes them (the weights sum to 1), Llama 4 takes the sigmoid of each (no
     * normalization; with one expert per token that is a gate in (0, 1)).
     *
     * @param sigmoid score each selected logit with a sigmoid instead of a softmax over the selected ones
     * @param bias    the router has a bias vector {@code [experts]}, added to the logits before the selection
     */
    record TopLogits(boolean sigmoid, boolean bias) implements Router {
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
