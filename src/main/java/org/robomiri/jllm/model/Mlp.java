package org.robomiri.jllm.model;

/** The feed-forward block of one decoder layer. */
public sealed interface Mlp permits Mlp.Dense, Mlp.Moe {

    /** The gate activation of a gated MLP: {@code down(act(gate(x)) * up(x))}. */
    enum Activation {
        SILU,
        /** {@code gelu_pytorch_tanh} (Gemma). */
        GELU_TANH,
        /** {@code relu(x)^2} (Nemotron-H): the MLP has no gate, {@code down(relu(up(x))^2)}. */
        RELU2;

        /** True if the MLP multiplies an activated gate with an up projection (all but {@link #RELU2}). */
        public boolean gated() {
            return this != RELU2;
        }
    }

    /** A dense gated MLP of width {@code hidden}. */
    record Dense(int hidden, Activation act) implements Mlp {
    }

    /**
     * What the routed experts of a layer compute besides the plain {@code down(silu(gate x) * up x)} weighted by the router.
     *
     * @param clampLimit  gpt-oss's clamped SwiGLU: {@code gate = min(gate, limit)}, {@code up = clamp(up, -limit, limit)},
     *                    {@code h = (up + 1) * gate * sigmoid(alpha * gate)}; 0 = the plain SwiGLU
     * @param alpha       the sigmoid gain of the clamped SwiGLU
     * @param bias        every expert has a gate/up bias (added before the activation) and a down bias (added to the
     *                    expert's output, so weighted by the router with it)
     * @param weightInput Llama 4: the routing weight multiplies the expert's <em>input</em> (so gate and up before the
     *                    activation) instead of its output
     * @param interleaved the checkpoint's fused gate/up output columns alternate gate, up, gate, up (gpt-oss) instead of
     *                    holding all gate columns then all up columns (Llama 4); only the converter cares
     */
    record ExpertSpec(float clampLimit, float alpha, boolean bias, boolean weightInput, boolean interleaved) {
        /** The plain experts of Qwen3-MoE / DeepSeek. */
        public static final ExpertSpec PLAIN = new ExpertSpec(0f, 0f, false, false, false);

        public boolean clamped() {
            return clampLimit > 0;
        }

        /** True if the experts need more than the plain SwiGLU weighted at the output. */
        public boolean special() {
            return clamped() || bias || weightInput;
        }
    }

    /**
     * A mixture of experts: each token runs the {@code topK} experts the {@link Router} picks (each a SwiGLU of
     * width {@code moeDim}) and, when {@code shared} is set, a dense SwiGLU of every token on top (DeepSeek, Llama 4).
     *
     * @param normTopK renormalize the selected routing weights to sum to 1
     * @param spec     the experts' arithmetic beyond the plain SwiGLU ({@link ExpertSpec#PLAIN} for most families)
     */
    record Moe(int experts, int topK, int moeDim, boolean normTopK, Router router, Dense shared, ExpertSpec spec)
            implements Mlp {

        public Moe {
            if (spec == null) {
                spec = ExpertSpec.PLAIN;
            }
        }

        public Moe(int experts, int topK, int moeDim, boolean normTopK, Router router, Dense shared) {
            this(experts, topK, moeDim, normTopK, router, shared, ExpertSpec.PLAIN);
        }
    }
}
