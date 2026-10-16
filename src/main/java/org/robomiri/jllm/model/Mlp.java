package org.robomiri.jllm.model;

/** The feed-forward block of one decoder layer. */
public sealed interface Mlp permits Mlp.Dense, Mlp.Moe {

    /** The gate activation of a gated MLP: {@code down(act(gate(x)) * up(x))}. */
    enum Activation {
        SILU,
        /** {@code gelu_pytorch_tanh} (Gemma). */
        GELU_TANH
    }

    /** A dense gated MLP of width {@code hidden}. */
    record Dense(int hidden, Activation act) implements Mlp {
    }

    /**
     * A mixture of experts: each token runs the {@code topK} experts the {@link Router} picks (each a SwiGLU of
     * width {@code moeDim}) and, when {@code shared} is set, a dense SwiGLU of every token on top (DeepSeek).
     *
     * @param normTopK renormalize the selected routing weights to sum to 1
     */
    record Moe(int experts, int topK, int moeDim, boolean normTopK, Router router, Dense shared) implements Mlp {
    }
}
