package org.robomiri.jllm.model;

/**
 * Mamba (v1): embedding, {@code layers} × (RMSNorm → selective-scan mixer → residual), final RMSNorm, LM head;
 * no attention, a fixed-size recurrent state per layer instead of a KV cache (docs/15).
 *
 * @param dState   {@code state_size}: SSM states per channel
 * @param dConv    {@code conv_kernel}: taps of the depthwise causal conv
 * @param dtRank   {@code time_step_rank} ("auto" = ceil(dim / 16))
 * @param dInner   {@code intermediate_size} ({@code expand * hidden_size}): channels of the mixer
 * @param convBias {@code use_conv_bias}
 */
public record MambaArch(Core core, int dState, int dConv, int dtRank, int dInner, boolean convBias) implements Architecture {

    @Override
    public String toString() {
        return core.description();
    }
}
