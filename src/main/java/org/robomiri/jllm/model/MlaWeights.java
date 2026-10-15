package org.robomiri.jllm.model;

/**
 * Per-layer DeepSeek tensors that replace the q/k/v projections of a standard layer: the Multi-head Latent
 * Attention projections. {@code kv_a_proj_with_mqa [kvLora + rope, dim]} is split by the converter into
 * {@link #kvA} (the first {@code kvLora} rows) and {@link #kPe} (the last {@code rope} rows), so that both
 * outputs are contiguous per token and the latent can be normalized in place.
 */
public final class MlaWeights {
    /** {@code q_proj [heads*qk, dim]} when {@code q_lora_rank} is null; otherwise null. */
    public QTensor qProj;
    /** {@code q_a_proj [qLora, dim]}, {@code q_a_layernorm [qLora]}, {@code q_b_proj [heads*qk, qLora]}. */
    public QTensor qA;
    public float[] qALayernorm;
    public QTensor qB;
    /** Latent rows of {@code kv_a_proj_with_mqa}, {@code [kvLora, dim]}. */
    public QTensor kvA;
    /** Rotary key rows of {@code kv_a_proj_with_mqa}, {@code [rope, dim]}; shared by all heads. */
    public QTensor kPe;
    public float[] kvALayernorm;
    /** {@code kv_b_proj [heads*(nope+v), kvLora]}: per head, {@code nope} key rows then {@code v} value rows. */
    public QTensor kvB;
    /** Shared experts (dense SwiGLU of width {@code nShared*moeDim}); gate rows then up rows. Null when none. */
    public QTensor sharedGateUp;
    public QTensor sharedDown;
    /** V3 {@code e_score_correction_bias [nRouted]}, null otherwise. */
    public float[] routerBias;
}
