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
    /**
     * {@code kv_b_proj [heads*(nope+v), kvLora]}: per head, {@code nope} key rows then {@code v} value rows. The
     * backends never multiply by it directly: its two halves are absorbed into the query and applied after the
     * softmax ({@link #kvBKeys}, {@link #kvBValues}), which is what lets the cache hold the latent.
     */
    public QTensor kvB;
    /** Shared experts (dense SwiGLU of width {@code nShared*moeDim}); gate rows then up rows. Null when none. */
    public QTensor sharedGateUp;
    public QTensor sharedDown;
    /** V3 {@code e_score_correction_bias [nRouted]}, null otherwise. */
    public float[] routerBias;

    /**
     * The key half of {@link #kvB} dequantized to f32, {@code W_kb_k[h] [nope, kvLora]} for every head: laid out
     * {@code [heads][nope][kvLora]} as stored, or {@code [heads][kvLora][nope]} when {@code transposed}. Absorbed
     * into the query: {@code q_abs[h] = W_kb_k[h]^T q_nope[h]} is {@code kvLora} wide and is dotted with the cached
     * latent directly. Small ({@code heads * nope * kvLora} floats per layer), computed once per backend at load.
     */
    public float[] kvBKeys(DeepSeekConfig d, int heads, boolean transposed) {
        return half(d, heads, 0, d.qkNopeDim(), transposed);
    }

    /**
     * The value half of {@link #kvB}, {@code W_kb_v[h] [v, kvLora]} per head: {@code [heads][v][kvLora]} as stored
     * or {@code [heads][kvLora][v]} when {@code transposed}. Applied after the softmax to the attention-weighted
     * latent: {@code out[h] = W_kb_v[h] (sum_s w_s c_kv[s])}.
     */
    public float[] kvBValues(DeepSeekConfig d, int heads, boolean transposed) {
        return half(d, heads, d.qkNopeDim(), d.vHeadDim(), transposed);
    }

    /** Rows {@code [first, first + count)} of every head's {@code nope + v} block of kvB, optionally transposed per head. */
    private float[] half(DeepSeekConfig d, int heads, int first, int count, boolean transposed) {
        int kvl = d.kvLoraRank();
        int block = d.qkNopeDim() + d.vHeadDim();
        float[] out = new float[heads * count * kvl];
        float[] row = new float[kvl];
        for (int h = 0; h < heads; h++) {
            int base = h * count * kvl;
            for (int r = 0; r < count; r++) {
                kvB.dequantizeRow(h * block + first + r, row, 0);
                if (transposed) {
                    for (int j = 0; j < kvl; j++) {
                        out[base + j * count + r] = row[j];
                    }
                } else {
                    System.arraycopy(row, 0, out, base + r * kvl, kvl);
                }
            }
        }
        return out;
    }
}
