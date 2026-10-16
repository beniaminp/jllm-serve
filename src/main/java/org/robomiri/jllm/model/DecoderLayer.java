package org.robomiri.jllm.model;

/**
 * The tensors of one decoder layer: the norms, the attention projections, {@code o_proj} and the feed-forward
 * block, each loaded for exactly what the layer's {@link LayerSpec} asks and checked against it.
 *
 * @param attnNorm     {@code input_layernorm [dim]}
 * @param ffnNorm      the norm before the MLP {@code [dim]} ({@link LayerSpec.Norms#ffnNorm})
 * @param postAttnNorm the norm on the attention output, null unless the layer's norms are a sandwich
 * @param postMlpNorm  the norm on the MLP output, null unless the layer's norms are a sandwich
 * @param wo           {@code o_proj [dim, attn.outDim()]}
 */
public record DecoderLayer(float[] attnNorm, float[] ffnNorm, float[] postAttnNorm, float[] postMlpNorm,
                           AttnWeights attn, QTensor wo, MlpWeights mlp) implements LayerWeights {

    static DecoderLayer load(TensorSource f, String p, LayerSpec s) {
        LayerSpec.Norms n = s.norms();
        return new DecoderLayer(
                f.floats(p + "input_layernorm.weight"),
                f.floats(p + n.ffnNorm() + ".weight"),
                n.post() ? f.floats(p + n.postAttnNorm() + ".weight") : null,
                n.post() ? f.floats(p + n.postMlpNorm() + ".weight") : null,
                AttnWeights.load(f, p + "self_attn.", s.attn()),
                f.tensor(p + "self_attn.o_proj.weight"),
                MlpWeights.load(f, p, s.mlp()));
    }

    /** Every shape a backend relies on: a mismatch here would otherwise surface as a silent out-of-bounds read. */
    boolean matches(LayerSpec s, int dim) {
        return attnNorm.length == dim && ffnNorm.length == dim
                && s.norms().post() == (postAttnNorm != null) && s.norms().post() == (postMlpNorm != null)
                && (postAttnNorm == null || (postAttnNorm.length == dim && postMlpNorm.length == dim))
                && attn.matches(s.attn(), dim)
                && wo.rows == dim && wo.cols == s.attn().outDim()
                && mlp.matches(s.mlp(), dim);
    }

    /** The grouped-query projections; throws if this is a latent-attention layer. */
    public AttnWeights.Gqa gqa() {
        if (attn instanceof AttnWeights.Gqa g) {
            return g;
        }
        throw new IllegalStateException("not a grouped-query attention layer: " + attn.getClass().getSimpleName());
    }

    /** The latent-attention projections; throws if this is a grouped-query layer. */
    public AttnWeights.Mla mla() {
        if (attn instanceof AttnWeights.Mla m) {
            return m;
        }
        throw new IllegalStateException("not a latent attention layer: " + attn.getClass().getSimpleName());
    }

    /** The dense MLP; throws if this is a MoE layer. */
    public MlpWeights.Dense dense() {
        if (mlp instanceof MlpWeights.Dense d) {
            return d;
        }
        throw new IllegalStateException("not a dense MLP layer");
    }

    /** The experts; throws if this is a dense layer. */
    public MlpWeights.Moe moe() {
        if (mlp instanceof MlpWeights.Moe m) {
            return m;
        }
        throw new IllegalStateException("not a MoE layer");
    }

    @Override
    public int maxCols() {
        return Math.max(Math.max(attn.maxCols(), wo.cols), mlp.maxCols());
    }
}
