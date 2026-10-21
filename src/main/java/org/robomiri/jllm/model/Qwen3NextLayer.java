package org.robomiri.jllm.model;

/**
 * The tensors of one Qwen3-Next layer under the names {@link ModelConverter} writes ({@code model.layers.N.}): the two
 * norms (with the zero-centred {@code 1 + w} folded in), the mixer and the MLP.
 *
 * @param attnNorm   {@code input_layernorm [dim]}
 * @param ffnNorm    {@code post_attention_layernorm [dim]}, the norm before the MLP
 * @param mixer      a Gated DeltaNet or a gated full-attention block
 * @param mlp        a dense SwiGLU or the routed experts plus the shared expert ({@link MlpWeights.Moe#shared})
 * @param sharedGate {@code mlp.shared_expert_gate [dim]}: the shared expert's output is scaled by
 *                   {@code sigmoid(sharedGate . x)}; null for a dense layer
 */
public record Qwen3NextLayer(float[] attnNorm, float[] ffnNorm, Mixer mixer, MlpWeights mlp, float[] sharedGate)
        implements LayerWeights {

    /** The token mixer of a layer. */
    public sealed interface Mixer permits DeltaNet, GatedAttention {
        int maxCols();
    }

    /**
     * Gated DeltaNet weights.
     *
     * @param inProj  {@code [projDim, dim]}: the checkpoint's {@code in_proj_qkvz} and {@code in_proj_ba}, whose rows are
     *                grouped per key head, reordered into [q (all heads) | k | v | z | b | a], so that the first
     *                {@code convDim} outputs are the conv channels in order
     * @param conv    depthwise conv weights {@code [convDim, convKernel]}, channel-major (no bias)
     * @param a       {@code A = -exp(A_log)} per value head
     * @param dtBias  {@code dt_bias} per value head
     * @param norm    the gated RMSNorm's gain {@code [valueDim]}, applied as is (not zero-centred)
     * @param outProj {@code [dim, valueWidth]}
     * @param z       null for the fused layout above. A GGUF file stores the projections separately, and keeps the tiny
     *                b / a one in floats: then {@code inProj} is {@code [convDim, dim]} (q, k, v only), {@code z} is
     *                {@code [valueWidth, dim]} (the gate of the gated norm) and {@code ba} is {@code [2 * valueHeads,
     *                dim]}, rows [b (all heads) | a], each product written to its own buffer
     * @param ba      see {@code z}; null exactly when {@code z} is
     */
    public record DeltaNet(QTensor inProj, float[] conv, float[] a, float[] dtBias, float[] norm, QTensor outProj,
                           QTensor z, QTensor ba) implements Mixer {

        public DeltaNet(QTensor inProj, float[] conv, float[] a, float[] dtBias, float[] norm, QTensor outProj) {
            this(inProj, conv, a, dtBias, norm, outProj, null, null);
        }

        public DeltaNet {
            if ((z == null) != (ba == null)) {
                throw new IllegalArgumentException("z and ba are both set or both null");
            }
        }

        /** True if the projections are separate tensors ({@link #z}, {@link #ba}) rather than one fused matrix. */
        public boolean split() {
            return z != null;
        }

        @Override
        public int maxCols() {
            return Math.max(inProj.cols, outProj.cols);
        }
    }

    /**
     * Gated full attention: q/k/v with the per-head norms, the output gate (the second half of every head of the
     * checkpoint's {@code q_proj}, split off by the converter) and {@code o_proj}.
     */
    public record GatedAttention(AttnWeights.Gqa gqa, QTensor gate, QTensor wo) implements Mixer {
        @Override
        public int maxCols() {
            return Math.max(Math.max(gqa.maxCols(), gate.cols), wo.cols);
        }
    }

    static Qwen3NextLayer load(TensorSource f, String p, Qwen3NextArch arch, int l) {
        Mixer mixer;
        if (arch.isFull(l)) {
            String a = p + "self_attn.";
            mixer = new GatedAttention(AttnWeights.Gqa.load(f, a, arch.attention()), f.tensor(a + "q_gate"),
                    f.tensor(a + "o_proj.weight"));
        } else {
            String q = p + "linear_attn.";
            mixer = new DeltaNet(f.tensor(q + "in_proj"), f.floats(q + "conv1d.weight"), f.floats(q + "A"),
                    f.floats(q + "dt_bias"), f.floats(q + "norm.weight"), f.tensor(q + "out_proj.weight"),
                    f.has(q + "in_proj_z") ? f.tensor(q + "in_proj_z") : null,
                    f.has(q + "in_proj_z") ? f.tensor(q + "in_proj_ba") : null);
        }
        Mlp spec = arch.mlps().get(l);
        Qwen3NextLayer L = new Qwen3NextLayer(f.floats(p + "input_layernorm.weight"),
                f.floats(p + "post_attention_layernorm.weight"), mixer, MlpWeights.load(f, p, spec),
                spec instanceof Mlp.Moe ? f.floats(p + "mlp.shared_gate") : null);
        if (!L.matches(arch, l)) {
            throw new IllegalArgumentException("layer " + l + " weight shapes do not match config " + arch
                    + ModelWeights.STALE_HINT);
        }
        return L;
    }

    /** Every shape a backend relies on. */
    boolean matches(Qwen3NextArch arch, int l) {
        int dim = arch.core().dim();
        boolean ok = attnNorm.length == dim && ffnNorm.length == dim && mlp.matches(arch.mlps().get(l), dim)
                && (arch.mlps().get(l) instanceof Mlp.Moe ? sharedGate != null && sharedGate.length == dim : sharedGate == null);
        Qwen3NextArch.DeltaNet d = arch.deltaNet();
        return ok && switch (mixer) {
            case GatedAttention g -> arch.isFull(l) && g.gqa().matches(arch.attention(), dim)
                    && g.gate().rows == arch.attention().qDim() && g.gate().cols == dim
                    && g.wo().rows == dim && g.wo().cols == arch.attention().qDim();
            case DeltaNet n -> !arch.isFull(l) && n.inProj().rows == (n.split() ? d.convDim() : d.projDim())
                    && n.inProj().cols == dim
                    && (!n.split() || n.z().rows == d.valueWidth() && n.z().cols == dim
                    && n.ba().rows == 2 * d.valueHeads() && n.ba().cols == dim)
                    && n.conv().length == d.convDim() * d.convKernel() && n.a().length == d.valueHeads()
                    && n.dtBias().length == d.valueHeads() && n.norm().length == d.valueDim()
                    && n.outProj().rows == dim && n.outProj().cols == d.valueWidth();
        };
    }

    public DeltaNet deltaNet() {
        return (DeltaNet) mixer;
    }

    public GatedAttention attention() {
        return (GatedAttention) mixer;
    }

    @Override
    public int maxCols() {
        return Math.max(mixer.maxCols(), mlp.maxCols());
    }
}
