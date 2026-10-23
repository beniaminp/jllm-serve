package org.robomiri.jllm.model;

/**
 * The tensors of one {@link HybridArch} layer under the names {@link ModelConverter} writes ({@code model.layers.N.}).
 *
 * @param norm1 {@code input_layernorm [dim]}, before the mixer (before the feed-forward block of a layer without mixer)
 * @param norm2 {@code post_attention_layernorm [dim]}, before the feed-forward block of a layer that has a mixer, else null
 * @param mixer the attention or state-space weights; null for a feed-forward-only layer
 * @param mlp   the feed-forward block, or null
 */
public record HybridLayer(float[] norm1, float[] norm2, Mixer mixer, MlpWeights mlp) implements LayerWeights {

    /** The token mixer of a layer. */
    public sealed interface Mixer permits Attn, M2, M1 {
        int maxCols();
    }

    /** Grouped-query attention: the projections ({@link AttnWeights.Gqa}) and {@code o_proj [dim, qDim]}. */
    public record Attn(AttnWeights.Gqa gqa, QTensor wo) implements Mixer {
        @Override
        public int maxCols() {
            return Math.max(gqa.maxCols(), wo.cols);
        }
    }

    /**
     * Mamba-2 weights.
     *
     * @param inProj  {@code [projDim, dim]}, rows {@code [z | x | B | C | dt]}
     * @param conv    depthwise conv weights {@code [convDim, convKernel]}, channel-major
     * @param convBias {@code [convDim]} (zeros when the checkpoint has none)
     * @param dtBias  {@code dt_bias [heads]}
     * @param a       {@code A = -exp(A_log) [heads]}
     * @param d       skip connection {@code [heads]}
     * @param norm    the gated RMSNorm's gain {@code [dInner]}
     * @param outProj {@code [dim, dInner]}
     */
    public record M2(QTensor inProj, float[] conv, float[] convBias, float[] dtBias, float[] a, float[] d, float[] norm,
                     QTensor outProj) implements Mixer {
        @Override
        public int maxCols() {
            return Math.max(inProj.cols, outProj.cols);
        }
    }

    /**
     * Mamba-1 weights (as {@link MambaLayer}) with Jamba's norms of dt, B and C.
     *
     * @param dtNorm {@code [dtRank]}
     * @param bNorm  {@code [state]}
     * @param cNorm  {@code [state]}
     */
    public record M1(QTensor inProj, float[] conv, float[] convBias, QTensor xProj, float[] dtProj, float[] dtBias, float[] a,
                     float[] d, QTensor outProj, float[] dtNorm, float[] bNorm, float[] cNorm) implements Mixer {
        @Override
        public int maxCols() {
            return Math.max(Math.max(inProj.cols, xProj.cols), outProj.cols);
        }
    }

    static HybridLayer load(TensorSource f, String p, HybridArch arch, int l) {
        HybridArch.Layer spec = arch.layers().get(l);
        Mixer mixer = switch (spec.mixer()) {
            case ATTENTION -> new Attn(AttnWeights.Gqa.load(f, p + "self_attn.", arch.attention()),
                    f.tensor(p + "self_attn.o_proj.weight"));
            case MAMBA2 -> {
                String q = p + "mixer.";
                yield new M2(f.tensor(q + "in_proj.weight"), f.floats(q + "conv1d.weight"), f.floats(q + "conv1d.bias"),
                        f.floats(q + "dt_bias"), f.floats(q + "A"), f.floats(q + "D"), f.floats(q + "norm.weight"),
                        f.tensor(q + "out_proj.weight"));
            }
            case MAMBA1 -> {
                String q = p + "mixer.";
                yield new M1(f.tensor(q + "in_proj.weight"), f.floats(q + "conv1d.weight"), f.floats(q + "conv1d.bias"),
                        f.tensor(q + "x_proj.weight"), f.floats(q + "dt_proj.weight"), f.floats(q + "dt_proj.bias"),
                        f.floats(q + "A"), f.floats(q + "D"), f.tensor(q + "out_proj.weight"), f.floats(q + "dt_norm.weight"),
                        f.floats(q + "b_norm.weight"), f.floats(q + "c_norm.weight"));
            }
            case NONE -> null;
        };
        HybridLayer L = new HybridLayer(f.floats(p + "input_layernorm.weight"),
                mixer != null && spec.ffn() != null ? f.floats(p + "post_attention_layernorm.weight") : null, mixer,
                spec.ffn() == null ? null : MlpWeights.load(f, p, spec.ffn()));
        if (!L.matches(arch, l)) {
            throw new IllegalArgumentException("layer " + l + " weight shapes do not match config " + arch
                    + ModelWeights.STALE_HINT);
        }
        return L;
    }

    /** Every shape a backend relies on. */
    boolean matches(HybridArch arch, int l) {
        HybridArch.Layer spec = arch.layers().get(l);
        int dim = arch.core().dim();
        boolean ok = norm1.length == dim && (mixer != null && spec.ffn() != null ? norm2 != null && norm2.length == dim : norm2 == null)
                && (spec.ffn() == null ? mlp == null : mlp != null && mlp.matches(spec.ffn(), dim));
        return ok && switch (mixer) {
            case null -> spec.mixer() == HybridArch.Mixer.NONE;
            case Attn a -> spec.mixer() == HybridArch.Mixer.ATTENTION && a.gqa().matches(arch.attention(), dim)
                    && a.wo().rows == dim && a.wo().cols == arch.attention().qDim();
            case M2 m -> {
                HybridArch.Mamba2 s = arch.mamba2();
                yield spec.mixer() == HybridArch.Mixer.MAMBA2 && m.inProj().rows == s.projDim() && m.inProj().cols == dim
                        && m.conv().length == s.convDim() * s.convKernel() && m.convBias().length == s.convDim()
                        && m.dtBias().length == s.heads() && m.a().length == s.heads() && m.d().length == s.heads()
                        && m.norm().length == s.dInner() && m.outProj().rows == dim && m.outProj().cols == s.dInner();
            }
            case M1 m -> {
                HybridArch.Mamba1 s = arch.mamba1();
                yield spec.mixer() == HybridArch.Mixer.MAMBA1 && m.inProj().rows == 2 * s.dInner() && m.inProj().cols == dim
                        && m.conv().length == s.dInner() * s.convKernel() && m.convBias().length == s.dInner()
                        && m.xProj().rows == s.dtRank() + 2 * s.state() && m.xProj().cols == s.dInner()
                        && m.dtProj().length == s.dInner() * s.dtRank() && m.dtBias().length == s.dInner()
                        && m.a().length == s.dInner() * s.state() && m.d().length == s.dInner()
                        && m.outProj().rows == dim && m.outProj().cols == s.dInner() && m.dtNorm().length == s.dtRank()
                        && m.bNorm().length == s.state() && m.cNorm().length == s.state();
            }
        };
    }

    public Attn attn() {
        return (Attn) mixer;
    }

    public M2 m2() {
        return (M2) mixer;
    }

    public M1 m1() {
        return (M1) mixer;
    }

    @Override
    public int maxCols() {
        return Math.max(mixer == null ? 0 : mixer.maxCols(), mlp == null ? 0 : mlp.maxCols());
    }
}
