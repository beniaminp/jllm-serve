package org.robomiri.jllm.model;

/**
 * Weights of one Mamba block ({@code backbone.layers.N}). Projections that read {@code d_model} or {@code d_inner}
 * features are quantized; everything with an awkward width (dt_proj has {@code dt_rank} columns), the small
 * per-channel tensors and the norm stay f32.
 *
 * @param norm     pre-norm RMSNorm weight {@code [d_model]}
 * @param inProj   {@code [2*d_inner, d_model]}: rows {@code [0, d_inner)} are the conv/SSM branch x, the rest the gate z
 * @param conv     depthwise causal conv weights {@code [d_inner, d_conv]}, channel-major; tap {@code k} sees input t-(d_conv-1)+k
 * @param convBias conv bias {@code [d_inner]} (zeros when the checkpoint has none)
 * @param xProj    {@code [dt_rank + 2*d_state, d_inner]}: rows are dt, B, C
 * @param dtProj   {@code [d_inner, dt_rank]} f32, row-major
 * @param dtBias   {@code [d_inner]}
 * @param a        {@code A = -exp(A_log)}, {@code [d_inner, d_state]} f32
 * @param d        skip connection {@code [d_inner]}
 * @param outProj  {@code [d_model, d_inner]}
 */
public record MambaLayer(float[] norm, QTensor inProj, float[] conv, float[] convBias, QTensor xProj, float[] dtProj,
                         float[] dtBias, float[] a, float[] d, QTensor outProj) implements LayerWeights {

    static MambaLayer load(TensorSource f, String p, MambaArch m) {
        String q = p + "mixer.";
        MambaLayer L = new MambaLayer(f.floats(p + "norm.weight"), f.tensor(q + "in_proj.weight"),
                f.floats(q + "conv1d.weight"), f.has(q + "conv1d.bias") ? f.floats(q + "conv1d.bias") : new float[m.dInner()],
                f.tensor(q + "x_proj.weight"), f.floats(q + "dt_proj.weight"), f.floats(q + "dt_proj.bias"),
                f.floats(q + "A"), f.floats(q + "D"), f.tensor(q + "out_proj.weight"));
        if (!L.matches(m)) {
            throw new IllegalArgumentException(p + " weight shapes do not match config " + m + ModelWeights.STALE_HINT);
        }
        return L;
    }

    boolean matches(MambaArch m) {
        int dim = m.core().dim();
        return norm.length == dim
                && inProj.rows == 2 * m.dInner() && inProj.cols == dim
                && conv.length == m.dInner() * m.dConv() && convBias.length == m.dInner()
                && xProj.rows == m.dtRank() + 2 * m.dState() && xProj.cols == m.dInner()
                && dtProj.length == m.dInner() * m.dtRank() && dtBias.length == m.dInner()
                && a.length == m.dInner() * m.dState() && d.length == m.dInner()
                && outProj.rows == dim && outProj.cols == m.dInner();
    }

    @Override
    public int maxCols() {
        return Math.max(Math.max(inProj.cols, xProj.cols), outProj.cols);
    }
}
