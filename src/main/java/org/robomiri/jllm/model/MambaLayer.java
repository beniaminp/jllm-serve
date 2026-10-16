package org.robomiri.jllm.model;

/**
 * Weights of one Mamba block ({@code backbone.layers.N}). Projections that read {@code d_model} or {@code d_inner}
 * features are quantized; everything with an awkward width (dt_proj has {@code dt_rank} columns), the small
 * per-channel tensors and the norm stay f32.
 */
public final class MambaLayer {
    /** Pre-norm RMSNorm weight {@code [d_model]}. */
    public float[] norm;
    /** {@code [2*d_inner, d_model]}: rows {@code [0, d_inner)} are the conv/SSM branch x, the rest the gate z. */
    public QTensor inProj;
    /** Depthwise causal conv weights {@code [d_inner, d_conv]}, channel-major; tap {@code k} sees input t-(d_conv-1)+k. */
    public float[] conv;
    /** Conv bias {@code [d_inner]} (zeros when the checkpoint has none). */
    public float[] convBias;
    /** {@code [dt_rank + 2*d_state, d_inner]}: rows are dt, B, C. */
    public QTensor xProj;
    /** {@code [d_inner, dt_rank]} f32, row-major. */
    public float[] dtProj;
    /** {@code [d_inner]} */
    public float[] dtBias;
    /** {@code A = -exp(A_log)}, {@code [d_inner, d_state]} f32. */
    public float[] a;
    /** Skip connection {@code [d_inner]} */
    public float[] d;
    /** {@code [d_model, d_inner]} */
    public QTensor outProj;

    static MambaLayer load(QuantFile f, String p, ModelConfig c) {
        ModelConfig.MambaParams m = c.mamba();
        MambaLayer L = new MambaLayer();
        L.norm = f.floats(p + "norm.weight");
        String q = p + "mixer.";
        L.inProj = f.tensor(q + "in_proj.weight");
        L.conv = f.floats(q + "conv1d.weight");
        L.convBias = f.has(q + "conv1d.bias") ? f.floats(q + "conv1d.bias") : new float[m.dInner()];
        L.xProj = f.tensor(q + "x_proj.weight");
        L.dtProj = f.floats(q + "dt_proj.weight");
        L.dtBias = f.floats(q + "dt_proj.bias");
        L.a = f.floats(q + "A");
        L.d = f.floats(q + "D");
        L.outProj = f.tensor(q + "out_proj.weight");
        boolean ok = L.norm.length == c.dim()
                && L.inProj.rows == 2 * m.dInner() && L.inProj.cols == c.dim()
                && L.conv.length == m.dInner() * m.dConv() && L.convBias.length == m.dInner()
                && L.xProj.rows == m.dtRank() + 2 * m.dState() && L.xProj.cols == m.dInner()
                && L.dtProj.length == m.dInner() * m.dtRank() && L.dtBias.length == m.dInner()
                && L.a.length == m.dInner() * m.dState() && L.d.length == m.dInner()
                && L.outProj.rows == c.dim() && L.outProj.cols == m.dInner();
        if (!ok) {
            throw new IllegalArgumentException(p + " weight shapes do not match config " + c + ModelWeights.STALE_HINT);
        }
        return L;
    }
}
