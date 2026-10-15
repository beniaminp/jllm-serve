package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.DeepSeekConfig;
import org.robomiri.jllm.model.MlaWeights;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.Rope;

import java.util.Arrays;

/**
 * DeepSeek Multi-head Latent Attention on the CPU, called by {@link CpuBackend} in place of the q/k/v projections
 * and the attention of a standard layer (the {@code o_proj} stays with the caller).
 *
 * <p>Per token: the query is {@code q_proj(x)} or {@code q_b(norm(q_a(x)))}; the key/value latent is
 * {@code norm(kv_a(x))} expanded by {@code kv_b} into per-head {@code k_nope} and {@code v}; one rotary key
 * {@code k_pe = kPe(x)} is shared by all heads. The rotary parts of q and k_pe are rotated (pairs
 * {@code (2i, 2i+1)}, see {@link Rope#deepseek}); key = {@code [k_nope, k_pe]}, query = {@code [q_nope, q_pe]}.
 *
 * <p>The cache holds the expanded per-head keys ({@code heads * (nope+rope)} floats) and values
 * ({@code heads * v}) in f32 - simple, but much larger than the compressed latent MLA is designed for
 * (DeepSeek-V2-Lite: 5120 floats = 20 KB per token and layer, 553 KB per token over 27 layers).
 */
final class MlaCpu {

    private final CpuBackend b;
    private final DeepSeekConfig d;
    private final Rope rope;
    private final int heads;
    private final int qk;
    private final int kStride;
    private final int vStride;

    private final float[][] key;
    private final float[][] val;
    private final float[] qa;
    private final float[] lat;
    private final float[] kpe;
    private final float[] kvb;

    MlaCpu(CpuBackend b, ModelConfig c, Rope rope, int ctx, int maxBatch) {
        this.b = b;
        this.d = c.deepseek();
        this.rope = rope;
        heads = c.heads();
        qk = d.qkDim();
        kStride = heads * qk;
        vStride = heads * d.vHeadDim();
        if ((long) ctx * kStride > Integer.MAX_VALUE - 16) {
            throw new IllegalArgumentException("context " + ctx + " is too large for the CPU MLA cache");
        }
        key = new float[c.layers()][ctx * kStride];
        val = new float[c.layers()][ctx * vStride];
        qa = new float[maxBatch * Math.max(d.qLoraRank(), 1)];
        lat = new float[maxBatch * d.kvLoraRank()];
        kpe = new float[maxBatch * d.qkRopeDim()];
        kvb = new float[maxBatch * heads * (d.qkNopeDim() + d.vHeadDim())];
    }

    /** Reads {@code b.xb} (normalized hidden states) and writes the attention output {@code [n, heads*v]} to {@code b.att}. */
    void run(int l, ModelWeights.Layer L, int n, int pos) {
        MlaWeights m = L.mla;
        int nope = d.qkNopeDim();
        int rd = d.qkRopeDim();
        int vd = d.vHeadDim();
        int kvl = d.kvLoraRank();
        if (m.qProj != null) {
            b.matmul(m.qProj, b.xb, b.q, 0, n, false);
        } else {
            int ql = d.qLoraRank();
            b.matmul(m.qA, b.xb, qa, 0, n, false);
            for (int t = 0; t < n; t++) {
                b.rmsnormRow(qa, t * ql, qa, t * ql, m.qALayernorm, ql);
            }
            b.matmul(m.qB, qa, b.q, 0, n, false);
        }
        b.matmul(m.kvA, b.xb, lat, 0, n, false);
        b.matmul(m.kPe, b.xb, kpe, 0, n, false);
        for (int t = 0; t < n; t++) {
            b.rmsnormRow(lat, t * kvl, lat, t * kvl, m.kvALayernorm, kvl);
        }
        b.matmul(m.kvB, lat, kvb, 0, n, false);

        int half = rd / 2;
        int kvbStride = heads * (nope + vd);
        float[] kc = key[l];
        float[] vc = val[l];
        for (int t = 0; t < n; t++) {
            int ro = (pos + t) * half;
            for (int h = 0; h < heads; h++) {
                rotate(b.q, t * kStride + h * qk + nope, ro, half);
            }
            rotate(kpe, t * rd, ro, half);
            int kd = (pos + t) * kStride;
            int vdst = (pos + t) * vStride;
            for (int h = 0; h < heads; h++) {
                int src = t * kvbStride + h * (nope + vd);
                System.arraycopy(kvb, src, kc, kd + h * qk, nope);
                System.arraycopy(kpe, t * rd, kc, kd + h * qk + nope, rd);
                System.arraycopy(kvb, src + nope, vc, vdst + h * vd, vd);
            }
        }
        attend(l, n, pos);
    }

    /** Rotates the pairs (2i, 2i+1) of {@code v[o .. o+2*half)} by the table row at {@code ro}. */
    private void rotate(float[] v, int o, int ro, int half) {
        float[] cos = rope.cos;
        float[] sin = rope.sin;
        for (int i = 0; i < half; i++) {
            float a = v[o + 2 * i];
            float bb = v[o + 2 * i + 1];
            float cs = cos[ro + i];
            float sn = sin[ro + i];
            v[o + 2 * i] = a * cs - bb * sn;
            v[o + 2 * i + 1] = bb * cs + a * sn;
        }
    }

    private void attend(int l, int n, int pos) {
        float[] kc = key[l];
        float[] vc = val[l];
        float[] q = b.q;
        float[] att = b.att;
        int vd = d.vHeadDim();
        float scale = (float) d.softmaxScale();
        b.pool.forRange(n * heads, 1, (i0, i1) -> {
            float[] sc = b.scratch.get();
            for (int i = i0; i < i1; i++) {
                int t = i / heads;
                int h = i % heads;
                int qo = t * kStride + h * qk;
                int ko = h * qk;
                int ao = t * vStride + h * vd;
                int len = pos + t + 1;
                float max = Float.NEGATIVE_INFINITY;
                for (int s = 0; s < len; s++) {
                    float dot = VectorKernels.dot(q, qo, kc, s * kStride + ko, qk) * scale;
                    sc[s] = dot;
                    max = Math.max(max, dot);
                }
                float sum = 0;
                for (int s = 0; s < len; s++) {
                    float e = (float) Math.exp(sc[s] - max);
                    sc[s] = e;
                    sum += e;
                }
                float inv = 1f / sum;
                Arrays.fill(att, ao, ao + vd, 0f);
                for (int s = 0; s < len; s++) {
                    VectorKernels.axpy(sc[s] * inv, vc, s * vStride + h * vd, att, ao, vd);
                }
            }
        });
    }
}
