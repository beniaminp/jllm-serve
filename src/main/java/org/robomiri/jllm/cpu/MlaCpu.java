package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.DeepSeekConfig;
import org.robomiri.jllm.model.InterleavedRope;
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
 * (DeepSeek-V2-Lite: 5120 floats = 20 KB per token and layer, 553 KB per token over 27 layers). The norms run
 * parallel over tokens, the rotation and cache scatter parallel over (token, head), the attention per
 * (token, head) with the heaviest tokens handed out first.
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
            b.rmsnorm(qa, qa, m.qALayernorm, n, ql);
            b.matmul(m.qB, qa, b.q, 0, n, false);
        }
        b.matmul(m.kvA, b.xb, lat, 0, n, false);
        b.matmul(m.kPe, b.xb, kpe, 0, n, false);
        b.rmsnorm(lat, lat, m.kvALayernorm, n, kvl);
        b.matmul(m.kvB, lat, kvb, 0, n, false);

        int half = rd / 2;
        int kvbStride = heads * (nope + vd);
        float[] kc = key[l];
        float[] vc = val[l];
        float[] cos = rope.cos;
        float[] sin = rope.sin;
        float[] q = b.q;
        // per (token, head): rotate q_pe in place; assemble the cached key [k_nope, k_pe] (the shared k_pe is
        // copied unrotated and rotated in its slot, so no head waits for another) and copy v
        b.pool.forRange(n * heads, 4, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int t = i / heads;
                int h = i - t * heads;
                int ro = (pos + t) * half;
                InterleavedRope.rotate(q, t * kStride + h * qk + nope, rd, cos, sin, ro);
                int src = t * kvbStride + h * (nope + vd);
                int kd = (pos + t) * kStride + h * qk;
                System.arraycopy(kvb, src, kc, kd, nope);
                System.arraycopy(kpe, t * rd, kc, kd + nope, rd);
                InterleavedRope.rotate(kc, kd + nope, rd, cos, sin, ro);
                System.arraycopy(kvb, src + nope, vc, (pos + t) * vStride + h * vd, vd);
            }
        });
        attend(l, n, pos);
    }

    private void attend(int l, int n, int pos) {
        float[] kc = key[l];
        float[] vc = val[l];
        float[] q = b.q;
        float[] att = b.att;
        int vd = d.vHeadDim();
        float scale = (float) d.softmaxScale();
        // one (token, head) per task, last tokens (most keys) first: a fine grain keeps the tail short
        b.pool.forRangeFine(n * heads, 1, (i0, i1) -> {
            float[] sc = b.scratch.get();
            for (int i = i0; i < i1; i++) {
                int t = n - 1 - i / heads;
                int h = i % heads;
                int qo = t * kStride + h * qk;
                int ko = h * qk;
                int ao = t * vStride + h * vd;
                int len = pos + t + 1;
                for (int s = 0; s < len; s++) {
                    sc[s] = VectorKernels.dot(q, qo, kc, s * kStride + ko, qk) * scale;
                }
                float inv = 1f / KvCache.softmax(sc, 0, len, 0f);
                Arrays.fill(att, ao, ao + vd, 0f);
                for (int s = 0; s < len; s++) {
                    VectorKernels.axpy(sc[s] * inv, vc, s * vStride + h * vd, att, ao, vd);
                }
            }
        });
    }
}
