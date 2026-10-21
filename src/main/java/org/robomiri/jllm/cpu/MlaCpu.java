package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.AttnWeights;
import org.robomiri.jllm.model.InterleavedRope;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.Rope;

import java.util.Arrays;

/**
 * DeepSeek Multi-head Latent Attention on the CPU, called by {@link CpuBackend} in place of the q/k/v projections
 * and the attention of a standard layer (the {@code o_proj} stays with the caller).
 *
 * <p>Per token: the query is {@code q_proj(x)} or {@code q_b(norm(q_a(x)))}, per head {@code [q_nope | q_pe]}; the
 * key/value latent is {@code c = norm(kv_a(x))} ({@code kvLora} wide); one rotary key {@code k_pe = kPe(x)} is
 * shared by all heads. The rotary parts of q and k_pe are rotated (pairs {@code (2i, 2i+1)}, the interleaved
 * table of the layer). In the paper's form every head expands {@code c} through {@code kv_b} into
 * {@code k_nope_h = W_kb_k[h] c} and {@code v_h = W_kb_v[h] c}; the engine uses the <b>absorbed</b> form instead,
 * which is the same arithmetic in a different order:
 * <pre>
 *   score_h(s) = q_nope_h . (W_kb_k[h] c_s) + q_pe_h . k_pe_s = (W_kb_k[h]^T q_nope_h) . c_s + q_pe_h . k_pe_s
 *   out_h      = sum_s w_s (W_kb_v[h] c_s) = W_kb_v[h] (sum_s w_s c_s)
 * </pre>
 * So the query of head h becomes {@code q_abs_h = [W_kb_k[h]^T q_nope_h | rope(q_pe_h)]} ({@code kvLora + rope}
 * wide), the key of every position is the one row {@code [c_s | rope(k_pe_s)]} shared by all heads, the value is
 * its first {@code kvLora} entries, and one small per-head matmul with {@code W_kb_v[h]} follows the softmax.
 *
 * <p>The cache therefore holds one row of {@code kvLora + rope} numbers per token and layer (f32, or f16 with
 * {@code backend.kv-dtype=f16}: stored as half, expanded to f32 per row before the SIMD kernels read it; q8: int8
 * with one f32 scale for the whole row, as {@link KvCache.Q8} has per kv head row): 576 floats = 2.3 KB for
 * DeepSeek-V2-Lite instead of the 20 KB of the expanded per-head keys and values. Attention
 * is tiled like {@link KvCache}: each task owns a tile of tokens and a range of heads and reads every cached row
 * once for all of them (the row is shared by the heads, so this is multi-query attention with one kv head).
 */
final class MlaCpu {

    private final CpuBackend b;
    private final Attention.Mla m;
    private final AttnWeights.Mla[] layers;
    private final Rope rope;
    private final int heads;
    private final int qk;
    /** Absorbed query / cached row width: kvLora + rope. */
    private final int aw;
    private final int qaStride;
    private final int oStride;

    /** The latent cache, one of the three: rows of {@code aw} per token (q8: and one scale per row). */
    private final float[][] cache32;
    private final short[][] cache16;
    private final byte[][] cache8;
    private final float[][] scale8;
    private final int ctx;
    /** Per layer {@code W_kb_k[h]^T} as [heads][kvLora][nope] and {@code W_kb_v[h]} as [heads][v][kvLora], f32. */
    private final float[][] wkT;
    private final float[][] wv;
    private final float[] qLora;
    private final float[] lat;
    private final float[] kpe;
    private final float[] qAbs;
    /** Tokens per attention tile, as in {@link KvCache}: tile * heads * ctx score floats stay within 32 K. */
    private final int tile;
    /** Per thread: [tile * heads * ctx scores][aw expanded row][tile * heads * kvLora latent sums][tile * heads 1/sum]. */
    private final ThreadLocal<float[]> scratch;
    private final int rowOff;
    private final int latOff;
    private final int invOff;

    MlaCpu(CpuBackend b, ModelWeights w, Attention.Mla m, Rope rope, int ctx, int maxBatch, KvDtype kvDtype) {
        this.b = b;
        this.m = m;
        this.rope = rope;
        heads = m.heads();
        qk = m.qkDim();
        int kvl = m.kvLoraRank();
        aw = m.latentDim();
        qaStride = heads * aw;
        oStride = heads * m.vDim();
        if ((long) ctx * aw > Integer.MAX_VALUE - 16) {
            throw new IllegalArgumentException("context " + ctx + " is too large for the CPU MLA cache");
        }
        int nLayers = w.arch.core().layers();
        cache32 = kvDtype == KvDtype.F32 ? new float[nLayers][ctx * aw] : null;
        cache16 = kvDtype == KvDtype.F16 ? new short[nLayers][ctx * aw] : null;
        cache8 = kvDtype == KvDtype.Q8 ? new byte[nLayers][ctx * aw] : null;
        scale8 = kvDtype == KvDtype.Q8 ? new float[nLayers][ctx] : null;
        this.ctx = ctx;
        layers = new AttnWeights.Mla[nLayers];
        wkT = new float[nLayers][];
        wv = new float[nLayers][];
        for (int l = 0; l < nLayers; l++) {
            layers[l] = w.decoderLayer(l).mla();
            wkT[l] = layers[l].kvBKeys(m, true);
            wv[l] = layers[l].kvBValues(m, false);
        }
        qLora = new float[maxBatch * Math.max(m.qLoraRank(), 1)];
        lat = new float[maxBatch * kvl];
        kpe = new float[maxBatch * m.rope()];
        qAbs = new float[maxBatch * qaStride];
        tile = Math.max(1, Math.min(8, 32768 / (heads * ctx)));
        rowOff = tile * heads * ctx;
        latOff = rowOff + aw;
        invOff = latOff + tile * heads * kvl;
        int len = invOff + tile * heads;
        scratch = ThreadLocal.withInitial(() -> new float[len]);
    }

    /** Reads {@code b.xb} (normalized hidden states) and writes the attention output {@code [n, heads*v]} to {@code b.att}. */
    void run(int l, int n, int pos) {
        AttnWeights.Mla mw = layers[l];
        int nope = m.nope();
        int rd = m.rope();
        int kvl = m.kvLoraRank();
        switch (mw.query()) {
            case AttnWeights.Mla.Proj p -> b.matmul(p.w(), b.xb, b.q, 0, n, false);
            case AttnWeights.Mla.Lora q -> {
                b.matmul(q.a(), b.xb, qLora, 0, n, false);
                b.rmsnorm(qLora, qLora, q.norm(), n, m.qLoraRank());
                b.matmul(q.b(), qLora, b.q, 0, n, false);
            }
        }
        b.matmul(mw.kvA(), b.xb, lat, 0, n, false);
        b.matmul(mw.kPe(), b.xb, kpe, 0, n, false);
        b.rmsnorm(lat, lat, mw.kvALayernorm(), n, kvl);

        int half = rd / 2;
        int qStride = heads * qk;
        float[] cos = rope.cos;
        float[] sin = rope.sin;
        float[] q = b.q;
        float[] wk = wkT[l];
        // per (token, head): rotate q_pe in place and absorb W_kb_k into q_nope; the head-0 task of each token also
        // rotates the shared k_pe and stores the cache row [c | k_pe] (no other task touches that token's row)
        b.pool.forRange(n * heads, 1, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int t = i / heads;
                int h = i - t * heads;
                int ro = (pos + t) * half;
                int qo = t * qStride + h * qk;
                InterleavedRope.rotate(q, qo + nope, rd, cos, sin, ro);
                int ao = t * qaStride + h * aw;
                int wo = h * kvl * nope;
                for (int j = 0; j < kvl; j++) {
                    qAbs[ao + j] = VectorKernels.dot(q, qo, wk, wo + j * nope, nope);
                }
                System.arraycopy(q, qo + nope, qAbs, ao + kvl, rd);
                if (h == 0) {
                    InterleavedRope.rotate(kpe, t * rd, rd, cos, sin, ro);
                    store(l, pos + t, t);
                }
            }
        });
        attend(l, n, pos);
    }

    /** Cache row {@code s} of layer {@code l} = [lat[t] | kpe[t]]. */
    private void store(int l, int s, int t) {
        int kvl = m.kvLoraRank();
        int rd = m.rope();
        if (cache32 != null) {
            float[] kc = cache32[l];
            System.arraycopy(lat, t * kvl, kc, s * aw, kvl);
            System.arraycopy(kpe, t * rd, kc, s * aw + kvl, rd);
        } else if (cache16 != null) {
            short[] kc = cache16[l];
            KvCache.F16.toHalf(lat, t * kvl, kc, s * aw, kvl);
            KvCache.F16.toHalf(kpe, t * rd, kc, s * aw + kvl, rd);
        } else {
            float max = 0f;
            for (int i = 0; i < kvl; i++) {
                max = Math.max(max, Math.abs(lat[t * kvl + i]));
            }
            for (int i = 0; i < rd; i++) {
                max = Math.max(max, Math.abs(kpe[t * rd + i]));
            }
            float sc = KvDtype.q8Scale(max);
            scale8[l][s] = sc;
            byte[] kc = cache8[l];
            for (int i = 0; i < kvl; i++) {
                kc[s * aw + i] = KvDtype.q8(lat[t * kvl + i], sc);
            }
            for (int i = 0; i < rd; i++) {
                kc[s * aw + kvl + i] = KvDtype.q8(kpe[t * rd + i], sc);
            }
        }
    }

    /** The first {@code len} values of cache row {@code s} (f16 or q8) as floats into {@code dst[d ..]}. */
    private void expand(int l, int s, int len, float[] dst, int d) {
        if (cache16 != null) {
            KvCache.F16.toFloat(cache16[l], s * aw, dst, d, len);
            return;
        }
        byte[] kc = cache8[l];
        float sc = scale8[l][s];
        for (int i = 0; i < len; i++) {
            dst[d + i] = kc[s * aw + i] * sc;
        }
    }

    /**
     * One task per (token tile, head range): scores of every (token, head) of the task against each cached row
     * (read in place for f32, expanded once per task for f16 and q8), softmax, the weighted latent sum, then
     * {@code W_kb_v[h]} on it. Later tiles see more keys and are handed out first; a short chunk (decode) splits
     * the heads instead so that every thread has work.
     */
    private void attend(int l, int n, int pos) {
        int kvl = m.kvLoraRank();
        int vd = m.vDim();
        float scale = m.scale();
        int tiles = (n + tile - 1) / tile;
        int chunks = Math.min(heads, Math.max(1, (2 * b.pool.threads() + tiles - 1) / tiles)); // head ranges per tile
        int hpc = (heads + chunks - 1) / chunks;                                              // heads per range
        chunks = (heads + hpc - 1) / hpc;
        float[] rows32 = cache32 != null ? cache32[l] : null;
        float[] wvl = wv[l];
        float[] att = b.att;
        int hc = chunks;
        b.pool.forRangeFine(tiles * hc, 1, (i0, i1) -> {
            float[] sc = scratch.get();
            float[] rows = rows32 != null ? rows32 : sc;
            for (int i = i0; i < i1; i++) {
                int h0 = (i % hc) * hpc;
                int h1 = Math.min(heads, h0 + hpc);
                int t0 = (tiles - 1 - i / hc) * tile;
                int t1 = Math.min(n, t0 + tile);
                int sMax = pos + t1;
                for (int s = 0; s < sMax; s++) {
                    int so;
                    if (rows32 != null) {
                        so = s * aw;
                    } else {
                        expand(l, s, aw, sc, rowOff);
                        so = rowOff;
                    }
                    for (int t = Math.max(t0, s - pos); t < t1; t++) { // tokens that see position s
                        int row = (t - t0) * hpc;
                        for (int h = h0; h < h1; h++) {
                            sc[(row + h - h0) * ctx + s] = VectorKernels.dot(qAbs, t * qaStride + h * aw, rows, so, aw) * scale;
                        }
                    }
                }
                for (int t = t0; t < t1; t++) {
                    int len = pos + t + 1;
                    for (int h = h0; h < h1; h++) {
                        int row = (t - t0) * hpc + h - h0;
                        sc[invOff + row] = 1f / KvCache.softmax(sc, row * ctx, len, 0f);
                        Arrays.fill(sc, latOff + row * kvl, latOff + (row + 1) * kvl, 0f);
                    }
                }
                for (int s = 0; s < sMax; s++) {
                    int so;
                    if (rows32 != null) {
                        so = s * aw;
                    } else {
                        expand(l, s, kvl, sc, rowOff);
                        so = rowOff;
                    }
                    for (int t = Math.max(t0, s - pos); t < t1; t++) {
                        int row = (t - t0) * hpc;
                        for (int h = h0; h < h1; h++) {
                            int r = row + h - h0;
                            VectorKernels.axpy(sc[r * ctx + s] * sc[invOff + r], rows, so, sc, latOff + r * kvl, kvl);
                        }
                    }
                }
                for (int t = t0; t < t1; t++) {
                    for (int h = h0; h < h1; h++) {
                        int lo = latOff + ((t - t0) * hpc + h - h0) * kvl;
                        int ao = t * oStride + h * vd;
                        int wo = h * vd * kvl;
                        for (int j = 0; j < vd; j++) {
                            att[ao + j] = VectorKernels.dot(wvl, wo + j * kvl, sc, lo, kvl);
                        }
                    }
                }
            }
        });
    }
}
