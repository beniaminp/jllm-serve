package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.AttnWeights;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;

/**
 * DeepSeek multi-head latent attention on the GPU, driven by {@link MetalBackend} (see {@code MlaCpu} for the
 * math of the absorbed form). Projections, norms and the rotary embedding (the interleaved {@code rope_il} on the
 * slice after the nope part) reuse the generic kernels; the attention is the generic kernel's
 * {@code attention_mla[_h|_q8]} instantiation over one shared kv head with keys {@code kv_lora_rank + rope} and values
 * {@code kv_lora_rank} wide; the kernels specific to DeepSeek are {@code mla_lat_store}, {@code mla_absorb_q} and
 * {@code mla_apply_v} (the router and the shared experts are the backend's MoE step).
 *
 * <p>The cache holds the latent rows {@code [c | k_pe]}: {@code (kv_lora_rank + rope)} values per token and layer in
 * f32 or, with {@code backend.kv-dtype=f16}, half, or with {@code q8} int8 and one f32 scale per row (2.3 KB, 1.2 KB
 * resp. 0.6 KB per token and layer for DeepSeek-V2-Lite).
 * The two halves of {@code kv_b_proj} live on the GPU dequantized to f32, {@code heads * (nope + v) * kv_lora_rank}
 * floats per layer (8 MB for V2-Lite).
 */
final class MlaMetal {

    private static final class Bufs {
        AttnWeights.Mla w;
        Metal.Buffer qProj;
        Metal.Buffer qA;
        Metal.Buffer qALn;
        Metal.Buffer qB;
        Metal.Buffer kvA;
        Metal.Buffer kPe;
        Metal.Buffer kvALn;
        /** W_kb_k as [heads][nope][kvLora] and W_kb_v as [heads][kvLora][v], f32. */
        Metal.Buffer wk;
        Metal.Buffer wv;
    }

    private final MetalBackend mb;
    private final MetalKernels k;
    private final Attention.Mla m;
    private final int ctx;
    private final KvDtype kv;
    private final Bufs[] bufs;
    /** Per layer: ctx rows of kvLora + rope (f32, half, or int8 followed by the row's f32 scale). */
    private final Metal.Buffer[] cache;
    private final Metal.Buffer qa;
    private final Metal.Buffer qaNorm;
    private final Metal.Buffer lat;
    private final Metal.Buffer latNorm;
    private final Metal.Buffer kpe;
    /** Absorbed queries [n, heads * (kvLora + rope)] and the attention output [n, heads * kvLora]. */
    private final Metal.Buffer qAbs;
    private final Metal.Buffer oLat;

    MlaMetal(MetalBackend mb, ModelWeights w, Attention.Mla m, int ctx, int maxBatch, KvDtype kvDtype) {
        this.mb = mb;
        this.k = mb.k;
        this.m = m;
        this.ctx = ctx;
        this.kv = kvDtype;
        int kvl = m.kvLoraRank();
        int aw = m.latentDim();
        if (kvl % 32 != 0 || aw % 32 != 0 || aw > MetalConstants.MLA_MAX_PER * 32 || m.rope() / 2 > 256) {
            throw new IllegalArgumentException("Metal MLA needs kv_lora_rank (" + kvl + ") and kv_lora_rank + rope ("
                    + aw + ") that are multiples of 32 and <= " + MetalConstants.MLA_MAX_PER * 32);
        }
        int heads = m.heads();
        int layers = w.arch.core().layers();
        bufs = new Bufs[layers];
        for (int l = 0; l < layers; l++) {
            AttnWeights.Mla mw = w.decoderLayer(l).mla();
            Bufs b = new Bufs();
            b.w = mw;
            switch (mw.query()) {
                case AttnWeights.Mla.Proj p -> b.qProj = k.wrap(p.w().data);
                case AttnWeights.Mla.Lora q -> {
                    b.qA = k.wrap(q.a().data);
                    b.qALn = k.upload(q.norm());
                    b.qB = k.wrap(q.b().data);
                }
            }
            b.kvA = k.wrap(mw.kvA().data);
            b.kPe = k.wrap(mw.kPe().data);
            b.kvALn = k.upload(mw.kvALayernorm());
            b.wk = k.upload(mw.kvBKeys(m, false));
            b.wv = k.upload(mw.kvBValues(m, true));
            bufs[l] = b;
        }
        // Not zeroed, like the dense caches (see MetalBackend.init): mla_lat_store writes every element of rows
        // [pos, pos + n) before attention reads rows [0, pos + n), and the projection scratch below is written for
        // all n tokens of the chunk before it is read.
        cache = new Metal.Buffer[layers];
        for (int l = 0; l < layers; l++) {
            cache[l] = k.scratch((long) ctx * kvDtype.rowBytes(aw, 1));
            k.rowIndexed(cache[l], kvDtype.rowBytes(aw, 1)); // for decode replay
        }
        qa = k.scratch((long) maxBatch * Math.max(m.qLoraRank(), 1) * 4);
        qaNorm = k.scratch((long) maxBatch * Math.max(m.qLoraRank(), 1) * 4);
        lat = k.scratch((long) maxBatch * kvl * 4);
        latNorm = k.scratch((long) maxBatch * kvl * 4);
        kpe = k.scratch((long) maxBatch * m.rope() * 4);
        qAbs = k.scratch((long) maxBatch * heads * aw * 4);
        oLat = k.scratch((long) maxBatch * heads * kvl * 4);
    }

    /** The latent cache of layer {@code l}: ctx rows of {@link #rowBytes} (saved caches read and write it directly). */
    java.lang.foreign.MemorySegment cache(int l) {
        return cache[l].contents();
    }

    /**
     * Layout of the saved rows, the same as {@code MlaCpu.format} (IEEE half and little-endian f32 on both); a q8
     * latent cache is not saved ({@code MetalBackend.stateFormat} returns null for it).
     */
    String format() {
        return (kv == KvDtype.F16 ? "f16" : "f32") + " layers=" + cache.length + " width=" + m.latentDim();
    }

    long rowBytes() {
        return (long) m.latentDim() * (kv == KvDtype.F16 ? 2 : 4);
    }

    int layers() {
        return cache.length;
    }

    /**
     * Reads {@code in} (normalized with {@code norm} on the fly when non-null: a single token, the *_norm matmuls; else
     * already normalized), writes the attention output {@code [n, heads*v]} to {@code att} (o_proj is the caller's).
     */
    void run(int l, int n, int pos, Metal.Buffer in, Metal.Buffer norm) {
        Bufs b = bufs[l];
        AttnWeights.Mla mw = b.w;
        int heads = m.heads();
        int qk = m.qkDim();
        int nope = m.nope();
        int rd = m.rope();
        int kvl = m.kvLoraRank();
        int aw = m.latentDim();
        switch (mw.query()) {
            case AttnWeights.Mla.Proj p -> k.matmul(p.w(), b.qProj, in, 0, mb.q, 0, n, false, norm, 0f);
            case AttnWeights.Mla.Lora q -> {
                k.matmul(q.a(), b.qA, in, 0, qa, 0, n, false, norm, 0f);
                k.rmsnorm(qa, 0, b.qALn, qaNorm, n, m.qLoraRank());
                k.matmul(q.b(), b.qB, qaNorm, 0, mb.q, 0, n, false, null, 0f);
            }
        }
        k.matmul(mw.kvA(), b.kvA, in, 0, lat, 0, n, false, norm, 0f);
        k.matmul(mw.kPe(), b.kPe, in, 0, kpe, 0, n, false, norm, 0f);
        k.rmsnorm(lat, 0, b.kvALn, latNorm, n, kvl);

        // interleaved rotation of the rd-wide slice after the nope part of every q head, and of the shared
        // k_pe row, both in place through the q path of the rope kernel (no k heads)
        Metal.Buffer cos = mb.ropeCos[m.ropeTable()];
        Metal.Buffer sin = mb.ropeSin[m.ropeTable()];
        k.rope(true, false, mb.q, mb.q, 0, mb.q, 0, null, null, cos, sin, qk, heads * qk, heads * qk,
                heads, 0, pos, nope, rd / 2, n);
        k.rope(true, false, kpe, kpe, 0, kpe, 0, null, null, cos, sin, rd, rd, rd, 1, 0, pos, 0, rd / 2, n);

        k.mlaLatStore(kv, latNorm, kpe, cache[l], pos * kv.rowBytes(aw, 1), m, n);
        k.mlaAbsorbQ(mb.q, b.wk, qAbs, m, n);
        // one kv head (the cache row) shared by all query heads; the value is the leading kvl of the key row
        int stride = kv.rowStride(aw, 1);
        k.attention("attention_mla" + MetalKernels.kvSuffix(kv), 1, qAbs, cache[l], cache[l], oLat, mb.scores, heads, 1,
                aw, kvl, pos, ctx, m.scale(), heads * aw, stride, stride, heads * kvl, 0, 0f, n);
        k.mlaApplyV(oLat, b.wv, mb.att, m, n);
    }
}
