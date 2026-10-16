package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.DeepSeekConfig;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.MlaWeights;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelWeights;

/**
 * DeepSeek multi-head latent attention, expert router and shared experts on the GPU, driven by
 * {@link MetalBackend} (see {@code MlaCpu} for the math of the absorbed form). Projections, norms and the rotary
 * embedding (the interleaved {@code rope_il} on the slice after the nope part) reuse the generic kernels; the
 * attention is the generic kernel's {@code attention_mla[_h]} instantiation over one shared kv head with keys
 * {@code kv_lora_rank + rope} and values {@code kv_lora_rank} wide; the kernels specific to DeepSeek are
 * {@code mla_lat_store}, {@code mla_absorb_q}, {@code mla_apply_v} and {@code router_ds}.
 *
 * <p>The cache holds the latent rows {@code [c | k_pe]}: {@code (kv_lora_rank + rope)} values per token and layer in
 * f32 or, with {@code backend.kv-dtype=f16}, half (2.3 KB resp. 1.2 KB per token and layer for DeepSeek-V2-Lite).
 * The two halves of {@code kv_b_proj} live on the GPU dequantized to f32, {@code heads * (nope + v) * kv_lora_rank}
 * floats per layer (8 MB for V2-Lite).
 */
final class MlaMetal {

    private static final class Bufs {
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
        Metal.Buffer sharedGateUp;
        Metal.Buffer sharedDown;
        Metal.Buffer routerBias;
    }

    private final MetalBackend mb;
    private final MetalKernels k;
    private final ModelConfig c;
    private final DeepSeekConfig d;
    private final int ctx;
    private final boolean kvHalf;
    private final Bufs[] bufs;
    /** Per layer: ctx rows of kvLora + rope (f32 or half). */
    private final Metal.Buffer[] cache;
    private final Metal.Buffer qa;
    private final Metal.Buffer qaNorm;
    private final Metal.Buffer lat;
    private final Metal.Buffer latNorm;
    private final Metal.Buffer kpe;
    /** Absorbed queries [n, heads * (kvLora + rope)] and the attention output [n, heads * kvLora]. */
    private final Metal.Buffer qAbs;
    private final Metal.Buffer oLat;

    MlaMetal(MetalBackend mb, ModelWeights w, int ctx, int maxBatch, KvDtype kvDtype) {
        this.mb = mb;
        this.k = mb.k;
        this.c = w.cfg;
        this.d = c.deepseek();
        this.ctx = ctx;
        this.kvHalf = kvDtype == KvDtype.F16;
        int kvl = d.kvLoraRank();
        int aw = kvl + d.qkRopeDim();
        if (kvl % 32 != 0 || aw % 32 != 0 || aw > MetalConstants.MLA_MAX_PER * 32 || d.qkRopeDim() / 2 > 256) {
            throw new IllegalArgumentException("Metal MLA needs kv_lora_rank (" + kvl + ") and kv_lora_rank + rope ("
                    + aw + ") that are multiples of 32 and <= " + MetalConstants.MLA_MAX_PER * 32);
        }
        if (c.isMoe() && c.numExperts() > MetalConstants.MAX_EXPERTS) {
            throw new IllegalArgumentException("Metal router kernel supports at most " + MetalConstants.MAX_EXPERTS + " experts");
        }
        int heads = c.heads();
        bufs = new Bufs[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            MlaWeights m = w.layers[l].mla;
            Bufs b = new Bufs();
            if (m.qProj != null) {
                b.qProj = k.wrap(m.qProj.data);
            } else {
                b.qA = k.wrap(m.qA.data);
                b.qALn = k.upload(m.qALayernorm);
                b.qB = k.wrap(m.qB.data);
            }
            b.kvA = k.wrap(m.kvA.data);
            b.kPe = k.wrap(m.kPe.data);
            b.kvALn = k.upload(m.kvALayernorm);
            b.wk = k.upload(m.kvBKeys(d, heads, false));
            b.wv = k.upload(m.kvBValues(d, heads, true));
            if (m.sharedGateUp != null) {
                b.sharedGateUp = k.wrap(m.sharedGateUp.data);
                b.sharedDown = k.wrap(m.sharedDown.data);
            }
            if (m.routerBias != null) {
                b.routerBias = k.upload(m.routerBias);
            }
            bufs[l] = b;
        }
        // Not zeroed, like the dense caches (see MetalBackend.init): mla_lat_store writes every element of rows
        // [pos, pos + n) before attention reads rows [0, pos + n), and the projection scratch below is written for
        // all n tokens of the chunk before it is read.
        cache = new Metal.Buffer[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            cache[l] = k.scratch((long) ctx * aw * kvDtype.bytes);
        }
        qa = k.scratch((long) maxBatch * Math.max(d.qLoraRank(), 1) * 4);
        qaNorm = k.scratch((long) maxBatch * Math.max(d.qLoraRank(), 1) * 4);
        lat = k.scratch((long) maxBatch * kvl * 4);
        latNorm = k.scratch((long) maxBatch * kvl * 4);
        kpe = k.scratch((long) maxBatch * d.qkRopeDim() * 4);
        qAbs = k.scratch((long) maxBatch * heads * aw * 4);
        oLat = k.scratch((long) maxBatch * heads * kvl * 4);
    }

    /** Reads {@code xb}, writes the attention output {@code [n, heads*v]} to {@code att} (o_proj is the caller's). */
    void run(int l, ModelWeights.Layer W, int n, int pos) {
        MlaWeights m = W.mla;
        Bufs b = bufs[l];
        int heads = c.heads();
        int qk = d.qkDim();
        int nope = d.qkNopeDim();
        int rd = d.qkRopeDim();
        int kvl = d.kvLoraRank();
        int aw = kvl + rd;
        if (m.qProj != null) {
            k.matmul(m.qProj, b.qProj, mb.xb, 0, mb.q, 0, n, false, null, 0f);
        } else {
            k.matmul(m.qA, b.qA, mb.xb, 0, qa, 0, n, false, null, 0f);
            k.rmsnorm(qa, 0, b.qALn, qaNorm, n, d.qLoraRank());
            k.matmul(m.qB, b.qB, qaNorm, 0, mb.q, 0, n, false, null, 0f);
        }
        k.matmul(m.kvA, b.kvA, mb.xb, 0, lat, 0, n, false, null, 0f);
        k.matmul(m.kPe, b.kPe, mb.xb, 0, kpe, 0, n, false, null, 0f);
        k.rmsnorm(lat, 0, b.kvALn, latNorm, n, kvl);

        // interleaved rotation of the rd-wide slice after the nope part of every q head, and of the shared
        // k_pe row, both in place through the q path of the rope kernel (no k heads)
        k.rope(true, false, mb.q, mb.q, 0, mb.q, 0, null, null, mb.ropeCos, mb.ropeSin, qk, heads * qk, heads * qk,
                heads, 0, pos, nope, rd / 2, n);
        k.rope(true, false, kpe, kpe, 0, kpe, 0, null, null, mb.ropeCos, mb.ropeSin, rd, rd, rd, 1, 0, pos, 0, rd / 2, n);

        k.mlaLatStore(kvHalf, latNorm, kpe, cache[l], (long) pos * aw * (kvHalf ? 2 : 4), d, heads, n);
        k.mlaAbsorbQ(mb.q, b.wk, qAbs, d, heads, n);
        // one kv head (the cache row) shared by all query heads; the value is the leading kvl of the key row
        k.attention(kvHalf ? "attention_mla_h" : "attention_mla", 1, qAbs, cache[l], cache[l], oLat, mb.scores, heads, 1,
                aw, kvl, pos, ctx, (float) d.softmaxScale(), heads * aw, aw, aw, heads * kvl, 0, 0f, n);
        k.mlaApplyV(oLat, b.wv, mb.att, d, heads, n);
    }

    /** Routes the n tokens of {@code xb}: fills the backend's assign / routeWeight buffers (scaling included). */
    void router(int l, Metal.Buffer routerW, int n) {
        int method = switch (d.topkMethod()) {
            case "greedy" -> 0;
            case "group_limited_greedy" -> 1;
            default -> 2;
        };
        k.routerDs(mb.xb, routerW, mb.assign, mb.routeWeight, bufs[l].routerBias, c.numExperts(), c.dim(),
                c.expertsPerTok(), c.normTopK(), d.nGroup(), d.topkGroup(), d.sigmoid(), method, d.scaleAlways(),
                (float) d.routedScale(), n);
    }

    /** Shared experts: a dense SwiGLU of width nShared*moeDim on every token, added to x. */
    void sharedExperts(ModelWeights.Layer W, int l, int n) {
        MlaWeights m = W.mla;
        if (m.sharedGateUp == null) {
            return;
        }
        Bufs b = bufs[l];
        k.swiglu(m.sharedGateUp, b.sharedGateUp, mb.xb, null, mb.hidden, false, n); // writes silu(gate) * up to hidden
        k.matmul(m.sharedDown, b.sharedDown, mb.hidden, 0, mb.x, 0, n, true, null, 0f);
    }
}
