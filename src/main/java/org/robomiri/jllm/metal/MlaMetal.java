package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.DeepSeekConfig;
import org.robomiri.jllm.model.MlaWeights;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelWeights;

/**
 * DeepSeek multi-head latent attention, expert router and shared experts on the GPU, driven by
 * {@link MetalBackend} (see {@code MlaCpu} for the math). Projections, norms and attention (the runtime-width
 * {@code attention} instantiation: keys nope + rope wide, values vDim wide) and the rotary embedding (the
 * interleaved {@code rope_il} on the slice after the nope part) reuse the generic kernels; the kernels added
 * for DeepSeek are {@code mla_kv_store} and {@code router_ds}.
 *
 * <p>The cache keeps the expanded per-head keys ({@code heads*(nope+rope)} floats per token) and values
 * ({@code heads*v}) in f32, like the CPU backend: 553 KB per token for DeepSeek-V2-Lite.
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
        Metal.Buffer kvB;
        Metal.Buffer sharedGateUp;
        Metal.Buffer sharedDown;
        Metal.Buffer routerBias;
    }

    private final MetalBackend mb;
    private final MetalKernels k;
    private final ModelConfig c;
    private final DeepSeekConfig d;
    private final int ctx;
    private final Bufs[] bufs;
    private final Metal.Buffer[] key;
    private final Metal.Buffer[] val;
    private final Metal.Buffer qa;
    private final Metal.Buffer qaNorm;
    private final Metal.Buffer lat;
    private final Metal.Buffer latNorm;
    private final Metal.Buffer kpe;
    private final Metal.Buffer kvb;

    MlaMetal(MetalBackend mb, ModelWeights w, int ctx, int maxBatch) {
        this.mb = mb;
        this.k = mb.k;
        this.c = w.cfg;
        this.d = c.deepseek();
        this.ctx = ctx;
        if (d.qkDim() % 32 != 0 || d.qkDim() > 256 || d.vHeadDim() % 32 != 0 || d.vHeadDim() > 256
                || d.qkRopeDim() / 2 > 256) {
            throw new IllegalArgumentException("Metal MLA needs qk (" + d.qkDim() + ") and v (" + d.vHeadDim()
                    + ") head dims that are multiples of 32 and <= 256");
        }
        if (c.isMoe() && c.numExperts() > MetalKernels.MAX_EXPERTS) {
            throw new IllegalArgumentException("Metal router kernel supports at most " + MetalKernels.MAX_EXPERTS + " experts");
        }
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
            b.kvB = k.wrap(m.kvB.data);
            if (m.sharedGateUp != null) {
                b.sharedGateUp = k.wrap(m.sharedGateUp.data);
                b.sharedDown = k.wrap(m.sharedDown.data);
            }
            if (m.routerBias != null) {
                b.routerBias = k.upload(m.routerBias);
            }
            bufs[l] = b;
        }
        // Not zeroed, like the dense caches (see MetalBackend.init): mla_kv_store writes every key and value
        // element of rows [pos, pos + n) before attention reads rows [0, pos + n), and the projection scratch
        // below is written for all n tokens of the chunk before it is read.
        int heads = c.heads();
        key = new Metal.Buffer[c.layers()];
        val = new Metal.Buffer[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            key[l] = k.scratch((long) ctx * heads * d.qkDim() * 4);
            val[l] = k.scratch((long) ctx * heads * d.vHeadDim() * 4);
        }
        qa = k.scratch((long) maxBatch * Math.max(d.qLoraRank(), 1) * 4);
        qaNorm = k.scratch((long) maxBatch * Math.max(d.qLoraRank(), 1) * 4);
        lat = k.scratch((long) maxBatch * d.kvLoraRank() * 4);
        latNorm = k.scratch((long) maxBatch * d.kvLoraRank() * 4);
        kpe = k.scratch((long) maxBatch * d.qkRopeDim() * 4);
        kvb = k.scratch((long) maxBatch * heads * (d.qkNopeDim() + d.vHeadDim()) * 4);
    }

    /** Reads {@code xb}, writes the attention output {@code [n, heads*v]} to {@code att} (o_proj is the caller's). */
    void run(int l, ModelWeights.Layer W, int n, int pos) {
        MlaWeights m = W.mla;
        Bufs b = bufs[l];
        int heads = c.heads();
        int qk = d.qkDim();
        int nope = d.qkNopeDim();
        int rd = d.qkRopeDim();
        int vd = d.vHeadDim();
        if (m.qProj != null) {
            k.matmul(m.qProj, b.qProj, mb.xb, 0, mb.q, 0, n, false, null, 0f);
        } else {
            k.matmul(m.qA, b.qA, mb.xb, 0, qa, 0, n, false, null, 0f);
            k.rmsnorm(qa, 0, b.qALn, qaNorm, n, d.qLoraRank());
            k.matmul(m.qB, b.qB, qaNorm, 0, mb.q, 0, n, false, null, 0f);
        }
        k.matmul(m.kvA, b.kvA, mb.xb, 0, lat, 0, n, false, null, 0f);
        k.matmul(m.kPe, b.kPe, mb.xb, 0, kpe, 0, n, false, null, 0f);
        k.rmsnorm(lat, 0, b.kvALn, latNorm, n, d.kvLoraRank());
        k.matmul(m.kvB, b.kvB, latNorm, 0, kvb, 0, n, false, null, 0f);

        // interleaved rotation of the rd-wide slice after the nope part of every q head, and of the shared
        // k_pe row, both in place through the q path of the rope kernel (no k heads)
        k.rope(true, false, mb.q, mb.q, 0, mb.q, 0, null, null, mb.ropeCos, mb.ropeSin, qk, heads * qk, heads * qk,
                heads, 0, pos, nope, rd / 2, n);
        k.rope(true, false, kpe, kpe, 0, kpe, 0, null, null, mb.ropeCos, mb.ropeSin, rd, rd, rd, 1, 0, pos, 0, rd / 2, n);

        k.mlaKvStore(kvb, kpe, key[l], (long) pos * heads * qk * 4, val[l], (long) pos * heads * vd * 4, nope, rd, vd, heads, n);

        // the runtime-width attention kernel: keys heads * qk wide, values heads * vd, no GQA, window or cap
        k.attention("attention", 1, mb.q, key[l], val[l], mb.att, mb.scores, heads, heads, qk, vd, pos, ctx,
                (float) d.softmaxScale(), heads * qk, heads * qk, heads * vd, heads * vd, 0, 0f, n);
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
