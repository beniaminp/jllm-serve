package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.DeepSeekConfig;
import org.robomiri.jllm.model.MlaWeights;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelWeights;

import java.lang.foreign.ValueLayout;

/**
 * DeepSeek multi-head latent attention, expert router and shared experts on the GPU, driven by
 * {@link MetalBackend} (see {@code MlaCpu} for the math). Projections, norms and attention (the runtime-width
 * {@code attention} instantiation: keys nope + rope wide, values vDim wide) reuse the generic kernels; the
 * kernels added for DeepSeek are {@code mla_rope}, {@code mla_kv_store} and {@code router_ds}.
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
    private final Metal metal;
    private final ModelConfig c;
    private final DeepSeekConfig d;
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
        this.metal = mb.metal;
        this.c = w.cfg;
        this.d = c.deepseek();
        if (d.qkDim() % 32 != 0 || d.qkDim() > 256 || d.vHeadDim() % 32 != 0 || d.vHeadDim() > 256
                || d.qkRopeDim() / 2 > 256) {
            throw new IllegalArgumentException("Metal MLA needs qk (" + d.qkDim() + ") and v (" + d.vHeadDim()
                    + ") head dims that are multiples of 32 and <= 256");
        }
        if (c.isMoe() && c.numExperts() > 1024) {
            throw new IllegalArgumentException("Metal router kernel supports at most 1024 experts");
        }
        bufs = new Bufs[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            MlaWeights m = w.layers[l].mla;
            Bufs b = new Bufs();
            if (m.qProj != null) {
                b.qProj = metal.wrap(m.qProj.data);
            } else {
                b.qA = metal.wrap(m.qA.data);
                b.qALn = mb.upload(m.qALayernorm);
                b.qB = metal.wrap(m.qB.data);
            }
            b.kvA = metal.wrap(m.kvA.data);
            b.kPe = metal.wrap(m.kPe.data);
            b.kvALn = mb.upload(m.kvALayernorm);
            b.kvB = metal.wrap(m.kvB.data);
            if (m.sharedGateUp != null) {
                b.sharedGateUp = metal.wrap(m.sharedGateUp.data);
                b.sharedDown = metal.wrap(m.sharedDown.data);
            }
            if (m.routerBias != null) {
                b.routerBias = mb.upload(m.routerBias);
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
            key[l] = metal.newBuffer((long) ctx * heads * d.qkDim() * 4, false);
            val[l] = metal.newBuffer((long) ctx * heads * d.vHeadDim() * 4, false);
        }
        qa = metal.newBuffer((long) maxBatch * Math.max(d.qLoraRank(), 1) * 4, false);
        qaNorm = metal.newBuffer((long) maxBatch * Math.max(d.qLoraRank(), 1) * 4, false);
        lat = metal.newBuffer((long) maxBatch * d.kvLoraRank() * 4, false);
        latNorm = metal.newBuffer((long) maxBatch * d.kvLoraRank() * 4, false);
        kpe = metal.newBuffer((long) maxBatch * d.qkRopeDim() * 4, false);
        kvb = metal.newBuffer((long) maxBatch * heads * (d.qkNopeDim() + d.vHeadDim()) * 4, false);
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
            mb.matmul(m.qProj, b.qProj, mb.xb, 0, mb.q, 0, n, false, null);
        } else {
            mb.matmul(m.qA, b.qA, mb.xb, 0, qa, 0, n, false, null);
            mb.rmsnorm(qa, b.qALn, qaNorm, n, d.qLoraRank());
            mb.matmul(m.qB, b.qB, qaNorm, 0, mb.q, 0, n, false, null);
        }
        mb.matmul(m.kvA, b.kvA, mb.xb, 0, lat, 0, n, false, null);
        mb.matmul(m.kPe, b.kPe, mb.xb, 0, kpe, 0, n, false, null);
        mb.rmsnorm(lat, b.kvALn, latNorm, n, d.kvLoraRank());
        mb.matmul(m.kvB, b.kvB, latNorm, 0, kvb, 0, n, false, null);

        rope(mb.q, heads, qk, nope, rd, heads * qk, n, pos);
        rope(kpe, 1, rd, 0, rd, rd, n, pos);

        mb.bind(0, kvb, 0);
        mb.bind(1, kpe, 0);
        mb.bind(2, key[l], (long) pos * heads * qk * 4);
        mb.bind(3, val[l], (long) pos * heads * vd * 4);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 0, nope);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 1, rd);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 2, vd);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 3, heads * (nope + vd));
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 4, heads * qk);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 5, heads * vd);
        metal.dispatch(metal.pipeline("mla_kv_store"), mb.bufArgs, mb.offArgs, 4, mb.params, 24, heads, n, 1, 64, 1, 1);

        // the runtime-width attention kernel: keys heads * qk wide, values heads * vd, no GQA, window or cap
        mb.attention("attention", 1, mb.q, key[l], val[l], mb.att, heads, heads, qk, vd, pos, (float) d.softmaxScale(),
                heads * qk, heads * qk, heads * vd, heads * vd, 0, 0f, n);
    }

    private void rope(Metal.Buffer v, int heads, int qk, int nope, int rd, int rowStride, int n, int pos) {
        mb.bind(0, v, 0);
        mb.bind(1, mb.ropeCos, 0);
        mb.bind(2, mb.ropeSin, 0);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 0, qk);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 1, nope);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 2, rd);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 3, rowStride);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 4, pos);
        metal.dispatch(metal.pipeline("mla_rope"), mb.bufArgs, mb.offArgs, 3, mb.params, 20, heads, n, 1, rd / 2, 1, 1);
    }

    /** Routes the n tokens of {@code xb}: fills the backend's assign / routeWeight buffers (scaling included). */
    void router(int l, Metal.Buffer routerW, int n) {
        Bufs b = bufs[l];
        mb.bind(0, mb.xb, 0);
        mb.bind(1, routerW, 0);
        mb.bind(2, mb.assign, 0);
        mb.bind(3, mb.routeWeight, 0);
        mb.bind(4, b.routerBias != null ? b.routerBias : routerW, 0); // dummy unless noaux_tc
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 0, c.numExperts());
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 1, c.dim());
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 2, c.expertsPerTok());
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 3, c.normTopK() ? 1 : 0);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 4, d.nGroup());
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 5, d.topkGroup());
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 6, d.sigmoid() ? 1 : 0);
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 7, switch (d.topkMethod()) {
            case "greedy" -> 0;
            case "group_limited_greedy" -> 1;
            default -> 2;
        });
        mb.params.setAtIndex(ValueLayout.JAVA_INT, 8, d.scaleAlways() ? 1 : 0);
        mb.params.setAtIndex(ValueLayout.JAVA_FLOAT, 9, (float) d.routedScale());
        metal.dispatch(metal.pipeline("router_ds"), mb.bufArgs, mb.offArgs, 5, mb.params, 40, n, 1, 1, 256, 1, 1);
    }

    /** Shared experts: a dense SwiGLU of width nShared*moeDim on every token, added to x. */
    void sharedExperts(ModelWeights.Layer W, int l, int n) {
        MlaWeights m = W.mla;
        if (m.sharedGateUp == null) {
            return;
        }
        Bufs b = bufs[l];
        mb.swiglu(m.sharedGateUp, b.sharedGateUp, mb.xb, null, n); // writes silu(gate) * up to hidden
        mb.matmul(m.sharedDown, b.sharedDown, mb.hidden, 0, mb.x, 0, n, true, null);
    }
}
