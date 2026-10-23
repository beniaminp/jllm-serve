package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.HybridArch;
import org.robomiri.jllm.model.HybridLayer;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.MlpWeights;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.Rope;
import org.robomiri.jllm.model.Router;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * The attention / state-space hybrids (Granite 4, Nemotron-H, Jamba) on the GPU (docs/14.22); the state contract is
 * {@link org.robomiri.jllm.cpu.HybridCpuBackend}'s ({@code pos == 0} resets, no rewinding). The whole chunk is one command
 * buffer. Per layer: {@code rmsnorm}, the mixer, the residual add (the output projection accumulates into x when the
 * layer's multiplier is 1, else into a scratch that {@code vec_axpy} scales into x), then the feed-forward block the same way.
 * <ul>
 *   <li>attention: {@code matmul3} (q, k to f32 staging, v straight into the cache rows), {@code rope} (the identity table of a
 *       model without positions copies k into the cache and leaves q alone), the attention kernel, {@code o_proj};</li>
 *   <li>Mamba-2: {@code in_proj}, {@code m2_conv}, {@code m2_scan}, {@code m2_gnorm}, {@code out_proj};</li>
 *   <li>Mamba-1 (Jamba): the kernels of {@link MambaMetalBackend} with {@code rms3} on the output of {@code x_proj};</li>
 *   <li>feed-forward: {@code matmul_swiglu} + down, or {@code matmul} + {@code relu2} + down (Nemotron-H), or the router, the
 *       routed experts (per-slot kernels, or the expert-grouped ones for long chunks) and Granite's shared MLP.</li>
 * </ul>
 * Only the attention layers have KV buffers; the recurrent states are zero-initialized shared buffers.
 */
public final class HybridMetalBackend implements Backend {

    private final ModelWeights w;
    private final HybridArch arch;
    private final Architecture.Core c;
    private final HybridArch.Mamba2 m2;
    private final HybridArch.Mamba1 m1;
    private final int ctx;
    private final int maxBatch;
    private final boolean kvHalf;
    private final MetalKernels k;

    private Metal.Buffer embedQ;
    private Metal.Buffer lmHeadQ;
    private Metal.Buffer finalNorm;
    private Metal.Buffer ropeCos;
    private Metal.Buffer ropeSin;
    private String attnKernel;
    private LayerBufs[] layers;
    private Metal.Buffer[] keyCache;
    private Metal.Buffer[] valCache;
    private Metal.Buffer[] convState;
    private Metal.Buffer[] ssmState;
    private int statePos = 0;

    private Metal.Buffer x;
    private Metal.Buffer xb;
    private Metal.Buffer mixOut;
    private Metal.Buffer ffnOut;
    private Metal.Buffer proj;
    private Metal.Buffer conv;
    private Metal.Buffer dbc;
    private Metal.Buffer y;
    private Metal.Buffer q;
    private Metal.Buffer kStage;
    private Metal.Buffer att;
    private Metal.Buffer scores;
    private Metal.Buffer hidden;
    private Metal.Buffer assign;
    private Metal.Buffer routeWeight;
    private Metal.Buffer moeHidden;
    private Metal.Buffer moeOrder;
    private Metal.Buffer moeTiles;
    private Metal.Buffer moeOut;
    private Metal.Buffer logitsBuf;

    private static final class LayerBufs {
        Metal.Buffer norm1;
        Metal.Buffer norm2;
        // attention
        Metal.Buffer wq;
        Metal.Buffer wk;
        Metal.Buffer wv;
        Metal.Buffer wo;
        // state-space
        Metal.Buffer inProj;
        Metal.Buffer conv;
        Metal.Buffer convBias;
        Metal.Buffer dtBias;
        Metal.Buffer a;
        Metal.Buffer d;
        Metal.Buffer norm;
        Metal.Buffer outProj;
        Metal.Buffer xProj;
        Metal.Buffer dtProj;
        Metal.Buffer norms3;
        // feed-forward
        Metal.Buffer gateUp;
        Metal.Buffer down;
        Metal.Buffer router;
        Metal.Buffer routerBias;
        Metal.Buffer expGateUp;
        Metal.Buffer expDown;
        Metal.Buffer sharedGateUp;
        Metal.Buffer sharedDown;
    }

    /** See {@link MetalBackend#concurrent}. */
    public void concurrent(boolean on) {
        k.concurrent(on ? MetalKernels.concurrentOn() : 0);
    }

    /** Memory barriers the concurrent encoder has inserted so far. */
    public long barriers() {
        return k.metal.barriers();
    }

    public HybridMetalBackend(ModelWeights w, int ctx, int maxBatch, String metalLib, KvDtype kvDtype) {
        this.w = w;
        this.arch = w.hybrid();
        this.c = arch.core();
        this.m2 = arch.mamba2();
        this.m1 = arch.mamba1();
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        if (kvDtype == KvDtype.Q8 || kvDtype == KvDtype.Q4) { // no q8 / q4 attention path for this family on Metal
            System.err.println("note: the Metal hybrid backend has no " + kvDtype + " KV cache, using f16");
            kvDtype = KvDtype.F16;
        }
        this.kvHalf = kvDtype == KvDtype.F16;
        if (arch.attention() != null) {
            int hd = arch.attention().headDim();
            if (hd % 32 != 0 || hd > 256) {
                throw new IllegalArgumentException("Metal backend needs head_dim a multiple of 32 and <= 256, got " + hd);
            }
        }
        if (m2 != null && (m2.convKernel() > MetalConstants.MAMBA_MAX_CONV)) {
            throw new IllegalArgumentException("Metal Mamba conv supports conv_kernel <= " + MetalConstants.MAMBA_MAX_CONV
                    + ", got " + m2.convKernel());
        }
        if (m1 != null && (m1.convKernel() > MetalConstants.MAMBA_MAX_CONV || m1.state() > MetalConstants.MAMBA_MAX_STATE)) {
            throw new IllegalArgumentException("Metal Mamba supports conv_kernel <= " + MetalConstants.MAMBA_MAX_CONV
                    + " and state_size <= " + MetalConstants.MAMBA_MAX_STATE + ", got " + m1);
        }
        if (arch.maxExperts() > MetalConstants.MAX_EXPERTS) {
            throw new IllegalArgumentException("Metal router kernel supports at most " + MetalConstants.MAX_EXPERTS + " experts");
        }
        this.k = new MetalKernels(Metal.create(metalLib), c.normEps());
        try { // owns the Metal context from here on: closed again if anything below throws
            init(kvDtype);
        } catch (RuntimeException | Error e) {
            try {
                k.close();
            } catch (RuntimeException | Error suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    private void init(KvDtype kvDtype) {
        Attention.Gqa g = arch.attention();
        int dim = c.dim();
        int layersN = c.layers();
        embedQ = k.wrap(w.embed.data);
        lmHeadQ = w.lmHead == w.embed ? embedQ : k.wrap(w.lmHead.data);
        finalNorm = k.upload(w.finalNorm);
        layers = new LayerBufs[layersN];
        keyCache = new Metal.Buffer[arch.attentionLayers()];
        valCache = new Metal.Buffer[arch.attentionLayers()];
        convState = new Metal.Buffer[layersN];
        ssmState = new Metal.Buffer[layersN];
        if (g != null) {
            int hd = g.headDim();
            String vh = kvHalf ? "_h" : "";
            attnKernel = hd == 64 || hd == 128 || hd == 256 ? "attention_g1_d" + hd + vh : "attention" + vh;
            Rope rope = new Rope(arch.rope(), ctx); // the identity table of a model without positions: cos 1, sin 0
            ropeCos = k.upload(rope.cos);
            ropeSin = k.upload(rope.sin);
        }
        for (int l = 0; l < layersN; l++) {
            HybridArch.Layer spec = arch.layers().get(l);
            HybridLayer L = w.hybridLayer(l);
            LayerBufs b = new LayerBufs();
            b.norm1 = k.upload(L.norm1());
            b.norm2 = L.norm2() != null ? k.upload(L.norm2()) : null;
            int ki = arch.kindIndex(l);
            switch (L.mixer()) {
                case HybridLayer.Attn a -> {
                    b.wq = k.wrap(a.gqa().wq().data);
                    b.wk = k.wrap(a.gqa().wk().data);
                    b.wv = k.wrap(a.gqa().wv().data);
                    b.wo = k.wrap(a.wo().data);
                    long kvBytes = (long) ctx * g.kvDim() * kvDtype.bytes;
                    // rows [0, pos + n) are written (v by matmul3, k by rope) before the attention reads them
                    keyCache[ki] = k.scratch(kvBytes);
                    valCache[ki] = k.scratch(kvBytes);
                }
                case HybridLayer.M2 m -> {
                    b.inProj = k.wrap(m.inProj().data);
                    b.conv = k.upload(m.conv());
                    b.convBias = k.upload(m.convBias());
                    b.dtBias = k.upload(m.dtBias());
                    b.a = k.upload(m.a());
                    b.d = k.upload(m.d());
                    b.norm = k.upload(m.norm());
                    b.outProj = k.wrap(m.outProj().data);
                    convState[l] = k.metal.newBuffer((long) m2.convStateSize() * 4); // zeroed: the initial state
                    ssmState[l] = k.metal.newBuffer((long) m2.ssmStateSize() * 4);
                }
                case HybridLayer.M1 m -> {
                    b.inProj = k.wrap(m.inProj().data);
                    b.conv = k.upload(m.conv());
                    b.convBias = k.upload(m.convBias());
                    b.xProj = k.wrap(m.xProj().data);
                    b.dtProj = k.upload(m.dtProj());
                    b.dtBias = k.upload(m.dtBias());
                    b.a = k.upload(m.a());
                    b.d = k.upload(m.d());
                    b.outProj = k.wrap(m.outProj().data);
                    float[] norms = new float[m.dtNorm().length + m.bNorm().length + m.cNorm().length];
                    System.arraycopy(m.dtNorm(), 0, norms, 0, m.dtNorm().length);
                    System.arraycopy(m.bNorm(), 0, norms, m.dtNorm().length, m.bNorm().length);
                    System.arraycopy(m.cNorm(), 0, norms, m.dtNorm().length + m.bNorm().length, m.cNorm().length);
                    b.norms3 = k.upload(norms);
                    convState[l] = k.metal.newBuffer((long) m1.convStateSize() * 4);
                    ssmState[l] = k.metal.newBuffer((long) m1.ssmStateSize() * 4);
                }
                case null -> {
                }
            }
            switch (L.mlp()) {
                case null -> {
                }
                case MlpWeights.Dense d -> {
                    b.gateUp = k.wrap(d.gateUp().data);
                    b.down = k.wrap(d.down().data);
                }
                case MlpWeights.Moe mo -> {
                    b.router = k.upload(mo.router());
                    b.routerBias = mo.routerBias() != null ? k.upload(mo.routerBias()) : null;
                    b.expGateUp = k.wrap(mo.expGateUp().data);
                    b.expDown = k.wrap(mo.expDown().data);
                    if (mo.shared() != null) {
                        b.sharedGateUp = k.wrap(mo.shared().gateUp().data);
                        b.sharedDown = k.wrap(mo.shared().down().data);
                    }
                }
            }
            layers[l] = b;
        }
        x = k.scratch((long) maxBatch * dim * 4);
        xb = k.scratch((long) maxBatch * dim * 4);
        mixOut = k.scratch((long) maxBatch * dim * 4);
        ffnOut = k.scratch((long) maxBatch * dim * 4);
        int projWidth = Math.max(m2 != null ? m2.projDim() : 0, m1 != null ? 2 * m1.dInner() : 0);
        proj = k.scratch((long) maxBatch * Math.max(1, projWidth) * 4);
        conv = k.scratch((long) maxBatch * Math.max(1, Math.max(m2 != null ? m2.convDim() : 0, m1 != null ? m1.dInner() : 0)) * 4);
        dbc = m1 != null ? k.scratch((long) maxBatch * (m1.dtRank() + 2 * m1.state()) * 4) : null;
        y = k.scratch((long) maxBatch * Math.max(1, arch.maxMixerWidth()) * 4);
        if (g != null) {
            q = k.scratch((long) maxBatch * g.qDim() * 4);
            kStage = k.scratch((long) maxBatch * g.kvDim() * 4);
            att = k.scratch((long) maxBatch * g.qDim() * 4);
            scores = k.scratch((long) Math.min(MetalConstants.ATT_TILE, maxBatch) * g.heads() * ctx * 4);
        }
        hidden = k.scratch((long) maxBatch * Math.max(1, arch.maxMlpDim()) * 4);
        if (arch.anyMoe()) {
            int slots = maxBatch * arch.maxTopK();
            assign = k.scratch((long) slots * 4);
            routeWeight = k.scratch((long) slots * 4);
            moeHidden = k.scratch((long) slots * arch.maxMoeDim() * 4);
            moeOrder = k.scratch((long) slots * 4);
            moeTiles = k.scratch((long) (1 + 3 * MetalKernels.moeMaxTiles(slots, arch.maxExperts())) * 4);
            moeOut = k.scratch((long) slots * dim * 4);
        }
        logitsBuf = k.scratch((long) w.lmHead.rows * 4);
        long ws = k.metal.recommendedWorkingSet();
        if (w.byteSize() > ws) {
            System.err.printf("warning: weights (%.1f GB) exceed Metal's recommended working set (%.1f GB); expect paging%n",
                    w.byteSize() / 1e9, ws / 1e9);
        }
    }

    @Override
    public String name() {
        return "metal hybrid (" + k.metal.deviceName() + ")";
    }

    @Override
    public int contextLength() {
        return ctx;
    }

    @Override
    public int maxBatch() {
        return maxBatch;
    }

    /** The recurrent state of the state-space layers cannot be rewound to an earlier position. */
    @Override
    public boolean supportsPrefixReuse() {
        return false;
    }

    @Override
    public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
        if (n < 1 || n > maxBatch || pos + n > ctx) {
            throw new IllegalArgumentException("bad forward(n=" + n + ", pos=" + pos + ")");
        }
        if (pos == 0) {
            for (int l = 0; l < c.layers(); l++) {
                if (convState[l] != null) {
                    convState[l].contents().fill((byte) 0);
                    ssmState[l].contents().fill((byte) 0);
                }
            }
        } else if (pos != statePos) {
            throw new IllegalStateException("the hybrid state holds " + statePos + " tokens, cannot continue at " + pos);
        }
        statePos = -1;
        int dim = c.dim();
        float[] row = new float[dim];
        for (int t = 0; t < n; t++) {
            w.embed.dequantizeRow(tokens[off + t], row, 0);
            if (c.embedScale() != 1f) {
                for (int i = 0; i < dim; i++) {
                    row[i] *= c.embedScale();
                }
            }
            MemorySegment.copy(row, 0, x.contents(), ValueLayout.JAVA_FLOAT, (long) t * dim * 4, dim);
        }
        k.begin();
        boolean committed = false;
        try {
            for (int l = 0; l < c.layers(); l++) {
                HybridArch.Layer spec = arch.layers().get(l);
                HybridLayer L = w.hybridLayer(l);
                LayerBufs b = layers[l];
                if (L.mixer() != null) {
                    k.rmsnorm(x, 0, b.norm1, xb, n, dim);
                    switch (L.mixer()) {
                        case HybridLayer.Attn a -> attention(a, b, spec.residual(), arch.kindIndex(l), n, pos);
                        case HybridLayer.M2 m -> mamba2(m, b, spec.residual(), l, n);
                        case HybridLayer.M1 m -> mamba1(m, b, spec.residual(), l, n);
                    }
                }
                if (L.mlp() != null) {
                    k.rmsnorm(x, 0, L.mixer() != null ? b.norm2 : b.norm1, xb, n, dim);
                    ffn(spec, L.mlp(), b, n);
                }
                if (MetalKernels.COMMIT_LAYERS > 0 && (l + 1) % MetalKernels.COMMIT_LAYERS == 0 && l + 1 < c.layers()) {
                    k.commit(); // the GPU starts on these layers while the next ones are encoded
                }
            }
            if (logits != null) {
                k.rmsnorm(x, (long) (n - 1) * dim * 4, finalNorm, xb, 1, dim);
                k.matmul(w.lmHead, lmHeadQ, xb, 0, logitsBuf, 0, 1, false, null, 0f);
            }
            k.commitAndWait();
            committed = true;
        } finally {
            if (!committed) {
                k.abort(); // an un-ended encoder would make Metal abort the process on release
            }
        }
        if (logits != null) {
            int rows = Math.min(w.lmHead.rows, logits.length);
            MemorySegment.copy(logitsBuf.contents(), ValueLayout.JAVA_FLOAT, 0, logits, 0, rows);
            float d = arch.logitsDivisor();
            if (d != 1f) {
                for (int i = 0; i < rows; i++) {
                    logits[i] /= d;
                }
            }
        }
        statePos = pos + n;
    }

    /** {@code x += r * (W in)} for the n rows: the matmul accumulates into x for r == 1, else goes through a scratch. */
    private void project(QTensor wt, Metal.Buffer wbuf, Metal.Buffer in, float r, int n) {
        int dim = c.dim();
        if (r == 1f) {
            k.matmul(wt, wbuf, in, 0, x, 0, n, true, null, 0f);
        } else {
            k.matmul(wt, wbuf, in, 0, mixOut, 0, n, false, null, 0f);
            k.vecAxpy(mixOut, x, n * dim, r);
        }
    }

    private void attention(HybridLayer.Attn a, LayerBufs b, float r, int ai, int n, int pos) {
        Attention.Gqa g = arch.attention();
        int hd = g.headDim();
        int kvDim = g.kvDim();
        long cacheOff = (long) pos * kvDim * (kvHalf ? 2 : 4);
        k.matmul3(a.gqa().wq(), b.wq, a.gqa().wk(), b.wk, a.gqa().wv(), b.wv, xb, null, q, kStage, 0, valCache[ai], cacheOff,
                kvHalf, null, null, null, n);
        k.rope(false, kvHalf, q, kStage, 0, keyCache[ai], cacheOff, null, null, ropeCos, ropeSin, hd, g.qDim(), kvDim,
                g.heads(), g.kvHeads(), pos, 0, arch.rope().rotaryDim() / 2, n);
        k.attention(attnKernel, 1, q, keyCache[ai], valCache[ai], att, scores, g.heads(), g.kvHeads(), hd, hd, pos, ctx,
                g.scale(), g.qDim(), kvDim, kvDim, g.qDim(), 0, 0f, n);
        project(a.wo(), b.wo, att, r, n);
    }

    private void mamba2(HybridLayer.M2 m, LayerBufs b, float r, int layer, int n) {
        int di = m2.dInner();
        int cd = m2.convDim();
        int pw = m2.projDim();
        k.matmul(m.inProj(), b.inProj, xb, 0, proj, 0, n, false, null, 0f);
        k.m2Conv(proj, b.conv, b.convBias, convState[layer], conv, cd, m2.convKernel(), pw, di, n);
        k.m2Scan(conv, proj, b.dtBias, b.a, b.d, ssmState[layer], y, m2.heads(), m2.headDim(), m2.groups(), m2.state(), pw,
                di + cd, cd, di, m2.dtMin(), m2.dtMax(), n);
        k.m2Gnorm(y, proj, b.norm, di, di / m2.normGroups(), pw, m2.normEps(), n);
        project(m.outProj(), b.outProj, y, r, n);
    }

    private void mamba1(HybridLayer.M1 m, LayerBufs b, float r, int layer, int n) {
        int di = m1.dInner();
        k.matmul(m.inProj(), b.inProj, xb, 0, proj, 0, n, false, null, 0f);
        k.mambaConv(proj, b.conv, b.convBias, convState[layer], conv, di, m1.convKernel(), n);
        k.matmul(m.xProj(), b.xProj, conv, 0, dbc, 0, n, false, null, 0f);
        k.rms3(dbc, b.norms3, m1.dtRank(), m1.state(), m1.normEps(), n);
        k.mambaScan(conv, proj, dbc, b.dtProj, b.dtBias, b.a, b.d, ssmState[layer], y, di, m1.state(), m1.dtRank(), n);
        project(m.outProj(), b.outProj, y, r, n);
    }

    /** The feed-forward block on the normed rows of xb, added to x (times the layer's multiplier). */
    private void ffn(HybridArch.Layer spec, MlpWeights mlp, LayerBufs b, int n) {
        int dim = c.dim();
        boolean direct = spec.residual() == 1f;
        Metal.Buffer dst = direct ? x : ffnOut;
        if (!direct) {
            k.vecZero(ffnOut, n * dim);
        }
        switch (mlp) {
            case MlpWeights.Dense d -> {
                Mlp.Dense ds = (Mlp.Dense) spec.ffn();
                if (ds.act() == Mlp.Activation.RELU2) {
                    k.matmul(d.gateUp(), b.gateUp, xb, 0, hidden, 0, n, false, null, 0f);
                    k.relu2(hidden, hidden, n * ds.hidden());
                } else {
                    k.swiglu(d.gateUp(), b.gateUp, xb, null, hidden, false, n);
                }
                k.matmul(d.down(), b.down, hidden, 0, dst, 0, n, true, null, 0f);
            }
            case MlpWeights.Moe mo -> moe((Mlp.Moe) spec.ffn(), mo, b, dst, n);
        }
        if (!direct) {
            k.vecAxpy(ffnOut, x, n * dim, spec.residual());
        }
    }

    private void moe(Mlp.Moe spec, MlpWeights.Moe m, LayerBufs b, Metal.Buffer dst, int n) {
        int dim = c.dim();
        switch (spec.router()) {
            case Router.TopLogits t -> k.routerDs(xb, b.router, assign, routeWeight, b.routerBias, spec.experts(), dim,
                    spec.topK(), !t.sigmoid(), 1, 1, t.sigmoid(), b.routerBias != null ? 3 : 0, false, 1f, null, null, null, n);
            default -> k.router(xb, b.router, assign, routeWeight, spec.experts(), dim, spec.topK(), spec.normTopK(), null, null,
                    null, n);
        }
        if (MetalKernels.MOE_MMA && n >= MetalKernels.MOE_MMA_MIN) {
            k.moeGrouped(m.expGateUp(), b.expGateUp, m.expDown(), b.expDown, xb, assign, routeWeight, moeOrder, moeTiles,
                    moeHidden, moeOut, dst, spec.experts(), spec.moeDim(), dim, spec.topK(), n);
        } else {
            k.moeUp(m.expGateUp(), b.expGateUp, xb, assign, moeHidden, spec.moeDim(), spec.topK(), n);
            k.moeDown(m.expDown(), b.expDown, moeHidden, assign, routeWeight, dst, dim, spec.topK(), n);
        }
        if (m.shared() != null) {
            k.swiglu(m.shared().gateUp(), b.sharedGateUp, xb, null, hidden, false, n);
            k.matmul(m.shared().down(), b.sharedDown, hidden, 0, dst, 0, n, true, null, 0f);
        }
    }

    @Override
    public void close() {
        k.close();
    }
}
