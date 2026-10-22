package org.robomiri.jllm.metal;

import org.robomiri.jllm.cpu.Qwen3NextCpuBackend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.AttnWeights;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.MlpWeights;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.MtpFeeder;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.Qwen3NextArch;
import org.robomiri.jllm.model.Qwen3NextLayer;
import org.robomiri.jllm.model.Qwen3NextMtp;
import org.robomiri.jllm.model.Rope;

import java.io.EOFException;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;

/**
 * Qwen3-Next forward pass on the GPU (docs/14.14); the state contract is {@link Qwen3NextCpuBackend}'s ({@code pos == 0}
 * resets, no rewinding, exact save / load / copy of the KV rows plus the DeltaNet states, the same bytes as the CPU).
 * The whole chunk is one command buffer. Per layer:
 * <ul>
 *   <li>Gated DeltaNet: {@code rmsnorm}, one {@code matmul} for the fused [q k v | z | b a] projection, {@code gdn_conv},
 *       {@code gdn_rule} (the delta rule, tokens in order inside each thread: prefill is the token loop, see the
 *       kernel), {@code gdn_norm_gate}, {@code matmul} {@code out_proj} accumulating into x;</li>
 *   <li>gated attention: {@code rmsnorm}, {@code matmul3} (q, k to f32 staging, v straight into the cache rows),
 *       {@code matmul} for the gate, the per-head q / k norms with the plain {@code rmsnorm} kernel, {@code rope} on the
 *       rotary slice (which also stores k into the cache, as half for an f16 cache), the attention kernel,
 *       {@code sigmoid_mul}, {@code matmul} {@code o_proj};</li>
 *   <li>MLP: {@code rmsnorm} then a dense {@code matmul_swiglu} + down, or {@code router}, the routed experts (per-slot
 *       kernels, or the expert-grouped ones for long chunks) and the shared expert ({@code matmul_swiglu}, down,
 *       {@code shared_gate_add}).</li>
 * </ul>
 * Only the full-attention layers have KV buffers; the DeltaNet states are zero-initialized shared buffers.
 */
public final class Qwen3NextMetalBackend implements Backend {

    private final ModelWeights w;
    private final Qwen3NextArch arch;
    private final Architecture.Core c;
    private final Qwen3NextArch.DeltaNet dn;
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
    private Metal.Buffer[] state;
    private int statePos = 0;

    private Metal.Buffer x;
    private Metal.Buffer xb;
    private Metal.Buffer proj;
    /** Only for a GGUF model ({@link Qwen3NextLayer.DeltaNet#split()}): the z and the b | a products. */
    private Metal.Buffer projZ;
    private Metal.Buffer projBa;
    private Metal.Buffer qkv;
    private Metal.Buffer o;
    private Metal.Buffer q;
    private Metal.Buffer gate;
    private Metal.Buffer kStage;
    private Metal.Buffer att;
    private Metal.Buffer scores;
    private Metal.Buffer hidden;
    private Metal.Buffer sharedOut;
    private Metal.Buffer assign;
    private Metal.Buffer routeWeight;
    private Metal.Buffer moeHidden;
    private Metal.Buffer moeOrder;
    private Metal.Buffer moeTiles;
    private Metal.Buffer moeOut;
    private Metal.Buffer logitsBuf;
    // multi-token prediction (null mtp: the model has no head)
    private final Qwen3NextMtp mtp;
    private MtpFeeder feeder;
    private LayerBufs mtpBufs;
    private Metal.Buffer ehBuf;
    private Metal.Buffer headBuf;
    private Metal.Buffer headNorm;
    private Metal.Buffer mtpIn;      // [rows, 2 * dim]: the normed token embedding then the normed hidden state
    private float[] hid;             // [n, dim]: the normed hidden states of the last pass
    private Metal.Buffer logitsAll;  // [rows, vocab] for forwardAll
    private int logitsAllRows;
    private Metal.Buffer[] ckConv;
    private Metal.Buffer[] ckState;
    private int ckPos;
    private boolean armed;          // checkpointState(): the next pass of two tokens saves the state after its first token
    private boolean captured;       // ... and did (ckConv / ckState hold it)
    private int capturedPos;

    private static final class LayerBufs {
        Metal.Buffer attnNorm;
        Metal.Buffer ffnNorm;
        // DeltaNet
        Metal.Buffer inProj;
        Metal.Buffer inProjZ;
        Metal.Buffer inProjBa;
        Metal.Buffer conv;
        Metal.Buffer a;
        Metal.Buffer dtBias;
        Metal.Buffer gdnNorm;
        Metal.Buffer outProj;
        // attention
        Metal.Buffer wq;
        Metal.Buffer wk;
        Metal.Buffer wv;
        Metal.Buffer wGate;
        Metal.Buffer wo;
        Metal.Buffer qNorm;
        Metal.Buffer kNorm;
        // MLP
        Metal.Buffer gateUp;
        Metal.Buffer down;
        Metal.Buffer router;
        Metal.Buffer expGateUp;
        Metal.Buffer expDown;
        Metal.Buffer sharedGate;
    }

    /**
     * Selects the concurrent compute encoder (true: independent dispatches overlap, a barrier goes between dependent ones)
     * or the serial one (false) for the passes that follow; the results are bit-identical. The default is
     * {@code -Djllm.metal.concurrent} (docs/08 section 8.28).
     */
    public void concurrent(boolean on) {
        k.concurrent(on ? MetalKernels.concurrentOn() : 0);
    }

    /** Memory barriers the concurrent encoder has inserted so far. */
    public long barriers() {
        return k.metal.barriers();
    }

    public Qwen3NextMetalBackend(ModelWeights w, int ctx, int maxBatch, String metalLib, KvDtype kvDtype) {
        this.w = w;
        this.mtp = w.mtp;
        this.arch = w.qwen3Next();
        this.c = arch.core();
        this.dn = arch.deltaNet();
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        if (kvDtype == KvDtype.Q8 || kvDtype == KvDtype.Q4) { // no q8 / q4 attention path for this family on Metal: its cache rows are f32 or f16
            System.err.println("note: the Metal Qwen3-Next backend has no " + kvDtype + " KV cache, using f16");
            kvDtype = KvDtype.F16;
        }
        this.kvHalf = kvDtype == KvDtype.F16;
        int hd = arch.attention().headDim();
        if (hd % 32 != 0 || hd > 256) {
            throw new IllegalArgumentException("Metal backend needs head_dim a multiple of 32 and <= 256, got " + hd);
        }
        if (dn.keyDim() % 32 != 0 || dn.keyDim() > MetalKernels.GDN_MAX_KEY_DIM) {
            throw new IllegalArgumentException("Metal Gated DeltaNet needs linear_key_head_dim a multiple of 32 and <= "
                    + MetalKernels.GDN_MAX_KEY_DIM + ", got " + dn.keyDim());
        }
        if (dn.convKernel() > MetalKernels.GDN_MAX_CONV) {
            throw new IllegalArgumentException("Metal Gated DeltaNet supports linear_conv_kernel_dim <= "
                    + MetalKernels.GDN_MAX_CONV + ", got " + dn.convKernel());
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
        int hd = g.headDim();
        String vh = kvHalf ? "_h" : "";
        attnKernel = hd == 64 || hd == 128 || hd == 256 ? "attention_g1_d" + hd + vh : "attention" + vh;
        embedQ = k.wrap(w.embed.data);
        lmHeadQ = w.lmHead == w.embed ? embedQ : k.wrap(w.lmHead.data);
        finalNorm = k.upload(w.finalNorm);
        Rope rope = new Rope(arch.rope(), ctx);
        ropeCos = k.upload(rope.cos);
        ropeSin = k.upload(rope.sin);
        layers = new LayerBufs[c.layers()];
        boolean splitProj = false;
        int kvLayers = arch.fullLayers() + (mtp != null ? 1 : 0); // the head's block has the last KV slot
        keyCache = new Metal.Buffer[kvLayers];
        valCache = new Metal.Buffer[kvLayers];
        convState = new Metal.Buffer[arch.linearLayers()];
        state = new Metal.Buffer[arch.linearLayers()];
        long kvBytes = (long) ctx * g.kvDim() * kvDtype.bytes;
        for (int l = 0; l < c.layers(); l++) {
            Qwen3NextLayer L = w.qwen3NextLayer(l);
            LayerBufs b = new LayerBufs();
            b.attnNorm = k.upload(L.attnNorm());
            b.ffnNorm = k.upload(L.ffnNorm());
            int ki = arch.kindIndex(l);
            switch (L.mixer()) {
                case Qwen3NextLayer.DeltaNet d -> {
                    b.inProj = k.wrap(d.inProj().data);
                    if (d.split()) {
                        b.inProjZ = k.wrap(d.z().data);
                        b.inProjBa = k.wrap(d.ba().data);
                        splitProj = true;
                    }
                    b.conv = k.upload(d.conv());
                    b.a = k.upload(d.a());
                    b.dtBias = k.upload(d.dtBias());
                    b.gdnNorm = k.upload(d.norm());
                    b.outProj = k.wrap(d.outProj().data);
                    convState[ki] = k.metal.newBuffer((long) dn.convStateSize() * 4); // zeroed: the initial state
                    state[ki] = k.metal.newBuffer((long) dn.stateSize() * 4);
                }
                case Qwen3NextLayer.GatedAttention a -> {
                    AttnWeights.Gqa wg = a.gqa();
                    b.wq = k.wrap(wg.wq().data);
                    b.wk = k.wrap(wg.wk().data);
                    b.wv = k.wrap(wg.wv().data);
                    b.wGate = k.wrap(a.gate().data);
                    b.wo = k.wrap(a.wo().data);
                    b.qNorm = k.upload(wg.qNorm());
                    b.kNorm = k.upload(wg.kNorm());
                    // rows [0, pos + n) are written (v by matmul3, k by rope) before the attention reads them
                    keyCache[ki] = k.scratch(kvBytes);
                    valCache[ki] = k.scratch(kvBytes);
                }
            }
            switch (L.mlp()) {
                case MlpWeights.Dense d -> {
                    b.gateUp = k.wrap(d.gateUp().data);
                    b.down = k.wrap(d.down().data);
                }
                case MlpWeights.Moe m -> {
                    b.router = k.upload(m.router());
                    b.expGateUp = k.wrap(m.expGateUp().data);
                    b.expDown = k.wrap(m.expDown().data);
                    b.gateUp = k.wrap(m.shared().gateUp().data);
                    b.down = k.wrap(m.shared().down().data);
                    b.sharedGate = k.upload(L.sharedGate());
                }
            }
            layers[l] = b;
        }
        if (mtp != null) {
            Qwen3NextLayer L = mtp.layer();
            LayerBufs b = new LayerBufs();
            b.attnNorm = k.upload(L.attnNorm());
            b.ffnNorm = k.upload(L.ffnNorm());
            Qwen3NextLayer.GatedAttention a = L.attention();
            AttnWeights.Gqa wg = a.gqa();
            b.wq = k.wrap(wg.wq().data);
            b.wk = k.wrap(wg.wk().data);
            b.wv = k.wrap(wg.wv().data);
            b.wGate = k.wrap(a.gate().data);
            b.wo = k.wrap(a.wo().data);
            b.qNorm = k.upload(wg.qNorm());
            b.kNorm = k.upload(wg.kNorm());
            keyCache[kvLayers - 1] = k.scratch(kvBytes);
            valCache[kvLayers - 1] = k.scratch(kvBytes);
            switch (L.mlp()) {
                case MlpWeights.Dense d -> {
                    b.gateUp = k.wrap(d.gateUp().data);
                    b.down = k.wrap(d.down().data);
                }
                case MlpWeights.Moe m -> {
                    b.router = k.upload(m.router());
                    b.expGateUp = k.wrap(m.expGateUp().data);
                    b.expDown = k.wrap(m.expDown().data);
                    b.gateUp = k.wrap(m.shared().gateUp().data);
                    b.down = k.wrap(m.shared().down().data);
                    b.sharedGate = k.upload(L.sharedGate());
                }
            }
            mtpBufs = b;
            ehBuf = k.wrap(mtp.ehProj().data);
            headBuf = mtp.head() == w.lmHead ? lmHeadQ : k.wrap(mtp.head().data);
            headNorm = k.upload(mtp.headNorm());
            mtpIn = k.scratch((long) maxBatch * 2 * c.dim() * 4);
            hid = new float[maxBatch * c.dim()];
            feeder = new MtpFeeder(c.dim(), this::mtpRows);
        }
        int dim = c.dim();
        x = k.scratch((long) maxBatch * dim * 4);
        xb = k.scratch((long) maxBatch * dim * 4);
        proj = k.scratch((long) maxBatch * dn.projDim() * 4);
        if (splitProj) {
            projZ = k.scratch((long) maxBatch * dn.valueWidth() * 4);
            projBa = k.scratch((long) maxBatch * 2 * dn.valueHeads() * 4);
        }
        qkv = k.scratch((long) maxBatch * dn.convDim() * 4);
        o = k.scratch((long) maxBatch * dn.valueWidth() * 4);
        q = k.scratch((long) maxBatch * g.qDim() * 4);
        gate = k.scratch((long) maxBatch * g.qDim() * 4);
        kStage = k.scratch((long) maxBatch * g.kvDim() * 4);
        att = k.scratch((long) maxBatch * g.qDim() * 4);
        scores = k.scratch((long) Math.min(MetalConstants.ATT_TILE, maxBatch) * g.heads() * ctx * 4);
        hidden = k.scratch((long) maxBatch * Math.max(1, arch.maxMlpDim()) * 4);
        if (arch.anyMoe()) {
            int slots = maxBatch * arch.maxTopK();
            sharedOut = k.scratch((long) maxBatch * dim * 4);
            assign = k.scratch((long) slots * 4);
            routeWeight = k.scratch((long) slots * 4);
            moeHidden = k.scratch((long) slots * arch.maxMoeDim() * 4);
            moeOrder = k.scratch((long) slots * 4);
            moeTiles = k.scratch((long) (1 + 3 * MetalKernels.moeMaxTiles(slots, arch.maxExperts())) * 4);
            moeOut = k.scratch((long) slots * dim * 4);
        }
        logitsBuf = k.scratch((long) w.lmHead.rows * 4);
        if (ctx > c.maxPositions()) {
            System.err.printf("warning: ctx %d exceeds the model's max_position_embeddings (%d); RoPE positions "
                    + "beyond that are extrapolated and quality degrades%n", ctx, c.maxPositions());
        }
        long ws = k.metal.recommendedWorkingSet();
        if (w.byteSize() > ws) {
            System.err.printf("warning: weights (%.1f GB) exceed Metal's recommended working set (%.1f GB); expect paging%n",
                    w.byteSize() / 1e9, ws / 1e9);
        }
    }

    @Override
    public String name() {
        return "metal qwen3-next (" + k.metal.deviceName() + ")";
    }

    @Override
    public int contextLength() {
        return ctx;
    }

    @Override
    public int maxBatch() {
        return maxBatch;
    }

    /** The recurrent state of the DeltaNet layers cannot be rewound to an earlier position. */
    @Override
    public boolean supportsPrefixReuse() {
        return false;
    }

    @Override
    public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
        run(tokens, null, off, n, pos, logits, 0);
    }

    // ---- images (docs/20): the vision encoder feeds embeddings; Qwen3.5 has its own rotary positions (interleaved M-RoPE)

    /** First slot whose rotary rows are not the plain ones ({@link #setRopePositions}); MAX_VALUE = none. */
    private int ropeCustomFrom = Integer.MAX_VALUE;

    @Override
    public boolean supportsEmbeds() {
        return true;
    }

    @Override
    public void forwardEmbeds(float[] embeds, int off, int n, int pos, float[][] deepstack, float[] logits) {
        if (deepstack != null) {
            throw new UnsupportedOperationException("deepstack features belong to Qwen3-VL, not to the Qwen3-Next family");
        }
        run(null, embeds, off, n, pos, logits, 0);
    }

    @Override
    public boolean setRopePositions(int from, org.robomiri.jllm.model.RopePositions p) {
        int half = arch.rope().rotaryDim() / 2;
        float[] cos = new float[(ctx - from) * half];
        float[] sin = new float[(ctx - from) * half];
        Rope.rows(arch.rope(), ctx, from, p, cos, sin);
        // the tables are shared host memory the kernels read in place; no work is in flight (the engine is idle)
        MemorySegment.copy(cos, 0, ropeCos.contents(), ValueLayout.JAVA_FLOAT, (long) from * half * 4, cos.length);
        MemorySegment.copy(sin, 0, ropeSin.contents(), ValueLayout.JAVA_FLOAT, (long) from * half * 4, sin.length);
        ropeCustomFrom = p != null ? Math.min(ropeCustomFrom, from) : (from <= ropeCustomFrom ? Integer.MAX_VALUE : ropeCustomFrom);
        return true;
    }

    @Override
    public boolean ropePositionsCustom() {
        return ropeCustomFrom != Integer.MAX_VALUE;
    }

    /** Logits of every row (the verification of drafts): at most {@link #MAX_ALL_ROWS} rows per call. */
    @Override
    public void forwardAll(int[] tokens, int off, int n, int pos, float[] logitsOut, int stride) {
        if (n > MAX_ALL_ROWS) {
            Backend.super.forwardAll(tokens, off, n, pos, logitsOut, stride);
            return;
        }
        run(tokens, null, off, n, pos, logitsOut, stride);
    }

    private static final int MAX_ALL_ROWS = 16;

    /** This pass saves the DeltaNet states after its first token into {@link #ckConv} / {@link #ckState}. */
    private boolean capture;

    /** {@code stride == 0}: the logits of the last row into {@code logits}; else row i to {@code logits[i * stride ..)}. */
    private void run(int[] tokens, float[] embeds, int off, int n, int pos, float[] logits, int stride) {
        if (n < 1 || n > maxBatch || pos + n > ctx) {
            throw new IllegalArgumentException("bad forward(n=" + n + ", pos=" + pos + ")");
        }
        capture = armed && n == 2 && stride > 0 && tokens != null;
        armed = false;
        captured = capture;
        capturedPos = pos;
        if (pos == 0) {
            for (int i = 0; i < state.length; i++) {
                convState[i].contents().fill((byte) 0);
                state[i].contents().fill((byte) 0);
            }
        } else if (pos != statePos) {
            throw new IllegalStateException("Qwen3-Next state holds " + statePos + " tokens, cannot continue at " + pos);
        }
        statePos = -1;
        int dim = c.dim();
        float[] row = new float[dim];
        if (embeds != null) {
            MemorySegment.copy(embeds, off * dim, x.contents(), ValueLayout.JAVA_FLOAT, 0, n * dim);
        } else {
            for (int t = 0; t < n; t++) {
                w.embed.dequantizeRow(tokens[off + t], row, 0);
                MemorySegment.copy(row, 0, x.contents(), ValueLayout.JAVA_FLOAT, (long) t * dim * 4, dim);
            }
        }
        k.begin();
        boolean committed = false;
        try {
            for (int l = 0; l < c.layers(); l++) {
                Qwen3NextLayer L = w.qwen3NextLayer(l);
                LayerBufs b = layers[l];
                k.rmsnorm(x, 0, b.attnNorm, xb, n, dim);
                switch (L.mixer()) {
                    case Qwen3NextLayer.DeltaNet d -> deltaNet(d, b, arch.kindIndex(l), n);
                    case Qwen3NextLayer.GatedAttention a -> attention(a, b, arch.kindIndex(l), n, pos);
                }
                k.rmsnorm(x, 0, b.ffnNorm, xb, n, dim);
                mlp(L, b, arch.mlps().get(l), n);
                if (MetalKernels.COMMIT_LAYERS > 0 && (l + 1) % MetalKernels.COMMIT_LAYERS == 0 && l + 1 < c.layers()) {
                    k.commit(); // the GPU starts on these layers while the next ones are encoded
                }
            }
            boolean mtpOn = feeder != null && feeder.enabled();
            if (stride > 0 && logits != null) { // every row's logits
                if (logitsAll == null || logitsAllRows < n) {
                    logitsAllRows = Math.max(n, 4);
                    logitsAll = k.scratch((long) logitsAllRows * w.lmHead.rows * 4);
                }
                k.rmsnorm(x, 0, finalNorm, xb, n, dim);
                k.matmul(w.lmHead, lmHeadQ, xb, 0, logitsAll, 0, n, false, null, 0f);
            } else if (mtpOn) { // the head reads every row's normed hidden state; the logits come from the last
                k.rmsnorm(x, 0, finalNorm, xb, n, dim);
                if (logits != null) {
                    k.matmul(w.lmHead, lmHeadQ, xb, (long) (n - 1) * dim * 4, logitsBuf, 0, 1, false, null, 0f);
                }
            } else if (logits != null) {
                k.rmsnorm(x, (long) (n - 1) * dim * 4, finalNorm, xb, 1, dim);
                k.matmul(w.lmHead, lmHeadQ, xb, 0, logitsBuf, 0, 1, false, null, 0f);
            }
            k.commitAndWait();
            if (mtpOn) {
                MemorySegment.copy(xb.contents(), ValueLayout.JAVA_FLOAT, 0, hid, 0, n * dim);
            }
            committed = true;
        } finally {
            if (!committed) {
                k.abort(); // an un-ended encoder would make Metal abort the process on release
            }
        }
        if (logits != null && stride > 0) {
            int rows = Math.min(w.lmHead.rows, stride);
            for (int i = 0; i < n; i++) {
                MemorySegment.copy(logitsAll.contents(), ValueLayout.JAVA_FLOAT, (long) i * w.lmHead.rows * 4, logits,
                        i * stride, rows);
            }
        } else if (logits != null) {
            MemorySegment.copy(logitsBuf.contents(), ValueLayout.JAVA_FLOAT, 0, logits, 0,
                    Math.min(w.lmHead.rows, logits.length));
        }
        statePos = pos + n;
        if (feeder != null && feeder.enabled()) {
            // (a picture's rows have no token: the head sees token 0 there, which only costs it some draft quality)
            feeder.afterPass(tokens != null ? tokens : new int[off + n], off, n, pos, hid);
        }
    }

    /** The MLP of a layer (dense SwiGLU, or experts plus the gated shared expert) on the normed rows of xb, added to x. */
    private void mlp(Qwen3NextLayer L, LayerBufs b, Mlp spec, int n) {
        switch (L.mlp()) {
            case MlpWeights.Dense d -> {
                k.swiglu(d.gateUp(), b.gateUp, xb, null, hidden, false, n);
                k.matmul(d.down(), b.down, hidden, 0, x, 0, n, true, null, 0f);
            }
            case MlpWeights.Moe m -> moe((Mlp.Moe) spec, m, b, n);
        }
    }

    // ------------------------------------------------------------------ multi-token prediction

    /** {@link org.robomiri.jllm.cpu.Qwen3NextCpuBackend}'s head on {@code n} rows, the block on the GPU. */
    private void mtpRows(int[] tokens, float[] h, int n, int pos, float[] logits) {
        int dim = c.dim();
        float[] emb = new float[dim];
        float[] norm = new float[dim];
        MemorySegment in = mtpIn.contents();
        for (int t = 0; t < n; t++) {
            mtp.embed().dequantizeRow(tokens[t], emb, 0);
            rmsnormRow(emb, 0, norm, mtp.enorm(), dim);
            MemorySegment.copy(norm, 0, in, ValueLayout.JAVA_FLOAT, (long) t * 2 * dim * 4, dim);
            rmsnormRow(h, t * dim, norm, mtp.hnorm(), dim);
            MemorySegment.copy(norm, 0, in, ValueLayout.JAVA_FLOAT, ((long) t * 2 * dim + dim) * 4, dim);
        }
        Qwen3NextLayer L = mtp.layer();
        k.begin();
        boolean committed = false;
        try {
            k.matmul(mtp.ehProj(), ehBuf, mtpIn, 0, x, 0, n, false, null, 0f);
            k.rmsnorm(x, 0, mtpBufs.attnNorm, xb, n, dim);
            attention(L.attention(), mtpBufs, arch.fullLayers(), n, pos);
            k.rmsnorm(x, 0, mtpBufs.ffnNorm, xb, n, dim);
            mlp(L, mtpBufs, arch.mlps().getLast(), n);
            if (logits != null) {
                k.rmsnorm(x, (long) (n - 1) * dim * 4, headNorm, xb, 1, dim);
                k.matmul(mtp.head(), headBuf, xb, 0, logitsBuf, 0, 1, false, null, 0f);
            }
            k.commitAndWait();
            committed = true;
        } finally {
            if (!committed) {
                k.abort();
            }
        }
        if (logits != null) {
            MemorySegment.copy(logitsBuf.contents(), ValueLayout.JAVA_FLOAT, 0, logits, 0,
                    Math.min(mtp.head().rows, logits.length));
        }
    }

    /** {@code out = in * rsqrt(mean(in^2) + eps) * gain} (the plain RMSNorm of the head's two inputs). */
    private void rmsnormRow(float[] in, int off, float[] out, float[] gain, int width) {
        double ss = 0;
        for (int i = 0; i < width; i++) {
            ss += (double) in[off + i] * in[off + i];
        }
        float inv = (float) (1.0 / Math.sqrt(ss / width + c.normEps()));
        for (int i = 0; i < width; i++) {
            out[i] = in[off + i] * inv * gain[i];
        }
    }

    @Override
    public boolean hasMtp() {
        return mtp != null;
    }

    @Override
    public void enableMtp(boolean on) {
        if (mtp == null) {
            throw new UnsupportedOperationException(name() + ": the model has no multi-token-prediction head");
        }
        feeder.enable(on);
    }

    @Override
    public boolean mtpReady() {
        return feeder != null && feeder.ready(statePos);
    }

    @Override
    public void mtpDraft(int token, float[] logits) {
        if (!mtpReady()) {
            throw new IllegalStateException("the multi-token-prediction head does not follow the state at " + statePos);
        }
        feeder.draft(token, logits);
    }

    @Override
    public boolean supportsRollback() {
        return true;
    }

    /**
     * The next {@link #forwardAll} of two tokens writes the conv and recurrent states as they are after its first token
     * to a second set of buffers while it computes the second ({@code gdn_conv} / {@code gdn_rule} snapshot parameter),
     * so a rejected draft costs a swap of the two sets, not a replay.
     */
    @Override
    public void checkpointState() {
        if (ckConv == null) {
            ckConv = new Metal.Buffer[state.length];
            ckState = new Metal.Buffer[state.length];
            for (int l = 0; l < state.length; l++) {
                ckConv[l] = k.scratch((long) dn.convStateSize() * 4);
                ckState[l] = k.scratch((long) dn.stateSize() * 4);
            }
        }
        armed = true;
        captured = false;
    }

    @Override
    public void rollbackState(int keep) {
        if (!captured || keep != 1) {
            throw new IllegalStateException("no state was saved after " + keep + " tokens of the last pass");
        }
        for (int l = 0; l < state.length; l++) {
            Metal.Buffer c = convState[l];
            convState[l] = ckConv[l];
            ckConv[l] = c;
            Metal.Buffer s = state[l];
            state[l] = ckState[l];
            ckState[l] = s;
        }
        captured = false;
        statePos = capturedPos + keep;
        if (feeder != null) {
            feeder.truncate(capturedPos, keep, hid);
        }
    }

    private void deltaNet(Qwen3NextLayer.DeltaNet d, LayerBufs b, int li, int n) {
        int pd = dn.projDim();
        int cd = dn.convDim();
        k.matmul(d.inProj(), b.inProj, xb, 0, proj, 0, n, false, null, 0f);
        if (d.split()) { // q k v | z | b a are three products in three buffers
            k.matmul(d.z(), b.inProjZ, xb, 0, projZ, 0, n, false, null, 0f);
            k.matmul(d.ba(), b.inProjBa, xb, 0, projBa, 0, n, false, null, 0f);
            k.gdnConv(proj, b.conv, convState[li], qkv, cd, dn.convKernel(), cd, cd, n, capture ? ckConv[li] : null, 0);
            k.gdnRule(qkv, projBa, b.a, b.dtBias, state[li], o, dn.keyHeads(), dn.valueHeads(), dn.keyDim(), dn.valueDim(),
                    cd, 2 * dn.valueHeads(), 0, 1e-6f, dn.tiled(), n, capture ? ckState[li] : null, 0);
            k.gdnNormGate(o, projZ, b.gdnNorm, dn.valueDim(), dn.valueHeads(), dn.valueWidth(), 0, n);
        } else {
            k.gdnConv(proj, b.conv, convState[li], qkv, cd, dn.convKernel(), pd, cd, n, capture ? ckConv[li] : null, 0);
            k.gdnRule(qkv, proj, b.a, b.dtBias, state[li], o, dn.keyHeads(), dn.valueHeads(), dn.keyDim(), dn.valueDim(),
                    cd, pd, cd + dn.valueWidth(), 1e-6f, dn.tiled(), n, capture ? ckState[li] : null, 0);
            k.gdnNormGate(o, proj, b.gdnNorm, dn.valueDim(), dn.valueHeads(), pd, cd, n);
        }
        k.matmul(d.outProj(), b.outProj, o, 0, x, 0, n, true, null, 0f);
    }

    private void attention(Qwen3NextLayer.GatedAttention a, LayerBufs b, int fi, int n, int pos) {
        Attention.Gqa g = arch.attention();
        AttnWeights.Gqa wg = a.gqa();
        int hd = g.headDim();
        int kvDim = g.kvDim();
        long cacheOff = (long) pos * kvDim * (kvHalf ? 2 : 4);
        k.beginIndependent(); // the projections of xb write apart; so do the two norms below (concurrent encoder)
        k.matmul3(wg.wq(), b.wq, wg.wk(), b.wk, wg.wv(), b.wv, xb, null, q, kStage, 0, valCache[fi], cacheOff, kvHalf,
                null, null, null, n);
        k.matmul(a.gate(), b.wGate, xb, 0, gate, 0, n, false, null, 0f);
        k.endIndependent();
        // per-head q / k norms over the whole head (rows of headDim), then RoPE on the rotary slice only
        k.beginIndependent();
        k.rmsnorm(q, 0, b.qNorm, q, n * g.heads(), hd);
        k.rmsnorm(kStage, 0, b.kNorm, kStage, n * g.kvHeads(), hd);
        k.endIndependent();
        k.rope(false, kvHalf, q, kStage, 0, keyCache[fi], cacheOff, null, null, ropeCos, ropeSin, hd, g.qDim(), kvDim,
                g.heads(), g.kvHeads(), pos, 0, arch.rope().rotaryDim() / 2, n);
        k.attention(attnKernel, 1, q, keyCache[fi], valCache[fi], att, scores, g.heads(), g.kvHeads(), hd, hd, pos, ctx,
                g.scale(), g.qDim(), kvDim, kvDim, g.qDim(), 0, 0f, n);
        k.sigmoidMul(att, gate, n * g.qDim());
        k.matmul(a.wo(), b.wo, att, 0, x, 0, n, true, null, 0f);
    }

    private void moe(Mlp.Moe spec, MlpWeights.Moe m, LayerBufs b, int n) {
        int dim = c.dim();
        QTensor gu = m.expGateUp();
        QTensor dw = m.expDown();
        if (MetalKernels.MOE_MMA && n >= MetalKernels.MOE_MMA_MIN) {
            k.router(xb, b.router, assign, routeWeight, spec.experts(), dim, spec.topK(), spec.normTopK(), null, null, null, n);
            k.moeGrouped(gu, b.expGateUp, dw, b.expDown, xb, assign, routeWeight, moeOrder, moeTiles, moeHidden, moeOut, x,
                    spec.experts(), spec.moeDim(), dim, spec.topK(), n);
            k.swiglu(m.shared().gateUp(), b.gateUp, xb, null, hidden, false, n);
            k.matmul(m.shared().down(), b.down, hidden, 0, sharedOut, 0, n, false, null, 0f);
        } else {
            // the shared expert (up to sharedOut) and the routed ones read xb and write apart; the routed ones still add
            // into x before the shared one does, as before
            k.beginIndependent();
            k.router(xb, b.router, assign, routeWeight, spec.experts(), dim, spec.topK(), spec.normTopK(), null, null, null, n);
            k.swiglu(m.shared().gateUp(), b.gateUp, xb, null, hidden, false, n);
            k.endIndependent();
            k.beginIndependent();
            k.moeUp(gu, b.expGateUp, xb, assign, moeHidden, spec.moeDim(), spec.topK(), n);
            k.matmul(m.shared().down(), b.down, hidden, 0, sharedOut, 0, n, false, null, 0f);
            k.endIndependent();
            k.moeDown(dw, b.expDown, moeHidden, assign, routeWeight, x, dim, spec.topK(), n);
        }
        k.sharedGateAdd(xb, b.sharedGate, sharedOut, x, dim, n);
    }

    // ------------------------------------------------------------------ saved state

    private String kvFormat() {
        Attention.Gqa g = arch.attention();
        return arch.fullLayers() == 0 ? "none"
                : (kvHalf ? "f16" : "f32") + " layers=" + arch.fullLayers() + " kvDim=" + g.kvDim() + " headDim=" + g.headDim();
    }

    /** {@link Qwen3NextCpuBackend#stateFormat}: the same bytes on both backends. */
    @Override
    public String stateFormat() {
        return Qwen3NextCpuBackend.stateFormat(arch, kvFormat());
    }

    private void checkPositions(int n) {
        if (n != statePos) {
            throw new IllegalStateException("the Qwen3-Next state holds " + statePos + " tokens, not " + n
                    + " (a recurrent state cannot be cut)");
        }
    }

    private long rowBytes() {
        return (long) arch.attention().kvDim() * (kvHalf ? 2 : 4);
    }

    @Override
    public void saveState(int n, WritableByteChannel out) throws IOException {
        checkPositions(n);
        long row = rowBytes();
        int rows = (int) Math.max(1, Math.min(n, (1 << 22) / (2 * row)));
        ByteBuffer buf = ByteBuffer.allocateDirect((int) (rows * 2 * row));
        MemorySegment b = MemorySegment.ofBuffer(buf);
        for (int l = 0; l < arch.fullLayers(); l++) { // per layer, per position: the K row then the V row (KvCache.write)
            for (int s0 = 0; s0 < n; s0 += rows) {
                int m = Math.min(rows, n - s0);
                for (int r = 0; r < m; r++) {
                    MemorySegment.copy(keyCache[l].contents(), (s0 + r) * row, b, 2 * r * row, row);
                    MemorySegment.copy(valCache[l].contents(), (s0 + r) * row, b, (2L * r + 1) * row, row);
                }
                buf.clear().limit((int) (2 * m * row));
                while (buf.hasRemaining()) {
                    out.write(buf);
                }
            }
        }
        for (int l = 0; l < state.length; l++) {
            write(convState[l].contents().asSlice(0, (long) dn.convStateSize() * 4), out);
            write(state[l].contents().asSlice(0, (long) dn.stateSize() * 4), out);
        }
    }

    @Override
    public void loadState(int n, ReadableByteChannel in) throws IOException {
        if (n > ctx) {
            throw new IOException("the saved state holds " + n + " positions, the context is " + ctx);
        }
        statePos = -1;
        if (feeder != null) {
            feeder.invalidate();
        }
        long row = rowBytes();
        int rows = (int) Math.max(1, Math.min(n, (1 << 22) / (2 * row)));
        ByteBuffer buf = ByteBuffer.allocateDirect((int) (rows * 2 * row));
        MemorySegment b = MemorySegment.ofBuffer(buf);
        for (int l = 0; l < arch.fullLayers(); l++) {
            for (int s0 = 0; s0 < n; s0 += rows) {
                int m = Math.min(rows, n - s0);
                buf.clear().limit((int) (2 * m * row));
                while (buf.hasRemaining()) {
                    if (in.read(buf) < 0) {
                        throw new EOFException("saved Qwen3-Next state is truncated");
                    }
                }
                for (int r = 0; r < m; r++) {
                    MemorySegment.copy(b, 2 * r * row, keyCache[l].contents(), (s0 + r) * row, row);
                    MemorySegment.copy(b, (2L * r + 1) * row, valCache[l].contents(), (s0 + r) * row, row);
                }
            }
        }
        for (int l = 0; l < state.length; l++) {
            read(convState[l].contents().asSlice(0, (long) dn.convStateSize() * 4), in);
            read(state[l].contents().asSlice(0, (long) dn.stateSize() * 4), in);
        }
        statePos = n;
    }

    @Override
    public void copyStateFrom(Backend other, int n) {
        if (!(other instanceof Qwen3NextMetalBackend o) || !stateFormat().equals(o.stateFormat())) {
            Backend.super.copyStateFrom(other, n);
            return;
        }
        o.checkPositions(n);
        statePos = -1;
        if (feeder != null) {
            feeder.invalidate();
        }
        long bytes = n * rowBytes();
        for (int l = 0; l < arch.fullLayers(); l++) {
            MemorySegment.copy(o.keyCache[l].contents(), 0, keyCache[l].contents(), 0, bytes);
            MemorySegment.copy(o.valCache[l].contents(), 0, valCache[l].contents(), 0, bytes);
        }
        for (int l = 0; l < state.length; l++) {
            MemorySegment.copy(o.convState[l].contents(), 0, convState[l].contents(), 0, (long) dn.convStateSize() * 4);
            MemorySegment.copy(o.state[l].contents(), 0, state[l].contents(), 0, (long) dn.stateSize() * 4);
        }
        statePos = n;
    }

    /** The bytes of {@code seg} (little-endian on every Metal device) to {@code out}. */
    private static void write(MemorySegment seg, WritableByteChannel out) throws IOException {
        ByteBuffer b = seg.asByteBuffer();
        while (b.hasRemaining()) {
            out.write(b);
        }
    }

    private static void read(MemorySegment seg, ReadableByteChannel in) throws IOException {
        ByteBuffer b = seg.asByteBuffer();
        while (b.hasRemaining()) {
            if (in.read(b) < 0) {
                throw new EOFException("saved Qwen3-Next state is truncated");
            }
        }
    }

    @Override
    public void close() {
        k.close();
    }
}
