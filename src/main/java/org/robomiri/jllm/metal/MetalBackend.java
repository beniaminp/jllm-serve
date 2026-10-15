package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.Rope;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Decoder forward pass (Llama, Mistral, Qwen2, Qwen3) on the GPU via Metal. Quantized weights are wrapped zero-copy (unified memory); the whole
 * forward pass for a chunk of tokens is recorded into a single command buffer and executed with one round trip.
 */
public final class MetalBackend implements Backend {

    private static final int TOK_TILE = 8; // must match kernels.metal
    /**
     * Tokens per attention dispatch; the scores scratch holds one row of ctx floats per (tile token, head)
     * instead of one per (batch token, head). 32 keeps a 64-token prefill chunk at two dispatches per layer.
     */
    private static final int ATT_TILE = 32;
    private static final int MAX_BUFS = 8;

    private final ModelWeights w;
    private final ModelConfig c;
    private final int ctx;
    private final int maxBatch;
    private final Metal metal;
    private final Arena arena = Arena.ofShared();

    private int pMatmulQ4;
    private int pMatmulQ8;
    private int pRmsnorm;
    private int pNormRope;
    private int pAttention;
    private int attnGroup; // query heads per attention threadgroup
    private int pSilu;
    private int pAddBias;
    private int pRouter;
    private int pMoeUpQ4;
    private int pMoeUpQ8;
    private int pMoeDownQ4;
    private int pMoeDownQ8;
    // f16 KV cache (backend.kv-dtype=f16): k/v go through f32 staging buffers, kv_store_half rounds them
    // into the half caches and attention_h reads them; see the "f16 KV cache" section of kernels.metal
    private final boolean kvHalf;
    private int pKvStoreHalf;
    private Metal.Buffer kStage;
    private Metal.Buffer vStage;

    private Metal.Buffer embedQ;
    private Metal.Buffer lmHeadQ;
    private LayerBufs[] layers;
    private Metal.Buffer finalNorm;
    private Metal.Buffer ropeCos;
    private Metal.Buffer ropeSin;
    private Metal.Buffer[] keyCache;
    private Metal.Buffer[] valCache;

    private Metal.Buffer x;
    private Metal.Buffer xb;
    private Metal.Buffer q;
    private Metal.Buffer att;
    private Metal.Buffer gateUp;
    private Metal.Buffer hidden;
    private Metal.Buffer scores;
    private Metal.Buffer logitsBuf;
    // MoE (null for dense models)
    private Metal.Buffer assign;
    private Metal.Buffer routeWeight;
    private Metal.Buffer moeHidden;

    // dispatch argument scratch (re-used; the native side copies params at record time)
    private MemorySegment bufArgs;
    private MemorySegment offArgs;
    private MemorySegment params;

    private static final class LayerBufs {
        Metal.Buffer attnNorm;
        Metal.Buffer ffnNorm;
        Metal.Buffer qNorm;
        Metal.Buffer kNorm;
        Metal.Buffer bq;
        Metal.Buffer bk;
        Metal.Buffer bv;
        Metal.Buffer wq;
        Metal.Buffer wk;
        Metal.Buffer wv;
        Metal.Buffer wo;
        Metal.Buffer wGateUp;
        Metal.Buffer wDown;
        Metal.Buffer router;
        Metal.Buffer expGateUp;
        Metal.Buffer expDown;
    }

    public MetalBackend(ModelWeights w, int ctx, int maxBatch) {
        this(w, ctx, maxBatch, (String) null);
    }

    public MetalBackend(ModelWeights w, int ctx, int maxBatch, String metalLib) {
        this(w, ctx, maxBatch, metalLib, KvDtype.F32);
    }

    public MetalBackend(ModelWeights w, int ctx, int maxBatch, String metalLib, KvDtype kvDtype) {
        this(w, ctx, maxBatch, open(w.cfg, metalLib), kvDtype);
    }

    private static Metal open(ModelConfig c, String metalLib) {
        if (c.headDim() % 32 != 0 || c.headDim() > 256) {
            throw new IllegalArgumentException("Metal backend needs headDim multiple of 32 and <= 256");
        }
        return Metal.create(metalLib);
    }

    /** Owns {@code metal} from here on: it is closed again if anything below throws. */
    private MetalBackend(ModelWeights w, int ctx, int maxBatch, Metal metal, KvDtype kvDtype) {
        this.w = w;
        this.c = w.cfg;
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        this.kvHalf = kvDtype == KvDtype.F16;
        this.metal = metal;
        try {
            init(kvDtype);
        } catch (RuntimeException | Error e) {
            try {
                metal.close();
            } catch (RuntimeException | Error suppressed) {
                e.addSuppressed(suppressed);
            }
            arena.close();
            throw e;
        }
    }

    private void init(KvDtype kvDtype) {
        pMatmulQ4 = metal.pipeline("matmul_q4");
        pMatmulQ8 = metal.pipeline("matmul_q8");
        pRmsnorm = metal.pipeline("rmsnorm");
        pNormRope = metal.pipeline("norm_rope");
        // Attention specialised on headDim; attnGroup query heads of one KV head per threadgroup. Sharing K/V
        // rows across the group (attnGroup > 1) measured no faster on Qwen3-0.6B (G=2) and slower on
        // Qwen3-30B-A3B (G=4/8: 20-22 tok/s vs 25 with G=1) because only kvHeads threadgroups run per decode
        // token and each one's serial work grows with the group, so the default is 1;
        // -Djllm.metal.attnGroup=2|4|8 overrides (it must divide heads / kvHeads).
        int group = c.heads() / c.kvHeads();
        int g = 1;
        int override = Integer.getInteger("jllm.metal.attnGroup", 0);
        if (override > 0 && override <= 8 && Integer.bitCount(override) == 1 && group % override == 0) {
            g = override;
        }
        int hd = c.headDim();
        if (hd == 64 || hd == 128 || hd == 256) {
            attnGroup = g;
            pAttention = metal.pipeline("attention_g" + g + "_d" + hd + (kvHalf ? "_h" : ""));
        } else {
            attnGroup = 1;
            pAttention = metal.pipeline(kvHalf ? "attention_h" : "attention");
        }
        pSilu = metal.pipeline("silu_mul");
        pAddBias = metal.pipeline("add_bias");
        pRouter = metal.pipeline("router");
        pMoeUpQ4 = metal.pipeline("moe_up_q4");
        pMoeUpQ8 = metal.pipeline("moe_up_q8");
        pMoeDownQ4 = metal.pipeline("moe_down_q4");
        pMoeDownQ8 = metal.pipeline("moe_down_q8");
        pKvStoreHalf = metal.pipeline("kv_store_half");
        if (c.isMoe() && c.numExperts() > 1024) {
            throw new IllegalArgumentException("Metal router kernel supports at most 1024 experts");
        }

        embedQ = metal.wrap(w.embed.data);
        lmHeadQ = w.lmHead == w.embed ? embedQ : metal.wrap(w.lmHead.data);
        layers = new LayerBufs[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            ModelWeights.Layer L = w.layers[l];
            LayerBufs b = new LayerBufs();
            b.attnNorm = upload(L.attnNorm);
            b.ffnNorm = upload(L.ffnNorm);
            b.qNorm = L.qNorm != null ? upload(L.qNorm) : null;
            b.kNorm = L.kNorm != null ? upload(L.kNorm) : null;
            b.bq = L.bq != null ? upload(L.bq) : null;
            b.bk = L.bk != null ? upload(L.bk) : null;
            b.bv = L.bv != null ? upload(L.bv) : null;
            b.wq = metal.wrap(L.wq.data);
            b.wk = metal.wrap(L.wk.data);
            b.wv = metal.wrap(L.wv.data);
            b.wo = metal.wrap(L.wo.data);
            if (L.moe) {
                b.router = upload(L.router);
                b.expGateUp = metal.wrap(L.expGateUp.data);
                b.expDown = metal.wrap(L.expDown.data);
            } else {
                b.wGateUp = metal.wrap(L.wGateUp.data);
                b.wDown = metal.wrap(L.wDown.data);
            }
            layers[l] = b;
        }
        finalNorm = upload(w.finalNorm);
        Rope rope = Rope.of(c, ctx);
        ropeCos = upload(rope.cos);
        ropeSin = upload(rope.sin);

        // None of the buffers below is zeroed: every element is written before it is read. The KV cache rows
        // [0, pos + n) are written by the K/V projections (or kv_store_half) before attention reads them,
        // activations and staging rows are written for every token of the current chunk, and the scores
        // scratch is written for positions [first, len) before the softmax reads the same range. Skipping the
        // memset keeps start-up fast and only commits pages that are actually touched (a long context would
        // otherwise clear gigabytes).
        long kvBytes = (long) ctx * c.kvDim() * kvDtype.bytes;
        keyCache = new Metal.Buffer[c.layers()];
        valCache = new Metal.Buffer[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            keyCache[l] = metal.newBuffer(kvBytes, false);
            valCache[l] = metal.newBuffer(kvBytes, false);
        }
        kStage = kvHalf ? metal.newBuffer((long) maxBatch * c.kvDim() * 4, false) : null;
        vStage = kvHalf ? metal.newBuffer((long) maxBatch * c.kvDim() * 4, false) : null;
        x = metal.newBuffer((long) maxBatch * c.dim() * 4, false);
        xb = metal.newBuffer((long) maxBatch * c.dim() * 4, false);
        q = metal.newBuffer((long) maxBatch * c.qDim() * 4, false);
        att = metal.newBuffer((long) maxBatch * c.qDim() * 4, false);
        gateUp = metal.newBuffer((long) maxBatch * 2L * c.hiddenDim() * 4, false);
        hidden = metal.newBuffer((long) maxBatch * c.hiddenDim() * 4, false);
        scores = metal.newBuffer((long) Math.min(ATT_TILE, maxBatch) * c.heads() * ctx * 4, false);
        logitsBuf = metal.newBuffer((long) w.lmHead.rows * 4, false);
        if (c.isMoe()) {
            int slots = maxBatch * c.expertsPerTok();
            assign = metal.newBuffer((long) slots * 4, false);
            routeWeight = metal.newBuffer((long) slots * 4, false);
            moeHidden = metal.newBuffer((long) slots * c.moeDim() * 4, false);
        } else {
            assign = null;
            routeWeight = null;
            moeHidden = null;
        }
        if (ctx > c.maxPositions()) {
            System.err.printf("warning: ctx %d exceeds the model's max_position_embeddings (%d); RoPE positions "
                    + "beyond that are extrapolated and quality degrades%n", ctx, c.maxPositions());
        }
        long ws = metal.recommendedWorkingSet();
        if (w.byteSize() > ws) {
            System.err.printf("warning: weights (%.1f GB) exceed Metal's recommended working set (%.1f GB); expect paging%n",
                    w.byteSize() / 1e9, ws / 1e9);
        }

        bufArgs = arena.allocate(ValueLayout.ADDRESS, MAX_BUFS);
        offArgs = arena.allocate(ValueLayout.JAVA_LONG, MAX_BUFS);
        params = arena.allocate(64);
    }

    private Metal.Buffer upload(float[] data) {
        Metal.Buffer b = metal.newBuffer((long) data.length * 4, false); // fully overwritten below
        MemorySegment.copy(data, 0, b.contents(), ValueLayout.JAVA_FLOAT, 0, data.length);
        return b;
    }

    @Override
    public String name() {
        return "metal (" + metal.deviceName() + ")";
    }

    @Override
    public int contextLength() {
        return ctx;
    }

    @Override
    public int maxBatch() {
        return maxBatch;
    }

    @Override
    public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
        if (n < 1 || n > maxBatch || pos + n > ctx) {
            throw new IllegalArgumentException("bad forward(n=" + n + ", pos=" + pos + ")");
        }
        int dim = c.dim();
        int kvDim = c.kvDim();
        // embeddings are gathered on the CPU (tiny) and written straight into the shared buffer
        float[] row = new float[dim];
        for (int t = 0; t < n; t++) {
            w.embed.dequantizeRow(tokens[off + t], row, 0);
            MemorySegment.copy(row, 0, x.contents(), ValueLayout.JAVA_FLOAT, (long) t * dim * 4, dim);
        }

        metal.begin();
        boolean committed = false;
        try {
            for (int l = 0; l < c.layers(); l++) {
                LayerBufs L = layers[l];
                ModelWeights.Layer W = w.layers[l];
                rmsnorm(x, L.attnNorm, xb, n, dim);
                matmul(W.wq, L.wq, xb, q, 0, n, false);
                // f32: k/v are written straight into the cache rows; f16: into staging, rounded by kvStore below
                Metal.Buffer kDst = kvHalf ? kStage : keyCache[l];
                Metal.Buffer vDst = kvHalf ? vStage : valCache[l];
                long kvOff = kvHalf ? 0 : (long) pos * kvDim * 4;
                matmul(W.wk, L.wk, xb, kDst, kvOff, n, false);
                matmul(W.wv, L.wv, xb, vDst, kvOff, n, false);
                if (L.bq != null) {
                    addBias(q, 0, L.bq, n, c.qDim());
                    addBias(kDst, kvOff, L.bk, n, kvDim);
                    addBias(vDst, kvOff, L.bv, n, kvDim);
                }
                normRope(L, kDst, kvOff, n, pos);
                if (kvHalf) {
                    kvStore(l, n, pos);
                }
                attention(l, n, pos);
                matmul(W.wo, L.wo, att, x, 0, n, true);

                rmsnorm(x, L.ffnNorm, xb, n, dim);
                if (W.moe) {
                    moe(W, L, n);
                } else {
                    matmul(W.wGateUp, L.wGateUp, xb, gateUp, 0, n, false);
                    siluMul(n);
                    matmul(W.wDown, L.wDown, hidden, x, 0, n, true);
                }
            }
            if (logits != null) {
                // normalize only the last token: read x at row n-1, write xb row 0
                rmsnormAt(x, (long) (n - 1) * dim * 4, finalNorm, xb, 1, dim);
                matmul(w.lmHead, lmHeadQ, xb, logitsBuf, 0, 1, false);
            }
            metal.commitAndWait();
            committed = true;
        } finally {
            if (!committed) {
                metal.abort(); // an un-ended encoder would make Metal abort the process on release
            }
        }
        if (logits != null) {
            MemorySegment.copy(logitsBuf.contents(), ValueLayout.JAVA_FLOAT, 0, logits, 0,
                    Math.min(w.lmHead.rows, logits.length));
        }
    }

    // ------------------------------------------------------------------ dispatch helpers

    private void bind(int i, Metal.Buffer b, long offset) {
        bufArgs.setAtIndex(ValueLayout.ADDRESS, i, b.handle());
        offArgs.setAtIndex(ValueLayout.JAVA_LONG, i, offset);
    }

    private void matmul(QTensor t, Metal.Buffer wbuf, Metal.Buffer in, Metal.Buffer out, long outOff, int n, boolean acc) {
        bind(0, wbuf, 0);
        bind(1, wbuf, t.scalesOffset);
        bind(2, in, 0);
        bind(3, out, outOff);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, t.rows);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, t.cols);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, n);
        params.setAtIndex(ValueLayout.JAVA_INT, 3, acc ? 1 : 0);
        int pipe = t.type == QType.Q4_0 ? pMatmulQ4 : pMatmulQ8;
        int sgPerTg = 4; // 128 threads = 4 rows per threadgroup
        int gx = (t.rows + sgPerTg - 1) / sgPerTg;
        int gy = (n + TOK_TILE - 1) / TOK_TILE;
        metal.dispatch(pipe, bufArgs, offArgs, 4, params, 16, gx, gy, 1, 32 * sgPerTg, 1, 1);
    }

    private void addBias(Metal.Buffer v, long off, Metal.Buffer bias, int n, int width) {
        bind(0, v, off);
        bind(1, bias, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, width);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, n * width);
        metal.dispatch(pAddBias, bufArgs, offArgs, 2, params, 8, (n * width + 255) / 256, 1, 1, 256, 1, 1);
    }

    private void rmsnorm(Metal.Buffer in, Metal.Buffer weight, Metal.Buffer out, int n, int width) {
        rmsnormAt(in, 0, weight, out, n, width);
    }

    private void rmsnormAt(Metal.Buffer in, long inOff, Metal.Buffer weight, Metal.Buffer out, int n, int width) {
        bind(0, in, inOff);
        bind(1, weight, 0);
        bind(2, out, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, width);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 1, c.normEps());
        metal.dispatch(pRmsnorm, bufArgs, offArgs, 3, params, 8, n, 1, 1, 256, 1, 1);
    }

    /** QK-norm + RoPE for the q rows and the k rows just written at position pos, in one dispatch. */
    private void normRope(LayerBufs L, Metal.Buffer kDst, long kvOff, int n, int pos) {
        boolean useNorm = L.qNorm != null && L.kNorm != null;
        bind(0, q, 0);
        bind(1, kDst, kvOff);
        bind(2, useNorm ? L.qNorm : finalNorm, 0); // dummy bindings when unused
        bind(3, useNorm ? L.kNorm : finalNorm, 0);
        bind(4, ropeCos, 0);
        bind(5, ropeSin, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, c.headDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 1, c.qDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 2, c.kvDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 3, c.heads());
        params.setAtIndex(ValueLayout.JAVA_INT, 4, pos);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 5, c.normEps());
        params.setAtIndex(ValueLayout.JAVA_INT, 6, useNorm ? 1 : 0);
        metal.dispatch(pNormRope, bufArgs, offArgs, 6, params, 28, c.heads() + c.kvHeads(), n, 1,
                c.headDim() / 2, 1, 1);
    }

    /** One dispatch per tile of ATT_TILE tokens; the scores scratch is indexed by the tile-local token. */
    private void attention(int l, int n, int pos) {
        bind(0, q, 0);
        bind(1, keyCache[l], 0);
        bind(2, valCache[l], 0);
        bind(3, att, 0);
        bind(4, scores, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, c.heads());
        params.setAtIndex(ValueLayout.JAVA_INT, 1, c.kvHeads());
        params.setAtIndex(ValueLayout.JAVA_INT, 2, c.headDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 3, pos);
        params.setAtIndex(ValueLayout.JAVA_INT, 4, ctx);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 5, (float) (1.0 / Math.sqrt(c.headDim())));
        params.setAtIndex(ValueLayout.JAVA_INT, 6, c.qDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 7, c.kvDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 8, c.windowFor(l));
        for (int t0 = 0; t0 < n; t0 += ATT_TILE) {
            params.setAtIndex(ValueLayout.JAVA_INT, 9, t0);
            metal.dispatch(pAttention, bufArgs, offArgs, 5, params, 40, c.heads() / attnGroup,
                    Math.min(ATT_TILE, n - t0), 1, 256, 1, 1);
        }
    }

    /** f16 cache: rounds the n staged k/v rows into rows [pos, pos+n) of layer l's half caches. */
    private void kvStore(int l, int n, int pos) {
        int total = n * c.kvDim();
        bind(0, kStage, 0);
        bind(1, vStage, 0);
        bind(2, keyCache[l], 0);
        bind(3, valCache[l], 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, n);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, pos);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, c.kvDim());
        metal.dispatch(pKvStoreHalf, bufArgs, offArgs, 4, params, 12, (total + 255) / 256, 1, 1, 256, 1, 1);
    }

    private void siluMul(int n) {
        siluMul(gateUp, hidden, n, c.hiddenDim());
    }

    private void siluMul(Metal.Buffer in, Metal.Buffer out, int ntok, int width) {
        int total = ntok * width;
        bind(0, in, 0);
        bind(1, out, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, width);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, total);
        metal.dispatch(pSilu, bufArgs, offArgs, 2, params, 8, (total + 255) / 256, 1, 1, 256, 1, 1);
    }

    /** Router → per-slot gate/up → SwiGLU → weighted down projections accumulated into x. */
    private void moe(ModelWeights.Layer W, LayerBufs L, int n) {
        int topK = c.expertsPerTok();
        int m = c.moeDim();
        int slots = n * topK;
        int sgPerTg = 4;

        bind(0, xb, 0);
        bind(1, L.router, 0);
        bind(2, assign, 0);
        bind(3, routeWeight, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, c.numExperts());
        params.setAtIndex(ValueLayout.JAVA_INT, 1, c.dim());
        params.setAtIndex(ValueLayout.JAVA_INT, 2, topK);
        params.setAtIndex(ValueLayout.JAVA_INT, 3, c.normTopK() ? 1 : 0);
        metal.dispatch(pRouter, bufArgs, offArgs, 4, params, 16, n, 1, 1, 256, 1, 1);

        // gate/up projections with SwiGLU fused in: one simdgroup per (slot, hidden unit) writes moeHidden
        QTensor gu = W.expGateUp;
        bind(0, L.expGateUp, 0);
        bind(1, L.expGateUp, gu.scalesOffset);
        bind(2, xb, 0);
        bind(3, assign, 0);
        bind(4, moeHidden, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, 2 * m); // rows per expert in the stacked tensor (gate | up)
        params.setAtIndex(ValueLayout.JAVA_INT, 1, gu.cols);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, topK);
        metal.dispatch(gu.type == QType.Q4_0 ? pMoeUpQ4 : pMoeUpQ8, bufArgs, offArgs, 5, params, 12,
                (m + sgPerTg - 1) / sgPerTg, slots, 1, 32 * sgPerTg, 1, 1);

        QTensor dn = W.expDown;
        bind(0, L.expDown, 0);
        bind(1, L.expDown, dn.scalesOffset);
        bind(2, moeHidden, 0);
        bind(3, assign, 0);
        bind(4, routeWeight, 0);
        bind(5, x, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, c.dim());
        params.setAtIndex(ValueLayout.JAVA_INT, 1, dn.cols);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, topK);
        metal.dispatch(dn.type == QType.Q4_0 ? pMoeDownQ4 : pMoeDownQ8, bufArgs, offArgs, 6, params, 12,
                (c.dim() + sgPerTg - 1) / sgPerTg, n, 1, 32 * sgPerTg, 1, 1);
    }

    @Override
    public void close() {
        metal.close();
        arena.close();
    }
}
