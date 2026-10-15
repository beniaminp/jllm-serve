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
    private static final int MAX_BUFS = 16; // matmul3 binds 11 buffers

    private final ModelWeights w;
    private final ModelConfig c;
    private final int ctx;
    private final int maxBatch;
    final Metal metal;
    private final Arena arena = Arena.ofShared();

    private int pMatmulQ4;
    private int pMatmulQ8;
    private int pMatmulQ4Norm;
    private int pMatmulQ8Norm;
    private int pMatmul3Q4;
    private int pMatmul3Q8;
    private int pMatmul3Q4Norm;
    private int pMatmul3Q8Norm;
    private int pSwigluQ4;
    private int pSwigluQ8;
    private int pSwigluQ4Norm;
    private int pSwigluQ8Norm;
    private int pRmsnorm;
    private int pNormRope;
    private int pNormRopeIl;  // GLM: interleaved partial rope, -1 otherwise
    private int pRmsnormAdd;  // Glm4: post-norms, -1 otherwise
    private int pAttention;
    private int attnGroup; // query heads per attention threadgroup
    private int pAddBias;
    private int pSoftcap;     // Gemma 2 final-logit cap, -1 otherwise
    private int pRouter;
    private int pMoeUpQ4;
    private int pMoeUpQ8;
    private int pMoeDownQ4;
    private int pMoeDownQ8;
    // Q2_K variants of every quantized matmul (see the Q2_K section of kernels.metal)
    private int pMatmulQ2K;
    private int pMatmulQ2KNorm;
    private int pMatmul3Q2K;
    private int pMatmul3Q2KNorm;
    private int pSwigluQ2K;
    private int pSwigluQ2KNorm;
    private int pMoeUpQ2K;
    private int pMoeDownQ2K;
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
    Metal.Buffer ropeCos;
    Metal.Buffer ropeSin;
    private Metal.Buffer ropeCosLocal; // Gemma 3 sliding layers; the same buffers as above when there is no local table
    private Metal.Buffer ropeSinLocal;
    private Metal.Buffer[] keyCache;
    private Metal.Buffer[] valCache;

    Metal.Buffer x;
    Metal.Buffer xb;
    Metal.Buffer q;
    Metal.Buffer att;
    Metal.Buffer hidden;
    Metal.Buffer scores;
    private Metal.Buffer logitsBuf;
    // MoE (null for dense models)
    Metal.Buffer assign;
    Metal.Buffer routeWeight;
    private Metal.Buffer moeHidden;

    // dispatch argument scratch (re-used; the native side copies params at record time)
    MemorySegment bufArgs;
    MemorySegment offArgs;
    MemorySegment params;
    /** DeepSeek multi-head latent attention, router and shared experts (own caches); null for other families. */
    private MlaMetal mla;

    private static final class LayerBufs {
        Metal.Buffer attnNorm;
        Metal.Buffer ffnNorm;
        Metal.Buffer postAttnNorm;
        Metal.Buffer postMlpNorm;
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
        if (c.deepseek() != null && kvDtype != KvDtype.F32) {
            throw new IllegalArgumentException("backend.kv-dtype=" + kvDtype + " is not supported for DeepSeek MLA models (use f32)");
        }
        pMatmulQ4 = metal.pipeline("matmul_q4");
        pMatmulQ8 = metal.pipeline("matmul_q8");
        pMatmulQ4Norm = metal.pipeline("matmul_q4_norm");
        pMatmulQ8Norm = metal.pipeline("matmul_q8_norm");
        pMatmul3Q4 = metal.pipeline("matmul3_q4");
        pMatmul3Q8 = metal.pipeline("matmul3_q8");
        pMatmul3Q4Norm = metal.pipeline("matmul3_q4_norm");
        pMatmul3Q8Norm = metal.pipeline("matmul3_q8_norm");
        pSwigluQ4 = metal.pipeline("matmul_swiglu_q4");
        pSwigluQ8 = metal.pipeline("matmul_swiglu_q8");
        pSwigluQ4Norm = metal.pipeline("matmul_swiglu_q4_norm");
        pSwigluQ8Norm = metal.pipeline("matmul_swiglu_q8_norm");
        pRmsnorm = metal.pipeline("rmsnorm");
        pNormRope = metal.pipeline("norm_rope");
        pNormRopeIl = c.ropeInterleaved() || c.ropePartialHalf() ? metal.pipeline("norm_rope_il") : -1;
        pRmsnormAdd = c.postNorms() ? metal.pipeline("rmsnorm_add") : -1;
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
        pAddBias = metal.pipeline("add_bias");
        pSoftcap = c.finalSoftcap() > 0 ? metal.pipeline("softcap") : -1;
        pRouter = metal.pipeline("router");
        pMoeUpQ4 = metal.pipeline("moe_up_q4");
        pMoeUpQ8 = metal.pipeline("moe_up_q8");
        pMoeDownQ4 = metal.pipeline("moe_down_q4");
        pMoeDownQ8 = metal.pipeline("moe_down_q8");
        pMatmulQ2K = metal.pipeline("matmul_q2k");
        pMatmulQ2KNorm = metal.pipeline("matmul_q2k_norm");
        pMatmul3Q2K = metal.pipeline("matmul3_q2k");
        pMatmul3Q2KNorm = metal.pipeline("matmul3_q2k_norm");
        pSwigluQ2K = metal.pipeline("matmul_swiglu_q2k");
        pSwigluQ2KNorm = metal.pipeline("matmul_swiglu_q2k_norm");
        pMoeUpQ2K = metal.pipeline("moe_up_q2k");
        pMoeDownQ2K = metal.pipeline("moe_down_q2k");
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
            b.postAttnNorm = L.postAttnNorm != null ? upload(L.postAttnNorm) : null;
            b.postMlpNorm = L.postMlpNorm != null ? upload(L.postMlpNorm) : null;
            b.qNorm = L.qNorm != null ? upload(L.qNorm) : null;
            b.kNorm = L.kNorm != null ? upload(L.kNorm) : null;
            b.bq = L.bq != null ? upload(L.bq) : null;
            b.bk = L.bk != null ? upload(L.bk) : null;
            b.bv = L.bv != null ? upload(L.bv) : null;
            if (L.mla == null) {
                b.wq = metal.wrap(L.wq.data);
                b.wk = metal.wrap(L.wk.data);
                b.wv = metal.wrap(L.wv.data);
            }
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
        boolean localRope = false;
        for (int l = 0; l < c.layers(); l++) {
            localRope |= c.usesLocalRope(l);
        }
        if (localRope) {
            Rope local = Rope.local(c, ctx);
            ropeCosLocal = upload(local.cos);
            ropeSinLocal = upload(local.sin);
        } else {
            ropeCosLocal = ropeCos;
            ropeSinLocal = ropeSin;
        }

        // None of the buffers below is zeroed: every element is written before it is read. The KV cache rows
        // [0, pos + n) are written by the K/V projections (or kv_store_half) before attention reads them,
        // activations and staging rows are written for every token of the current chunk, and the scores
        // scratch is written for positions [first, len) before the softmax reads the same range. Skipping the
        // memset keeps start-up fast and only commits pages that are actually touched (a long context would
        // otherwise clear gigabytes).
        long kvBytes = (long) ctx * c.kvDim() * kvDtype.bytes;
        keyCache = new Metal.Buffer[c.layers()];
        valCache = new Metal.Buffer[c.layers()];
        for (int l = 0; l < c.layers() && c.deepseek() == null; l++) { // DeepSeek: MlaMetal owns its caches
            keyCache[l] = metal.newBuffer(kvBytes, false);
            valCache[l] = metal.newBuffer(kvBytes, false);
        }
        kStage = kvHalf ? metal.newBuffer((long) maxBatch * c.kvDim() * 4, false) : null;
        vStage = kvHalf ? metal.newBuffer((long) maxBatch * c.kvDim() * 4, false) : null;
        x = metal.newBuffer((long) maxBatch * c.dim() * 4, false);
        xb = metal.newBuffer((long) maxBatch * c.dim() * 4, false);
        q = metal.newBuffer((long) maxBatch * c.qDim() * 4, false);
        att = metal.newBuffer((long) maxBatch * (c.deepseek() == null ? c.qDim()
                : Math.max(c.qDim(), c.heads() * c.deepseek().vHeadDim())) * 4, false);
        hidden = metal.newBuffer((long) maxBatch * c.maxMlpDim() * 4, false);
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
        mla = c.deepseek() != null ? new MlaMetal(this, w, ctx, maxBatch) : null;
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

    Metal.Buffer upload(float[] data) {
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
            if (c.embedScale() != 1f) {
                for (int i = 0; i < dim; i++) {
                    row[i] *= c.embedScale();
                }
            }
            MemorySegment.copy(row, 0, x.contents(), ValueLayout.JAVA_FLOAT, (long) t * dim * 4, dim);
        }

        metal.begin();
        boolean committed = false;
        try {
            // For a single token the rmsnorm preceding a projection is folded into the projection kernel
            // (the *_norm variants read x and normalise on the fly); otherwise it writes xb first.
            boolean fuseNorm = n == 1;
            for (int l = 0; l < c.layers(); l++) {
                LayerBufs L = layers[l];
                ModelWeights.Layer W = w.layers[l];
                if (mla != null) {
                    rmsnorm(x, L.attnNorm, xb, n, dim);
                    mla.run(l, W, n, pos);
                } else {
                    // f32: k/v are written straight into the cache rows; f16: into staging, rounded by kvStore below
                    Metal.Buffer kDst = kvHalf ? kStage : keyCache[l];
                    Metal.Buffer vDst = kvHalf ? vStage : valCache[l];
                    long kvOff = kvHalf ? 0 : (long) pos * kvDim * 4;
                    if (fuseNorm) {
                        qkv(W, L, x, L.attnNorm, kDst, vDst, kvOff, n);
                    } else {
                        rmsnorm(x, L.attnNorm, xb, n, dim);
                        qkv(W, L, xb, null, kDst, vDst, kvOff, n);
                    }
                    if (L.bq != null) {
                        addBias(q, 0, L.bq, n, c.qDim());
                        addBias(kDst, kvOff, L.bk, n, kvDim);
                        addBias(vDst, kvOff, L.bv, n, kvDim);
                    }
                    normRope(L, kDst, kvOff, n, pos, c.usesLocalRope(l));
                    if (kvHalf) {
                        kvStore(l, n, pos);
                    }
                    attention(l, n, pos);
                }
                if (L.postAttnNorm != null) {
                    // Glm4: x += rmsnorm(o_proj(attention)), so o_proj cannot accumulate into x directly
                    matmul(W.wo, L.wo, att, 0, xb, 0, n, false, null);
                    rmsnormAdd(xb, L.postAttnNorm, x, n, dim);
                } else {
                    matmul(W.wo, L.wo, att, 0, x, 0, n, true, null);
                }

                if (W.moe) {
                    rmsnorm(x, L.ffnNorm, xb, n, dim);
                    moe(W, L, l, n);
                    if (mla != null) {
                        mla.sharedExperts(W, l, n);
                    }
                } else {
                    if (fuseNorm) {
                        swiglu(W, L, x, L.ffnNorm, n);
                    } else {
                        rmsnorm(x, L.ffnNorm, xb, n, dim);
                        swiglu(W, L, xb, null, n);
                    }
                    if (L.postMlpNorm != null) {
                        matmul(W.wDown, L.wDown, hidden, 0, xb, 0, n, false, null);
                        rmsnormAdd(xb, L.postMlpNorm, x, n, dim);
                    } else {
                        matmul(W.wDown, L.wDown, hidden, 0, x, 0, n, true, null);
                    }
                }
            }
            if (logits != null) {
                // normalise only the last token (read x at row n-1, write xb row 0), then the LM head. Folding
                // the norm into the LM-head kernel measured ~5% slower at decode: its prologue runs in all
                // ~38k threadgroups, which costs more than the one dispatch it saves.
                rmsnormAt(x, (long) (n - 1) * dim * 4, finalNorm, xb, 1, dim);
                matmul(w.lmHead, lmHeadQ, xb, 0, logitsBuf, 0, 1, false, null);
                if (pSoftcap >= 0) {
                    bind(0, logitsBuf, 0);
                    params.setAtIndex(ValueLayout.JAVA_INT, 0, w.lmHead.rows);
                    params.setAtIndex(ValueLayout.JAVA_FLOAT, 1, c.finalSoftcap());
                    metal.dispatch(pSoftcap, bufArgs, offArgs, 1, params, 8, (w.lmHead.rows + 255) / 256, 1, 1, 256, 1, 1);
                }
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

    void bind(int i, Metal.Buffer b, long offset) {
        bufArgs.setAtIndex(ValueLayout.ADDRESS, i, b.handle());
        offArgs.setAtIndex(ValueLayout.JAVA_LONG, i, offset);
    }

    private static final int SG_PER_TG = 4; // 128 threads = 4 rows per matmul threadgroup (kernels assume it)

    /**
     * out[t, row] (+)= W[row] . in[t]. With {@code norm != null} (single token only) the kernel first
     * applies rmsnorm with those weights to the input row, exactly as the rmsnorm kernel would.
     */
    void matmul(QTensor t, Metal.Buffer wbuf, Metal.Buffer in, long inOff, Metal.Buffer out, long outOff,
                        int n, boolean acc, Metal.Buffer norm) {
        if (norm != null && n != 1) {
            throw new IllegalArgumentException("fused rmsnorm needs a single token");
        }
        bind(0, wbuf, 0);
        bind(1, wbuf, scaleOff(t));
        bind(2, in, inOff);
        bind(3, out, outOff);
        bind(4, norm != null ? norm : finalNorm, 0); // dummy binding when unused
        params.setAtIndex(ValueLayout.JAVA_INT, 0, t.rows);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, t.cols);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, n);
        params.setAtIndex(ValueLayout.JAVA_INT, 3, acc ? 1 : 0);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 4, c.normEps());
        int pipe = norm != null ? byType(t.type, pMatmulQ4Norm, pMatmulQ8Norm, pMatmulQ2KNorm)
                                : byType(t.type, pMatmulQ4, pMatmulQ8, pMatmulQ2K);
        int gx = (t.rows + SG_PER_TG - 1) / SG_PER_TG;
        int gy = (n + TOK_TILE - 1) / TOK_TILE;
        metal.dispatch(pipe, bufArgs, offArgs, 5, params, 20, normTgMem(norm, t.cols), gx, gy, 1, 32 * SG_PER_TG, 1, 1);
    }

    /** Dynamic threadgroup memory for the normalised input row of the *_norm kernels (0 when unused). */
    private static int normTgMem(Metal.Buffer norm, int cols) {
        return norm != null ? cols * 4 : 0;
    }

    /** Offset of the f32 scale region to bind as buffer(1); Q2_K keeps its scales inside the blocks (unused binding). */
    static long scaleOff(QTensor t) {
        return t.type.scaleBytes == 0 ? 0 : t.scalesOffset;
    }

    /** The pipeline variant for a tensor's quantization type. */
    static int byType(QType type, int q4, int q8, int q2k) {
        return switch (type) {
            case Q4_0 -> q4;
            case Q8_0 -> q8;
            case Q2_K -> q2k;
        };
    }

    /** q, k and v projections of {@code in} in one dispatch (k and v straight into the caches at pos). */
    private void qkv(ModelWeights.Layer W, LayerBufs L, Metal.Buffer in, Metal.Buffer norm,
                     Metal.Buffer kDst, Metal.Buffer vDst, long kvOff, int n) {
        if (W.wq.type != W.wk.type || W.wk.type != W.wv.type || W.wq.cols != W.wk.cols || W.wk.cols != W.wv.cols) {
            matmul(W.wq, L.wq, in, 0, q, 0, n, false, norm);
            matmul(W.wk, L.wk, in, 0, kDst, kvOff, n, false, norm);
            matmul(W.wv, L.wv, in, 0, vDst, kvOff, n, false, norm);
            return;
        }
        bind(0, L.wq, 0);
        bind(1, L.wq, scaleOff(W.wq));
        bind(2, L.wk, 0);
        bind(3, L.wk, scaleOff(W.wk));
        bind(4, L.wv, 0);
        bind(5, L.wv, scaleOff(W.wv));
        bind(6, in, 0);
        bind(7, q, 0);
        bind(8, kDst, kvOff);
        bind(9, vDst, kvOff);
        bind(10, norm != null ? norm : finalNorm, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, W.wq.rows);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, W.wk.rows);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, W.wv.rows);
        params.setAtIndex(ValueLayout.JAVA_INT, 3, W.wq.cols);
        params.setAtIndex(ValueLayout.JAVA_INT, 4, n);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 5, c.normEps());
        int pipe = norm != null ? byType(W.wq.type, pMatmul3Q4Norm, pMatmul3Q8Norm, pMatmul3Q2KNorm)
                                : byType(W.wq.type, pMatmul3Q4, pMatmul3Q8, pMatmul3Q2K);
        int rows = W.wq.rows + W.wk.rows + W.wv.rows;
        int gx = (rows + SG_PER_TG - 1) / SG_PER_TG;
        int gy = (n + TOK_TILE - 1) / TOK_TILE;
        metal.dispatch(pipe, bufArgs, offArgs, 11, params, 24, normTgMem(norm, W.wq.cols), gx, gy, 1,
                32 * SG_PER_TG, 1, 1);
    }

    /** hidden[t] = silu(Wgate . in[t]) * (Wup . in[t]) from the stacked [2 * hidden, dim] gate|up matrix. */
    private void swiglu(ModelWeights.Layer W, LayerBufs L, Metal.Buffer in, Metal.Buffer norm, int n) {
        swiglu(W.wGateUp, L.wGateUp, in, norm, n);
    }

    /** Same for any stacked gate|up tensor {@code t} (also the DeepSeek shared experts); writes {@code hidden}. */
    void swiglu(QTensor t, Metal.Buffer wbuf, Metal.Buffer in, Metal.Buffer norm, int n) {
        int hiddenDim = t.rows / 2;
        bind(0, wbuf, 0);
        bind(1, wbuf, scaleOff(t));
        bind(2, in, 0);
        bind(3, hidden, 0);
        bind(4, norm != null ? norm : finalNorm, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, hiddenDim);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, t.cols);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, n);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 3, c.normEps());
        params.setAtIndex(ValueLayout.JAVA_INT, 4, c.isGemma() ? 1 : 0);
        int pipe = norm != null ? byType(t.type, pSwigluQ4Norm, pSwigluQ8Norm, pSwigluQ2KNorm)
                                : byType(t.type, pSwigluQ4, pSwigluQ8, pSwigluQ2K);
        int gx = (hiddenDim + SG_PER_TG - 1) / SG_PER_TG;
        int gy = (n + TOK_TILE - 1) / TOK_TILE;
        metal.dispatch(pipe, bufArgs, offArgs, 5, params, 20, normTgMem(norm, t.cols), gx, gy, 1,
                32 * SG_PER_TG, 1, 1);
    }

    private void addBias(Metal.Buffer v, long off, Metal.Buffer bias, int n, int width) {
        bind(0, v, off);
        bind(1, bias, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, width);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, n * width);
        metal.dispatch(pAddBias, bufArgs, offArgs, 2, params, 8, (n * width + 255) / 256, 1, 1, 256, 1, 1);
    }

    void rmsnorm(Metal.Buffer in, Metal.Buffer weight, Metal.Buffer out, int n, int width) {
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

    /** out[t] += rmsnorm(y[t]) * weight for n rows of the given width. */
    private void rmsnormAdd(Metal.Buffer y, Metal.Buffer weight, Metal.Buffer out, int n, int width) {
        bind(0, y, 0);
        bind(1, weight, 0);
        bind(2, out, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, width);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 1, c.normEps());
        metal.dispatch(pRmsnormAdd, bufArgs, offArgs, 3, params, 8, n, 1, 1, 256, 1, 1);
    }

    /** GLM interleaved pairs, or rotate-half pairs for partial rotary (Phi-4-mini): the first rotaryDim dims of every q and k head (no qk-norm). */
    private void normRopeInterleaved(Metal.Buffer kDst, long kvOff, int n, int pos) {
        bind(0, q, 0);
        bind(1, kDst, kvOff);
        bind(2, ropeCos, 0);
        bind(3, ropeSin, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, c.headDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 1, c.qDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 2, c.kvDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 3, c.heads());
        params.setAtIndex(ValueLayout.JAVA_INT, 4, pos);
        params.setAtIndex(ValueLayout.JAVA_INT, 5, c.rotaryDim() / 2);
        params.setAtIndex(ValueLayout.JAVA_INT, 6, c.ropeInterleaved() ? 0 : 1);
        metal.dispatch(pNormRopeIl, bufArgs, offArgs, 4, params, 28, c.heads() + c.kvHeads(), n, 1,
                c.rotaryDim() / 2, 1, 1);
    }

    /** QK-norm + RoPE for the q rows and the k rows just written at position pos, in one dispatch. */
    private void normRope(LayerBufs L, Metal.Buffer kDst, long kvOff, int n, int pos, boolean localRope) {
        if (c.ropeInterleaved() || c.ropePartialHalf()) {
            if (L.qNorm != null) {
                throw new IllegalStateException("partial rotary embedding combined with q/k norm is not implemented");
            }
            normRopeInterleaved(kDst, kvOff, n, pos);
            return;
        }
        boolean useNorm = L.qNorm != null && L.kNorm != null;
        bind(0, q, 0);
        bind(1, kDst, kvOff);
        bind(2, useNorm ? L.qNorm : finalNorm, 0); // dummy bindings when unused
        bind(3, useNorm ? L.kNorm : finalNorm, 0);
        bind(4, localRope ? ropeCosLocal : ropeCos, 0);
        bind(5, localRope ? ropeSinLocal : ropeSin, 0);
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
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 5, c.attnScale());
        params.setAtIndex(ValueLayout.JAVA_INT, 6, c.qDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 7, c.kvDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 8, c.windowFor(l));
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 10, c.attnSoftcap());
        for (int t0 = 0; t0 < n; t0 += ATT_TILE) {
            params.setAtIndex(ValueLayout.JAVA_INT, 9, t0);
            metal.dispatch(pAttention, bufArgs, offArgs, 5, params, 44, c.heads() / attnGroup,
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

    /** Router → per-slot gate/up → SwiGLU → weighted down projections accumulated into x. */
    private void moe(ModelWeights.Layer W, LayerBufs L, int l, int n) {
        int topK = c.expertsPerTok();
        int m = c.moeDim();
        int slots = n * topK;
        int sgPerTg = 4;

        if (mla != null) {
            mla.router(l, L.router, n); // DeepSeek rule; fills assign / routeWeight like `router`
        } else {
            bind(0, xb, 0);
            bind(1, L.router, 0);
            bind(2, assign, 0);
            bind(3, routeWeight, 0);
            params.setAtIndex(ValueLayout.JAVA_INT, 0, c.numExperts());
            params.setAtIndex(ValueLayout.JAVA_INT, 1, c.dim());
            params.setAtIndex(ValueLayout.JAVA_INT, 2, topK);
            params.setAtIndex(ValueLayout.JAVA_INT, 3, c.normTopK() ? 1 : 0);
            metal.dispatch(pRouter, bufArgs, offArgs, 4, params, 16, n, 1, 1, 256, 1, 1);
        }

        // gate/up projections with SwiGLU fused in: one simdgroup per (slot, hidden unit) writes moeHidden
        QTensor gu = W.expGateUp;
        bind(0, L.expGateUp, 0);
        bind(1, L.expGateUp, scaleOff(gu));
        bind(2, xb, 0);
        bind(3, assign, 0);
        bind(4, moeHidden, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, 2 * m); // rows per expert in the stacked tensor (gate | up)
        params.setAtIndex(ValueLayout.JAVA_INT, 1, gu.cols);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, topK);
        metal.dispatch(byType(gu.type, pMoeUpQ4, pMoeUpQ8, pMoeUpQ2K), bufArgs, offArgs, 5, params, 12,
                (m + sgPerTg - 1) / sgPerTg, slots, 1, 32 * sgPerTg, 1, 1);

        QTensor dn = W.expDown;
        bind(0, L.expDown, 0);
        bind(1, L.expDown, scaleOff(dn));
        bind(2, moeHidden, 0);
        bind(3, assign, 0);
        bind(4, routeWeight, 0);
        bind(5, x, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, c.dim());
        params.setAtIndex(ValueLayout.JAVA_INT, 1, dn.cols);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, topK);
        metal.dispatch(byType(dn.type, pMoeDownQ4, pMoeDownQ8, pMoeDownQ2K), bufArgs, offArgs, 6, params, 12,
                (c.dim() + sgPerTg - 1) / sgPerTg, n, 1, 32 * sgPerTg, 1, 1);
    }

    @Override
    public void close() {
        metal.close();
        arena.close();
    }
}
