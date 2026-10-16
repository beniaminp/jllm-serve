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
    /** Rows (matmul_swiglu_pf: units) and tokens per simdgroup of the *_pf prefill matmuls; must match kernels.metal. */
    static final int PF_ROWS = 2;
    static final int PF_UNITS = 1;
    static final int PF_TOKS = 8;
    /**
     * Chunks of at least this many tokens use the tiled prefill matmuls (bit-identical results, see the prefill
     * section of kernels.metal); smaller ones the single-row kernels. A huge value disables the tiled kernels.
     */
    static final int PREFILL_MIN_TOKENS = Integer.getInteger("jllm.metal.prefillMinTokens", 8);
    /**
     * Flash decoding (see the "flash decoding" section of kernels.metal): a decode step over at least
     * FLASH_MIN_LEN cached positions splits every head's key range across up to FLASH_MAX_SPLITS threadgroups of
     * FLASH_CHUNK keys and combines the partial softmaxes. Opt-in because the combine sums in a different order
     * from the single-pass kernel (results differ at the 1e-6 level, not bit-identical).
     */
    static final boolean FLASH_DECODE = Boolean.getBoolean("jllm.metal.flashDecode");
    static final int FLASH_MIN_LEN = Integer.getInteger("jllm.metal.flashMinLen", 512);
    static final int FLASH_CHUNK = Integer.getInteger("jllm.metal.flashChunk", 128); // 64-128 measured best, 256+ slower
    static final int FLASH_MAX_SPLITS = 32; // the combine kernel holds 64 weights
    /**
     * Tokens per attention dispatch; the scores scratch holds one row of ctx floats per (tile token, head)
     * instead of one per (batch token, head). 32 keeps a 64-token prefill chunk at two dispatches per layer.
     */
    private static final int ATT_TILE = 32;
    private static final int MAX_BUFS = 16; // matmul3 binds 14 buffers
    /**
     * Layers per command buffer: the dispatches of every COMMIT_LAYERS layers are submitted (without waiting)
     * while the following layers are still being encoded, so GPU execution overlaps the CPU-side encoding
     * instead of starting only after the whole pass is recorded. 0 = one command buffer per forward pass.
     */
    static final int COMMIT_LAYERS = Integer.getInteger("jllm.metal.commitLayers", 8);

    private final ModelWeights w;
    private final ModelConfig c;
    private final int ctx;
    private final int maxBatch;
    final Metal metal;
    private final Arena arena = Arena.ofShared();

    // Kernels are looked up by name at dispatch time (Metal.pipeline memoizes and compiles on first use), so
    // only the variants a model runs are built; the names that depend on the model are fixed here.
    private String attnKernel;     // attention variant for headDim / attnGroup / cache type
    private int attnGroup;         // query heads per attention threadgroup
    private final boolean flashDecode; // M6 flash decoding enabled for this model (dense attention only)
    private Metal.Buffer flashPart;
    // f16 KV cache (backend.kv-dtype=f16): matmul3's *_h variants write v straight into the half cache, k goes
    // through an f32 staging row and rope_h rounds it into the cache after the rotation, attention_h
    // reads half; see the "f16 KV cache" section of kernels.metal
    private final boolean kvHalf;
    private final String vh;       // "_h" kernel-name suffix for the half-cache variants, "" otherwise
    /**
     * Whether the *_norm matmul variants may be used: they keep the normalised input row (dim floats) in dynamic
     * threadgroup memory next to their 32-byte reduction scratch, which must fit the device's threadgroup memory
     * limit (32 KB: dim <= 8176). Otherwise a decode step runs rmsnorm + plain matmul like a prefill chunk.
     */
    private boolean fusedNormFits;
    private Metal.Buffer kStage;

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
        // the grouped attention instantiations (G = 2, 4, 8) are only compiled when the override asks for them
        return Metal.create(metalLib, attnGroupOverride(c) > 1 ? "#define JLLM_ATTN_GROUPS 1\n" : "");
    }

    /**
     * Query heads per attention threadgroup. Sharing K/V rows across the group (> 1) measured no faster on
     * Qwen3-0.6B (G=2) and slower on Qwen3-30B-A3B (G=4/8: 20-22 tok/s vs 25 with G=1) because only kvHeads
     * threadgroups run per decode token and each one's serial work grows with the group, so the default is 1;
     * -Djllm.metal.attnGroup=2|4|8 overrides (it must divide heads / kvHeads; headDim 64, 128 or 256 only).
     */
    private static int attnGroupOverride(ModelConfig c) {
        int group = c.heads() / c.kvHeads();
        int override = Integer.getInteger("jllm.metal.attnGroup", 0);
        int hd = c.headDim();
        boolean specialised = hd == 64 || hd == 128 || hd == 256;
        return specialised && override > 0 && override <= 8 && Integer.bitCount(override) == 1 && group % override == 0
                ? override : 1;
    }

    /** Owns {@code metal} from here on: it is closed again if anything below throws. */
    private MetalBackend(ModelWeights w, int ctx, int maxBatch, Metal metal, KvDtype kvDtype) {
        this.w = w;
        this.c = w.cfg;
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        this.kvHalf = kvDtype == KvDtype.F16;
        this.vh = kvHalf ? "_h" : "";
        this.flashDecode = FLASH_DECODE && c.deepseek() == null;
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
        // Attention specialised on headDim (64 / 128 / 256) and the group size, see attnGroupOverride
        attnGroup = attnGroupOverride(c);
        int hd = c.headDim();
        attnKernel = hd == 64 || hd == 128 || hd == 256 ? "attention_g" + attnGroup + "_d" + hd + vh : "attention" + vh;
        if (c.isMoe() && c.numExperts() > 1024) {
            throw new IllegalArgumentException("Metal router kernel supports at most 1024 experts");
        }
        fusedNormFits = (long) c.dim() * 4 + NORM_STATIC_TG_BYTES <= metal.maxThreadgroupMemory();

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
        kStage = kvHalf ? metal.newBuffer((long) maxBatch * c.kvDim() * 4, false) : null; // f32 k rows before RoPE
        x = metal.newBuffer((long) maxBatch * c.dim() * 4, false);
        xb = metal.newBuffer((long) maxBatch * c.dim() * 4, false);
        q = metal.newBuffer((long) maxBatch * c.qDim() * 4, false);
        att = metal.newBuffer((long) maxBatch * (c.deepseek() == null ? c.qDim()
                : Math.max(c.qDim(), c.heads() * c.deepseek().vHeadDim())) * 4, false);
        hidden = metal.newBuffer((long) maxBatch * c.maxMlpDim() * 4, false);
        scores = metal.newBuffer((long) Math.min(ATT_TILE, maxBatch) * c.heads() * ctx * 4, false);
        logitsBuf = metal.newBuffer((long) w.lmHead.rows * 4, false);
        flashPart = flashDecode ? metal.newBuffer((long) c.heads() * FLASH_MAX_SPLITS * (c.headDim() + 2) * 4, false) : null;
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
            boolean fuseNorm = n == 1 && fusedNormFits;
            for (int l = 0; l < c.layers(); l++) {
                LayerBufs L = layers[l];
                ModelWeights.Layer W = w.layers[l];
                if (mla != null) {
                    rmsnorm(x, L.attnNorm, xb, n, dim);
                    mla.run(l, W, n, pos);
                } else {
                    // v is written straight into the cache rows at pos (as half for the f16 cache); k too for
                    // the f32 cache, while the f16 cache stages the f32 k rows and the rope kernel rounds them in
                    long cacheOff = (long) pos * kvDim * (kvHalf ? 2 : 4);
                    Metal.Buffer kDst = kvHalf ? kStage : keyCache[l];
                    long kOff = kvHalf ? 0 : cacheOff;
                    if (fuseNorm) {
                        qkv(W, L, x, L.attnNorm, kDst, kOff, valCache[l], cacheOff, n);
                    } else {
                        rmsnorm(x, L.attnNorm, xb, n, dim);
                        qkv(W, L, xb, null, kDst, kOff, valCache[l], cacheOff, n);
                    }
                    normRope(L, kDst, kOff, keyCache[l], cacheOff, n, pos, c.usesLocalRope(l));
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
                if (COMMIT_LAYERS > 0 && (l + 1) % COMMIT_LAYERS == 0 && l + 1 < c.layers()) {
                    metal.commit(); // the GPU starts on these layers while the next ones are encoded
                }
            }
            if (logits != null) {
                // normalise only the last token (read x at row n-1, write xb row 0), then the LM head. Folding
                // the norm into the LM-head kernel measured ~5% slower at decode: its prologue runs in all
                // ~38k threadgroups, which costs more than the one dispatch it saves.
                rmsnormAt(x, (long) (n - 1) * dim * 4, finalNorm, xb, 1, dim);
                matmul(w.lmHead, lmHeadQ, xb, 0, logitsBuf, 0, 1, false, null, c.finalSoftcap()); // Gemma: capped logits
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
    /** Static threadgroup memory of the *_norm matmul kernels (their `threadgroup float buf[8]`). */
    private static final int NORM_STATIC_TG_BYTES = 32;

    /**
     * out[t, row] (+)= W[row] . in[t]. With {@code norm != null} (single token only) the kernel first
     * applies rmsnorm with those weights to the input row, exactly as the rmsnorm kernel would.
     */
    void matmul(QTensor t, Metal.Buffer wbuf, Metal.Buffer in, long inOff, Metal.Buffer out, long outOff,
                int n, boolean acc, Metal.Buffer norm) {
        matmul(t, wbuf, in, inOff, out, outOff, n, acc, norm, 0f);
    }

    /** As above; {@code softcap > 0} applies tanh(out / softcap) * softcap in the epilogue (not with acc). */
    private void matmul(QTensor t, Metal.Buffer wbuf, Metal.Buffer in, long inOff, Metal.Buffer out, long outOff,
                        int n, boolean acc, Metal.Buffer norm, float softcap) {
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
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 5, softcap);
        if (norm == null && n >= PREFILL_MIN_TOKENS) {
            int pipe = metal.pipeline(kernel("matmul_pf", t.type, ""));
            metal.dispatch(pipe, bufArgs, offArgs, 5, params, 24, pfGrid(t.rows), pfTiles(n), 1, 32 * SG_PER_TG, 1, 1);
            return;
        }
        int pipe = metal.pipeline(kernel("matmul", t.type, norm != null ? "_norm" : ""));
        int gx = (t.rows + SG_PER_TG - 1) / SG_PER_TG;
        int gy = (n + TOK_TILE - 1) / TOK_TILE;
        metal.dispatch(pipe, bufArgs, offArgs, 5, params, 24, normTgMem(norm, t.cols), gx, gy, 1, 32 * SG_PER_TG, 1, 1);
    }

    /** Threadgroups along the row axis for the tiled prefill matmuls (PF_ROWS rows per simdgroup). */
    static int pfGrid(int rows) {
        return (rows + SG_PER_TG * PF_ROWS - 1) / (SG_PER_TG * PF_ROWS);
    }

    /** The same for matmul_swiglu_pf (PF_UNITS gate/up units per simdgroup). */
    private static int pfUnitGrid(int units) {
        return (units + SG_PER_TG * PF_UNITS - 1) / (SG_PER_TG * PF_UNITS);
    }

    static int pfTiles(int n) {
        return (n + PF_TOKS - 1) / PF_TOKS;
    }

    /**
     * Length bound to the matmul kernels' [[threadgroup(0)]] argument when the instantiation does not use it:
     * the plain (non-NORM) variants still declare the argument, and Metal requires every declared threadgroup
     * argument to have a length set before the dispatch.
     */
    static final int UNUSED_TG_BYTES = 16;

    /** Dynamic threadgroup memory for the normalised input row of the *_norm kernels (a dummy length otherwise). */
    private static int normTgMem(Metal.Buffer norm, int cols) {
        return norm != null ? cols * 4 : UNUSED_TG_BYTES;
    }

    /** Offset of the f32 scale region to bind as buffer(1); Q2_K keeps its scales inside the blocks (unused binding). */
    static long scaleOff(QTensor t) {
        return t.type.scaleBytes == 0 ? 0 : t.scalesOffset;
    }

    /** Kernel name {@code base_<qt><suffix>} for a tensor's quantization type (q4 / q8 / q2k). */
    static String kernel(String base, QType type, String suffix) {
        String qt = switch (type) {
            case Q4_0 -> "q4";
            case Q8_0 -> "q8";
            case Q2_K -> "q2k";
        };
        return base + "_" + qt + suffix;
    }

    /**
     * q, k and v projections of {@code in} in one dispatch (k and v straight into the caches at pos), with the
     * q/k/v biases added in the kernel's epilogue when the layer has them.
     */
    private void qkv(ModelWeights.Layer W, LayerBufs L, Metal.Buffer in, Metal.Buffer norm,
                     Metal.Buffer kDst, long kOff, Metal.Buffer vDst, long vOff, int n) {
        if (W.wq.type != W.wk.type || W.wk.type != W.wv.type || W.wq.cols != W.wk.cols || W.wk.cols != W.wv.cols) {
            throw new IllegalStateException("q/k/v projections differ in type or width"); // the loader guarantees they match
        }
        boolean bias = L.bq != null;
        bind(0, L.wq, 0);
        bind(1, L.wq, scaleOff(W.wq));
        bind(2, L.wk, 0);
        bind(3, L.wk, scaleOff(W.wk));
        bind(4, L.wv, 0);
        bind(5, L.wv, scaleOff(W.wv));
        bind(6, in, 0);
        bind(7, q, 0);
        bind(8, kDst, kOff);
        bind(9, vDst, vOff);
        bind(10, norm != null ? norm : finalNorm, 0);
        bind(11, bias ? L.bq : finalNorm, 0); // dummy bindings without bias
        bind(12, bias ? L.bk : finalNorm, 0);
        bind(13, bias ? L.bv : finalNorm, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, W.wq.rows);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, W.wk.rows);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, W.wv.rows);
        params.setAtIndex(ValueLayout.JAVA_INT, 3, W.wq.cols);
        params.setAtIndex(ValueLayout.JAVA_INT, 4, n);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 5, c.normEps());
        params.setAtIndex(ValueLayout.JAVA_INT, 6, bias ? 1 : 0);
        int rows = W.wq.rows + W.wk.rows + W.wv.rows;
        if (norm == null && n >= PREFILL_MIN_TOKENS) {
            int pipe = metal.pipeline(kernel("matmul3_pf", W.wq.type, vh));
            metal.dispatch(pipe, bufArgs, offArgs, 14, params, 28, pfGrid(rows), pfTiles(n), 1, 32 * SG_PER_TG, 1, 1);
            return;
        }
        int pipe = metal.pipeline(kernel("matmul3", W.wq.type, (norm != null ? "_norm" : "") + vh));
        int gx = (rows + SG_PER_TG - 1) / SG_PER_TG;
        int gy = (n + TOK_TILE - 1) / TOK_TILE;
        metal.dispatch(pipe, bufArgs, offArgs, 14, params, 28, normTgMem(norm, W.wq.cols), gx, gy, 1,
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
        if (norm == null && n >= PREFILL_MIN_TOKENS) {
            int pipe = metal.pipeline(kernel("matmul_swiglu_pf", t.type, ""));
            metal.dispatch(pipe, bufArgs, offArgs, 5, params, 20, pfUnitGrid(hiddenDim), pfTiles(n), 1, 32 * SG_PER_TG, 1, 1);
            return;
        }
        int pipe = metal.pipeline(kernel("matmul_swiglu", t.type, norm != null ? "_norm" : ""));
        int gx = (hiddenDim + SG_PER_TG - 1) / SG_PER_TG;
        int gy = (n + TOK_TILE - 1) / TOK_TILE;
        metal.dispatch(pipe, bufArgs, offArgs, 5, params, 20, normTgMem(norm, t.cols), gx, gy, 1,
                32 * SG_PER_TG, 1, 1);
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
        metal.dispatch(metal.pipeline("rmsnorm"), bufArgs, offArgs, 3, params, 8, n, 1, 1, 256, 1, 1);
    }

    /** out[t] += rmsnorm(y[t]) * weight for n rows of the given width. */
    private void rmsnormAdd(Metal.Buffer y, Metal.Buffer weight, Metal.Buffer out, int n, int width) {
        bind(0, y, 0);
        bind(1, weight, 0);
        bind(2, out, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, width);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 1, c.normEps());
        metal.dispatch(metal.pipeline("rmsnorm_add"), bufArgs, offArgs, 3, params, 8, n, 1, 1, 256, 1, 1);
    }

    /**
     * QK-norm + RoPE for the q rows and the k rows of this chunk, in one dispatch: k is read from
     * {@code kSrc} (the f32 cache rows at pos, or the f32 staging rows of an f16 cache) and written to the cache
     * rows {@code kOut} at pos (the same rows for f32, rounded to half for f16). GLM rotates interleaved pairs
     * of the first rotaryDim dims, Phi rotate-half pairs of them; both without a qk-norm.
     */
    private void normRope(LayerBufs L, Metal.Buffer kSrc, long kSrcOff, Metal.Buffer kOut, long kOutOff, int n, int pos,
                          boolean localRope) {
        boolean partial = c.ropeInterleaved() || c.ropePartialHalf();
        if (partial && L.qNorm != null) {
            throw new IllegalStateException("partial rotary embedding combined with q/k norm is not implemented");
        }
        boolean useNorm = !partial && L.qNorm != null && L.kNorm != null;
        int rotHalf = partial ? c.rotaryDim() / 2 : c.headDim() / 2;
        rope(c.ropeInterleaved(), q, kSrc, kSrcOff, kOut, kOutOff, useNorm ? L.qNorm : null, useNorm ? L.kNorm : null,
                localRope ? ropeCosLocal : ropeCos, localRope ? ropeSinLocal : ropeSin,
                c.headDim(), c.qDim(), c.kvDim(), c.heads(), c.kvHeads(), pos, 0, rotHalf, n);
    }

    /**
     * The one writer of RopeParams: {@code rope} (rotate-half pairs) or {@code rope_il} (adjacent pairs) over
     * nHeads q heads in place and nKHeads k heads from kSrc into kOut (the half variant when the cache is f16),
     * rotating the {@code 2 * rotHalf} dims from {@code rotOff} of every head; qw/kw non-null = qk-norm first.
     */
    void rope(boolean adjacent, Metal.Buffer qb, Metal.Buffer kSrc, long kSrcOff, Metal.Buffer kOut, long kOutOff,
              Metal.Buffer qw, Metal.Buffer kw, Metal.Buffer cos, Metal.Buffer sin, int headDim, int qStride, int kvStride,
              int nHeads, int nKHeads, int pos, int rotOff, int rotHalf, int n) {
        bind(0, qb, 0);
        bind(1, kSrc, kSrcOff);
        bind(2, qw != null ? qw : finalNorm, 0); // dummy bindings when unused
        bind(3, kw != null ? kw : finalNorm, 0);
        bind(4, cos, 0);
        bind(5, sin, 0);
        bind(6, kOut, kOutOff);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, headDim);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, qStride);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, kvStride);
        params.setAtIndex(ValueLayout.JAVA_INT, 3, nHeads);
        params.setAtIndex(ValueLayout.JAVA_INT, 4, pos);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 5, c.normEps());
        params.setAtIndex(ValueLayout.JAVA_INT, 6, qw != null ? 1 : 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 7, rotOff);
        params.setAtIndex(ValueLayout.JAVA_INT, 8, rotHalf);
        String kernel = (adjacent ? "rope_il" : "rope") + (nKHeads > 0 ? vh : "");
        metal.dispatch(metal.pipeline(kernel), bufArgs, offArgs, 7, params, 36, nHeads + nKHeads, n, 1, rotHalf, 1, 1);
    }

    /** Dense attention of layer l for the n tokens at pos (the group-specialised kernel, or flash decoding). */
    private void attention(int l, int n, int pos) {
        if (flashDecode && n == 1 && pos + 1 >= FLASH_MIN_LEN) {
            flashAttention(l, pos);
            return;
        }
        int hd = c.headDim();
        attention(attnKernel, attnGroup, q, keyCache[l], valCache[l], att, c.heads(), c.kvHeads(), hd, hd, pos,
                c.attnScale(), c.qDim(), c.kvDim(), c.kvDim(), c.qDim(), c.windowFor(l), c.attnSoftcap(), n);
    }

    /**
     * The one writer of AttnParams: {@code attention_t} for n tokens, one dispatch per tile of ATT_TILE tokens
     * (the scores scratch is indexed by the tile-local token). {@code groups} query heads share a threadgroup;
     * kc/vc are the full caches, out receives rows of oStride floats.
     */
    void attention(String kernel, int groups, Metal.Buffer qb, Metal.Buffer kc, Metal.Buffer vc, Metal.Buffer out,
                   int nHeads, int nKvHeads, int qkDim, int vDim, int pos, float scale, int qStride, int kStride,
                   int vStride, int oStride, int window, float softcap, int n) {
        bind(0, qb, 0);
        bind(1, kc, 0);
        bind(2, vc, 0);
        bind(3, out, 0);
        bind(4, scores, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, nHeads);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, nKvHeads);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, qkDim);
        params.setAtIndex(ValueLayout.JAVA_INT, 3, vDim);
        params.setAtIndex(ValueLayout.JAVA_INT, 4, pos);
        params.setAtIndex(ValueLayout.JAVA_INT, 5, ctx);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 6, scale);
        params.setAtIndex(ValueLayout.JAVA_INT, 7, qStride);
        params.setAtIndex(ValueLayout.JAVA_INT, 8, kStride);
        params.setAtIndex(ValueLayout.JAVA_INT, 9, vStride);
        params.setAtIndex(ValueLayout.JAVA_INT, 10, oStride);
        params.setAtIndex(ValueLayout.JAVA_INT, 11, window);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 13, softcap);
        int pipe = metal.pipeline(kernel);
        for (int t0 = 0; t0 < n; t0 += ATT_TILE) {
            params.setAtIndex(ValueLayout.JAVA_INT, 12, t0);
            metal.dispatch(pipe, bufArgs, offArgs, 5, params, 56, nHeads / groups, Math.min(ATT_TILE, n - t0), 1, 256, 1, 1);
        }
    }

    /** Flash decoding for the single token at pos: attention_split over (heads, nSplit), then attention_combine. */
    private void flashAttention(int l, int pos) {
        int len = pos + 1;
        int window = c.windowFor(l);
        int first = window > 0 ? Math.max(0, len - window) : 0;
        int keys = len - first;
        int nSplit = Math.min(FLASH_MAX_SPLITS, (keys + FLASH_CHUNK - 1) / FLASH_CHUNK);
        int chunk = (keys + nSplit - 1) / nSplit;
        bind(0, q, 0);
        bind(1, keyCache[l], 0);
        bind(2, valCache[l], 0);
        bind(3, flashPart, 0);
        bind(4, scores, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, c.heads());
        params.setAtIndex(ValueLayout.JAVA_INT, 1, c.kvHeads());
        params.setAtIndex(ValueLayout.JAVA_INT, 2, c.headDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 3, pos);
        params.setAtIndex(ValueLayout.JAVA_INT, 4, ctx);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 5, c.attnScale());
        params.setAtIndex(ValueLayout.JAVA_INT, 6, c.qDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 7, c.kvDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 8, window);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 9, c.attnSoftcap());
        params.setAtIndex(ValueLayout.JAVA_INT, 10, nSplit);
        params.setAtIndex(ValueLayout.JAVA_INT, 11, chunk);
        metal.dispatch(metal.pipeline("attention_split" + vh), bufArgs, offArgs, 5, params, 48, c.heads(), nSplit, 1, 256, 1, 1);
        bind(0, flashPart, 0);
        bind(1, att, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, c.heads());
        params.setAtIndex(ValueLayout.JAVA_INT, 1, c.headDim());
        params.setAtIndex(ValueLayout.JAVA_INT, 2, nSplit);
        metal.dispatch(metal.pipeline("attention_combine"), bufArgs, offArgs, 2, params, 12, c.heads(), 1, 1, 256, 1, 1);
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
            metal.dispatch(metal.pipeline("router"), bufArgs, offArgs, 4, params, 16, n, 1, 1, 256, 1, 1);
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
        metal.dispatch(metal.pipeline(kernel("moe_up", gu.type, "")), bufArgs, offArgs, 5, params, 12,
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
        metal.dispatch(metal.pipeline(kernel("moe_down", dn.type, "")), bufArgs, offArgs, 6, params, 12,
                (c.dim() + sgPerTg - 1) / sgPerTg, n, 1, 32 * sgPerTg, 1, 1);
    }

    @Override
    public void close() {
        metal.close();
        arena.close();
    }
}
