package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.Rope;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Decoder forward pass (Llama, Mistral, Qwen2, Qwen3, Gemma, GLM, Phi, DeepSeek) on the GPU via Metal. Owns
 * the model's buffers (weights wrapped zero-copy in unified memory, caches, activations) and records the forward
 * pass of a chunk of tokens through the kernel writers of {@link MetalKernels}; the GPU runs it with one wait
 * at the end.
 */
public final class MetalBackend implements Backend {

    private final ModelWeights w;
    private final ModelConfig c;
    private final int ctx;
    private final int maxBatch;
    final MetalKernels k;

    private String attnKernel;         // attention_t instantiation for headDim / attnGroup / cache type
    private int attnGroup;             // query heads per attention threadgroup
    private final boolean flashDecode; // flash decoding enabled for this model (dense attention only)
    private Metal.Buffer flashPart;
    // f16 KV cache (backend.kv-dtype=f16): matmul3's *_h variants write v straight into the half cache, k goes
    // through an f32 staging row and rope_h rounds it into the cache after the rotation, attention_h
    // reads half; see the "f16 KV cache" section of kernels.metal
    private final boolean kvHalf;
    private boolean fusedNormFits;     // see MetalKernels.fusedNormFits
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
        this.flashDecode = MetalKernels.FLASH_DECODE && c.deepseek() == null;
        this.k = new MetalKernels(metal, c.normEps());
        try {
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
        if (c.deepseek() != null && kvDtype != KvDtype.F32) {
            throw new IllegalArgumentException("backend.kv-dtype=" + kvDtype + " is not supported for DeepSeek MLA models (use f32)");
        }
        // Attention specialised on headDim (64 / 128 / 256) and the group size, see attnGroupOverride
        attnGroup = attnGroupOverride(c);
        int hd = c.headDim();
        String vh = kvHalf ? "_h" : "";
        attnKernel = hd == 64 || hd == 128 || hd == 256 ? "attention_g" + attnGroup + "_d" + hd + vh : "attention" + vh;
        if (c.isMoe() && c.numExperts() > MetalKernels.MAX_EXPERTS) {
            throw new IllegalArgumentException("Metal router kernel supports at most " + MetalKernels.MAX_EXPERTS + " experts");
        }
        fusedNormFits = k.fusedNormFits(c.dim());

        embedQ = k.wrap(w.embed.data);
        lmHeadQ = w.lmHead == w.embed ? embedQ : k.wrap(w.lmHead.data);
        layers = new LayerBufs[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            ModelWeights.Layer L = w.layers[l];
            LayerBufs b = new LayerBufs();
            b.attnNorm = k.upload(L.attnNorm);
            b.ffnNorm = k.upload(L.ffnNorm);
            b.postAttnNorm = L.postAttnNorm != null ? k.upload(L.postAttnNorm) : null;
            b.postMlpNorm = L.postMlpNorm != null ? k.upload(L.postMlpNorm) : null;
            b.qNorm = L.qNorm != null ? k.upload(L.qNorm) : null;
            b.kNorm = L.kNorm != null ? k.upload(L.kNorm) : null;
            b.bq = L.bq != null ? k.upload(L.bq) : null;
            b.bk = L.bk != null ? k.upload(L.bk) : null;
            b.bv = L.bv != null ? k.upload(L.bv) : null;
            if (L.mla == null) {
                b.wq = k.wrap(L.wq.data);
                b.wk = k.wrap(L.wk.data);
                b.wv = k.wrap(L.wv.data);
            }
            b.wo = k.wrap(L.wo.data);
            if (L.moe) {
                b.router = k.upload(L.router);
                b.expGateUp = k.wrap(L.expGateUp.data);
                b.expDown = k.wrap(L.expDown.data);
            } else {
                b.wGateUp = k.wrap(L.wGateUp.data);
                b.wDown = k.wrap(L.wDown.data);
            }
            layers[l] = b;
        }
        finalNorm = k.upload(w.finalNorm);
        Rope rope = Rope.of(c, ctx);
        ropeCos = k.upload(rope.cos);
        ropeSin = k.upload(rope.sin);
        boolean localRope = false;
        for (int l = 0; l < c.layers(); l++) {
            localRope |= c.usesLocalRope(l);
        }
        if (localRope) {
            Rope local = Rope.local(c, ctx);
            ropeCosLocal = k.upload(local.cos);
            ropeSinLocal = k.upload(local.sin);
        } else {
            ropeCosLocal = ropeCos;
            ropeSinLocal = ropeSin;
        }

        // None of the buffers below is zeroed: every element is written before it is read. The KV cache rows
        // [0, pos + n) are written by the K/V projections (and the rope kernel) before attention reads them,
        // activations and staging rows are written for every token of the current chunk, and the scores
        // scratch is written for positions [first, len) before the softmax reads the same range. Skipping the
        // memset keeps start-up fast and only commits pages that are actually touched (a long context would
        // otherwise clear gigabytes).
        long kvBytes = (long) ctx * c.kvDim() * kvDtype.bytes;
        keyCache = new Metal.Buffer[c.layers()];
        valCache = new Metal.Buffer[c.layers()];
        for (int l = 0; l < c.layers() && c.deepseek() == null; l++) { // DeepSeek: MlaMetal owns its caches
            keyCache[l] = k.scratch(kvBytes);
            valCache[l] = k.scratch(kvBytes);
        }
        kStage = kvHalf ? k.scratch((long) maxBatch * c.kvDim() * 4) : null; // f32 k rows before RoPE
        x = k.scratch((long) maxBatch * c.dim() * 4);
        xb = k.scratch((long) maxBatch * c.dim() * 4);
        q = k.scratch((long) maxBatch * c.qDim() * 4);
        att = k.scratch((long) maxBatch * (c.deepseek() == null ? c.qDim()
                : Math.max(c.qDim(), c.heads() * c.deepseek().vHeadDim())) * 4);
        hidden = k.scratch((long) maxBatch * c.maxMlpDim() * 4);
        scores = k.scratch((long) Math.min(MetalKernels.ATT_TILE, maxBatch) * c.heads() * ctx * 4);
        logitsBuf = k.scratch((long) w.lmHead.rows * 4);
        flashPart = flashDecode ? k.scratch((long) c.heads() * MetalKernels.FLASH_MAX_SPLITS * (c.headDim() + 2) * 4) : null;
        if (c.isMoe()) {
            int slots = maxBatch * c.expertsPerTok();
            assign = k.scratch((long) slots * 4);
            routeWeight = k.scratch((long) slots * 4);
            moeHidden = k.scratch((long) slots * c.moeDim() * 4);
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
        long ws = k.metal.recommendedWorkingSet();
        if (w.byteSize() > ws) {
            System.err.printf("warning: weights (%.1f GB) exceed Metal's recommended working set (%.1f GB); expect paging%n",
                    w.byteSize() / 1e9, ws / 1e9);
        }
    }

    @Override
    public String name() {
        return "metal (" + k.metal.deviceName() + ")";
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

        k.begin();
        boolean committed = false;
        try {
            // For a single token the rmsnorm preceding a projection is folded into the projection kernel
            // (the *_norm variants read x and normalise on the fly); otherwise it writes xb first.
            boolean fuseNorm = n == 1 && fusedNormFits;
            for (int l = 0; l < c.layers(); l++) {
                LayerBufs L = layers[l];
                ModelWeights.Layer W = w.layers[l];
                if (mla != null) {
                    k.rmsnorm(x, 0, L.attnNorm, xb, n, dim);
                    mla.run(l, W, n, pos);
                } else {
                    // v is written straight into the cache rows at pos (as half for the f16 cache); k too for
                    // the f32 cache, while the f16 cache stages the f32 k rows and the rope kernel rounds them in
                    long cacheOff = (long) pos * kvDim * (kvHalf ? 2 : 4);
                    Metal.Buffer kDst = kvHalf ? kStage : keyCache[l];
                    long kOff = kvHalf ? 0 : cacheOff;
                    Metal.Buffer in = x;
                    Metal.Buffer norm = L.attnNorm;
                    if (!fuseNorm) {
                        k.rmsnorm(x, 0, L.attnNorm, xb, n, dim);
                        in = xb;
                        norm = null;
                    }
                    k.matmul3(W.wq, L.wq, W.wk, L.wk, W.wv, L.wv, in, norm, q, kDst, kOff, valCache[l], cacheOff, kvHalf,
                            L.bq, L.bk, L.bv, n);
                    normRope(L, kDst, kOff, keyCache[l], cacheOff, n, pos, c.usesLocalRope(l));
                    attention(l, n, pos);
                }
                if (L.postAttnNorm != null) {
                    // Glm4: x += rmsnorm(o_proj(attention)), so o_proj cannot accumulate into x directly
                    k.matmul(W.wo, L.wo, att, 0, xb, 0, n, false, null, 0f);
                    k.rmsnormAdd(xb, L.postAttnNorm, x, n, dim);
                } else {
                    k.matmul(W.wo, L.wo, att, 0, x, 0, n, true, null, 0f);
                }

                if (W.moe) {
                    k.rmsnorm(x, 0, L.ffnNorm, xb, n, dim);
                    moe(W, L, l, n);
                    if (mla != null) {
                        mla.sharedExperts(W, l, n);
                    }
                } else {
                    if (fuseNorm) {
                        k.swiglu(W.wGateUp, L.wGateUp, x, L.ffnNorm, hidden, c.isGemma(), n);
                    } else {
                        k.rmsnorm(x, 0, L.ffnNorm, xb, n, dim);
                        k.swiglu(W.wGateUp, L.wGateUp, xb, null, hidden, c.isGemma(), n);
                    }
                    if (L.postMlpNorm != null) {
                        k.matmul(W.wDown, L.wDown, hidden, 0, xb, 0, n, false, null, 0f);
                        k.rmsnormAdd(xb, L.postMlpNorm, x, n, dim);
                    } else {
                        k.matmul(W.wDown, L.wDown, hidden, 0, x, 0, n, true, null, 0f);
                    }
                }
                if (MetalKernels.COMMIT_LAYERS > 0 && (l + 1) % MetalKernels.COMMIT_LAYERS == 0 && l + 1 < c.layers()) {
                    k.commit(); // the GPU starts on these layers while the next ones are encoded
                }
            }
            if (logits != null) {
                // normalise only the last token (read x at row n-1, write xb row 0), then the LM head. Folding
                // the norm into the LM-head kernel measured ~5% slower at decode: its prologue runs in all
                // ~38k threadgroups, which costs more than the one dispatch it saves.
                k.rmsnorm(x, (long) (n - 1) * dim * 4, finalNorm, xb, 1, dim);
                k.matmul(w.lmHead, lmHeadQ, xb, 0, logitsBuf, 0, 1, false, null, c.finalSoftcap()); // Gemma: capped logits
            }
            k.commitAndWait();
            committed = true;
        } finally {
            if (!committed) {
                k.abort(); // an un-ended encoder would make Metal abort the process on release
            }
        }
        if (logits != null) {
            MemorySegment.copy(logitsBuf.contents(), ValueLayout.JAVA_FLOAT, 0, logits, 0,
                    Math.min(w.lmHead.rows, logits.length));
        }
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
        k.rope(c.ropeInterleaved(), kvHalf, q, kSrc, kSrcOff, kOut, kOutOff, useNorm ? L.qNorm : null, useNorm ? L.kNorm : null,
                localRope ? ropeCosLocal : ropeCos, localRope ? ropeSinLocal : ropeSin,
                c.headDim(), c.qDim(), c.kvDim(), c.heads(), c.kvHeads(), pos, 0, rotHalf, n);
    }

    /** Dense attention of layer l for the n tokens at pos (the group-specialised kernel, or flash decoding). */
    private void attention(int l, int n, int pos) {
        int hd = c.headDim();
        if (flashDecode && n == 1 && pos + 1 >= MetalKernels.FLASH_MIN_LEN) {
            k.flashAttention(kvHalf, q, keyCache[l], valCache[l], flashPart, scores, att, c.heads(), c.kvHeads(), hd, pos, ctx,
                    c.attnScale(), c.qDim(), c.kvDim(), c.windowFor(l), c.attnSoftcap());
            return;
        }
        k.attention(attnKernel, attnGroup, q, keyCache[l], valCache[l], att, scores, c.heads(), c.kvHeads(), hd, hd, pos, ctx,
                c.attnScale(), c.qDim(), c.kvDim(), c.kvDim(), c.qDim(), c.windowFor(l), c.attnSoftcap(), n);
    }

    /** Router → per-slot gate/up → SwiGLU → weighted down projections accumulated into x. */
    private void moe(ModelWeights.Layer W, LayerBufs L, int l, int n) {
        int topK = c.expertsPerTok();
        if (mla != null) {
            mla.router(l, L.router, n); // DeepSeek rule; fills assign / routeWeight like `router`
        } else {
            k.router(xb, L.router, assign, routeWeight, c.numExperts(), c.dim(), topK, c.normTopK(), n);
        }
        k.moeUp(W.expGateUp, L.expGateUp, xb, assign, moeHidden, c.moeDim(), topK, n);
        k.moeDown(W.expDown, L.expDown, moeHidden, assign, routeWeight, x, c.dim(), topK, n);
    }

    @Override
    public void close() {
        k.close();
    }
}
