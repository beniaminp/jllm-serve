package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.InterleavedRope;
import org.robomiri.jllm.model.GemmaMath;
import org.robomiri.jllm.model.DeepSeekRouter;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.Rope;

import java.util.Arrays;

/**
 * Decoder forward pass on the CPU (Llama, Mistral, Qwen2, Qwen3, GLM, Phi-3, Gemma, DeepSeek): quantized matmuls
 * with the Vector API ({@link CpuMatmul}: multithreaded over output rows for decode, row groups for prefill), f32
 * or f16 KV cache on the Java heap ({@link KvCache}).
 */
public final class CpuBackend implements Backend {

    /**
     * {@code -Djllm.cpu.deterministic=true}: every multi-token matmul (and MoE layer) is run token by token through
     * the single-token kernels, so a token's hidden state no longer depends on whether it was prefilled or decoded
     * (docs/07 §7.14). Slower prefill; meant for comparisons, not for serving.
     */
    static final boolean DETERMINISTIC = Boolean.getBoolean("jllm.cpu.deterministic");

    private final ModelWeights w;
    private final ModelConfig c;
    private final int ctx;
    private final int maxBatch;
    final ThreadPool pool;
    private final Rope rope;
    /** Gemma 3 sliding layers use another RoPE base; same object as {@link #rope} for every other model. */
    private final Rope ropeLocal;

    private final KvCache kv;
    /** DeepSeek multi-head latent attention (own cache); null for every other family. */
    private final MlaCpu mla;
    // k/v of the current chunk, [maxBatch, kvDim]: projected, normalized and rotated here, then stored
    private final float[] kStage;
    private final float[] vStage;

    // activations, [maxBatch, width] row-major
    private final float[] x;
    final float[] xb;
    final float[] q;
    final float[] att;
    private final float[] gateUp;
    private final float[] hidden;

    // MoE scratch (null for dense models); per-slot buffers are indexed by slot = t * topK + k
    private final float[] routerLogits;
    private final int[] assign;
    private final float[] routeWeight;
    private final float[] moeGateUp;
    private final float[] moeHidden;
    private final float[] dsScratch;
    private final float[] moeDown;
    private final int[] expertStart;
    private final int[] expertFill;
    private final int[] usedExperts;
    private final int[] slotOrder;

    /** The quantized matmul (float and int8 paths) and the thread-local float scratch, shared with {@link MlaCpu}. */
    final CpuMatmul mm;
    final ThreadLocal<float[]> scratch;

    public CpuBackend(ModelWeights w, int ctx, int maxBatch, int threads) {
        this(w, ctx, maxBatch, threads, KvDtype.F32);
    }

    public CpuBackend(ModelWeights w, int ctx, int maxBatch, int threads, KvDtype kvDtype) {
        this(w, ctx, maxBatch, threads, kvDtype, false);
    }

    /**
     * @param int8Activations quantize activations to int8 per 32-block and use integer dot products for the
     *                        single-token (decode / MoE) matmuls instead of dequantizing weights to float
     *                        (Q4_0 and Q8_0 weights; other formats keep the float path)
     */
    public CpuBackend(ModelWeights w, int ctx, int maxBatch, int threads, KvDtype kvDtype, boolean int8Activations) {
        this.w = w;
        this.c = w.cfg;
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        this.pool = new ThreadPool(threads);
        this.rope = Rope.of(c, ctx);
        this.ropeLocal = anyLocalRope(c) ? Rope.local(c, ctx) : rope;
        if (ctx > c.maxPositions()) {
            System.err.println("warning: context " + ctx + " exceeds the model's max_position_embeddings ("
                    + c.maxPositions() + "); RoPE positions beyond the training length are extrapolated");
        }
        if (c.deepseek() != null) {
            if (kvDtype != KvDtype.F32) {
                throw new IllegalArgumentException("backend.kv-dtype=" + kvDtype + " is not supported for DeepSeek MLA models (use f32)");
            }
            kv = null;
            mla = new MlaCpu(this, c, rope, ctx, maxBatch);
        } else {
            kv = KvCache.create(kvDtype, c, ctx);
            mla = null;
        }
        kStage = new float[maxBatch * c.kvDim()];
        vStage = new float[maxBatch * c.kvDim()];
        x = new float[maxBatch * c.dim()];
        xb = new float[maxBatch * c.dim()];
        q = new float[maxBatch * c.qDim()];
        att = new float[maxBatch * (c.deepseek() == null ? c.qDim() : Math.max(c.qDim(), c.heads() * c.deepseek().vHeadDim()))];
        gateUp = new float[maxBatch * 2 * c.maxMlpDim()];
        hidden = new float[maxBatch * c.maxMlpDim()];
        if (c.isMoe()) {
            int slots = maxBatch * c.expertsPerTok();
            routerLogits = new float[c.numExperts()];
            assign = new int[slots];
            routeWeight = new float[slots];
            moeGateUp = new float[slots * 2 * c.moeDim()];
            moeHidden = new float[slots * c.moeDim()];
            dsScratch = new float[c.numExperts()];
            moeDown = new float[slots * c.dim()];
            expertStart = new int[c.numExperts() + 1];
            expertFill = new int[c.numExperts()];
            usedExperts = new int[c.numExperts()];
            slotOrder = new int[slots];
        } else {
            dsScratch = null;
            routerLogits = null;
            assign = null;
            routeWeight = null;
            moeGateUp = null;
            moeHidden = null;
            moeDown = null;
            expertStart = null;
            expertFill = null;
            usedExperts = null;
            slotOrder = null;
        }
        int scratchLen = Math.max(Math.max(ctx, c.numExperts()),
                4 * Math.max(Math.max(c.maxMlpDim(), c.moeDim()), Math.max(c.dim(), c.qDim())));
        mm = new CpuMatmul(pool, int8Activations, DETERMINISTIC, scratchLen);
        scratch = mm.scratch;
    }

    private static boolean anyLocalRope(ModelConfig c) {
        for (int l = 0; l < c.layers(); l++) {
            if (c.usesLocalRope(l)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String name() {
        return "cpu (" + VectorKernels.describe() + (mm.int8() ? ", " + Int8Kernels.describe() : "")
                + (DETERMINISTIC ? ", deterministic" : "") + ", " + pool.threads() + " threads)";
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
        for (int t = 0; t < n; t++) {
            w.embed.dequantizeRow(tokens[off + t], x, t * dim);
        }
        if (c.embedScale() != 1f) {
            for (int i = 0, m = n * dim; i < m; i++) {
                x[i] *= c.embedScale();
            }
        }
        for (int l = 0; l < c.layers(); l++) {
            ModelWeights.Layer L = w.layers[l];
            rmsnorm(x, xb, L.attnNorm, n, dim);
            if (mla != null) {
                mla.run(l, L, n, pos);
            } else {
                matmul(L.wq, xb, q, 0, n, false);
                matmul(L.wk, xb, kStage, 0, n, false);
                matmul(L.wv, xb, vStage, 0, n, false);
                if (L.bq != null) {
                    addBias(q, 0, L.bq, n);
                    addBias(kStage, 0, L.bk, n);
                    addBias(vStage, 0, L.bv, n);
                }
                Rope r = c.usesLocalRope(l) ? ropeLocal : rope;
                normRope(q, 0, c.qDim(), c.heads(), L.qNorm, n, pos, r);
                normRope(kStage, 0, kvDim, c.kvHeads(), L.kNorm, n, pos, r);
                kv.store(l, pos, kStage, vStage, n);
                attention(l, n, pos);
            }
            if (L.postAttnNorm != null) {
                // Glm4: x += rmsnorm(o_proj(attention))
                matmul(L.wo, att, xb, 0, n, false);
                addNormed(xb, L.postAttnNorm, n);
            } else {
                matmul(L.wo, att, x, 0, n, true);
            }

            rmsnorm(x, xb, L.ffnNorm, n, dim);
            if (L.moe) {
                moe(L, n);
                if (L.mla != null && L.mla.sharedGateUp != null) {
                    // DeepSeek shared experts: a dense SwiGLU of every token, added to the routed output
                    int sh = c.deepseek().nShared() * c.moeDim();
                    matmul(L.mla.sharedGateUp, xb, gateUp, 0, n, false);
                    siluMul(n, sh);
                    matmul(L.mla.sharedDown, hidden, x, 0, n, true);
                }
            } else {
                matmul(L.wGateUp, xb, gateUp, 0, n, false);
                if (c.isGemma()) {
                    geluMul(n);
                } else {
                    siluMul(n);
                }
                if (L.postMlpNorm != null) {
                    // Glm4: x += rmsnorm(down_proj(mlp)); xb is free again once gate/up are computed
                    matmul(L.wDown, hidden, xb, 0, n, false);
                    addNormed(xb, L.postMlpNorm, n);
                } else {
                    matmul(L.wDown, hidden, x, 0, n, true);
                }
            }
        }
        if (logits != null) {
            rmsnormRow(x, (n - 1) * dim, xb, 0, w.finalNorm, dim);
            // the caller sizes logits by cfg.vocabSize(); the LM head may have fewer (or, padded, more) rows
            int rows = Math.min(w.lmHead.rows, logits.length);
            mm.mul(w.lmHead, rows, xb, logits, 0, 1, false);
            float cap = c.finalSoftcap();
            if (cap > 0) {
                pool.forRange(rows, 4096, (i0, i1) -> {
                    for (int i = i0; i < i1; i++) {
                        logits[i] = GemmaMath.softcap(logits[i], cap);
                    }
                });
            }
        }
    }

    // ------------------------------------------------------------------ layers

    void rmsnorm(float[] in, float[] out, float[] weight, int n, int width) {
        if (n == 1) {
            rmsnormRow(in, 0, out, 0, weight, width);
            return;
        }
        pool.forRange(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                rmsnormRow(in, t * width, out, t * width, weight, width);
            }
        });
    }

    /** x[t] += rmsnorm(y[t]) * weight for n rows; y is overwritten with the normalized rows. */
    private void addNormed(float[] y, float[] weight, int n) {
        int dim = c.dim();
        rmsnorm(y, y, weight, n, dim);
        for (int i = 0; i < n * dim; i++) {
            x[i] += y[i];
        }
    }

    void rmsnormRow(float[] in, int io, float[] out, int oo, float[] weight, int width) {
        float ss = VectorKernels.sumSquares(in, io, width);
        float scale = (float) (1.0 / Math.sqrt(ss / width + c.normEps()));
        VectorKernels.scaleMul(in, io, scale, weight, out, oo, width);
    }

    /** {@code out[t, r] (+)= dot(W[r, :], in[t, :])} for {@code t < n}: {@link CpuMatmul#mul}. */
    void matmul(QTensor wt, float[] in, float[] out, int outOff, int n, boolean acc) {
        mm.mul(wt, wt.rows, in, out, outOff, n, acc);
    }

    /** buf[off + t*bias.length + i] += bias[i] for n rows (Qwen2 q/k/v bias). */
    private static void addBias(float[] buf, int off, float[] bias, int n) {
        for (int t = 0; t < n; t++) {
            int o = off + t * bias.length;
            for (int i = 0; i < bias.length; i++) {
                buf[o + i] += bias[i];
            }
        }
    }

    /** Per-head RMSNorm (if {@code norm != null}) followed by rotary embedding, in place. */
    private void normRope(float[] v, int base, int stride, int nHeads, float[] norm, int n, int pos, Rope rope) {
        if (n == 1) {
            normRopeHeads(v, base, stride, nHeads, norm, pos, rope, 0, nHeads);
            return;
        }
        pool.forRange(n * nHeads, 4, (i0, i1) -> normRopeHeads(v, base, stride, nHeads, norm, pos, rope, i0, i1));
    }

    /** {@link #normRope} over the flat (token, head) indices {@code [i0, i1)}. */
    private void normRopeHeads(float[] v, int base, int stride, int nHeads, float[] norm, int pos, Rope rope,
                               int i0, int i1) {
        int hd = c.headDim();
        int half = c.rotaryDim() / 2; // rope table width; = hd / 2 unless partial_rotary_factor < 1
        boolean interleaved = c.ropeInterleaved();
        float[] cos = rope.cos;
        float[] sin = rope.sin;
        for (int i = i0; i < i1; i++) {
            int t = i / nHeads;
            int h = i - t * nHeads;
            int o = base + t * stride + h * hd;
            int ro = (pos + t) * half;
            if (norm != null) {
                rmsnormRow(v, o, v, o, norm, hd);
            }
            if (interleaved) {
                InterleavedRope.rotate(v, o, c.rotaryDim(), cos, sin, ro);
            } else {
                VectorKernels.rope(v, o, half, cos, sin, ro);
            }
        }
    }

    /** Causal grouped-query attention for tokens at positions [pos, pos+n): {@link KvCache#attention}. */
    private void attention(int layer, int n, int pos) {
        kv.attention(pool, layer, n, pos, q, att, c.windowFor(layer), c.attnScale(), c.attnSoftcap());
    }

    /**
     * Mixture of experts: for each token, softmax over the router logits, pick the top-k experts, run each
     * expert's SwiGLU MLP on the token and add the results to x weighted by the (renormalized) router probabilities.
     */
    private void moe(ModelWeights.Layer L, int n) {
        if (n > 1 && !DETERMINISTIC) {
            moeBatched(L, n);
            return;
        }
        for (int t = 0; t < n; t++) {
            moeToken(L, t);
        }
    }

    /** The single-token MoE layer for token {@code t}: {@code x[t] += Σ_k w_k · expert_k(xb[t])}. */
    private void moeToken(ModelWeights.Layer L, int t) {
        int dim = c.dim();
        int e = c.numExperts();
        int topK = c.expertsPerTok();
        int m = c.moeDim();
        int gu = 2 * m;
        int xo = t * dim;
        if (L.mla != null) {
            routeDeepseek(L, xo, e, topK);
        } else {
            route(L.router, xo, e, topK, routerLogits, 0);
        }
        // all selected experts' gate/up rows in one parallel loop
        if (mm.useInt8(L.expGateUp)) {
            short[] xq = mm.quantizeInt8(xb, xo, dim);
            float[] xs = mm.xs();
            pool.forRange(topK * gu, 32, (i0, i1) -> {
                for (int i = i0; i < i1; i++) {
                    int k = i / gu;
                    int r = i - k * gu;
                    moeGateUp[i] = Int8Kernels.dotRow(L.expGateUp, assign[k] * gu + r, xq, xs, 0);
                }
            });
        } else {
            pool.forRange(topK * gu, 32, (i0, i1) -> {
                for (int i = i0; i < i1; i++) {
                    int k = i / gu;
                    int r = i - k * gu;
                    moeGateUp[i] = VectorKernels.dotRow(L.expGateUp, assign[k] * gu + r, xb, xo);
                }
            });
        }
        for (int k = 0; k < topK; k++) {
            int gi = k * gu;
            int hi = k * m;
            for (int i = 0; i < m; i++) {
                float g = moeGateUp[gi + i];
                moeHidden[hi + i] = g / (1f + (float) Math.exp(-g)) * moeGateUp[gi + m + i];
            }
        }
        if (mm.useInt8(L.expDown)) {
            short[] xq = mm.quantizeInt8(moeHidden, 0, topK * m);   // slot k starts at block offset k * m
            float[] xs = mm.xs();
            pool.forRange(dim, 8, (r0, r1) -> {
                for (int r = r0; r < r1; r++) {
                    float sum = 0;
                    for (int k = 0; k < topK; k++) {
                        sum += routeWeight[k] * Int8Kernels.dotRow(L.expDown, assign[k] * dim + r, xq, xs, k * m);
                    }
                    x[xo + r] += sum;
                }
            });
        } else {
            pool.forRange(dim, 8, (r0, r1) -> {
                for (int r = r0; r < r1; r++) {
                    float sum = 0;
                    for (int k = 0; k < topK; k++) {
                        sum += routeWeight[k] * VectorKernels.dotRow(L.expDown, assign[k] * dim + r, moeHidden, k * m);
                    }
                    x[xo + r] += sum;
                }
            });
        }
    }

    /**
     * Prefill MoE: route every token, group the {@code n * topK} slots by expert (counting sort), then run each
     * expert once over all tokens routed to it with the dense prefill kernel (four rows dequantized once, reused
     * for every token). Down-projection results are kept per slot and combined into x per token in slot order.
     */
    private void moeBatched(ModelWeights.Layer L, int n) {
        int dim = c.dim();
        int e = c.numExperts();
        int topK = c.expertsPerTok();
        int m = c.moeDim();
        int gu = 2 * m;
        int slots = n * topK;

        if (L.mla != null) {
            // DeepSeekRouter.select writes slots [0, topK): route the last token first and move each result to
            // its own slots, so token 0 (which stays at offset 0) is written last
            for (int t = n - 1; t >= 0; t--) {
                routeDeepseek(L, t * dim, e, topK);
                System.arraycopy(assign, 0, assign, t * topK, topK);
                System.arraycopy(routeWeight, 0, routeWeight, t * topK, topK);
            }
        } else {
            pool.forRange(n, 1, (t0, t1) -> {
                float[] logits = scratch.get();
                for (int t = t0; t < t1; t++) {
                    route(L.router, t * dim, e, topK, logits, t * topK);
                }
            });
        }

        Arrays.fill(expertStart, 0, e + 1, 0);
        for (int s = 0; s < slots; s++) {
            expertStart[assign[s] + 1]++;
        }
        int used = 0;
        for (int i = 0; i < e; i++) {
            if (expertStart[i + 1] > 0) {
                usedExperts[used++] = i;
            }
            expertStart[i + 1] += expertStart[i];
            expertFill[i] = expertStart[i];
        }
        for (int s = 0; s < slots; s++) {
            slotOrder[expertFill[assign[s]]++] = s;
        }

        expertMatmul(L.expGateUp, gu, used, xb, dim, false, moeGateUp, gu);

        pool.forRange(slots * m, 256, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int s = i / m;
                int j = i - s * m;
                int gi = s * gu;
                float g = moeGateUp[gi + j];
                moeHidden[s * m + j] = g / (1f + (float) Math.exp(-g)) * moeGateUp[gi + m + j];
            }
        });

        expertMatmul(L.expDown, dim, used, moeHidden, m, true, moeDown, dim);

        pool.forRange(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                int xo = t * dim;
                int s0 = t * topK;
                for (int r = 0; r < dim; r++) {
                    float sum = 0;
                    for (int k = 0; k < topK; k++) {
                        sum += routeWeight[s0 + k] * moeDown[(s0 + k) * dim + r];
                    }
                    x[xo + r] += sum;
                }
            }
        });
    }

    /**
     * {@code out[slot, r] = dot(W[expert(slot) * rowsPer + r, :], in[src(slot), :])} for every slot of every
     * used expert, where {@code src(slot)} is the slot itself ({@code inBySlot}) or its token. Parallel over
     * (expert, 4-row group) pairs; each group is dequantized once and reused for all of the expert's slots.
     */
    private void expertMatmul(QTensor wt, int rowsPer, int used, float[] in, int inStride, boolean inBySlot,
                              float[] out, int outStride) {
        int cols = wt.cols;
        int topK = c.expertsPerTok();
        int groups = (rowsPer + 3) / 4;
        pool.forRange(used * groups, 1, (g0, g1) -> {
            float[] tmp = scratch.get();
            for (int g = g0; g < g1; g++) {
                int ex = usedExperts[g / groups];
                int r = (g % groups) * 4;
                int rowBase = ex * rowsPer;
                int sBeg = expertStart[ex];
                int sEnd = expertStart[ex + 1];
                if (r + 4 <= rowsPer) {
                    for (int k = 0; k < 4; k++) {
                        VectorKernels.dequantRow(wt, rowBase + r + k, tmp, k * cols);
                    }
                    int s = sBeg;
                    for (; s + 4 <= sEnd; s += 4) {
                        int sa = slotOrder[s];
                        int sb = slotOrder[s + 1];
                        int sc = slotOrder[s + 2];
                        int sd = slotOrder[s + 3];
                        VectorKernels.dot4x4(tmp, 0, cols, in,
                                (inBySlot ? sa : sa / topK) * inStride, (inBySlot ? sb : sb / topK) * inStride,
                                (inBySlot ? sc : sc / topK) * inStride, (inBySlot ? sd : sd / topK) * inStride,
                                out, sa * outStride + r, sb * outStride + r, sc * outStride + r, sd * outStride + r,
                                false);
                    }
                    for (; s < sEnd; s++) {
                        int slot = slotOrder[s];
                        int io = (inBySlot ? slot : slot / topK) * inStride;
                        VectorKernels.dot4(tmp, 0, cols, in, io, out, slot * outStride + r, false);
                    }
                } else {
                    for (int rr = r; rr < rowsPer; rr++) {
                        for (int s = sBeg; s < sEnd; s++) {
                            int slot = slotOrder[s];
                            int io = (inBySlot ? slot : slot / topK) * inStride;
                            out[slot * outStride + rr] = VectorKernels.dotRow(wt, rowBase + rr, in, io);
                        }
                    }
                }
            }
        });
    }

    /**
     * DeepSeek router for one token at {@code xb[xo..]} (see {@link DeepSeekRouter}): fills
     * {@code assign[0..topK)} / {@code routeWeight[0..topK)}. Uses the backend's own scratch, so one token at a time.
     */
    private void routeDeepseek(ModelWeights.Layer L, int xo, int e, int topK) {
        int dim = c.dim();
        for (int i = 0; i < e; i++) {
            routerLogits[i] = VectorKernels.dot(L.router, i * dim, xb, xo, dim);
        }
        DeepSeekRouter.select(routerLogits, L.mla.routerBias, c.deepseek(), topK, c.normTopK(), assign, routeWeight,
                dsScratch);
    }

    /**
     * Router for one token at {@code xb[xo..]}: fills {@code assign[ao..ao+topK)} / {@code routeWeight[ao..]}
     * (same rule as transformers' Qwen3MoeSparseMoeBlock). {@code logits} is scratch of at least {@code e} floats.
     */
    private void route(float[] router, int xo, int e, int topK, float[] logits, int ao) {
        int dim = c.dim();
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < e; i++) {
            float v = VectorKernels.dot(router, i * dim, xb, xo, dim);
            logits[i] = v;
            max = Math.max(max, v);
        }
        float sum = 0;
        for (int i = 0; i < e; i++) {
            float p = (float) Math.exp(logits[i] - max);
            logits[i] = p;
            sum += p;
        }
        float selected = 0;
        for (int k = 0; k < topK; k++) {
            int best = -1;
            float bv = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < e; i++) {
                if (logits[i] > bv) {
                    bv = logits[i];
                    best = i;
                }
            }
            assign[ao + k] = best;
            routeWeight[ao + k] = bv / sum;
            selected += bv / sum;
            logits[best] = Float.NEGATIVE_INFINITY;
        }
        if (c.normTopK() && selected > 0) {
            for (int k = 0; k < topK; k++) {
                routeWeight[ao + k] /= selected;
            }
        }
    }

    /** hidden = gelu_tanh(gate) * up, Gemma's GeGLU; parallel over flat (token, i) indices like {@link #siluMul}. */
    private void geluMul(int n) {
        int hdim = c.hiddenDim();
        pool.forRange(n * hdim, 512, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int t = i / hdim;
                int j = i - t * hdim;
                int gi = t * 2 * hdim;
                hidden[t * hdim + j] = GemmaMath.geluTanh(gateUp[gi + j]) * gateUp[gi + hdim + j];
            }
        });
    }

    private void siluMul(int n) {
        siluMul(n, c.hiddenDim());
    }

    /**
     * {@code hidden[t, i] = silu(gate[t, i]) * up[t, i]} over all tokens, parallel over flat (token, i) indices.
     * {@code Math.exp} stays scalar: the Vector API's EXP is not bit-identical to it.
     */
    private void siluMul(int n, int hdim) {
        pool.forRange(n * hdim, 512, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int t = i / hdim;
                int j = i - t * hdim;
                int gi = t * 2 * hdim;
                float g = gateUp[gi + j];
                hidden[t * hdim + j] = g / (1f + (float) Math.exp(-g)) * gateUp[gi + hdim + j];
            }
        });
    }

    @Override
    public void close() {
        pool.close();
    }
}
