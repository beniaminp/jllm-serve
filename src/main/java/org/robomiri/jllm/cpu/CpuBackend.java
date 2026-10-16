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
    private final int[] assign;
    private final float[] routeWeight;
    private final float[] moeGateUp;
    private final float[] moeHidden;
    /** DeepSeek router working space, one per thread (null for other models). */
    private final ThreadLocal<DsBuf> dsBuf;
    private final float[] moeDown;
    private final int[] expertStart;
    private final int[] expertFill;
    private final int[] usedExperts;
    private final int[] slotOrder;

    /** The quantized matmul (float and int8 paths) and the thread-local float scratch, shared with {@link MlaCpu}. */
    final CpuMatmul mm;
    final ThreadLocal<float[]> scratch;
    /** What each layer does (attention, residual, feed-forward), resolved once from the config and weights. */
    private final LayerPlan[] plans;

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
        kStage = mla == null ? new float[maxBatch * c.kvDim()] : null; // MLA keeps its own cache and staging
        vStage = mla == null ? new float[maxBatch * c.kvDim()] : null;
        x = new float[maxBatch * c.dim()];
        xb = new float[maxBatch * c.dim()];
        q = new float[maxBatch * c.qDim()];
        att = new float[maxBatch * (c.deepseek() == null ? c.qDim() : Math.max(c.qDim(), c.heads() * c.deepseek().vHeadDim()))];
        gateUp = new float[maxBatch * 2 * c.maxMlpDim()];
        hidden = new float[maxBatch * c.maxMlpDim()];
        if (c.isMoe()) {
            int slots = maxBatch * c.expertsPerTok();
            assign = new int[slots];
            routeWeight = new float[slots];
            moeGateUp = new float[slots * 2 * c.moeDim()];
            moeHidden = new float[slots * c.moeDim()];
            dsBuf = c.deepseek() != null ? ThreadLocal.withInitial(() -> new DsBuf(c.numExperts(), c.expertsPerTok())) : null;
            moeDown = new float[slots * c.dim()];
            expertStart = new int[c.numExperts() + 1];
            expertFill = new int[c.numExperts()];
            usedExperts = new int[c.numExperts()];
            slotOrder = new int[slots];
        } else {
            dsBuf = null;
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
        // attention scores (ctx), router logits (experts), or four dequantized rows of the widest tensor
        int scratchLen = Math.max(Math.max(ctx, c.numExperts()), 4 * CpuMatmul.maxCols(w));
        mm = new CpuMatmul(pool, int8Activations, DETERMINISTIC, scratchLen);
        scratch = mm.scratch;
        plans = new LayerPlan[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            plans[l] = plan(l, w.layers[l]);
        }
    }

    // ------------------------------------------------------------------ layer plans

    /**
     * One decoder layer as three strategy objects, so {@link #forward} contains no per-layer dispatch on the
     * family's optional pieces: attention (GQA over the {@link KvCache}, or DeepSeek MLA), how a projection joins
     * the residual stream (plain add, or Glm4 / Gemma post-norm add), and the feed-forward (dense GLU with SiLU
     * or GELU, routed experts, or routed plus shared experts). Every implementation runs exactly the statements
     * the straight-line code ran before, in the same order.
     */
    private record LayerPlan(Attention attention, Residual residual, Ffn ffn) {
    }

    /** Reads the normalized hidden states {@link #xb}, writes the attention output {@link #att}. */
    private interface Attention {
        void run(int layer, ModelWeights.Layer L, int n, int pos);
    }

    /** {@code x += proj(in)}, or {@code x += rmsnorm(proj(in)) * norm} where the family normalizes the output. */
    private interface Residual {
        void add(QTensor proj, float[] in, float[] norm, int n);
    }

    /** Reads {@link #xb}, adds the feed-forward output to {@link #x}. */
    private interface Ffn {
        void run(ModelWeights.Layer L, int n);
    }

    private LayerPlan plan(int l, ModelWeights.Layer L) {
        Attention attention = mla != null ? mla::run : new Gqa(c.usesLocalRope(l) ? ropeLocal : rope);
        Residual residual = L.postAttnNorm != null ? this::addPostNormed : this::addPlain;
        Ffn ffn;
        if (!L.moe) {
            ffn = new DenseGlu(c.isGemma(), residual);
        } else if (L.mla != null && L.mla.sharedGateUp != null) {
            ffn = this::moeWithShared;
        } else {
            ffn = this::moe;
        }
        return new LayerPlan(attention, residual, ffn);
    }

    /** Grouped-query attention: q/k/v projections (+ bias), QK-norm and RoPE, store into the cache, attend. */
    private final class Gqa implements Attention {
        private final Rope rope;

        Gqa(Rope rope) {
            this.rope = rope;
        }

        @Override
        public void run(int layer, ModelWeights.Layer L, int n, int pos) {
            int kvDim = c.kvDim();
            matmul(L.wq, xb, q, 0, n, false);
            matmul(L.wk, xb, kStage, 0, n, false);
            matmul(L.wv, xb, vStage, 0, n, false);
            if (L.bq != null) {
                addBias(q, 0, L.bq, n);
                addBias(kStage, 0, L.bk, n);
                addBias(vStage, 0, L.bv, n);
            }
            normRope(q, 0, c.qDim(), c.heads(), L.qNorm, n, pos, rope);
            normRope(kStage, 0, kvDim, c.kvHeads(), L.kNorm, n, pos, rope);
            kv.store(layer, pos, kStage, vStage, n);
            kv.attention(pool, layer, n, pos, q, att, c.windowFor(layer), c.attnScale(), c.attnSoftcap());
        }
    }

    private void addPlain(QTensor proj, float[] in, float[] norm, int n) {
        matmul(proj, in, x, 0, n, true);
    }

    /** Glm4 / Gemma: {@code x += rmsnorm(proj(in)) * norm}; {@link #xb} is free as the temporary at both call sites. */
    private void addPostNormed(QTensor proj, float[] in, float[] norm, int n) {
        matmul(proj, in, xb, 0, n, false);
        addNormed(xb, norm, n);
    }

    /** Dense gated MLP: {@code x += down(act(gate(xb)) * up(xb))} through the layer's residual rule. */
    private final class DenseGlu implements Ffn {
        private final boolean gelu;
        private final Residual residual;

        DenseGlu(boolean gelu, Residual residual) {
            this.gelu = gelu;
            this.residual = residual;
        }

        @Override
        public void run(ModelWeights.Layer L, int n) {
            matmul(L.wGateUp, xb, gateUp, 0, n, false);
            if (gelu) {
                geluMul(n);
            } else {
                siluMul(n);
            }
            residual.add(L.wDown, hidden, L.postMlpNorm, n);
        }
    }

    /** DeepSeek: the routed experts plus a dense SwiGLU of every token over the shared experts. */
    private void moeWithShared(ModelWeights.Layer L, int n) {
        moe(L, n);
        int sh = c.deepseek().nShared() * c.moeDim();
        matmul(L.mla.sharedGateUp, xb, gateUp, 0, n, false);
        siluMul(n, sh);
        matmul(L.mla.sharedDown, hidden, x, 0, n, true);
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
                + (DETERMINISTIC ? ", deterministic" : "") + (VectorKernels.FASTMATH ? ", fastmath" : "")
                + ", " + pool.threads() + " threads)";
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
        for (int t = 0; t < n; t++) {
            w.embed.dequantizeRow(tokens[off + t], x, t * dim);
        }
        if (c.embedScale() != 1f) {
            float es = c.embedScale();
            pool.forRange(n, 1, (t0, t1) -> {
                for (int i = t0 * dim, m = t1 * dim; i < m; i++) {
                    x[i] *= es;
                }
            });
        }
        for (int l = 0; l < c.layers(); l++) {
            ModelWeights.Layer L = w.layers[l];
            LayerPlan p = plans[l];
            rmsnorm(x, xb, L.attnNorm, n, dim);
            p.attention().run(l, L, n, pos);
            p.residual().add(L.wo, att, L.postAttnNorm, n);
            rmsnorm(x, xb, L.ffnNorm, n, dim);
            p.ffn().run(L, n);
        }
        if (logits != null) {
            rmsnormRow(x, (n - 1) * dim, xb, 0, w.finalNorm, dim);
            // the caller sizes logits by cfg.vocabSize(); the LM head may have fewer (or, padded, more) rows
            int rows = Math.min(w.lmHead.rows, logits.length);
            mm.mul(w.lmHead, rows, xb, logits, 0, 1, false);
            float cap = c.finalSoftcap();
            if (cap > 0) {
                pool.forRange(rows, 4096, (i0, i1) -> {
                    if (VectorKernels.FASTMATH) {
                        VectorKernels.softcapFast(logits, i0, i1 - i0, cap);
                        return;
                    }
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

    /** x[t] += rmsnorm(y[t]) * weight for n rows, parallel over tokens; y is overwritten with the normalized rows. */
    private void addNormed(float[] y, float[] weight, int n) {
        int dim = c.dim();
        pool.forRange(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                rmsnormRow(y, t * dim, y, t * dim, weight, dim);
                for (int i = t * dim, m = i + dim; i < m; i++) {
                    x[i] += y[i];
                }
            }
        });
    }

    void rmsnormRow(float[] in, int io, float[] out, int oo, float[] weight, int width) {
        VectorKernels.rmsnormRow(in, io, out, oo, weight, width, c.normEps());
    }

    /** {@code out[t, r] (+)= dot(W[r, :], in[t, :])} for {@code t < n}: {@link CpuMatmul#mul}. */
    void matmul(QTensor wt, float[] in, float[] out, int outOff, int n, boolean acc) {
        mm.mul(wt, wt.rows, in, out, outOff, n, acc);
    }

    /** buf[off + t*bias.length + i] += bias[i] for n rows (Qwen2 q/k/v bias), parallel over tokens. */
    private void addBias(float[] buf, int off, float[] bias, int n) {
        pool.forRange(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                int o = off + t * bias.length;
                for (int i = 0; i < bias.length; i++) {
                    buf[o + i] += bias[i];
                }
            }
        });
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
            routeDeepseek(L, xo, e, topK, 0, true);
        } else {
            route(L.router, xo, e, topK, scratch.get(), 0, true);
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
                moeHidden[hi + i] = VectorKernels.silu(g) * moeGateUp[gi + m + i];
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

        // one token per thread; each routes in its own scratch
        pool.forRange(n, 1, (t0, t1) -> {
            float[] logits = scratch.get();
            for (int t = t0; t < t1; t++) {
                if (L.mla != null) {
                    routeDeepseek(L, t * dim, e, topK, t * topK, false);
                } else {
                    route(L.router, t * dim, e, topK, logits, t * topK, false);
                }
            }
        });

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
                moeHidden[s * m + j] = VectorKernels.silu(g) * moeGateUp[gi + m + j];
            }
        });

        expertMatmul(L.expDown, dim, used, moeHidden, m, true, moeDown, dim);

        pool.forRange(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                int s0 = t * topK;
                VectorKernels.combine(routeWeight, s0, topK, moeDown, s0 * dim, dim, x, t * dim, dim);
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
     * Router logits of the token at {@code xb[xo..]}: {@code logits[i] = dot(router[i], xb[t])} for the {@code e}
     * experts, parallel over experts for a single token (decode) or serial when the caller already runs one token
     * per thread (prefill).
     */
    private void routerLogits(float[] router, int xo, int e, float[] logits, boolean parallel) {
        int dim = c.dim();
        if (parallel) {
            pool.forRange(e, 4, (i0, i1) -> {
                for (int i = i0; i < i1; i++) {
                    logits[i] = VectorKernels.dot(router, i * dim, xb, xo, dim);
                }
            });
            return;
        }
        for (int i = 0; i < e; i++) {
            logits[i] = VectorKernels.dot(router, i * dim, xb, xo, dim);
        }
    }

    /**
     * DeepSeek router for the token at {@code xb[xo..]} (see {@link DeepSeekRouter}): fills
     * {@code assign[ao..ao+topK)} / {@code routeWeight[ao..]}. Works in the calling thread's own buffers
     * ({@link #dsBuf}: logits, the router's working space, and the top-k ids / weights), so tokens can be routed
     * in parallel.
     */
    private void routeDeepseek(ModelWeights.Layer L, int xo, int e, int topK, int ao, boolean parallel) {
        DsBuf b = dsBuf.get();
        routerLogits(L.router, xo, e, b.logits, parallel);
        DeepSeekRouter.select(b.logits, L.mla.routerBias, c.deepseek(), topK, c.normTopK(), b.idx, b.weight, b.work);
        System.arraycopy(b.idx, 0, assign, ao, topK);
        System.arraycopy(b.weight, 0, routeWeight, ao, topK);
    }

    /** Per-thread working space of {@link #routeDeepseek}; {@code DeepSeekRouter.select} sizes the experts by {@code logits.length}. */
    private static final class DsBuf {
        final float[] logits;
        final float[] work;
        final int[] idx;
        final float[] weight;

        DsBuf(int e, int topK) {
            logits = new float[e];
            work = new float[e];
            idx = new int[topK];
            weight = new float[topK];
        }
    }

    /**
     * Router for the token at {@code xb[xo..]}: fills {@code assign[ao..ao+topK)} / {@code routeWeight[ao..]}
     * (same rule as transformers' Qwen3MoeSparseMoeBlock). {@code logits} is scratch of at least {@code e} floats.
     */
    private void route(float[] router, int xo, int e, int topK, float[] logits, int ao, boolean parallel) {
        routerLogits(router, xo, e, logits, parallel);
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < e; i++) {
            max = Math.max(max, logits[i]);
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
                hidden[t * hdim + j] = VectorKernels.silu(gateUp[gi + j]) * gateUp[gi + hdim + j];
            }
        });
    }

    @Override
    public void close() {
        pool.close();
    }
}
