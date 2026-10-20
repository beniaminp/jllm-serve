package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.AttnWeights;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.DecoderLayer;
import org.robomiri.jllm.model.DeepSeekRouter;
import org.robomiri.jllm.model.GemmaMath;
import org.robomiri.jllm.model.InterleavedRope;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.LayerSpec;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.MlpWeights;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.Rope;
import org.robomiri.jllm.model.RopeTable;
import org.robomiri.jllm.model.Router;

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
    private final Decoder arch;
    private final Architecture.Core c;
    private final int ctx;
    private final int maxBatch;
    final ThreadPool pool;
    /** One table per {@link Decoder#ropeTables()} entry. */
    private final Rope[] ropes;

    /** The key/value cache of the grouped-query layers; null for a latent-attention model. */
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
    private final float[] moeDown;
    private final int[] expertStart;
    private final int[] expertFill;
    private final int[] usedExperts;
    private final int[] slotOrder;

    /** The quantized matmul (float and int8 paths) and the thread-local float scratch, shared with {@link MlaCpu}. */
    final CpuMatmul mm;
    final ThreadLocal<float[]> scratch;
    /** What each layer does (attention, residual, feed-forward), resolved once from its spec and weights. */
    private final LayerPlan[] plans;

    /**
     * @param int8Activations quantize activations to int8 per 32-block and use integer dot products for the
     *                        single-token (decode / MoE) matmuls instead of dequantizing weights to float
     *                        (Q4_0 and Q8_0 weights; other formats keep the float path)
     */
    public CpuBackend(ModelWeights w, int ctx, int maxBatch, int threads, KvDtype kvDtype, boolean int8Activations) {
        this.w = w;
        this.arch = w.decoder();
        this.c = arch.core();
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        this.pool = new ThreadPool(threads);
        ropes = new Rope[arch.ropeTables().size()];
        for (int i = 0; i < ropes.length; i++) {
            ropes[i] = new Rope(arch.ropeTables().get(i), ctx);
        }
        if (ctx > c.maxPositions()) {
            System.err.println("warning: context " + ctx + " exceeds the model's max_position_embeddings ("
                    + c.maxPositions() + "); RoPE positions beyond the training length are extrapolated");
        }
        Attention a = arch.attention();
        switch (a) {
            case Attention.Gqa g -> {
                kv = KvCache.create(kvDtype, c.layers(), g, ctx);
                mla = null;
                kStage = new float[maxBatch * g.kvDim()];
                vStage = new float[maxBatch * g.kvDim()];
            }
            case Attention.Mla m -> {
                kv = null;
                mla = new MlaCpu(this, w, m, ropes[m.ropeTable()], ctx, maxBatch, kvDtype);
                kStage = null; // MLA keeps its own cache and staging
                vStage = null;
            }
        }
        x = new float[maxBatch * c.dim()];
        xb = new float[maxBatch * c.dim()];
        q = new float[maxBatch * a.qDim()];
        att = new float[maxBatch * Math.max(a.qDim(), a.outDim())];
        gateUp = new float[maxBatch * 2 * arch.maxMlpDim()];
        hidden = new float[maxBatch * arch.maxMlpDim()];
        if (arch.anyMoe()) {
            int slots = maxBatch * arch.maxTopK();
            assign = new int[slots];
            routeWeight = new float[slots];
            moeGateUp = new float[slots * 2 * arch.maxMoeDim()];
            moeHidden = new float[slots * arch.maxMoeDim()];
            moeDown = new float[slots * c.dim()];
            expertStart = new int[arch.maxExperts() + 1];
            expertFill = new int[arch.maxExperts()];
            usedExperts = new int[arch.maxExperts()];
            slotOrder = new int[slots];
        } else {
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
        int scratchLen = Math.max(Math.max(ctx, arch.maxExperts()), 4 * CpuMatmul.maxCols(w));
        mm = new CpuMatmul(pool, int8Activations, DETERMINISTIC, scratchLen);
        scratch = mm.scratch;
        plans = new LayerPlan[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            plans[l] = plan(arch.layers().get(l), w.decoderLayer(l));
        }
    }

    // ------------------------------------------------------------------ layer plans

    /**
     * One decoder layer as its weights plus three strategy objects, so {@link #forward} contains no per-layer
     * dispatch on the family's optional pieces: attention (GQA over the {@link KvCache}, or DeepSeek MLA), how a
     * projection joins the residual stream (plain add, or Glm4 / Gemma post-norm add), and the feed-forward
     * (dense GLU with SiLU or GELU, routed experts, or routed plus shared experts). Each is picked by a switch
     * over the layer's {@link LayerSpec}; every implementation runs exactly the statements the straight-line
     * code ran before, in the same order.
     */
    private record LayerPlan(DecoderLayer weights, AttnStep attention, Residual residual, FfnStep ffn) {
    }

    /** Reads the normalized hidden states {@link #xb}, writes the attention output {@link #att}. */
    private interface AttnStep {
        void run(int layer, int n, int pos);
    }

    /** {@code x += proj(in)}, or {@code x += rmsnorm(proj(in)) * norm} where the family normalizes the output. */
    private interface Residual {
        void add(QTensor proj, float[] in, float[] norm, int n);
    }

    /** Reads {@link #xb}, adds the feed-forward output to {@link #x}. */
    private interface FfnStep {
        void run(int n);
    }

    private LayerPlan plan(LayerSpec spec, DecoderLayer L) {
        AttnStep attention = switch (spec.attn()) {
            case Attention.Gqa g -> new GqaStep(g, L.gqa());
            case Attention.Mla m -> mla::run;
        };
        Residual residual = spec.norms().post() ? this::addPostNormed : this::addPlain;
        FfnStep ffn = switch (spec.mlp()) {
            case Mlp.Dense d -> new DenseGlu(d, L.dense(), residual, L.postMlpNorm());
            case Mlp.Moe m -> new MoeFfn(m, L.moe());
        };
        return new LayerPlan(L, attention, residual, ffn);
    }

    /** Grouped-query attention: q/k/v projections (+ bias), QK-norm and RoPE, store into the cache, attend. */
    private final class GqaStep implements AttnStep {
        private final Attention.Gqa g;
        private final AttnWeights.Gqa w;
        private final RopeTable table;
        private final Rope rope;

        GqaStep(Attention.Gqa g, AttnWeights.Gqa w) {
            this.g = g;
            this.w = w;
            this.table = arch.ropeTables().get(g.ropeTable());
            this.rope = ropes[g.ropeTable()];
        }

        @Override
        public void run(int layer, int n, int pos) {
            matmul(w.wq(), xb, q, 0, n, false);
            matmul(w.wk(), xb, kStage, 0, n, false);
            matmul(w.wv(), xb, vStage, 0, n, false);
            if (w.bq() != null) {
                addBias(q, 0, w.bq(), n);
                addBias(kStage, 0, w.bk(), n);
                addBias(vStage, 0, w.bv(), n);
            }
            normRope(q, 0, g.qDim(), g.heads(), w.qNorm(), n, pos);
            normRope(kStage, 0, g.kvDim(), g.kvHeads(), w.kNorm(), n, pos);
            kv.store(layer, pos, kStage, vStage, n);
            kv.attention(pool, layer, n, pos, q, att, g.window(), g.scale(), g.softcap());
        }

        /** Per-head RMSNorm (if {@code norm != null}) followed by rotary embedding, in place. */
        private void normRope(float[] v, int base, int stride, int nHeads, float[] norm, int n, int pos) {
            if (n == 1) {
                normRopeHeads(v, base, stride, nHeads, norm, pos, 0, nHeads);
                return;
            }
            pool.forRange(n * nHeads, 4, (i0, i1) -> normRopeHeads(v, base, stride, nHeads, norm, pos, i0, i1));
        }

        /** {@link #normRope} over the flat (token, head) indices {@code [i0, i1)}. */
        private void normRopeHeads(float[] v, int base, int stride, int nHeads, float[] norm, int pos, int i0, int i1) {
            int hd = g.headDim();
            int half = table.rotaryDim() / 2; // rope table width; = hd / 2 unless partial_rotary_factor < 1
            boolean interleaved = table.pairing() == RopeTable.Pairing.INTERLEAVED;
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
                    InterleavedRope.rotate(v, o, table.rotaryDim(), cos, sin, ro);
                } else {
                    VectorKernels.rope(v, o, half, cos, sin, ro);
                }
            }
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
    private final class DenseGlu implements FfnStep {
        private final MlpWeights.Dense w;
        private final int hdim;
        private final boolean gelu;
        private final Residual residual;
        private final float[] postMlpNorm;

        DenseGlu(Mlp.Dense d, MlpWeights.Dense w, Residual residual, float[] postMlpNorm) {
            this.w = w;
            this.hdim = d.hidden();
            this.gelu = d.act() == Mlp.Activation.GELU_TANH;
            this.residual = residual;
            this.postMlpNorm = postMlpNorm;
        }

        @Override
        public void run(int n) {
            matmul(w.gateUp(), xb, gateUp, 0, n, false);
            if (gelu) {
                geluMul(n, hdim);
            } else {
                siluMul(n, hdim);
            }
            residual.add(w.down(), hidden, postMlpNorm, n);
        }
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
        run(tokens, off, n, pos, logits, 0);
    }

    @Override
    public void forwardAll(int[] tokens, int off, int n, int pos, float[] logitsOut, int stride) {
        run(tokens, off, n, pos, logitsOut, stride);
    }

    /** {@code stride == 0}: logits of the last token only, else of every token (row {@code i} at {@code i * stride}). */
    private void run(int[] tokens, int off, int n, int pos, float[] logits, int stride) {
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
            LayerPlan p = plans[l];
            DecoderLayer L = p.weights();
            rmsnorm(x, xb, L.attnNorm(), n, dim);
            p.attention().run(l, n, pos);
            p.residual().add(L.wo(), att, L.postAttnNorm(), n);
            rmsnorm(x, xb, L.ffnNorm(), n, dim);
            p.ffn().run(n);
        }
        if (logits != null && stride == 0) {
            rmsnormRow(x, (n - 1) * dim, xb, 0, w.finalNorm, dim);
            // the caller sizes logits by vocabSize; the LM head may have fewer (or, padded, more) rows
            int rows = Math.min(w.lmHead.rows, logits.length);
            mm.mul(w.lmHead, rows, xb, logits, 0, 1, false);
            softcap(logits, 0, rows);
        } else if (logits != null) {
            lmHeadAll(n, logits, stride);
        }
    }

    /** The LM head over all {@code n} rows of {@link #x}, row {@code i} stored at {@code out[i * stride ..]}. */
    private void lmHeadAll(int n, float[] out, int stride) {
        rmsnorm(x, xb, w.finalNorm, n, c.dim());
        int headRows = w.lmHead.rows;
        int rows = Math.min(headRows, stride);
        if (headRows == stride) { // the usual case: the matmul writes the final layout directly
            mm.mul(w.lmHead, rows, xb, out, 0, n, false);
        } else {
            float[] tmp = new float[n * headRows];
            mm.mul(w.lmHead, rows, xb, tmp, 0, n, false);
            for (int i = 0; i < n; i++) {
                System.arraycopy(tmp, i * headRows, out, i * stride, rows);
            }
        }
        for (int i = 0; i < n; i++) {
            softcap(out, i * stride, rows);
        }
    }

    private void softcap(float[] logits, int base, int rows) {
        float cap = c.finalSoftcap();
        if (cap > 0) {
            pool.forRange(rows, 4096, (i0, i1) -> {
                if (VectorKernels.FASTMATH) {
                    VectorKernels.softcapFast(logits, base + i0, i1 - i0, cap);
                    return;
                }
                for (int i = base + i0; i < base + i1; i++) {
                    logits[i] = GemmaMath.softcap(logits[i], cap);
                }
            });
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

    // ------------------------------------------------------------------ mixture of experts

    /**
     * Mixture of experts: for each token the router picks the top-k experts and their weights, each expert's
     * SwiGLU MLP runs on the token and the results are added to x weighted by the router; DeepSeek layers add a
     * dense SwiGLU of every token over the shared experts.
     */
    private final class MoeFfn implements FfnStep {
        /** Fills {@code assign[ao..ao+topK)} / {@code routeWeight[ao..]} for the token at {@code xb[xo..]}. */
        private interface Route {
            void route(int xo, int ao, boolean parallel);
        }

        private final Mlp.Moe spec;
        private final MlpWeights.Moe w;
        private final Route route;
        private final int sharedDim;

        MoeFfn(Mlp.Moe spec, MlpWeights.Moe w) {
            this.spec = spec;
            this.w = w;
            this.sharedDim = spec.shared() == null ? 0 : spec.shared().hidden();
            this.route = switch (spec.router()) {
                case Router.Softmax s -> (xo, ao, parallel) -> routeSoftmax(xo, scratch.get(), ao, parallel);
                case Router.DeepSeek r -> {
                    // per-thread working space (logits, the router's scratch, the top-k ids / weights), so
                    // tokens can be routed in parallel
                    ThreadLocal<DsBuf> buf = ThreadLocal.withInitial(() -> new DsBuf(spec.experts(), spec.topK()));
                    yield (xo, ao, parallel) -> routeDeepseek(r, buf.get(), xo, ao, parallel);
                }
            };
        }

        @Override
        public void run(int n) {
            if (n > 1 && !DETERMINISTIC) {
                moeBatched(n);
            } else {
                for (int t = 0; t < n; t++) {
                    moeToken(t);
                }
            }
            if (sharedDim > 0) {
                matmul(w.shared().gateUp(), xb, gateUp, 0, n, false);
                siluMul(n, sharedDim);
                matmul(w.shared().down(), hidden, x, 0, n, true);
            }
        }

        /** The single-token MoE layer for token {@code t}: {@code x[t] += Σ_k w_k · expert_k(xb[t])}. */
        private void moeToken(int t) {
            int dim = c.dim();
            int topK = spec.topK();
            int m = spec.moeDim();
            int gu = 2 * m;
            int xo = t * dim;
            route.route(xo, 0, true);
            QTensor expGateUp = w.expGateUp();
            QTensor expDown = w.expDown();
            // all selected experts' gate/up rows in one parallel loop
            if (mm.useInt8(expGateUp)) {
                short[] xq = mm.quantizeInt8(xb, xo, dim);
                float[] xs = mm.xs();
                pool.forRange(topK * gu, 32, (i0, i1) -> {
                    for (int i = i0; i < i1; i++) {
                        int k = i / gu;
                        int r = i - k * gu;
                        moeGateUp[i] = mm.int8Dot(expGateUp, assign[k] * gu + r, xq, xs, 0);
                    }
                });
            } else {
                pool.forRange(topK * gu, 32, (i0, i1) -> {
                    for (int i = i0; i < i1; i++) {
                        int k = i / gu;
                        int r = i - k * gu;
                        moeGateUp[i] = VectorKernels.dotRow(expGateUp, assign[k] * gu + r, xb, xo);
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
            if (mm.useInt8(expDown)) {
                short[] xq = mm.quantizeInt8(moeHidden, 0, topK * m);   // slot k starts at block offset k * m
                float[] xs = mm.xs();
                pool.forRange(dim, 8, (r0, r1) -> {
                    for (int r = r0; r < r1; r++) {
                        float sum = 0;
                        for (int k = 0; k < topK; k++) {
                            sum += routeWeight[k] * mm.int8Dot(expDown, assign[k] * dim + r, xq, xs, k * m);
                        }
                        x[xo + r] += sum;
                    }
                });
            } else {
                pool.forRange(dim, 8, (r0, r1) -> {
                    for (int r = r0; r < r1; r++) {
                        float sum = 0;
                        for (int k = 0; k < topK; k++) {
                            sum += routeWeight[k] * VectorKernels.dotRow(expDown, assign[k] * dim + r, moeHidden, k * m);
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
        private void moeBatched(int n) {
            int dim = c.dim();
            int e = spec.experts();
            int topK = spec.topK();
            int m = spec.moeDim();
            int gu = 2 * m;
            int slots = n * topK;

            // one token per thread; each routes in its own scratch
            pool.forRange(n, 1, (t0, t1) -> {
                for (int t = t0; t < t1; t++) {
                    route.route(t * dim, t * topK, false);
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

            expertMatmul(w.expGateUp(), gu, used, xb, dim, false, moeGateUp, gu);

            pool.forRange(slots * m, 256, (i0, i1) -> {
                for (int i = i0; i < i1; i++) {
                    int s = i / m;
                    int j = i - s * m;
                    int gi = s * gu;
                    float g = moeGateUp[gi + j];
                    moeHidden[s * m + j] = VectorKernels.silu(g) * moeGateUp[gi + m + j];
                }
            });

            expertMatmul(w.expDown(), dim, used, moeHidden, m, true, moeDown, dim);

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
            int topK = spec.topK();
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
         * Router logits of the token at {@code xb[xo..]}: {@code logits[i] = dot(router[i], xb[t])} for the experts,
         * parallel over experts for a single token (decode) or serial when the caller already runs one token per
         * thread (prefill).
         */
        private void routerLogits(int xo, float[] logits, boolean parallel) {
            int dim = c.dim();
            int e = spec.experts();
            float[] router = w.router();
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

        /** The DeepSeek rule ({@link DeepSeekRouter}) in the calling thread's own buffers. */
        private void routeDeepseek(Router.DeepSeek r, DsBuf b, int xo, int ao, boolean parallel) {
            int topK = spec.topK();
            routerLogits(xo, b.logits, parallel);
            DeepSeekRouter.select(b.logits, w.routerBias(), r, topK, spec.normTopK(), b.idx, b.weight, b.work);
            System.arraycopy(b.idx, 0, assign, ao, topK);
            System.arraycopy(b.weight, 0, routeWeight, ao, topK);
        }

        /**
         * Softmax over the router logits, the top-k probabilities, optionally renormalized (same rule as transformers'
         * Qwen3MoeSparseMoeBlock). {@code logits} is scratch of at least {@code experts} floats.
         */
        private void routeSoftmax(int xo, float[] logits, int ao, boolean parallel) {
            int e = spec.experts();
            int topK = spec.topK();
            routerLogits(xo, logits, parallel);
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
            if (spec.normTopK() && selected > 0) {
                for (int k = 0; k < topK; k++) {
                    routeWeight[ao + k] /= selected;
                }
            }
        }
    }

    /** Per-thread working space of the DeepSeek router; {@code DeepSeekRouter.select} sizes the experts by {@code logits.length}. */
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

    /** hidden = gelu_tanh(gate) * up, Gemma's GeGLU; parallel over flat (token, i) indices like {@link #siluMul}. */
    private void geluMul(int n, int hdim) {
        pool.forRange(n * hdim, 512, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int t = i / hdim;
                int j = i - t * hdim;
                int gi = t * 2 * hdim;
                hidden[t * hdim + j] = GemmaMath.geluTanh(gateUp[gi + j]) * gateUp[gi + hdim + j];
            }
        });
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
