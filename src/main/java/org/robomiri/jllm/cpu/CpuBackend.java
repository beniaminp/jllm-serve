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
    private final boolean ownsPool;
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
    private final MoeExperts experts;

    /** The quantized matmul (float and int8 paths) and the thread-local float scratch, shared with {@link MlaCpu}. */
    final CpuMatmul mm;
    final ThreadLocal<float[]> scratch;
    /** What each layer does (attention, residual, feed-forward), resolved once from its spec and weights. */
    private final LayerPlan[] plans;

    /**
     * Resolves {@code backend.cpu.int8-activations}: {@code true} / {@code false} as given, {@code auto} by timing both
     * paths on {@code w} once (see {@link CpuMatmul#int8Faster}).
     */
    public static boolean int8Activations(String setting, ModelWeights w, int threads) {
        return switch (setting.trim().toLowerCase(java.util.Locale.ROOT)) {
            case "true" -> true;
            case "false" -> false;
            case "auto" -> CpuMatmul.int8Faster(w, threads);
            default -> throw new IllegalArgumentException("backend.cpu.int8-activations must be auto, true or false, got '"
                    + setting + "'");
        };
    }

    /**
     * @param int8Activations quantize activations to int8 per 32-block and use integer dot products for the
     *                        single-token (decode / MoE) matmuls instead of dequantizing weights to float
     *                        (Q4_0 and Q8_0 weights; other formats keep the float path)
     */
    public CpuBackend(ModelWeights w, int ctx, int maxBatch, int threads, KvDtype kvDtype, boolean int8Activations) {
        this(w, ctx, maxBatch, new ThreadPool(threads), true, kvDtype, int8Activations);
    }

    /**
     * On a thread pool lent by the caller (not closed with this backend): the slots of a batch group, which never
     * compute at the same time, share one pool instead of keeping one set of threads each.
     */
    public CpuBackend(ModelWeights w, int ctx, int maxBatch, ThreadPool pool, KvDtype kvDtype, boolean int8Activations) {
        this(w, ctx, maxBatch, pool, false, kvDtype, int8Activations);
    }

    private CpuBackend(ModelWeights w, int ctx, int maxBatch, ThreadPool pool, boolean ownsPool, KvDtype kvDtype,
                       boolean int8Activations) {
        this.w = w;
        this.arch = w.decoder();
        this.c = arch.core();
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        this.pool = pool;
        this.ownsPool = ownsPool;
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
                kv = KvCache.create(kvDtype, c.layers(), g, ctx, maxBatch, KvDtype.q4Recent());
                mla = null;
                kStage = new float[maxBatch * g.kvDim()];
                vStage = new float[maxBatch * g.kvDim()];
            }
            case Attention.Mla m -> {
                kv = null;
                if (kvDtype == KvDtype.Q4) {
                    System.err.println("note: backend.kv-dtype=q4 is not implemented for DeepSeek's latent cache, using q8");
                }
                mla = new MlaCpu(this, w, m, ropes[m.ropeTable()], ctx, maxBatch, kvDtype == KvDtype.Q4 ? KvDtype.Q8 : kvDtype);
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
        } else {
            assign = null;
            routeWeight = null;
        }
        // attention scores (ctx), router logits (experts), or four dequantized rows of the widest tensor
        int scratchLen = Math.max(Math.max(ctx, arch.maxExperts()), 4 * CpuMatmul.maxCols(w));
        mm = new CpuMatmul(pool, int8Activations, DETERMINISTIC, scratchLen);
        scratch = mm.scratch;
        experts = arch.anyMoe() ? new MoeExperts(pool, mm, arch, maxBatch) : null;
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

        /**
         * One token per row, each row of another backend's conversation ({@link #forwardBatch}): row t belongs to
         * {@code owners[t]} at {@code positions[t]}. Null when the attention keeps no per-row state this way (MLA).
         */
        default boolean runRows(int layer, CpuBackend[] owners, int[] positions) {
            return false;
        }
    }

    /**
     * {@code x += proj(in) + bias}, or {@code x += rmsnorm(proj(in) + bias) * norm} where the family normalizes the
     * output; {@code bias} (o_proj's) is usually null.
     */
    private interface Residual {
        void add(QTensor proj, float[] bias, float[] in, float[] norm, int n);
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
            if (bidirectional && n > 1) {
                // every row attends to all rows of the picture: with all n keys stored, each is a "last row" at pos + n - 1
                int qDim = g.qDim();
                int outDim = arch.attention().outDim();
                float[] qRow = new float[qDim];
                float[] aRow = new float[outDim];
                for (int t = 0; t < n; t++) {
                    System.arraycopy(q, t * qDim, qRow, 0, qDim);
                    kv.attention(pool, layer, 1, pos + n - 1, qRow, aRow, g.window(), g.scale(), g.softcap());
                    System.arraycopy(aRow, 0, att, t * outDim, outDim);
                }
                return;
            }
            kv.attention(pool, layer, n, pos, q, att, g.window(), g.scale(), g.softcap());
        }

        @Override
        public boolean runRows(int layer, CpuBackend[] owners, int[] positions) {
            int n = owners.length;
            matmul(w.wq(), xb, q, 0, n, false);
            matmul(w.wk(), xb, kStage, 0, n, false);
            matmul(w.wv(), xb, vStage, 0, n, false);
            if (w.bq() != null) {
                addBias(q, 0, w.bq(), n);
                addBias(kStage, 0, w.bk(), n);
                addBias(vStage, 0, w.bv(), n);
            }
            int qDim = g.qDim();
            int kvDim = g.kvDim();
            int outDim = arch.attention().outDim();
            for (int t = 0; t < n; t++) {
                // row t at its own position, into its own cache, through that backend's (idle) staging rows
                CpuBackend o = owners[t];
                int pos = positions[t];
                normRopeHeads(q, t * qDim, qDim, g.heads(), w.qNorm(), pos, 0, g.heads());
                normRopeHeads(kStage, t * kvDim, kvDim, g.kvHeads(), w.kNorm(), pos, 0, g.kvHeads());
                System.arraycopy(kStage, t * kvDim, o.kStage, 0, kvDim);
                System.arraycopy(vStage, t * kvDim, o.vStage, 0, kvDim);
                System.arraycopy(q, t * qDim, o.q, 0, qDim);
                o.kv.store(layer, pos, o.kStage, o.vStage, 1);
                o.kv.attention(pool, layer, 1, pos, o.q, o.att, g.window(), g.scale(), g.softcap());
                System.arraycopy(o.att, 0, att, t * outDim, outDim);
            }
            return true;
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

    /** {@code x += proj(in)}; with a bias {@code x += (proj(in) + bias)}, the projection staged in {@link #xb}. */
    private void addPlain(QTensor proj, float[] bias, float[] in, float[] norm, int n) {
        if (bias == null) {
            matmul(proj, in, x, 0, n, true);
            return;
        }
        int dim = c.dim();
        matmul(proj, in, xb, 0, n, false);
        pool.forRange(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                for (int i = 0, o = t * dim; i < dim; i++, o++) {
                    x[o] += xb[o] + bias[i];
                }
            }
        });
    }

    /** Glm4 / Gemma: {@code x += rmsnorm(proj(in) + bias) * norm}; {@link #xb} is free as the temporary at both call sites. */
    private void addPostNormed(QTensor proj, float[] bias, float[] in, float[] norm, int n) {
        matmul(proj, in, xb, 0, n, false);
        if (bias != null) {
            addBias(xb, 0, bias, n);
        }
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
            residual.add(w.down(), null, hidden, postMlpNorm, n);
        }
    }

    @Override
    public String name() {
        return "cpu (" + VectorKernels.describe() + (mm.int8() ? ", " + mm.describeInt8() : "")
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
        run(tokens, null, off, n, pos, logits, 0);
    }

    // ---- images (vision encoders feed embeddings; Qwen-VL needs its own rotary positions)

    /** First slot whose rotary rows are not the plain ones ({@link #setRopePositions}); MAX_VALUE = none. */
    private int ropeCustomFrom = Integer.MAX_VALUE;

    @Override
    public boolean supportsEmbeds() {
        return true;
    }

    @Override
    public void forwardEmbeds(float[] embeds, int off, int n, int pos, float[][] deepstack, float[] logits) {
        this.deepstack = deepstack;
        try {
            run(null, embeds, off, n, pos, logits, 0);
        } finally {
            this.deepstack = null;
        }
    }

    /** Qwen3-VL: per-layer features added to the hidden state after the first layers of the current {@link #run}. */
    private float[][] deepstack;

    /** The current {@link #run} is a picture whose rows attend to each other both ways (Gemma 3). */
    private boolean bidirectional;

    @Override
    public boolean supportsBidirectionalSpan() {
        return true;
    }

    @Override
    public void forwardEmbedsBidirectional(float[] embeds, int off, int n, int pos, float[] logits) {
        bidirectional = true;
        try {
            forwardEmbeds(embeds, off, n, pos, null, logits);
        } finally {
            bidirectional = false;
        }
    }

    @Override
    public boolean setRopePositions(int from, org.robomiri.jllm.model.RopePositions p) {
        for (int i = 0; i < ropes.length; i++) {
            ropes[i].refill(arch.ropeTables().get(i), ctx, from, p);
        }
        ropeCustomFrom = p != null ? Math.min(ropeCustomFrom, from) : (from <= ropeCustomFrom ? Integer.MAX_VALUE : ropeCustomFrom);
        return true;
    }

    @Override
    public boolean ropePositionsCustom() {
        return ropeCustomFrom != Integer.MAX_VALUE;
    }

    @Override
    public void forwardAll(int[] tokens, int off, int n, int pos, float[] logitsOut, int stride) {
        run(tokens, null, off, n, pos, logitsOut, stride);
    }

    @Override
    public boolean forwardBatch(Backend[] members, int[] tokens, int[] positions, float[][] logits) {
        int n = members.length;
        if (n < 1 || n > maxBatch || kv == null) {
            return false;
        }
        CpuBackend[] owners = new CpuBackend[n];
        for (int t = 0; t < n; t++) {
            if (!(members[t] instanceof CpuBackend o) || o.w != w || o.kv == null || positions[t] >= o.ctx
                    || o.ropeCustomFrom != Integer.MAX_VALUE) { // a row with its own rotary table cannot share the lender's
                return false;
            }
            owners[t] = o;
        }
        int dim = c.dim();
        for (int t = 0; t < n; t++) {
            w.embed.dequantizeRow(tokens[t], x, t * dim);
        }
        if (c.embedScale() != 1f) {
            for (int i = 0; i < n * dim; i++) {
                x[i] *= c.embedScale();
            }
        }
        for (int l = 0; l < c.layers(); l++) {
            LayerPlan p = plans[l];
            DecoderLayer L = p.weights();
            rmsnorm(x, xb, L.attnNorm(), n, dim);
            if (!p.attention().runRows(l, owners, positions)) {
                throw new IllegalStateException("layer " + l + " cannot attend per row"); // kv != null: GQA only
            }
            p.residual().add(L.wo(), L.bo(), att, L.postAttnNorm(), n);
            rmsnorm(x, xb, L.ffnNorm(), n, dim);
            p.ffn().run(n);
        }
        int vocab = w.lmHead.rows;
        float[] all = new float[n * vocab];
        lmHeadAll(n, all, vocab);
        for (int t = 0; t < n; t++) {
            System.arraycopy(all, t * vocab, logits[t], 0, Math.min(vocab, logits[t].length));
        }
        return true;
    }

    /**
     * {@code stride == 0}: logits of the last token only, else of every token (row {@code i} at {@code i * stride}).
     * With {@code embeds != null} the input rows are those embeddings (rows {@code off ..}) and {@code tokens} is unused.
     */
    private void run(int[] tokens, float[] embeds, int off, int n, int pos, float[] logits, int stride) {
        if (n < 1 || n > maxBatch || pos + n > ctx) {
            throw new IllegalArgumentException("bad forward(n=" + n + ", pos=" + pos + ")");
        }
        int dim = c.dim();
        if (embeds != null) {
            System.arraycopy(embeds, off * dim, x, 0, n * dim); // image embeddings are used as they are
        } else {
            for (int t = 0; t < n; t++) {
                w.embed.dequantizeRow(tokens[off + t], x, t * dim);
            }
        }
        if (embeds == null && c.embedScale() != 1f) {
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
            p.residual().add(L.wo(), L.bo(), att, L.postAttnNorm(), n);
            rmsnorm(x, xb, L.ffnNorm(), n, dim);
            p.ffn().run(n);
            if (deepstack != null && l < deepstack.length) { // Qwen3-VL: x += this layer's picture features
                float[] ds = deepstack[l];
                int base = off * dim;
                pool.forRange(n * dim, 4096, (i0, i1) -> {
                    for (int i = i0; i < i1; i++) {
                        x[i] += ds[base + i];
                    }
                });
            }
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
            int topK = spec.topK();
            if (n > 1 && !DETERMINISTIC) {
                // one token per thread; each routes in its own scratch
                pool.forRange(n, 1, (t0, t1) -> {
                    for (int t = t0; t < t1; t++) {
                        route.route(t * c.dim(), t * topK, false);
                    }
                });
                experts.batched(w, spec.experts(), topK, spec.moeDim(), n, xb, assign, routeWeight, x);
            } else {
                for (int t = 0; t < n; t++) {
                    route.route(t * c.dim(), t * topK, true);
                    experts.token(w, topK, spec.moeDim(), t, xb, assign, routeWeight, x);
                }
            }
            if (sharedDim > 0) {
                matmul(w.shared().gateUp(), xb, gateUp, 0, n, false);
                siluMul(n, sharedDim);
                matmul(w.shared().down(), hidden, x, 0, n, true);
            }
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
        // parallel over (token, 512-wide piece) so a single token still spreads over the threads
        int pieces = (hdim + 511) / 512;
        pool.forRange(n * pieces, 1, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int t = i / pieces;
                int j = (i - t * pieces) * 512;
                int gi = t * 2 * hdim;
                VectorKernels.siluMul(gateUp, gi + j, gateUp, gi + hdim + j, hidden, t * hdim + j, Math.min(512, hdim - j));
            }
        });
    }

    @Override
    public String stateFormat() { // the same rows as MetalBackend writes
        return kv != null ? "kv " + kv.format() : mla != null && mla.format() != null ? "mla " + mla.format() : null;
    }

    @Override
    public void saveState(int n, java.nio.channels.WritableByteChannel out) throws java.io.IOException {
        if (kv != null) {
            kv.write(n, out);
        } else if (mla != null) {
            mla.write(n, out);
        } else {
            Backend.super.saveState(n, out);
        }
    }

    @Override
    public boolean canSaveWhileServing() {
        return stateFormat() != null;
    }

    /** {@link #saveState} only reads the cache arrays (rows below {@code n}): safe while later rows are written. */
    @Override
    public void saveStateWhileServing(int n, java.nio.channels.WritableByteChannel out) throws java.io.IOException {
        saveState(n, out);
    }

    @Override
    public void copyStateFrom(Backend other, int n) {
        if (!(other instanceof CpuBackend o) || stateFormat() == null || !stateFormat().equals(o.stateFormat())) {
            Backend.super.copyStateFrom(other, n);
            return;
        }
        if (kv != null) {
            kv.copyFrom(o.kv, n);
        } else {
            mla.copyFrom(o.mla, n);
        }
    }

    @Override
    public void loadState(int n, java.nio.channels.ReadableByteChannel in) throws java.io.IOException {
        loadState(n, n, in);
    }

    @Override
    public void loadState(int n, int stored, java.nio.channels.ReadableByteChannel in) throws java.io.IOException {
        if (kv != null) {
            kv.read(n, stored, in);
        } else if (mla != null) {
            mla.read(n, stored, in);
        } else {
            throw new UnsupportedOperationException(name() + " cannot load a saved cache");
        }
    }

    @Override
    public void close() {
        if (ownsPool) {
            pool.close();
        }
    }
}
