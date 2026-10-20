package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.AttnWeights;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.DecoderLayer;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.LayerSpec;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.MlpWeights;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.Rope;
import org.robomiri.jllm.model.RopeTable;
import org.robomiri.jllm.model.Router;

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
    private final Decoder arch;
    private final Architecture.Core c;
    private final int ctx;
    private final int maxBatch;
    final MetalKernels k;

    private String attnKernel;         // attention_t instantiation for headDim / attnGroup / cache type
    private int attnGroup;             // query heads per attention threadgroup
    private boolean flashDecode;       // flash decoding for long decode steps (decided by MetalTuner, dense attention only)
    private boolean flashPrefill;      // attention_fa for prefill chunks (decided by MetalTuner)
    private Metal.Buffer flashPart;
    // f16 KV cache (backend.kv-dtype=f16): matmul3's *_h variants write v straight into the half cache, k goes
    // through an f32 staging row and rope_h rounds it into the cache after the rotation, attention_h
    // reads half; see the "f16 KV cache" section of kernels.metal
    private final boolean kvHalf;
    private boolean fusedNormFits;     // see MetalKernels.fusedNormFits
    private Metal.Buffer kStage;

    private Metal.Buffer embedQ;
    private Metal.Buffer lmHeadQ;
    private LayerPlan[] plans;
    private Metal.Buffer finalNorm;
    /** One cos/sin pair per {@link Decoder#ropeTables()} entry. */
    Metal.Buffer[] ropeCos;
    Metal.Buffer[] ropeSin;
    private Metal.Buffer[] keyCache;
    private Metal.Buffer[] valCache;

    Metal.Buffer x;
    Metal.Buffer xb;
    Metal.Buffer q;
    Metal.Buffer att;
    Metal.Buffer hidden;
    Metal.Buffer scores;
    private Metal.Buffer logitsBuf;
    private Metal.Buffer logitsAllBuf; // n rows of logits for forwardAll, grown on demand
    private int logitsAllRows;
    // MoE (null for dense models)
    Metal.Buffer assign;
    Metal.Buffer routeWeight;
    private Metal.Buffer moeHidden;
    /** DeepSeek multi-head latent attention (own caches); null for other families. */
    private MlaMetal mla;

    /**
     * One decoder layer: its norm buffers, {@code o_proj}, and two strategy objects picked once from the layer's
     * {@link LayerSpec} (like {@code CpuBackend.LayerPlan}), so {@link #forward} contains no per-layer dispatch on
     * the family's optional pieces: the attention (grouped-query over the KV cache, or MLA) and the feed-forward
     * (dense GLU with SiLU or GELU, routed experts, routed plus shared experts). Each step runs exactly the
     * dispatches the straight-line code recorded before, in the same order.
     */
    private record LayerPlan(Metal.Buffer attnNorm, Metal.Buffer postAttnNorm, QTensor wo, Metal.Buffer woBuf,
                             AttnStep attention, FfnStep ffn) {
    }

    /** Reads {@code x} (normalizing it with the layer's attention norm), writes the attention output to {@link #att}. */
    private interface AttnStep {
        void run(int layer, int n, int pos, boolean fuseNorm);
    }

    /** Reads {@code x} (normalizing it with the layer's MLP norm), adds the feed-forward output to {@link #x}. */
    private interface FfnStep {
        void run(int layer, int n, boolean fuseNorm);
    }

    public MetalBackend(ModelWeights w, int ctx, int maxBatch) {
        this(w, ctx, maxBatch, (String) null);
    }

    public MetalBackend(ModelWeights w, int ctx, int maxBatch, String metalLib) {
        this(w, ctx, maxBatch, metalLib, KvDtype.F32);
    }

    public MetalBackend(ModelWeights w, int ctx, int maxBatch, String metalLib, KvDtype kvDtype) {
        this(w, ctx, maxBatch, open(w.decoder(), metalLib), kvDtype);
    }

    private static Metal open(Decoder d, String metalLib) {
        int hd = d.attention().headDim();
        if (hd % 32 != 0 || hd > 256) {
            throw new IllegalArgumentException("Metal backend needs headDim multiple of 32 and <= 256");
        }
        // the grouped attention instantiations (G = 2, 4, 8) are only compiled when the override asks for them
        return Metal.create(metalLib, attnGroupOverride(d.attention()) > 1 ? "#define JLLM_ATTN_GROUPS 1\n" : "");
    }

    /**
     * Query heads per attention threadgroup. Sharing K/V rows across the group (> 1) measured no faster on
     * Qwen3-0.6B (G=2) and slower on Qwen3-30B-A3B (G=4/8: 20-22 tok/s vs 25 with G=1) because only kvHeads
     * threadgroups run per decode token and each one's serial work grows with the group, so the default is 1;
     * -Djllm.metal.attnGroup=2|4|8 overrides (it must divide heads / kvHeads; headDim 64, 128 or 256 only).
     */
    private static int attnGroupOverride(Attention a) {
        if (!(a instanceof Attention.Gqa g)) {
            return 1; // MLA attends with its own instantiation
        }
        int group = g.heads() / g.kvHeads();
        int override = Integer.getInteger("jllm.metal.attnGroup", 0);
        int hd = g.headDim();
        boolean specialised = hd == 64 || hd == 128 || hd == 256;
        return specialised && override > 0 && override <= 8 && Integer.bitCount(override) == 1 && group % override == 0
                ? override : 1;
    }

    /** Owns {@code metal} from here on: it is closed again if anything below throws. */
    private MetalBackend(ModelWeights w, int ctx, int maxBatch, Metal metal, KvDtype kvDtype) {
        this.w = w;
        this.arch = w.decoder();
        this.c = arch.core();
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        this.kvHalf = kvDtype == KvDtype.F16;
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
        Attention a = arch.attention();
        // Attention specialised on headDim (64 / 128 / 256) and the group size, see attnGroupOverride
        attnGroup = attnGroupOverride(a);
        int hd = a.headDim();
        String vh = kvHalf ? "_h" : "";
        attnKernel = hd == 64 || hd == 128 || hd == 256 ? "attention_g" + attnGroup + "_d" + hd + vh : "attention" + vh;
        if (arch.maxExperts() > MetalConstants.MAX_EXPERTS) {
            throw new IllegalArgumentException("Metal router kernel supports at most " + MetalConstants.MAX_EXPERTS + " experts");
        }
        fusedNormFits = k.fusedNormFits(c.dim());

        embedQ = k.wrap(w.embed.data);
        lmHeadQ = w.lmHead == w.embed ? embedQ : k.wrap(w.lmHead.data);
        finalNorm = k.upload(w.finalNorm);
        int tables = arch.ropeTables().size();
        ropeCos = new Metal.Buffer[tables];
        ropeSin = new Metal.Buffer[tables];
        for (int t = 0; t < tables; t++) {
            Rope rope = new Rope(arch.ropeTables().get(t), ctx);
            ropeCos[t] = k.upload(rope.cos);
            ropeSin[t] = k.upload(rope.sin);
        }

        // None of the buffers below is zeroed: every element is written before it is read. The KV cache rows
        // [0, pos + n) are written by the K/V projections (and the rope kernel) before attention reads them,
        // activations and staging rows are written for every token of the current chunk, and the scores
        // scratch is written for positions [first, len) before the softmax reads the same range. Skipping the
        // memset keeps start-up fast and only commits pages that are actually touched (a long context would
        // otherwise clear gigabytes).
        keyCache = new Metal.Buffer[c.layers()];
        valCache = new Metal.Buffer[c.layers()];
        if (a instanceof Attention.Gqa g) { // DeepSeek: MlaMetal owns its latent cache
            long kvBytes = (long) ctx * g.kvDim() * kvDtype.bytes;
            for (int l = 0; l < c.layers(); l++) {
                keyCache[l] = k.scratch(kvBytes);
                valCache[l] = k.scratch(kvBytes);
            }
            kStage = kvHalf ? k.scratch((long) maxBatch * g.kvDim() * 4) : null; // f32 k rows before RoPE
        }
        x = k.scratch((long) maxBatch * c.dim() * 4);
        xb = k.scratch((long) maxBatch * c.dim() * 4);
        q = k.scratch((long) maxBatch * a.qDim() * 4);
        att = k.scratch((long) maxBatch * Math.max(a.qDim(), a.outDim()) * 4);
        hidden = k.scratch((long) maxBatch * arch.maxMlpDim() * 4);
        scores = k.scratch((long) Math.min(MetalConstants.ATT_TILE, maxBatch) * a.heads() * ctx * 4);
        logitsBuf = k.scratch((long) w.lmHead.rows * 4);
        if (a instanceof Attention.Gqa g) {
            MetalTuner.Decision d = MetalTuner.decide(k, g, kvHalf, attnKernel, attnGroup);
            flashPrefill = d.flashPrefill();
            flashDecode = d.flashDecode();
            System.err.println("Metal kernels: flash prefill " + (flashPrefill ? "on" : "off") + ", flash decode "
                    + (flashDecode ? "on" : "off") + " (" + d.note() + ")");
        }
        flashPart = flashDecode ? k.scratch((long) a.heads() * MetalKernels.FLASH_SPLITS * (hd + 2) * 4) : null;
        if (arch.anyMoe()) {
            int slots = maxBatch * arch.maxTopK();
            assign = k.scratch((long) slots * 4);
            routeWeight = k.scratch((long) slots * 4);
            moeHidden = k.scratch((long) slots * arch.maxMoeDim() * 4);
        } else {
            assign = null;
            routeWeight = null;
            moeHidden = null;
        }
        mla = a instanceof Attention.Mla m ? new MlaMetal(this, w, m, ctx, maxBatch, kvDtype) : null;
        plans = new LayerPlan[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            plans[l] = plan(arch.layers().get(l), w.decoderLayer(l));
        }
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

    // ------------------------------------------------------------------ layer plans

    private LayerPlan plan(LayerSpec spec, DecoderLayer L) {
        Metal.Buffer attnNorm = k.upload(L.attnNorm());
        Metal.Buffer ffnNorm = k.upload(L.ffnNorm());
        Metal.Buffer postAttnNorm = L.postAttnNorm() != null ? k.upload(L.postAttnNorm()) : null;
        Metal.Buffer postMlpNorm = L.postMlpNorm() != null ? k.upload(L.postMlpNorm()) : null;
        AttnStep attention = switch (spec.attn()) {
            case Attention.Gqa g -> new GqaStep(g, L.gqa(), attnNorm);
            case Attention.Mla m -> (layer, n, pos, fuseNorm) -> {
                k.rmsnorm(x, 0, attnNorm, xb, n, c.dim());
                mla.run(layer, n, pos);
            };
        };
        FfnStep ffn = switch (spec.mlp()) {
            case Mlp.Dense d -> new DenseStep(d, L.dense(), ffnNorm, postMlpNorm);
            case Mlp.Moe m -> new MoeStep(m, L.moe(), ffnNorm);
        };
        return new LayerPlan(attnNorm, postAttnNorm, L.wo(), k.wrap(L.wo().data), attention, ffn);
    }

    /**
     * Grouped-query attention: the fused q/k/v projection (v written straight into the cache rows at pos, as
     * half for the f16 cache; k too for the f32 cache, while the f16 cache stages the f32 k rows and the rope
     * kernel rounds them in), QK-norm + RoPE, then the attention kernel.
     */
    private final class GqaStep implements AttnStep {
        private final Attention.Gqa g;
        private final RopeTable table;
        private final QTensor wq;
        private final QTensor wk;
        private final QTensor wv;
        private final Metal.Buffer attnNorm;
        private final Metal.Buffer bwq;
        private final Metal.Buffer bwk;
        private final Metal.Buffer bwv;
        private final Metal.Buffer bq;
        private final Metal.Buffer bk;
        private final Metal.Buffer bv;
        private final Metal.Buffer qNorm;
        private final Metal.Buffer kNorm;
        private final Metal.Buffer cos;
        private final Metal.Buffer sin;
        /** GLM rotates interleaved pairs of the first rotaryDim dims, Phi rotate-half pairs of them. */
        private final boolean partial;

        GqaStep(Attention.Gqa g, AttnWeights.Gqa w, Metal.Buffer attnNorm) {
            this.g = g;
            this.table = arch.ropeTables().get(g.ropeTable());
            this.attnNorm = attnNorm;
            wq = w.wq();
            wk = w.wk();
            wv = w.wv();
            bwq = k.wrap(wq.data);
            bwk = k.wrap(wk.data);
            bwv = k.wrap(wv.data);
            bq = w.bq() != null ? k.upload(w.bq()) : null;
            bk = w.bk() != null ? k.upload(w.bk()) : null;
            bv = w.bv() != null ? k.upload(w.bv()) : null;
            qNorm = w.qNorm() != null ? k.upload(w.qNorm()) : null;
            kNorm = w.kNorm() != null ? k.upload(w.kNorm()) : null;
            cos = ropeCos[g.ropeTable()];
            sin = ropeSin[g.ropeTable()];
            partial = table.pairing() == RopeTable.Pairing.INTERLEAVED || table.rotaryDim() != g.headDim();
            if (partial && g.qkNorm()) {
                throw new IllegalArgumentException("partial rotary embedding combined with q/k norm is not implemented");
            }
        }

        @Override
        public void run(int layer, int n, int pos, boolean fuseNorm) {
            int dim = c.dim();
            int kvDim = g.kvDim();
            long cacheOff = (long) pos * kvDim * (kvHalf ? 2 : 4);
            Metal.Buffer kDst = kvHalf ? kStage : keyCache[layer];
            long kOff = kvHalf ? 0 : cacheOff;
            Metal.Buffer in = x;
            Metal.Buffer norm = attnNorm;
            if (!fuseNorm) {
                k.rmsnorm(x, 0, attnNorm, xb, n, dim);
                in = xb;
                norm = null;
            }
            k.matmul3(wq, bwq, wk, bwk, wv, bwv, in, norm, q, kDst, kOff, valCache[layer], cacheOff, kvHalf, bq, bk, bv, n);
            // QK-norm + RoPE for the q rows and the k rows of this chunk in one dispatch: k is read from kDst
            // and written to the cache rows at pos (the same rows for f32, rounded to half for f16)
            boolean useNorm = !partial && qNorm != null && kNorm != null;
            int rotHalf = partial ? table.rotaryDim() / 2 : g.headDim() / 2;
            k.rope(table.pairing() == RopeTable.Pairing.INTERLEAVED, kvHalf, q, kDst, kOff, keyCache[layer], cacheOff,
                    useNorm ? qNorm : null, useNorm ? kNorm : null, cos, sin,
                    g.headDim(), g.qDim(), kvDim, g.heads(), g.kvHeads(), pos, 0, rotHalf, n);
            int hd = g.headDim();
            if (flashDecode && n == 1 && pos + 1 >= MetalKernels.FLASH_MIN_LEN) {
                k.flashAttention(kvHalf, q, keyCache[layer], valCache[layer], flashPart, scores, att, g.heads(), g.kvHeads(),
                        hd, pos, ctx, g.scale(), g.qDim(), kvDim, g.window(), g.softcap());
                return;
            }
            if (flashPrefill && n >= MetalKernels.FLASH_PREFILL_MIN) {
                k.flashPrefill(kvHalf, hd, q, keyCache[layer], valCache[layer], att, g.heads(), g.kvHeads(), pos, n,
                        g.scale(), g.qDim(), kvDim, g.qDim(), g.window(), g.softcap());
                return;
            }
            k.attention(attnKernel, attnGroup, q, keyCache[layer], valCache[layer], att, scores, g.heads(), g.kvHeads(), hd, hd,
                    pos, ctx, g.scale(), g.qDim(), kvDim, kvDim, g.qDim(), g.window(), g.softcap(), n);
        }
    }

    /** Dense gated MLP: the fused norm + gate/up + activation kernel, then the down projection (post-normed for Glm4 / Gemma). */
    private final class DenseStep implements FfnStep {
        private final QTensor gateUp;
        private final QTensor down;
        private final Metal.Buffer bGateUp;
        private final Metal.Buffer bDown;
        private final Metal.Buffer ffnNorm;
        private final Metal.Buffer postMlpNorm;
        private final boolean gelu;

        DenseStep(Mlp.Dense d, MlpWeights.Dense w, Metal.Buffer ffnNorm, Metal.Buffer postMlpNorm) {
            gateUp = w.gateUp();
            down = w.down();
            bGateUp = k.wrap(gateUp.data);
            bDown = k.wrap(down.data);
            this.ffnNorm = ffnNorm;
            this.postMlpNorm = postMlpNorm;
            this.gelu = d.act() == Mlp.Activation.GELU_TANH;
        }

        @Override
        public void run(int layer, int n, boolean fuseNorm) {
            int dim = c.dim();
            if (fuseNorm) {
                k.swiglu(gateUp, bGateUp, x, ffnNorm, hidden, gelu, n);
            } else {
                k.rmsnorm(x, 0, ffnNorm, xb, n, dim);
                k.swiglu(gateUp, bGateUp, xb, null, hidden, gelu, n);
            }
            if (postMlpNorm != null) {
                k.matmul(down, bDown, hidden, 0, xb, 0, n, false, null, 0f);
                k.rmsnormAdd(xb, postMlpNorm, x, n, dim);
            } else {
                k.matmul(down, bDown, hidden, 0, x, 0, n, true, null, 0f);
            }
        }
    }

    /** Router → per-slot gate/up → SwiGLU → weighted down projections accumulated into x, then DeepSeek's shared experts. */
    private final class MoeStep implements FfnStep {
        private interface Route {
            void run(int n);
        }

        private final Mlp.Moe spec;
        private final QTensor expGateUp;
        private final QTensor expDown;
        private final Metal.Buffer bExpGateUp;
        private final Metal.Buffer bExpDown;
        private final Metal.Buffer ffnNorm;
        private final Route route;
        private final QTensor sharedGateUp;
        private final QTensor sharedDown;
        private final Metal.Buffer bSharedGateUp;
        private final Metal.Buffer bSharedDown;

        MoeStep(Mlp.Moe spec, MlpWeights.Moe w, Metal.Buffer ffnNorm) {
            this.spec = spec;
            this.ffnNorm = ffnNorm;
            expGateUp = w.expGateUp();
            expDown = w.expDown();
            bExpGateUp = k.wrap(expGateUp.data);
            bExpDown = k.wrap(expDown.data);
            Metal.Buffer router = k.upload(w.router());
            int e = spec.experts();
            int dim = c.dim();
            int topK = spec.topK();
            route = switch (spec.router()) {
                case Router.Softmax s -> n -> k.router(xb, router, assign, routeWeight, e, dim, topK, spec.normTopK(), n);
                case Router.DeepSeek r -> {
                    Metal.Buffer bias = w.routerBias() != null ? k.upload(w.routerBias()) : null;
                    int method = switch (r.topK()) {
                        case GREEDY -> 0;
                        case GROUP_LIMITED_GREEDY -> 1;
                        case NOAUX_TC -> 2;
                    };
                    yield n -> k.routerDs(xb, router, assign, routeWeight, bias, e, dim, topK, spec.normTopK(),
                            r.nGroup(), r.topkGroup(), r.sigmoid(), method, r.scaleAlways(), (float) r.routedScale(), n);
                }
            };
            if (w.shared() != null) {
                sharedGateUp = w.shared().gateUp();
                sharedDown = w.shared().down();
                bSharedGateUp = k.wrap(sharedGateUp.data);
                bSharedDown = k.wrap(sharedDown.data);
            } else {
                sharedGateUp = null;
                sharedDown = null;
                bSharedGateUp = null;
                bSharedDown = null;
            }
        }

        @Override
        public void run(int layer, int n, boolean fuseNorm) {
            k.rmsnorm(x, 0, ffnNorm, xb, n, c.dim());
            route.run(n);
            k.moeUp(expGateUp, bExpGateUp, xb, assign, moeHidden, spec.moeDim(), spec.topK(), n);
            k.moeDown(expDown, bExpDown, moeHidden, assign, routeWeight, x, c.dim(), spec.topK(), n);
            if (sharedGateUp != null) {
                // shared experts: a dense SwiGLU of width nShared*moeDim on every token, added to x
                k.swiglu(sharedGateUp, bSharedGateUp, xb, null, hidden, false, n); // writes silu(gate) * up to hidden
                k.matmul(sharedDown, bSharedDown, hidden, 0, x, 0, n, true, null, 0f);
            }
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
                LayerPlan L = plans[l];
                L.attention().run(l, n, pos, fuseNorm);
                if (L.postAttnNorm() != null) {
                    // Glm4 / Gemma: x += rmsnorm(o_proj(attention)), so o_proj cannot accumulate into x directly
                    k.matmul(L.wo(), L.woBuf(), att, 0, xb, 0, n, false, null, 0f);
                    k.rmsnormAdd(xb, L.postAttnNorm(), x, n, dim);
                } else {
                    k.matmul(L.wo(), L.woBuf(), att, 0, x, 0, n, true, null, 0f);
                }
                L.ffn().run(l, n, fuseNorm);
                if (MetalKernels.COMMIT_LAYERS > 0 && (l + 1) % MetalKernels.COMMIT_LAYERS == 0 && l + 1 < c.layers()) {
                    k.commit(); // the GPU starts on these layers while the next ones are encoded
                }
            }
            if (logits != null && stride > 0) {
                // every row (speculative verification): rmsnorm and LM head over the n tokens
                if (logitsAllBuf == null || logitsAllRows < n) {
                    logitsAllRows = Math.min(maxBatch, Math.max(n, 16)); // buffers live until the backend closes: size once
                    logitsAllBuf = k.scratch((long) logitsAllRows * w.lmHead.rows * 4);
                }
                k.rmsnorm(x, 0, finalNorm, xb, n, dim);
                k.matmul(w.lmHead, lmHeadQ, xb, 0, logitsAllBuf, 0, n, false, null, c.finalSoftcap());
            } else if (logits != null) {
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
        if (logits != null && stride > 0) {
            int rows = Math.min(w.lmHead.rows, stride);
            for (int i = 0; i < n; i++) {
                MemorySegment.copy(logitsAllBuf.contents(), ValueLayout.JAVA_FLOAT, (long) i * w.lmHead.rows * 4,
                        logits, i * stride, rows);
            }
        } else if (logits != null) {
            MemorySegment.copy(logitsBuf.contents(), ValueLayout.JAVA_FLOAT, 0, logits, 0,
                    Math.min(w.lmHead.rows, logits.length));
        }
    }

    // ------------------------------------------------------------------ saved caches

    /**
     * The KV cache rows as CpuBackend saves them ({@code KvCache.write}): per layer, per position, the K row then the V
     * row, little-endian f32 or IEEE half; Metal's {@code half} has the same bits, so a cache saved by either backend
     * has the same layout. DeepSeek's latent cache is not saved.
     */
    @Override
    public String stateFormat() {
        if (!(arch.attention() instanceof Attention.Gqa g)) {
            return null;
        }
        return "kv " + (kvHalf ? "f16" : "f32") + " layers=" + c.layers() + " kvDim=" + g.kvDim() + " headDim=" + g.headDim();
    }

    private long rowBytes() {
        return (long) ((Attention.Gqa) arch.attention()).kvDim() * (kvHalf ? 2 : 4);
    }

    @Override
    public void saveState(int n, java.nio.channels.WritableByteChannel out) throws java.io.IOException {
        if (stateFormat() == null) {
            Backend.super.saveState(n, out);
        }
        long row = rowBytes();
        int rows = (int) Math.max(1, Math.min(n, (1 << 22) / (2 * row)));
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocateDirect((int) (rows * 2 * row));
        MemorySegment b = MemorySegment.ofBuffer(buf);
        for (int l = 0; l < c.layers(); l++) {
            MemorySegment kc = keyCache[l].contents();
            MemorySegment vc = valCache[l].contents();
            for (int s0 = 0; s0 < n; s0 += rows) {
                int m = Math.min(rows, n - s0);
                for (int r = 0; r < m; r++) {
                    MemorySegment.copy(kc, (s0 + r) * row, b, 2 * r * row, row);
                    MemorySegment.copy(vc, (s0 + r) * row, b, (2L * r + 1) * row, row);
                }
                buf.clear().limit((int) (2 * m * row));
                while (buf.hasRemaining()) {
                    out.write(buf);
                }
            }
        }
    }

    @Override
    public void loadState(int n, java.nio.channels.ReadableByteChannel in) throws java.io.IOException {
        if (stateFormat() == null) {
            Backend.super.loadState(n, in);
        }
        if (n > ctx) {
            throw new java.io.IOException("the saved cache holds " + n + " positions, the context is " + ctx);
        }
        long row = rowBytes();
        int rows = (int) Math.max(1, Math.min(n, (1 << 22) / (2 * row)));
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocateDirect((int) (rows * 2 * row));
        MemorySegment b = MemorySegment.ofBuffer(buf);
        for (int l = 0; l < c.layers(); l++) {
            MemorySegment kc = keyCache[l].contents();
            MemorySegment vc = valCache[l].contents();
            for (int s0 = 0; s0 < n; s0 += rows) {
                int m = Math.min(rows, n - s0);
                buf.clear().limit((int) (2 * m * row));
                while (buf.hasRemaining()) {
                    if (in.read(buf) < 0) {
                        throw new java.io.EOFException("the saved cache file is truncated");
                    }
                }
                // straight into the cache buffers: on unified memory they are the GPU's memory
                for (int r = 0; r < m; r++) {
                    MemorySegment.copy(b, 2 * r * row, kc, (s0 + r) * row, row);
                    MemorySegment.copy(b, (2L * r + 1) * row, vc, (s0 + r) * row, row);
                }
            }
        }
    }

    @Override
    public void copyStateFrom(Backend other, int n) {
        if (stateFormat() == null || !(other instanceof MetalBackend o) || !stateFormat().equals(o.stateFormat())) {
            Backend.super.copyStateFrom(other, n);
        }
        MetalBackend o = (MetalBackend) other;
        long bytes = (long) n * rowBytes();
        for (int l = 0; l < c.layers(); l++) {
            MemorySegment.copy(o.keyCache[l].contents(), 0, keyCache[l].contents(), 0, bytes);
            MemorySegment.copy(o.valCache[l].contents(), 0, valCache[l].contents(), 0, bytes);
        }
    }

    @Override
    public void close() {
        k.close();
    }
}
