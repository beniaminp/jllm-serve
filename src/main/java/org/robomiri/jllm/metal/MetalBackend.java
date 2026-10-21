package org.robomiri.jllm.metal;

import org.robomiri.jllm.cpu.MoeExperts;
import org.robomiri.jllm.cpu.ThreadPool;
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
    // reads half; see the "f16 KV cache" section of kernels.metal. q8 (backend.kv-dtype=q8): k and v both go
    // through f32 staging rows (rope rotates k in place) and kv_store_q8 quantizes each (token, kv head) row into
    // the cache, whose position rows are kvDim int8 values followed by one f32 scale per kv head; the *_q8
    // attention kernels read them
    private final KvDtype kvDtype;
    private final boolean kvHalf;
    private boolean fusedNormFits;     // see MetalKernels.fusedNormFits
    private boolean replay;            // decode replay (MetalKernels.step) on: see stepKey
    private int[] windows;             // the distinct attention windows of the GQA layers (0 = none), for stepKey
    private int keyedVocab = -1;       // the pipe's vocabulary the recorded pipe steps were made with
    private Metal.Buffer kStage;
    private Metal.Buffer vStage;

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
    private Metal.Buffer moeOrder;  // grouped MoE: slots sorted by expert
    private Metal.Buffer moeTiles;  // grouped MoE: tiles of one expert's slots
    private Metal.Buffer moeOut;    // grouped MoE: each slot's weighted output row
    /** DeepSeek multi-head latent attention (own caches); null for other families. */
    private MlaMetal mla;
    // experts left on the CPU (backend.metal.cpu-moe); null when every expert is on the GPU
    private final MoeSplit split;
    private ThreadPool cpuPool;
    private MoeExperts cpuExperts;
    private float[] cpuXb;
    private float[] cpuY;
    private int[] cpuAssign;
    private float[] cpuWeight;
    private int cpuLayers;            // MoE layers with at least one expert on the CPU
    private Placement placement;      // where the experts are, and the packed hot experts of split layers
    private boolean placementShared;  // placement is the weights' entry in PLACEMENTS (else this backend's own)
    private Metal.Buffer gassign;     // the router's remap: each slot's expert among the layer's GPU experts, or -1
    // how often each expert was picked, per MoE layer, while any expert is on the CPU: counted by the router on the GPU
    // (layers with GPU experts) or here (layers entirely on the CPU, from the router's choices the CPU reads anyway)
    private Metal.Buffer[] counts;
    private long[][] hostCounts;
    private boolean countStep;        // this forward pass counts the layers entirely on the GPU (every COUNT_EVERY-th)
    private long steps;
    private java.nio.file.Path statsFile;
    private String modelKey;
    private long lastStatsSave;
    private boolean closed;
    /** This backend in the {@link DeviceLedger}; its placement is set before it is added. */
    private final DeviceLedger.Resident ledger = new DeviceLedger.Resident() {
        @Override
        public ModelWeights weights() {
            return w;
        }

        @Override
        public long gpuWeightBytes() {
            return gpuBytes(placement != null ? placement.cold : new boolean[c.layers()][]) - k.metal.allocatedBytes();
        }

        @Override
        public long ownBytes() {
            return k.metal.allocatedBytes();
        }
    };

    /**
     * Layers whose experts are all on the GPU are counted on one forward pass in this many (the counting atomics in
     * every layer's router on every token would cost more than they tell): the planner turns each layer's counts into
     * shares of that layer's picks, so a sample is enough.
     */
    private static final int COUNT_EVERY = Math.max(1, Integer.getInteger("jllm.metal.expertCountEvery", 8));

    /**
     * Where the experts of one {@link ModelWeights} live, and the packed copies of the hot experts of its split layers.
     * Shared by every MetalBackend on those weights with the same {@link MoeSplit} (a server's extra slots), so the
     * copies exist once and every slot runs the same plan; freed with the last of them.
     */
    private static final class Placement {
        final String config;
        final boolean[][] cold;   // rows replaced by rebalanceExperts
        final QTensor[] gateUp;   // per layer: the packed hot experts of a split layer, else null
        final QTensor[] down;
        /** Per layer: the memory of its packed experts, so a re-placed layer frees its old copy on its own. */
        final java.lang.foreign.Arena[] arenas;
        /** The CPU threads for the experts, one set for all the backends (a pool runs one caller's job at a time). */
        ThreadPool pool;
        int refs = 1;

        Placement(String config, boolean[][] cold) {
            this.config = config;
            this.cold = cold;
            this.gateUp = new QTensor[cold.length];
            this.down = new QTensor[cold.length];
            this.arenas = new java.lang.foreign.Arena[cold.length];
        }

        /** Packs layer l's hot experts (those not in {@code coldRow}) into a new arena; nothing for a layer not split. */
        void pack(int l, MlpWeights.Moe mw, boolean[] coldRow, int moeDim, int dim) {
            int[] hot = new int[coldRow.length];
            int nHot = 0;
            for (int e = 0; e < coldRow.length; e++) {
                if (!coldRow[e]) {
                    hot[nHot++] = e;
                }
            }
            gateUp[l] = null;
            down[l] = null;
            arenas[l] = null;
            if (nHot == 0 || nHot == coldRow.length) {
                return;
            }
            java.lang.foreign.Arena a = java.lang.foreign.Arena.ofShared();
            try {
                int[] h = java.util.Arrays.copyOf(hot, nHot);
                gateUp[l] = mw.expGateUp().gatherExperts(a, h, 2 * moeDim);
                down[l] = mw.expDown().gatherExperts(a, h, dim);
                arenas[l] = a;
            } catch (RuntimeException | Error e) {
                gateUp[l] = null;
                down[l] = null;
                a.close();
                throw e;
            }
        }

        void close() {
            for (java.lang.foreign.Arena a : arenas) {
                if (a != null) {
                    a.close();
                }
            }
        }
    }

    private static final java.util.Map<ModelWeights, Placement> PLACEMENTS = new java.util.IdentityHashMap<>();


    /**
     * One decoder layer: its norm buffers, {@code o_proj}, and two strategy objects picked once from the layer's
     * {@link LayerSpec} (like {@code CpuBackend.LayerPlan}), so {@link #forward} contains no per-layer dispatch on
     * the family's optional pieces: the attention (grouped-query over the KV cache, or MLA) and the feed-forward
     * (dense GLU with SiLU or GELU, routed experts, routed plus shared experts). Each step runs exactly the
     * dispatches the straight-line code recorded before, in the same order.
     */
    private record LayerPlan(Metal.Buffer attnNorm, Metal.Buffer postAttnNorm, QTensor wo, Metal.Buffer woBuf,
                             Metal.Buffer woBias, AttnStep attention, FfnStep ffn) {
    }

    /** Reads {@code x} (normalizing it with the layer's attention norm), writes the attention output to {@link #att}. */
    private interface AttnStep {
        void run(int layer, int n, int pos, boolean fuseNorm);

        /**
         * Batched decoding across slots ({@link #forwardBatch}): row t of the n rows of x is one token of
         * {@code owners[t]}'s conversation at {@code positions[t]}; the projections run over all n rows, RoPE, the
         * cache store and attention once per row in that owner's cache.
         */
        void runRows(int layer, int n, MetalBackend[] owners, int[] positions);
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
        this(w, ctx, maxBatch, metalLib, kvDtype, MoeSplit.OFF);
    }

    /** {@code split}: which MoE layers run their routed experts on the CPU (see {@link MoeSplit}). */
    public MetalBackend(ModelWeights w, int ctx, int maxBatch, String metalLib, KvDtype kvDtype, MoeSplit split) {
        this(w, ctx, maxBatch, open(w.decoder(), metalLib), kvDtype, split);
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
    private MetalBackend(ModelWeights w, int ctx, int maxBatch, Metal metal, KvDtype kvDtype, MoeSplit split) {
        this.w = w;
        this.split = split;
        this.arch = w.decoder();
        this.c = arch.core();
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        this.kvDtype = kvDtype;
        this.kvHalf = kvDtype == KvDtype.F16;
        this.k = new MetalKernels(metal, c.normEps());
        try {
            init(kvDtype);
        } catch (RuntimeException | Error e) {
            try {
                close();
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
        String vh = MetalKernels.kvSuffix(kvDtype);
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
            long kvBytes = ctx * kvDtype.rowBytes(g.kvDim(), g.kvHeads());
            for (int l = 0; l < c.layers(); l++) {
                keyCache[l] = k.scratch(kvBytes);
                valCache[l] = k.scratch(kvBytes);
                k.rowIndexed(keyCache[l], kvDtype.rowBytes(g.kvDim(), g.kvHeads())); // for decode replay
                k.rowIndexed(valCache[l], kvDtype.rowBytes(g.kvDim(), g.kvHeads()));
            }
            // f32 k rows before RoPE (f16, q8) and v rows before the quantization (q8)
            kStage = kvDtype != KvDtype.F32 ? k.scratch((long) maxBatch * g.kvDim() * 4) : null;
            vStage = kvDtype == KvDtype.Q8 ? k.scratch((long) maxBatch * g.kvDim() * 4) : null;
        }
        x = k.scratch((long) maxBatch * c.dim() * 4);
        xb = k.scratch((long) maxBatch * c.dim() * 4);
        q = k.scratch((long) maxBatch * a.qDim() * 4);
        att = k.scratch((long) maxBatch * Math.max(a.qDim(), a.outDim()) * 4);
        hidden = k.scratch((long) maxBatch * arch.maxMlpDim() * 4);
        scores = k.scratch((long) Math.min(MetalConstants.ATT_TILE, maxBatch) * a.heads() * ctx * 4);
        logitsBuf = k.scratch((long) w.lmHead.rows * 4);
        if (a instanceof Attention.Gqa g) {
            MetalTuner.Decision d = MetalTuner.decide(k, g, kvDtype, attnKernel, attnGroup);
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
            moeOrder = k.scratch((long) slots * 4);
            moeTiles = k.scratch((long) (1 + 3 * MetalKernels.moeMaxTiles(slots, arch.maxExperts())) * 4);
            moeOut = k.scratch((long) slots * c.dim() * 4);
        } else {
            assign = null;
            routeWeight = null;
            moeHidden = null;
        }
        mla = a instanceof Attention.Mla m ? new MlaMetal(this, w, m, ctx, maxBatch, kvDtype) : null;
        boolean[][] cold = coldExperts();
        plans = new LayerPlan[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            plans[l] = plan(arch.layers().get(l), w.decoderLayer(l), cold[l], l);
        }
        k.decodeThreads = MetalTuner.decodeThreads(k, plans[0].wo(), plans[0].woBuf());
        MetalTuner.prefillMatmuls(k, w); // f32 or half-weight MMA prefill kernels, per matrix shape
        java.util.TreeSet<Integer> wins = new java.util.TreeSet<>();
        for (LayerSpec s : arch.layers()) {
            if (s.attn() instanceof Attention.Gqa g) {
                wins.add(g.window());
            }
        }
        windows = wins.stream().mapToInt(Integer::intValue).toArray();
        replay = MetalKernels.REPLAY && k.metal.canRun() && windows.length <= 8;
        if (ctx > c.maxPositions()) {
            System.err.printf("warning: ctx %d exceeds the model's max_position_embeddings (%d); RoPE positions "
                    + "beyond that are extrapolated and quality degrades%n", ctx, c.maxPositions());
        }
        long gpu = gpuBytes(cold);
        long others = DeviceLedger.others(ledger);
        DeviceLedger.add(ledger);
        long ws = workingSet();
        if (gpu + others > ws) {
            System.err.printf("warning: the GPU holds %.1f GB (this model's share and caches %.1f GB, other backends %.1f GB), "
                            + "more than Metal's recommended working set (%.1f GB); expect paging%s%n", (gpu + others) / 1e9,
                    gpu / 1e9, others / 1e9, ws / 1e9, arch.anyMoe()
                            ? " (backend.metal.cpu-moe=auto leaves experts on the CPU; a placement shared by slots is made "
                            + "when the first slot loads)" : "");
        }
    }

    // ------------------------------------------------------------------ experts on the CPU

    /** Metal's recommendedMaxWorkingSetSize, or {@code -Djllm.metal.workingSetGb} (to leave memory to other programs). */
    private long workingSet() {
        double gb = ExpertPlacement.positiveProperty("jllm.metal.workingSetGb", -1);
        return gb > 0 ? (long) (gb * 1e9) : k.metal.recommendedWorkingSet();
    }

    private static long bytes(QTensor t) {
        return t.data.byteSize() + (t.second != null ? t.second.data.byteSize() : 0);
    }

    /** Bytes of layer {@code l}'s routed experts (0 for a dense layer). */
    private long expertBytes(int l) {
        return arch.layers().get(l).mlp() instanceof Mlp.Moe
                ? bytes(w.decoderLayer(l).moe().expGateUp()) + bytes(w.decoderLayer(l).moe().expDown()) : 0;
    }

    /** Bytes of one expert of layer {@code l} (gate, up and down rows; 0 for a dense layer). */
    private long bytesPerExpert(int l) {
        return arch.layers().get(l).mlp() instanceof Mlp.Moe m ? expertBytes(l) / m.experts() : 0;
    }

    /** What the GPU keeps resident: the weights minus the experts left on the CPU, plus its own buffers. */
    private long gpuBytes(boolean[][] cold) {
        long b = w.byteSize() + k.metal.allocatedBytes();
        for (int l = 0; l < cold.length; l++) {
            if (cold[l] != null) {
                for (boolean v : cold[l]) {
                    b -= v ? bytesPerExpert(l) : 0;
                }
            }
        }
        return b;
    }

    private static final long STATS_SAVE_NANOS = 10L * 60 * 1_000_000_000L;

    /** {@code -Djllm.metal.hotExperts=false}: plan from uniform usage (whole layers) even when statistics exist. */
    static final boolean HOT_EXPERTS = !"false".equals(System.getProperty("jllm.metal.hotExperts"));

    /**
     * The experts left on the CPU, {@code cold[l][e]} (null for a dense layer): the whole last {@code split.layers()}
     * MoE layers, or with {@link MoeSplit#AUTO} the cheapest set that makes the GPU's share fit the working set, chosen
     * by {@link ExpertPlacement#plan} from how often earlier runs picked each expert (none when the model fits). Sets
     * up the CPU side and the usage counting when any expert is on the CPU.
     */
    private boolean[][] coldExperts() {
        int layers = c.layers();
        long[] perExpert = new long[layers];
        int[] experts = new int[layers];
        int[] topK = new int[layers];
        for (int l = 0; l < layers; l++) {
            if (arch.layers().get(l).mlp() instanceof Mlp.Moe m) {
                perExpert[l] = bytesPerExpert(l);
                experts[l] = m.experts();
                topK[l] = m.topK();
            }
        }
        modelKey = modelKey();
        statsFile = ExpertPlacement.statsFile(modelKey);
        String config = split.config();
        // planning and packing (seconds, gigabytes) hold only this model's lock, not the other models' loads
        synchronized (w) {
            Placement shared;
            synchronized (PLACEMENTS) {
                shared = PLACEMENTS.get(w);
                if (shared != null && shared.config.equals(config)) {
                    shared.refs++;
                }
            }
            if (shared != null && shared.config.equals(config)) {
                placement = shared;
                placementShared = true;
            } else {
                placement = newPlacement(config, perExpert, experts, topK);
                synchronized (PLACEMENTS) {
                    if (!PLACEMENTS.containsKey(w)) {
                        PLACEMENTS.put(w, placement);
                        placementShared = true;
                    }
                }
                if (!placementShared && placement.cold != null) {
                    System.err.println("note: this backend's backend.metal.cpu-moe differs from the other backends on the "
                            + "same weights, so it keeps its own expert placement (and copies, and CPU threads)");
                }
            }
        }
        boolean[][] cold = placement.cold;
        for (int l = 0; l < layers; l++) {
            int n = 0;
            for (int e = 0; cold[l] != null && e < experts[l]; e++) {
                n += cold[l][e] ? 1 : 0;
            }
            cpuLayers += n > 0 ? 1 : 0;
        }
        if (cpuLayers > 0) {
            synchronized (PLACEMENTS) {
                if (placement.pool == null) {
                    placement.pool = new ThreadPool(split.threads());
                }
            }
            cpuPool = placement.pool;
            cpuExperts = MoeExperts.create(cpuPool, w, maxBatch, split.int8Activations());
            int slots = maxBatch * arch.maxTopK();
            cpuXb = new float[maxBatch * c.dim()];
            cpuY = new float[maxBatch * c.dim()];
            cpuAssign = new int[slots];
            cpuWeight = new float[slots];
            gassign = k.scratch((long) slots * 4);
            counts = new Metal.Buffer[layers];
            hostCounts = new long[layers][];
            for (int l = 0; l < layers; l++) {
                if (cold[l] != null) { // both: rebalanceExperts can turn a layer from all-CPU to split and back
                    hostCounts[l] = new long[experts[l]];
                    counts[l] = k.metal.newBuffer((long) experts[l] * 4); // zeroed
                }
            }
            lastStatsSave = System.nanoTime();
        }
        return cold;
    }

    /**
     * Plans where the experts go (see {@link #coldExperts}), packs the hot experts of split layers into memory of
     * their own, and logs the plan.
     */
    private Placement newPlacement(String config, long[] perExpert, int[] experts, int[] topK) {
        int layers = c.layers();
        long[][] usage = null;
        boolean[][] cold;
        if (split.cold() != null) {
            cold = new boolean[layers][];
            for (int l = 0; l < layers; l++) {
                cold[l] = perExpert[l] > 0 ? (l < split.cold().length && split.cold()[l] != null
                        ? split.cold()[l].clone() : new boolean[experts[l]]) : null;
            }
        } else if (split.layers() == MoeSplit.AUTO) {
            // the device holds this model's weights, every slot's buffers (estimated as this one's, the later slots
            // are not created yet) and whatever the other open backends hold
            long need = w.byteSize() + split.backends() * (k.metal.allocatedBytes() + laterUploads()) + DeviceLedger.others(ledger)
                    - workingSet();
            usage = need > 0 && HOT_EXPERTS ? ExpertPlacement.load(statsFile, layers, modelKey) : null;
            cold = ExpertPlacement.plan(need, perExpert, experts, topK, usage);
        } else {
            cold = ExpertPlacement.lastLayers(split.layers(), perExpert, experts);
        }
        Placement p = new Placement(config, cold);
        int moeLayers = 0;
        int full = 0;
        int mixed = 0;
        long moved = 0;
        double cpuPicks = 0;
        try {
            for (int l = 0; l < layers; l++) {
                if (cold[l] == null) {
                    continue;
                }
                moeLayers++;
                double[] share = ExpertPlacement.shares(usage != null ? usage[l] : null, experts[l]);
                int[] hot = new int[experts[l]];
                int nHot = 0;
                for (int e = 0; e < experts[l]; e++) {
                    if (cold[l][e]) {
                        cpuPicks += share[e];
                        moved += perExpert[l];
                    } else {
                        hot[nHot++] = e;
                    }
                }
                full += nHot == 0 ? 1 : 0;
                if (nHot > 0 && nHot < experts[l]) {
                    mixed++;
                    // the hot experts, packed into memory of their own; the mapped tensors stay untouched for the CPU
                    p.pack(l, w.decoderLayer(l).moe(), cold[l], ((Mlp.Moe) arch.layers().get(l).mlp()).moeDim(), c.dim());
                }
            }
        } catch (RuntimeException | Error e) {
            p.close();
            throw e;
        }
        if (full + mixed > 0) {
            System.err.printf("Metal: %.1f GB of experts run on the CPU (%d threads): %d of %d MoE layers entirely, %d in part "
                            + "(%s); expected share of expert picks on the CPU %.0f %%; GPU share %.1f GB of a %.1f GB "
                            + "working set%n", moved / 1e9, split.threads(), full, moeLayers, mixed,
                    usage != null ? "placed by the usage of earlier runs" : "no usage statistics yet: whole layers",
                    100 * cpuPicks / Math.max(moeLayers, 1), gpuBytes(cold) / 1e9, workingSet() / 1e9);
        }
        return p;
    }

    /**
     * What the layer plans will still upload after the placement is made (routers, norms, biases: f32 copies, not
     * wrapped weights), so the budget includes them; a slight overestimate.
     */
    private long laterUploads() {
        long b = 0;
        for (int l = 0; l < c.layers(); l++) {
            b += 7L * c.dim() * 4; // up to four norms, the o_proj bias; q/k/v biases are smaller than the rest together
            if (arch.layers().get(l).mlp() instanceof Mlp.Moe m) {
                b += (long) m.experts() * (c.dim() + 1) * 4; // router (+ DeepSeek's bias)
            }
        }
        return b;
    }

    /**
     * Names the model for its usage statistics: the architecture, the size, and a hash of the first MoE layer's router
     * weights, so two fine-tunes of one base model (same shapes and size) keep their own statistics.
     */
    private String modelKey() {
        int router = 0;
        for (int l = 0; l < c.layers(); l++) {
            if (arch.layers().get(l).mlp() instanceof Mlp.Moe) {
                router = java.util.Arrays.hashCode(w.decoderLayer(l).moe().router());
                break;
            }
        }
        return arch + " | " + w.byteSize() + " bytes | router " + Integer.toHexString(router);
    }

    /** Adds the expert picks counted since the last save to the statistics file and resets the counters. */
    private void saveExpertStats(boolean background) {
        if (counts == null) {
            return;
        }
        long[][] delta = new long[counts.length][];
        boolean any = false;
        for (int l = 0; l < counts.length; l++) {
            if (counts[l] != null) { // the GPU's counts (layers with GPU experts) plus the CPU's (layers without)
                MemorySegment cs = counts[l].contents();
                delta[l] = hostCounts[l].clone();
                java.util.Arrays.fill(hostCounts[l], 0);
                for (int i = 0; i < delta[l].length; i++) {
                    delta[l][i] += Integer.toUnsignedLong(cs.getAtIndex(ValueLayout.JAVA_INT, i));
                }
                cs.fill((byte) 0);
            }
            for (int i = 0; delta[l] != null && i < delta[l].length; i++) {
                any |= delta[l][i] != 0;
            }
        }
        if (any) {
            // the counters are read and reset here (the GPU is idle); the file, with its cross-process lock, is
            // written off the decode thread except at close
            if (background) {
                ExpertPlacement.addLater(statsFile, modelKey, delta);
            } else {
                ExpertPlacement.add(statsFile, modelKey, delta);
            }
        }
        lastStatsSave = System.nanoTime();
    }

    /**
     * {@code cpuY = routed experts(xb)} on the CPU for the n tokens of the current chunk, after the GPU has written xb
     * and the router's choices (the command buffer that recorded them must be finished): all of them, or with
     * {@code cold} only the slots whose expert is cold. Results go to {@link #cpuY} so the GPU may add its own experts
     * into x meanwhile ({@code addToX} later). False when no slot was the CPU's (cpuY is then not written).
     * {@code hostCount}: adds the router's choices to it (a layer counted here instead of by the GPU router).
     */
    private boolean cpuRouted(MlpWeights.Moe mw, Mlp.Moe spec, int n, boolean[] cold, long[] hostCount) {
        int dim = c.dim();
        int topK = spec.topK();
        MemorySegment.copy(assign.contents(), ValueLayout.JAVA_INT, 0, cpuAssign, 0, n * topK);
        for (int s = 0; hostCount != null && s < n * topK; s++) {
            hostCount[cpuAssign[s]]++;
        }
        boolean any = cold == null;
        if (cold != null) { // only the slots whose expert stays on the CPU; the GPU computes the others
            for (int s = 0; s < n * topK; s++) {
                if (cold[cpuAssign[s]]) {
                    any = true;
                } else {
                    cpuAssign[s] = -1;
                }
            }
        }
        if (!any) {
            return false;
        }
        MemorySegment.copy(xb.contents(), ValueLayout.JAVA_FLOAT, 0, cpuXb, 0, n * dim);
        MemorySegment.copy(routeWeight.contents(), ValueLayout.JAVA_FLOAT, 0, cpuWeight, 0, n * topK);
        java.util.Arrays.fill(cpuY, 0, n * dim, 0f);
        synchronized (cpuPool) { // shared by the slots of this model; it runs one job at a time
            if (n > 1) {
                cpuExperts.batched(mw, spec.experts(), topK, spec.moeDim(), n, cpuXb, cpuAssign, cpuWeight, cpuY);
            } else {
                cpuExperts.token(mw, topK, spec.moeDim(), 0, cpuXb, cpuAssign, cpuWeight, cpuY);
            }
        }
        return true;
    }

    /** {@code x[0, n·dim) += cpuY}: the CPU's experts joining the residual stream (the GPU is idle here). */
    private void addToX(int n) {
        int len = n * c.dim();
        // two bulk copies around a plain loop the JIT vectorizes (cpuXb is free again: the experts are done)
        MemorySegment.copy(x.contents(), ValueLayout.JAVA_FLOAT, 0, cpuXb, 0, len);
        for (int i = 0; i < len; i++) {
            cpuXb[i] += cpuY[i];
        }
        MemorySegment.copy(cpuXb, 0, x.contents(), ValueLayout.JAVA_FLOAT, 0, len);
    }

    // ------------------------------------------------------------------ layer plans

    private LayerPlan plan(LayerSpec spec, DecoderLayer L, boolean[] cold, int index) {
        Metal.Buffer attnNorm = k.upload(L.attnNorm());
        Metal.Buffer ffnNorm = k.upload(L.ffnNorm());
        Metal.Buffer postAttnNorm = L.postAttnNorm() != null ? k.upload(L.postAttnNorm()) : null;
        Metal.Buffer postMlpNorm = L.postMlpNorm() != null ? k.upload(L.postMlpNorm()) : null;
        AttnStep attention = switch (spec.attn()) {
            case Attention.Gqa g -> new GqaStep(g, L.gqa(), attnNorm);
            case Attention.Mla m -> new AttnStep() {
                @Override
                public void run(int layer, int n, int pos, boolean fuseNorm) {
                    if (fuseNorm && MetalKernels.FUSE) { // the projections normalize x themselves (*_norm matmuls)
                        mla.run(layer, n, pos, x, attnNorm);
                    } else {
                        k.rmsnorm(x, 0, attnNorm, xb, n, c.dim());
                        mla.run(layer, n, pos, xb, null);
                    }
                }

                @Override
                public void runRows(int layer, int n, MetalBackend[] owners, int[] positions) {
                    k.rmsnorm(x, 0, attnNorm, xb, n, c.dim());
                    MlaMetal[] caches = new MlaMetal[n];
                    for (int t = 0; t < n; t++) {
                        caches[t] = owners[t].mla;
                    }
                    mla.runRows(layer, n, caches, positions);
                }
            };
        };
        FfnStep ffn = switch (spec.mlp()) {
            case Mlp.Dense d -> new DenseStep(d, L.dense(), ffnNorm, postMlpNorm);
            case Mlp.Moe m -> new MoeStep(m, L.moe(), ffnNorm, cold, index);
        };
        return new LayerPlan(attnNorm, postAttnNorm, L.wo(), k.wrap(L.wo().data), L.bo() != null ? k.upload(L.bo()) : null,
                attention, ffn);
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
            boolean q8 = kvDtype == KvDtype.Q8;
            int stride = kvDtype.rowStride(kvDim, g.kvHeads()); // cache row stride in elements (q8: bytes)
            long cacheOff = pos * kvDtype.rowBytes(kvDim, g.kvHeads());
            Metal.Buffer kDst = kvDtype != KvDtype.F32 ? kStage : keyCache[layer];
            long kOff = kvDtype != KvDtype.F32 ? 0 : cacheOff;
            Metal.Buffer vDst = q8 ? vStage : valCache[layer];
            long vOff = q8 ? 0 : cacheOff;
            Metal.Buffer in = x;
            Metal.Buffer norm = attnNorm;
            if (!fuseNorm) {
                k.rmsnorm(x, 0, attnNorm, xb, n, dim);
                in = xb;
                norm = null;
            }
            k.matmul3(wq, bwq, wk, bwk, wv, bwv, in, norm, q, kDst, kOff, vDst, vOff, kvHalf, bq, bk, bv, n);
            // QK-norm + RoPE for the q rows and the k rows of this chunk in one dispatch: k is read from kDst
            // and written to the cache rows at pos (the same rows for f32, rounded to half for f16; for q8 back
            // into the staging rows, which kv_store_q8 then quantizes together with v)
            boolean useNorm = !partial && qNorm != null && kNorm != null;
            int rotHalf = partial ? table.rotaryDim() / 2 : g.headDim() / 2;
            boolean adjacent = table.pairing() == RopeTable.Pairing.INTERLEAVED;
            int hd = g.headDim();
            if (q8 && MetalKernels.FUSE) { // the rotation and the quantization of k and v in one dispatch
                k.ropeStoreQ8(adjacent, q, kStage, vStage, keyCache[layer], valCache[layer], cacheOff,
                        useNorm ? qNorm : null, useNorm ? kNorm : null, cos, sin, hd, g.qDim(), g.heads(), g.kvHeads(), pos,
                        0, rotHalf, n);
            } else {
                k.rope(adjacent, kvHalf, q, kDst, kOff, q8 ? kStage : keyCache[layer], q8 ? 0 : cacheOff,
                        useNorm ? qNorm : null, useNorm ? kNorm : null, cos, sin,
                        hd, g.qDim(), kvDim, g.heads(), g.kvHeads(), pos, 0, rotHalf, n);
                if (q8) {
                    k.kvStoreQ8(kStage, vStage, keyCache[layer], valCache[layer], cacheOff, hd, g.kvHeads(), n);
                }
            }
            if (flashDecode && n == 1 && pos + 1 >= k.flashMinLen) {
                k.flashAttention(kvDtype, q, keyCache[layer], valCache[layer], flashPart, scores, att, g.heads(), g.kvHeads(),
                        hd, pos, ctx, g.scale(), g.qDim(), stride, g.window(), g.softcap());
                return;
            }
            if (flashPrefill && n >= MetalKernels.FLASH_PREFILL_MIN) {
                k.flashPrefill(kvDtype, hd, q, keyCache[layer], valCache[layer], att, g.heads(), g.kvHeads(), pos, n,
                        g.scale(), g.qDim(), stride, g.qDim(), g.window(), g.softcap());
                return;
            }
            k.attention(attnKernel, attnGroup, q, keyCache[layer], valCache[layer], att, scores, g.heads(), g.kvHeads(), hd, hd,
                    pos, ctx, g.scale(), g.qDim(), stride, stride, g.qDim(), g.window(), g.softcap(), n);
        }

        /**
         * The rows' q, k and v in one projection over the n rows (k and v to the f32 rows rowK / rowV, since every
         * row goes to another cache), then per row: QK-norm + RoPE of its q row and of its k row into its owner's
         * cache at its position (rounded to half for f16; for q8 back into rowK), its v row into the cache
         * (kv_put, or kv_store_q8 together with k), and attention over its owner's cache. The per-row dispatches are
         * the single-sequence kernels with the row's offsets, so a row computes exactly what the owner's own
         * decode step would.
         */
        @Override
        public void runRows(int layer, int n, MetalBackend[] owners, int[] positions) {
            int dim = c.dim();
            int kvDim = g.kvDim();
            int hd = g.headDim();
            boolean q8 = kvDtype == KvDtype.Q8;
            int stride = kvDtype.rowStride(kvDim, g.kvHeads());
            long rowBytes = kvDtype.rowBytes(kvDim, g.kvHeads());
            k.rmsnorm(x, 0, attnNorm, xb, n, dim);
            k.matmul3(wq, bwq, wk, bwk, wv, bwv, xb, null, q, rowK, 0, rowV, 0, false, bq, bk, bv, n);
            boolean useNorm = !partial && qNorm != null && kNorm != null;
            int rotHalf = partial ? table.rotaryDim() / 2 : hd / 2;
            boolean adjacent = table.pairing() == RopeTable.Pairing.INTERLEAVED;
            // The rows are independent (each has its own q, k, v and att row and its own owner's cache), so each stage is
            // issued for all rows before the next: with the concurrent encoder the rows of a stage overlap, the barriers
            // go between stages (the serial encoder runs them in this order too, to the same results).
            k.beginIndependent();
            for (int t = 0; t < n; t++) {
                Metal.Buffer kc = owners[t].keyCache[layer];
                long cacheOff = positions[t] * rowBytes;
                long kvOff = (long) t * kvDim * 4;
                k.ropeRow(adjacent, kvHalf, q, (long) t * g.qDim() * 4, rowK, kvOff, q8 ? rowK : kc, q8 ? kvOff : cacheOff,
                        useNorm ? qNorm : null, useNorm ? kNorm : null, cos, sin, hd, g.qDim(), kvDim, g.heads(),
                        g.kvHeads(), positions[t], 0, rotHalf);
                if (!q8) { // (writes the cache's v rows: independent of the rope above, which writes q and the k rows)
                    k.kvPut(kvHalf, rowV, kvOff, owners[t].valCache[layer], cacheOff, kvDim);
                }
            }
            k.endIndependent();
            if (q8) {
                k.beginIndependent();
                for (int t = 0; t < n; t++) {
                    k.kvStoreQ8Row(rowK, (long) t * kvDim * 4, rowV, (long) t * kvDim * 4, owners[t].keyCache[layer],
                            owners[t].valCache[layer], positions[t] * rowBytes, hd, g.kvHeads());
                }
                k.endIndependent();
            }
            // attention: one slice of the scores scratch per row (it holds a tile's worth), so rows can overlap; the
            // flash form shares its partial-result buffer between rows and stays one after the other
            boolean anyFlash = false;
            for (int t = 0; t < n; t++) {
                anyFlash |= flashDecode && positions[t] + 1 >= MetalKernels.FLASH_MIN_LEN;
            }
            int sliceRows = Math.max(1, (int) (scores.size() / ((long) g.heads() * ctx * 4)));
            for (int t = 0; t < n; t++) {
                Metal.Buffer kc = owners[t].keyCache[layer];
                Metal.Buffer vc = owners[t].valCache[layer];
                int pos = positions[t];
                long qOff = (long) t * g.qDim() * 4;
                if (anyFlash) {
                    if (flashDecode && pos + 1 >= MetalKernels.FLASH_MIN_LEN) {
                        k.flashAttentionRow(kvDtype, q, qOff, kc, vc, flashPart, scores, att, qOff, g.heads(), g.kvHeads(),
                                hd, pos, ctx, g.scale(), g.qDim(), stride, g.window(), g.softcap());
                        continue;
                    }
                } else if (t % sliceRows == 0) {
                    k.endIndependent();
                    k.beginIndependent();
                }
                k.rowScores(anyFlash ? 0 : (long) (t % sliceRows) * g.heads() * ctx * 4);
                k.attentionRow(attnKernel, attnGroup, q, qOff, kc, vc, att, qOff, scores, g.heads(), g.kvHeads(), hd,
                        hd, pos, ctx, g.scale(), g.qDim(), stride, stride, g.qDim(), g.window(), g.softcap());
            }
            k.endIndependent();
            k.rowScores(0);
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

    /**
     * Router → per-slot gate/up → SwiGLU → weighted down projections accumulated into x, then DeepSeek's shared experts.
     * The routed experts run on the GPU, on the CPU, or split between the two (hot ones on the GPU, see
     * {@link ExpertPlacement}).
     */
    private final class MoeStep implements FfnStep {
        /** {@code counted}: also write gassign and count the picks (see remap_slot in kernels.metal). */
        private interface Route {
            void run(int n, boolean counted);
        }

        private final Mlp.Moe spec;
        private final MlpWeights.Moe weights;
        private final Metal.Buffer map; // the router's remap: expert -> index among the GPU experts, or -1
        private final Metal.Buffer counts;   // the router counts this layer's picks here (null: not counted)
        private final long[] hostCount;      // the CPU counts them while the layer is entirely on the CPU
        // where the experts are (set by place, again by rebalanceExperts)
        private boolean[] cold;         // null: all on the GPU; else true for an expert left on the CPU
        private boolean allCold;
        private int gpuExperts;         // experts in expGateUp / expDown (all, or the packed hot ones)
        private QTensor expGateUp;
        private QTensor expDown;
        private Metal.Buffer bExpGateUp; // null when the layer has no expert on the GPU
        private Metal.Buffer bExpDown;
        private final Metal.Buffer ffnNorm;
        private final Route route;
        private final QTensor sharedGateUp;
        private final QTensor sharedDown;
        private final Metal.Buffer bSharedGateUp;
        private final Metal.Buffer bSharedDown;

        MoeStep(Mlp.Moe spec, MlpWeights.Moe w, Metal.Buffer ffnNorm, boolean[] cold, int layer) {
            this.spec = spec;
            this.weights = w;
            this.ffnNorm = ffnNorm;
            int e = spec.experts();
            counts = MetalBackend.this.counts != null ? MetalBackend.this.counts[layer] : null;
            hostCount = hostCounts != null ? hostCounts[layer] : null;
            map = counts != null ? k.scratch((long) e * 4) : null;
            place(cold, placement.gateUp[layer], placement.down[layer]);
            Metal.Buffer router = k.upload(w.router());
            int dim = c.dim();
            int topK = spec.topK();
            // the layer's norm (x -> xb) runs inside the router kernel (router_norm / router_ds_norm), unless
            // -Djllm.metal.fuse=false, which brings back rmsnorm + router
            route = switch (spec.router()) {
                case Router.Softmax s -> (n, counted) -> {
                    if (MetalKernels.FUSE) {
                        k.routerNorm(x, ffnNorm, xb, router, assign, routeWeight, e, dim, topK, spec.normTopK(),
                                counted ? map : null, counted ? gassign : null, counted ? counts : null, n);
                    } else {
                        k.rmsnorm(x, 0, ffnNorm, xb, n, dim);
                        k.router(xb, router, assign, routeWeight, e, dim, topK, spec.normTopK(), counted ? map : null,
                                counted ? gassign : null, counted ? counts : null, n);
                    }
                };
                case Router.DeepSeek r -> {
                    Metal.Buffer bias = w.routerBias() != null ? k.upload(w.routerBias()) : null;
                    int method = switch (r.topK()) {
                        case GREEDY -> 0;
                        case GROUP_LIMITED_GREEDY -> 1;
                        case NOAUX_TC -> 2;
                    };
                    yield (n, counted) -> {
                        if (MetalKernels.FUSE) {
                            k.routerDsNorm(x, ffnNorm, xb, router, assign, routeWeight, bias, e, dim, topK,
                                    spec.normTopK(), r.nGroup(), r.topkGroup(), r.sigmoid(), method, r.scaleAlways(),
                                    (float) r.routedScale(), counted ? map : null, counted ? gassign : null,
                                    counted ? counts : null, n);
                        } else {
                            k.rmsnorm(x, 0, ffnNorm, xb, n, dim);
                            k.routerDs(xb, router, assign, routeWeight, bias, e, dim, topK, spec.normTopK(), r.nGroup(),
                                    r.topkGroup(), r.sigmoid(), method, r.scaleAlways(), (float) r.routedScale(),
                                    counted ? map : null, counted ? gassign : null, counted ? counts : null, n);
                        }
                    };
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

        /**
         * Puts the layer's experts where {@code coldRow} says: all on the GPU (null or no cold expert; the mapped tensors
         * are wrapped), all on the CPU (nothing on the GPU), or split ({@code packedGateUp} / {@code packedDown} hold the
         * hot experts). Releases what the previous placement gave the GPU.
         */
        void place(boolean[] coldRow, QTensor packedGateUp, QTensor packedDown) {
            if (bExpGateUp != null) {
                k.release(expGateUp, bExpGateUp);
                k.release(expDown, bExpDown);
            }
            int e = spec.experts();
            int nCold = 0;
            for (int i = 0; coldRow != null && i < e; i++) {
                nCold += coldRow[i] ? 1 : 0;
            }
            cold = nCold > 0 ? coldRow.clone() : null;
            allCold = nCold == e;
            int[] toGpu = new int[e];
            int nHot = 0;
            for (int i = 0; i < e; i++) {
                toGpu[i] = cold != null && cold[i] ? -1 : nHot++;
            }
            if (map != null) {
                MemorySegment.copy(toGpu, 0, map.contents(), ValueLayout.JAVA_INT, 0, e);
            }
            gpuExperts = nHot;
            if (allCold) {
                expGateUp = weights.expGateUp();
                expDown = weights.expDown();
                bExpGateUp = null;   // not on the GPU at all
                bExpDown = null;
            } else {
                // split: the packed hot experts (see Placement.pack); else the mapped tensors, wrapped zero-copy
                expGateUp = cold != null ? packedGateUp : weights.expGateUp();
                expDown = cold != null ? packedDown : weights.expDown();
                bExpGateUp = k.wrap(expGateUp.data);
                bExpDown = k.wrap(expDown.data);
            }
        }

        @Override
        public void run(int layer, int n, boolean fuseNorm) {
            // the router normalizes x into xb (the experts' input) itself; the GPU router counts a split layer on every step (it needs gassign anyway) and a GPU-only layer on sampled
            // steps; a layer entirely on the CPU is counted by the CPU (runOnCpu), from the choices it reads anyway
            route.run(n, counts != null && (cold != null ? !allCold : countStep));
            if (allCold) {
                runOnCpu(n);
            } else if (cold != null) {
                runSplit(n);
            } else {
                gpuRouted(assign, n);
                shared(n);
            }
        }

        /** The routed experts the GPU holds, for the slots of {@code slotExperts} (index into expGateUp / expDown, -1 = skip). */
        private void gpuRouted(Metal.Buffer slotExperts, int n) {
            if (MetalKernels.MOE_MMA && n >= MetalKernels.MOE_MMA_MIN) {
                k.moeGrouped(expGateUp, bExpGateUp, expDown, bExpDown, xb, slotExperts, routeWeight, moeOrder, moeTiles,
                        moeHidden, moeOut, x, gpuExperts, spec.moeDim(), c.dim(), spec.topK(), n);
            } else {
                k.moeUp(expGateUp, bExpGateUp, xb, slotExperts, moeHidden, spec.moeDim(), spec.topK(), n);
                k.moeDown(expDown, bExpDown, moeHidden, slotExperts, routeWeight, x, c.dim(), spec.topK(), n);
            }
        }

        /** Shared experts: a dense SwiGLU of width nShared*moeDim on every token, added to x. */
        private void shared(int n) {
            if (sharedGateUp != null) {
                k.swiglu(sharedGateUp, bSharedGateUp, xb, null, hidden, false, n); // writes silu(gate) * up to hidden
                k.matmul(sharedDown, bSharedDown, hidden, 0, x, 0, n, true, null, 0f);
            }
        }

        /**
         * The GPU stops after the router; the CPU runs the routed experts and adds them to x; the GPU continues.
         * With shared experts (and a native library with jm_wait_pending) the GPU computes those while the CPU
         * computes the routed ones: router | shared go in two command buffers, the CPU waits for the first only.
         */
        private void runOnCpu(int n) {
            if (sharedGateUp != null && k.metal.canWaitPending()) {
                k.commit();               // ... the router
                shared(n);
                k.commit();               // the shared experts, running while ...
                k.metal.waitPending(1);   // ... the CPU, once the router is done,
                cpuRouted(weights, spec, n, null, hostCount);
                k.commitAndWait();        // (the empty buffer the last commit opened: waits for the shared experts)
                addToX(n);
                k.begin();
            } else {
                k.commitAndWait();
                cpuRouted(weights, spec, n, null, hostCount);
                addToX(n);
                k.begin();
                shared(n);
            }
        }

        /**
         * Hot experts on the GPU, cold ones on the CPU, at the same time: the router (with its remap) goes in one command
         * buffer, the GPU's experts and the shared ones in the next; once the first is done the CPU computes the slots
         * that chose a cold expert while the GPU runs the second, then adds them to x after it. When no token chose a
         * cold expert the GPU is not waited for at all. Without jm_wait_pending: router, then CPU, then GPU.
         */
        private void runSplit(int n) {
            if (k.metal.canWaitPending()) {
                k.commit();               // the router (and its remap)
                gpuRouted(gassign, n);
                shared(n);
                k.commit();               // the hot experts, running while ...
                k.metal.waitPending(1);   // ... the CPU takes the cold ones, once the router is done
                if (cpuRouted(weights, spec, n, cold, null)) {
                    k.commitAndWait();    // the GPU's experts are in x: add the CPU's
                    addToX(n);
                    k.begin();
                }                         // else the GPU just goes on; the open buffer records the next layer
            } else {
                k.commitAndWait();
                boolean any = cpuRouted(weights, spec, n, cold, null);
                if (any) {
                    addToX(n);
                }
                k.begin();
                gpuRouted(gassign, n);
                shared(n);
            }
        }
    }

    @Override
    public String name() {
        return "metal (" + k.metal.deviceName() + ")";
    }

    @Override
    public int thermalState() {
        return k.metal.thermalState();
    }

    private long thermalCheck;
    private int lastThermal;

    /** Every 10 s at most: a warning when the machine becomes too hot (the GPU is then slowed down), and when it cools. */
    private void watchThermal() {
        long now = System.nanoTime();
        if (now - thermalCheck < 10_000_000_000L) {
            return;
        }
        thermalCheck = now;
        int t = thermalState();
        if (t >= 2 && t > lastThermal) {
            System.err.println("warning: thermal state " + (t == 3 ? "critical" : "serious")
                    + ": macOS is slowing the GPU down, expect fewer tokens per second");
        } else if (t >= 0 && t < 2 && lastThermal >= 2) {
            System.err.println("thermal state back to " + (t == 0 ? "nominal" : "fair"));
        }
        lastThermal = t;
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
    /**
     * Records the decoder layers for n tokens at pos (x holds their embeddings). For a single token the rmsnorm
     * preceding a projection is folded into the projection kernel (the *_norm variants read x and normalise on the
     * fly); otherwise it writes xb first. {@code split}: submit every COMMIT_LAYERS layers so the GPU starts while the
     * rest is encoded (not in a pipelined step, which must stay one command buffer).
     */
    private void encodeLayers(int n, int pos, boolean split) {
        int dim = c.dim();
        boolean fuseNorm = n == 1 && fusedNormFits;
        for (int l = 0; l < c.layers(); l++) {
            LayerPlan L = plans[l];
            L.attention().run(l, n, pos, fuseNorm);
            if (L.postAttnNorm() != null) {
                // Glm4 / Gemma: x += rmsnorm(o_proj(attention)), so o_proj cannot accumulate into x directly
                k.matmul(L.wo(), L.woBuf(), att, 0, xb, 0, n, false, null, 0f, L.woBias());
                k.rmsnormAdd(xb, L.postAttnNorm(), x, n, dim);
            } else {
                k.matmul(L.wo(), L.woBuf(), att, 0, x, 0, n, true, null, 0f, L.woBias()); // x += o_proj(att) (+ bias)
            }
            L.ffn().run(l, n, fuseNorm);
            if (split && MetalKernels.COMMIT_LAYERS > 0 && (l + 1) % MetalKernels.COMMIT_LAYERS == 0 && l + 1 < c.layers()) {
                k.commit(); // the GPU starts on these layers while the next ones are encoded
            }
        }
    }

    // ------------------------------------------------------------------ decode replay

    /**
     * Whether single-token decode steps are recorded and replayed ({@link MetalKernels#step}): on by default, off with
     * {@code -Djllm.metal.replay=false}, with a native library that has no {@code jm_run}, and while any expert runs on
     * the CPU (such a step waits for the CPU halfway, so it is not one program).
     */
    private boolean replayable() {
        return replay && cpuLayers == 0;
    }

    /**
     * What a recorded decode step depends on besides its position and ring slots: pipelined or not, whether this pass
     * counts the experts' picks (the router binds the counters), and flash decoding: on or off at this position and,
     * when on, the number of splits for each attention window (the grid of attention_split; the keys per split follow
     * the position inside the kernel). Everything else (weights, buffers, kernel choices) is fixed for the backend's
     * life or invalidates the recordings when it changes (rebalanceExperts, a new pipe vocabulary).
     */
    private long stepKey(boolean pipelined, int pos) {
        long key = (pipelined ? 1 : 0) | (countStep ? 2 : 0);
        if (flashDecode && pos + 1 >= k.flashMinLen) {
            key |= 4;
            for (int i = 0; i < windows.length; i++) {
                key |= (long) k.flashSplits(pos, windows[i]) << (3 + 7 * i);
            }
        }
        return key;
    }

    /**
     * Selects the concurrent compute encoder (true: independent dispatches overlap, a barrier goes between dependent ones)
     * or the serial one (false) for the passes that follow; the results are bit-identical. The default is
     * {@code -Djllm.metal.concurrent} (docs/08 section 8.26).
     */
    public void concurrent(boolean on) {
        k.concurrent(on ? MetalKernels.concurrentOn() : 0);
    }

    /** Memory barriers the concurrent encoder has inserted so far. */
    public long barriers() {
        return k.metal.barriers();
    }

    /** Turns decode replay on or off for this backend (on only where it can work: see {@link #replayable}). */
    public void replay(boolean on) {
        replay = on && k.metal.canRun() && windows.length <= 8;
        k.invalidatePrograms();
    }

    /** Whether decode replay is on for this backend now. */
    public boolean replaying() {
        return replayable();
    }

    /** Java → native calls of the forward passes so far (see {@link Metal#downcalls}). */
    public long downcalls() {
        return k.metal.downcalls();
    }

    /** Kernel dispatches encoded so far. */
    public long dispatches() {
        return k.metal.dispatches();
    }

    /** Decode steps run from a recording / recorded so far. */
    public long replayedSteps() {
        return k.replayedSteps();
    }

    public long recordedSteps() {
        return k.recordedSteps();
    }

    /**
     * Forces flash decoding on for steps over at least {@code minLen} positions, with {@code chunk} keys per split, so a
     * test can cross the kernel-variant thresholds of a long context within a few tokens. Dense attention only.
     */
    public void flashDecodeThresholds(int minLen, int chunk) {
        if (!(arch.attention() instanceof Attention.Gqa)) {
            return;
        }
        if (flashPart == null) {
            flashPart = k.scratch((long) arch.attention().heads() * MetalKernels.FLASH_SPLITS * (arch.attention().headDim() + 2) * 4);
        }
        flashDecode = true;
        k.flashMinLen = minLen;
        k.flashChunk = chunk;
        k.invalidatePrograms();
    }

    // ------------------------------------------------------------------ pipelined greedy decoding

    private Pipe pipe; // the one pipe of this backend, created on first use

    @Override
    public GreedyPipe greedyPipe(int vocab) {
        if (!k.metal.canWaitPending() || !PIPELINE || cpuLayers > 0) {
            return null; // a step with experts on the CPU cannot run ahead of the CPU
        }
        if (pipe == null) {
            pipe = new Pipe();
        }
        pipe.vocab = Math.min(vocab, w.lmHead.rows);
        return pipe;
    }

    /** {@code -Djllm.metal.pipeline=false} turns pipelined greedy decoding off. */
    static final boolean PIPELINE = !"false".equals(System.getProperty("jllm.metal.pipeline"));

    /**
     * Step j reads its token from ring slot j % R (written by step j - 1's argmax, or by the CPU for an explicit token,
     * which is only allowed when nothing is in flight) and its argmax goes to slot (j + 1) % R. Each step is one
     * command buffer, so "step j is done" is "at most submitted - j - 1 buffers are unfinished" (jm_wait_pending).
     */
    private final class Pipe implements GreedyPipe {
        private static final int R = 4;
        private final Metal.Buffer ring = k.scratch(R * 4);
        private long submitted;
        private long read;
        private boolean open; // a command buffer is being recorded (between begin / commit and commitAndWait)
        int vocab;

        @Override
        public void submit(int token, int pos) {
            if (pos >= ctx) {
                throw new IllegalArgumentException("position " + pos + " is past the context");
            }
            int slot = (int) (submitted % R);
            if (token >= 0) {
                if (submitted != read) {
                    throw new IllegalStateException("an explicit token while steps are in flight");
                }
                ring.contents().setAtIndex(ValueLayout.JAVA_INT, slot, token);
            }
            if (!open) {
                k.begin();
                open = true;
            }
            boolean ok = false;
            try {
                int out = (slot + 1) % R;
                Runnable body = () -> {
                    k.embedToken(w.embed, embedQ, ring, slot, x, c.dim(), c.embedScale());
                    encodeLayers(1, pos, false);
                    k.rmsnorm(x, 0, finalNorm, xb, 1, c.dim());
                    k.matmul(w.lmHead, lmHeadQ, xb, 0, logitsBuf, 0, 1, false, null, c.finalSoftcap());
                    k.argmax(logitsBuf, vocab, ring, out);
                };
                if (replayable()) {
                    if (keyedVocab != vocab) { // the argmax range is part of every recorded pipe step
                        k.invalidatePrograms();
                        keyedVocab = vocab;
                    }
                    k.step(stepKey(true, pos), pos, slot, out, body);
                } else {
                    body.run();
                }
                k.commit(); // runs now; a fresh command buffer is open for the next step
                ok = true;
            } finally {
                if (!ok) {
                    k.abort();
                    open = false;
                    read = submitted;
                }
            }
            submitted++;
        }

        @Override
        public int next() {
            if (read >= submitted) {
                throw new IllegalStateException("no step in flight");
            }
            k.metal.waitPending((int) (submitted - read - 1));
            int t = ring.contents().getAtIndex(ValueLayout.JAVA_INT, (int) ((read + 1) % R));
            read++;
            return t;
        }

        @Override
        public int pending() {
            return (int) (submitted - read);
        }

        @Override
        public void drain() {
            if (open) {
                open = false;
                k.commitAndWait(); // the empty buffer opened by the last commit, after every step before it
            }
            read = submitted;
        }
    }

    private void run(int[] tokens, int off, int n, int pos, float[] logits, int stride) {
        if (pipe != null) {
            pipe.drain(); // never interleave with pipelined steps
        }
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

        countStep = counts != null && steps++ % COUNT_EVERY == 0;
        watchThermal();
        // a decode step: recorded once per shape, then replayed with its position patched in (one native call)
        boolean decodeReplay = n == 1 && logits != null && stride == 0 && replayable();
        k.begin();
        boolean committed = false;
        try {
            if (decodeReplay) {
                k.step(stepKey(false, pos), pos, 0, 0, () -> {
                    encodeLayers(1, pos, false);
                    k.rmsnorm(x, 0, finalNorm, xb, 1, dim);
                    k.matmul(w.lmHead, lmHeadQ, xb, 0, logitsBuf, 0, 1, false, null, c.finalSoftcap());
                });
            } else {
                encodeLayers(n, pos, true);
            }
            if (decodeReplay) {
                // the final norm and the LM head are part of the step
            } else if (logits != null && stride > 0) {
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
        if (counts != null && System.nanoTime() - lastRebalance > REBALANCE_NANOS) {
            rebalanceExperts(); // saves the statistics first
        } else if (counts != null && System.nanoTime() - lastStatsSave > STATS_SAVE_NANOS) {
            saveExpertStats(true); // a long-running server keeps its statistics current, not only at shutdown
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

    // ------------------------------------------------------------------ batched decoding across slots

    /** {@code -Djllm.metal.batchDecode=false}: no batched passes (the batch group's slots then take turns). */
    static final boolean BATCH_DECODE = !"false".equals(System.getProperty("jllm.metal.batchDecode"));

    /** The f32 k / v rows of a batched step ([maxBatch, kvDim] each), created by the first batched pass. */
    private Metal.Buffer rowK;
    private Metal.Buffer rowV;

    /**
     * One decode step for each of several slots on these weights, in one pass on this backend's command queue and
     * activation buffers: the projections, the MLP (dense or MoE, with experts on the CPU too) and the LM head run over
     * the n rows with the multi-row kernels, so every weight is read once for the whole batch; RoPE, the cache store
     * and attention run per row in the member's own KV cache at its own position ({@code AttnStep.runRows}). The members
     * must be idle MetalBackends of the same weights and cache type on the same device, which {@link
     * org.robomiri.jllm.engine.BatchGroup} guarantees by holding its device lock. Every member's pipelined steps are
     * drained first.
     */
    @Override
    public boolean forwardBatch(Backend[] members, int[] tokens, int[] positions, float[][] logits) {
        int n = members.length;
        if (!BATCH_DECODE || closed || n < 1 || n > maxBatch) {
            return false;
        }
        MetalBackend[] owners = new MetalBackend[n];
        boolean self = false;
        for (int t = 0; t < n; t++) {
            if (!(members[t] instanceof MetalBackend o) || o.w != w || o.kvDtype != kvDtype || o.closed
                    || positions[t] < 0 || positions[t] >= Math.min(ctx, o.ctx) || (mla == null) != (o.mla == null)) {
                return false;
            }
            for (int u = 0; u < t; u++) {
                if (owners[u] == o) {
                    return false; // one cache cannot take two rows of a batch
                }
            }
            owners[t] = o;
            self |= o == this;
        }
        if (!self) {
            return false;
        }
        for (MetalBackend o : owners) {
            if (o.pipe != null) {
                o.pipe.drain(); // never interleave with pipelined steps (of any member: the group lock is held)
            }
        }
        if (arch.attention() instanceof Attention.Gqa g && rowK == null) {
            rowK = k.scratch((long) maxBatch * g.kvDim() * 4);
            rowV = k.scratch((long) maxBatch * g.kvDim() * 4);
        }
        if (logitsAllBuf == null || logitsAllRows < n) {
            logitsAllRows = Math.min(maxBatch, Math.max(n, 16)); // lives until the backend closes: size once
            logitsAllBuf = k.scratch((long) logitsAllRows * w.lmHead.rows * 4);
        }
        int dim = c.dim();
        float[] row = new float[dim];
        for (int t = 0; t < n; t++) {
            w.embed.dequantizeRow(tokens[t], row, 0);
            if (c.embedScale() != 1f) {
                for (int i = 0; i < dim; i++) {
                    row[i] *= c.embedScale();
                }
            }
            MemorySegment.copy(row, 0, x.contents(), ValueLayout.JAVA_FLOAT, (long) t * dim * 4, dim);
        }
        countStep = counts != null && steps++ % COUNT_EVERY == 0;
        watchThermal();
        k.begin();
        boolean committed = false;
        try {
            for (int l = 0; l < c.layers(); l++) {
                LayerPlan L = plans[l];
                L.attention().runRows(l, n, owners, positions);
                if (L.postAttnNorm() != null) {
                    k.matmul(L.wo(), L.woBuf(), att, 0, xb, 0, n, false, null, 0f, L.woBias());
                    k.rmsnormAdd(xb, L.postAttnNorm(), x, n, dim);
                } else {
                    k.matmul(L.wo(), L.woBuf(), att, 0, x, 0, n, true, null, 0f, L.woBias());
                }
                L.ffn().run(l, n, false);
                if (MetalKernels.COMMIT_LAYERS > 0 && (l + 1) % MetalKernels.COMMIT_LAYERS == 0 && l + 1 < c.layers()) {
                    k.commit();
                }
            }
            k.rmsnorm(x, 0, finalNorm, xb, n, dim);
            k.matmul(w.lmHead, lmHeadQ, xb, 0, logitsAllBuf, 0, n, false, null, c.finalSoftcap());
            k.commitAndWait();
            committed = true;
        } finally {
            if (!committed) {
                k.abort();
            }
        }
        if (counts != null && System.nanoTime() - lastRebalance > REBALANCE_NANOS) {
            rebalanceExperts();
        } else if (counts != null && System.nanoTime() - lastStatsSave > STATS_SAVE_NANOS) {
            saveExpertStats(true);
        }
        for (int t = 0; t < n; t++) {
            MemorySegment.copy(logitsAllBuf.contents(), ValueLayout.JAVA_FLOAT, (long) t * w.lmHead.rows * 4, logits[t], 0,
                    Math.min(w.lmHead.rows, logits[t].length));
        }
        return true;
    }

    /** Bytes this backend keeps resident on the GPU: the weights given to it (minus experts on the CPU) and its buffers. */
    public long gpuResidentBytes() {
        return gpuBytes(placement != null ? placement.cold : new boolean[c.layers()][]);
    }

    private static final long REBALANCE_NANOS = 30L * 60 * 1_000_000_000L;
    private long lastRebalance = System.nanoTime();

    /**
     * Plans the expert placement again from the usage counted so far and, if the new plan is clearly cheaper (by the
     * planner's own cost, at least 20 %), moves the experts of the layers that change, one layer at a time: the new
     * hot experts are packed, the GPU switches to them, then the old copy is freed, so at most one extra layer's
     * copy exists at any moment. Only for {@code backend.metal.cpu-moe=auto} with usage statistics, and only while
     * this backend is the placement's sole user (slots sharing one keep the plan made at load). Runs on its own every
     * 30 minutes of use; between forward passes, so the GPU is idle.
     *
     * @return the number of layers whose experts moved
     */
    public int rebalanceExperts() {
        if (counts == null || split.cold() != null || split.layers() != MoeSplit.AUTO || !HOT_EXPERTS || closed) {
            return 0;
        }
        synchronized (PLACEMENTS) {
            if (placement.refs > 1) {
                return 0;
            }
        }
        lastRebalance = System.nanoTime();
        saveExpertStats(false);
        int layers = c.layers();
        long[][] usage = ExpertPlacement.load(statsFile, layers, modelKey);
        if (usage == null) {
            return 0;
        }
        long[] perExpert = new long[layers];
        int[] experts = new int[layers];
        int[] topK = new int[layers];
        for (int l = 0; l < layers; l++) {
            if (arch.layers().get(l).mlp() instanceof Mlp.Moe m) {
                perExpert[l] = bytesPerExpert(l);
                experts[l] = m.experts();
                topK[l] = m.topK();
            }
        }
        long need = gpuBytes(new boolean[layers][]) + DeviceLedger.others(ledger) - workingSet();
        boolean[][] next = ExpertPlacement.plan(need, perExpert, experts, topK, usage);
        double before = ExpertPlacement.cost(placement.cold, perExpert, experts, topK, usage);
        double after = ExpertPlacement.cost(next, perExpert, experts, topK, usage);
        if (!(after < 0.8 * before)) {
            return 0;
        }
        int changed = 0;
        for (int l = 0; l < layers; l++) {
            if (next[l] == null || java.util.Arrays.equals(next[l], placement.cold[l])) {
                continue;
            }
            java.lang.foreign.Arena old = placement.arenas[l];
            placement.pack(l, w.decoderLayer(l).moe(), next[l], ((Mlp.Moe) arch.layers().get(l).mlp()).moeDim(), c.dim());
            ((MoeStep) plans[l].ffn()).place(next[l], placement.gateUp[l], placement.down[l]);
            placement.cold[l] = next[l];
            if (old != null) {
                old.close(); // the GPU's buffers over it were released by place
            }
            changed++;
        }
        k.invalidatePrograms(); // recorded steps bind the experts' old buffers (place released them, too)
        cpuLayers = 0;
        for (boolean[] row : placement.cold) {
            boolean any = false;
            for (int e = 0; row != null && e < row.length; e++) {
                any |= row[e];
            }
            cpuLayers += any ? 1 : 0;
        }
        System.err.printf("Metal: experts re-placed from the usage so far: %d layers changed, estimated cost per token "
                + "%.1f -> %.1f us%n", changed, before, after);
        return changed;
    }

    /** Drops this backend's hold on its placement; the last backend on it frees the packed experts. */
    private void releasePlacement() {
        if (placement == null) {
            return;
        }
        synchronized (PLACEMENTS) {
            if (--placement.refs == 0) {
                if (placementShared) {
                    PLACEMENTS.remove(w, placement);
                }
                if (placement.pool != null) {
                    placement.pool.close();
                }
                placement.close();
            }
        }
        placement = null;
    }

    // ------------------------------------------------------------------ saved caches

    /**
     * The cache rows as CpuBackend saves them: for standard attention ({@code KvCache.write}) per layer, per position,
     * the K row then the V row, little-endian f32 or IEEE half, or q8's kvDim int8 values followed by the kvHeads f32
     * scales; for DeepSeek's latent attention ({@code MlaCpu.write}) per layer, per position, the latent row
     * {@code [c | k_pe]} in f32 or half. Metal's {@code half} has the same bits and its q8 cache rows are that layout
     * already, so a cache saved by either backend has the same layout. A q8 latent cache is not saved.
     */
    @Override
    public String stateFormat() {
        if (arch.attention() instanceof Attention.Gqa g) {
            return "kv " + kvDtype + " layers=" + c.layers() + " kvDim=" + g.kvDim() + " headDim=" + g.headDim();
        }
        return mla != null && kvDtype != KvDtype.Q8 ? "mla " + mla.format() : null;
    }

    /** Bytes of one row of one cache segment. */
    private long rowBytes() {
        if (mla != null) {
            return mla.rowBytes();
        }
        Attention.Gqa g = (Attention.Gqa) arch.attention();
        return kvDtype.rowBytes(g.kvDim(), g.kvHeads());
    }

    /** The cache segments of layer {@code l} whose rows are saved one after the other per position: K and V, or the latent rows. */
    private MemorySegment[] segments(int l) {
        return mla != null ? new MemorySegment[]{mla.cache(l)}
                : new MemorySegment[]{keyCache[l].contents(), valCache[l].contents()};
    }

    @Override
    public void saveState(int n, java.nio.channels.WritableByteChannel out) throws java.io.IOException {
        if (pipe != null) {
            pipe.drain();
        }
        saveRows(n, out);
    }

    @Override
    public boolean canSaveWhileServing() {
        return stateFormat() != null;
    }

    /** Copies rows {@code [0, n)} out of the shared buffers without touching the command queue or the pipe. */
    @Override
    public void saveStateWhileServing(int n, java.nio.channels.WritableByteChannel out) throws java.io.IOException {
        saveRows(n, out);
    }

    private void saveRows(int n, java.nio.channels.WritableByteChannel out) throws java.io.IOException {
        if (stateFormat() == null) {
            Backend.super.saveState(n, out);
            return;
        }
        long row = rowBytes();
        int segs = mla != null ? 1 : 2;
        int rows = (int) Math.max(1, Math.min(n, (1 << 22) / (segs * row)));
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocateDirect((int) (rows * segs * row));
        MemorySegment b = MemorySegment.ofBuffer(buf);
        for (int l = 0; l < c.layers(); l++) {
            MemorySegment[] seg = segments(l);
            for (int s0 = 0; s0 < n; s0 += rows) {
                int m = Math.min(rows, n - s0);
                for (int r = 0; r < m; r++) {
                    for (int j = 0; j < segs; j++) {
                        MemorySegment.copy(seg[j], (s0 + r) * row, b, ((long) segs * r + j) * row, row);
                    }
                }
                buf.clear().limit((int) (segs * m * row));
                while (buf.hasRemaining()) {
                    out.write(buf);
                }
            }
        }
    }

    @Override
    public void loadState(int n, java.nio.channels.ReadableByteChannel in) throws java.io.IOException {
        loadState(n, n, in);
    }

    @Override
    public void loadState(int n, int stored, java.nio.channels.ReadableByteChannel in) throws java.io.IOException {
        if (pipe != null) {
            pipe.drain();
        }
        if (stateFormat() == null) {
            throw new UnsupportedOperationException(name() + " cannot load a saved cache");
        }
        if (n > ctx) {
            throw new java.io.IOException("the saved cache holds " + n + " positions, the context is " + ctx);
        }
        if (n > stored) {
            throw new IllegalArgumentException(n + " positions requested, " + stored + " saved");
        }
        long row = rowBytes();
        int segs = mla != null ? 1 : 2;
        int rows = (int) Math.max(1, Math.min(n, (1 << 22) / (segs * row)));
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocateDirect((int) (rows * segs * row));
        MemorySegment b = MemorySegment.ofBuffer(buf);
        for (int l = 0; l < c.layers(); l++) {
            MemorySegment[] seg = segments(l);
            for (int s0 = 0; s0 < n; s0 += rows) {
                int m = Math.min(rows, n - s0);
                buf.clear().limit((int) (segs * m * row));
                while (buf.hasRemaining()) {
                    if (in.read(buf) < 0) {
                        throw new java.io.EOFException("the saved cache file is truncated");
                    }
                }
                // straight into the cache buffers: on unified memory they are the GPU's memory
                for (int r = 0; r < m; r++) {
                    for (int j = 0; j < segs; j++) {
                        MemorySegment.copy(b, ((long) segs * r + j) * row, seg[j], (s0 + r) * row, row);
                    }
                }
            }
            Backend.skipState(in, (long) (stored - n) * segs * row);
        }
    }

    @Override
    public void copyStateFrom(Backend other, int n) {
        if (stateFormat() == null || !(other instanceof MetalBackend o) || !stateFormat().equals(o.stateFormat())) {
            Backend.super.copyStateFrom(other, n);
            return;
        }
        if (pipe != null) {
            pipe.drain();
        }
        if (o.pipe != null) {
            o.pipe.drain();
        }
        long bytes = (long) n * rowBytes();
        for (int l = 0; l < c.layers(); l++) {
            MemorySegment[] from = o.segments(l);
            MemorySegment[] to = segments(l);
            for (int j = 0; j < to.length; j++) {
                MemorySegment.copy(from[j], 0, to[j], 0, bytes);
            }
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (pipe != null) {
            try {
                pipe.drain();
            } catch (RuntimeException ignored) {
                // closing anyway
            }
        }
        try {
            saveExpertStats(false);
        } catch (RuntimeException e) {
            System.err.println("warning: expert statistics not saved: " + e);
        }
        try {
            k.close();
        } finally {
            DeviceLedger.remove(ledger);
            releasePlacement();
        }
    }
}
