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
    private Metal.Buffer gassign;     // moe_remap output: each slot's expert among the layer's GPU experts, or -1
    // how often each expert was picked, per MoE layer, while any expert is on the CPU: counted by moe_remap on the GPU
    // (layers with GPU experts) or here (layers entirely on the CPU, from the router's choices the CPU reads anyway)
    private Metal.Buffer[] counts;
    private long[][] hostCounts;
    private boolean countStep;        // this forward pass counts the layers entirely on the GPU (every COUNT_EVERY-th)
    private long steps;
    private java.nio.file.Path statsFile;
    private String modelKey;
    private long lastStatsSave;
    private boolean closed;

    /**
     * Layers whose experts are all on the GPU are counted on one forward pass in this many (a moe_remap dispatch per
     * layer per token would cost more than it tells): the planner compares experts within a layer, so a sample is enough.
     */
    private static final int COUNT_EVERY = Math.max(1, Integer.getInteger("jllm.metal.expertCountEvery", 8));

    /**
     * Where the experts of one {@link ModelWeights} live, and the packed copies of the hot experts of its split layers.
     * Shared by every MetalBackend on those weights with the same {@link MoeSplit} (a server's extra slots), so the
     * copies exist once and every slot runs the same plan; freed with the last of them.
     */
    private static final class Placement {
        final String config;
        final boolean[][] cold;
        final QTensor[] gateUp;   // per layer: the packed hot experts of a split layer, else null
        final QTensor[] down;
        final java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofShared();
        /** The CPU threads for the experts, one set for all the backends (a pool runs one caller's job at a time). */
        ThreadPool pool;
        int refs = 1;

        Placement(String config, boolean[][] cold) {
            this.config = config;
            this.cold = cold;
            this.gateUp = new QTensor[cold.length];
            this.down = new QTensor[cold.length];
        }
    }

    private static final java.util.Map<ModelWeights, Placement> PLACEMENTS = new java.util.IdentityHashMap<>();

    /**
     * Every open MetalBackend of the process: Metal's working set is the device's, so a placement must leave room for
     * the other backends' weights and buffers (another model, a draft model, a server's other slots).
     */
    private static final java.util.Set<MetalBackend> LIVE = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());

    /** GPU-resident bytes of all open backends but {@code self}: their buffers, and once per weights their GPU share. */
    private static long othersOnDevice(MetalBackend self) {
        synchronized (LIVE) {
            long b = 0;
            java.util.Set<ModelWeights> counted = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
            for (MetalBackend o : LIVE) {
                if (o == self) {
                    continue;
                }
                b += o.k.metal.allocatedBytes();
                if (o.w != self.w && counted.add(o.w)) {
                    b += o.gpuBytes(o.placement.cold) - o.k.metal.allocatedBytes();
                }
            }
            return b;
        }
    }

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
        if (ctx > c.maxPositions()) {
            System.err.printf("warning: ctx %d exceeds the model's max_position_embeddings (%d); RoPE positions "
                    + "beyond that are extrapolated and quality degrades%n", ctx, c.maxPositions());
        }
        long gpu = gpuBytes(cold);
        long others = othersOnDevice(this);
        synchronized (LIVE) {
            LIVE.add(this);
        }
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
                if (cold[l] == null) {
                    continue;
                }
                boolean allCold = true;
                for (boolean v : cold[l]) {
                    allCold &= v;
                }
                if (allCold) {
                    hostCounts[l] = new long[experts[l]];
                } else {
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
            long need = w.byteSize() + split.backends() * (k.metal.allocatedBytes() + laterUploads()) + othersOnDevice(this)
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
                    MlpWeights.Moe mw = w.decoderLayer(l).moe();
                    int[] h = java.util.Arrays.copyOf(hot, nHot);
                    p.gateUp[l] = mw.expGateUp().gatherExperts(p.arena, h, 2 * ((Mlp.Moe) arch.layers().get(l).mlp()).moeDim());
                    p.down[l] = mw.expDown().gatherExperts(p.arena, h, c.dim());
                }
            }
        } catch (RuntimeException | Error e) {
            p.arena.close();
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
            b += 6L * c.dim() * 4; // up to four norms, q/k/v biases are smaller than the rest together
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
            if (hostCounts[l] != null) {
                delta[l] = hostCounts[l].clone();
                java.util.Arrays.fill(hostCounts[l], 0);
            } else if (counts[l] != null) {
                MemorySegment cs = counts[l].contents();
                delta[l] = new long[(int) (counts[l].size() / 4)];
                for (int i = 0; i < delta[l].length; i++) {
                    delta[l][i] = Integer.toUnsignedLong(cs.getAtIndex(ValueLayout.JAVA_INT, i));
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
     * {@code hostCount}: adds the router's choices to it (a layer counted here instead of by moe_remap).
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
            case Attention.Mla m -> (layer, n, pos, fuseNorm) -> {
                k.rmsnorm(x, 0, attnNorm, xb, n, c.dim());
                mla.run(layer, n, pos);
            };
        };
        FfnStep ffn = switch (spec.mlp()) {
            case Mlp.Dense d -> new DenseStep(d, L.dense(), ffnNorm, postMlpNorm);
            case Mlp.Moe m -> new MoeStep(m, L.moe(), ffnNorm, cold, index);
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

    /**
     * Router → per-slot gate/up → SwiGLU → weighted down projections accumulated into x, then DeepSeek's shared experts.
     * The routed experts run on the GPU, on the CPU, or split between the two (hot ones on the GPU, see
     * {@link ExpertPlacement}).
     */
    private final class MoeStep implements FfnStep {
        private interface Route {
            void run(int n);
        }

        private final Mlp.Moe spec;
        private final MlpWeights.Moe weights;
        private final boolean[] cold;   // null: all on the GPU; else true for an expert left on the CPU
        private final boolean allCold;
        private final Metal.Buffer map; // moe_remap: expert -> index among the GPU experts, or -1
        private final Metal.Buffer counts;   // moe_remap counts this layer's picks (null: not counted on the GPU)
        private final long[] hostCount;      // the CPU counts them (a layer entirely on the CPU)
        private final int gpuExperts;   // experts in expGateUp / expDown (all, or the packed hot ones)
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

        MoeStep(Mlp.Moe spec, MlpWeights.Moe w, Metal.Buffer ffnNorm, boolean[] cold, int layer) {
            this.spec = spec;
            this.weights = w;
            this.ffnNorm = ffnNorm;
            int e = spec.experts();
            int nCold = 0;
            for (int i = 0; cold != null && i < e; i++) {
                nCold += cold[i] ? 1 : 0;
            }
            this.cold = nCold > 0 ? cold : null;
            allCold = nCold == e;
            counts = MetalBackend.this.counts != null ? MetalBackend.this.counts[layer] : null;
            hostCount = hostCounts != null ? hostCounts[layer] : null;
            int[] hot = new int[e - nCold];
            int[] toGpu = new int[e];
            for (int i = 0, j = 0; i < e; i++) {
                toGpu[i] = this.cold != null && cold[i] ? -1 : j;
                if (toGpu[i] >= 0) {
                    hot[j++] = i;
                }
            }
            if (counts != null) {
                map = k.scratch((long) e * 4);
                MemorySegment.copy(toGpu, 0, map.contents(), ValueLayout.JAVA_INT, 0, e);
            } else {
                map = null;
            }
            if (allCold) {
                expGateUp = w.expGateUp();
                expDown = w.expDown();
                bExpGateUp = null;   // not on the GPU at all
                bExpDown = null;
            } else if (this.cold != null) {
                expGateUp = placement.gateUp[layer]; // the packed hot experts (see newPlacement)
                expDown = placement.down[layer];
                bExpGateUp = k.wrap(expGateUp.data);
                bExpDown = k.wrap(expDown.data);
            } else {
                expGateUp = w.expGateUp();
                expDown = w.expDown();
                bExpGateUp = k.wrap(expGateUp.data);
                bExpDown = k.wrap(expDown.data);
            }
            gpuExperts = hot.length;
            Metal.Buffer router = k.upload(w.router());
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
            if (counts != null && (cold != null || countStep)) { // a split layer needs gassign on every step
                k.moeRemap(assign, map, gassign, counts, spec.experts(), spec.topK(), n);
            }
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
         * Hot experts on the GPU, cold ones on the CPU, at the same time: the router (and moe_remap) go in one command
         * buffer, the GPU's experts and the shared ones in the next; once the first is done the CPU computes the slots
         * that chose a cold expert while the GPU runs the second, then adds them to x after it. When no token chose a
         * cold expert the GPU is not waited for at all. Without jm_wait_pending: router, then CPU, then GPU.
         */
        private void runSplit(int n) {
            if (k.metal.canWaitPending()) {
                k.commit();               // the router and moe_remap
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
                k.matmul(L.wo(), L.woBuf(), att, 0, xb, 0, n, false, null, 0f);
                k.rmsnormAdd(xb, L.postAttnNorm(), x, n, dim);
            } else {
                k.matmul(L.wo(), L.woBuf(), att, 0, x, 0, n, true, null, 0f);
            }
            L.ffn().run(l, n, fuseNorm);
            if (split && MetalKernels.COMMIT_LAYERS > 0 && (l + 1) % MetalKernels.COMMIT_LAYERS == 0 && l + 1 < c.layers()) {
                k.commit(); // the GPU starts on these layers while the next ones are encoded
            }
        }
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
                k.embedToken(w.embed, embedQ, ring, slot, x, c.dim(), c.embedScale());
                encodeLayers(1, pos, false);
                k.rmsnorm(x, 0, finalNorm, xb, 1, c.dim());
                k.matmul(w.lmHead, lmHeadQ, xb, 0, logitsBuf, 0, 1, false, null, c.finalSoftcap());
                k.argmax(logitsBuf, vocab, ring, (slot + 1) % R);
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
        k.begin();
        boolean committed = false;
        try {
            encodeLayers(n, pos, true);
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
        if (counts != null && System.nanoTime() - lastStatsSave > STATS_SAVE_NANOS) {
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
                placement.arena.close();
            }
        }
        placement = null;
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
        if (pipe != null) {
            pipe.drain();
        }
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
        if (pipe != null) {
            pipe.drain();
        }
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
        if (pipe != null) {
            pipe.drain();
        }
        if (o.pipe != null) {
            o.pipe.drain();
        }
        long bytes = (long) n * rowBytes();
        for (int l = 0; l < c.layers(); l++) {
            MemorySegment.copy(o.keyCache[l].contents(), 0, keyCache[l].contents(), 0, bytes);
            MemorySegment.copy(o.valCache[l].contents(), 0, valCache[l].contents(), 0, bytes);
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
            synchronized (LIVE) {
                LIVE.remove(this);
            }
            releasePlacement();
        }
    }
}
