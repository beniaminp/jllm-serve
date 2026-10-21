package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.robomiri.jllm.metal.MetalConstants.ATT_THREADS;
import static org.robomiri.jllm.metal.MetalConstants.ATT_TILE;
import static org.robomiri.jllm.metal.MetalConstants.CHANNEL_THREADS;
import static org.robomiri.jllm.metal.MetalConstants.FLASH_MAX_SPLITS;
import static org.robomiri.jllm.metal.MetalConstants.MM_THREADS;
import static org.robomiri.jllm.metal.MetalConstants.MMA_BR;
import static org.robomiri.jllm.metal.MetalConstants.MMA_BT;
import static org.robomiri.jllm.metal.MetalConstants.PF_ROWS;
import static org.robomiri.jllm.metal.MetalConstants.PF_TOKS;
import static org.robomiri.jllm.metal.MetalConstants.PF_UNITS;
import static org.robomiri.jllm.metal.MetalConstants.ROW_THREADS;
import static org.robomiri.jllm.metal.MetalConstants.TOK_TILE;

/**
 * The kernel-side half of the Metal backends: owns the {@link Metal} context, the dispatch argument scratch and
 * a writer per kernel of {@code kernels.metal} (buffers, parameter struct, grid). {@link MetalBackend},
 * {@link MlaMetal} and {@link MambaMetalBackend} own the model's buffers and orchestrate the forward pass by
 * calling these writers; nothing else touches parameter layouts, so each kernel's struct has exactly one writer
 * here, written through {@link Params} against its {@link KernelParams} layout (checked at the dispatch and, against
 * the shader source, by {@code MetalParamsTest}).
 *
 * <p>Kernels are looked up by name at dispatch time ({@link Metal#pipeline} memoizes and compiles on first use),
 * so only the variants a model runs are built.
 */
final class MetalKernels implements AutoCloseable {

    // ---------------------------------------------------------------- derived from the kernels' constants
    /** Simdgroups (= rows) per matmul threadgroup. */
    static final int SG_PER_TG = MM_THREADS / 32;
    /** Static threadgroup memory of the *_norm matmul kernels (their {@code threadgroup float buf[8]}). */
    static final int NORM_STATIC_TG_BYTES = 32;
    /**
     * Length bound to the matmul kernels' [[threadgroup(0)]] argument when the instantiation does not use it:
     * the plain (non-NORM) variants still declare the argument, and Metal requires every declared threadgroup
     * argument to have a length set before the dispatch.
     */
    static final int UNUSED_TG_BYTES = 16;
    private static final int MAX_BUFS = 16; // matmul3 binds 14 buffers

    // ---------------------------------------------------------------- tunables
    /**
     * Chunks of at least this many tokens use the tiled prefill matmuls (bit-identical results, see the prefill
     * section of kernels.metal); smaller ones the single-row kernels. A huge value disables the tiled kernels.
     */
    static final int PREFILL_MIN_TOKENS = Integer.getInteger("jllm.metal.prefillMinTokens", 8);
    /**
     * Chunks of at least MMA_MIN_TOKENS tokens use the simdgroup-matrix (8x8 MMA) prefill matmuls; the summation
     * order differs from the scalar kernels (f32 rounding level). -Djllm.metal.mmaPrefill=false turns them off.
     */
    static final boolean MMA_PREFILL = Boolean.parseBoolean(System.getProperty("jllm.metal.mmaPrefill", "true"));
    static final int MMA_MIN_TOKENS = Integer.getInteger("jllm.metal.mmaMinTokens", 32);
    /**
     * Which form of the simdgroup-matrix prefill matmuls runs (section 11 of kernels.metal): "auto" (default) = per
     * matrix shape, as {@link MetalTuner#prefillMatmuls} decided on this device (the f32 {@code *_mma} kernels for a
     * shape it did not check); "true" = the half-weight {@code *_hmma} kernels everywhere; "false" = never.
     */
    static final String HMMA = System.getProperty("jllm.metal.hmma", "auto");
    /**
     * Layers per command buffer: the dispatches of every COMMIT_LAYERS layers are submitted (without waiting)
     * while the following layers are still being encoded, so GPU execution overlaps the CPU-side encoding
     * instead of starting only after the whole pass is recorded. 0 = one command buffer per forward pass.
     */
    static final int COMMIT_LAYERS = Integer.getInteger("jllm.metal.commitLayers", 8);
    /**
     * Flash decoding (see the "flash decoding" section of kernels.metal): a decode step over at least
     * FLASH_MIN_LEN cached positions splits every head's key range across up to FLASH_SPLITS threadgroups of
     * FLASH_CHUNK keys and combines the partial softmaxes. Opt-in because the combine sums in a different order
     * from the single-pass kernel (results differ at the 1e-6 level, not bit-identical).
     */
    /** {@code -Djllm.metal.flashDecode=true|false} forces flash decoding on or off; "auto": {@link MetalTuner} decides. */
    static final String FLASH_DECODE_MODE = System.getProperty("jllm.metal.flashDecode", "auto");
    /**
     * Flash-attention prefill ({@code attention_fa}) for chunks of at least {@link #FLASH_PREFILL_MIN} tokens:
     * {@code -Djllm.metal.flashPrefill=false} turns it off; otherwise the start-up self-check ({@link MetalTuner})
     * decides on the device.
     */
    static final String FLASH_PREFILL = System.getProperty("jllm.metal.flashPrefill", "auto");
    static final int FLASH_PREFILL_MIN = Integer.getInteger("jllm.metal.flashPrefillMin", 16);
    static final int FLASH_MIN_LEN = Integer.getInteger("jllm.metal.flashMinLen", 512);
    static final int FLASH_CHUNK = Integer.getInteger("jllm.metal.flashChunk", 128); // 64-128 measured best, 256+ slower
    static final int FLASH_SPLITS = Math.min(FLASH_MAX_SPLITS, Integer.getInteger("jllm.metal.flashSplits", 32));
    /**
     * Decode replay: a single-token decode step is recorded once per shape and later steps re-run the recording with
     * their position (and token ring slots) patched in, one native call per step instead of one per dispatch (see
     * {@link #step}). {@code -Djllm.metal.replay=false} encodes every step dispatch by dispatch.
     */
    static final boolean REPLAY = !"false".equals(System.getProperty("jllm.metal.replay"));
    /** {@code -Djllm.metal.replayVerify=all}: compare every replayed step with a fresh recording (tests; slow). */
    static final boolean REPLAY_VERIFY_ALL = "all".equals(System.getProperty("jllm.metal.replayVerify"));
    /**
     * Decode-path fusions: q8 cache store in the rope kernel, the MoE norm in the router, DeepSeek MLA's attention norm
     * in its projections. Bit-identical; {@code -Djllm.metal.fuse=false} brings the separate dispatches back (A/B).
     */
    static final boolean FUSE = !"false".equals(System.getProperty("jllm.metal.fuse"));

    final Metal metal;
    private final float eps;
    private final Arena arena = Arena.ofShared();
    // dispatch argument scratch (re-used; the native side copies params and records the bindings at dispatch time)
    private final MemorySegment bufArgs;
    private final MemorySegment offArgs;
    private final Params p;
    /** Bound wherever a kernel declares a buffer the dispatch does not use (norm weights, biases, ...). */
    private final Metal.Buffer dummy;
    /** Flash decoding thresholds (see FLASH_MIN_LEN, FLASH_CHUNK); per instance so a test can cross them early. */
    int flashMinLen = FLASH_MIN_LEN;
    int flashChunk = FLASH_CHUNK;

    /** Owns {@code metal} from here on. */
    MetalKernels(Metal metal, float normEps) {
        this.metal = metal;
        this.eps = normEps;
        bufArgs = arena.allocate(ValueLayout.ADDRESS, MAX_BUFS);
        offArgs = arena.allocate(ValueLayout.JAVA_LONG, MAX_BUFS);
        p = new Params(arena.allocate(64));
        dummy = metal.newBuffer(64, true);
    }

    @Override
    public void close() {
        try {
            metal.close();
        } finally {
            arena.close();
        }
    }

    // ---------------------------------------------------------------- buffers and command buffers

    Metal.Buffer upload(float[] data) {
        Metal.Buffer b = metal.newBuffer((long) data.length * 4, false); // fully overwritten below
        MemorySegment.copy(data, 0, b.contents(), ValueLayout.JAVA_FLOAT, 0, data.length);
        return b;
    }

    /** Un-zeroed scratch: the caller guarantees every element is written before it is read. */
    Metal.Buffer scratch(long bytes) {
        return metal.newBuffer(bytes, false);
    }

    /** Zero-copy buffer over a weight tensor's bytes (page aligned, or a slice of a mapped GGUF file). */
    Metal.Buffer wrap(MemorySegment seg) {
        return metal.wrapRange(seg);
    }

    /**
     * Whether the *_norm matmul variants may be used for rows of {@code cols} floats: they keep the normalised
     * input row in dynamic threadgroup memory next to their static reduction scratch, which must fit the device's
     * threadgroup memory limit (32 KB: cols <= 8184). Otherwise a decode step runs rmsnorm + plain matmul.
     */
    boolean fusedNormFits(int cols) {
        return (long) cols * 4 + NORM_STATIC_TG_BYTES <= metal.maxThreadgroupMemory();
    }

    void begin() {
        notRecording();
        metal.begin();
    }

    void commit() {
        notRecording();
        metal.commit();
    }

    void commitAndWait() {
        notRecording();
        metal.commitAndWait();
    }

    void abort() {
        metal.abort();
    }

    // ---------------------------------------------------------------- decode replay

    /** Thrown while recording a step that cannot be replayed (as it is, or at this position: {@code retry}). */
    static final class NotReplayable extends RuntimeException {
        final boolean retry;

        NotReplayable(String why, boolean retry) {
            super(why, null, false, false);
            this.retry = retry;
        }
    }

    private void notRecording() {
        if (rec != null) { // a step that synchronizes with the CPU midway (experts on the CPU) is not one program
            throw new NotReplayable("the step submits or waits for command buffers", false);
        }
    }

    private DecodeProgram rec;                       // the program being recorded, or null
    private final int[] recVars = new int[DecodeProgram.VARS];
    // per binding slot: the step variable its offset follows (-1: none), and how
    private final int[] bindVar = new int[MAX_BUFS];
    private final long[] bindBase = new long[MAX_BUFS];
    private final long[] bindCoeff = new long[MAX_BUFS];
    /** Buffers with one row per position (KV caches): a binding at the step position's row follows the position. */
    private final java.util.Map<Metal.Buffer, Long> rowBytes = new java.util.IdentityHashMap<>();
    private final java.util.LinkedHashMap<Long, DecodeProgram> programs = new java.util.LinkedHashMap<>(16, 0.75f, true);
    private final java.util.Set<Long> unreplayable = new java.util.HashSet<>();
    private static final int MAX_PROGRAMS = 8;
    private long replayed;
    private long recorded;

    /** Declares {@code b} a per-position cache: position p's row starts at byte {@code p * bytesPerRow}. */
    void rowIndexed(Metal.Buffer b, long bytesPerRow) {
        rowBytes.put(b, bytesPerRow);
    }

    /** Drops every recorded step (a buffer they bind was released or replaced, a kernel choice changed). */
    void invalidatePrograms() {
        programs.clear();
        unreplayable.clear();
    }

    /** Decode steps run from a recording made at an earlier step. */
    long replayedSteps() {
        return replayed;
    }

    /** Decode steps recorded (each then ran from its fresh recording). */
    long recordedSteps() {
        return recorded;
    }

    /**
     * Encodes one single-token decode step: {@code body} issues its dispatches through the writers of this class. A
     * step with the same {@code key} (everything that changes which dispatches run, their grids or their buffers, except
     * the variables) recorded before is replayed instead: its position and ring-slot fields are patched to {@code pos},
     * {@code slotIn}, {@code slotOut} and the whole step goes to the native side in one {@link Metal#run} call. A new key
     * is recorded (the body runs in recording mode, nothing crosses to the native side) and run the same way. The first
     * replay of every recording is checked against a fresh recording of that step, field by field; a mismatch (a
     * per-step value the writers did not declare) disables replay for the key with a warning, and the fresh recording
     * runs instead.
     */
    void step(long key, int pos, int slotIn, int slotOut, Runnable body) {
        int[] vars = {pos, slotIn, slotOut};
        if (unreplayable.contains(key)) {
            body.run();
            return;
        }
        DecodeProgram prog = programs.get(key);
        if (prog == null) {
            DecodeProgram fresh = record(key, vars, body);
            if (fresh == null) {
                body.run();
                return;
            }
            if (programs.size() >= MAX_PROGRAMS) {
                programs.pollFirstEntry(); // least recently used: positions only grow within a conversation
            }
            programs.put(key, fresh);
            recorded++;
            metal.run(fresh.commands(), fresh.size());
            return;
        }
        prog.apply(vars);
        if (!prog.verified || REPLAY_VERIFY_ALL) {
            DecodeProgram fresh = record(key, vars, body);
            int bad = fresh == null ? 0 : prog.firstDifference(fresh);
            if (fresh == null || bad >= 0) {
                programs.remove(key);
                unreplayable.add(key);
                System.err.println("warning: Metal decode replay disabled for a step shape: "
                        + (fresh == null ? "it could not be recorded again" : "command " + bad + " ("
                        + (bad < prog.size() ? prog.kernel(bad) : "end") + ") differs from a fresh recording")
                        + "; those steps are encoded dispatch by dispatch");
                if (fresh != null) {
                    metal.run(fresh.commands(), fresh.size());
                } else {
                    body.run();
                }
                return;
            }
            prog.verified = true;
        }
        replayed++;
        metal.run(prog.commands(), prog.size());
    }

    /** Runs {@code body} in recording mode; null if the step cannot be replayed (nothing was encoded then). */
    private DecodeProgram record(long key, int[] vars, Runnable body) {
        DecodeProgram prog = new DecodeProgram();
        System.arraycopy(vars, 0, recVars, 0, vars.length);
        java.util.Arrays.fill(bindVar, -1);
        rec = prog;
        try {
            body.run();
            return prog;
        } catch (NotReplayable e) {
            if (!e.retry) {
                unreplayable.add(key);
                System.err.println("note: Metal decode replay off for a step shape: " + e.getMessage());
            }
            return null;
        } finally {
            rec = null;
        }
    }

    // ---------------------------------------------------------------- dispatch plumbing

    private void bind(int i, Metal.Buffer b, long offset) {
        bufArgs.setAtIndex(ValueLayout.ADDRESS, i, b.handle());
        offArgs.setAtIndex(ValueLayout.JAVA_LONG, i, offset + b.offset());
        if (rec != null) {
            bindVar[i] = -1;
            Long row = rowBytes.get(b);
            int pos = recVars[DecodeProgram.VAR_POS];
            if (row != null && (offset != 0 || pos == 0)) { // offset 0 at pos > 0: the whole cache (attention)
                if (pos == 0) {
                    throw new NotReplayable("position 0: a cache row and the whole cache look the same", true);
                }
                if (offset != pos * row) {
                    throw new NotReplayable("a cache bound at " + offset + ", not at the step's row " + pos, false);
                }
                bindVar[i] = DecodeProgram.VAR_POS;
                bindBase[i] = b.offset();
                bindCoeff[i] = row;
            }
        }
    }

    private void bindOrDummy(int i, Metal.Buffer b) {
        bind(i, b != null ? b : dummy, 0);
    }

    /** Records {@code kernel} with the buffers bound so far and the params written since the last reset. */
    private void dispatch(String kernel, int nbuf, int tgmem, int gx, int gy, int gz, int tx, int ty, int tz) {
        int plen = p.done();
        if (rec != null) {
            int cmd = rec.add(metal.pipeline(kernel), kernel, bufArgs, offArgs, nbuf, p.segment(), plen, tgmem, gx, gy, gz,
                    tx, ty, tz);
            for (int i = 0; i < nbuf; i++) {
                if (bindVar[i] >= 0) {
                    rec.patch(cmd, DecodeProgram.KIND_OFFSET, i, bindVar[i], bindBase[i], bindCoeff[i]);
                }
            }
            for (int j = 0; j < p.vars(); j++) {
                int f = p.varField(j);
                int var = p.varOf(j);
                rec.patch(cmd, DecodeProgram.KIND_PARAM, f, var, (long) p.intAt(f) - recVars[var], 1);
            }
            return;
        }
        metal.dispatch(metal.pipeline(kernel), bufArgs, offArgs, nbuf, p.segment(), plen, tgmem, gx, gy, gz, tx, ty, tz);
    }

    /** Kernel name {@code base_<qt><suffix>} for a tensor's quantization type (q4 / q8 / q2k / q6k, and q4g / q8g / q4k / q5k for GGUF tensors). */
    static String kernel(String base, QType type, String suffix) {
        String qt = switch (type) {
            case Q4_0 -> "q4";
            case Q8_0 -> "q8";
            case Q2_K -> "q2k";
            case Q4_0G -> "q4g";
            case Q8_0G -> "q8g";
            case Q4_K -> "q4k";
            case Q5_K -> "q5k";
            case Q6_K -> "q6k";
        };
        return base + "_" + qt + suffix;
    }

    /** Zero-copy buffers of the second views of split (GGUF gate | up) tensors, created on first use. */
    private final java.util.Map<QTensor, Metal.Buffer> upBuffers = new java.util.IdentityHashMap<>();

    private Metal.Buffer upBuffer(QTensor t) {
        return upBuffers.computeIfAbsent(t, x -> wrap(x.second.data));
    }

    /** Releases {@code buf} (a wrap of {@code t}) and the wrap of t's second view, if any: t is no longer used. */
    void release(QTensor t, Metal.Buffer buf) {
        invalidatePrograms(); // recorded steps may bind it
        metal.release(buf);
        Metal.Buffer up = upBuffers.remove(t);
        if (up != null) {
            metal.release(up);
        }
    }

    /** Offset of the f32 scale region to bind as buffer(1); Q2_K / Q6_K keep their scales inside the blocks (unused binding). */
    private static long scaleOff(QTensor t) {
        return t.type.scaleBytes == 0 ? 0 : t.scalesOffset;
    }

    /** Threadgroups along the row axis for the tiled prefill matmuls (PF_ROWS rows per simdgroup). */
    private static int pfGrid(int rows) {
        return (rows + SG_PER_TG * PF_ROWS - 1) / (SG_PER_TG * PF_ROWS);
    }

    /**
     * True if a chunk of n tokens of weights of type {@code type} runs on the simdgroup-matrix prefill kernels: every
     * format has an MMA instantiation ({@code matmul_mma_q4/q8/q2k/q6k} and the GGUF ones).
     */
    private static boolean useMma(int n, QType type) {
        return MMA_PREFILL && n >= MMA_MIN_TOKENS
                && (type == QType.Q4_0 || type == QType.Q8_0 || type == QType.Q2_K || type == QType.Q6_K || type.gguf);
    }

    /** The MMA prefill families whose two forms ({@code <family>_mma}, {@code <family>_hmma}) are chosen per shape. */
    enum MmaFamily { MATMUL, MATMUL3, SWIGLU, MOE_UP, MOE_DOWN }

    /** Per-shape choice of the half-weight MMA kernels (key: {@link #hmmaKey}), filled by {@link MetalTuner#prefillMatmuls}. */
    final java.util.Map<String, Boolean> hmmaChoice = new java.util.HashMap<>();
    /** Non-null while {@link MetalTuner} times one form against the other: overrides every choice. */
    Boolean hmmaForce;

    /** {@code rows}: output rows of the dispatch (q + k + v for matmul3, hidden units for swiglu, rows per expert for MoE). */
    static String hmmaKey(MmaFamily f, QType type, int rows, int cols) {
        return f + "|" + type + "|" + rows + "x" + cols;
    }

    /** Whether a prefill MMA dispatch of this family and shape uses the half-weight kernels. */
    boolean hmma(MmaFamily f, QType type, int rows, int cols) {
        if (hmmaForce != null) {
            return hmmaForce;
        }
        if (!"auto".equals(HMMA)) {
            return Boolean.parseBoolean(HMMA);
        }
        return hmmaChoice.getOrDefault(hmmaKey(f, type, rows, cols), false);
    }

    private String mmaKernel(String base, MmaFamily f, QType type, int rows, int cols, String suffix) {
        return kernel(base + (hmma(f, type, rows, cols) ? "_hmma" : "_mma"), type, suffix);
    }

    /** Threadgroups along the row axis for the MMA prefill matmuls (MMA_BR rows per threadgroup). */
    private static int mmaGrid(int rows, int perTg) {
        return (rows + perTg - 1) / perTg;
    }

    private static int mmaTiles(int n) {
        return (n + MMA_BT - 1) / MMA_BT;
    }

    private static int pfTiles(int n) {
        return (n + PF_TOKS - 1) / PF_TOKS;
    }

    private static int rowGrid(int rows) {
        return (rows + SG_PER_TG - 1) / SG_PER_TG;
    }

    /**
     * Threads per threadgroup of the plain single-token matmul (one simdgroup per output row, so 64, 128 or 256 is
     * 2, 4 or 8 rows per threadgroup; every row's arithmetic is the same, the results are bit-identical). Chosen per
     * chip by {@link MetalTuner#decodeThreads}: small matrices fill the GPU better with smaller threadgroups.
     */
    int decodeThreads = MetalConstants.MM_THREADS;

    private static int tokTiles(int n) {
        return (n + TOK_TILE - 1) / TOK_TILE;
    }

    /** Threadgroups of CHANNEL_THREADS covering one thread per channel. */
    private static int channelGrid(int channels) {
        return (channels + CHANNEL_THREADS - 1) / CHANNEL_THREADS;
    }

    /** Dynamic threadgroup memory for the normalised input row of the *_norm kernels (a dummy length otherwise). */
    private static int normTgMem(Metal.Buffer norm, int cols) {
        return norm != null ? cols * 4 : UNUSED_TG_BYTES;
    }

    // ---------------------------------------------------------------- matmul family

    /**
     * out[t, row] (+)= W[row] . in[t]. With {@code norm != null} (single token only) the kernel first applies
     * rmsnorm with those weights to the input row, exactly as the rmsnorm kernel would; {@code softcap > 0}
     * applies tanh(out / softcap) * softcap in the epilogue (not with acc).
     */
    void matmul(QTensor t, Metal.Buffer wbuf, Metal.Buffer in, long inOff, Metal.Buffer out, long outOff,
                int n, boolean acc, Metal.Buffer norm, float softcap) {
        matmul(t, wbuf, in, inOff, out, outOff, n, acc, norm, softcap, null);
    }

    /** As above with a per-row {@code bias} (f32, {@code t.rows}) added to every product before it is stored or accumulated. */
    void matmul(QTensor t, Metal.Buffer wbuf, Metal.Buffer in, long inOff, Metal.Buffer out, long outOff,
                int n, boolean acc, Metal.Buffer norm, float softcap, Metal.Buffer bias) {
        if (norm != null && n != 1) {
            throw new IllegalArgumentException("fused rmsnorm needs a single token");
        }
        if (t.isSplit()) {
            throw new IllegalStateException("a split gate|up tensor is only multiplied by swiglu / moe_up");
        }
        bind(0, wbuf, 0);
        bind(1, wbuf, scaleOff(t));
        bind(2, in, inOff);
        bind(3, out, outOff);
        bindOrDummy(4, norm);
        bindOrDummy(5, bias);
        p.reset(KernelParams.MATMUL).i(t.rows).i(t.cols).i(n).i(acc).f(eps).f(softcap).i(bias != null);
        if (norm == null && useMma(n, t.type)) {
            dispatch(mmaKernel("matmul", MmaFamily.MATMUL, t.type, t.rows, t.cols, ""), 6, 0, mmaGrid(t.rows, MMA_BR),
                    mmaTiles(n), 1, MM_THREADS, 1, 1);
        } else if (norm == null && n >= PREFILL_MIN_TOKENS) {
            dispatch(kernel("matmul_pf", t.type, ""), 6, 0, pfGrid(t.rows), pfTiles(n), 1, MM_THREADS, 1, 1);
        } else if (norm == null && n == 1) {
            int sgs = decodeThreads / 32;
            dispatch(kernel("matmul", t.type, ""), 6, 0, (t.rows + sgs - 1) / sgs, 1, 1, decodeThreads, 1, 1);
        } else {
            dispatch(kernel("matmul", t.type, norm != null ? "_norm" : ""), 6, normTgMem(norm, t.cols),
                    rowGrid(t.rows), tokTiles(n), 1, MM_THREADS, 1, 1);
        }
    }

    /**
     * q, k and v projections of {@code in} in one dispatch: q to {@code qOut}, k to {@code kOut} at kOff, v to
     * {@code vOut} at vOff (as half when {@code vHalf}: the f16 cache rows). The three weights share the quant
     * type and width; {@code norm} folds the preceding rmsnorm (single token); the biases are optional.
     */
    void matmul3(QTensor wq, Metal.Buffer bwq, QTensor wk, Metal.Buffer bwk, QTensor wv, Metal.Buffer bwv,
                 Metal.Buffer in, Metal.Buffer norm, Metal.Buffer qOut, Metal.Buffer kOut, long kOff,
                 Metal.Buffer vOut, long vOff, boolean vHalf, Metal.Buffer bq, Metal.Buffer bk, Metal.Buffer bv, int n) {
        if (wq.cols != wk.cols || wk.cols != wv.cols) {
            throw new IllegalStateException("q/k/v projections differ in width"); // the loader guarantees they match
        }
        if (norm != null && n != 1) {
            throw new IllegalArgumentException("fused rmsnorm needs a single token");
        }
        if (wq.type != wk.type || wk.type != wv.type) {
            // A GGUF file may quantize the three projections differently (Q4_K_M: v in Q6_K in half the layers),
            // and a kernel instantiation has one block format: one single-matrix dispatch per projection, each
            // with the kernel of its own type, writing where the fused dispatch would.
            matmul3Part(wq, bwq, in, norm, qOut, 0, null, 0, false, bq, 0, n);
            matmul3Part(wk, bwk, in, norm, kOut, kOff, null, 0, false, bk, 1, n);
            matmul3Part(wv, bwv, in, norm, null, 0, vOut, vOff, vHalf, bv, 2, n);
            return;
        }
        bind(0, bwq, 0);
        bind(1, bwq, scaleOff(wq));
        bind(2, bwk, 0);
        bind(3, bwk, scaleOff(wk));
        bind(4, bwv, 0);
        bind(5, bwv, scaleOff(wv));
        bind(6, in, 0);
        bind(7, qOut, 0);
        bind(8, kOut, kOff);
        bind(9, vOut, vOff);
        bindOrDummy(10, norm);
        bindOrDummy(11, bq);
        bindOrDummy(12, bk);
        bindOrDummy(13, bv);
        p.reset(KernelParams.MATMUL3).i(wq.rows).i(wk.rows).i(wv.rows).i(wq.cols).i(n).f(eps).i(bq != null);
        int rows = wq.rows + wk.rows + wv.rows;
        String vh = vHalf ? "_h" : "";
        if (norm == null && useMma(n, wq.type)) {
            dispatch(mmaKernel("matmul3", MmaFamily.MATMUL3, wq.type, rows, wq.cols, vh), 14, 0, mmaGrid(rows, MMA_BR),
                    mmaTiles(n), 1, MM_THREADS, 1, 1);
        } else if (norm == null && n >= PREFILL_MIN_TOKENS) {
            dispatch(kernel("matmul3_pf", wq.type, vh), 14, 0, pfGrid(rows), pfTiles(n), 1, MM_THREADS, 1, 1);
        } else {
            dispatch(kernel("matmul3", wq.type, (norm != null ? "_norm" : "") + vh), 14, normTgMem(norm, wq.cols),
                    rowGrid(rows), tokTiles(n), 1, MM_THREADS, 1, 1);
        }
    }

    /**
     * One projection of {@link #matmul3} as a dispatch of its own: {@code which} 0 / 1 = q / k (float output at
     * {@code out}, {@code outOff}), 2 = v (rows past the first two ranges, written to {@code vOut} at {@code vOff} as half when
     * {@code vHalf}). The unused weight bindings point at the same buffer; their row ranges are empty.
     */
    private void matmul3Part(QTensor w, Metal.Buffer bw, Metal.Buffer in, Metal.Buffer norm, Metal.Buffer out, long outOff,
                             Metal.Buffer vOut, long vOff, boolean vHalf, Metal.Buffer bias, int which, int n) {
        bind(0, bw, 0);
        bind(1, bw, scaleOff(w));
        bind(2, bw, 0);
        bind(3, bw, scaleOff(w));
        bind(4, bw, 0);
        bind(5, bw, scaleOff(w));
        bind(6, in, 0);
        if (which == 2) {
            bind(7, in, 0);   // out0 / out1 are not written: their row ranges are empty
            bind(8, in, 0);
            bind(9, vOut, vOff);
        } else {
            bind(7, out, outOff);
            bind(8, out, outOff);
            bind(9, out, outOff);
        }
        bindOrDummy(10, norm);
        bindOrDummy(11, which == 0 ? bias : null);
        bindOrDummy(12, which == 1 ? bias : null);
        bindOrDummy(13, which == 2 ? bias : null);
        p.reset(KernelParams.MATMUL3).i(which == 0 ? w.rows : 0).i(which == 1 ? w.rows : 0).i(which == 2 ? w.rows : 0)
                .i(w.cols).i(n).f(eps).i(bias != null);
        String vh = which == 2 && vHalf ? "_h" : "";
        if (norm == null && useMma(n, w.type)) {
            dispatch(mmaKernel("matmul3", MmaFamily.MATMUL3, w.type, w.rows, w.cols, vh), 14, 0, mmaGrid(w.rows, MMA_BR),
                    mmaTiles(n), 1, MM_THREADS, 1, 1);
        } else if (norm == null && n >= PREFILL_MIN_TOKENS) {
            dispatch(kernel("matmul3_pf", w.type, vh), 14, 0, pfGrid(w.rows), pfTiles(n), 1, MM_THREADS, 1, 1);
        } else {
            dispatch(kernel("matmul3", w.type, (norm != null ? "_norm" : "") + vh), 14, normTgMem(norm, w.cols),
                    rowGrid(w.rows), tokTiles(n), 1, MM_THREADS, 1, 1);
        }
    }

    /**
     * out[t] = act(Wgate . in[t]) * (Wup . in[t]) from a stacked [2 * hidden, cols] gate|up tensor; act is SiLU,
     * or Gemma's gelu_tanh with {@code gelu}. {@code norm} folds the preceding rmsnorm (single token).
     */
    void swiglu(QTensor t, Metal.Buffer wbuf, Metal.Buffer in, Metal.Buffer norm, Metal.Buffer out, boolean gelu, int n) {
        if (norm != null && n != 1) {
            throw new IllegalArgumentException("fused rmsnorm needs a single token");
        }
        int hidden = t.rows / 2;
        bind(0, wbuf, 0);
        if (t.isSplit()) {
            bind(1, upBuffer(t), 0);   // the up rows' view of a GGUF file (the kernels read it in place of the scales)
        } else {
            bind(1, wbuf, scaleOff(t));
        }
        bind(2, in, 0);
        bind(3, out, 0);
        bindOrDummy(4, norm);
        p.reset(KernelParams.SWIGLU).i(hidden).i(t.cols).i(n).f(eps).i(gelu);
        if (norm == null && useMma(n, t.type)) {
            dispatch(mmaKernel("matmul_swiglu", MmaFamily.SWIGLU, t.type, hidden, t.cols, ""), 5, 0,
                    mmaGrid(hidden, MMA_BR / 2), mmaTiles(n), 1, MM_THREADS, 1, 1);
        } else if (norm == null && n >= PREFILL_MIN_TOKENS) {
            int gx = (hidden + SG_PER_TG * PF_UNITS - 1) / (SG_PER_TG * PF_UNITS);
            dispatch(kernel("matmul_swiglu_pf", t.type, ""), 5, 0, gx, pfTiles(n), 1, MM_THREADS, 1, 1);
        } else {
            dispatch(kernel("matmul_swiglu", t.type, norm != null ? "_norm" : ""), 5, normTgMem(norm, t.cols),
                    rowGrid(hidden), tokTiles(n), 1, MM_THREADS, 1, 1);
        }
    }

    // ---------------------------------------------------------------- norms and rope

    /** out[t] = rmsnorm(in[t]) * weight for n rows of the given width (in read from the byte offset inOff). */
    void rmsnorm(Metal.Buffer in, long inOff, Metal.Buffer weight, Metal.Buffer out, int n, int width) {
        bind(0, in, inOff);
        bind(1, weight, 0);
        bind(2, out, 0);
        p.reset(KernelParams.NORM).i(width).f(eps);
        dispatch("rmsnorm", 3, 0, n, 1, 1, ROW_THREADS, 1, 1);
    }

    /** out[t] += rmsnorm(y[t]) * weight for n rows of the given width (Glm4's post-norms). */
    void rmsnormAdd(Metal.Buffer y, Metal.Buffer weight, Metal.Buffer out, int n, int width) {
        bind(0, y, 0);
        bind(1, weight, 0);
        bind(2, out, 0);
        p.reset(KernelParams.NORM).i(width).f(eps);
        dispatch("rmsnorm_add", 3, 0, n, 1, 1, ROW_THREADS, 1, 1);
    }

    /**
     * Rotary embedding ({@code rope}: rotate-half pairs, {@code rope_il}: adjacent pairs) of the {@code 2 * rotHalf}
     * dims from {@code rotOff} of every head: nHeads q heads in place, nKHeads k heads read from kSrc and written
     * to kOut (as half when {@code kHalf}: the f16 cache rows); qw/kw non-null = qk-norm first.
     */
    void rope(boolean adjacent, boolean kHalf, Metal.Buffer q, Metal.Buffer kSrc, long kSrcOff, Metal.Buffer kOut,
              long kOutOff, Metal.Buffer qw, Metal.Buffer kw, Metal.Buffer cos, Metal.Buffer sin, int headDim,
              int qStride, int kvStride, int nHeads, int nKHeads, int pos, int rotOff, int rotHalf, int n) {
        bind(0, q, 0);
        bind(1, kSrc, kSrcOff);
        bindOrDummy(2, qw);
        bindOrDummy(3, kw);
        bind(4, cos, 0);
        bind(5, sin, 0);
        bind(6, kOut, kOutOff);
        p.reset(KernelParams.ROPE).i(headDim).i(qStride).i(kvStride).i(nHeads).iv(DecodeProgram.VAR_POS, pos).f(eps).i(qw != null)
                .i(rotOff).i(rotHalf);
        dispatch((adjacent ? "rope_il" : "rope") + (kHalf && nKHeads > 0 ? "_h" : ""), 7, 0,
                nHeads + nKHeads, n, 1, rotHalf, 1, 1);
    }

    /**
     * The q8 cache's rope: {@link #rope} for the q heads and the k heads of the f32 staging rows {@code kStage}, and
     * {@link #kvStoreQ8} of the rotated k and of {@code vStage} into the cache rows at byte offset {@code cacheOff} of
     * kc / vc, in one dispatch ({@code rope_store_q8}; the same cache bytes).
     */
    void ropeStoreQ8(boolean adjacent, Metal.Buffer q, Metal.Buffer kStage, Metal.Buffer vStage, Metal.Buffer kc,
                     Metal.Buffer vc, long cacheOff, Metal.Buffer qw, Metal.Buffer kw, Metal.Buffer cos, Metal.Buffer sin,
                     int headDim, int qStride, int nHeads, int nKvHeads, int pos, int rotOff, int rotHalf, int n) {
        int kvDim = headDim * nKvHeads;
        bind(0, q, 0);
        bind(1, kStage, 0);
        bindOrDummy(2, qw);
        bindOrDummy(3, kw);
        bind(4, cos, 0);
        bind(5, sin, 0);
        bind(6, kc, cacheOff);
        bind(7, vStage, 0);
        bind(8, vc, cacheOff);
        p.reset(KernelParams.ROPE_Q8).i(headDim).i(qStride).i(kvDim).i(nHeads).iv(DecodeProgram.VAR_POS, pos).f(eps)
                .i(qw != null).i(rotOff).i(rotHalf).i(nKvHeads).i((int) KvDtype.Q8.rowBytes(kvDim, nKvHeads));
        dispatch(adjacent ? "rope_il_store_q8" : "rope_store_q8", 9, 0, nHeads + 2 * nKvHeads, n, 1, rotHalf, 1, 1);
    }

    // ---------------------------------------------------------------- attention

    /**
     * {@code attention_t} ({@code kernel} names the instantiation) for n tokens at pos, one dispatch per tile of
     * ATT_TILE tokens (the scores scratch holds a row of ctx floats per tile token and head). {@code groups}
     * query heads share a threadgroup; kc/vc are the full caches; out receives rows of oStride floats.
     */
    void attention(String kernel, int groups, Metal.Buffer q, Metal.Buffer kc, Metal.Buffer vc, Metal.Buffer out,
                   Metal.Buffer scores, int nHeads, int nKvHeads, int qkDim, int vDim, int pos, int ctx, float scale,
                   int qStride, int kStride, int vStride, int oStride, int window, float softcap, int n) {
        bind(0, q, 0);
        bind(1, kc, 0);
        bind(2, vc, 0);
        bind(3, out, 0);
        bind(4, scores, 0);
        for (int t0 = 0; t0 < n; t0 += ATT_TILE) {
            p.reset(KernelParams.ATTN).i(nHeads).i(nKvHeads).i(qkDim).i(vDim).iv(DecodeProgram.VAR_POS, pos).i(ctx)
                    .f(scale)
                    .i(qStride).i(kStride).i(vStride).i(oStride).i(window).i(t0).f(softcap);
            dispatch(kernel, 5, 0, nHeads / groups, Math.min(ATT_TILE, n - t0), 1, ATT_THREADS, 1, 1);
        }
    }

    /** x row 0 = scale * embedding row tokens[slot] (the token a previous step's argmax chose). */
    void embedToken(QTensor embed, Metal.Buffer embedBuf, Metal.Buffer tokens, int slot, Metal.Buffer x, int dim,
                    float scale) {
        bind(0, embedBuf, 0);
        bind(1, embedBuf, scaleOff(embed));
        bind(2, tokens, 0);
        bind(3, x, 0);
        p.reset(KernelParams.EMBED).i(dim).iv(DecodeProgram.VAR_SLOT_IN, slot).f(scale);
        dispatch(kernel("embed_token", embed.type, ""), 4, 0, 1, 1, 1, ROW_THREADS, 1, 1);
    }

    /** tokens[slot] = argmax of v[0 .. n) (lowest index among equal maxima). */
    void argmax(Metal.Buffer v, int n, Metal.Buffer tokens, int slot) {
        bind(0, v, 0);
        bind(1, tokens, 0);
        p.reset(KernelParams.ARGMAX).i(n).iv(DecodeProgram.VAR_SLOT_OUT, slot);
        dispatch("argmax", 2, 0, 1, 1, 1, ROW_THREADS, 1, 1);
    }

    /** Suffix of the attention / cache-store instantiations for a cache type: none (f32), {@code _h}, {@code _q8}. */
    static String kvSuffix(KvDtype kv) {
        return switch (kv) {
            case F32 -> "";
            case F16 -> "_h";
            case Q8 -> "_q8";
        };
    }

    /**
     * {@code kv_store_q8}: quantizes the n f32 staging rows of k and v ({@code [n, kvDim]}) into the q8 cache rows at
     * byte offset {@code off} of kc / vc (row stride {@code kvDim + 4 * kvHeads}: int8 values, then a scale per head).
     */
    void kvStoreQ8(Metal.Buffer kStage, Metal.Buffer vStage, Metal.Buffer kc, Metal.Buffer vc, long off, int headDim,
                   int kvHeads, int n) {
        int kvDim = headDim * kvHeads;
        bind(0, kStage, 0);
        bind(1, vStage, 0);
        bind(2, kc, off);
        bind(3, vc, off);
        p.reset(KernelParams.KV_STORE).i(headDim).i(kvDim).i(kvHeads).i((int) KvDtype.Q8.rowBytes(kvDim, kvHeads));
        dispatch("kv_store_q8", 4, 0, 2 * kvHeads, n, 1, 32, 1, 1);
    }

    /** Whether {@code attention_fa} exists for this head width. */
    static boolean flashPrefillFits(int headDim) {
        return headDim == 64 || headDim == 128 || headDim == 256;
    }

    /**
     * Flash-attention prefill of the n tokens at pos ({@code attention_fa}): one dispatch over (heads, n / FA_ROWS
     * tiles), no scores scratch. kc / vc are the full caches with row stride kStride (elements; q8: bytes).
     */
    /**
     * The flash prefill kernel {@link #flashPrefill} dispatches: {@code attention_fa}, or its second form
     * {@code attention_fb} (K and V staged together, half staging for the f16 / q8 caches) when {@link MetalTuner#decide}
     * found it correct and faster on this device ({@code -Djllm.metal.flashPrefillKernel=fa|fb} forces one).
     */
    String flashPrefillKernel = "attention_fa";

    void flashPrefill(KvDtype kv, int headDim, Metal.Buffer q, Metal.Buffer kc, Metal.Buffer vc, Metal.Buffer out,
                      int nHeads, int nKvHeads, int pos, int n, float scale, int qStride, int kStride, int oStride,
                      int window, float softcap) {
        bind(0, q, 0);
        bind(1, kc, 0);
        bind(2, vc, 0);
        bind(3, out, 0);
        p.reset(KernelParams.FLASH).i(nHeads).i(nKvHeads).i(pos).i(n).f(scale).i(qStride).i(kStride).i(oStride)
                .i(window).f(softcap);
        dispatch(flashPrefillKernel + "_d" + headDim + kvSuffix(kv), 4, 0, nHeads,
                (n + MetalConstants.FA_ROWS - 1) / MetalConstants.FA_ROWS, 1, MetalConstants.FA_THREADS, 1, 1);
    }

    /** Threadgroups per head of flash decoding for the token at pos: one per flashChunk keys it sees, at most FLASH_SPLITS. */
    int flashSplits(int pos, int window) {
        int len = pos + 1;
        int first = window > 0 ? Math.max(0, len - window) : 0;
        int keys = len - first;
        return Math.min(FLASH_SPLITS, (keys + flashChunk - 1) / flashChunk);
    }

    /**
     * Flash decoding for the single token at pos:{@code attention_split} over (heads, nSplit) into the partials
     * buffer {@code part} (heads * FLASH_SPLITS * (headDim + 2) floats), then {@code attention_combine} into out.
     */
    void flashAttention(KvDtype kv, Metal.Buffer q, Metal.Buffer kc, Metal.Buffer vc, Metal.Buffer part,
                        Metal.Buffer scores, Metal.Buffer out, int nHeads, int nKvHeads, int headDim, int pos, int ctx,
                        float scale, int qStride, int kvStride, int window, float softcap) {
        int nSplit = flashSplits(pos, window);
        bind(0, q, 0);
        bind(1, kc, 0);
        bind(2, vc, 0);
        bind(3, part, 0);
        bind(4, scores, 0);
        // chunk 0: the kernel derives the keys per split from the position, ceil(keys / nSplit), so a recorded decode
        // step stays valid while nSplit does (MetalBackend's step key)
        p.reset(KernelParams.ATTN_SPLIT).i(nHeads).i(nKvHeads).i(headDim).iv(DecodeProgram.VAR_POS, pos).i(ctx).f(scale)
                .i(qStride).i(kvStride).i(window).f(softcap).i(nSplit).i(0);
        dispatch("attention_split" + kvSuffix(kv), 5, 0, nHeads, nSplit, 1, ATT_THREADS, 1, 1);
        bind(0, part, 0);
        bind(1, out, 0);
        p.reset(KernelParams.ATTN_COMBINE).i(nHeads).i(headDim).i(nSplit);
        dispatch("attention_combine", 2, 0, nHeads, 1, 1, ROW_THREADS, 1, 1);
    }

    // ---------------------------------------------------------------- mixture of experts

    /**
     * Softmax top-k routing of n rows of x: assign[t * topK + k] and the (optionally renormalised) weights. With
     * {@code counts} (experts on the CPU): also gassign[slot] = map[expert] and counts[expert] += 1 per pick.
     */
    void router(Metal.Buffer x, Metal.Buffer w, Metal.Buffer assign, Metal.Buffer rw, int nExperts, int dim, int topK,
                boolean normalize, Metal.Buffer map, Metal.Buffer gassign, Metal.Buffer counts, int n) {
        bind(0, x, 0);
        bind(1, w, 0);
        bind(2, assign, 0);
        bind(3, rw, 0);
        bindOrDummy(4, map);
        bindOrDummy(5, gassign);
        bindOrDummy(6, counts);
        p.reset(KernelParams.ROUTER).i(nExperts).i(dim).i(topK).i(normalize).i(counts != null);
        dispatch("router", 7, 0, n, 1, 1, ROW_THREADS, 1, 1);
    }

    /** DeepSeek routing (see router_ds in kernels.metal and DeepSeekRouter.java); {@code bias} only for noaux_tc. */
    void routerDs(Metal.Buffer x, Metal.Buffer w, Metal.Buffer assign, Metal.Buffer rw, Metal.Buffer bias, int nExperts,
                  int dim, int topK, boolean normalize, int nGroup, int topkGroup, boolean sigmoid, int method,
                  boolean scaleAlways, float routedScale, Metal.Buffer map, Metal.Buffer gassign, Metal.Buffer counts,
                  int n) {
        bind(0, x, 0);
        bind(1, w, 0);
        bind(2, assign, 0);
        bind(3, rw, 0);
        bindOrDummy(4, bias);
        bindOrDummy(5, map);
        bindOrDummy(6, gassign);
        bindOrDummy(7, counts);
        p.reset(KernelParams.ROUTER_DS).i(nExperts).i(dim).i(topK).i(normalize).i(nGroup).i(topkGroup).i(sigmoid)
                .i(method).i(scaleAlways).f(routedScale).i(counts != null);
        dispatch("router_ds", 8, 0, n, 1, 1, ROW_THREADS, 1, 1);
    }

    /**
     * {@link #rmsnorm} of the n rows of x with {@code normW} into xn, and {@link #router} on xn, in one dispatch
     * ({@code router_norm}, bit-identical): the norm before a MoE layer's experts.
     */
    void routerNorm(Metal.Buffer x, Metal.Buffer normW, Metal.Buffer xn, Metal.Buffer w, Metal.Buffer assign,
                    Metal.Buffer rw, int nExperts, int dim, int topK, boolean normalize, Metal.Buffer map,
                    Metal.Buffer gassign, Metal.Buffer counts, int n) {
        bind(0, x, 0);
        bind(1, w, 0);
        bind(2, assign, 0);
        bind(3, rw, 0);
        bindOrDummy(4, map);
        bindOrDummy(5, gassign);
        bindOrDummy(6, counts);
        bind(7, normW, 0);
        bind(8, xn, 0);
        p.reset(KernelParams.ROUTER_NORM).i(nExperts).i(dim).i(topK).i(normalize).i(counts != null).f(eps);
        dispatch("router_norm", 9, 0, n, 1, 1, ROW_THREADS, 1, 1);
    }

    /** {@link #rmsnorm} into xn and {@link #routerDs} on it in one dispatch ({@code router_ds_norm}, bit-identical). */
    void routerDsNorm(Metal.Buffer x, Metal.Buffer normW, Metal.Buffer xn, Metal.Buffer w, Metal.Buffer assign,
                      Metal.Buffer rw, Metal.Buffer bias, int nExperts, int dim, int topK, boolean normalize, int nGroup,
                      int topkGroup, boolean sigmoid, int method, boolean scaleAlways, float routedScale,
                      Metal.Buffer map, Metal.Buffer gassign, Metal.Buffer counts, int n) {
        bind(0, x, 0);
        bind(1, w, 0);
        bind(2, assign, 0);
        bind(3, rw, 0);
        bindOrDummy(4, bias);
        bindOrDummy(5, map);
        bindOrDummy(6, gassign);
        bindOrDummy(7, counts);
        bind(8, normW, 0);
        bind(9, xn, 0);
        p.reset(KernelParams.ROUTER_DS_NORM).i(nExperts).i(dim).i(topK).i(normalize).i(nGroup).i(topkGroup).i(sigmoid)
                .i(method).i(scaleAlways).f(routedScale).i(counts != null).f(eps);
        dispatch("router_ds_norm", 10, 0, n, 1, 1, ROW_THREADS, 1, 1);
    }

    /** Per-slot gate/up projections with SwiGLU: out[slot, :] for the n * topK slots of assign (moe_up). */
    void moeUp(QTensor gateUp, Metal.Buffer wbuf, Metal.Buffer x, Metal.Buffer assign, Metal.Buffer out, int moeDim,
               int topK, int n) {
        bind(0, wbuf, 0);
        if (gateUp.isSplit()) {
            bind(1, upBuffer(gateUp), 0);   // the up experts' view of a GGUF file
        } else {
            bind(1, wbuf, scaleOff(gateUp));
        }
        bind(2, x, 0);
        bind(3, assign, 0);
        bind(4, out, 0);
        p.reset(KernelParams.MOE).i(2 * moeDim).i(gateUp.cols).i(topK); // rows per expert: gate | up
        dispatch(kernel("moe_up", gateUp.type, ""), 5, 0, rowGrid(moeDim), n * topK, 1, MM_THREADS, 1, 1);
    }

    /**
     * Expert-grouped MoE for chunks of at least this many tokens (moe_sort + moe_up_mma + moe_down_mma + moe_combine);
     * {@code -Djllm.metal.moeMma=false} keeps the per-slot kernels.
     */
    static final boolean MOE_MMA = !"false".equals(System.getProperty("jllm.metal.moeMma"));
    static final int MOE_MMA_MIN = Integer.getInteger("jllm.metal.moeMmaMin", 16);

    /** Tiles of up to MMA_BT slots of one expert that a chunk can need: whole tiles plus one partial per expert. */
    static int moeMaxTiles(int slots, int experts) {
        return slots / MetalConstants.MMA_BT + Math.min(experts, slots) + 1;
    }

    /**
     * The grouped MoE block of n tokens: slots sorted by expert into tiles, gate/up and down projections per tile on the
     * simdgroup-matrix core, then x[t] += the weighted rows of the token's topK slots.
     */
    void moeGrouped(QTensor gateUp, Metal.Buffer bGateUp, QTensor down, Metal.Buffer bDown, Metal.Buffer xb,
                    Metal.Buffer assign, Metal.Buffer rw, Metal.Buffer order, Metal.Buffer tiles, Metal.Buffer hidden,
                    Metal.Buffer ys, Metal.Buffer x, int experts, int moeDim, int dim, int topK, int n) {
        int slots = n * topK;
        int maxTiles = moeMaxTiles(slots, experts);
        bind(0, assign, 0);
        bind(1, order, 0);
        bind(2, tiles, 0);
        p.reset(KernelParams.MOE_SORT).i(slots).i(experts).i(maxTiles);
        dispatch("moe_sort", 3, 0, 1, 1, 1, ROW_THREADS, 1, 1);
        moeUpTiles(gateUp, bGateUp, xb, order, tiles, hidden, moeDim, topK, maxTiles);
        moeDownTiles(down, bDown, hidden, order, tiles, rw, ys, dim, topK, maxTiles);

        bind(0, ys, 0);
        bind(1, assign, 0);
        bind(2, x, 0);
        p.reset(KernelParams.MOE).i(dim).i(moeDim).i(topK);
        dispatch("moe_combine", 3, 0, (dim + ROW_THREADS - 1) / ROW_THREADS, n, 1, ROW_THREADS, 1, 1);
    }

    /**
     * hidden[s] = silu(Wgate_e . x[s / topK]) * (Wup_e . x[s / topK]) for the slots of the expert tiles in {@code tiles}
     * (moe_sort's list, at most {@code maxTiles}): moe_up_mma or moe_up_hmma, as chosen for this shape.
     */
    void moeUpTiles(QTensor gateUp, Metal.Buffer bGateUp, Metal.Buffer xb, Metal.Buffer order, Metal.Buffer tiles,
                    Metal.Buffer hidden, int moeDim, int topK, int maxTiles) {
        bind(0, bGateUp, 0);
        if (gateUp.isSplit()) {
            bind(1, upBuffer(gateUp), 0);
        } else {
            bind(1, bGateUp, scaleOff(gateUp));
        }
        bind(2, xb, 0);
        bind(3, order, 0);
        bind(4, tiles, 0);
        bind(5, hidden, 0);
        p.reset(KernelParams.MOE).i(2 * moeDim).i(gateUp.cols).i(topK);
        dispatch(mmaKernel("moe_up", MmaFamily.MOE_UP, gateUp.type, moeDim, gateUp.cols, ""), 6, 0,
                mmaGrid(moeDim, MetalConstants.MMA_BR / 2), maxTiles, 1, MM_THREADS, 1, 1);
    }

    /** ys[s] = rw[s] * W_e . hidden[s] for the slots of the expert tiles: moe_down_mma or moe_down_hmma. */
    void moeDownTiles(QTensor down, Metal.Buffer bDown, Metal.Buffer hidden, Metal.Buffer order, Metal.Buffer tiles,
                      Metal.Buffer rw, Metal.Buffer ys, int dim, int topK, int maxTiles) {
        bind(0, bDown, 0);
        bind(1, bDown, scaleOff(down));
        bind(2, hidden, 0);
        bind(3, order, 0);
        bind(4, tiles, 0);
        bind(5, rw, 0);
        bind(6, ys, 0);
        p.reset(KernelParams.MOE).i(dim).i(down.cols).i(topK);
        dispatch(mmaKernel("moe_down", MmaFamily.MOE_DOWN, down.type, dim, down.cols, ""), 7, 0,
                mmaGrid(dim, MetalConstants.MMA_BR), maxTiles, 1, MM_THREADS, 1, 1);
    }

    /** x[t] += sum_k rw[t, k] * W_assign[t, k] . h[t * topK + k] (moe_down). */
    void moeDown(QTensor down, Metal.Buffer wbuf, Metal.Buffer h, Metal.Buffer assign, Metal.Buffer rw, Metal.Buffer x,
                 int dim, int topK, int n) {
        bind(0, wbuf, 0);
        bind(1, wbuf, scaleOff(down));
        bind(2, h, 0);
        bind(3, assign, 0);
        bind(4, rw, 0);
        bind(5, x, 0);
        p.reset(KernelParams.MOE).i(dim).i(down.cols).i(topK);
        dispatch(kernel("moe_down", down.type, ""), 6, 0, rowGrid(dim), n, 1, MM_THREADS, 1, 1);
    }

    // ---------------------------------------------------------------- DeepSeek MLA

    /**
     * Stores the latent cache rows [c | k_pe] of n tokens from lat / kpe at byte offset off of the cache (f32, half,
     * or q8 with one scale per row).
     */
    void mlaLatStore(KvDtype kv, Metal.Buffer lat, Metal.Buffer kpe, Metal.Buffer cache, long off, Attention.Mla m,
                     int n) {
        bind(0, lat, 0);
        bind(1, kpe, 0);
        bind(2, cache, off);
        mlaParams(m);
        dispatch("mla_lat_store" + kvSuffix(kv), 3, 0, n, 1, 1, CHANNEL_THREADS, 1, 1);
    }

    /** qa[t, h] = [W_kb_k[h]^T q_nope | q_pe] for the n rows of q (mla_absorb_q); wk is [heads][nope][kvLora] f32. */
    void mlaAbsorbQ(Metal.Buffer q, Metal.Buffer wk, Metal.Buffer qa, Attention.Mla m, int n) {
        bind(0, q, 0);
        bind(1, wk, 0);
        bind(2, qa, 0);
        mlaParams(m);
        dispatch("mla_absorb_q", 3, 0, m.heads(), n, 1, ROW_THREADS, 1, 1);
    }

    /** att[t, h] = W_kb_v[h] o[t, h] for the n rows of o (mla_apply_v); wv is [heads][kvLora][vDim] f32. */
    void mlaApplyV(Metal.Buffer o, Metal.Buffer wv, Metal.Buffer att, Attention.Mla m, int n) {
        bind(0, o, 0);
        bind(1, wv, 0);
        bind(2, att, 0);
        mlaParams(m);
        dispatch("mla_apply_v", 3, 0, m.heads(), n, 1, ROW_THREADS, 1, 1);
    }

    private void mlaParams(Attention.Mla m) {
        p.reset(KernelParams.MLA).i(m.nope()).i(m.rope()).i(m.kvLoraRank()).i(m.vDim()).i(m.heads());
    }

    // ---------------------------------------------------------------- Mamba

    /** Causal conv over the chunk with the carried state (mamba_conv). */
    void mambaConv(Metal.Buffer xz, Metal.Buffer w, Metal.Buffer bias, Metal.Buffer state, Metal.Buffer xc,
                   int dInner, int dConv, int n) {
        bind(0, xz, 0);
        bind(1, w, 0);
        bind(2, bias, 0);
        bind(3, state, 0);
        bind(4, xc, 0);
        p.reset(KernelParams.MAMBA_CONV).i(dInner).i(dConv).i(n);
        dispatch("mamba_conv", 5, 0, channelGrid(dInner), 1, 1, CHANNEL_THREADS, 1, 1);
    }

    /** Selective scan over the chunk with the carried SSM state (mamba_scan). */
    void mambaScan(Metal.Buffer xc, Metal.Buffer xz, Metal.Buffer dbc, Metal.Buffer dtw, Metal.Buffer dtb, Metal.Buffer a,
                   Metal.Buffer d, Metal.Buffer state, Metal.Buffer y, int dInner, int dState, int dtRank, int n) {
        bind(0, xc, 0);
        bind(1, xz, 0);
        bind(2, dbc, 0);
        bind(3, dtw, 0);
        bind(4, dtb, 0);
        bind(5, a, 0);
        bind(6, d, 0);
        bind(7, state, 0);
        bind(8, y, 0);
        p.reset(KernelParams.MAMBA_SCAN).i(dInner).i(dState).i(dtRank).i(n);
        dispatch("mamba_scan", 9, 0, channelGrid(dInner), 1, 1, CHANNEL_THREADS, 1, 1);
    }

    // ---------------------------------------------------------------- Qwen3-Next

    /** Gated DeltaNet conv state width per gdn_rule threadgroup (GDN_COLS) and the widest key head (32 * GDN_MAX_PER). */
    static final int GDN_COLS = 4;
    static final int GDN_MAX_KEY_DIM = 32 * 8;
    static final int GDN_MAX_CONV = 8;

    /** y[t, c] = silu(causal conv of channel c) over n tokens with the carried conv state (gdn_conv). */
    void gdnConv(Metal.Buffer x, Metal.Buffer w, Metal.Buffer state, Metal.Buffer y, int channels, int kernel,
                 int inStride, int outStride, int n) {
        gdnConv(x, w, state, y, channels, kernel, inStride, outStride, n, null, -1);
    }

    /** As above; with {@code snapTok >= 0} the state after that token also goes to {@code snap} (same layout as {@code state}). */
    void gdnConv(Metal.Buffer x, Metal.Buffer w, Metal.Buffer state, Metal.Buffer y, int channels, int kernel,
                 int inStride, int outStride, int n, Metal.Buffer snap, int snapTok) {
        bind(0, x, 0);
        bind(1, w, 0);
        bind(2, state, 0);
        bind(3, y, 0);
        bind(4, snap != null ? snap : state, 0);
        p.reset(KernelParams.GDN_CONV).i(channels).i(kernel).i(n).i(inStride).i(outStride).i(snap != null ? snapTok : -1);
        dispatch("gdn_conv", 5, 0, channelGrid(channels), 1, 1, CHANNEL_THREADS, 1, 1);
    }

    /** The gated delta rule over n tokens for every (value head, state column) (gdn_rule). */
    void gdnRule(Metal.Buffer qkv, Metal.Buffer proj, Metal.Buffer a, Metal.Buffer dtBias, Metal.Buffer state,
                 Metal.Buffer out, int keyHeads, int valueHeads, int keyDim, int valueDim, int qkvStride, int projStride,
                 int bOff, float l2Eps, int n) {
        gdnRule(qkv, proj, a, dtBias, state, out, keyHeads, valueHeads, keyDim, valueDim, qkvStride, projStride, bOff, l2Eps,
                false, n, null, -1);
    }

    /** As above; {@code tiled}: value head h reads key head h % keyHeads (Qwen3.5 GGUF order) instead of h / group. */
    void gdnRule(Metal.Buffer qkv, Metal.Buffer proj, Metal.Buffer a, Metal.Buffer dtBias, Metal.Buffer state,
                 Metal.Buffer out, int keyHeads, int valueHeads, int keyDim, int valueDim, int qkvStride, int projStride,
                 int bOff, float l2Eps, boolean tiled, int n) {
        gdnRule(qkv, proj, a, dtBias, state, out, keyHeads, valueHeads, keyDim, valueDim, qkvStride, projStride, bOff, l2Eps,
                tiled, n, null, -1);
    }

    /** As above; with {@code snapTok >= 0} the state after that token also goes to {@code snap} (same layout as {@code state}). */
    void gdnRule(Metal.Buffer qkv, Metal.Buffer proj, Metal.Buffer a, Metal.Buffer dtBias, Metal.Buffer state,
                 Metal.Buffer out, int keyHeads, int valueHeads, int keyDim, int valueDim, int qkvStride, int projStride,
                 int bOff, float l2Eps, boolean tiled, int n, Metal.Buffer snap, int snapTok) {
        bind(0, qkv, 0);
        bind(1, proj, 0);
        bind(2, a, 0);
        bind(3, dtBias, 0);
        bind(4, state, 0);
        bind(5, out, 0);
        bind(6, snap != null ? snap : state, 0);
        p.reset(KernelParams.GDN_RULE).i(keyHeads).i(valueHeads).i(keyDim).i(valueDim).i(n).i(qkvStride).i(projStride)
                .i(bOff).f(l2Eps).i(tiled ? 1 : 0).i(snap != null ? snapTok : -1);
        dispatch("gdn_rule", 7, 0, (valueDim + GDN_COLS - 1) / GDN_COLS, valueHeads, 1, GDN_COLS * 32, 1, 1);
    }

    /** o[row] = rmsnorm(o[row]) * w * silu(z[row]) for the n * valueHeads rows of valueDim (gdn_norm_gate). */
    void gdnNormGate(Metal.Buffer o, Metal.Buffer proj, Metal.Buffer w, int valueDim, int valueHeads, int projStride,
                     int zOff, int n) {
        bind(0, o, 0);
        bind(1, proj, 0);
        bind(2, w, 0);
        p.reset(KernelParams.GDN_NORM).i(valueDim).i(valueHeads).i(projStride).i(zOff).f(eps);
        int threads = Math.max(32, Math.min(ROW_THREADS, (valueDim + 31) / 32 * 32));
        dispatch("gdn_norm_gate", 3, 0, n * valueHeads, 1, 1, threads, 1, 1);
    }

    /** a[i] *= sigmoid(g[i]) for i < count (sigmoid_mul). */
    void sigmoidMul(Metal.Buffer a, Metal.Buffer g, int count) {
        bind(0, a, 0);
        bind(1, g, 0);
        p.reset(KernelParams.GATE).i(count);
        dispatch("sigmoid_mul", 2, 0, (count + ROW_THREADS - 1) / ROW_THREADS, 1, 1, ROW_THREADS, 1, 1);
    }

    /** x[t] += sigmoid(w . xb[t]) * y[t] for n rows of dim (shared_gate_add). */
    void sharedGateAdd(Metal.Buffer xb, Metal.Buffer w, Metal.Buffer y, Metal.Buffer x, int dim, int n) {
        bind(0, xb, 0);
        bind(1, w, 0);
        bind(2, y, 0);
        bind(3, x, 0);
        p.reset(KernelParams.GATE).i(dim);
        dispatch("shared_gate_add", 4, 0, n, 1, 1, ROW_THREADS, 1, 1);
    }

    // ---------------------------------------------------------------- batched decoding across slots (one row each)
    // MetalBackend.forwardBatch: the projections and the MLP run over the n rows of a batch at once (the writers
    // above), while RoPE, the cache store and attention run once per row, each in its own conversation's cache at its
    // own position. These are the kernels of the single-sequence writers above, dispatched for one token whose
    // activation rows start at a byte offset (qOff, outOff, ...) instead of row 0.

    /** {@link #rope} for the one token at pos whose q row starts at byte {@code qOff} of q. */
    void ropeRow(boolean adjacent, boolean kHalf, Metal.Buffer q, long qOff, Metal.Buffer kSrc, long kSrcOff,
                 Metal.Buffer kOut, long kOutOff, Metal.Buffer qw, Metal.Buffer kw, Metal.Buffer cos, Metal.Buffer sin,
                 int headDim, int qStride, int kvStride, int nHeads, int nKHeads, int pos, int rotOff, int rotHalf) {
        bind(0, q, qOff);
        bind(1, kSrc, kSrcOff);
        bindOrDummy(2, qw);
        bindOrDummy(3, kw);
        bind(4, cos, 0);
        bind(5, sin, 0);
        bind(6, kOut, kOutOff);
        p.reset(KernelParams.ROPE).i(headDim).i(qStride).i(kvStride).i(nHeads).i(pos).f(eps).i(qw != null).i(rotOff).i(rotHalf);
        dispatch((adjacent ? "rope_il" : "rope") + (kHalf && nKHeads > 0 ? "_h" : ""), 7, 0,
                nHeads + nKHeads, 1, 1, rotHalf, 1, 1);
    }

    /** {@link #attention} of the one token at pos whose q row starts at byte qOff; its output row goes to byte outOff. */
    void attentionRow(String kernel, int groups, Metal.Buffer q, long qOff, Metal.Buffer kc, Metal.Buffer vc,
                      Metal.Buffer out, long outOff, Metal.Buffer scores, int nHeads, int nKvHeads, int qkDim, int vDim,
                      int pos, int ctx, float scale, int qStride, int kStride, int vStride, int oStride, int window,
                      float softcap) {
        bind(0, q, qOff);
        bind(1, kc, 0);
        bind(2, vc, 0);
        bind(3, out, outOff);
        bind(4, scores, 0);
        p.reset(KernelParams.ATTN).i(nHeads).i(nKvHeads).i(qkDim).i(vDim).i(pos).i(ctx).f(scale)
                .i(qStride).i(kStride).i(vStride).i(oStride).i(window).i(0).f(softcap);
        dispatch(kernel, 5, 0, nHeads / groups, 1, 1, ATT_THREADS, 1, 1);
    }

    /** {@link #flashAttention} of the one token at pos whose q row starts at byte qOff; its output row goes to outOff. */
    void flashAttentionRow(KvDtype kv, Metal.Buffer q, long qOff, Metal.Buffer kc, Metal.Buffer vc, Metal.Buffer part,
                           Metal.Buffer scores, Metal.Buffer out, long outOff, int nHeads, int nKvHeads, int headDim,
                           int pos, int ctx, float scale, int qStride, int kvStride, int window, float softcap) {
        int len = pos + 1;
        int first = window > 0 ? Math.max(0, len - window) : 0;
        int keys = len - first;
        int nSplit = Math.min(FLASH_SPLITS, (keys + FLASH_CHUNK - 1) / FLASH_CHUNK);
        int chunk = (keys + nSplit - 1) / nSplit;
        bind(0, q, qOff);
        bind(1, kc, 0);
        bind(2, vc, 0);
        bind(3, part, 0);
        bind(4, scores, 0);
        p.reset(KernelParams.ATTN_SPLIT).i(nHeads).i(nKvHeads).i(headDim).i(pos).i(ctx).f(scale).i(qStride).i(kvStride)
                .i(window).f(softcap).i(nSplit).i(chunk);
        dispatch("attention_split" + kvSuffix(kv), 5, 0, nHeads, nSplit, 1, ATT_THREADS, 1, 1);
        bind(0, part, 0);
        bind(1, out, outOff);
        p.reset(KernelParams.ATTN_COMBINE).i(nHeads).i(headDim).i(nSplit);
        dispatch("attention_combine", 2, 0, nHeads, 1, 1, ROW_THREADS, 1, 1);
    }

    /** {@link #kvStoreQ8} of one token: its f32 staging rows at bytes kOff / vOff, its cache rows at byte offset off. */
    void kvStoreQ8Row(Metal.Buffer kStage, long kOff, Metal.Buffer vStage, long vOff, Metal.Buffer kc, Metal.Buffer vc,
                      long off, int headDim, int kvHeads) {
        int kvDim = headDim * kvHeads;
        bind(0, kStage, kOff);
        bind(1, vStage, vOff);
        bind(2, kc, off);
        bind(3, vc, off);
        p.reset(KernelParams.KV_STORE).i(headDim).i(kvDim).i(kvHeads).i((int) KvDtype.Q8.rowBytes(kvDim, kvHeads));
        dispatch("kv_store_q8", 4, 0, 2 * kvHeads, 1, 1, 32, 1, 1);
    }

    /**
     * {@code kv_put}: copies the f32 row of kvDim values at byte srcOff of src into the cache row at byte dstOff of dst,
     * rounded to half when {@code half} (the f16 cache).
     */
    void kvPut(boolean half, Metal.Buffer src, long srcOff, Metal.Buffer dst, long dstOff, int kvDim) {
        bind(0, src, srcOff);
        bind(1, dst, dstOff);
        p.reset(KernelParams.KV_PUT).i(kvDim).i(kvDim);
        dispatch(half ? "kv_put_h" : "kv_put", 2, 0, channelGrid(kvDim), 1, 1, CHANNEL_THREADS, 1, 1);
    }

    /** {@link #mlaLatStore} of one token: its latent row at byte latOff, its k_pe row at kpeOff, its cache row at off. */
    void mlaLatStoreRow(KvDtype kv, Metal.Buffer lat, long latOff, Metal.Buffer kpe, long kpeOff, Metal.Buffer cache,
                        long off, Attention.Mla m) {
        bind(0, lat, latOff);
        bind(1, kpe, kpeOff);
        bind(2, cache, off);
        mlaParams(m);
        dispatch("mla_lat_store" + kvSuffix(kv), 3, 0, 1, 1, 1, CHANNEL_THREADS, 1, 1);
    }
}
