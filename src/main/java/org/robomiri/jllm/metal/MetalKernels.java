package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * The kernel-side half of the Metal backends: owns the {@link Metal} context, the dispatch argument scratch and
 * a writer per kernel of {@code kernels.metal} (buffers, parameter struct, grid). {@link MetalBackend},
 * {@link MlaMetal} and {@link MambaMetalBackend} own the model's buffers and orchestrate the forward pass by
 * calling these writers; nothing else touches parameter layouts, so each kernel's struct has exactly one writer
 * here (checked by {@code MetalParamsTest}).
 *
 * <p>Kernels are looked up by name at dispatch time ({@link Metal#pipeline} memoizes and compiles on first use),
 * so only the variants a model runs are built.
 */
final class MetalKernels implements AutoCloseable {

    // ---------------------------------------------------------------- constants shared with kernels.metal
    /** Tokens per simdgroup of the single-row matmuls (TOK_TILE in kernels.metal). */
    static final int TOK_TILE = 8;
    /** Rows (matmul_swiglu_pf: units) and tokens per simdgroup of the *_pf prefill matmuls (PF_* in kernels.metal). */
    static final int PF_ROWS = 2;
    static final int PF_UNITS = 1;
    static final int PF_TOKS = 8;
    /** Simdgroups per matmul threadgroup: 128 threads = 4 rows (the kernels assume it). */
    static final int SG_PER_TG = 4;
    /** Tokens per attention dispatch; the scores scratch holds one row of ctx floats per (tile token, head). */
    static final int ATT_TILE = 32;
    /** Static threadgroup memory of the *_norm matmul kernels (their {@code threadgroup float buf[8]}). */
    static final int NORM_STATIC_TG_BYTES = 32;
    /**
     * Length bound to the matmul kernels' [[threadgroup(0)]] argument when the instantiation does not use it:
     * the plain (non-NORM) variants still declare the argument, and Metal requires every declared threadgroup
     * argument to have a length set before the dispatch.
     */
    static final int UNUSED_TG_BYTES = 16;
    /** Experts the router kernels can hold in threadgroup memory. */
    static final int MAX_EXPERTS = 1024;
    private static final int MAX_BUFS = 16; // matmul3 binds 14 buffers

    // ---------------------------------------------------------------- tunables
    /**
     * Chunks of at least this many tokens use the tiled prefill matmuls (bit-identical results, see the prefill
     * section of kernels.metal); smaller ones the single-row kernels. A huge value disables the tiled kernels.
     */
    static final int PREFILL_MIN_TOKENS = Integer.getInteger("jllm.metal.prefillMinTokens", 8);
    /**
     * Layers per command buffer: the dispatches of every COMMIT_LAYERS layers are submitted (without waiting)
     * while the following layers are still being encoded, so GPU execution overlaps the CPU-side encoding
     * instead of starting only after the whole pass is recorded. 0 = one command buffer per forward pass.
     */
    static final int COMMIT_LAYERS = Integer.getInteger("jllm.metal.commitLayers", 8);
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

    final Metal metal;
    private final float eps;
    private final Arena arena = Arena.ofShared();
    // dispatch argument scratch (re-used; the native side copies params and records the bindings at dispatch time)
    private final MemorySegment bufArgs;
    private final MemorySegment offArgs;
    private final MemorySegment params;
    /** Bound wherever a kernel declares a buffer the dispatch does not use (norm weights, biases, ...). */
    private final Metal.Buffer dummy;

    /** Owns {@code metal} from here on. */
    MetalKernels(Metal metal, float normEps) {
        this.metal = metal;
        this.eps = normEps;
        bufArgs = arena.allocate(ValueLayout.ADDRESS, MAX_BUFS);
        offArgs = arena.allocate(ValueLayout.JAVA_LONG, MAX_BUFS);
        params = arena.allocate(64);
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

    Metal.Buffer wrap(MemorySegment seg) {
        return metal.wrap(seg);
    }

    /**
     * Whether the *_norm matmul variants may be used for rows of {@code cols} floats: they keep the normalised
     * input row in dynamic threadgroup memory next to their static reduction scratch, which must fit the device's
     * threadgroup memory limit (32 KB: cols <= 8176). Otherwise a decode step runs rmsnorm + plain matmul.
     */
    boolean fusedNormFits(int cols) {
        return (long) cols * 4 + NORM_STATIC_TG_BYTES <= metal.maxThreadgroupMemory();
    }

    void begin() {
        metal.begin();
    }

    void commit() {
        metal.commit();
    }

    void commitAndWait() {
        metal.commitAndWait();
    }

    void abort() {
        metal.abort();
    }

    // ---------------------------------------------------------------- dispatch plumbing

    private void bind(int i, Metal.Buffer b, long offset) {
        bufArgs.setAtIndex(ValueLayout.ADDRESS, i, b.handle());
        offArgs.setAtIndex(ValueLayout.JAVA_LONG, i, offset);
    }

    private void bindOrDummy(int i, Metal.Buffer b) {
        bind(i, b != null ? b : dummy, 0);
    }

    private void dispatch(String kernel, int nbuf, int plen, int tgmem, int gx, int gy, int gz, int tx, int ty, int tz) {
        metal.dispatch(metal.pipeline(kernel), bufArgs, offArgs, nbuf, params, plen, tgmem, gx, gy, gz, tx, ty, tz);
    }

    private void i(int index, int v) {
        params.setAtIndex(ValueLayout.JAVA_INT, index, v);
    }

    private void f(int index, float v) {
        params.setAtIndex(ValueLayout.JAVA_FLOAT, index, v);
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

    /** Offset of the f32 scale region to bind as buffer(1); Q2_K keeps its scales inside the blocks (unused binding). */
    private static long scaleOff(QTensor t) {
        return t.type.scaleBytes == 0 ? 0 : t.scalesOffset;
    }

    /** Threadgroups along the row axis for the tiled prefill matmuls (PF_ROWS rows per simdgroup). */
    private static int pfGrid(int rows) {
        return (rows + SG_PER_TG * PF_ROWS - 1) / (SG_PER_TG * PF_ROWS);
    }

    private static int pfTiles(int n) {
        return (n + PF_TOKS - 1) / PF_TOKS;
    }

    private static int rowGrid(int rows) {
        return (rows + SG_PER_TG - 1) / SG_PER_TG;
    }

    private static int tokTiles(int n) {
        return (n + TOK_TILE - 1) / TOK_TILE;
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
        if (norm != null && n != 1) {
            throw new IllegalArgumentException("fused rmsnorm needs a single token");
        }
        bind(0, wbuf, 0);
        bind(1, wbuf, scaleOff(t));
        bind(2, in, inOff);
        bind(3, out, outOff);
        bindOrDummy(4, norm);
        i(0, t.rows);
        i(1, t.cols);
        i(2, n);
        i(3, acc ? 1 : 0);
        f(4, eps);
        f(5, softcap);
        if (norm == null && n >= PREFILL_MIN_TOKENS) {
            dispatch(kernel("matmul_pf", t.type, ""), 5, 24, 0, pfGrid(t.rows), pfTiles(n), 1, 32 * SG_PER_TG, 1, 1);
        } else {
            dispatch(kernel("matmul", t.type, norm != null ? "_norm" : ""), 5, 24, normTgMem(norm, t.cols),
                    rowGrid(t.rows), tokTiles(n), 1, 32 * SG_PER_TG, 1, 1);
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
        if (wq.type != wk.type || wk.type != wv.type || wq.cols != wk.cols || wk.cols != wv.cols) {
            throw new IllegalStateException("q/k/v projections differ in type or width"); // the loader guarantees they match
        }
        if (norm != null && n != 1) {
            throw new IllegalArgumentException("fused rmsnorm needs a single token");
        }
        boolean bias = bq != null;
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
        i(0, wq.rows);
        i(1, wk.rows);
        i(2, wv.rows);
        i(3, wq.cols);
        i(4, n);
        f(5, eps);
        i(6, bias ? 1 : 0);
        int rows = wq.rows + wk.rows + wv.rows;
        String vh = vHalf ? "_h" : "";
        if (norm == null && n >= PREFILL_MIN_TOKENS) {
            dispatch(kernel("matmul3_pf", wq.type, vh), 14, 28, 0, pfGrid(rows), pfTiles(n), 1, 32 * SG_PER_TG, 1, 1);
        } else {
            dispatch(kernel("matmul3", wq.type, (norm != null ? "_norm" : "") + vh), 14, 28, normTgMem(norm, wq.cols),
                    rowGrid(rows), tokTiles(n), 1, 32 * SG_PER_TG, 1, 1);
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
        bind(1, wbuf, scaleOff(t));
        bind(2, in, 0);
        bind(3, out, 0);
        bindOrDummy(4, norm);
        i(0, hidden);
        i(1, t.cols);
        i(2, n);
        f(3, eps);
        i(4, gelu ? 1 : 0);
        if (norm == null && n >= PREFILL_MIN_TOKENS) {
            int gx = (hidden + SG_PER_TG * PF_UNITS - 1) / (SG_PER_TG * PF_UNITS);
            dispatch(kernel("matmul_swiglu_pf", t.type, ""), 5, 20, 0, gx, pfTiles(n), 1, 32 * SG_PER_TG, 1, 1);
        } else {
            dispatch(kernel("matmul_swiglu", t.type, norm != null ? "_norm" : ""), 5, 20, normTgMem(norm, t.cols),
                    rowGrid(hidden), tokTiles(n), 1, 32 * SG_PER_TG, 1, 1);
        }
    }

    // ---------------------------------------------------------------- norms and rope

    /** out[t] = rmsnorm(in[t]) * weight for n rows of the given width (in read from the byte offset inOff). */
    void rmsnorm(Metal.Buffer in, long inOff, Metal.Buffer weight, Metal.Buffer out, int n, int width) {
        bind(0, in, inOff);
        bind(1, weight, 0);
        bind(2, out, 0);
        i(0, width);
        f(1, eps);
        dispatch("rmsnorm", 3, 8, 0, n, 1, 1, 256, 1, 1);
    }

    /** out[t] += rmsnorm(y[t]) * weight for n rows of the given width (Glm4's post-norms). */
    void rmsnormAdd(Metal.Buffer y, Metal.Buffer weight, Metal.Buffer out, int n, int width) {
        bind(0, y, 0);
        bind(1, weight, 0);
        bind(2, out, 0);
        i(0, width);
        f(1, eps);
        dispatch("rmsnorm_add", 3, 8, 0, n, 1, 1, 256, 1, 1);
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
        i(0, headDim);
        i(1, qStride);
        i(2, kvStride);
        i(3, nHeads);
        i(4, pos);
        f(5, eps);
        i(6, qw != null ? 1 : 0);
        i(7, rotOff);
        i(8, rotHalf);
        dispatch((adjacent ? "rope_il" : "rope") + (kHalf && nKHeads > 0 ? "_h" : ""), 7, 36, 0,
                nHeads + nKHeads, n, 1, rotHalf, 1, 1);
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
        i(0, nHeads);
        i(1, nKvHeads);
        i(2, qkDim);
        i(3, vDim);
        i(4, pos);
        i(5, ctx);
        f(6, scale);
        i(7, qStride);
        i(8, kStride);
        i(9, vStride);
        i(10, oStride);
        i(11, window);
        f(13, softcap);
        for (int t0 = 0; t0 < n; t0 += ATT_TILE) {
            i(12, t0);
            dispatch(kernel, 5, 56, 0, nHeads / groups, Math.min(ATT_TILE, n - t0), 1, 256, 1, 1);
        }
    }

    /**
     * Flash decoding for the single token at pos: {@code attention_split} over (heads, nSplit) into the partials
     * buffer {@code part} (heads * FLASH_MAX_SPLITS * (headDim + 2) floats), then {@code attention_combine} into out.
     */
    void flashAttention(boolean kvHalf, Metal.Buffer q, Metal.Buffer kc, Metal.Buffer vc, Metal.Buffer part,
                        Metal.Buffer scores, Metal.Buffer out, int nHeads, int nKvHeads, int headDim, int pos, int ctx,
                        float scale, int qStride, int kvStride, int window, float softcap) {
        int len = pos + 1;
        int first = window > 0 ? Math.max(0, len - window) : 0;
        int keys = len - first;
        int nSplit = Math.min(FLASH_MAX_SPLITS, (keys + FLASH_CHUNK - 1) / FLASH_CHUNK);
        int chunk = (keys + nSplit - 1) / nSplit;
        bind(0, q, 0);
        bind(1, kc, 0);
        bind(2, vc, 0);
        bind(3, part, 0);
        bind(4, scores, 0);
        i(0, nHeads);
        i(1, nKvHeads);
        i(2, headDim);
        i(3, pos);
        i(4, ctx);
        f(5, scale);
        i(6, qStride);
        i(7, kvStride);
        i(8, window);
        f(9, softcap);
        i(10, nSplit);
        i(11, chunk);
        dispatch(kvHalf ? "attention_split_h" : "attention_split", 5, 48, 0, nHeads, nSplit, 1, 256, 1, 1);
        bind(0, part, 0);
        bind(1, out, 0);
        i(0, nHeads);
        i(1, headDim);
        i(2, nSplit);
        dispatch("attention_combine", 2, 12, 0, nHeads, 1, 1, 256, 1, 1);
    }

    // ---------------------------------------------------------------- mixture of experts

    /** Softmax top-k routing of n rows of x: assign[t * topK + k] and the (optionally renormalised) weights. */
    void router(Metal.Buffer x, Metal.Buffer w, Metal.Buffer assign, Metal.Buffer rw, int nExperts, int dim, int topK,
                boolean normalize, int n) {
        bind(0, x, 0);
        bind(1, w, 0);
        bind(2, assign, 0);
        bind(3, rw, 0);
        i(0, nExperts);
        i(1, dim);
        i(2, topK);
        i(3, normalize ? 1 : 0);
        dispatch("router", 4, 16, 0, n, 1, 1, 256, 1, 1);
    }

    /** DeepSeek routing (see router_ds in kernels.metal and DeepSeekRouter.java); {@code bias} only for noaux_tc. */
    void routerDs(Metal.Buffer x, Metal.Buffer w, Metal.Buffer assign, Metal.Buffer rw, Metal.Buffer bias, int nExperts,
                  int dim, int topK, boolean normalize, int nGroup, int topkGroup, boolean sigmoid, int method,
                  boolean scaleAlways, float routedScale, int n) {
        bind(0, x, 0);
        bind(1, w, 0);
        bind(2, assign, 0);
        bind(3, rw, 0);
        bindOrDummy(4, bias);
        i(0, nExperts);
        i(1, dim);
        i(2, topK);
        i(3, normalize ? 1 : 0);
        i(4, nGroup);
        i(5, topkGroup);
        i(6, sigmoid ? 1 : 0);
        i(7, method);
        i(8, scaleAlways ? 1 : 0);
        f(9, routedScale);
        dispatch("router_ds", 5, 40, 0, n, 1, 1, 256, 1, 1);
    }

    /** Per-slot gate/up projections with SwiGLU: out[slot, :] for the n * topK slots of assign (moe_up). */
    void moeUp(QTensor gateUp, Metal.Buffer wbuf, Metal.Buffer x, Metal.Buffer assign, Metal.Buffer out, int moeDim,
               int topK, int n) {
        bind(0, wbuf, 0);
        bind(1, wbuf, scaleOff(gateUp));
        bind(2, x, 0);
        bind(3, assign, 0);
        bind(4, out, 0);
        i(0, 2 * moeDim); // rows per expert in the stacked tensor (gate | up)
        i(1, gateUp.cols);
        i(2, topK);
        dispatch(kernel("moe_up", gateUp.type, ""), 5, 12, 0, rowGrid(moeDim), n * topK, 1, 32 * SG_PER_TG, 1, 1);
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
        i(0, dim);
        i(1, down.cols);
        i(2, topK);
        dispatch(kernel("moe_down", down.type, ""), 6, 12, 0, rowGrid(dim), n, 1, 32 * SG_PER_TG, 1, 1);
    }

    // ---------------------------------------------------------------- DeepSeek MLA

    /** Assembles the per-head keys [k_nope | k_pe] and values of n tokens into the cache rows at kOff / vOff. */
    void mlaKvStore(Metal.Buffer kvb, Metal.Buffer kpe, Metal.Buffer kc, long kOff, Metal.Buffer vc, long vOff,
                    int nope, int rope, int vDim, int heads, int n) {
        bind(0, kvb, 0);
        bind(1, kpe, 0);
        bind(2, kc, kOff);
        bind(3, vc, vOff);
        i(0, nope);
        i(1, rope);
        i(2, vDim);
        i(3, heads * (nope + vDim));
        i(4, heads * (nope + rope));
        i(5, heads * vDim);
        dispatch("mla_kv_store", 4, 24, 0, heads, n, 1, 64, 1, 1);
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
        i(0, dInner);
        i(1, dConv);
        i(2, n);
        dispatch("mamba_conv", 5, 12, 0, (dInner + 63) / 64, 1, 1, 64, 1, 1);
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
        i(0, dInner);
        i(1, dState);
        i(2, dtRank);
        i(3, n);
        dispatch("mamba_scan", 9, 16, 0, (dInner + 63) / 64, 1, 1, 64, 1, 1);
    }
}
