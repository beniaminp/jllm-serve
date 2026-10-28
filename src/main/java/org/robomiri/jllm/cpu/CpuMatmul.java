package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.LayerWeights;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

/**
 * The quantized matmul of the CPU backends, shared by {@link CpuBackend}, {@link MlaCpu} and
 * {@link MambaCpuBackend}: {@code out[t, r] (+)= dot(W[r, :], in[t, :])} for {@code t < n}, parallel over output
 * rows (single token) or four-row groups (several tokens), see docs/07 §7.4.
 *
 * <ul>
 *   <li>{@code n == 1}: every row is one {@link VectorKernels#dotRow} streaming the quantized blocks, or, with int8
 *       activations on and a Q4_0 / Q8_0 / K-quant weight ({@link #useInt8}), one {@link Int8Kernels#dotRow} /
 *       {@link KQuantKernels#dotRow} against the activation quantized once per call ({@link #quantizeInt8}). With the
 *       native library ({@link NativeCpu}) loaded, each thread's range of rows is one call into it instead.</li>
 *   <li>{@code 1 < n <=} {@link #fewRows()} ({@code backend.cpu.batch-int8-rows}: a decode step batched across slots,
 *       a speculative verification run, a short prefill) with the native int8 kernel for the tensor: every token is
 *       quantized once and run through the single-token kernel, chunk of weight rows by chunk ({@link #CHUNK_BYTES}),
 *       so a chunk read from memory serves all the tokens from the cache; each token's result is the single-token
 *       result bit for bit (docs/07 §7.4.1).</li>
 *   <li>{@code n > 1} otherwise: four rows are dequantized into thread-local scratch once and reused for every token
 *       ({@link VectorKernels#dot4x4} / {@link VectorKernels#dot4}); rows left over when {@code rows % 4 != 0}
 *       fall back to {@code dotRow}. With {@link CpuBackend#DETERMINISTIC} the tokens go one by one through the
 *       single-token path instead (docs/07 §7.14).</li>
 * </ul>
 *
 * <p>Owns the thread-local float scratch (at least {@code 4 * maxCols} floats, the dequantized row group) that the
 * backends also use for attention scores and router logits, and the int8 activation buffers, which are written by
 * the calling thread before a row loop is published and therefore need no synchronization.
 */
final class CpuMatmul {

    final ThreadPool pool;
    final ThreadLocal<float[]> scratch;
    private final boolean int8;
    private final boolean deterministic;
    private short[] xq;
    private float[] xs;
    private int[] xsum;
    private float[] permuted;
    /** The native int8 kernels (null: the Java ones) and their activation buffer. */
    private final NativeCpu nat;
    private final NativeCpu.Activation act = new NativeCpu.Activation();
    /** One activation per token of a few-rows pass ({@link #mulFewNative}). */
    private NativeCpu.Activation[] acts = new NativeCpu.Activation[0];

    /**
     * Passes of up to this many tokens run every token through the native single-token int8 kernel
     * ({@code backend.cpu.batch-int8-rows}, {@link #configureFewRows}); 0 = the float multi-row kernels for every n > 1.
     */
    private static volatile int fewRows = 3; // = BATCH_DEFAULT

    /**
     * Weight bytes of the row chunk a few-rows pass reuses from the cache for all its tokens: within a core's L2. Measured
     * on Qwen3-0.6B, 4 AVX512-VNNI cores (docs/07 §7.4.1): 8 KB and 32 KB chunks slower, 512 KB no better.
     */
    static final long CHUNK_BYTES = Long.getLong("jllm.cpu.fewRowsChunkBytes", 128 * 1024);

    /** Applies {@code backend.cpu.batch-int8-rows} (0..64) to every CPU backend of the process. */
    static void configureFewRows(int rows) {
        if (rows < 0 || rows > 64) {
            throw new IllegalArgumentException("backend.cpu.batch-int8-rows must be 0..64, got " + rows);
        }
        fewRows = rows;
    }

    /** The current {@code backend.cpu.batch-int8-rows}. */
    static int fewRows() {
        return fewRows;
    }

    /**
     * @param int8       quantize single-token activations to int8 and use the integer kernels for Q4_0 / Q8_0 weights
     * @param scratchLen floats per thread-local scratch array; at least four times the widest {@code cols} that
     *                   will be multiplied with {@code n > 1}
     */
    CpuMatmul(ThreadPool pool, boolean int8, boolean deterministic, int scratchLen) {
        this(pool, int8, deterministic, scratchLen, int8 ? NativeCpu.get() : null);
    }

    /** @param nat the native int8 kernels to use, or null for the Java ones */
    CpuMatmul(ThreadPool pool, boolean int8, boolean deterministic, int scratchLen, NativeCpu nat) {
        this.pool = pool;
        this.int8 = int8;
        this.deterministic = deterministic;
        this.scratch = ThreadLocal.withInitial(() -> new float[scratchLen]);
        this.nat = int8 ? nat : null;
    }

    /** What computes the int8 dot products, for the backends' names. */
    String describeInt8() {
        return nat != null ? "int8 activations (native " + nat.level() + ")" : Int8Kernels.describe();
    }

    /** The widest {@code cols} of any tensor of {@code w}: the row-group scratch must hold four rows of it. */
    static int maxCols(ModelWeights w) {
        int m = Math.max(w.embed.cols, w.lmHead.cols);
        for (LayerWeights L : w.layers) {
            m = Math.max(m, L.maxCols());
        }
        if (w.mtp != null) {
            m = Math.max(m, Math.max(w.mtp.ehProj().cols, w.mtp.layer().maxCols()));
        }
        return m;
    }

    boolean int8() {
        return int8;
    }

    /** Probe results per weights ({@link #int8Faster}): every slot of a model would measure the same thing. */
    private static final java.util.Map<ModelWeights, Boolean> PROBED =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    /**
     * Whether single-token matmuls of {@code w} run faster with int8 activations than with float ones on this machine
     * ({@code backend.cpu.int8-activations=auto}). Integer dot products win where the CPU multiplies bytes cheaply
     * (ARM NEON's {@code sdot}, AVX-VNNI); on AVX-512 without them the float kernels can be faster, as the weights are
     * streamed either way and the int8 path adds the activation's quantization. Times one token's single-token
     * matmuls both ways on {@code threads} threads (every tensor of the first decoder layer, weighted by the layer
     * count, plus the LM head) and keeps the faster; a model with no type the int8 kernels cover gives false.
     */
    static boolean int8Faster(ModelWeights w, int threads) {
        Boolean known = PROBED.get(w);
        if (known != null) {
            return known;
        }
        // one token's matmuls: the first decoder layer's tensors (times the layer count) and the LM head
        java.util.List<QTensor> layer = new java.util.ArrayList<>();
        for (LayerWeights L : w.layers) {
            if (L instanceof org.robomiri.jllm.model.DecoderLayer d) {
                collect(d, layer, 0);
                break;
            }
        }
        boolean faster = false;
        int maxCols = w.lmHead.cols;
        for (QTensor t : layer) {
            maxCols = Math.max(maxCols, t.cols);
        }
        try (ThreadPool pool = new ThreadPool(threads)) {
            CpuMatmul i8 = new CpuMatmul(pool, true, false, 4 * maxCols);
            CpuMatmul f32 = new CpuMatmul(pool, false, false, 4 * maxCols);
            double t8 = 0;
            double tf = 0;
            boolean any = false;
            boolean nat = false;
            int layers = w.layers.length;
            java.util.List<QTensor> all = new java.util.ArrayList<>(layer);
            all.add(w.lmHead);
            for (int k = 0; k < all.size(); k++) {
                QTensor t = all.get(k);
                double weight = k < layer.size() ? layers : 1;
                any |= i8.useInt8(t);
                nat |= i8.nativeInt8(t);
                long[] best = time(i8, f32, t);
                t8 += weight * best[0];
                tf += weight * best[1];
            }
            if (any) {
                faster = t8 < tf;
                // with the native library loaded, i8 runs its kernels: the probe times what decode would run
                System.err.printf("int8 activations: auto -> %s (one token's matmuls: int8 %.1f ms [%s], float %.1f ms)%n",
                        faster ? "on" : "off", t8 / 1e6, nat ? "native " + i8.nat.level() : "java", tf / 1e6);
            }
        }
        PROBED.put(w, faster);
        return faster;
    }

    /** Probe results of {@link #batchRows} per weights. */
    private static final java.util.Map<ModelWeights, Integer> BATCH_PROBED =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    /** The pass sizes {@link #batchRows} tries, in ascending order; the last is the cap of {@code auto}. */
    private static final int[] BATCH_CANDIDATES = {2, 3, 4, 6, 8, 12, 16};

    /** The value {@code backend.cpu.batch-int8-rows=auto} takes where it cannot measure (the 4-core x86 VM of docs/07 §7.4.1). */
    static final int BATCH_DEFAULT = 3;

    /**
     * {@code backend.cpu.batch-int8-rows=auto}: the largest pass size n (of {@link #BATCH_CANDIDATES}) up to which running
     * the n tokens one by one through the native int8 kernel ({@link #mulFewNative}) is still faster than the float
     * 4x4 tiles, measured on this machine and these weights once. Both ways time the matmuls of the first decoder layer
     * (layer count times, plus the LM head's first 2048 rows scaled up) at n = 2, 3, 4, 6, ... 16, best of 4 alternating runs after 2
     * warm-up ones; a binary search over the candidates (the per-row cost grows linearly with n, the
     * tiles' much more slowly, so they cross once). Without the native library, or with no native int8 tensor in the
     * first layer, {@link #BATCH_DEFAULT}. Logs one line.
     */
    static int batchRows(ModelWeights w, int threads) {
        Integer known = BATCH_PROBED.get(w);
        if (known != null) {
            return known;
        }
        java.util.List<QTensor> layer = new java.util.ArrayList<>();
        for (LayerWeights L : w.layers) {
            if (L instanceof org.robomiri.jllm.model.DecoderLayer d) {
                collect(d, layer, 0);
                break;
            }
        }
        int best = BATCH_DEFAULT;
        if (NativeCpu.get() != null && !layer.isEmpty()) {
            long t0 = System.nanoTime();
            int maxCols = 0;
            for (QTensor t : layer) {
                maxCols = Math.max(maxCols, t.cols);
            }
            StringBuilder trail = new StringBuilder();
            try (ThreadPool pool = new ThreadPool(threads)) {
                CpuMatmul i8 = new CpuMatmul(pool, true, false, 4 * maxCols);
                CpuMatmul f32 = new CpuMatmul(pool, false, false, 4 * maxCols);
                java.util.List<QTensor> used = new java.util.ArrayList<>();
                for (QTensor t : layer) {
                    if (i8.nativeInt8(t)) {
                        used.add(t);
                    }
                }
                boolean head = i8.nativeInt8(w.lmHead);
                int headRows = Math.min(w.lmHead.rows, 2048);
                if (!used.isEmpty()) {
                    // the first runs fault the mapped weights in and compile the kernels: take them out of the timings
                    for (QTensor t : used) {
                        timeRows(i8, f32, t, t.rows, 2);
                    }
                    if (head) {
                        timeRows(i8, f32, w.lmHead, headRows, 2);
                    }
                    // binary search for the last candidate where per-row int8 wins (best 1: it loses even at 2 rows)
                    best = 1;
                    int lo = 0;
                    int hi = BATCH_CANDIDATES.length - 1;
                    while (lo <= hi) {
                        int mid = (lo + hi) >>> 1;
                        int n = BATCH_CANDIDATES[mid];
                        double t8 = 0;
                        double tf = 0;
                        for (QTensor t : used) {
                            long[] ns = timeRows(i8, f32, t, t.rows, n);
                            t8 += w.layers.length * ns[0];
                            tf += w.layers.length * ns[1];
                        }
                        if (head) { // the LM head's first rows, scaled to all of them: the tiles dequantize every one
                            long[] ns = timeRows(i8, f32, w.lmHead, headRows, n);
                            t8 += (double) w.lmHead.rows / headRows * ns[0];
                            tf += (double) w.lmHead.rows / headRows * ns[1];
                        }
                        trail.append(String.format(java.util.Locale.ROOT, "%s%d:%.1f/%.1f", trail.length() > 0 ? " " : "", n,
                                t8 / 1e6, tf / 1e6));
                        if (t8 < tf) {
                            best = n;
                            lo = mid + 1;
                        } else {
                            hi = mid - 1;
                        }
                    }
                }
            }
            System.err.printf("batch int8 rows: auto -> %d (one pass's matmuls, rows:int8/float ms: %s; probe %d ms)%n",
                    best, trail, (System.nanoTime() - t0) / 1_000_000);
        }
        BATCH_PROBED.put(w, best);
        return best;
    }

    /** Best of 4 alternating runs (after 2 warm-up ones) of an {@code n}-token matmul of the first {@code rows} rows of {@code t}: per-row int8, float tiles, in ns. */
    private static long[] timeRows(CpuMatmul i8, CpuMatmul f32, QTensor t, int rows, int n) {
        float[] in = new float[n * t.cols];
        java.util.Random r = new java.util.Random(1);
        for (int i = 0; i < in.length; i++) {
            in[i] = (float) r.nextGaussian();
        }
        float[] out = new float[n * t.rows];
        long best8 = Long.MAX_VALUE;
        long bestF = Long.MAX_VALUE;
        for (int round = 0; round < 6; round++) {
            long a = System.nanoTime();
            i8.mulFewNative(t, rows, in, out, 0, n, false);
            long b = System.nanoTime();
            f32.mul(t, rows, in, out, 0, n, false);
            long c = System.nanoTime();
            if (round >= 2) {
                best8 = Math.min(best8, b - a);
                bestF = Math.min(bestF, c - b);
            }
        }
        return new long[]{best8, bestF};
    }

    /** Best of 12 alternating runs (after 4 warm-up rounds) of a single-token {@code t} matmul each way, in ns. */
    private static long[] time(CpuMatmul i8, CpuMatmul f32, QTensor t) {
        float[] in = new float[t.cols];
        java.util.Random r = new java.util.Random(1);
        for (int i = 0; i < in.length; i++) {
            in[i] = (float) r.nextGaussian();
        }
        float[] out = new float[t.rows];
        long best8 = Long.MAX_VALUE;
        long bestF = Long.MAX_VALUE;
        for (int round = 0; round < 16; round++) { // alternating, so both see the same clocks and caches
            long a = System.nanoTime();
            i8.mul(t, in, out, 0, 1, false);
            long b = System.nanoTime();
            f32.mul(t, in, out, 0, 1, false);
            long c = System.nanoTime();
            if (round >= 4) { // the first rounds warm up the JIT
                best8 = Math.min(best8, b - a);
                bestF = Math.min(bestF, c - b);
            }
        }
        return new long[]{best8, bestF};
    }

    /** The quantized tensors of a layer's records (attention, MLP or experts), in declaration order. */
    private static void collect(Object o, java.util.List<QTensor> out, int depth) {
        if (o == null || depth > 3) {
            return;
        }
        if (o instanceof QTensor q) {
            out.add(q);
            return;
        }
        if (!o.getClass().isRecord()) {
            return;
        }
        for (var c : o.getClass().getRecordComponents()) {
            try {
                collect(c.getAccessor().invoke(o), out, depth + 1);
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException(e);
            }
        }
    }

    /**
     * The int8 path applies to the formats {@link Int8Kernels} has kernels for (Q4_0, Q8_0 and their ggml layouts
     * Q4_0G, Q8_0G), the K-quants of {@link KQuantKernels} (Q4_K, Q5_K, Q6_K) and the other ggml formats of
     * {@link GgufInt8Kernels} (Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4, IQ4_XS, Q3_K); Q2_K, F16 and BF16 stay on the float kernels.
     */
    boolean useInt8(QTensor wt) {
        return int8 && ((wt.type == QType.Q4_0 || wt.type == QType.Q8_0 || wt.type == QType.Q4_0G || wt.type == QType.Q8_0G)
                && wt.cols % QType.BLOCK == 0 || KQuantKernels.supports(wt) || GgufInt8Kernels.supports(wt));
    }

    /** Whether the single-token int8 matmul of {@code wt} runs in the native library. */
    boolean nativeInt8(QTensor wt) {
        return nat != null && useInt8(wt) && nat.supports(wt);
    }

    /** Quantizes {@code in[off .. off + n)} into the native activation buffer (for {@link #nativeRows}). */
    void quantizeNative(float[] in, int off, int n) {
        nat.quantize(in, off, n, act);
    }

    /**
     * {@code out[outOff + i] (+)= dot(W[r0 + i, :], x)} in the native library, x the last {@link #quantizeNative}
     * activation from 32-block {@code xBlock} on.
     */
    void nativeRows(QTensor wt, int r0, int r1, int xBlock, float[] out, int outOff, boolean acc) {
        nat.dotRows(wt, r0, r1, act, xBlock, out, outOff, acc);
    }

    /** Row {@code r} of {@code wt} (a {@link #useInt8} type) times the last {@link #quantizeInt8} activation from {@code xo}. */
    float int8Dot(QTensor wt, int r, short[] xq, float[] xs, int xo) {
        if (GgufInt8Kernels.supports(wt)) {
            return GgufInt8Kernels.dotRow(wt, r, xq, xs, xsum, xo);
        }
        return KQuantKernels.supports(wt) ? KQuantKernels.dotRow(wt, r, xq, xs, xsum, xo) : Int8Kernels.dotRow(wt, r, xq, xs, xo);
    }

    /** Quantizes {@code in[off .. off + n)} into the int8 scratch (grown if needed) and returns it; scales are in {@link #xs()}. */
    short[] quantizeInt8(float[] in, int off, int n) {
        if (xq == null || xq.length < n) {
            xq = new short[n];
            xs = new float[n / QType.BLOCK];
            xsum = new int[n / QType.BLOCK];
        }
        Int8Kernels.quantize(in, off, n, xq, xs);
        KQuantKernels.blockSums(xq, n, xsum); // the K-quants' min terms; n adds, next to nothing
        return xq;
    }

    /** Block scales of the last {@link #quantizeInt8}. */
    float[] xs() {
        return xs;
    }

    /** {@code out[outOff + t * wt.rows + r] (+)= dot(W[r, :], in[t * cols, :])} over all rows of {@code wt}. */
    void mul(QTensor wt, float[] in, float[] out, int outOff, int n, boolean acc) {
        mul(wt, wt.rows, in, out, outOff, n, acc);
    }

    /** {@link #mul} over the first {@code rows} rows of {@code wt}; for {@code n > 1} the output row stride stays {@code wt.rows}. */
    void mul(QTensor wt, int rows, float[] in, float[] out, int outOff, int n, boolean acc) {
        int cols = wt.cols;
        int stride = wt.rows;
        if (n == 1) {
            mulOne(wt, rows, in, 0, out, outOff, acc);
            return;
        }
        if (deterministic) {
            for (int t = 0; t < n; t++) {
                mulOne(wt, rows, in, t * cols, out, outOff + t * stride, acc);
            }
            return;
        }
        if (n <= fewRows && nativeInt8(wt)) {
            mulFewNative(wt, rows, in, out, outOff, n, acc);
            return;
        }
        if (KQuantKernels.supports(wt)) {
            mulPermuted(wt, rows, in, out, outOff, n, acc);
            return;
        }
        int groups = (rows + 3) / 4;
        pool.forRange(groups, 1, (g0, g1) -> {
            float[] tmp = scratch.get();
            for (int g = g0; g < g1; g++) {
                int r = g * 4;
                if (r + 4 <= rows) {
                    for (int k = 0; k < 4; k++) {
                        VectorKernels.dequantRow(wt, r + k, tmp, k * cols);
                    }
                    int t = 0;
                    for (; t + 4 <= n; t += 4) {
                        int xo = t * cols;
                        int oo = outOff + t * stride + r;
                        VectorKernels.dot4x4(tmp, 0, cols, in, xo, xo + cols, xo + 2 * cols, xo + 3 * cols,
                                out, oo, oo + stride, oo + 2 * stride, oo + 3 * stride, acc);
                    }
                    for (; t < n; t++) {
                        VectorKernels.dot4(tmp, 0, cols, in, t * cols, out, outOff + t * stride + r, acc);
                    }
                } else {
                    for (int rr = r; rr < rows; rr++) {
                        for (int t = 0; t < n; t++) {
                            float v = VectorKernels.dotRow(wt, rr, in, t * cols);
                            int o = outOff + t * stride + rr;
                            out[o] = acc ? out[o] + v : v;
                        }
                    }
                }
            }
        });
    }

    /**
     * {@link #mul} for a few tokens with the native int8 kernel: each token quantized once (as {@link #mulOne} does), then
     * per thread, chunk of {@link #CHUNK_BYTES} weight rows by chunk, the single-token kernel for every token. The chunk
     * comes from memory once and from the cache for the other tokens, so the pass costs about one token's weight
     * traffic plus the extra tokens' integer arithmetic, where the float path dequantizes every weight row first; and
     * each token's output is {@link #mulOne}'s bit for bit (a row's dot product does not depend on the chunking).
     */
    private void mulFewNative(QTensor wt, int rows, float[] in, float[] out, int outOff, int n, boolean acc) {
        int cols = wt.cols;
        int stride = wt.rows;
        if (acts.length < n) {
            NativeCpu.Activation[] grown = java.util.Arrays.copyOf(acts, n);
            for (int t = acts.length; t < n; t++) {
                grown[t] = new NativeCpu.Activation();
            }
            acts = grown;
        }
        NativeCpu.Activation[] a = acts;
        for (int t = 0; t < n; t++) {
            nat.quantize(in, t * cols, cols, a[t]);
        }
        long rowBytes = (long) wt.blocksPerRow() * (wt.type.blockBytes + wt.type.scaleBytes);
        int chunk = (int) Math.max(16, CHUNK_BYTES / Math.max(1, rowBytes));
        pool.forRange(rows, 16, (r0, r1) -> {
            for (int c0 = r0; c0 < r1; c0 += chunk) {
                int c1 = Math.min(r1, c0 + chunk);
                for (int t = 0; t < n; t++) {
                    nat.dotRows(wt, c0, c1, a[t], 0, out, outOff + t * stride + c0, acc);
                }
            }
        });
    }

    /**
     * {@link #mul} for the K-quants: rows dequantized in {@link KQuantKernels}' permuted order (no widening casts) and
     * the activations permuted the same way, once, by the calling thread before the row loop is published.
     */
    private void mulPermuted(QTensor wt, int rows, float[] in, float[] out, int outOff, int n, boolean acc) {
        int cols = wt.cols;
        int stride = wt.rows;
        if (permuted == null || permuted.length < n * cols) {
            permuted = new float[n * cols];
        }
        float[] xp = permuted;
        KQuantKernels.permute(in, n, cols, xp);
        int groups = (rows + 3) / 4;
        pool.forRange(groups, 1, (g0, g1) -> {
            float[] tmp = scratch.get();
            for (int g = g0; g < g1; g++) {
                int r = g * 4;
                if (r + 4 <= rows) {
                    for (int k = 0; k < 4; k++) {
                        KQuantKernels.dequantPermuted(wt, r + k, tmp, k * cols);
                    }
                    int t = 0;
                    for (; t + 4 <= n; t += 4) {
                        int xo = t * cols;
                        int oo = outOff + t * stride + r;
                        VectorKernels.dot4x4(tmp, 0, cols, xp, xo, xo + cols, xo + 2 * cols, xo + 3 * cols,
                                out, oo, oo + stride, oo + 2 * stride, oo + 3 * stride, acc);
                    }
                    for (; t < n; t++) {
                        VectorKernels.dot4(tmp, 0, cols, xp, t * cols, out, outOff + t * stride + r, acc);
                    }
                } else {
                    for (int rr = r; rr < rows; rr++) {
                        KQuantKernels.dequantPermuted(wt, rr, tmp, 0);
                        for (int t = 0; t < n; t++) {
                            float v = VectorKernels.dot(tmp, 0, xp, t * cols, cols);
                            int o = outOff + t * stride + rr;
                            out[o] = acc ? out[o] + v : v;
                        }
                    }
                }
            }
        });
    }

    /** The single-token kernel: {@code out[outOff + r] (+)= dot(W[r, :], in[inOff, :])} for {@code r < rows}, parallel over rows. */
    void mulOne(QTensor wt, int rows, float[] in, int inOff, float[] out, int outOff, boolean acc) {
        if (nativeInt8(wt)) { // one downcall per chunk of rows
            nat.quantize(in, inOff, wt.cols, act);
            pool.forRange(rows, 16, (r0, r1) -> nat.dotRows(wt, r0, r1, act, 0, out, outOff + r0, acc));
            return;
        }
        if (useInt8(wt)) {
            short[] xq = quantizeInt8(in, inOff, wt.cols);
            float[] xs = this.xs;
            pool.forRange(rows, 16, (r0, r1) -> {
                for (int r = r0; r < r1; r++) {
                    float v = int8Dot(wt, r, xq, xs, 0);
                    out[outOff + r] = acc ? out[outOff + r] + v : v;
                }
            });
            return;
        }
        if (wt.type == QType.Q8_0 && wt.second == null) {
            // dequantize, then a float dot: C2 can fail to intrinsify dotRow's byte-to-float conversion for Q8_0 in
            // the full engine (a 70x slower LM head on AVX-512, docs/07 §7.10.1); the dequantization kernel is unaffected
            int cols = wt.cols;
            pool.forRange(rows, 16, (r0, r1) -> {
                float[] tmp = scratch.get();
                for (int r = r0; r < r1; r++) {
                    VectorKernels.dequantRow(wt, r, tmp, 0);
                    float v = VectorKernels.dot(tmp, 0, in, inOff, cols);
                    out[outOff + r] = acc ? out[outOff + r] + v : v;
                }
            });
            return;
        }
        pool.forRange(rows, 16, (r0, r1) -> {
            for (int r = r0; r < r1; r++) {
                float v = VectorKernels.dotRow(wt, r, in, inOff);
                out[outOff + r] = acc ? out[outOff + r] + v : v;
            }
        });
    }
}
