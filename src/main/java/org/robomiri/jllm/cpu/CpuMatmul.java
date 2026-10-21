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
 *       {@link KQuantKernels#dotRow} against the activation quantized once per call ({@link #quantizeInt8}).</li>
 *   <li>{@code n > 1}: four rows are dequantized into thread-local scratch once and reused for every token
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

    /**
     * @param int8       quantize single-token activations to int8 and use the integer kernels for Q4_0 / Q8_0 weights
     * @param scratchLen floats per thread-local scratch array; at least four times the widest {@code cols} that
     *                   will be multiplied with {@code n > 1}
     */
    CpuMatmul(ThreadPool pool, boolean int8, boolean deterministic, int scratchLen) {
        this.pool = pool;
        this.int8 = int8;
        this.deterministic = deterministic;
        this.scratch = ThreadLocal.withInitial(() -> new float[scratchLen]);
    }

    /** The widest {@code cols} of any tensor of {@code w}: the row-group scratch must hold four rows of it. */
    static int maxCols(ModelWeights w) {
        int m = Math.max(w.embed.cols, w.lmHead.cols);
        for (LayerWeights L : w.layers) {
            m = Math.max(m, L.maxCols());
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
            int layers = w.layers.length;
            java.util.List<QTensor> all = new java.util.ArrayList<>(layer);
            all.add(w.lmHead);
            for (int k = 0; k < all.size(); k++) {
                QTensor t = all.get(k);
                double weight = k < layer.size() ? layers : 1;
                any |= i8.useInt8(t);
                long[] best = time(i8, f32, t);
                t8 += weight * best[0];
                tf += weight * best[1];
            }
            if (any) {
                faster = t8 < tf;
                System.err.printf("int8 activations: auto -> %s (one token's matmuls: int8 %.1f ms, float %.1f ms)%n",
                        faster ? "on" : "off", t8 / 1e6, tf / 1e6);
            }
        }
        PROBED.put(w, faster);
        return faster;
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
     * Q4_0G, Q8_0G) and the K-quants of {@link KQuantKernels} (Q4_K, Q5_K, Q6_K); Q2_K stays on the float kernels.
     */
    boolean useInt8(QTensor wt) {
        return int8 && ((wt.type == QType.Q4_0 || wt.type == QType.Q8_0 || wt.type == QType.Q4_0G || wt.type == QType.Q8_0G)
                && wt.cols % QType.BLOCK == 0 || KQuantKernels.supports(wt));
    }

    /** Row {@code r} of {@code wt} (a {@link #useInt8} type) times the last {@link #quantizeInt8} activation from {@code xo}. */
    float int8Dot(QTensor wt, int r, short[] xq, float[] xs, int xo) {
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
