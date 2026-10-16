package org.robomiri.jllm.cpu;

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
 *       activations on and a Q4_0 / Q8_0 weight ({@link #useInt8}), one {@link Int8Kernels#dotRow} against the
 *       activation quantized once per call ({@link #quantizeInt8}).</li>
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
        for (ModelWeights.Layer L : w.layers) {
            for (QTensor t : new QTensor[]{L.wq, L.wk, L.wv, L.wo, L.wGateUp, L.wDown, L.expGateUp, L.expDown}) {
                m = Math.max(m, cols(t));
            }
            if (L.mla != null) {
                for (QTensor t : new QTensor[]{L.mla.qProj, L.mla.qA, L.mla.qB, L.mla.kvA, L.mla.kPe, L.mla.kvB,
                        L.mla.sharedGateUp, L.mla.sharedDown}) {
                    m = Math.max(m, cols(t));
                }
            }
            if (L.mamba != null) {
                for (QTensor t : new QTensor[]{L.mamba.inProj, L.mamba.xProj, L.mamba.outProj}) {
                    m = Math.max(m, cols(t));
                }
            }
        }
        return m;
    }

    private static int cols(QTensor t) {
        return t == null ? 0 : t.cols;
    }

    boolean int8() {
        return int8;
    }

    /** The int8 path applies to the formats {@link Int8Kernels} has kernels for (Q4_0, Q8_0); anything else stays float. */
    boolean useInt8(QTensor wt) {
        return int8 && (wt.type == QType.Q4_0 || wt.type == QType.Q8_0) && wt.cols % QType.BLOCK == 0;
    }

    /** Quantizes {@code in[off .. off + n)} into the int8 scratch (grown if needed) and returns it; scales are in {@link #xs()}. */
    short[] quantizeInt8(float[] in, int off, int n) {
        if (xq == null || xq.length < n) {
            xq = new short[n];
            xs = new float[n / QType.BLOCK];
        }
        Int8Kernels.quantize(in, off, n, xq, xs);
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

    /** The single-token kernel: {@code out[outOff + r] (+)= dot(W[r, :], in[inOff, :])} for {@code r < rows}, parallel over rows. */
    void mulOne(QTensor wt, int rows, float[] in, int inOff, float[] out, int outOff, boolean acc) {
        if (useInt8(wt)) {
            short[] xq = quantizeInt8(in, inOff, wt.cols);
            float[] xs = this.xs;
            pool.forRange(rows, 16, (r0, r1) -> {
                for (int r = r0; r < r1; r++) {
                    float v = Int8Kernels.dotRow(wt, r, xq, xs, 0);
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
