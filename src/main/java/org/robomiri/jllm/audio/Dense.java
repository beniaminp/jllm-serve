package org.robomiri.jllm.audio;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.cpu.ThreadPool;
import org.robomiri.jllm.safetensors.SafeTensors;

/**
 * Dense float matrices and the CPU kernels of the audio models: {@code out[t, r] (+)= bias[r] + dot(W[r, :], x[t, :])}
 * for a weight matrix {@code W} of {@code rows x cols} (the Hugging Face {@code [out, in]} layout), parallel over groups
 * of output rows. Four weight rows are multiplied with four input rows at once (16 accumulators, each weight and input
 * vector loaded once per four uses), the tile every dense GEMM of a Whisper layer reduces to. A weight kept in a
 * half-precision checkpoint is converted to float four rows at a time into a per-thread scratch (the conversion is
 * amortised over all the input rows), so big models can stay memory-mapped and f16 on disk.
 */
public final class Dense {

    private Dense() {
    }

    static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
    private static final int LANES = F.length();

    /** A weight matrix: resident floats, or a view of the checkpoint converted on the fly. */
    public static final class Mat {
        public final int rows;
        public final int cols;
        final float[] f;                       // row-major, or null
        final SafeTensors.Tensor t;            // the source when f == null
        public final float[] bias;             // null for none

        private Mat(int rows, int cols, float[] f, SafeTensors.Tensor t, float[] bias) {
            this.rows = rows;
            this.cols = cols;
            this.f = f;
            this.t = t;
            this.bias = bias;
        }

        public static Mat of(float[] w, int rows, int cols, float[] bias) {
            if (w.length != rows * cols) {
                throw new IllegalArgumentException("matrix " + rows + "x" + cols + " has " + w.length + " values");
            }
            return new Mat(rows, cols, w, null, bias);
        }

        /** @param resident copy the weights into a float[] (fast), else convert from the mapped checkpoint per use */
        public static Mat of(SafeTensors.Tensor t, int rows, int cols, float[] bias, boolean resident) {
            if (t.numel() != (long) rows * cols) {
                throw new IllegalArgumentException(t.name() + " has " + t.numel() + " values, expected " + rows + "x" + cols);
            }
            if (resident) {
                return new Mat(rows, cols, t.toFloatArray(), t, bias);
            }
            return new Mat(rows, cols, null, t, bias);
        }

        public Mat withBias(float[] b) {
            return new Mat(rows, cols, f, t, b);
        }

        /** The checkpoint tensor behind the matrix, or null for a matrix made from floats. */
        public SafeTensors.Tensor tensor() {
            return t;
        }

        /** Resident floats, or null when the matrix is converted from the checkpoint per use. */
        public float[] floats() {
            return f;
        }

        public long bytesResident() {
            return f != null ? 4L * f.length : 0;
        }

        /** Row {@code r} as floats into {@code dst} (for embeddings and tests). */
        public void row(int r, float[] dst, int off) {
            if (f != null) {
                System.arraycopy(f, r * cols, dst, off, cols);
            } else {
                t.toFloat((long) r * cols, dst, off, cols);
            }
        }
    }

    private static final ThreadLocal<float[]> SCRATCH = new ThreadLocal<>();

    private static float[] scratch(int n) {
        float[] s = SCRATCH.get();
        if (s == null || s.length < n) {
            s = new float[n];
            SCRATCH.set(s);
        }
        return s;
    }

    /**
     * {@code out[t * os + r] = (add ? out[...] : 0) + bias[r] + dot(W[r], x[t * xs ..])} for {@code t < n},
     * {@code r < rows}.
     */
    public static void mul(ThreadPool pool, Mat w, float[] x, int xo, int xs, int n, float[] out, int oo, int os, boolean add) {
        int groups = (w.rows + 3) / 4;
        // chunks of 4-row groups so a thread's weights stay in cache while it walks all the tokens
        int per = Math.max(1, Math.min(8, groups / Math.max(1, pool.threads() * 2)));
        int tasks = (groups + per - 1) / per;
        pool.forRangeFine(tasks, 1, (s, e) -> {
            for (int task = s; task < e; task++) {
                int g0 = task * per;
                int g1 = Math.min(groups, g0 + per);
                for (int g = g0; g < g1; g++) {
                    rowGroup(w, g * 4, x, xo, xs, n, out, oo, os, add);
                }
            }
        });
    }

    private static void rowGroup(Mat w, int r0, float[] x, int xo, int xs, int n, float[] out, int oo, int os, boolean add) {
        int nr = Math.min(4, w.rows - r0);
        int cols = w.cols;
        if (w.f != null) {
            rowGroup(w.f, r0 * cols, nr, cols, w.bias, r0, x, xo, xs, n, out, oo + r0, os, add);
        } else {
            float[] wa = scratch(4 * cols);
            w.t.toFloat((long) r0 * cols, wa, 0, nr * cols);
            rowGroup(wa, 0, nr, cols, w.bias, r0, x, xo, xs, n, out, oo + r0, os, add);
        }
    }

    /**
     * Single-threaded {@code out[oo + t * os + r] (+)= dot(w[wo + r * cols ..], x[xo + t * xs ..])} for the {@code rows}
     * rows of a plain float matrix at offset {@code wo} (a head's keys in attention, say); used inside parallel tasks.
     */
    static void block(float[] w, int wo, int rows, int cols, float[] x, int xo, int xs, int n, float[] out, int oo, int os,
                      boolean add) {
        for (int r0 = 0; r0 < rows; r0 += 4) {
            rowGroup(w, wo + r0 * cols, Math.min(4, rows - r0), cols, null, 0, x, xo, xs, n, out, oo + r0, os, add);
        }
    }

    /** {@code nr <= 4} rows of {@code wa} at {@code wo}; their outputs go to {@code out[oo + t * os + rr]}. */
    private static void rowGroup(float[] wa, int wo, int nr, int cols, float[] bias, int bo, float[] x, int xo, int xs, int n,
                                 float[] out, int oo, int os, boolean add) {
        int bound = F.loopBound(cols);
        int t = 0;
        if (nr == 4) {
            for (; t + 4 <= n; t += 4) {
                int x0 = xo + t * xs;
                FloatVector a00 = FloatVector.zero(F), a01 = a00, a02 = a00, a03 = a00;
                FloatVector a10 = a00, a11 = a00, a12 = a00, a13 = a00;
                FloatVector a20 = a00, a21 = a00, a22 = a00, a23 = a00;
                FloatVector a30 = a00, a31 = a00, a32 = a00, a33 = a00;
                int k = 0;
                for (; k < bound; k += LANES) {
                    FloatVector w0 = FloatVector.fromArray(F, wa, wo + k);
                    FloatVector w1 = FloatVector.fromArray(F, wa, wo + cols + k);
                    FloatVector w2 = FloatVector.fromArray(F, wa, wo + 2 * cols + k);
                    FloatVector w3 = FloatVector.fromArray(F, wa, wo + 3 * cols + k);
                    FloatVector v0 = FloatVector.fromArray(F, x, x0 + k);
                    a00 = w0.fma(v0, a00);
                    a10 = w1.fma(v0, a10);
                    a20 = w2.fma(v0, a20);
                    a30 = w3.fma(v0, a30);
                    FloatVector v1 = FloatVector.fromArray(F, x, x0 + xs + k);
                    a01 = w0.fma(v1, a01);
                    a11 = w1.fma(v1, a11);
                    a21 = w2.fma(v1, a21);
                    a31 = w3.fma(v1, a31);
                    FloatVector v2 = FloatVector.fromArray(F, x, x0 + 2 * xs + k);
                    a02 = w0.fma(v2, a02);
                    a12 = w1.fma(v2, a12);
                    a22 = w2.fma(v2, a22);
                    a32 = w3.fma(v2, a32);
                    FloatVector v3 = FloatVector.fromArray(F, x, x0 + 3 * xs + k);
                    a03 = w0.fma(v3, a03);
                    a13 = w1.fma(v3, a13);
                    a23 = w2.fma(v3, a23);
                    a33 = w3.fma(v3, a33);
                }
                float[] r = {
                        a00.reduceLanes(VectorOperators.ADD), a10.reduceLanes(VectorOperators.ADD),
                        a20.reduceLanes(VectorOperators.ADD), a30.reduceLanes(VectorOperators.ADD),
                        a01.reduceLanes(VectorOperators.ADD), a11.reduceLanes(VectorOperators.ADD),
                        a21.reduceLanes(VectorOperators.ADD), a31.reduceLanes(VectorOperators.ADD),
                        a02.reduceLanes(VectorOperators.ADD), a12.reduceLanes(VectorOperators.ADD),
                        a22.reduceLanes(VectorOperators.ADD), a32.reduceLanes(VectorOperators.ADD),
                        a03.reduceLanes(VectorOperators.ADD), a13.reduceLanes(VectorOperators.ADD),
                        a23.reduceLanes(VectorOperators.ADD), a33.reduceLanes(VectorOperators.ADD)};
                for (; k < cols; k++) {
                    for (int tt = 0; tt < 4; tt++) {
                        float xv = x[x0 + tt * xs + k];
                        for (int rr = 0; rr < 4; rr++) {
                            r[tt * 4 + rr] += wa[wo + rr * cols + k] * xv;
                        }
                    }
                }
                for (int tt = 0; tt < 4; tt++) {
                    int o = oo + (t + tt) * os;
                    for (int rr = 0; rr < 4; rr++) {
                        float v = r[tt * 4 + rr] + (bias != null ? bias[bo + rr] : 0f);
                        out[o + rr] = add ? out[o + rr] + v : v;
                    }
                }
            }
        }
        for (; t < n; t++) {          // remaining tokens, or a partial row group
            int x0 = xo + t * xs;
            for (int rr = 0; rr < nr; rr++) {
                FloatVector acc = FloatVector.zero(F);
                int k = 0;
                int wbase = wo + rr * cols;
                for (; k < bound; k += LANES) {
                    acc = FloatVector.fromArray(F, wa, wbase + k).fma(FloatVector.fromArray(F, x, x0 + k), acc);
                }
                float s = acc.reduceLanes(VectorOperators.ADD);
                for (; k < cols; k++) {
                    s += wa[wbase + k] * x[x0 + k];
                }
                if (bias != null) {
                    s += bias[bo + rr];
                }
                int o = oo + t * os + rr;
                out[o] = add ? out[o] + s : s;
            }
        }
    }

    // ------------------------------------------------------------------ element-wise pieces

    /** In-place layer norm of each of the {@code n} rows of length {@code d} of {@code x}. */
    public static void layerNorm(float[] x, int xo, int n, int d, float[] gamma, float[] beta, float eps, float[] out, int oo) {
        for (int t = 0; t < n; t++) {
            int b = xo + t * d;
            double mean = 0;
            for (int i = 0; i < d; i++) {
                mean += x[b + i];
            }
            mean /= d;
            double var = 0;
            for (int i = 0; i < d; i++) {
                double c = x[b + i] - mean;
                var += c * c;
            }
            var /= d;
            float inv = (float) (1.0 / Math.sqrt(var + eps));
            float m = (float) mean;
            int ob = oo + t * d;
            for (int i = 0; i < d; i++) {
                out[ob + i] = (x[b + i] - m) * inv * gamma[i] + (beta != null ? beta[i] : 0f);
            }
        }
    }

    /** Exact (erf) GELU, in place over {@code n} values; the erf is the float rational approximation of Eigen / XLA. */
    public static void gelu(float[] x, int off, int n) {
        int bound = F.loopBound(n);
        FloatVector half = FloatVector.broadcast(F, 0.5f);
        FloatVector one = FloatVector.broadcast(F, 1f);
        FloatVector invSqrt2 = FloatVector.broadcast(F, 0.70710678118654752f);
        int i = 0;
        for (; i < bound; i += LANES) {
            FloatVector v = FloatVector.fromArray(F, x, off + i);
            half.mul(v).mul(one.add(erf(v.mul(invSqrt2)))).intoArray(x, off + i);
        }
        for (; i < n; i++) {
            float v = x[off + i];
            x[off + i] = 0.5f * v * (1f + erf(v * 0.70710678118654752f));
        }
    }

    static FloatVector erf(FloatVector x) {
        x = x.max(-4f).min(4f);
        FloatVector x2 = x.mul(x);
        FloatVector p = FloatVector.broadcast(F, -2.72614225801306e-10f);
        p = p.fma(x2, FloatVector.broadcast(F, 2.77068142495902e-08f));
        p = p.fma(x2, FloatVector.broadcast(F, -2.10102402082508e-06f));
        p = p.fma(x2, FloatVector.broadcast(F, -5.69250639462346e-05f));
        p = p.fma(x2, FloatVector.broadcast(F, -7.34990630326855e-04f));
        p = p.fma(x2, FloatVector.broadcast(F, -2.95459980854025e-03f));
        p = p.fma(x2, FloatVector.broadcast(F, -1.60960333262415e-02f)).mul(x);
        FloatVector q = FloatVector.broadcast(F, -1.45660718464996e-05f);
        q = q.fma(x2, FloatVector.broadcast(F, -2.13374055278905e-04f));
        q = q.fma(x2, FloatVector.broadcast(F, -1.68282697438203e-03f));
        q = q.fma(x2, FloatVector.broadcast(F, -7.37332916720468e-03f));
        q = q.fma(x2, FloatVector.broadcast(F, -1.42647390514189e-02f));
        return p.div(q);
    }

    static float erf(float x) {
        x = Math.max(-4f, Math.min(4f, x));
        float x2 = x * x;
        float p = -2.72614225801306e-10f;
        p = p * x2 + 2.77068142495902e-08f;
        p = p * x2 - 2.10102402082508e-06f;
        p = p * x2 - 5.69250639462346e-05f;
        p = p * x2 - 7.34990630326855e-04f;
        p = p * x2 - 2.95459980854025e-03f;
        p = (p * x2 - 1.60960333262415e-02f) * x;
        float q = -1.45660718464996e-05f;
        q = q * x2 - 2.13374055278905e-04f;
        q = q * x2 - 1.68282697438203e-03f;
        q = q * x2 - 7.37332916720468e-03f;
        q = q * x2 - 1.42647390514189e-02f;
        return p / q;
    }

    /** {@code y[i] += x[i]}. */
    public static void add(float[] y, int yo, float[] x, int xo, int n) {
        int bound = F.loopBound(n);
        int i = 0;
        for (; i < bound; i += LANES) {
            FloatVector.fromArray(F, y, yo + i).add(FloatVector.fromArray(F, x, xo + i)).intoArray(y, yo + i);
        }
        for (; i < n; i++) {
            y[yo + i] += x[xo + i];
        }
    }

    static float dot(float[] a, int ao, float[] b, int bo, int n) {
        int bound = F.loopBound(n);
        FloatVector acc = FloatVector.zero(F);
        int i = 0;
        for (; i < bound; i += LANES) {
            acc = FloatVector.fromArray(F, a, ao + i).fma(FloatVector.fromArray(F, b, bo + i), acc);
        }
        float s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            s += a[ao + i] * b[bo + i];
        }
        return s;
    }

    /** {@code y[0..n) += a * x[0..n)}. */
    static void axpy(float a, float[] x, int xo, float[] y, int yo, int n) {
        int bound = F.loopBound(n);
        FloatVector av = FloatVector.broadcast(F, a);
        int i = 0;
        for (; i < bound; i += LANES) {
            av.fma(FloatVector.fromArray(F, x, xo + i), FloatVector.fromArray(F, y, yo + i)).intoArray(y, yo + i);
        }
        for (; i < n; i++) {
            y[yo + i] += a * x[xo + i];
        }
    }
}
