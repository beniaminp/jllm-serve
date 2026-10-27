package org.robomiri.jllm.vision;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.cpu.ThreadPool;
import org.robomiri.jllm.cpu.VectorKernels;

/**
 * Float kernels of the CPU vision encoder: register-tiled matrix products (4 weight rows x 4 activation rows per
 * tile, the same shape as the text engine's float path), layer / RMS norms, activations, a softmax attention over
 * blocks of keys, and the rotary embeddings of the Qwen-VL encoders. All of them work on row-major {@code float[]}
 * arrays and split their work over a {@link ThreadPool}.
 *
 * <p>Row lengths that feed the matrix kernels are multiples of {@link #PAD} (16, so any vector width fits); callers
 * pad with zeros where a dimension is not.
 */
final class VMath {

    static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
    static final int FL = F.length();
    static final int PAD = 16;

    private VMath() {
    }

    static int pad(int n) {
        return (n + PAD - 1) / PAD * PAD;
    }

    // ------------------------------------------------------------------ matrix products

    /**
     * {@code out[i * ldo + r] = dot(w[wo + r * k ..], x[xo + i * k ..])} for {@code r < nw} (a multiple of 4) and
     * {@code i < nx}: a tile of a product of two row-major matrices that share the inner dimension {@code k} (a
     * multiple of {@link #PAD}).
     */
    static void tile(float[] w, int wo, int nw, int k, float[] x, int xo, int nx, float[] out, int oo, int ldo) {
        int i = 0;
        for (; i + 4 <= nx; i += 4) {
            for (int r = 0; r < nw; r += 4) {
                dot4x4(w, wo + r * k, k, x, xo + i * k, xo + (i + 1) * k, xo + (i + 2) * k, xo + (i + 3) * k, out,
                        oo + i * ldo + r, oo + (i + 1) * ldo + r, oo + (i + 2) * ldo + r, oo + (i + 3) * ldo + r);
            }
        }
        for (; i < nx; i++) {
            for (int r = 0; r < nw; r += 4) {
                dot4(w, wo + r * k, k, x, xo + i * k, out, oo + i * ldo + r);
            }
        }
    }

    private static void dot4(float[] w, int wo, int n, float[] x, int xo, float[] out, int oo) {
        FloatVector s0 = FloatVector.zero(F);
        FloatVector s1 = FloatVector.zero(F);
        FloatVector s2 = FloatVector.zero(F);
        FloatVector s3 = FloatVector.zero(F);
        for (int i = 0; i < n; i += FL) {
            FloatVector xv = FloatVector.fromArray(F, x, xo + i);
            s0 = FloatVector.fromArray(F, w, wo + i).fma(xv, s0);
            s1 = FloatVector.fromArray(F, w, wo + n + i).fma(xv, s1);
            s2 = FloatVector.fromArray(F, w, wo + 2 * n + i).fma(xv, s2);
            s3 = FloatVector.fromArray(F, w, wo + 3 * n + i).fma(xv, s3);
        }
        if (FL == 4) {
            out[oo] = VectorKernels.laneSum(s0);
            out[oo + 1] = VectorKernels.laneSum(s1);
            out[oo + 2] = VectorKernels.laneSum(s2);
            out[oo + 3] = VectorKernels.laneSum(s3);
            return;
        }
        // 8 / 16 lanes: VectorKernels.laneSum's butterfly written out on each accumulator, stored, lane 0 read (the same bits)
        s0 = s0.add(s0.rearrange(VectorKernels.SWH));
        s0 = s0.add(s0.rearrange(VectorKernels.SWQ));
        s0 = s0.add(s0.rearrange(VectorKernels.SWE));
        s1 = s1.add(s1.rearrange(VectorKernels.SWH));
        s1 = s1.add(s1.rearrange(VectorKernels.SWQ));
        s1 = s1.add(s1.rearrange(VectorKernels.SWE));
        s2 = s2.add(s2.rearrange(VectorKernels.SWH));
        s2 = s2.add(s2.rearrange(VectorKernels.SWQ));
        s2 = s2.add(s2.rearrange(VectorKernels.SWE));
        s3 = s3.add(s3.rearrange(VectorKernels.SWH));
        s3 = s3.add(s3.rearrange(VectorKernels.SWQ));
        s3 = s3.add(s3.rearrange(VectorKernels.SWE));
        if (FL == 16) { // the fourth step of 16 lanes
            s0 = s0.add(s0.rearrange(VectorKernels.SW1));
            s1 = s1.add(s1.rearrange(VectorKernels.SW1));
            s2 = s2.add(s2.rearrange(VectorKernels.SW1));
            s3 = s3.add(s3.rearrange(VectorKernels.SW1));
        }
        float[] t = VectorKernels.tile();
        s0.intoArray(t, 0);
        s1.intoArray(t, FL);
        s2.intoArray(t, 2 * FL);
        s3.intoArray(t, 3 * FL);
        for (int j = 0; j < 4; j++) {
            out[oo + j] = t[j * FL];
        }
    }

    private static void dot4x4(float[] w, int wo, int n, float[] x, int xa, int xb, int xc, int xd,
                               float[] out, int oa, int ob, int oc, int od) {
        FloatVector a0 = FloatVector.zero(F), a1 = FloatVector.zero(F), a2 = FloatVector.zero(F), a3 = FloatVector.zero(F);
        FloatVector b0 = FloatVector.zero(F), b1 = FloatVector.zero(F), b2 = FloatVector.zero(F), b3 = FloatVector.zero(F);
        FloatVector c0 = FloatVector.zero(F), c1 = FloatVector.zero(F), c2 = FloatVector.zero(F), c3 = FloatVector.zero(F);
        FloatVector d0 = FloatVector.zero(F), d1 = FloatVector.zero(F), d2 = FloatVector.zero(F), d3 = FloatVector.zero(F);
        for (int i = 0; i < n; i += FL) {
            FloatVector w0 = FloatVector.fromArray(F, w, wo + i);
            FloatVector w1 = FloatVector.fromArray(F, w, wo + n + i);
            FloatVector w2 = FloatVector.fromArray(F, w, wo + 2 * n + i);
            FloatVector w3 = FloatVector.fromArray(F, w, wo + 3 * n + i);
            FloatVector xv = FloatVector.fromArray(F, x, xa + i);
            a0 = w0.fma(xv, a0);
            a1 = w1.fma(xv, a1);
            a2 = w2.fma(xv, a2);
            a3 = w3.fma(xv, a3);
            xv = FloatVector.fromArray(F, x, xb + i);
            b0 = w0.fma(xv, b0);
            b1 = w1.fma(xv, b1);
            b2 = w2.fma(xv, b2);
            b3 = w3.fma(xv, b3);
            xv = FloatVector.fromArray(F, x, xc + i);
            c0 = w0.fma(xv, c0);
            c1 = w1.fma(xv, c1);
            c2 = w2.fma(xv, c2);
            c3 = w3.fma(xv, c3);
            xv = FloatVector.fromArray(F, x, xd + i);
            d0 = w0.fma(xv, d0);
            d1 = w1.fma(xv, d1);
            d2 = w2.fma(xv, d2);
            d3 = w3.fma(xv, d3);
        }
        if (FL == 4) {
            store4(out, oa, a0, a1, a2, a3);
            store4(out, ob, b0, b1, b2, b3);
            store4(out, oc, c0, c1, c2, c3);
            store4(out, od, d0, d1, d2, d3);
            return;
        }
        // 8 / 16 lanes: VectorKernels.laneSum's butterfly written out on each accumulator, then stored and lane 0 read (the
        // same bits). Passed to store4 instead, C2 inlined only the first call (NodeCountInliningCutoff) and boxed the others.
        a0 = a0.add(a0.rearrange(VectorKernels.SWH));
        a0 = a0.add(a0.rearrange(VectorKernels.SWQ));
        a0 = a0.add(a0.rearrange(VectorKernels.SWE));
        a1 = a1.add(a1.rearrange(VectorKernels.SWH));
        a1 = a1.add(a1.rearrange(VectorKernels.SWQ));
        a1 = a1.add(a1.rearrange(VectorKernels.SWE));
        a2 = a2.add(a2.rearrange(VectorKernels.SWH));
        a2 = a2.add(a2.rearrange(VectorKernels.SWQ));
        a2 = a2.add(a2.rearrange(VectorKernels.SWE));
        a3 = a3.add(a3.rearrange(VectorKernels.SWH));
        a3 = a3.add(a3.rearrange(VectorKernels.SWQ));
        a3 = a3.add(a3.rearrange(VectorKernels.SWE));
        b0 = b0.add(b0.rearrange(VectorKernels.SWH));
        b0 = b0.add(b0.rearrange(VectorKernels.SWQ));
        b0 = b0.add(b0.rearrange(VectorKernels.SWE));
        b1 = b1.add(b1.rearrange(VectorKernels.SWH));
        b1 = b1.add(b1.rearrange(VectorKernels.SWQ));
        b1 = b1.add(b1.rearrange(VectorKernels.SWE));
        b2 = b2.add(b2.rearrange(VectorKernels.SWH));
        b2 = b2.add(b2.rearrange(VectorKernels.SWQ));
        b2 = b2.add(b2.rearrange(VectorKernels.SWE));
        b3 = b3.add(b3.rearrange(VectorKernels.SWH));
        b3 = b3.add(b3.rearrange(VectorKernels.SWQ));
        b3 = b3.add(b3.rearrange(VectorKernels.SWE));
        c0 = c0.add(c0.rearrange(VectorKernels.SWH));
        c0 = c0.add(c0.rearrange(VectorKernels.SWQ));
        c0 = c0.add(c0.rearrange(VectorKernels.SWE));
        c1 = c1.add(c1.rearrange(VectorKernels.SWH));
        c1 = c1.add(c1.rearrange(VectorKernels.SWQ));
        c1 = c1.add(c1.rearrange(VectorKernels.SWE));
        c2 = c2.add(c2.rearrange(VectorKernels.SWH));
        c2 = c2.add(c2.rearrange(VectorKernels.SWQ));
        c2 = c2.add(c2.rearrange(VectorKernels.SWE));
        c3 = c3.add(c3.rearrange(VectorKernels.SWH));
        c3 = c3.add(c3.rearrange(VectorKernels.SWQ));
        c3 = c3.add(c3.rearrange(VectorKernels.SWE));
        d0 = d0.add(d0.rearrange(VectorKernels.SWH));
        d0 = d0.add(d0.rearrange(VectorKernels.SWQ));
        d0 = d0.add(d0.rearrange(VectorKernels.SWE));
        d1 = d1.add(d1.rearrange(VectorKernels.SWH));
        d1 = d1.add(d1.rearrange(VectorKernels.SWQ));
        d1 = d1.add(d1.rearrange(VectorKernels.SWE));
        d2 = d2.add(d2.rearrange(VectorKernels.SWH));
        d2 = d2.add(d2.rearrange(VectorKernels.SWQ));
        d2 = d2.add(d2.rearrange(VectorKernels.SWE));
        d3 = d3.add(d3.rearrange(VectorKernels.SWH));
        d3 = d3.add(d3.rearrange(VectorKernels.SWQ));
        d3 = d3.add(d3.rearrange(VectorKernels.SWE));
        if (FL == 16) { // the fourth step of 16 lanes
            a0 = a0.add(a0.rearrange(VectorKernels.SW1));
            a1 = a1.add(a1.rearrange(VectorKernels.SW1));
            a2 = a2.add(a2.rearrange(VectorKernels.SW1));
            a3 = a3.add(a3.rearrange(VectorKernels.SW1));
            b0 = b0.add(b0.rearrange(VectorKernels.SW1));
            b1 = b1.add(b1.rearrange(VectorKernels.SW1));
            b2 = b2.add(b2.rearrange(VectorKernels.SW1));
            b3 = b3.add(b3.rearrange(VectorKernels.SW1));
            c0 = c0.add(c0.rearrange(VectorKernels.SW1));
            c1 = c1.add(c1.rearrange(VectorKernels.SW1));
            c2 = c2.add(c2.rearrange(VectorKernels.SW1));
            c3 = c3.add(c3.rearrange(VectorKernels.SW1));
            d0 = d0.add(d0.rearrange(VectorKernels.SW1));
            d1 = d1.add(d1.rearrange(VectorKernels.SW1));
            d2 = d2.add(d2.rearrange(VectorKernels.SW1));
            d3 = d3.add(d3.rearrange(VectorKernels.SW1));
        }
        float[] t = VectorKernels.tile();
        a0.intoArray(t, 0);
        a1.intoArray(t, FL);
        a2.intoArray(t, 2 * FL);
        a3.intoArray(t, 3 * FL);
        b0.intoArray(t, 4 * FL);
        b1.intoArray(t, 5 * FL);
        b2.intoArray(t, 6 * FL);
        b3.intoArray(t, 7 * FL);
        c0.intoArray(t, 8 * FL);
        c1.intoArray(t, 9 * FL);
        c2.intoArray(t, 10 * FL);
        c3.intoArray(t, 11 * FL);
        d0.intoArray(t, 12 * FL);
        d1.intoArray(t, 13 * FL);
        d2.intoArray(t, 14 * FL);
        d3.intoArray(t, 15 * FL);
        int[] o = {oa, ob, oc, od};
        for (int g = 0; g < 4; g++) {
            for (int j = 0; j < 4; j++) {
                out[o[g] + j] = t[(4 * g + j) * FL];
            }
        }
    }

    private static void store4(float[] out, int oo, FloatVector s0, FloatVector s1, FloatVector s2, FloatVector s3) {
        out[oo] = VectorKernels.laneSum(s0);
        out[oo + 1] = VectorKernels.laneSum(s1);
        out[oo + 2] = VectorKernels.laneSum(s2);
        out[oo + 3] = VectorKernels.laneSum(s3);
    }

    /** Dequantizes one block of weight rows into a float array with row stride {@code kPad}. */
    @FunctionalInterface
    interface WeightRows {
        /** Rows {@code [r0, r0 + n)} into {@code dst[0 ..)}, row stride {@code kPad}, zero past {@code cols}. */
        void rows(int r0, int n, float[] dst);
    }

    /**
     * {@code y[t * ldy + r] = dot(W[r], x[t]) + bias[r]} for {@code t < nTok}, {@code r < nOut}. {@code x} has row
     * stride {@code kPad}. Weight rows are dequantized once, in blocks of 4, by the task that owns them.
     */
    static void gemm(ThreadPool pool, WeightRows w, int nOut, int kPad, float[] bias, float[] x, int nTok, float[] y,
                     int ldy) {
        int groups = (nOut + 3) / 4;
        pool.forRange(groups, 1, (g0, g1) -> {
            float[] wb = new float[4 * kPad * 8];
            float[] tmp = new float[nTok * 4 * 2];
            for (int g = g0; g < g1; ) {
                int gn = Math.min(8, g1 - g);
                int r0 = g * 4;
                int rows = Math.min(nOut - r0, gn * 4);
                w.rows(r0, rows, wb);
                int rp = (rows + 3) / 4 * 4;
                for (int j = rows; j < rp; j++) { // the last block may be short: zero rows
                    java.util.Arrays.fill(wb, j * kPad, (j + 1) * kPad, 0f);
                }
                if (tmp.length < nTok * rp) {
                    tmp = new float[nTok * rp];
                }
                tile(wb, 0, rp, kPad, x, 0, nTok, tmp, 0, rp);
                for (int t = 0; t < nTok; t++) {
                    int yo = t * ldy + r0;
                    int to = t * rp;
                    if (bias != null) {
                        for (int j = 0; j < rows; j++) {
                            y[yo + j] = tmp[to + j] + bias[r0 + j];
                        }
                    } else {
                        System.arraycopy(tmp, to, y, yo, rows);
                    }
                }
                g += gn;
            }
        });
    }

    // ------------------------------------------------------------------ norms and activations

    /** LayerNorm over rows of width {@code n}: {@code (x - mean) / sqrt(var + eps) * w + b} (b may be null). */
    static void layerNorm(ThreadPool pool, float[] x, int rows, int n, float[] w, float[] b, float eps, float[] out) {
        pool.forRange(rows, 8, (r0, r1) -> {
            for (int r = r0; r < r1; r++) {
                int o = r * n;
                double sum = 0;
                for (int i = 0; i < n; i++) {
                    sum += x[o + i];
                }
                float mean = (float) (sum / n);
                double sq = 0;
                for (int i = 0; i < n; i++) {
                    float d = x[o + i] - mean;
                    sq += d * d;
                }
                float scale = (float) (1.0 / Math.sqrt(sq / n + eps));
                for (int i = 0; i < n; i++) {
                    float v = (x[o + i] - mean) * scale;
                    if (w != null) {
                        v *= w[i];
                    }
                    if (b != null) {
                        v += b[i];
                    }
                    out[o + i] = v;
                }
            }
        });
    }

    /** RMSNorm over rows of width {@code n}: {@code x / sqrt(mean(x^2) + eps) * w}. */
    static void rmsNorm(ThreadPool pool, float[] x, int rows, int n, float[] w, float eps, float[] out) {
        pool.forRange(rows, 8, (r0, r1) -> {
            for (int r = r0; r < r1; r++) {
                int o = r * n;
                double sq = 0;
                for (int i = 0; i < n; i++) {
                    sq += (double) x[o + i] * x[o + i];
                }
                float scale = (float) (1.0 / Math.sqrt(sq / n + eps));
                for (int i = 0; i < n; i++) {
                    float v = x[o + i] * scale;
                    out[o + i] = w != null ? v * w[i] : v;
                }
            }
        });
    }

    static float geluTanh(float x) {
        final float sqrt2OverPi = 0.79788456080286535588f;
        return 0.5f * x * (1.0f + org.robomiri.jllm.cpu.VectorKernels.tanhFast(sqrt2OverPi * x * (1.0f + 0.044715f * x * x)));
    }

    static float geluQuick(float x) {
        return x * (1.0f / (1.0f + org.robomiri.jllm.cpu.VectorKernels.expFast(-1.702f * x)));
    }

    static float silu(float x) {
        return x / (1.0f + org.robomiri.jllm.cpu.VectorKernels.expFast(-x));
    }

    /** {@code y = act(x)} elementwise; {@code kind}: 0 gelu (tanh form), 1 silu, 2 gelu_quick, 3 gelu (erf form). */
    static void activate(ThreadPool pool, float[] x, int n, int kind) {
        pool.forRange(n, 4096, (i0, i1) -> {
            switch (kind) {
                case 0 -> {
                    for (int i = i0; i < i1; i++) {
                        x[i] = geluTanh(x[i]);
                    }
                }
                case 1 -> {
                    for (int i = i0; i < i1; i++) {
                        x[i] = silu(x[i]);
                    }
                }
                case 2 -> {
                    for (int i = i0; i < i1; i++) {
                        x[i] = geluQuick(x[i]);
                    }
                }
                default -> {
                    for (int i = i0; i < i1; i++) {
                        x[i] = (float) (0.5 * x[i] * (1.0 + erf(x[i] / Math.sqrt(2.0))));
                    }
                }
            }
        });
    }

    /** {@code gate[i] = act(gate[i]) * up[i]}. */
    static void gated(ThreadPool pool, float[] gate, float[] up, int n, int kind) {
        activate(pool, gate, n, kind);
        pool.forRange(n, 4096, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                gate[i] *= up[i];
            }
        });
    }

    static void addTo(ThreadPool pool, float[] dst, float[] src, int n) {
        pool.forRange(n, 4096, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                dst[i] += src[i];
            }
        });
    }

    /** Abramowitz-Stegun style erf accurate to ~1e-7 (enough for a float activation). */
    static double erf(double x) {
        double t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
        double y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
                * Math.exp(-x * x);
        return x >= 0 ? y : -y;
    }

    // ------------------------------------------------------------------ attention

    /**
     * Bidirectional multi-head attention: each query row attends to the keys of its own segment (a window; one
     * segment spanning every row is full attention). {@code q, k, v} are {@code [t][heads * dh]}; the result goes to
     * {@code out} with the same layout. Scores are scaled by {@code scale}, softmax in float.
     *
     * <p>Per head and segment the keys are copied once into a padded row-major block and the values transposed
     * ({@code [dh][keys]}), so both products (scores = Q K^T, output = P V) are row-by-row dot products of the
     * register-tiled kernel; queries are processed in blocks of 64 rows.
     *
     * @param segStart first row of each segment; segment {@code s} is {@code [segStart[s], segStart[s+1])}
     */
    static void attention(ThreadPool pool, float[] q, float[] k, float[] v, int t, int heads, int dh, float scale,
                          int[] segStart, int nSeg, float[] out) {
        final int dhp = pad(dh);
        final int dh4 = (dh + 3) / 4 * 4;
        final int dim = heads * dh;
        float[][] qh = new float[heads][];
        float[][][] ks = new float[heads][nSeg][];
        float[][][] vs = new float[heads][nSeg][];
        pool.forRange(heads * nSeg, 1, (a, b) -> {
            for (int hs = a; hs < b; hs++) {
                int h = hs / nSeg;
                int s = hs % nSeg;
                int s0 = segStart[s];
                int s1 = segStart[s + 1];
                int nk = s1 - s0;
                int nk4 = (nk + 3) / 4 * 4;
                int nkp = pad(nk);
                float[] kk = new float[nk4 * dhp];
                float[] vv = new float[dh4 * nkp];
                for (int r = 0; r < nk; r++) {
                    System.arraycopy(k, (s0 + r) * dim + h * dh, kk, r * dhp, dh);
                    for (int d = 0; d < dh; d++) {
                        vv[d * nkp + r] = v[(s0 + r) * dim + h * dh + d];
                    }
                }
                ks[h][s] = kk;
                vs[h][s] = vv;
            }
        });
        pool.forRange(heads, 1, (h0, h1) -> {
            for (int h = h0; h < h1; h++) {
                float[] qq = new float[t * dhp];
                for (int r = 0; r < t; r++) {
                    System.arraycopy(q, r * dim + h * dh, qq, r * dhp, dh);
                }
                qh[h] = qq;
            }
        });
        final int blockQ = 64;
        java.util.ArrayList<int[]> tasks = new java.util.ArrayList<>();
        for (int s = 0; s < nSeg; s++) {
            for (int b = segStart[s]; b < segStart[s + 1]; b += blockQ) {
                for (int h = 0; h < heads; h++) {
                    tasks.add(new int[]{h, s, b, Math.min(segStart[s + 1], b + blockQ)});
                }
            }
        }
        pool.forRangeFine(tasks.size(), 1, (a, b) -> {
            for (int ti = a; ti < b; ti++) {
                int[] tk = tasks.get(ti);
                int h = tk[0];
                int s = tk[1];
                int q0 = tk[2];
                int nq = tk[3] - q0;
                int nk = segStart[s + 1] - segStart[s];
                int nk4 = (nk + 3) / 4 * 4;
                int nkp = pad(nk);
                float[] scores = new float[nq * nk4];
                tile(ks[h][s], 0, nk4, dhp, qh[h], q0 * dhp, nq, scores, 0, nk4);
                float[] p = new float[nq * nkp]; // softmax rows, zero past nk
                for (int i = 0; i < nq; i++) {
                    int so = i * nk4;
                    float mx = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < nk; j++) {
                        float sv = scores[so + j] * scale;
                        scores[so + j] = sv;
                        if (sv > mx) {
                            mx = sv;
                        }
                    }
                    float sum = org.robomiri.jllm.cpu.VectorKernels.softmaxExp(scores, so, nk, mx);
                    float inv = 1f / sum;
                    for (int j = 0; j < nk; j++) {
                        p[i * nkp + j] = scores[so + j] * inv;
                    }
                }
                float[] res = new float[nq * dh4];
                tile(vs[h][s], 0, dh4, nkp, p, 0, nq, res, 0, dh4);
                for (int i = 0; i < nq; i++) {
                    System.arraycopy(res, i * dh4, out, (q0 + i) * dim + h * dh, dh);
                }
            }
        });
    }

    // ------------------------------------------------------------------ rotary embeddings

    /**
     * 2-D rotary embedding of a vision transformer in place, NEOX pairing (the Qwen-VL encoders): per head of size
     * {@code dh}, rotation pair {@code j < dh/2} is {@code (x[j], x[j + dh/2])}; the first half of the pairs
     * ({@code j < dh/4}) turns by {@code posA * theta^(-2j / (dh/2))}, the second half by
     * {@code posB * theta^(-2(j - dh/4) / (dh/2))}.
     */
    static void rope2d(ThreadPool pool, float[] x, int rows, int heads, int dh, int[] posA, int[] posB, float theta) {
        int half = dh / 2;
        int quarter = dh / 4;
        double[] inv = new double[quarter];
        for (int j = 0; j < quarter; j++) {
            inv[j] = 1.0 / Math.pow(theta, 2.0 * j / half);
        }
        int dim = heads * dh;
        pool.forRange(rows, 8, (r0, r1) -> {
            float[] cs = new float[half];
            float[] sn = new float[half];
            for (int r = r0; r < r1; r++) {
                for (int j = 0; j < quarter; j++) {
                    double a = posA[r] * inv[j];
                    double b = posB[r] * inv[j];
                    cs[j] = (float) Math.cos(a);
                    sn[j] = (float) Math.sin(a);
                    cs[quarter + j] = (float) Math.cos(b);
                    sn[quarter + j] = (float) Math.sin(b);
                }
                for (int h = 0; h < heads; h++) {
                    int o = r * dim + h * dh;
                    for (int j = 0; j < half; j++) {
                        float x0 = x[o + j];
                        float x1 = x[o + j + half];
                        x[o + j] = x0 * cs[j] - x1 * sn[j];
                        x[o + j + half] = x0 * sn[j] + x1 * cs[j];
                    }
                }
            }
        });
    }
}
