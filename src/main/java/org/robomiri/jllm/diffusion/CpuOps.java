package org.robomiri.jllm.diffusion;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.stream.IntStream;

/**
 * {@link Ops} on the CPU in float32 (Vector API): the correctness reference of the GPU path and a fallback without Metal.
 *
 * <p>Linear layers and convolutions are matrix products {@code C[M,N] = A[M,K] · W[K,N]}: a convolution gathers its
 * input patches (im2col) tile by tile into a per-thread buffer, so the same register-tiled micro-kernel (4 rows × 4
 * vectors of float accumulators) serves both. Weights are packed once at load into panels of {@code NR} columns:
 * {@code panel[p][k][j] = W[k][p·NR + j]}, so the kernel streams them linearly. They live outside the Java heap.
 */
public final class CpuOps implements Ops {

    static final VectorSpecies<Float> SP = FloatVector.SPECIES_PREFERRED;
    static final int L = SP.length();
    /** Columns of one packed weight panel: four vectors. */
    static final int NR = 4 * L;
    private static final ByteOrder NATIVE = ByteOrder.nativeOrder();
    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT.withOrder(NATIVE);
    /** Rows per task of the matrix product. */
    private static final int MB = 32;

    /** A float32 activation or vector. */
    public static final class T extends Tn {
        public final float[] a;

        T(float[] a, int... shape) {
            super(shape);
            this.a = a;
        }
    }

    /** A packed weight: {@code K} inputs, {@code N} outputs. {@link Tn#shape} is the checkpoint's [N, K] or [Co, kh, kw, Ci]. */
    static final class W extends Tn {
        final MemorySegment seg;
        final int k, n, npad;

        W(MemorySegment seg, int k, int n, int[] shape) {
            super(shape);
            this.seg = seg;
            this.k = k;
            this.n = n;
            this.npad = (n + NR - 1) / NR * NR;
        }
    }

    private final Arena arena = Arena.ofShared();
    private final ThreadLocal<float[]> scratch = ThreadLocal.withInitial(() -> new float[0]);

    @Override
    public String name() {
        return "cpu";
    }

    @Override
    public boolean half() {
        return false;
    }

    @Override
    public Tn create(int... shape) {
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n > Integer.MAX_VALUE - 16) {
            throw new IllegalArgumentException("tensor too large: " + java.util.Arrays.toString(shape));
        }
        return new T(new float[(int) n], shape);
    }

    @Override
    public Tn upload(float[] data, int... shape) {
        return new T(data.clone(), shape);
    }

    @Override
    public float[] download(Tn t) {
        return ((T) t).a.clone();
    }

    @Override
    public void free(Tn t) {
        // garbage collected
    }

    @Override
    public void sync() {
    }

    @Override
    public void close() {
        arena.close();
    }

    // ------------------------------------------------------------------ weights

    @Override
    public Tn weight(SafeTensors.Tensor t, Kind kind) {
        float[] src = t.toFloatArray();
        if (kind == Kind.VECTOR) {
            int[] shape = new int[t.shape().length];
            for (int i = 0; i < shape.length; i++) {
                shape[i] = (int) t.shape()[i];
            }
            return new T(src, shape);
        }
        int[] shape = new int[t.shape().length];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = (int) t.shape()[i];
        }
        int n = shape[0];
        int k;
        float[] rows = src; // [n][k] with k in the order the kernel walks it
        if (kind == Kind.CONV) {
            int ci = shape[1];
            int kh = shape[2];
            int kw = shape[3];
            k = ci * kh * kw;
            rows = new float[src.length];
            for (int co = 0; co < n; co++) {
                for (int c = 0; c < ci; c++) {
                    for (int y = 0; y < kh; y++) {
                        for (int x = 0; x < kw; x++) {
                            rows[co * k + (y * kw + x) * ci + c] = src[((co * ci + c) * kh + y) * kw + x];
                        }
                    }
                }
            }
            shape = new int[]{n, kh, kw, ci};
        } else {
            k = src.length / n; // a 1x1 convolution weight [O, I, 1, 1] is a linear weight
            shape = new int[]{n, k};
        }
        int npad = (n + NR - 1) / NR * NR;
        MemorySegment seg = arena.allocate((long) k * npad * 4L, 64);
        final int kk = k;
        final float[] r = rows;
        IntStream.range(0, npad / NR).parallel().forEach(p -> {
            for (int j = 0; j < NR; j++) {
                int col = p * NR + j;
                for (int q = 0; q < kk; q++) {
                    float v = col < n ? r[col * kk + q] : 0f;
                    seg.set(F32, ((long) p * kk + q) * NR * 4L + j * 4L, v);
                }
            }
        });
        return new W(seg, k, n, shape);
    }

    // ------------------------------------------------------------------ matrix product

    /** Epilogue shared by the tile writers. */
    private static final class Epi {
        float[] bias, chan, res;
        int chanRowsPer; // rows per batch element for chanBias lookup
        Act act = Act.NONE;
        float[] out;
        int n;
    }

    @FunctionalInterface
    private interface RowSource {
        /** Writes the A rows [m0, m1) into dst (row length K) and returns the array and offset holding them. */
        int fill(int m0, int m1, float[] dst);
    }

    private static void micro(float[] a, int o0, int o1, int o2, int o3, int k, MemorySegment w, long wByte, float[] tmp) {
        FloatVector c00 = FloatVector.zero(SP), c01 = c00, c02 = c00, c03 = c00;
        FloatVector c10 = c00, c11 = c00, c12 = c00, c13 = c00;
        FloatVector c20 = c00, c21 = c00, c22 = c00, c23 = c00;
        FloatVector c30 = c00, c31 = c00, c32 = c00, c33 = c00;
        final long step = (long) NR * 4;
        final long l4 = (long) L * 4;
        for (int q = 0; q < k; q++) {
            long p = wByte + q * step;
            FloatVector w0 = FloatVector.fromMemorySegment(SP, w, p, NATIVE);
            FloatVector w1 = FloatVector.fromMemorySegment(SP, w, p + l4, NATIVE);
            FloatVector w2 = FloatVector.fromMemorySegment(SP, w, p + 2 * l4, NATIVE);
            FloatVector w3 = FloatVector.fromMemorySegment(SP, w, p + 3 * l4, NATIVE);
            FloatVector a0 = FloatVector.broadcast(SP, a[o0 + q]);
            c00 = a0.fma(w0, c00);
            c01 = a0.fma(w1, c01);
            c02 = a0.fma(w2, c02);
            c03 = a0.fma(w3, c03);
            FloatVector a1 = FloatVector.broadcast(SP, a[o1 + q]);
            c10 = a1.fma(w0, c10);
            c11 = a1.fma(w1, c11);
            c12 = a1.fma(w2, c12);
            c13 = a1.fma(w3, c13);
            FloatVector a2 = FloatVector.broadcast(SP, a[o2 + q]);
            c20 = a2.fma(w0, c20);
            c21 = a2.fma(w1, c21);
            c22 = a2.fma(w2, c22);
            c23 = a2.fma(w3, c23);
            FloatVector a3 = FloatVector.broadcast(SP, a[o3 + q]);
            c30 = a3.fma(w0, c30);
            c31 = a3.fma(w1, c31);
            c32 = a3.fma(w2, c32);
            c33 = a3.fma(w3, c33);
        }
        c00.intoArray(tmp, 0);
        c01.intoArray(tmp, L);
        c02.intoArray(tmp, 2 * L);
        c03.intoArray(tmp, 3 * L);
        c10.intoArray(tmp, NR);
        c11.intoArray(tmp, NR + L);
        c12.intoArray(tmp, NR + 2 * L);
        c13.intoArray(tmp, NR + 3 * L);
        c20.intoArray(tmp, 2 * NR);
        c21.intoArray(tmp, 2 * NR + L);
        c22.intoArray(tmp, 2 * NR + 2 * L);
        c23.intoArray(tmp, 2 * NR + 3 * L);
        c30.intoArray(tmp, 3 * NR);
        c31.intoArray(tmp, 3 * NR + L);
        c32.intoArray(tmp, 3 * NR + 2 * L);
        c33.intoArray(tmp, 3 * NR + 3 * L);
    }

    static float gelu(float x) {
        return 0.5f * x * (1f + erf(x * 0.70710678118654752f));
    }

    private static final float[] EA = {-2.72614225801306e-10f, 2.77068142495902e-08f, -2.10102402082508e-06f,
            -5.69250639462346e-05f, -7.34990630326855e-04f, -2.95459980854025e-03f, -1.60960333262415e-02f};
    private static final float[] EB = {-1.45660718464996e-05f, -2.13374055278905e-04f, -1.68282697438203e-03f,
            -7.37332916720468e-03f, -1.42647390514189e-02f};

    /** erf with an absolute error below 7e-8 (a rational approximation; the GPU kernels use the same one). */
    static float erf(float x) {
        float c = Math.max(-4f, Math.min(4f, x));
        float x2 = c * c;
        float p = EA[0];
        for (int i = 1; i < EA.length; i++) {
            p = p * x2 + EA[i];
        }
        float q = EB[0];
        for (int i = 1; i < EB.length; i++) {
            q = q * x2 + EB[i];
        }
        return c * p / q;
    }

    static float silu(float x) {
        return x / (1f + (float) Math.exp(-x));
    }

    private static float activate(float v, Act act) {
        return switch (act) {
            case NONE -> v;
            case SILU -> silu(v);
            case GELU -> gelu(v);
            case QUICK_GELU -> v / (1f + (float) Math.exp(-1.702f * v));
        };
    }

    /** Runs the product of a [M,K] matrix given by {@code src} with the packed weight, in parallel tiles. */
    private void gemm(int m, W w, RowSource src, float[] direct, int directLd, Epi e) {
        final int k = w.k;
        final int panels = w.npad / NR;
        final int pg = 4; // panels per task
        final int mTiles = (m + MB - 1) / MB;
        final int pTiles = (panels + pg - 1) / pg;
        IntStream.range(0, mTiles * pTiles).parallel().forEach(task -> {
            int mt = task / pTiles;
            int pt = task % pTiles;
            int m0 = mt * MB;
            int m1 = Math.min(m, m0 + MB);
            float[] a;
            int base; // offset of row m0 in a
            int ld;
            if (direct != null) {
                a = direct;
                base = m0 * directLd;
                ld = directLd;
            } else {
                float[] sc = scratch.get();
                if (sc.length < MB * k) {
                    sc = new float[MB * k];
                    scratch.set(sc);
                }
                base = src.fill(m0, m1, sc);
                a = sc;
                ld = k;
            }
            float[] tmp = new float[4 * NR];
            for (int p = pt * pg; p < Math.min(panels, (pt + 1) * pg); p++) {
                long wByte = (long) p * k * NR * 4L;
                int c0 = p * NR;
                int nc = Math.min(NR, w.n - c0);
                for (int r = m0; r < m1; r += 4) {
                    int mr = Math.min(4, m1 - r);
                    int o0 = base + (r - m0) * ld;
                    int o1 = mr > 1 ? o0 + ld : o0;
                    int o2 = mr > 2 ? o0 + 2 * ld : o0;
                    int o3 = mr > 3 ? o0 + 3 * ld : o0;
                    micro(a, o0, o1, o2, o3, k, w.seg, wByte, tmp);
                    for (int i = 0; i < mr; i++) {
                        store(e, r + i, c0, nc, tmp, i * NR);
                    }
                }
            }
        });
    }

    private static void store(Epi e, int row, int c0, int nc, float[] tmp, int to) {
        int o = row * e.n + c0;
        int chanBase = e.chan == null ? 0 : (row / e.chanRowsPer) * e.n + c0;
        for (int j = 0; j < nc; j++) {
            float v = tmp[to + j];
            if (e.bias != null) {
                v += e.bias[c0 + j];
            }
            if (e.chan != null) {
                v += e.chan[chanBase + j];
            }
            v = activate(v, e.act);
            if (e.res != null) {
                v += e.res[o + j];
            }
            e.out[o + j] = v;
        }
    }

    @Override
    public Tn linear(Tn x, Tn wt, Tn bias, Tn residual, Act act) {
        W w = (W) wt;
        int rows = x.rows();
        if (x.last() != w.k) {
            throw new IllegalArgumentException("linear: input " + x + " does not fit weight " + w);
        }
        int[] shape = x.shape.clone();
        shape[shape.length - 1] = w.n;
        T out = (T) create(shape);
        Epi e = new Epi();
        e.bias = bias == null ? null : ((T) bias).a;
        e.res = residual == null ? null : ((T) residual).a;
        e.act = act;
        e.out = out.a;
        e.n = w.n;
        gemm(rows, w, null, ((T) x).a, w.k, e);
        return out;
    }

    @Override
    public Tn conv2d(Tn xt, Tn wt, Tn bias, int stride, int padT, int padL, int padB, int padR, Tn chanBias, Tn residual) {
        W w = (W) wt;
        T x = (T) xt;
        int b = x.dim(0);
        int h = x.dim(1);
        int wd = x.dim(2);
        int ci = x.dim(3);
        int kh = w.shape[1];
        int kw = w.shape[2];
        if (w.shape[3] != ci) {
            throw new IllegalArgumentException("conv2d: input " + x + " does not fit weight " + w);
        }
        int ho = (h + padT + padB - kh) / stride + 1;
        int wo = (wd + padL + padR - kw) / stride + 1;
        T out = (T) create(b, ho, wo, w.n);
        Epi e = new Epi();
        e.bias = bias == null ? null : ((T) bias).a;
        e.chan = chanBias == null ? null : ((T) chanBias).a;
        e.chanRowsPer = ho * wo;
        e.res = residual == null ? null : ((T) residual).a;
        e.out = out.a;
        e.n = w.n;
        int m = b * ho * wo;
        if (kh == 1 && kw == 1 && stride == 1 && padT == 0 && padL == 0 && padB == 0 && padR == 0) {
            gemm(m, w, null, x.a, ci, e);
            return out;
        }
        final float[] xa = x.a;
        final int k = w.k;
        gemm(m, w, (m0, m1, dst) -> {
            for (int r = m0; r < m1; r++) {
                int bi = r / (ho * wo);
                int rem = r % (ho * wo);
                int oy = rem / wo;
                int ox = rem % wo;
                int d = (r - m0) * k;
                for (int ky = 0; ky < kh; ky++) {
                    int iy = oy * stride - padT + ky;
                    for (int kx = 0; kx < kw; kx++) {
                        int ix = ox * stride - padL + kx;
                        if (iy < 0 || iy >= h || ix < 0 || ix >= wd) {
                            java.util.Arrays.fill(dst, d, d + ci, 0f);
                        } else {
                            System.arraycopy(xa, ((bi * h + iy) * wd + ix) * ci, dst, d, ci);
                        }
                        d += ci;
                    }
                }
            }
            return 0;
        }, null, 0, e);
        return out;
    }

    // ------------------------------------------------------------------ normalisation

    @Override
    public Tn groupNorm(Tn xt, Tn gamma, Tn beta, int groups, float eps, boolean silu) {
        T x = (T) xt;
        int b = x.dim(0);
        int c = x.last();
        int hw = x.rows() / b;
        int gs = c / groups;
        T out = (T) create(x.shape);
        float[] g = ((T) gamma).a;
        float[] be = ((T) beta).a;
        IntStream.range(0, b * groups).parallel().forEach(i -> {
            int bi = i / groups;
            int gi = i % groups;
            int c0 = gi * gs;
            double sum = 0;
            for (int p = 0; p < hw; p++) {
                int o = (bi * hw + p) * c + c0;
                for (int j = 0; j < gs; j++) {
                    sum += x.a[o + j];
                }
            }
            double n = (double) hw * gs;
            float mean = (float) (sum / n);
            double sq = 0;
            for (int p = 0; p < hw; p++) {
                int o = (bi * hw + p) * c + c0;
                for (int j = 0; j < gs; j++) {
                    double d = x.a[o + j] - mean;
                    sq += d * d;
                }
            }
            float rstd = (float) (1.0 / Math.sqrt(sq / n + eps));
            for (int p = 0; p < hw; p++) {
                int o = (bi * hw + p) * c + c0;
                for (int j = 0; j < gs; j++) {
                    float v = (x.a[o + j] - mean) * rstd * g[c0 + j] + be[c0 + j];
                    out.a[o + j] = silu ? silu(v) : v;
                }
            }
        });
        return out;
    }

    @Override
    public Tn layerNorm(Tn xt, Tn gamma, Tn beta, float eps) {
        T x = (T) xt;
        int c = x.last();
        int rows = x.rows();
        T out = (T) create(x.shape);
        float[] g = gamma == null ? null : ((T) gamma).a;
        float[] be = beta == null ? null : ((T) beta).a;
        IntStream.range(0, rows).parallel().forEach(r -> {
            int o = r * c;
            double s = 0;
            for (int j = 0; j < c; j++) {
                s += x.a[o + j];
            }
            float mean = (float) (s / c);
            double q = 0;
            for (int j = 0; j < c; j++) {
                double d = x.a[o + j] - mean;
                q += d * d;
            }
            float rstd = (float) (1.0 / Math.sqrt(q / c + eps));
            for (int j = 0; j < c; j++) {
                float v = (x.a[o + j] - mean) * rstd;
                if (g != null) {
                    v *= g[j];
                }
                if (be != null) {
                    v += be[j];
                }
                out.a[o + j] = v;
            }
        });
        return out;
    }

    // ------------------------------------------------------------------ attention

    private static float dot(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector acc = FloatVector.zero(SP);
        int i = 0;
        for (; i < SP.loopBound(n); i += L) {
            acc = FloatVector.fromArray(SP, a, ao + i).fma(FloatVector.fromArray(SP, b, bo + i), acc);
        }
        float s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            s += a[ao + i] * b[bo + i];
        }
        return s;
    }

    @Override
    public Tn attention(Tn qt, Tn kt, Tn vt, int heads, boolean causal) {
        T q = (T) qt;
        T k = (T) kt;
        T v = (T) vt;
        int batch = q.dim(0);
        int lq = q.rows() / batch;
        int lk = k.rows() / batch;
        int dm = q.last();
        int hd = dm / heads;
        T out = (T) create(q.shape);
        final float scale = (float) (1.0 / Math.sqrt(hd));
        final int qb = 16;
        int qBlocks = (lq + qb - 1) / qb;
        IntStream.range(0, batch * heads * qBlocks).parallel().forEach(task -> {
            int bh = task / qBlocks;
            int blk = task % qBlocks;
            int bi = bh / heads;
            int hi = bh % heads;
            int i0 = blk * qb;
            int i1 = Math.min(lq, i0 + qb);
            float[] s = new float[lk];
            for (int i = i0; i < i1; i++) {
                int qo = (bi * lq + i) * dm + hi * hd;
                int n = causal ? Math.min(lk, i + 1) : lk;
                float mx = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < n; j++) {
                    float d = dot(q.a, qo, k.a, (bi * lk + j) * dm + hi * hd, hd) * scale;
                    s[j] = d;
                    mx = Math.max(mx, d);
                }
                float sum = 0;
                for (int j = 0; j < n; j++) {
                    float e = (float) Math.exp(s[j] - mx);
                    s[j] = e;
                    sum += e;
                }
                float inv = 1f / sum;
                int oo = qo;
                for (int d = 0; d < hd; d++) {
                    out.a[oo + d] = 0f;
                }
                for (int j = 0; j < n; j++) {
                    float p = s[j] * inv;
                    int vo = (bi * lk + j) * dm + hi * hd;
                    FloatVector pv = FloatVector.broadcast(SP, p);
                    int d = 0;
                    for (; d < SP.loopBound(hd); d += L) {
                        FloatVector acc = FloatVector.fromArray(SP, out.a, oo + d);
                        pv.fma(FloatVector.fromArray(SP, v.a, vo + d), acc).intoArray(out.a, oo + d);
                    }
                    for (; d < hd; d++) {
                        out.a[oo + d] += p * v.a[vo + d];
                    }
                }
            }
        });
        return out;
    }

    // ------------------------------------------------------------------ elementwise and layout

    @Override
    public Tn geglu(Tn xt) {
        T x = (T) xt;
        int c2 = x.last();
        int n = c2 / 2;
        int rows = x.rows();
        int[] shape = x.shape.clone();
        shape[shape.length - 1] = n;
        T out = (T) create(shape);
        IntStream.range(0, rows).parallel().forEach(r -> {
            for (int j = 0; j < n; j++) {
                out.a[r * n + j] = x.a[r * c2 + j] * gelu(x.a[r * c2 + n + j]);
            }
        });
        return out;
    }

    @Override
    public Tn add(Tn at, Tn bt) {
        T a = (T) at;
        T b = (T) bt;
        T out = (T) create(a.shape);
        for (int i = 0; i < out.a.length; i++) {
            out.a[i] = a.a[i] + b.a[i];
        }
        return out;
    }

    @Override
    public Tn silu(Tn xt) {
        T x = (T) xt;
        T out = (T) create(x.shape);
        for (int i = 0; i < out.a.length; i++) {
            out.a[i] = silu(x.a[i]);
        }
        return out;
    }

    @Override
    public Tn upsample2x(Tn xt) {
        T x = (T) xt;
        int b = x.dim(0);
        int h = x.dim(1);
        int w = x.dim(2);
        int c = x.dim(3);
        T out = (T) create(b, h * 2, w * 2, c);
        IntStream.range(0, b * h * 2).parallel().forEach(r -> {
            int bi = r / (h * 2);
            int oy = r % (h * 2);
            int iy = oy / 2;
            for (int ox = 0; ox < w * 2; ox++) {
                System.arraycopy(x.a, ((bi * h + iy) * w + ox / 2) * c, out.a, ((bi * h * 2 + oy) * w * 2 + ox) * c, c);
            }
        });
        return out;
    }

    @Override
    public Tn concat(Tn at, Tn bt) {
        T a = (T) at;
        T b = (T) bt;
        int ca = a.last();
        int cb = b.last();
        int rows = a.rows();
        int[] shape = a.shape.clone();
        shape[shape.length - 1] = ca + cb;
        T out = (T) create(shape);
        for (int r = 0; r < rows; r++) {
            System.arraycopy(a.a, r * ca, out.a, r * (ca + cb), ca);
            System.arraycopy(b.a, r * cb, out.a, r * (ca + cb) + ca, cb);
        }
        return out;
    }

    @Override
    public Tn padChannels(Tn xt, int newC) {
        T x = (T) xt;
        int c = x.last();
        int rows = x.rows();
        int[] shape = x.shape.clone();
        shape[shape.length - 1] = newC;
        T out = (T) create(shape);
        for (int r = 0; r < rows; r++) {
            System.arraycopy(x.a, r * c, out.a, r * newC, c);
        }
        return out;
    }
}
