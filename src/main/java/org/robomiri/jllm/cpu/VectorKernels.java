package org.robomiri.jllm.cpu;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * SIMD kernels written with the Vector API. Quantized blocks are expanded byte -> float in registers and
 * consumed with FMAs, so weights are streamed from memory exactly once per matrix-vector product.
 */
public final class VectorKernels {

    static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
    static final VectorSpecies<Byte> B16 = ByteVector.SPECIES_128;
    static final int FL = F.length();
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    private VectorKernels() {
    }

    public static String describe() {
        return "Vector API " + F.vectorBitSize() + "-bit (" + FL + " float lanes)";
    }

    // ------------------------------------------------------------------ quantized row dot products

    /** Dot product of row {@code r} of {@code w} with {@code x[xo .. xo + cols)}. */
    static float dotRow(QTensor w, int r, float[] x, int xo) {
        if (w.second != null) {
            return dotRow(w.partOf(r), w.partRow(r), x, xo);
        }
        if (w.type == QType.Q2_K) {
            return Q2KKernels.dot(w, r, x, xo);
        }
        if (w.type.gguf) {
            return GgufKernels.dot(w, r, x, xo);
        }
        if (w.type == QType.Q6_K) {
            return Q6KKernels.dot(w, r, x, xo);
        }
        int nb = w.blocksPerRow();
        if (FL < 4 || FL > 16) {
            return dotRowScalar(w, r, x, xo);
        }
        return w.type == QType.Q4_0
                ? dotQ4(w.data, w.rowOffset(r), w.scaleOffset(r), nb, x, xo)
                : dotQ8(w.data, w.rowOffset(r), w.scaleOffset(r), nb, x, xo);
    }

    private static float dotQ4(MemorySegment seg, long q, long s, int nb, float[] x, int xo) {
        FloatVector acc = FloatVector.zero(F);
        for (int b = 0; b < nb; b++) {
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + (long) b * 16, LE);
            ByteVector lo = v.and((byte) 0x0F).sub((byte) 8);
            ByteVector hi = v.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            FloatVector scale = FloatVector.broadcast(F, seg.get(ValueLayout.JAVA_FLOAT, s + (long) b * 4));
            int xb = xo + b * QType.BLOCK;
            acc = halfDot(lo, x, xb).add(halfDot(hi, x, xb + 16)).fma(scale, acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static float dotQ8(MemorySegment seg, long q, long s, int nb, float[] x, int xo) {
        FloatVector acc = FloatVector.zero(F);
        for (int b = 0; b < nb; b++) {
            long base = q + (long) b * 32;
            ByteVector v0 = ByteVector.fromMemorySegment(B16, seg, base, LE);
            ByteVector v1 = ByteVector.fromMemorySegment(B16, seg, base + 16, LE);
            FloatVector scale = FloatVector.broadcast(F, seg.get(ValueLayout.JAVA_FLOAT, s + (long) b * 4));
            int xb = xo + b * QType.BLOCK;
            acc = halfDot(v0, x, xb).add(halfDot(v1, x, xb + 16)).fma(scale, acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /**
     * sum(bytes[0..16) * x[xb .. xb+16)) as a float vector (lanes not yet reduced).
     *
     * <p>{@code castShape(F, part)} with {@code part > 0} is implemented as {@code slice(part * FL)} followed by a
     * part-0 cast; on AArch64 the slice is a single {@code ext} and the JIT compiles it as cheaply as part 0
     * (measured: equal cost). Restructuring this to narrower byte loads so that only part-0 casts remain was
     * measured slower on 128-bit NEON (1.5x per Q4 row in isolation, and the wider Q8 body fell off the intrinsic
     * path entirely), so the part-based form is kept deliberately.
     */
    static FloatVector halfDot(ByteVector bytes, float[] x, int xb) {
        if (FL == 16) {
            return ((FloatVector) bytes.castShape(F, 0)).mul(FloatVector.fromArray(F, x, xb));
        } else if (FL == 8) {
            FloatVector a = ((FloatVector) bytes.castShape(F, 0)).mul(FloatVector.fromArray(F, x, xb));
            return ((FloatVector) bytes.castShape(F, 1)).fma(FloatVector.fromArray(F, x, xb + 8), a);
        } else {
            FloatVector a = ((FloatVector) bytes.castShape(F, 0)).mul(FloatVector.fromArray(F, x, xb));
            FloatVector b = ((FloatVector) bytes.castShape(F, 1)).mul(FloatVector.fromArray(F, x, xb + 4));
            a = ((FloatVector) bytes.castShape(F, 2)).fma(FloatVector.fromArray(F, x, xb + 8), a);
            b = ((FloatVector) bytes.castShape(F, 3)).fma(FloatVector.fromArray(F, x, xb + 12), b);
            return a.add(b);
        }
    }

    private static float dotRowScalar(QTensor w, int r, float[] x, int xo) {
        float[] tmp = new float[w.cols];
        w.dequantizeRow(r, tmp, 0);
        float s = 0;
        for (int i = 0; i < w.cols; i++) {
            s += tmp[i] * x[xo + i];
        }
        return s;
    }

    /** Dequantizes row {@code r} into {@code dst[off .. off + cols)} using SIMD conversions. */
    static void dequantRow(QTensor w, int r, float[] dst, int off) {
        if (w.second != null) {
            dequantRow(w.partOf(r), w.partRow(r), dst, off);
            return;
        }
        if (w.type == QType.Q2_K) {
            Q2KKernels.dequant(w, r, dst, off);
            return;
        }
        if (w.type.gguf) {
            GgufKernels.dequant(w, r, dst, off);
            return;
        }
        if (w.type == QType.Q6_K) {
            Q6KKernels.dequant(w, r, dst, off);
            return;
        }
        if (FL < 4 || FL > 16) {
            w.dequantizeRow(r, dst, off);
            return;
        }
        MemorySegment seg = w.data;
        long q = w.rowOffset(r);
        long s = w.scaleOffset(r);
        int nb = w.blocksPerRow();
        for (int b = 0; b < nb; b++) {
            FloatVector scale = FloatVector.broadcast(F, seg.get(ValueLayout.JAVA_FLOAT, s + (long) b * 4));
            int o = off + b * QType.BLOCK;
            ByteVector first;
            ByteVector second;
            if (w.type == QType.Q4_0) {
                ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + (long) b * 16, LE);
                first = v.and((byte) 0x0F).sub((byte) 8);
                second = v.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            } else {
                first = ByteVector.fromMemorySegment(B16, seg, q + (long) b * 32, LE);
                second = ByteVector.fromMemorySegment(B16, seg, q + (long) b * 32 + 16, LE);
            }
            for (int p = 0; p < 16 / FL; p++) {
                ((FloatVector) first.castShape(F, p)).mul(scale).intoArray(dst, o + p * FL);
                ((FloatVector) second.castShape(F, p)).mul(scale).intoArray(dst, o + 16 + p * FL);
            }
        }
    }

    // ------------------------------------------------------------------ float kernels

    static float dot(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(F);
        FloatVector a1 = FloatVector.zero(F);
        int i = 0;
        int ub = n - 2 * FL;
        for (; i <= ub; i += 2 * FL) {
            a0 = FloatVector.fromArray(F, a, ao + i).fma(FloatVector.fromArray(F, b, bo + i), a0);
            a1 = FloatVector.fromArray(F, a, ao + i + FL).fma(FloatVector.fromArray(F, b, bo + i + FL), a1);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            s += a[ao + i] * b[bo + i];
        }
        return s;
    }

    /** {@code y[yo..] += a * x[xo..]} */
    static void axpy(float a, float[] x, int xo, float[] y, int yo, int n) {
        FloatVector av = FloatVector.broadcast(F, a);
        int i = 0;
        int ub = F.loopBound(n);
        for (; i < ub; i += FL) {
            FloatVector.fromArray(F, x, xo + i).fma(av, FloatVector.fromArray(F, y, yo + i)).intoArray(y, yo + i);
        }
        for (; i < n; i++) {
            y[yo + i] += a * x[xo + i];
        }
    }

    /**
     * Four simultaneous dot products of rows {@code w[wo + k*n ..]} (k = 0..3) with {@code x[xo..]}, written to
     * {@code out[oo .. oo+4)} (or accumulated into it). Sharing the x loads quarters activation bandwidth.
     */
    static void dot4(float[] w, int wo, int n, float[] x, int xo, float[] out, int oo, boolean acc) {
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
        store4(out, oo, s0, s1, s2, s3, acc);
    }

    /**
     * 4 x 4 register tile: the four rows {@code w[wo + k*n ..]} against four activation vectors
     * {@code x[xa..], x[xb..], x[xc..], x[xd..]}; results for vector {@code x?} go to {@code out[o? .. o?+4)}.
     * Eight loads feed sixteen FMAs. Every accumulator follows the same lane-wise order as {@link #dot4}, so each
     * output is bit-identical to the untiled kernel.
     */
    static void dot4x4(float[] w, int wo, int n, float[] x, int xa, int xb, int xc, int xd,
                       float[] out, int oa, int ob, int oc, int od, boolean acc) {
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
        store4(out, oa, a0, a1, a2, a3, acc);
        store4(out, ob, b0, b1, b2, b3, acc);
        store4(out, oc, c0, c1, c2, c3, acc);
        store4(out, od, d0, d1, d2, d3, acc);
    }

    private static void store4(float[] out, int oo, FloatVector s0, FloatVector s1, FloatVector s2, FloatVector s3,
                               boolean acc) {
        float r0 = s0.reduceLanes(VectorOperators.ADD);
        float r1 = s1.reduceLanes(VectorOperators.ADD);
        float r2 = s2.reduceLanes(VectorOperators.ADD);
        float r3 = s3.reduceLanes(VectorOperators.ADD);
        if (acc) {
            out[oo] += r0;
            out[oo + 1] += r1;
            out[oo + 2] += r2;
            out[oo + 3] += r3;
        } else {
            out[oo] = r0;
            out[oo + 1] = r1;
            out[oo + 2] = r2;
            out[oo + 3] = r3;
        }
    }

    // ------------------------------------------------------------------ fast math (opt-in, not bit-exact)

    /**
     * {@code -Djllm.cpu.fastmath=true}: the attention softmax and the soft-capping use the polynomial
     * {@link #expFast} / {@link #tanhFast} on float lanes instead of scalar {@link Math#exp} / {@link Math#tanh}.
     * Relative error about 2e-7 (exp) and 1e-6 (tanh), so not bit-identical to the default path; off by default.
     */
    public static final boolean FASTMATH = Boolean.getBoolean("jllm.cpu.fastmath");

    private static final float LOG2E = 1.4426950408889634f;
    private static final float LN2_HI = 0.693145751953125f;    // ln 2 split so that n * LN2_HI is exact for |n| < 2^10
    private static final float LN2_LO = 1.428606765330187e-6f;
    private static final float EXP_MIN = -87.0f;               // e^-87 = 1.6e-38, the smallest normal magnitude
    private static final float EXP_MAX = 88.0f;
    private static final float ROUND_MAGIC = 12582912f;        // 1.5 * 2^23: v + MAGIC - MAGIC rounds v to the nearest integer
    private static final int ROUND_MAGIC_BITS = Float.floatToRawIntBits(ROUND_MAGIC);

    /**
     * {@code e^x} per lane: {@code x = n ln 2 + r} with {@code |r| <= ln 2 / 2}, a degree-6 Taylor polynomial for
     * {@code e^r} (Horner, FMAs) and {@code 2^n} built from the exponent bits. {@code x} is clamped to
     * {@code [-87, 88]}, so underflow stops at {@code 1.6e-38} instead of 0 (harmless in a softmax). The integer
     * {@code n} comes out of the 1.5 * 2^23 rounding trick as in {@link Int8Kernels}, without a conversion node.
     */
    public static FloatVector expFast(FloatVector x) {
        x = x.max(EXP_MIN).min(EXP_MAX);
        FloatVector t = x.mul(LOG2E).add(ROUND_MAGIC);
        FloatVector n = t.sub(ROUND_MAGIC);
        FloatVector r = n.fma(FloatVector.broadcast(F, -LN2_HI), x);
        r = n.fma(FloatVector.broadcast(F, -LN2_LO), r);
        FloatVector p = FloatVector.broadcast(F, 1f / 720);
        p = p.fma(r, FloatVector.broadcast(F, 1f / 120));
        p = p.fma(r, FloatVector.broadcast(F, 1f / 24));
        p = p.fma(r, FloatVector.broadcast(F, 1f / 6));
        p = p.fma(r, FloatVector.broadcast(F, 0.5f));
        p = p.fma(r, FloatVector.broadcast(F, 1f));
        p = p.fma(r, FloatVector.broadcast(F, 1f));
        FloatVector scale = t.reinterpretAsInts().sub(ROUND_MAGIC_BITS).add(127).lanewise(VectorOperators.LSHL, 23)
                .reinterpretAsFloats();
        return p.mul(scale);
    }

    /** Scalar twin of {@link #expFast} (same arithmetic, for loop tails). */
    public static float expFast(float x) {
        x = Math.min(Math.max(x, EXP_MIN), EXP_MAX);
        float t = x * LOG2E + ROUND_MAGIC;
        float n = t - ROUND_MAGIC;
        float r = Math.fma(n, -LN2_HI, x);
        r = Math.fma(n, -LN2_LO, r);
        float p = 1f / 720;
        p = Math.fma(p, r, 1f / 120);
        p = Math.fma(p, r, 1f / 24);
        p = Math.fma(p, r, 1f / 6);
        p = Math.fma(p, r, 0.5f);
        p = Math.fma(p, r, 1f);
        p = Math.fma(p, r, 1f);
        return p * Float.intBitsToFloat((Float.floatToRawIntBits(t) - ROUND_MAGIC_BITS + 127) << 23);
    }

    /** {@code tanh(x) = 1 - 2 / (e^{2x} + 1)} per lane with {@link #expFast}; saturates correctly at both ends. */
    public static FloatVector tanhFast(FloatVector x) {
        FloatVector e = expFast(x.add(x));
        return FloatVector.broadcast(F, 1f).sub(FloatVector.broadcast(F, 2f).div(e.add(1f)));
    }

    public static float tanhFast(float x) {
        return 1f - 2f / (expFast(x + x) + 1f);
    }

    /**
     * Softmax numerator in place: {@code x[i] = exp(x[i] - max)} for {@code i in [o, o + n)}, returning the sum.
     * {@link #FASTMATH}: vector lanes and {@link #expFast}, the sum accumulated lane-wise and reduced once;
     * otherwise scalar {@link Math#exp} summed in order, which is what the attention loops always did.
     */
    public static float softmaxExp(float[] x, int o, int n, float max) {
        if (!FASTMATH) {
            float sum = 0;
            for (int i = o; i < o + n; i++) {
                float e = (float) Math.exp(x[i] - max);
                x[i] = e;
                sum += e;
            }
            return sum;
        }
        FloatVector mv = FloatVector.broadcast(F, max);
        FloatVector acc = FloatVector.zero(F);
        int i = o;
        int ub = o + F.loopBound(n);
        for (; i < ub; i += FL) {
            FloatVector e = expFast(FloatVector.fromArray(F, x, i).sub(mv));
            e.intoArray(x, i);
            acc = acc.add(e);
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < o + n; i++) {
            float e = expFast(x[i] - max);
            x[i] = e;
            sum += e;
        }
        return sum;
    }

    /** {@code x[i] = tanh(x[i] * (1 / cap)) * cap} for {@code i in [o, o + n)} with {@link #tanhFast} (callers check {@link #FASTMATH}). */
    public static void softcapFast(float[] x, int o, int n, float cap) {
        float inv = 1f / cap; // lanes and the scalar tail both multiply by the reciprocal, so they agree bit for bit
        FloatVector iv = FloatVector.broadcast(F, inv);
        FloatVector cv = FloatVector.broadcast(F, cap);
        int i = o;
        int ub = o + F.loopBound(n);
        for (; i < ub; i += FL) {
            tanhFast(FloatVector.fromArray(F, x, i).mul(iv)).mul(cv).intoArray(x, i);
        }
        for (; i < o + n; i++) {
            x[i] = tanhFast(x[i] * inv) * cap;
        }
    }

    /**
     * MoE combine of one token: {@code x[xo + i] += Σ_k w[wo + k] · d[dOff + k * stride + i]} for {@code i < n}.
     * Each lane follows the scalar evaluation exactly - the products are summed in {@code k} order starting from
     * {@code w_0 · d_0}, with separate multiply and add (no FMA), and the total is added to {@code x} last - so the
     * result is bit-identical to the scalar triple loop it replaces.
     */
    static void combine(float[] w, int wo, int topK, float[] d, int dOff, int stride, float[] x, int xo, int n) {
        int i = 0;
        int ub = F.loopBound(n);
        for (; i < ub; i += FL) {
            FloatVector acc = FloatVector.fromArray(F, d, dOff + i).mul(w[wo]);
            for (int k = 1; k < topK; k++) {
                acc = acc.add(FloatVector.fromArray(F, d, dOff + k * stride + i).mul(w[wo + k]));
            }
            acc.add(FloatVector.fromArray(F, x, xo + i)).intoArray(x, xo + i);
        }
        for (; i < n; i++) {
            float sum = 0;
            for (int k = 0; k < topK; k++) {
                sum += w[wo + k] * d[dOff + k * stride + i];
            }
            x[xo + i] += sum;
        }
    }

    static float sumSquares(float[] x, int xo, int n) {
        return dot(x, xo, x, xo, n);
    }

    /** RMSNorm of one row: {@code out[oo + i] = (in[io + i] * rsqrt(mean(in^2) + eps)) * w[i]}. */
    static void rmsnormRow(float[] in, int io, float[] out, int oo, float[] w, int n, float eps) {
        float ss = sumSquares(in, io, n);
        float scale = (float) (1.0 / Math.sqrt(ss / n + eps));
        scaleMul(in, io, scale, w, out, oo, n);
    }

    /** {@code x / (1 + e^-x)}, the scalar SiLU every CPU path uses (so they all round the same way). */
    static float silu(float x) {
        return x / (1f + (float) Math.exp(-x));
    }

    /** {@code out[oo + i] = (in[io + i] * s) * w[i]}, the RMSNorm scaling step (same rounding as the scalar loop). */
    static void scaleMul(float[] in, int io, float s, float[] w, float[] out, int oo, int n) {
        FloatVector sv = FloatVector.broadcast(F, s);
        int i = 0;
        int ub = F.loopBound(n);
        for (; i < ub; i += FL) {
            FloatVector.fromArray(F, in, io + i).mul(sv).mul(FloatVector.fromArray(F, w, i)).intoArray(out, oo + i);
        }
        for (; i < n; i++) {
            out[oo + i] = in[io + i] * s * w[i];
        }
    }

    /**
     * Rotary embedding of one head in place: for {@code i < half}, {@code (a, b) = (v[o+i], v[o+i+half])} becomes
     * {@code (a*cos - b*sin, b*cos + a*sin)} with {@code cos/sin[co + i]}. Separate multiplies and add/subtract
     * (no FMA), so the result matches the scalar expression bit for bit.
     */
    static void rope(float[] v, int o, int half, float[] cos, float[] sin, int co) {
        int i = 0;
        int ub = F.loopBound(half);
        for (; i < ub; i += FL) {
            FloatVector a = FloatVector.fromArray(F, v, o + i);
            FloatVector b = FloatVector.fromArray(F, v, o + i + half);
            FloatVector cs = FloatVector.fromArray(F, cos, co + i);
            FloatVector sn = FloatVector.fromArray(F, sin, co + i);
            a.mul(cs).sub(b.mul(sn)).intoArray(v, o + i);
            b.mul(cs).add(a.mul(sn)).intoArray(v, o + i + half);
        }
        for (; i < half; i++) {
            float a = v[o + i];
            float b = v[o + i + half];
            float cs = cos[co + i];
            float sn = sin[co + i];
            v[o + i] = a * cs - b * sn;
            v[o + i + half] = b * cs + a * sn;
        }
    }
}
