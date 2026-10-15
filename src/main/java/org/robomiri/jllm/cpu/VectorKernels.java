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
        if (w.type == QType.Q2_K) {
            return Q2KKernels.dot(w, r, x, xo);
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

    /** sum(bytes[0..16) * x[xb .. xb+16)) as a float vector (lanes not yet reduced). */
    private static FloatVector halfDot(ByteVector bytes, float[] x, int xb) {
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
        if (w.type == QType.Q2_K) {
            Q2KKernels.dequant(w, r, dst, off);
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

    static float sumSquares(float[] x, int xo, int n) {
        return dot(x, xo, x, xo, n);
    }
}
