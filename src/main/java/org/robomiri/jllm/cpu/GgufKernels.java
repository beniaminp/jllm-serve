package org.robomiri.jllm.cpu;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.model.Ggml;
import org.robomiri.jllm.model.QTensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Vector API kernels for the ggml block layouts of {@link Ggml} (Q4_0, Q8_0, Q4_K, Q5_K), reading a GGUF file's
 * tensors in place. The structure is that of {@link VectorKernels} (Q4_0 / Q8_0) and {@link Q2KKernels} (K-quants):
 * quantized bytes are widened to floats in registers and consumed with FMAs; the K-quant dot uses
 * {@code Σ (d·sc·q − dmin·m)·x = d·Σ sc·(q·x) − dmin·Σ m·(Σ x)} per 32-weight sub-block, so the codes meet the
 * activation without a per-weight dequantization. The f16 scale of a legacy block sits at an odd 2-byte offset,
 * so scales are read with unaligned loads.
 */
final class GgufKernels {

    private static final VectorSpecies<Float> F = VectorKernels.F;
    private static final VectorSpecies<Byte> B16 = VectorKernels.B16;
    private static final int FL = VectorKernels.FL;
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;
    private static final ValueLayout.OfInt LE32 = ValueLayout.JAVA_INT_UNALIGNED.withOrder(LE);

    private GgufKernels() {
    }

    private static boolean simd() {
        return FL >= 4 && FL <= 16;
    }

    /** Dot product of row {@code r} of {@code w} (a GGUF-layout type) with {@code x[xo .. xo + cols)}. */
    static float dot(QTensor w, int r, float[] x, int xo) {
        if (!simd()) {
            return dotScalar(w, r, x, xo);
        }
        MemorySegment seg = w.data;
        long q = w.rowOffset(r);
        int nb = w.blocksPerRow();
        return switch (w.type) {
            case Q4_0G -> dotQ4(seg, q, nb, x, xo);
            case Q8_0G -> dotQ8(seg, q, nb, x, xo);
            case Q4_K -> dotK(seg, q, nb, x, xo, false);
            case Q5_K -> dotK(seg, q, nb, x, xo, true);
            default -> throw new IllegalStateException(w.type.name());
        };
    }

    /** Dequantizes row {@code r} into {@code dst[off .. off + cols)}. */
    static void dequant(QTensor w, int r, float[] dst, int off) {
        if (!simd()) {
            w.dequantizeRow(r, dst, off);
            return;
        }
        MemorySegment seg = w.data;
        long q = w.rowOffset(r);
        int nb = w.blocksPerRow();
        switch (w.type) {
            case Q4_0G -> dequantQ4(seg, q, nb, dst, off);
            case Q8_0G -> dequantQ8(seg, q, nb, dst, off);
            case Q4_K -> dequantK(seg, q, nb, dst, off, false);
            case Q5_K -> dequantK(seg, q, nb, dst, off, true);
            default -> throw new IllegalStateException(w.type.name());
        }
    }

    private static float dotScalar(QTensor w, int r, float[] x, int xo) {
        float[] tmp = new float[w.cols];
        w.dequantizeRow(r, tmp, 0);
        float s = 0;
        for (int i = 0; i < w.cols; i++) {
            s += tmp[i] * x[xo + i];
        }
        return s;
    }

    // ------------------------------------------------------------------ Q4_0 / Q8_0

    private static float dotQ4(MemorySegment seg, long q, int nb, float[] x, int xo) {
        FloatVector acc = FloatVector.zero(F);
        for (int b = 0; b < nb; b++, q += Ggml.Q4_0_BYTES) {
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + 2, LE);
            ByteVector lo = v.and((byte) 0x0F).sub((byte) 8);
            ByteVector hi = v.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            FloatVector scale = FloatVector.broadcast(F, Ggml.f16(seg, q));
            int xb = xo + b * 32;
            acc = VectorKernels.halfDot(lo, x, xb).add(VectorKernels.halfDot(hi, x, xb + 16)).fma(scale, acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static float dotQ8(MemorySegment seg, long q, int nb, float[] x, int xo) {
        FloatVector acc = FloatVector.zero(F);
        for (int b = 0; b < nb; b++, q += Ggml.Q8_0_BYTES) {
            ByteVector v0 = ByteVector.fromMemorySegment(B16, seg, q + 2, LE);
            ByteVector v1 = ByteVector.fromMemorySegment(B16, seg, q + 18, LE);
            FloatVector scale = FloatVector.broadcast(F, Ggml.f16(seg, q));
            int xb = xo + b * 32;
            acc = VectorKernels.halfDot(v0, x, xb).add(VectorKernels.halfDot(v1, x, xb + 16)).fma(scale, acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static void dequantQ4(MemorySegment seg, long q, int nb, float[] dst, int off) {
        for (int b = 0; b < nb; b++, q += Ggml.Q4_0_BYTES) {
            FloatVector scale = FloatVector.broadcast(F, Ggml.f16(seg, q));
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + 2, LE);
            store16(v.and((byte) 0x0F).sub((byte) 8), scale, dst, off + b * 32);
            store16(v.lanewise(VectorOperators.LSHR, 4).sub((byte) 8), scale, dst, off + b * 32 + 16);
        }
    }

    private static void dequantQ8(MemorySegment seg, long q, int nb, float[] dst, int off) {
        for (int b = 0; b < nb; b++, q += Ggml.Q8_0_BYTES) {
            FloatVector scale = FloatVector.broadcast(F, Ggml.f16(seg, q));
            store16(ByteVector.fromMemorySegment(B16, seg, q + 2, LE), scale, dst, off + b * 32);
            store16(ByteVector.fromMemorySegment(B16, seg, q + 18, LE), scale, dst, off + b * 32 + 16);
        }
    }

    /** {@code dst[o .. o+16) = bytes * scale}. */
    private static void store16(ByteVector bytes, FloatVector scale, float[] dst, int o) {
        for (int p = 0; p < 16 / FL; p++) {
            ((FloatVector) bytes.castShape(F, p)).mul(scale).intoArray(dst, o + p * FL);
        }
    }

    // ------------------------------------------------------------------ Q4_K / Q5_K

    /**
     * The eight 6-bit sub-block scales ({@link #scalesFull}, sub-block {@code j} in byte {@code j} of the result)
     * and mins ({@link #mins}) of a K-quant block whose 12 scale bytes start at {@code sc}: llama.cpp's word-wise
     * form of {@code get_scale_min_k4} (bytes 0..3 hold scales 0..3 and mins 0..3 in their low 6 bits, bytes 8..11
     * the low nibbles of scales 4..7 and the high nibbles of mins 4..7, and the top two bits of bytes 0..7 the
     * high bits of scales 4..7 / mins 4..7).
     */
    private static long mins(MemorySegment seg, long sc) {
        int u0 = seg.get(LE32, sc);
        int u1 = seg.get(LE32, sc + 4);
        int u2 = seg.get(LE32, sc + 8);
        int k2 = 0x0f0f0f0f;
        int k3 = 0x03030303;
        int lo = (u1 & 0x3f3f3f3f);                                          // mins 0..3
        int hi = ((u2 >>> 4) & k2) | (((u1 >>> 6) & k3) << 4);               // mins 4..7
        return (lo & 0xFFFFFFFFL) | ((long) hi << 32);
    }

    /** Scales 4..7 differ from the plain 6-bit form: they take two high bits from the first four bytes. */
    private static long scalesFull(MemorySegment seg, long sc) {
        int u0 = seg.get(LE32, sc);
        int u1 = seg.get(LE32, sc + 4);
        int u2 = seg.get(LE32, sc + 8);
        int k2 = 0x0f0f0f0f;
        int k3 = 0x03030303;
        int lo = u0 & 0x3f3f3f3f;                                            // scales 0..3
        int hi = (u2 & k2) | (((u0 >>> 6) & k3) << 4);                       // scales 4..7
        return (lo & 0xFFFFFFFFL) | ((long) hi << 32);
    }

    private static float dotK(MemorySegment seg, long q, int nb, float[] x, int xo, boolean q5) {
        int bytes = q5 ? Ggml.Q5_K_BYTES : Ggml.Q4_K_BYTES;
        FloatVector acc = FloatVector.zero(F);
        for (int b = 0; b < nb; b++, q += bytes) {
            float d = Ggml.f16(seg, q);
            float dmin = Ggml.f16(seg, q + Ggml.K_DMIN);
            long sc = scalesFull(seg, q + Ggml.K_SCALES);
            long mn = mins(seg, q + Ggml.K_SCALES);
            long qs = q + (q5 ? Ggml.Q5_K_QS : Ggml.Q4_K_QS);
            ByteVector h0 = null;
            ByteVector h1 = null;
            if (q5) {
                h0 = ByteVector.fromMemorySegment(B16, seg, q + Ggml.Q5_K_QH, LE);
                h1 = ByteVector.fromMemorySegment(B16, seg, q + Ggml.Q5_K_QH + 16, LE);
            }
            int xb = xo + b * Ggml.QK_K;
            FloatVector accQ = FloatVector.zero(F);
            FloatVector accM = FloatVector.zero(F);
            for (int g = 0; g < 4; g++, qs += 32, xb += 64) {
                ByteVector v0 = ByteVector.fromMemorySegment(B16, seg, qs, LE);
                ByteVector v1 = ByteVector.fromMemorySegment(B16, seg, qs + 16, LE);
                ByteVector lo0 = v0.and((byte) 0x0F);
                ByteVector lo1 = v1.and((byte) 0x0F);
                ByteVector hi0 = v0.lanewise(VectorOperators.LSHR, 4);
                ByteVector hi1 = v1.lanewise(VectorOperators.LSHR, 4);
                if (q5) {
                    lo0 = lo0.add(h0.lanewise(VectorOperators.LSHR, 2 * g).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    lo1 = lo1.add(h1.lanewise(VectorOperators.LSHR, 2 * g).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    hi0 = hi0.add(h0.lanewise(VectorOperators.LSHR, 2 * g + 1).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    hi1 = hi1.add(h1.lanewise(VectorOperators.LSHR, 2 * g + 1).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                }
                int s0 = (int) (sc >>> (16 * g)) & 0xFF;
                int s1 = (int) (sc >>> (16 * g + 8)) & 0xFF;
                int m0 = (int) (mn >>> (16 * g)) & 0xFF;
                int m1 = (int) (mn >>> (16 * g + 8)) & 0xFF;
                accQ = VectorKernels.halfDot(lo0, x, xb).add(VectorKernels.halfDot(lo1, x, xb + 16))
                        .fma(FloatVector.broadcast(F, (float) s0), accQ);
                accM = sum32(x, xb).fma(FloatVector.broadcast(F, (float) m0), accM);
                accQ = VectorKernels.halfDot(hi0, x, xb + 32).add(VectorKernels.halfDot(hi1, x, xb + 48))
                        .fma(FloatVector.broadcast(F, (float) s1), accQ);
                accM = sum32(x, xb + 32).fma(FloatVector.broadcast(F, (float) m1), accM);
            }
            acc = accQ.fma(FloatVector.broadcast(F, d), acc);
            acc = accM.fma(FloatVector.broadcast(F, -dmin), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static void dequantK(MemorySegment seg, long q, int nb, float[] dst, int off, boolean q5) {
        int bytes = q5 ? Ggml.Q5_K_BYTES : Ggml.Q4_K_BYTES;
        for (int b = 0; b < nb; b++, q += bytes) {
            float d = Ggml.f16(seg, q);
            float dmin = Ggml.f16(seg, q + Ggml.K_DMIN);
            long sc = scalesFull(seg, q + Ggml.K_SCALES);
            long mn = mins(seg, q + Ggml.K_SCALES);
            long qs = q + (q5 ? Ggml.Q5_K_QS : Ggml.Q4_K_QS);
            ByteVector h0 = null;
            ByteVector h1 = null;
            if (q5) {
                h0 = ByteVector.fromMemorySegment(B16, seg, q + Ggml.Q5_K_QH, LE);
                h1 = ByteVector.fromMemorySegment(B16, seg, q + Ggml.Q5_K_QH + 16, LE);
            }
            int o = off + b * Ggml.QK_K;
            for (int g = 0; g < 4; g++, qs += 32, o += 64) {
                ByteVector v0 = ByteVector.fromMemorySegment(B16, seg, qs, LE);
                ByteVector v1 = ByteVector.fromMemorySegment(B16, seg, qs + 16, LE);
                ByteVector lo0 = v0.and((byte) 0x0F);
                ByteVector lo1 = v1.and((byte) 0x0F);
                ByteVector hi0 = v0.lanewise(VectorOperators.LSHR, 4);
                ByteVector hi1 = v1.lanewise(VectorOperators.LSHR, 4);
                if (q5) {
                    lo0 = lo0.add(h0.lanewise(VectorOperators.LSHR, 2 * g).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    lo1 = lo1.add(h1.lanewise(VectorOperators.LSHR, 2 * g).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    hi0 = hi0.add(h0.lanewise(VectorOperators.LSHR, 2 * g + 1).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    hi1 = hi1.add(h1.lanewise(VectorOperators.LSHR, 2 * g + 1).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                }
                float d1 = d * ((int) (sc >>> (16 * g)) & 0xFF);
                float d2 = d * ((int) (sc >>> (16 * g + 8)) & 0xFF);
                float m1 = dmin * ((int) (mn >>> (16 * g)) & 0xFF);
                float m2 = dmin * ((int) (mn >>> (16 * g + 8)) & 0xFF);
                storeAffine(lo0, d1, -m1, dst, o);
                storeAffine(lo1, d1, -m1, dst, o + 16);
                storeAffine(hi0, d2, -m2, dst, o + 32);
                storeAffine(hi1, d2, -m2, dst, o + 48);
            }
        }
    }

    /** {@code dst[o .. o+16) = q * scale + negMin} for the 16 byte lanes of {@code q}. */
    private static void storeAffine(ByteVector q, float scale, float negMin, float[] dst, int o) {
        FloatVector s = FloatVector.broadcast(F, scale);
        FloatVector m = FloatVector.broadcast(F, negMin);
        for (int p = 0; p < 16 / FL; p++) {
            ((FloatVector) q.castShape(F, p)).fma(s, m).intoArray(dst, o + p * FL);
        }
    }

    /** sum(x[xb .. xb+32)) as a float vector (lanes not yet reduced). */
    private static FloatVector sum32(float[] x, int xb) {
        FloatVector a = FloatVector.zero(F);
        for (int i = 0; i < 32; i += FL) {
            a = a.add(FloatVector.fromArray(F, x, xb + i));
        }
        return a;
    }
}
