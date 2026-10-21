package org.robomiri.jllm.cpu;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;
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
    private static final ValueLayout.OfShort LE16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(LE);

    private GgufKernels() {
    }

    // lanes 0..7 take byte 0 (or 2) of a 4-byte group, lanes 8..15 byte 1 (or 3): the bit j of a 16-bit run is lane j
    private static final VectorShuffle<Byte> SH_LO = VectorShuffle.fromValues(B16, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    private static final VectorShuffle<Byte> SH_HI = VectorShuffle.fromValues(B16, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    private static final ByteVector BIT_OF_LANE = ByteVector.fromArray(B16, new byte[]{1, 2, 4, 8, 16, 32, 64, (byte) 128,
            1, 2, 4, 8, 16, 32, 64, (byte) 128}, 0);
    private static final ByteVector TAB_IQ4 = ByteVector.fromArray(B16, Ggml.KVALUES_IQ4NL, 0);
    private static final ByteVector TAB_FP4 = ByteVector.fromArray(B16, Ggml.KVALUES_MXFP4, 0);

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
            case Q4_1 -> dotQ41(seg, q, nb, x, xo, false);
            case Q5_0 -> dotQ50(seg, q, nb, x, xo);
            case Q5_1 -> dotQ41(seg, q, nb, x, xo, true);
            case Q3_K -> dotQ3K(seg, q, nb, x, xo);
            case IQ4_NL -> dotTab(seg, q, nb, x, xo, false);
            case MXFP4 -> dotTab(seg, q, nb, x, xo, true);
            case IQ4_XS -> dotIQ4XS(seg, q, nb, x, xo);
            case F16 -> dotHalves(seg, q, nb * 32, x, xo, false);
            case BF16 -> dotHalves(seg, q, nb * 32, x, xo, true);
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
            case Q4_1 -> dequantQ41(seg, q, nb, dst, off, false);
            case Q5_0 -> dequantQ50(seg, q, nb, dst, off);
            case Q5_1 -> dequantQ41(seg, q, nb, dst, off, true);
            case Q3_K -> dequantQ3K(seg, q, nb, dst, off);
            case IQ4_NL -> dequantTab(seg, q, nb, dst, off, false);
            case MXFP4 -> dequantTab(seg, q, nb, dst, off, true);
            case IQ4_XS -> dequantIQ4XS(seg, q, nb, dst, off);
            case F16 -> dequantHalves(seg, q, nb * 32, dst, off, false);
            case BF16 -> dequantHalves(seg, q, nb * 32, dst, off, true);
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

    // ------------------------------------------------------------------ Q4_1 / Q5_0 / Q5_1 (32-weight blocks, 5th bit in qh)

    /** Lanes whose bit is clear: lane j of the half {@code hi ? 1 : 0} tests bit {@code 16 hi + j} of the block's 32 high bits. */
    private static VectorMask<Byte> bitClear(ByteVector qhBytes, boolean hi) {
        return qhBytes.rearrange(hi ? SH_HI : SH_LO).and(BIT_OF_LANE).eq((byte) 0);
    }

    /** Q4_1 (20 B: d, m, 16 nibble bytes) and Q5_1 (24 B: d, m, qh[4], 16 nibble bytes): {@code Σ (d q + m) x = d Σ q x + m Σ x}. */
    private static float dotQ41(MemorySegment seg, long q, int nb, float[] x, int xo, boolean q5) {
        int bytes = q5 ? Ggml.Q5_1_BYTES : Ggml.Q4_1_BYTES;
        FloatVector acc = FloatVector.zero(F);
        FloatVector accM = FloatVector.zero(F);
        for (int b = 0; b < nb; b++, q += bytes) {
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + (q5 ? 8 : 4), LE);
            ByteVector lo = v.and((byte) 0x0F);
            ByteVector hi = v.lanewise(VectorOperators.LSHR, 4);
            if (q5) {
                ByteVector qh = ByteVector.fromMemorySegment(B16, seg, q + 4, LE);
                lo = lo.add((byte) 16, bitClear(qh, false).not());
                hi = hi.add((byte) 16, bitClear(qh, true).not());
            }
            int xb = xo + b * 32;
            acc = VectorKernels.halfDot(lo, x, xb).add(VectorKernels.halfDot(hi, x, xb + 16))
                    .fma(FloatVector.broadcast(F, Ggml.f16(seg, q)), acc);
            accM = sum32(x, xb).fma(FloatVector.broadcast(F, Ggml.f16(seg, q + 2)), accM);
        }
        return acc.add(accM).reduceLanes(VectorOperators.ADD);
    }

    private static void dequantQ41(MemorySegment seg, long q, int nb, float[] dst, int off, boolean q5) {
        int bytes = q5 ? Ggml.Q5_1_BYTES : Ggml.Q4_1_BYTES;
        for (int b = 0; b < nb; b++, q += bytes) {
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + (q5 ? 8 : 4), LE);
            ByteVector lo = v.and((byte) 0x0F);
            ByteVector hi = v.lanewise(VectorOperators.LSHR, 4);
            if (q5) {
                ByteVector qh = ByteVector.fromMemorySegment(B16, seg, q + 4, LE);
                lo = lo.add((byte) 16, bitClear(qh, false).not());
                hi = hi.add((byte) 16, bitClear(qh, true).not());
            }
            float d = Ggml.f16(seg, q);
            float m = Ggml.f16(seg, q + 2);
            storeAffine(lo, d, m, dst, off + b * 32);
            storeAffine(hi, d, m, dst, off + b * 32 + 16);
        }
    }

    /** Q5_0 (22 B: d, qh[4], 16 nibble bytes): the 5-bit code minus 16 (a clear high bit costs 16), like Q4_0 minus 8. */
    private static float dotQ50(MemorySegment seg, long q, int nb, float[] x, int xo) {
        FloatVector acc = FloatVector.zero(F);
        for (int b = 0; b < nb; b++, q += Ggml.Q5_0_BYTES) {
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + 6, LE);
            ByteVector qh = ByteVector.fromMemorySegment(B16, seg, q + 2, LE);
            ByteVector lo = v.and((byte) 0x0F).sub((byte) 16, bitClear(qh, false));
            ByteVector hi = v.lanewise(VectorOperators.LSHR, 4).sub((byte) 16, bitClear(qh, true));
            int xb = xo + b * 32;
            acc = VectorKernels.halfDot(lo, x, xb).add(VectorKernels.halfDot(hi, x, xb + 16))
                    .fma(FloatVector.broadcast(F, Ggml.f16(seg, q)), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static void dequantQ50(MemorySegment seg, long q, int nb, float[] dst, int off) {
        for (int b = 0; b < nb; b++, q += Ggml.Q5_0_BYTES) {
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + 6, LE);
            ByteVector qh = ByteVector.fromMemorySegment(B16, seg, q + 2, LE);
            FloatVector scale = FloatVector.broadcast(F, Ggml.f16(seg, q));
            store16(v.and((byte) 0x0F).sub((byte) 16, bitClear(qh, false)), scale, dst, off + b * 32);
            store16(v.lanewise(VectorOperators.LSHR, 4).sub((byte) 16, bitClear(qh, true)), scale, dst, off + b * 32 + 16);
        }
    }

    // ------------------------------------------------------------------ IQ4_NL / MXFP4 / IQ4_XS (nibbles index a 16-entry table)

    /** IQ4_NL (18 B: f16 d, 16 nibble bytes) and MXFP4 (17 B: e8m0 byte, 16 nibble bytes): {@code d · table[nibble]}. */
    private static float dotTab(MemorySegment seg, long q, int nb, float[] x, int xo, boolean fp4) {
        int bytes = fp4 ? Ggml.MXFP4_BYTES : Ggml.IQ4_NL_BYTES;
        ByteVector tab = fp4 ? TAB_FP4 : TAB_IQ4;
        FloatVector acc = FloatVector.zero(F);
        for (int b = 0; b < nb; b++, q += bytes) {
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + (fp4 ? 1 : 2), LE);
            ByteVector lo = tab.rearrange(v.and((byte) 0x0F).toShuffle());
            ByteVector hi = tab.rearrange(v.lanewise(VectorOperators.LSHR, 4).toShuffle());
            float d = fp4 ? Ggml.e8m0Half(seg.get(ValueLayout.JAVA_BYTE, q) & 0xFF) : Ggml.f16(seg, q);
            int xb = xo + b * 32;
            acc = VectorKernels.halfDot(lo, x, xb).add(VectorKernels.halfDot(hi, x, xb + 16))
                    .fma(FloatVector.broadcast(F, d), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static void dequantTab(MemorySegment seg, long q, int nb, float[] dst, int off, boolean fp4) {
        int bytes = fp4 ? Ggml.MXFP4_BYTES : Ggml.IQ4_NL_BYTES;
        ByteVector tab = fp4 ? TAB_FP4 : TAB_IQ4;
        for (int b = 0; b < nb; b++, q += bytes) {
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + (fp4 ? 1 : 2), LE);
            float d = fp4 ? Ggml.e8m0Half(seg.get(ValueLayout.JAVA_BYTE, q) & 0xFF) : Ggml.f16(seg, q);
            FloatVector scale = FloatVector.broadcast(F, d);
            store16(tab.rearrange(v.and((byte) 0x0F).toShuffle()), scale, dst, off + b * 32);
            store16(tab.rearrange(v.lanewise(VectorOperators.LSHR, 4).toShuffle()), scale, dst, off + b * 32 + 16);
        }
    }

    /** The 6-bit scale (minus 32) of sub-block {@code ib} of an IQ4_XS block at {@code q}: low nibble of {@code scales_l}, 2 bits of {@code scales_h}. */
    private static int iq4xsScale(MemorySegment seg, long q, int sh, int ib) {
        int lo = (seg.get(ValueLayout.JAVA_BYTE, q + 4 + (ib >> 1)) >>> (4 * (ib & 1))) & 0xF;
        return (lo | (((sh >>> (2 * ib)) & 3) << 4)) - 32;
    }

    private static float dotIQ4XS(MemorySegment seg, long q, int nb, float[] x, int xo) {
        FloatVector acc = FloatVector.zero(F);
        for (int b = 0; b < nb; b++, q += Ggml.IQ4_XS_BYTES) {
            int sh = seg.get(LE16, q + 2) & 0xFFFF;
            FloatVector accQ = FloatVector.zero(F);
            int xb = xo + b * Ggml.QK_K;
            for (int ib = 0; ib < 8; ib++, xb += 32) {
                ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + 8 + 16L * ib, LE);
                ByteVector lo = TAB_IQ4.rearrange(v.and((byte) 0x0F).toShuffle());
                ByteVector hi = TAB_IQ4.rearrange(v.lanewise(VectorOperators.LSHR, 4).toShuffle());
                accQ = VectorKernels.halfDot(lo, x, xb).add(VectorKernels.halfDot(hi, x, xb + 16))
                        .fma(FloatVector.broadcast(F, (float) iq4xsScale(seg, q, sh, ib)), accQ);
            }
            acc = accQ.fma(FloatVector.broadcast(F, Ggml.f16(seg, q)), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static void dequantIQ4XS(MemorySegment seg, long q, int nb, float[] dst, int off) {
        for (int b = 0; b < nb; b++, q += Ggml.IQ4_XS_BYTES) {
            float d = Ggml.f16(seg, q);
            int sh = seg.get(LE16, q + 2) & 0xFFFF;
            int o = off + b * Ggml.QK_K;
            for (int ib = 0; ib < 8; ib++, o += 32) {
                ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + 8 + 16L * ib, LE);
                FloatVector scale = FloatVector.broadcast(F, d * iq4xsScale(seg, q, sh, ib));
                store16(TAB_IQ4.rearrange(v.and((byte) 0x0F).toShuffle()), scale, dst, o);
                store16(TAB_IQ4.rearrange(v.lanewise(VectorOperators.LSHR, 4).toShuffle()), scale, dst, o + 16);
            }
        }
    }

    // ------------------------------------------------------------------ Q3_K

    /**
     * The sixteen 6-bit scales of a Q3_K block at {@code q} packed four to an int (scale {@code s} is byte {@code s & 3}
     * of word {@code s >> 2}, before the "minus 32"): llama.cpp's {@code kmask1 / kmask2} shuffle of the 12 scale bytes.
     */
    private static void q3kScaleWords(MemorySegment seg, long q, int[] w) {
        int a0 = seg.get(LE32, q + 96);
        int a1 = seg.get(LE32, q + 100);
        int t = seg.get(LE32, q + 104);
        w[0] = (a0 & 0x0f0f0f0f) | (((t) & 0x03030303) << 4);
        w[1] = (a1 & 0x0f0f0f0f) | (((t >>> 2) & 0x03030303) << 4);
        w[2] = ((a0 >>> 4) & 0x0f0f0f0f) | (((t >>> 4) & 0x03030303) << 4);
        w[3] = ((a1 >>> 4) & 0x0f0f0f0f) | (((t >>> 6) & 0x03030303) << 4);
    }

    private static final ThreadLocal<int[]> Q3_SCALES = ThreadLocal.withInitial(() -> new int[4]);

    private static int q3kScale(int[] w, int s) {
        return ((w[s >> 2] >>> (8 * (s & 3))) & 0xFF) - 32;
    }

    /**
     * The 16 weights of one half of unit {@code u} (0..7) of a Q3_K block: the 2-bit code {@code (qs >> 2 (u & 3)) & 3}
     * of the 16 bytes {@code qs} minus 4 where the high-mask bit {@code u} of the matching 16 bytes {@code hm} is clear.
     */
    private static ByteVector q3kHalf(ByteVector qs, ByteVector hm, int u) {
        byte bit = (byte) (1 << u);
        return qs.lanewise(VectorOperators.LSHR, 2 * (u & 3)).and((byte) 3).sub((byte) 4, hm.and(bit).eq((byte) 0));
    }

    private static float dotQ3K(MemorySegment seg, long q, int nb, float[] x, int xo) {
        int[] w = Q3_SCALES.get();
        FloatVector acc = FloatVector.zero(F);
        for (int b = 0; b < nb; b++, q += Ggml.Q3_K_BYTES) {
            q3kScaleWords(seg, q, w);
            ByteVector h0 = ByteVector.fromMemorySegment(B16, seg, q, LE);
            ByteVector h1 = ByteVector.fromMemorySegment(B16, seg, q + 16, LE);
            FloatVector accQ = FloatVector.zero(F);
            int xb = xo + b * Ggml.QK_K;
            for (int u = 0; u < 8; u++, xb += 32) {
                long qs = q + 32 + 32L * (u >> 2);
                ByteVector v0 = ByteVector.fromMemorySegment(B16, seg, qs, LE);
                ByteVector v1 = ByteVector.fromMemorySegment(B16, seg, qs + 16, LE);
                accQ = VectorKernels.halfDot(q3kHalf(v0, h0, u), x, xb).fma(FloatVector.broadcast(F, (float) q3kScale(w, 2 * u)), accQ);
                accQ = VectorKernels.halfDot(q3kHalf(v1, h1, u), x, xb + 16).fma(FloatVector.broadcast(F, (float) q3kScale(w, 2 * u + 1)), accQ);
            }
            acc = accQ.fma(FloatVector.broadcast(F, Ggml.f16(seg, q + 108)), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static void dequantQ3K(MemorySegment seg, long q, int nb, float[] dst, int off) {
        int[] w = Q3_SCALES.get();
        for (int b = 0; b < nb; b++, q += Ggml.Q3_K_BYTES) {
            q3kScaleWords(seg, q, w);
            float d = Ggml.f16(seg, q + 108);
            ByteVector h0 = ByteVector.fromMemorySegment(B16, seg, q, LE);
            ByteVector h1 = ByteVector.fromMemorySegment(B16, seg, q + 16, LE);
            int o = off + b * Ggml.QK_K;
            for (int u = 0; u < 8; u++, o += 32) {
                long qs = q + 32 + 32L * (u >> 2);
                ByteVector v0 = ByteVector.fromMemorySegment(B16, seg, qs, LE);
                ByteVector v1 = ByteVector.fromMemorySegment(B16, seg, qs + 16, LE);
                store16(q3kHalf(v0, h0, u), FloatVector.broadcast(F, d * q3kScale(w, 2 * u)), dst, o);
                store16(q3kHalf(v1, h1, u), FloatVector.broadcast(F, d * q3kScale(w, 2 * u + 1)), dst, o + 16);
            }
        }
    }

    // ------------------------------------------------------------------ F16 / BF16 (the weights as they are)

    private static final VectorSpecies<Short> SS = VectorSpecies.of(short.class, jdk.incubator.vector.VectorShape.forBitSize(16 * FL));
    private static final VectorSpecies<Integer> IS = jdk.incubator.vector.IntVector.SPECIES_PREFERRED;

    /** FL weights (halves or bfloat16) at byte {@code q} of {@code seg} widened to floats. */
    private static FloatVector halves(MemorySegment seg, long q, boolean bf16) {
        jdk.incubator.vector.IntVector bits = (jdk.incubator.vector.IntVector) jdk.incubator.vector.ShortVector
                .fromMemorySegment(SS, seg, q, LE).convertShape(VectorOperators.S2I, IS, 0).reinterpretAsInts();
        bits = bits.and(0xFFFF);
        return bf16 ? bits.lanewise(VectorOperators.LSHL, 16).reinterpretAsFloats() : KvCache.F16.lowHalf(bits);
    }

    private static float dotHalves(MemorySegment seg, long q, int cols, float[] x, int xo, boolean bf16) {
        FloatVector a0 = FloatVector.zero(F);
        FloatVector a1 = FloatVector.zero(F);
        for (int i = 0; i < cols; i += 2 * FL) {
            a0 = halves(seg, q + 2L * i, bf16).fma(FloatVector.fromArray(F, x, xo + i), a0);
            a1 = halves(seg, q + 2L * (i + FL), bf16).fma(FloatVector.fromArray(F, x, xo + i + FL), a1);
        }
        return a0.add(a1).reduceLanes(VectorOperators.ADD);
    }

    private static void dequantHalves(MemorySegment seg, long q, int cols, float[] dst, int off, boolean bf16) {
        for (int i = 0; i < cols; i += FL) {
            halves(seg, q + 2L * i, bf16).intoArray(dst, off + i);
        }
    }
}
