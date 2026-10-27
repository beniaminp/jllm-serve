package org.robomiri.jllm.cpu;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.model.Q6K;
import org.robomiri.jllm.model.QTensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Vector API kernels for {@link Q6K} super-blocks (layout in that class).
 *
 * <p>Per 128-weight half the 64 {@code ql} and 32 {@code qh} bytes are loaded as six 16-byte vectors. The eight
 * 16-weight sub-blocks of the half are group {@code g = 0..3} (weights {@code 32g ..}) times {@code lb = 0, 1}
 * (the two 16-byte lane ranges): the low nibble ({@code g < 2}) or high nibble of {@code ql[32·(g&1) + 16·lb ..]}
 * is or-ed with {@code (qh[16·lb ..] >> 2g) & 3} shifted to bits 4..5 and offset by -32, giving a signed byte code
 * per lane. The dot product multiplies the codes with the activation directly and weights each sub-block's
 * lane-wise partial sum by its int8 scale; {@code d} is applied once per super-block. Only full super-blocks go
 * through SIMD; a partial trailing super-block uses the scalar helpers in {@link Q6K}.
 */
public final class Q6KKernels {

    private static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> B16 = ByteVector.SPECIES_128;
    private static final int FL = F.length();
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    private Q6KKernels() {
    }

    /** The signed codes (-32..31) of sub-block {@code (g, lb)} from the half's loaded vectors. */
    private static ByteVector codes(ByteVector[] ql, ByteVector[] qh, int g, int lb) {
        ByteVector q = ql[(g & 1) * 2 + lb];
        ByteVector low = g < 2 ? q.and((byte) 0x0F) : q.lanewise(VectorOperators.LSHR, 4);
        ByteVector high = qh[lb].lanewise(VectorOperators.LSHR, 2 * g).and((byte) 3).lanewise(VectorOperators.LSHL, 4);
        return low.or(high).sub((byte) 32);
    }

    private static void load(MemorySegment seg, long base, int h, ByteVector[] ql, ByteVector[] qh) {
        long l = base + Q6K.QL_OFFSET + h * 64L;
        for (int i = 0; i < 4; i++) {
            ql[i] = ByteVector.fromMemorySegment(B16, seg, l + 16L * i, LE);
        }
        long hq = base + Q6K.QH_OFFSET + h * 32L;
        qh[0] = ByteVector.fromMemorySegment(B16, seg, hq, LE);
        qh[1] = ByteVector.fromMemorySegment(B16, seg, hq + 16, LE);
    }

    /** Dot product of row {@code r} of {@code w} (Q6_K) with {@code x[xo .. xo + cols)}. */
    public static float dot(QTensor w, int r, float[] x, int xo) {
        MemorySegment seg = w.data;
        long base = w.rowOffset(r);
        int cols = w.cols;
        if (FL < 4 || FL > 16) {
            return Q6K.dotRow(seg, base, cols, x, xo);
        }
        int full = cols / Q6K.BLOCK;
        FloatVector acc = FloatVector.zero(F);
        ByteVector[] ql = new ByteVector[4];
        ByteVector[] qh = new ByteVector[2];
        for (int b = 0; b < full; b++, base += Q6K.BLOCK_BYTES) {
            float d = Q6K.f16(seg, base + Q6K.D_OFFSET);
            FloatVector accQ = FloatVector.zero(F); // Σ_j sc_j · (q·x)_j, lanes not reduced
            int xb = xo + b * Q6K.BLOCK;
            for (int h = 0; h < 2; h++) {
                load(seg, base, h, ql, qh);
                long sc = base + Q6K.SCALES_OFFSET + 8L * h;
                int xh = xb + h * 128;
                for (int g = 0; g < 4; g++) {
                    for (int lb = 0; lb < 2; lb++) {
                        float s = seg.get(ValueLayout.JAVA_BYTE, sc + 2 * g + lb);
                        accQ = Q2KKernels.subDot(codes(ql, qh, g, lb), x, xh + 32 * g + 16 * lb)
                                .fma(FloatVector.broadcast(F, s), accQ);
                    }
                }
            }
            acc = accQ.fma(FloatVector.broadcast(F, d), acc);
        }
        float sum = VectorKernels.laneSum(acc);
        int tail = cols - full * Q6K.BLOCK;
        if (tail > 0) {
            sum += Q6K.dotBlock(seg, base, tail, x, xo + full * Q6K.BLOCK);
        }
        return sum;
    }

    /** Dequantizes row {@code r} of {@code w} (Q6_K) into {@code dst[off .. off + cols)}. */
    public static void dequant(QTensor w, int r, float[] dst, int off) {
        MemorySegment seg = w.data;
        long base = w.rowOffset(r);
        int cols = w.cols;
        if (FL < 4 || FL > 16) {
            Q6K.dequantizeRow(seg, base, cols, dst, off);
            return;
        }
        int full = cols / Q6K.BLOCK;
        ByteVector[] ql = new ByteVector[4];
        ByteVector[] qh = new ByteVector[2];
        for (int b = 0; b < full; b++, base += Q6K.BLOCK_BYTES) {
            float d = Q6K.f16(seg, base + Q6K.D_OFFSET);
            int ob = off + b * Q6K.BLOCK;
            for (int h = 0; h < 2; h++) {
                load(seg, base, h, ql, qh);
                long sc = base + Q6K.SCALES_OFFSET + 8L * h;
                int oh = ob + h * 128;
                for (int g = 0; g < 4; g++) {
                    for (int lb = 0; lb < 2; lb++) {
                        float s = seg.get(ValueLayout.JAVA_BYTE, sc + 2 * g + lb);
                        Q2KKernels.store16(codes(ql, qh, g, lb), d * s, 0f, dst, oh + 32 * g + 16 * lb);
                    }
                }
            }
        }
        int tail = cols - full * Q6K.BLOCK;
        if (tail > 0) {
            Q6K.dequantizeBlock(seg, base, tail, dst, off + full * Q6K.BLOCK);
        }
    }
}
