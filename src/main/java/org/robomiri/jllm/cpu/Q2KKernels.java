package org.robomiri.jllm.cpu;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.model.Q2K;
import org.robomiri.jllm.model.QTensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Vector API kernels for {@link Q2K} super-blocks (layout in that class).
 *
 * <p>Per 128-weight half the 32 quant bytes are loaded as two 16-byte vectors; shifting by 0/2/4/6 and masking
 * with 3 yields the eight 16-weight sub-blocks of the half as byte vectors with lanes in {0,1,2,3}. The dot product
 * uses {@code Σ (d·sc_j·q − dmin·m_j)·x = d·Σ_j sc_j·(Σ q·x)_j − dmin·Σ_j m_j·(Σ x)_j}: the codes are multiplied
 * with the activation directly (no per-weight dequantization), and the min term only needs the per-sub-block sum
 * of the activation, which is accumulated lane-wise alongside. Only full super-blocks go through SIMD; a partial
 * trailing super-block (rows not a multiple of 256) uses the scalar helpers in {@link Q2K}.
 */
public final class Q2KKernels {

    private static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> B16 = ByteVector.SPECIES_128;
    private static final int FL = F.length();
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    private Q2KKernels() {
    }

    /** Dot product of row {@code r} of {@code w} (Q2_K) with {@code x[xo .. xo + cols)}. */
    public static float dot(QTensor w, int r, float[] x, int xo) {
        MemorySegment seg = w.data;
        long base = w.rowOffset(r);
        int cols = w.cols;
        if (FL < 4 || FL > 16) {
            return Q2K.dotRow(seg, base, cols, x, xo);
        }
        int full = cols / Q2K.BLOCK;
        FloatVector acc = FloatVector.zero(F);
        for (int b = 0; b < full; b++, base += Q2K.BLOCK_BYTES) {
            float d = Q2K.f16(seg, base + Q2K.D_OFFSET);
            float dmin = Q2K.f16(seg, base + Q2K.DMIN_OFFSET);
            FloatVector accQ = FloatVector.zero(F); // Σ_j sc_j · (q·x)_j, lanes not reduced
            FloatVector accM = FloatVector.zero(F); // Σ_j m_j · (Σ x)_j
            int xb = xo + b * Q2K.BLOCK;
            long sc = base + Q2K.SCALES_OFFSET;
            for (int h = 0; h < 2; h++, sc += 8) {
                long qs = base + Q2K.QS_OFFSET + h * 32;
                ByteVector v0 = ByteVector.fromMemorySegment(B16, seg, qs, LE);
                ByteVector v1 = ByteVector.fromMemorySegment(B16, seg, qs + 16, LE);
                int xh = xb + h * 128;
                for (int k = 0; k < 4; k++) {
                    int s0 = seg.get(ValueLayout.JAVA_BYTE, sc + 2 * k) & 0xFF;
                    int s1 = seg.get(ValueLayout.JAVA_BYTE, sc + 2 * k + 1) & 0xFF;
                    ByteVector q0 = v0.lanewise(VectorOperators.LSHR, 2 * k).and((byte) 3);
                    ByteVector q1 = v1.lanewise(VectorOperators.LSHR, 2 * k).and((byte) 3);
                    int x0 = xh + k * 32;
                    accQ = subDot(q0, x, x0).fma(FloatVector.broadcast(F, (float) (s0 & 0xF)), accQ);
                    accM = subSum(x, x0).fma(FloatVector.broadcast(F, (float) (s0 >>> 4)), accM);
                    accQ = subDot(q1, x, x0 + 16).fma(FloatVector.broadcast(F, (float) (s1 & 0xF)), accQ);
                    accM = subSum(x, x0 + 16).fma(FloatVector.broadcast(F, (float) (s1 >>> 4)), accM);
                }
            }
            acc = accQ.fma(FloatVector.broadcast(F, d), acc);
            acc = accM.fma(FloatVector.broadcast(F, -dmin), acc);
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        int tail = cols - full * Q2K.BLOCK;
        if (tail > 0) {
            sum += Q2K.dotBlock(seg, base, tail, x, xo + full * Q2K.BLOCK);
        }
        return sum;
    }

    /** Dequantizes row {@code r} of {@code w} (Q2_K) into {@code dst[off .. off + cols)}. */
    public static void dequant(QTensor w, int r, float[] dst, int off) {
        MemorySegment seg = w.data;
        long base = w.rowOffset(r);
        int cols = w.cols;
        if (FL < 4 || FL > 16) {
            Q2K.dequantizeRow(seg, base, cols, dst, off);
            return;
        }
        int full = cols / Q2K.BLOCK;
        for (int b = 0; b < full; b++, base += Q2K.BLOCK_BYTES) {
            float d = Q2K.f16(seg, base + Q2K.D_OFFSET);
            float dmin = Q2K.f16(seg, base + Q2K.DMIN_OFFSET);
            int ob = off + b * Q2K.BLOCK;
            long sc = base + Q2K.SCALES_OFFSET;
            for (int h = 0; h < 2; h++, sc += 8) {
                long qs = base + Q2K.QS_OFFSET + h * 32;
                ByteVector v0 = ByteVector.fromMemorySegment(B16, seg, qs, LE);
                ByteVector v1 = ByteVector.fromMemorySegment(B16, seg, qs + 16, LE);
                int oh = ob + h * 128;
                for (int k = 0; k < 4; k++) {
                    int s0 = seg.get(ValueLayout.JAVA_BYTE, sc + 2 * k) & 0xFF;
                    int s1 = seg.get(ValueLayout.JAVA_BYTE, sc + 2 * k + 1) & 0xFF;
                    ByteVector q0 = v0.lanewise(VectorOperators.LSHR, 2 * k).and((byte) 3);
                    ByteVector q1 = v1.lanewise(VectorOperators.LSHR, 2 * k).and((byte) 3);
                    store16(q0, d * (s0 & 0xF), -dmin * (s0 >>> 4), dst, oh + k * 32);
                    store16(q1, d * (s1 & 0xF), -dmin * (s1 >>> 4), dst, oh + k * 32 + 16);
                }
            }
        }
        int tail = cols - full * Q2K.BLOCK;
        if (tail > 0) {
            Q2K.dequantizeBlock(seg, base, tail, dst, off + full * Q2K.BLOCK);
        }
    }

    /** {@code dst[o .. o+16) = q * scale + negMin} for the 16 byte lanes of {@code q}. */
    private static void store16(ByteVector q, float scale, float negMin, float[] dst, int o) {
        FloatVector s = FloatVector.broadcast(F, scale);
        FloatVector m = FloatVector.broadcast(F, negMin);
        for (int p = 0; p < 16 / FL; p++) {
            ((FloatVector) q.castShape(F, p)).fma(s, m).intoArray(dst, o + p * FL);
        }
    }

    /** sum(q[0..16) * x[xb .. xb+16)) as a float vector (lanes not yet reduced). */
    private static FloatVector subDot(ByteVector q, float[] x, int xb) {
        if (FL == 16) {
            return ((FloatVector) q.castShape(F, 0)).mul(FloatVector.fromArray(F, x, xb));
        } else if (FL == 8) {
            FloatVector a = ((FloatVector) q.castShape(F, 0)).mul(FloatVector.fromArray(F, x, xb));
            return ((FloatVector) q.castShape(F, 1)).fma(FloatVector.fromArray(F, x, xb + 8), a);
        } else {
            FloatVector a = ((FloatVector) q.castShape(F, 0)).mul(FloatVector.fromArray(F, x, xb));
            FloatVector b = ((FloatVector) q.castShape(F, 1)).mul(FloatVector.fromArray(F, x, xb + 4));
            a = ((FloatVector) q.castShape(F, 2)).fma(FloatVector.fromArray(F, x, xb + 8), a);
            b = ((FloatVector) q.castShape(F, 3)).fma(FloatVector.fromArray(F, x, xb + 12), b);
            return a.add(b);
        }
    }

    /** sum(x[xb .. xb+16)) as a float vector (lanes not yet reduced). */
    private static FloatVector subSum(float[] x, int xb) {
        if (FL == 16) {
            return FloatVector.fromArray(F, x, xb);
        } else if (FL == 8) {
            return FloatVector.fromArray(F, x, xb).add(FloatVector.fromArray(F, x, xb + 8));
        } else {
            FloatVector a = FloatVector.fromArray(F, x, xb).add(FloatVector.fromArray(F, x, xb + 4));
            FloatVector b = FloatVector.fromArray(F, x, xb + 8).add(FloatVector.fromArray(F, x, xb + 12));
            return a.add(b);
        }
    }
}
