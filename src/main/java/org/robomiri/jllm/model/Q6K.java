package org.robomiri.jllm.model;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * The Q6_K block format (byte-compatible with llama.cpp's {@code block_q6_K}) and its scalar helpers.
 *
 * <p>A <b>super-block</b> covers {@link #BLOCK} = 256 weights in {@link #BLOCK_BYTES} = 210 bytes (6.5625 bits per
 * weight):
 * <pre>
 *   offset   0: uint8 ql[128]    low 4 bits of the 6-bit quants, two per byte
 *   offset 128: uint8 qh[64]     high 2 bits of the quants, four per byte
 *   offset 192: int8  scales[16] one signed scale per 16-weight sub-block
 *   offset 208: f16   d          super-block scale multiplying the sub-block scales
 * </pre>
 * A weight of sub-block {@code j} is {@code w = d·scales_j·(q − 32)} with {@code q ∈ 0..63}. The 256 weights are two
 * 128-weight halves; per half, with {@code l = 0..31}: {@code ql[l]} holds the low nibbles of weights {@code l} (low
 * nibble) and {@code l+64} (high nibble), {@code ql[l+32]} those of {@code l+32} and {@code l+96}, and
 * {@code qh[l]} the high 2 bits of weights {@code l, l+32, l+64, l+96} at shifts 0, 2, 4, 6. A half advances
 * {@code ql} by 64, {@code qh} by 32 and the scales by 8.
 *
 * <p>Rows whose length is not a multiple of 256 end with a partial super-block: only the first {@code n/16}
 * sub-blocks are meaningful, the remaining scale, ql and qh bytes are zero.
 *
 * <p>Quantization is llama.cpp's {@code quantize_row_q6_K_ref}: each 16-weight sub-block gets a scale from
 * {@link #makeQxQuants} (codes in -32..31, x²-weighted least-squares search over 18 candidate scales), then the 16
 * scales are quantized to int8 with {@code d = max_scale / -128} and the codes are recomputed from the stored
 * (dequantized) scales.
 */
public final class Q6K {

    /** Weights per super-block. */
    public static final int BLOCK = 256;
    /** Weights per sub-block (one int8 scale each). */
    public static final int SUB = 16;
    /** Bytes per super-block. */
    public static final int BLOCK_BYTES = 210;
    public static final int QL_OFFSET = 0;
    public static final int QH_OFFSET = 128;
    public static final int SCALES_OFFSET = 192;
    public static final int D_OFFSET = 208;

    private static final ValueLayout.OfShort F16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static final int NMAX = 32;
    private static final float GROUP_MAX_EPS = 1e-15f;

    private Q6K() {
    }

    /** Per-thread scratch so quantizing millions of super-blocks allocates nothing. */
    private static final class Scratch {
        final byte[] L = new byte[BLOCK];
        final float[] scales = new float[BLOCK / SUB];
        final int[] sc = new int[BLOCK / SUB];
    }

    private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);

    public static float f16(MemorySegment seg, long off) {
        return Float.float16ToFloat(seg.get(F16, off));
    }

    // ------------------------------------------------------------------ quantization

    /**
     * Quantizes {@code x[off .. off + n)} ({@code n ≤ 256}, a multiple of 16) into the 210-byte super-block at
     * {@code out[outOff .. outOff + 210)}. Unused sub-blocks of a partial super-block are written as zero bytes.
     */
    public static void quantizeBlock(float[] x, int off, int n, byte[] out, int outOff) {
        if (n < SUB || n > BLOCK || n % SUB != 0) {
            throw new IllegalArgumentException("Q6_K super-block of " + n + " weights");
        }
        Scratch s = SCRATCH.get();
        byte[] L = s.L;
        Arrays.fill(L, (byte) 0);
        Arrays.fill(out, outOff, outOff + BLOCK_BYTES, (byte) 0);
        int nsub = n / SUB;
        float maxScale = 0;
        float maxAbsScale = 0;
        for (int ib = 0; ib < nsub; ib++) {
            float scale = makeQxQuants(x, off + ib * SUB, L, ib * SUB);
            s.scales[ib] = scale;
            float abs = Math.abs(scale);
            if (abs > maxAbsScale) {
                maxAbsScale = abs;
                maxScale = scale;
            }
        }
        if (maxAbsScale < GROUP_MAX_EPS) {
            return; // all zero: d = 0, every byte 0
        }
        float iscale = -128f / maxScale;
        short dBits = Float.floatToFloat16(1 / iscale);
        out[outOff + D_OFFSET] = (byte) dBits; // f16 little-endian
        out[outOff + D_OFFSET + 1] = (byte) (dBits >>> 8);
        int[] sc = s.sc;
        for (int ib = 0; ib < nsub; ib++) {
            sc[ib] = Math.min(127, (int) Math.rint(iscale * s.scales[ib]));
            out[outOff + SCALES_OFFSET + ib] = (byte) sc[ib];
        }
        // codes from the stored (dequantized) scales; a sub-block whose scale is 0 keeps the codes of
        // makeQxQuants, as llama.cpp's `if (!d) continue;` does
        float d = Float.float16ToFloat(dBits);
        for (int j = 0; j < nsub; j++) {
            float dl = d * sc[j];
            if (dl == 0) {
                continue;
            }
            for (int i = 0; i < SUB; i++) {
                int l = (int) Math.rint(x[off + SUB * j + i] / dl);
                l = Math.max(-NMAX, Math.min(NMAX - 1, l));
                L[SUB * j + i] = (byte) (l + NMAX);
            }
        }
        int ql = outOff + QL_OFFSET;
        int qh = outOff + QH_OFFSET;
        for (int j = 0; j < BLOCK; j += 128) {
            for (int l = 0; l < 32; l++) {
                int q1 = L[j + l] & 0xF;
                int q2 = L[j + l + 32] & 0xF;
                int q3 = L[j + l + 64] & 0xF;
                int q4 = L[j + l + 96] & 0xF;
                out[ql + l] = (byte) (q1 | (q3 << 4));
                out[ql + l + 32] = (byte) (q2 | (q4 << 4));
                out[qh + l] = (byte) ((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4)
                        | ((L[j + l + 96] >> 4) << 6));
            }
            ql += 64;
            qh += 32;
        }
    }

    /**
     * llama.cpp's {@code make_qx_quants(16, 32, x, L, rmse_type = 1, qw = NULL)}: finds {@code scale} with
     * {@code x ≈ scale·(L − 32)}, {@code L ∈ 0..63}, by an x²-weighted least-squares search. Writes the codes to
     * {@code L[lo .. lo+16)} and returns the (possibly negative) scale.
     */
    static float makeQxQuants(float[] x, int xo, byte[] L, int lo) {
        float max = 0;
        float amax = 0;
        for (int i = 0; i < SUB; i++) {
            float ax = Math.abs(x[xo + i]);
            if (ax > amax) {
                amax = ax;
                max = x[xo + i];
            }
        }
        if (amax < GROUP_MAX_EPS) {
            for (int i = 0; i < SUB; i++) {
                L[lo + i] = 0;
            }
            return 0f;
        }
        float iscale = -NMAX / max;
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < SUB; i++) {
            int l = (int) Math.rint(iscale * x[xo + i]);
            l = Math.max(-NMAX, Math.min(NMAX - 1, l));
            L[lo + i] = (byte) (l + NMAX);
            float w = x[xo + i] * x[xo + i];
            sumlx += w * x[xo + i] * l;
            suml2 += w * l * l;
        }
        float scale = suml2 != 0 ? sumlx / suml2 : 0.0f;
        float best = scale * sumlx;
        for (int is = -9; is <= 9; is++) {
            if (is == 0) {
                continue;
            }
            iscale = -(NMAX + 0.1f * is) / max;
            sumlx = 0;
            suml2 = 0;
            for (int i = 0; i < SUB; i++) {
                int l = (int) Math.rint(iscale * x[xo + i]);
                l = Math.max(-NMAX, Math.min(NMAX - 1, l));
                float w = x[xo + i] * x[xo + i];
                sumlx += w * x[xo + i] * l;
                suml2 += w * l * l;
            }
            if (suml2 > 0 && sumlx * sumlx > best * suml2) {
                for (int i = 0; i < SUB; i++) {
                    int l = (int) Math.rint(iscale * x[xo + i]);
                    L[lo + i] = (byte) (NMAX + Math.max(-NMAX, Math.min(NMAX - 1, l)));
                }
                scale = sumlx / suml2;
                best = scale * sumlx;
            }
        }
        return scale;
    }

    // ------------------------------------------------------------------ dequantization / scalar dot

    /** The 6-bit code minus 32 of weight {@code i} (0..255) of the super-block at {@code base}. */
    private static int code(MemorySegment seg, long base, int i) {
        int h = i >>> 7;
        int r = i & 127;
        int group = r >>> 5;
        int l = r & 31;
        int lo = seg.get(ValueLayout.JAVA_BYTE, base + QL_OFFSET + h * 64 + (group & 1) * 32 + l) & 0xFF;
        int hi = seg.get(ValueLayout.JAVA_BYTE, base + QH_OFFSET + h * 32 + l) & 0xFF;
        return (((lo >>> (4 * (group >>> 1))) & 0xF) | (((hi >>> (2 * group)) & 3) << 4)) - 32;
    }

    /** Dequantizes the first {@code n} weights of the super-block at {@code base} into {@code dst[off .. off + n)}. */
    public static void dequantizeBlock(MemorySegment seg, long base, int n, float[] dst, int off) {
        float d = f16(seg, base + D_OFFSET);
        for (int j = 0; j < n / SUB; j++) {
            float dl = d * seg.get(ValueLayout.JAVA_BYTE, base + SCALES_OFFSET + j);
            for (int k = 0; k < SUB; k++) {
                dst[off + j * SUB + k] = dl * code(seg, base, j * SUB + k);
            }
        }
    }

    /** Dequantizes a row of {@code cols} weights whose first super-block starts at {@code base}. */
    public static void dequantizeRow(MemorySegment seg, long base, int cols, float[] dst, int off) {
        for (int c = 0; c < cols; c += BLOCK) {
            dequantizeBlock(seg, base + (long) (c / BLOCK) * BLOCK_BYTES, Math.min(BLOCK, cols - c), dst, off + c);
        }
    }

    /** Dot product of the first {@code n} weights of the super-block at {@code base} with {@code x[xo .. xo + n)}. */
    public static float dotBlock(MemorySegment seg, long base, int n, float[] x, int xo) {
        float d = f16(seg, base + D_OFFSET);
        float sum = 0;
        for (int j = 0; j < n / SUB; j++) {
            float qx = 0;
            for (int k = 0; k < SUB; k++) {
                qx += code(seg, base, j * SUB + k) * x[xo + j * SUB + k];
            }
            sum += seg.get(ValueLayout.JAVA_BYTE, base + SCALES_OFFSET + j) * qx;
        }
        return d * sum;
    }

    /** Scalar reference dot product of a {@code cols}-wide row starting at {@code base} with {@code x[xo ..)}. */
    public static float dotRow(MemorySegment seg, long base, int cols, float[] x, int xo) {
        float sum = 0;
        for (int c = 0; c < cols; c += BLOCK) {
            sum += dotBlock(seg, base + (long) (c / BLOCK) * BLOCK_BYTES, Math.min(BLOCK, cols - c), x, xo + c);
        }
        return sum;
    }
}
