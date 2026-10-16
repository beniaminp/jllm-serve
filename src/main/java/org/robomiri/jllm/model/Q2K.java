package org.robomiri.jllm.model;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * The Q2_K block format (byte-compatible with llama.cpp's {@code block_q2_K}) and its scalar helpers.
 *
 * <p>A <b>super-block</b> covers {@link #BLOCK} = 256 weights in {@link #BLOCK_BYTES} = 84 bytes (2.625 bits per
 * weight):
 * <pre>
 *   offset  0: uint8 scales[16]   one byte per 16-weight sub-block: low nibble = 4-bit scale sc, high nibble = 4-bit min m
 *   offset 16: uint8 qs[64]       2-bit quants q in {0,1,2,3}, four per byte (see below)
 *   offset 80: f16   d            super-block scale multiplying the sub-block scales
 *   offset 82: f16   dmin         super-block scale multiplying the sub-block mins
 * </pre>
 * A weight of sub-block {@code j} is reconstructed as {@code w = d·sc_j·q − dmin·m_j}: the min is <i>subtracted</i>,
 * so {@code m ≥ 0} encodes a non-positive minimum. The 64 quant bytes hold two 128-weight halves of 32 bytes each;
 * within a half, byte {@code l} (0..31) holds weights {@code l, l+32, l+64, l+96} at bit shifts 0, 2, 4, 6. Sub-block
 * {@code j} therefore lives at byte offset {@code 16 + (j/8)·32 + (j%2)·16} with shift {@code 2·((j%8)/2)}.
 *
 * <p>Rows whose length is not a multiple of 256 end with a partial super-block: only the first {@code n/16} sub-blocks
 * are meaningful, the remaining scale and quant bytes are zero. Rows that are a multiple of 256 (every linear layer
 * of Qwen3-0.6B and Qwen3-30B-A3B) are laid out exactly as llama.cpp would.
 *
 * <p>Quantization follows llama.cpp's {@code quantize_row_q2_K_ref}: each 16-weight sub-block gets a (scale, min)
 * pair from {@link #makeQkx2Quants} (closed-form start {@code scale = (max − min)/3}, then 16 candidate scales
 * between {@code (max − min)/4} and {@code (max − min)/2.5}, each refined by a weighted least-squares fit of
 * (scale, min) to the rounded codes and kept if it lowers the |x|-weighted absolute error). The 16 scales and mins
 * are then quantized to 4 bits with {@code d = max(scale)/15}, {@code dmin = max(min)/15}, and the codes are
 * recomputed from the <i>dequantized</i> sub-block scale/min so the stored values are self-consistent (a
 * sub-block whose 4-bit scale is 0 keeps the codes of the fit, as llama.cpp does, so the bytes match it).
 */
public final class Q2K {

    /** Weights per super-block. */
    public static final int BLOCK = 256;
    /** Weights per sub-block (one 4-bit scale and one 4-bit min each). */
    public static final int SUB = 16;
    /** Bytes per super-block. */
    public static final int BLOCK_BYTES = 84;
    public static final int SCALES_OFFSET = 0;
    public static final int QS_OFFSET = 16;
    public static final int D_OFFSET = 80;
    public static final int DMIN_OFFSET = 82;

    private static final ValueLayout.OfShort F16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    // make_qkx2_quants parameters used by llama.cpp for Q2_K
    private static final int NMAX = 3;
    private static final float RMIN = -0.5f;
    private static final float RDELTA = 0.1f;
    private static final int NSTEP = 15;

    private Q2K() {
    }

    /** Per-thread scratch so quantizing millions of super-blocks allocates nothing. */
    private static final class Scratch {
        final byte[] L = new byte[BLOCK];
        final byte[] laux = new byte[SUB];
        final float[] weights = new float[SUB];
        final float[] scales = new float[BLOCK / SUB];
        final float[] mins = new float[BLOCK / SUB];
        final byte[] qs = new byte[64];
        final int[] sc = new int[BLOCK / SUB];
        final int[] mn = new int[BLOCK / SUB];
    }

    private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);

    public static float f16(MemorySegment seg, long off) {
        return Float.float16ToFloat(seg.get(F16, off));
    }

    // ------------------------------------------------------------------ quantization

    /**
     * Quantizes {@code x[off .. off + n)} ({@code n ≤ 256}, a multiple of 16) into the 84-byte super-block at
     * {@code out[outOff .. outOff + 84)}. Unused sub-blocks of a partial super-block are written as zero bytes.
     */
    public static void quantizeBlock(float[] x, int off, int n, byte[] out, int outOff) {
        if (n < SUB || n > BLOCK || n % SUB != 0) {
            throw new IllegalArgumentException("Q2_K super-block of " + n + " weights");
        }
        Scratch s = SCRATCH.get();
        int nsub = n / SUB;
        float maxScale = 0;
        float maxMin = 0;
        for (int j = 0; j < nsub; j++) {
            for (int l = 0; l < SUB; l++) {
                s.weights[l] = Math.abs(x[off + j * SUB + l]);
            }
            s.scales[j] = makeQkx2Quants(x, off + j * SUB, s.weights, s.L, j * SUB, s.mins, j, s.laux);
            maxScale = Math.max(maxScale, s.scales[j]);
            maxMin = Math.max(maxMin, s.mins[j]);
        }

        // 16 sub-block scales / mins -> 4 bits each, relative to the super-block's largest
        int[] sc = s.sc;
        int[] mn = s.mn;
        Arrays.fill(sc, 0);
        Arrays.fill(mn, 0);
        short dBits = 0;
        short dminBits = 0;
        if (maxScale > 0) {
            float iscale = 15f / maxScale;
            for (int j = 0; j < nsub; j++) {
                sc[j] = clamp((int) Math.rint(iscale * s.scales[j]), 0, 15);
            }
            dBits = Float.floatToFloat16(maxScale / 15f);
        }
        if (maxMin > 0) {
            float iscale = 15f / maxMin;
            for (int j = 0; j < nsub; j++) {
                mn[j] = clamp((int) Math.rint(iscale * s.mins[j]), 0, 15);
            }
            dminBits = Float.floatToFloat16(maxMin / 15f);
        }
        for (int j = 0; j < BLOCK / SUB; j++) {
            out[outOff + SCALES_OFFSET + j] = (byte) (sc[j] | (mn[j] << 4));
        }
        out[outOff + D_OFFSET] = (byte) dBits;              // f16 little-endian
        out[outOff + D_OFFSET + 1] = (byte) (dBits >>> 8);
        out[outOff + DMIN_OFFSET] = (byte) dminBits;
        out[outOff + DMIN_OFFSET + 1] = (byte) (dminBits >>> 8);

        // final codes from the dequantized (4-bit) sub-block scale and min, so storage is self-consistent. A
        // sub-block whose 4-bit scale rounds to 0 keeps the codes of the make_qkx2_quants fit, exactly as
        // llama.cpp's `if (!d) continue;` does: every code reconstructs to -dmin*m there, so only byte-level
        // parity with llama.cpp depends on it. Sub-blocks beyond a partial super-block are zero.
        float d = Float.float16ToFloat(dBits);
        float dmin = Float.float16ToFloat(dminBits);
        Arrays.fill(s.L, n, BLOCK, (byte) 0);
        for (int j = 0; j < nsub; j++) {
            float dl = d * sc[j];
            if (dl == 0) {
                continue;
            }
            float ml = dmin * mn[j];
            for (int l = 0; l < SUB; l++) {
                int q = (int) Math.rint((x[off + j * SUB + l] + ml) / dl);
                s.L[j * SUB + l] = (byte) clamp(q, 0, NMAX);
            }
        }

        // pack: per 128-weight half, byte l = L[l] | L[l+32] << 2 | L[l+64] << 4 | L[l+96] << 6
        byte[] L = s.L;
        for (int h = 0; h < BLOCK; h += 128) {
            for (int l = 0; l < 32; l++) {
                s.qs[h / 4 + l] = (byte) (L[h + l] | (L[h + l + 32] << 2) | (L[h + l + 64] << 4) | (L[h + l + 96] << 6));
            }
        }
        System.arraycopy(s.qs, 0, out, outOff + QS_OFFSET, 64);
    }

    /**
     * llama.cpp's {@code make_qkx2_quants} for {@code nmax = 3}: finds (scale, min) with
     * {@code x ≈ scale·L − min}, {@code L ∈ 0..3}, minimizing {@code Σ weights·|error|}. Writes the codes to
     * {@code L[lo .. lo+16)}, the (non-negative) min to {@code mins[j]} and returns the scale.
     */
    static float makeQkx2Quants(float[] x, int xo, float[] weights, byte[] L, int lo, float[] mins, int j, byte[] laux) {
        float min = x[xo];
        float max = x[xo];
        float sumW = weights[0];
        float sumX = sumW * x[xo];
        for (int i = 1; i < SUB; i++) {
            float v = x[xo + i];
            min = Math.min(min, v);
            max = Math.max(max, v);
            float w = weights[i];
            sumW += w;
            sumX += w * v;
        }
        if (min > 0) {
            min = 0;
        }
        if (max == min) {
            for (int i = 0; i < SUB; i++) {
                L[lo + i] = 0;
            }
            mins[j] = -min;
            return 0f;
        }
        // closed-form start: the span maps onto the 4 levels
        float iscale = NMAX / (max - min);
        float scale = 1 / iscale;
        float bestMad = 0;
        for (int i = 0; i < SUB; i++) {
            int l = clamp((int) Math.rint(iscale * (x[xo + i] - min)), 0, NMAX);
            L[lo + i] = (byte) l;
            bestMad += weights[i] * Math.abs(scale * l + min - x[xo + i]);
        }
        // candidate steps: try 16 slightly different rounding grids and re-fit (scale, min) to each by least squares
        for (int is = 0; is <= NSTEP; is++) {
            iscale = (RMIN + RDELTA * is + NMAX) / (max - min);
            float sumL = 0;
            float sumL2 = 0;
            float sumXL = 0;
            for (int i = 0; i < SUB; i++) {
                int l = clamp((int) Math.rint(iscale * (x[xo + i] - min)), 0, NMAX);
                laux[i] = (byte) l;
                float w = weights[i];
                sumL += w * l;
                sumL2 += w * l * l;
                sumXL += w * l * x[xo + i];
            }
            float det = sumW * sumL2 - sumL * sumL;
            if (det > 0) {
                float thisScale = (sumW * sumXL - sumX * sumL) / det;
                float thisMin = (sumL2 * sumX - sumL * sumXL) / det;
                if (thisMin > 0) {
                    thisMin = 0;
                    thisScale = sumXL / sumL2;
                }
                float mad = 0;
                for (int i = 0; i < SUB; i++) {
                    mad += weights[i] * Math.abs(thisScale * laux[i] + thisMin - x[xo + i]);
                }
                if (mad < bestMad) {
                    System.arraycopy(laux, 0, L, lo, SUB);
                    bestMad = mad;
                    scale = thisScale;
                    min = thisMin;
                }
            }
        }
        mins[j] = -min;
        return scale;
    }

    // ------------------------------------------------------------------ dequantization / scalar dot

    /** Dequantizes the first {@code n} weights of the super-block at {@code base} into {@code dst[off .. off + n)}. */
    public static void dequantizeBlock(MemorySegment seg, long base, int n, float[] dst, int off) {
        float d = f16(seg, base + D_OFFSET);
        float dmin = f16(seg, base + DMIN_OFFSET);
        for (int j = 0; j < n / SUB; j++) {
            int s = seg.get(ValueLayout.JAVA_BYTE, base + SCALES_OFFSET + j) & 0xFF;
            float dl = d * (s & 0xF);
            float ml = dmin * (s >>> 4);
            long q = base + QS_OFFSET + (j >>> 3) * 32 + (j & 1) * 16;
            int shift = ((j & 7) >>> 1) << 1;
            for (int l = 0; l < SUB; l++) {
                int code = (seg.get(ValueLayout.JAVA_BYTE, q + l) >>> shift) & 3;
                dst[off + j * SUB + l] = dl * code - ml;
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
        float dmin = f16(seg, base + DMIN_OFFSET);
        float sumQ = 0; // Σ_j sc_j · Σ q·x
        float sumM = 0; // Σ_j m_j · Σ x
        for (int j = 0; j < n / SUB; j++) {
            int s = seg.get(ValueLayout.JAVA_BYTE, base + SCALES_OFFSET + j) & 0xFF;
            long q = base + QS_OFFSET + (j >>> 3) * 32 + (j & 1) * 16;
            int shift = ((j & 7) >>> 1) << 1;
            float qx = 0;
            float xs = 0;
            for (int l = 0; l < SUB; l++) {
                float xv = x[xo + j * SUB + l];
                qx += ((seg.get(ValueLayout.JAVA_BYTE, q + l) >>> shift) & 3) * xv;
                xs += xv;
            }
            sumQ += (s & 0xF) * qx;
            sumM += (s >>> 4) * xs;
        }
        return d * sumQ - dmin * sumM;
    }

    /** Scalar reference dot product of a {@code cols}-wide row starting at {@code base} with {@code x[xo ..)}. */
    public static float dotRow(MemorySegment seg, long base, int cols, float[] x, int xo) {
        float sum = 0;
        for (int c = 0; c < cols; c += BLOCK) {
            sum += dotBlock(seg, base + (long) (c / BLOCK) * BLOCK_BYTES, Math.min(BLOCK, cols - c), x, xo + c);
        }
        return sum;
    }

    private static int clamp(int v, int lo, int hi) {
        return v < lo ? lo : Math.min(v, hi);
    }
}
