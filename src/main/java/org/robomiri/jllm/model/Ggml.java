package org.robomiri.jllm.model;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * The ggml block formats the engine computes with in place (byte-compatible with llama.cpp's {@code block_q4_0},
 * {@code block_q8_0}, {@code block_q4_K} and {@code block_q5_K}; {@link Q2K} is the fifth), with scalar reference
 * dequantizers. They are the semantic definition the SIMD and Metal kernels are checked against, and the code
 * behind {@link QTensor#dequantizeRow} for these types. Layouts (all little endian, scales are f16):
 *
 * <pre>
 *   Q4_0  18 B / 32 w   f16 d | u8 qs[16]                      w[j] = (lo(qs[j]) - 8) d,  w[j+16] = (hi(qs[j]) - 8) d
 *   Q8_0  34 B / 32 w   f16 d | i8 qs[32]                      w[j] = qs[j] d
 *   Q4_K 144 B / 256 w  f16 d | f16 dmin | u8 sc[12] | u8 qs[128]
 *   Q5_K 176 B / 256 w  f16 d | f16 dmin | u8 sc[12] | u8 qh[32] | u8 qs[128]
 * </pre>
 * In the K-quants eight 32-weight sub-blocks have a 6-bit scale and a 6-bit min packed into {@code sc} (see
 * {@link #scaleMin}); {@code w = d sc q - dmin m}. Sub-blocks {@code 2g} and {@code 2g+1} share the 32 bytes
 * {@code qs[32g ..)} as low and high nibbles; in Q5_K the fifth bit of weight {@code l} of sub-block {@code j} is
 * bit {@code j} of {@code qh[l]}.
 */
public final class Ggml {

    public static final int Q4_0_BYTES = 18;
    public static final int Q8_0_BYTES = 34;
    public static final int QK_K = 256;
    public static final int Q4_K_BYTES = 144;
    public static final int Q5_K_BYTES = 176;
    /** Offset of {@code dmin} in a K-quant block; {@code d} is at 0. */
    public static final int K_DMIN = 2;
    public static final int K_SCALES = 4;
    public static final int Q4_K_QS = 16;
    public static final int Q5_K_QH = 16;
    public static final int Q5_K_QS = 48;

    private static final ValueLayout.OfShort LE16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE32 = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private Ggml() {
    }

    /** The f16 at byte offset {@code off} (2-byte alignment is not guaranteed in a 18- or 34-byte block layout). */
    public static float f16(MemorySegment seg, long off) {
        return Float.float16ToFloat(seg.get(LE16, off));
    }

    public static float bf16(MemorySegment seg, long off) {
        return Float.intBitsToFloat((seg.get(LE16, off) & 0xFFFF) << 16);
    }

    private static int u8(MemorySegment seg, long off) {
        return seg.get(ValueLayout.JAVA_BYTE, off) & 0xFF;
    }

    /** Dequantizes the Q4_0 row of {@code cols} weights at {@code off} into {@code dst[o ..)}. */
    public static void dequantQ4_0(MemorySegment seg, long off, int cols, float[] dst, int o) {
        for (int b = 0; b < cols / 32; b++, off += Q4_0_BYTES, o += 32) {
            float d = f16(seg, off);
            for (int j = 0; j < 16; j++) {
                int v = u8(seg, off + 2 + j);
                dst[o + j] = ((v & 0xF) - 8) * d;
                dst[o + j + 16] = ((v >>> 4) - 8) * d;
            }
        }
    }

    public static void dequantQ8_0(MemorySegment seg, long off, int cols, float[] dst, int o) {
        for (int b = 0; b < cols / 32; b++, off += Q8_0_BYTES, o += 32) {
            float d = f16(seg, off);
            for (int j = 0; j < 32; j++) {
                dst[o + j] = seg.get(ValueLayout.JAVA_BYTE, off + 2 + j) * d;
            }
        }
    }

    /**
     * The 6-bit scale (index 0) and min (index 1) of sub-block {@code j} of a K-quant block whose 12 scale bytes
     * start at {@code sc} (llama.cpp's {@code get_scale_min_k4}).
     */
    public static void scaleMin(MemorySegment seg, long sc, int j, int[] out) {
        if (j < 4) {
            out[0] = u8(seg, sc + j) & 63;
            out[1] = u8(seg, sc + j + 4) & 63;
        } else {
            int a = u8(seg, sc + j + 4);
            out[0] = (a & 0xF) | ((u8(seg, sc + j - 4) >>> 6) << 4);
            out[1] = (a >>> 4) | ((u8(seg, sc + j) >>> 6) << 4);
        }
    }

    public static void dequantQ4_K(MemorySegment seg, long off, int cols, float[] dst, int o) {
        int[] sm = new int[2];
        for (int b = 0; b < cols / QK_K; b++, off += Q4_K_BYTES) {
            float d = f16(seg, off);
            float dmin = f16(seg, off + K_DMIN);
            long qs = off + Q4_K_QS;
            for (int g = 0; g < 4; g++, qs += 32) {
                scaleMin(seg, off + K_SCALES, 2 * g, sm);
                float d1 = d * sm[0];
                float m1 = dmin * sm[1];
                scaleMin(seg, off + K_SCALES, 2 * g + 1, sm);
                float d2 = d * sm[0];
                float m2 = dmin * sm[1];
                for (int l = 0; l < 32; l++) {
                    int v = u8(seg, qs + l);
                    dst[o + l] = d1 * (v & 0xF) - m1;
                    dst[o + l + 32] = d2 * (v >>> 4) - m2;
                }
                o += 64;
            }
        }
    }

    public static void dequantQ5_K(MemorySegment seg, long off, int cols, float[] dst, int o) {
        int[] sm = new int[2];
        for (int b = 0; b < cols / QK_K; b++, off += Q5_K_BYTES) {
            float d = f16(seg, off);
            float dmin = f16(seg, off + K_DMIN);
            long qh = off + Q5_K_QH;
            long qs = off + Q5_K_QS;
            for (int g = 0; g < 4; g++, qs += 32) {
                scaleMin(seg, off + K_SCALES, 2 * g, sm);
                float d1 = d * sm[0];
                float m1 = dmin * sm[1];
                scaleMin(seg, off + K_SCALES, 2 * g + 1, sm);
                float d2 = d * sm[0];
                float m2 = dmin * sm[1];
                for (int l = 0; l < 32; l++) {
                    int v = u8(seg, qs + l);
                    int h = u8(seg, qh + l);
                    dst[o + l] = d1 * ((v & 0xF) + (((h >>> (2 * g)) & 1) << 4)) - m1;
                    dst[o + l + 32] = d2 * ((v >>> 4) + (((h >>> (2 * g + 1)) & 1) << 4)) - m2;
                }
                o += 64;
            }
        }
    }

    // ------------------------------------------------------------------ formats the engine only converts (to Q8_0)

    /** Q4_1: f16 d | f16 m | u8 qs[16] (20 B / 32 w), w = q d + m. */
    public static void dequantQ4_1(MemorySegment seg, long off, int cols, float[] dst, int o) {
        for (int b = 0; b < cols / 32; b++, off += 20, o += 32) {
            float d = f16(seg, off);
            float m = f16(seg, off + 2);
            for (int j = 0; j < 16; j++) {
                int v = u8(seg, off + 4 + j);
                dst[o + j] = (v & 0xF) * d + m;
                dst[o + j + 16] = (v >>> 4) * d + m;
            }
        }
    }

    /** Q5_0: f16 d | u8 qh[4] | u8 qs[16] (22 B / 32 w), w = (q5 - 16) d. */
    public static void dequantQ5_0(MemorySegment seg, long off, int cols, float[] dst, int o) {
        for (int b = 0; b < cols / 32; b++, off += 22, o += 32) {
            float d = f16(seg, off);
            int qh = seg.get(LE32, off + 2);
            for (int j = 0; j < 16; j++) {
                int v = u8(seg, off + 6 + j);
                dst[o + j] = (((v & 0xF) | (((qh >>> j) & 1) << 4)) - 16) * d;
                dst[o + j + 16] = (((v >>> 4) | (((qh >>> (j + 16)) & 1) << 4)) - 16) * d;
            }
        }
    }

    /** Q5_1: f16 d | f16 m | u8 qh[4] | u8 qs[16] (24 B / 32 w), w = q5 d + m. */
    public static void dequantQ5_1(MemorySegment seg, long off, int cols, float[] dst, int o) {
        for (int b = 0; b < cols / 32; b++, off += 24, o += 32) {
            float d = f16(seg, off);
            float m = f16(seg, off + 2);
            int qh = seg.get(LE32, off + 4);
            for (int j = 0; j < 16; j++) {
                int v = u8(seg, off + 8 + j);
                dst[o + j] = ((v & 0xF) | (((qh >>> j) & 1) << 4)) * d + m;
                dst[o + j + 16] = ((v >>> 4) | (((qh >>> (j + 16)) & 1) << 4)) * d + m;
            }
        }
    }

    /**
     * Q3_K: u8 hmask[32] | u8 qs[64] | u8 scales[12] | f16 d (110 B / 256 w). Sixteen 6-bit scales (minus 32), each
     * for 16 weights; a weight is the 2-bit quant minus 4 unless its hmask bit is set.
     */
    public static void dequantQ3_K(MemorySegment seg, long off, int cols, float[] dst, int o) {
        int[] sc = new int[16];
        for (int b = 0; b < cols / QK_K; b++, off += 110) {
            float d = f16(seg, off + 108);
            // unpack the 12 scale bytes into 16 6-bit values (llama.cpp's kmask1/kmask2 shuffle)
            int a0 = seg.get(LE32, off + 96);
            int a1 = seg.get(LE32, off + 100);
            int tmp = seg.get(LE32, off + 104);
            int k1 = 0x03030303;
            int k2 = 0x0f0f0f0f;
            int s2 = ((a0 >>> 4) & k2) | (((tmp >>> 4) & k1) << 4);
            int s3 = ((a1 >>> 4) & k2) | (((tmp >>> 6) & k1) << 4);
            int s0 = (a0 & k2) | (((tmp >>> 0) & k1) << 4);
            int s1 = (a1 & k2) | (((tmp >>> 2) & k1) << 4);
            int[] words = {s0, s1, s2, s3};
            for (int i = 0; i < 16; i++) {
                sc[i] = (byte) ((words[i >> 2] >>> (8 * (i & 3))) & 0xFF);
            }
            int is = 0;
            long q = off + 32;
            int m = 1;
            for (int n = 0; n < 2; n++, q += 32) {
                int shift = 0;
                for (int j = 0; j < 4; j++, shift += 2, m <<= 1) {
                    float dl = d * (sc[is++] - 32);
                    for (int l = 0; l < 16; l++) {
                        int hm = u8(seg, off + l);
                        dst[o++] = dl * (((u8(seg, q + l) >>> shift) & 3) - ((hm & m) != 0 ? 0 : 4));
                    }
                    dl = d * (sc[is++] - 32);
                    for (int l = 0; l < 16; l++) {
                        int hm = u8(seg, off + 16 + l);
                        dst[o++] = dl * (((u8(seg, q + 16 + l) >>> shift) & 3) - ((hm & m) != 0 ? 0 : 4));
                    }
                }
            }
        }
    }

    /** Q6_K: u8 ql[128] | u8 qh[64] | i8 scales[16] | f16 d (210 B / 256 w), w = d sc (q6 - 32). */
    public static void dequantQ6_K(MemorySegment seg, long off, int cols, float[] dst, int o) {
        for (int b = 0; b < cols / QK_K; b++, off += 210, o += QK_K) {
            float d = f16(seg, off + 208);
            long ql = off;
            long qh = off + 128;
            long sc = off + 192;
            int y = o;
            for (int n = 0; n < QK_K; n += 128, ql += 64, qh += 32, sc += 8, y += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int h = u8(seg, qh + l);
                    int lo0 = u8(seg, ql + l);
                    int lo1 = u8(seg, ql + l + 32);
                    int q1 = ((lo0 & 0xF) | (((h >>> 0) & 3) << 4)) - 32;
                    int q2 = ((lo1 & 0xF) | (((h >>> 2) & 3) << 4)) - 32;
                    int q3 = ((lo0 >>> 4) | (((h >>> 4) & 3) << 4)) - 32;
                    int q4 = ((lo1 >>> 4) | (((h >>> 6) & 3) << 4)) - 32;
                    dst[y + l] = d * seg.get(ValueLayout.JAVA_BYTE, sc + is) * q1;
                    dst[y + l + 32] = d * seg.get(ValueLayout.JAVA_BYTE, sc + is + 2) * q2;
                    dst[y + l + 64] = d * seg.get(ValueLayout.JAVA_BYTE, sc + is + 4) * q3;
                    dst[y + l + 96] = d * seg.get(ValueLayout.JAVA_BYTE, sc + is + 6) * q4;
                }
            }
        }
    }
}
