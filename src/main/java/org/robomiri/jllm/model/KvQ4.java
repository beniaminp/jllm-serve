package org.robomiri.jllm.model;

/**
 * The arithmetic of the q4 KV cache ({@code backend.kv-dtype=q4}), shared by the CPU cache, the Metal host code and
 * the tests; the Metal kernels {@code kv_store_q4} and the {@code *_q4} attention kernels compute the same operations
 * in the same order, so a value stored by either backend has the same bits.
 *
 * <p><b>Rotation.</b> Each kv head's K row (after QK-norm and RoPE) is multiplied by the Walsh–Hadamard matrix of
 * {@link #block} = the largest power of two dividing {@code headDim} (the whole row for 64 / 128 / 256, blocks of 32 for
 * 96), <em>unnormalized</em>: only additions and subtractions, {@code H·H = B·I}. The queries get the same rotation
 * inside attention and the score scale is divided by {@code B} (a power of two, so exactly):
 * {@code (Hq)·(Hk) / B = q·k}. The rotation spreads a few large channels over the whole block, which is what makes
 * 4 bits enough (docs/07 §7.13).
 *
 * <p><b>Quantization.</b> Groups of {@link #GROUP} = 32 consecutive values of a (rotated) row, symmetric with 16
 * levels at the odd multiples of half a step ("mid-rise"): {@code d = half(max|x| / 7.5)},
 * {@code code = clamp(rint(x / d + 7.5), 0, 15)}, {@code x̂ = (code - 7.5) · d}. The levels are
 * {@code ±0.5d, ±1.5d, ..., ±7.5d}: both signs reach {@code ±max|x|} and every value is within half a step
 * {@code d / 2 = max|x| / 15} (unlike the ggml Q4_0 rule {@code d = m / -8}, which leaves the side opposite to the
 * largest value 7 levels and clips it, and measured clearly worse here: docs/07 §7.13). In a stored group of 16 bytes,
 * byte {@code j} holds element {@code j} in its low nibble and element {@code j + 16} in its high nibble (the ggml
 * Q4_0 order), and the scale is an IEEE half.
 *
 * <p><b>Row layout</b> (Metal cache rows and saved caches): {@code kvDim / 2} bytes of nibbles (head after head, group
 * after group) followed by {@code kvDim / 32} halves (the scales in the same order): 0.5625 bytes per value.
 */
public final class KvQ4 {
    /** Values per quantization group (and per scale). */
    public static final int GROUP = 32;

    /**
     * Whether keys (and queries) are Hadamard-rotated: on, except for the accuracy study of docs/07 §7.13, which sets
     * {@code -Djllm.kv.q4.rotate=false} to measure what the rotation buys. Read once; the CPU cache and the Metal
     * kernels follow it alike.
     */
    public static final boolean ROTATE = !"false".equalsIgnoreCase(System.getProperty("jllm.kv.q4.rotate"));

    private KvQ4() {
    }

    /** What the score scale is divided by: {@link #block} with the rotation, 1 without. */
    public static int scaleDivisor(int headDim) {
        return ROTATE ? block(headDim) : 1;
    }

    /** Whether a head width can be stored as q4 (whole groups); other widths fall back to q8. */
    public static boolean supports(int headDim) {
        return headDim % GROUP == 0;
    }

    /** The Walsh–Hadamard block: the largest power of two dividing {@code headDim}. */
    public static int block(int headDim) {
        return Integer.lowestOneBit(headDim);
    }

    /**
     * Unnormalized Walsh–Hadamard transform of each block of {@link #block} values of {@code x[o .. o + headDim)}, in
     * place: butterflies of stride 1, 2, 4, ... in that order, {@code (a, b) -> (a + b, a - b)}.
     */
    public static void rotate(float[] x, int o, int headDim) {
        int b = block(headDim);
        for (int bo = o; bo < o + headDim; bo += b) {
            for (int h = 1; h < b; h <<= 1) {
                for (int i = bo; i < bo + b; i += 2 * h) {
                    for (int j = i; j < i + h; j++) {
                        float u = x[j];
                        float w = x[j + h];
                        x[j] = u + w;
                        x[j + h] = u - w;
                    }
                }
            }
        }
    }

    /** The scale of the group {@code x[o .. o + 32)} as half bits: {@code half(max|x| / 7.5)}. */
    public static short scaleBits(float[] x, int o) {
        float a = 0f;
        for (int i = o; i < o + GROUP; i++) {
            a = Math.max(a, Math.abs(x[i]));
        }
        return Float.floatToFloat16(a / 7.5f);
    }

    /** The 4-bit code of {@code x} with scale {@code d}: {@code clamp(rint(x / d + 7.5), 0, 15)}, 8 when d is 0. */
    public static int code(float x, float d) {
        if (d == 0f) {
            return 8;
        }
        return (int) Math.max(0f, Math.min(15f, (float) Math.rint(x / d + 7.5f)));
    }

    /** The value a code stands for: {@code (code - 7.5) · d}. */
    public static float value(int code, float d) {
        return (code - 7.5f) * d;
    }

    /**
     * Quantizes the {@code n} values {@code x[o ..]} (whole groups) into the row layout: nibbles at
     * {@code nib[no + group * 16 ..]}, scales as halves at {@code sc[so + group]}.
     */
    public static void quantize(float[] x, int o, int n, byte[] nib, int no, short[] sc, int so) {
        for (int gi = 0; gi < n / GROUP; gi++) {
            int go = o + gi * GROUP;
            short bits = scaleBits(x, go);
            float d = Float.float16ToFloat(bits);
            sc[so + gi] = bits;
            for (int j = 0; j < GROUP / 2; j++) {
                nib[no + gi * 16 + j] = (byte) (code(x[go + j], d) | code(x[go + j + 16], d) << 4);
            }
        }
    }

    /** Inverse of {@link #quantize}: the {@code n} dequantized values into {@code x[o ..]}. */
    public static void dequantize(byte[] nib, int no, short[] sc, int so, int n, float[] x, int o) {
        for (int gi = 0; gi < n / GROUP; gi++) {
            float d = Float.float16ToFloat(sc[so + gi]);
            for (int j = 0; j < GROUP / 2; j++) {
                int b = nib[no + gi * 16 + j] & 0xFF;
                x[o + gi * GROUP + j] = value(b & 15, d);
                x[o + gi * GROUP + j + 16] = value(b >>> 4, d);
            }
        }
    }
}
