package org.robomiri.jllm.cpu;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.model.Ggml;
import org.robomiri.jllm.model.Q6K;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * The K-quants of GGUF files (Q4_K, Q5_K, Q6_K: what Q4_K_M / Q5_K_M models are made of) on the operations the
 * {@link Int8Kernels} use: 128-bit loads, lanewise integer arithmetic, same-size reinterpretation and the 2^23
 * magic constant instead of int-to-float conversions. The float kernels of {@link GgufKernels} widen bytes with
 * {@code castShape}; with 512-bit vectors C2 spent minutes compiling them (loop predication on the widening casts,
 * see docs/07 §7.10), during which the model ran interpreted at a tenth of a token per second.
 *
 * <p><b>Decode</b> ({@code n == 1}) multiplies each row with the int8 activation of {@link Int8Kernels#quantize}
 * (Q8_0 blocks of 32 in its permuted layout): per 16 weights an exact 16-bit integer dot product, scaled per
 * sub-block. Q4_K / Q5_K store {@code w = d·sc·q − dmin·m}; the min term needs the sum of each activation block,
 * which {@link #blockSums} computes once per matmul (llama.cpp's q4_K × q8_K scheme).
 *
 * <p><b>Prefill</b> dequantizes rows to floats for the float GEMM ({@link VectorKernels#dot4x4}). Writing them in
 * natural order would need a lane shuffle or a widening cast; instead each 32-weight block is written in the order
 * the integer unpacking produces ({@link #permutedIndex}) and the activations are permuted the same way
 * ({@link #permute}), once per matmul: a dot product does not depend on the order of its terms.
 */
public final class KQuantKernels {

    private static final VectorSpecies<Short> S8 = ShortVector.SPECIES_128;
    private static final VectorSpecies<Integer> I4 = IntVector.SPECIES_128;
    private static final VectorSpecies<Float> F4 = FloatVector.SPECIES_128;
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;
    private static final ValueLayout.OfInt LE32 = ValueLayout.JAVA_INT_UNALIGNED.withOrder(LE);
    private static final float MAGIC = 12582912f; // 1.5 * 2^23
    private static final int MAGIC_BITS = Float.floatToRawIntBits(MAGIC);

    private KQuantKernels() {
    }

    /**
     * Whether {@code w} can use these kernels: a K-quant with whole super-blocks per row; for a split tensor (two
     * tensors seen as one, such as fused gate / up projections) both parts.
     */
    public static boolean supports(QTensor w) {
        if (w.second != null) {
            return supports(w.partOf(0)) && supports(w.second);
        }
        return Int8Kernels.SUPPORTED && w.cols % Ggml.QK_K == 0
                && (w.type == QType.Q4_K || w.type == QType.Q5_K || w.type == QType.Q6_K);
    }

    // ------------------------------------------------------------------ helpers

    /** Pairs of adjacent short lanes summed into int lanes (sign-extended by shifts). */
    private static IntVector widenSum(ShortVector p) {
        IntVector pi = p.reinterpretAsInts();
        return pi.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16).add(pi.lanewise(VectorOperators.ASHR, 16));
    }

    /** Exact int-to-float for |i| < 2^22. */
    private static FloatVector toFloat(IntVector i) {
        return i.add(MAGIC_BITS).reinterpretAsFloats().sub(MAGIC);
    }

    /** Scales (byte j = scale of sub-block j) of a Q4_K / Q5_K super-block, 6 bits each. */
    static long scales(MemorySegment seg, long sc) {
        int u0 = seg.get(LE32, sc);
        int u2 = seg.get(LE32, sc + 8);
        int lo = u0 & 0x3f3f3f3f;
        int hi = (u2 & 0x0f0f0f0f) | (((u0 >>> 6) & 0x03030303) << 4);
        return (lo & 0xFFFFFFFFL) | ((long) hi << 32);
    }

    /** Mins (byte j = min of sub-block j) of a Q4_K / Q5_K super-block. */
    static long mins(MemorySegment seg, long sc) {
        int u1 = seg.get(LE32, sc + 4);
        int u2 = seg.get(LE32, sc + 8);
        int lo = u1 & 0x3f3f3f3f;
        int hi = ((u2 >>> 4) & 0x0f0f0f0f) | (((u1 >>> 6) & 0x03030303) << 4);
        return (lo & 0xFFFFFFFFL) | ((long) hi << 32);
    }

    private static int b(long packed, int j) {
        return (int) (packed >>> (8 * j)) & 0xFF;
    }

    /** {@code sums[b] = Σ xq} of each 32-element block of the quantized activation (for the K-quant min terms). */
    public static void blockSums(short[] xq, int n, int[] sums) {
        for (int b = 0; b < n / QType.BLOCK; b++) {
            ShortVector a = ShortVector.zero(S8);
            for (int i = 0; i < QType.BLOCK; i += 8) {
                a = a.add(ShortVector.fromArray(S8, xq, b * QType.BLOCK + i)); // |sum| <= 4 * 127 per lane
            }
            sums[b] = widenSum(a).reduceLanes(VectorOperators.ADD);
        }
    }

    // ------------------------------------------------------------------ decode: int8 activations

    /**
     * Dot product of row {@code r} of {@code w} with the int8 activation {@code xq[xo ..]} (block scales {@code xs},
     * block sums {@code xsum}, both from block {@code xo / 32}); {@code xo} is a multiple of 32.
     */
    public static float dotRow(QTensor w, int r, short[] xq, float[] xs, int[] xsum, int xo) {
        if (w.second != null) {
            return dotRow(w.partOf(r), w.partRow(r), xq, xs, xsum, xo);
        }
        int xb = xo / QType.BLOCK;
        return switch (w.type) {
            case Q4_K -> dotQ4K(w.data, w.rowOffset(r), w.cols / Ggml.QK_K, xq, xs, xsum, xb);
            case Q5_K -> dotQ5K(w.data, w.rowOffset(r), w.cols / Ggml.QK_K, xq, xs, xsum, xb);
            case Q6_K -> dotQ6K(w.data, w.rowOffset(r), w.cols / Ggml.QK_K, xq, xs, xb);
            default -> throw new IllegalArgumentException(w.type.name());
        };
    }

    /** Per-sub-block scales (weight scale × activation scale) of the current super-block, filled before its loop. */
    private static final ThreadLocal<float[]> WS = ThreadLocal.withInitial(() -> new float[16]);

    /**
     * Q4_K row. A 32-byte group {@code g} of a super-block holds sub-block {@code 2g} in its low nibbles and
     * {@code 2g+1} in its high nibbles; loaded as 16-byte halves of 8 shorts, the four nibbles of a short lane are
     * (sub-block, even / odd byte) pairs whose activations sit contiguously in the permuted layout. One segment load
     * per inner iteration; the per-sub-block scales and the min terms are computed in a scalar prologue per
     * super-block, so the vector loop only broadcasts a float.
     */
    static float dotQ4K(MemorySegment seg, long q, int nb, short[] xq, float[] xs, int[] xsum, int blk0) {
        float[] ws = WS.get();
        FloatVector acc = FloatVector.zero(F4);
        FloatVector magic = FloatVector.broadcast(F4, MAGIC);
        IntVector magicBits = IntVector.broadcast(I4, MAGIC_BITS);
        float minus = 0;
        for (int sb = 0; sb < nb; sb++, q += Ggml.Q4_K_BYTES) {
            int blk = blk0 + sb * 8;
            float d = Ggml.f16(seg, q);
            float dmin = Ggml.f16(seg, q + Ggml.K_DMIN);
            long sc = scales(seg, q + Ggml.K_SCALES);
            long mn = mins(seg, q + Ggml.K_SCALES);
            for (int j = 0; j < 8; j++) {
                float x = xs[blk + j];
                minus += dmin * b(mn, j) * x * xsum[blk + j];
                ws[j] = d * b(sc, j) * x;
            }
            long qs = q + Ggml.Q4_K_QS;
            for (int i = 0; i < 8; i++) { // (group g = i/2, half h = i%2)
                int g = i >> 1;
                ShortVector v = ShortVector.fromMemorySegment(S8, seg, qs + 16L * i, LE);
                int xa = (blk + 2 * g) * QType.BLOCK + 16 * (i & 1);
                int xb = xa + QType.BLOCK;
                ShortVector pa = v.and((short) 0x0F).mul(ShortVector.fromArray(S8, xq, xa))
                        .add(v.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F).mul(ShortVector.fromArray(S8, xq, xa + 8)));
                ShortVector pb = v.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F).mul(ShortVector.fromArray(S8, xq, xb))
                        .add(v.lanewise(VectorOperators.LSHR, 12).mul(ShortVector.fromArray(S8, xq, xb + 8)));
                acc = widenSum(pa).add(magicBits).reinterpretAsFloats().sub(magic).fma(FloatVector.broadcast(F4, ws[2 * g]), acc);
                acc = widenSum(pb).add(magicBits).reinterpretAsFloats().sub(magic).fma(FloatVector.broadcast(F4, ws[2 * g + 1]), acc);
            }
        }
        return VectorKernels.laneSum4(acc) - minus;
    }

    /**
     * Q5_K row: Q4_K plus a fifth bit per weight from {@code qh} (bit {@code 2g} for the low sub-block, {@code 2g+1}
     * for the high). The 16-byte half {@code h} of every group shares one half of {@code qh}: the half is the outer
     * loop, so each vector has one definition (a vector chosen by a condition is boxed by C2).
     */
    static float dotQ5K(MemorySegment seg, long q, int nb, short[] xq, float[] xs, int[] xsum, int blk0) {
        float[] ws = WS.get();
        FloatVector acc = FloatVector.zero(F4);
        FloatVector magic = FloatVector.broadcast(F4, MAGIC);
        IntVector magicBits = IntVector.broadcast(I4, MAGIC_BITS);
        float minus = 0;
        for (int sb = 0; sb < nb; sb++, q += Ggml.Q5_K_BYTES) {
            int blk = blk0 + sb * 8;
            float d = Ggml.f16(seg, q);
            float dmin = Ggml.f16(seg, q + Ggml.K_DMIN);
            long sc = scales(seg, q + Ggml.K_SCALES);
            long mn = mins(seg, q + Ggml.K_SCALES);
            for (int j = 0; j < 8; j++) {
                float x = xs[blk + j];
                minus += dmin * b(mn, j) * x * xsum[blk + j];
                ws[j] = d * b(sc, j) * x;
            }
            long qs = q + Ggml.Q5_K_QS;
            for (int h = 0; h < 2; h++) {
                ShortVector hv = ShortVector.fromMemorySegment(S8, seg, q + Ggml.Q5_K_QH + 16L * h, LE);
                for (int g = 0; g < 4; g++) {
                    ShortVector v = ShortVector.fromMemorySegment(S8, seg, qs + 32L * g + 16L * h, LE);
                    for (int sub = 0; sub < 2; sub++) { // low nibbles: sub-block 2g, high: 2g+1
                        int xa = (blk + 2 * g + sub) * QType.BLOCK + 16 * h;
                        // both bytes of each short lane at once: 5-bit codes in bits 0..4 and 8..12
                        ShortVector w = v.lanewise(VectorOperators.LSHR, 4 * sub).and((short) 0x0F0F)
                                .or(hv.lanewise(VectorOperators.LSHR, 2 * g + sub).and((short) 0x0101).lanewise(VectorOperators.LSHL, 4));
                        ShortVector p = w.and((short) 0xFF).mul(ShortVector.fromArray(S8, xq, xa))
                                .add(w.lanewise(VectorOperators.LSHR, 8).mul(ShortVector.fromArray(S8, xq, xa + 8)));
                        acc = widenSum(p).add(magicBits).reinterpretAsFloats().sub(magic).fma(FloatVector.broadcast(F4, ws[2 * g + sub]), acc);
                    }
                }
            }
        }
        return VectorKernels.laneSum4(acc) - minus;
    }

    /**
     * Q6_K row: per 128-weight half of a super-block, group {@code g} (32 weights) takes the low ({@code g < 2}) or
     * high nibbles of {@code ql[32·(g&1) ..]} and bits {@code 2g, 2g+1} of {@code qh}; each 16-byte half of a group
     * is one 16-weight sub-block with its own int8 scale. Codes are {@code (nibble | high << 4) − 32}.
     */
    static float dotQ6K(MemorySegment seg, long q, int nb, short[] xq, float[] xs, int blk0) {
        float[] ws = WS.get();
        FloatVector acc = FloatVector.zero(F4);
        FloatVector magic = FloatVector.broadcast(F4, MAGIC);
        IntVector magicBits = IntVector.broadcast(I4, MAGIC_BITS);
        for (int sb = 0; sb < nb; sb++, q += Q6K.BLOCK_BYTES) {
            int blk0s = blk0 + sb * 8;
            float d = Q6K.f16(seg, q + Q6K.D_OFFSET);
            for (int k = 0; k < 16; k++) { // sub-block k of 16 weights: activation block k / 2
                ws[k] = d * seg.get(ValueLayout.JAVA_BYTE, q + Q6K.SCALES_OFFSET + k) * xs[blk0s + (k >> 1)];
            }
            for (int part = 0; part < 4; part++) { // (128-weight half, 16-byte half of its groups)
                int half = part >> 1;
                int hh = part & 1;
                long ql = q + Q6K.QL_OFFSET + 64L * half + 16L * hh;
                ShortVector hv = ShortVector.fromMemorySegment(S8, seg, q + Q6K.QH_OFFSET + 32L * half + 16L * hh, LE);
                int blk = blk0s + half * 4;
                for (int g = 0; g < 4; g++) {
                    ShortVector v = ShortVector.fromMemorySegment(S8, seg, ql + 32L * (g & 1), LE);
                    int lo = (g >> 1) * 4;
                    int xa = (blk + g) * QType.BLOCK + 16 * hh;
                    // both bytes of each short lane at once: 6-bit codes in bits 0..5 and 8..13
                    ShortVector w = v.lanewise(VectorOperators.LSHR, lo).and((short) 0x0F0F)
                            .or(hv.lanewise(VectorOperators.LSHR, 2 * g).and((short) 0x0303).lanewise(VectorOperators.LSHL, 4));
                    ShortVector p = w.and((short) 0xFF).sub((short) 32).mul(ShortVector.fromArray(S8, xq, xa))
                            .add(w.lanewise(VectorOperators.LSHR, 8).sub((short) 32).mul(ShortVector.fromArray(S8, xq, xa + 8)));
                    acc = widenSum(p).add(magicBits).reinterpretAsFloats().sub(magic).fma(FloatVector.broadcast(F4, ws[8 * half + 2 * g + hh]), acc);
                }
            }
        }
        return VectorKernels.laneSum4(acc);
    }

    // ------------------------------------------------------------------ prefill: permuted float rows

    /**
     * Where element {@code j} (0..31) of a 32-weight block lands in a permuted row: halves of 16, then even / odd
     * elements ({@code Int8Kernels}' short-lane layout), then the even / odd short lanes of each int lane.
     */
    public static int permutedIndex(int j) {
        int k = (j & 15) >> 1;
        return (j & 16) + (j & 1) * 8 + (k & 1) * 4 + (k >> 1);
    }

    /** {@code dst[t * cols + p(i)] = src[t * cols + i]} for {@code n} rows of {@code cols} (a multiple of 32). */
    public static void permute(float[] src, int n, int cols, float[] dst) {
        for (int t = 0; t < n; t++) {
            int o = t * cols;
            for (int b = 0; b < cols; b += QType.BLOCK) {
                for (int j = 0; j < QType.BLOCK; j++) {
                    dst[o + b + PERM[j]] = src[o + b + j];
                }
            }
        }
    }

    private static final int[] PERM = new int[QType.BLOCK];

    static {
        for (int j = 0; j < QType.BLOCK; j++) {
            PERM[j] = permutedIndex(j);
        }
    }

    /** Dequantizes row {@code r} of {@code w} into {@code dst[off ..]} in the permuted order. */
    public static void dequantPermuted(QTensor w, int r, float[] dst, int off) {
        if (w.second != null) {
            dequantPermuted(w.partOf(r), w.partRow(r), dst, off);
            return;
        }
        MemorySegment seg = w.data;
        long q = w.rowOffset(r);
        int nb = w.cols / Ggml.QK_K;
        switch (w.type) {
            case Q4_K -> dequantQ4K(seg, q, nb, dst, off);
            case Q5_K -> dequantQ5K(seg, q, nb, dst, off);
            case Q6_K -> dequantQ6K(seg, q, nb, dst, off);
            default -> throw new IllegalArgumentException(w.type.name());
        }
    }

    /** Stores the 8 short lanes of {@code v} as {@code v * scale + bias} at {@code dst[o .. o+8)}: even lanes, then odd lanes. */
    private static void store8(ShortVector v, float scale, float bias, float[] dst, int o) {
        IntVector vi = v.reinterpretAsInts();
        FloatVector s = FloatVector.broadcast(F4, scale);
        FloatVector m = FloatVector.broadcast(F4, bias);
        toFloat(vi.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)).fma(s, m).intoArray(dst, o);
        toFloat(vi.lanewise(VectorOperators.ASHR, 16)).fma(s, m).intoArray(dst, o + 4);
    }

    private static void dequantQ4K(MemorySegment seg, long q, int nb, float[] dst, int off) {
        for (int sb = 0; sb < nb; sb++, q += Ggml.Q4_K_BYTES) {
            float d = Ggml.f16(seg, q);
            float dmin = Ggml.f16(seg, q + Ggml.K_DMIN);
            long sc = scales(seg, q + Ggml.K_SCALES);
            long mn = mins(seg, q + Ggml.K_SCALES);
            long qs = q + Ggml.Q4_K_QS;
            for (int i = 0; i < 8; i++) {
                int g = i >> 1;
                int h = i & 1;
                ShortVector v = ShortVector.fromMemorySegment(S8, seg, qs + 16L * i, LE);
                int oa = off + (sb * 8 + 2 * g) * QType.BLOCK + 16 * h;
                int ob = oa + QType.BLOCK;
                float da = d * b(sc, 2 * g);
                float ma = -dmin * b(mn, 2 * g);
                float db = d * b(sc, 2 * g + 1);
                float mb = -dmin * b(mn, 2 * g + 1);
                store8(v.and((short) 0x0F), da, ma, dst, oa);
                store8(v.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F), da, ma, dst, oa + 8);
                store8(v.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F), db, mb, dst, ob);
                store8(v.lanewise(VectorOperators.LSHR, 12).and((short) 0x0F), db, mb, dst, ob + 8);
            }
        }
    }

    private static void dequantQ5K(MemorySegment seg, long q, int nb, float[] dst, int off) {
        for (int sb = 0; sb < nb; sb++, q += Ggml.Q5_K_BYTES) {
            float d = Ggml.f16(seg, q);
            float dmin = Ggml.f16(seg, q + Ggml.K_DMIN);
            long sc = scales(seg, q + Ggml.K_SCALES);
            long mn = mins(seg, q + Ggml.K_SCALES);
            long qs = q + Ggml.Q5_K_QS;
            for (int h = 0; h < 2; h++) {
                ShortVector hv = ShortVector.fromMemorySegment(S8, seg, q + Ggml.Q5_K_QH + 16L * h, LE);
                for (int g = 0; g < 4; g++) {
                    ShortVector v = ShortVector.fromMemorySegment(S8, seg, qs + 32L * g + 16L * h, LE);
                    int oa = off + (sb * 8 + 2 * g) * QType.BLOCK + 16 * h;
                    int ob = oa + QType.BLOCK;
                    float da = d * b(sc, 2 * g);
                    float ma = -dmin * b(mn, 2 * g);
                    float db = d * b(sc, 2 * g + 1);
                    float mb = -dmin * b(mn, 2 * g + 1);
                    store8(v.and((short) 0x0F).or(hv.lanewise(VectorOperators.LSHR, 2 * g).and((short) 1).lanewise(VectorOperators.LSHL, 4)),
                            da, ma, dst, oa);
                    store8(v.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
                            .or(hv.lanewise(VectorOperators.LSHR, 8 + 2 * g).and((short) 1).lanewise(VectorOperators.LSHL, 4)), da, ma, dst, oa + 8);
                    store8(v.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F)
                            .or(hv.lanewise(VectorOperators.LSHR, 2 * g + 1).and((short) 1).lanewise(VectorOperators.LSHL, 4)), db, mb, dst, ob);
                    store8(v.lanewise(VectorOperators.LSHR, 12).and((short) 0x0F)
                            .or(hv.lanewise(VectorOperators.LSHR, 9 + 2 * g).and((short) 1).lanewise(VectorOperators.LSHL, 4)), db, mb, dst, ob + 8);
                }
            }
        }
    }

    private static void dequantQ6K(MemorySegment seg, long q, int nb, float[] dst, int off) {
        for (int sb = 0; sb < nb; sb++, q += Q6K.BLOCK_BYTES) {
            float d = Q6K.f16(seg, q + Q6K.D_OFFSET);
            for (int part = 0; part < 4; part++) {
                int half = part >> 1;
                int hh = part & 1;
                long ql = q + Q6K.QL_OFFSET + 64L * half + 16L * hh;
                long scs = q + Q6K.SCALES_OFFSET + 8L * half + hh;
                ShortVector hv = ShortVector.fromMemorySegment(S8, seg, q + Q6K.QH_OFFSET + 32L * half + 16L * hh, LE);
                for (int g = 0; g < 4; g++) {
                    ShortVector v = ShortVector.fromMemorySegment(S8, seg, ql + 32L * (g & 1), LE);
                    int lo = (g >> 1) * 4;
                    float s = d * seg.get(ValueLayout.JAVA_BYTE, scs + 2 * g);
                    int oo = off + (sb * 8 + half * 4 + g) * QType.BLOCK + 16 * hh;
                    store8(v.lanewise(VectorOperators.LSHR, lo).and((short) 0x0F)
                            .or(hv.lanewise(VectorOperators.LSHR, 2 * g).and((short) 3).lanewise(VectorOperators.LSHL, 4)).sub((short) 32), s, 0f, dst, oo);
                    store8(v.lanewise(VectorOperators.LSHR, 8 + lo).and((short) 0x0F)
                            .or(hv.lanewise(VectorOperators.LSHR, 8 + 2 * g).and((short) 3).lanewise(VectorOperators.LSHL, 4)).sub((short) 32), s, 0f, dst, oo + 8);
                }
            }
        }
    }
}
