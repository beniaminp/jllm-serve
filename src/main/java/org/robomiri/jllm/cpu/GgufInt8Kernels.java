package org.robomiri.jllm.cpu;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.model.Ggml;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Int8-activation dot products ({@link Int8Kernels}' scheme: the activation quantized once per matmul to Q8_0 in the
 * permuted block layout, every weight block's exact integer product scaled by {@code weightScale * activationScale})
 * for the other ggml 32-weight formats of GGUF files: Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4 and the 32-weight sub-blocks of
 * IQ4_XS.
 *
 * <ul>
 *   <li><b>Q4_1 / Q5_1</b>: {@code Σ (d q + m) x = d Σ q x + m Σ x}; the integer sum of the activation block comes
 *       from {@link KQuantKernels#blockSums} (the same {@code xsum} the K-quants use).</li>
 *   <li><b>Q5_0</b>: {@code (q5 - 16) d}; the fifth bit of weight {@code j} is bit {@code j} of {@code qh}, expanded
 *       in the byte domain (a byte shuffle and per-lane shifts) and handled like the table formats' bytes.</li>
 *   <li><b>IQ4_NL, MXFP4, IQ4_XS</b>: the nibbles index a 16-entry int8 table (a byte shuffle); the looked-up bytes
 *       are then multiplied as Q8_0 bytes are in {@link Int8Kernels#dotRowQ8GxQ8}.</li>
 * </ul>
 * Lane arithmetic never overflows a short: 5-bit codes give at most 4 x 31 x 127 per lane, the table bytes at most
 * 2 x 127 x 127 before they are widened to ints.
 */
public final class GgufInt8Kernels {

    private static final VectorSpecies<Short> S8 = ShortVector.SPECIES_128;
    private static final VectorSpecies<Byte> B16 = ByteVector.SPECIES_128;
    private static final VectorSpecies<Integer> I4 = IntVector.SPECIES_128;
    private static final VectorSpecies<Float> F4 = FloatVector.SPECIES_128;
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;
    private static final ValueLayout.OfInt LE32 = ValueLayout.JAVA_INT_UNALIGNED.withOrder(LE);
    private static final ValueLayout.OfShort LE16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(LE);
    private static final ValueLayout.OfLong LE64 = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(LE);
    private static final float MAGIC = 12582912f; // 1.5 * 2^23
    private static final int MAGIC_BITS = Float.floatToRawIntBits(MAGIC);

    private static final ByteVector TAB_IQ4 = ByteVector.fromArray(B16, Ggml.KVALUES_IQ4NL, 0);
    private static final ByteVector TAB_FP4 = ByteVector.fromArray(B16, Ggml.KVALUES_MXFP4, 0);

    private GgufInt8Kernels() {
    }

    /** Whether {@code w} has a kernel here (for a split tensor: both parts). */
    public static boolean supports(QTensor w) {
        if (w.second != null) {
            return supports(w.partOf(0)) && supports(w.second);
        }
        if (!Int8Kernels.SUPPORTED) {
            return false;
        }
        return switch (w.type) {
            case Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4 -> w.cols % QType.BLOCK == 0;
            case IQ4_XS, Q3_K -> w.cols % Ggml.QK_K == 0;
            default -> false;
        };
    }

    private static IntVector widenSum(ShortVector p) {
        IntVector pi = p.reinterpretAsInts();
        return pi.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16).add(pi.lanewise(VectorOperators.ASHR, 16));
    }

    private static FloatVector toFloat(IntVector i) {
        return i.add(MAGIC_BITS).reinterpretAsFloats().sub(MAGIC);
    }

    /**
     * Dot product of row {@code r} of {@code w} with the int8 activation {@code xq[xo ..]}; block scales {@code xs}
     * and block sums {@code xsum} are indexed from block {@code xo / 32}.
     */
    public static float dotRow(QTensor w, int r, short[] xq, float[] xs, int[] xsum, int xo) {
        if (w.second != null) {
            return dotRow(w.partOf(r), w.partRow(r), xq, xs, xsum, xo);
        }
        MemorySegment seg = w.data;
        long q = w.rowOffset(r);
        int nb = w.blocksPerRow();
        int xb = xo / QType.BLOCK;
        return switch (w.type) {
            case Q4_1 -> dotQ41(seg, q, nb, xq, xs, xsum, xo, xb);
            case Q5_1 -> dotQ51(seg, q, nb, xq, xs, xsum, xo, xb);
            case Q5_0 -> dotQ50(seg, q, nb, xq, xs, xo, xb);
            case IQ4_NL -> dotTab(seg, q, nb, xq, xs, xo, xb, false);
            case MXFP4 -> dotTab(seg, q, nb, xq, xs, xo, xb, true);
            case IQ4_XS -> dotIQ4XS(seg, q, nb, xq, xs, xo, xb);
            case Q3_K -> dotQ3K(seg, q, nb, xq, xs, xo, xb);
            default -> throw new IllegalArgumentException(w.type.name());
        };
    }

    /** Q4_1 (20 B: d, m, 16 nibble bytes): {@code Σ (d q + m) x = d Σ q x + m Σ x}. */
    private static float dotQ41(MemorySegment seg, long q, int nb, short[] xq, float[] xs, int[] xsum, int xo, int sb) {
        FloatVector acc = FloatVector.zero(F4);
        float minus = 0;
        for (int b = 0; b < nb; b++, q += Ggml.Q4_1_BYTES) {
            ShortVector v = ShortVector.fromMemorySegment(S8, seg, q + 4, LE);
            int x = xo + b * QType.BLOCK;
            ShortVector p = v.and((short) 0x0F).mul(ShortVector.fromArray(S8, xq, x));
            p = v.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F).mul(ShortVector.fromArray(S8, xq, x + 8)).add(p);
            p = v.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F).mul(ShortVector.fromArray(S8, xq, x + 16)).add(p);
            p = v.lanewise(VectorOperators.LSHR, 12).and((short) 0x0F).mul(ShortVector.fromArray(S8, xq, x + 24)).add(p);
            float xd = xs[sb + b];
            int dm = seg.get(LE32, q); // d and m in one load: two separate segment reads per block push C2 over its inlining budget
            acc = toFloat(widenSum(p)).fma(FloatVector.broadcast(F4, Float.float16ToFloat((short) dm) * xd), acc);
            minus += Float.float16ToFloat((short) (dm >>> 16)) * xd * xsum[sb + b];
        }
        return acc.reduceLanes(VectorOperators.ADD) + minus;
    }

    // the fifth bit of the 5-bit formats, expanded in the byte domain: lane j takes bit (16 hi + j) of qh
    private static final VectorShuffle<Byte> SH_LO = VectorShuffle.fromValues(B16, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    private static final VectorShuffle<Byte> SH_HI = VectorShuffle.fromValues(B16, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    private static final ByteVector BIT_OF_LANE = ByteVector.fromArray(B16, new byte[]{1, 2, 4, 8, 16, 32, 64, (byte) 128,
            1, 2, 4, 8, 16, 32, 64, (byte) 128}, 0);
    private static final ByteVector ZERO = ByteVector.zero(B16);

    /** 16 in the lanes whose fifth bit is set, else 0. */
    private static ByteVector bit16(ByteVector qhBytes, boolean hi) {
        return ZERO.blend((byte) 16, qhBytes.rearrange(hi ? SH_HI : SH_LO).and(BIT_OF_LANE).compare(VectorOperators.NE, (byte) 0));
    }

    /** Q5_1 (24 B: d, m, qh[4], 16 nibble bytes): Q4_1 with the fifth bit of weight j taken from bit j of qh. */
    private static float dotQ51(MemorySegment seg, long q, int nb, short[] xq, float[] xs, int[] xsum, int xo, int sb) {
        FloatVector acc = FloatVector.zero(F4);
        float minus = 0;
        for (int b = 0; b < nb; b++, q += Ggml.Q5_1_BYTES) {
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + 8, LE);
            ByteVector qh = ByteVector.fromMemorySegment(B16, seg, q + 4, LE);
            IntVector isum = tabDot(v.and((byte) 0x0F).add(bit16(qh, false)), v.lanewise(VectorOperators.LSHR, 4).add(bit16(qh, true)),
                    xq, xo + b * QType.BLOCK);
            float xd = xs[sb + b];
            int dm = seg.get(LE32, q);
            acc = toFloat(isum).fma(FloatVector.broadcast(F4, Float.float16ToFloat((short) dm) * xd), acc);
            minus += Float.float16ToFloat((short) (dm >>> 16)) * xd * xsum[sb + b];
        }
        return acc.reduceLanes(VectorOperators.ADD) + minus;
    }

    /** Q5_0 (22 B: d, qh[4], 16 nibble bytes): {@code (q5 - 16) d}. */
    private static float dotQ50(MemorySegment seg, long q, int nb, short[] xq, float[] xs, int xo, int sb) {
        FloatVector acc = FloatVector.zero(F4);
        for (int b = 0; b < nb; b++, q += Ggml.Q5_0_BYTES) {
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + 6, LE);
            ByteVector qh = ByteVector.fromMemorySegment(B16, seg, q + 2, LE);
            IntVector isum = tabDot(v.and((byte) 0x0F).add(bit16(qh, false)).sub((byte) 16),
                    v.lanewise(VectorOperators.LSHR, 4).add(bit16(qh, true)).sub((byte) 16), xq, xo + b * QType.BLOCK);
            acc = toFloat(isum).fma(FloatVector.broadcast(F4, Ggml.f16(seg, q) * xs[sb + b]), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /**
     * The integer product of 32 table-looked-up int8 weights (two 16-byte halves, natural order) with the permuted
     * activation shorts at {@code x}: each half as 8 short lanes of (even, odd) byte pairs, as {@link Int8Kernels#dotRowQ8GxQ8}.
     */
    private static IntVector tabDot(ByteVector loW, ByteVector hiW, short[] xq, int x) {
        ShortVector lv = loW.reinterpretAsShorts();
        ShortVector hv = hiW.reinterpretAsShorts();
        ShortVector pl = lv.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, x))
                .add(lv.lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, x + 8)));
        ShortVector ph = hv.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, x + 16))
                .add(hv.lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, x + 24)));
        return widenSum(pl).add(widenSum(ph));
    }

    /** IQ4_NL (18 B: f16 d, 16 nibble bytes) and MXFP4 (17 B: e8m0, 16 nibble bytes). */
    private static float dotTab(MemorySegment seg, long q, int nb, short[] xq, float[] xs, int xo, int sb, boolean fp4) {
        int bytes = fp4 ? Ggml.MXFP4_BYTES : Ggml.IQ4_NL_BYTES;
        ByteVector tab = fp4 ? TAB_FP4 : TAB_IQ4;
        FloatVector acc = FloatVector.zero(F4);
        for (int b = 0; b < nb; b++, q += bytes) {
            ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + (fp4 ? 1 : 2), LE);
            IntVector isum = tabDot(v.and((byte) 0x0F).selectFrom(tab),
                    v.lanewise(VectorOperators.LSHR, 4).selectFrom(tab), xq, xo + b * QType.BLOCK);
            float d = fp4 ? Ggml.e8m0Half(seg.get(ValueLayout.JAVA_BYTE, q) & 0xFF) : Ggml.f16(seg, q);
            acc = toFloat(isum).fma(FloatVector.broadcast(F4, d * xs[sb + b]), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /** The integer product of 16 signed byte weights (natural order) with the 16 permuted activation shorts at {@code x}. */
    private static IntVector dot16(ByteVector w, short[] xq, int x) {
        ShortVector v = w.reinterpretAsShorts();
        return widenSum(v.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, x))
                .add(v.lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, x + 8))));
    }

    /** Q3_K weights of one half of unit u: {@code ((qs >> 2 (u & 3)) & 3) - 4 + 4 (hmask bit u)}. */
    private static ByteVector q3kHalf(ByteVector qs, ByteVector hm, int u) {
        return qs.lanewise(VectorOperators.LSHR, 2 * (u & 3)).and((byte) 3).sub((byte) 4)
                .add(hm.lanewise(VectorOperators.LSHR, u).and((byte) 1).lanewise(VectorOperators.LSHL, 2));
    }

    /** Q3_K (110 B per 256 weights): sixteen 6-bit scales, one per 16 weights (llama.cpp's kmask1 / kmask2 shuffle). */
    private static float dotQ3K(MemorySegment seg, long q, int nb, short[] xq, float[] xs, int xo, int sb) {
        FloatVector acc = FloatVector.zero(F4);
        for (int s = 0; s < nb; s++, q += Ggml.Q3_K_BYTES) {
            long a = seg.get(LE64, q + 96);
            int a0 = (int) a;
            int a1 = (int) (a >>> 32);
            int t = seg.get(LE32, q + 104);
            float d = Float.float16ToFloat(seg.get(LE16, q + 108));
            ByteVector h0 = ByteVector.fromMemorySegment(B16, seg, q, LE);
            ByteVector h1 = ByteVector.fromMemorySegment(B16, seg, q + 16, LE);
            for (int g = 0; g < 4; g++) {
                int word = switch (g) {
                    case 0 -> (a0 & 0x0f0f0f0f) | ((t & 0x03030303) << 4);
                    case 1 -> (a1 & 0x0f0f0f0f) | (((t >>> 2) & 0x03030303) << 4);
                    case 2 -> ((a0 >>> 4) & 0x0f0f0f0f) | (((t >>> 4) & 0x03030303) << 4);
                    default -> ((a1 >>> 4) & 0x0f0f0f0f) | (((t >>> 6) & 0x03030303) << 4);
                };
                for (int h = 0; h < 2; h++) {
                    int u = 2 * g + h;
                    int blk = s * 8 + u;
                    long qs = q + 32 + 32L * (u >> 2);
                    float xd = xs[sb + blk];
                    IntVector i0 = dot16(q3kHalf(ByteVector.fromMemorySegment(B16, seg, qs, LE), h0, u), xq, xo + blk * QType.BLOCK);
                    IntVector i1 = dot16(q3kHalf(ByteVector.fromMemorySegment(B16, seg, qs + 16, LE), h1, u), xq, xo + blk * QType.BLOCK + 16);
                    acc = toFloat(i0).fma(FloatVector.broadcast(F4, d * (((word >>> (16 * h)) & 0xFF) - 32) * xd), acc);
                    acc = toFloat(i1).fma(FloatVector.broadcast(F4, d * (((word >>> (16 * h + 8)) & 0xFF) - 32) * xd), acc);
                }
            }
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /** IQ4_XS (136 B per 256 weights): eight IQ4_NL-style 32-weight sub-blocks with their own 6-bit scales. */
    private static float dotIQ4XS(MemorySegment seg, long q, int nb, short[] xq, float[] xs, int xo, int sb) {
        FloatVector acc = FloatVector.zero(F4);
        for (int s = 0; s < nb; s++, q += Ggml.IQ4_XS_BYTES) {
            long h = seg.get(LE64, q); // d, scales_h and scales_l in one load
            float d = Float.float16ToFloat((short) h);
            int sh = (int) (h >>> 16) & 0xFFFF;
            int sl = (int) (h >>> 32);
            for (int ib = 0; ib < 8; ib++) {
                ByteVector v = ByteVector.fromMemorySegment(B16, seg, q + 8 + 16L * ib, LE);
                int blk = s * 8 + ib;
                IntVector isum = tabDot(v.and((byte) 0x0F).selectFrom(TAB_IQ4),
                        v.lanewise(VectorOperators.LSHR, 4).selectFrom(TAB_IQ4), xq, xo + blk * QType.BLOCK);
                int ls = ((sl >>> (8 * (ib >> 1) + 4 * (ib & 1))) & 0xF) | (((sh >>> (2 * ib)) & 3) << 4);
                acc = toFloat(isum).fma(FloatVector.broadcast(F4, d * (ls - 32) * xs[sb + blk]), acc);
            }
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }
}
