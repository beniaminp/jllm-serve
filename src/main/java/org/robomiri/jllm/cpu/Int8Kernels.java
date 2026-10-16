package org.robomiri.jllm.cpu;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.model.Ggml;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Int8 activation path (llama.cpp's q4_0 x q8_0 / q8_0 x q8_0 scheme). The activation vector is quantized once
 * per matmul to Q8_0 (32-element blocks, one f32 scale each) and every weight row is multiplied with it on
 * 16-bit integer lanes: eight products per 128-bit multiply instead of four float FMAs, and no byte-to-float
 * conversion in the inner loop. Per block the exact integer dot product is scaled by
 * {@code weightScale * activationScale} and accumulated in float lanes; one horizontal reduction per row.
 *
 * <p>Layout of a quantized activation block (32 shorts): {@code [x0 x2 .. x14] [x1 x3 .. x15] [x16 x18 .. x30]
 * [x17 x19 .. x31]}. A 16-byte group of weights is loaded as eight shorts, so short lane {@code k} holds
 * bytes {@code 2k} (low half) and {@code 2k+1} (high half); the four 8-lane weight vectors a shift-and-mask
 * unpack produces are therefore "even bytes" and "odd bytes" of each half, and the activations are stored in
 * that order so no shuffle is needed.
 *
 * <p>Implementation notes (docs/07 §7.10): the kernels use only loads, lanewise arithmetic, same-size
 * reinterpretation and one reduction per row. Lane-widening conversions ({@code convertShape},
 * {@code castShape}) are avoided because they all run through one shared JDK template with profiled call
 * sites; widening is done with shifts and int-to-float with the 2^23 magic constant. The loop bodies are kept
 * to one memory-segment load per iteration: the same arithmetic with two loads per iteration made C2 take
 * minutes to compile the method, during which it ran at C1 speed, ~1000x slower. Only C2-compiled Vector API
 * code is fast, so the first second of a run is warm-up.
 */
public final class Int8Kernels {

    private static final VectorSpecies<Short> S8 = ShortVector.SPECIES_128;
    private static final VectorSpecies<Integer> I4 = IntVector.SPECIES_128;
    private static final VectorSpecies<Float> F4 = FloatVector.SPECIES_128;
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    /** 1.5 * 2^23: adding it to a float in (-2^22, 2^22) leaves the rounded integer in the low mantissa bits. */
    private static final float MAGIC = 12582912f;
    private static final int MAGIC_BITS = Float.floatToRawIntBits(MAGIC);

    /** The kernels need 128-bit vectors; narrower hardware falls back to the scalar references. */
    public static final boolean SUPPORTED = FloatVector.SPECIES_PREFERRED.vectorBitSize() >= 128;

    private Int8Kernels() {
    }

    // ------------------------------------------------------------------ activation quantization

    /**
     * Quantizes {@code x[xo .. xo + n)} (n a multiple of 32) to Q8_0 in the permuted block layout described
     * above: {@code q = rint(x / d)} with {@code d = max|x| / 127} per block, into {@code q[0 .. n)} and the
     * scales into {@code s[0 .. n/32)}. |q| is at most 127; rounding is to nearest, ties to even.
     */
    public static void quantize(float[] x, int xo, int n, short[] q, float[] s) {
        if (!SUPPORTED) {
            quantizeScalar(x, xo, n, q, s);
            return;
        }
        int nb = n / QType.BLOCK;
        int[] tmp = TMP.get();
        for (int b = 0; b < nb; b++) {
            int o = xo + b * QType.BLOCK;
            FloatVector m = FloatVector.zero(F4);
            for (int p = 0; p < QType.BLOCK; p += 4) {
                m = m.max(FloatVector.fromArray(F4, x, o + p).abs());
            }
            float amax = m.reduceLanes(VectorOperators.MAX);
            float d = amax / 127f;
            float id = d != 0 ? 1f / d : 0f;
            s[b] = d;
            FloatVector idv = FloatVector.broadcast(F4, id);
            int qo = b * QType.BLOCK;
            for (int h = 0; h < 2; h++) {
                // 16 inputs -> rounded ints in the low 16 bits of each lane; pack lane pairs into shorts
                int io = o + h * 16;
                IntVector i0 = roundToInt(FloatVector.fromArray(F4, x, io).mul(idv));
                IntVector i1 = roundToInt(FloatVector.fromArray(F4, x, io + 4).mul(idv));
                IntVector i2 = roundToInt(FloatVector.fromArray(F4, x, io + 8).mul(idv));
                IntVector i3 = roundToInt(FloatVector.fromArray(F4, x, io + 12).mul(idv));
                // tmp[j] = rounded x[io + j]; even j go to the first 8 shorts of the half, odd j to the next 8
                i0.intoArray(tmp, 0);
                i1.intoArray(tmp, 4);
                i2.intoArray(tmp, 8);
                i3.intoArray(tmp, 12);
                int eo = qo + h * 16;
                for (int k = 0; k < 8; k++) {
                    q[eo + k] = (short) tmp[2 * k];
                    q[eo + 8 + k] = (short) tmp[2 * k + 1];
                }
            }
        }
    }

    private static final ThreadLocal<int[]> TMP = ThreadLocal.withInitial(() -> new int[16]);

    /** Round to nearest (ties to even) for |v| < 2^22, without a float-to-int conversion node. */
    private static IntVector roundToInt(FloatVector v) {
        return v.add(MAGIC).reinterpretAsInts().sub(MAGIC_BITS);
    }

    /** Scalar reference for {@link #quantize}; bit-identical results. */
    public static void quantizeScalar(float[] x, int xo, int n, short[] q, float[] s) {
        int nb = n / QType.BLOCK;
        for (int b = 0; b < nb; b++) {
            int o = xo + b * QType.BLOCK;
            float amax = 0;
            for (int j = 0; j < QType.BLOCK; j++) {
                amax = Math.max(amax, Math.abs(x[o + j]));
            }
            float d = amax / 127f;
            float id = d != 0 ? 1f / d : 0f;
            s[b] = d;
            for (int j = 0; j < QType.BLOCK; j++) {
                q[b * QType.BLOCK + permutedIndex(j)] = (short) (int) Math.rint(x[o + j] * id);
            }
        }
    }

    /** Position of element {@code j} of a block inside the quantized block. */
    public static int permutedIndex(int j) {
        return (j & 16) + (j & 1) * 8 + ((j & 15) >> 1);
    }

    // ------------------------------------------------------------------ row dot products

    /**
     * Dot product of row {@code r} of {@code w} with the quantized activation {@code xq[xo .. xo + cols)} whose
     * block scales are {@code xs[xo/32 ..]}. {@code xo} must be a multiple of 32.
     */
    public static float dotRow(QTensor w, int r, short[] xq, float[] xs, int xo) {
        if (w.second != null) {
            return dotRow(w.partOf(r), w.partRow(r), xq, xs, xo);
        }
        if (!SUPPORTED) {
            return dotRowScalar(w, r, xq, xs, xo);
        }
        return switch (w.type) {
            case Q4_0 -> dotRowQ4xQ8(w, r, xq, xs, xo);
            case Q4_0G -> dotRowQ4GxQ8(w, r, xq, xs, xo);
            case Q8_0G -> dotRowQ8GxQ8(w, r, xq, xs, xo);
            default -> dotRowQ8xQ8(w, r, xq, xs, xo);
        };
    }

    /**
     * {@link #dotRowQ4xQ8} for ggml's Q4_0 layout: 18-byte blocks, f16 scale first, then the same 16 nibble bytes
     * (so {@link #blockQ4} applies unchanged, loaded from byte 2 of the block).
     */
    public static float dotRowQ4GxQ8(QTensor w, int r, short[] xq, float[] xs, int xo) {
        MemorySegment seg = w.data;
        long q = w.rowOffset(r);
        int nb = w.blocksPerRow();
        int sb = xo / QType.BLOCK;
        FloatVector acc = FloatVector.zero(F4);
        FloatVector magic = FloatVector.broadcast(F4, MAGIC);
        for (int b = 0; b < nb; b++, q += Ggml.Q4_0_BYTES) {
            IntVector isum = blockQ4(ShortVector.fromMemorySegment(S8, seg, q + 2, LE), xq, xo + b * QType.BLOCK);
            float scale = Ggml.f16(seg, q) * xs[sb + b];
            acc = toFloat(isum, magic).fma(FloatVector.broadcast(F4, scale), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /** {@link #dotRowQ8xQ8} for ggml's Q8_0 layout: 34-byte blocks, f16 scale then 32 int8, 16-byte halves at byte 2 and 18. */
    public static float dotRowQ8GxQ8(QTensor w, int r, short[] xq, float[] xs, int xo) {
        MemorySegment seg = w.data;
        long q = w.rowOffset(r);
        int ng = w.blocksPerRow() * 2;
        int sb = xo / QType.BLOCK;
        FloatVector acc = FloatVector.zero(F4);
        FloatVector magic = FloatVector.broadcast(F4, MAGIC);
        for (int g = 0; g < ng; g++) {
            long blk = q + (long) (g >> 1) * Ggml.Q8_0_BYTES;
            ShortVector v = ShortVector.fromMemorySegment(S8, seg, blk + 2 + (g & 1) * 16, LE);
            int xb = xo + g * 16;
            ShortVector p = v.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, xb))
                    .add(v.lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, xb + 8)));
            IntVector isum = widenSum(p);
            float scale = Ggml.f16(seg, blk) * xs[sb + (g >> 1)];
            acc = toFloat(isum, magic).fma(FloatVector.broadcast(F4, scale), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /**
     * Q4_0 row x Q8_0 activation. Per block the 16 packed bytes are loaded as 8 shorts; nibble {@code n} of
     * each short ({@code (v >>> 4n) & 0xF}, minus 8) is one 8-lane weight vector: n = 0 even bytes' low
     * nibbles (weights 0,2..14), n = 2 odd bytes' low nibbles (1,3..15), n = 1 even bytes' high nibbles
     * (16,18..30), n = 3 odd bytes' high nibbles (17,19..31). Four 8-lane products are summed in short lanes
     * (|w| <= 8, |x| <= 127: at most 4 x 1016 per lane, no overflow), widened to int lanes and scaled.
     */
    public static float dotRowQ4xQ8(QTensor w, int r, short[] xq, float[] xs, int xo) {
        MemorySegment seg = w.data;
        long q = w.rowOffset(r);
        long s = w.scaleOffset(r);
        int nb = w.blocksPerRow();
        int sb = xo / QType.BLOCK;
        FloatVector acc = FloatVector.zero(F4);
        FloatVector magic = FloatVector.broadcast(F4, MAGIC);
        for (int b = 0; b < nb; b++) {
            IntVector isum = blockQ4(ShortVector.fromMemorySegment(S8, seg, q + (long) b * 16, LE), xq, xo + b * QType.BLOCK);
            float scale = seg.get(ValueLayout.JAVA_FLOAT, s + (long) b * 4) * xs[sb + b];
            acc = toFloat(isum, magic).fma(FloatVector.broadcast(F4, scale), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /**
     * Q8_0 row x Q8_0 activation. The loop runs over 16-byte half blocks, each loaded as 8 shorts and split into
     * the sign-extended even bytes ({@code (v << 8) >> 8}) and odd bytes ({@code v >> 8}); the block scale is
     * applied to each half. Products reach 127 x 127 = 16129, so only two are summed per short lane
     * (32258 < 32767) before widening.
     *
     * <p>One segment load per iteration is deliberate: the same arithmetic written as one iteration per block
     * (two loads) made C2 take minutes instead of milliseconds to compile the method (see 07 §7.10).
     */
    public static float dotRowQ8xQ8(QTensor w, int r, short[] xq, float[] xs, int xo) {
        MemorySegment seg = w.data;
        long q = w.rowOffset(r);
        long s = w.scaleOffset(r);
        int ng = w.blocksPerRow() * 2;
        int sb = xo / QType.BLOCK;
        FloatVector acc = FloatVector.zero(F4);
        FloatVector magic = FloatVector.broadcast(F4, MAGIC);
        for (int g = 0; g < ng; g++) {
            ShortVector v = ShortVector.fromMemorySegment(S8, seg, q + (long) g * 16, LE);
            int xb = xo + g * 16;
            ShortVector p = v.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, xb))
                    .add(v.lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, xb + 8)));
            IntVector isum = widenSum(p);
            int b = g >> 1;
            float scale = seg.get(ValueLayout.JAVA_FLOAT, s + (long) b * 4) * xs[sb + b];
            acc = toFloat(isum, magic).fma(FloatVector.broadcast(F4, scale), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /** Exact integer dot product of one Q4_0 block (16 packed bytes as 8 shorts) with 32 activation shorts, as 4 int lanes. */
    private static IntVector blockQ4(ShortVector v, short[] xq, int xb) {
        ShortVector p = v.and((short) 0x0F).sub((short) 8).mul(ShortVector.fromArray(S8, xq, xb));
        p = v.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F).sub((short) 8).mul(ShortVector.fromArray(S8, xq, xb + 8)).add(p);
        p = v.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F).sub((short) 8).mul(ShortVector.fromArray(S8, xq, xb + 16)).add(p);
        p = v.lanewise(VectorOperators.LSHR, 12).and((short) 0x0F).sub((short) 8).mul(ShortVector.fromArray(S8, xq, xb + 24)).add(p);
        return widenSum(p);
    }

    /** Sums each pair of adjacent short lanes into one int lane (sign-extending both halves by shifts). */
    private static IntVector widenSum(ShortVector p) {
        IntVector pi = p.reinterpretAsInts();
        return pi.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16).add(pi.lanewise(VectorOperators.ASHR, 16));
    }

    /** Exact int-to-float for |i| < 2^22 without a conversion node: float bits of (i + 1.5 * 2^23) minus the constant. */
    private static FloatVector toFloat(IntVector i, FloatVector magic) {
        return i.add(MAGIC_BITS).reinterpretAsFloats().sub(magic);
    }

    // ------------------------------------------------------------------ references (tests, unsupported hardware)

    /** Exact integer dot product of block {@code b} of row {@code r} with the (permuted) activation shorts. */
    public static int blockDotScalar(QTensor w, int r, int b, short[] xq, int xo) {
        MemorySegment seg = w.data;
        int qb = xo + b * QType.BLOCK;
        int sum = 0;
        if (w.type == QType.Q4_0 || w.type == QType.Q4_0G) {
            long base = w.type == QType.Q4_0G ? w.rowOffset(r) + (long) b * Ggml.Q4_0_BYTES + 2 : w.rowOffset(r) + (long) b * 16;
            for (int j = 0; j < 16; j++) {
                int v = seg.get(ValueLayout.JAVA_BYTE, base + j) & 0xFF;
                sum += ((v & 0xF) - 8) * xq[qb + permutedIndex(j)];
                sum += ((v >>> 4) - 8) * xq[qb + permutedIndex(j + 16)];
            }
        } else {
            long base = w.type == QType.Q8_0G ? w.rowOffset(r) + (long) b * Ggml.Q8_0_BYTES + 2 : w.rowOffset(r) + (long) b * 32;
            for (int j = 0; j < QType.BLOCK; j++) {
                sum += seg.get(ValueLayout.JAVA_BYTE, base + j) * xq[qb + permutedIndex(j)];
            }
        }
        return sum;
    }

    /** Same arithmetic as the SIMD kernels (per-block integer sum, then float scale), accumulated in double. */
    public static float dotRowScalar(QTensor w, int r, short[] xq, float[] xs, int xo) {
        double acc = 0;
        int nb = w.blocksPerRow();
        long s = w.scaleOffset(r);
        for (int b = 0; b < nb; b++) {
            float ws = w.type.gguf ? Ggml.f16(w.data, w.rowOffset(r) + (long) b * w.type.blockBytes)
                    : w.data.get(ValueLayout.JAVA_FLOAT, s + (long) b * 4);
            acc += (double) (ws * xs[xo / QType.BLOCK + b]) * blockDotScalar(w, r, b, xq, xo);
        }
        return (float) acc;
    }

    /**
     * Exact integer dot product of a whole row (no scales) through the SIMD integer lanes, for tests: the same
     * per-block ({@link #blockQ4}) and per-half-block ({@link #dotRowQ8xQ8}) lane arithmetic as the kernels.
     */
    public static long intDotRow(QTensor w, int r, short[] xq, int xo) {
        long sum = 0;
        if (!SUPPORTED) {
            for (int b = 0; b < w.blocksPerRow(); b++) {
                sum += blockDotScalar(w, r, b, xq, xo);
            }
            return sum;
        }
        MemorySegment seg = w.data;
        long q = w.rowOffset(r);
        if (w.type == QType.Q4_0) {
            for (int b = 0; b < w.blocksPerRow(); b++) {
                sum += blockQ4(ShortVector.fromMemorySegment(S8, seg, q + (long) b * 16, LE), xq, xo + b * QType.BLOCK)
                        .reduceLanes(VectorOperators.ADD);
            }
            return sum;
        }
        for (int g = 0; g < w.blocksPerRow() * 2; g++) {
            ShortVector v = ShortVector.fromMemorySegment(S8, seg, q + (long) g * 16, LE);
            int xb = xo + g * 16;
            ShortVector p = v.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, xb))
                    .add(v.lanewise(VectorOperators.ASHR, 8).mul(ShortVector.fromArray(S8, xq, xb + 8)));
            sum += widenSum(p).reduceLanes(VectorOperators.ADD);
        }
        return sum;
    }

    public static String describe() {
        return SUPPORTED ? "int8 activations (8 short lanes)" : "int8 activations (scalar fallback)";
    }
}
