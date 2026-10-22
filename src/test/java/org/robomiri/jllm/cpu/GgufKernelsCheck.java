package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Ggml;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

/**
 * Byte-parity of the ggml block dequantizers and of the CPU kernels with the reference implementation
 * ({@code gguf.quants} of llama.cpp's gguf-py): {@code src/test/resources/gguf/<type>.blocks} are rows of random
 * blocks (valid f16 scales) and {@code <type>.ref} their f32 dequantization by the reference
 * ({@code src/test/python/gen_gguf_fixtures.py}).
 *
 * <p>Checks per type: the scalar dequantizer ({@link Ggml}) against the reference (the formulas are the same
 * single-precision operations, so equality is expected bit for bit), the SIMD dequantizer and the SIMD dot product
 * against the scalar reference within float rounding.
 */
public final class GgufKernelsCheck {

    static int passed;
    static int failed;

    static void check(boolean ok, String what) {
        if (ok) {
            passed++;
        } else {
            failed++;
            System.out.println("FAIL: " + what);
        }
    }

    private static final int ROWS = 4;
    private static final int COLS = 512;

    interface Dequant {
        void row(MemorySegment seg, long off, int cols, float[] dst, int o);
    }

    private static byte[] resource(String name) throws IOException {
        try (InputStream in = GgufKernelsCheck.class.getResourceAsStream("/gguf/" + name)) {
            if (in == null) {
                throw new IOException("missing test resource /gguf/" + name);
            }
            return in.readAllBytes();
        }
    }

    private static float[] floats(byte[] b) {
        float[] f = new float[b.length / 4];
        ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(f);
        return f;
    }

    public static void main(String[] args) throws Exception {
        try (Arena a = Arena.ofShared()) {
            // formats the engine computes with in place: scalar == reference, SIMD within rounding
            native_(a, "Q4_0", QType.Q4_0G);
            native_(a, "Q8_0", QType.Q8_0G);
            native_(a, "Q4_K", QType.Q4_K);
            native_(a, "Q5_K", QType.Q5_K);
            native_(a, "Q2_K", QType.Q2_K);
            native_(a, "Q6_K", QType.Q6_K);
            native_(a, "Q4_1", QType.Q4_1);
            native_(a, "Q5_0", QType.Q5_0);
            native_(a, "Q5_1", QType.Q5_1);
            native_(a, "Q3_K", QType.Q3_K);
            native_(a, "IQ4_NL", QType.IQ4_NL);
            native_(a, "IQ4_XS", QType.IQ4_XS);
            native_(a, "MXFP4", QType.MXFP4);
            scalarOnly(a, "Q6_K", Ggml::dequantQ6_K); // the scalar definition used by the converting fallbacks
            floatTypes(a);
        }
        halfConversion();
        System.out.println("ggml kernels: " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    /** F16 / BF16 matrices used as they are: the dequantizers are the conversions, the SIMD kernels must agree with them. */
    private static void floatTypes(Arena a) {
        Random rnd = new Random(7);
        for (QType qt : new QType[]{QType.F16, QType.BF16}) {
            MemorySegment seg = a.allocate(ROWS * COLS * 2L + 2, 1).asSlice(1, ROWS * COLS * 2L);
            float[] ref = new float[ROWS * COLS];
            for (int i = 0; i < ref.length; i++) {
                float v = (float) (rnd.nextGaussian() * (i % 11 == 0 ? 50 : 0.5));
                if (qt == QType.F16) {
                    short h = Float.floatToFloat16(v);
                    seg.set(java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED, 2L * i, h);
                    ref[i] = Float.float16ToFloat(h);
                } else {
                    short h = (short) (Float.floatToRawIntBits(v) >>> 16);
                    seg.set(java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED, 2L * i, h);
                    ref[i] = Float.intBitsToFloat((h & 0xFFFF) << 16);
                }
            }
            QTensor t = QTensor.mapped(qt, ROWS, COLS, seg);
            float[] x = new float[COLS + 4];
            for (int i = 0; i < x.length; i++) {
                x[i] = rnd.nextFloat() * 2 - 1;
            }
            float[] row = new float[COLS];
            float[] simd = new float[COLS];
            int bad = 0;
            double worstDot = 0;
            for (int r = 0; r < ROWS; r++) {
                t.dequantizeRow(r, row, 0);
                VectorKernels.dequantRow(t, r, simd, 0);
                double want = 0;
                double mag = 0;
                for (int i = 0; i < COLS; i++) {
                    float e = ref[r * COLS + i];
                    bad += (row[i] == e ? 0 : 1) + (simd[i] == e ? 0 : 1);
                    want += (double) e * x[4 + i];
                    mag += Math.abs((double) e * x[4 + i]);
                }
                worstDot = Math.max(worstDot, Math.abs(VectorKernels.dotRow(t, r, x, 4) - want) / (mag + 1e-6));
            }
            check(bad == 0, qt + " scalar and SIMD dequantizers equal the conversion: " + bad + " differ");
            check(worstDot < 1e-5, qt + " SIMD dot vs reference: " + worstDot);
            System.out.printf("%-5s scalar and SIMD == conversion (%d differ), dot %.2e%n", qt, bad, worstDot);
        }
    }

    private static MemorySegment load(Arena a, String type) throws IOException {
        byte[] blocks = resource(type + ".blocks");
        MemorySegment seg = a.allocate(blocks.length + 2, 1).asSlice(1, blocks.length); // deliberately odd-aligned, as in a GGUF
        MemorySegment.copy(blocks, 0, seg, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, blocks.length);
        return seg;
    }

    private static void scalarOnly(Arena a, String type, Dequant d) throws IOException {
        MemorySegment seg = load(a, type);
        float[] ref = floats(resource(type + ".ref"));
        long rowBytes = seg.byteSize() / ROWS;
        float[] row = new float[COLS];
        int exact = 0;
        double worst = 0;
        for (int r = 0; r < ROWS; r++) {
            d.row(seg, r * rowBytes, COLS, row, 0);
            for (int i = 0; i < COLS; i++) {
                float e = ref[r * COLS + i];
                if (row[i] == e) {
                    exact++;
                }
                worst = Math.max(worst, Math.abs(row[i] - e) / (Math.abs(e) + 1e-3));
            }
        }
        check(worst < 1e-5, type + " scalar dequantizer vs gguf.quants: max rel err " + worst);
        System.out.printf("%-5s scalar == reference for %d/%d weights, max rel err %.2e%n", type, exact, ROWS * COLS, worst);
    }

    private static void native_(Arena a, String type, QType qt) throws IOException {
        MemorySegment seg = load(a, type);
        float[] ref = floats(resource(type + ".ref"));
        QTensor t = QTensor.mapped(qt, ROWS, COLS, seg);
        float[] row = new float[COLS];
        float[] simd = new float[COLS];
        Random rnd = new Random(42);
        float[] x = new float[COLS + 8];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 2 - 1;
        }
        int exact = 0;
        double worstScalar = 0;
        double worstSimd = 0;
        double worstDot = 0;
        for (int r = 0; r < ROWS; r++) {
            t.dequantizeRow(r, row, 0);
            VectorKernels.dequantRow(t, r, simd, 0);
            double want = 0;
            double mag = 0;
            for (int i = 0; i < COLS; i++) {
                float e = ref[r * COLS + i];
                if (row[i] == e) {
                    exact++;
                }
                worstScalar = Math.max(worstScalar, Math.abs(row[i] - e) / (Math.abs(e) + 1e-3));
                worstSimd = Math.max(worstSimd, Math.abs(simd[i] - e) / (Math.abs(e) + 1e-3));
                want += (double) e * x[4 + i];
                mag += Math.abs((double) e * x[4 + i]);
            }
            float got = VectorKernels.dotRow(t, r, x, 4);
            worstDot = Math.max(worstDot, Math.abs(got - want) / (mag + 1e-6));
        }
        if (qt == QType.Q4_0G || qt == QType.Q8_0G) {
            // int8 activations: the SIMD integer kernel against its scalar reference and against the float product
            short[] xq = new short[COLS];
            float[] xs = new float[COLS / 32];
            float[] xa = java.util.Arrays.copyOfRange(x, 4, 4 + COLS);
            Int8Kernels.quantize(xa, 0, COLS, xq, xs);
            double worstInt = 0;
            double worstVsFloat = 0;
            for (int r = 0; r < ROWS; r++) {
                float got = Int8Kernels.dotRow(t, r, xq, xs, 0);
                float want = Int8Kernels.dotRowScalar(t, r, xq, xs, 0);
                double mag = 0;
                double fl = 0;
                for (int i = 0; i < COLS; i++) {
                    mag += Math.abs((double) ref[r * COLS + i] * xa[i]);
                    fl += (double) ref[r * COLS + i] * xa[i];
                }
                worstInt = Math.max(worstInt, Math.abs(got - want) / (mag + 1e-9));
                worstVsFloat = Math.max(worstVsFloat, Math.abs(got - fl) / (mag + 1e-9));
            }
            check(worstInt < 1e-5, type + " int8-activation SIMD dot vs scalar reference: " + worstInt);
            check(worstVsFloat < 0.02, type + " int8-activation dot vs float dot (activation rounding): " + worstVsFloat);
            System.out.printf("%-5s int8 activations: SIMD vs scalar %.2e, vs float dot %.2e%n", type, worstInt, worstVsFloat);
        }
        if (qt == QType.Q4_K || qt == QType.Q5_K || qt == QType.Q6_K) {
            kquants(type, t, ref, java.util.Arrays.copyOfRange(x, 4, 4 + COLS));
        }
        if (GgufInt8Kernels.supports(t)) {
            otherInt8(type, t, ref, java.util.Arrays.copyOfRange(x, 4, 4 + COLS));
        }
        check(worstScalar < 1e-5, type + " scalar dequantizer vs gguf.quants: max rel err " + worstScalar);
        check(worstSimd < 1e-5, type + " SIMD dequantizer vs gguf.quants: max rel err " + worstSimd);
        check(worstDot < 1e-5, type + " SIMD dot vs reference: max err relative to sum|w x| " + worstDot);
        System.out.printf("%-5s scalar == reference for %d/%d weights, max rel err scalar %.2e simd %.2e, dot %.2e (%s)%n", type,
                exact, ROWS * COLS, worstScalar, worstSimd, worstDot, VectorKernels.describe());
    }

    /**
     * The f16 KV cache's vector half-to-float conversion ({@link KvCache.F16#lowHalf} / {@code highHalf}) against
     * {@link Float#float16ToFloat} for every finite half (zeros, subnormals, normals, both signs), in both lane halves.
     */
    private static void halfConversion() {
        var sp = jdk.incubator.vector.IntVector.SPECIES_PREFERRED;
        int[] lanes = new int[sp.length()];
        float[] lo = new float[sp.length()];
        float[] hi = new float[sp.length()];
        int wrong = 0;
        int tested = 0;
        for (int h = 0; h < 65536; h += sp.length()) {
            for (int l = 0; l < lanes.length; l++) {
                int a = h + l;
                int b = (h + l) ^ 0x8000; // the other lane half holds a different value with the opposite sign
                lanes[l] = (a & 0xFFFF) | (b << 16);
            }
            jdk.incubator.vector.IntVector x = jdk.incubator.vector.IntVector.fromArray(sp, lanes, 0);
            KvCache.F16.lowHalf(x).intoArray(lo, 0);
            KvCache.F16.highHalf(x).intoArray(hi, 0);
            for (int l = 0; l < lanes.length; l++) {
                short a = (short) (h + l);
                short b = (short) ((h + l) ^ 0x8000);
                if ((a & 0x7C00) == 0x7C00) {
                    continue; // infinities and NaNs are not supported
                }
                tested++;
                if (Float.floatToRawIntBits(lo[l]) != Float.floatToRawIntBits(Float.float16ToFloat(a))
                        || Float.floatToRawIntBits(hi[l]) != Float.floatToRawIntBits(Float.float16ToFloat(b))) {
                    wrong++;
                }
            }
        }
        check(wrong == 0, "vector half-to-float conversion: " + wrong + " of " + tested + " halves differ");
        System.out.printf("f16   vector half-to-float == Float.float16ToFloat for %d/%d finite halves%n", tested - wrong, tested);
    }

    /**
     * {@link GgufInt8Kernels}: the int8-activation dot against the exact product of the reference weights with the
     * dequantized int8 activation (only float summation order differs), and against the float dot (activation rounding).
     */
    private static void otherInt8(String type, QTensor t, float[] ref, float[] xa) {
        short[] xq = new short[COLS];
        float[] xs = new float[COLS / 32];
        int[] sums = new int[COLS / 32];
        Int8Kernels.quantize(xa, 0, COLS, xq, xs);
        KQuantKernels.blockSums(xq, COLS, sums);
        double worstExact = 0;
        double worstFloat = 0;
        for (int r = 0; r < ROWS; r++) {
            double exact = 0;
            double fl = 0;
            double mag = 0;
            for (int i = 0; i < COLS; i++) {
                double w = ref[r * COLS + i];
                exact += w * (double) xs[i / 32] * xq[(i / 32) * 32 + Int8Kernels.permutedIndex(i % 32)];
                fl += w * xa[i];
                mag += Math.abs(w * xa[i]);
            }
            float got = GgufInt8Kernels.dotRow(t, r, xq, xs, sums, 0);
            worstExact = Math.max(worstExact, Math.abs(got - exact) / (mag + 1e-9));
            worstFloat = Math.max(worstFloat, Math.abs(got - fl) / (mag + 1e-9));
        }
        check(worstExact < 1e-5, type + " int8 dot vs exact product with the quantized activation: " + worstExact);
        check(worstFloat < 0.02, type + " int8 dot vs float dot (activation rounding): " + worstFloat);
        System.out.printf("%-6s int8 activations: vs exact %.2e, vs float dot %.2e%n", type, worstExact, worstFloat);
    }

    /**
     * {@link KQuantKernels}: the int8-activation dot against the exact product of the reference weights with the
     * dequantized int8 activation, and against the float dot (only activation rounding differs); the permuted
     * dequantization element by element; a dot of permuted row and permuted activation against the plain one.
     */
    private static void kquants(String type, QTensor t, float[] ref, float[] xa) {
        short[] xq = new short[COLS];
        float[] xs = new float[COLS / 32];
        int[] sums = new int[COLS / 32];
        Int8Kernels.quantize(xa, 0, COLS, xq, xs);
        KQuantKernels.blockSums(xq, COLS, sums);
        double worstExact = 0;
        double worstFloat = 0;
        double worstDeq = 0;
        double worstPermDot = 0;
        float[] row = new float[COLS];
        float[] xp = new float[COLS];
        KQuantKernels.permute(xa, 1, COLS, xp);
        for (int r = 0; r < ROWS; r++) {
            double exact = 0;
            double fl = 0;
            double mag = 0;
            for (int i = 0; i < COLS; i++) {
                double w = ref[r * COLS + i];
                double xi = (double) xs[i / 32] * xq[(i / 32) * 32 + Int8Kernels.permutedIndex(i % 32)];
                exact += w * xi;
                fl += w * xa[i];
                mag += Math.abs(w * xa[i]);
            }
            float got = KQuantKernels.dotRow(t, r, xq, xs, sums, 0);
            worstExact = Math.max(worstExact, Math.abs(got - exact) / (mag + 1e-9));
            worstFloat = Math.max(worstFloat, Math.abs(got - fl) / (mag + 1e-9));
            KQuantKernels.dequantPermuted(t, r, row, 0);
            double pd = 0;
            for (int i = 0; i < COLS; i++) {
                float e = ref[r * COLS + i];
                float g = row[(i / 32) * 32 + KQuantKernels.permutedIndex(i % 32)];
                worstDeq = Math.max(worstDeq, Math.abs(g - e) / (Math.abs(e) + 1e-3));
                pd += (double) row[i] * xp[i];
            }
            worstPermDot = Math.max(worstPermDot, Math.abs(pd - fl) / (mag + 1e-9));
        }
        check(worstExact < 1e-5, type + " int8 K-quant dot vs exact product with the quantized activation: " + worstExact);
        check(worstFloat < 0.02, type + " int8 K-quant dot vs float dot (activation rounding): " + worstFloat);
        check(worstDeq < 1e-5, type + " permuted dequantization vs gguf.quants: " + worstDeq);
        check(worstPermDot < 1e-6, type + " permuted row . permuted activation vs plain dot: " + worstPermDot);
        System.out.printf("%-5s int8 K-quant dot: vs exact %.2e, vs float %.2e; permuted dequant %.2e, permuted dot %.2e%n",
                type, worstExact, worstFloat, worstDeq, worstPermDot);
    }
}
