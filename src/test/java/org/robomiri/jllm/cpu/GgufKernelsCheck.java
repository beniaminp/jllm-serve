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
            // formats that are only converted to Q8_0 at load: scalar dequantizer == reference
            scalarOnly(a, "Q4_1", Ggml::dequantQ4_1);
            scalarOnly(a, "Q5_0", Ggml::dequantQ5_0);
            scalarOnly(a, "Q5_1", Ggml::dequantQ5_1);
            scalarOnly(a, "Q3_K", Ggml::dequantQ3_K);
            scalarOnly(a, "Q6_K", Ggml::dequantQ6_K); // the scalar definition used by the converting fallbacks
        }
        System.out.println("ggml kernels: " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
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
        check(worstScalar < 1e-5, type + " scalar dequantizer vs gguf.quants: max rel err " + worstScalar);
        check(worstSimd < 1e-5, type + " SIMD dequantizer vs gguf.quants: max rel err " + worstSimd);
        check(worstDot < 1e-5, type + " SIMD dot vs reference: max err relative to sum|w x| " + worstDot);
        System.out.printf("%-5s scalar == reference for %d/%d weights, max rel err scalar %.2e simd %.2e, dot %.2e (%s)%n", type,
                exact, ROWS * COLS, worstScalar, worstSimd, worstDot, VectorKernels.describe());
    }
}
