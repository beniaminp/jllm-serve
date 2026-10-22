package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.Random;

/**
 * The CPU kernels that sum a vector's lanes (the quantized row dots and the 4-row / 4 x 4 tiles of the prefill matmuls and
 * the CPU-side MoE experts) must give the same bits in the interpreter, in C1 and in C2. {@code reduceLanes(ADD)} does not:
 * its order is unspecified, the Java fallback adds the lanes one after the other and C2 on NEON in pairs, so a kernel's
 * first calls in a process summed differently from the rest. That made the concurrent-encoder check of arch smoke
 * (docs 08 8.28), whose serial reference is the first run of the CPU experts of a fresh JVM, fail about once in thirty
 * runs on a tiny Qwen3-MoE with experts on the CPU, although the encoder was innocent. Run in a fresh JVM, so that the
 * first calls are interpreted: the results of those calls are recorded, the kernels are then run until C2 has compiled
 * them, and every result must equal the first.
 *
 * <p>The float {@code dot} (the f32 matmuls, the norms' sums of squares, the dense attention helpers) and the softmax sum
 * of attention ({@code softmaxExp}) are covered as well: they are what the flat and the paged KV cache both use around the
 * attention scores, and a cold-JVM difference there made the flat-versus-paged comparison of {@code PagedEngineTest} fail
 * about once in thirty runs (docs 24 section 24.6).
 */
public final class JitStableKernelsTest {

    private static int failures;

    public static void main(String[] args) {
        int cols = 256;
        int rows = 16;
        Random rnd = new Random(42);
        float[] x = new float[4 * cols]; // four activation vectors
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() - 0.5f;
        }
        float[] w = new float[4 * cols];
        for (int i = 0; i < w.length; i++) {
            w[i] = rnd.nextFloat() - 0.5f;
        }
        short[] xq = new short[cols]; // the int8 activation path of the decode matmuls
        float[] xs = new float[cols / QType.BLOCK];
        Int8Kernels.quantize(x, 0, cols, xq, xs);
        try (Arena arena = Arena.ofShared()) {
            for (QType type : new QType[]{QType.Q4_0, QType.Q8_0}) {
                Random r = new Random(7);
                QTensor t = QTensor.quantize(arena, type, rows, cols, (row, dst) -> {
                    for (int i = 0; i < dst.length; i++) {
                        dst[i] = r.nextFloat() - 0.5f;
                    }
                });
                float[] first = null;
                for (int round = 0; round < 40_000; round++) {
                    float[] got = run(t, x, w, cols, xq, xs);
                    if (first == null) {
                        first = got;
                    } else if (!Arrays.equals(first, got)) {
                        failures++;
                        int i = 0;
                        while (first[i] == got[i]) {
                            i++;
                        }
                        System.out.println("FAIL " + type + ": result " + i + " of round " + round + " is " + got[i]
                                + ", the first call gave " + first[i]);
                        break;
                    }
                }
            }
        }
        if (failures > 0) {
            System.out.println("FAILED: " + failures);
            System.exit(1);
        }
        System.out.println("JIT-stable kernels passed (the lane sums of dotRow, dot4 and dot4x4 give the interpreter's bits in C2)");
    }

    /** Every result of the kernels under test, in one vector. */
    private static float[] run(QTensor t, float[] x, float[] w, int cols, short[] xq, float[] xs) {
        float[] out = new float[t.rows + 4 + 16 + 4 + 2 + 77 + t.rows];
        for (int r = 0; r < t.rows; r++) {
            out[r] = VectorKernels.dotRow(t, r, x, 0);
        }
        VectorKernels.dot4(w, 0, cols, x, cols, out, t.rows, false);
        float[] tile = new float[16];
        VectorKernels.dot4x4(w, 0, cols, x, 0, cols, 2 * cols, 3 * cols, tile, 0, 4, 8, 12, false);
        System.arraycopy(tile, 0, out, t.rows + 4, 16);
        int o = t.rows + 4 + 16;
        out[o++] = VectorKernels.dot(w, 0, x, 0, cols);
        out[o++] = VectorKernels.dot(w, cols, x, cols, cols - 3); // with a scalar tail
        out[o++] = VectorKernels.dot(w, 0, x, cols, 61);
        out[o++] = VectorKernels.sumSquares(x, 0, cols + 5);
        float[] sc = Arrays.copyOfRange(x, 0, 77); // an attention row: the softmax numerator and its sum
        out[o++] = VectorKernels.softmaxExp(sc, 0, 77, 0.5f);
        out[o++] = VectorKernels.softmaxExp(sc, 0, 16, 0.25f);
        System.arraycopy(sc, 0, out, o, 77);
        o += 77;
        for (int r = 0; r < t.rows; r++) {
            out[o++] = Int8Kernels.dotRow(t, r, xq, xs, 0);
        }
        return out;
    }
}
