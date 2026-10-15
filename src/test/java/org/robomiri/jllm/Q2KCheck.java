package org.robomiri.jllm;

import org.robomiri.jllm.cpu.Q2KKernels;
import org.robomiri.jllm.model.Q2K;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
import java.util.Random;

/**
 * Q2_K checks (no model needed), called from {@link SmokeTest}:
 * <ol>
 *   <li>rows built on the Q2_K grid (f16 d/dmin, 4-bit sc/m, every sub-block using codes 0 and 3) round-trip exactly;</li>
 *   <li>random rows reconstruct within the analytic bound of the format;</li>
 *   <li>the SIMD dot and dequantize kernels agree with the scalar reference.</li>
 * </ol>
 * Runs for a row length that is a multiple of 256 (llama.cpp layout) and one with a partial trailing super-block.
 */
final class Q2KCheck {

    private Q2KCheck() {
    }

    static void run() {
        for (int cols : new int[]{1024, 320}) {
            exactRoundTrip(cols);
            randomRows(cols);
        }
    }

    /** Rows whose sub-blocks are exactly representable must come back bit-for-bit (up to float rounding). */
    private static void exactRoundTrip(int cols) {
        Random rnd = new Random(7);
        int rows = 8;
        float[][] src = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int sb = 0; sb * Q2K.BLOCK < cols; sb++) {
                float d = f16(0.0005f + 0.004f * rnd.nextFloat());
                float dmin = f16(0.0005f + 0.004f * rnd.nextFloat());
                int nsub = Math.min(Q2K.BLOCK, cols - sb * Q2K.BLOCK) / Q2K.SUB;
                int[] sc = new int[nsub];
                int[] m = new int[nsub];
                for (int j = 0; j < nsub; j++) {
                    sc[j] = 1 + rnd.nextInt(15);
                    m[j] = rnd.nextInt(16);
                }
                sc[rnd.nextInt(nsub)] = 15; // d = max(scale)/15 and dmin = max(min)/15 must land on the generated d/dmin
                m[rnd.nextInt(nsub)] = 15;
                for (int j = 0; j < nsub; j++) {
                    for (int l = 0; l < Q2K.SUB; l++) {
                        int q = l == 0 ? 0 : l == 1 ? 3 : rnd.nextInt(4); // span the full code range
                        src[r][sb * Q2K.BLOCK + j * Q2K.SUB + l] = d * sc[j] * q - dmin * m[j];
                    }
                }
            }
        }
        try (Arena a = Arena.ofShared()) {
            QTensor t = QTensor.quantize(a, QType.Q2_K, rows, cols, (r, dst) -> System.arraycopy(src[r], 0, dst, 0, cols));
            int nsb = (cols + Q2K.BLOCK - 1) / Q2K.BLOCK;
            SmokeTest.check(t.blocksPerRow() == nsb && t.byteSize() == QTensor.PAGE
                            && t.rowOffset(1) == (long) nsb * Q2K.BLOCK_BYTES,
                    "q2k layout: " + nsb + " super-blocks of 84 bytes per row, no scale region (cols " + cols + ")");
            float[] back = new float[cols];
            float[] simd = new float[cols];
            float maxErr = 0;
            float maxDiff = 0;
            for (int r = 0; r < rows; r++) {
                t.dequantizeRow(r, back, 0);
                Q2KKernels.dequant(t, r, simd, 0);
                for (int i = 0; i < cols; i++) {
                    maxErr = Math.max(maxErr, Math.abs(back[i] - src[r][i]));
                    maxDiff = Math.max(maxDiff, Math.abs(back[i] - simd[i]));
                }
            }
            SmokeTest.check(maxErr <= 1e-6f, "q2k exact round trip of on-grid rows (cols " + cols + "): max error " + maxErr);
            SmokeTest.check(maxDiff <= 1e-6f, "q2k SIMD dequant == scalar dequant (cols " + cols + "): max diff " + maxDiff);
        }
    }

    /**
     * Gaussian rows with outliers. Per sub-block the format's error is at most about {@code range/6} (4 levels over
     * the span) plus the 4-bit rounding of the sub-block scale ({@code ≤ d/2 · 3 = max(scale)/10}) and min
     * ({@code ≤ max(min)/30}); {@code 0.3 · widest sub-block range} of the super-block covers all three terms.
     */
    private static void randomRows(int cols) {
        Random rnd = new Random(11);
        int rows = 64;
        float[][] src = new float[rows][cols];
        for (float[] row : src) {
            for (int i = 0; i < cols; i++) {
                row[i] = (float) (rnd.nextGaussian() * 0.02);
            }
            row[rnd.nextInt(cols)] *= 6; // one outlier per row, as in real attention matrices
        }
        float[] x = new float[cols];
        for (int i = 0; i < cols; i++) {
            x[i] = rnd.nextFloat() * 2 - 1;
        }
        try (Arena a = Arena.ofShared()) {
            QTensor t = QTensor.quantize(a, QType.Q2_K, rows, cols, (r, dst) -> System.arraycopy(src[r], 0, dst, 0, cols));
            float[] back = new float[cols];
            double sumErr2 = 0;
            double sumW2 = 0;
            float worstRatio = 0;
            float worstDot = 0;
            float worstDotRel = 0;
            for (int r = 0; r < rows; r++) {
                t.dequantizeRow(r, back, 0);
                for (int sb = 0; sb * Q2K.BLOCK < cols; sb++) {
                    int n = Math.min(Q2K.BLOCK, cols - sb * Q2K.BLOCK);
                    float widest = 0;
                    float maxErr = 0;
                    for (int j = 0; j < n / Q2K.SUB; j++) {
                        float lo = Float.MAX_VALUE;
                        float hi = -Float.MAX_VALUE;
                        for (int l = 0; l < Q2K.SUB; l++) {
                            int i = sb * Q2K.BLOCK + j * Q2K.SUB + l;
                            lo = Math.min(lo, Math.min(src[r][i], 0)); // the min is clamped to <= 0 by the format
                            hi = Math.max(hi, src[r][i]);
                            float e = Math.abs(back[i] - src[r][i]);
                            maxErr = Math.max(maxErr, e);
                            sumErr2 += e * e;
                            sumW2 += (double) src[r][i] * src[r][i];
                        }
                        widest = Math.max(widest, hi - lo);
                    }
                    worstRatio = Math.max(worstRatio, maxErr / widest);
                }
                // SIMD dot vs scalar dequantize-then-dot (double accumulation)
                double ref = 0;
                double mag = 0;
                for (int i = 0; i < cols; i++) {
                    ref += (double) back[i] * x[i];
                    mag += Math.abs((double) back[i] * x[i]);
                }
                float simd = Q2KKernels.dot(t, r, x, 0);
                float scalar = Q2K.dotRow(t.data, t.rowOffset(r), cols, x, 0);
                worstDot = Math.max(worstDot, (float) (Math.max(Math.abs(simd - ref), Math.abs(scalar - ref)) / mag));
                worstDotRel = Math.max(worstDotRel, (float) (Math.abs(simd - ref) / Math.abs(ref)));
            }
            double rmsRel = Math.sqrt(sumErr2 / sumW2);
            SmokeTest.check(worstRatio <= 0.3f, String.format("q2k random rows (cols %d): max error = %.3f x widest sub-block range, "
                    + "rms error = %.1f%% of rms weight", cols, worstRatio, 100 * rmsRel));
            SmokeTest.check(worstDot <= 1e-4f, String.format("q2k SIMD dot == scalar dequantize-then-dot (cols %d): "
                    + "max |diff| = %.1e x sum|w.x| (%.1e relative to the dot)", cols, worstDot, worstDotRel));
        }
    }

    private static float f16(float v) {
        return Float.float16ToFloat(Float.floatToFloat16(v));
    }
}
