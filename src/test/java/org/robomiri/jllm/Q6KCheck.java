package org.robomiri.jllm;

import org.robomiri.jllm.cpu.Q6KKernels;
import org.robomiri.jllm.model.Q6K;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
import java.util.Random;

/**
 * Q6_K checks (no model needed), called from {@link SmokeTest}:
 * <ol>
 *   <li>rows built on the Q6_K grid (f16 d, int8 scales, every sub-block using codes -32 and 31) round-trip exactly;</li>
 *   <li>random rows reconstruct within a bound of the format and a zero row stays zero;</li>
 *   <li>the SIMD dot and dequantize kernels agree with the scalar reference.</li>
 * </ol>
 * Runs for a row length that is a multiple of 256 (llama.cpp layout) and one with a partial trailing super-block.
 */
final class Q6KCheck {

    private Q6KCheck() {
    }

    static void run() {
        for (int cols : new int[]{1024, 320}) {
            exactRoundTrip(cols);
            randomRows(cols);
        }
        zeroRow();
    }

    /** Rows whose sub-blocks are exactly representable must come back (up to float rounding). */
    private static void exactRoundTrip(int cols) {
        Random rnd = new Random(7);
        int rows = 8;
        float[][] src = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int sb = 0; sb * Q6K.BLOCK < cols; sb++) {
                // one sub-block has |scale| = 128 d so that the quantizer's d = -max_scale / 128 lands on d0
                float d0 = f16(0.0002f + 0.002f * rnd.nextFloat());
                int nsub = Math.min(Q6K.BLOCK, cols - sb * Q6K.BLOCK) / Q6K.SUB;
                int[] k = new int[nsub];
                for (int j = 0; j < nsub; j++) {
                    k[j] = 1 + rnd.nextInt(127);
                }
                k[rnd.nextInt(nsub)] = 128;
                for (int j = 0; j < nsub; j++) {
                    int minusAt = rnd.nextInt(Q6K.SUB);
                    for (int l = 0; l < Q6K.SUB; l++) {
                        int q = l == minusAt ? -32 : l == (minusAt + 1) % Q6K.SUB ? 31 : rnd.nextInt(64) - 32;
                        src[r][sb * Q6K.BLOCK + j * Q6K.SUB + l] = d0 * k[j] * q;
                    }
                }
            }
        }
        try (Arena a = Arena.ofShared()) {
            QTensor t = QTensor.quantize(a, QType.Q6_K, rows, cols, (r, dst) -> System.arraycopy(src[r], 0, dst, 0, cols));
            int nsb = (cols + Q6K.BLOCK - 1) / Q6K.BLOCK;
            SmokeTest.check(t.blocksPerRow() == nsb && t.byteSize() == QTensor.PAGE
                            && t.rowOffset(1) == (long) nsb * Q6K.BLOCK_BYTES,
                    "q6k layout: " + nsb + " super-blocks of 210 bytes per row, no scale region (cols " + cols + ")");
            float[] back = new float[cols];
            float[] simd = new float[cols];
            float maxRel = 0;
            float maxDiff = 0;
            for (int r = 0; r < rows; r++) {
                t.dequantizeRow(r, back, 0);
                Q6KKernels.dequant(t, r, simd, 0);
                float amax = 0;
                for (float v : src[r]) {
                    amax = Math.max(amax, Math.abs(v));
                }
                for (int i = 0; i < cols; i++) {
                    maxRel = Math.max(maxRel, Math.abs(back[i] - src[r][i]) / amax);
                    maxDiff = Math.max(maxDiff, Math.abs(back[i] - simd[i]));
                }
            }
            SmokeTest.check(maxRel <= 1e-5f, "q6k exact round trip of on-grid rows (cols " + cols + "): max error " + maxRel + " of the row max");
            SmokeTest.check(maxDiff <= 1e-6f, "q6k SIMD dequant == scalar dequant (cols " + cols + "): max diff " + maxDiff);
        }
    }

    /**
     * Gaussian rows with outliers. The step of a sub-block is {@code |d·sc_j|}; a code is off by at most half a
     * step, plus the int8 rounding of the sub-block scale (clipping at the ends of -32..31), so 4% of the
     * super-block's largest magnitude covers every sub-block of these rows.
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
            QTensor t = QTensor.quantize(a, QType.Q6_K, rows, cols, (r, dst) -> System.arraycopy(src[r], 0, dst, 0, cols));
            float[] back = new float[cols];
            double sumErr2 = 0;
            double sumW2 = 0;
            float worstRatio = 0;
            float worstDot = 0;
            float worstDotRel = 0;
            for (int r = 0; r < rows; r++) {
                t.dequantizeRow(r, back, 0);
                for (int sb = 0; sb * Q6K.BLOCK < cols; sb++) {
                    int n = Math.min(Q6K.BLOCK, cols - sb * Q6K.BLOCK);
                    float amax = 0;
                    float maxErr = 0;
                    for (int i = sb * Q6K.BLOCK; i < sb * Q6K.BLOCK + n; i++) {
                        amax = Math.max(amax, Math.abs(src[r][i]));
                        float e = Math.abs(back[i] - src[r][i]);
                        maxErr = Math.max(maxErr, e);
                        sumErr2 += e * e;
                        sumW2 += (double) src[r][i] * src[r][i];
                    }
                    worstRatio = Math.max(worstRatio, maxErr / amax);
                }
                // SIMD dot vs scalar dequantize-then-dot (double accumulation)
                double ref = 0;
                double mag = 0;
                for (int i = 0; i < cols; i++) {
                    ref += (double) back[i] * x[i];
                    mag += Math.abs((double) back[i] * x[i]);
                }
                float simd = Q6KKernels.dot(t, r, x, 0);
                float scalar = Q6K.dotRow(t.data, t.rowOffset(r), cols, x, 0);
                worstDot = Math.max(worstDot, (float) (Math.max(Math.abs(simd - ref), Math.abs(scalar - ref)) / mag));
                worstDotRel = Math.max(worstDotRel, (float) (Math.abs(simd - ref) / Math.abs(ref)));
            }
            double rmsRel = Math.sqrt(sumErr2 / sumW2);
            SmokeTest.check(worstRatio <= 0.04f, String.format("q6k random rows (cols %d): max error = %.4f x super-block max, "
                    + "rms error = %.2f%% of rms weight", cols, worstRatio, 100 * rmsRel));
            SmokeTest.check(rmsRel <= 0.03, "q6k rms error under 3% of the rms weight (" + rmsRel + ")");
            SmokeTest.check(worstDot <= 1e-4f, String.format("q6k SIMD dot == scalar dequantize-then-dot (cols %d): "
                    + "max |diff| = %.1e x sum|w.x| (%.1e relative to the dot)", cols, worstDot, worstDotRel));
        }
    }

    /** An all-zero row quantizes to all-zero bytes (d = 0) and dequantizes to zeros. */
    private static void zeroRow() {
        try (Arena a = Arena.ofShared()) {
            QTensor t = QTensor.quantize(a, QType.Q6_K, 2, 512, (r, dst) -> java.util.Arrays.fill(dst, 0f));
            float[] back = new float[512];
            java.util.Arrays.fill(back, 1f);
            t.dequantizeRow(1, back, 0);
            boolean zero = true;
            for (float v : back) {
                zero &= v == 0f;
            }
            SmokeTest.check(zero, "q6k zero row dequantizes to zeros");
        }
    }

    private static float f16(float v) {
        return Float.float16ToFloat(Float.floatToFloat16(v));
    }
}
