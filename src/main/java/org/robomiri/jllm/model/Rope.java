package org.robomiri.jllm.model;

/**
 * Rotary position embedding tables in the Hugging Face "rotate half" convention: dimension {@code i} is paired
 * with {@code i + headDim/2} and rotated by {@code pos * theta^(-2i/headDim)}.
 */
public final class Rope {

    /** {@code [ctx * headDim/2]} cosines. */
    public final float[] cos;
    /** {@code [ctx * headDim/2]} sines. */
    public final float[] sin;

    public Rope(int headDim, double theta, int ctx) {
        int half = headDim / 2;
        cos = new float[ctx * half];
        sin = new float[ctx * half];
        for (int i = 0; i < half; i++) {
            double freq = 1.0 / Math.pow(theta, (2.0 * i) / headDim);
            for (int p = 0; p < ctx; p++) {
                double a = p * freq;
                cos[p * half + i] = (float) Math.cos(a);
                sin[p * half + i] = (float) Math.sin(a);
            }
        }
    }
}
