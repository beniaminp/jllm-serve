package org.robomiri.jllm.model;

/**
 * Rotary position embedding tables in the Hugging Face "rotate half" convention: dimension {@code i} is paired
 * with {@code i + headDim/2} and rotated by {@code pos * invFreq[i]}, where {@code invFreq[i]} is
 * {@code theta^(-2i/headDim)} optionally rescaled for long-context models (see {@link #invFreq}).
 */
public final class Rope {

    /** {@code [ctx * headDim/2]} cosines. */
    public final float[] cos;
    /** {@code [ctx * headDim/2]} sines. */
    public final float[] sin;

    public Rope(int headDim, double theta, int ctx) {
        this(headDim, theta, ctx, null);
    }

    public Rope(int headDim, double theta, int ctx, ModelConfig.RopeScaling scaling) {
        int half = headDim / 2;
        cos = new float[ctx * half];
        sin = new float[ctx * half];
        double[] inv = invFreq(headDim, theta, scaling);
        for (int i = 0; i < half; i++) {
            double freq = inv[i];
            for (int p = 0; p < ctx; p++) {
                double a = p * freq;
                cos[p * half + i] = (float) Math.cos(a);
                sin[p * half + i] = (float) Math.sin(a);
            }
        }
    }

    public static Rope of(ModelConfig c, int ctx) {
        return new Rope(c.headDim(), c.ropeTheta(), ctx, c.ropeScaling());
    }

    /**
     * The {@code headDim/2} inverse frequencies. Plain RoPE: {@code theta^(-2i/headDim)}. With {@code linear}
     * scaling they are divided by the factor. With {@code llama3} scaling (transformers'
     * {@code _compute_llama3_parameters}) each frequency is treated by its wavelength {@code 2π/f}: short
     * wavelengths (below {@code original/high_freq_factor}) are kept, long ones (above
     * {@code original/low_freq_factor}) are divided by {@code factor}, and in between the two are blended
     * with {@code smooth = (original/wavelen - low) / (high - low)}.
     */
    public static double[] invFreq(int headDim, double theta, ModelConfig.RopeScaling sc) {
        int half = headDim / 2;
        double[] f = new double[half];
        for (int i = 0; i < half; i++) {
            f[i] = 1.0 / Math.pow(theta, (2.0 * i) / headDim);
        }
        if (sc == null) {
            return f;
        }
        if (sc.type().equals("linear")) {
            for (int i = 0; i < half; i++) {
                f[i] /= sc.factor();
            }
            return f;
        }
        double lowWavelen = sc.originalMaxPositions() / sc.lowFreqFactor();
        double highWavelen = sc.originalMaxPositions() / sc.highFreqFactor();
        for (int i = 0; i < half; i++) {
            double wavelen = 2 * Math.PI / f[i];
            if (wavelen < highWavelen) {
                continue;
            }
            if (wavelen > lowWavelen) {
                f[i] /= sc.factor();
            } else {
                double smooth = (sc.originalMaxPositions() / wavelen - sc.lowFreqFactor())
                        / (sc.highFreqFactor() - sc.lowFreqFactor());
                f[i] = (1 - smooth) * f[i] / sc.factor() + smooth * f[i];
            }
        }
        return f;
    }
}
