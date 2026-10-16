package org.robomiri.jllm.model;

/**
 * Rotary position embedding tables in the Hugging Face "rotate half" convention: dimension {@code i} is paired
 * with {@code i + headDim/2} and rotated by {@code pos * invFreq[i]}, where {@code invFreq[i]} is
 * {@code theta^(-2i/headDim)} optionally rescaled for long-context models (see {@link #invFreq}).
 */
public final class Rope {
    // GLM (ModelConfig.ropeInterleaved) uses the same tables, built for the rotated slice only: rotaryDim/2
    // entries per position and frequencies theta^(-2i/rotaryDim). The pair (2i, 2i+1) of the head is rotated by
    // entry i instead of the pair (i, i + rotaryDim/2).

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
        double[] inv = invFreq(headDim, theta, scaling, ctx);
        double amp = attentionFactor(scaling);
        for (int i = 0; i < half; i++) {
            double freq = inv[i];
            for (int p = 0; p < ctx; p++) {
                double a = p * freq;
                cos[p * half + i] = (float) (Math.cos(a) * amp);
                sin[p * half + i] = (float) (Math.sin(a) * amp);
            }
        }
    }

    public static Rope of(ModelConfig c, int ctx) {
        if (c.deepseek() != null) {
            return deepseek(c.deepseek(), c.ropeTheta(), ctx);
        }
        return new Rope(c.rotaryDim(), c.ropeTheta(), ctx, c.ropeScaling());
    }

    private Rope(double[] inv, int ctx, double mscale) {
        int half = inv.length;
        cos = new float[ctx * half];
        sin = new float[ctx * half];
        for (int i = 0; i < half; i++) {
            for (int p = 0; p < ctx; p++) {
                double a = p * inv[i];
                cos[p * half + i] = (float) (Math.cos(a) * mscale);
                sin[p * half + i] = (float) (Math.sin(a) * mscale);
            }
        }
    }

    /**
     * The table for the {@code qk_rope_head_dim} wide rotary part of DeepSeek MLA, as in modeling_deepseek.py:
     * YaRN frequencies ({@link #yarnInvFreq}) and cos/sin multiplied by
     * {@code yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)}. The pairs are the
     * interleaved dimensions (2i, 2i+1) of the checkpoint layout, rotated by {@code pos * invFreq[i]}.
     */
    public static Rope deepseek(DeepSeekConfig d, double theta, int ctx) {
        DeepSeekConfig.Yarn y = d.yarn();
        if (y == null) {
            return new Rope(invFreq(d.qkRopeDim(), theta, null), ctx, 1.0);
        }
        double[] inv = yarnInvFreq(d.qkRopeDim(), theta, y.factor(), y.originalMaxPositions(), y.betaFast(), y.betaSlow());
        return new Rope(inv, ctx, yarnMscale(y.factor(), y.mscale()) / yarnMscale(y.factor(), y.mscaleAllDim()));
    }

    /** {@code yarn_get_mscale}: {@code 0.1 * mscale * ln(scale) + 1}, and 1 when {@code scale <= 1}. */
    public static double yarnMscale(double scale, double mscale) {
        return scale <= 1 ? 1.0 : 0.1 * mscale * Math.log(scale) + 1.0;
    }

    /** {@code yarn_find_correction_dim}: the dimension whose wavelength makes {@code rotations} turns in {@code origMax}. */
    static double yarnCorrectionDim(double rotations, int dim, double base, int origMax) {
        return dim * Math.log(origMax / (rotations * 2 * Math.PI)) / (2 * Math.log(base));
    }

    /**
     * YaRN inverse frequencies ({@code DeepseekV2YarnRotaryEmbedding}): dimension pair {@code i} below the
     * correction range keeps the plain frequency (extrapolation), above it the frequency is divided by
     * {@code factor} (interpolation), in between the two are blended with a linear ramp. The range is
     * {@code [max(floor(dim_fast), 0), min(ceil(dim_slow), dim-1)]} with {@code dim_x = yarnCorrectionDim(beta_x)}.
     */
    public static double[] yarnInvFreq(int dim, double base, double factor, int origMax, double betaFast, double betaSlow) {
        int half = dim / 2;
        double low = Math.max(Math.floor(yarnCorrectionDim(betaFast, dim, base, origMax)), 0);
        double high = Math.min(Math.ceil(yarnCorrectionDim(betaSlow, dim, base, origMax)), dim - 1);
        if (low == high) {
            high += 0.001; // prevent a singularity
        }
        double[] f = new double[half];
        for (int i = 0; i < half; i++) {
            double extra = 1.0 / Math.pow(base, (2.0 * i) / dim);
            double inter = 1.0 / (factor * Math.pow(base, (2.0 * i) / dim));
            double ramp = Math.min(Math.max((i - low) / (high - low), 0), 1);
            f[i] = inter * ramp + extra * (1 - ramp);
        }
        return f;
    }

    /**
     * The table for layers with {@link ModelConfig#usesLocalRope local RoPE} (Gemma 3 sliding layers): base
     * {@code rope_local_base_freq}, never scaled, over the same {@link ModelConfig#rotaryDim rotated width} as
     * {@link #of}. Other layers use {@link #of}.
     */
    public static Rope local(ModelConfig c, int ctx) {
        return new Rope(c.rotaryDim(), c.gemma().localRopeTheta(), ctx, null);
    }

    /**
     * LongRoPE's multiplier on cos and sin (so on q.k it appears squared). transformers'
     * {@code _compute_longrope_parameters}: {@code attention_factor} if given, else 1 for {@code factor <= 1}, else
     * {@code sqrt(1 + ln(factor) / ln(original_max_position_embeddings))}. 1 for every other scaling type.
     */
    public static double attentionFactor(ModelConfig.RopeScaling sc) {
        if (sc == null || !sc.type().equals("longrope")) {
            return 1.0;
        }
        if (sc.attentionFactor() > 0) {
            return sc.attentionFactor();
        }
        return sc.factor() <= 1.0 ? 1.0 : Math.sqrt(1 + Math.log(sc.factor()) / Math.log(sc.originalMaxPositions()));
    }

    /**
     * LongRoPE: {@code inv_freq[i] = 1 / (ext[i] * theta^(2i/rotDim))} with {@code ext} the long factors when the
     * context exceeds {@code original_max_position_embeddings} and the short ones otherwise. transformers picks per
     * sequence length; here the choice is made once from the context size the backend is created with.
     */
    public static double[] invFreq(int rotDim, double theta, ModelConfig.RopeScaling sc, int ctx) {
        if (sc == null || !sc.type().equals("longrope")) {
            return invFreq(rotDim, theta, sc);
        }
        double[] ext = ctx > sc.originalMaxPositions() ? sc.longFactor() : sc.shortFactor();
        int half = rotDim / 2;
        double[] f = new double[half];
        for (int i = 0; i < half; i++) {
            f[i] = 1.0 / (ext[i] * Math.pow(theta, (2.0 * i) / rotDim));
        }
        return f;
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
