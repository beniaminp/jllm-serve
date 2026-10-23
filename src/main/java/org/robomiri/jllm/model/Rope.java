package org.robomiri.jllm.model;

/**
 * The cos/sin table of one {@link RopeTable} over a context length: entry {@code i} of position {@code p} is
 * {@code p * invFreq[i]}, with {@code invFreq[i] = theta^(-2i/rotaryDim)} optionally rescaled for long-context
 * models ({@link #invFreq}) and the table multiplied by the scaling's amplitude ({@link #amplitude}).
 *
 * <p>The table does not know how a head pairs its dimensions: with {@link RopeTable.Pairing#HALF} the backends
 * rotate {@code (i, i + rotaryDim/2)} by entry {@code i}, with {@link RopeTable.Pairing#INTERLEAVED}
 * ({@link InterleavedRope}) the pair {@code (2i, 2i+1)}. Dimensions {@code >= rotaryDim} of a head stay put.
 */
public final class Rope {

    /** {@code [ctx * rotaryDim/2]} cosines. */
    public final float[] cos;
    /** {@code [ctx * rotaryDim/2]} sines. */
    public final float[] sin;

    /** A plain table: {@code rotaryDim/2} frequencies {@code theta^(-2i/rotaryDim)}, no scaling. */
    public Rope(int rotaryDim, double theta, int ctx) {
        this(new RopeTable(rotaryDim, theta, Scaling.NONE, RopeTable.Pairing.HALF), ctx);
    }

    public Rope(RopeTable t, int ctx) {
        int half = t.rotaryDim() / 2;
        cos = new float[ctx * half];
        sin = new float[ctx * half];
        double[] inv = invFreq(t, ctx);
        double amp = amplitude(t.scaling());
        for (int i = 0; i < half; i++) {
            double freq = inv[i];
            for (int p = 0; p < ctx; p++) {
                double a = p * freq;
                cos[p * half + i] = (float) (Math.cos(a) * amp);
                sin[p * half + i] = (float) (Math.sin(a) * amp);
            }
        }
    }

    /**
     * The cos/sin rows of slots {@code [from, ctx)}, laid out like {@link #cos} / {@link #sin} (row {@code i - from}
     * at {@code (i - from) * half}), when the position of slot {@code i} pair {@code j} is {@code p.position(i, j)}
     * instead of {@code i} ({@link RopePositions}); {@code p == null} gives the plain table's rows. A backend whose
     * table sits in device memory copies these rows over its own.
     */
    public static void rows(RopeTable t, int ctx, int from, RopePositions p, float[] cos, float[] sin) {
        int half = t.rotaryDim() / 2;
        double[] inv = invFreq(t, ctx);
        double amp = amplitude(t.scaling());
        for (int i = from; i < ctx; i++) {
            int o = (i - from) * half;
            for (int j = 0; j < half; j++) {
                double a = (double) (p == null ? i : p.position(i, j)) * inv[j];
                cos[o + j] = (float) (Math.cos(a) * amp);
                sin[o + j] = (float) (Math.sin(a) * amp);
            }
        }
    }

    /** Replaces the rows of slots {@code [from, ctx)} of this table ({@link #rows}); {@code ctx} is the table's context. */
    public void refill(RopeTable t, int ctx, int from, RopePositions p) {
        int half = t.rotaryDim() / 2;
        float[] c = new float[(ctx - from) * half];
        float[] s = new float[(ctx - from) * half];
        rows(t, ctx, from, p, c, s);
        System.arraycopy(c, 0, cos, from * half, c.length);
        System.arraycopy(s, 0, sin, from * half, s.length);
    }

    /**
     * Multiplier of cos and sin (so of {@code q.k} squared): LongRoPE's {@link #attentionFactor}, YaRN's
     * {@code attention_factor} or else {@code yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)}
     * (modeling_deepseek.py), 1 for every other scaling.
     */
    public static double amplitude(Scaling s) {
        return switch (s) {
            case Scaling.LongRope lr -> attentionFactor(lr);
            case Scaling.Yarn y -> y.attentionFactor() > 0 ? y.attentionFactor()
                    : yarnMscale(y.factor(), y.mscale()) / yarnMscale(y.factor(), y.mscaleAllDim());
            case Scaling.None n -> 1.0;
            case Scaling.Identity i -> 1.0;
            case Scaling.Linear l -> 1.0;
            case Scaling.Llama3 l -> 1.0;
        };
    }

    /**
     * LongRoPE's multiplier on cos and sin. transformers' {@code _compute_longrope_parameters}:
     * {@code attention_factor} if given, else 1 for {@code factor <= 1}, else
     * {@code sqrt(1 + ln(factor) / ln(original_max_position_embeddings))}.
     */
    public static double attentionFactor(Scaling.LongRope sc) {
        if (sc.attentionFactor() > 0) {
            return sc.attentionFactor();
        }
        return sc.factor() <= 1.0 ? 1.0 : Math.sqrt(1 + Math.log(sc.factor()) / Math.log(sc.originalMaxPositions()));
    }

    /**
     * The {@code rotaryDim/2} inverse frequencies of {@code t} for a context of {@code ctx} tokens; the pairs from
     * {@link RopeTable#activePairs()} on are 0 (Gemma 4's proportional RoPE leaves the rest of the head unrotated).
     */
    public static double[] invFreq(RopeTable t, int ctx) {
        double[] f = invFreq(t.rotaryDim(), t.theta(), t.scaling(), ctx);
        if (t.activePairs() < f.length) {
            java.util.Arrays.fill(f, t.activePairs(), f.length, 0.0);
        }
        return f;
    }

    /**
     * The {@code rotaryDim/2} inverse frequencies. Plain RoPE: {@code theta^(-2i/rotaryDim)}. {@code linear}:
     * divided by the factor. {@code llama3} (transformers' {@code _compute_llama3_parameters}): each frequency is
     * treated by its wavelength {@code 2π/f}: short wavelengths (below {@code original/high_freq_factor}) are
     * kept, long ones (above {@code original/low_freq_factor}) are divided by {@code factor}, and in between the
     * two are blended with {@code smooth = (original/wavelen - low) / (high - low)}. {@code longrope}:
     * {@code 1 / (ext[i] * theta^(2i/rotDim))} with {@code ext} the long factors when {@code ctx} exceeds
     * {@code original_max_position_embeddings} and the short ones otherwise (transformers picks per sequence
     * length; here the choice is made once from the context size the backend is created with). {@code yarn}:
     * {@link #yarnInvFreq}.
     */
    public static double[] invFreq(int rotaryDim, double theta, Scaling scaling, int ctx) {
        int half = rotaryDim / 2;
        double[] f = new double[half];
        for (int i = 0; i < half; i++) {
            f[i] = 1.0 / Math.pow(theta, (2.0 * i) / rotaryDim);
        }
        switch (scaling) {
            case Scaling.None n -> {
            }
            case Scaling.Identity i -> {
                return new double[half]; // frequency 0: cos 1, sin 0
            }
            case Scaling.Linear l -> {
                for (int i = 0; i < half; i++) {
                    f[i] /= l.factor();
                }
            }
            case Scaling.Llama3 sc -> {
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
                        double span = sc.highFreqFactor() - sc.lowFreqFactor();
                        double smooth = span > 0 ? (sc.originalMaxPositions() / wavelen - sc.lowFreqFactor()) / span : 0;
                        f[i] = (1 - smooth) * f[i] / sc.factor() + smooth * f[i];
                    }
                }
            }
            case Scaling.LongRope sc -> {
                double[] ext = ctx > sc.originalMaxPositions() ? sc.longFactor() : sc.shortFactor();
                for (int i = 0; i < half; i++) {
                    f[i] = 1.0 / (ext[i] * Math.pow(theta, (2.0 * i) / rotaryDim));
                }
            }
            case Scaling.Yarn y -> {
                return yarnInvFreq(rotaryDim, theta, y.factor(), y.originalMaxPositions(), y.betaFast(), y.betaSlow(), y.truncate());
            }
        }
        return f;
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
        return yarnInvFreq(dim, base, factor, origMax, betaFast, betaSlow, true);
    }

    /** As above; {@code truncate = false} keeps the (clamped) correction range fractional (transformers' {@code truncate}). */
    public static double[] yarnInvFreq(int dim, double base, double factor, int origMax, double betaFast, double betaSlow,
                                       boolean truncate) {
        int half = dim / 2;
        double lowDim = yarnCorrectionDim(betaFast, dim, base, origMax);
        double highDim = yarnCorrectionDim(betaSlow, dim, base, origMax);
        double low = Math.max(truncate ? Math.floor(lowDim) : lowDim, 0);
        double high = Math.min(truncate ? Math.ceil(highDim) : highDim, dim - 1);
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
}
