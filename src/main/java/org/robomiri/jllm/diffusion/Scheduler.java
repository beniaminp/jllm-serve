package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.json.Json;

import java.util.Locale;
import java.util.Map;

/**
 * Noise schedules and sampler updates of the diffusion pipelines, following the diffusers implementations (their
 * {@code EulerDiscreteScheduler}, {@code EulerAncestralDiscreteScheduler} and {@code DPMSolverMultistepScheduler}) so
 * the results can be compared number for number.
 *
 * <p>The model was trained with a fixed noise schedule: {@code x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε}, with
 * {@code β_t} running from {@code beta_start} to {@code beta_end} over 1000 steps ("scaled linear" in β^½ for Stable
 * Diffusion) and ᾱ_t = ∏(1−β_s). In the Euler parametrisation the state is {@code x̃ = x_t / √ᾱ_t = x_0 + σ ε}
 * with {@code σ = √((1−ᾱ)/ᾱ)}; generation walks σ from σ_max down to 0 in a few steps.
 *
 * <ul>
 *   <li>Euler: {@code x_0 = x − σε}, derivative {@code d = (x − x_0)/σ}, {@code x ← x + d (σ_next − σ)};</li>
 *   <li>Euler ancestral: step down to {@code σ_down} then add fresh noise of size {@code σ_up}
 *       ({@code σ_up² = σ_next² (σ² − σ_next²)/σ²}, {@code σ_down² = σ_next² − σ_up²});</li>
 *   <li>DPM-Solver++ (2M): second-order multistep in the log-SNR variable λ = −ln σ, reusing the previous step's
 *       prediction; the first and the last step are first-order.</li>
 * </ul>
 */
public final class Scheduler {

    public enum Kind {
        EULER, EULER_ANCESTRAL, DPMPP_2M, DPMPP_2M_KARRAS;

        public static Kind parse(String s) {
            String n = s == null ? "" : s.toLowerCase(Locale.ROOT).replace("-", "_").replace("+", "p").replace(" ", "_");
            return switch (n) {
                case "", "default", "euler", "eulerdiscrete" -> EULER;
                case "euler_a", "euler_ancestral", "eulerancestral", "k_euler_a" -> EULER_ANCESTRAL;
                case "dpmpp_2m", "dpmpp2m", "dpmsolver++", "dpmsolver_multistep", "dpmpp_2m_k_off" -> DPMPP_2M;
                case "dpmpp_2m_karras", "dpmpp2m_karras", "karras", "k_dpmpp_2m" -> DPMPP_2M_KARRAS;
                default -> throw new IllegalArgumentException("unknown scheduler '" + s
                        + "' (euler, euler_a, dpmpp_2m, dpmpp_2m_karras)");
            };
        }
    }

    public final Kind kind;
    private final int trainSteps;
    private final String spacing;
    private final int stepsOffset;
    private final boolean vPrediction;
    private final double[] sigmaTable; // sigma of every training step, ascending
    private float[] timesteps = new float[0];
    private float[] sigmas = new float[0];
    // DPM-Solver++ history
    private float[] prevX0;
    private int lowerOrderNums;

    public Scheduler(Map<String, Object> config, Kind kind) {
        this(config, kind, null);
    }

    /** @param spacingOverride {@code leading}, {@code trailing} or {@code linspace}; null: the model's own */
    public Scheduler(Map<String, Object> config, Kind kind, String spacingOverride) {
        this.kind = kind;
        trainSteps = (int) Json.num(config, "num_train_timesteps", 1000);
        double bs = Json.dbl(config, "beta_start", 0.00085);
        double be = Json.dbl(config, "beta_end", 0.012);
        String sched = Json.str(config, "beta_schedule", "scaled_linear");
        String sp = spacingOverride != null ? spacingOverride : Json.str(config, "timestep_spacing", "linspace");
        spacing = sp;
        stepsOffset = (int) Json.num(config, "steps_offset", 0);
        String pred = Json.str(config, "prediction_type", "epsilon");
        if (!pred.equals("epsilon") && !pred.equals("v_prediction")) {
            throw new IllegalArgumentException("unsupported prediction_type " + pred);
        }
        vPrediction = pred.equals("v_prediction");
        double[] betas = new double[trainSteps];
        for (int i = 0; i < trainSteps; i++) {
            double f = trainSteps == 1 ? 0 : (double) i / (trainSteps - 1);
            betas[i] = switch (sched) {
                case "linear" -> bs + (be - bs) * f;
                case "scaled_linear" -> {
                    double a = Math.sqrt(bs) + (Math.sqrt(be) - Math.sqrt(bs)) * f;
                    yield a * a;
                }
                case "squaredcos_cap_v2" -> cosineBeta(i, trainSteps);
                default -> throw new IllegalArgumentException("unsupported beta_schedule " + sched);
            };
        }
        sigmaTable = new double[trainSteps];
        double cum = 1;
        for (int i = 0; i < trainSteps; i++) {
            cum *= 1 - (float) betas[i];
            double ac = (float) cum;
            sigmaTable[i] = (float) Math.sqrt((1 - ac) / ac);
        }
    }

    private static double cosineBeta(int i, int n) {
        java.util.function.DoubleUnaryOperator bar = t -> Math.pow(Math.cos((t + 0.008) / 1.008 * Math.PI / 2), 2);
        double t1 = (double) i / n;
        double t2 = (double) (i + 1) / n;
        return Math.min(1 - bar.applyAsDouble(t2) / bar.applyAsDouble(t1), 0.999);
    }

    // ------------------------------------------------------------------ timesteps

    public int steps() {
        return timesteps.length;
    }

    public float timestep(int i) {
        return timesteps[i];
    }

    public float sigma(int i) {
        return sigmas[i];
    }

    /** Chooses {@code n} timesteps (and the sigmas between them) and clears the sampler history. */
    public void setTimesteps(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("steps must be at least 1");
        }
        double[] ts = new double[n];
        boolean dpm = kind == Kind.DPMPP_2M || kind == Kind.DPMPP_2M_KARRAS;
        switch (spacing) {
            case "linspace" -> {
                if (dpm) {
                    double[] l = linspace(0, trainSteps - 1, n + 1);
                    for (int i = 0; i < n; i++) {
                        ts[i] = Math.rint(l[n - i]);
                    }
                } else {
                    double[] l = linspace(0, trainSteps - 1, n);
                    for (int i = 0; i < n; i++) {
                        ts[i] = (float) l[n - 1 - i];
                    }
                }
            }
            case "leading" -> {
                if (dpm) {
                    int ratio = trainSteps / (n + 1);
                    for (int i = 0; i < n; i++) {
                        ts[i] = Math.rint((double) (n - i) * ratio) + stepsOffset;
                    }
                } else {
                    int ratio = trainSteps / n;
                    for (int i = 0; i < n; i++) {
                        ts[i] = Math.rint((double) (n - 1 - i) * ratio) + stepsOffset;
                    }
                }
            }
            case "trailing" -> {
                double ratio = (double) trainSteps / n;
                for (int i = 0; i < n; i++) {
                    ts[i] = Math.rint(trainSteps - i * ratio) - 1;
                }
            }
            default -> throw new IllegalArgumentException("unsupported timestep_spacing " + spacing);
        }
        double[] sg = new double[n + 1];
        if (kind == Kind.DPMPP_2M_KARRAS) {
            double smin = sigmaTable[0];
            double smax = sigmaTable[trainSteps - 1];
            double rho = 7.0;
            for (int i = 0; i < n; i++) {
                double ramp = n == 1 ? 0 : (double) i / (n - 1);
                sg[i] = Math.pow(Math.pow(smax, 1 / rho) + ramp * (Math.pow(smin, 1 / rho) - Math.pow(smax, 1 / rho)), rho);
                ts[i] = Math.rint(sigmaToT(sg[i]));
            }
        } else {
            for (int i = 0; i < n; i++) {
                sg[i] = interp(ts[i]);
            }
        }
        sg[n] = 0;
        timesteps = new float[n];
        sigmas = new float[n + 1];
        for (int i = 0; i < n; i++) {
            timesteps[i] = (float) ts[i];
        }
        for (int i = 0; i <= n; i++) {
            sigmas[i] = (float) sg[i];
        }
        reset();
    }

    public void reset() {
        prevX0 = null;
        lowerOrderNums = 0;
    }

    private static double[] linspace(double a, double b, int n) {
        double[] r = new double[n];
        for (int i = 0; i < n; i++) {
            r[i] = n == 1 ? a : a + (b - a) * i / (n - 1);
        }
        return r;
    }

    /** {@code np.interp(t, arange(N), sigmaTable)}. */
    private double interp(double t) {
        if (t <= 0) {
            return sigmaTable[0];
        }
        if (t >= trainSteps - 1) {
            return sigmaTable[trainSteps - 1];
        }
        int lo = (int) Math.floor(t);
        double w = t - lo;
        return sigmaTable[lo] * (1 - w) + sigmaTable[lo + 1] * w;
    }

    /** diffusers {@code _sigma_to_t}: the (fractional) training timestep whose sigma is {@code sigma}, by log-interpolation. */
    private double sigmaToT(double sigma) {
        double ls = Math.log(Math.max(sigma, 1e-10));
        int low = 0;
        for (int i = 0; i < trainSteps; i++) {
            if (ls - Math.log(sigmaTable[i]) >= 0) {
                low = i;
            }
        }
        low = Math.min(low, trainSteps - 2);
        double lo = Math.log(sigmaTable[low]);
        double hi = Math.log(sigmaTable[low + 1]);
        double w = Math.max(0, Math.min(1, (lo - ls) / (lo - hi)));
        return (1 - w) * low + w * (low + 1);
    }

    /** The scale of the initial noise: σ_max (trailing / linspace) or √(σ_max²+1) (leading) for Euler, 1 for DPM-Solver. */
    public float initNoiseSigma() {
        if (kind == Kind.DPMPP_2M || kind == Kind.DPMPP_2M_KARRAS) {
            return 1f;
        }
        float max = sigmas[0];
        return spacing.equals("linspace") || spacing.equals("trailing") ? max : (float) Math.sqrt((double) max * max + 1);
    }

    // ------------------------------------------------------------------ sampling

    private boolean ve() {
        return kind == Kind.EULER || kind == Kind.EULER_ANCESTRAL;
    }

    /** What the network sees: {@code x / √(σ²+1)} for the Euler samplers, {@code x} for DPM-Solver. */
    public float[] scaleInput(float[] x, int i) {
        if (!ve()) {
            return x;
        }
        float s = (float) (1.0 / Math.sqrt((double) sigmas[i] * sigmas[i] + 1));
        float[] out = new float[x.length];
        for (int k = 0; k < x.length; k++) {
            out[k] = x[k] * s;
        }
        return out;
    }

    /** The noisy version of {@code x0} at step {@code i} (img2img, inpainting): {@code x0 + σ ε} or {@code α x0 + σ_t ε}. */
    public float[] addNoise(float[] x0, float[] noise, int i) {
        float[] out = new float[x0.length];
        float s = sigmas[i];
        if (ve()) {
            for (int k = 0; k < out.length; k++) {
                out[k] = x0[k] + noise[k] * s;
            }
        } else {
            float alpha = (float) (1.0 / Math.sqrt((double) s * s + 1));
            float sg = s * alpha;
            for (int k = 0; k < out.length; k++) {
                out[k] = alpha * x0[k] + sg * noise[k];
            }
        }
        return out;
    }

    /** The network's estimate of the clean latents x_0 at step {@code i}, from its output {@code out} and the state {@code x} before the step. */
    public float[] predictOriginal(float[] out, int i, float[] x) {
        int n = x.length;
        float sigma = sigmas[i];
        float[] x0 = new float[n];
        if (ve()) {
            if (vPrediction) {
                float c = (float) (-sigma / Math.sqrt((double) sigma * sigma + 1));
                float sk = (float) (1.0 / ((double) sigma * sigma + 1));
                for (int k = 0; k < n; k++) {
                    x0[k] = out[k] * c + x[k] * sk;
                }
            } else {
                for (int k = 0; k < n; k++) {
                    x0[k] = x[k] - sigma * out[k];
                }
            }
        } else {
            double alpha = 1.0 / Math.sqrt((double) sigma * sigma + 1);
            double sg = sigma * alpha;
            for (int k = 0; k < n; k++) {
                x0[k] = vPrediction ? (float) (alpha * x[k] - sg * out[k]) : (float) ((x[k] - sg * out[k]) / alpha);
            }
        }
        return x0;
    }

    /**
     * One update: {@code x} is the state before the step, {@code out} the network's output for its scaled version.
     *
     * @param noiseSource fresh noise (in the layout of {@code x}) for the ancestral sampler; called once per step, only when needed
     */
    public float[] step(float[] out, int i, float[] x, java.util.function.Supplier<float[]> noiseSource) {
        int n = x.length;
        float sigma = sigmas[i];
        float sigmaNext = sigmas[i + 1];
        float[] res = new float[n];
        float[] x0 = predictOriginal(out, i, x);
        if (ve()) {
            if (kind == Kind.EULER) {
                float dt = sigmaNext - sigma;
                for (int k = 0; k < n; k++) {
                    res[k] = x[k] + (x[k] - x0[k]) / sigma * dt;
                }
            } else {
                double up2 = (double) sigmaNext * sigmaNext * ((double) sigma * sigma - (double) sigmaNext * sigmaNext) / ((double) sigma * sigma);
                float sigmaUp = (float) Math.sqrt(Math.max(0, up2));
                float sigmaDown = (float) Math.sqrt(Math.max(0, (double) sigmaNext * sigmaNext - (double) sigmaUp * sigmaUp));
                float dt = sigmaDown - sigma;
                float[] noise = noiseSource.get();
                for (int k = 0; k < n; k++) {
                    res[k] = x[k] + (x[k] - x0[k]) / sigma * dt + noise[k] * sigmaUp;
                }
            }
            return res;
        }
        // DPM-Solver++ (data prediction), midpoint second order
        double alphaS = 1.0 / Math.sqrt((double) sigma * sigma + 1);
        double sigS = sigma * alphaS;
        boolean last = i == sigmas.length - 2;
        double alphaT = 1.0 / Math.sqrt((double) sigmaNext * sigmaNext + 1);
        double sigT = sigmaNext * alphaT;
        double lamT = Math.log(alphaT) - Math.log(sigT);
        double lamS = Math.log(alphaS) - Math.log(sigS);
        double h = lamT - lamS;
        double em1 = Math.exp(-h) - 1.0; // −1 when σ_next = 0
        if (lowerOrderNums < 1 || last || prevX0 == null) {
            double c1 = sigT / sigS;
            double c2 = alphaT * em1;
            for (int k = 0; k < n; k++) {
                res[k] = (float) (c1 * x[k] - c2 * x0[k]);
            }
        } else {
            float sigmaPrev = sigmas[i - 1];
            double alphaP = 1.0 / Math.sqrt((double) sigmaPrev * sigmaPrev + 1);
            double sigP = sigmaPrev * alphaP;
            double lamP = Math.log(alphaP) - Math.log(sigP);
            double h0 = lamS - lamP;
            double r0 = h0 / h;
            double c1 = sigT / sigS;
            double c2 = alphaT * em1;
            for (int k = 0; k < n; k++) {
                double d0 = x0[k];
                double d1 = (1.0 / r0) * (x0[k] - prevX0[k]);
                res[k] = (float) (c1 * x[k] - c2 * d0 - 0.5 * c2 * d1);
            }
        }
        prevX0 = x0;
        lowerOrderNums = Math.min(lowerOrderNums + 1, 2);
        return res;
    }
}
