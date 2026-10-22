package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.json.Json;

import java.util.Map;

/**
 * Samplers for <em>flow-matching</em> models (Wan): the network predicts the velocity {@code v = ε − x₀} of the straight path
 * {@code x_σ = (1 − σ) x₀ + σ ε} and generation walks σ from about 1 down to 0.
 *
 * <p>The sigmas are those of diffusers' {@code UniPCMultistepScheduler} with {@code use_flow_sigmas}: {@code linspace(1, 1/1000, n)}
 * (without the last point), shifted by {@code σ ← s σ / (1 + (s − 1) σ)} with {@code s = flow_shift} (3.0 for Wan 1.3B at 480p), the first
 * one nudged by 1e-6 below 1, a final 0; the network sees the integer {@code ⌊1000 σ⌋}.
 *
 * <ul>
 *   <li>{@link Kind#EULER}: {@code x ← x + (σ_next − σ) v};</li>
 *   <li>{@link Kind#UNIPC}: UniPC with the B(h) solver of type {@code bh2} in data prediction mode ({@code x̂₀ = x − σ v}), order 2 with
 *       the lower orders at the start and the last step ({@code lower_order_final}), a predictor (UniP) and, from the second step on,
 *       a corrector (UniC) that re-evaluates the previous step's update with the new model output — the same arithmetic as diffusers,
 *       in double precision. With the flow parametrisation {@code α = 1 − σ}, {@code λ = ln α − ln σ}.</li>
 * </ul>
 */
public final class FlowScheduler {

    public enum Kind {
        UNIPC, EULER;

        public static Kind parse(String s) {
            String n = s == null ? "" : s.toLowerCase(java.util.Locale.ROOT).replace("-", "_");
            return switch (n) {
                case "", "unipc", "uni_pc", "default" -> UNIPC;
                case "euler", "flow_euler" -> EULER;
                default -> throw new IllegalArgumentException("unknown scheduler '" + s + "' for a flow-matching model (unipc, euler)");
            };
        }
    }

    public final Kind kind;
    private final int trainSteps;
    private final double shift;
    private final int order;
    private double[] sigmas = new double[0];
    private int[] timesteps = new int[0];
    // UniPC state
    private float[][] outputs = new float[2][];
    private float[] lastSample;
    private int lowerOrderNums;
    private int thisOrder;

    public FlowScheduler(Map<String, Object> config, Kind kind) {
        this.kind = kind;
        trainSteps = (int) Json.num(config, "num_train_timesteps", 1000);
        shift = Json.dbl(config, "flow_shift", 3.0);
        order = (int) Json.num(config, "solver_order", 2);
        if (order < 1 || order > 2) {
            throw new IllegalArgumentException("only solver_order 1 and 2 are supported");
        }
    }

    public int steps() {
        return timesteps.length;
    }

    public int timestep(int i) {
        return timesteps[i];
    }

    public double sigma(int i) {
        return sigmas[i];
    }

    public void setTimesteps(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("steps must be at least 1");
        }
        sigmas = new double[n + 1];
        timesteps = new int[n];
        for (int i = 0; i < n; i++) {
            double s0 = 1.0 + (1.0 / trainSteps - 1.0) * i / n; // np.linspace(1, 1/N, n + 1)[:-1]
            double s = shift * s0 / (1 + (shift - 1) * s0);
            if (i == 0 && Math.abs(s - 1) < 1e-6) {
                s -= 1e-6;
            }
            sigmas[i] = (float) s; // diffusers keeps the sigmas as float32
            timesteps[i] = (int) (s * trainSteps);
        }
        sigmas[n] = 0;
        outputs = new float[2][];
        lastSample = null;
        lowerOrderNums = 0;
        thisOrder = 0;
    }

    /** One update: {@code v} is the network's output for {@code x} at step {@code i}. */
    public float[] step(float[] v, int i, float[] x) {
        int n = x.length;
        float[] out = new float[n];
        if (kind == Kind.EULER) {
            float dt = (float) (sigmas[i + 1] - sigmas[i]);
            for (int k = 0; k < n; k++) {
                out[k] = x[k] + dt * v[k];
            }
            return out;
        }
        float sg = (float) sigmas[i];
        float[] m = new float[n];
        for (int k = 0; k < n; k++) {
            m[k] = x[k] - sg * v[k];
        }
        float[] cur = x;
        if (i > 0 && lastSample != null) {
            cur = corrector(m, lastSample, x, thisOrder, i);
        }
        outputs[0] = outputs[1];
        outputs[1] = m;
        int to = Math.min(order, sigmas.length - 1 - i); // lower_order_final
        thisOrder = Math.min(to, lowerOrderNums + 1);
        lastSample = cur;
        float[] prev = predictor(cur, thisOrder, i);
        if (lowerOrderNums < order) {
            lowerOrderNums++;
        }
        return prev;
    }

    private static double lambda(double sigma) {
        return Math.log(1 - sigma) - Math.log(sigma);
    }

    /** UniP (order 1 or 2): x at sigma_i to sigma_{i+1}. */
    private float[] predictor(float[] x, int p, int i) {
        double sT = sigmas[i + 1];
        double sS = sigmas[i];
        double aT = 1 - sT;
        double h = lambda(sT) - lambda(sS);
        double hh = -h;
        double hPhi1 = Math.expm1(hh);
        double bH = Math.expm1(hh);
        float[] m0 = outputs[1];
        float[] out = new float[x.length];
        double c1 = sT / sS;
        double c2 = aT * hPhi1;
        if (p == 1) {
            for (int k = 0; k < x.length; k++) {
                out[k] = (float) (c1 * x[k] - c2 * m0[k]);
            }
            return out;
        }
        double rk = (lambda(sigmas[i - 1]) - lambda(sS)) / h;
        float[] mi = outputs[0];
        for (int k = 0; k < x.length; k++) {
            double d1 = (mi[k] - m0[k]) / rk;
            out[k] = (float) (c1 * x[k] - c2 * m0[k] - aT * bH * 0.5 * d1);
        }
        return out;
    }

    /** UniC: corrects the sample the previous predictor made (from {@code last} at sigma_{i-1}) with the new model output at sigma_i. */
    private float[] corrector(float[] modelT, float[] last, float[] thisSample, int p, int i) {
        double sT = sigmas[i];
        double sS = sigmas[i - 1];
        double aT = 1 - sT;
        double h = lambda(sT) - lambda(sS);
        double hh = -h;
        double hPhi1 = Math.expm1(hh);
        double bH = Math.expm1(hh);
        float[] m0 = outputs[1];
        double c1 = sT / sS;
        double c2 = aT * hPhi1;
        float[] out = new float[last.length];
        double rho0;
        double rho1;
        double rk = 1;
        if (p == 1) {
            rho1 = 0.5;
            rho0 = 0;
        } else {
            rk = (lambda(sigmas[i - 2]) - lambda(sS)) / h;
            // R = [[1, 1], [rk, 1]], b_i = h_phi_k · i! / B_h (h_phi_k recursed as diffusers does)
            double hPhiK = hPhi1 / hh - 1;
            double fact = 1;
            double[] b = new double[2];
            for (int j = 1; j <= 2; j++) {
                b[j - 1] = hPhiK * fact / bH;
                fact *= j + 1;
                hPhiK = hPhiK / hh - 1.0 / fact;
            }
            double det = 1 * 1 - 1 * rk;
            rho0 = (b[0] * 1 - 1 * b[1]) / det;
            rho1 = (1 * b[1] - rk * b[0]) / det;
        }
        float[] mi = outputs[0];
        for (int k = 0; k < out.length; k++) {
            double d1t = modelT[k] - m0[k];
            double corr = p == 1 ? 0 : rho0 * (mi[k] - m0[k]) / rk;
            out[k] = (float) (c1 * last[k] - c2 * m0[k] - aT * bH * (corr + rho1 * d1t));
        }
        return out;
    }
}
