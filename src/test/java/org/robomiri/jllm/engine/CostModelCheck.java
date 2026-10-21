package org.robomiri.jllm.engine;

import java.util.ArrayList;
import java.util.List;

/**
 * {@link Speculator.CostModel} on synthetic measurements: the draft length it picks is the brute-force best of
 * {@code E(k) / (k d + a + b k)}, it pauses speculation when no length beats a plain step, and it fits the pass cost
 * {@code V(n) = a + b (n - 1)} from passes of several lengths.
 */
public final class CostModelCheck {

    private CostModelCheck() {
    }

    /** @return descriptions of the failed checks (empty when all pass) */
    public static List<String> run() {
        List<String> bad = new ArrayList<>();
        double ms = 1e6;
        double a = 39.2 * ms;
        double d = 8 * ms;
        // the 30B MoE of docs/11 §11.6 (b = 0.55 a) and a cheap multi-token pass (b = 0.1 a)
        for (double bShare : new double[]{0.55, 0.1}) {
            double b = bShare * a;
            for (double alpha : new double[]{0.6, 0.9, 0.98}) {
                Speculator.CostModel m = new Speculator.CostModel();
                for (int pass = 0; pass < 200; pass++) {
                    int n = 2 + pass % 4; // passes of 2 to 5 positions
                    m.verify(n, (long) (a + b * (n - 1)));
                    m.draft((long) (3 * d), 3);
                    m.plain((long) a);
                    // acceptance at rate alpha: of 100 examined positions, 100 alpha accepted
                    m.acceptance((int) Math.round(100 * alpha), 100);
                }
                double[] fit = m.fit();
                if (Math.abs(fit[0] - a) > 0.01 * a || Math.abs(fit[1] - b) > 0.01 * a) {
                    bad.add(String.format("fit a=%.1f b=%.1f ms, want %.1f, %.1f", fit[0] / ms, fit[1] / ms, a / ms, b / ms));
                }
                int want = 0;
                double best = 1 / a;
                for (int k = 1; k <= 16; k++) {
                    double e = 0;
                    for (int i = 0; i <= k; i++) {
                        e += Math.pow(alpha, i);
                    }
                    double rate = e / (k * d + a + b * k);
                    if (rate > best) {
                        best = rate;
                        want = k;
                    }
                }
                int got = m.bestK(16);
                if (got != want) {
                    bad.add("b = " + bShare + " a, alpha " + alpha + ": draft length " + got + ", want " + want);
                }
            }
        }
        if (new Speculator.CostModel().bestK(8) != -1) {
            bad.add("a cost model without measurements must leave the choice to the doubling rule (-1)");
        }
        return bad;
    }
}
