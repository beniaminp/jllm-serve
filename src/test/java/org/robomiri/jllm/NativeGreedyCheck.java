package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;

import java.nio.file.Path;
import java.util.List;

/**
 * End to end, int8 activations on: the native int8 kernels ({@code backend.cpu.native=true}) against the Java ones
 * ({@code false}) on a real model, both loaded at once (docs/07 §7.16).
 *
 * <ol>
 *   <li>Teacher-forced decode: the same tokens go through both backends one at a time (the single-token path, where
 *       the int8 kernels run); per step the logits are compared (largest difference relative to the logits' range)
 *       and so is the arg-max.</li>
 *   <li>Free-running greedy generation: the token sequences of both, and where they first differ.</li>
 * </ol>
 *
 * The two compute the same integer block sums and the same scales, and add the scaled block results in different
 * orders, so the logits differ by float rounding only; a greedy run can still take another path after a near-tie
 * (two top logits closer than that rounding). The check fails if the teacher-forced arg-max differs where the top two
 * logits of the Java run are more than 1e-3 of the logit range apart, or if a logit differs by more than 1e-3 of the
 * range.
 *
 * <pre>java ... org.robomiri.jllm.NativeGreedyCheck models/Qwen_Qwen3-0.6B [tokens]</pre>
 */
public final class NativeGreedyCheck {

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]).toAbsolutePath();
        int steps = args.length > 1 ? Integer.parseInt(args[1]) : 64;
        String[] prompts = {"The capital of France is", "Write a haiku about autumn leaves.", "def fibonacci(n):",
                "Explain why the sky is blue in two sentences."};
        try (ModelLoader.Loaded nat = load(dir, true); ModelLoader.Loaded java = load(dir, false)) {
            Backend bn = nat.engine().backend();
            Backend bj = java.engine().backend();
            System.out.println("native: " + bn.name());
            System.out.println("java:   " + bj.name());
            int vocab = nat.weights().lmHead.rows;
            boolean ok = true;
            int sameSeq = 0;
            // 1. teacher forced, every prompt from position 0 (before the engines keep any cached prefix)
            for (String p : prompts) {
                int[] prompt = chat(nat.engine(), p);
                float[] ln = new float[vocab];
                float[] lj = new float[vocab];
                bn.forward(prompt, 0, prompt.length, 0, ln);
                bj.forward(prompt, 0, prompt.length, 0, lj);
                int agree = 0;
                double worst = 0;
                int pos = prompt.length;
                for (int s = 0; s < steps; s++) {
                    int an = argmax(ln);
                    int aj = argmax(lj);
                    float range = max(lj) - min(lj);
                    double diff = 0;
                    for (int i = 0; i < vocab; i++) {
                        diff = Math.max(diff, Math.abs(ln[i] - lj[i]));
                    }
                    worst = Math.max(worst, diff / range);
                    if (an == aj) {
                        agree++;
                    } else {
                        float margin = lj[aj] - lj[an];
                        System.out.printf("  step %d: arg-max %d (native) vs %d (java), java margin %.4g (%.2g of range)%n",
                                s, an, aj, margin, margin / range);
                        if (margin > 1e-3 * range) {
                            ok = false;
                        }
                    }
                    if (diff > 1e-3 * range) {
                        ok = false;
                    }
                    int next = aj;
                    bn.forward(new int[]{next}, 0, 1, pos, ln);
                    bj.forward(new int[]{next}, 0, 1, pos, lj);
                    pos++;
                }
                System.out.printf("teacher-forced %-48s arg-max agrees %d/%d, max |dlogit| %.2g of the logit range%n",
                        "\"" + p + "\"", agree, steps, worst);
            }
            // 2. free running greedy
            SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, steps, List.of());
            for (String p : prompts) {
                int[] prompt = chat(nat.engine(), p);
                int[] tn = nat.engine().generate(prompt, greedy, null, () -> false).tokens();
                int[] tj = java.engine().generate(prompt, greedy, null, () -> false).tokens();
                int first = firstDifference(tn, tj);
                if (first < 0) {
                    sameSeq++;
                }
                System.out.printf("greedy %-48s %s%n", "\"" + p + "\"",
                        first < 0 ? "identical (" + tn.length + " tokens)" : "differs from token " + first + " of "
                                + Math.min(tn.length, tj.length));
            }
            System.out.println("greedy sequences identical: " + sameSeq + "/" + prompts.length);
            if (!ok) {
                System.out.println("native greedy check FAILED");
                System.exit(1);
            }
            System.out.println("native greedy check passed");
        }
    }

    private static ModelLoader.Loaded load(Path dir, boolean nativeKernels) throws Exception {
        Settings st = Settings.load(new String[]{"--backend", "cpu", "--model.dir", dir.getParent().toString(),
                "--backend.ctx", "1024", "--engine.speculative", "off", "--engine.batch-decode", "false",
                "--backend.cpu.int8-activations", "true", "--backend.cpu.native", String.valueOf(nativeKernels)});
        return ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString());
    }

    private static int[] chat(Engine e, String user) {
        return e.renderChat(List.of(new ChatTemplate.Message("user", user + " /no_think")), false);
    }

    private static int firstDifference(int[] a, int[] b) {
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            if (a[i] != b[i]) {
                return i;
            }
        }
        return a.length == b.length ? -1 : Math.min(a.length, b.length);
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    private static float max(float[] v) {
        float m = v[0];
        for (float f : v) {
            m = Math.max(m, f);
        }
        return m;
    }

    private static float min(float[] v) {
        float m = v[0];
        for (float f : v) {
            m = Math.min(m, f);
        }
        return m;
    }
}
