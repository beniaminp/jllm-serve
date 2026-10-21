package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;

import java.nio.file.Path;
import java.util.List;

/**
 * End to end on a real model (docs/07 §7.16): int8 activations with the native kernels ({@code backend.cpu.native=true})
 * against int8 with the Java kernels ({@code false}), with float activations as the yardstick. All three are loaded at
 * once.
 *
 * <ol>
 *   <li>Teacher-forced decode: the same tokens go through all three one at a time (the single-token path, where the
 *       int8 kernels run). Per step, the largest logit difference relative to the logits' range and the arg-max are
 *       compared: native vs Java int8, and Java int8 vs float.</li>
 *   <li>Free-running greedy generation, native vs Java int8: where the token sequences first differ.</li>
 * </ol>
 *
 * Native and Java int8 compute the same integer block sums and the same scales and add the scaled results in a
 * different order, so one matmul differs by float rounding. That is not the end of it: the next layer quantizes its
 * input to int8 again, and a last-bit difference that moves a value across a rounding boundary changes that int8 value
 * by one step (up to 1/254 of the block's largest value). Over 28 layers and the growing KV cache the two runs drift
 * apart much further than float rounding: on Qwen3-0.6B, about as far as int8 drifts from float (the logits of one
 * decode step already differ by ~1 % of their range, while each matmul alone agrees to ~1e-7 of {@code Σ|w_i x_i|},
 * {@code NativeRealCheck}). The check therefore requires the native-vs-Java difference to stay within twice the
 * Java-int8-vs-float difference (the noise int8 activations add anyway), and every arg-max disagreement to be a
 * near-tie: the Java run's top two logits closer than 1 % of the logit range.
 *
 * <pre>java ... org.robomiri.jllm.NativeGreedyCheck models/Qwen_Qwen3-0.6B [tokens]</pre>
 */
public final class NativeGreedyCheck {

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]).toAbsolutePath();
        int steps = args.length > 1 ? Integer.parseInt(args[1]) : 64;
        String[] prompts = {"The capital of France is", "Write a haiku about autumn leaves.", "def fibonacci(n):",
                "Explain why the sky is blue in two sentences."};
        try (ModelLoader.Loaded nat = load(dir, "true", true); ModelLoader.Loaded java = load(dir, "true", false);
             ModelLoader.Loaded flt = load(dir, "false", false)) {
            Backend bn = nat.engine().backend();
            Backend bj = java.engine().backend();
            Backend bf = flt.engine().backend();
            System.out.println("native: " + bn.name());
            System.out.println("java:   " + bj.name());
            System.out.println("float:  " + bf.name());
            int vocab = nat.weights().lmHead.rows;
            double worstNJ = 0;
            double worstJF = 0;
            int disagreeNJ = 0;
            int disagreeJF = 0;
            boolean ties = true;
            // 1. teacher forced, every prompt from position 0 (before the engines keep any cached prefix)
            for (String p : prompts) {
                int[] prompt = chat(nat.engine(), p);
                float[] ln = new float[vocab];
                float[] lj = new float[vocab];
                float[] lf = new float[vocab];
                bn.forward(prompt, 0, prompt.length, 0, ln);
                bj.forward(prompt, 0, prompt.length, 0, lj);
                bf.forward(prompt, 0, prompt.length, 0, lf);
                int dnj = 0;
                int djf = 0;
                double nj = 0;
                double jf = 0;
                double first = 0;
                int pos = prompt.length;
                for (int s = 0; s < steps; s++) {
                    float range = max(lj) - min(lj);
                    double a = maxDiff(ln, lj) / range;
                    double b = maxDiff(lj, lf) / range;
                    if (s == 1) {
                        first = a; // the first decode step: one int8 pass over a KV cache both filled identically
                    }
                    nj = Math.max(nj, a);
                    jf = Math.max(jf, b);
                    if (argmax(ln) != argmax(lj)) {
                        dnj++;
                        float margin = lj[argmax(lj)] - lj[argmax(ln)];
                        System.out.printf("  step %d: arg-max differs, margin of the java run %.4f of the range%n", s,
                                margin / range);
                        ties &= margin < 0.01 * range;
                    }
                    if (argmax(lj) != argmax(lf)) {
                        djf++;
                    }
                    int next = argmax(lj);
                    bn.forward(new int[]{next}, 0, 1, pos, ln);
                    bj.forward(new int[]{next}, 0, 1, pos, lj);
                    bf.forward(new int[]{next}, 0, 1, pos, lf);
                    pos++;
                }
                System.out.printf("teacher-forced %-48s max |dlogit|/range: native-java %.4f (first decode step %.5f),"
                        + " java-float %.4f; arg-max differs: native-java %d, java-float %d of %d%n",
                        "\"" + p + "\"", nj, first, jf, dnj, djf, steps);
                worstNJ = Math.max(worstNJ, nj);
                worstJF = Math.max(worstJF, jf);
                disagreeNJ += dnj;
                disagreeJF += djf;
            }
            // 2. free running greedy
            SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, steps, List.of());
            int sameSeq = 0;
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
            System.out.printf("greedy sequences identical: %d/%d; teacher-forced worst |dlogit|/range native-java %.4f vs"
                            + " java-float %.4f, arg-max disagreements %d vs %d%n", sameSeq, prompts.length, worstNJ,
                    worstJF, disagreeNJ, disagreeJF);
            if (worstNJ > 2 * worstJF || !ties) {
                System.out.println("native greedy check FAILED: native differs from java int8 more than twice as much as"
                        + " int8 from float, or an arg-max differs at more than a near-tie");
                System.exit(1);
            }
            System.out.println("native greedy check passed");
        }
    }

    private static ModelLoader.Loaded load(Path dir, String int8, boolean nativeKernels) throws Exception {
        Settings st = Settings.load(new String[]{"--backend", "cpu", "--model.dir", dir.getParent().toString(),
                "--backend.ctx", "1024", "--engine.speculative", "off", "--engine.batch-decode", "false",
                "--backend.cpu.int8-activations", int8, "--backend.cpu.native", String.valueOf(nativeKernels)});
        return ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString());
    }

    private static int[] chat(Engine e, String user) {
        return e.renderChat(List.of(new ChatTemplate.Message("user", user + " /no_think")), false);
    }

    private static double maxDiff(float[] a, float[] b) {
        double d = 0;
        for (int i = 0; i < a.length; i++) {
            d = Math.max(d, Math.abs(a[i] - b[i]));
        }
        return d;
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
