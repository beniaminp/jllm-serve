package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.Priority;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * What the request priority buys (docs/10 §10.14): a background job (a long prompt, then a long answer) is running on
 * one slot when a short interactive request arrives on another slot of the same model. The interactive request's time to
 * its first token and its whole time are the median of the rounds, with the priority honoured
 * ({@code --engine.priority-order true}, the default) or not ({@code false}), and with the job marked background or left
 * interactive.
 *
 * <pre>java ... org.robomiri.jllm.PriorityBench [--rounds 5] [--long-tokens 1500] [settings...]  (--model Qwen/Qwen3-0.6B)</pre>
 */
public final class PriorityBench {

    public static void main(String[] args) throws Exception {
        int rounds = 5;
        int longTokens = 1500;
        List<String> rest = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--rounds" -> rounds = Integer.parseInt(args[++i]);
                case "--long-tokens" -> longTokens = Integer.parseInt(args[++i]);
                default -> rest.add(args[i]);
            }
        }
        if (!rest.contains("--ctx")) {
            rest.add("--ctx");
            rest.add("4096");
        }
        rest.add("--temp");
        rest.add("0");
        rest.add("--engine.speculative");
        rest.add("off");
        Settings st = Settings.load(rest.toArray(new String[0]));
        try (ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), st.get("model.repo"), 2);
             Engine slotB = ModelLoader.extraEngine(st, l)) {
            Engine slotA = l.engine();
            System.out.printf("PRIO model=%s backend=%s priority-order=%s long prompt ~%d tokens rounds=%d%n", st.get("model.repo"),
                    slotA.backend().name(), st.get("engine.priority-order"), longTokens, rounds);
            Random rnd = new Random(1);
            SamplingParams jobParams = st.sampling().withMaxTokens(96);
            SamplingParams quickParams = st.sampling().withMaxTokens(24);
            warm(slotA, slotB, st);
            for (Priority job : new Priority[]{Priority.BACKGROUND, Priority.INTERACTIVE}) {
                double[] first = new double[rounds];
                double[] total = new double[rounds];
                double[] jobTotal = new double[rounds];
                for (int r = 0; r < rounds; r++) {
                    int[] longPrompt = slotA.renderChat(List.of(new ChatTemplate.Message("user", filler(rnd, longTokens)
                            + "\n\nSummarize the text above in one paragraph.")), false);
                    int[] quick = slotB.renderChat(List.of(new ChatTemplate.Message("user", "Say hello in five words, run "
                            + rnd.nextInt(1_000_000) + ".")), false);
                    long[] t = new long[4];
                    Thread a = new Thread(() -> {
                        try (Priority.Scope s = Priority.enter(job)) {
                            long t0 = System.nanoTime();
                            slotA.generate(longPrompt, jobParams, null, () -> false);
                            t[3] = System.nanoTime() - t0;
                        }
                    });
                    long[] firstAt = new long[1];
                    Thread b = new Thread(() -> {
                        try (Priority.Scope s = Priority.enter(Priority.INTERACTIVE)) {
                            long t0 = System.nanoTime();
                            slotB.generate(quick, quickParams, piece -> {
                                if (firstAt[0] == 0) {
                                    firstAt[0] = System.nanoTime() - t0;
                                }
                            }, () -> false);
                            t[1] = System.nanoTime() - t0;
                        }
                    });
                    a.start();
                    Thread.sleep(150); // the job is in its prefill
                    b.start();
                    a.join();
                    b.join();
                    first[r] = firstAt[0] / 1e6;
                    total[r] = t[1] / 1e6;
                    jobTotal[r] = t[3] / 1e6;
                }
                System.out.printf("PRIO job=%s interactive request: first token %.0f ms (min %.0f, max %.0f), whole request %.0f ms; "
                                + "the job took %.0f ms%n", job.name().toLowerCase(), median(first), min(first), max(first), median(total),
                        median(jobTotal));
            }
        }
    }

    private static void warm(Engine a, Engine b, Settings st) {
        for (Engine e : new Engine[]{a, b}) {
            int[] p = e.renderChat(List.of(new ChatTemplate.Message("user", "Warm up " + System.nanoTime())), false);
            e.generate(p, st.sampling().withMaxTokens(8), null, () -> false);
        }
    }

    /** About {@code tokens} tokens of text no earlier prompt shares (a prefix cache hit would skip the prefill). */
    private static String filler(Random rnd, int tokens) {
        String[] words = {"engine", "kernel", "buffer", "token", "cache", "matrix", "shader", "thread", "layer", "weight", "vector", "memory"};
        StringBuilder sb = new StringBuilder("Notes " + rnd.nextInt(1_000_000) + ": ");
        for (int i = 0; i < tokens; i++) {
            sb.append(words[rnd.nextInt(words.length)]).append(' ');
        }
        return sb.toString();
    }

    private static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s[s.length / 2];
    }

    private static double min(double[] v) {
        return Arrays.stream(v).min().orElse(0);
    }

    private static double max(double[] v) {
        return Arrays.stream(v).max().orElse(0);
    }
}
