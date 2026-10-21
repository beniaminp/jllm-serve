package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.BatchGroup;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.SamplingParams;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Batched decoding across slots ({@link BatchGroup}) on the real model, CPU backend: three slots generating at the same
 * time give the same greedy answers as when each runs alone, their steps really are computed together, and the total
 * throughput is printed for one slot alone and three at once.
 *
 * <p>usage: BatchDecodeTest &lt;model dir&gt; (models/Qwen_Qwen3-0.6B)
 */
public final class BatchDecodeTest {

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        // float activations: a batched step runs the multi-token kernels, which add in another order than the int8
        // single-token ones; with float activations both are the same arithmetic up to summation order
        Settings st = Settings.load(new String[]{"--backend", "cpu", "--model.dir", dir.toAbsolutePath().getParent().toString(),
                "--backend.ctx", "512", "--engine.speculative", "off", "--backend.cpu.int8-activations", "false"});
        String[] prompts = {"The capital of France is", "Write a haiku about autumn leaves.", "def fibonacci(n):"};
        SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 24, List.of());
        try (ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString());
             Engine e2 = ModelLoader.extraEngine(st, l);
             Engine e3 = ModelLoader.extraEngine(st, l)) {
            Engine[] engines = {l.engine(), e2, e3};
            check(engines[0].backend() instanceof BatchGroup.Member, "the slots' backends are members of a batch group");
            BatchGroup group = null;

            // alone: one after another
            String[] alone = new String[3];
            for (int i = 0; i < 3; i++) {
                alone[i] = engines[i].generate(chat(engines[i], prompts[i]), greedy, null, () -> false).text();
            }
            // together: three threads at once
            String[] together = new String[3];
            long[] before = statsOf(engines[0]);
            ExecutorService ex = Executors.newFixedThreadPool(3);
            List<Future<String>> fs = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                int k = i;
                fs.add(ex.submit(() -> engines[k].generate(chat(engines[k], prompts[k]), greedy, null, () -> false).text()));
            }
            for (int i = 0; i < 3; i++) {
                together[i] = fs.get(i).get();
            }
            long[] after = statsOf(engines[0]);
            for (int i = 0; i < 3; i++) {
                check(alone[i].equals(together[i]), "slot " + i + " gives the same answer alone and batched: "
                        + abbreviate(together[i]));
            }
            long batched = after[1] - before[1];
            check(batched > 0, "decode steps were computed together (" + batched + " steps in shared passes, "
                    + (after[0] - before[0]) + " passes)");

            // throughput: one slot alone, then three at once (int8 activations as by default would be faster for
            // both; this compares like with like)
            SamplingParams longer = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 64, List.of());
            long t0 = System.nanoTime();
            Engine.Result r = engines[0].generate(chat(engines[0], "Tell me a long story about a robot."), longer, null,
                    () -> false);
            double one = r.tokens().length / ((System.nanoTime() - t0) / 1e9);
            t0 = System.nanoTime();
            fs.clear();
            for (int i = 0; i < 3; i++) {
                int k = i;
                fs.add(ex.submit(() -> engines[k].generate(chat(engines[k], "Tell me a long story about robot number "
                        + k + "."), longer, null, () -> false).text()));
            }
            int total = 0;
            for (Future<String> f : fs) {
                total += engines[0].tokenizer().encode(f.get()).length;
            }
            double three = total / ((System.nanoTime() - t0) / 1e9);
            ex.shutdown();
            System.out.printf("throughput (prompt included): one slot %.1f tok/s, three slots together %.1f tok/s (%.1fx)%n",
                    one, three, three / one);
        }
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("batch decode test passed");
    }

    private static int[] chat(Engine e, String user) {
        return e.renderChat(List.of(new org.robomiri.jllm.tokenizer.ChatTemplate.Message("user", user + " /no_think")),
                false);
    }

    private static long[] statsOf(Engine e) {
        Backend b = e.backend();
        return b instanceof BatchGroup.Member m ? m.group().stats() : new long[2];
    }

    private static String abbreviate(String s) {
        String one = s.replace('\n', ' ');
        return Arrays.toString(new String[]{one.length() > 50 ? one.substring(0, 50) + "…" : one});
    }
}
