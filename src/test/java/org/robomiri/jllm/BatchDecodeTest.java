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
 * throughput is printed for one slot alone and three at once. With int8 activations and the native kernels a batched step
 * gives each slot its lone step's logits bit for bit ({@code backend.cpu.batch-int8-rows}).
 *
 * <p>usage: BatchDecodeTest &lt;model dir&gt; [bench] [--setting value ...] (models/Qwen_Qwen3-0.6B; {@code bench} adds
 * the throughput run; settings override the test's, e.g. {@code --backend metal} on a Mac, as scripts/mac-validate.sh
 * runs it). {@code BatchDecodeTest <model dir> passes}: the time of one CPU decode pass of 1 to 4 rows, int8 activations,
 * with {@code backend.cpu.batch-int8-rows} 4 (every pass of up to 4 rows token by token through the int8 kernel) and 0
 * (the float multi-row kernels), docs/07 §7.4.1.
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
        if (args.length > 1 && args[1].equals("passes")) {
            passes(dir);
            return;
        }
        boolean bench = args.length > 1 && args[1].equals("bench");
        String[] extra = java.util.Arrays.copyOfRange(args, bench ? 2 : 1, args.length);
        run(dir, bench, extra);
        if (extra.length == 0) {
            int8Rows(dir);
        }
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("batch decode test passed");
    }

    private static void run(Path dir, boolean bench, String[] extra) throws Exception {
        // float activations: a batched step runs the multi-token kernels, which add in another order than the int8
        // single-token ones; with float activations both are the same arithmetic up to summation order
        String[] base = {"--backend", "cpu", "--model.dir", dir.toAbsolutePath().getParent().toString(),
                "--backend.ctx", "512", "--engine.speculative", "off", "--backend.cpu.int8-activations", "false"};
        String[] all = java.util.Arrays.copyOf(base, base.length + extra.length);
        System.arraycopy(extra, 0, all, base.length, extra.length);
        Settings st = Settings.load(all);
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

            if (!bench) {
                ex.shutdown();
                return;
            }
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
    }

    /**
     * With int8 activations (the CPU default) and the native kernels, a batched step of two slots
     * ({@code backend.cpu.batch-int8-rows}, docs/07 §7.4.1) gives each slot the logits of its lone step bit for bit; with
     * the switch at 0 (the float multi-row kernels) they differ by the int8 rounding.
     */
    private static void int8Rows(Path dir) throws Exception {
        for (String rows : new String[]{"3", "0"}) {
            Settings st = Settings.load(new String[]{"--backend", "cpu", "--model.dir", dir.toAbsolutePath().getParent().toString(),
                    "--backend.ctx", "512", "--engine.speculative", "off", "--backend.cpu.int8-activations", "true",
                    "--backend.cpu.batch-int8-rows", rows});
            SamplingParams one = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 1, List.of());
            try (ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString());
                 Engine e2 = ModelLoader.extraEngine(st, l)) {
                if (org.robomiri.jllm.cpu.NativeCpu.get() == null) {
                    System.out.println("skip int8 batched rows: no native CPU kernels (scripts/build-native-cpu.sh)");
                    return;
                }
                Engine[] engines = {l.engine(), e2};
                Backend[] inner = new Backend[2];
                for (int i = 0; i < 2; i++) {
                    engines[i].generate(chat(engines[i], "Tell me about robot " + i + "."), one, null, () -> false);
                    inner[i] = ((BatchGroup.Member) engines[i].backend()).inner();
                }
                int vocab = engines[0].tokenizer().vocabSize();
                int pos = 20;
                float[][] lone = {new float[vocab], new float[vocab]};
                inner[0].forward(new int[]{100}, 0, 1, pos, lone[0]);
                inner[1].forward(new int[]{200}, 0, 1, pos + 1, lone[1]);
                float[][] batched = {new float[vocab], new float[vocab]};
                boolean ran = inner[0].forwardBatch(inner, new int[]{100, 200}, new int[]{pos, pos + 1}, batched);
                int differ = 0;
                for (int r = 0; r < 2; r++) {
                    for (int i = 0; i < vocab; i++) {
                        differ += Float.floatToRawIntBits(lone[r][i]) != Float.floatToRawIntBits(batched[r][i]) ? 1 : 0;
                    }
                }
                if (rows.equals("3")) {
                    check(ran && differ == 0, "int8 activations, batch-int8-rows 3: both rows of a batched step have their"
                            + " lone step's logits bit for bit (" + differ + " differ)");
                } else {
                    check(ran && differ > 0, "int8 activations, batch-int8-rows 0: the float multi-row kernels differ by the"
                            + " int8 rounding (" + differ + " of " + 2 * vocab + " logits differ)");
                }
            }
        }
    }

    /** The pass-time table of docs/07 §7.4.1: lone step and forwardBatch of 1..4 rows, median of 5 rounds of 20 passes. */
    private static void passes(Path dir) throws Exception {
        System.out.println("| backend.cpu.batch-int8-rows | lone step | 1 row | 2 rows | 3 rows | 4 rows |");
        System.out.println("|---|---|---|---|---|---|");
        for (String rows : new String[]{"4", "0"}) {
            Settings st = Settings.load(new String[]{"--backend", "cpu", "--model.dir", dir.toAbsolutePath().getParent().toString(),
                    "--backend.ctx", "1024", "--engine.speculative", "off", "--backend.cpu.int8-activations", "true",
                    "--backend.cpu.batch-int8-rows", rows});
            SamplingParams one = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 1, List.of());
            try (ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString(), 4);
                 Engine e2 = ModelLoader.extraEngine(st, l);
                 Engine e3 = ModelLoader.extraEngine(st, l);
                 Engine e4 = ModelLoader.extraEngine(st, l)) {
                Engine[] engines = {l.engine(), e2, e3, e4};
                Backend[] inner = new Backend[4];
                for (int i = 0; i < 4; i++) {
                    engines[i].generate(chat(engines[i], "Tell me a story about robot " + i + " and its friends."), one, null,
                            () -> false);
                    Backend b = engines[i].backend();
                    inner[i] = b instanceof BatchGroup.Member m ? m.inner() : b;
                }
                int vocab = engines[0].tokenizer().vocabSize();
                float[][] logits = new float[4][vocab];
                int pos = 30;
                double[][] ms = new double[5][5];
                for (int round = 0; round < 5; round++) {
                    long t0 = System.nanoTime();
                    for (int r = 0; r < 20; r++) {
                        inner[0].forward(new int[]{100}, 0, 1, pos, logits[0]);
                    }
                    ms[0][round] = (System.nanoTime() - t0) / 1e6 / 20;
                    for (int n = 1; n <= 4; n++) {
                        Backend[] bs = Arrays.copyOf(inner, n);
                        int[] toks = new int[n];
                        int[] ps = new int[n];
                        Arrays.fill(toks, 100);
                        Arrays.fill(ps, pos);
                        float[][] out = Arrays.copyOf(logits, n);
                        t0 = System.nanoTime();
                        for (int r = 0; r < 20; r++) {
                            inner[0].forwardBatch(bs, toks, ps, out);
                        }
                        ms[n][round] = (System.nanoTime() - t0) / 1e6 / 20;
                    }
                }
                StringBuilder row = new StringBuilder("| " + rows + " |");
                for (double[] m : ms) {
                    Arrays.sort(m);
                    row.append(String.format(java.util.Locale.ROOT, " %.1f ms |", m[2]));
                }
                System.out.println(row);
            }
        }
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
