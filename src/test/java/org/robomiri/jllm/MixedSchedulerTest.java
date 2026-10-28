package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.BatchGroup;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Mixed passes through the engine (docs/25) on the real model: slots with a paged KV cache under
 * {@code engine.scheduler=mixed}. A long prompt prefills in one slot while two other slots generate; every conversation
 * must give the greedy answer it gives alone, passes must really have carried a prefill chunk together with decode
 * tokens, and a generation paused in the middle (its blocks given back to the pool) must go on to the same answer.
 *
 * <p>usage: MixedSchedulerTest &lt;model dir&gt; [--setting value ...] (e.g. {@code --backend metal}, as
 * scripts/mac-validate.sh runs it)
 */
public final class MixedSchedulerTest {

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        run(dir, java.util.Arrays.copyOfRange(args, 1, args.length));
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("mixed scheduler test passed");
    }

    private static void run(Path dir, String[] extra) throws Exception {
        // float activations: a mixed pass runs the multi-token kernels for every row, which add in another order than the int8
        // single-token ones; with float activations both are the same arithmetic up to summation order
        String[] base = {"--backend", "cpu", "--model.dir", dir.toAbsolutePath().getParent().toString(),
                "--backend.ctx", "2048", "--engine.speculative", "off", "--backend.cpu.int8-activations", "false",
                "--engine.scheduler", "mixed", "--backend.batch", "64"};
        String[] all = java.util.Arrays.copyOf(base, base.length + extra.length);
        System.arraycopy(extra, 0, all, base.length, extra.length);
        Settings st = Settings.load(all);
        SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 32, List.of());
        try (ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString(), 3);
             Engine e2 = ModelLoader.extraEngine(st, l);
             Engine e3 = ModelLoader.extraEngine(st, l)) {
            Engine[] engines = {l.engine(), e2, e3};
            check(engines[0].backend() instanceof BatchGroup.Member m && m.group().mixed(), "the slots form a mixed group");
            check(engines[0].backend().kvSequence() != null, "the slots have a paged cache");
            StringBuilder longText = new StringBuilder("Read this carefully and then summarize it in one sentence: ");
            for (int i = 0; i < 14; i++) {
                longText.append("Item ").append(i).append(" is a small wooden box that holds exactly ").append(i * 3 + 1)
                        .append(" brass keys. ");
            }
            int[][] prompts = {chat(engines[0], longText.toString()), chat(engines[1], "The capital of France is"),
                    chat(engines[2], "Write a haiku about autumn leaves.")};
            check(prompts[0].length > 150, "the long prompt has " + prompts[0].length + " tokens (several chunks)");

            String[] alone = new String[3];
            for (int i = 0; i < 3; i++) {
                alone[i] = engines[i].generate(prompts[i], greedy, null, () -> false).text();
                reset(engines[i]);
            }
            // together: the two short chats start first and are generating when the long prompt arrives
            BatchGroup group = ((BatchGroup.Member) engines[0].backend()).group();
            long[] before = group.mixedStats();
            String[] together = new String[3];
            ExecutorService ex = Executors.newFixedThreadPool(3);
            CountDownLatch shortsRunning = new CountDownLatch(2);
            SamplingParams longer = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 32, List.of());
            List<Future<String>> fs = new ArrayList<>();
            for (int i = 1; i < 3; i++) {
                int k = i;
                fs.add(ex.submit(() -> engines[k].generate(prompts[k], longer, t -> shortsRunning.countDown(), () -> false)
                        .text()));
            }
            shortsRunning.await();
            Future<String> big = ex.submit(() -> engines[0].generate(prompts[0], greedy, null, () -> false).text());
            together[1] = fs.get(0).get();
            together[2] = fs.get(1).get();
            together[0] = big.get();
            long[] after = group.mixedStats();
            for (int i = 0; i < 3; i++) {
                check(alone[i].equals(together[i]), "slot " + i + " gives the same answer alone and in mixed passes: "
                        + together[i].replace('\n', ' ').substring(0, Math.min(40, together[i].length())));
            }
            System.out.println("     mixed passes " + (after[0] - before[0]) + ", rider rows " + (after[1] - before[1])
                    + ", chunks run in parts " + (after[2] - before[2]));

            // the same two chats again, at once: each prompt is cached up to its last token, which is fed as a one-token
            // forward call at its offset in the prompt, batched with the other slot's step (it once ran the prompt's first token)
            CountDownLatch go = new CountDownLatch(1);
            List<Future<String>> again = new ArrayList<>();
            for (int i = 1; i < 3; i++) {
                int k = i;
                again.add(ex.submit(() -> {
                    go.await();
                    return engines[k].generate(prompts[k], greedy, null, () -> false).text();
                }));
            }
            go.countDown();
            for (int i = 1; i < 3; i++) {
                check(alone[i].equals(again.get(i - 1).get()), "slot " + i + ", its prompt cached up to the last token, gives the"
                        + " same answer alone and batched with another slot");
            }
            ex.shutdown();

            // pause a generation in the middle and let it go on
            pauseCheck(engines, prompts, greedy, alone);
        }
        speculationCheck(dir, extra);
    }

    /**
     * Batched speculative verification: with prompt-lookup speculation every slot's draft is verified as a segment of a
     * mixed pass. Three slots repeat a sentence (so the drafts are accepted), generate at once, and give the tokens they give
     * alone; some drafts were verified in a pass shared with another segment.
     */
    private static void speculationCheck(Path dir, String[] extra) throws Exception {
        String[] base = {"--backend", "cpu", "--model.dir", dir.toAbsolutePath().getParent().toString(),
                "--backend.ctx", "1024", "--engine.speculative", "ngram", "--engine.draft-tokens", "6",
                "--backend.cpu.int8-activations", "false", "--engine.scheduler", "mixed", "--backend.batch", "64"};
        String[] all = java.util.Arrays.copyOf(base, base.length + extra.length);
        System.arraycopy(extra, 0, all, base.length, extra.length);
        Settings st = Settings.load(all);
        SamplingParams p = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 60, List.of());
        try (ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString(), 3);
             Engine e2 = ModelLoader.extraEngine(st, l);
             Engine e3 = ModelLoader.extraEngine(st, l)) {
            Engine[] engines = {l.engine(), e2, e3};
            String[] sentences = {"The quick brown fox jumps over the lazy dog.", "Pack my box with five dozen liquor jugs.",
                    "How vexingly quick daft zebras jump."};
            int[][] prompts = new int[3][];
            for (int i = 0; i < 3; i++) {
                prompts[i] = chat(engines[i], "Repeat this sentence exactly ten times, one per line: " + sentences[i]);
            }
            String[] alone = new String[3];
            for (int i = 0; i < 3; i++) {
                alone[i] = engines[i].generate(prompts[i], p, null, () -> false).text();
                reset(engines[i]);
            }
            BatchGroup group = ((BatchGroup.Member) engines[0].backend()).group();
            long[] before = group.mixedStats();
            ExecutorService ex = Executors.newFixedThreadPool(3);
            List<Future<Engine.Result>> fs = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                int k = i;
                fs.add(ex.submit(() -> engines[k].generate(prompts[k], p, null, () -> false)));
            }
            int drafted = 0;
            for (int i = 0; i < 3; i++) {
                Engine.Result r = fs.get(i).get();
                drafted += r.stats().draftedTokens();
                check(alone[i].equals(r.text()), "speculation: slot " + i + " gives the same answer alone and in mixed passes");
            }
            ex.shutdown();
            long[] after = group.mixedStats();
            System.out.println("     drafted " + drafted + " tokens; drafts verified in shared passes: " + (after[3] - before[3]));
            check(drafted > 0, "prompt-lookup speculation drafted tokens");
            check(after[3] > before[3], "drafts were verified in a pass shared with other slots' segments");
        }
    }

    private static void pauseCheck(Engine[] engines, int[][] prompts, SamplingParams greedy, String[] alone) throws Exception {
        BatchGroup.Member member = (BatchGroup.Member) engines[1].backend();
        SamplingParams p = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 32, List.of());
        ExecutorService ex = Executors.newSingleThreadExecutor();
        CountDownLatch started = new CountDownLatch(3);
        Future<String> f = ex.submit(() -> engines[1].generate(prompts[1], p, t -> started.countDown(), () -> false).text());
        started.await();
        boolean asked = member.requestPause();
        long t0 = System.nanoTime();
        while (asked && !member.paused() && !f.isDone() && System.nanoTime() - t0 < 5_000_000_000L) {
            Thread.sleep(2);
        }
        boolean wasPaused = member.paused();
        var seq = member.kvSequence();
        int heldWhilePaused = seq.blocks();
        Thread.sleep(50);
        member.resume();
        String out = f.get();
        ex.shutdown();
        check(asked, "a slot with a paged cache can be asked to pause");
        check(wasPaused || f.isDone(), "the generation paused between two steps (or had finished)");
        check(!wasPaused || heldWhilePaused <= 1, "while paused it holds at most its partly filled block (" + heldWhilePaused + ")");
        check(out.equals(alone[1]), "the paused and resumed generation gives the answer it gives alone");
    }

    /** Forgets the conversation of this slot and the blocks the pool kept for it, so the next run prefills for real. */
    private static void reset(Engine e) {
        SamplingParams one = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 1, List.of());
        e.generate(chat(e, "zzz unrelated"), one, null, () -> false);
        e.backend().kvSequence().pool().evictAllCached();
    }

    private static int[] chat(Engine e, String user) {
        return e.renderChat(List.of(new ChatTemplate.Message("user", user + " /no_think")), false);
    }
}
