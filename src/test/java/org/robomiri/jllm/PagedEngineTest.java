package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.kv.FileSpill;
import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.SamplingParams;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * The paged KV cache against the flat one on the real model (CPU backend), through the engine: greedy output and logits
 * equal for f32 / f16 / q8 caches; two slots with a shared prefix store it once and the second prefills only its tail; a
 * fork in the middle of a block and a retry reuse what they share and give the flat answer; an evicted prefix comes
 * back from the disk block store; speculation rollbacks, batched rows and saved caches (flat to paged and back) give the
 * flat answers. Run with {@code -Djllm.cpu.deterministic=true} (a token's rows then do not depend on how its prompt was
 * chunked, which prefix sharing changes).
 *
 * <p>usage: PagedEngineTest &lt;model dir&gt; (models/Qwen_Qwen3-0.6B)
 */
public final class PagedEngineTest {

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    private static final SamplingParams GREEDY = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 16, List.of());

    private static String[] base(Path dir, String... extra) {
        List<String> a = new ArrayList<>(List.of("--backend", "cpu", "--model.dir", dir.toAbsolutePath().getParent().toString(),
                "--backend.ctx", "1024", "--engine.speculative", "off", "--backend.cpu.int8-activations", "false",
                "--backend.threads", "4"));
        a.addAll(List.of(extra));
        return a.toArray(new String[0]);
    }

    private static ModelLoader.Loaded load(Path dir, int slots, String... extra) throws Exception {
        Settings st = Settings.load(base(dir, extra));
        return ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString(), slots);
    }

    private static Engine extra(Path dir, ModelLoader.Loaded l, String... extra) throws Exception {
        return ModelLoader.extraEngine(Settings.load(base(dir, extra)), l);
    }

    /** {@code n} token ids of ordinary text, starting at word {@code from}. */
    private static int[] text(Engine e, int from, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = from; sb.length() < n * 8; i++) {
            sb.append("item ").append(i).append(" of the list, ");
        }
        int[] ids = e.tokenizer().encode(sb.toString());
        return Arrays.copyOf(ids, n);
    }

    private static int[] concat(int[] a, int[] b) {
        int[] r = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        // A kernel's first calls (interpreted, or compiled at a lower tier) can round differently from its later ones: the flat
        // cache differs from itself there. The first round only warms the JVM up; its logits comparison is not counted.
        int warmFailures = failures;
        sameAsFlat(dir, "f32");
        failures = warmFailures;
        for (String dt : new String[]{"f32", "f16", "q8"}) {
            sameAsFlat(dir, dt);
        }
        sharing(dir);
        evictionAndSpill(dir);
        speculation(dir);
        batched(dir);
        savedCaches(dir);
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("paged engine test passed");
    }

    /** The same calls on a flat and on a paged backend: identical logits, and the engine's greedy answers match. */
    static void sameAsFlat(Path dir, String dt) throws Exception {
        try (ModelLoader.Loaded flat = load(dir, 1, "--backend.kv-dtype", dt);
             ModelLoader.Loaded paged = load(dir, 1, "--backend.kv-dtype", dt, "--backend.kv-paged", "true")) {
            Backend bf = flat.engine().backend();
            Backend bp = paged.engine().backend();
            check(bf.kvSequence() == null && bp.kvSequence() != null, dt + ": the paged backend has a block table, the flat one not");
            check(bf.stateFormat().equals(bp.stateFormat()), dt + ": the same saved-cache format " + bf.stateFormat());
            int[] ids = text(flat.engine(), 0, 150);
            int vocab = flat.weights().arch.core().vocabSize();
            float[] lf = new float[vocab];
            float[] lp = new float[vocab];
            boolean same = true;
            for (int pos = 0; pos < ids.length; pos += 64) {
                int n = Math.min(64, ids.length - pos);
                bf.forward(ids, pos, n, pos, lf);
                bp.forward(ids, pos, n, pos, lp);
                same &= Arrays.equals(lf, lp);
            }
            int pos = ids.length;
            int[] one = new int[1];
            for (int i = 0; i < 12 && same; i++) { // decode, then a rewind of 5 positions and a verification-shaped chunk
                one[0] = argmax(lf);
                bf.forward(one, 0, 1, pos, lf);
                bp.forward(one, 0, 1, pos, lp);
                same &= Arrays.equals(lf, lp);
                pos++;
            }
            int[] chunk = Arrays.copyOfRange(ids, 20, 25);
            bf.forward(chunk, 0, 5, pos - 5, lf);
            bp.forward(chunk, 0, 5, pos - 5, lp);
            same &= Arrays.equals(lf, lp);
            check(same, dt + ": prefill, decode and a rewind give bit-identical logits on the flat and the paged cache");
            int[] prompt = text(flat.engine(), 7, 100);
            Engine.Result rf = flat.engine().generate(prompt, GREEDY, null, () -> false);
            Engine.Result rp = paged.engine().generate(prompt, GREEDY, null, () -> false);
            check(Arrays.equals(rf.tokens(), rp.tokens()), dt + ": the engine's greedy answer is the same (" + abbreviate(rp.text()) + ")");
        }
    }

    /** Two slots with a shared prefix store it once; a fork in the middle of a block, a retry; all give the flat answers. */
    static void sharing(Path dir) throws Exception {
        try (ModelLoader.Loaded flat = load(dir, 1);
             ModelLoader.Loaded paged = load(dir, 3, "--backend.kv-paged", "true")) {
            Engine f = flat.engine();
            Engine p1 = paged.engine();
            Engine p2 = extra(dir, paged, "--backend.kv-paged", "true");
            Engine p3 = extra(dir, paged, "--backend.kv-paged", "true");
            KvPool pool = p1.backend().kvSequence().pool();
            check(p2.backend().kvSequence().pool() == pool && p3.backend().kvSequence().pool() == pool, "the slots of a model share one pool");
            int[] sys = text(f, 0, 112);       // 7 blocks of 16
            int[] a = concat(sys, text(f, 500, 40));
            int[] b = concat(sys, text(f, 900, 45));
            Engine.Result ra = p1.generate(a, GREEDY, null, () -> false);
            int afterA = pool.residentBlocks();
            Engine.Result rb = p2.generate(b, GREEDY, null, () -> false);
            int afterB = pool.residentBlocks();
            check(rb.stats().cachedTokens() >= 112, "the second slot found the shared prefix: " + rb.stats().cachedTokens() + " of "
                    + b.length + " tokens cached");
            check(afterB - afterA < afterA - 4, "the shared prefix is stored once: " + afterA + " blocks for the first conversation, "
                    + (afterB - afterA) + " more for the second");
            Engine.Result fa = f.generate(a, GREEDY, null, () -> false);
            Engine.Result fb = f.generate(b, GREEDY, null, () -> false);
            check(Arrays.equals(ra.tokens(), fa.tokens()), "first conversation == flat");
            check(Arrays.equals(rb.tokens(), fb.tokens()), "second conversation (prefix from the first slot) == flat");
            // a retry in the same slot: everything is cached, the answer is the same
            Engine.Result rr = p1.generate(a, GREEDY, null, () -> false);
            check(rr.stats().cachedTokens() == a.length - 1 && Arrays.equals(rr.tokens(), ra.tokens()),
                    "a retry reuses the whole prompt (" + rr.stats().cachedTokens() + " of " + (a.length - 1) + ") and answers the same");
            // a fork in the middle of a block: shares 120 tokens of a, diverges inside block 7
            int[] c = concat(Arrays.copyOf(a, 123), text(f, 1300, 30));
            Engine.Result rc = p3.generate(c, GREEDY, null, () -> false);
            Engine.Result fc = f.generate(c, GREEDY, null, () -> false);
            check(rc.stats().cachedTokens() >= 120, "a fork at 123 reuses " + rc.stats().cachedTokens() + " tokens (copy on write of the shared block)");
            check(Arrays.equals(rc.tokens(), fc.tokens()), "the fork's answer == flat");
            // the original conversation was not disturbed by the fork
            Engine.Result ra2 = p2.generate(a, GREEDY, null, () -> false);
            check(Arrays.equals(ra2.tokens(), fa.tokens()), "a conversation re-run after the fork still answers like flat");
            KvPool.Stats s = pool.stats();
            check(s.usedBlocks() > 0 && s.copiesOnWrite() > 0, "copies on write happened: " + s.copiesOnWrite());
            p2.close();
            p3.close();
        }
    }

    /** A pool too small for everything evicts, spills to disk, and brings a prefix back. */
    static void evictionAndSpill(Path dir) throws Exception {
        Path spillDir = Files.createTempDirectory("kvspill");
        try (ModelLoader.Loaded flat = load(dir, 1);
             ModelLoader.Loaded paged = load(dir, 1, "--backend.kv-paged", "true", "--backend.kv-pool-tokens", "480")) {
            Engine f = flat.engine();
            Engine p = paged.engine();
            KvPool pool = p.backend().kvSequence().pool();
            pool.setSpill(new FileSpill(spillDir, "paged-engine-test", 1L << 30));
            int[] a = text(f, 0, 160);
            Engine.Result ra = p.generate(a, GREEDY, null, () -> false);
            for (int i = 1; i <= 4; i++) { // other conversations push it out of the 30-block pool
                p.generate(text(f, 1000 * i, 160), GREEDY, null, () -> false);
            }
            KvPool.Stats s = pool.stats();
            check(s.evictions() > 0 && s.spills() > 0 && s.spillFiles() > 0, "blocks were evicted (" + s.evictions() + ") and spilled ("
                    + s.spillFiles() + " files, " + s.spillBytes() / 1000000 + " MB)");
            int[] a2 = concat(a, text(f, 77, 20));
            Engine.Result rb = p.generate(a2, GREEDY, null, () -> false);
            check(pool.stats().reloads() > 0 && rb.stats().cachedTokens() >= 128, "the first conversation's prefix came back from disk: "
                    + rb.stats().cachedTokens() + " tokens, " + pool.stats().reloads() + " blocks reloaded");
            Engine.Result fb = f.generate(a2, GREEDY, null, () -> false);
            check(Arrays.equals(rb.tokens(), fb.tokens()), "and the answer == flat");
            check(pool.residentBlocks() <= pool.maxBlocks(), "the pool stayed within " + pool.maxBlocks() + " blocks");
        } finally {
            try (var walk = Files.walk(spillDir)) {
                walk.sorted(java.util.Comparator.reverseOrder()).forEach(x -> x.toFile().delete());
            }
        }
    }

    /** Prompt-lookup speculation rolls positions back: the paged answer is the flat one. */
    static void speculation(Path dir) throws Exception {
        try (ModelLoader.Loaded flat = load(dir, 1, "--engine.speculative", "ngram");
             ModelLoader.Loaded paged = load(dir, 1, "--engine.speculative", "ngram", "--backend.kv-paged", "true")) {
            int[] prompt = text(flat.engine(), 0, 120); // repetitive: the n-gram drafts hit and miss
            SamplingParams longer = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 48, List.of());
            Engine.Result rf = flat.engine().generate(prompt, longer, null, () -> false);
            Engine.Result rp = paged.engine().generate(prompt, longer, null, () -> false);
            check(Arrays.equals(rf.tokens(), rp.tokens()), "speculation: paged == flat (" + rp.stats().draftedTokens() + " drafted, "
                    + rp.stats().acceptedTokens() + " accepted)");
        }
    }

    /** Three slots decoding together (batched rows through the block tables) answer as they do alone on the flat cache. */
    static void batched(Path dir) throws Exception {
        try (ModelLoader.Loaded flat = load(dir, 1);
             ModelLoader.Loaded paged = load(dir, 3, "--backend.kv-paged", "true")) {
            Engine[] pe = {paged.engine(), extra(dir, paged, "--backend.kv-paged", "true"), extra(dir, paged, "--backend.kv-paged", "true")};
            int[][] prompts = {text(flat.engine(), 0, 60), text(flat.engine(), 300, 70), text(flat.engine(), 600, 50)};
            String[] want = new String[3];
            for (int i = 0; i < 3; i++) {
                want[i] = Arrays.toString(flat.engine().generate(prompts[i], GREEDY, null, () -> false).tokens());
            }
            ExecutorService ex = Executors.newFixedThreadPool(3);
            List<Future<String>> fs = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                int k = i;
                fs.add(ex.submit(() -> Arrays.toString(pe[k].generate(prompts[k], GREEDY, null, () -> false).tokens())));
            }
            for (int i = 0; i < 3; i++) {
                check(want[i].equals(fs.get(i).get()), "slot " + i + " decoded in shared passes answers as flat alone");
            }
            ex.shutdown();
            pe[1].close();
            pe[2].close();
        }
    }

    /** A cache saved from the flat engine loads into the paged one and back; the continuation is the same. */
    static void savedCaches(Path dir) throws Exception {
        Path tmp = Files.createTempDirectory("kvsave");
        try (ModelLoader.Loaded flat = load(dir, 1);
             ModelLoader.Loaded paged = load(dir, 1, "--backend.kv-paged", "true")) {
            Engine f = flat.engine();
            Engine p = paged.engine();
            int[] a = text(f, 0, 100);
            f.generate(a, GREEDY, null, () -> false);
            Path ff = tmp.resolve("flat.kv");
            check(f.saveCache(ff, "fp") > 0, "the flat engine saved its cache");
            check(p.loadCache(ff, "fp") > 0, "the paged engine loaded the flat engine's file");
            int[] a2 = concat(a, text(f, 55, 10));
            Engine.Result rp = p.generate(a2, GREEDY, null, () -> false);
            Engine.Result rf = f.generate(a2, GREEDY, null, () -> false);
            check(rp.stats().cachedTokens() >= 90 && Arrays.equals(rp.tokens(), rf.tokens()), "continuing from the loaded cache == flat ("
                    + rp.stats().cachedTokens() + " tokens reused)");
            // paged saves (in the background, from a snapshot) while the next request writes; flat loads it
            Path pf = tmp.resolve("paged.kv");
            Engine.PendingSave ps = p.beginSave(pf, "fp");
            Engine.Result next = p.generate(concat(a2, text(f, 5, 12)), GREEDY, null, () -> false); // runs while the save is pending
            int saved = ps.write();
            check(saved > 100,"the paged engine saved " + saved
                    + " tokens from a snapshot while it went on serving");
            check(f.loadCache(pf, "fp") == saved, "the flat engine loaded the paged engine's file");
            int[] a3 = concat(Arrays.copyOf(a2, 105), text(f, 9, 6));
            Engine.Result r1 = f.generate(a3, GREEDY, null, () -> false);
            Engine.Result r2 = paged.engine().generate(a3, GREEDY, null, () -> false);
            check(Arrays.equals(r1.tokens(), r2.tokens()), "after the round trip both answer alike (" + next.tokens().length + ")");
        } finally {
            try (var walk = Files.walk(tmp)) {
                walk.sorted(java.util.Comparator.reverseOrder()).forEach(x -> x.toFile().delete());
            }
        }
    }

    private static int argmax(float[] l) {
        int b = 0;
        for (int i = 1; i < l.length; i++) {
            if (l[i] > l[b]) {
                b = i;
            }
        }
        return b;
    }

    private static String abbreviate(String s) {
        String one = s.replace('\n', ' ');
        return one.length() > 40 ? one.substring(0, 40) + "…" : one;
    }
}
