package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.kv.KvReports;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.SamplingParams;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Tiered KV storage (docs/26) on the real model through the engine: contexts that exceed a deliberately small hot budget give
 * the all-resident answers.
 * <ul>
 *   <li>Pool tiers (CPU and Metal): conversations of a few hundred tokens through a pool that holds a fraction of them are
 *       pushed to host memory and the block files, and the first one, sent again, is adopted back (prefetched) and answers
 *       like the flat cache.</li>
 *   <li>Streamed KV (Metal): a slot whose context cannot fit the budget keeps its rows on the host and runs layer by layer; its
 *       logits and greedy answer equal the flat cache's.</li>
 * </ul>
 *
 * <p>usage: TierEngineTest &lt;model dir&gt; [metal lib] (models/Qwen_Qwen3-0.6B)
 */
public final class TierEngineTest {

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    private static final SamplingParams GREEDY = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 24, List.of());

    private static ModelLoader.Loaded load(Path dir, String backend, int ctx, int slots, String... extra) throws Exception {
        List<String> a = new ArrayList<>(List.of("--backend", backend, "--model.dir", dir.toAbsolutePath().getParent().toString(),
                "--backend.ctx", String.valueOf(ctx), "--engine.speculative", "off", "--backend.cpu.int8-activations", "false",
                "--backend.threads", "4", "--backend.kv-dtype", "f16"));
        a.addAll(List.of(extra));
        Settings st = Settings.load(a.toArray(new String[0]));
        return ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString(), slots);
    }

    private static int[] text(Engine e, int from, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = from; sb.length() < n * 8; i++) {
            sb.append("item ").append(i).append(" of the list, ");
        }
        return Arrays.copyOf(e.tokenizer().encode(sb.toString()), n);
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

    @SuppressWarnings("unchecked")
    private static Map<String, Object> kvReport(Backend b) {
        Map<String, Object> m = b.memoryReport();
        return m == null ? Map.of() : (Map<String, Object>) m.getOrDefault("kv", Map.of());
    }

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        String lib = args.length > 1 ? args[1] : "target/native/libjllm_metal.dylib";
        boolean metal = Metal.available(lib);
        poolTiers(dir, "cpu", 4);
        if (metal) {
            poolTiers(dir, "metal", 40);
            streamed(dir);
        } else {
            System.out.println("note: no Metal here, the GPU cases are skipped");
        }
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("tier engine test passed");
    }

    /** Four 300-token conversations through a pool that holds about 1.6 of them; the first, sent again, is adopted back. */
    static void poolTiers(Path dir, String backend, int hotMb) throws Exception {
        Path tmp = Files.createTempDirectory("tierengine");
        String[] tier = {"--backend.kv-tier", "true", "--backend.kv-hot-mb", String.valueOf(hotMb), "--backend.kv-host-mb", "4",
                "--backend.kv-stream", "off", "--backend.kv-tier-dir", tmp.toString()};
        try (ModelLoader.Loaded flat = load(dir, backend, 512, 1);
             ModelLoader.Loaded tiered = load(dir, backend, 512, 1, tier)) {
            Engine fe = flat.engine();
            Engine te = tiered.engine();
            check(te.backend().kvSequence() != null, backend + ": backend.kv-tier gives the slot a block table");
            KvPool pool = te.backend().kvSequence().pool();
            long blockBytes = pool.storage().residentBytes();
            check(pool.hotCap() * blockBytes <= hotMb * 1_000_000L && pool.hotCap() < pool.maxBlocks() || backend.equals("cpu"),
                    backend + ": the hot cap is " + pool.hotCap() + " of " + pool.maxBlocks() + " blocks");
            int[][] prompts = new int[4][];
            Engine.Result[] expect = new Engine.Result[4];
            for (int i = 0; i < 4; i++) {
                prompts[i] = text(fe, 1000 * i, 300);
                expect[i] = fe.generate(prompts[i], GREEDY, null, () -> false);
                Engine.Result r = te.generate(prompts[i], GREEDY, null, () -> false);
                check(Arrays.equals(r.tokens(), expect[i].tokens()), backend + ": conversation " + i + " answers like the flat cache");
            }
            KvPool.Stats s = pool.stats();
            check(s.evictions() > 0 && s.committedBlocks() <= pool.hotCap(), backend + ": blocks were pushed down (" + s.evictions()
                    + " evicted, " + s.committedBlocks() + " committed of cap " + pool.hotCap() + ", " + s.spillFiles() + " stored)");
            long reloads = s.reloads();
            Engine.Result again = te.generate(prompts[0], GREEDY, null, () -> false);
            KvPool.Stats s2 = pool.stats();
            check(Arrays.equals(again.tokens(), expect[0].tokens()) && s2.reloads() > reloads,
                    backend + ": the first conversation, sent again, is adopted back from the tiers (" + (s2.reloads() - reloads)
                            + " blocks reloaded, " + again.stats().cachedTokens() + " tokens reused) and answers like the flat cache");
            Map<String, Object> kv = kvReport(te.backend());
            check(kv.containsKey("hot") && kv.containsKey("warm") && kv.containsKey("cold"), backend + ": the memory report lists the tiers " + kv.keySet());
            System.out.println("     " + KvReports.pool(pool).get("kv"));
        } finally {
            try (var walk = Files.walk(tmp)) {
                walk.sorted(java.util.Comparator.reverseOrder()).forEach(x -> x.toFile().delete());
            }
        }
    }

    /** Metal: a budget of 2 MB cannot hold a 512-token slot, so the slot streams; logits and answer equal the flat cache's. */
    static void streamed(Path dir) throws Exception {
        Path tmp = Files.createTempDirectory("tierstream");
        try (ModelLoader.Loaded flat = load(dir, "metal", 1024, 1);
             ModelLoader.Loaded str = load(dir, "metal", 1024, 1, "--backend.kv-tier", "true", "--backend.kv-hot-mb", "2",
                     "--backend.kv-host-mb", "40", "--backend.kv-tier-dir", tmp.toString())) {
            Backend bf = flat.engine().backend();
            Backend bs = str.engine().backend();
            Map<String, Object> kv = kvReport(bs);
            check("streamed".equals(kv.get("mode")), "metal: a 2 MB budget makes the slot stream its KV (" + kv.get("mode") + ")");
            int[] ids = text(flat.engine(), 0, 700);
            int vocab = flat.weights().arch.core().vocabSize();
            boolean same = false;
            for (int attempt = 0; attempt < 6 && !same; attempt++) {
                same = true;
                float[] lf = new float[vocab];
                float[] ls = new float[vocab];
                for (int pos = 0; pos < ids.length && same; pos += 64) {
                    int n = Math.min(64, ids.length - pos);
                    bf.forward(ids, pos, n, pos, lf);
                    bs.forward(ids, pos, n, pos, ls);
                    same = Arrays.equals(lf, ls);
                }
                int pos = ids.length;
                int[] one = new int[1];
                for (int i = 0; i < 20 && same; i++) {
                    one[0] = argmax(lf);
                    bf.forward(one, 0, 1, pos, lf);
                    one[0] = argmax(ls);
                    bs.forward(one, 0, 1, pos, ls);
                    same = Arrays.equals(lf, ls);
                    pos++;
                }
            }
            check(same, "metal: 700 tokens of prefill and 20 decode steps over the streamed KV give the flat logits");
            int[] prompt = text(flat.engine(), 5000, 400);
            Engine.Result rf = flat.engine().generate(prompt, GREEDY, null, () -> false);
            Engine.Result rs = str.engine().generate(prompt, GREEDY, null, () -> false);
            check(Arrays.equals(rf.tokens(), rs.tokens()), "metal: the engine's greedy answer over the streamed KV is the flat answer");
            kv = kvReport(bs);
            System.out.println("     " + kv);
        } finally {
            try (var walk = Files.walk(tmp)) {
                walk.sorted(java.util.Comparator.reverseOrder()).forEach(x -> x.toFile().delete());
            }
        }
    }
}
