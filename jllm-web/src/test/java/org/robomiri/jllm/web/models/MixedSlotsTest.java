package org.robomiri.jllm.web.models;

import org.junit.jupiter.api.Test;
import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.engine.Admission;
import org.robomiri.jllm.engine.BatchGroup;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.Priority;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.kv.KvSequence;
import org.robomiri.jllm.kv.KvStorage;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;
import org.robomiri.jllm.web.WebProperties;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The mixed scheduler (docs/25) through {@link LoadedModel}: slots of one model in a mixed {@link BatchGroup} over a paged KV
 * cache. The backends are fakes whose cache rows are a rolling hash of the tokens up to each position and whose next token
 * is a function of the last row, so a row that is shared, spilled or reloaded wrongly changes the generated text. Checked:
 * a pass carries a prefill chunk together with the decode tokens of others and every conversation generates what it
 * generates alone; an interactive request overtakes background ones in the queue; a background generation is paused for an
 * interactive one, its blocks spilled to disk while the pool is small, and continues to the same text; a request whose client
 * left is dropped from the queue at once.
 */
class MixedSlotsTest {

    static final int CTX = 8192;
    static final int B = 16;

    static final class RowStorage implements KvStorage {
        final long[][] rows;

        RowStorage(int blocks) {
            rows = new long[blocks][B];
        }

        @Override
        public int blockTokens() {
            return B;
        }

        @Override
        public int spillBytes() {
            return 8 * B;
        }

        @Override
        public long residentBytes() {
            return 8L * B;
        }

        @Override
        public void grow(int blocks) {
        }

        @Override
        public void copy(int src, int dst, int n) {
            System.arraycopy(rows[src], 0, rows[dst], 0, B);
        }

        @Override
        public void save(int block, ByteBuffer dst) {
            for (long v : rows[block]) {
                dst.putLong(v);
            }
        }

        @Override
        public void load(int block, ByteBuffer src) {
            for (int i = 0; i < B; i++) {
                rows[block][i] = src.getLong();
            }
        }

        @Override
        public void close() {
        }
    }

    /** The compositions of the passes ({@code forwardMixed} calls) of every backend of one test: rows per segment. */
    static final List<int[]> PASSES = Collections.synchronizedList(new ArrayList<>());
    /** Milliseconds one pass takes (the "device" is busy), so that several conversations overlap. */
    static volatile int passMillis = 3;

    static final class MixBackend implements Backend {
        final KvSequence seq;
        final RowStorage st;

        MixBackend(KvPool pool) {
            this.seq = pool.newSequence(CTX);
            this.st = (RowStorage) pool.storage();
        }

        @Override
        public String name() {
            return "mix-fake";
        }

        @Override
        public int contextLength() {
            return CTX;
        }

        @Override
        public int maxBatch() {
            return 64 + 32;
        }

        private long row(int p) {
            return st.rows[seq.block(p / B)][p % B];
        }

        /** Row p = f(row p - 1, token): like a key, it depends on every token before. */
        private void compute(int[] tokens, int off, int n, int pos, float[] logits, int stride) {
            seq.beginWrite(pos, n);
            long h = pos == 0 ? 1469598103934665603L : row(pos - 1);
            for (int i = 0; i < n; i++) {
                h = (h ^ (tokens[off + i] + 1)) * 1099511628211L + (pos + i);
                st.rows[seq.block((pos + i) / B)][(pos + i) % B] = h;
                if (logits != null && (stride > 0 || i == n - 1)) {
                    int base = stride > 0 ? i * stride : 0;
                    Arrays.fill(logits, base, base + Math.min(logits.length - base, stride > 0 ? stride : logits.length), 0f);
                    logits[base + (int) Math.floorMod(h >>> 7, 1000L) + 10] = 1f;
                }
            }
        }

        @Override
        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            compute(tokens, off, n, pos, logits, 0);
        }

        @Override
        public void forwardAll(int[] tokens, int off, int n, int pos, float[] logitsOut, int stride) {
            compute(tokens, off, n, pos, logitsOut, stride);
        }

        @Override
        public boolean forwardMixed(Segment[] segs) {
            int[] sizes = new int[segs.length];
            for (int i = 0; i < segs.length; i++) {
                Segment s = segs[i];
                sizes[i] = s.n();
                ((MixBackend) s.owner()).compute(s.tokens(), s.off(), s.n(), s.pos(), s.logits(), s.stride());
            }
            PASSES.add(sizes);
            try {
                Thread.sleep(passMillis);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return true;
        }

        @Override
        public KvSequence kvSequence() {
            return seq;
        }

        @Override
        public void close() {
            seq.close();
        }
    }

    private static Path tokenizerJson() {
        for (Path d = Path.of("").toAbsolutePath(); d != null; d = d.getParent()) {
            Path p = d.resolve("models/Qwen_Qwen3-0.6B/tokenizer.json");
            if (Files.isRegularFile(p)) {
                return p;
            }
        }
        return null;
    }

    private static int[] tokens(Random r, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 100 + r.nextInt(5000);
        }
        return t;
    }

    private record Setup(LoadedModel model, KvPool pool, KvStore store, List<Engine> engines, BatchGroup group, Path dir) {
    }

    private static Setup setup(int slots, int blocks, boolean spill, Map<String, String> settings) throws Exception {
        Path tj = tokenizerJson();
        assumeTrue(tj != null, "models/Qwen_Qwen3-0.6B/tokenizer.json not found: skipped");
        Tokenizer tok = Tokenizer.load(tj);
        Architecture cfg = Architecture.parse(Json.parseObject("{\"model_type\":\"qwen3\",\"hidden_size\":64,"
                + "\"num_hidden_layers\":1,\"num_attention_heads\":2,\"intermediate_size\":128,\"vocab_size\":"
                + tok.vocabSize() + ",\"eos_token_id\":" + tok.tokenId("<|im_end|>") + "}"));
        Path dir = Files.createTempDirectory("mixed-slots");
        WebProperties props = new WebProperties("x", "", List.of(), 100, "y", "deepseek", false, dir.toString(), 1.0, 512, 0,
                List.of(), spill, 1.0);
        KvStore store = new KvStore(props);
        KvPool pool = new KvPool(new RowStorage(blocks), blocks, "long rows");
        BatchGroup group = new BatchGroup();
        group.enableMixed(new BatchGroup.MixedConfig(0, 32, 16, 10_000_000_000L, 0));
        List<Engine> engines = new ArrayList<>();
        for (int i = 0; i < slots; i++) {
            engines.add(new Engine(group.join(new MixBackend(pool), 32), tok, cfg, Set.of(), 0, ChatTemplate.forModelType("qwen3")));
        }
        Map<String, String> st = new java.util.HashMap<>(settings);
        st.put("engine.scheduler", "mixed");
        LoadedModel model = new LoadedModel("m", "mixed-test", new ModelLoader.Loaded(null, engines.getFirst()),
                engines.subList(1, engines.size()), st, 0, store);
        store.attachSpill(model);
        PASSES.clear();
        passMillis = 3;
        return new Setup(model, pool, store, engines, group, dir);
    }

    private static void cleanup(Setup s) throws Exception {
        for (Engine e : s.engines()) {
            e.close();
        }
        s.store().clear();
        try (var w = Files.walk(s.dir())) {
            for (Path p : w.sorted(java.util.Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(p);
            }
        }
    }

    private static final SamplingParams GREEDY40 = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 40, List.of());

    private static Engine.Result run(LoadedModel model, int[] prompt, Priority prio, SamplingParams p, java.util.function.BooleanSupplier gone) {
        try (Priority.Scope ignored = Priority.enter(prio); LoadedModel.Slot slot = model.pick(prompt, null, gone)) {
            return slot.engine().generate(prompt, p, null, gone);
        }
    }

    /** What a conversation generates when it runs alone (a fresh pool, no other slot busy). */
    private static int[] alone(int[] prompt, SamplingParams p) throws Exception {
        Setup s = setup(1, 4000, false, Map.of());
        try {
            return run(s.model(), prompt, Priority.INTERACTIVE, p, () -> false).tokens();
        } finally {
            cleanup(s);
        }
    }

    @Test
    void aPassCarriesAPrefillChunkAndTheDecodeTokensOfOthers() throws Exception {
        Random r = new Random(3);
        int[] shortA = tokens(r, 30);
        int[] shortB = tokens(r, 40);
        int[] big = tokens(r, 900);
        SamplingParams longGen = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 120, List.of());
        int[] wantA = alone(shortA, longGen);
        int[] wantB = alone(shortB, longGen);
        int[] wantBig = alone(big, GREEDY40);
        Setup s = setup(3, 4000, false, Map.of());
        try {
            ExecutorService ex = Executors.newFixedThreadPool(3);
            CountDownLatch generating = new CountDownLatch(2);
            Future<Engine.Result> fa = ex.submit(() -> runWithSink(s.model(), shortA, longGen, generating));
            Future<Engine.Result> fb = ex.submit(() -> runWithSink(s.model(), shortB, longGen, generating));
            assertTrue(generating.await(10, TimeUnit.SECONDS));
            Future<Engine.Result> fbig = ex.submit(() -> run(s.model(), big, Priority.INTERACTIVE, GREEDY40, () -> false));
            assertArrayEquals(wantA, fa.get(30, TimeUnit.SECONDS).tokens(), "slot A generates what it generates alone");
            assertArrayEquals(wantB, fb.get(30, TimeUnit.SECONDS).tokens(), "slot B generates what it generates alone");
            assertArrayEquals(wantBig, fbig.get(30, TimeUnit.SECONDS).tokens(), "the long prompt's answer is the same too");
            int withChunk = 0;
            for (int[] pass : PASSES) {
                boolean chunk = pass[0] > 1;
                boolean rider = pass.length > 1 && Arrays.stream(pass).skip(1).anyMatch(n -> n == 1);
                withChunk += chunk && rider ? 1 : 0;
            }
            long[] st = s.group().mixedStats();
            System.out.println("mixed passes with a chunk and decode tokens: " + withChunk + " of " + PASSES.size() + " (group: "
                    + Arrays.toString(st) + ")");
            assertTrue(withChunk >= 3, "prefill chunks of the long prompt rode with the others' decode tokens: " + withChunk);
            assertTrue(st[0] >= 3 && st[1] >= 3, "the group counted them");
            ex.shutdown();
        } finally {
            cleanup(s);
        }
    }

    private static Engine.Result runWithSink(LoadedModel model, int[] prompt, SamplingParams p, CountDownLatch started) {
        try (LoadedModel.Slot slot = model.pick(prompt)) {
            return slot.engine().generate(prompt, p, t -> started.countDown(), () -> false);
        }
    }

    @Test
    void anInteractiveRequestOvertakesBackgroundOnesInTheQueue() throws Exception {
        Random r = new Random(4);
        Setup s = setup(1, 4000, false, Map.of());
        try {
            passMillis = 6;
            SamplingParams p = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 30, List.of());
            ExecutorService ex = Executors.newFixedThreadPool(4);
            CountDownLatch holding = new CountDownLatch(1);
            Future<Engine.Result> first = ex.submit(() -> runWithSink(s.model(), tokens(r, 20), p, holding));
            assertTrue(holding.await(10, TimeUnit.SECONDS));
            List<String> order = Collections.synchronizedList(new ArrayList<>());
            int[] bgPrompt = tokens(r, 25);
            int[] iaPrompt = tokens(r, 26);
            Future<?> bg = ex.submit(() -> {
                run(s.model(), bgPrompt, Priority.BACKGROUND, new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 2, List.of()), () -> false);
                order.add("background");
            });
            waitUntil(() -> s.model().admission().queued() == 1);
            Future<?> ia = ex.submit(() -> {
                run(s.model(), iaPrompt, Priority.INTERACTIVE, new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 2, List.of()), () -> false);
                order.add("interactive");
            });
            waitUntil(() -> s.model().admission().queued() == 2);
            first.get(30, TimeUnit.SECONDS);
            bg.get(30, TimeUnit.SECONDS);
            ia.get(30, TimeUnit.SECONDS);
            assertEquals(List.of("interactive", "background"), order, "the interactive request, which came later, was served first");
            ex.shutdown();
        } finally {
            cleanup(s);
        }
    }

    @Test
    void aBackgroundGenerationIsPausedForAnInteractiveOneAndItsBlocksAreSpilled() throws Exception {
        Random r = new Random(7);
        // 2 slots, one running place: the interactive request can only run by pausing the background one. The pool is
        // small, so the paused blocks are evicted to the block store while the interactive request runs.
        int[] bgPrompt = tokens(r, 1500);
        int[] iaPrompt = tokens(r, 1500);
        SamplingParams longGen = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 150, List.of());
        SamplingParams shortGen = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 20, List.of());
        int[] wantBg = alone(bgPrompt, longGen);
        int[] wantIa = alone(iaPrompt, shortGen);
        Setup s = setup(2, 150, true, Map.of("engine.max-running", "1", "engine.preempt", "true")); // 2400 positions
        try {
            passMillis = 4;
            ExecutorService ex = Executors.newFixedThreadPool(2);
            CountDownLatch generating = new CountDownLatch(5);
            Future<Engine.Result> bg = ex.submit(() -> {
                try (Priority.Scope ignored = Priority.enter(Priority.BACKGROUND); LoadedModel.Slot slot = s.model().pick(bgPrompt, null, () -> false)) {
                    return slot.engine().generate(bgPrompt, longGen, t -> generating.countDown(), () -> false);
                }
            });
            assertTrue(generating.await(20, TimeUnit.SECONDS), "the background request is generating");
            long t0 = System.nanoTime();
            Engine.Result ia = run(s.model(), iaPrompt, Priority.INTERACTIVE, shortGen, () -> false);
            double iaMs = (System.nanoTime() - t0) / 1e6;
            Engine.Result bgr = bg.get(60, TimeUnit.SECONDS);
            Admission<?> q = s.model().admission();
            long[] st = q.stats();
            KvPool.Stats ps = s.pool().stats();
            System.out.printf("interactive request took %.0f ms; admission grants=%d pauses=%d resumes=%d; pool evictions=%d spills=%d reloads=%d%n",
                    iaMs, st[0], st[1], st[2], ps.evictions(), ps.spills(), ps.reloads());
            assertEquals(1, st[1], "the background slot was paused once");
            assertTrue(st[2] >= 1, "and resumed");
            assertArrayEquals(wantIa, ia.tokens(), "the interactive request generated what it generates alone");
            assertArrayEquals(wantBg, bgr.tokens(), "the paused and resumed background generation continued to the very same tokens");
            assertTrue(ps.spills() > 0, "its blocks were spilled to the block store while it was paused: " + ps.spills());
            assertTrue(ps.reloads() > 0, "and read back when it resumed: " + ps.reloads());
            ex.shutdown();
        } finally {
            cleanup(s);
        }
    }

    @Test
    void aQueuedRequestWhoseClientLeftIsWokenAndDropped() throws Exception {
        Random r = new Random(8);
        Setup s = setup(1, 4000, false, Map.of());
        try {
            passMillis = 6;
            ExecutorService ex = Executors.newFixedThreadPool(2);
            CountDownLatch holding = new CountDownLatch(1);
            Future<Engine.Result> first = ex.submit(() -> runWithSink(s.model(), tokens(r, 20),
                    new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 150, List.of()), holding));
            assertTrue(holding.await(10, TimeUnit.SECONDS));
            AtomicBoolean gone = new AtomicBoolean();
            AtomicReference<Throwable> thrown = new AtomicReference<>();
            Thread waiter = new Thread(() -> {
                try (LoadedModel.Slot slot = s.model().pick(tokens(new Random(1), 30), null, gone::get)) {
                    thrown.set(new IllegalStateException("got a slot"));
                } catch (Admission.Abandoned e) {
                    thrown.set(e);
                }
            });
            waiter.start();
            waitUntil(() -> s.model().admission().queued() == 1);
            long t0 = System.nanoTime();
            gone.set(true);
            waiter.join(2000);
            double ms = (System.nanoTime() - t0) / 1e6;
            System.out.printf("a queued request is dropped %.0f ms after its client left%n", ms);
            assertTrue(thrown.get() instanceof Admission.Abandoned, "the waiting request gave up: " + thrown.get());
            assertTrue(ms < 500, "within half a second: " + ms);
            assertEquals(0, s.model().admission().queued());
            assertThrows(Admission.Abandoned.class, () -> {
                try (LoadedModel.Slot slot = s.model().pick(tokens(new Random(2), 30), null, () -> true)) {
                    assertTrue(false, "a request whose client is gone must not get the busy slot");
                }
            });
            first.get(30, TimeUnit.SECONDS);
            ex.shutdown();
        } finally {
            cleanup(s);
        }
    }

    private static void waitUntil(java.util.function.BooleanSupplier c) throws InterruptedException {
        long end = System.nanoTime() + 10_000_000_000L;
        while (!c.getAsBoolean() && System.nanoTime() < end) {
            Thread.sleep(2);
        }
        assertTrue(c.getAsBoolean(), "condition not reached in 10 s");
    }
}
