package org.robomiri.jllm.web.models;

import org.junit.jupiter.api.Test;
import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.engine.Engine;
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
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Slots over a paged KV cache (docs/24) through {@link LoadedModel}: two conversations that share a long system prompt
 * store it once (the pool's resident bytes are counted, at the byte size of a Qwen3-0.6B f16 cache position), the second
 * prefills only its tail whichever slot it lands on, and with the block store ({@link KvStore#attachSpill}) a
 * conversation pushed out of a small pool comes back from disk. The backends are fakes whose "rows" are the token ids fed
 * at each position, so a block that is shared or reloaded wrongly shows as a wrong row.
 */
class PagedSlotsTest {

    /** Qwen3-0.6B, f16: 28 layers x (K + V) x 1024 values x 2 bytes. */
    static final long BYTES_PER_TOKEN = 28L * 2 * 1024 * 2;
    static final int CTX = 8192;

    static final class IntStorage implements KvStorage {
        final int B = 16;
        final int[][] rows;

        IntStorage(int blocks) {
            rows = new int[blocks][B];
        }

        @Override
        public int blockTokens() {
            return B;
        }

        @Override
        public int spillBytes() {
            return 4 * B;
        }

        @Override
        public long residentBytes() {
            return BYTES_PER_TOKEN * B;
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
            for (int v : rows[block]) {
                dst.putInt(v);
            }
        }

        @Override
        public void load(int block, ByteBuffer src) {
            for (int i = 0; i < B; i++) {
                rows[block][i] = src.getInt();
            }
        }

        @Override
        public void close() {
        }
    }

    /** A backend whose cache is a block table of token ids. */
    static final class PagedBackend implements Backend {
        final KvSequence seq;
        final IntStorage st;

        PagedBackend(KvPool pool) {
            this.seq = pool.newSequence(CTX);
            this.st = (IntStorage) pool.storage();
        }

        @Override
        public String name() {
            return "paged-fake";
        }

        @Override
        public int contextLength() {
            return CTX;
        }

        @Override
        public int maxBatch() {
            return 256;
        }

        @Override
        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            seq.beginWrite(pos, n);
            for (int i = 0; i < n; i++) {
                st.rows[seq.block((pos + i) / st.B)][(pos + i) % st.B] = tokens[off + i];
            }
            if (logits != null) {
                Arrays.fill(logits, 0f);
                logits[(tokens[off + n - 1] * 31 + pos) % 1000 + 10] = 1f;
            }
        }

        /** The rows the table holds for positions [0, n). */
        int[] rows(int n) {
            int[] r = new int[n];
            for (int p = 0; p < n; p++) {
                r[p] = st.rows[seq.block(p / st.B)][p % st.B];
            }
            return r;
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

    private static int[] concat(int[] a, int[] b) {
        int[] c = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }

    private static Engine.Stats request(LoadedModel model, int[] prompt, PagedBackend[] outBackend) {
        SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 4, List.of());
        try (LoadedModel.Slot s = model.pick(prompt)) {
            Engine.Result r = s.engine().generate(prompt, greedy, null, () -> false);
            model.record(s, r.stats());
            if (outBackend != null) {
                outBackend[0] = (PagedBackend) ((Engine) s.engine()).backend();
            }
            return r.stats();
        }
    }

    private record Setup(LoadedModel model, KvPool pool, KvStore store, Engine e0, Engine e1, Path dir) {
    }

    private static Setup setup(int blocks, boolean spill) throws Exception {
        Path tj = tokenizerJson();
        assumeTrue(tj != null, "models/Qwen_Qwen3-0.6B/tokenizer.json not found: skipped");
        Tokenizer tok = Tokenizer.load(tj);
        Architecture cfg = Architecture.parse(Json.parseObject("{\"model_type\":\"qwen3\",\"hidden_size\":64,"
                + "\"num_hidden_layers\":1,\"num_attention_heads\":2,\"intermediate_size\":128,\"vocab_size\":"
                + tok.vocabSize() + ",\"eos_token_id\":" + tok.tokenId("<|im_end|>") + "}"));
        Path dir = Files.createTempDirectory("paged-slots");
        WebProperties props = new WebProperties("x", "", List.of(), 100, "y", "deepseek", false, dir.toString(), 1.0, 512, 0,
                List.of(), spill, 1.0, 120);
        KvStore store = new KvStore(props);
        KvPool pool = new KvPool(new IntStorage(blocks), blocks, "int rows");
        Engine e0 = new Engine(new PagedBackend(pool), tok, cfg, Set.of(), 0, ChatTemplate.forModelType("qwen3"));
        Engine e1 = new Engine(new PagedBackend(pool), tok, cfg, Set.of(), 0, ChatTemplate.forModelType("qwen3"));
        LoadedModel model = new LoadedModel("m", "paged-test", new ModelLoader.Loaded(null, e0), List.of(e1), Map.of(), 0, store);
        store.attachSpill(model);
        return new Setup(model, pool, store, e0, e1, dir);
    }

    private static void cleanup(Setup s) throws Exception {
        s.e0().close();
        s.e1().close();
        s.store().clear();
        try (var w = Files.walk(s.dir())) {
            for (Path p : w.sorted(java.util.Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(p);
            }
        }
    }

    @Test
    void twoConversationsWithASharedPrefixStoreItOnce() throws Exception {
        Setup s = setup(2000, false);
        try {
            Random r = new Random(5);
            int[] system = tokens(r, 4000);
            int[] a = concat(system, tokens(r, 120));
            int[] b = concat(system, tokens(r, 150));
            PagedBackend[] used = new PagedBackend[1];
            Engine.Stats sa = request(s.model(), a, used);
            PagedBackend ba = used[0];
            long oneConversation = s.pool().stats().residentBytes();
            Engine.Stats sb = request(s.model(), b, used);
            PagedBackend bb = used[0];
            long both = s.pool().stats().residentBytes();
            long flat = (long) (a.length + 4 + b.length + 4) * BYTES_PER_TOKEN; // what two flat slots hold for these tokens
            System.out.printf("one conversation %.1f MB, two sharing a 4000-token prefix %.1f MB (flat slots: %.1f MB)%n",
                    oneConversation / 1e6, both / 1e6, flat / 1e6);
            assertEquals(0, sa.cachedTokens(), "the first conversation prefills everything");
            assertTrue(sb.cachedTokens() >= 3984, "the second takes the system prompt from the pool: " + sb.cachedTokens() + " tokens");
            assertTrue(both < oneConversation * 1.1, "the shared prefix is stored once");
            assertTrue(both < flat * 0.55, "about half of what two flat caches hold");
            assertArrayEquals(Arrays.copyOf(a, 4000), Arrays.copyOf(ba.rows(4000), 4000), "the first slot reads its rows");
            assertArrayEquals(Arrays.copyOf(b, 4000), Arrays.copyOf(bb.rows(4000), 4000), "the second reads the shared rows");
            // the first conversation's next turn is still cached whole, in its own slot
            int[] a2 = concat(s.e0().cachedTokens().length >= a.length ? s.e0().cachedTokens() : s.e1().cachedTokens(), tokens(r, 30));
            Engine.Stats s2 = request(s.model(), a2, null);
            assertTrue(s2.cachedTokens() >= a.length, "a retry/continuation reuses the whole conversation: " + s2.cachedTokens());
        } finally {
            cleanup(s);
        }
    }

    @Test
    void anIdleSlotGivesItsBlocksBackAndTheConversationIsStillFound() throws Exception {
        Setup s = setup(600, false);
        try {
            Random r = new Random(21);
            int[] a = tokens(r, 4100);
            request(s.model(), a, null);
            long held = s.pool().stats().usedBlocks();
            assertTrue(held >= 256, "the slot holds its conversation: " + held + " blocks");
            // a slot that has not been idle long enough keeps them; after the timeout (here 0) it releases them
            assertTrue(!s.model().releaseIdle(60_000), "not idle for a minute yet");
            assertEquals(held, s.pool().stats().usedBlocks());
            assertTrue(s.model().releaseIdle(0), "idle: released");
            KvPool.Stats st = s.pool().stats();
            assertEquals(0, st.usedBlocks(), "no block is referenced any more");
            assertTrue(st.cachedBlocks() >= 256, "the conversation's full blocks stay in the tree: " + st.cachedBlocks());
            Engine.Stats back = request(s.model(), concat(a, tokens(r, 30)), null);
            assertTrue(back.cachedTokens() >= 4080, "the next turn finds it through the tree: " + back.cachedTokens() + " tokens");
        } finally {
            cleanup(s);
        }
    }

    @Test
    void aFullPoolAsksIdleSlotsForTheirBlocks() throws Exception {
        Setup s = setup(300, false); // room for one 4100-token conversation, not two
        try {
            s.model().enablePagedHousekeeping(0); // only the memory-pressure path
            Random r = new Random(23);
            int[] a = tokens(r, 4100);
            int[] c = tokens(r, 4100);
            request(s.model(), a, null);
            Engine.Stats sc = request(s.model(), c, null); // runs on the other slot; without the reclaimer the pool would be exhausted
            assertEquals(0, sc.cachedTokens());
            assertTrue(s.pool().stats().residentBlocks() <= 300);
        } finally {
            cleanup(s);
        }
    }

    @Test
    void anEvictedConversationComesBackFromTheBlockStore() throws Exception {
        Setup s = setup(560, true); // 8960 positions: two conversations of ~4100 fit, a third pushes the oldest out
        try {
            Random r = new Random(9);
            int[] a = tokens(r, 4100);
            int[] c = tokens(r, 4100);
            int[] d = tokens(r, 4100);
            request(s.model(), a, null);
            request(s.model(), c, null);
            request(s.model(), d, null); // both slots are in use: the oldest published blocks are evicted
            KvPool.Stats st = s.pool().stats();
            assertTrue(st.evictions() > 0 && st.spills() > 0, "blocks were evicted and spilled: " + st.evictions() + " / " + st.spills());
            assertTrue(st.residentBlocks() <= 560);
            Engine.Stats back = request(s.model(), concat(a, tokens(r, 20)), null);
            System.out.printf("evicted conversation: %d of %d tokens came back (%d blocks reloaded from %d files)%n",
                    back.cachedTokens(), a.length, s.pool().stats().reloads(), s.pool().stats().spillFiles());
            assertTrue(back.cachedTokens() >= 3000, "most of the first conversation came back: " + back.cachedTokens());
            assertTrue(s.pool().stats().reloads() > 0, "from disk");
        } finally {
            cleanup(s);
        }
    }
}
