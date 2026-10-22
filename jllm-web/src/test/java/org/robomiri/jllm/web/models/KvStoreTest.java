package org.robomiri.jllm.web.models;

import org.junit.jupiter.api.Test;
import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;
import org.robomiri.jllm.web.WebProperties;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * {@link KvStore} end to end on a fake backend whose "cache" is the token id fed at each position (so a restored
 * state can be compared with what was saved): saves after a request (written in the background, skipped while the
 * conversation has not grown enough), a grown conversation replacing its old file, a restore that loads only the
 * prefix a prompt shares with a longer saved conversation, and the sorted index against a linear scan.
 */
class KvStoreTest {

    /** The state of position p is the token fed at p; saved as little-endian ints. */
    static final class StateBackend implements Backend {
        final int[] state = new int[8192];
        final AtomicInteger lastLoadN = new AtomicInteger(-1);
        final AtomicInteger lastLoadStored = new AtomicInteger(-1);

        @Override
        public String name() {
            return "state-fake";
        }

        @Override
        public int contextLength() {
            return state.length;
        }

        @Override
        public int maxBatch() {
            return 256;
        }

        @Override
        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            System.arraycopy(tokens, off, state, pos, n);
            if (logits != null) {
                Arrays.fill(logits, 0f);
                logits[(tokens[off + n - 1] * 31 + pos) % 1000 + 10] = 1f; // some token below the vocabulary size
            }
        }

        @Override
        public String stateFormat() {
            return "fake ints";
        }

        @Override
        public void saveState(int n, WritableByteChannel out) throws IOException {
            ByteBuffer b = ByteBuffer.allocate(4 * n).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            b.asIntBuffer().put(state, 0, n);
            while (b.hasRemaining()) {
                out.write(b);
            }
        }

        @Override
        public boolean canSaveWhileServing() {
            return true;
        }

        @Override
        public void saveStateWhileServing(int n, WritableByteChannel out) throws IOException {
            saveState(n, out);
        }

        @Override
        public void loadState(int n, ReadableByteChannel in) throws IOException {
            loadState(n, n, in);
        }

        @Override
        public void loadState(int n, int stored, ReadableByteChannel in) throws IOException {
            lastLoadN.set(n);
            lastLoadStored.set(stored);
            ByteBuffer b = ByteBuffer.allocate(4 * n).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            while (b.hasRemaining()) {
                if (in.read(b) < 0) {
                    throw new EOFException();
                }
            }
            b.flip();
            b.asIntBuffer().get(state, 0, n);
            Backend.skipState(in, 4L * (stored - n));
        }

        @Override
        public void copyStateFrom(Backend other, int n) {
            System.arraycopy(((StateBackend) other).state, 0, state, 0, n);
        }

        @Override
        public void close() {
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

    /** Runs a request on the model (pick, generate 4 tokens, record) and returns the slot's cached tokens after it. */
    private static int[] request(LoadedModel model, int[] prompt) {
        SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 4, List.of());
        try (LoadedModel.Slot s = model.pick(prompt)) {
            Engine.Result r = s.engine().generate(prompt, greedy, null, () -> false);
            model.record(s, r.stats());
            return s.engine().cachedTokens();
        }
    }

    /** Waits for the background saves to settle at {@code files} files. */
    private static void awaitFiles(KvStore store, int files) throws InterruptedException {
        long t0 = System.nanoTime();
        while (store.view().files() != files) {
            assertTrue(System.nanoTime() - t0 < 20_000_000_000L, "expected " + files + " saved files, have "
                    + store.view().files());
            Thread.sleep(10);
        }
    }

    @Test
    void saveSkipReplaceAndPartialRestore() throws Exception {
        Path tj = tokenizerJson();
        assumeTrue(tj != null, "models/Qwen_Qwen3-0.6B/tokenizer.json not found: skipped");
        Tokenizer tok = Tokenizer.load(tj);
        Architecture cfg = Architecture.parse(Json.parseObject("{\"model_type\":\"qwen3\",\"hidden_size\":64,"
                + "\"num_hidden_layers\":1,\"num_attention_heads\":2,\"intermediate_size\":128,\"vocab_size\":"
                + tok.vocabSize() + ",\"eos_token_id\":" + tok.tokenId("<|im_end|>") + "}"));
        Path dir = Files.createTempDirectory("kvstore-test");
        WebProperties props = new WebProperties("x", "", List.of(), 100, "y", "deepseek", true, dir.toString(), 1.0, 512, 0, List.of(), true, 1.0, 120);
        KvStore store = new KvStore(props);
        StateBackend backend = new StateBackend();
        Engine engine = new Engine(backend, tok, cfg, Set.of(), 0, ChatTemplate.forModelType("qwen3"));
        LoadedModel model = new LoadedModel("m", "test-source", new ModelLoader.Loaded(null, engine), List.of(),
                Map.of(), 0, store);
        Random r = new Random(3);
        try {
            int[] a = tokens(r, 1200);
            int[] cachedA = request(model, a);
            awaitFiles(store, 1);
            assertEquals(cachedA.length, store.view().saved().getFirst().tokens(), "the whole cache is saved");

            // the next turn adds 100 tokens: less than max(512, 1200 / 4) new, not saved again
            int[] a2 = concat(cachedA, tokens(r, 100));
            int[] cachedA2 = request(model, a2);
            Thread.sleep(200);
            assertEquals(1, store.view().files(), "a little growth is not saved");

            // 700 more: saved, and the old file of the conversation goes
            int[] a3 = concat(cachedA2, tokens(r, 700));
            int[] cachedA3 = request(model, a3);
            long t0 = System.nanoTime();
            while (store.view().files() != 1 || store.view().saved().getFirst().tokens() != cachedA3.length) {
                assertTrue(System.nanoTime() - t0 < 20_000_000_000L, "the grown conversation replaced its file: "
                        + store.view().saved());
                Thread.sleep(10);
            }

            // another conversation takes the slot (A is on disk already: nothing to save before), and is saved
            int[] b = tokens(r, 900);
            request(model, b);
            awaitFiles(store, 2);

            // a prompt sharing the first 1500 tokens with A: only those are loaded from A's file
            int[] c = concat(Arrays.copyOf(cachedA3, 1500), tokens(r, 300));
            try (LoadedModel.Slot s = model.pick(c)) {
                assertEquals(1500, s.engine().cachedLength(), "restored the shared prefix only");
                assertEquals(1500, backend.lastLoadN.get());
                assertEquals(cachedA3.length, backend.lastLoadStored.get(), "... of the longer saved conversation");
                assertArrayEquals(Arrays.copyOf(cachedA3, 1500), Arrays.copyOf(backend.state, 1500),
                        "the restored rows are the saved ones");
            }
        } finally {
            engine.close();
            store.clear();
            try (var w = Files.walk(dir)) {
                for (Path p : w.sorted(java.util.Comparator.reverseOrder()).toList()) {
                    Files.deleteIfExists(p);
                }
            }
        }
    }

    @Test
    void sortedIndexFindsTheLongestPrefix() {
        Random r = new Random(11);
        for (int round = 0; round < 200; round++) {
            TreeMap<int[], Integer> map = new TreeMap<>(Arrays::compare);
            List<int[]> lists = new java.util.ArrayList<>();
            int[] base = new int[40];
            for (int i = 0; i < base.length; i++) {
                base[i] = r.nextInt(3);
            }
            for (int k = 0; k < 1 + r.nextInt(12); k++) { // lists that share prefixes of various lengths
                int[] t = Arrays.copyOf(base, r.nextInt(base.length + 1) + r.nextInt(8));
                for (int i = r.nextInt(t.length + 1); i < t.length; i++) {
                    t[i] = r.nextInt(3);
                }
                map.put(t, k);
                lists.add(t);
            }
            int[] q = Arrays.copyOf(base, r.nextInt(base.length + 5));
            for (int i = r.nextInt(q.length + 1); i < q.length; i++) {
                q[i] = r.nextInt(3);
            }
            int best = 0;
            for (int[] t : map.keySet()) {
                best = Math.max(best, KvStore.commonPrefix(t, q));
            }
            Map.Entry<int[], Integer> got = KvStore.longestPrefix(map, q);
            assertNotNull(got);
            assertEquals(best, KvStore.commonPrefix(got.getKey(), q), "round " + round);
        }
    }
}
