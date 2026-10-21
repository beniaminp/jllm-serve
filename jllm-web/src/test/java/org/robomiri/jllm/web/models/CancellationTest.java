package org.robomiri.jllm.web.models;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A client that goes away stops its generation: over a real socket (Tomcat on a random port), a long generation on
 * a fake backend (each forward call sleeps) is abandoned by closing the connection, during the prefill or during
 * decoding, streaming or not; the slot must be free again within about one forward call, and the backend must not be
 * fed another token afterwards.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, properties = {
        "jllm.web.settings-file=target/test-settings-cancel/jllm-web.properties",
        "jllm.web.tools-file=target/test-settings-cancel/jllm-web-tools.json",
        "jllm.web.kv-cache-dir=target/test-settings-cancel/kv-cache"})
class CancellationTest {

    /** Sleeps {@code delayMs} per forward call and always predicts {@code token}. */
    static final class SlowBackend implements Backend {
        final AtomicInteger fed = new AtomicInteger();
        final AtomicInteger calls = new AtomicInteger();
        final int token;
        volatile long delayMs;

        SlowBackend(int token, long delayMs) {
            this.token = token;
            this.delayMs = delayMs;
        }

        @Override
        public String name() {
            return "slow-fake";
        }

        @Override
        public int contextLength() {
            return 16384;
        }

        @Override
        public int maxBatch() {
            return 64;
        }

        @Override
        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            calls.incrementAndGet();
            try {
                Thread.sleep(delayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            fed.addAndGet(n);
            if (logits != null) {
                Arrays.fill(logits, 0f);
                logits[token] = 1f;
            }
        }

        @Override
        public void close() {
        }
    }

    @Autowired
    ModelManager models;

    @LocalServerPort
    int port;

    Tokenizer tok;
    SlowBackend backend;
    Engine engine;

    @BeforeEach
    void setUp() throws Exception {
        Path tj = tokenizerJson();
        assumeTrue(tj != null, "models/Qwen_Qwen3-0.6B/tokenizer.json not found: skipped");
        tok = Tokenizer.load(tj);
        Architecture cfg = Architecture.parse(Json.parseObject("{\"model_type\":\"qwen3\",\"hidden_size\":64,"
                + "\"num_hidden_layers\":1,\"num_attention_heads\":2,\"intermediate_size\":128,\"vocab_size\":"
                + tok.vocabSize() + ",\"eos_token_id\":" + tok.tokenId("<|im_end|>") + "}"));
        backend = new SlowBackend(tok.encode(" hello")[0], 20);
        engine = new Engine(backend, tok, cfg, Set.of(tok.tokenId("<|im_end|>")), 0, ChatTemplate.forModelType("qwen3"));
        models.registerForTest("slow", engine);
    }

    @AfterEach
    void tearDown() {
        if (engine != null) {
            models.removeForTest("slow");
            engine.close();
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

    @Test
    void nonStreamingDuringDecode() throws Exception {
        abandon(false, 300, 20, 2000);
    }

    @Test
    void nonStreamingDuringPrefill() throws Exception {
        abandon(false, 3000, 150, 50); // ~47 chunks of 150 ms: the client leaves half-way through the prefill
    }

    @Test
    void streamingDuringDecode() throws Exception {
        abandon(true, 300, 20, 2000);
    }

    @Test
    void streamingDuringPrefill() throws Exception {
        abandon(true, 3000, 150, 50);
    }

    /**
     * Sends a chat request of about {@code promptWords} words, waits until the backend is busy with it (half of the
     * prefill done when {@code maxTokens} is small, else 30 tokens into decoding), closes the connection and checks
     * that the slot is free soon and nothing more is fed.
     */
    private void abandon(boolean stream, int promptWords, long delayMs, int maxTokens) throws Exception {
        backend.delayMs = delayMs;
        String body = Json.write(Json.obj("model", "slow", "stream", stream, "max_tokens", maxTokens, "temperature", 0,
                "messages", java.util.List.of(Map.of("role", "user", "content", "word ".repeat(promptWords)))));
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        int promptTokens = tok.encode("word ".repeat(promptWords)).length;
        LoadedModel model = models.resolve("slow");
        long tLeave;
        try (Socket s = new Socket("127.0.0.1", port)) {
            OutputStream out = s.getOutputStream();
            out.write(("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                    + "Content-Length: " + bytes.length + "\r\n\r\n").getBytes(StandardCharsets.US_ASCII));
            out.write(bytes);
            out.flush();
            if (stream) { // the headers come at once (keep-alive comments follow during the prefill)
                InputStream in = s.getInputStream();
                s.setSoTimeout(10_000);
                assertTrue(in.read() >= 0, "the stream started");
            }
            int target = maxTokens < 100 ? promptTokens / 2 : promptTokens + 30;
            long t0 = System.nanoTime();
            while (backend.fed.get() < target) {
                assertTrue(System.nanoTime() - t0 < 60_000_000_000L, "the generation got going");
                Thread.sleep(5);
            }
            tLeave = System.nanoTime();
        } // the client goes away
        long t0 = System.nanoTime();
        while (model.busySlots() > 0 || engine.busy()) {
            assertTrue(System.nanoTime() - t0 < 10_000_000_000L, "the slot is freed");
            Thread.sleep(5);
        }
        double freedMs = (System.nanoTime() - tLeave) / 1e6;
        int fedAtStop = backend.fed.get();
        Thread.sleep(3 * delayMs + 200);
        assertEquals(fedAtStop, backend.fed.get(), "nothing is fed after the slot is free");
        int generated = Math.max(0, fedAtStop - promptTokens);
        System.err.printf("cancellation (%s, %s): slot free %.0f ms after the client left (one forward call: %d ms);"
                        + " fed %d of %d prompt tokens, %d generated of max %d%n", stream ? "stream" : "non-stream",
                maxTokens < 100 ? "prefill" : "decode", freedMs, delayMs, Math.min(fedAtStop, promptTokens), promptTokens,
                generated, maxTokens);
        // one forward call in flight, the probe interval, and scheduling slack on a busy machine
        assertTrue(freedMs < 2 * delayMs + 1000, "freed within about one forward call: " + freedMs + " ms");
        if (maxTokens < 100) {
            assertTrue(fedAtStop < promptTokens, "the prefill stopped half-way: " + fedAtStop + " of " + promptTokens);
        } else {
            assertTrue(generated < maxTokens / 4, "decoding stopped: " + generated + " of " + maxTokens);
        }
    }
}
