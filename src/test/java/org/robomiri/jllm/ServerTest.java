package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.server.ApiServer;
import org.robomiri.jllm.server.HttpServer;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Socket-level checks of {@link HttpServer} / {@link ApiServer} against a fake backend (instant, deterministic
 * tokens) and the real Qwen3 tokenizer. Dependency-free like {@link SmokeTest}; run with:
 * {@code java --add-modules jdk.incubator.vector -cp target/classes:target/test-classes org.robomiri.jllm.ServerTest}
 */
public final class ServerTest {

    /** Timeouts of the server under test (ms); every slow-client check is measured against these. */
    static final int READ_TIMEOUT = 1000;
    static final int WRITE_TIMEOUT = 1000;
    static final int QUEUE_TIMEOUT = 5000;

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "models/Qwen_Qwen3-0.6B");
        settings();
        Tokenizer tok = Tokenizer.load(dir.resolve("tokenizer.json"));
        try (Fixture f = new Fixture(tok, "qwen3", 1 << 20)) {
            basics(f);
            badRequests(f);
            keepAlive(f);
            stalledStreamClient(f);
            disconnectedClient(f, false);
            disconnectedClient(f, true);
            slowRequest(f);
            midStreamFailure(f);
        }
        // a tokenizer that adds BOS (as Llama/Mistral tokenizer configs do): the BOS rule is the tokenizer's, not the family's
        Tokenizer tokBos = Tokenizer.fromJson(Files.readString(dir.resolve("tokenizer.json")),
                "{\"add_bos_token\": true, \"bos_token\": \"<|im_start|>\"}");
        try (Fixture f = new Fixture(tokBos, "llama", 64)) {
            bosContextCheck(f);
        }
        System.out.println("all server tests passed");
    }

    // ------------------------------------------------------------------ checks

    /** L9: every typed setting is validated up front, with a message naming the key. */
    static void settings() throws Exception {
        Settings ok = Settings.load(new String[]{"serve", "--ctx", "2048", "--port=9000"}).validate();
        check(ok.getInt("backend.ctx") == 2048 && ok.getInt("server.port") == 9000, "settings: aliases and --key=value");
        for (String[] bad : new String[][]{{"--ctx", "abc"}, {"--ctx", "0"}, {"--backend", "gpu"}, {"--quant", "q3"},
                {"--backend.kv-dtype", "f64"}, {"--hub.timeout-ms", "0"}, {"--top-p", "2"}, {"--temp", "x"},
                {"--server.port", "70000"}}) {
            String[] args = {"serve", bad[0], bad[1]};
            String key = Settings.ALIASES.getOrDefault(bad[0].substring(2), bad[0].substring(2));
            try {
                Settings.load(args).validate();
                check(false, "settings: " + bad[0] + " " + bad[1] + " rejected");
            } catch (IllegalArgumentException e) {
                check(e.getMessage().contains(key.startsWith("sampling.") ? "sampling." : key),
                        "settings: " + bad[0] + " " + bad[1] + " rejected up front: " + e.getMessage());
            }
        }
        // L10: --key=value for values that start with --, boolean flags, strict booleans, CLI-only keys in a file
        Settings s = Settings.load(new String[]{"run", "--system=--weird", "--raw", "--prompt", "hi", "--think", "no"}).validate();
        check(s.get("chat.system").equals("--weird") && s.getBool("raw") && s.get("prompt").equals("hi") && !s.getBool("chat.thinking"),
                "settings: --key=--value, bare --raw before another flag, --think no");
        s = Settings.load(new String[]{"run", "--raw", "0", "--low-disk", "FALSE", "--think"}).validate();
        check(!s.getBool("raw") && !s.getBool("model.low-disk") && s.getBool("chat.thinking"),
                "settings: boolean flags consume a following boolean literal (0, FALSE) or stand alone");
        for (String[] args : new String[][]{{"serve", "--chat.thinking", "flase"}, {"serve", "--chat.thinking=flase"},
                {"run", "--raw", "tell me a story"}}) {
            try {
                Settings.load(args).validate();
                check(false, "settings: " + String.join(" ", args) + " rejected");
            } catch (IllegalArgumentException e) {
                check(e.getMessage().contains(args[args.length - 1].replace("--chat.thinking=", "")),
                        "settings: " + String.join(" ", args) + " rejected: " + e.getMessage());
            }
        }
        Path file = Files.createTempFile("jllm-test", ".properties");
        try {
            Files.writeString(file, "prompt=hello\n");
            try {
                Settings.load(new String[]{"run", "--config", file.toString()});
                check(false, "settings: prompt= in the properties file rejected");
            } catch (IllegalArgumentException e) {
                check(e.getMessage().contains("prompt"), "settings: prompt= in the properties file rejected: " + e.getMessage());
            }
            Files.writeString(file, "backend.ctx=512\n");
            check(Settings.load(new String[]{"run", "--config", file.toString()}).validate().getInt("backend.ctx") == 512,
                    "settings: the external file is still applied");
        } finally {
            Files.deleteIfExists(file);
        }
    }

    static void basics(Fixture f) throws Exception {
        Resp r = f.get("/health");
        check(r.status == 200 && r.body.contains("\"ok\""), "GET /health -> 200");
        r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5,\"temperature\":0}");
        Map<String, Object> j = Json.parseObject(r.body);
        check(r.status == 200 && Json.num(Json.map(j, "usage"), "completion_tokens", -1) == 5,
                "non-stream chat: 200 with 5 completion tokens");
        // L18: an empty prompt becomes the tokenizer's EOS token (deterministically)
        r = f.post("/v1/completions", "{\"prompt\":\"\",\"max_tokens\":2,\"temperature\":0}");
        int eos = f.tok.eosId() >= 0 ? f.tok.eosId() : f.tok.tokenId("<|im_end|>");
        check(r.status == 200 && Json.num(Json.map(Json.parseObject(r.body), "usage"), "prompt_tokens", -1) == 1
                && f.backend.firstToken == eos, "empty prompt: one EOS token (" + f.backend.firstToken + ") is fed");
        r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3,\"temperature\":0,\"stream\":true}");
        check(r.status == 200 && r.headers.getOrDefault("transfer-encoding", "").equals("chunked")
                && r.body.endsWith("data: [DONE]\n\n") && r.body.contains("\"finish_reason\":\"length\""),
                "stream chat: chunked SSE ending in [DONE]");
    }

    /** Request validation: each of these must be a 400 with a JSON error, never a 500 or a dropped connection. */
    static void badRequests(Fixture f) throws Exception {
        String msgs = "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]";
        Resp r = f.post("/v1/chat/completions", "{" + msgs + ",\"temperature\":1e400}");
        check(r.status == 400 && r.body.contains("finite"), "temperature 1e400 (infinite): 400");
        r = f.get("/nope");
        check(r.status == 404 && r.headers.get("content-type").equals("application/json") && r.body.contains("\"message\""),
                "unknown route: 404 JSON error");
        // L16: HttpServer's own fallback for a handler that does not answer uses the same JSON error shape
        try (HttpServer bare = new HttpServer("127.0.0.1", 0, (q, s) -> { })) {
            Thread.ofPlatform().daemon().start(bare::serve);
            try (Socket s = new Socket("127.0.0.1", bare.port())) {
                s.getOutputStream().write(request("GET", "/x", null));
                s.getOutputStream().flush();
                r = readResponse(s.getInputStream());
                check(r.status == 404 && r.headers.get("content-type").equals("application/json") && r.body.contains("not found: /x"),
                        "handler without a response: 404 JSON error");
            }
        }
        // L13: integer fields are range-checked before the int cast
        for (String topK : new String[]{"10000000000", "1.5", "-1", "\"5\""}) {
            r = f.post("/v1/chat/completions", "{" + msgs + ",\"top_k\":" + topK + "}");
            check(r.status == 400 && r.body.contains("top_k"), "top_k " + topK + ": 400");
        }
        // L3: a chunk size that overflows when added to what was already received
        String chunked = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                + "Transfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n7fffffffffffffff\r\n";
        r = f.exchange(chunked.getBytes(StandardCharsets.ISO_8859_1));
        check(r.status == 413, "chunk size 0x7fffffffffffffff after 5 bytes: 413 (got " + r.status + ")");
        // L4: one header line beyond the 64 KB limit
        r = f.exchange(request("GET", "/health", null, "X-Long: " + "y".repeat(70 * 1024)));
        check(r.status == 431, "70 KB header line: 431 (got " + r.status + ")");
        // L5: ambiguous framing
        r = f.exchange(request("POST", "/v1/chat/completions", "{}", "Content-Type: application/json", "Content-Length: 2"));
        check(r.status == 400 && r.body.contains("duplicate"), "two Content-Length headers: 400 (got " + r.status + ")");
        r = f.exchange(("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n"
                + "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n").getBytes(StandardCharsets.ISO_8859_1));
        check(r.status == 400 && r.body.contains("duplicate"), "two Transfer-Encoding headers: 400 (got " + r.status + ")");
        // L8: unsupported request features
        r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"},"
                + "{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}]}]}");
        check(r.status == 400 && r.body.contains("image_url"), "image content part: 400");
        for (String field : new String[]{"\"logit_bias\":{\"1\":5}", "\"echo\":true", "\"suffix\":\"x\"", "\"best_of\":2"}) {
            r = f.post("/v1/completions", "{\"prompt\":\"hi\"," + field + "}");
            check(r.status == 400 && r.body.contains("not supported"), field + ": 400");
        }
    }

    /** Two requests on one connection (the second must not be disturbed by the first's bookkeeping). */
    static void keepAlive(Fixture f) throws Exception {
        try (Socket s = f.connect()) {
            OutputStream out = s.getOutputStream();
            String body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"temperature\":0}";
            out.write(request("POST", "/v1/chat/completions", body, "Content-Type: application/json"));
            out.flush();
            Resp a = readResponse(s.getInputStream());
            out.write(request("GET", "/health", null));
            out.flush();
            Resp b = readResponse(s.getInputStream());
            check(a.status == 200 && b.status == 200 && a.headers.get("connection").equals("keep-alive"),
                    "keep-alive: second request on the same connection is served");
        }
    }

    /**
     * H1: a streaming client that stops reading must not hold the engine. The stream is queued for at most
     * the write timeout, then the connection is dropped and the generation cancelled; a second client is served
     * well within the engine's queue timeout.
     */
    static void stalledStreamClient(Fixture f) throws Exception {
        int before = f.backend.forwards.get();
        Socket stalled = new Socket();
        stalled.setReceiveBufferSize(8 * 1024); // small window: the server's buffers fill up sooner
        stalled.connect(new InetSocketAddress("127.0.0.1", f.server.port()));
        String body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1000000,"
                + "\"temperature\":0,\"stream\":true}";
        stalled.getOutputStream().write(request("POST", "/v1/chat/completions", body, "Content-Type: application/json"));
        stalled.getOutputStream().flush();
        InputStream in = stalled.getInputStream();
        readHeaders(in); // the stream has started; from here on the client never reads again
        long t0 = System.nanoTime();
        Resp r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"next\"}],\"max_tokens\":1,\"temperature\":0}");
        long ms = (System.nanoTime() - t0) / 1_000_000;
        check(r.status == 200, "stalled stream client: the next request is served (status " + r.status + ")");
        check(ms < QUEUE_TIMEOUT, "stalled stream client: served in " + ms + " ms (< engine queue timeout)");
        int afterFirst = f.backend.forwards.get();
        Thread.sleep(200);
        check(f.backend.forwards.get() == afterFirst, "stalled stream client: its generation was cancelled");
        check(afterFirst - before < 1_000_000, "stalled stream client: generation did not run to max_tokens");
        boolean closedByServer;
        try {
            byte[] buf = new byte[1 << 16];
            while (in.read(buf) >= 0) { // drain whatever was queued before the drop
                // keep reading
            }
            closedByServer = true;
        } catch (IOException e) {
            closedByServer = true; // reset by peer is also a close
        }
        check(closedByServer, "stalled stream client: server closed the connection");
        stalled.close();
    }

    /**
     * M1: a client that closes its connection mid-generation is noticed by the connection probe, for a
     * non-streaming request (nothing is written until the end) as well as a streaming one (before any write
     * fails), and the engine stops generating within a few tokens.
     */
    static void disconnectedClient(Fixture f, boolean stream) throws Exception {
        String what = (stream ? "stream" : "non-stream") + " client disconnect";
        f.backend.delayMs = 2; // ~2 ms per token: 100k tokens would take minutes
        try {
            Socket s = f.connect();
            String body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":100000,"
                    + "\"temperature\":0,\"stream\":" + stream + "}";
            s.getOutputStream().write(request("POST", "/v1/chat/completions", body, "Content-Type: application/json"));
            s.getOutputStream().flush();
            if (stream) {
                readHeaders(s.getInputStream());
            } else {
                Thread.sleep(150); // let the generation start
            }
            int atClose = f.backend.forwards.get();
            s.close();
            long t0 = System.nanoTime();
            Resp r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"next\"}],\"max_tokens\":1,\"temperature\":0}");
            long ms = (System.nanoTime() - t0) / 1_000_000;
            check(r.status == 200 && ms < 1500, what + ": the next request is served in " + ms + " ms");
            int settled = f.backend.forwards.get();
            Thread.sleep(200);
            check(f.backend.forwards.get() == settled, what + ": generation stopped");
            check(settled - atClose < 50, what + ": stopped within a few tokens (" + (settled - atClose) + " forwards after the close)");
        } finally {
            f.backend.delayMs = 0;
        }
    }

    /**
     * M2: a client that keeps a request alive by sending one header byte at a time (each within the per-read
     * timeout) gets a 408 once the request as a whole has taken longer than the read timeout; and a keep-alive
     * connection that goes quiet after a response is closed silently.
     */
    static void slowRequest(Fixture f) throws Exception {
        try (Socket s = f.connect()) {
            OutputStream out = s.getOutputStream();
            out.write("GET /health HTTP/1.1\r\nHost: x\r\nX-Slow: ".getBytes(StandardCharsets.ISO_8859_1));
            out.flush();
            long t0 = System.nanoTime();
            try {
                for (int i = 0; i < 10; i++) { // 3 s of trickling, one byte every 300 ms < READ_TIMEOUT
                    Thread.sleep(300);
                    out.write('x');
                    out.flush();
                }
            } catch (IOException e) {
                // the server has already answered and closed: expected
            }
            Resp r = readHeaders(s.getInputStream());
            long ms = (System.nanoTime() - t0) / 1_000_000;
            check(r.status == 408, "slow request: 408 after " + ms + " ms (per-read timeout never fired)");
        }
        try (Socket s = f.connect()) {
            s.getOutputStream().write(request("GET", "/health", null));
            s.getOutputStream().flush();
            check(readResponse(s.getInputStream()).status == 200, "idle keep-alive: first request served");
            long t0 = System.nanoTime();
            int c = s.getInputStream().read(); // nothing sent: the server closes after the idle timeout
            long ms = (System.nanoTime() - t0) / 1_000_000;
            check(c == -1 && ms >= READ_TIMEOUT - 50 && ms < 3 * READ_TIMEOUT,
                    "idle keep-alive: closed silently after " + ms + " ms");
        }
    }

    /** M4: a backend failure after the stream has started ends it with an in-band error event and no [DONE]. */
    static void midStreamFailure(Fixture f) throws Exception {
        f.backend.failAfter = f.backend.forwards.get() + 4; // prompt, then a few tokens, then the failure
        System.err.println("(a stack trace of the injected failure follows: expected)");
        try {
            Resp r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":50,\"temperature\":0,\"stream\":true}");
            String[] events = r.body.split("\n\n");
            String last = events[events.length - 1];
            check(r.status == 200 && last.startsWith("data: {\"error\":") && last.contains("\"server_error\"")
                    && !r.body.contains("[DONE]") && r.body.contains("\"content\":\" hello\""),
                    "mid-stream failure: tokens, then an error event, then a proper end without [DONE]");
        } finally {
            f.backend.failAfter = -1;
        }
        check(f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"temperature\":0}").status == 200,
                "mid-stream failure: the server keeps serving afterwards");
    }

    /**
     * M3: for a family with a prompt BOS token (here a llama config, ctx 64) the context check must count the
     * BOS the engine prepends: a raw prompt of ctx-1 tokens is a 400, not a 500 from the engine.
     */
    static void bosContextCheck(Fixture f) throws Exception {
        int ctx = f.backend.contextLength();
        Resp r = f.post("/v1/completions", "{\"prompt\":" + Json.write(promptOf(f.tok, ctx - 1)) + ",\"max_tokens\":1,\"temperature\":0}");
        check(r.status == 400 && r.body.contains("context_length_exceeded"),
                "prompt of ctx-1 tokens plus BOS: 400 context_length_exceeded (got " + r.status + ")");
        r = f.post("/v1/completions", "{\"prompt\":" + Json.write(promptOf(f.tok, ctx - 2)) + ",\"max_tokens\":5,\"temperature\":0}");
        Map<String, Object> usage = Json.map(Json.parseObject(r.body), "usage");
        check(r.status == 200 && Json.num(usage, "prompt_tokens", -1) == ctx - 1 && Json.num(usage, "completion_tokens", -1) == 1,
                "prompt of ctx-2 tokens plus BOS: served, one token of room");
    }

    /** Text that the tokenizer encodes to exactly {@code n} tokens. */
    static String promptOf(Tokenizer tok, int n) {
        StringBuilder sb = new StringBuilder("x");
        while (tok.encode(sb.toString()).length < n) {
            sb.append(" x");
        }
        check(tok.encode(sb.toString()).length == n, "built a " + n + "-token prompt");
        return sb.toString();
    }

    // ------------------------------------------------------------------ fixture: engine on a fake backend

    /** A backend that produces the same token every step, instantly (or after {@code delayMs}). */
    static final class FakeBackend implements Backend {
        final int ctx;
        final int token;
        final AtomicInteger forwards = new AtomicInteger();
        volatile long delayMs;
        volatile int failAfter = -1; // throw from forward() once this many calls happened (-1 = never)
        volatile int firstToken = -1; // the token at position 0 of the most recent prefill

        FakeBackend(int ctx, int token) {
            this.ctx = ctx;
            this.token = token;
        }

        @Override
        public String name() {
            return "fake";
        }

        @Override
        public int contextLength() {
            return ctx;
        }

        @Override
        public int maxBatch() {
            return 64;
        }

        @Override
        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            int calls = forwards.incrementAndGet();
            if (pos == 0) {
                firstToken = tokens[off];
            }
            if (failAfter >= 0 && calls > failAfter) {
                throw new IllegalStateException("fake backend failure");
            }
            if (delayMs > 0) {
                try {
                    Thread.sleep(delayMs);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            if (logits != null) {
                Arrays.fill(logits, 0f);
                logits[token] = 1f;
            }
        }

        @Override
        public void close() {
        }
    }

    static final class Fixture implements AutoCloseable {
        final Tokenizer tok;
        final FakeBackend backend;
        final Engine engine;
        final ApiServer server;
        final Thread serveThread;
        final Path tmp;

        Fixture(Tokenizer tok, String modelType, int ctx) throws Exception {
            this.tok = tok;
            tmp = Files.createTempDirectory("jllm-server-test");
            String bos = modelType.equals("qwen3") ? "" : ",\"bos_token_id\":" + tok.tokenId("<|im_start|>");
            Files.writeString(tmp.resolve("config.json"), "{\"model_type\":\"" + modelType + "\",\"hidden_size\":64,"
                    + "\"num_hidden_layers\":1,\"num_attention_heads\":2,\"intermediate_size\":128,\"vocab_size\":" + tok.vocabSize()
                    + ",\"eos_token_id\":" + tok.tokenId("<|im_end|>") + bos + "}");
            Architecture cfg = Architecture.load(tmp);
            backend = new FakeBackend(ctx, tok.encode(" hello")[0]);
            engine = new Engine(backend, tok, cfg, Set.of(tok.tokenId("<|im_end|>")), QUEUE_TIMEOUT,
                    ChatTemplate.forModelType("qwen3"));
            HttpServer.Options http = new HttpServer.Options(READ_TIMEOUT, WRITE_TIMEOUT, 16, 64 * 1024, null);
            server = new ApiServer(engine, "test-model", "127.0.0.1", 0,
                    new ApiServer.Options(SamplingParams.defaults(), false, 0, 0, http));
            serveThread = Thread.ofPlatform().daemon().start(server::serve);
        }

        Socket connect() throws IOException {
            Socket s = new Socket("127.0.0.1", server.port());
            s.setSoTimeout(30_000);
            return s;
        }

        Resp get(String path) throws IOException {
            return exchange(request("GET", path, null));
        }

        Resp post(String path, String json) throws IOException {
            return exchange(request("POST", path, json, "Content-Type: application/json"));
        }

        Resp exchange(byte[] raw) throws IOException {
            try (Socket s = connect()) {
                s.getOutputStream().write(raw);
                s.getOutputStream().flush();
                return readResponse(s.getInputStream());
            }
        }

        @Override
        public void close() throws Exception {
            server.close();
            engine.close();
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    // ------------------------------------------------------------------ a minimal HTTP/1.1 client

    record Resp(int status, Map<String, String> headers, String body) {
    }

    static byte[] request(String method, String path, String body, String... headers) {
        StringBuilder sb = new StringBuilder();
        sb.append(method).append(' ').append(path).append(" HTTP/1.1\r\nHost: localhost\r\n");
        for (String h : headers) {
            sb.append(h).append("\r\n");
        }
        byte[] b = body == null ? new byte[0] : body.getBytes(StandardCharsets.UTF_8);
        if (body != null) {
            sb.append("Content-Length: ").append(b.length).append("\r\n");
        }
        sb.append("\r\n");
        byte[] head = sb.toString().getBytes(StandardCharsets.ISO_8859_1);
        byte[] all = Arrays.copyOf(head, head.length + b.length);
        System.arraycopy(b, 0, all, head.length, b.length);
        return all;
    }

    /** Reads the status line and headers; returns the status with the lower-cased header names. */
    static Resp readHeaders(InputStream in) throws IOException {
        String status = readLine(in);
        if (status == null) {
            throw new IOException("connection closed before the status line");
        }
        int code = Integer.parseInt(status.split(" ")[1]);
        Map<String, String> headers = new LinkedHashMap<>();
        String line;
        while ((line = readLine(in)) != null && !line.isEmpty()) {
            int c = line.indexOf(':');
            headers.put(line.substring(0, c).trim().toLowerCase(), line.substring(c + 1).trim());
        }
        return new Resp(code, headers, "");
    }

    static Resp readResponse(InputStream in) throws IOException {
        Resp h = readHeaders(in);
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        if ("chunked".equals(h.headers.get("transfer-encoding"))) {
            while (true) {
                String size = readLine(in);
                int n = Integer.parseInt(size.trim(), 16);
                if (n == 0) {
                    readLine(in);
                    break;
                }
                body.write(in.readNBytes(n));
                readLine(in);
            }
        } else {
            int n = Integer.parseInt(h.headers.getOrDefault("content-length", "0"));
            body.write(in.readNBytes(n));
        }
        return new Resp(h.status, h.headers, body.toString(StandardCharsets.UTF_8));
    }

    static String readLine(InputStream in) throws IOException {
        StringBuilder sb = new StringBuilder();
        int c;
        while ((c = in.read()) >= 0) {
            if (c == '\n') {
                int n = sb.length();
                if (n > 0 && sb.charAt(n - 1) == '\r') {
                    sb.setLength(n - 1);
                }
                return sb.toString();
            }
            sb.append((char) c);
        }
        return sb.isEmpty() ? null : sb.toString();
    }

    static void check(boolean ok, String what) {
        if (!ok) {
            throw new AssertionError(what);
        }
        System.out.println("ok  " + what);
    }
}
