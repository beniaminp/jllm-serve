package org.robomiri.jllm;

import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.ModelConfig;
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
        Tokenizer tok = Tokenizer.load(dir.resolve("tokenizer.json"));
        try (Fixture f = new Fixture(tok, "qwen3", 1 << 20)) {
            basics(f);
            keepAlive(f);
            stalledStreamClient(f);
        }
        System.out.println("all server tests passed");
    }

    // ------------------------------------------------------------------ checks

    static void basics(Fixture f) throws Exception {
        Resp r = f.get("/health");
        check(r.status == 200 && r.body.contains("\"ok\""), "GET /health -> 200");
        r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5,\"temperature\":0}");
        Map<String, Object> j = Json.parseObject(r.body);
        check(r.status == 200 && Json.num(Json.map(j, "usage"), "completion_tokens", -1) == 5,
                "non-stream chat: 200 with 5 completion tokens");
        r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3,\"temperature\":0,\"stream\":true}");
        check(r.status == 200 && r.headers.getOrDefault("transfer-encoding", "").equals("chunked")
                && r.body.endsWith("data: [DONE]\n\n") && r.body.contains("\"finish_reason\":\"length\""),
                "stream chat: chunked SSE ending in [DONE]");
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

    // ------------------------------------------------------------------ fixture: engine on a fake backend

    /** A backend that produces the same token every step, instantly (or after {@code delayMs}). */
    static final class FakeBackend implements Backend {
        final int ctx;
        final int token;
        final AtomicInteger forwards = new AtomicInteger();
        volatile long delayMs;
        volatile int failAfter = -1; // throw from forward() once this many calls happened (-1 = never)

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
            ModelConfig cfg = ModelConfig.load(tmp);
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
