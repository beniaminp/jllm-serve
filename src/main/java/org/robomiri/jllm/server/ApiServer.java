package org.robomiri.jllm.server;

import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;

/**
 * OpenAI-compatible endpoints on top of {@link Engine}:
 * {@code POST /v1/chat/completions}, {@code POST /v1/completions} (both with {@code "stream": true} SSE support),
 * {@code GET /v1/models}, {@code GET /health}, plus a tiny chat page at {@code /}.
 */
public final class ApiServer implements AutoCloseable {

    /**
     * @param sampling        sampling values used when a request omits a field
     * @param thinkingDefault whether Qwen3 thinking is on when a request does not say
     * @param maxPromptBytes   reject prompts larger than this (text, before tokenizing); 0 = 16 * context length
     * @param requestTimeoutMs wall-clock limit for one request's generation (finish_reason "length" when hit); 0 = none
     * @param http             connection limits of the underlying HTTP server
     */
    public record Options(SamplingParams sampling, boolean thinkingDefault, long maxPromptBytes, long requestTimeoutMs,
                          HttpServer.Options http) {
        public static Options defaults() {
            return new Options(SamplingParams.defaults(), false, 0, 0, HttpServer.Options.defaults());
        }
    }

    /**
     * Cancels generation once the request has run for longer than the limit. The clock starts at the first
     * poll, which the engine makes right after acquiring its lock, so time spent queued does not count.
     */
    private static final class Deadline implements BooleanSupplier {
        private final long limitNanos;
        private long start;
        private boolean started;
        boolean expired;

        Deadline(long limitMs) {
            this.limitNanos = limitMs * 1_000_000L;
        }

        @Override
        public boolean getAsBoolean() {
            if (limitNanos <= 0) {
                return false;
            }
            long now = System.nanoTime();
            if (!started) {
                started = true;
                start = now;
                return false;
            }
            expired = now - start >= limitNanos;
            return expired;
        }

        /** Maps the engine's "cancelled" to the OpenAI finish reason "length" when it was this deadline that fired. */
        String finishReason(Engine.Result r) {
            return expired && r.finishReason().equals("cancelled") ? "length" : r.finishReason();
        }
    }

    /** A client error: carried out of the request parsing helpers and turned into a JSON error response. */
    private static final class RequestError extends RuntimeException {
        final int status;
        final String type;
        final String code;

        RequestError(int status, String message, String type, String code) {
            super(message);
            this.status = status;
            this.type = type;
            this.code = code;
        }

        static RequestError invalid(String message) {
            return new RequestError(400, message, "invalid_request_error", null);
        }
    }

    private final Engine engine;
    private final String modelId;
    private final HttpServer http;
    private final AtomicLong counter = new AtomicLong();
    private final SamplingParams defaults;
    private final boolean thinkingDefault;
    private final long maxPromptBytes;
    private final long requestTimeoutMs;

    public ApiServer(Engine engine, String modelId, String host, int port) throws IOException {
        this(engine, modelId, host, port, Options.defaults());
    }

    public ApiServer(Engine engine, String modelId, String host, int port, SamplingParams defaults,
                     boolean thinkingDefault) throws IOException {
        this(engine, modelId, host, port, new Options(defaults, thinkingDefault, 0, 0, HttpServer.Options.defaults()));
    }

    public ApiServer(Engine engine, String modelId, String host, int port, Options options) throws IOException {
        this.engine = engine;
        this.modelId = modelId;
        this.defaults = options.sampling();
        this.thinkingDefault = options.thinkingDefault();
        this.maxPromptBytes = options.maxPromptBytes() > 0
                ? options.maxPromptBytes() : 16L * engine.backend().contextLength();
        this.requestTimeoutMs = options.requestTimeoutMs();
        this.http = new HttpServer(host, port, options.http(), this::handle);
    }

    public int port() {
        return http.port();
    }

    public void serve() {
        http.serve();
    }

    private void handle(HttpServer.Request req, HttpServer.Response res) throws Exception {
        String path = req.path;
        int qm = path.indexOf('?');
        if (qm >= 0) {
            path = path.substring(0, qm);
        }
        if (req.method.equals("OPTIONS")) { // CORS preflight; the headers only make sense when an origin is configured
            if (res.cors()) {
                res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                        .header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                        .header("Access-Control-Max-Age", "86400");
            }
            res.send(204, "text/plain", "");
            return;
        }
        switch (path) {
            case "/", "/index.html" -> res.send(200, "text/html; charset=utf-8", WebUi.HTML);
            case "/health" -> res.send(200, "application/json", Json.write(Json.obj("status", "ok", "model", modelId)));
            case "/v1/models" -> res.send(200, "application/json", Json.write(Json.obj(
                    "object", "list",
                    "data", List.of(Json.obj("id", modelId, "object", "model", "owned_by", "jllm-serve")))));
            case "/v1/chat/completions" -> completion(req, res, true);
            case "/v1/completions" -> completion(req, res, false);
            default -> res.send(404, "application/json", error("not found: " + path));
        }
    }

    private void completion(HttpServer.Request req, HttpServer.Response res, boolean chat) throws Exception {
        if (!req.method.equals("POST")) {
            res.send(405, "application/json", error("use POST"));
            return;
        }
        if (!isJson(req.headers.get("content-type"))) {
            res.send(415, "application/json", error("Content-Type must be application/json"));
            return;
        }
        Map<String, Object> body;
        try {
            body = Json.parseObject(req.bodyText());
        } catch (RuntimeException e) {
            res.send(400, "application/json", error("invalid JSON: " + e.getMessage()));
            return;
        }
        int[] prompt;
        SamplingParams params;
        try {
            checkUnsupported(body, chat);
            prompt = engine.preparePrompt(chat ? chatPrompt(body) : rawPrompt(body)); // with the family BOS, if any
            int ctx = engine.backend().contextLength();
            if (prompt.length >= ctx) {
                throw new RequestError(400, "prompt has " + prompt.length + " tokens, context is " + ctx
                        + " (at least one token must be left for the completion)",
                        "invalid_request_error", "context_length_exceeded");
            }
            params = params(body, ctx - prompt.length);
        } catch (RequestError e) {
            res.send(e.status, "application/json", error(e));
            return;
        } catch (RuntimeException e) {
            res.send(400, "application/json", error(e.getMessage()));
            return;
        }
        boolean stream = Json.bool(body, "stream", false);
        // OpenAI: with stream_options the usage goes into a separate final chunk (only when include_usage is true);
        // without it we keep the legacy inline usage on the last choice chunk
        Map<String, Object> streamOptions = Json.map(body, "stream_options");
        boolean separateUsage = streamOptions != null;
        boolean includeUsage = streamOptions == null || Json.bool(streamOptions, "include_usage", false);
        String id = (chat ? "chatcmpl-" : "cmpl-") + Long.toHexString(System.currentTimeMillis()) + "-" + counter.incrementAndGet();
        long created = System.currentTimeMillis() / 1000;
        String object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";

        Deadline deadline = new Deadline(requestTimeoutMs);
        if (!stream) {
            Engine.Result r;
            try {
                r = engine.generate(prompt, params, null, () -> req.clientGone() || deadline.getAsBoolean());
            } catch (Engine.OverloadedException e) {
                res.send(503, "application/json", error(e.getMessage(), "server_error", "overloaded"));
                return;
            } catch (IllegalArgumentException e) { // the engine's own context check (should not get past ours)
                res.send(400, "application/json", error(e.getMessage(), "invalid_request_error", "context_length_exceeded"));
                return;
            }
            String finish = deadline.finishReason(r);
            Map<String, Object> choice = chat
                    ? Json.obj("index", 0, "message", Json.obj("role", "assistant", "content", r.text()), "finish_reason", finish)
                    : Json.obj("index", 0, "text", r.text(), "finish_reason", finish);
            res.send(200, "application/json", Json.write(Json.obj(
                    "id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(choice),
                    "usage", usage(r.stats()))));
            return;
        }

        // The stream headers are sent lazily, at the first token: until then a queue timeout can still be a 503.
        IOException[] failure = new IOException[1];
        Engine.Result r;
        try {
            r = engine.generate(prompt, params, text -> {
                if (failure[0] != null) {
                    return;
                }
                Map<String, Object> choice = chat
                        ? Json.obj("index", 0, "delta", Json.obj("content", text), "finish_reason", null)
                        : Json.obj("index", 0, "text", text, "finish_reason", null);
                try {
                    startStream(res, chat, id, object, created);
                    sse(res, Json.obj("id", id, "object", object, "created", created, "model", modelId, "choices", List.of(choice)));
                } catch (IOException e) {
                    failure[0] = e; // client disconnected: stop generating
                }
            }, () -> failure[0] != null || req.clientGone() || deadline.getAsBoolean());
        } catch (Engine.OverloadedException e) {
            if (res.started()) {
                throw e;
            }
            res.send(503, "application/json", error(e.getMessage(), "server_error", "overloaded"));
            return;
        } catch (IllegalArgumentException e) {
            if (res.started()) {
                throw e;
            }
            res.send(400, "application/json", error(e.getMessage(), "invalid_request_error", "context_length_exceeded"));
            return;
        } catch (RuntimeException e) {
            if (res.started()) {
                // The status line is long gone: tell the client in-band (OpenAI SDKs raise on an "error" event),
                // then end the stream without [DONE]. HttpServer logs the exception and drops the connection.
                sse(res, Json.obj("error", Json.obj("message", "internal server error", "type", "server_error")));
                res.end();
            }
            throw e;
        }
        if (failure[0] != null) {
            throw failure[0];
        }
        if (req.clientGone()) {
            throw new IOException("client disconnected");
        }
        startStream(res, chat, id, object, created);
        String finish = deadline.finishReason(r);
        Map<String, Object> last = chat
                ? Json.obj("index", 0, "delta", Json.obj(), "finish_reason", finish)
                : Json.obj("index", 0, "text", "", "finish_reason", finish);
        Map<String, Object> event = Json.obj("id", id, "object", object, "created", created, "model", modelId,
                "choices", List.of(last));
        if (includeUsage && !separateUsage) {
            event.put("usage", usage(r.stats()));
        }
        sse(res, event);
        if (includeUsage && separateUsage) {
            sse(res, Json.obj("id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(), "usage", usage(r.stats())));
        }
        res.write("data: [DONE]\n\n");
        res.end();
    }

    /** Rejects request features this server cannot honour rather than silently ignoring them. */
    private static void checkUnsupported(Map<String, Object> b, boolean chat) {
        if (Json.num(b, "n", 1) != 1) {
            throw RequestError.invalid("'n' other than 1 is not supported");
        }
        Object tools = b.get("tools");
        if (tools instanceof List<?> l && !l.isEmpty()) {
            throw RequestError.invalid("'tools' is not supported");
        }
        Object toolChoice = b.get("tool_choice");
        if (toolChoice != null && !"none".equals(toolChoice)) {
            throw RequestError.invalid("'tool_choice' is not supported");
        }
        Map<String, Object> format = Json.map(b, "response_format");
        if (format != null && !"text".equals(Json.str(format, "type", "text"))) {
            throw RequestError.invalid("'response_format' other than text is not supported");
        }
        Object logprobs = b.get("logprobs"); // boolean on chat, an integer (top n) on completions
        if (Boolean.TRUE.equals(logprobs) || logprobs instanceof Number n && n.intValue() != 0) {
            throw RequestError.invalid("'logprobs' is not supported");
        }
        if (!chat && b.get("prompt") instanceof List<?> l) {
            if (l.size() != 1 || !(l.getFirst() instanceof String)) {
                throw RequestError.invalid("'prompt' must be a string or a single-element array of strings"
                        + " (batches and token-id arrays are not supported)");
            }
        }
    }

    /** Sends the SSE headers (and, for chat, the initial role event) unless already done. */
    private void startStream(HttpServer.Response res, boolean chat, String id, String object, long created) throws IOException {
        if (res.started()) {
            return;
        }
        res.startStream(200, "text/event-stream; charset=utf-8");
        if (chat) {
            sse(res, Json.obj("id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(Json.obj("index", 0, "delta", Json.obj("role", "assistant", "content", ""), "finish_reason", null))));
        }
    }

    /** {@code application/json}, case-insensitively and with optional parameters such as {@code ; charset=utf-8}. */
    private static boolean isJson(String contentType) {
        if (contentType == null) {
            return false;
        }
        int semi = contentType.indexOf(';');
        String media = (semi >= 0 ? contentType.substring(0, semi) : contentType).trim();
        return media.equalsIgnoreCase("application/json");
    }

    private static void sse(HttpServer.Response res, Map<String, Object> event) throws IOException {
        res.write("data: " + Json.write(event) + "\n\n");
    }

    private static Map<String, Object> usage(Engine.Stats s) {
        return Json.obj(
                "prompt_tokens", s.promptTokens(),
                "completion_tokens", s.generatedTokens(),
                "total_tokens", s.promptTokens() + s.generatedTokens(),
                "cached_tokens", s.cachedTokens(),
                "prefill_ms", Math.round(s.prefillMs()),
                "decode_ms", Math.round(s.decodeMs()),
                "tokens_per_second", Math.round(s.tokPerSec() * 10) / 10.0);
    }

    private int[] rawPrompt(Map<String, Object> body) {
        Object p = body.get("prompt");
        if (p instanceof List<?> l) { // a single-element string array (validated in checkUnsupported)
            p = l.getFirst();
        }
        String text = p instanceof String str ? str : "";
        checkPromptBytes(utf8Length(text));
        return engine.tokenizer().encode(text);
    }

    /** Enforces the byte limit on the raw text so that oversized requests never reach the tokenizer. */
    private void checkPromptBytes(long bytes) {
        if (bytes > maxPromptBytes) {
            throw new RequestError(400, "prompt is " + bytes + " bytes, the limit is " + maxPromptBytes,
                    "invalid_request_error", "context_length_exceeded");
        }
    }

    private static long utf8Length(String s) {
        long n = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c < 0x80) {
                n += 1;
            } else if (c < 0x800) {
                n += 2;
            } else if (Character.isHighSurrogate(c) && i + 1 < s.length() && Character.isLowSurrogate(s.charAt(i + 1))) {
                n += 4;
                i++;
            } else {
                n += 3;
            }
        }
        return n;
    }

    private int[] chatPrompt(Map<String, Object> body) {
        List<Object> msgs = Json.list(body, "messages");
        if (msgs == null || msgs.isEmpty()) {
            throw new IllegalArgumentException("'messages' is required");
        }
        List<ChatTemplate.Message> list = new ArrayList<>();
        long bytes = 0;
        for (Object m : msgs) {
            String role = Json.str(m, "role", "user");
            Object content = m instanceof Map<?, ?> mm ? mm.get("content") : null;
            String text;
            if (content instanceof String s) {
                text = s;
            } else if (content instanceof List<?> parts) { // OpenAI multi-part content: concatenate text parts
                StringBuilder sb = new StringBuilder();
                for (Object p : parts) {
                    if ("text".equals(Json.str(p, "type", "text"))) {
                        sb.append(Json.str(p, "text", ""));
                    }
                }
                text = sb.toString();
            } else {
                text = "";
            }
            bytes += utf8Length(role) + utf8Length(text);
            checkPromptBytes(bytes);
            list.add(new ChatTemplate.Message(role, text));
        }
        boolean thinking = Json.bool(body, "enable_thinking", thinkingDefault);
        Map<String, Object> extra = Json.map(body, "chat_template_kwargs");
        if (extra != null) {
            thinking = Json.bool(extra, "enable_thinking", thinking);
        }
        return engine.renderChat(list, thinking);
    }

    /** @param room how many tokens the context has left after the prompt (at least 1) */
    private SamplingParams params(Map<String, Object> b, int room) {
        SamplingParams d = defaults;
        List<String> stop = new ArrayList<>();
        Object s = b.get("stop");
        if (s instanceof String str) {
            stop.add(str);
        } else if (s instanceof List<?> l) {
            for (Object o : l) {
                stop.add(String.valueOf(o));
            }
        }
        long maxTokens = Json.num(b, "max_completion_tokens", Json.num(b, "max_tokens", d.maxTokens()));
        if (maxTokens <= 0) {
            throw RequestError.invalid("max_tokens must be at least 1");
        }
        maxTokens = Math.min(maxTokens, room); // the engine stops at the context edge anyway; this also avoids int overflow
        return new SamplingParams(
                (float) Json.dbl(b, "temperature", d.temperature()),
                (int) Json.num(b, "top_k", d.topK()),
                (float) Json.dbl(b, "top_p", d.topP()),
                (float) Json.dbl(b, "min_p", d.minP()),
                (float) Json.dbl(b, "repetition_penalty", d.repetitionPenalty()),
                (float) Json.dbl(b, "presence_penalty", d.presencePenalty()),
                (float) Json.dbl(b, "frequency_penalty", d.frequencyPenalty()),
                Json.num(b, "seed", d.seed()),
                (int) maxTokens,
                stop);
    }

    private static String error(String msg) {
        return error(msg, "invalid_request_error", null);
    }

    private static String error(RequestError e) {
        return error(e.getMessage(), e.type, e.code);
    }

    private static String error(String msg, String type, String code) {
        Map<String, Object> e = Json.obj("message", msg, "type", type);
        if (code != null) {
            e.put("code", code);
        }
        return Json.write(Json.obj("error", e));
    }

    @Override
    public void close() {
        http.close();
    }
}
