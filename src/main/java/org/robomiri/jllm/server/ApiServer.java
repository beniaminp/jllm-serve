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

/**
 * OpenAI-compatible endpoints on top of {@link Engine}:
 * {@code POST /v1/chat/completions}, {@code POST /v1/completions} (both with {@code "stream": true} SSE support),
 * {@code GET /v1/models}, {@code GET /health}, plus a tiny chat page at {@code /}.
 */
public final class ApiServer implements AutoCloseable {

    private final Engine engine;
    private final String modelId;
    private final HttpServer http;
    private final AtomicLong counter = new AtomicLong();
    private final SamplingParams defaults;
    private final boolean thinkingDefault;

    public ApiServer(Engine engine, String modelId, String host, int port) throws IOException {
        this(engine, modelId, host, port, SamplingParams.defaults(), false);
    }

    /**
     * @param defaults        sampling values used when a request omits a field
     * @param thinkingDefault whether Qwen3 thinking is on when a request does not say
     */
    public ApiServer(Engine engine, String modelId, String host, int port, SamplingParams defaults,
                     boolean thinkingDefault) throws IOException {
        this.engine = engine;
        this.modelId = modelId;
        this.defaults = defaults;
        this.thinkingDefault = thinkingDefault;
        this.http = new HttpServer(host, port, this::handle);
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
        if (req.method.equals("OPTIONS")) {
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
        Map<String, Object> body;
        try {
            body = Json.parseObject(req.bodyText());
        } catch (RuntimeException e) {
            res.send(400, "application/json", error("invalid JSON: " + e.getMessage()));
            return;
        }
        int[] prompt;
        try {
            prompt = chat ? chatPrompt(body) : engine.tokenizer().encode(Json.str(body, "prompt", ""));
        } catch (RuntimeException e) {
            res.send(400, "application/json", error(e.getMessage()));
            return;
        }
        SamplingParams params;
        try {
            params = params(body);
        } catch (RuntimeException e) {
            res.send(400, "application/json", error(e.getMessage()));
            return;
        }
        boolean stream = Json.bool(body, "stream", false);
        String id = (chat ? "chatcmpl-" : "cmpl-") + Long.toHexString(System.currentTimeMillis()) + "-" + counter.incrementAndGet();
        long created = System.currentTimeMillis() / 1000;
        String object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";

        if (!stream) {
            Engine.Result r = engine.generate(prompt, params, null, () -> false);
            Map<String, Object> choice = chat
                    ? Json.obj("index", 0, "message", Json.obj("role", "assistant", "content", r.text()), "finish_reason", r.finishReason())
                    : Json.obj("index", 0, "text", r.text(), "finish_reason", r.finishReason());
            res.send(200, "application/json", Json.write(Json.obj(
                    "id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(choice),
                    "usage", usage(r.stats()))));
            return;
        }

        res.startStream(200, "text/event-stream; charset=utf-8");
        if (chat) {
            sse(res, Json.obj("id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(Json.obj("index", 0, "delta", Json.obj("role", "assistant", "content", ""), "finish_reason", null))));
        }
        IOException[] failure = new IOException[1];
        Engine.Result r = engine.generate(prompt, params, text -> {
            if (failure[0] != null) {
                return;
            }
            Map<String, Object> choice = chat
                    ? Json.obj("index", 0, "delta", Json.obj("content", text), "finish_reason", null)
                    : Json.obj("index", 0, "text", text, "finish_reason", null);
            try {
                sse(res, Json.obj("id", id, "object", object, "created", created, "model", modelId, "choices", List.of(choice)));
            } catch (IOException e) {
                failure[0] = e; // client disconnected: stop generating
            }
        }, () -> failure[0] != null);
        if (failure[0] != null) {
            throw failure[0];
        }
        Map<String, Object> last = chat
                ? Json.obj("index", 0, "delta", Json.obj(), "finish_reason", r.finishReason())
                : Json.obj("index", 0, "text", "", "finish_reason", r.finishReason());
        sse(res, Json.obj("id", id, "object", object, "created", created, "model", modelId,
                "choices", List.of(last), "usage", usage(r.stats())));
        res.write("data: [DONE]\n\n");
        res.end();
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

    private int[] chatPrompt(Map<String, Object> body) {
        List<Object> msgs = Json.list(body, "messages");
        if (msgs == null || msgs.isEmpty()) {
            throw new IllegalArgumentException("'messages' is required");
        }
        List<ChatTemplate.Message> list = new ArrayList<>();
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
            list.add(new ChatTemplate.Message(role, text));
        }
        boolean thinking = Json.bool(body, "enable_thinking", thinkingDefault);
        Map<String, Object> extra = Json.map(body, "chat_template_kwargs");
        if (extra != null) {
            thinking = Json.bool(extra, "enable_thinking", thinking);
        }
        return ChatTemplate.renderTokens(engine.tokenizer(), list, true, thinking);
    }

    private SamplingParams params(Map<String, Object> b) {
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
        return new SamplingParams(
                (float) Json.dbl(b, "temperature", d.temperature()),
                (int) Json.num(b, "top_k", d.topK()),
                (float) Json.dbl(b, "top_p", d.topP()),
                (float) Json.dbl(b, "min_p", d.minP()),
                (float) Json.dbl(b, "repetition_penalty", d.repetitionPenalty()),
                (float) Json.dbl(b, "presence_penalty", d.presencePenalty()),
                (float) Json.dbl(b, "frequency_penalty", d.frequencyPenalty()),
                Json.num(b, "seed", -1),
                (int) Math.max(1, maxTokens),
                stop);
    }

    private static String error(String msg) {
        return Json.write(Json.obj("error", Json.obj("message", msg, "type", "invalid_request_error")));
    }

    @Override
    public void close() {
        http.close();
    }
}
