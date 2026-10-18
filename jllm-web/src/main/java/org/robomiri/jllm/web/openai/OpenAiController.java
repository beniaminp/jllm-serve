package org.robomiri.jllm.web.openai;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.web.models.LoadedModel;
import org.robomiri.jllm.web.models.ModelManager;
import org.robomiri.jllm.web.settings.SettingsService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;

/**
 * The OpenAI-compatible API over the loaded models: {@code GET /v1/models}, {@code POST /v1/chat/completions} and
 * {@code POST /v1/completions}, both with {@code "stream": true} (server-sent events). The request handling follows
 * the engine's own server ({@code org.robomiri.jllm.server.ApiServer}) field by field, so a client sees the same
 * behaviour from {@code jllm serve} and from the web console; the {@code model} field picks the loaded model.
 */
@RestController
public class OpenAiController {

    /** A client error, turned into an OpenAI-style JSON error response. */
    static final class RequestError extends RuntimeException {
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

    /** Cancels generation once it has run longer than the limit (the clock starts when the engine starts). */
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

        String finishReason(Engine.Result r) {
            return expired && r.finishReason().equals("cancelled") ? "length" : r.finishReason();
        }
    }

    private final ModelManager models;
    private final SettingsService settings;
    private final AtomicLong counter = new AtomicLong();
    private final AtomicBoolean thinkingNoted = new AtomicBoolean();

    public OpenAiController(ModelManager models, SettingsService settings) {
        this.models = models;
        this.settings = settings;
    }

    @GetMapping(value = "/v1/models", produces = "application/json")
    public String listModels() {
        List<Object> data = new ArrayList<>();
        for (LoadedModel m : models.loaded()) {
            data.add(modelObject(m));
        }
        return Json.write(Json.obj("object", "list", "data", data));
    }

    @GetMapping(value = "/v1/models/{*id}", produces = "application/json")
    public void getModel(@PathVariable String id, HttpServletResponse res) throws IOException {
        String name = id.startsWith("/") ? id.substring(1) : id;
        for (LoadedModel m : models.loaded()) {
            if (m.id().equals(name)) {
                send(res, 200, Json.write(modelObject(m)));
                return;
            }
        }
        send(res, 404, error("model '" + name + "' is not loaded", "invalid_request_error", "model_not_found"));
    }

    private static Map<String, Object> modelObject(LoadedModel m) {
        LoadedModel.View v = m.view();
        return Json.obj("id", v.id(), "object", "model", "created", v.loadedAt() / 1000, "owned_by", "jllm-serve",
                "context_length", v.contextLength(), "backend", v.backend(), "quant", v.quant());
    }

    @PostMapping("/v1/chat/completions")
    public void chatCompletions(@RequestBody(required = false) String body, HttpServletRequest req,
                                HttpServletResponse res) throws IOException {
        completion(body, req, res, true);
    }

    @PostMapping("/v1/completions")
    public void completions(@RequestBody(required = false) String body, HttpServletRequest req,
                            HttpServletResponse res) throws IOException {
        completion(body, req, res, false);
    }

    private void completion(String text, HttpServletRequest req, HttpServletResponse res, boolean chat) throws IOException {
        if (!isJson(req.getContentType())) {
            send(res, 415, error("Content-Type must be application/json", "invalid_request_error", null));
            return;
        }
        Map<String, Object> body;
        try {
            body = Json.parseObject(text == null ? "" : text);
        } catch (RuntimeException e) {
            send(res, 400, error("invalid JSON: " + e.getMessage(), "invalid_request_error", null));
            return;
        }
        LoadedModel model;
        try {
            model = models.resolve(Json.str(body, "model", null));
        } catch (NoSuchElementException e) {
            send(res, 404, error(e.getMessage(), "invalid_request_error", "model_not_found"));
            return;
        }
        try (LoadedModel.Lease lease = model.lease()) {
            completion(body, model, res, chat);
        } catch (LoadedModel.UnavailableException e) {
            send(res, 503, error(e.getMessage(), "server_error", "model_unavailable"));
        }
    }

    private void completion(Map<String, Object> body, LoadedModel model, HttpServletResponse res, boolean chat) throws IOException {
        Engine engine = model.engine();
        Settings s = settings.current();
        String modelId = model.id();
        int[] prompt;
        SamplingParams params;
        try {
            checkUnsupported(body, chat);
            long maxPromptBytes = s.getLong("engine.max-prompt-bytes") > 0
                    ? s.getLong("engine.max-prompt-bytes") : 16L * engine.backend().contextLength();
            prompt = engine.preparePrompt(chat ? chatPrompt(engine, body, s.getBool("chat.thinking"), maxPromptBytes)
                    : rawPrompt(engine, body, maxPromptBytes));
            int ctx = engine.backend().contextLength();
            if (prompt.length >= ctx) {
                throw new RequestError(400, "prompt has " + prompt.length + " tokens, context is " + ctx
                        + " (at least one token must be left for the completion)",
                        "invalid_request_error", "context_length_exceeded");
            }
            params = params(body, s.sampling(), ctx - prompt.length);
        } catch (RequestError e) {
            send(res, e.status, error(e.getMessage(), e.type, e.code));
            return;
        } catch (RuntimeException e) {
            send(res, 400, error(e.getMessage(), "invalid_request_error", null));
            return;
        }
        boolean stream = Json.bool(body, "stream", false);
        Map<String, Object> streamOptions = Json.map(body, "stream_options");
        boolean separateUsage = streamOptions != null;
        boolean includeUsage = streamOptions == null || Json.bool(streamOptions, "include_usage", false);
        String id = (chat ? "chatcmpl-" : "cmpl-") + Long.toHexString(System.currentTimeMillis()) + "-" + counter.incrementAndGet();
        long created = System.currentTimeMillis() / 1000;
        String object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";
        Deadline deadline = new Deadline(s.getLong("server.request-timeout-ms"));

        if (!stream) {
            Engine.Result r;
            try {
                r = engine.generate(prompt, params, null, () -> model.closing() || deadline.getAsBoolean());
            } catch (Engine.OverloadedException e) {
                send(res, 503, error(e.getMessage(), "server_error", "overloaded"));
                return;
            } catch (IllegalArgumentException e) {
                send(res, 400, error(e.getMessage(), "invalid_request_error", "context_length_exceeded"));
                return;
            }
            model.record(r.stats());
            String finish = deadline.finishReason(r);
            Map<String, Object> choice = chat
                    ? Json.obj("index", 0, "message", Json.obj("role", "assistant", "content", r.text()), "finish_reason", finish)
                    : Json.obj("index", 0, "text", r.text(), "finish_reason", finish);
            send(res, 200, Json.write(Json.obj("id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(choice), "usage", usage(r.stats()))));
            return;
        }

        // the stream starts at the first token: until then a queue timeout can still be a 503
        Sse sse = new Sse(res);
        IOException[] failure = new IOException[1];
        Engine.Result r;
        try {
            r = engine.generate(prompt, params, piece -> {
                if (failure[0] != null) {
                    return;
                }
                Map<String, Object> choice = chat
                        ? Json.obj("index", 0, "delta", Json.obj("content", piece), "finish_reason", null)
                        : Json.obj("index", 0, "text", piece, "finish_reason", null);
                try {
                    startStream(sse, chat, id, object, created, modelId);
                    sse.event(Json.obj("id", id, "object", object, "created", created, "model", modelId, "choices", List.of(choice)));
                } catch (IOException e) {
                    failure[0] = e; // the client went away: stop generating
                }
            }, () -> failure[0] != null || model.closing() || deadline.getAsBoolean());
        } catch (RuntimeException e) {
            if (sse.started) {
                sse.event(Json.obj("error", Json.obj("message", "internal server error", "type", "server_error")));
                throw e;
            }
            if (e instanceof Engine.OverloadedException) {
                send(res, 503, error(e.getMessage(), "server_error", "overloaded"));
            } else if (e instanceof IllegalArgumentException) {
                send(res, 400, error(e.getMessage(), "invalid_request_error", "context_length_exceeded"));
            } else {
                throw e;
            }
            return;
        }
        model.record(r.stats());
        if (failure[0] != null) {
            return; // nothing more can be sent
        }
        startStream(sse, chat, id, object, created, modelId);
        String finish = deadline.finishReason(r);
        Map<String, Object> last = chat
                ? Json.obj("index", 0, "delta", Json.obj(), "finish_reason", finish)
                : Json.obj("index", 0, "text", "", "finish_reason", finish);
        Map<String, Object> event = Json.obj("id", id, "object", object, "created", created, "model", modelId,
                "choices", List.of(last));
        if (includeUsage && !separateUsage) {
            event.put("usage", usage(r.stats()));
        }
        sse.event(event);
        if (includeUsage && separateUsage) {
            sse.event(Json.obj("id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(), "usage", usage(r.stats())));
        }
        sse.raw("data: [DONE]\n\n");
    }

    /** Server-sent events written straight to the servlet response, flushed per event. */
    private static final class Sse {
        private final HttpServletResponse res;
        private OutputStream out;
        boolean started;

        Sse(HttpServletResponse res) {
            this.res = res;
        }

        void start() throws IOException {
            if (started) {
                return;
            }
            res.setStatus(200);
            res.setContentType("text/event-stream;charset=UTF-8");
            res.setHeader("Cache-Control", "no-cache");
            res.setHeader("X-Accel-Buffering", "no"); // nginx: do not buffer the stream
            out = res.getOutputStream();
            started = true;
        }

        void event(Map<String, Object> e) throws IOException {
            raw("data: " + Json.write(e) + "\n\n");
        }

        void raw(String s) throws IOException {
            start();
            out.write(s.getBytes(StandardCharsets.UTF_8));
            out.flush();
        }
    }

    private static void startStream(Sse sse, boolean chat, String id, String object, long created, String modelId) throws IOException {
        if (sse.started) {
            return;
        }
        sse.start();
        if (chat) {
            sse.event(Json.obj("id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(Json.obj("index", 0, "delta", Json.obj("role", "assistant", "content", ""), "finish_reason", null))));
        }
    }

    /** {@code application/json}, case-insensitively and with optional parameters such as {@code ; charset=utf-8}. */
    private static boolean isJson(String contentType) {
        if (contentType == null) {
            return false;
        }
        int semi = contentType.indexOf(';');
        return (semi >= 0 ? contentType.substring(0, semi) : contentType).trim().equalsIgnoreCase("application/json");
    }

    static void send(HttpServletResponse res, int status, String json) throws IOException {
        res.setStatus(status);
        res.setContentType("application/json");
        byte[] b = json.getBytes(StandardCharsets.UTF_8);
        res.setContentLength(b.length);
        res.getOutputStream().write(b);
    }

    static String error(String msg, String type, String code) {
        Map<String, Object> e = Json.obj("message", msg, "type", type);
        if (code != null) {
            e.put("code", code);
        }
        return Json.write(Json.obj("error", e));
    }

    /** Rejects request features the engine cannot honour rather than silently ignoring them. */
    static void checkUnsupported(Map<String, Object> b, boolean chat) {
        if (Json.num(b, "n", 1) != 1) {
            throw RequestError.invalid("'n' other than 1 is not supported");
        }
        if (b.get("tools") instanceof List<?> l && !l.isEmpty()) {
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
        Object logprobs = b.get("logprobs");
        if (Boolean.TRUE.equals(logprobs) || logprobs instanceof Number n && n.intValue() != 0) {
            throw RequestError.invalid("'logprobs' is not supported");
        }
        Map<String, Object> bias = Json.map(b, "logit_bias");
        if (bias != null && !bias.isEmpty()) {
            throw RequestError.invalid("'logit_bias' is not supported");
        }
        if (Boolean.TRUE.equals(b.get("echo"))) {
            throw RequestError.invalid("'echo' is not supported");
        }
        if (b.get("suffix") != null) {
            throw RequestError.invalid("'suffix' is not supported");
        }
        if (Json.num(b, "best_of", 1) != 1) {
            throw RequestError.invalid("'best_of' other than 1 is not supported");
        }
        if (!chat && b.get("prompt") instanceof List<?> l && (l.size() != 1 || !(l.getFirst() instanceof String))) {
            throw RequestError.invalid("'prompt' must be a string or a single-element array of strings"
                    + " (batches and token-id arrays are not supported)");
        }
    }

    static Map<String, Object> usage(Engine.Stats s) {
        return Json.obj(
                "prompt_tokens", s.promptTokens(),
                "completion_tokens", s.generatedTokens(),
                "total_tokens", s.promptTokens() + s.generatedTokens(),
                "cached_tokens", s.cachedTokens(),
                "prefill_ms", Math.round(s.prefillMs()),
                "decode_ms", Math.round(s.decodeMs()),
                "drafted_tokens", s.draftedTokens(),
                "accepted_draft_tokens", s.acceptedTokens(),
                "tokens_per_second", Math.round(s.tokPerSec() * 10) / 10.0);
    }

    private static int[] rawPrompt(Engine engine, Map<String, Object> body, long maxPromptBytes) {
        Object p = body.get("prompt");
        if (p instanceof List<?> l) {
            p = l.getFirst();
        }
        String text = p instanceof String str ? str : "";
        checkPromptBytes(utf8Length(text), maxPromptBytes);
        return engine.tokenizer().encode(text);
    }

    private static void checkPromptBytes(long bytes, long max) {
        if (bytes > max) {
            throw new RequestError(400, "prompt is " + bytes + " bytes, the limit is " + max,
                    "invalid_request_error", "context_length_exceeded");
        }
    }

    static long utf8Length(String s) {
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

    private int[] chatPrompt(Engine engine, Map<String, Object> body, boolean thinkingDefault, long maxPromptBytes) {
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
            if (content instanceof String str) {
                text = str;
            } else if (content instanceof List<?> parts) { // multi-part content: text parts only
                StringBuilder sb = new StringBuilder();
                for (Object p : parts) {
                    String type = p instanceof Map<?, ?> ? Json.str(p, "type", "text") : "?";
                    if (!type.equals("text")) {
                        throw RequestError.invalid("message content parts must be of type \"text\" (got \"" + type
                                + "\"); images and audio are not supported");
                    }
                    sb.append(Json.str(p, "text", ""));
                }
                text = sb.toString();
            } else {
                text = "";
            }
            bytes += utf8Length(role) + utf8Length(text);
            checkPromptBytes(bytes, maxPromptBytes);
            list.add(new ChatTemplate.Message(role, text));
        }
        boolean thinking = Json.bool(body, "enable_thinking", thinkingDefault);
        Map<String, Object> extra = Json.map(body, "chat_template_kwargs");
        if (extra != null) {
            thinking = Json.bool(extra, "enable_thinking", thinking);
        }
        boolean given = body.containsKey("enable_thinking") || (extra != null && extra.containsKey("enable_thinking"));
        String template = engine.template().name();
        if (given && !template.equals("chatml-qwen3") && thinkingNoted.compareAndSet(false, true)) {
            System.err.println("note: enable_thinking only applies to the Qwen3 chat template; this model uses \""
                    + template + "\", the field is ignored (reported once)");
        }
        return engine.renderChat(list, thinking);
    }

    static SamplingParams params(Map<String, Object> b, SamplingParams d, int room) {
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
        maxTokens = Math.min(maxTokens, room);
        return new SamplingParams(
                (float) Json.dbl(b, "temperature", d.temperature()),
                intField(b, "top_k", d.topK()),
                (float) Json.dbl(b, "top_p", d.topP()),
                (float) Json.dbl(b, "min_p", d.minP()),
                (float) Json.dbl(b, "repetition_penalty", d.repetitionPenalty()),
                (float) Json.dbl(b, "presence_penalty", d.presencePenalty()),
                (float) Json.dbl(b, "frequency_penalty", d.frequencyPenalty()),
                Json.num(b, "seed", d.seed()),
                (int) maxTokens,
                stop);
    }

    private static int intField(Map<String, Object> b, String key, int def) {
        Object v = b.get(key);
        if (v == null) {
            return def;
        }
        if (!(v instanceof Number n) || (v instanceof Double x && x != Math.rint(x))) {
            throw RequestError.invalid("'" + key + "' must be an integer");
        }
        double x = n.doubleValue();
        if (x < 0 || x > Integer.MAX_VALUE) {
            throw RequestError.invalid("'" + key + "' must be between 0 and " + Integer.MAX_VALUE);
        }
        return (int) x;
    }
}
