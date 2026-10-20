package org.robomiri.jllm.web.chat;

import jakarta.servlet.http.HttpServletResponse;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.SamplingParams;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;

/**
 * What the OpenAI endpoints and the console's chat endpoint share: request validation, sampling parameters,
 * usage, errors and server-sent events. The semantics follow the engine's own server
 * ({@code org.robomiri.jllm.server.ApiServer}) field by field.
 */
public final class ChatSupport {

    private ChatSupport() {
    }

    /** A client error, turned into an OpenAI-style JSON error response. */
    public static final class RequestError extends RuntimeException {
        public final int status;
        public final String type;
        public final String code;

        public RequestError(int status, String message, String type, String code) {
            super(message);
            this.status = status;
            this.type = type;
            this.code = code;
        }

        public static RequestError invalid(String message) {
            return new RequestError(400, message, "invalid_request_error", null);
        }

        public static RequestError contextExceeded(String message) {
            return new RequestError(400, message, "invalid_request_error", "context_length_exceeded");
        }
    }

    /** Cancels generation once it has run longer than the limit (the clock starts when the engine starts). */
    public static final class Deadline implements BooleanSupplier {
        private final long limitNanos;
        private long start;
        private boolean started;
        private boolean expired;

        public Deadline(long limitMs) {
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

        /** The engine's "cancelled" is the OpenAI finish reason "length" when it was this deadline that fired. */
        public String finishReason(Engine.Result r) {
            return expired && r.finishReason().equals("cancelled") ? "length" : r.finishReason();
        }
    }

    /** Server-sent events written straight to the servlet response, flushed per event; started lazily. */
    public static final class Sse {
        private final HttpServletResponse res;
        private OutputStream out;
        private boolean started;

        public Sse(HttpServletResponse res) {
            this.res = res;
        }

        public boolean started() {
            return started;
        }

        public void start() throws IOException {
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

        public void event(Map<String, Object> e) throws IOException {
            raw("data: " + Json.write(e) + "\n\n");
        }

        public void raw(String s) throws IOException {
            start();
            out.write(s.getBytes(StandardCharsets.UTF_8));
            out.flush();
        }
    }

    public static void send(HttpServletResponse res, int status, String json) throws IOException {
        res.setStatus(status);
        res.setContentType("application/json");
        byte[] b = json.getBytes(StandardCharsets.UTF_8);
        res.setContentLength(b.length);
        res.getOutputStream().write(b);
    }

    public static String error(String msg, String type, String code) {
        Map<String, Object> e = Json.obj("message", msg, "type", type);
        if (code != null) {
            e.put("code", code);
        }
        return Json.write(Json.obj("error", e));
    }

    /** {@code application/json}, case-insensitively and with optional parameters such as {@code ; charset=utf-8}. */
    public static boolean isJson(String contentType) {
        if (contentType == null) {
            return false;
        }
        int semi = contentType.indexOf(';');
        return (semi >= 0 ? contentType.substring(0, semi) : contentType).trim().equalsIgnoreCase("application/json");
    }

    /** Rejects request features the engine cannot honour rather than silently ignoring them. */
    public static void checkUnsupported(Map<String, Object> b, boolean chat) {
        if (Json.num(b, "n", 1) != 1) {
            throw RequestError.invalid("'n' other than 1 is not supported");
        }
        Object tools = b.get("tools");
        if (tools != null && !chat) {
            throw RequestError.invalid("'tools' is only supported on /v1/chat/completions");
        }
        if (tools != null && !(tools instanceof List<?>)) {
            throw RequestError.invalid("'tools' must be an array");
        }
        if (tools instanceof List<?> l) {
            for (Object t : l) {
                if (!"function".equals(Json.str(t, "type", "function")) || Json.map(t, "function") == null
                        || Json.str(Json.map(t, "function"), "name", null) == null) {
                    throw RequestError.invalid("each tool must be {\"type\": \"function\", \"function\": {\"name\": ...}}");
                }
            }
        }
        Object choice = b.get("tool_choice");
        if (choice != null && !(choice instanceof String s && (s.equals("none") || s.equals("auto") || s.equals("required")))
                && !(choice instanceof Map<?, ?> && Json.str(Json.map(choice, "function"), "name", null) != null)) {
            throw RequestError.invalid("'tool_choice' must be \"none\", \"auto\", \"required\" or {\"type\": \"function\", \"function\": {\"name\": ...}}");
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

    public static Map<String, Object> usage(Engine.Stats s) {
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

    public static void checkPromptBytes(long bytes, long max) {
        if (bytes > max) {
            throw RequestError.contextExceeded("prompt is " + bytes + " bytes, the limit is " + max);
        }
    }

    public static long utf8Length(String s) {
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

    /** The prompt ids (with the family BOS) after checking that at least one token is left for the answer. */
    public static int[] checkContext(Engine engine, int[] prompt) {
        int[] p = engine.preparePrompt(prompt);
        int ctx = engine.backend().contextLength();
        if (p.length >= ctx) {
            throw RequestError.contextExceeded("prompt has " + p.length + " tokens, context is " + ctx
                    + " (at least one token must be left for the completion)");
        }
        return p;
    }

    /** @param room how many tokens the context has left after the prompt (at least 1) */
    public static SamplingParams params(Map<String, Object> b, SamplingParams d, int room) {
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

    /** Whether Qwen3 thinking is on: the request's {@code enable_thinking} (or chat_template_kwargs), else the default. */
    public static boolean thinking(Map<String, Object> body, boolean def) {
        boolean thinking = Json.bool(body, "enable_thinking", def);
        Map<String, Object> extra = Json.map(body, "chat_template_kwargs");
        return extra != null ? Json.bool(extra, "enable_thinking", thinking) : thinking;
    }

    /** Validates the messages array and its size in bytes (before tokenizing). */
    public static List<Object> messages(Map<String, Object> body, long maxPromptBytes) {
        List<Object> msgs = Json.list(body, "messages");
        if (msgs == null || msgs.isEmpty()) {
            throw new IllegalArgumentException("'messages' is required");
        }
        long bytes = 0;
        for (Object m : msgs) {
            if (!(m instanceof Map<?, ?> mm)) {
                throw RequestError.invalid("each message must be an object");
            }
            bytes += utf8Length(String.valueOf(mm.get("role"))) + utf8Length(org.robomiri.jllm.web.tools.ToolCallFormat.text(mm.get("content")));
            Object calls = mm.get("tool_calls");
            if (calls != null) {
                bytes += utf8Length(Json.write(calls));
            }
            checkPromptBytes(bytes, maxPromptBytes);
        }
        return msgs;
    }
}
