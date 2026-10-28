package org.robomiri.jllm.server;

import org.robomiri.jllm.engine.Admission;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.Priority;
import org.robomiri.jllm.engine.SlotPool;
import org.robomiri.jllm.engine.TokenLogprob;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

/**
 * OpenAI-compatible endpoints on top of {@link Engine}:
 * {@code POST /v1/chat/completions}, {@code POST /v1/completions} (both with {@code "stream": true} SSE support),
 * {@code GET /v1/models}, {@code GET /health}, plus a tiny chat page at {@code /}.
 *
 * <p>The text model is a {@link SlotPool}: with one slot (the default) requests take turns on its engine; with
 * {@code server.slots} of them (docs/10 §10.9c) each request picks a slot, so several users are served at once, their
 * decode steps sharing forward passes ({@code engine.batch-decode}) and their prompts the paged cache's blocks.
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

    private final SlotPool slots;
    /** The first slot's engine: tokenizer, template, context and vision encoder (the same for every slot); null without a text model. */
    private final Engine engine;
    private final String modelId;
    private volatile AudioRoutes audio; // null: no /v1/audio endpoints
    private final HttpServer http;
    private final AtomicLong counter = new AtomicLong();
    private final AtomicBoolean thinkingNoted = new AtomicBoolean();
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
        this(engine == null ? null : new SlotPool(List.of(engine), null, null, modelId), options, modelId, host, port);
    }

    /**
     * A server whose requests run on the slots of {@code slots} ({@code server.slots}, docs/10 §10.9c): each request takes
     * the slot {@link SlotPool#pick} chooses for its prompt. Null: no text model, images / video / audio only.
     */
    public static ApiServer onSlots(SlotPool slots, String modelId, String host, int port, Options options) throws IOException {
        return new ApiServer(slots, options, modelId, host, port);
    }

    private ApiServer(SlotPool slots, Options options, String modelId, String host, int port) throws IOException {
        this.slots = slots;
        this.engine = slots == null ? null : slots.first();
        this.modelId = modelId;
        this.defaults = options.sampling();
        this.thinkingDefault = options.thinkingDefault();
        this.maxPromptBytes = options.maxPromptBytes() > 0
                ? options.maxPromptBytes() : (engine == null ? 0 : 16L * engine.backend().contextLength());
        this.requestTimeoutMs = options.requestTimeoutMs();
        this.http = new HttpServer(host, port, options.http(), this::handle);
    }

    public int port() {
        return http.port();
    }

    private volatile ImageHttp images;

    /** Adds the image endpoints ({@code /v1/images/*}) served by {@code images}; the text engine may be null (images only). */
    public ApiServer withImages(ImageHttp images) {
        this.images = images;
        return this;
    }

    private volatile VideoHttp video;

    /** Adds the video endpoints ({@code /v1/videos/*}) served by {@code video}; the text engine may be null (video only). */
    public ApiServer withVideo(VideoHttp video) {
        this.video = video;
        return this;
    }

    // ---- pictures in chat messages (docs/20)
    private long imageMaxBytes = 32L << 20;
    private long imageMaxPixels = org.robomiri.jllm.vision.ImageDecoder.MAX_PIXELS;
    private boolean imageAllowPrivate;

    /** Limits for the pictures of a request: file size, decoded pixels, and whether http(s) URLs on private networks may be fetched. */
    public ApiServer imageLimits(long maxBytes, long maxPixels, boolean allowPrivateUrls) {
        this.imageMaxBytes = maxBytes;
        this.imageMaxPixels = maxPixels;
        this.imageAllowPrivate = allowPrivateUrls;
        return this;
    }

    /** Offers the OpenAI audio endpoints of {@code audio} (speech-to-text / text-to-speech models) as well. */
    public void setAudio(AudioRoutes audio) {
        this.audio = audio;
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
                        .header("Access-Control-Allow-Headers", "Content-Type, Authorization, " + OpenAiFeatures.PRIORITY_HEADER)
                        .header("Access-Control-Max-Age", "86400");
            }
            res.send(204, "text/plain", "");
            return;
        }
        ImageHttp img = images;
        if (img != null && img.handle(path, req, res)) {
            return;
        }
        VideoHttp vid = video;
        if (vid != null && vid.handle(path, req, res)) {
            return;
        }
        AudioRoutes ar = audio;
        if (ar != null && ar.handle(path, req, res)) {
            return;
        }
        switch (path) {
            case "/", "/index.html" -> res.send(200, "text/html; charset=utf-8", WebUi.HTML);
            case "/health" -> res.send(200, "application/json", Json.write(slots == null ? Json.obj("status", "ok", "model", modelId)
                    : Json.obj("status", "ok", "model", modelId, "slots", slots.size(), "busy_slots", slots.busy(),
                    "queued", slots.admission() == null ? 0 : slots.admission().queued())));
            case "/v1/models" -> {
                List<Object> data = new ArrayList<>();
                if (engine != null) {
                    data.add(Json.obj("id", modelId, "object", "model", "owned_by", "jllm-serve", "kind", "text"));
                }
                if (img != null) {
                    data.add(img.modelObject());
                }
                if (vid != null) {
                    data.add(vid.modelObject());
                }
                if (ar != null) {
                    data.addAll(ar.modelObjects());
                }
                res.send(200, "application/json", Json.write(Json.obj("object", "list", "data", data)));
            }
            case "/v1/chat/completions" -> completion(req, res, true);
            case "/v1/completions" -> completion(req, res, false);
            default -> res.send(404, "application/json", error("not found: " + path));
        }
    }

    private void completion(HttpServer.Request req, HttpServer.Response res, boolean chat) throws Exception {
        if (engine == null) {
            res.send(404, "application/json", error("no text model is loaded (this server was started with --model none)"));
            return;
        }
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
        OpenAiFeatures.headerPriority(body, req.headers.get(OpenAiFeatures.PRIORITY_HEADER.toLowerCase(java.util.Locale.ROOT)));
        Tokenizer tok = engine.tokenizer();
        int[] prompt;
        org.robomiri.jllm.engine.MediaPrompt media = null;
        SamplingParams params;
        OpenAiFeatures.Options o;
        int echoFrom; // the prompt tokens an echo returns start here (after a BOS the engine put in front)
        String echoText = "";
        try {
            checkUnsupported(body, chat);
            o = OpenAiFeatures.parse(body, chat, tok.vocabSize());
            int[] ids;
            List<org.robomiri.jllm.vision.Rgb> images = List.of();
            if (chat) {
                List<Object> raw = Json.list(body, "messages");
                if (raw != null && org.robomiri.jllm.vision.ImageParts.hasImages(raw)) {
                    if (engine.vision() == null) {
                        throw RequestError.invalid("this model cannot see images: it was loaded without a vision projector"
                                + " (mmproj); see model.mmproj");
                    }
                    org.robomiri.jllm.vision.ImageParts.Extracted ex = org.robomiri.jllm.vision.ImageParts.extract(raw,
                            new org.robomiri.jllm.vision.ImageFetcher(imageAllowPrivate, imageMaxBytes), imageMaxBytes, imageMaxPixels);
                    body = new java.util.LinkedHashMap<>(body);
                    body.put("messages", ex.messages());
                    images = ex.images();
                }
                ids = chatPrompt(body, o.json());
            } else if (o.suffix() != null) {
                checkPromptBytes(utf8Length(promptText(body)) + utf8Length(o.suffix()));
                ids = OpenAiFeatures.fimPrompt(tok, promptText(body), o.suffix());
            } else {
                ids = rawPrompt(body);
            }
            int[] prepared = engine.preparePrompt(ids); // with the family BOS, if any
            if (!images.isEmpty()) {
                media = engine.vision().build(prepared, images);
            }
            prompt = media != null ? media.identity() : prepared;
            echoFrom = prompt.length - ids.length;
            if (o.echo()) {
                echoText = promptText(body);
            }
            int ctx = engine.backend().contextLength();
            if (prompt.length >= ctx) {
                throw new RequestError(400, "prompt has " + prompt.length + " tokens, context is " + ctx
                        + " (at least one token must be left for the completion)",
                        "invalid_request_error", "context_length_exceeded");
            }
            params = params(body, ctx - prompt.length, o.echo());
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
        Engine.Decoding decoding = o.decoding(tok);
        org.robomiri.jllm.engine.MediaPrompt mediaFinal = media;
        try (SlotPool.Slot slot = pick(prompt, Json.str(body, "user", null), decoding.priority(), req, res)) {
            if (slot == null) {
                return;
            }
            generate(slot.engine(), req, res, chat, stream, o, prompt, mediaFinal, params, decoding, echoFrom, echoText,
                    separateUsage, includeUsage, id, created, object);
        }
    }

    /**
     * The slot that serves a request: {@link SlotPool#pick} in the request's priority class, which orders the mixed
     * scheduler's queue; a client that leaves while queued there ends the request like a disconnect, and one queued for
     * longer than {@code server.queue-timeout-ms} gets the 503 a busy engine gives (null: answered).
     */
    private SlotPool.Slot pick(int[] prompt, String session, Priority priority, HttpServer.Request req, HttpServer.Response res)
            throws IOException {
        try (Priority.Scope ignored = Priority.enter(priority)) {
            return slots.pick(prompt, session, req::clientGone, engine.queueTimeoutMs());
        } catch (Admission.Abandoned e) {
            throw new IOException("client disconnected while queued for a slot");
        } catch (Engine.OverloadedException e) {
            res.send(503, "application/json", error(e.getMessage(), "server_error", "overloaded"));
            return null;
        }
    }

    /** Runs a parsed request on {@code engine} (its slot's) and writes the response. */
    private void generate(Engine engine, HttpServer.Request req, HttpServer.Response res, boolean chat, boolean stream,
                          OpenAiFeatures.Options o, int[] prompt, org.robomiri.jllm.engine.MediaPrompt mediaFinal,
                          SamplingParams params, Engine.Decoding decoding, int echoFrom, String echoText,
                          boolean separateUsage, boolean includeUsage, String id, long created, String object) throws Exception {
        Tokenizer tok = engine.tokenizer();
        Deadline deadline = new Deadline(requestTimeoutMs);
        if (!stream) {
            List<Engine.Result> all;
            try {
                all = mediaFinal == null
                        ? engine.generate(prompt, params, decoding, o.count(), null,
                        () -> req.clientGone() || deadline.getAsBoolean())
                        : engine.generate(mediaFinal, params, decoding, o.count(), null,
                        () -> req.clientGone() || deadline.getAsBoolean());
            } catch (Engine.OverloadedException e) {
                res.send(503, "application/json", error(e.getMessage(), "server_error", "overloaded"));
                return;
            } catch (IllegalArgumentException e) { // the engine's own context check (should not get past ours)
                res.send(400, "application/json", error(e.getMessage(), "invalid_request_error", "context_length_exceeded"));
                return;
            }
            List<Object> choices = new ArrayList<>();
            List<Engine.Result> chosen = OpenAiFeatures.select(all, o);
            for (int i = 0; i < chosen.size(); i++) {
                Engine.Result r = chosen.get(i);
                Map<String, Object> choice = chat
                        ? Json.obj("index", i, "message", Json.obj("role", "assistant", "content", r.text()))
                        : Json.obj("index", i, "text", echoText + r.text());
                if (o.logprobs()) {
                    choice.put("logprobs", OpenAiFeatures.choiceLogprobs(chat, r, o, prompt, echoFrom, tok));
                }
                choice.put("finish_reason", deadline.finishReason(r));
                choices.add(choice);
            }
            res.send(200, "application/json", Json.write(Json.obj(
                    "id", id, "object", object, "created", created, "model", modelId,
                    "choices", choices,
                    "usage", usage(OpenAiFeatures.total(all)))));
            return;
        }

        // The stream headers are sent lazily, once the engine has the request: until then a queue timeout can still
        // be a 503. The choices run one after the other; every chunk carries its choice's index.
        IOException[] failure = new IOException[1];
        int count = o.count(); // (best_of is not allowed with stream)
        boolean[] begun = new boolean[count];
        List<List<TokenLogprob>> pending = new ArrayList<>();
        int[][] offsets = new int[count][1];
        for (int i = 0; i < count; i++) {
            pending.add(new ArrayList<>());
        }
        List<Engine.Result> finished = new ArrayList<>();
        AtomicReference<List<TokenLogprob>> promptLps = new AtomicReference<>();
        String echoed = echoText;
        Engine.ChoiceSink sink = new Engine.ChoiceSink() {
            /** Starts the stream and the choice: the role (chat) or the echoed prompt (completions). */
            private void begin(int i) throws IOException {
                startStream(res);
                if (begun[i]) {
                    return;
                }
                begun[i] = true;
                if (chat) {
                    sse(res, chunk(id, object, created, Json.obj("index", i,
                            "delta", Json.obj("role", "assistant", "content", ""), "finish_reason", null)));
                } else if (o.echo()) {
                    Map<String, Object> c = Json.obj("index", i, "text", echoed);
                    if (o.logprobs()) {
                        c.put("logprobs", OpenAiFeatures.echoLogprobs(prompt, echoFrom, promptLps.get(), tok, offsets[i]));
                    }
                    c.put("finish_reason", null);
                    sse(res, chunk(id, object, created, c));
                }
            }

            private void send(int i, String text, String finish, Engine.Result last) {
                if (failure[0] != null) {
                    return;
                }
                try {
                    begin(i);
                    Map<String, Object> c = chat
                            ? Json.obj("index", i, "delta", text == null ? Json.obj() : Json.obj("content", text))
                            : Json.obj("index", i, "text", text == null ? "" : text);
                    if (o.logprobs()) {
                        c.put("logprobs", OpenAiFeatures.drainLogprobs(chat, pending.get(i), tok, offsets[i]));
                    }
                    c.put("finish_reason", finish);
                    Map<String, Object> event = chunk(id, object, created, c);
                    if (last != null && i == count - 1 && includeUsage && !separateUsage) {
                        event.put("usage", usage(OpenAiFeatures.total(finished)));
                    }
                    sse(res, event);
                } catch (IOException e) {
                    failure[0] = e; // client disconnected: stop generating
                }
            }

            @Override
            public void prefilled(int i, List<TokenLogprob> lps) {
                if (i == 0) {
                    promptLps.set(lps);
                }
                if (failure[0] == null && o.echo()) {
                    try {
                        begin(i);
                    } catch (IOException e) {
                        failure[0] = e;
                    }
                }
            }

            @Override
            public void text(int i, String text) {
                send(i, text, null, null);
            }

            @Override
            public void token(int i, TokenLogprob lp) {
                pending.get(i).add(lp);
            }

            @Override
            public void done(int i, Engine.Result r) {
                finished.add(r);
                if (!req.clientGone()) {
                    send(i, null, deadline.finishReason(r), r);
                }
            }
        };
        try {
            if (mediaFinal == null) {
                engine.generate(prompt, params, decoding, count, sink,
                        () -> failure[0] != null || req.clientGone() || deadline.getAsBoolean());
            } else {
                engine.generate(mediaFinal, params, decoding, count, sink,
                        () -> failure[0] != null || req.clientGone() || deadline.getAsBoolean());
            }
        } catch (RuntimeException e) {
            if (res.started()) {
                // The status line is long gone: tell the client in-band (OpenAI SDKs raise on an "error" event),
                // then end the stream without [DONE]. HttpServer logs the exception and drops the connection.
                sse(res, Json.obj("error", Json.obj("message", "internal server error", "type", "server_error")));
                res.end();
                throw e;
            }
            // nothing was sent yet: the engine refused the request before producing a token
            if (e instanceof Engine.OverloadedException) {
                res.send(503, "application/json", error(e.getMessage(), "server_error", "overloaded"));
            } else if (e instanceof IllegalArgumentException) {
                res.send(400, "application/json", error(e.getMessage(), "invalid_request_error", "context_length_exceeded"));
            } else {
                throw e;
            }
            return;
        }
        if (failure[0] != null) {
            throw failure[0];
        }
        if (req.clientGone()) {
            throw new IOException("client disconnected");
        }
        startStream(res);
        if (includeUsage && separateUsage) {
            sse(res, Json.obj("id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(), "usage", usage(OpenAiFeatures.total(finished))));
        }
        res.write("data: [DONE]\n\n");
        res.end();
    }

    private Map<String, Object> chunk(String id, String object, long created, Map<String, Object> choice) {
        return Json.obj("id", id, "object", object, "created", created, "model", modelId, "choices", List.of(choice));
    }

    /** Rejects request features this server cannot honour rather than silently ignoring them. */
    private static void checkUnsupported(Map<String, Object> b, boolean chat) {
        Object tools = b.get("tools");
        if (tools instanceof List<?> l && !l.isEmpty()) {
            throw RequestError.invalid("'tools' is not supported");
        }
        Object toolChoice = b.get("tool_choice");
        if (toolChoice != null && !"none".equals(toolChoice)) {
            throw RequestError.invalid("'tool_choice' is not supported");
        }
        if (!chat && b.get("prompt") instanceof List<?> l) {
            if (l.size() != 1 || !(l.getFirst() instanceof String)) {
                throw RequestError.invalid("'prompt' must be a string or a single-element array of strings"
                        + " (batches and token-id arrays are not supported)");
            }
        }
    }

    /** Sends the SSE headers unless already done. */
    private static void startStream(HttpServer.Response res) throws IOException {
        if (!res.started()) {
            res.startStream(200, "text/event-stream; charset=utf-8");
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
                "prompt_tokens_details", Json.obj("cached_tokens", s.cachedTokens()), // where OpenAI clients read the cache hits
                "prefill_ms", Math.round(s.prefillMs()),
                "decode_ms", Math.round(s.decodeMs()),
                "drafted_tokens", s.draftedTokens(),
                "accepted_draft_tokens", s.acceptedTokens(),
                "forced_tokens", s.forcedTokens(),
                "tokens_per_second", Math.round(s.tokPerSec() * 10) / 10.0);
    }

    /** The completions prompt: a string, or a single-element string array (validated in checkUnsupported). */
    private static String promptText(Map<String, Object> body) {
        Object p = body.get("prompt");
        if (p instanceof List<?> l) {
            p = l.getFirst();
        }
        return p instanceof String str ? str : "";
    }

    private int[] rawPrompt(Map<String, Object> body) {
        String text = promptText(body);
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

    /** @param json whether the output is constrained to JSON: Qwen3 thinking is then off (a think block is not JSON) */
    private int[] chatPrompt(Map<String, Object> body, boolean json) {
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
            } else if (content instanceof List<?> parts) { // OpenAI multi-part content: text parts only
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
            checkPromptBytes(bytes);
            list.add(new ChatTemplate.Message(role, text));
        }
        boolean thinking = Json.bool(body, "enable_thinking", thinkingDefault);
        Map<String, Object> extra = Json.map(body, "chat_template_kwargs");
        if (extra != null) {
            thinking = Json.bool(extra, "enable_thinking", thinking);
        }
        boolean thinkingGiven = body.containsKey("enable_thinking") || (extra != null && extra.containsKey("enable_thinking"));
        String template = engine.template().name();
        if (thinkingGiven && !template.equals("chatml-qwen3") && thinkingNoted.compareAndSet(false, true)) {
            System.err.println("note: enable_thinking only applies to the Qwen3 chat template; this model uses \""
                    + template + "\", the field is ignored (reported once)");
        }
        return engine.renderChat(list, thinking && !json);
    }

    /**
     * @param room      how many tokens the context has left after the prompt (at least 1)
     * @param allowZero whether {@code max_tokens: 0} is valid (an echo that only scores the prompt)
     */
    private SamplingParams params(Map<String, Object> b, int room, boolean allowZero) {
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
        if (maxTokens < (allowZero ? 0 : 1)) {
            throw RequestError.invalid("max_tokens must be at least " + (allowZero ? 0 : 1));
        }
        maxTokens = Math.min(maxTokens, room); // the engine stops at the context edge anyway; this also avoids int overflow
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

    /** An integer request field in {@code [0, Integer.MAX_VALUE]}; fractions, negatives and larger values are a 400. */
    private static int intField(Map<String, Object> b, String key, int def) {
        Object v = b.get(key);
        if (v == null) {
            return def;
        }
        if (!(v instanceof Number n) || (v instanceof Double x && x != Math.rint(x))) {
            throw RequestError.invalid("'" + key + "' must be an integer");
        }
        double x = n.doubleValue(); // Long or Double; exact enough for the range check
        if (x < 0 || x > Integer.MAX_VALUE) {
            throw RequestError.invalid("'" + key + "' must be between 0 and " + Integer.MAX_VALUE);
        }
        return (int) x;
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
