package org.robomiri.jllm.web.openai;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.MediaPrompt;
import org.robomiri.jllm.engine.SlotPool;
import org.robomiri.jllm.engine.TokenLogprob;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.server.OpenAiFeatures;
import org.robomiri.jllm.tokenizer.Tokenizer;
import org.robomiri.jllm.web.WebProperties;
import org.robomiri.jllm.web.chat.ChatSupport;
import org.robomiri.jllm.web.chat.WebImages;
import org.robomiri.jllm.web.chat.ClientGone;
import org.robomiri.jllm.web.chat.ReasoningSplitter;
import org.robomiri.jllm.web.chat.ChatSupport.RequestError;
import org.robomiri.jllm.web.models.LoadedModel;
import org.robomiri.jllm.web.models.ModelManager;
import org.robomiri.jllm.web.settings.SettingsService;
import org.robomiri.jllm.web.tools.ToolCallFormat;
import org.robomiri.jllm.web.tools.ToolsConfig;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;

/**
 * The OpenAI-compatible API over the loaded models: {@code GET /v1/models}, {@code POST /v1/chat/completions} and
 * {@code POST /v1/completions}, both with {@code "stream": true} (server-sent events). The {@code model} field picks
 * the loaded model. Chat requests may pass {@code tools}: the model's calls come back as {@code tool_calls} with
 * finish reason {@code tool_calls}, and the client sends the results as {@code tool} messages (see
 * {@link ToolCallFormat}). {@code n}, {@code best_of}, {@code logprobs}, {@code logit_bias}, {@code response_format},
 * {@code echo} and {@code suffix} work as on the engine's own server ({@link OpenAiFeatures}).
 */
@RestController
public class OpenAiController {

    private final ModelManager models;
    private final SettingsService settings;
    private final WebProperties props;
    private final ToolsConfig toolsConfig;
    private final AtomicLong counter = new AtomicLong();
    private final AtomicBoolean thinkingNoted = new AtomicBoolean();
    private final org.robomiri.jllm.web.images.ImageService images;

    public OpenAiController(ModelManager models, SettingsService settings, WebProperties props, ToolsConfig toolsConfig,
                            org.robomiri.jllm.web.images.ImageService images) {
        this.models = models;
        this.settings = settings;
        this.props = props;
        this.toolsConfig = toolsConfig;
        this.images = images;
    }

    // loaded video models are listed too (kind: "video")
    @org.springframework.beans.factory.annotation.Autowired(required = false)
    private org.robomiri.jllm.web.video.VideoService videos;

    // loaded speech-to-text / text-to-speech models are listed too (kind: "speech-to-text" / "text-to-speech")
    @org.springframework.beans.factory.annotation.Autowired(required = false)
    private org.robomiri.jllm.web.audio.AudioService audio;

    @GetMapping(value = "/v1/models", produces = "application/json")
    public String listModels() {
        List<Object> data = new ArrayList<>();
        for (LoadedModel m : models.loaded()) {
            data.add(modelObject(m));
        }
        for (var m : images.loaded()) {
            var v = m.view();
            data.add(Json.obj("id", v.id(), "object", "model", "created", v.loadedAt() / 1000, "owned_by", "jllm-serve", "kind", "image",
                    "backend", v.backend(), "defaults", v.defaults(), "size_multiple", v.sizeMultiple(), "inpainting_unet", v.inpaintUnet()));
        }
        if (videos != null) {
            for (var m : videos.loaded()) {
                var v = m.view();
                data.add(Json.obj("id", v.id(), "object", "model", "created", v.loadedAt() / 1000, "owned_by", "jllm-serve", "kind", "video",
                        "backend", v.backend(), "defaults", v.defaults(), "size_multiple", v.sizeMultiple()));
            }
        }
        if (audio != null) {
            data.addAll(org.robomiri.jllm.audio.AudioApi.modelObjects(audio.models()));
        }
        return Json.write(Json.obj("object", "list", "data", data));
    }

    @GetMapping(value = "/v1/models/{*id}", produces = "application/json")
    public void getModel(@PathVariable String id, HttpServletResponse res) throws IOException {
        String name = id.startsWith("/") ? id.substring(1) : id;
        for (LoadedModel m : models.loaded()) {
            if (m.id().equals(name)) {
                ChatSupport.send(res, 200, Json.write(modelObject(m)));
                return;
            }
        }
        ChatSupport.send(res, 404, ChatSupport.error("model '" + name + "' is not loaded", "invalid_request_error", "model_not_found"));
    }

    private static Map<String, Object> modelObject(LoadedModel m) {
        LoadedModel.View v = m.view();
        return Json.obj("id", v.id(), "object", "model", "created", v.loadedAt() / 1000, "owned_by", "jllm-serve", "kind", "text",
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
        if (!ChatSupport.isJson(req.getContentType())) {
            ChatSupport.send(res, 415, ChatSupport.error("Content-Type must be application/json", "invalid_request_error", null));
            return;
        }
        Map<String, Object> body;
        try {
            body = Json.parseObject(text == null ? "" : text);
        } catch (RuntimeException e) {
            ChatSupport.send(res, 400, ChatSupport.error("invalid JSON: " + e.getMessage(), "invalid_request_error", null));
            return;
        }
        OpenAiFeatures.headerPriority(body, req.getHeader(OpenAiFeatures.PRIORITY_HEADER));
        LoadedModel model;
        try {
            model = models.resolve(Json.str(body, "model", null));
        } catch (NoSuchElementException e) {
            ChatSupport.send(res, 404, ChatSupport.error(e.getMessage(), "invalid_request_error", "model_not_found"));
            return;
        }
        try (LoadedModel.Lease lease = model.lease()) {
            completion(body, model, res, chat, ClientGone.of(req));
        } catch (LoadedModel.UnavailableException e) {
            ChatSupport.send(res, 503, ChatSupport.error(e.getMessage(), "server_error", "model_unavailable"));
        } catch (org.robomiri.jllm.engine.Admission.Abandoned e) {
            if (model.closing()) { // the model is being unloaded: the waiting request is told
                ChatSupport.send(res, 503, ChatSupport.error("model " + model.id() + " is being unloaded", "server_error", "model_unavailable"));
            } // else the client left while the request waited for a slot: nothing to answer
        }
    }

    /** @param gone whether the client has closed the connection: generation stops and nothing is sent */
    private void completion(Map<String, Object> body, LoadedModel model, HttpServletResponse res, boolean chat,
                            BooleanSupplier gone) throws IOException {
        Engine engine = model.engine();
        Tokenizer tok = engine.tokenizer();
        Settings s = settings.current();
        String modelId = model.id();
        int[] prompt;
        int[] textIds; // the prompt without the pictures' identities: what the text-side helpers read
        MediaPrompt media = null; // the prompt with pictures in it (docs/20)
        SamplingParams params;
        OpenAiFeatures.Options o;
        Engine.Decoding decoding;
        int echoFrom; // the prompt tokens an echo returns start here (after a BOS the engine put in front)
        String echoText = "";
        Set<String> toolNames = new HashSet<>();
        try {
            ChatSupport.checkUnsupported(body, chat);
            o = OpenAiFeatures.parse(body, chat, tok.vocabSize());
            long maxPromptBytes = s.getLong("engine.max-prompt-bytes") > 0
                    ? s.getLong("engine.max-prompt-bytes") : 16L * engine.backend().contextLength();
            int[] ids;
            List<org.robomiri.jllm.vision.Rgb> images = List.of();
            if (chat) {
                List<Object> raw = Json.list(body, "messages");
                Map<String, Object> chatBody = body;
                if (raw != null && org.robomiri.jllm.vision.ImageParts.hasImages(raw)) {
                    org.robomiri.jllm.vision.ImageParts.Extracted ex = WebImages.extract(raw, engine, s,
                            toolsConfig.general().fetch().allowPrivateAddresses());
                    chatBody = WebImages.withMessages(body, ex.messages());
                    images = ex.images();
                }
                ids = chatPrompt(engine, chatBody, s.getBool("chat.thinking"), o.json(), maxPromptBytes, toolNames);
            } else if (o.suffix() != null) {
                String text = rawText(body);
                ChatSupport.checkPromptBytes(ChatSupport.utf8Length(text) + ChatSupport.utf8Length(o.suffix()), maxPromptBytes);
                ids = OpenAiFeatures.fimPrompt(tok, text, o.suffix());
            } else {
                ids = rawPrompt(engine, body, maxPromptBytes);
            }
            if (!images.isEmpty()) {
                media = WebImages.prompt(engine, ids, images);
                prompt = media.identity();
                textIds = media.tokens();
                echoFrom = 0;
            } else {
                prompt = ChatSupport.checkContext(engine, ids);
                textIds = prompt;
                echoFrom = prompt.length - ids.length;
            }
            if (o.echo()) {
                echoText = rawText(body);
            }
            params = ChatSupport.params(body, s.sampling(), engine.backend().contextLength() - prompt.length, o.echo());
            // tool calls are constrained to the offered functions and their schemas (docs/17 §17.7)
            boolean constrainTools = chat && s.getBool("engine.constrained-tools");
            if (chat && !constrainTools) {
                ChatSupport.checkFormatWithoutToolConstraint(body); // the old behaviour: tools and response_format exclude each other
            }
            decoding = constrainTools ? ToolCallFormat.decoding(o.decoding(tok), body, engine, textIds, o.format(),
                    ChatSupport.thinking(body, s.getBool("chat.thinking")) && !o.json()) : o.decoding(tok);
        } catch (RequestError e) {
            ChatSupport.send(res, e.status, ChatSupport.error(e.getMessage(), e.type, e.code));
            return;
        } catch (RuntimeException e) {
            ChatSupport.send(res, 400, ChatSupport.error(e.getMessage(), "invalid_request_error", null));
            return;
        }
        boolean tools = !toolNames.isEmpty();
        boolean stream = Json.bool(body, "stream", false);
        Map<String, Object> streamOptions = Json.map(body, "stream_options");
        boolean separateUsage = streamOptions != null;
        boolean includeUsage = streamOptions == null || Json.bool(streamOptions, "include_usage", false);
        String id = (chat ? "chatcmpl-" : "cmpl-") + Long.toHexString(System.currentTimeMillis()) + "-" + counter.incrementAndGet();
        long created = System.currentTimeMillis() / 1000;
        String object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";
        ChatSupport.Deadline deadline = new ChatSupport.Deadline(s.getLong("server.request-timeout-ms"));
        // the thinking block goes to reasoning_content unless the server or the request asks for it inline
        boolean splitReasoning = chat && !"none".equalsIgnoreCase(Json.str(body, "reasoning_format", props.reasoningFormat()));
        boolean startsInThink = splitReasoning && ReasoningSplitter.promptOpensThink(engine, textIds);
        MediaPrompt mediaPrompt = media;
        int messages = chat && Json.list(body, "messages") != null ? Json.list(body, "messages").size() : 1;
        int count = o.count();

        try (org.robomiri.jllm.engine.Priority.Scope ignored = org.robomiri.jllm.engine.Priority.enter(decoding.priority()); // the queue's class
             SlotPool.Slot slot = model.pick(prompt, Json.str(body, "user", null), gone)) {
            Engine e = slot.engine();
            if (!stream) {
                List<Engine.Result> all;
                try {
                    all = mediaPrompt == null
                            ? e.generate(prompt, params, decoding, count, null,
                            () -> model.closing() || deadline.getAsBoolean() || gone.getAsBoolean())
                            : e.generate(mediaPrompt, params, decoding, count, null,
                            () -> model.closing() || deadline.getAsBoolean() || gone.getAsBoolean());
                } catch (Engine.OverloadedException ex) {
                    ChatSupport.send(res, 503, ChatSupport.error(ex.getMessage(), "server_error", "overloaded"));
                    return;
                } catch (IllegalArgumentException ex) {
                    ChatSupport.send(res, 400, ChatSupport.error(ex.getMessage(), "invalid_request_error", "context_length_exceeded"));
                    return;
                }
                Engine.Stats total = OpenAiFeatures.total(all);
                model.record(slot, total);
                if (gone.getAsBoolean()) { // nobody to answer: the slot is free again right away
                    log(modelId, slot, chat, messages, toolNames.size(), count, total, "cancelled: the client went away");
                    return;
                }
                List<Engine.Result> chosen = OpenAiFeatures.select(all, o);
                List<Object> choices = new ArrayList<>();
                String lastFinish = null;
                for (int i = 0; i < chosen.size(); i++) {
                    Engine.Result r = chosen.get(i);
                    String finish = deadline.finishReason(r);
                    Map<String, Object> choice;
                    if (chat) {
                        String text = r.text();
                        List<Object> calls = null;
                        if (tools) {
                            ToolCallFormat.Parsed parsed = ToolCallFormat.parse(text, toolNames);
                            if (!parsed.calls().isEmpty()) {
                                text = parsed.content();
                                calls = toolCalls(parsed.calls(), false);
                                finish = "tool_calls";
                            }
                        }
                        Map<String, Object> message = Json.obj("role", "assistant");
                        if (splitReasoning) {
                            String[] parts = ReasoningSplitter.split(text, startsInThink);
                            if (!parts[0].isEmpty()) {
                                message.put("reasoning_content", parts[0]);
                            }
                            text = parts[1];
                        }
                        message.put("content", calls != null && text.isBlank() ? null : text);
                        if (calls != null) {
                            message.put("tool_calls", calls);
                        }
                        choice = Json.obj("index", i, "message", message);
                    } else {
                        choice = Json.obj("index", i, "text", echoText + r.text());
                    }
                    if (o.logprobs()) {
                        choice.put("logprobs", OpenAiFeatures.choiceLogprobs(chat, r, o, prompt, echoFrom, tok));
                    }
                    choice.put("finish_reason", finish);
                    choices.add(choice);
                    lastFinish = finish;
                }
                log(modelId, slot, chat, messages, toolNames.size(), count, total, lastFinish);
                ChatSupport.send(res, 200, Json.write(Json.obj("id", id, "object", object, "created", created, "model", modelId,
                        "choices", choices, "usage", ChatSupport.usage(total))));
                return;
            }

            // the headers go out now and keep-alive comments while the prompt is prefilled (minutes for a coding
            // agent's prompt on a CPU); a refusal after this point becomes an error event. The choices run one
            // after the other; every chunk carries its choice's index.
            ChatSupport.Sse sse = new ChatSupport.Sse(res);
            IOException[] failure = new IOException[1];
            sse.start();
            List<Engine.Result> finished = new ArrayList<>();
            String echoed = echoText;
            String[] lastFinish = new String[1];
            Engine.ChoiceSink sink = new Engine.ChoiceSink() {
                /** The text path of the running choice: tool-call holdback, then reasoning split, then events. */
                private ReasoningSplitter splitter;
                private ToolCallFormat.Holdback holdback;
                private Engine.TextSink entry;
                private final List<TokenLogprob> pending = new ArrayList<>();
                private final int[] offset = new int[1];
                private List<TokenLogprob> promptLps;

                private void event(Map<String, Object> choice, boolean last) {
                    if (failure[0] != null) {
                        return;
                    }
                    Map<String, Object> ev = Json.obj("id", id, "object", object, "created", created, "model", modelId,
                            "choices", List.of(choice));
                    if (last && includeUsage && !separateUsage) {
                        ev.put("usage", ChatSupport.usage(OpenAiFeatures.total(finished)));
                    }
                    try {
                        sse.event(ev);
                    } catch (IOException ex) {
                        failure[0] = ex; // the client went away: stop generating
                    }
                }

                private void send(int i, String field, String piece) {
                    Map<String, Object> choice = chat
                            ? Json.obj("index", i, "delta", Json.obj(field, piece))
                            : Json.obj("index", i, "text", piece);
                    if (o.logprobs()) {
                        choice.put("logprobs", OpenAiFeatures.drainLogprobs(chat, pending, tok, offset));
                    }
                    choice.put("finish_reason", null);
                    event(choice, false);
                }

                @Override
                public void start(int i) {
                    pending.clear();
                    offset[0] = 0;
                    splitter = splitReasoning ? new ReasoningSplitter(piece -> send(i, "reasoning_content", piece),
                            piece -> send(i, "content", piece), startsInThink) : null;
                    Engine.TextSink deltas = splitter != null ? splitter : piece -> send(i, "content", piece);
                    holdback = tools ? new ToolCallFormat.Holdback(deltas) : null;
                    entry = holdback != null ? holdback : deltas;
                    if (chat) {
                        event(Json.obj("index", i, "delta", Json.obj("role", "assistant", "content", ""), "finish_reason", null), false);
                    }
                }

                @Override
                public void prefilled(int i, List<TokenLogprob> lps) {
                    if (i == 0) {
                        promptLps = lps;
                    }
                    if (o.echo()) {
                        Map<String, Object> choice = Json.obj("index", i, "text", echoed);
                        if (o.logprobs()) {
                            choice.put("logprobs", OpenAiFeatures.echoLogprobs(prompt, echoFrom, promptLps, tok, offset));
                        }
                        choice.put("finish_reason", null);
                        event(choice, false);
                    }
                }

                @Override
                public void text(int i, String text) {
                    entry.accept(text);
                }

                @Override
                public void token(int i, TokenLogprob lp) {
                    pending.add(lp);
                }

                @Override
                public void done(int i, Engine.Result r) {
                    finished.add(r);
                    String finish = deadline.finishReason(r);
                    List<Object> calls = null;
                    if (holdback != null) {
                        ToolCallFormat.Parsed parsed = ToolCallFormat.parse(r.text(), toolNames);
                        if (!holdback.callStarted()) {
                            holdback.finish(); // (a bare JSON call was streamed as text; the structured call follows anyway)
                        }
                        if (!parsed.calls().isEmpty()) {
                            calls = toolCalls(parsed.calls(), true);
                            finish = "tool_calls";
                        }
                    }
                    if (splitter != null) {
                        splitter.finish();
                    }
                    lastFinish[0] = finish;
                    if (calls != null) {
                        event(Json.obj("index", i, "delta", Json.obj("tool_calls", calls), "finish_reason", null), false);
                    }
                    Map<String, Object> last = chat
                            ? Json.obj("index", i, "delta", Json.obj())
                            : Json.obj("index", i, "text", "");
                    if (o.logprobs()) {
                        last.put("logprobs", OpenAiFeatures.drainLogprobs(chat, pending, tok, offset));
                    }
                    last.put("finish_reason", finish);
                    event(last, i == count - 1);
                }
            };
            try (AutoCloseable alive = sse.keepAlive()) {
                if (mediaPrompt == null) {
                    e.generate(prompt, params, decoding, count, sink,
                            () -> failure[0] != null || sse.failed() || model.closing() || deadline.getAsBoolean()
                                    || gone.getAsBoolean());
                } else {
                    e.generate(mediaPrompt, params, decoding, count, sink,
                            () -> failure[0] != null || sse.failed() || model.closing() || deadline.getAsBoolean()
                                    || gone.getAsBoolean());
                }
            } catch (Engine.OverloadedException | IllegalArgumentException ex) {
                sse.event(Json.obj("error", Json.obj("message", ex.getMessage(), "type",
                        ex instanceof Engine.OverloadedException ? "server_error" : "invalid_request_error",
                        "code", ex instanceof Engine.OverloadedException ? "overloaded" : "context_length_exceeded")));
                return;
            } catch (RuntimeException ex) {
                sse.event(Json.obj("error", Json.obj("message", "internal server error", "type", "server_error")));
                throw ex;
            } catch (Exception ex) { // AutoCloseable.close: cannot happen
                throw new IllegalStateException(ex);
            }
            Engine.Stats total = OpenAiFeatures.total(finished);
            model.record(slot, total);
            boolean away = failure[0] != null || sse.failed() || gone.getAsBoolean();
            log(modelId, slot, chat, messages, toolNames.size(), count, total,
                    away ? "cancelled: the client went away" : lastFinish[0]);
            if (away) {
                return; // nothing more can be sent
            }
            if (includeUsage && separateUsage) {
                sse.event(Json.obj("id", id, "object", object, "created", created, "model", modelId,
                        "choices", List.of(), "usage", ChatSupport.usage(total)));
            }
            sse.raw("data: [DONE]\n\n");
        }
    }

    /** One line per request in the log: enough to see what an agent is doing and where the time goes. */
    private static void log(String model, SlotPool.Slot slot, boolean chat, int messages, int tools, int choices,
                            Engine.Stats st, String finish) {
        System.err.printf("%s %s [slot %d] %d msg%s%s%s | prompt %d tok (%d cached) %.1f s | %d tok %.1f tok/s | %s%n",
                chat ? "chat" : "completion", model, slot.index(), messages, messages == 1 ? "" : "s",
                tools > 0 ? ", " + tools + " tools" : "", choices > 1 ? ", " + choices + " choices" : "",
                st.promptTokens(), st.cachedTokens(), st.prefillMs() / 1000, st.generatedTokens(), st.tokPerSec(), finish);
    }

    /** OpenAI {@code tool_calls}; calls the model wrote unreadably are passed on with their raw error as arguments. */
    private static List<Object> toolCalls(List<ToolCallFormat.Call> calls, boolean indexed) {
        List<Object> out = new ArrayList<>();
        for (int i = 0; i < calls.size(); i++) {
            ToolCallFormat.Call c = calls.get(i);
            Map<String, Object> call = Json.obj("id", c.id(), "type", "function", "function",
                    Json.obj("name", c.name() == null ? "" : c.name(), "arguments", c.argumentsJson()));
            if (indexed) {
                call.put("index", i);
            }
            out.add(call);
        }
        return out;
    }

    /** The completions prompt: a string, or a single-element string array (validated in checkUnsupported). */
    private static String rawText(Map<String, Object> body) {
        Object p = body.get("prompt");
        if (p instanceof List<?> l) {
            p = l.getFirst();
        }
        return p instanceof String str ? str : "";
    }

    private static int[] rawPrompt(Engine engine, Map<String, Object> body, long maxPromptBytes) {
        String text = rawText(body);
        ChatSupport.checkPromptBytes(ChatSupport.utf8Length(text), maxPromptBytes);
        return engine.tokenizer().encode(text);
    }

    /**
     * @param thinkingDefault whether Qwen3 thinking is on when the request does not say
     * @param json            whether the output is constrained to JSON: thinking is then off (a think block is not JSON)
     * @param toolNames       receives the names of the functions offered (empty when no tools are in play)
     */
    private int[] chatPrompt(Engine engine, Map<String, Object> body, boolean thinkingDefault, boolean json,
                             long maxPromptBytes, Set<String> toolNames) {
        List<Object> msgs = ChatSupport.messages(body, maxPromptBytes);
        String block = null;
        Object choice = body.get("tool_choice");
        List<Object> tools = Json.list(body, "tools");
        if (tools != null && !tools.isEmpty() && !"none".equals(choice)) {
            List<Map<String, Object>> specs = new ArrayList<>();
            for (Object t : tools) {
                @SuppressWarnings("unchecked")
                Map<String, Object> spec = (Map<String, Object>) t;
                specs.add(Json.obj("type", "function", "function", spec.get("function")));
                toolNames.add(Json.str(Json.map(t, "function"), "name", ""));
            }
            String instruction = "required".equals(choice) ? "You must call at least one of these functions."
                    : choice instanceof Map<?, ?> ? "You must call the function " + Json.str(Json.map(choice, "function"), "name", "") + "."
                    : null;
            block = ToolCallFormat.toolsBlock(specs, instruction);
            ChatSupport.checkPromptBytes(ChatSupport.utf8Length(block), maxPromptBytes);
        }
        boolean given = body.containsKey("enable_thinking") || Json.map(body, "chat_template_kwargs") != null;
        String template = engine.template().name();
        if (given && !template.equals("chatml-qwen3") && thinkingNoted.compareAndSet(false, true)) {
            System.err.println("note: enable_thinking only applies to the Qwen3 chat template; this model uses \""
                    + template + "\", the field is ignored (reported once)");
        }
        return engine.renderChat(ToolCallFormat.toTemplate(msgs, block, template), ChatSupport.thinking(body, thinkingDefault) && !json);
    }
}
