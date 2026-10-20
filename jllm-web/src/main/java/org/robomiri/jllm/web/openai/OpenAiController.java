package org.robomiri.jllm.web.openai;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.web.chat.ChatSupport;
import org.robomiri.jllm.web.chat.ChatSupport.RequestError;
import org.robomiri.jllm.web.models.LoadedModel;
import org.robomiri.jllm.web.models.ModelManager;
import org.robomiri.jllm.web.settings.SettingsService;
import org.robomiri.jllm.web.tools.ToolCallFormat;
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

/**
 * The OpenAI-compatible API over the loaded models: {@code GET /v1/models}, {@code POST /v1/chat/completions} and
 * {@code POST /v1/completions}, both with {@code "stream": true} (server-sent events). The {@code model} field picks
 * the loaded model. Chat requests may pass {@code tools}: the model's calls come back as {@code tool_calls} with
 * finish reason {@code tool_calls}, and the client sends the results as {@code tool} messages (see
 * {@link ToolCallFormat}).
 */
@RestController
public class OpenAiController {

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
                ChatSupport.send(res, 200, Json.write(modelObject(m)));
                return;
            }
        }
        ChatSupport.send(res, 404, ChatSupport.error("model '" + name + "' is not loaded", "invalid_request_error", "model_not_found"));
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
        LoadedModel model;
        try {
            model = models.resolve(Json.str(body, "model", null));
        } catch (NoSuchElementException e) {
            ChatSupport.send(res, 404, ChatSupport.error(e.getMessage(), "invalid_request_error", "model_not_found"));
            return;
        }
        try (LoadedModel.Lease lease = model.lease()) {
            completion(body, model, res, chat);
        } catch (LoadedModel.UnavailableException e) {
            ChatSupport.send(res, 503, ChatSupport.error(e.getMessage(), "server_error", "model_unavailable"));
        }
    }

    private void completion(Map<String, Object> body, LoadedModel model, HttpServletResponse res, boolean chat) throws IOException {
        Engine engine = model.engine();
        Settings s = settings.current();
        String modelId = model.id();
        int[] prompt;
        SamplingParams params;
        Set<String> toolNames = new HashSet<>();
        try {
            ChatSupport.checkUnsupported(body, chat);
            long maxPromptBytes = s.getLong("engine.max-prompt-bytes") > 0
                    ? s.getLong("engine.max-prompt-bytes") : 16L * engine.backend().contextLength();
            prompt = ChatSupport.checkContext(engine, chat ? chatPrompt(engine, body, s.getBool("chat.thinking"), maxPromptBytes, toolNames)
                    : rawPrompt(engine, body, maxPromptBytes));
            params = ChatSupport.params(body, s.sampling(), engine.backend().contextLength() - prompt.length);
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

        if (!stream) {
            Engine.Result r;
            try {
                r = engine.generate(prompt, params, null, () -> model.closing() || deadline.getAsBoolean());
            } catch (Engine.OverloadedException e) {
                ChatSupport.send(res, 503, ChatSupport.error(e.getMessage(), "server_error", "overloaded"));
                return;
            } catch (IllegalArgumentException e) {
                ChatSupport.send(res, 400, ChatSupport.error(e.getMessage(), "invalid_request_error", "context_length_exceeded"));
                return;
            }
            model.record(r.stats());
            String finish = deadline.finishReason(r);
            Map<String, Object> choice;
            if (chat) {
                Map<String, Object> message = Json.obj("role", "assistant", "content", r.text());
                ToolCallFormat.Parsed parsed = tools ? ToolCallFormat.parse(r.text(), toolNames) : null;
                if (parsed != null && !parsed.calls().isEmpty()) {
                    message.put("content", parsed.content().isEmpty() ? null : parsed.content());
                    message.put("tool_calls", toolCalls(parsed.calls(), false));
                    finish = "tool_calls";
                }
                choice = Json.obj("index", 0, "message", message, "finish_reason", finish);
            } else {
                choice = Json.obj("index", 0, "text", r.text(), "finish_reason", finish);
            }
            ChatSupport.send(res, 200, Json.write(Json.obj("id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(choice), "usage", ChatSupport.usage(r.stats()))));
            return;
        }

        // the stream starts at the first token: until then a queue timeout can still be a 503
        ChatSupport.Sse sse = new ChatSupport.Sse(res);
        IOException[] failure = new IOException[1];
        Engine.TextSink deltas = piece -> {
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
        };
        ToolCallFormat.Holdback holdback = tools ? new ToolCallFormat.Holdback(deltas) : null;
        Engine.Result r;
        try {
            r = engine.generate(prompt, params, holdback != null ? holdback : deltas,
                    () -> failure[0] != null || model.closing() || deadline.getAsBoolean());
        } catch (RuntimeException e) {
            if (sse.started()) {
                sse.event(Json.obj("error", Json.obj("message", "internal server error", "type", "server_error")));
                throw e;
            }
            if (e instanceof Engine.OverloadedException) {
                ChatSupport.send(res, 503, ChatSupport.error(e.getMessage(), "server_error", "overloaded"));
            } else if (e instanceof IllegalArgumentException) {
                ChatSupport.send(res, 400, ChatSupport.error(e.getMessage(), "invalid_request_error", "context_length_exceeded"));
            } else {
                throw e;
            }
            return;
        }
        model.record(r.stats());
        String finish = deadline.finishReason(r);
        if (holdback != null) {
            ToolCallFormat.Parsed parsed = ToolCallFormat.parse(r.text(), toolNames);
            if (!parsed.calls().isEmpty()) {
                if (!holdback.callStarted()) {
                    // a bare JSON call was streamed as text already; clients see the structured call below as well
                    holdback.finish();
                }
                if (failure[0] == null) {
                    startStream(sse, chat, id, object, created, modelId);
                    sse.event(Json.obj("id", id, "object", object, "created", created, "model", modelId, "choices",
                            List.of(Json.obj("index", 0, "delta", Json.obj("tool_calls", toolCalls(parsed.calls(), true)), "finish_reason", null))));
                }
                finish = "tool_calls";
            } else {
                holdback.finish();
            }
        }
        if (failure[0] != null) {
            return; // nothing more can be sent
        }
        startStream(sse, chat, id, object, created, modelId);
        Map<String, Object> last = chat
                ? Json.obj("index", 0, "delta", Json.obj(), "finish_reason", finish)
                : Json.obj("index", 0, "text", "", "finish_reason", finish);
        Map<String, Object> event = Json.obj("id", id, "object", object, "created", created, "model", modelId,
                "choices", List.of(last));
        if (includeUsage && !separateUsage) {
            event.put("usage", ChatSupport.usage(r.stats()));
        }
        sse.event(event);
        if (includeUsage && separateUsage) {
            sse.event(Json.obj("id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(), "usage", ChatSupport.usage(r.stats())));
        }
        sse.raw("data: [DONE]\n\n");
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

    private static void startStream(ChatSupport.Sse sse, boolean chat, String id, String object, long created, String modelId) throws IOException {
        if (sse.started()) {
            return;
        }
        sse.start();
        if (chat) {
            sse.event(Json.obj("id", id, "object", object, "created", created, "model", modelId,
                    "choices", List.of(Json.obj("index", 0, "delta", Json.obj("role", "assistant", "content", ""), "finish_reason", null))));
        }
    }

    private static int[] rawPrompt(Engine engine, Map<String, Object> body, long maxPromptBytes) {
        Object p = body.get("prompt");
        if (p instanceof List<?> l) {
            p = l.getFirst();
        }
        String text = p instanceof String str ? str : "";
        ChatSupport.checkPromptBytes(ChatSupport.utf8Length(text), maxPromptBytes);
        return engine.tokenizer().encode(text);
    }

    /** @param toolNames receives the names of the functions offered (empty when no tools are in play) */
    private int[] chatPrompt(Engine engine, Map<String, Object> body, boolean thinkingDefault, long maxPromptBytes,
                             Set<String> toolNames) {
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
        return engine.renderChat(ToolCallFormat.toTemplate(msgs, block, template), ChatSupport.thinking(body, thinkingDefault));
    }
}
