package org.robomiri.jllm.web.api;

import jakarta.servlet.http.HttpServletRequest;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.WebProperties;
import org.robomiri.jllm.web.models.LoadedModel;
import org.robomiri.jllm.web.models.ModelManager;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Configuration for coding agents that speak the OpenAI API, built for the loaded models:
 * <ul>
 *   <li>{@code GET /api/integrations/opencode}: an {@code opencode.json} (with {@code ?raw=true}, only the file, e.g.
 *       {@code curl -s localhost:8090/api/integrations/opencode?raw=true > opencode.json});</li>
 *   <li>{@code GET /api/integrations/continue}: a Continue {@code config.yaml} ({@code ~/.continue/config.yaml});</li>
 *   <li>{@code GET /api/integrations/aider}: {@code .aider.conf.yml}, {@code .aider.model.metadata.json} and
 *       {@code .aider.model.settings.yml} ({@code ?file=conf|metadata|settings} returns one of them raw);</li>
 *   <li>{@code GET /api/integrations/cline}: the values to enter in Cline's "OpenAI Compatible" provider settings
 *       (Cline has no configuration file).</li>
 * </ul>
 * Each answer carries warnings for the models that will struggle (context too small for the agent's prompt, a
 * single slot, a family that follows the tool-call format less reliably).
 */
@RestController
@RequestMapping("/api/integrations")
public class IntegrationsController {

    /** opencode's system prompt plus its tool definitions, measured with opencode 1.19 (about 44 000 characters). */
    static final int OPENCODE_PROMPT_TOKENS = 11_000;
    static final String PROVIDER = "jllm";

    private final ModelManager models;
    private final WebProperties props;

    public IntegrationsController(ModelManager models, WebProperties props) {
        this.models = models;
        this.props = props;
    }

    @GetMapping("/opencode")
    public ResponseEntity<String> opencode(@RequestParam(required = false) String baseUrl,
                                           @RequestParam(required = false) String model,
                                           @RequestParam(defaultValue = "false") boolean raw,
                                           HttpServletRequest req) {
        List<LoadedModel> loaded = models.loaded();
        if (loaded.isEmpty()) {
            throw new IllegalStateException("no model is loaded: load one first (Models page; for coding use a context of 32768 and 2 slots)");
        }
        String base = baseUrl != null && !baseUrl.isBlank() ? baseUrl.replaceAll("/+$", "") : origin(req) + "/v1";
        // no overall request limit: the first prefill of opencode's ~11k-token prompt can take minutes on a CPU; the
        // server sends keep-alive comments meanwhile, so the header and idle-chunk timeouts still catch a dead server
        Map<String, Object> options = Json.obj("baseURL", base, "timeout", false);
        if (props.apiKeyRequired()) {
            options.put("apiKey", "{env:JLLM_API_KEY}");
        }
        Map<String, Object> modelMap = new LinkedHashMap<>();
        List<String> warnings = new ArrayList<>();
        String main = null;
        for (LoadedModel m : loaded) {
            LoadedModel.View v = m.view();
            String template = v.template();
            boolean qwen3 = template.equals("chatml-qwen3");
            boolean reasoning = qwen3 || template.startsWith("deepseek");
            Map<String, Object> entry = Json.obj(
                    "name", v.id() + " (jllm)",
                    "limit", Json.obj("context", v.contextLength(), "output", Math.max(256, Math.min(8192, v.contextLength() / 4))),
                    "tool_call", true,
                    "reasoning", reasoning);
            if (qwen3) {
                // thinking makes every step several times slower on a CPU; switch it on for hard problems
                entry.put("options", Json.obj("enable_thinking", false));
            }
            modelMap.put(v.id(), entry);
            if (main == null && (model == null || model.equals(v.id()))) {
                main = v.id();
            }
            if (v.contextLength() < 16_384) {
                warnings.add(v.id() + ": context is " + v.contextLength() + " tokens; opencode's instructions and tool"
                        + " definitions alone take about " + OPENCODE_PROMPT_TOKENS + ". Load it with a context of 32768.");
            }
            if (v.slots() < 2) {
                warnings.add(v.id() + ": 1 slot. opencode sends a title request with each message, which pushes the"
                        + " conversation out of the KV cache, so the next step prefills everything again. Load it with 2 slots.");
            }
            if (!template.startsWith("chatml-qwen")) {
                warnings.add(v.id() + ": tool calls use the Qwen / Hermes format, which " + template
                        + " models follow less reliably; Qwen2.5 / Qwen3 (4B or larger) work best for coding.");
            }
        }
        if (main == null) {
            throw new java.util.NoSuchElementException("model '" + model + "' is not loaded");
        }
        Map<String, Object> config = Json.obj(
                "$schema", "https://opencode.ai/config.json",
                "provider", Json.obj(PROVIDER, Json.obj(
                        "npm", "@ai-sdk/openai-compatible",
                        "name", "jllm-serve (local)",
                        "options", options,
                        "models", modelMap)),
                "model", PROVIDER + "/" + main,
                "small_model", PROVIDER + "/" + main);
        String file = org.robomiri.jllm.web.tools.ToolsConfig.pretty(config, 0) + "\n";
        if (raw) {
            return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON)
                    .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"opencode.json\"").body(file);
        }
        return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(Json.write(Json.obj(
                "file", file, "warnings", warnings, "baseUrl", base, "apiKeyRequired", props.apiKeyRequired())));
    }

    /** Continue's agent-mode system prompt and tool definitions, roughly. */
    static final int CONTINUE_PROMPT_TOKENS = 5_000;
    /** Aider's system prompt and repository map (1024 tokens by default), roughly. */
    static final int AIDER_PROMPT_TOKENS = 4_000;
    /** Cline's system prompt with its tool descriptions, roughly. */
    static final int CLINE_PROMPT_TOKENS = 12_000;

    private List<LoadedModel.View> views(String model) {
        List<LoadedModel> loaded = models.loaded();
        if (loaded.isEmpty()) {
            throw new IllegalStateException("no model is loaded: load one first (Models page; for coding use a context of 32768 and 2 slots)");
        }
        List<LoadedModel.View> views = new ArrayList<>();
        for (LoadedModel m : loaded) {
            views.add(m.view());
        }
        if (model != null && !model.isBlank()) { // the chosen model first
            LoadedModel.View chosen = views.stream().filter(v -> v.id().equals(model)).findFirst()
                    .orElseThrow(() -> new java.util.NoSuchElementException("model '" + model + "' is not loaded"));
            views.remove(chosen);
            views.addFirst(chosen);
        }
        return views;
    }

    private static String base(String baseUrl, HttpServletRequest req) {
        return baseUrl != null && !baseUrl.isBlank() ? baseUrl.replaceAll("/+$", "") : origin(req) + "/v1";
    }

    private static int maxOutput(LoadedModel.View v) {
        return Math.max(256, Math.min(8192, v.contextLength() / 4));
    }

    private static List<String> warnings(List<LoadedModel.View> views, String agent, int promptTokens, boolean slotsMatter) {
        List<String> w = new ArrayList<>();
        for (LoadedModel.View v : views) {
            if (v.contextLength() < 2 * promptTokens) {
                w.add(v.id() + ": context is " + v.contextLength() + " tokens; " + agent + "'s instructions alone take about "
                        + promptTokens + ". Load it with a context of 32768.");
            }
            if (slotsMatter && v.slots() < 2) {
                w.add(v.id() + ": 1 slot. A side request (autocomplete, a title) pushes the conversation out of the KV cache,"
                        + " so the next step prefills everything again. Load it with 2 slots.");
            }
            if (!v.template().startsWith("chatml-qwen")) {
                w.add(v.id() + ": " + v.template() + " models follow tool and edit formats less reliably than Qwen2.5 /"
                        + " Qwen3 (4B or larger), which work best for coding.");
            }
        }
        return w;
    }

    private static ResponseEntity<String> json(Object o) {
        return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(Json.write(o));
    }

    private static ResponseEntity<String> rawFile(String name, String type, String body) {
        return ResponseEntity.ok().contentType(MediaType.parseMediaType(type + ";charset=UTF-8"))
                .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"" + name + "\"").body(body);
    }

    /** A YAML scalar: plain when safe, else double-quoted with JSON escapes (valid YAML). */
    static String yaml(String s) {
        return s.matches("[A-Za-z0-9_./:@-]+") && !s.matches("(?i)true|false|null|yes|no|on|off|~|[0-9.e+-]+")
                ? s : Json.write(s);
    }

    @GetMapping("/continue")
    public ResponseEntity<String> continueConfig(@RequestParam(required = false) String baseUrl,
                                                 @RequestParam(required = false) String model,
                                                 @RequestParam(defaultValue = "false") boolean raw,
                                                 HttpServletRequest req) {
        List<LoadedModel.View> views = views(model);
        String base = base(baseUrl, req);
        StringBuilder y = new StringBuilder();
        y.append("# Continue configuration for jllm-serve (").append(base).append(")\n");
        y.append("# Save as ~/.continue/config.yaml, or copy the models into the one you have.\n");
        y.append("name: jllm-serve\nversion: 1.0.0\nschema: v1\nmodels:\n");
        for (LoadedModel.View v : views) {
            y.append("  - name: ").append(yaml(v.id() + " (jllm)")).append('\n');
            y.append("    provider: openai\n");
            y.append("    model: ").append(yaml(v.id())).append('\n');
            y.append("    apiBase: ").append(yaml(base)).append('\n');
            y.append("    apiKey: ").append(props.apiKeyRequired() ? "REPLACE_WITH_JLLM_API_KEY # the console's jllm.web.api-key" : "jllm")
                    .append('\n');
            y.append("    roles:\n      - chat\n      - edit\n      - apply\n");
            y.append("    capabilities:\n      - tool_use\n");
            y.append("    defaultCompletionOptions:\n");
            y.append("      contextLength: ").append(v.contextLength()).append('\n');
            y.append("      maxTokens: ").append(maxOutput(v)).append('\n');
            y.append("    requestOptions:\n");
            y.append("      timeout: 3600 # seconds: a long first prefill on a CPU takes minutes\n");
            if (v.template().equals("chatml-qwen3")) {
                y.append("      extraBodyProperties:\n        enable_thinking: false # true for hard problems (much slower)\n");
            }
        }
        String file = y.toString();
        if (raw) {
            return rawFile("config.yaml", "application/yaml", file);
        }
        return json(Json.obj("file", file, "warnings", warnings(views, "Continue's agent mode", CONTINUE_PROMPT_TOKENS, true),
                "baseUrl", base, "apiKeyRequired", props.apiKeyRequired()));
    }

    @GetMapping("/aider")
    public ResponseEntity<String> aider(@RequestParam(required = false) String baseUrl,
                                        @RequestParam(required = false) String model,
                                        @RequestParam(required = false) String file,
                                        HttpServletRequest req) {
        List<LoadedModel.View> views = views(model);
        String base = base(baseUrl, req);
        LoadedModel.View main = views.getFirst();
        StringBuilder conf = new StringBuilder();
        conf.append("# aider configuration for jllm-serve (").append(base).append("): save in your repository as .aider.conf.yml\n");
        conf.append("model: ").append(yaml("openai/" + main.id())).append('\n');
        conf.append("weak-model: ").append(yaml("openai/" + main.id())).append('\n');
        conf.append("openai-api-base: ").append(yaml(base)).append('\n');
        if (props.apiKeyRequired()) {
            conf.append("# the key comes from the OPENAI_API_KEY environment variable (export OPENAI_API_KEY=<jllm key>)\n");
        } else {
            conf.append("openai-api-key: jllm\n");
        }
        conf.append("model-metadata-file: .aider.model.metadata.json\n");
        conf.append("model-settings-file: .aider.model.settings.yml\n");
        conf.append("timeout: 3600 # seconds: a long first prefill on a CPU takes minutes\n");
        conf.append("show-model-warnings: false\n");
        conf.append("map-tokens: 1024\n");

        Map<String, Object> meta = new LinkedHashMap<>();
        StringBuilder settings = new StringBuilder();
        settings.append("# aider model settings for jllm-serve: save in your repository as .aider.model.settings.yml\n");
        for (LoadedModel.View v : views) {
            String name = "openai/" + v.id();
            meta.put(name, Json.obj("max_tokens", maxOutput(v), "max_input_tokens", v.contextLength(),
                    "max_output_tokens", maxOutput(v), "input_cost_per_token", 0, "output_cost_per_token", 0,
                    "litellm_provider", "openai", "mode", "chat"));
            // small models rewrite whole files more reliably than they produce search/replace blocks
            boolean small = v.weightBytes() < 4_000_000_000L;
            settings.append("- name: ").append(yaml(name)).append('\n');
            settings.append("  edit_format: ").append(small ? "whole" : "diff").append('\n');
            settings.append("  use_repo_map: true\n");
            settings.append("  examples_as_sys_msg: true\n");
            settings.append("  extra_params:\n    max_tokens: ").append(maxOutput(v)).append('\n');
            if (v.template().equals("chatml-qwen3")) {
                settings.append("    extra_body:\n      enable_thinking: false # true for hard problems (much slower)\n");
            }
        }
        String metadata = org.robomiri.jllm.web.tools.ToolsConfig.pretty(meta, 0) + "\n";
        if (file != null) {
            return switch (file) {
                case "conf" -> rawFile(".aider.conf.yml", "application/yaml", conf.toString());
                case "metadata" -> rawFile(".aider.model.metadata.json", "application/json", metadata);
                case "settings" -> rawFile(".aider.model.settings.yml", "application/yaml", settings.toString());
                default -> throw new IllegalArgumentException("file must be conf, metadata or settings");
            };
        }
        return json(Json.obj(
                "files", Json.obj(".aider.conf.yml", conf.toString(), ".aider.model.metadata.json", metadata,
                        ".aider.model.settings.yml", settings.toString()),
                "warnings", warnings(views, "aider", AIDER_PROMPT_TOKENS, false),
                "baseUrl", base, "apiKeyRequired", props.apiKeyRequired()));
    }

    @GetMapping("/cline")
    public ResponseEntity<String> cline(@RequestParam(required = false) String baseUrl,
                                        @RequestParam(required = false) String model,
                                        HttpServletRequest req) {
        List<LoadedModel.View> views = views(model);
        String base = base(baseUrl, req);
        List<Object> list = new ArrayList<>();
        for (LoadedModel.View v : views) {
            list.add(Json.obj("modelId", v.id(), "contextWindow", v.contextLength(), "maxOutputTokens", maxOutput(v),
                    "thinkingTemplate", v.template().equals("chatml-qwen3")));
        }
        List<String> warnings = warnings(views, "Cline", CLINE_PROMPT_TOKENS, true);
        if (views.stream().anyMatch(v -> v.template().equals("chatml-qwen3"))) {
            warnings.add("Cline cannot send enable_thinking: Qwen3 thinks unless chat.thinking is false (Settings page),"
                    + " which makes every step several times slower on a CPU.");
        }
        return json(Json.obj("provider", "OpenAI Compatible", "baseUrl", base,
                "apiKey", props.apiKeyRequired() ? "(your jllm.web.api-key)" : "jllm",
                "models", list, "supportsImages", false, "supportsComputerUse", false,
                "warnings", warnings, "apiKeyRequired", props.apiKeyRequired()));
    }

    private static String origin(HttpServletRequest req) {
        String scheme = req.getHeader("X-Forwarded-Proto") != null ? req.getHeader("X-Forwarded-Proto") : req.getScheme();
        String host = req.getHeader("X-Forwarded-Host") != null ? req.getHeader("X-Forwarded-Host")
                : req.getServerName() + ((req.getServerPort() == 80 && scheme.equals("http")) || (req.getServerPort() == 443
                && scheme.equals("https")) ? "" : ":" + req.getServerPort());
        return scheme + "://" + host;
    }
}
