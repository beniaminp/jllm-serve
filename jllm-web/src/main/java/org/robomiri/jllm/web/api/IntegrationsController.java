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
 * Configuration for coding agents that speak the OpenAI API. {@code GET /api/integrations/opencode} builds an
 * {@code opencode.json} for the loaded models (with {@code ?raw=true}, only the file, e.g.
 * {@code curl -s localhost:8090/api/integrations/opencode?raw=true > opencode.json}).
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

    private static String origin(HttpServletRequest req) {
        String scheme = req.getHeader("X-Forwarded-Proto") != null ? req.getHeader("X-Forwarded-Proto") : req.getScheme();
        String host = req.getHeader("X-Forwarded-Host") != null ? req.getHeader("X-Forwarded-Host")
                : req.getServerName() + ((req.getServerPort() == 80 && scheme.equals("http")) || (req.getServerPort() == 443
                && scheme.equals("https")) ? "" : ":" + req.getServerPort());
        return scheme + "://" + host;
    }
}
