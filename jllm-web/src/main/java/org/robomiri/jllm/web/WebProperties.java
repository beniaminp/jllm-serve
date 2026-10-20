package org.robomiri.jllm.web;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

import java.util.List;

/**
 * {@code jllm.web.*} application properties.
 *
 * @param settingsFile where the engine settings changed in the UI are stored (a properties file of overrides)
 * @param apiKey       when set, {@code /v1/**} and {@code /api/**} require {@code Authorization: Bearer <key>}
 * @param autoload     models (Hub repositories, links or local .gguf paths) loaded at startup
 * @param logLines     how many log lines the console keeps
 * @param toolsFile    tool settings and MCP servers (JSON)
 * @param reasoningFormat how {@code /v1/chat/completions} returns a {@code <think>} block: {@code deepseek} (in
 *                        {@code reasoning_content}, apart from the answer) or {@code none} (inline in the content);
 *                        a request may override it with {@code "reasoning_format"}
 * @param kvCache            save conversation KV caches to disk and restore them ({@code KvStore})
 * @param kvCacheDir         where the saved caches live
 * @param kvCacheMaxGb       disk budget; the least recently used caches are deleted beyond it
 * @param kvCacheMinTokens   shorter caches are not saved
 * @param kvCacheSaveAfterMs a request whose prefill took at least this long saves its slot's cache
 */
@ConfigurationProperties("jllm.web")
public record WebProperties(
        @DefaultValue("jllm-web.properties") String settingsFile,
        @DefaultValue("") String apiKey,
        @DefaultValue List<String> autoload,
        @DefaultValue("2000") int logLines,
        @DefaultValue("jllm-web-tools.json") String toolsFile,
        @DefaultValue("deepseek") String reasoningFormat,
        @DefaultValue("true") boolean kvCache,
        @DefaultValue("kv-cache") String kvCacheDir,
        @DefaultValue("10") double kvCacheMaxGb,
        @DefaultValue("1024") int kvCacheMinTokens,
        @DefaultValue("10000") long kvCacheSaveAfterMs) {

    public boolean apiKeyRequired() {
        return apiKey != null && !apiKey.isBlank();
    }
}
