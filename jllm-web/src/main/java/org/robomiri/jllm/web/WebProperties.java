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
 * @param kvCacheMinTokens   shorter caches are not saved (default 512; also the least growth that saves a conversation
 *                           again)
 * @param kvCacheSaveAfterMs a request whose prefill took at least this long saves its slot's cache (default 2000: the
 *                           save only holds the slot to copy the token list, the rows are written in the background)
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
        @DefaultValue("512") int kvCacheMinTokens,
        @DefaultValue("2000") long kvCacheSaveAfterMs) {

    public WebProperties {
        if (!(kvCacheMaxGb >= 0)) {
            throw new IllegalArgumentException("jllm.web.kv-cache-max-gb must be 0 or more, got " + kvCacheMaxGb);
        }
        if (kvCacheMinTokens < 1) {
            throw new IllegalArgumentException("jllm.web.kv-cache-min-tokens must be at least 1, got " + kvCacheMinTokens);
        }
        if (kvCacheSaveAfterMs < 0) {
            throw new IllegalArgumentException("jllm.web.kv-cache-save-after-ms must be 0 or more, got " + kvCacheSaveAfterMs);
        }
    }

    public boolean apiKeyRequired() {
        return apiKey != null && !apiKey.isBlank();
    }
}
