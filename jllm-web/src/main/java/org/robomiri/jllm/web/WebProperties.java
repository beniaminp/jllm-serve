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
 */
@ConfigurationProperties("jllm.web")
public record WebProperties(
        @DefaultValue("jllm-web.properties") String settingsFile,
        @DefaultValue("") String apiKey,
        @DefaultValue List<String> autoload,
        @DefaultValue("2000") int logLines) {

    public boolean apiKeyRequired() {
        return apiKey != null && !apiKey.isBlank();
    }
}
