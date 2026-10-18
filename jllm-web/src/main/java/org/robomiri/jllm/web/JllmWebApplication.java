package org.robomiri.jllm.web;

import org.robomiri.jllm.web.logs.LogBuffer;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

/**
 * Web console for jllm-serve: chat, model management (Hugging Face links, load / unload with per-model
 * settings), the engine settings, a live log and the OpenAI-compatible API under {@code /v1}.
 */
@SpringBootApplication
@ConfigurationPropertiesScan
public class JllmWebApplication {

    public static void main(String[] args) {
        LogBuffer.install(); // before anything prints: the engine reports progress on stderr
        SpringApplication.run(JllmWebApplication.class, args);
    }
}
