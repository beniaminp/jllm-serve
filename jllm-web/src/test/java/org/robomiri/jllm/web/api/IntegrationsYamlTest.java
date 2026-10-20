package org.robomiri.jllm.web.api;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** The YAML scalars of the generated Continue / aider files. */
class IntegrationsYamlTest {

    @Test
    void plainWhenSafeQuotedOtherwise() {
        assertEquals("qwen3-0.6b", IntegrationsController.yaml("qwen3-0.6b"));
        assertEquals("openai/qwen3-0.6b", IntegrationsController.yaml("openai/qwen3-0.6b"));
        assertEquals("http://127.0.0.1:8090/v1", IntegrationsController.yaml("http://127.0.0.1:8090/v1"));
        assertEquals("\"qwen3 (jllm)\"", IntegrationsController.yaml("qwen3 (jllm)"));
        assertEquals("\"true\"", IntegrationsController.yaml("true"));   // would be a boolean
        assertEquals("\"1.5\"", IntegrationsController.yaml("1.5"));     // would be a number
        assertEquals("\"a: b # c\"", IntegrationsController.yaml("a: b # c"));
    }
}
