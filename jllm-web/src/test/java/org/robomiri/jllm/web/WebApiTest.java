package org.robomiri.jllm.web;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.hamcrest.Matchers.containsString;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** The console's endpoints without a model: settings round trip, validation and the OpenAI errors. */
@SpringBootTest(properties = {"jllm.web.settings-file=target/test-settings/jllm-web.properties",
        "jllm.web.api-key=secret"})
@AutoConfigureMockMvc
class WebApiTest {

    private static final String AUTH = "Bearer secret";

    @Autowired
    MockMvc mvc;

    @Test
    void apiKeyIsRequired() throws Exception {
        mvc.perform(get("/v1/models")).andExpect(status().isUnauthorized());
        mvc.perform(get("/api/status").header("Authorization", "Bearer wrong")).andExpect(status().isUnauthorized());
        mvc.perform(get("/api/status").header("Authorization", AUTH)).andExpect(status().isOk())
                .andExpect(jsonPath("$.apiKeyRequired").value(true));
        mvc.perform(get("/index.html")).andExpect(status().isOk()); // the UI itself is public
    }

    @Test
    void noModelLoaded() throws Exception {
        mvc.perform(get("/v1/models").header("Authorization", AUTH)).andExpect(status().isOk())
                .andExpect(jsonPath("$.object").value("list")).andExpect(jsonPath("$.data").isEmpty());
        mvc.perform(post("/v1/chat/completions").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"))
                .andExpect(status().isNotFound()).andExpect(jsonPath("$.error.code").value("model_not_found"));
        mvc.perform(post("/v1/chat/completions").header("Authorization", AUTH).contentType(MediaType.TEXT_PLAIN).content("{}"))
                .andExpect(status().isUnsupportedMediaType());
        mvc.perform(post("/v1/completions").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON).content("{nope"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.message", containsString("invalid JSON")));
    }

    @Test
    void settingsRoundTrip() throws Exception {
        mvc.perform(get("/api/settings").header("Authorization", AUTH)).andExpect(status().isOk())
                .andExpect(content().string(containsString("\"key\":\"backend.ctx\"")));
        mvc.perform(put("/api/settings").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"changes\":{\"sampling.temperature\":\"0.3\"}}"))
                .andExpect(status().isOk());
        String stored = Files.readString(Path.of("target/test-settings/jllm-web.properties"));
        org.junit.jupiter.api.Assertions.assertTrue(stored.contains("sampling.temperature=0.3"), stored);
        // invalid values are rejected as a whole and nothing is stored
        mvc.perform(put("/api/settings").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"changes\":{\"backend.ctx\":\"0\",\"sampling.top-k\":\"5\"}}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error", containsString("backend.ctx")));
        mvc.perform(put("/api/settings").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"changes\":{\"no.such.key\":\"1\"}}"))
                .andExpect(status().isBadRequest());
        mvc.perform(put("/api/settings").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"reset\":[\"sampling.temperature\"]}"))
                .andExpect(status().isOk());
        stored = Files.readString(Path.of("target/test-settings/jllm-web.properties"));
        org.junit.jupiter.api.Assertions.assertFalse(stored.contains("sampling.temperature"), stored);
    }

    @Test
    void loadValidation() throws Exception {
        mvc.perform(post("/api/models/load").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"source\":\"https://huggingface.co/datasets/x/y\"}"))
                .andExpect(status().isBadRequest());
        mvc.perform(post("/api/models/load").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"source\":\"Qwen/Qwen3-0.6B\",\"settings\":{\"backend.ctx\":\"-5\"}}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error", containsString("backend.ctx")));
        mvc.perform(post("/api/models/load").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"source\":\"Qwen/Qwen3-0.6B\",\"settings\":{\"model.repo\":\"x/y\"}}"))
                .andExpect(status().isBadRequest());
        mvc.perform(get("/api/jobs").header("Authorization", AUTH)).andExpect(status().isOk());
    }
}
