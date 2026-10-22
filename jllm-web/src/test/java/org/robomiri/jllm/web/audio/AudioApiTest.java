package org.robomiri.jllm.web.audio;

import org.junit.jupiter.api.Test;
import org.robomiri.jllm.audio.AudioModels;
import org.robomiri.jllm.audio.Wav;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.settings.SettingsService;
import org.robomiri.jllm.web.tools.ToolRegistry;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.hamcrest.Matchers.containsString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * The audio API of the console on a tiny random Whisper (the engine's test fixture): the OpenAI endpoints, the clip
 * store, the model listings and the chat tools {@code transcribe_audio} and {@code speak}.
 */
@SpringBootTest(properties = {"jllm.web.settings-file=target/test-settings/audio-web.properties",
        "jllm.web.tools-file=target/test-settings/audio-tools.json", "jllm.web.kv-cache-dir=target/test-settings/audio-kv"})
@AutoConfigureMockMvc
class AudioApiTest {

    static final Path FIXTURE = Path.of("../src/test/resources/audio/tiny-whisper");

    @Autowired
    MockMvc mvc;
    @Autowired
    AudioService audio;
    @Autowired
    SettingsService settings;
    @Autowired
    ToolRegistry tools;

    private static byte[] wav(double seconds) {
        float[] x = new float[(int) (seconds * 16000)];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) (0.3 * Math.sin(2 * Math.PI * 440 * i / 16000.0));
        }
        return Wav.write16(x, 16000);
    }

    @Test
    void audioApi() throws Exception {
        // nothing loaded: the OpenAI error shape
        mvc.perform(get("/api/audio/models")).andExpect(status().isOk()).andExpect(content().string("[]"));
        mvc.perform(multipart("/v1/audio/transcriptions").file(new MockMultipartFile("file", "a.wav", "audio/wav", wav(1))))
                .andExpect(status().isNotFound()).andExpect(jsonPath("$.error.code").value("model_not_found"))
                .andExpect(jsonPath("$.error.message", containsString("no speech-to-text model is loaded")));
        mvc.perform(post("/v1/audio/speech").contentType(MediaType.APPLICATION_JSON).content("{\"input\":\"hi\"}"))
                .andExpect(status().isNotFound()).andExpect(jsonPath("$.error.message", containsString("no text-to-speech model")));
        mvc.perform(post("/v1/audio/speech").contentType(MediaType.APPLICATION_JSON).content("{nope"))
                .andExpect(status().isBadRequest());

        AudioModels.Entry e = audio.load(settings.current(), "tiny-stt", "tiny-stt", FIXTURE);
        assertEquals(AudioModels.Kind.SPEECH_TO_TEXT, e.kind());
        try {
            mvc.perform(get("/api/audio/models")).andExpect(status().isOk()).andExpect(jsonPath("$[0].id").value("tiny-stt"))
                    .andExpect(jsonPath("$[0].kind").value("speech-to-text")).andExpect(jsonPath("$[0].multilingual").value(false));
            mvc.perform(get("/v1/models")).andExpect(status().isOk()).andExpect(jsonPath("$.data[0].id").value("tiny-stt"))
                    .andExpect(jsonPath("$.data[0].kind").value("speech-to-text"));

            MockMultipartFile file = new MockMultipartFile("file", "a.wav", "audio/wav", wav(1.5));
            mvc.perform(multipart("/v1/audio/transcriptions").file(file).param("model", "whisper-1"))
                    .andExpect(status().isOk()).andExpect(jsonPath("$.text").exists());
            mvc.perform(multipart("/v1/audio/transcriptions").file(file).param("response_format", "verbose_json")
                            .param("timestamp_granularities[]", "word").param("timestamp_granularities[]", "segment"))
                    .andExpect(status().isOk()).andExpect(jsonPath("$.task").value("transcribe"))
                    .andExpect(jsonPath("$.duration").value(1.5)).andExpect(jsonPath("$.segments").isArray());
            mvc.perform(multipart("/v1/audio/transcriptions").file(file).param("response_format", "srt"))
                    .andExpect(status().isOk()).andExpect(content().contentTypeCompatibleWith("text/plain"));
            mvc.perform(multipart("/v1/audio/transcriptions").file(file).param("response_format", "vtt"))
                    .andExpect(status().isOk()).andExpect(content().string(containsString("WEBVTT")));
            mvc.perform(multipart("/v1/audio/transcriptions").file(file).param("stream", "true"))
                    .andExpect(status().isOk()).andExpect(content().contentTypeCompatibleWith("text/event-stream"))
                    .andExpect(content().string(containsString("\"type\":\"transcript.text.done\"")))
                    .andExpect(content().string(containsString("data: [DONE]")));
            mvc.perform(multipart("/v1/audio/transcriptions").file(file).param("response_format", "xml"))
                    .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.type").value("invalid_request_error"));
            mvc.perform(multipart("/v1/audio/transcriptions").param("model", "x")).andExpect(status().isBadRequest());
            mvc.perform(multipart("/v1/audio/translations").file(file)).andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.error.message", containsString("English-only")));
            mvc.perform(multipart("/v1/audio/transcriptions").file(new MockMultipartFile("file", "x.mp3", "audio/mpeg",
                    "ID3 not a real mp3".getBytes(StandardCharsets.UTF_8)))).andExpect(status().isBadRequest());

            // the clip store
            MvcResult up = mvc.perform(post("/api/audio/files").param("label", "mine").contentType("audio/wav").content(wav(0.5)))
                    .andExpect(status().isOk()).andExpect(jsonPath("$.seconds").value(0.5)).andReturn();
            String id = (String) Json.parseObject(up.getResponse().getContentAsString()).get("id");
            byte[] back = mvc.perform(get("/api/audio/files/" + id)).andExpect(status().isOk())
                    .andExpect(content().contentTypeCompatibleWith("audio/wav")).andReturn().getResponse().getContentAsByteArray();
            assertEquals(8000, Wav.read(back).length());
            mvc.perform(post("/api/audio/files").contentType("audio/wav").content("garbage".getBytes(StandardCharsets.UTF_8)))
                    .andExpect(status().isBadRequest());

            // the tools
            var names = tools.describe().stream().map(ToolRegistry.Info::name).toList();
            assertTrue(names.contains("transcribe_audio") && names.contains("speak"), "audio tools are registered: " + names);
            var out = tools.call(tools.find("transcribe_audio"), Map.of("audio_id", id));
            assertTrue(!out.error(), "transcribe_audio on a stored clip: " + out.content());
            var spoken = tools.call(tools.find("speak"), Map.of("text", "hello"));
            assertTrue(spoken.error() && spoken.content().contains("no text-to-speech model"), spoken.content());
            var missing = tools.call(tools.find("transcribe_audio"), Map.of("audio_id", "nope"));
            assertTrue(missing.error(), "an unknown clip is a tool error the model can read: " + missing.content());
            mvc.perform(delete("/api/audio/files/" + id)).andExpect(jsonPath("$.deleted").value(true));
            mvc.perform(get("/api/audio/files/" + id)).andExpect(status().isNotFound());

            // unloading through the models endpoint
            mvc.perform(delete("/api/models/loaded").param("id", "tiny-stt")).andExpect(status().isOk());
            mvc.perform(get("/api/audio/models")).andExpect(content().string("[]"));
        } finally {
            if (audio.has("tiny-stt")) {
                audio.unload("tiny-stt");
            }
        }
    }
}
