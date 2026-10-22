package org.robomiri.jllm.web.video;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.robomiri.jllm.diffusion.Devices;
import org.robomiri.jllm.diffusion.Imaging;
import org.robomiri.jllm.diffusion.VideoEncoder;
import org.robomiri.jllm.diffusion.VideoPipeline;
import org.robomiri.jllm.json.Json;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.containsString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * The video API and the {@code generate_video} tool on a tiny random AnimateDiff pipeline (the engine's test fixtures, on the
 * CPU): generations in both response formats and every file format, streaming, frames for the editor, the encode endpoint, the
 * files endpoint, the models list, the tool, and the errors.
 */
@SpringBootTest(properties = {"jllm.web.settings-file=target/test-settings/video-web.properties",
        "jllm.web.tools-file=target/test-settings/video-tools.json", "jllm.web.api-key=secret",
        "jllm.web.kv-cache-dir=target/test-settings/video-kv"})
@AutoConfigureMockMvc
class VideosApiTest {

    private static final String AUTH = "Bearer secret";
    private static final Path BASE = Path.of("../src/test/resources/diffusion/tiny-sd");
    private static final Path ADAPTER = Path.of("../src/test/resources/diffusion/tiny-motion-adapter");

    @Autowired
    MockMvc mvc;
    @Autowired
    VideoService videos;

    private static boolean registered;

    private synchronized void model() throws Exception {
        if (!registered) {
            videos.register("tiny-video", "test/tiny-motion-adapter", "test/tiny-sd", new VideoPipeline(BASE, ADAPTER, Devices.cpu()), Map.of(), 0);
            registered = true;
        }
    }

    @AfterAll
    static void done() {
        registered = false;
    }

    private static String req(Object... kv) {
        return Json.write(Json.obj(kv));
    }

    private static boolean ffmpeg() {
        return VideoEncoder.resolveFfmpeg("auto") != null;
    }

    @Test
    void noModelLoaded() throws Exception {
        if (videos.loaded().isEmpty()) {
            mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                            .content("{\"prompt\":\"a cat\"}"))
                    .andExpect(status().isNotFound()).andExpect(jsonPath("$.error.code").value("model_not_found"));
        }
    }

    @Test
    void generateWebpGifAndFrames() throws Exception {
        model();
        MvcResult r = mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content(req("model", "tiny-video", "prompt", "a balloon over a hill", "n", 2, "size", "64x64", "num_frames", 4, "fps", 8, "steps", 2,
                                "guidance", 4, "seed", 5, "output_format", "webp", "include_frames", "jpeg")))
                .andExpect(status().isOk()).andExpect(jsonPath("$.data.length()").value(2))
                .andExpect(jsonPath("$.data[0].seed").value(5)).andExpect(jsonPath("$.data[1].seed").value(6))
                .andExpect(jsonPath("$.data[0].format").value("webp")).andExpect(jsonPath("$.data[0].frames").value(4))
                .andExpect(jsonPath("$.data[0].width").value(64)).andExpect(jsonPath("$.data[0].duration").value(0.5))
                .andExpect(jsonPath("$.data[0].frame_images.length()").value(4)).andExpect(jsonPath("$.usage.frames").value(8))
                .andReturn();
        Map<String, Object> body = Json.parseObject(r.getResponse().getContentAsString());
        Object first = ((List<?>) body.get("data")).get(0);
        byte[] webp = Base64.getDecoder().decode(Json.str(first, "b64_json", ""));
        assertEquals("RIFF", new String(webp, 0, 4, StandardCharsets.US_ASCII));
        assertEquals("WEBP", new String(webp, 8, 4, StandardCharsets.US_ASCII));
        byte[] frame = Base64.getDecoder().decode((String) ((List<?>) Json.list(first, "frame_images")).get(0));
        assertEquals(64, Imaging.decode(frame).width());
        // the same request again gives the same clip (seeded noise)
        MvcResult again = mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content(req("prompt", "a balloon over a hill", "size", "64x64", "num_frames", 4, "steps", 2, "guidance", 4, "seed", 5,
                                "output_format", "webp")))
                .andExpect(status().isOk()).andReturn();
        assertEquals(Json.str(first, "b64_json", ""),
                Json.str(((List<?>) Json.parseObject(again.getResponse().getContentAsString()).get("data")).get(0), "b64_json", ""));
        // a GIF, with the default scheduler (ddim) and with euler
        for (String sch : new String[]{"ddim", "euler"}) {
            MvcResult g = mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                            .content(req("prompt", "waves", "size", "64x64", "num_frames", 3, "steps", 2, "seed", 1, "scheduler", sch, "output_format", "gif")))
                    .andExpect(status().isOk()).andExpect(jsonPath("$.data[0].format").value("gif")).andExpect(jsonPath("$.data[0].mime_type").value("image/gif"))
                    .andReturn();
            byte[] gif = Base64.getDecoder().decode(Json.str(((List<?>) Json.parseObject(g.getResponse().getContentAsString()).get("data")).get(0), "b64_json", ""));
            assertEquals("GIF89a", new String(gif, 0, 6, StandardCharsets.US_ASCII));
        }
        if (ffmpeg()) {
            MvcResult m = mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                            .content(req("prompt", "waves", "size", "64x64", "num_frames", 3, "steps", 2, "seed", 1, "output_format", "mp4")))
                    .andExpect(status().isOk()).andExpect(jsonPath("$.data[0].format").value("mp4")).andReturn();
            byte[] mp4 = Base64.getDecoder().decode(Json.str(((List<?>) Json.parseObject(m.getResponse().getContentAsString()).get("data")).get(0), "b64_json", ""));
            assertEquals("ftyp", new String(mp4, 4, 4, StandardCharsets.US_ASCII));
        }
    }

    @Test
    void urlAndFiles() throws Exception {
        model();
        MvcResult url = mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content(req("prompt", "a dog", "size", "64x64", "num_frames", 2, "steps", 1, "response_format", "url", "seed", 1, "output_format", "webp")))
                .andExpect(status().isOk()).andReturn();
        String u = Json.str(((List<?>) Json.parseObject(url.getResponse().getContentAsString()).get("data")).get(0), "url", "");
        assertTrue(u.contains("/v1/videos/files/") && u.endsWith(".webp"), u);
        String path = u.substring(u.indexOf("/v1/"));
        // the clip is fetched without a key (a <video> or <img> tag cannot send one)
        MvcResult file = mvc.perform(get(path)).andExpect(status().isOk()).andExpect(content().contentType("image/webp")).andReturn();
        assertEquals("RIFF", new String(file.getResponse().getContentAsByteArray(), 0, 4, StandardCharsets.US_ASCII));
        mvc.perform(get("/v1/videos/files/00ff.webp")).andExpect(status().isNotFound());
    }

    @Test
    void streaming() throws Exception {
        model();
        MvcResult r = mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content(req("prompt", "a cat", "size", "64x64", "num_frames", 3, "steps", 3, "guidance", 3, "stream", true, "seed", 2, "output_format", "gif")))
                .andExpect(status().isOk()).andReturn();
        String text = r.getResponse().getContentAsString();
        assertTrue(text.contains("video_generation.progress"), text);
        assertTrue(text.contains("video_generation.partial_image"), "previews");
        assertTrue(text.contains("video_generation.decoding"), "decoding progress");
        assertTrue(text.contains("video_generation.completed"), "completed");
        assertTrue(text.trim().endsWith("[DONE]"), "ends with [DONE]");
        int progress = text.split("video_generation.progress", -1).length - 1;
        assertTrue(progress >= 3, "one progress event per step plus the start: " + progress);
    }

    @Test
    void encodeEndpoint() throws Exception {
        model();
        List<Object> frames = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            byte[] rgb = new byte[48 * 32 * 3];
            for (int p = 0; p < rgb.length; p++) {
                rgb[p] = (byte) (i * 40 + p % 50);
            }
            frames.add(Base64.getEncoder().encodeToString(Imaging.png(new Imaging.Image(48, 32, rgb))));
        }
        MvcResult r = mvc.perform(post("/v1/videos/encode").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content(req("frames", frames, "fps", 10, "output_format", "webp")))
                .andExpect(status().isOk()).andExpect(jsonPath("$.data[0].frames").value(5)).andExpect(jsonPath("$.data[0].fps").value(10.0))
                .andExpect(jsonPath("$.data[0].width").value(48)).andExpect(jsonPath("$.data[0].format").value("webp")).andReturn();
        byte[] webp = Base64.getDecoder().decode(Json.str(((List<?>) Json.parseObject(r.getResponse().getContentAsString()).get("data")).get(0), "b64_json", ""));
        assertEquals("WEBP", new String(webp, 8, 4, StandardCharsets.US_ASCII));
        mvc.perform(post("/v1/videos/encode").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON).content(req("frames", List.of())))
                .andExpect(status().isBadRequest());
        mvc.perform(post("/v1/videos/encode").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content(req("frames", List.of("not base64 !!"), "output_format", "gif")))
                .andExpect(status().isBadRequest());
        mvc.perform(post("/v1/videos/encode").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content(req("frames", frames, "fps", 0)))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.message", containsString("fps")));
    }

    @Test
    void validation() throws Exception {
        model();
        String ok = "\"size\":\"64x64\"";
        mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON).content("{}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.message", containsString("prompt")));
        mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"x\",\"size\":\"500x500\"}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.message", containsString("multiple of")));
        mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"x\"," + ok + ",\"num_frames\":99}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.message", containsString("num_frames")));
        mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"x\"," + ok + ",\"n\":99}"))
                .andExpect(status().isBadRequest());
        mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"x\"," + ok + ",\"scheduler\":\"nope\"}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.message", containsString("scheduler")));
        mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"x\"," + ok + ",\"output_format\":\"avi\"}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.message", containsString("format")));
        mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"x\"," + ok + ",\"fps\":500}"))
                .andExpect(status().isBadRequest());
        mvc.perform(post("/v1/videos/generations").contentType(MediaType.APPLICATION_JSON).content("{\"prompt\":\"x\"}"))
                .andExpect(status().isUnauthorized());
        mvc.perform(post("/v1/videos/generations").header("Authorization", AUTH).contentType(MediaType.TEXT_PLAIN).content("x"))
                .andExpect(status().isUnsupportedMediaType());
    }

    @Test
    void modelsAndTool() throws Exception {
        model();
        mvc.perform(get("/v1/models").header("Authorization", AUTH)).andExpect(status().isOk())
                .andExpect(jsonPath("$.data[?(@.id == 'tiny-video')].kind").value("video"));
        mvc.perform(get("/api/models/loaded").header("Authorization", AUTH)).andExpect(status().isOk())
                .andExpect(jsonPath("$[?(@.id == 'tiny-video')].kind").value("video"))
                .andExpect(jsonPath("$[?(@.id == 'tiny-video')].defaults.max_frames").exists());
        mvc.perform(get("/api/tools").header("Authorization", AUTH)).andExpect(status().isOk())
                .andExpect(jsonPath("$.tools[?(@.name == 'generate_video')]").exists());
        MvcResult r = mvc.perform(post("/api/tools/call").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"generate_video\",\"arguments\":{\"prompt\":\"a lighthouse\",\"size\":\"64x64\",\"num_frames\":3,\"steps\":2,\"seed\":9}}"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.error").value(false)).andReturn();
        String text = Json.str(Json.parseObject(r.getResponse().getContentAsString()), "content", "");
        assertTrue(text.contains("![generated video](/v1/videos/files/"), text);
        String link = text.substring(text.indexOf("(/v1/") + 1, text.indexOf(")", text.indexOf("(/v1/")));
        mvc.perform(get(link)).andExpect(status().isOk());
        mvc.perform(post("/api/tools/call").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"generate_video\",\"arguments\":{\"prompt\":\"x\",\"size\":\"65x65\"}}"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.error").value(true)).andExpect(jsonPath("$.content", containsString("multiple of")));
    }
}
