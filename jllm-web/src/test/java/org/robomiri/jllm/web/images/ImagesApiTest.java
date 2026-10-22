package org.robomiri.jllm.web.images;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.robomiri.jllm.diffusion.Devices;
import org.robomiri.jllm.diffusion.ImagePipeline;
import org.robomiri.jllm.diffusion.Imaging;
import org.robomiri.jllm.json.Json;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockPart;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.containsString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * The image API and the image tools on a tiny random Stable Diffusion pipeline (the engine's test fixture, on the CPU):
 * generations in both response formats and streaming, edits as JSON and as multipart (image to image and inpainting), the
 * files endpoint, the models list, the tools {@code generate_image} / {@code edit_image}, and the errors.
 */
@SpringBootTest(properties = {"jllm.web.settings-file=target/test-settings/images-web.properties",
        "jllm.web.tools-file=target/test-settings/images-tools.json", "jllm.web.api-key=secret",
        "jllm.web.kv-cache-dir=target/test-settings/images-kv"})
@AutoConfigureMockMvc
class ImagesApiTest {

    private static final String AUTH = "Bearer secret";
    private static final Path FIXTURE = Path.of("../src/test/resources/diffusion/tiny-sd");

    @Autowired
    MockMvc mvc;
    @Autowired
    ImageService images;

    private static boolean registered;

    private synchronized void model() throws Exception {
        if (!registered) {
            images.register("tiny-sd", "test/tiny-sd", new ImagePipeline(FIXTURE, Devices.cpu()), Map.of(), 0);
            registered = true;
        }
    }

    @AfterAll
    static void done() {
        registered = false;
    }

    @Test
    void noModelLoaded() throws Exception {
        if (images.loaded().isEmpty()) {
            mvc.perform(post("/v1/images/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                            .content("{\"prompt\":\"a cat\"}"))
                    .andExpect(status().isNotFound()).andExpect(jsonPath("$.error.code").value("model_not_found"));
        }
    }

    @Test
    void generateB64AndUrl() throws Exception {
        model();
        MvcResult r = mvc.perform(post("/v1/images/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"model\":\"tiny-sd\",\"prompt\":\"a cat on a sofa\",\"n\":2,\"size\":\"64x64\",\"steps\":2,"
                                + "\"guidance\":4,\"seed\":5,\"scheduler\":\"euler_a\"}"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.data.length()").value(2))
                .andExpect(jsonPath("$.data[0].seed").value(5)).andExpect(jsonPath("$.data[1].seed").value(6))
                .andExpect(jsonPath("$.data[0].width").value(64)).andReturn();
        Map<String, Object> body = Json.parseObject(r.getResponse().getContentAsString());
        byte[] png = Base64.getDecoder().decode(Json.str(((List<?>) body.get("data")).get(0), "b64_json", ""));
        Imaging.Image img = Imaging.decode(png);
        assertEquals(64, img.width());
        // the same request again gives the same picture (seeded noise)
        MvcResult again = mvc.perform(post("/v1/images/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"a cat on a sofa\",\"n\":1,\"size\":\"64x64\",\"steps\":2,\"guidance\":4,\"seed\":5,\"scheduler\":\"euler_a\"}"))
                .andExpect(status().isOk()).andReturn();
        assertEquals(Json.str(((List<?>) body.get("data")).get(0), "b64_json", ""),
                Json.str(((List<?>) Json.parseObject(again.getResponse().getContentAsString()).get("data")).get(0), "b64_json", ""));

        MvcResult url = mvc.perform(post("/v1/images/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"a dog\",\"size\":\"64x64\",\"steps\":1,\"response_format\":\"url\",\"seed\":1}"))
                .andExpect(status().isOk()).andReturn();
        String u = Json.str(((List<?>) Json.parseObject(url.getResponse().getContentAsString()).get("data")).get(0), "url", "");
        assertTrue(u.contains("/v1/images/files/") && u.endsWith(".png"), u);
        String path = u.substring(u.indexOf("/v1/"));
        // the picture is fetched without a key (an <img> tag cannot send one)
        MvcResult file = mvc.perform(get(path)).andExpect(status().isOk()).andExpect(content().contentType("image/png")).andReturn();
        assertEquals(64, Imaging.decode(file.getResponse().getContentAsByteArray()).width());
        mvc.perform(get("/v1/images/files/00ff.png")).andExpect(status().isNotFound());
    }

    @Test
    void streaming() throws Exception {
        model();
        MvcResult r = mvc.perform(post("/v1/images/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"a cat\",\"size\":\"64x64\",\"steps\":3,\"guidance\":3,\"stream\":true,\"seed\":2}"))
                .andExpect(status().isOk()).andReturn();
        String text = r.getResponse().getContentAsString();
        assertTrue(text.contains("image_generation.progress"), text);
        assertTrue(text.contains("image_generation.partial_image"), "previews");
        assertTrue(text.contains("image_generation.completed"), "completed");
        assertTrue(text.trim().endsWith("[DONE]"), "ends with [DONE]");
        int progress = text.split("image_generation.progress", -1).length - 1;
        assertTrue(progress >= 3, "one progress event per step plus the start: " + progress);
    }

    @Test
    void edits() throws Exception {
        model();
        Imaging.Image src = new Imaging.Image(64, 64, new byte[64 * 64 * 3]);
        byte[] png = Imaging.png(src);
        byte[] maskPng = Imaging.png(new Imaging.Image(64, 64, maskBytes()));
        // image to image as JSON
        mvc.perform(post("/v1/images/edits").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content(Json.write(Json.obj("prompt", "a red house", "image", Base64.getEncoder().encodeToString(png), "steps", 4,
                                "strength", 0.5, "guidance", 3, "seed", 3))))
                .andExpect(status().isOk()).andExpect(jsonPath("$.data[0].width").value(64));
        // inpainting as multipart, the way the OpenAI SDKs send it
        mvc.perform(multipart("/v1/images/edits").part(new MockPart("image", "image.png", png))
                        .part(new MockPart("mask", "mask.png", maskPng))
                        .param("prompt", "a blue door").param("steps", "3").param("seed", "4").param("size", "64x64")
                        .param("guidance", "3").header("Authorization", AUTH))
                .andExpect(status().isOk()).andExpect(jsonPath("$.data[0].b64_json").exists());
        mvc.perform(post("/v1/images/edits").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"x\"}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.message", containsString("image is required")));
    }

    private static byte[] maskBytes() {
        byte[] m = new byte[64 * 64 * 3];
        for (int y = 16; y < 48; y++) {
            for (int x = 16; x < 48; x++) {
                m[(y * 64 + x) * 3] = m[(y * 64 + x) * 3 + 1] = m[(y * 64 + x) * 3 + 2] = (byte) 255;
            }
        }
        return m;
    }

    @Test
    void validation() throws Exception {
        model();
        mvc.perform(post("/v1/images/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON).content("{}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.message", containsString("prompt")));
        mvc.perform(post("/v1/images/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"x\",\"size\":\"500x500\"}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.message", containsString("multiple of")));
        mvc.perform(post("/v1/images/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"x\",\"n\":99}"))
                .andExpect(status().isBadRequest());
        mvc.perform(post("/v1/images/generations").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"x\",\"scheduler\":\"nope\"}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error.message", containsString("scheduler")));
        mvc.perform(post("/v1/images/generations").contentType(MediaType.APPLICATION_JSON).content("{\"prompt\":\"x\"}"))
                .andExpect(status().isUnauthorized());
        mvc.perform(post("/v1/images/generations").header("Authorization", AUTH).contentType(MediaType.TEXT_PLAIN).content("x"))
                .andExpect(status().isUnsupportedMediaType());
    }

    @Test
    void modelsAndTools() throws Exception {
        model();
        mvc.perform(get("/v1/models").header("Authorization", AUTH)).andExpect(status().isOk())
                .andExpect(jsonPath("$.data[?(@.id == 'tiny-sd')].kind").value("image"));
        mvc.perform(get("/api/models/loaded").header("Authorization", AUTH)).andExpect(status().isOk())
                .andExpect(jsonPath("$[?(@.id == 'tiny-sd')].kind").value("image"))
                .andExpect(jsonPath("$[?(@.id == 'tiny-sd')].defaults.steps").exists());
        mvc.perform(get("/api/tools").header("Authorization", AUTH)).andExpect(status().isOk())
                .andExpect(jsonPath("$.tools[?(@.name == 'generate_image')]").exists())
                .andExpect(jsonPath("$.tools[?(@.name == 'edit_image')]").exists());
        MvcResult r = mvc.perform(post("/api/tools/call").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"generate_image\",\"arguments\":{\"prompt\":\"a lighthouse\",\"size\":\"64x64\",\"steps\":2,\"seed\":9}}"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.error").value(false)).andReturn();
        String text = Json.str(Json.parseObject(r.getResponse().getContentAsString()), "content", "");
        assertTrue(text.contains("![generated image 1](/v1/images/files/"), text);
        String link = text.substring(text.indexOf("(/v1/") + 1, text.indexOf(".png)") + 4);
        mvc.perform(get(link)).andExpect(status().isOk());
        // edit it: the whole picture, then a region
        mvc.perform(post("/api/tools/call").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content(Json.write(Json.obj("name", "edit_image", "arguments",
                                Json.obj("image", link, "prompt", "at night", "steps", 4, "strength", 0.6, "seed", 1)))))
                .andExpect(status().isOk()).andExpect(jsonPath("$.error").value(false))
                .andExpect(jsonPath("$.content", containsString("/v1/images/files/")));
        mvc.perform(post("/api/tools/call").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content(Json.write(Json.obj("name", "edit_image", "arguments",
                                Json.obj("image", link, "prompt", "a boat", "steps", 3, "region", Json.obj("x", 0.25, "y", 0.25, "width", 0.5, "height", 0.5))))))
                .andExpect(status().isOk()).andExpect(jsonPath("$.error").value(false));
        mvc.perform(post("/api/tools/call").header("Authorization", AUTH).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"edit_image\",\"arguments\":{\"image\":\"/v1/images/files/ffffffffffffffffffffffff.png\",\"prompt\":\"x\"}}"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.error").value(true))
                .andExpect(jsonPath("$.content", containsString("no longer available")));
    }
}
