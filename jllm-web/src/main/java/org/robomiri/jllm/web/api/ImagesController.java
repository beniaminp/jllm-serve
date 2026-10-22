package org.robomiri.jllm.web.api;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.chat.ChatSupport;
import org.robomiri.jllm.web.chat.ImageStore;
import org.robomiri.jllm.web.settings.SettingsService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.io.InputStream;

/**
 * {@code POST /api/images}: the body is a picture file (PNG, JPEG, GIF, BMP). It is kept for the {@code describe_image}
 * tool, which names it by the {@code id} returned here (docs/20).
 */
@RestController
public class ImagesController {

    private final ImageStore store;
    private final SettingsService settings;

    public ImagesController(ImageStore store, SettingsService settings) {
        this.store = store;
        this.settings = settings;
    }

    @PostMapping("/api/images")
    public void upload(HttpServletRequest req, HttpServletResponse res) throws IOException {
        Settings s = settings.current();
        long max = s.getLong("vision.max-image-bytes");
        byte[] bytes;
        try (InputStream in = req.getInputStream()) {
            bytes = in.readNBytes((int) Math.min(max + 1, Integer.MAX_VALUE - 8));
        }
        if (bytes.length > max) {
            ChatSupport.send(res, 413, ChatSupport.error("the picture is larger than the limit of " + max + " bytes (vision.max-image-bytes)",
                    "invalid_request_error", null));
            return;
        }
        try {
            String id = store.put(bytes, s.getLong("vision.max-image-pixels"));
            ChatSupport.send(res, 200, Json.write(Json.obj("id", id, "bytes", bytes.length)));
        } catch (IOException e) {
            ChatSupport.send(res, 400, ChatSupport.error(e.getMessage(), "invalid_request_error", null));
        }
    }
}
