package org.robomiri.jllm.web.video;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.diffusion.ImageApi;
import org.robomiri.jllm.diffusion.VideoApi;
import org.robomiri.jllm.diffusion.VideoStore;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.chat.ChatSupport;
import org.robomiri.jllm.web.chat.ClientGone;
import org.robomiri.jllm.web.settings.SettingsService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * The video API: {@code POST /v1/videos/generations}, {@code POST /v1/videos/encode} and
 * {@code GET /v1/videos/files/<id>.<ext>}; the semantics, the extra fields and the streaming events are those of
 * {@link VideoApi}. The {@code model} field picks a loaded video model (the only one when it is omitted).
 */
@RestController
public class VideoGenerationController {

    private final VideoService videos;
    private final SettingsService settings;

    public VideoGenerationController(VideoService videos, SettingsService settings) {
        this.videos = videos;
        this.settings = settings;
    }

    @PostMapping("/v1/videos/generations")
    public void generations(HttpServletRequest req, HttpServletResponse res) throws IOException {
        handle(req, res, false);
    }

    @PostMapping("/v1/videos/encode")
    public void encode(HttpServletRequest req, HttpServletResponse res) throws IOException {
        handle(req, res, true);
    }

    @GetMapping("/v1/videos/files/{name}")
    public void file(@PathVariable String name, HttpServletResponse res) throws IOException {
        int dot = name.lastIndexOf('.');
        String id = dot >= 0 ? name.substring(0, dot) : name;
        VideoStore.Clip clip = videos.store().get(id);
        if (clip == null) {
            ChatSupport.send(res, 404, ChatSupport.error("no such video (it may have been evicted)", "invalid_request_error", "not_found"));
            return;
        }
        res.setHeader("Cache-Control", "private, max-age=3600");
        res.setContentType(clip.mime());
        res.setContentLength(clip.data().length);
        res.getOutputStream().write(clip.data());
    }

    private void handle(HttpServletRequest req, HttpServletResponse res, boolean encode) throws IOException {
        Settings s = settings.current();
        if (!s.getBool("video.enabled")) {
            ChatSupport.send(res, 404, ChatSupport.error("video generation is switched off (video.enabled=false)", "invalid_request_error", "disabled"));
            return;
        }
        Map<String, Object> body;
        String ct = req.getContentType();
        try {
            if (!ChatSupport.isJson(ct)) {
                ChatSupport.send(res, 415, ChatSupport.error("Content-Type must be application/json", "invalid_request_error", null));
                return;
            }
            body = Json.parseObject(new String(req.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8));
        } catch (RuntimeException e) {
            ChatSupport.send(res, 400, ChatSupport.error("invalid request body: " + e.getMessage(), "invalid_request_error", null));
            return;
        }
        VideoApi.Config cfg = videos.config(s.get("video.ffmpeg"), s.get("video.format"));
        String base = req.getScheme() + "://" + req.getHeader("Host");
        if (encode) { // pictures to a file: needs no model, only the encoders
            try {
                ChatSupport.send(res, 200, Json.write(VideoApi.encode("jllm-video-encoder", body, videos.store(), base, cfg)));
            } catch (ImageApi.ApiException e) {
                ChatSupport.send(res, e.status, ChatSupport.error(e.getMessage(), e.status == 400 ? "invalid_request_error" : "server_error", e.code));
            } catch (RuntimeException e) {
                e.printStackTrace();
                ChatSupport.send(res, 500, ChatSupport.error("encoding failed: " + e.getMessage(), "server_error", null));
            }
            return;
        }
        VideoService.Loaded model;
        try {
            model = videos.resolve(Json.str(body, "model", null));
        } catch (NoSuchElementException e) {
            ChatSupport.send(res, 404, ChatSupport.error(e.getMessage(), "invalid_request_error", "model_not_found"));
            return;
        }
        java.util.function.BooleanSupplier gone = ClientGone.of(req);
        boolean stream = Json.bool(body, "stream", false);
        ChatSupport.Sse sse = stream ? new ChatSupport.Sse(res) : null;
        try (VideoService.Lease lease = model.lease()) {
            ImageApi.Events events = sse == null ? new ImageApi.Events() {
                @Override
                public void send(Map<String, Object> event) {
                }

                @Override
                public boolean cancelled() {
                    return gone.getAsBoolean() || model.closing();
                }
            } : new ImageApi.Events() {
                @Override
                public void send(Map<String, Object> event) throws IOException {
                    sse.event(event);
                }

                @Override
                public boolean cancelled() {
                    return sse.failed() || gone.getAsBoolean() || model.closing();
                }
            };
            Map<String, Object> out = VideoApi.generate(model.pipeline(), model.id(), body, videos.store(), base, events, cfg);
            Object data = out.get("data");
            model.countVideos(data instanceof java.util.List<?> l ? l.size() : 0);
            if (sse == null) {
                ChatSupport.send(res, 200, Json.write(out));
            } else {
                sse.event(Json.obj("type", "video_generation.done", "usage", out.get("usage"), "timing", out.get("timing"), "model", model.id()));
                sse.raw("data: [DONE]\n\n");
            }
        } catch (ImageApi.ApiException e) {
            fail(res, sse, e.status, e.getMessage(), e.status == 400 ? "invalid_request_error" : "server_error", e.code);
        } catch (IllegalStateException e) { // being unloaded
            fail(res, sse, 503, e.getMessage(), "server_error", "model_unavailable");
        } catch (OutOfMemoryError e) {
            fail(res, sse, 507, "out of memory generating this clip: use a smaller size or fewer frames, or restart with a larger heap (JLLM_XMX)", "server_error", "out_of_memory");
        } catch (RuntimeException e) {
            e.printStackTrace();
            fail(res, sse, 500, "video generation failed: " + e.getMessage(), "server_error", null);
        }
    }

    private static void fail(HttpServletResponse res, ChatSupport.Sse sse, int status, String message, String type, String code) throws IOException {
        if (sse != null && sse.started()) {
            sse.event(Json.obj("type", "error", "error", Json.obj("message", message, "type", type)));
            sse.raw("data: [DONE]\n\n");
        } else {
            ChatSupport.send(res, status, ChatSupport.error(message, type, code));
        }
    }
}
