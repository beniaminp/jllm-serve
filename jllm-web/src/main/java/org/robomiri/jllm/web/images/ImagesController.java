package org.robomiri.jllm.web.images;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.Part;
import org.robomiri.jllm.diffusion.ImageApi;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.chat.ChatSupport;
import org.robomiri.jllm.web.chat.ClientGone;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * The OpenAI-compatible image API: {@code POST /v1/images/generations}, {@code POST /v1/images/edits} (JSON or multipart)
 * and {@code GET /v1/images/files/<id>.png}; the semantics, the extra fields and the streaming events are those of
 * {@link ImageApi}. The {@code model} field picks a loaded image model (the only one when it is omitted).
 */
@RestController
public class ImagesController {

    private final ImageService images;

    public ImagesController(ImageService images) {
        this.images = images;
    }

    @PostMapping("/v1/images/generations")
    public void generations(HttpServletRequest req, HttpServletResponse res) throws IOException {
        handle(req, res, false);
    }

    @PostMapping("/v1/images/edits")
    public void edits(HttpServletRequest req, HttpServletResponse res) throws IOException {
        handle(req, res, true);
    }

    @GetMapping("/v1/images/files/{name}")
    public void file(@PathVariable String name, HttpServletResponse res) throws IOException {
        String id = name.endsWith(".png") ? name.substring(0, name.length() - 4) : name;
        byte[] png = images.store().get(id);
        if (png == null) {
            ChatSupport.send(res, 404, ChatSupport.error("no such image (it may have been evicted)", "invalid_request_error", "not_found"));
            return;
        }
        res.setHeader("Cache-Control", "private, max-age=3600");
        res.setContentType("image/png");
        res.setContentLength(png.length);
        res.getOutputStream().write(png);
    }

    private void handle(HttpServletRequest req, HttpServletResponse res, boolean edit) throws IOException {
        Map<String, Object> body = new HashMap<>();
        byte[] image = null;
        byte[] mask = null;
        String ct = req.getContentType();
        try {
            if (ct != null && ct.toLowerCase(java.util.Locale.ROOT).startsWith("multipart/form-data")) {
                for (Part p : req.getParts()) {
                    String name = p.getName().endsWith("[]") ? p.getName().substring(0, p.getName().length() - 2) : p.getName();
                    byte[] data = p.getInputStream().readAllBytes();
                    if (p.getSubmittedFileName() != null) {
                        if (name.equals("image") && image == null) {
                            image = data;
                        } else if (name.equals("mask") && mask == null) {
                            mask = data;
                        }
                    } else {
                        body.put(name, field(name, new String(data, java.nio.charset.StandardCharsets.UTF_8)));
                    }
                }
                for (Map.Entry<String, String[]> e : req.getParameterMap().entrySet()) { // text fields a container exposes as parameters
                    if (!body.containsKey(e.getKey()) && e.getValue().length > 0) {
                        body.put(e.getKey(), field(e.getKey(), e.getValue()[0]));
                    }
                }
            } else if (ChatSupport.isJson(ct)) {
                body = Json.parseObject(new String(req.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8));
            } else {
                ChatSupport.send(res, 415, ChatSupport.error("Content-Type must be application/json or multipart/form-data", "invalid_request_error", null));
                return;
            }
        } catch (jakarta.servlet.ServletException | RuntimeException e) {
            ChatSupport.send(res, 400, ChatSupport.error("invalid request body: " + e.getMessage(), "invalid_request_error", null));
            return;
        }
        ImageService.Loaded model;
        try {
            model = images.resolve(Json.str(body, "model", null));
        } catch (NoSuchElementException e) {
            ChatSupport.send(res, 404, ChatSupport.error(e.getMessage(), "invalid_request_error", "model_not_found"));
            return;
        }
        String base = req.getScheme() + "://" + req.getHeader("Host");
        java.util.function.BooleanSupplier gone = ClientGone.of(req);
        boolean stream = Json.bool(body, "stream", false);
        ChatSupport.Sse sse = stream ? new ChatSupport.Sse(res) : null;
        try (ImageService.Lease lease = model.lease()) {
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
            Map<String, Object> out = ImageApi.run(model.pipeline(), model.id(), body, image, mask, edit, images.store(), base, events);
            Object data = out.get("data");
            model.countImages(data instanceof java.util.List<?> l ? l.size() : 0);
            if (sse == null) {
                ChatSupport.send(res, 200, Json.write(out));
            } else {
                sse.event(Json.obj("type", "image_generation.done", "usage", out.get("usage"), "timing", out.get("timing"), "model", model.id()));
                sse.raw("data: [DONE]\n\n");
            }
        } catch (ImageApi.ApiException e) {
            fail(res, sse, e.status, e.getMessage(), e.status == 400 ? "invalid_request_error" : "server_error", e.code);
        } catch (IllegalStateException e) { // being unloaded
            fail(res, sse, 503, e.getMessage(), "server_error", "model_unavailable");
        } catch (OutOfMemoryError e) {
            fail(res, sse, 507, "out of memory generating this image: use a smaller size or restart with a larger heap (JLLM_XMX)", "server_error", "out_of_memory");
        } catch (RuntimeException e) {
            e.printStackTrace();
            fail(res, sse, 500, "image generation failed: " + e.getMessage(), "server_error", null);
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

    /** Multipart text fields arrive as strings: numbers and booleans become what the JSON form carries. */
    private static Object field(String name, String text) {
        String t = text.strip();
        try {
            return switch (name) {
                case "n", "steps", "seed", "clip_skip", "partial_images" -> Long.parseLong(t);
                case "guidance", "guidance_scale", "strength" -> Double.parseDouble(t);
                case "stream", "preview" -> t.equalsIgnoreCase("true") || t.equals("1");
                default -> text;
            };
        } catch (NumberFormatException e) {
            return text;
        }
    }
}
