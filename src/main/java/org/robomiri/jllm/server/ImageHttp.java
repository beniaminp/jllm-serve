package org.robomiri.jllm.server;

import org.robomiri.jllm.diffusion.ImageApi;
import org.robomiri.jllm.diffusion.ImagePipeline;
import org.robomiri.jllm.diffusion.ImageStore;
import org.robomiri.jllm.diffusion.Multipart;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The image routes of the engine's own HTTP server: {@code POST /v1/images/generations}, {@code POST /v1/images/edits}
 * (JSON, or multipart as the OpenAI SDKs send it) and {@code GET /v1/images/files/<id>.png} (the pictures that
 * {@code "response_format": "url"} points at), all on top of {@link ImageApi}.
 */
public final class ImageHttp {

    private final ImagePipeline pipeline;
    private final String modelId;
    private final ImageStore store = new ImageStore(64, 256L << 20);

    public ImageHttp(ImagePipeline pipeline, String modelId) {
        this.pipeline = pipeline;
        this.modelId = modelId;
    }

    public String modelId() {
        return modelId;
    }

    public ImagePipeline pipeline() {
        return pipeline;
    }

    /** The {@code /v1/models} entry of the image model. */
    public Map<String, Object> modelObject() {
        return Json.obj("id", modelId, "object", "model", "owned_by", "jllm-serve", "kind", "image",
                "description", pipeline.description(), "defaults", defaults(pipeline), "size_multiple", pipeline.sizeMultiple(),
                "inpainting_unet", pipeline.hasInpaintUnet());
    }

    public static Map<String, Object> defaults(ImagePipeline p) {
        return Json.obj("steps", p.defaults.steps(), "guidance", p.defaults.guidance(), "size", p.defaults.size(),
                "scheduler", p.defaults.scheduler(), "distilled", p.defaults.distilled());
    }

    /** @return true if the path was one of ours and has been answered */
    public boolean handle(String path, HttpServer.Request req, HttpServer.Response res) throws IOException {
        if (path.startsWith("/v1/images/files/")) {
            String id = path.substring("/v1/images/files/".length());
            if (id.endsWith(".png")) {
                id = id.substring(0, id.length() - 4);
            }
            byte[] png = store.get(id);
            if (png == null) {
                res.send(404, "application/json", error("no such image (it may have been evicted)", "not_found"));
            } else {
                res.header("Cache-Control", "private, max-age=3600").send(200, "image/png", png);
            }
            return true;
        }
        boolean gen = path.equals("/v1/images/generations");
        boolean edit = path.equals("/v1/images/edits");
        if (!gen && !edit) {
            return false;
        }
        if (!req.method.equals("POST")) {
            res.send(405, "application/json", error("use POST", "invalid_request_error"));
            return true;
        }
        Map<String, Object> body = new HashMap<>();
        byte[] image = null;
        byte[] mask = null;
        String ct = req.headers.get("content-type");
        try {
            String boundary = Multipart.boundary(ct);
            if (boundary != null) {
                for (Multipart.Part p : Multipart.parse(req.body, boundary).values()) {
                    String name = p.name().endsWith("[]") ? p.name().substring(0, p.name().length() - 2) : p.name();
                    if (p.filename() != null) {
                        if (name.equals("image") && image == null) {
                            image = p.data();
                        } else if (name.equals("mask") && mask == null) {
                            mask = p.data();
                        }
                    } else {
                        body.put(name, field(name, p.text()));
                    }
                }
            } else if (ct != null && ct.toLowerCase(java.util.Locale.ROOT).startsWith("application/json")) {
                body = Json.parseObject(req.bodyText());
            } else {
                res.send(415, "application/json", error("Content-Type must be application/json or multipart/form-data", "invalid_request_error"));
                return true;
            }
        } catch (RuntimeException e) {
            res.send(400, "application/json", error("invalid request body: " + e.getMessage(), "invalid_request_error"));
            return true;
        }
        String host = req.headers.getOrDefault("host", "127.0.0.1");
        String base = "http://" + host;
        boolean stream = Json.bool(body, "stream", false);
        try {
            if (!stream) {
                Map<String, Object> out = ImageApi.run(pipeline, modelId, body, image, mask, edit, store, base, new ImageApi.Events() {
                    @Override
                    public void send(Map<String, Object> event) {
                    }

                    @Override
                    public boolean cancelled() {
                        return req.clientGone();
                    }
                });
                res.send(200, "application/json", Json.write(out));
                return true;
            }
            // streaming: validation errors can still be a plain 400, so the stream starts with the first event
            boolean[] started = {false};
            ImageApi.Events events = new ImageApi.Events() {
                @Override
                public void send(Map<String, Object> event) throws IOException {
                    if (!started[0]) {
                        res.startStream(200, "text/event-stream");
                        started[0] = true;
                    }
                    res.write("data: " + Json.write(event) + "\n\n");
                }

                @Override
                public boolean cancelled() {
                    return req.clientGone();
                }
            };
            Map<String, Object> out = ImageApi.run(pipeline, modelId, body, image, mask, edit, store, base, events);
            if (!started[0]) {
                res.startStream(200, "text/event-stream");
            }
            res.write("data: " + Json.write(Json.obj("type", "image_generation.done", "usage", out.get("usage"),
                    "timing", out.get("timing"), "model", modelId)) + "\n\ndata: [DONE]\n\n");
            res.end();
        } catch (ImageApi.ApiException e) {
            res.send(e.status, "application/json", error(e.getMessage(), e.status == 400 ? "invalid_request_error" : "server_error"));
        } catch (RuntimeException e) {
            e.printStackTrace();
            res.send(500, "application/json", error("image generation failed: " + e.getMessage(), "server_error"));
        }
        return true;
    }

    /** Multipart text fields arrive as strings: numbers and booleans are turned back into what the JSON form carries. */
    private static Object field(String name, String text) {
        String t = text.strip();
        switch (name) {
            case "n", "steps", "seed", "clip_skip", "partial_images" -> {
                try {
                    return Long.parseLong(t);
                } catch (NumberFormatException e) {
                    return text;
                }
            }
            case "guidance", "guidance_scale", "strength" -> {
                try {
                    return Double.parseDouble(t);
                } catch (NumberFormatException e) {
                    return text;
                }
            }
            case "stream", "preview" -> {
                return t.equalsIgnoreCase("true") || t.equals("1");
            }
            default -> {
                return text;
            }
        }
    }

    private static String error(String msg, String type) {
        return Json.write(Json.obj("error", Json.obj("message", msg, "type", type)));
    }

    /** For tests and the console: the models list entry in a list. */
    public List<Object> modelList() {
        return List.of(modelObject());
    }
}
