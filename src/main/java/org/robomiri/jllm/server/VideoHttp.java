package org.robomiri.jllm.server;

import org.robomiri.jllm.diffusion.ImageApi;
import org.robomiri.jllm.diffusion.VideoApi;
import org.robomiri.jllm.diffusion.VideoPipeline;
import org.robomiri.jllm.diffusion.VideoStore;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.util.Map;

/**
 * The video routes of the engine's own HTTP server: {@code POST /v1/videos/generations}, {@code POST /v1/videos/encode} and
 * {@code GET /v1/videos/files/<id>.<ext>} (the clips that {@code "response_format": "url"} points at), all on top of
 * {@link VideoApi}.
 */
public final class VideoHttp {

    private final VideoPipeline pipeline;
    private final String modelId;
    private final VideoApi.Config config;
    private final VideoStore store = new VideoStore(16, 256L << 20);

    public VideoHttp(VideoPipeline pipeline, String modelId, VideoApi.Config config) {
        this.pipeline = pipeline;
        this.modelId = modelId;
        this.config = config;
    }

    public String modelId() {
        return modelId;
    }

    public VideoPipeline pipeline() {
        return pipeline;
    }

    /** The {@code /v1/models} entry of the video model. */
    public Map<String, Object> modelObject() {
        return Json.obj("id", modelId, "object", "model", "owned_by", "jllm-serve", "kind", "video",
                "description", pipeline.description(), "defaults", pipeline.info(), "size_multiple", pipeline.sizeMultiple(),
                "formats", config.ffmpegAvailable() ? java.util.List.of("mp4", "webp", "gif") : java.util.List.of("webp", "gif"));
    }

    /** @return true if the path was one of ours and has been answered */
    public boolean handle(String path, HttpServer.Request req, HttpServer.Response res) throws IOException {
        if (path.startsWith("/v1/videos/files/")) {
            String name = path.substring("/v1/videos/files/".length());
            int dot = name.lastIndexOf('.');
            String id = dot >= 0 ? name.substring(0, dot) : name;
            VideoStore.Clip clip = store.get(id);
            if (clip == null) {
                res.send(404, "application/json", error("no such video (it may have been evicted)", "not_found"));
            } else {
                res.header("Cache-Control", "private, max-age=3600").send(200, clip.mime(), clip.data());
            }
            return true;
        }
        boolean gen = path.equals("/v1/videos/generations");
        boolean enc = path.equals("/v1/videos/encode");
        if (!gen && !enc) {
            return false;
        }
        if (!req.method.equals("POST")) {
            res.send(405, "application/json", error("use POST", "invalid_request_error"));
            return true;
        }
        Map<String, Object> body;
        String ct = req.headers.get("content-type");
        try {
            if (ct != null && ct.toLowerCase(java.util.Locale.ROOT).startsWith("application/json")) {
                body = Json.parseObject(req.bodyText());
            } else {
                res.send(415, "application/json", error("Content-Type must be application/json", "invalid_request_error"));
                return true;
            }
        } catch (RuntimeException e) {
            res.send(400, "application/json", error("invalid request body: " + e.getMessage(), "invalid_request_error"));
            return true;
        }
        String base = "http://" + req.headers.getOrDefault("host", "127.0.0.1");
        try {
            if (enc) {
                res.send(200, "application/json", Json.write(VideoApi.encode(modelId, body, store, base, config)));
                return true;
            }
            boolean stream = Json.bool(body, "stream", false);
            if (!stream) {
                Map<String, Object> out = VideoApi.generate(pipeline, modelId, body, store, base, new ImageApi.Events() {
                    @Override
                    public void send(Map<String, Object> event) {
                    }

                    @Override
                    public boolean cancelled() {
                        return req.clientGone();
                    }
                }, config);
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
            Map<String, Object> out = VideoApi.generate(pipeline, modelId, body, store, base, events, config);
            if (!started[0]) {
                res.startStream(200, "text/event-stream");
            }
            res.write("data: " + Json.write(Json.obj("type", "video_generation.done", "usage", out.get("usage"),
                    "timing", out.get("timing"), "model", modelId)) + "\n\ndata: [DONE]\n\n");
            res.end();
        } catch (ImageApi.ApiException e) {
            res.send(e.status, "application/json", error(e.getMessage(), e.status == 400 ? "invalid_request_error" : "server_error"));
        } catch (RuntimeException e) {
            e.printStackTrace();
            res.send(500, "application/json", error("video generation failed: " + e.getMessage(), "server_error"));
        }
        return true;
    }

    private static String error(String msg, String type) {
        return Json.write(Json.obj("error", Json.obj("message", msg, "type", type)));
    }
}
