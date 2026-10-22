package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.diffusion.ImageApi.ApiException;
import org.robomiri.jllm.diffusion.Imaging.Image;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * The video endpoints on top of a {@link VideoPipeline}, independent of the HTTP server (the engine's own server and the web
 * console both call it). OpenAI has no video-generation endpoint that fits a synchronous call, so the shape follows
 * {@code /v1/images/generations} (see {@link ImageApi}):
 *
 * <ul>
 *   <li>{@code POST /v1/videos/generations}: {@code prompt}, {@code n} (1 to 4 clips), {@code size} ("512x512"),
 *       {@code num_frames} (or {@code frames}, default 16), {@code fps} (default 8), {@code output_format} ({@code mp4},
 *       {@code webp}, {@code gif}, {@code auto}), {@code response_format} ({@code b64_json} or {@code url}); extras
 *       {@code negative_prompt}, {@code steps}, {@code guidance} (or {@code guidance_scale}), {@code seed},
 *       {@code scheduler} ({@code ddim}, {@code euler}, {@code euler_a}, {@code dpmpp_2m}, {@code dpmpp_2m_karras}),
 *       {@code clip_skip}, {@code include_frames} ({@code png} or {@code jpeg}: every frame as base64, for editors);</li>
 *   <li>{@code POST /v1/videos/encode}: {@code frames} (a list of base64 / data-URL pictures of one size), {@code fps},
 *       {@code output_format}, {@code response_format}: writes the pictures as a video file (used by the console to export
 *       an edited clip);</li>
 *   <li>{@code "stream": true}: server-sent events {@code video_generation.progress} (step n of m),
 *       {@code video_generation.partial_image} (a small preview of the first frame), {@code video_generation.decoding}
 *       (frame n of m decoded) and {@code video_generation.completed}, then {@code [DONE]}.</li>
 * </ul>
 *
 * The response is {@code {created, model, data: [{b64_json | url, format, mime_type, width, height, frames, fps, duration, seed,
 * revised_prompt, [frame_images]}], usage, timing}}. Clip k of a request uses seed {@code seed + k}.
 */
public final class VideoApi {

    /** What the API needs to know about the server's video output. */
    public record Config(String ffmpeg, String defaultFormat) {
        public boolean ffmpegAvailable() {
            return ffmpeg != null;
        }
    }

    private VideoApi() {
    }

    /** Runs one generation request. */
    public static Map<String, Object> generate(VideoPipeline pipe, String modelId, Map<String, Object> body, VideoStore store, String baseUrl,
                                               ImageApi.Events events, Config cfg) throws IOException {
        String prompt = Json.str(body, "prompt", "");
        if (prompt.isBlank()) {
            throw ApiException.invalid("prompt is required");
        }
        int n = (int) num(body, "n", 1);
        if (n < 1 || n > 4) {
            throw ApiException.invalid("n must be 1 to 4");
        }
        String format = Json.str(body, "response_format", "b64_json");
        if (!format.equals("b64_json") && !format.equals("url")) {
            throw ApiException.invalid("response_format must be b64_json or url");
        }
        VideoEncoder.Format out = outputFormat(body, cfg);
        int mult = pipe.sizeMultiple();
        int w;
        int h;
        String size = Json.str(body, "size", "auto");
        if (size.equals("auto") || size.isBlank()) {
            w = h = pipe.defaults.size();
        } else {
            String[] p = size.toLowerCase(java.util.Locale.ROOT).split("x");
            try {
                w = Integer.parseInt(p[0].trim());
                h = Integer.parseInt(p[1].trim());
            } catch (RuntimeException e) {
                throw ApiException.invalid("size must look like 512x512");
            }
        }
        if (w % mult != 0 || h % mult != 0 || w < mult || h < mult) {
            throw ApiException.invalid("size must be a multiple of " + mult + " on both sides for this model, got " + w + "x" + h);
        }
        int frames = (int) num(body, body.containsKey("num_frames") ? "num_frames" : "frames", pipe.defaults.frames());
        if (frames < 1 || frames > pipe.maxFrames()) {
            throw ApiException.invalid("num_frames must be 1 to " + pipe.maxFrames() + " for this model");
        }
        double fps = num(body, "fps", pipe.defaults.fps());
        if (!(fps > 0 && fps <= 60)) {
            throw ApiException.invalid("fps must be above 0 and at most 60");
        }
        int steps = (int) num(body, "steps", pipe.defaults.steps());
        double guidance = num(body, body.containsKey("guidance") ? "guidance" : "guidance_scale", pipe.defaults.guidance());
        long seed = body.containsKey("seed") && body.get("seed") != null ? (long) num(body, "seed", 0)
                : ThreadLocalRandom.current().nextLong(0, 1L << 31);
        String scheduler = Json.str(body, "scheduler", pipe.defaults.scheduler());
        int clipSkip = (int) num(body, "clip_skip", 0);
        String negative = Json.str(body, "negative_prompt", "");
        try {
            Scheduler.Kind.parse(scheduler);
        } catch (IllegalArgumentException e) {
            throw ApiException.invalid(e.getMessage());
        }
        String withFrames = frameFormat(body);
        boolean stream = events != null && Json.bool(body, "stream", false);
        boolean previews = stream && Json.bool(body, "preview", true);
        long t0 = System.nanoTime();
        boolean[] gone = new boolean[1];
        int[] clipIndex = new int[1];
        VideoPipeline.Listener listener = new VideoPipeline.Listener() {
            @Override
            public void progress(int step, int total, Image preview) {
                if (!stream || gone[0]) {
                    return;
                }
                try {
                    events.send(Json.obj("type", "video_generation.progress", "video", clipIndex[0], "step", step, "steps", total));
                    if (preview != null) {
                        events.send(Json.obj("type", "video_generation.partial_image", "video", clipIndex[0], "step", step, "steps", total,
                                "width", preview.width(), "height", preview.height(),
                                "b64_json", Base64.getEncoder().encodeToString(Imaging.png(preview))));
                    }
                } catch (IOException e) {
                    gone[0] = true;
                }
            }

            @Override
            public void decoded(int frame, int total) {
                if (!stream || gone[0]) {
                    return;
                }
                try {
                    events.send(Json.obj("type", "video_generation.decoding", "video", clipIndex[0], "frame", frame, "frames", total));
                } catch (IOException e) {
                    gone[0] = true;
                }
            }

            @Override
            public boolean wantsPreview() {
                return previews;
            }

            @Override
            public boolean cancelled() {
                return gone[0] || (events != null && events.cancelled());
            }
        };
        List<Object> data = new ArrayList<>();
        long textNs = 0;
        long unetNs = 0;
        long vaeNs = 0;
        long encodeNs = 0;
        for (int i = 0; i < n; i++) {
            clipIndex[0] = i;
            VideoPipeline.Request req = new VideoPipeline.Request(prompt, negative, w, h, frames, steps, guidance, seed + i, scheduler, clipSkip);
            List<Image> clip;
            try {
                clip = pipe.generate(req, listener);
            } catch (VideoPipeline.CancellationException e) {
                return Json.obj("created", System.currentTimeMillis() / 1000, "model", modelId, "data", data, "cancelled", true);
            } catch (IllegalArgumentException e) {
                throw ApiException.invalid(e.getMessage());
            }
            textNs += pipe.textNs;
            unetNs += pipe.unetNs;
            vaeNs += pipe.vaeNs;
            long e0 = System.nanoTime();
            Map<String, Object> item = file(clip, fps, out, cfg, format, store, baseUrl);
            encodeNs += System.nanoTime() - e0;
            item.put("seed", seed + i);
            item.put("revised_prompt", prompt);
            if (withFrames != null) {
                item.put("frame_images", frameImages(clip, withFrames));
            }
            data.add(item);
            if (stream && !gone[0]) {
                Map<String, Object> ev = Json.obj("type", "video_generation.completed", "video", i);
                ev.putAll(item);
                try {
                    events.send(ev);
                } catch (IOException e) {
                    gone[0] = true;
                }
            }
        }
        double seconds = (System.nanoTime() - t0) / 1e9;
        return Json.obj("created", System.currentTimeMillis() / 1000, "model", modelId, "data", data,
                "usage", Json.obj("videos", n, "frames", n * frames, "steps", steps),
                "timing", Json.obj("seconds", Math.round(seconds * 1000) / 1000.0, "text_s", textNs / 1e9, "unet_s", unetNs / 1e9,
                        "vae_s", vaeNs / 1e9, "encode_s", encodeNs / 1e9));
    }

    /** {@code POST /v1/videos/encode}: pictures in, a video file out. */
    public static Map<String, Object> encode(String modelId, Map<String, Object> body, VideoStore store, String baseUrl, Config cfg) throws IOException {
        Object fr = body.get("frames");
        if (!(fr instanceof List<?> list) || list.isEmpty()) {
            throw ApiException.invalid("frames must be a non-empty list of base64 pictures");
        }
        if (list.size() > 256) {
            throw ApiException.invalid("at most 256 frames");
        }
        double fps = num(body, "fps", 8);
        if (!(fps > 0 && fps <= 60)) {
            throw ApiException.invalid("fps must be above 0 and at most 60");
        }
        String format = Json.str(body, "response_format", "b64_json");
        if (!format.equals("b64_json") && !format.equals("url")) {
            throw ApiException.invalid("response_format must be b64_json or url");
        }
        VideoEncoder.Format out = outputFormat(body, cfg);
        List<Image> frames = new ArrayList<>();
        try {
            for (Object o : list) {
                byte[] bytes = decodeField(o, "frames");
                if (bytes == null) {
                    throw ApiException.invalid("frames contains an empty entry");
                }
                frames.add(Imaging.decode(bytes));
            }
        } catch (IOException e) {
            throw ApiException.invalid(e.getMessage());
        }
        Image f0 = frames.get(0);
        for (int i = 0; i < frames.size(); i++) {
            Image f = frames.get(i);
            if (f.width() != f0.width() || f.height() != f0.height()) {
                frames.set(i, Imaging.resize(f, f0.width(), f0.height()));
            }
        }
        if (f0.width() > 4096 || f0.height() > 4096) {
            throw ApiException.invalid("frames larger than 4096 pixels on a side are not supported");
        }
        long t0 = System.nanoTime();
        Map<String, Object> item = file(frames, fps, out, cfg, format, store, baseUrl);
        return Json.obj("created", System.currentTimeMillis() / 1000, "model", modelId, "data", List.of(item),
                "timing", Json.obj("seconds", Math.round((System.nanoTime() - t0) / 1e6) / 1000.0));
    }

    // ------------------------------------------------------------------ helpers

    private static Map<String, Object> file(List<Image> clip, double fps, VideoEncoder.Format out, Config cfg, String responseFormat,
                                            VideoStore store, String baseUrl) throws IOException {
        byte[] bytes;
        try {
            bytes = VideoEncoder.encode(clip, fps, out, cfg.ffmpeg());
        } catch (IllegalArgumentException e) {
            throw ApiException.invalid(e.getMessage());
        } catch (IOException e) {
            throw new ApiException(500, null, "could not write the " + out.ext + " file: " + e.getMessage());
        }
        Image f0 = clip.get(0);
        Map<String, Object> item = responseFormat.equals("url")
                ? Json.obj("url", baseUrl + "/v1/videos/files/" + store.put(bytes, out) + "." + out.ext)
                : Json.obj("b64_json", Base64.getEncoder().encodeToString(bytes));
        item.put("format", out.ext);
        item.put("mime_type", out.mime);
        item.put("width", f0.width());
        item.put("height", f0.height());
        item.put("frames", clip.size());
        item.put("fps", fps);
        item.put("duration", Math.round(clip.size() / fps * 1000) / 1000.0);
        item.put("bytes", bytes.length);
        return item;
    }

    private static VideoEncoder.Format outputFormat(Map<String, Object> body, Config cfg) {
        String f = Json.str(body, body.containsKey("output_format") ? "output_format" : "format", cfg.defaultFormat());
        VideoEncoder.Format out;
        try {
            out = VideoEncoder.Format.parse(f, cfg.ffmpegAvailable());
        } catch (IllegalArgumentException e) {
            throw ApiException.invalid(e.getMessage());
        }
        if (out == VideoEncoder.Format.MP4 && !cfg.ffmpegAvailable()) {
            throw ApiException.invalid("mp4 output needs ffmpeg on the server (video.ffmpeg); use webp or gif");
        }
        return out;
    }

    private static String frameFormat(Map<String, Object> body) {
        Object v = body.get("include_frames");
        if (v == null || Boolean.FALSE.equals(v) || "false".equals(v) || "".equals(v)) {
            return null;
        }
        if (Boolean.TRUE.equals(v)) {
            return "jpeg";
        }
        String s = String.valueOf(v).toLowerCase(java.util.Locale.ROOT);
        if (s.equals("png") || s.equals("jpeg") || s.equals("jpg")) {
            return s.equals("png") ? "png" : "jpeg";
        }
        throw ApiException.invalid("include_frames must be png, jpeg or false");
    }

    private static List<Object> frameImages(List<Image> clip, String format) throws IOException {
        List<Object> out = new ArrayList<>();
        for (Image f : clip) {
            byte[] b = format.equals("png") ? Imaging.png(f) : Imaging.jpeg(f, 0.92f);
            out.add(Base64.getEncoder().encodeToString(b));
        }
        return out;
    }

    private static double num(Map<String, Object> body, String key, double def) {
        Object v = body.get(key);
        if (v == null) {
            return def;
        }
        if (v instanceof Number x) {
            return x.doubleValue();
        }
        if (v instanceof String s) {
            try {
                return Double.parseDouble(s.trim());
            } catch (NumberFormatException e) {
                throw ApiException.invalid(key + " must be a number");
            }
        }
        throw ApiException.invalid(key + " must be a number");
    }

    /** A picture field given as base64 or a data URL (not as a URL: the server does not fetch anything on a client's behalf). */
    private static byte[] decodeField(Object v, String name) {
        if (v == null) {
            return null;
        }
        if (!(v instanceof String s)) {
            throw ApiException.invalid(name + " entries must be base64 strings or data URLs");
        }
        if (s.isBlank()) {
            return null;
        }
        if (s.startsWith("http://") || s.startsWith("https://")) {
            throw ApiException.invalid(name + " as a URL is not supported: send the pictures as base64 or data URLs");
        }
        int comma = s.indexOf(',');
        String b64 = s.startsWith("data:") && comma >= 0 ? s.substring(comma + 1) : s;
        try {
            return Base64.getMimeDecoder().decode(b64.strip());
        } catch (IllegalArgumentException e) {
            throw ApiException.invalid(name + " is not valid base64");
        }
    }
}
