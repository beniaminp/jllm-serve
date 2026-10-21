package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.diffusion.Imaging.Image;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * The OpenAI-compatible image endpoints on top of an {@link ImagePipeline}, independent of the HTTP server (the engine's
 * own server and the web console both call it):
 *
 * <ul>
 *   <li>{@code POST /v1/images/generations}: {@code prompt}, {@code n}, {@code size} ("512x512"), {@code response_format}
 *       ({@code b64_json} or {@code url}); extras {@code negative_prompt}, {@code steps}, {@code guidance} (or
 *       {@code guidance_scale}), {@code seed}, {@code scheduler} ({@code euler}, {@code euler_a}, {@code dpmpp_2m},
 *       {@code dpmpp_2m_karras}), {@code clip_skip}, and for image-to-image {@code image} (base64 or data URL) with
 *       {@code strength};</li>
 *   <li>{@code POST /v1/images/edits}: {@code image}, optional {@code mask} (white / transparent = repaint), {@code prompt}
 *       and the same options: image-to-image without a mask, inpainting with one;</li>
 *   <li>{@code "stream": true}: server-sent events {@code image_generation.progress} (step n of m),
 *       {@code image_generation.partial_image} (a small preview of the current estimate) and
 *       {@code image_generation.completed} (the finished image), then {@code [DONE]}.</li>
 * </ul>
 *
 * Image k of a request is generated with seed {@code seed + k}; the seeds used are returned.
 */
public final class ImageApi {

    /** A request the client got wrong: becomes an HTTP 400 (or the given status). */
    public static final class ApiException extends RuntimeException {
        public final int status;
        public final String code;

        public ApiException(int status, String code, String message) {
            super(message);
            this.status = status;
            this.code = code;
        }

        public static ApiException invalid(String message) {
            return new ApiException(400, null, message);
        }
    }

    /** Where progress events of a streaming request go. */
    public interface Events {
        void send(Map<String, Object> event) throws IOException;

        /** True once the client has gone away. */
        boolean cancelled();
    }

    private ImageApi() {
    }

    /**
     * Runs one generation or edit request.
     *
     * @param body    the JSON fields (for a multipart request: the text fields)
     * @param image   the uploaded input image (edits), or null: then {@code body.image} may carry one as base64
     * @param mask    the uploaded mask, or null: then {@code body.mask} may carry one
     * @param edit    {@code /v1/images/edits}: an input image is required
     * @param store   where {@code response_format: "url"} images are kept
     * @param baseUrl prefix of the returned URLs, e.g. {@code http://127.0.0.1:8080}
     * @param events  progress sink for {@code "stream": true}, or null
     * @return the response object (for a streaming request: the final summary, also sent as the last event)
     */
    public static Map<String, Object> run(ImagePipeline pipe, String modelId, Map<String, Object> body, byte[] image, byte[] mask,
                                          boolean edit, ImageStore store, String baseUrl, Events events) throws IOException {
        String prompt = Json.str(body, "prompt", "");
        if (prompt.isBlank() && !edit) {
            throw ApiException.invalid("prompt is required");
        }
        int n = (int) num(body, "n", 1);
        if (n < 1 || n > 10) {
            throw ApiException.invalid("n must be 1 to 10");
        }
        String format = Json.str(body, "response_format", "b64_json");
        if (!format.equals("b64_json") && !format.equals("url")) {
            throw ApiException.invalid("response_format must be b64_json or url");
        }
        Image init = null;
        float[] maskPx = null;
        int[] maskSize = new int[2];
        try {
            byte[] imgBytes = image != null ? image : decodeField(body.get("image"), "image");
            if (imgBytes != null) {
                init = Imaging.decode(imgBytes);
            }
            byte[] maskBytes = mask != null ? mask : decodeField(body.get("mask"), "mask");
            if (maskBytes != null) {
                maskPx = Imaging.decodeMask(maskBytes, maskSize);
            }
        } catch (IOException e) {
            throw ApiException.invalid(e.getMessage());
        }
        if (edit && init == null) {
            throw ApiException.invalid("image is required");
        }
        if (maskPx != null && init == null) {
            throw ApiException.invalid("a mask needs an image");
        }
        int mult = pipe.sizeMultiple();
        int w;
        int h;
        String size = Json.str(body, "size", "auto");
        if (size.equals("auto") || size.isBlank()) {
            if (init != null) { // the input picture's size, rounded down to what the network can take, within 1024 on a side
                double f = Math.min(1.0, 1024.0 / Math.max(init.width(), init.height()));
                w = Math.max(mult, (int) (init.width() * f) / mult * mult);
                h = Math.max(mult, (int) (init.height() * f) / mult * mult);
            } else {
                w = h = pipe.defaults.size();
            }
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
        int steps = (int) num(body, "steps", pipe.defaults.steps());
        double guidance = dbl(body, body.containsKey("guidance") ? "guidance" : "guidance_scale", pipe.defaults.guidance());
        long seed = body.containsKey("seed") && body.get("seed") != null ? (long) num(body, "seed", 0)
                : ThreadLocalRandom.current().nextLong(0, 1L << 31);
        String scheduler = Json.str(body, "scheduler", pipe.defaults.scheduler());
        int clipSkip = (int) num(body, "clip_skip", 0);
        double strength = dbl(body, "strength", maskPx != null ? 1.0 : 0.8);
        String negative = Json.str(body, "negative_prompt", "");
        try {
            Scheduler.Kind.parse(scheduler);
        } catch (IllegalArgumentException e) {
            throw ApiException.invalid(e.getMessage());
        }
        boolean stream = events != null && Json.bool(body, "stream", false);
        boolean previews = stream && Json.bool(body, "preview", true) && num(body, "partial_images", 1) > 0;
        ImagePipeline.Request req = new ImagePipeline.Request(prompt, negative, w, h, steps, guidance, seed, scheduler, clipSkip, init,
                maskPx, maskSize[0], maskSize[1], strength);
        long t0 = System.nanoTime();
        boolean[] gone = new boolean[1];
        ImagePipeline.Listener listener = new ImagePipeline.Listener() {
            @Override
            public void progress(int imageIndex, int step, int total, Image preview) {
                if (!stream || gone[0]) {
                    return;
                }
                try {
                    events.send(Json.obj("type", "image_generation.progress", "image", imageIndex, "step", step, "steps", total));
                    if (preview != null) {
                        events.send(Json.obj("type", "image_generation.partial_image", "image", imageIndex, "step", step, "steps", total,
                                "width", preview.width(), "height", preview.height(),
                                "b64_json", Base64.getEncoder().encodeToString(Imaging.png(preview))));
                    }
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
        List<Image> images;
        try {
            images = pipe.generate(req, n, listener);
        } catch (ImagePipeline.CancellationException e) {
            return Json.obj("created", System.currentTimeMillis() / 1000, "model", modelId, "data", List.of(), "cancelled", true);
        } catch (IllegalArgumentException e) {
            throw ApiException.invalid(e.getMessage());
        }
        List<Object> data = new ArrayList<>();
        for (int i = 0; i < images.size(); i++) {
            Image img = images.get(i);
            byte[] png = Imaging.png(img);
            Map<String, Object> item = format.equals("url")
                    ? Json.obj("url", baseUrl + "/v1/images/files/" + store.put(png) + ".png")
                    : Json.obj("b64_json", Base64.getEncoder().encodeToString(png));
            item.put("seed", seed + i);
            item.put("width", img.width());
            item.put("height", img.height());
            if (!prompt.isBlank()) {
                item.put("revised_prompt", prompt);
            }
            data.add(item);
            if (stream && !gone[0]) {
                Map<String, Object> ev = Json.obj("type", "image_generation.completed", "image", i);
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
                "usage", Json.obj("images", images.size(), "steps", req.steps()),
                "timing", Json.obj("seconds", Math.round(seconds * 1000) / 1000.0, "text_s", pipe.textNs / 1e9,
                        "unet_s", pipe.unetNs / 1e9, "vae_s", pipe.vaeNs / 1e9));
    }

    // ------------------------------------------------------------------ parsing helpers

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

    private static double dbl(Map<String, Object> body, String key, double def) {
        return num(body, key, def);
    }

    /** An image field given as base64 or a data URL (not as a URL: the server does not fetch anything on a client's behalf). */
    private static byte[] decodeField(Object v, String name) {
        if (v == null) {
            return null;
        }
        if (v instanceof List<?> l) { // the SDKs may send a list of images; the first one counts
            return l.isEmpty() ? null : decodeField(l.get(0), name);
        }
        if (!(v instanceof String s)) {
            throw ApiException.invalid(name + " must be a base64 string or data URL");
        }
        if (s.isBlank()) {
            return null;
        }
        if (s.startsWith("http://") || s.startsWith("https://")) {
            throw ApiException.invalid(name + " as a URL is not supported: send the picture as base64, a data URL or a multipart file");
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
