package org.robomiri.jllm.web.tools;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.diffusion.ImageApi;
import org.robomiri.jllm.diffusion.VideoApi;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.web.settings.SettingsService;
import org.robomiri.jllm.web.video.VideoService;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * {@code generate_video}: a short clip from a text prompt with the loaded video model (AnimateDiff); the chat shows it inline,
 * wherever a tool result contains {@code ![...](/v1/videos/files/<id>.<ext>)}. Gated by {@code video.enabled}.
 */
public final class GenerateVideoTool implements Tool {

    private final VideoService videos;
    private final SettingsService settings;

    GenerateVideoTool(VideoService videos, SettingsService settings) {
        this.videos = videos;
        this.settings = settings;
    }

    @Override
    public String name() {
        return "generate_video";
    }

    @Override
    public String description() {
        return "Make a short video clip (a few seconds, no sound) from a text description with the video model that is loaded "
                + "(AnimateDiff on Stable Diffusion). The clip is shown to the user in the chat. Describe one scene and its motion in "
                + "English as a short list, e.g. 'a red balloon floating over a green hill, white clouds moving, photo'. "
                + "Generation takes minutes: call it once. Returns the link of the clip.";
    }

    @Override
    public Map<String, Object> parameters() {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("prompt", Json.obj("type", "string", "description", "the scene and its motion (English works best)"));
        properties.put("negative_prompt", Json.obj("type", "string", "description", "what it should not show, e.g. 'blurry, low quality' (optional)"));
        properties.put("num_frames", Json.obj("type", "integer", "description", "number of frames, 8 to 32 (optional, default 16; at fps 8 that is 2 seconds)"));
        properties.put("fps", Json.obj("type", "integer", "description", "frames per second of playback (optional, default 8)"));
        properties.put("size", Json.obj("type", "string", "description", "width x height in pixels, e.g. 512x512 (optional; a multiple of 64; smaller is faster)"));
        properties.put("steps", Json.obj("type", "integer", "description", "denoising steps (optional; the model's default is used when left out)"));
        properties.put("seed", Json.obj("type", "integer", "description", "random seed for a reproducible clip (optional)"));
        return Json.obj("type", "object", "properties", properties, "required", List.of("prompt"));
    }

    @Override
    public String source() {
        return "builtin";
    }

    /** A clip takes minutes, not seconds: at least half an hour whatever the general limit is. */
    @Override
    public int timeoutSeconds(int configured) {
        return Math.max(configured, 1800);
    }

    @Override
    @SuppressWarnings("unchecked")
    public String call(Map<String, Object> args) throws Exception {
        Settings s = settings.current();
        if (!s.getBool("video.enabled")) {
            throw new IllegalStateException("video generation is switched off (video.enabled=false)");
        }
        String prompt = Json.str(args, "prompt", "").strip();
        if (prompt.isEmpty()) {
            throw new IllegalArgumentException("prompt is required");
        }
        Map<String, Object> body = new HashMap<>();
        body.put("prompt", prompt);
        for (String k : new String[]{"negative_prompt", "num_frames", "fps", "size", "steps", "seed"}) {
            if (args.get(k) != null) {
                body.put(k, args.get(k));
            }
        }
        body.put("n", 1);
        body.put("response_format", "url");
        VideoService.Loaded model;
        try {
            model = videos.resolve(Json.str(body, "model", null));
        } catch (NoSuchElementException e) {
            throw new IllegalStateException("no video model is loaded; ask the user to load one on the Models page (for example guoyww/animatediff-motion-adapter-v1-5-2)");
        }
        VideoApi.Config cfg = videos.config(s.get("video.ffmpeg"), s.get("video.format"));
        try (VideoService.Lease lease = model.lease()) {
            Map<String, Object> out = VideoApi.generate(model.pipeline(), model.id(), body, videos.store(), "", null, cfg);
            List<Object> data = (List<Object>) out.get("data");
            model.countVideos(data.size());
            Object first = data.get(0);
            Map<String, Object> timing = Json.map(out, "timing");
            return "Generated a video with " + model.id() + " (" + Json.num(first, "width", 0) + "x" + Json.num(first, "height", 0) + ", "
                    + Json.num(first, "frames", 0) + " frames at " + Json.dbl(first, "fps", 0) + " fps, seed " + Json.num(first, "seed", 0) + ", "
                    + String.format("%.0f s", Json.dbl(timing, "seconds", 0)) + "):\n![generated video](" + Json.str(first, "url", "") + ")\n"
                    + "The clip is shown to the user in the chat.";
        } catch (ImageApi.ApiException e) {
            throw new IllegalArgumentException(e.getMessage());
        }
    }
}
