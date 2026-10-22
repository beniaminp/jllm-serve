package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.diffusion.Imaging.Image;
import org.robomiri.jllm.hub.HuggingFaceHub;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * {@code jllm video}: makes a clip from the command line with the {@code video.*} settings (the shared flags {@code --steps},
 * {@code --guidance}, {@code --width}, {@code --height}, {@code --seed}, {@code --scheduler}, {@code --negative} and
 * {@code --out} count when given, else the {@code video.*} values do).
 */
public final class VideoCli {

    private VideoCli() {
    }

    /** The value of a flag shared with {@code jllm image} when it was given, else the {@code video.*} setting. */
    private static String pick(Settings s, String imageKey, String videoKey) {
        return !s.source(imageKey).equals("default") ? s.get(imageKey) : s.get(videoKey);
    }

    /** True when {@code source} is a local folder holding a Wan pipeline ({@code model_index.json} of class {@code WanPipeline}). */
    static boolean isWanFolder(String source) {
        Path p = Path.of(source).resolve("model_index.json");
        try {
            return Files.isRegularFile(p) && "WanPipeline".equals(org.robomiri.jllm.json.Json.str(
                    org.robomiri.jllm.json.Json.parseObject(Files.readString(p)), "_class_name", ""));
        } catch (java.io.IOException | RuntimeException e) {
            return false;
        }
    }

    /**
     * Wan 2.1 text-to-video from a local folder (docs/23 section 23.12): frames must be 4k + 1 (rounded down), sizes multiples of 16, the
     * text encoder is quantized to int8 on the CPU while it is read, the transformer runs in float32 on the GPU unless
     * {@code video.precision} says f16.
     */
    private static void runWan(Settings s, String source) throws Exception {
        String prompt = s.has("prompt") ? s.get("prompt") : "a red balloon floating over a green hill, white clouds moving, photo";
        String ffmpeg = VideoEncoder.resolveFfmpeg(s.get("video.ffmpeg"));
        String prec = s.get("video.precision").equals("auto") ? "f32" : s.get("video.precision");
        int frames = (Math.max(1, s.getInt("video.frames")) - 1) / 4 * 4 + 1;
        int steps = Integer.parseInt(pick(s, "image.steps", "video.steps"));
        double guidance = Double.parseDouble(pick(s, "image.guidance", "video.guidance"));
        long seed = Long.parseLong(pick(s, "image.seed", "video.seed"));
        int w = Integer.parseInt(pick(s, "image.width", "video.width"));
        int h = Integer.parseInt(pick(s, "image.height", "video.height"));
        w = w == 0 ? 480 : w;
        h = h == 0 ? (w == 480 ? 272 : w) : h;
        Devices dev = Devices.open(s.get("video.backend"), prec, false);
        try (WanPipeline pipe = new WanPipeline(Path.of(source), dev, true)) {
            System.err.println("Wan 2.1 text-to-video, " + dev.description() + ", text encoder int8 on the CPU");
            WanPipeline.Request r = new WanPipeline.Request(prompt, pick(s, "image.negative", "video.negative"), w, h, frames, steps <= 0 ? 30 : steps,
                    guidance < 0 ? 5.0 : guidance, seed < 0 ? ThreadLocalRandom.current().nextLong(0, 1L << 31) : seed,
                    pick(s, "image.scheduler", "video.scheduler").isEmpty() ? "unipc" : pick(s, "image.scheduler", "video.scheduler"));
            long t0 = System.nanoTime();
            List<Image> clip = pipe.generate(r, (step, total) -> {
                if (step > 0) {
                    System.err.printf("\rstep %d/%d ", step, total);
                }
            });
            System.err.printf("%n%d frames %dx%d, %d step(s), guidance %.1f, seed %d: %.1f s (text %.1f, transformer %.1f = %.1f per step, vae %.1f)%n", clip.size(), w, h,
                    r.steps(), r.guidance(), r.seed(), (System.nanoTime() - t0) / 1e9, pipe.textNs / 1e9, pipe.ditNs / 1e9, pipe.ditNs / 1e9 / r.steps(), pipe.vaeNs / 1e9);
            VideoEncoder.Format format = VideoEncoder.Format.parse(s.get("video.format"), ffmpeg != null);
            String outName = !s.source("image.out").equals("default") ? s.get("image.out") : !s.get("video.out").isEmpty() ? s.get("video.out") : "video." + format.ext;
            int dot = outName.lastIndexOf('.');
            if (dot > 0) {
                format = VideoEncoder.Format.parse(outName.substring(dot + 1), ffmpeg != null);
            } else {
                outName = outName + "." + format.ext;
            }
            double fps = s.source("video.fps").equals("default") ? 16.0 : s.getFloat("video.fps"); // Wan's frame rate
            Files.write(Path.of(outName), VideoEncoder.encode(clip, fps, format, ffmpeg));
            System.out.println(outName);
        }
    }

    public static void run(Settings s, HuggingFaceHub hub, String source) throws Exception {
        if (isWanFolder(source)) {
            runWan(s, source);
            return;
        }
        String prompt = s.has("prompt") ? s.get("prompt") : "a red balloon floating over a green hill, white clouds moving, photo";
        String ffmpeg = VideoEncoder.resolveFfmpeg(s.get("video.ffmpeg"));
        try (VideoPipeline pipe = VideoModels.load(s, hub, source)) {
            System.err.println(pipe.description());
            VideoPipeline.Defaults d = pipe.defaults;
            int steps = Integer.parseInt(pick(s, "image.steps", "video.steps"));
            if (steps <= 0) {
                steps = d.steps();
            }
            double guidance = Double.parseDouble(pick(s, "image.guidance", "video.guidance"));
            if (guidance < 0) {
                guidance = d.guidance();
            }
            long seed = Long.parseLong(pick(s, "image.seed", "video.seed"));
            if (seed < 0) {
                seed = s.getLong("sampling.seed") >= 0 ? s.getLong("sampling.seed") : ThreadLocalRandom.current().nextLong(0, 1L << 31);
            }
            String scheduler = pick(s, "image.scheduler", "video.scheduler");
            if (scheduler.isEmpty()) {
                scheduler = d.scheduler();
            }
            String negative = pick(s, "image.negative", "video.negative");
            int w = Integer.parseInt(pick(s, "image.width", "video.width"));
            int h = Integer.parseInt(pick(s, "image.height", "video.height"));
            if (w == 0 && h == 0) {
                w = h = d.size();
            } else if (w == 0 || h == 0) {
                w = h = Math.max(w, h);
            }
            int frames = s.getInt("video.frames");
            double fps = s.getFloat("video.fps");
            VideoEncoder.Format format = VideoEncoder.Format.parse(s.get("video.format"), ffmpeg != null);
            if (format == VideoEncoder.Format.MP4 && ffmpeg == null) {
                throw new IllegalArgumentException("video.format=mp4 needs ffmpeg (video.ffmpeg=" + s.get("video.ffmpeg") + " found none)");
            }
            VideoPipeline.Request r = new VideoPipeline.Request(prompt, negative, w, h, frames, steps, guidance, seed, scheduler, 0);
            long t0 = System.nanoTime();
            List<Image> clip = pipe.generate(r, new VideoPipeline.Listener() {
                @Override
                public void progress(int step, int total, Image preview) {
                    if (step > 0) {
                        System.err.printf("\rstep %d/%d ", step, total);
                    }
                }

                @Override
                public void decoded(int frame, int total) {
                    System.err.printf("\rdecoded frame %d/%d ", frame, total);
                }
            });
            double gen = (System.nanoTime() - t0) / 1e9;
            System.err.printf("%n%d frames %dx%d, %d step(s), guidance %.1f, seed %d: %.1f s (text %.2f, unet %.1f = %.2f per step, vae %.1f)%n",
                    clip.size(), w, h, steps, guidance, seed, gen, pipe.textNs / 1e9, pipe.unetNs / 1e9, pipe.unetNs / 1e9 / steps, pipe.vaeNs / 1e9);
            String outName = !s.source("image.out").equals("default") ? s.get("image.out")
                    : !s.get("video.out").isEmpty() ? s.get("video.out") : "video." + format.ext;
            Path target = Path.of(outName);
            int dot = outName.lastIndexOf('.');
            if (dot > 0 && dot > outName.lastIndexOf('/')) { // the extension of --out picks the format
                String ext = outName.substring(dot + 1).toLowerCase(java.util.Locale.ROOT);
                if (ext.equals("mp4") || ext.equals("webp") || ext.equals("gif")) {
                    format = VideoEncoder.Format.parse(ext, ffmpeg != null);
                }
            } else {
                target = Path.of(outName + "." + format.ext);
            }
            if (format == VideoEncoder.Format.MP4 && ffmpeg == null) {
                throw new IllegalArgumentException("mp4 output needs ffmpeg: choose a .webp or .gif file name");
            }
            Files.write(target, VideoEncoder.encode(clip, fps, format, ffmpeg));
            System.out.println(target);
        }
    }
}
