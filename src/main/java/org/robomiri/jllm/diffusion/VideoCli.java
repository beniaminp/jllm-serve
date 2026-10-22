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

    public static void run(Settings s, HuggingFaceHub hub, String source) throws Exception {
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
