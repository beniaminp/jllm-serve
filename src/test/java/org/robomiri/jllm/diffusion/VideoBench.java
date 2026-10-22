package org.robomiri.jllm.diffusion;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Times video generation: {@code VideoBench BASE_DIR ADAPTER_DIR backend precision steps width height frames reps [guidance [scheduler [outFile]]]}.
 * The first clip includes shader compilation and warm-up and is reported apart. Run it under the machine's heavy lock.
 * {@code -Djllm.diffusion.profile=true} prints the GPU time of every kernel (each dispatch waited for: slower overall).
 * With {@code outFile} the last clip is written (format from the extension: webp, gif, mp4).
 */
public final class VideoBench {

    public static void main(String[] a) throws Exception {
        Path base = Path.of(a[0]);
        Path adapter = Path.of(a[1]);
        String backend = a.length > 2 ? a[2] : "metal";
        String precision = a.length > 3 ? a[3] : "f16";
        int steps = a.length > 4 ? Integer.parseInt(a[4]) : 10;
        int w = a.length > 5 ? Integer.parseInt(a[5]) : 512;
        int h = a.length > 6 ? Integer.parseInt(a[6]) : 512;
        int frames = a.length > 7 ? Integer.parseInt(a[7]) : 16;
        int reps = a.length > 8 ? Integer.parseInt(a[8]) : 1;
        double guidance = a.length > 9 ? Double.parseDouble(a[9]) : 7.5;
        String scheduler = a.length > 10 ? a[10] : "ddim";
        String out = a.length > 11 ? a[11] : null;
        long t0 = System.nanoTime();
        Devices dev = Devices.open(backend, precision, false);
        try (VideoPipeline pipe = new VideoPipeline(base, adapter, dev)) {
            System.out.printf("loaded %s + %s in %.1f s (%s)%n", base.getFileName(), adapter.getFileName(), (System.nanoTime() - t0) / 1e9,
                    pipe.description());
            java.util.List<Imaging.Image> clip = null;
            for (int i = 0; i < reps; i++) {
                VideoPipeline.Request r = new VideoPipeline.Request("a red balloon floating over a green hill, white clouds moving, photo", "blurry, low quality",
                        w, h, frames, steps, guidance, 42 + i, scheduler, 0);
                if (MetalProfile.reset(dev)) {
                    System.out.println("(profiling: each dispatch is waited for)");
                }
                clip = pipe.generate(r, (step, total, preview) -> {
                    if (Boolean.getBoolean("bench.progress")) {
                        System.err.printf("\rstep %d/%d ", step, total);
                    }
                });
                System.out.printf("%s %dx%d, %d frames, %d step(s), guidance %.1f: %.2f s total = text %.2f + unet %.2f (%.2f s/step) + vae %.2f (%.2f s/frame) + other %.2f%n",
                        i == 0 ? "first " : "warm  ", w, h, frames, steps, guidance, pipe.totalNs / 1e9, pipe.textNs / 1e9, pipe.unetNs / 1e9,
                        pipe.unetNs / 1e9 / steps, pipe.vaeNs / 1e9, pipe.vaeNs / 1e9 / frames,
                        (pipe.totalNs - pipe.textNs - pipe.unetNs - pipe.vaeNs) / 1e9);
                String prof = MetalProfile.report(dev);
                if (!prof.isEmpty()) {
                    System.out.print(prof);
                }
            }
            Runtime rt = Runtime.getRuntime();
            System.out.printf("heap used %.0f MB, GPU buffers %.0f MB (peak %.0f MB)%n", (rt.totalMemory() - rt.freeMemory()) / 1e6,
                    MetalProfile.live(dev) / 1e6, MetalProfile.peak(dev) / 1e6);
            if (out != null && clip != null) {
                VideoEncoder.Format f = VideoEncoder.Format.parse(out.substring(out.lastIndexOf('.') + 1), true);
                Files.write(Path.of(out), VideoEncoder.encode(clip, 8, f, VideoEncoder.resolveFfmpeg("auto")));
                for (int i = 0; i < clip.size(); i += Math.max(1, clip.size() / 4)) {
                    Files.write(Path.of(out + ".f" + i + ".png"), Imaging.png(clip.get(i)));
                }
                System.out.println("wrote " + out);
            }
        }
    }
}
