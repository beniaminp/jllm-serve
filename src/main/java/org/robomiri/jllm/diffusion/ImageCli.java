package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.diffusion.Imaging.Image;
import org.robomiri.jllm.hub.HuggingFaceHub;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/** {@code jllm image}: generates pictures from the command line with the {@code image.*} settings. */
public final class ImageCli {

    private ImageCli() {
    }

    public static void run(Settings s, HuggingFaceHub hub, String source) throws Exception {
        String prompt = s.has("prompt") ? s.get("prompt") : "a photo of an astronaut riding a horse on the moon";
        try (ImagePipeline pipe = ImageModels.load(s, hub, source)) {
            System.err.println(pipe.description());
            ImagePipeline.Defaults d = pipe.defaults;
            int steps = s.getInt("image.steps") > 0 ? s.getInt("image.steps") : d.steps();
            double guidance = s.getFloat("image.guidance") >= 0 ? s.getFloat("image.guidance") : d.guidance();
            // --seed (sampling.seed) works here too
            long seed = s.getLong("image.seed") >= 0 ? s.getLong("image.seed") : s.getLong("sampling.seed") >= 0 ? s.getLong("sampling.seed")
                    : ThreadLocalRandom.current().nextLong(0, 1L << 31);
            String scheduler = s.opt("image.scheduler") != null ? s.opt("image.scheduler") : d.scheduler();
            Image init = null;
            float[] mask = null;
            int[] msz = new int[2];
            if (s.opt("image.init") != null) {
                init = Imaging.decode(Files.readAllBytes(Path.of(s.get("image.init"))));
            }
            if (s.opt("image.mask") != null) {
                mask = Imaging.decodeMask(Files.readAllBytes(Path.of(s.get("image.mask"))), msz);
            }
            int mult = pipe.sizeMultiple();
            int w = s.getInt("image.width");
            int h = s.getInt("image.height");
            if (w == 0 && h == 0) {
                w = h = d.size();
                if (init != null) {
                    w = Math.max(mult, init.width() / mult * mult);
                    h = Math.max(mult, init.height() / mult * mult);
                }
            } else if (w == 0 || h == 0) {
                w = h = Math.max(w, h);
            }
            double strength = mask != null && s.getFloat("image.strength") == 0.8f ? 1.0 : s.getFloat("image.strength");
            int n = s.getInt("image.count");
            ImagePipeline.Request r = new ImagePipeline.Request(prompt, s.get("image.negative"), w, h, steps, guidance, seed, scheduler, 0,
                    init, mask, msz[0], msz[1], strength);
            long t0 = System.nanoTime();
            List<Image> out = pipe.generate(r, n, (image, step, total, preview) -> {
                if (step > 0) {
                    System.err.printf("\rimage %d/%d  step %d/%d ", image + 1, n, step, total);
                }
            });
            System.err.printf("%n%d image(s) %dx%d, %d step(s), guidance %.1f, seeds %d..%d: %.2f s (text %.2f, unet %.2f, vae %.2f)%n",
                    out.size(), w, h, steps, guidance, seed, seed + n - 1, (System.nanoTime() - t0) / 1e9, pipe.textNs / 1e9,
                    pipe.unetNs / 1e9, pipe.vaeNs / 1e9);
            Path target = Path.of(s.get("image.out"));
            for (int i = 0; i < out.size(); i++) {
                Path p = n == 1 ? target : numbered(target, i + 1);
                Files.write(p, Imaging.png(out.get(i)));
                System.out.println(p);
            }
        }
    }

    private static Path numbered(Path p, int i) {
        String name = p.getFileName().toString();
        int dot = name.lastIndexOf('.');
        String stem = dot < 0 ? name : name.substring(0, dot);
        String ext = dot < 0 ? ".png" : name.substring(dot);
        return p.resolveSibling(stem + "-" + i + ext);
    }
}
