package org.robomiri.jllm.diffusion;

import java.nio.file.Path;

/**
 * Times image generation: {@code DiffusionBench MODEL_DIR backend precision steps width height reps [guidance]}.
 * The first generation includes shader compilation and warm-up and is reported apart. Run it under the machine's heavy lock.
 * {@code -Djllm.diffusion.profile=true} prints the GPU time of every kernel (each dispatch waited for: slower overall).
 */
public final class DiffusionBench {

    public static void main(String[] a) throws Exception {
        Path model = Path.of(a[0]);
        String backend = a.length > 1 ? a[1] : "metal";
        String precision = a.length > 2 ? a[2] : "f16";
        int steps = a.length > 3 ? Integer.parseInt(a[3]) : 4;
        int w = a.length > 4 ? Integer.parseInt(a[4]) : 512;
        int h = a.length > 5 ? Integer.parseInt(a[5]) : 512;
        int reps = a.length > 6 ? Integer.parseInt(a[6]) : 2;
        double guidance = a.length > 7 ? Double.parseDouble(a[7]) : 0.0;
        long t0 = System.nanoTime();
        boolean xl = java.nio.file.Files.exists(model.resolve("text_encoder_2"));
        Devices dev = Devices.open(backend, precision, xl);
        try (ImagePipeline pipe = new ImagePipeline(model, dev)) {
            System.out.printf("loaded %s in %.1f s (%s)%n", model.getFileName(), (System.nanoTime() - t0) / 1e9, pipe.description());
            for (int i = 0; i < reps; i++) {
                ImagePipeline.Request r = new ImagePipeline.Request("a photo of an astronaut riding a horse on the moon", "", w, h,
                        steps, guidance, 42 + i, "euler", 0, null, null, 0, 0, 1.0);
                if (MetalProfile.reset(dev)) {
                    System.out.println("(profiling: each dispatch is waited for)");
                }
                pipe.generate(r, 1, null);
                System.out.printf("%s %dx%d, %d step(s), guidance %.1f: %.2f s total = text %.2f + unet %.2f (%.2f s/eval) + vae %.2f + other %.2f%n",
                        i == 0 ? "first " : "warm  ", w, h, steps, guidance, pipe.totalNs / 1e9, pipe.textNs / 1e9, pipe.unetNs / 1e9,
                        pipe.unetNs / 1e9 / steps, pipe.vaeNs / 1e9,
                        (pipe.totalNs - pipe.textNs - pipe.unetNs - pipe.vaeNs) / 1e9);
                String prof = MetalProfile.report(dev);
                if (!prof.isEmpty()) {
                    System.out.print(prof);
                }
            }
            Runtime rt = Runtime.getRuntime();
            System.out.printf("heap used %.0f MB, GPU buffers %.0f MB (peak %.0f MB)%n", (rt.totalMemory() - rt.freeMemory()) / 1e6,
                    MetalProfile.live(dev) / 1e6, MetalProfile.peak(dev) / 1e6);
        }
    }
}
