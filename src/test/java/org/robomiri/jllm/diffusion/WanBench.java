package org.robomiri.jllm.diffusion;

import java.nio.file.Path;

/**
 * Times the Wan 2.1 transformer and VAE decoder on random inputs: {@code WanBench MODEL_DIR backend precision latentFrames latentH latentW reps [textTokens]}
 * (the transformer sees {@code latentFrames · latentH/2 · latentW/2} tokens; the decoder turns the latent into {@code 1 + 4 (frames − 1)}
 * frames of {@code 8 latentH × 8 latentW} pixels). The first call includes shader compilation. Run it under the machine's heavy lock.
 */
public final class WanBench {

    public static void main(String[] a) throws Exception {
        Path model = Path.of(a[0]);
        String backend = a.length > 1 ? a[1] : "metal";
        String precision = a.length > 2 ? a[2] : "f32";
        int f = a.length > 3 ? Integer.parseInt(a[3]) : 5;
        int h = a.length > 4 ? Integer.parseInt(a[4]) : 24;
        int w = a.length > 5 ? Integer.parseInt(a[5]) : 40;
        int reps = a.length > 6 ? Integer.parseInt(a[6]) : 2;
        int tl = a.length > 7 ? Integer.parseInt(a[7]) : 64;
        Devices dev = Devices.open(backend, precision, false);
        long t0 = System.nanoTime();
        try (WanTransformer dit = new WanTransformer(dev.main(), model.resolve("transformer")); WanVae vae = new WanVae(dev.vae(), model.resolve("vae"))) {
            System.out.printf("loaded in %.1f s (%s, %.1f GB of weights on disk)%n", (System.nanoTime() - t0) / 1e9, dev.description(), (dit.weightBytes() + vae.weightBytes()) / 1e9);
            Rng rng = new Rng(1);
            float[] x = rng.normal(f * h * w * dit.inChannels);
            Tn text = dit.encodeText(rng.normal(tl * dit.textDim), tl);
            for (int i = 0; i < reps; i++) {
                long u0 = System.nanoTime();
                dit.forward(x, f, h, w, 700f, text);
                System.out.printf("%s transformer %d x %d x %d latent = %d tokens: %.2f s per evaluation%n", i == 0 ? "first " : "warm  ", f, h, w, f * (h / 2) * (w / 2),
                        (System.nanoTime() - u0) / 1e9);
            }
            float[] z = rng.normal(f * h * w * vae.zDim);
            for (int i = 0; i < reps; i++) {
                long v0 = System.nanoTime();
                vae.decode(z, f, h, w);
                System.out.printf("%s vae decode %d frames of %dx%d: %.2f s%n", i == 0 ? "first " : "warm  ", 1 + 4 * (f - 1), 8 * w, 8 * h, (System.nanoTime() - v0) / 1e9);
            }
            Runtime rt = Runtime.getRuntime();
            System.out.printf("heap used %.0f MB, GPU buffers %.0f MB (peak %.0f MB)%n", (rt.totalMemory() - rt.freeMemory()) / 1e6, MetalProfile.live(dev) / 1e6,
                    MetalProfile.peak(dev) / 1e6);
        }
    }
}
