package org.robomiri.jllm.diffusion;

import java.nio.file.Path;

/**
 * Times the Wan 2.1 transformer and VAE decoder on random inputs: {@code WanBench MODEL_DIR backend precision latentFrames latentH latentW reps [textTokens]}
 * (the transformer sees {@code latentFrames · latentH/2 · latentW/2} tokens; the decoder turns the latent into {@code 1 + 4 (frames − 1)}
 * frames of {@code 8 latentH × 8 latentW} pixels). The first call includes shader compilation. Run it under the machine's heavy lock.
 *
 * <p>{@code WanBench MODEL_DIR t5 cache(true|false) [tokens]} times the int8 UMT5 text encoder instead: two encoders one after the other,
 * each read (quantized from the float32 files, or mapped from {@code jllm-q8.bin} with the cache on) and run on {@code tokens} ids
 * (default 64). With the cache on, the first writes the int8 copy into the model's {@code text_encoder} folder and the second maps it.
 */
public final class WanBench {

    private static void t5(Path model, boolean cache, int tokens) throws Exception {
        int[] ids = new int[tokens];
        for (int i = 0; i < tokens; i++) {
            ids[i] = 2 + i * 37 % 100; // any ids work for timing; these fit the tiny test vocabulary too
        }
        for (String round : new String[]{"first", "warm "}) {
            long t0 = System.nanoTime();
            try (Umt5Encoder enc = new Umt5Encoder(model.resolve("text_encoder"), true, cache)) {
                double load = (System.nanoTime() - t0) / 1e9;
                long t1 = System.nanoTime();
                enc.encode(ids);
                System.out.printf("%s umt5 cache %s: read %.1f s, encode %.2f s (%d tokens, %.1f GB of int8 weights%s)%n", round, cache, load,
                        (System.nanoTime() - t1) / 1e9, tokens, enc.residentBytes() / 1e9, enc.fromCache() ? ", mapped from the copy" : "");
            }
        }
    }

    public static void main(String[] a) throws Exception {
        if (a.length > 1 && a[1].equals("t5")) {
            t5(Path.of(a[0]), a.length > 2 && Boolean.parseBoolean(a[2]), a.length > 3 ? Integer.parseInt(a[3]) : 64);
            return;
        }
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
