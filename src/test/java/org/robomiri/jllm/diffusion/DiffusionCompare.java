package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Compares the diffusion stack with the outputs of Hugging Face diffusers stage by stage
 * ({@code scripts/diffusion/reference.py} writes them): token ids, text embeddings, one UNet evaluation, the VAE decoder
 * and encoder on fixed inputs, and the whole pipeline from the same noise.
 *
 * <pre>
 *   java ... DiffusionCompare MODEL_DIR REF_DIR [cpu|metal] [f16|f32]
 * </pre>
 *
 * It prints, per stage, the largest and the mean absolute difference; with {@code -Dcompare.tol=…} (a limit on the
 * largest difference relative to the largest reference value, per stage as {@code stage=limit,...}) it fails when a
 * stage is off.
 */
public final class DiffusionCompare {

    private static int failures;
    private static Map<String, Double> limits = Map.of();

    public static float[] floats(Path f) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(f)).order(ByteOrder.LITTLE_ENDIAN);
        float[] a = new float[b.remaining() / 4];
        b.asFloatBuffer().get(a);
        return a;
    }

    public static int[] ints(Path f) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(f)).order(ByteOrder.LITTLE_ENDIAN);
        int[] a = new int[b.remaining() / 4];
        b.asIntBuffer().get(a);
        return a;
    }

    static float[] nchwToNhwc(float[] a, int b, int c, int h, int w) {
        float[] o = new float[a.length];
        for (int n = 0; n < b; n++) {
            for (int ch = 0; ch < c; ch++) {
                for (int p = 0; p < h * w; p++) {
                    o[(n * h * w + p) * c + ch] = a[(n * c + ch) * h * w + p];
                }
            }
        }
        return o;
    }

    static double[] diff(float[] got, float[] want) {
        if (got.length != want.length) {
            throw new IllegalStateException("length " + got.length + " vs " + want.length);
        }
        double max = 0;
        double sum = 0;
        double ref = 0;
        for (int i = 0; i < got.length; i++) {
            double d = Math.abs(got[i] - want[i]);
            max = Math.max(max, d);
            sum += d;
            ref = Math.max(ref, Math.abs(want[i]));
        }
        return new double[]{max, sum / got.length, ref};
    }

    static void report(String stage, float[] got, float[] want) {
        double[] d = diff(got, want);
        double rel = d[0] / Math.max(d[2], 1e-12);
        Double lim = limits.get(stage);
        boolean ok = lim == null || rel <= lim;
        System.out.printf("%-14s max |diff| %.3e  mean |diff| %.3e  max |ref| %.3e  relative max %.3e%s%n", stage, d[0], d[1], d[2], rel,
                lim == null ? "" : (ok ? "  ok (<= " + lim + ")" : "  FAIL (limit " + lim + ")"));
        if (!ok) {
            failures++;
        }
    }

    static double psnr(float[] a, float[] b) {
        double mse = 0;
        for (int i = 0; i < a.length; i++) {
            double x = Math.max(0, Math.min(1, a[i] / 2 + 0.5));
            double y = Math.max(0, Math.min(1, b[i] / 2 + 0.5));
            mse += (x - y) * (x - y);
        }
        mse /= a.length;
        return mse == 0 ? 99 : 10 * Math.log10(1 / mse);
    }

    public static void main(String[] args) throws Exception {
        Path model = Path.of(args[0]);
        Path ref = Path.of(args[1]);
        String backend = args.length > 2 ? args[2] : "cpu";
        String precision = args.length > 3 ? args[3] : "f32";
        String tol = System.getProperty("compare.tol", "");
        if (!tol.isEmpty()) {
            Map<String, Double> m = new java.util.HashMap<>();
            for (String kv : tol.split(",")) {
                String[] p = kv.split("=");
                m.put(p[0], Double.parseDouble(p[1]));
            }
            limits = m;
        }
        Map<String, Object> meta = Json.parseObject(Files.readString(ref.resolve("ref.json")));
        boolean xl = Json.bool(meta, "xl", false);
        boolean cfg = Json.bool(meta, "cfg", true);
        int w = (int) Json.num(meta, "width", 64);
        int h = (int) Json.num(meta, "height", 64);
        int lh = h / 8;
        int lw = w / 8;
        int steps = (int) Json.num(meta, "steps", 3);
        double guidance = Json.dbl(meta, "guidance", 5);
        long seed = Json.num(meta, "seed", 7);
        String prompt = Json.str(meta, "prompt", "");
        String negative = Json.str(meta, "negative", "");
        String scheduler = Json.str(meta, "scheduler", "euler");
        long t0 = System.nanoTime();
        try (Devices dev = Devices.open(backend, precision, xl)) {
            Ops ops = dev.main();
            // ---- tokenizer
            ClipTokenizer tok = new ClipTokenizer(model.resolve("tokenizer"));
            int[] ids = ints(ref.resolve("ids.i32"));
            int[] mine = cfg ? concat(tok.encode(negative), tok.encode(prompt)) : tok.encode(prompt);
            System.out.println("tokens         " + (java.util.Arrays.equals(ids, mine) ? "identical" : "DIFFERENT"));
            if (!java.util.Arrays.equals(ids, mine)) {
                failures++;
            }
            for (Object o : Json.list(meta, "tok_cases")) {
                String text = Json.str(o, "text", "");
                List<Object> want = Json.list(o, "ids");
                int[] got = tok.encode(text);
                boolean same = got.length == want.size();
                for (int i = 0; same && i < got.length; i++) {
                    same = got[i] == ((Number) want.get(i)).intValue();
                }
                if (!same) {
                    System.out.println("tokens         DIFFERENT for case " + text.substring(0, Math.min(20, text.length())));
                    failures++;
                }
            }
            // ---- text encoders
            float[] wantText = floats(ref.resolve("text.f32"));
            Tn enc;
            try (ClipText c1 = new ClipText(ops, model.resolve("text_encoder"))) {
                int[][] rows = cfg ? new int[][]{tok.encode(negative), tok.encode(prompt)} : new int[][]{tok.encode(prompt)};
                if (!xl) {
                    enc = c1.encode(rows, 0, true, false).hidden();
                } else {
                    ClipTokenizer tok2 = new ClipTokenizer(model.resolve("tokenizer_2"));
                    int[][] rows2 = cfg ? new int[][]{tok2.encode(negative), tok2.encode(prompt)} : new int[][]{tok2.encode(prompt)};
                    try (ClipText c2 = new ClipText(ops, model.resolve("text_encoder_2"))) {
                        ClipText.Output a = c1.encode(rows, 1, false, false);
                        ClipText.Output b = c2.encode(rows2, 1, false, true);
                        enc = ops.concat(a.hidden(), b.hidden());
                        float[] wantPooled = floats(ref.resolve("pooled.f32"));
                        // an empty negative prompt is zeros in SDXL; the reference does that too, so compare the prompt half
                        int half = wantPooled.length / (cfg ? 2 : 1);
                        float[] gp = b.pooled();
                        report("pooled", cfg ? java.util.Arrays.copyOfRange(gp, gp.length / 2, gp.length) : gp,
                                cfg ? java.util.Arrays.copyOfRange(wantPooled, half, wantPooled.length) : wantPooled);
                    }
                }
            }
            float[] gotText = ops.download(enc);
            if (xl && cfg && negative.isBlank()) {
                int per = gotText.length / 2;
                gotText = java.util.Arrays.copyOfRange(gotText, per, gotText.length);
                wantText = java.util.Arrays.copyOfRange(wantText, per, wantText.length);
            }
            report("text", gotText, wantText);
            // ---- one UNet evaluation
            float[] noiseN = floats(ref.resolve("noise.f32"));
            Rng rng = new Rng(seed);
            float[] mineNoise = rng.normal(4 * lh * lw);
            report("noise", mineNoise, noiseN);
            try (UNet unet = new UNet(ops, model.resolve("unet"))) {
                int b = cfg ? 2 : 1;
                float[] uin = nchwToNhwc(floats(ref.resolve("unet_in.f32")), b, 4, lh, lw);
                float[] wantOut = nchwToNhwc(floats(ref.resolve("unet_out.f32")), b, 4, lh, lw);
                int inCh = 4;
                if (Files.exists(ref.resolve("unet_extra.f32"))) { // an inpainting UNet: mask + masked-image latents follow
                    float[] extra = nchwToNhwc(floats(ref.resolve("unet_extra.f32")), 1, 5, lh, lw);
                    float[] full = new float[b * lh * lw * 9];
                    for (int n = 0; n < b; n++) {
                        for (int px = 0; px < lh * lw; px++) {
                            System.arraycopy(uin, (n * lh * lw + px) * 4, full, (n * lh * lw + px) * 9, 4);
                            System.arraycopy(extra, px * 5, full, (n * lh * lw + px) * 9 + 4, 5);
                        }
                    }
                    uin = full;
                    inCh = 9;
                }
                Tn x = ops.upload(uin, b, lh, lw, inCh);
                float[] pooled = null;
                float[] timeIds = null;
                if (xl) {
                    pooled = floats(ref.resolve("pooled.f32"));
                    timeIds = new float[b * 6];
                    for (int i = 0; i < b; i++) {
                        float[] v = {h, w, 0, 0, h, w};
                        System.arraycopy(v, 0, timeIds, i * 6, 6);
                    }
                }
                Tn encFull = enc;
                if (xl && cfg && negative.isBlank()) {
                    float[] e = floats(ref.resolve("text.f32"));
                    encFull = ops.upload(e, 2, 77, e.length / 154);
                }
                long u0 = System.nanoTime();
                Tn out = unet.forward(x, (float) Json.dbl(meta, "t0", 0), new UNet.Cond(encFull, pooled, timeIds));
                float[] got = ops.download(out);
                System.out.printf("unet eval      %.2f s%n", (System.nanoTime() - u0) / 1e9);
                report("unet", got, wantOut);
            }
            // ---- VAE
            try (Vae vae = new Vae(dev.vae(), model.resolve("vae"))) {
                float[] z = nchwToNhwc(floats(ref.resolve("vae_z.f32")), 1, 4, lh, lw);
                long v0 = System.nanoTime();
                for (int i = 0; i < z.length; i++) {
                    z[i] /= vae.scalingFactor;
                }
                float[] dec = vae.decode(z, 1, lh, lw);
                System.out.printf("vae decode     %.2f s%n", (System.nanoTime() - v0) / 1e9);
                report("vae_decode", dec, floats(ref.resolve("vae_dec.f32")));
                float[] img = floats(ref.resolve("vae_img.f32"));
                Vae.Posterior p = vae.encode(img, 1, h, w);
                report("vae_mean", p.mean(), nchwToNhwc(floats(ref.resolve("vae_mean.f32")), 1, 4, lh, lw));
                report("vae_std", p.std(), nchwToNhwc(floats(ref.resolve("vae_std.f32")), 1, 4, lh, lw));
            }
            // ---- the pipeline
            if (Files.exists(ref.resolve("latents.f32"))) {
                try (ImagePipeline pipe = new ImagePipeline(model, new Devices(dev.main(), dev.vae(), dev.description(), null))) {
                    long p0 = System.nanoTime();
                    String mode = Json.str(meta, "mode", "txt2img");
                    double strength = mode.equals("txt2img") ? 1.0 : Json.dbl(meta, "strength", 0.6);
                    Imaging.Image init = null;
                    float[] mask = null;
                    if (!mode.equals("txt2img")) {
                        init = Imaging.decode(Files.readAllBytes(ref.resolve("init.png")));
                    }
                    if (mode.equals("inpaint")) {
                        int[] sz = new int[2];
                        mask = Imaging.decodeMask(Files.readAllBytes(ref.resolve("mask.png")), sz);
                    }
                    ImagePipeline.Request r = new ImagePipeline.Request(prompt, negative, w, h, steps, guidance, seed, scheduler,
                            0, init, mask, w, h, strength);
                    pipe.generate(r, 1, null);
                    System.out.printf("pipeline       %.2f s%n", (System.nanoTime() - p0) / 1e9);
                    report("latents", pipe.lastLatents, nchwToNhwc(floats(ref.resolve("latents.f32")), 1, 4, lh, lw));
                    float[] wantImg = floats(ref.resolve("image.f32"));
                    report("image", pipe.lastPixels, wantImg);
                    System.out.printf("image PSNR vs diffusers: %.2f dB%n", psnr(pipe.lastPixels, wantImg));
                }
            }
        }
        System.out.printf("total %.1f s, %d failure(s)%n", (System.nanoTime() - t0) / 1e9, failures);
        if (failures > 0) {
            System.exit(1);
        }
    }

    private static int[] concat(int[] a, int[] b) {
        int[] o = new int[a.length + b.length];
        System.arraycopy(a, 0, o, 0, a.length);
        System.arraycopy(b, 0, o, a.length, b.length);
        return o;
    }
}
