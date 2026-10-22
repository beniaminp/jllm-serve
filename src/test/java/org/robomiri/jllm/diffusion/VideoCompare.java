package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.json.Json;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Compares the video pipeline (AnimateDiff on Stable Diffusion 1.x) with the outputs of diffusers' {@code AnimateDiffPipeline}
 * stage by stage ({@code scripts/diffusion/reference_video.py} writes them): the text embeddings, the initial noise, one UNet
 * evaluation on a whole clip (temporal attention included), the latents after every step and at the end, and the decoded frames.
 *
 * <pre>
 *   java ... VideoCompare BASE_DIR ADAPTER_DIR REF_DIR [cpu|metal] [f16|f32]
 * </pre>
 *
 * With {@code -Dcompare.tol=stage=limit,...} (limits on the largest difference relative to the largest reference value) it fails
 * when a stage is off.
 */
public final class VideoCompare {

    private final DiffusionCompare rep;
    /** PSNR of the decoded frames against the reference (all frames together), dB. */
    double framesPsnr = Double.NaN;
    /** Per-frame PSNR, dB. */
    double[] framePsnr;

    VideoCompare(Map<String, Double> limits, boolean quiet) {
        rep = new DiffusionCompare(limits, quiet);
    }

    /** {@code [B, C, F, h, w]} (PyTorch's video layout) to {@code [B·F, h·w, C]}. */
    static float[] bcfhwToBfhwc(float[] a, int b, int c, int f, int hw) {
        float[] o = new float[a.length];
        for (int n = 0; n < b; n++) {
            for (int ch = 0; ch < c; ch++) {
                for (int fr = 0; fr < f; fr++) {
                    for (int p = 0; p < hw; p++) {
                        o[((n * f + fr) * hw + p) * c + ch] = a[((n * c + ch) * f + fr) * hw + p];
                    }
                }
            }
        }
        return o;
    }

    static float[] floats(Path f) throws java.io.IOException {
        return DiffusionCompare.floats(f);
    }

    /** @return the number of stages that exceeded their limit */
    int run(Path base, Path adapter, Path ref, String backend, String precision) throws Exception {
        Map<String, Object> meta = Json.parseObject(Files.readString(ref.resolve("ref.json")));
        boolean cfg = Json.bool(meta, "cfg", true);
        int w = (int) Json.num(meta, "width", 64);
        int h = (int) Json.num(meta, "height", 64);
        int nf = (int) Json.num(meta, "frames", 4);
        int lh = h / 8;
        int lw = w / 8;
        int hw = lh * lw;
        int steps = (int) Json.num(meta, "steps", 3);
        double guidance = Json.dbl(meta, "guidance", 5);
        long seed = Json.num(meta, "seed", 7);
        String prompt = Json.str(meta, "prompt", "");
        String negative = Json.str(meta, "negative", "");
        String scheduler = Json.str(meta, "scheduler", "ddim");
        long t0 = System.nanoTime();
        try (Devices dev = Devices.open(backend, precision, false);
             VideoPipeline pipe = new VideoPipeline(base, adapter, dev)) {
            Ops ops = dev.main();
            if (Files.exists(ref.resolve("noise.f32"))) {
                rep.report("noise", new Rng(seed).normal(4 * nf * hw), floats(ref.resolve("noise.f32")));
            }
            if (Files.exists(ref.resolve("unet_out.f32"))) {
                // one UNet call on the whole clip, with the text embeddings repeated per frame like diffusers' repeat_interleave
                int b = cfg ? 2 : 1;
                float[] uin = bcfhwToBfhwc(floats(ref.resolve("unet_in.f32")), b, 4, nf, hw);
                float[] wantOut = bcfhwToBfhwc(floats(ref.resolve("unet_out.f32")), b, 4, nf, hw);
                float[] text = floats(ref.resolve("text.f32"));
                int per = text.length / b;
                float[] rep2 = new float[text.length * nf];
                for (int i = 0; i < b; i++) {
                    for (int f = 0; f < nf; f++) {
                        System.arraycopy(text, i * per, rep2, (i * nf + f) * per, per);
                    }
                }
                Tn enc = ops.upload(rep2, b * nf, 77, per / 77);
                Tn x = ops.upload(uin, b * nf, lh, lw, 4);
                long u0 = System.nanoTime();
                Tn out = pipe.unetForTest().forward(x, (float) Json.dbl(meta, "t0", 0), new UNet.Cond(enc, null, null), nf);
                float[] got = ops.download(out);
                if (!Boolean.getBoolean("compare.quiet")) {
                    System.out.printf("unet eval      %.2f s%n", (System.nanoTime() - u0) / 1e9);
                }
                rep.report("unet", got, wantOut);
            }
            if (Files.exists(ref.resolve("step_0.f32"))) {
                pipe.stepHook = (i, xs) -> {
                    try {
                        rep.report("step " + i, xs, bcfhwToBfhwc(floats(ref.resolve("step_" + i + ".f32")), 1, 4, nf, hw));
                    } catch (java.io.IOException e) {
                        throw new IllegalStateException(e);
                    }
                };
            }
            long p0 = System.nanoTime();
            VideoPipeline.Request r = new VideoPipeline.Request(prompt, negative, w, h, nf, steps, guidance, seed, scheduler, 0);
            pipe.generate(r, null);
            if (!Boolean.getBoolean("compare.quiet")) {
                System.out.printf("pipeline       %.2f s (text %.2f, unet %.2f, vae %.2f)%n", (System.nanoTime() - p0) / 1e9,
                        pipe.textNs / 1e9, pipe.unetNs / 1e9, pipe.vaeNs / 1e9);
            }
            rep.report("latents", pipe.lastLatents, bcfhwToBfhwc(floats(ref.resolve("latents.f32")), 1, 4, nf, hw));
            float[] wantFrames = floats(ref.resolve("frames.f32"));
            rep.report("frames", pipe.lastPixels, wantFrames);
            framesPsnr = DiffusionCompare.psnr(pipe.lastPixels, wantFrames);
            int fl = w * h * 3;
            framePsnr = new double[nf];
            for (int f = 0; f < nf; f++) {
                framePsnr[f] = DiffusionCompare.psnr(java.util.Arrays.copyOfRange(pipe.lastPixels, f * fl, (f + 1) * fl),
                        java.util.Arrays.copyOfRange(wantFrames, f * fl, (f + 1) * fl));
            }
            if (!Boolean.getBoolean("compare.quiet")) {
                System.out.printf("frames PSNR vs diffusers: %.2f dB (worst frame %.2f dB)%n", framesPsnr,
                        java.util.Arrays.stream(framePsnr).min().orElse(0));
            }
        }
        if (!Boolean.getBoolean("compare.quiet")) {
            System.out.printf("total %.1f s%n", (System.nanoTime() - t0) / 1e9);
        }
        return rep.failureCount();
    }

    public static void main(String[] args) throws Exception {
        Path base = Path.of(args[0]);
        Path adapter = Path.of(args[1]);
        Path ref = Path.of(args[2]);
        String backend = args.length > 3 ? args[3] : "cpu";
        String precision = args.length > 4 ? args[4] : "f32";
        Map<String, Double> m = new HashMap<>();
        String tol = System.getProperty("compare.tol", "");
        if (!tol.isEmpty()) {
            for (String kv : tol.split(",")) {
                String[] p = kv.split("=");
                m.put(p[0], Double.parseDouble(p[1]));
            }
        }
        int failures = new VideoCompare(m, false).run(base, adapter, ref, backend, precision);
        if (failures > 0) {
            System.exit(1);
        }
    }
}
