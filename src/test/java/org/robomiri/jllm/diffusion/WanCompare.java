package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.json.Json;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Compares the Wan 2.1 components with Hugging Face diffusers' outputs ({@code scripts/diffusion/reference_wan.py}):
 * <pre>
 *   java ... WanCompare dit|vae MODEL_DIR REF_DIR [cpu|metal] [f16|f32]
 * </pre>
 * {@code dit}: the transformer on a seeded latent and seeded text embeddings; {@code vae}: the causal 3-D decoder on a seeded latent.
 * With {@code -Dcompare.tol=stage=limit} it fails when the largest difference relative to the largest reference value is over the limit.
 */
public final class WanCompare {

    private final DiffusionCompare rep;
    double relative = Double.NaN;

    WanCompare(Map<String, Double> limits, boolean quiet) {
        rep = new DiffusionCompare(limits, quiet);
    }

    /** {@code [B=1, C, F, H, W]} (PyTorch) to {@code [F, H, W, C]}. */
    static float[] ncfhwToFhwc(float[] a, int c, int f, int h, int w) {
        float[] o = new float[a.length];
        for (int ch = 0; ch < c; ch++) {
            for (int fr = 0; fr < f; fr++) {
                for (int p = 0; p < h * w; p++) {
                    o[(fr * h * w + p) * c + ch] = a[(ch * f + fr) * h * w + p];
                }
            }
        }
        return o;
    }

    int dit(Path model, Path ref, String backend, String precision) throws Exception {
        Map<String, Object> meta = Json.parseObject(Files.readString(ref.resolve("ref.json")));
        int f = (int) Json.num(meta, "frames", 3);
        int h = (int) Json.num(meta, "height", 4);
        int w = (int) Json.num(meta, "width", 6);
        int c = (int) Json.num(meta, "channels", 4);
        int tl = (int) Json.num(meta, "text_len", 7);
        int td = (int) Json.num(meta, "text_dim", 32);
        float t = (float) Json.dbl(meta, "t", 700);
        long t0 = System.nanoTime();
        try (Devices dev = Devices.open(backend, precision, false); WanTransformer dit = new WanTransformer(dev.main(), model.resolve("transformer"))) {
            float[] x = ncfhwToFhwc(DiffusionCompare.floats(ref.resolve("x.f32")), c, f, h, w);
            float[] text = DiffusionCompare.floats(ref.resolve("text.f32"));
            Tn enc = dit.encodeText(text, tl);
            long u0 = System.nanoTime();
            float[] got = dit.forward(x, f, h, w, t, enc);
            if (!Boolean.getBoolean("compare.quiet")) {
                System.out.printf("dit eval       %.2f s (%d tokens)%n", (System.nanoTime() - u0) / 1e9, f * (h / 2) * (w / 2));
            }
            float[] want = ncfhwToFhwc(DiffusionCompare.floats(ref.resolve("out.f32")), dit.outChannels, f, h, w);
            rep.report("dit", got, want);
            relative = DiffusionCompare.diff(got, want)[0] / Math.max(DiffusionCompare.diff(got, want)[2], 1e-12);
        }
        if (!Boolean.getBoolean("compare.quiet")) {
            System.out.printf("total %.1f s%n", (System.nanoTime() - t0) / 1e9);
        }
        return rep.failureCount();
    }

    int vae(Path model, Path ref, String backend, String precision) throws Exception {
        Map<String, Object> meta = Json.parseObject(Files.readString(ref.resolve("ref.json")));
        int f = (int) Json.num(meta, "frames", 3);
        int h = (int) Json.num(meta, "height", 4);
        int w = (int) Json.num(meta, "width", 6);
        int z = (int) Json.num(meta, "z_dim", 4);
        java.util.List<Object> shape = Json.list(meta, "out_shape");
        int of = ((Number) shape.get(2)).intValue();
        int oh = ((Number) shape.get(3)).intValue();
        int ow = ((Number) shape.get(4)).intValue();
        long t0 = System.nanoTime();
        try (Devices dev = Devices.open(backend, precision, false); WanVae vae = new WanVae(dev.main(), model.resolve("vae"))) {
            float[] zin = ncfhwToFhwc(DiffusionCompare.floats(ref.resolve("z.f32")), z, f, h, w);
            long u0 = System.nanoTime();
            float[] got = vae.decode(zin, f, h, w);
            if (!Boolean.getBoolean("compare.quiet")) {
                System.out.printf("vae decode     %.2f s (%d frames of %dx%d)%n", (System.nanoTime() - u0) / 1e9, of, ow, oh);
            }
            float[] want = ncfhwToFhwc(DiffusionCompare.floats(ref.resolve("video.f32")), 3, of, oh, ow);
            rep.report("vae", got, want);
            relative = DiffusionCompare.psnr(got, want);
            if (!Boolean.getBoolean("compare.quiet")) {
                System.out.printf("PSNR vs diffusers: %.2f dB%n", relative);
            }
        }
        if (!Boolean.getBoolean("compare.quiet")) {
            System.out.printf("total %.1f s%n", (System.nanoTime() - t0) / 1e9);
        }
        return rep.failureCount();
    }

    /** The tokenizer on the reference's cases and the encoder (float, or int8 with {@code precision = q8}) on the reference prompt. */
    int t5(Path model, Path ref, String backend, String precision) throws Exception {
        Map<String, Object> meta = Json.parseObject(Files.readString(ref.resolve("ref.json")));
        UnigramTokenizer tok = new UnigramTokenizer(model.resolve("tokenizer").resolve("tokenizer.json"));
        int bad = 0;
        for (Object o : Json.list(meta, "cases")) {
            String text = Json.str(o, "text", "");
            java.util.List<Object> want = Json.list(o, "ids");
            int[] got = tok.encode(text);
            boolean same = got.length == want.size();
            for (int i = 0; same && i < got.length; i++) {
                same = got[i] == ((Number) want.get(i)).intValue();
            }
            if (!same) {
                System.out.println("FAIL tokens differ for '" + text + "': " + java.util.Arrays.toString(got) + " vs " + want);
                bad++;
            }
        }
        if (precision.equals("tok")) {
            return bad;
        }
        int[] ids = DiffusionCompare.ints(ref.resolve("ids.i32"));
        float[] want = DiffusionCompare.floats(ref.resolve("emb.f32"));
        long u0 = System.nanoTime();
        try (Umt5Encoder enc = new Umt5Encoder(model.resolve("text_encoder"), precision.equals("q8"))) {
            float[] got = enc.encode(ids);
            if (!Boolean.getBoolean("compare.quiet")) {
                System.out.printf("umt5 encode    %.2f s (%d tokens, %.1f MB resident, %s)%n", (System.nanoTime() - u0) / 1e9, ids.length, enc.residentBytes() / 1e6, precision);
            }
            rep.report("umt5", got, want);
            relative = DiffusionCompare.diff(got, want)[0] / Math.max(DiffusionCompare.diff(got, want)[2], 1e-12);
        }
        return rep.failureCount() + bad;
    }

    /** The whole pipeline from the same noise: the final latents and the decoded frames. */
    int pipe(Path model, Path ref, String backend, String precision) throws Exception {
        Map<String, Object> meta = Json.parseObject(Files.readString(ref.resolve("ref.json")));
        java.util.List<Object> ls = Json.list(meta, "lat_shape");
        java.util.List<Object> os = Json.list(meta, "out_shape");
        int c = ((Number) ls.get(1)).intValue();
        int lf = ((Number) ls.get(2)).intValue();
        int lh = ((Number) ls.get(3)).intValue();
        int lw = ((Number) ls.get(4)).intValue();
        int of = ((Number) os.get(2)).intValue();
        int oh = ((Number) os.get(3)).intValue();
        int ow = ((Number) os.get(4)).intValue();
        if (Files.exists(ref.resolve("prompt_embeds.f32"))) { // the text encoder with the pipeline's padding
            UnigramTokenizer tok = new UnigramTokenizer(model.resolve("tokenizer").resolve("tokenizer.json"));
            try (Umt5Encoder enc = new Umt5Encoder(model.resolve("text_encoder"), false)) {
                for (String[] pn : new String[][]{{Json.str(meta, "prompt", ""), "prompt_embeds.f32"}, {Json.str(meta, "negative", ""), "negative_embeds.f32"}}) {
                    int[] ids = tok.encode(pn[0].replaceAll("\\s+", " ").strip());
                    float[] e = enc.encode(ids);
                    float[] want = DiffusionCompare.floats(ref.resolve(pn[1]));
                    int d = e.length / ids.length;
                    float[] padded = new float[want.length];
                    System.arraycopy(e, 0, padded, 0, e.length);
                    rep.report("text " + pn[1], padded, want);
                }
            }
        }
        try (Devices dev = Devices.open(backend, precision, false); WanPipeline p = new WanPipeline(model, dev, false)) {
            WanPipeline.Request req = new WanPipeline.Request(Json.str(meta, "prompt", ""), Json.str(meta, "negative", ""), ow, oh, of,
                    (int) Json.num(meta, "steps", 3), Json.dbl(meta, "guidance", 5), Json.num(meta, "seed", 7), "unipc");
            long t0 = System.nanoTime();
            if (Files.exists(ref.resolve("step_0.f32"))) {
                p.stepHook = (i, xs) -> {
                    try {
                        rep.report("step " + i, xs, ncfhwToFhwc(DiffusionCompare.floats(ref.resolve("step_" + i + ".f32")), c, lf, lh, lw));
                    } catch (java.io.IOException e) {
                        throw new IllegalStateException(e);
                    }
                };
            }
            p.generate(req, null);
            if (!Boolean.getBoolean("compare.quiet")) {
                System.out.printf("pipeline       %.2f s (text %.2f, dit %.2f, vae %.2f)%n", (System.nanoTime() - t0) / 1e9, p.textNs / 1e9, p.ditNs / 1e9, p.vaeNs / 1e9);
            }
            rep.report("latents", p.lastLatents, ncfhwToFhwc(DiffusionCompare.floats(ref.resolve("latents.f32")), c, lf, lh, lw));
            float[] want = ncfhwToFhwc(DiffusionCompare.floats(ref.resolve("video.f32")), 3, of, oh, ow);
            rep.report("video", p.lastPixels, want);
            relative = DiffusionCompare.psnr(p.lastPixels, want);
            if (!Boolean.getBoolean("compare.quiet")) {
                System.out.printf("video PSNR vs diffusers: %.2f dB%n", relative);
            }
        }
        return rep.failureCount();
    }

    public static void main(String[] args) throws Exception {
        String stage = args[0];
        Path model = Path.of(args[1]);
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
        WanCompare wc = new WanCompare(m, false);
        int failures = switch (stage) {
            case "dit" -> wc.dit(model, ref, backend, precision);
            case "vae" -> wc.vae(model, ref, backend, precision);
            case "t5" -> wc.t5(model, ref, backend, precision);
            case "tok" -> wc.t5(model, ref, backend, "tok");
            case "pipe" -> wc.pipe(model, ref, backend, precision);
            default -> throw new IllegalArgumentException("stage " + stage);
        };
        if (failures > 0) {
            System.exit(1);
        }
    }
}
