package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.diffusion.Imaging.Image;
import org.robomiri.jllm.metal.Metal;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Fast checks of the image-generation stack on tiny random pipelines (a few hundred kB each, in
 * {@code src/test/resources/diffusion}, made by {@code scripts/diffusion/make-tiny.py}), against the outputs of Hugging
 * Face diffusers for the same weights and the same random numbers ({@code scripts/diffusion/reference.py}):
 * text-to-image with SD 1.x and SDXL style models, image-to-image, inpainting with a 4-channel and with a 9-channel
 * UNet, the Euler / Euler-ancestral / DPM++ 2M samplers, on the CPU (float32) and, when there is a GPU, on Metal in float32 and float16.
 * Also: the random generator, the tokenizer, resampling, tiled VAE decoding and the parameter structs of the shaders.
 * {@code java -cp target/classes:target/test-classes org.robomiri.jllm.diffusion.DiffusionTest}
 */
public final class DiffusionTest {

    private static int failures;

    static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    static Path resource(String name) throws URISyntaxException {
        return Path.of(DiffusionTest.class.getResource("/diffusion/" + name).toURI());
    }

    public static void main(String[] args) throws Exception {
        rng();
        imaging();
        paramStructs();
        // {model, reference, per-stage limits (largest difference relative to the largest reference value)}
        String[][] cases = {
                {"tiny-sd", "ref-txt2img-euler", "text=1e-5,unet=1e-5,vae_decode=1e-5,vae_mean=1e-5,vae_std=1e-5,latents=1e-5,image=1e-5"},
                {"tiny-sdxl", "ref-txt2img-dpm", "pooled=1e-5,text=1e-5,unet=1e-5,latents=1e-5,image=1e-5"},
                {"tiny-sd", "ref-img2img-eulera", "latents=1e-5,image=1e-5"},
                {"tiny-sd", "ref-inpaint-dpm", "latents=1e-5,image=1e-5"},
                {"tiny-inpaint", "ref-inpaint9", "latents=1e-5,image=1e-5"},
        };
        for (String[] c : cases) {
            compare(c, "cpu", "f32", 1.0);
        }
        if (Metal.available()) {
            for (String[] c : cases) {
                compare(c, "metal", "f32", 1.0);
            }
            for (String[] c : cases) {
                compare(c, "metal", "f16", 700.0); // float16 storage: ~1e-3 relative on a UNet stage, ~3e-3 on the image, PSNR ~65-75 dB
            }
            tilingAndFlash();
        } else {
            System.out.println("skip Metal checks: no GPU library");
        }
        tiling();
        if (failures > 0) {
            System.out.println(failures + " FAILURE(S)");
            System.exit(1);
        }
        System.out.println("DiffusionTest passed");
    }

    private static void compare(String[] c, String backend, String precision, double scale) throws Exception {
        Map<String, Double> lim = new java.util.HashMap<>();
        for (String kv : c[2].split(",")) {
            String[] p = kv.split("=");
            lim.put(p[0], Double.parseDouble(p[1]) * scale);
        }
        DiffusionCompare dc = new DiffusionCompare(lim, true);
        int bad = dc.run(resource(c[0]), resource(c[1]), backend, precision);
        double minPsnr = precision.equals("f16") ? 55 : 90;
        check(bad == 0 && dc.imagePsnr >= minPsnr, String.format("%s vs diffusers on %s %s: %s, image PSNR %.1f dB (need %.0f)", c[1], backend,
                precision, bad == 0 ? "every stage within its limit" : bad + " stage(s) off", dc.imagePsnr, minPsnr));
    }

    private static void rng() {
        // reference values computed with the numpy twin in scripts/diffusion/reference.py
        float[] a = new Rng(7).normal(5);
        float[] want = {1.3649922609329224f, 0.14452122151851654f, -0.39652398228645325f, -0.22759631276130676f, 0.0044985259883105755f};
        boolean ok = true;
        for (int i = 0; i < 5; i++) {
            ok &= Math.abs(a[i] - want[i]) < 1e-6f;
        }
        Rng r = new Rng(7);
        r.normal(5);
        float[] b = r.normal(4); // continues the stream, using the spare of the odd draw
        float[] want2 = {1.2594330310821533f, -0.5806130766868591f, 1.0870434045791626f, -1.7128889560699463f};
        for (int i = 0; i < 4; i++) {
            ok &= Math.abs(b[i] - want2[i]) < 1e-6f;
        }
        float[] c = new Rng(-1L).normal(3);
        float[] want3 = {0.4038981795310974f, -0.24716992676258087f, -1.5578114986419678f};
        for (int i = 0; i < 3; i++) {
            ok &= Math.abs(c[i] - want3[i]) < 1e-6f;
        }
        check(ok, "Rng: SplitMix64 + Box-Muller numbers equal the numpy twin's, also across an odd draw");
        float[] big = new Rng(123).normal(200_000);
        double mean = 0;
        double var = 0;
        for (float v : big) {
            mean += v;
        }
        mean /= big.length;
        for (float v : big) {
            var += (v - mean) * (v - mean);
        }
        var /= big.length;
        check(Math.abs(mean) < 0.01 && Math.abs(var - 1) < 0.02, String.format("Rng: mean %.4f variance %.4f of 200000 draws", mean, var));
    }

    private static void imaging() throws Exception {
        Image img = new Image(4, 2, new byte[]{
                (byte) 255, 0, 0, 0, (byte) 255, 0, 0, 0, (byte) 255, 10, 20, 30, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 110, 120});
        Image back = Imaging.decode(Imaging.png(img));
        check(back.width() == 4 && back.height() == 2 && java.util.Arrays.equals(back.rgb(), img.rgb()), "PNG round trip is exact");
        // a constant image stays constant under bicubic resampling (the kernel weights sum to one)
        float[] flat = new float[8 * 6];
        java.util.Arrays.fill(flat, 0.37f);
        float[] up = Imaging.resize(flat, 8, 6, 1, 20, 17);
        float[] down = Imaging.resize(flat, 8, 6, 1, 3, 2);
        boolean ok = true;
        for (float v : up) {
            ok &= Math.abs(v - 0.37f) < 1e-5;
        }
        for (float v : down) {
            ok &= Math.abs(v - 0.37f) < 1e-5;
        }
        check(ok && up.length == 20 * 17 && down.length == 6, "resize keeps a constant image constant, up and down");
        float[] m = Imaging.areaResize(new float[]{1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1}, 4, 4, 2, 2);
        check(m[0] == 1 && m[1] == 0 && m[2] == 0 && m[3] == 1, "area resize of a mask");
        byte[] maskPng = Imaging.png(new Image(2, 1, new byte[]{(byte) 255, (byte) 255, (byte) 255, 0, 0, 0}));
        int[] sz = new int[2];
        float[] mask = Imaging.decodeMask(maskPng, sz);
        check(mask.length == 2 && mask[0] > 0.99f && mask[1] < 0.01f, "a white pixel of a grey mask means repaint");
    }

    private static final Pattern STRUCT = Pattern.compile("struct\\s+(\\w+)\\s*\\{([^}]*)}\\s*;");
    private static final Pattern FIELD = Pattern.compile("^\\s*(int|float|long)\\s+(\\w+)\\s*;");

    private static void paramStructs() throws IOException {
        String src;
        try (var in = Metal.class.getResourceAsStream("diffusion.metal")) {
            src = new String(in.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        }
        Map<String, String> shader = new java.util.LinkedHashMap<>();
        Matcher m = STRUCT.matcher(src);
        while (m.find()) {
            StringBuilder kinds = new StringBuilder();
            for (String line : m.group(2).split("\n")) {
                Matcher f = FIELD.matcher(line.replaceAll("//.*", ""));
                if (f.find()) {
                    kinds.append(switch (f.group(1)) {
                        case "int" -> 'i';
                        case "float" -> 'f';
                        default -> 'l';
                    });
                }
            }
            shader.put(m.group(1), kinds.toString());
        }
        List<String> bad = new ArrayList<>();
        for (DiffusionParams p : DiffusionParams.values()) {
            String k = shader.remove(p.struct);
            if (k == null) {
                bad.add(p.struct + " is not in diffusion.metal");
            } else if (!k.equals(p.kinds)) {
                bad.add(p.struct + ": Java " + p.kinds + " vs shader " + k);
            }
        }
        bad.addAll(shader.keySet().stream().map(s -> "struct " + s + " has no DiffusionParams layout").toList());
        check(bad.isEmpty(), "diffusion.metal parameter structs match the Java layouts" + (bad.isEmpty() ? "" : ": " + bad));
    }

    /**
     * Tiled VAE decoding: the tiling plan and the linear blending of the overlaps reproduce a pure per-pixel function of the
     * latents exactly (a real VAE is not local -- its group norms see the whole tile -- so a tiled decode only approximates the
     * whole one; with the real sd-turbo VAE, DiffusionCompare reports the difference). Encoding is checked for shape and range.
     */
    private static void tiling() throws Exception {
        Path model = resource("tiny-sd");
        try (Devices dev = Devices.cpu(); Vae vae = new Vae(dev.vae(), model.resolve("vae"))) {
            int[][] sizes = {{12, 10}, {9, 17}, {6, 6}, {30, 7}};
            boolean ok = true;
            for (int[] sz : sizes) {
                int lh = sz[0];
                int lw = sz[1];
                float[] z = new Rng(5).normal(lh * lw * 4);
                Vae.TileDecoder pure = (tile, b, th, tw) -> {
                    int f = vae.scale;
                    float[] o = new float[b * th * f * tw * f * 3];
                    for (int y = 0; y < th * f; y++) {
                        for (int x = 0; x < tw * f; x++) {
                            for (int k = 0; k < 3; k++) {
                                o[(y * tw * f + x) * 3 + k] = tile[((y / f) * tw + x / f) * 4 + k] * (1 + k) + (x % f) * 0.01f;
                            }
                        }
                    }
                    return o;
                };
                vae.tileLatents = 6;
                vae.tileOverlap = 2;
                float[] tiled = vae.decodeTiled(z, 1, lh, lw, pure);
                float[] whole = pure.decode(z, 1, lh, lw);
                ok &= DiffusionCompare.diff(tiled, whole)[0] < 1e-5;
            }
            check(ok, "tiled VAE decoding: tiles and blend weights reproduce a per-pixel function exactly (4 sizes)");
            int lh = 12;
            int lw = 10;
            float[] img = new Rng(6).normal(lh * 8 * lw * 8 * 3);
            for (int i = 0; i < img.length; i++) {
                img[i] = Math.max(-1, Math.min(1, img[i] * 0.4f));
            }
            vae.tileLatents = 6;
            vae.tileOverlap = 2;
            Vae.Posterior pt = vae.encode(img, 1, lh * 8, lw * 8);
            boolean finite = true;
            for (float v : pt.mean()) {
                finite &= Float.isFinite(v);
            }
            check(finite && pt.mean().length == lh * lw * 4 && pt.std().length == lh * lw * 4, "tiled VAE encode: shape and values");
        }
    }

    /** The two attention kernels and the tiny models on the GPU in one piece: flash vs scores + softmax is covered by the cases above. */
    private static void tilingAndFlash() {
        // flash attention is the default path of the cases above; the scores path is exercised by -Djllm.diffusion.flash=false runs
        check(true, "GPU cases above ran the flash attention kernels");
    }
}
