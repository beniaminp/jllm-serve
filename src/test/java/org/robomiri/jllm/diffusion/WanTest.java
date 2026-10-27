package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.metal.Metal;

import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Fast checks of the Wan 2.1 text-to-video components on a tiny random pipeline (a diffusion transformer, a causal 3-D VAE and a UMT5
 * text encoder with a Unigram tokenizer: {@code src/test/resources/diffusion/tiny-wan}, made by {@code scripts/diffusion/make-tiny-wan.py})
 * against the outputs of Hugging Face diffusers / transformers for the same weights and noise ({@code scripts/diffusion/reference_wan.py}):
 * the tokenizer, the encoder in float and int8, one transformer evaluation, the decoder, the flow sampler, and the whole pipeline; plus the
 * tokenizer on the real 256 000-piece vocabulary when it is on disk. {@code java -cp target/classes:target/test-classes org.robomiri.jllm.diffusion.WanTest}
 */
public final class WanTest {

    private static int failures;

    static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    static Path resource(String name) throws URISyntaxException {
        return Path.of(WanTest.class.getResource("/diffusion/" + name).toURI());
    }

    private static Map<String, Double> limits(String spec, double scale) {
        Map<String, Double> m = new HashMap<>();
        for (String kv : spec.split(",")) {
            String[] p = kv.split("=");
            m.put(p[0], Double.parseDouble(p[1]) * scale);
        }
        return m;
    }

    /**
     * The int8 copy of the UMT5 encoder on disk ({@code video.text-encoder-cache}), in a copy of the tiny encoder's folder: written by the
     * first encoder, mapped by the next, the same bits as a fresh quantization either way, and rewritten when the float file changes or
     * the copy is damaged; without the switch nothing is written.
     */
    private static void textCache(Path src) throws Exception {
        Path dir = Files.createTempDirectory("jllm-umt5-cache");
        try {
            for (String f : new String[]{"config.json", "model.safetensors"}) {
                Files.copy(src.resolve(f), dir.resolve(f));
            }
            int[] ids = {5, 17, 3, 99, 42, 1};
            float[] fresh;
            try (Umt5Encoder e = new Umt5Encoder(dir, true, false)) {
                fresh = e.encode(ids);
            }
            check(!Files.exists(dir.resolve(Umt5Encoder.CACHE_FILE)) && !Files.exists(dir.resolve(Umt5Encoder.CACHE_INDEX)),
                    "UMT5 cache off: nothing written next to the float file");
            try (Umt5Encoder e = new Umt5Encoder(dir, true, true)) {
                check(!e.fromCache() && Files.isRegularFile(dir.resolve(Umt5Encoder.CACHE_FILE)) && Files.isRegularFile(dir.resolve(Umt5Encoder.CACHE_INDEX)),
                        "UMT5 cache on, first encoder: quantizes and writes jllm-q8.bin + jllm-q8.json");
                check(java.util.Arrays.equals(e.encode(ids), fresh), "UMT5 cache on, first encoder: the same bits as without the cache");
            }
            try (Umt5Encoder e = new Umt5Encoder(dir, true, true)) {
                check(e.fromCache() && java.util.Arrays.equals(e.encode(ids), fresh), "UMT5 cache on, next encoder: maps the int8 copy, the same bits");
            }
            Path weights = dir.resolve("model.safetensors");
            Files.setLastModifiedTime(weights, java.nio.file.attribute.FileTime.fromMillis(Files.getLastModifiedTime(weights).toMillis() + 5000));
            try (Umt5Encoder e = new Umt5Encoder(dir, true, true)) {
                check(!e.fromCache() && java.util.Arrays.equals(e.encode(ids), fresh), "UMT5 cache: a changed float file makes it quantize (and write) again");
            }
            try (Umt5Encoder e = new Umt5Encoder(dir, true, true)) {
                check(e.fromCache(), "UMT5 cache: the rewritten copy is mapped by the next encoder");
            }
            Path bin = dir.resolve(Umt5Encoder.CACHE_FILE);
            try (java.nio.channels.FileChannel ch = java.nio.channels.FileChannel.open(bin, java.nio.file.StandardOpenOption.WRITE)) {
                ch.truncate(ch.size() / 2);
            }
            try (Umt5Encoder e = new Umt5Encoder(dir, true, true)) {
                check(!e.fromCache() && java.util.Arrays.equals(e.encode(ids), fresh), "UMT5 cache: a truncated copy is not read (quantized again)");
            }
            Files.writeString(dir.resolve(Umt5Encoder.CACHE_INDEX), "{\"version\": 0}");
            try (Umt5Encoder e = new Umt5Encoder(dir, true, true)) {
                check(!e.fromCache(), "UMT5 cache: an index of another version is not read");
            }
            try (Umt5Encoder e = new Umt5Encoder(dir, false, true)) {
                check(!e.fromCache(), "UMT5 cache: a float encoder never reads the int8 copy");
            }
        } finally {
            try (java.util.stream.Stream<Path> files = Files.list(dir)) {
                for (Path f : files.toList()) {
                    Files.delete(f);
                }
            }
            Files.delete(dir);
        }
    }

    public static void main(String[] args) throws Exception {
        scheduler();
        Path model = resource("tiny-wan");
        // tokenizer and encoder: float, then int8 weights (the quantisation error of a random 2-layer model, ~1e-2)
        {
            WanCompare c = new WanCompare(limits("umt5=1e-5", 1), true);
            check(c.t5(model, resource("ref-wan-t5"), "cpu", "f32") == 0, "tokenizer = HF tokenizers on 8 prompts (accents, CJK, runs of spaces, empty) and the UMT5 encoder = transformers (1e-5)");
            WanCompare q = new WanCompare(limits("umt5=5e-2", 1), true);
            check(q.t5(model, resource("ref-wan-t5"), "cpu", "q8") == 0, String.format("UMT5 encoder with int8 weights within 5e-2 of float (%.1e)", q.relative));
        }
        textCache(model.resolve("text_encoder"));
        Path real = Path.of(System.getProperty("jllm.wan.dir", "models/Wan-AI_Wan2.1-T2V-1.3B-Diffusers"));
        if (Files.isRegularFile(real.resolve("tokenizer").resolve("tokenizer.json"))) {
            WanCompare c = new WanCompare(Map.of(), true);
            check(c.t5(real, resource("ref-wan-realtok"), "cpu", "tok") == 0, "tokenizer on the real umt5-xxl vocabulary = HF tokenizers");
        } else {
            System.out.println("skip real-vocabulary tokenizer check (tokenizer.json not downloaded)");
        }
        for (String[] c : new String[][]{{"ref-wan-dit", "dit=1e-5"}, {"ref-wan-dit2", "dit=1e-5"}}) {
            WanCompare w = new WanCompare(limits(c[1], 1), true);
            check(w.dit(model, resource(c[0]), "cpu", "f32") == 0, c[0] + ": transformer on CPU f32 within 1e-5 of diffusers");
        }
        for (String r : new String[]{"ref-wan-vae", "ref-wan-vae2"}) {
            WanCompare w = new WanCompare(limits("vae=1e-4", 1), true);
            check(w.vae(model, resource(r), "cpu", "f32") == 0, String.format("%s: causal 3-D VAE decoder on CPU f32 within 1e-4 (PSNR %.1f dB)", r, w.relative));
        }
        for (String r : new String[]{"ref-wan-pipe", "ref-wan-pipe2"}) {
            WanCompare w = new WanCompare(limits("latents=1e-4,video=1e-4", 1), true);
            check(w.pipe(model, resource(r), "cpu", "f32") == 0 && w.relative > 100, String.format("%s: pipeline (UMT5, transformer, UniPC, VAE) on CPU f32: PSNR %.1f dB", r, w.relative));
        }
        if (Metal.available()) {
            for (String[] pr : new String[][]{{"f32", "1"}, {"f16", "700"}}) {
                double scale = Double.parseDouble(pr[1]);
                for (String r : new String[]{"ref-wan-dit", "ref-wan-dit2"}) {
                    WanCompare w = new WanCompare(limits("dit=1e-5", scale), true);
                    check(w.dit(model, resource(r), "metal", pr[0]) == 0, r + ": transformer on Metal " + pr[0]);
                }
                for (String r : new String[]{"ref-wan-vae", "ref-wan-vae2"}) {
                    double min = pr[0].equals("f16") ? 55 : 90;
                    WanCompare w = new WanCompare(limits("vae=1e-4", scale), true);
                    check(w.vae(model, resource(r), "metal", pr[0]) == 0 && w.relative > min, String.format("%s: VAE decoder on Metal %s (PSNR %.1f dB, need %.0f)", r, pr[0], w.relative, min));
                }
                for (String r : new String[]{"ref-wan-pipe", "ref-wan-pipe2"}) {
                    double min = pr[0].equals("f16") ? 55 : 100;
                    WanCompare w = new WanCompare(limits("latents=1e-4,video=1e-4", scale * (pr[0].equals("f16") ? 4 : 1)), true);
                    check(w.pipe(model, resource(r), "metal", pr[0]) == 0 && w.relative > min, String.format("%s: pipeline on Metal %s (PSNR %.1f dB, need %.0f)", r, pr[0], w.relative, min));
                }
            }
            geluTanhHuge("f32", 1e-5);
            geluTanhHuge("f16", 1.5e-2);
        } else {
            System.out.println("skip Metal checks: no GPU library");
        }
        if (failures > 0) {
            System.out.println(failures + " FAILURE(S)");
            System.exit(1);
        }
        System.out.println("WanTest passed");
    }

    /**
     * The tanh GELU of pre-activations in the hundreds (the real transformer's text projection has them; the tiny random one does not): tanh of a
     * huge argument is 1, and must not be inf / inf = NaN on the GPU.
     */
    private static void geluTanhHuge(String precision, double relTol) {
        int m = 24, k = 64, n = 72;
        java.util.Random rnd = new java.util.Random(5);
        float[] a = new float[m * k];
        float[] w = new float[n * k];
        float[] b = new float[n];
        for (int i = 0; i < a.length; i++) {
            a[i] = (float) (rnd.nextGaussian() * 12);
        }
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (rnd.nextGaussian() * 0.5);
        }
        float[][] out = new float[2][];
        try (Devices cpu = Devices.cpu(); Devices gpu = Devices.open("metal", precision, false)) {
            for (int dev = 0; dev < 2; dev++) {
                Ops ops = dev == 0 ? cpu.main() : gpu.main();
                Tn wt = ops.matrix(w, Ops.Kind.LINEAR, n, k);
                out[dev] = ops.download(ops.linear(ops.upload(a, 1, m, k), wt, ops.vector(b, 1, n), null, Ops.Act.GELU_TANH));
            }
        }
        int bad = 0;
        for (float f : out[1]) {
            bad += Float.isFinite(f) ? 0 : 1;
        }
        double[] d = DiffusionCompare.diff(out[0], out[1]);
        check(bad == 0 && d[0] <= relTol * Math.max(1, d[2]), String.format("tanh GELU of pre-activations up to %.0f on Metal %s: %d non-finite, max diff %.2e vs CPU", d[2], precision, bad, d[0]));
    }

    private static void scheduler() {
        // diffusers' UniPCMultistepScheduler(use_flow_sigmas, flow_shift 3).set_timesteps(5): timesteps and sigmas
        FlowScheduler s = new FlowScheduler(Map.of("flow_shift", 3.0, "solver_order", 2, "num_train_timesteps", 1000), FlowScheduler.Kind.UNIPC);
        s.setTimesteps(5);
        int[] ts = {999, 923, 818, 667, 429};
        double[] sg = {0.9999989867210388, 0.9231656789779663, 0.8184296488761902, 0.6672218441963196, 0.4297945201396942, 0.0};
        boolean ok = s.steps() == 5;
        for (int i = 0; ok && i < 5; i++) {
            ok &= s.timestep(i) == ts[i];
        }
        for (int i = 0; ok && i < 6; i++) {
            ok &= Math.abs((i < 5 ? s.sigma(i) : 0.0) - sg[i]) < 1e-7;
        }
        check(ok, "flow sigmas and timesteps for 5 steps equal diffusers' (999, 923, 818, 667, 429)");
    }
}
