package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.diffusion.Imaging.Image;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Text-to-video with Wan 2.1 (diffusers layout): UMT5 text encoder → diffusion transformer over the video latent, flow matching with
 * classifier-free guidance → causal 3-D VAE.
 *
 * <pre>
 *   prompt ──UnigramTokenizer──▶ ids ──UMT5 (CPU, int8)──▶ [L, 4096] (zero-padded to 512 rows) ─┐
 *   seed ──▶ noise [F', h, w, 16] with F' = (frames − 1)/4 + 1, h = H/8, w = W/8                  │ cross-attention
 *   for each step:  v = v_neg + g (v_pos − v_neg)  with v = DiT(x, t, text)                       ◀┘   two network calls per step
 *                   x = FlowScheduler.step(v, i, x)
 *   x · std + mean ──VAE decode──▶ 1 + 4 (F' − 1) frames
 * </pre>
 *
 * The text encoder is large (5.7 B parameters): it is read and quantized to int8 when a request arrives, used for the prompt and the
 * negative prompt and released before the transformer starts, so that its 4.6 GB and the transformer's weights are never needed at once.
 */
public final class WanPipeline implements AutoCloseable {

    /** One generation request. */
    public record Request(String prompt, String negativePrompt, int width, int height, int frames, int steps, double guidance, long seed,
                          String scheduler) {
    }

    /** Called while a clip is generated. */
    public interface Listener {
        void progress(int step, int steps);

        default boolean cancelled() {
            return false;
        }
    }

    public static final int MAX_TEXT = 512;

    public final Path dir;
    private final Devices devices;
    private final WanTransformer dit;
    private final WanVae vae;
    private final Map<String, Object> schedulerConfig;
    private final boolean q8;
    private final UnigramTokenizer tokenizer;
    public volatile long textNs, ditNs, vaeNs, totalNs;
    public volatile float[] lastLatents;

    /**
     * @param q8 quantize the text encoder to int8 while reading it (needed for the real 5.7 B model; the tiny test model runs in float)
     */
    public WanPipeline(Path dir, Devices devices, boolean q8) throws IOException {
        this.dir = dir;
        this.devices = devices;
        this.q8 = q8;
        Map<String, Object> index = Json.parseObject(Files.readString(dir.resolve("model_index.json"), StandardCharsets.UTF_8));
        if (!"WanPipeline".equals(Json.str(index, "_class_name", ""))) {
            throw new IOException("unsupported pipeline " + Json.str(index, "_class_name", "") + " in " + dir + " (Wan 2.1 text-to-video: WanPipeline)");
        }
        tokenizer = new UnigramTokenizer(dir.resolve("tokenizer").resolve("tokenizer.json"));
        dit = new WanTransformer(devices.main(), dir.resolve("transformer"));
        WanVae v;
        try {
            v = new WanVae(devices.vae(), dir.resolve("vae"));
        } catch (IOException | RuntimeException e) {
            dit.close();
            throw e;
        }
        vae = v;
        Path sc = dir.resolve("scheduler").resolve("scheduler_config.json");
        schedulerConfig = Files.isRegularFile(sc) ? Json.parseObject(Files.readString(sc, StandardCharsets.UTF_8)) : Map.of();
    }

    public int sizeMultiple() {
        return 16;
    }

    public long weightBytes() {
        return dit.weightBytes() + vae.weightBytes();
    }

    private static String clean(String s) {
        return s.replaceAll("\\s+", " ").strip();
    }

    private int[] ids(String text) {
        int[] ids = tokenizer.encode(clean(text));
        if (ids.length > MAX_TEXT) {
            ids = java.util.Arrays.copyOf(ids, MAX_TEXT);
            ids[MAX_TEXT - 1] = tokenizer.eos();
        }
        return ids;
    }

    /** {@code [N, 4096]} UMT5 rows of the valid tokens, zero-padded to {@link #MAX_TEXT} rows (diffusers' {@code _get_t5_prompt_embeds}). */
    private float[] padded(float[] emb, int tokens, int d) {
        float[] out = new float[MAX_TEXT * d];
        System.arraycopy(emb, 0, out, 0, tokens * d);
        return out;
    }

    public synchronized List<Image> generate(Request r, Listener listener) throws IOException {
        int m = sizeMultiple();
        if (r.width() % m != 0 || r.height() % m != 0 || r.width() < m || r.height() < m) {
            throw new IllegalArgumentException("width and height must be multiples of " + m + ", got " + r.width() + "x" + r.height());
        }
        if (r.frames() < 1 || (r.frames() - 1) % 4 != 0) {
            throw new IllegalArgumentException("frames must be 4k + 1 (1, 5, 9, ... 81), got " + r.frames());
        }
        if (r.steps() < 1 || r.steps() > 200) {
            throw new IllegalArgumentException("steps must be 1..200");
        }
        boolean cfg = r.guidance() > 1.0 + 1e-9;
        long t0 = System.nanoTime();
        // 1. text
        Tn cond;
        Tn uncond = null;
        int[] pid = ids(r.prompt());
        int[] nid = ids(r.negativePrompt() == null ? "" : r.negativePrompt());
        float[] pe;
        float[] ne = null;
        try (Umt5Encoder enc = new Umt5Encoder(dir.resolve("text_encoder"), q8)) {
            pe = enc.encode(pid);
            if (cfg) {
                ne = enc.encode(nid);
            }
        }
        int d = pe.length / pid.length;
        cond = dit.encodeText(padded(pe, pid.length, d), MAX_TEXT);
        if (cfg) {
            uncond = dit.encodeText(padded(ne, nid.length, d), MAX_TEXT);
        }
        devices.main().sync();
        textNs = System.nanoTime() - t0;
        ditNs = 0;
        vaeNs = 0;
        try {
            int lf = (r.frames() - 1) / 4 + 1;
            int lh = r.height() / 8;
            int lw = r.width() / 8;
            int c = dit.inChannels;
            float[] nchw = new Rng(r.seed()).normal(c * lf * lh * lw);
            float[] x = new float[nchw.length];
            for (int ch = 0; ch < c; ch++) { // PyTorch's [C, F, h, w] to [F, h, w, C]
                for (int p = 0; p < lf * lh * lw; p++) {
                    x[p * c + ch] = nchw[ch * lf * lh * lw + p];
                }
            }
            FlowScheduler sched = new FlowScheduler(schedulerConfig, FlowScheduler.Kind.parse(r.scheduler()));
            sched.setTimesteps(r.steps());
            if (listener != null) {
                listener.progress(0, sched.steps());
            }
            for (int i = 0; i < sched.steps(); i++) {
                if (listener != null && listener.cancelled()) {
                    throw new VideoPipeline.CancellationException("cancelled");
                }
                long u0 = System.nanoTime();
                float[] v = dit.forward(x, lf, lh, lw, sched.timestep(i), cond);
                if (cfg) {
                    float[] vu = dit.forward(x, lf, lh, lw, sched.timestep(i), uncond);
                    float g = (float) r.guidance();
                    for (int k = 0; k < v.length; k++) {
                        v[k] = vu[k] + g * (v[k] - vu[k]);
                    }
                }
                ditNs += System.nanoTime() - u0;
                x = sched.step(v, i, x);
                if (stepHook != null) {
                    stepHook.accept(i, x);
                }
                if (listener != null) {
                    listener.progress(i + 1, sched.steps());
                }
            }
            lastLatents = x;
            // 2. decode: x / (1/std) + mean per channel
            float[] z = new float[x.length];
            for (int p = 0; p < x.length / c; p++) {
                for (int ch = 0; ch < c; ch++) {
                    z[p * c + ch] = x[p * c + ch] * vae.latentsStd[ch] + vae.latentsMean[ch];
                }
            }
            long v0 = System.nanoTime();
            float[] px = vae.decode(z, lf, lh, lw);
            vaeNs = System.nanoTime() - v0;
            lastPixels = px;
            int of = 1 + 4 * (lf - 1);
            int w = lw * 8;
            int h = lh * 8;
            List<Image> out = new ArrayList<>();
            for (int f = 0; f < of; f++) {
                out.add(Imaging.fromUnit(java.util.Arrays.copyOfRange(px, f * w * h * 3, (f + 1) * w * h * 3), w, h));
            }
            totalNs = System.nanoTime() - t0;
            return out;
        } finally {
            devices.main().free(cond);
            if (uncond != null) {
                devices.main().free(uncond);
            }
            devices.main().trim(1L << 30);
        }
    }

    /** Debugging: called with the latents ({@code [F', h, w, 16]}) after every sampler step. */
    public volatile java.util.function.BiConsumer<Integer, float[]> stepHook;

    /** The decoded frames before 8-bit rounding ({@code [F', H, W, 3]}, −1..1), for comparisons. */
    public volatile float[] lastPixels;

    @Override
    public void close() {
        dit.close();
        vae.close();
        devices.close();
    }
}
