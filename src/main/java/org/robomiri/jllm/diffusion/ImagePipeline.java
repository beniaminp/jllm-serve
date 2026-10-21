package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.diffusion.Imaging.Image;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Text-to-image, image-to-image and inpainting with a Stable Diffusion family model in the diffusers folder layout
 * (SD 1.x, SD 2.x including SD-Turbo, SDXL including SDXL-Turbo).
 *
 * <pre>
 *   text ──CLIP──▶ embeddings ─┐
 *   noise (or noised photo) ───▶ UNet ◀── repeated per step; classifier-free guidance runs it on [negative, prompt]
 *                               │   and combines  ε = ε_neg + g (ε_prompt − ε_neg)
 *                               ▼   scheduler update
 *                            latents ──VAE decoder──▶ pixels
 * </pre>
 *
 * Each image uses the random stream of {@code seed + index} (see {@link Rng}). A pipeline serves one request at a time.
 */
public final class ImagePipeline implements AutoCloseable {

    /** One generation request; every image of it shares prompt, size and settings. */
    public record Request(String prompt, String negativePrompt, int width, int height, int steps, double guidance,
                          long seed, String scheduler, int clipSkip, Image init, float[] mask, int maskWidth,
                          int maskHeight, double strength) {
        public Request withInit(Image i, float[] m, int mw, int mh, double s) {
            return new Request(prompt, negativePrompt, width, height, steps, guidance, seed, scheduler, clipSkip, i, m, mw, mh, s);
        }
    }

    /** Called while an image is generated. */
    public interface Listener {
        /**
         * @param image   index of the image of the request
         * @param step    steps done (0..steps)
         * @param preview a cheap preview of the current estimate at latent resolution, or null if {@link #wantsPreview} is false
         */
        void progress(int image, int step, int steps, Image preview);

        default boolean wantsPreview() {
            return false;
        }

        /** True to stop early (the client left). */
        default boolean cancelled() {
            return false;
        }
    }

    /** Recommended settings of this model. */
    public record Defaults(int steps, double guidance, int size, String scheduler, boolean distilled) {
    }

    public final Path dir;
    public final String className;
    public final boolean xl;
    private final Devices devices;
    private final ClipTokenizer tok1;
    private final ClipTokenizer tok2;
    private final ClipText clip1;
    private final ClipText clip2;
    private final UNet unet;
    private final Vae vae;
    private final Map<String, Object> schedulerConfig;
    private final boolean forceZerosForEmpty;
    public final Defaults defaults;

    public ImagePipeline(Path dir, Devices devices) throws IOException {
        this.dir = dir;
        this.devices = devices;
        Map<String, Object> index = Json.parseObject(Files.readString(dir.resolve("model_index.json"), StandardCharsets.UTF_8));
        className = Json.str(index, "_class_name", "");
        if (!className.startsWith("StableDiffusion")) {
            throw new IOException("unsupported pipeline " + className + " in " + dir + " (Stable Diffusion 1.x / 2.x / XL are supported)");
        }
        xl = index.containsKey("text_encoder_2");
        forceZerosForEmpty = Json.bool(index, "force_zeros_for_empty_prompt", xl);
        Ops main = devices.main();
        tok1 = new ClipTokenizer(dir.resolve("tokenizer"));
        clip1 = new ClipText(main, dir.resolve("text_encoder"));
        if (xl) {
            tok2 = new ClipTokenizer(dir.resolve("tokenizer_2"));
            clip2 = new ClipText(main, dir.resolve("text_encoder_2"));
        } else {
            tok2 = null;
            clip2 = null;
        }
        unet = new UNet(main, dir.resolve("unet"));
        vae = new Vae(devices.vae(), dir.resolve("vae"));
        Path sc = dir.resolve("scheduler").resolve("scheduler_config.json");
        schedulerConfig = Files.isRegularFile(sc) ? Json.parseObject(Files.readString(sc, StandardCharsets.UTF_8)) : Map.of();
        String name = dir.getFileName().toString().toLowerCase(Locale.ROOT);
        boolean turbo = name.contains("turbo") || Json.str(schedulerConfig, "timestep_spacing", "").equals("trailing")
                && name.contains("lcm");
        int size = xl ? 1024 : (unetSampleSize() >= 96 ? 768 : 512);
        if (turbo) {
            defaults = new Defaults(1, 0.0, xl ? 512 : 512, "euler", true);
        } else {
            defaults = new Defaults(25, 7.5, size, "dpmpp_2m_karras", false);
        }
    }

    private int unetSampleSize() {
        return unet.sampleSize * vae.scale;
    }

    /** Images larger than {@code tile} latent pixels per side go through the VAE in tiles that overlap by {@code overlap}. */
    public void setVaeTiling(int tile, int overlap) {
        vae.tileLatents = tile;
        vae.tileOverlap = overlap;
    }

    /** Pixel size must be a multiple of this (the UNet halves the latent several times). */
    public int sizeMultiple() {
        return vae.scale << (unet.blockCount - 1);
    }

    public String description() {
        return className + (xl ? " (XL)" : "") + ", unet in_channels=" + unet.inChannels + ", " + devices.description();
    }

    public boolean hasInpaintUnet() {
        return unet.inChannels == 9;
    }

    // ------------------------------------------------------------------ text

    private record Embeds(Tn encoder, float[] pooled, float[] timeIds) {
    }

    private Embeds embed(String prompt, String negative, boolean cfg, int width, int height, int clipSkip) {
        Ops ops = devices.main();
        String neg = negative == null ? "" : negative;
        boolean zeroNeg = xl && forceZerosForEmpty && neg.isBlank();
        int[][] ids1 = cfg ? new int[][]{tok1.encode(neg), tok1.encode(prompt)} : new int[][]{tok1.encode(prompt)};
        if (!xl) {
            ClipText.Output o = clip1.encode(ids1, clipSkip, true, false);
            return new Embeds(o.hidden(), null, null);
        }
        int[][] ids2 = cfg ? new int[][]{tok2.encode(neg), tok2.encode(prompt)} : new int[][]{tok2.encode(prompt)};
        ClipText.Output a = clip1.encode(ids1, 1 + clipSkip, false, false);
        ClipText.Output b = clip2.encode(ids2, 1 + clipSkip, false, true);
        Tn cat = ops.concat(a.hidden(), b.hidden());
        ops.free(a.hidden());
        ops.free(b.hidden());
        float[] pooled = b.pooled();
        if (zeroNeg && cfg) {
            // the empty negative prompt of SDXL is all zeros rather than the embedding of ""
            float[] hc = ops.download(cat);
            int per = hc.length / 2;
            java.util.Arrays.fill(hc, 0, per, 0f);
            ops.free(cat);
            cat = ops.upload(hc, cat.shape);
            java.util.Arrays.fill(pooled, 0, pooled.length / 2, 0f);
        }
        int bs = cfg ? 2 : 1;
        float[] t = new float[bs * 6];
        for (int i = 0; i < bs; i++) {
            float[] v = {height, width, 0, 0, height, width};
            System.arraycopy(v, 0, t, i * 6, 6);
        }
        return new Embeds(cat, pooled, t);
    }

    // ------------------------------------------------------------------ generation

    /** Generates {@code n} images (image k from the random stream of {@code seed + k}). */
    public synchronized List<Image> generate(Request r, int n, Listener listener) {
        validate(r);
        Ops ops = devices.main();
        boolean cfg = r.guidance() > 1.0 + 1e-9;
        long t0 = System.nanoTime();
        unetNs = 0;
        vaeNs = 0;
        Embeds emb = embed(r.prompt(), r.negativePrompt(), cfg, r.width(), r.height(), r.clipSkip());
        devices.main().sync();
        textNs = System.nanoTime() - t0;
        List<Image> out = new ArrayList<>();
        try {
            for (int k = 0; k < n; k++) {
                if (listener != null && listener.cancelled()) {
                    break;
                }
                out.add(one(r, k, cfg, emb, listener));
            }
        } finally {
            ops.free(emb.encoder());
        }
        totalNs = System.nanoTime() - t0;
        return out;
    }

    private void validate(Request r) {
        int m = sizeMultiple();
        if (r.width() < m || r.height() < m || r.width() % m != 0 || r.height() % m != 0) {
            throw new IllegalArgumentException("width and height must be multiples of " + m + " for this model, got "
                    + r.width() + "x" + r.height());
        }
        if (r.width() * (long) r.height() > 2048L * 2048L) {
            throw new IllegalArgumentException("images larger than 2048x2048 pixels are not supported");
        }
        if (r.steps() < 1 || r.steps() > 200) {
            throw new IllegalArgumentException("steps must be 1..200");
        }
        if (r.guidance() < 0 || r.guidance() > 50) {
            throw new IllegalArgumentException("guidance must be 0..50");
        }
        if (r.init() == null && r.mask() != null) {
            throw new IllegalArgumentException("a mask needs an input image");
        }
        if (r.strength() < 0 || r.strength() > 1) {
            throw new IllegalArgumentException("strength must be 0..1");
        }
    }

    /** NCHW noise of PyTorch's generators, as the NHWC layout used here. */
    private static float[] toNhwc(float[] nchw, int c, int h, int w) {
        float[] out = new float[nchw.length];
        for (int ch = 0; ch < c; ch++) {
            for (int p = 0; p < h * w; p++) {
                out[p * c + ch] = nchw[ch * h * w + p];
            }
        }
        return out;
    }

    private Image one(Request r, int index, boolean cfg, Embeds emb, Listener listener) {
        int w = r.width();
        int h = r.height();
        int lh = h / vae.scale;
        int lw = w / vae.scale;
        int lc = vae.latentChannels;
        int lp = lh * lw;
        Rng rng = new Rng(r.seed() + index);
        Scheduler sched = new Scheduler(schedulerConfig, Scheduler.Kind.parse(r.scheduler()), spacingFor(r));
        sched.setTimesteps(r.steps());
        boolean nine = hasInpaintUnet();           // the 9-channel inpainting UNet takes mask + masked image as input
        boolean inpaint = r.init() != null && r.mask() != null;
        float[] x;
        int start = 0;
        float[] initLatents = null;                // clean latents of the input image
        float[] latentMask = null;                 // [lh*lw], 1 = repaint
        float[] maskedLatents = null;              // 9-channel inpainting UNet input
        float[] pixelMask = null;
        float[] noiseForBlend = null;
        Image initImage = null;
        if (r.init() != null) {
            initImage = Imaging.resize(r.init(), w, h);
            float[] px = Imaging.toUnit(initImage);
            if (inpaint) {
                pixelMask = r.maskWidth() == w && r.maskHeight() == h ? r.mask()
                        : Imaging.resize(r.mask(), r.maskWidth(), r.maskHeight(), 1, w, h);
                latentMask = Imaging.areaResize(pixelMask, w, h, lw, lh);
            }
            // the random numbers are drawn in the order of diffusers' pipelines: the encoded image's sample (when its
            // latents are needed), the noise, then the masked image's sample (inpainting)
            boolean needImage = !inpaint || !nine || r.strength() < 1.0;
            if (needImage) {
                long e0 = System.nanoTime();
                Vae.Posterior post0 = vae.encode(px, 1, h, w);
                vaeNs += System.nanoTime() - e0;
                initLatents = sampleLatents(post0, rng, lc, lh, lw);
            }
            int initSteps = Math.min((int) (r.steps() * r.strength()), r.steps());
            start = Math.max(r.steps() - initSteps, 0);
            noiseForBlend = toNhwc(rng.normal(lc * lp), lc, lh, lw);
            if (nine) {
                float[] masked = px.clone();
                if (inpaint) {
                    for (int i = 0; i < pixelMask.length; i++) {
                        if (pixelMask[i] >= 0.5f) {
                            masked[i * 3] = masked[i * 3 + 1] = masked[i * 3 + 2] = 0f;
                        }
                    }
                } else {
                    java.util.Arrays.fill(masked, 0f);
                    latentMask = new float[lp];
                    java.util.Arrays.fill(latentMask, 1f);
                }
                long e1 = System.nanoTime();
                Vae.Posterior mp = vae.encode(masked, 1, h, w);
                vaeNs += System.nanoTime() - e1;
                maskedLatents = sampleLatents(mp, rng, lc, lh, lw);
            } else if (inpaint) {
                rng.normal(lc * lp); // the reference pipeline also samples the (unused) masked image here
            }
            if (initSteps == 0) {
                return finish(initLatents, lh, lw, initImage, pixelMask, w, h);
            }
            if (inpaint && r.strength() >= 1.0) { // the inpainting pipelines start from pure noise at full strength
                x = noiseForBlend.clone();
                float s = sched.initNoiseSigma();
                for (int i = 0; i < x.length; i++) {
                    x[i] *= s;
                }
            } else {
                x = sched.addNoise(initLatents, noiseForBlend, start);
            }
        } else {
            x = toNhwc(rng.normal(lc * lp), lc, lh, lw);
            float s = sched.initNoiseSigma();
            for (int i = 0; i < x.length; i++) {
                x[i] *= s;
            }
            if (nine) { // an inpainting UNet used without an image: everything is "to be painted"
                latentMask = new float[lp];
                java.util.Arrays.fill(latentMask, 1f);
                float[] black = new float[w * h * 3];
                java.util.Arrays.fill(black, 0f);
                maskedLatents = sampleLatents(vae.encode(black, 1, h, w), rng, lc, lh, lw);
            }
        }
        Ops ops = devices.main();
        int total = sched.steps() - start;
        if (listener != null) {
            listener.progress(index, 0, total, null);
        }
        for (int i = start; i < sched.steps(); i++) {
            if (listener != null && listener.cancelled()) {
                throw new CancellationException("cancelled");
            }
            float[] in = sched.scaleInput(x, i);
            float[] unetIn = in;
            int inC = lc;
            if (nine) {
                inC = lc + 1 + lc;
                unetIn = new float[lp * inC];
                for (int p = 0; p < lp; p++) {
                    System.arraycopy(in, p * lc, unetIn, p * inC, lc);
                    unetIn[p * inC + lc] = latentMask[p];
                    System.arraycopy(maskedLatents, p * lc, unetIn, p * inC + lc + 1, lc);
                }
            }
            int bs = cfg ? 2 : 1;
            float[] batch = new float[bs * unetIn.length];
            for (int b = 0; b < bs; b++) {
                System.arraycopy(unetIn, 0, batch, b * unetIn.length, unetIn.length);
            }
            long u0 = System.nanoTime();
            Tn sample = ops.upload(batch, bs, lh, lw, inC);
            Tn eps = unet.forward(sample, sched.timestep(i), new UNet.Cond(emb.encoder(), emb.pooled(), emb.timeIds()));
            ops.free(sample);
            float[] e = ops.download(eps);
            ops.free(eps);
            unetNs += System.nanoTime() - u0;
            int oc = unet.outChannels;
            float[] guided = new float[lp * lc];
            int per = lp * oc;
            for (int p = 0; p < lp; p++) {
                for (int c = 0; c < lc; c++) {
                    if (cfg) {
                        float eu = e[p * oc + c];
                        float ec = e[per + p * oc + c];
                        guided[p * lc + c] = eu + (float) r.guidance() * (ec - eu);
                    } else {
                        guided[p * lc + c] = e[p * oc + c];
                    }
                }
            }
            float[] prev = sched.step(guided, i, x, () -> toNhwc(rng.normal(lc * lp), lc, lh, lw));
            if (inpaint && !nine) {
                // blend the noised original back into the part that is to be kept
                float[] keep = i + 1 < sched.steps() ? sched.addNoise(initLatents, noiseForBlend, i + 1) : initLatents;
                for (int p = 0; p < lp; p++) {
                    float m = latentMask[p];
                    for (int c = 0; c < lc; c++) {
                        prev[p * lc + c] = (1 - m) * keep[p * lc + c] + m * prev[p * lc + c];
                    }
                }
            }
            Image pv = null;
            if (listener != null && listener.wantsPreview() && i + 1 < sched.steps()) {
                pv = LatentPreview.render(sched.predictOriginal(guided, i, x), lh, lw, xl);
            }
            x = prev;
            if (listener != null) {
                listener.progress(index, i + 1 - start, total, pv);
            }
        }
        return finish(x, lh, lw, initImage, pixelMask, w, h);
    }

    /** {@code (mean + std · ε) · scaling_factor} with ε from the generator's stream. */
    private float[] sampleLatents(Vae.Posterior post, Rng rng, int lc, int lh, int lw) {
        float[] eps = toNhwc(rng.normal(lc * lh * lw), lc, lh, lw);
        float[] z = new float[eps.length];
        for (int i = 0; i < z.length; i++) {
            z[i] = (post.mean()[i] + post.std()[i] * eps[i]) * vae.scalingFactor;
        }
        return z;
    }

    /** Thrown when a listener asks to stop. */
    public static final class CancellationException extends RuntimeException {
        public CancellationException(String m) {
            super(m);
        }
    }

    private String spacingFor(Request r) {
        // few-step distilled models are trained on the "trailing" schedule: keep it unless the user asks otherwise
        return defaults.distilled() ? "trailing" : null;
    }

    /** Time spent in the last {@link #generate} call: text encoders, UNet evaluations, VAE (encode and decode), everything. */
    public volatile long textNs, unetNs, vaeNs, totalNs;

    /** Diagnostics of the last image: the final latents (NHWC) and the decoded pixels before clamping (-1..1). */
    public volatile float[] lastLatents;
    public volatile float[] lastPixels;

    private Image finish(float[] x, int lh, int lw, Image initImage, float[] pixelMask, int w, int h) {
        lastLatents = x;
        float[] z = new float[x.length];
        float inv = 1f / vae.scalingFactor;
        for (int i = 0; i < z.length; i++) {
            z[i] = x[i] * inv;
        }
        long v0 = System.nanoTime();
        float[] px = vae.decode(z, 1, lh, lw);
        vaeNs += System.nanoTime() - v0;
        lastPixels = px;
        Image img = Imaging.fromUnit(px, w, h);
        if (initImage != null && pixelMask != null) {
            // keep the pixels outside the mask exactly as they were (the VAE round trip would change them slightly)
            byte[] o = img.rgb().clone();
            for (int p = 0; p < w * h; p++) {
                float m = pixelMask[p];
                if (m < 1f) {
                    for (int c = 0; c < 3; c++) {
                        float v = m * (o[p * 3 + c] & 255) + (1 - m) * (initImage.rgb()[p * 3 + c] & 255);
                        o[p * 3 + c] = (byte) Math.round(v);
                    }
                }
            }
            img = new Image(w, h, o);
        }
        return img;
    }

    @Override
    public void close() {
        unet.close();
        vae.close();
        clip1.close();
        if (clip2 != null) {
            clip2.close();
        }
        devices.close();
    }
}
