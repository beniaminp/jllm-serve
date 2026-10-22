package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.diffusion.Imaging.Image;
import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Text-to-video with AnimateDiff: a Stable Diffusion 1.x model whose UNet got a <em>motion adapter</em> (a temporal
 * transformer after every ResNet / attention pair, see {@link UNet#attachMotion}).
 *
 * <pre>
 *   text ──CLIP──▶ embeddings (repeated for every frame)
 *   noise [F frames × h × w × 4] ──▶ UNet(frames in the batch, temporal attention across them) ◀── repeated per step;
 *                                    guidance runs it on [negative, prompt] clips and combines ε = ε_neg + g (ε_prompt − ε_neg)
 *                                    scheduler update of the whole clip
 *   latents ──VAE decoder, one frame at a time──▶ F pictures
 * </pre>
 *
 * The random numbers are drawn in the order of diffusers' {@code AnimateDiffPipeline}: one tensor of shape
 * {@code [4, F, h, w]} (channels first, then frames) from the stream of {@code seed}. Like {@link ImagePipeline} (whose text
 * encoder, UNet weights, VAE and device it uses) it serves one request at a time.
 */
public final class VideoPipeline implements AutoCloseable {

    /** One generation request. */
    public record Request(String prompt, String negativePrompt, int width, int height, int frames, int steps, double guidance,
                          long seed, String scheduler, int clipSkip) {
    }

    /** Called while a clip is generated. */
    public interface Listener {
        /**
         * @param step    steps done (0..steps)
         * @param preview a cheap preview (first frame at latent resolution), or null if {@link #wantsPreview} is false
         */
        void progress(int step, int steps, Image preview);

        /** Called after each frame is decoded to pixels (0 &lt; frame &le; frames). */
        default void decoded(int frame, int frames) {
        }

        default boolean wantsPreview() {
            return false;
        }

        /** True to stop early (the client left). */
        default boolean cancelled() {
            return false;
        }
    }

    /** Recommended settings of this model. */
    public record Defaults(int frames, int fps, int steps, double guidance, int size, String scheduler) {
    }

    public final Path baseDir;
    public final Path adapterDir;
    public final Defaults defaults;
    /** The SD 1.x pipeline whose text encoder, UNet, VAE and device are used. */
    private final ImagePipeline base;
    private final UNet unet;
    private final Vae vae;
    private final Devices devices;
    private final Map<String, Object> schedulerConfig;
    /** Frames the VAE decodes at once. */
    private int decodeChunk = 1;

    public VideoPipeline(Path baseDir, Path adapterDir, Devices devices) throws IOException {
        this.baseDir = baseDir;
        this.adapterDir = adapterDir;
        this.devices = devices;
        base = new ImagePipeline(baseDir, devices);
        try {
            if (base.xl) {
                throw new IOException("AnimateDiff motion adapters exist for Stable Diffusion 1.x models; " + baseDir + " is an SDXL model");
            }
            if (base.hasInpaintUnet()) {
                throw new IOException(baseDir + " is an inpainting model: use a plain text-to-image Stable Diffusion 1.x model as the base");
            }
            unet = base.unet();
            vae = base.vae();
            unet.attachMotion(adapterDir);
        } catch (IOException | RuntimeException e) {
            base.close();
            throw e;
        }
        // AnimateDiff was trained with the "linear" beta schedule (not Stable Diffusion's scaled linear) and linspace timesteps
        Map<String, Object> c = new LinkedHashMap<>(base.schedulerConfig());
        c.put("beta_schedule", "linear");
        c.put("beta_start", 0.00085);
        c.put("beta_end", 0.012);
        c.put("clip_sample", false);
        c.put("timestep_spacing", "linspace");
        c.put("steps_offset", 1);
        schedulerConfig = c;
        int size = base.defaults.size();
        defaults = new Defaults(Math.min(16, unet.motionMaxFrames()), 8, 25, 7.5, Math.min(size, 512), "ddim");
    }

    /** The UNet with the motion adapter attached (tests compare single evaluations). */
    UNet unetForTest() {
        return unet;
    }

    /** Frames the VAE decodes in one call (memory against speed; 1 keeps the peak lowest). */
    public void setDecodeChunk(int frames) {
        decodeChunk = Math.max(1, frames);
    }

    /** Pixel size must be a multiple of this. */
    public int sizeMultiple() {
        return base.sizeMultiple();
    }

    /** The longest clip the motion adapter supports. */
    public int maxFrames() {
        return unet.motionMaxFrames();
    }

    public String description() {
        return "AnimateDiff on " + base.className + ", motion adapter " + adapterDir.getFileName() + ", " + devices.description();
    }

    public long weightBytes() {
        return base.weightBytes();
    }

    public long memoryBytes() {
        return base.memoryBytes();
    }

    /** Images larger than {@code tile} latent pixels per side go through the VAE in tiles that overlap by {@code overlap}. */
    public void setVaeTiling(int tile, int overlap) {
        base.setVaeTiling(tile, overlap);
    }

    /** Time spent in the last {@link #generate} call: text encoder, UNet evaluations, VAE decoding, everything. */
    /** Benchmark switch (-Djllm.video.nomotion=true): run the frames as independent images, to time what the motion modules add. The clip is garbage. */
    private static final boolean NO_MOTION = Boolean.getBoolean("jllm.video.nomotion");
    public volatile long textNs, unetNs, vaeNs, totalNs;

    /** Diagnostics of the last clip: the final latents (NHWC per frame: {@code [F, h, w, 4]}) and the decoded pixels before clamping. */
    public volatile float[] lastLatents;
    public volatile float[] lastPixels;

    /** Debugging: called with the latents ({@code [F, h, w, 4]}) after every sampler step. */
    public volatile java.util.function.BiConsumer<Integer, float[]> stepHook;

    /** Thrown when a listener asks to stop. */
    public static final class CancellationException extends RuntimeException {
        public CancellationException(String m) {
            super(m);
        }
    }

    private void validate(Request r) {
        int m = sizeMultiple();
        if (r.width() < m || r.height() < m || r.width() % m != 0 || r.height() % m != 0) {
            throw new IllegalArgumentException("width and height must be multiples of " + m + " for this model, got "
                    + r.width() + "x" + r.height());
        }
        if (r.width() * (long) r.height() > 1024L * 1024L) {
            throw new IllegalArgumentException("clips larger than 1024x1024 pixels are not supported");
        }
        if (r.frames() < 1 || r.frames() > maxFrames()) {
            throw new IllegalArgumentException("frames must be 1.." + maxFrames() + " for this motion adapter");
        }
        if (r.steps() < 1 || r.steps() > 200) {
            throw new IllegalArgumentException("steps must be 1..200");
        }
        if (r.guidance() < 0 || r.guidance() > 50) {
            throw new IllegalArgumentException("guidance must be 0..50");
        }
    }

    /**
     * {@code [C, F, h, w]} (PyTorch's order) to the layout used here, {@code [F, h·w, C]}.
     */
    static float[] toFrames(float[] cfhw, int c, int f, int hw) {
        float[] out = new float[cfhw.length];
        for (int ch = 0; ch < c; ch++) {
            for (int fr = 0; fr < f; fr++) {
                int src = (ch * f + fr) * hw;
                for (int p = 0; p < hw; p++) {
                    out[(fr * hw + p) * c + ch] = cfhw[src + p];
                }
            }
        }
        return out;
    }

    /** Generates one clip. */
    public synchronized List<Image> generate(Request r, Listener listener) {
        validate(r);
        Ops ops = devices.main();
        int f = r.frames();
        int lh = r.height() / vae.scale;
        int lw = r.width() / vae.scale;
        int lc = vae.latentChannels;
        int lp = lh * lw;
        int frameLen = lp * lc;
        boolean cfg = r.guidance() > 1.0 + 1e-9;
        long t0 = System.nanoTime();
        unetNs = 0;
        vaeNs = 0;
        // text: [bs, 77, C], then each row repeated for every frame of its clip (diffusers' repeat_interleave)
        Tn enc0 = base.textEmbeddings(r.prompt(), r.negativePrompt(), cfg, r.clipSkip());
        int bs = cfg ? 2 : 1;
        float[] te = ops.download(enc0);
        ops.free(enc0);
        int perClip = te.length / bs;
        float[] rep = new float[te.length * f];
        for (int b = 0; b < bs; b++) {
            for (int fr = 0; fr < f; fr++) {
                System.arraycopy(te, b * perClip, rep, (b * f + fr) * perClip, perClip);
            }
        }
        Tn enc = ops.upload(rep, bs * f, enc0.dim(1), enc0.dim(2));
        ops.sync();
        textNs = System.nanoTime() - t0;
        try {
            Rng rng = new Rng(r.seed());
            Scheduler sched = new Scheduler(schedulerConfig, Scheduler.Kind.parse(r.scheduler()), null);
            sched.setTimesteps(r.steps());
            float[] x = toFrames(rng.normal(lc * f * lp), lc, f, lp);
            float init = sched.initNoiseSigma();
            for (int i = 0; i < x.length; i++) {
                x[i] *= init;
            }
            if (listener != null) {
                listener.progress(0, sched.steps(), null);
            }
            for (int i = 0; i < sched.steps(); i++) {
                if (listener != null && listener.cancelled()) {
                    throw new CancellationException("cancelled");
                }
                float[] in = sched.scaleInput(x, i);
                float[] batch = in;
                if (cfg) {
                    batch = new float[2 * in.length];
                    System.arraycopy(in, 0, batch, 0, in.length);
                    System.arraycopy(in, 0, batch, in.length, in.length);
                }
                long u0 = System.nanoTime();
                Tn sample = ops.upload(batch, bs * f, lh, lw, lc);
                Tn eps = unet.forward(sample, sched.timestep(i), new UNet.Cond(enc, null, null), NO_MOTION ? 1 : f);
                ops.free(sample);
                float[] e = ops.download(eps);
                ops.free(eps);
                unetNs += System.nanoTime() - u0;
                float[] guided = new float[x.length];
                if (cfg) {
                    float g = (float) r.guidance();
                    for (int k = 0; k < guided.length; k++) {
                        float eu = e[k];
                        float ec = e[x.length + k];
                        guided[k] = eu + g * (ec - eu);
                    }
                } else {
                    System.arraycopy(e, 0, guided, 0, guided.length);
                }
                Image pv = null;
                if (listener != null && listener.wantsPreview() && i + 1 < sched.steps()) {
                    float[] x0 = sched.predictOriginal(guided, i, x);
                    pv = LatentPreview.render(java.util.Arrays.copyOf(x0, frameLen), lh, lw, false);
                }
                x = sched.step(guided, i, x, () -> toFrames(rng.normal(lc * f * lp), lc, f, lp));
                if (stepHook != null) {
                    stepHook.accept(i, x);
                }
                if (listener != null) {
                    listener.progress(i + 1, sched.steps(), pv);
                }
            }
            lastLatents = x;
            return decode(x, f, lh, lw, r.width(), r.height(), listener, t0);
        } finally {
            ops.free(enc);
            devices.main().trim(1L << 30);
            devices.vae().trim(1L << 30);
        }
    }

    private List<Image> decode(float[] x, int f, int lh, int lw, int w, int h, Listener listener, long t0) {
        int frameLen = lh * lw * vae.latentChannels;
        float inv = 1f / vae.scalingFactor;
        List<Image> out = new ArrayList<>();
        float[] allPx = new float[f * w * h * 3];
        long v0 = System.nanoTime();
        for (int fr = 0; fr < f; ) {
            if (listener != null && listener.cancelled()) {
                throw new CancellationException("cancelled");
            }
            int n = Math.min(decodeChunk, f - fr);
            float[] z = new float[n * frameLen];
            for (int k = 0; k < z.length; k++) {
                z[k] = x[fr * frameLen + k] * inv;
            }
            float[] px = vae.decode(z, n, lh, lw);
            System.arraycopy(px, 0, allPx, fr * w * h * 3, px.length);
            for (int k = 0; k < n; k++) {
                out.add(Imaging.fromUnit(java.util.Arrays.copyOfRange(px, k * w * h * 3, (k + 1) * w * h * 3), w, h));
            }
            fr += n;
            if (listener != null) {
                listener.decoded(fr, f);
            }
        }
        vaeNs = System.nanoTime() - v0;
        lastPixels = allPx;
        totalNs = System.nanoTime() - t0;
        return out;
    }

    @Override
    public void close() {
        base.close();
    }

    /** Describes the model for {@code /v1/models}-style listings. */
    public Map<String, Object> info() {
        return Json.obj("steps", defaults.steps(), "guidance", defaults.guidance(), "size", defaults.size(), "frames", defaults.frames(),
                "fps", defaults.fps(), "scheduler", defaults.scheduler(), "max_frames", maxFrames());
    }
}
