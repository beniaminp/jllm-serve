package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.diffusion.Imaging.Image;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.safetensors.SafeTensors;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Fast checks of video generation on a tiny random AnimateDiff pipeline (the tiny Stable Diffusion 1.x pipeline of the image
 * tests plus a tiny random motion adapter, {@code scripts/diffusion/make-tiny-video.py}) against the outputs of diffusers'
 * {@code AnimateDiffPipeline} for the same weights and random numbers ({@code scripts/diffusion/reference_video.py}); the
 * GPU operations the temporal layers need against the CPU; the file writers (animated WebP, GIF, MP4) by round trips; and the
 * JSON API. {@code java -cp target/classes:target/test-classes org.robomiri.jllm.diffusion.VideoTest}
 */
public final class VideoTest {

    private static int failures;

    /** The delay (cs) of each of the first {@code n} frames of a GIF, from its graphic control extensions. */
    static int[] gifDelays(ImageReader rd, int n) throws java.io.IOException {
        int[] d = new int[n];
        for (int i = 0; i < n; i++) {
            org.w3c.dom.Node root = rd.getImageMetadata(i).getAsTree("javax_imageio_gif_image_1.0");
            for (org.w3c.dom.Node c = root.getFirstChild(); c != null; c = c.getNextSibling()) {
                if (c.getNodeName().equals("GraphicControlExtension")) {
                    d[i] = Integer.parseInt(c.getAttributes().getNamedItem("delayTime").getNodeValue());
                }
            }
        }
        return d;
    }

    static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    static Path resource(String name) throws URISyntaxException {
        return Path.of(VideoTest.class.getResource("/diffusion/" + name).toURI());
    }

    public static void main(String[] args) throws Exception {
        scheduler();
        encoders();
        engineServer();
        modelDownload();
        // {reference, per-stage limits (largest difference relative to the largest reference value)}
        String[][] cases = {
                {"ref-video-ddim", "unet=1e-5,step 0=1e-5,step 1=1e-5,step 2=1e-5,latents=1e-5,frames=1e-5"},
                {"ref-video-euler", "unet=1e-5,latents=1e-5,frames=1e-5"},
                {"ref-video-dpmpp_2m", "unet=1e-5,latents=1e-5,frames=1e-5"},
        };
        for (String[] c : cases) {
            compare(c, "cpu", "f32", 1.0);
        }
        if (Metal.available()) {
            for (String[] c : cases) {
                compare(c, "metal", "f32", 1.0);
            }
            for (String[] c : cases) {
                compare(c, "metal", "f16", 1500.0); // float16 storage: ~1e-3 relative on a UNet stage, ~1e-2 on Euler latents (sigma up to 14.6), PSNR 60-65 dB on the frames
            }
            opsParity("f32", 3e-5);
            opsParity("f16", 1.5e-2);
        } else {
            System.out.println("skip Metal checks: no GPU library");
        }
        if (failures > 0) {
            System.out.println(failures + " FAILURE(S)");
            System.exit(1);
        }
        System.out.println("VideoTest passed");
    }

    private static void compare(String[] c, String backend, String precision, double scale) throws Exception {
        Map<String, Double> lim = new HashMap<>();
        for (String kv : c[1].split(",")) {
            String[] p = kv.split("=");
            lim.put(p[0], Double.parseDouble(p[1]) * scale);
        }
        VideoCompare vc = new VideoCompare(lim, true);
        int bad = vc.run(resource("tiny-sd"), resource("tiny-motion-adapter"), resource(c[0]), backend, precision);
        double minPsnr = precision.equals("f16") ? 55 : 90;
        double worst = java.util.Arrays.stream(vc.framePsnr).min().orElse(0);
        check(bad == 0 && worst >= minPsnr, String.format("%s vs diffusers on %s %s: %s, frames PSNR %.1f dB (worst frame %.1f, need %.0f)", c[0],
                backend, precision, bad == 0 ? "every stage within its limit" : bad + " stage(s) off", vc.framesPsnr, worst, minPsnr));
    }

    // ------------------------------------------------------------------ scheduler

    private static void scheduler() {
        // DDIM timesteps as diffusers' DDIMScheduler(linspace, steps_offset 1) makes them
        Map<String, Object> cfg = new HashMap<>(Map.of("_class_name", "DDIMScheduler", "beta_schedule", "linear", "beta_start", 0.00085,
                "beta_end", 0.012, "timestep_spacing", "linspace", "steps_offset", 1, "clip_sample", false, "set_alpha_to_one", false));
        Scheduler s = new Scheduler(cfg, Scheduler.Kind.DDIM, null);
        s.setTimesteps(4);
        check(s.steps() == 4 && s.timestep(0) == 999 && s.timestep(1) == 666 && s.timestep(2) == 333 && s.timestep(3) == 0,
                "DDIM linspace timesteps for 4 steps are 999, 666, 333, 0");
        s.setTimesteps(25);
        check(s.timestep(0) == 999 && s.timestep(24) == 0 && s.timestep(1) == 957 && s.timestep(2) == 916, "DDIM linspace timesteps for 25 steps are numpy.linspace(0, 999, 25).round() reversed: 999, 957, 916, ... 0");
        Map<String, Object> lead = new HashMap<>(cfg);
        lead.put("timestep_spacing", "leading");
        Scheduler l = new Scheduler(lead, Scheduler.Kind.DDIM, null);
        l.setTimesteps(4);
        check(l.timestep(0) == 751 && l.timestep(3) == 1, "DDIM leading timesteps with steps_offset 1 for 4 steps: 751, 501, 251, 1");
        // at the final step DDIM lands on x0 (prev alpha 1 or alpha[0]) and x0 is recovered from the exact noise
        float[] x0 = {0.3f, -0.7f, 1.2f};
        float[] noise = {0.5f, 1.5f, -0.2f};
        s.setTimesteps(3);
        float[] xt = s.addNoise(x0, noise, 0);
        float[] back = s.predictOriginal(noise, 0, xt);
        boolean ok = true;
        for (int i = 0; i < 3; i++) {
            ok &= Math.abs(back[i] - x0[i]) < 1e-5;
        }
        check(ok, "DDIM addNoise then predictOriginal with the true noise returns the clean latents");
        check(Scheduler.Kind.parse("DDIM") == Scheduler.Kind.DDIM, "scheduler name ddim is known");
    }

    // ------------------------------------------------------------------ ops of the temporal layers

    private static float[] noise(long seed, int n, float scale) {
        float[] f = new Rng(seed).normal(n);
        for (int i = 0; i < n; i++) {
            f[i] *= scale;
        }
        return f;
    }

    private static SafeTensors.Tensor tensor(float[] data, long... shape) {
        return DiffusionTest.tensor(data, shape);
    }

    private static void opsParity(String precision, double relTol) {
        try (Devices cpu = Devices.cpu(); Devices gpu = Devices.open("metal", precision, false)) {
            Ops c = cpu.main();
            Ops g = gpu.main();
            // swapAxes: [outer, d1, d2, C] -> [outer, d2, d1, C]
            for (int[] k : new int[][]{{2, 5, 7, 12}, {1, 16, 64, 40}, {3, 1, 9, 8}}) {
                int o = k[0], d1 = k[1], d2 = k[2], ch = k[3];
                float[] xin = noise(31, o * d1 * d2 * ch, 1f);
                float[][] out = new float[2][];
                for (int dev = 0; dev < 2; dev++) {
                    Ops ops = dev == 0 ? c : g;
                    out[dev] = ops.download(ops.swapAxes(ops.upload(xin, o * d1, d2, ch), o, d1, d2, o * d2, d1, ch));
                }
                double[] d = DiffusionCompare.diff(out[0], out[1]);
                check(d[0] <= relTol * Math.max(1, d[2]), String.format("swapAxes %dx%dx%dx%d %s vs CPU: max diff %.2e", o, d1, d2, ch, precision, d[0]));
                // and it really moves the element: out[o][j2][j1][c] = in[o][j1][j2][c]
                boolean right = true;
                for (int oo = 0; oo < o && right; oo++) {
                    for (int j1 = 0; j1 < d1; j1++) {
                        for (int j2 = 0; j2 < d2; j2++) {
                            right &= out[0][((oo * d2 + j2) * d1 + j1) * ch] == xin[((oo * d1 + j1) * d2 + j2) * ch];
                        }
                    }
                }
                check(right, String.format("swapAxes %dx%dx%dx%d places element (j1, j2) at (j2, j1)", o, d1, d2, ch));
            }
            // addRows: x[r] + table[r mod period]
            {
                int rows = 70, ch = 48, period = 16;
                float[] xin = noise(32, rows * ch, 1f);
                float[] tab = noise(33, 32 * ch, 1f);
                float[][] out = new float[2][];
                for (int dev = 0; dev < 2; dev++) {
                    Ops ops = dev == 0 ? c : g;
                    out[dev] = ops.download(ops.addRows(ops.upload(xin, rows, ch), ops.vector(tab, 32, ch), period));
                }
                double[] d = DiffusionCompare.diff(out[0], out[1]);
                check(d[0] <= relTol * Math.max(1, d[2]), String.format("addRows %s vs CPU: max diff %.2e", precision, d[0]));
                boolean right = true;
                for (int r = 0; r < rows; r++) {
                    right &= Math.abs(out[0][r * ch + 5] - (xin[r * ch + 5] + tab[(r % period) * ch + 5])) < 1e-6;
                }
                check(right, "addRows adds row r mod period of the table");
            }
            // group norm over several frames of one clip: [B*F, H, W, C] with samples = B
            for (int[] k : new int[][]{{2, 3, 6, 6, 32, 8}, {2, 8, 24, 20, 64, 16}}) {
                int b = k[0], f = k[1], h = k[2], w = k[3], ch = k[4], gr = k[5];
                float[] xin = noise(34, b * f * h * w * ch, 1.5f);
                float[] gam = noise(35, ch, 0.5f);
                float[] bet = noise(36, ch, 0.5f);
                float[][] out = new float[2][];
                for (int dev = 0; dev < 2; dev++) {
                    Ops ops = dev == 0 ? c : g;
                    out[dev] = ops.download(ops.groupNorm(ops.upload(xin, b * f, h, w, ch), ops.weight(tensor(gam, ch), Ops.Kind.VECTOR),
                            ops.weight(tensor(bet, ch), Ops.Kind.VECTOR), gr, 1e-6f, false, b));
                }
                double[] d = DiffusionCompare.diff(out[0], out[1]);
                check(d[0] <= relTol * Math.max(1, d[2]), String.format("groupNorm over %d frames (%dx%d, c=%d, %d samples) %s vs CPU: max diff %.2e",
                        f, h, w, ch, b, precision, d[0]));
            }
        }
    }

    // ------------------------------------------------------------------ file writers

    static List<Image> sampleFrames(int w, int h, int n, boolean noisy) {
        List<Image> frames = new ArrayList<>();
        Rng rng = new Rng(11);
        for (int f = 0; f < n; f++) {
            byte[] rgb = new byte[w * h * 3];
            float[] nz = noisy ? rng.normal(w * h) : null;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int o = (y * w + x) * 3;
                    double cx = w * (0.2 + 0.6 * f / Math.max(1, n - 1));
                    boolean disc = (x - cx) * (x - cx) + (y - h * 0.5) * (y - h * 0.5) < (h * 0.2) * (h * 0.2);
                    double r = 0.5 + 0.5 * Math.sin(x / (double) w * 6 + f * 0.3);
                    double g = 0.5 + 0.5 * Math.cos(y / (double) h * 5);
                    double b = (x + y) / (double) (w + h);
                    if (disc) {
                        r = 0.9;
                        g = 0.2;
                        b = 0.1;
                    }
                    double nv = noisy ? nz[y * w + x] * 0.04 : 0;
                    rgb[o] = (byte) Math.max(0, Math.min(255, Math.round((r + nv) * 255)));
                    rgb[o + 1] = (byte) Math.max(0, Math.min(255, Math.round((g + nv) * 255)));
                    rgb[o + 2] = (byte) Math.max(0, Math.min(255, Math.round((b + nv) * 255)));
                }
            }
            frames.add(new Image(w, h, rgb));
        }
        return frames;
    }

    private static double psnr(byte[] a, byte[] b) {
        double mse = 0;
        for (int i = 0; i < a.length; i++) {
            double d = (a[i] & 255) - (b[i] & 255);
            mse += d * d;
        }
        mse /= a.length;
        return mse == 0 ? 99 : 10 * Math.log10(255.0 * 255.0 / mse);
    }

    private static void encoders() throws Exception {
        for (boolean noisy : new boolean[]{false, true}) {
            for (int[] sz : new int[][]{{64, 48}, {37, 29}, {1, 1}, {3, 200}}) {
                List<Image> frames = sampleFrames(sz[0], sz[1], 4, noisy);
                byte[] webp = VideoEncoder.webp(frames, 8);
                List<Vp8lDecoder.Frame> back = Vp8lDecoder.parseWebp(webp);
                boolean same = back.size() == frames.size();
                int total = 0;
                for (int i = 0; same && i < back.size(); i++) {
                    same &= java.util.Arrays.equals(back.get(i).rgb(), frames.get(i).rgb());
                    total += back.get(i).durationMs();
                }
                check(same, String.format("animated WebP %dx%d%s: %d frames decode to exactly the pixels written (%d bytes, raw %d)", sz[0], sz[1],
                        noisy ? " noisy" : "", frames.size(), webp.length, sz[0] * sz[1] * 3 * frames.size()));
                check(total == 500, "animated WebP durations add up to 4 frames at 8 fps = 500 ms (" + total + ")");
            }
        }
        List<Image> one = sampleFrames(40, 30, 1, true);
        List<Vp8lDecoder.Frame> still = Vp8lDecoder.parseWebp(VideoEncoder.webp(one, 8));
        check(still.size() == 1 && java.util.Arrays.equals(still.get(0).rgb(), one.get(0).rgb()), "a single frame is written as a still lossless WebP");
        // a larger, smooth frame: the predictor and the run-length coding must make it much smaller than raw
        List<Image> big = sampleFrames(256, 192, 2, false);
        byte[] bigWebp = VideoEncoder.webp(big, 8);
        List<Vp8lDecoder.Frame> bigBack = Vp8lDecoder.parseWebp(bigWebp);
        check(java.util.Arrays.equals(bigBack.get(1).rgb(), big.get(1).rgb()) && bigWebp.length < 256 * 192 * 3 * 2 / 4,
                "smooth 256x192 frames: lossless and under a quarter of the raw size (" + bigWebp.length + " of " + 256 * 192 * 3 * 2 + " bytes)");

        // GIF: header, frame count, delays and picture content through the JDK's own decoder
        List<Image> frames = sampleFrames(64, 48, 5, false);
        byte[] gif = VideoEncoder.gif(frames, 8);
        check(new String(gif, 0, 6, java.nio.charset.StandardCharsets.US_ASCII).equals("GIF89a") && gif[gif.length - 1] == 0x3B, "GIF has its header and trailer");
        try (ImageInputStream in = ImageIO.createImageInputStream(new ByteArrayInputStream(gif))) {
            Iterator<ImageReader> it = ImageIO.getImageReadersByFormatName("gif");
            ImageReader rd = it.next();
            rd.setInput(in);
            int n = rd.getNumImages(true);
            check(n == 5, "GIF decodes to 5 frames (" + n + ")");
            double worst = 99;
            for (int i = 0; i < n; i++) {
                BufferedImage bi = rd.read(i);
                byte[] rgb = new byte[64 * 48 * 3];
                for (int y = 0; y < 48; y++) {
                    for (int x = 0; x < 64; x++) {
                        int p = bi.getRGB(x, y);
                        rgb[(y * 64 + x) * 3] = (byte) (p >> 16);
                        rgb[(y * 64 + x) * 3 + 1] = (byte) (p >> 8);
                        rgb[(y * 64 + x) * 3 + 2] = (byte) p;
                    }
                }
                worst = Math.min(worst, psnr(rgb, frames.get(i).rgb()));
            }
            check(worst > 25, String.format("GIF frames are close to the originals (worst PSNR %.1f dB)", worst));
            check(java.util.Arrays.equals(gifDelays(rd, n), new int[]{13, 12, 13, 12, 13}), "GIF delays at 8 fps are 13, 12, 13, 12, 13 cs (the running total stays exact)");
            rd.dispose();
        }
        // above 50 fps a frame would last 1 cs, which browsers play as 10 cs: frames are dropped instead, keeping the clip's length
        boolean same = true;
        for (double fps : new double[]{1, 7.5, 8, 12, 16, 24, 25, 30, 33.3, 49.9, 50}) {
            int[][] t = VideoEncoder.gifTiming(40, fps);
            same &= t.length == 40;
            for (int i = 0; same && i < t.length; i++) {
                same = t[i][0] == i && t[i][1] == VideoEncoder.delay(i, fps, 100);
            }
        }
        check(same, "GIF timing up to 50 fps keeps every frame with the exact-total delays");
        boolean fast = true;
        for (double fps : new double[]{50.1, 55, 60, 75, 99, 100, 144, 240}) {
            for (int n : new int[]{1, 2, 3, 7, 16, 61}) {
                int[][] t = VideoEncoder.gifTiming(n, fps);
                int total = 0;
                for (int j = 0; j < t.length; j++) {
                    fast &= t[j][1] >= 2 && (j == 0 ? t[j][0] == 0 : t[j][0] > t[j - 1][0]) && t[j][0] < n;
                    total += t[j][1];
                }
                fast &= total == Math.max(2, Math.round(n * 100 / fps)); // the clip's length, or the shortest still a viewer shows
            }
        }
        check(fast, "GIF timing above 50 fps: every delay at least 2 cs, frames in order, total = the clip's length");
        List<Image> quick = sampleFrames(32, 24, 12, false);
        try (ImageInputStream in = ImageIO.createImageInputStream(new ByteArrayInputStream(VideoEncoder.gif(quick, 60)))) {
            ImageReader rd = ImageIO.getImageReadersByFormatName("gif").next();
            rd.setInput(in);
            int n = rd.getNumImages(true);
            int[] d = gifDelays(rd, n);
            check(n == 8 && java.util.Arrays.equals(d, new int[]{2, 3, 2, 3, 2, 3, 2, 3}),
                    "12 frames at 60 fps: 8 frames of 2 or 3 cs, 20 cs in all (" + n + " frames, " + java.util.Arrays.toString(d) + ")");
            rd.dispose();
        }
        // a clip with a full 256-colour noise picture exercises LZW table resets and code widths
        List<Image> nz = sampleFrames(120, 100, 2, true);
        byte[] gifNoise = VideoEncoder.gif(nz, 12);
        try (ImageInputStream in = ImageIO.createImageInputStream(new ByteArrayInputStream(gifNoise))) {
            ImageReader rd = ImageIO.getImageReadersByFormatName("gif").next();
            rd.setInput(in);
            BufferedImage bi = rd.read(1);
            check(bi.getWidth() == 120 && bi.getHeight() == 100, "GIF of a noisy clip (LZW resets) decodes");
            rd.dispose();
        }
        String dump = System.getProperty("video.dump");
        if (dump != null) { // for looking at the files with other tools (PIL, ffprobe, a browser)
            List<Image> d = sampleFrames(200, 150, 6, true);
            Files.write(Path.of(dump, "sample.webp"), VideoEncoder.webp(d, 8));
            Files.write(Path.of(dump, "sample.gif"), VideoEncoder.gif(d, 8));
            for (int i = 0; i < d.size(); i++) {
                Files.write(Path.of(dump, "frame" + i + ".png"), Imaging.png(d.get(i)));
            }
            String f = VideoEncoder.resolveFfmpeg("auto");
            if (f != null) {
                Files.write(Path.of(dump, "sample.mp4"), VideoEncoder.mp4(d, 8, f));
            }
        }
        String ff = VideoEncoder.resolveFfmpeg("auto");
        if (ff != null) {
            byte[] mp4 = VideoEncoder.mp4(sampleFrames(64, 48, 6, false), 8, ff);
            check(mp4.length > 100 && mp4[4] == 'f' && mp4[5] == 't' && mp4[6] == 'y' && mp4[7] == 'p', "MP4 from ffmpeg starts with an ftyp box (" + mp4.length + " bytes)");
        } else {
            System.out.println("skip MP4 check: no ffmpeg");
        }
        check(VideoEncoder.Format.parse("auto", true) == VideoEncoder.Format.MP4 && VideoEncoder.Format.parse("auto", false) == VideoEncoder.Format.WEBP
                && VideoEncoder.Format.parse("GIF", false) == VideoEncoder.Format.GIF, "format names: auto picks MP4 only with ffmpeg");
        boolean threw = false;
        try {
            VideoEncoder.mp4(frames, 8, null);
        } catch (java.io.IOException e) {
            threw = e.getMessage().contains("ffmpeg");
        }
        check(threw, "MP4 without ffmpeg is a clear error");
    }

    // ------------------------------------------------------------------ fetching a motion adapter from a (fake) Hub

    private static void modelDownload() throws Exception {
        Path adapter = resource("tiny-motion-adapter");
        byte[] cfg = Files.readAllBytes(adapter.resolve("config.json"));
        byte[] weights = Files.readAllBytes(adapter.resolve("diffusion_pytorch_model.safetensors"));
        com.sun.net.httpserver.HttpServer hub = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress("127.0.0.1", 0), 0);
        java.util.concurrent.atomic.AtomicInteger weightRequests = new java.util.concurrent.atomic.AtomicInteger();
        hub.createContext("/", ex -> {
            String path = ex.getRequestURI().getPath();
            byte[] body;
            if (path.equals("/api/models/test/adapter/revision/main")) {
                body = ("{\"sha\":\"abc\",\"siblings\":[{\"rfilename\":\"README.md\",\"size\":3},{\"rfilename\":\"config.json\",\"size\":" + cfg.length
                        + "},{\"rfilename\":\"diffusion_pytorch_model.fp16.safetensors\",\"size\":" + weights.length + "},{\"rfilename\":\"diffusion_pytorch_model.safetensors\",\"size\":"
                        + weights.length + "}]}").getBytes(java.nio.charset.StandardCharsets.UTF_8);
            } else if (path.equals("/test/adapter/resolve/main/config.json")) {
                body = cfg;
            } else if (path.equals("/test/adapter/resolve/main/diffusion_pytorch_model.fp16.safetensors")) {
                weightRequests.incrementAndGet();
                body = weights;
            } else {
                ex.sendResponseHeaders(404, -1);
                ex.close();
                return;
            }
            ex.sendResponseHeaders(200, body.length);
            ex.getResponseBody().write(body);
            ex.close();
        });
        hub.start();
        Path tmp = Files.createTempDirectory("jllm-video-hub");
        try {
            org.robomiri.jllm.config.Settings s = org.robomiri.jllm.config.Settings.load(new String[]{"video", "--model.dir", tmp.toString()});
            org.robomiri.jllm.hub.HuggingFaceHub h = new org.robomiri.jllm.hub.HuggingFaceHub("http://127.0.0.1:" + hub.getAddress().getPort(), null);
            Path dir = VideoModels.resolveAdapter(s, h, "test/adapter");
            check(dir.equals(tmp.resolve("test_adapter")) && VideoModels.isAdapterDir(dir) && Files.size(dir.resolve("diffusion_pytorch_model.fp16.safetensors")) == weights.length
                            && !Files.exists(dir.resolve("diffusion_pytorch_model.safetensors")) && !Files.exists(dir.resolve("README.md")),
                    "motion adapter download: config.json and only the fp16 weights, into model.dir/owner_name");
            VideoModels.resolveAdapter(s, h, "test/adapter");
            check(weightRequests.get() == 1, "a motion adapter already on disk is not downloaded again");
            check(VideoModels.isAdapterRepo(h.listFiles("test/adapter", "main"), new String(cfg, java.nio.charset.StandardCharsets.UTF_8))
                    && !VideoModels.isAdapterRepo(java.util.List.of(), "{}") && VideoModels.defaultId("guoyww/animatediff-motion-adapter-v1-5-2").equals("animatediff-motion-adapter-v1-5-2"),
                    "a repository with a MotionAdapter config.json is recognised");
        } finally {
            hub.stop(0);
        }
    }

    // ------------------------------------------------------------------ the engine's own HTTP server

    private static void engineServer() throws Exception {
        VideoStore store = new VideoStore(2, 1000);
        String a = store.put(new byte[400], VideoEncoder.Format.GIF);
        String b = store.put(new byte[400], VideoEncoder.Format.WEBP);
        String c = store.put(new byte[400], VideoEncoder.Format.MP4);
        check(store.get(a) == null && store.get(b).ext().equals("webp") && store.get(c).mime().equals("video/mp4") && a.length() == 24,
                "clip store: oldest evicted beyond its limits, 24-hex ids, the file type is kept");
        try (Devices dev = Devices.cpu();
             VideoPipeline pipe = new VideoPipeline(resource("tiny-sd"), resource("tiny-motion-adapter"), dev);
             org.robomiri.jllm.server.ApiServer server = new org.robomiri.jllm.server.ApiServer(null, "none", "127.0.0.1", 0,
                     org.robomiri.jllm.server.ApiServer.Options.defaults())
                     .withVideo(new org.robomiri.jllm.server.VideoHttp(pipe, "tiny-video", new VideoApi.Config(null, "auto")))) {
            Thread t = Thread.ofVirtual().start(server::serve);
            String base = "http://127.0.0.1:" + server.port();
            java.net.http.HttpClient http = java.net.http.HttpClient.newHttpClient();
            java.util.function.Function<String, java.net.http.HttpRequest.Builder> req = path -> java.net.http.HttpRequest.newBuilder(java.net.URI.create(base + path));
            java.net.http.HttpResponse<String> r = http.send(req.apply("/v1/videos/generations").header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"prompt\":\"a cat\",\"size\":\"64x64\",\"num_frames\":3,\"steps\":2,\"guidance\":3,\"seed\":7,\"n\":2,"
                            + "\"response_format\":\"url\"}")).build(), java.net.http.HttpResponse.BodyHandlers.ofString());
            Map<String, Object> out = Json.parseObject(r.body());
            List<Object> data = Json.list(out, "data");
            String url = data == null ? "" : Json.str(data.get(0), "url", "");
            check(r.statusCode() == 200 && data.size() == 2 && url.contains("/v1/videos/files/") && url.endsWith(".webp") && Json.num(data.get(1), "seed", 0) == 8,
                    "engine server: POST /v1/videos/generations with n=2 and response_format url (webp without ffmpeg)");
            java.net.http.HttpResponse<byte[]> f = http.send(req.apply(url.substring(url.indexOf("/v1/"))).GET().build(), java.net.http.HttpResponse.BodyHandlers.ofByteArray());
            check(f.statusCode() == 200 && Vp8lDecoder.parseWebp(f.body()).size() == 3 && f.headers().firstValue("content-type").orElse("").equals("image/webp"),
                    "engine server: GET /v1/videos/files/<id>.webp serves the clip with its 3 frames");
            java.net.http.HttpResponse<String> models = http.send(req.apply("/v1/models").GET().build(), java.net.http.HttpResponse.BodyHandlers.ofString());
            check(models.body().contains("\"kind\":\"video\"") && !models.body().contains("\"kind\":\"text\""), "engine server: /v1/models lists the video model");
            java.net.http.HttpResponse<String> s = http.send(req.apply("/v1/videos/generations").header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"prompt\":\"a cat\",\"size\":\"64x64\",\"num_frames\":2,\"steps\":3,\"guidance\":3,\"seed\":1,\"stream\":true,"
                            + "\"output_format\":\"gif\"}")).build(), java.net.http.HttpResponse.BodyHandlers.ofString());
            check(s.body().contains("video_generation.progress") && s.body().contains("video_generation.partial_image") && s.body().contains("video_generation.decoding")
                            && s.body().contains("video_generation.completed") && s.body().trim().endsWith("[DONE]"),
                    "engine server: streaming progress, previews, decoding, completion, [DONE]");
            java.net.http.HttpResponse<String> bad = http.send(req.apply("/v1/videos/generations").header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"prompt\":\"x\",\"size\":\"50x50\"}")).build(), java.net.http.HttpResponse.BodyHandlers.ofString());
            check(bad.statusCode() == 400 && bad.body().contains("multiple of"), "engine server: a bad size is a 400 with the reason");
            java.net.http.HttpResponse<String> mp4 = http.send(req.apply("/v1/videos/generations").header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"prompt\":\"x\",\"size\":\"64x64\",\"output_format\":\"mp4\"}")).build(), java.net.http.HttpResponse.BodyHandlers.ofString());
            check(mp4.statusCode() == 400 && mp4.body().contains("ffmpeg"), "engine server: mp4 without ffmpeg is a 400 that says so");
            String frame = java.util.Base64.getEncoder().encodeToString(Imaging.png(sampleFrames(32, 24, 1, false).get(0)));
            java.net.http.HttpResponse<String> enc = http.send(req.apply("/v1/videos/encode").header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"frames\":[\"" + frame + "\",\"" + frame + "\"],\"fps\":4,\"output_format\":\"gif\"}")).build(),
                    java.net.http.HttpResponse.BodyHandlers.ofString());
            check(enc.statusCode() == 200 && enc.body().contains("\"frames\":2") && enc.body().contains("\"format\":\"gif\""), "engine server: POST /v1/videos/encode turns pictures into a GIF");
            java.net.http.HttpResponse<String> chat = http.send(req.apply("/v1/chat/completions").header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"messages\":[]}")).build(), java.net.http.HttpResponse.BodyHandlers.ofString());
            check(chat.statusCode() == 404 && chat.body().contains("no text model"), "engine server: video only, the chat endpoint says there is no text model");
        }
    }

    static Map<String, Object> obj(Object... kv) {
        return Json.obj(kv);
    }
}
