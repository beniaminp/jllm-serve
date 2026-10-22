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
        repoAndHttp();
        engineServer();
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
            opsParity("f32", 3e-5);
            opsParity("f16", 1.5e-2);
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

    /** The engine's own HTTP server with an image model only (`--model none`): generation, streaming, edits (JSON and multipart), files. */
    private static void engineServer() throws Exception {
        try (Devices dev = Devices.cpu();
             ImagePipeline pipe = new ImagePipeline(resource("tiny-sd"), dev);
             org.robomiri.jllm.server.ApiServer server = new org.robomiri.jllm.server.ApiServer(null, "none", "127.0.0.1", 0,
                     org.robomiri.jllm.server.ApiServer.Options.defaults())
                     .withImages(new org.robomiri.jllm.server.ImageHttp(pipe, "tiny-sd"))) {
            Thread t = Thread.ofVirtual().start(server::serve);
            String base = "http://127.0.0.1:" + server.port();
            java.net.http.HttpClient http = java.net.http.HttpClient.newHttpClient();
            java.util.function.Function<String, java.net.http.HttpRequest.Builder> req = path -> java.net.http.HttpRequest.newBuilder(java.net.URI.create(base + path));
            java.net.http.HttpResponse<String> r = http.send(req.apply("/v1/images/generations").header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"prompt\":\"a cat\",\"size\":\"64x64\",\"steps\":2,\"guidance\":3,\"seed\":7,\"n\":2,\"response_format\":\"url\"}")).build(),
                    java.net.http.HttpResponse.BodyHandlers.ofString());
            Map<String, Object> out = org.robomiri.jllm.json.Json.parseObject(r.body());
            List<Object> data = org.robomiri.jllm.json.Json.list(out, "data");
            String url = data == null ? "" : org.robomiri.jllm.json.Json.str(data.get(0), "url", "");
            check(r.statusCode() == 200 && data.size() == 2 && url.contains("/v1/images/files/") && org.robomiri.jllm.json.Json.num(data.get(1), "seed", 0) == 8,
                    "engine server: POST /v1/images/generations with n=2 and response_format url");
            java.net.http.HttpResponse<byte[]> f = http.send(req.apply(url.substring(url.indexOf("/v1/"))).GET().build(), java.net.http.HttpResponse.BodyHandlers.ofByteArray());
            check(f.statusCode() == 200 && Imaging.decode(f.body()).width() == 64, "engine server: GET /v1/images/files/<id>.png serves the picture");
            java.net.http.HttpResponse<String> models = http.send(req.apply("/v1/models").GET().build(), java.net.http.HttpResponse.BodyHandlers.ofString());
            check(models.body().contains("\"kind\":\"image\"") && !models.body().contains("\"kind\":\"text\""), "engine server: /v1/models lists the image model");
            // streaming
            java.net.http.HttpResponse<String> s = http.send(req.apply("/v1/images/generations").header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"prompt\":\"a cat\",\"size\":\"64x64\",\"steps\":3,\"guidance\":3,\"seed\":1,\"stream\":true}")).build(),
                    java.net.http.HttpResponse.BodyHandlers.ofString());
            check(s.body().contains("image_generation.progress") && s.body().contains("image_generation.partial_image")
                            && s.body().contains("image_generation.completed") && s.body().trim().endsWith("[DONE]"),
                    "engine server: streaming progress, previews, completion, [DONE]");
            // edits: multipart (image + mask) and JSON
            byte[] png = Imaging.png(new Imaging.Image(64, 64, new byte[64 * 64 * 3]));
            byte[] maskRgb = new byte[64 * 64 * 3];
            for (int i = 0; i < 64 * 64 / 2; i++) {
                maskRgb[i * 3] = maskRgb[i * 3 + 1] = maskRgb[i * 3 + 2] = (byte) 255;
            }
            byte[] mask = Imaging.png(new Imaging.Image(64, 64, maskRgb));
            String boundary = "xBOUNDARYx";
            java.io.ByteArrayOutputStream body = new java.io.ByteArrayOutputStream();
            java.util.function.BiConsumer<String, String> text = (k, v) -> body.writeBytes(("--" + boundary + "\r\nContent-Disposition: form-data; name=\"" + k + "\"\r\n\r\n" + v + "\r\n").getBytes());
            java.util.function.BiConsumer<String, byte[]> file = (k, v) -> {
                body.writeBytes(("--" + boundary + "\r\nContent-Disposition: form-data; name=\"" + k + "\"; filename=\"" + k + ".png\"\r\nContent-Type: image/png\r\n\r\n").getBytes());
                body.writeBytes(v);
                body.writeBytes("\r\n".getBytes());
            };
            text.accept("prompt", "a dog");
            text.accept("steps", "3");
            text.accept("guidance", "3");
            text.accept("seed", "5");
            file.accept("image", png);
            file.accept("mask", mask);
            body.writeBytes(("--" + boundary + "--\r\n").getBytes());
            java.net.http.HttpResponse<String> e = http.send(req.apply("/v1/images/edits").header("Content-Type", "multipart/form-data; boundary=" + boundary)
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofByteArray(body.toByteArray())).build(), java.net.http.HttpResponse.BodyHandlers.ofString());
            check(e.statusCode() == 200 && e.body().contains("b64_json"), "engine server: POST /v1/images/edits as multipart (inpainting)");
            java.net.http.HttpResponse<String> bad = http.send(req.apply("/v1/images/generations").header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"prompt\":\"x\",\"size\":\"50x50\"}")).build(), java.net.http.HttpResponse.BodyHandlers.ofString());
            check(bad.statusCode() == 400 && bad.body().contains("multiple of"), "engine server: a bad size is a 400 with the reason");
            java.net.http.HttpResponse<String> chat = http.send(req.apply("/v1/chat/completions").header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"messages\":[]}")).build(), java.net.http.HttpResponse.BodyHandlers.ofString());
            check(chat.statusCode() == 404 && chat.body().contains("no text model"), "engine server: images only, the chat endpoint says there is no text model");
        }
    }

    /** Which files of a diffusers repository are downloaded, the multipart reader and the picture store. */
    private static void repoAndHttp() throws Exception {
        // a listing like stabilityai/sdxl-turbo's, plus what must be left out
        String[] names = {"model_index.json", "sd_xl_turbo_1.0_fp16.safetensors", "scheduler/scheduler_config.json",
                "text_encoder/config.json", "text_encoder/model.safetensors", "text_encoder/model.fp16.safetensors", "text_encoder/model.onnx",
                "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors", "tokenizer/vocab.json", "tokenizer/merges.txt",
                "tokenizer_2/vocab.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
                "unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.fp16.bin", "unet/openvino/model.xml",
                "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors", "safety_checker/model.safetensors",
                "safety_checker/config.json", "vae_decoder/config.json"};
        List<org.robomiri.jllm.hub.HuggingFaceHub.RepoFile> listing = new ArrayList<>();
        for (String n : names) {
            listing.add(new org.robomiri.jllm.hub.HuggingFaceHub.RepoFile(n, 100, null));
        }
        Map<String, Object> index = org.robomiri.jllm.json.Json.parseObject("{\"_class_name\":\"StableDiffusionXLPipeline\","
                + "\"scheduler\":[\"diffusers\",\"EulerDiscreteScheduler\"],\"text_encoder\":[\"transformers\",\"CLIPTextModel\"],"
                + "\"text_encoder_2\":[\"transformers\",\"CLIPTextModelWithProjection\"],\"tokenizer\":[\"transformers\",\"CLIPTokenizer\"],"
                + "\"tokenizer_2\":[\"transformers\",\"CLIPTokenizer\"],\"unet\":[\"diffusers\",\"UNet2DConditionModel\"],"
                + "\"vae\":[\"diffusers\",\"AutoencoderKL\"],\"safety_checker\":[null,null],\"force_zeros_for_empty_prompt\":true}");
        java.util.Set<String> chosen = DiffusersRepo.select(listing, index).keySet();
        check(chosen.contains("unet/diffusion_pytorch_model.fp16.safetensors") && !chosen.contains("unet/diffusion_pytorch_model.safetensors")
                        && chosen.contains("text_encoder/model.fp16.safetensors") && !chosen.contains("text_encoder/model.safetensors")
                        && chosen.contains("text_encoder_2/model.fp16.safetensors") && chosen.contains("vae/diffusion_pytorch_model.fp16.safetensors"),
                "diffusers download: the fp16 weights of unet, vae and both text encoders, not the full-precision ones");
        check(chosen.contains("model_index.json") && chosen.contains("tokenizer_2/vocab.json") && chosen.contains("scheduler/scheduler_config.json")
                        && !chosen.contains("sd_xl_turbo_1.0_fp16.safetensors") && !chosen.contains("unet/openvino/model.xml")
                        && !chosen.contains("unet/diffusion_pytorch_model.fp16.bin") && !chosen.contains("text_encoder/model.onnx")
                        && !chosen.contains("safety_checker/model.safetensors") && !chosen.contains("vae_decoder/config.json"),
                "diffusers download: configs and tokenizers yes; the single-file checkpoint, .bin / .onnx / openvino files and the safety checker no");
        check(DiffusersRepo.isDiffusers(listing) && !DiffusersRepo.isDiffusers(List.of()), "a repository with a model_index.json is a diffusers repository");

        String boundary = "----jllmBoundary";
        byte[] png = {(byte) 0x89, 'P', 'N', 'G', 13, 10, 26, 10, 0, 1, 2, '\r', '\n', 3};
        java.io.ByteArrayOutputStream body = new java.io.ByteArrayOutputStream();
        String head = "--" + boundary + "\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\na cat\r\n"
                + "--" + boundary + "\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n";
        body.write(head.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        body.write(png);
        body.write(("\r\n--" + boundary + "\r\nContent-Disposition: form-data; name=\"n\"\r\n\r\n2\r\n--" + boundary + "--\r\n")
                .getBytes(java.nio.charset.StandardCharsets.UTF_8));
        Map<String, Multipart.Part> parts = Multipart.parse(body.toByteArray(), Multipart.boundary("multipart/form-data; boundary=\"" + boundary + "\""));
        check(parts.size() == 3 && parts.get("prompt").text().equals("a cat") && parts.get("n").text().equals("2")
                        && parts.get("image").filename().equals("a.png") && java.util.Arrays.equals(parts.get("image").data(), png),
                "multipart/form-data: text fields and a binary file part (with CR LF inside it) are read back exactly");
        ImageStore store = new ImageStore(2, 1000);
        String a = store.put(new byte[400]);
        String b = store.put(new byte[400]);
        String c = store.put(new byte[400]);
        check(store.get(a) == null && store.get(b) != null && store.get(c) != null && a.length() == 24, "picture store: oldest evicted beyond its limits, 24-hex ids");
    }

    static org.robomiri.jllm.safetensors.SafeTensors.Tensor tensor(float[] data, long... shape) {
        java.lang.foreign.MemorySegment seg = java.lang.foreign.Arena.ofAuto().allocate(data.length * 4L);
        java.lang.foreign.MemorySegment.copy(data, 0, seg, java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, 0, data.length);
        return new org.robomiri.jllm.safetensors.SafeTensors.Tensor("t", org.robomiri.jllm.safetensors.SafeTensors.DType.F32, shape, seg);
    }

    private static float[] noise(long seed, int n, float scale) {
        float[] f = new Rng(seed).normal(n);
        for (int i = 0; i < n; i++) {
            f[i] *= scale;
        }
        return f;
    }

    /** Every {@link Ops} operation on the GPU against the CPU reference, at sizes that exercise the big-tensor kernel variants. */
    private static void opsParity(String precision, double relTol) {
        try (Devices cpu = Devices.cpu(); Devices gpu = Devices.open("metal", precision, false)) {
            Ops c = cpu.main();
            Ops g = gpu.main();
            java.util.function.BiConsumer<String, float[][]> cmp = (what, pair) -> {
                double[] d = DiffusionCompare.diff(pair[0], pair[1]);
                check(d[0] <= relTol * Math.max(1.0, d[2]), String.format("%s %s vs CPU: max diff %.2e (of %.2f)", what, precision, d[0], d[2]));
            };
            // convolutions: plain, strided, asymmetric padding, with channel bias and residual, a first layer with 4 channels
            for (int[] k : new int[][]{{1, 20, 18, 32, 40, 3, 1, 1, 1}, {2, 16, 16, 24, 70, 3, 2, 1, 1}, {1, 16, 16, 16, 16, 3, 2, 0, 1}, {1, 9, 11, 4, 8, 3, 1, 1, 1}, {1, 12, 12, 16, 3, 3, 1, 1, 1}}) {
                int b = k[0], h = k[1], w = k[2], ci = k[3], co = k[4], ks = k[5], st = k[6], pt = k[7], pb = k[8];
                float[] xin = noise(1, b * h * w * ci, 1f);
                float[] win = noise(2, co * ci * ks * ks, 0.2f);
                float[] bin = noise(3, co, 0.5f);
                int ho = (h + pt + pb - ks) / st + 1;
                int wo = (w + pt + pb - ks) / st + 1;
                float[] chan = noise(4, b * co, 0.5f);
                float[] res = noise(5, b * ho * wo * co, 0.5f);
                float[][] out = new float[2][];
                for (int dev = 0; dev < 2; dev++) {
                    Ops o = dev == 0 ? c : g;
                    Tn x = o.upload(xin, b, h, w, ci);
                    Tn wt = o.weight(tensor(win, co, ci, ks, ks), Ops.Kind.CONV);
                    Tn bt = o.weight(tensor(bin, co), Ops.Kind.VECTOR);
                    Tn ct = o.upload(chan, b, co);
                    Tn rt = o.upload(res, b, ho, wo, co);
                    out[dev] = o.download(o.conv2d(x, wt, bt, st, pt, pt, pb, pb, ct, rt));
                }
                cmp.accept(String.format("conv %dx%dx%d %d->%d k%d s%d", b, h, w, ci, co, ks, st), out);
            }
            // linear with GELU and residual; GEGLU
            {
                float[] xin = noise(6, 37 * 48, 1f);
                float[] win = noise(7, 70 * 48, 0.3f);
                float[] bin = noise(8, 70, 0.3f);
                float[] res = noise(9, 37 * 70, 1f);
                float[][] out = new float[2][];
                float[][] ge = new float[2][];
                for (int dev = 0; dev < 2; dev++) {
                    Ops o = dev == 0 ? c : g;
                    Tn y = o.linear(o.upload(xin, 37, 48), o.weight(tensor(win, 70, 48), Ops.Kind.LINEAR), o.weight(tensor(bin, 70), Ops.Kind.VECTOR),
                            o.upload(res, 37, 70), Ops.Act.GELU);
                    out[dev] = o.download(y);
                    ge[dev] = o.download(o.geglu(y));
                }
                cmp.accept("linear gelu+residual", out);
                cmp.accept("geglu", ge);
            }
            // normalisation: group norm small (one threadgroup) and large (three kernels), layer norm
            for (int[] k : new int[][]{{2, 6, 6, 32, 8}, {2, 48, 40, 64, 16}, {1, 128, 100, 32, 32}}) {
                int b = k[0], h = k[1], w = k[2], ch = k[3], gr = k[4];
                float[] xin = noise(10, b * h * w * ch, 1.5f);
                float[] gam = noise(11, ch, 0.5f);
                float[] bet = noise(12, ch, 0.5f);
                float[][] out = new float[2][];
                for (int dev = 0; dev < 2; dev++) {
                    Ops o = dev == 0 ? c : g;
                    out[dev] = o.download(o.groupNorm(o.upload(xin, b, h, w, ch), o.weight(tensor(gam, ch), Ops.Kind.VECTOR),
                            o.weight(tensor(bet, ch), Ops.Kind.VECTOR), gr, 1e-5f, true));
                }
                cmp.accept(String.format("groupnorm %dx%dx%d c=%d g=%d (%s)", b, h, w, ch, gr, (long) h * w * ch <= 65536 ? "one threadgroup" : "chunked"), out);
            }
            {
                float[] xin = noise(13, 21 * 77, 2f);
                float[] gam = noise(14, 77, 0.5f);
                float[] bet = noise(15, 77, 0.5f);
                float[][] out = new float[2][];
                for (int dev = 0; dev < 2; dev++) {
                    Ops o = dev == 0 ? c : g;
                    out[dev] = o.download(o.layerNorm(o.upload(xin, 21, 77), o.weight(tensor(gam, 77), Ops.Kind.VECTOR), o.weight(tensor(bet, 77), Ops.Kind.VECTOR), 1e-5f));
                }
                cmp.accept("layernorm", out);
            }
            // attention: self (flash widths), cross with 77 keys, causal, and a head width without a flash kernel
            for (int[] k : new int[][]{{2, 5, 64, 100, 100, 0}, {1, 8, 40, 64, 77, 0}, {2, 4, 32, 77, 77, 1}, {1, 2, 100, 30, 50, 0}}) {
                int b = k[0], hd = k[2], hs = k[1], lq = k[3], lk = k[4];
                int dm = hs * hd;
                float[] qin = noise(16, b * lq * dm, 1f);
                float[] kin = noise(17, b * lk * dm, 1f);
                float[] vin = noise(18, b * lk * dm, 1f);
                float[][] out = new float[2][];
                for (int dev = 0; dev < 2; dev++) {
                    Ops o = dev == 0 ? c : g;
                    out[dev] = o.download(o.attention(o.upload(qin, b, lq, dm), o.upload(kin, b, lk, dm), o.upload(vin, b, lk, dm), hs, k[5] == 1));
                }
                cmp.accept(String.format("attention b=%d h=%d hd=%d lq=%d lk=%d causal=%d", b, hs, hd, lq, lk, k[5]), out);
            }
            // layout: fused upsample + convolution, concat, add, silu
            {
                float[] xin = noise(19, 2 * 7 * 9 * 16, 1f);
                float[] win = noise(20, 24 * 16 * 9, 0.2f);
                float[] bin = noise(21, 24, 0.5f);
                float[] other = noise(22, 2 * 7 * 9 * 8, 1f);
                float[][] up = new float[2][];
                float[][] cat = new float[2][];
                float[][] mix = new float[2][];
                for (int dev = 0; dev < 2; dev++) {
                    Ops o = dev == 0 ? c : g;
                    Tn x = o.upload(xin, 2, 7, 9, 16);
                    up[dev] = o.download(o.upsampleConv(x, o.weight(tensor(win, 24, 16, 3, 3), Ops.Kind.CONV), o.weight(tensor(bin, 24), Ops.Kind.VECTOR)));
                    Tn ct = o.concat(x, o.upload(other, 2, 7, 9, 8));
                    cat[dev] = o.download(ct);
                    mix[dev] = o.download(o.silu(o.add(ct, ct)));
                }
                cmp.accept("upsample + conv", up);
                cmp.accept("concat", cat);
                cmp.accept("add + silu", mix);
            }
        }
    }
}
