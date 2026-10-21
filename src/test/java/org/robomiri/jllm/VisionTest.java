package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.MediaPrompt;
import org.robomiri.jllm.gguf.GgmlType;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.Rope;
import org.robomiri.jllm.model.RopePositions;
import org.robomiri.jllm.model.RopeTable;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.GgufTokenizer;
import org.robomiri.jllm.tokenizer.Tokenizer;
import org.robomiri.jllm.vision.ClipCpu;
import org.robomiri.jllm.vision.ClipMetal;
import org.robomiri.jllm.vision.ClipModel;
import org.robomiri.jllm.vision.ImageDecoder;
import org.robomiri.jllm.vision.ImageFetcher;
import org.robomiri.jllm.vision.ImageParts;
import org.robomiri.jllm.vision.MediaChat;
import org.robomiri.jllm.vision.Mmproj;
import org.robomiri.jllm.vision.Preprocess;
import org.robomiri.jllm.vision.Resize;
import org.robomiri.jllm.vision.Rgb;
import org.robomiri.jllm.vision.VisionContext;
import org.robomiri.jllm.vision.VisionEncoder;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Images as input (docs/20): the resize and size rules against Pillow / llama.cpp, picture decoding, the picture parts
 * of a chat request, projector discovery, the chat template's picture marks, the rotary positions of Qwen-VL, and the
 * whole path on a tiny random vision-language model (a Qwen2-VL text model and a Qwen2.5-VL style projector written as
 * GGUF files): prompt layout, deterministic generation, KV cache reuse keyed by the picture, M-RoPE rows restored for
 * text, embeddings as input equal to token ids, and the Metal backends (encoder and text) against the CPU ones when
 * Metal is available.
 */
public final class VisionTest {

    static int passed;
    static int failed;

    static void check(boolean ok, String what) {
        if (ok) {
            passed++;
        } else {
            failed++;
            System.out.println("FAIL: " + what);
        }
    }

    public static void main(String[] args) throws Exception {
        Path dir = Files.createTempDirectory("vision-test");
        try {
            resize();
            sizeRules();
            projectorChoice();
            decoding();
            imageParts();
            ropePositions();
            tinyModel(dir);
            qwen3Vl(dir);
        } finally {
            try (var s = Files.list(dir)) {
                for (Path p : s.toList()) {
                    Files.deleteIfExists(p);
                }
            }
            Files.deleteIfExists(dir);
        }
        System.out.println("vision: " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    // ------------------------------------------------------------------ resize (expected values: Pillow's Image.resize)

    private static Rgb synthetic(int w, int h) {
        byte[] d = new byte[3 * w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int c = 0; c < 3; c++) {
                    d[3 * (y * w + x) + c] = (byte) ((x * 37 + y * 91 + c * 53 + x * y) % 256);
                }
            }
        }
        return new Rgb(w, h, d);
    }

    private static void resize() {
        Rgb src = synthetic(13, 9);
        Object[][] cases = {
                {Resize.Algo.BILINEAR, 5, 4, "6271f4d3805ee270f41810f9e9dcd1f0f3c67da273b3244f5a81771d19f7bcb0"},
                {Resize.Algo.BICUBIC, 5, 4, "1750db10f5ece284c377f7bdb29fc2762d214aa3ffbd48aee839d68134fcd6b2"},
                {Resize.Algo.LANCZOS, 5, 4, "6ec3a1b0485f8a6d01e889474d58d44ec034c182ef7a38f36f906deb9e4a3212"},
                {Resize.Algo.BICUBIC, 26, 18, "e5b19d4311520a54362c18d688d9621d35eeb6329aa92d5044d516ccae7a51d7"},
                {Resize.Algo.LANCZOS, 13, 4, "4c4f33c9054da2746028b63f416b2ad07fc5e12a87aa61ccf46fe13aa53cfdcc"},
                {Resize.Algo.BILINEAR, 20, 12, "e02024be6a70acc70d82c2e078968f937227017a01ed9c53179d6b8c34ef1af7"},
        };
        // (a mixed case, one axis growing and the other shrinking, is not here: recent Pillow versions order the two passes
        // differently, which changes single pixels by one level; llama.cpp and this port run the horizontal pass first)
        for (Object[] c : cases) {
            Rgb r = Resize.resize(src, new Resize.Size((Integer) c[1], (Integer) c[2]), (Resize.Algo) c[0], Resize.Pad.NONE, null);
            check(r.hash().equals(c[3]), "resize " + c[0] + " 13x9 -> " + c[1] + "x" + c[2] + " equals Pillow's bytes");
        }
        // padding keeps the aspect ratio and centres on the pad colour
        Rgb padded = Resize.resize(synthetic(8, 4), new Resize.Size(8, 8), Resize.Algo.BILINEAR, Resize.Pad.CEIL, new int[]{9, 8, 7});
        check(padded.width() == 8 && padded.height() == 8 && padded.r(0, 0) == 9 && padded.g(7, 7) == 8 && padded.b(3, 7) == 7
                && padded.r(0, 2) != 9, "CEIL padding: 8x4 in 8x8 is centred on the pad colour");
        check(Resize.crop(src, 2, 1, 5, 4).width() == 5 && Resize.crop(src, 2, 1, 5, 4).r(0, 0) == src.r(2, 1), "crop");
    }

    private static void sizeRules() {
        // llama.cpp's calc_size_preserved_ratio (Qwen-VL: align 28, 8..4096 merged tokens of 28 x 28)
        Resize.SizeRule qwen = new Resize.SizeRule(28, 6272, 3211264, 0);
        check(Resize.calcSizePreservedRatio(new Resize.Size(640, 480), qwen).equals(new Resize.Size(644, 476)), "size rule 640x480 -> 644x476");
        check(Resize.calcSizePreservedRatio(new Resize.Size(100, 50), qwen).equals(new Resize.Size(112, 56)), "size rule 100x50 -> 112x56");
        check(Resize.calcSizePreservedRatio(new Resize.Size(3000, 2000), qwen).equals(new Resize.Size(2184, 1456)), "size rule 3000x2000 -> 2184x1456 (over the budget)");
        check(Resize.calcSizePreservedRatio(new Resize.Size(10, 10), qwen).equals(new Resize.Size(84, 84)), "size rule 10x10 -> 84x84 (under the minimum)");
        // SmolVLM without a preprocessor size: whole 512 tiles
        Resize.SizeRule tiles = new Resize.SizeRule(512, 0, 0, 0);
        check(Resize.calcSizePreservedRatio(new Resize.Size(480, 200), tiles).equals(new Resize.Size(512, 512)), "tile rule 480x200 -> 512x512");
        check(Resize.calcSizePreservedRatio(new Resize.Size(1500, 900), tiles).equals(new Resize.Size(1536, 1024)), "tile rule 1500x900 -> 1536x1024");
    }

    // ------------------------------------------------------------------ projector choice

    private static void projectorChoice() {
        check(Mmproj.isMmproj("mmproj-SmolVLM-256M-Instruct-f16.gguf") && Mmproj.isMmproj("sub/MMPROJ-F16.gguf")
                && !Mmproj.isMmproj("SmolVLM-256M-Instruct-Q8_0.gguf"), "mmproj file names");
        List<String> files = List.of("SmolVLM-256M-Instruct-Q8_0.gguf", "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
                "mmproj-SmolVLM-256M-Instruct-f16.gguf", "mmproj-Other-f16.gguf");
        check("mmproj-SmolVLM-256M-Instruct-f16.gguf".equals(Mmproj.choose(files, "SmolVLM-256M-Instruct-Q8_0.gguf")),
                "the F16 projector with the model's name wins over Q8_0 and over another model's");
        check("mmproj-model-f16.gguf".equals(Mmproj.choose(List.of("gemma-3-4b-it-Q4_K_M.gguf", "mmproj-model-f16.gguf"), "gemma-3-4b-it-Q4_K_M.gguf")),
                "a single projector is taken whatever its name");
        check(Mmproj.choose(List.of("model.gguf"), "model.gguf") == null, "no projector, no choice");
        check("mmproj-X-Q8_0.gguf".equals(Mmproj.choose(List.of("mmproj-X-Q8_0.gguf", "mmproj-X-Q8_0.gguf.incomplete"), "x.gguf")),
                "a partial download is not a candidate");
    }

    // ------------------------------------------------------------------ decoding

    private static byte[] png(BufferedImage img) throws Exception {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        ImageIO.write(img, "png", o);
        return o.toByteArray();
    }

    private static void decoding() throws Exception {
        BufferedImage rgb = new BufferedImage(4, 3, BufferedImage.TYPE_INT_RGB);
        BufferedImage argb = new BufferedImage(4, 3, BufferedImage.TYPE_INT_ARGB);
        BufferedImage gray = new BufferedImage(4, 3, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 4; x++) {
                int c = (x * 60) << 16 | (y * 100) << 8 | 77;
                rgb.setRGB(x, y, c);
                argb.setRGB(x, y, 0x80000000 | c);
                gray.getRaster().setSample(x, y, 0, x * 60 + y);
            }
        }
        Rgb a = ImageDecoder.decode(png(rgb));
        check(a.width() == 4 && a.height() == 3 && a.r(2, 1) == 120 && a.g(2, 1) == 100 && a.b(2, 1) == 77, "RGB PNG decodes to its pixels");
        Rgb b = ImageDecoder.decode(png(argb));
        check(b.r(2, 1) == 120 && b.g(2, 1) == 100 && b.b(2, 1) == 77, "alpha is dropped, colours kept");
        Rgb g = ImageDecoder.decode(png(gray));
        check(g.r(3, 2) == 182 && g.g(3, 2) == 182 && g.b(3, 2) == 182, "gray values are replicated, not colour-converted");
        try {
            ImageDecoder.decode(new byte[]{1, 2, 3, 4});
            check(false, "garbage is refused");
        } catch (java.io.IOException e) {
            check(true, "garbage is refused");
        }
        try {
            ImageDecoder.decode(png(rgb), 5);
            check(false, "a picture over the pixel limit is refused before decoding");
        } catch (java.io.IOException e) {
            check(e.getMessage().contains("limit"), "a picture over the pixel limit is refused before decoding");
        }
        String url = "data:image/png;base64," + Base64.getEncoder().encodeToString(png(rgb));
        check(Arrays.equals(ImageDecoder.dataUrlBytes(url), png(rgb)), "data: URL");
    }

    // ------------------------------------------------------------------ the picture parts of a request

    @SuppressWarnings("unchecked")
    private static void imageParts() throws Exception {
        BufferedImage img = new BufferedImage(6, 5, BufferedImage.TYPE_INT_RGB);
        String url = "data:image/png;base64," + Base64.getEncoder().encodeToString(png(img));
        Map<String, Object> part1 = new LinkedHashMap<>(Map.of("type", "text", "text", "see "));
        Map<String, Object> part2 = new LinkedHashMap<>(Map.of("type", "image_url", "image_url", Map.of("url", url, "detail", "low")));
        Map<String, Object> part3 = new LinkedHashMap<>(Map.of("type", "text", "text", "and " + MediaChat.MARK + "this"));
        Map<String, Object> msg = new LinkedHashMap<>();
        msg.put("role", "user");
        msg.put("content", List.of(part1, part2, part3));
        List<Object> msgs = new ArrayList<>(List.of(Map.of("role", "system", "content", "be brief"), msg));
        check(ImageParts.hasImages(msgs) && !ImageParts.hasImages(List.of(Map.of("role", "user", "content", "hi"))), "hasImages");
        ImageParts.Extracted ex = ImageParts.extract(msgs, null, 1 << 20, 1 << 20);
        String content = (String) ((Map<String, Object>) ex.messages().get(1)).get("content");
        check(ex.images().size() == 1 && ex.images().getFirst().width() == 6 && content.equals("see " + MediaChat.MARK + "and this"),
                "parts become text with one mark per picture, in order; a mark typed by the client is stripped: '" + content + "'");
        check("be brief".equals(((Map<String, Object>) ex.messages().get(0)).get("content")), "other messages are untouched");
        expectBad(() -> ImageParts.extract(List.of(Map.of("role", "assistant", "content", List.of(part2))), null, 1 << 20, 1 << 20),
                "user messages", "images in an assistant message");
        expectBad(() -> ImageParts.extract(List.of(Map.of("role", "user", "content", List.of(Map.of("type", "image_url", "image_url", Map.of("url", "https://example.com/a.png"))))),
                null, 1 << 20, 1 << 20), "does not fetch", "a URL without a fetcher");
        expectBad(() -> ImageParts.extract(List.of(Map.of("role", "user", "content", List.of(Map.of("type", "input_audio")))), null, 1 << 20, 1 << 20),
                "audio", "an audio part");
        expectBad(() -> ImageParts.extract(List.of(Map.of("role", "user", "content", List.of(Map.of("type", "image_url", "image_url", Map.of("url", "data:image/png;base64,AAAA"))))),
                null, 1 << 20, 1 << 20), "cannot load", "broken picture data");
        expectBad(() -> ImageParts.extract(msgs, null, 10, 1 << 20), "limit", "a picture over the byte limit");
        List<Object> many = new ArrayList<>();
        for (int i = 0; i <= ImageParts.MAX_IMAGES; i++) {
            many.add(part2);
        }
        expectBad(() -> ImageParts.extract(List.of(Map.of("role", "user", "content", many)), null, 1 << 20, 1 << 20), "at most", "too many pictures");
        // fetch rules: private addresses refused unless allowed
        try {
            ImageFetcher.check(URI.create("http://127.0.0.1:1/x.png"), false);
            check(false, "loopback refused");
        } catch (IllegalArgumentException e) {
            check(e.getMessage().contains("private"), "loopback is refused by default");
        }
        ImageFetcher.check(URI.create("http://127.0.0.1:1/x.png"), true);
        try {
            ImageFetcher.check(URI.create("file:///etc/passwd"), true);
            check(false, "file: refused");
        } catch (IllegalArgumentException e) {
            check(true, "only http and https are fetched");
        }
    }

    private static void expectBad(Runnable r, String contains, String what) {
        try {
            r.run();
            check(false, what + " is refused");
        } catch (ImageParts.BadImage e) {
            check(e.getMessage().contains(contains), what + " is refused with a message naming the problem: " + e.getMessage());
        }
    }

    // ------------------------------------------------------------------ rotary positions

    private static void ropePositions() {
        RopeTable plain = new RopeTable(16, 10000.0, org.robomiri.jllm.model.Scaling.NONE, RopeTable.Pairing.HALF);
        Rope base = new Rope(plain, 12);
        float[] c = new float[12 * 8];
        float[] s = new float[12 * 8];
        Rope.rows(plain, 12, 0, null, c, s);
        check(Arrays.equals(c, base.cos) && Arrays.equals(s, base.sin), "Rope.rows without positions equals the table, bit for bit");
        // sections 2 / 3 / 3 over 8 pairs: t = pairs 0-1, h = 2-4, w = 5-7
        RopePositions p = new RopePositions(new int[]{0, 5}, new int[]{0, 6}, new int[]{0, 7}, new int[]{2, 3, 3}, false, 0);
        check(p.axisOf(0) == 0 && p.axisOf(1) == 0 && p.axisOf(2) == 1 && p.axisOf(4) == 1 && p.axisOf(5) == 2 && p.axisOf(7) == 2, "M-RoPE sections");
        check(p.position(1, 0) == 5 && p.position(1, 3) == 6 && p.position(1, 6) == 7 && p.position(0, 6) == 0, "M-RoPE position of a slot and pair");
        RopePositions inter = new RopePositions(new int[]{0}, new int[]{0}, new int[]{0}, new int[]{3, 2, 2}, true, 0);
        check(inter.axisOf(0) == 0 && inter.axisOf(1) == 1 && inter.axisOf(2) == 2 && inter.axisOf(3) == 0 && inter.axisOf(4) == 1
                && inter.axisOf(5) == 2 && inter.axisOf(6) == 0, "interleaved M-RoPE axes");
        RopePositions tail = new RopePositions(new int[]{0, 1}, new int[]{0, 1}, new int[]{0, 1}, new int[]{2, 3, 3}, false, 3);
        check(tail.position(7, 4) == 4, "after the known slots the position is slot - tailShift");
        float[] c2 = new float[6 * 8];
        float[] s2 = new float[6 * 8];
        Rope.rows(plain, 12, 6, new RopePositions(new int[0], new int[0], new int[0], new int[]{2, 3, 3}, false, 2), c2, s2);
        check(Math.abs(c2[0] - Math.cos(4 * 1.0)) < 1e-6, "rows past the known slots are shifted by tailShift (slot 6 has position 4)");
    }

    // ------------------------------------------------------------------ the tiny vision-language model

    static final int DIM = 128, HEADS = 4, KV = 2, HD = 32, FF = 192, LAYERS = 2, VOCAB = 264;
    static final int V_E = 32, V_FF = 64, V_LAYERS = 2, V_HEADS = 2, V_PATCH = 4, V_IMAGE = 16;

    private static void vec(GgufWriter w, Random r, String name, int n, double base, double std) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) (base + std * r.nextGaussian());
        }
        w.f32Tensor(name, new long[]{n}, v);
    }

    /** A matrix [out][in] as ggml dims {in, out}, F16 or F32. */
    private static void mat(GgufWriter w, Random r, String name, int out, int in, boolean f16) {
        float[] v = new float[out * in];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (r.nextGaussian() * 0.15);
        }
        if (!f16) {
            w.f32Tensor(name, new long[]{in, out}, v);
            return;
        }
        ByteBuffer b = ByteBuffer.allocate(2 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) {
            b.putShort(Float.floatToFloat16(x));
        }
        w.tensor(name, GgmlType.F16, new long[]{in, out}, b.array());
    }

    private static void writeText(Path p) throws Exception {
        Random r = new Random(5);
        GgufWriter w = new GgufWriter().str("general.architecture", "qwen2vl").str("general.name", "tiny-vl");
        String a = "qwen2vl.";
        w.u32(a + "block_count", LAYERS).u32(a + "embedding_length", DIM).u32(a + "feed_forward_length", FF)
                .u32(a + "attention.head_count", HEADS).u32(a + "attention.head_count_kv", KV)
                .u32(a + "context_length", 256).f32(a + "rope.freq_base", 10000f)
                .f32(a + "attention.layer_norm_rms_epsilon", 1e-6f).u32(a + "attention.key_length", HD)
                .u32(a + "rope.dimension_count", HD).i32s(a + "rope.dimension_sections", 4, 6, 6, 0)
                .u32("tokenizer.ggml.bos_token_id", 256).u32("tokenizer.ggml.eos_token_id", 257);
        mat(w, r, "token_embd.weight", VOCAB, DIM, false);
        vec(w, r, "output_norm.weight", DIM, 1.0, 0.1);
        for (int l = 0; l < LAYERS; l++) {
            String b = "blk." + l + ".";
            vec(w, r, b + "attn_norm.weight", DIM, 1.0, 0.1);
            mat(w, r, b + "attn_q.weight", HEADS * HD, DIM, false);
            mat(w, r, b + "attn_k.weight", KV * HD, DIM, false);
            mat(w, r, b + "attn_v.weight", KV * HD, DIM, false);
            vec(w, r, b + "attn_q.bias", HEADS * HD, 0, 0.1);
            vec(w, r, b + "attn_k.bias", KV * HD, 0, 0.1);
            vec(w, r, b + "attn_v.bias", KV * HD, 0, 0.1);
            mat(w, r, b + "attn_output.weight", DIM, HEADS * HD, false);
            vec(w, r, b + "ffn_norm.weight", DIM, 1.0, 0.1);
            mat(w, r, b + "ffn_gate.weight", FF, DIM, false);
            mat(w, r, b + "ffn_up.weight", FF, DIM, false);
            mat(w, r, b + "ffn_down.weight", DIM, FF, false);
        }
        w.write(p);
    }

    /** A Qwen2.5-VL style projector: windowed ViT with RMSNorm, SwiGLU, 2-D rotary, 2 x 2 merger; 16 x 16 pixels, 4-pixel patches. */
    private static void writeProjector(Path p) throws Exception {
        Random r = new Random(9);
        GgufWriter w = new GgufWriter().str("general.architecture", "clip").bool("clip.has_vision_encoder", true)
                .str("clip.projector_type", "qwen2.5vl_merger").u32("clip.vision.embedding_length", V_E)
                .u32("clip.vision.feed_forward_length", V_FF).u32("clip.vision.block_count", V_LAYERS)
                .u32("clip.vision.attention.head_count", V_HEADS).u32("clip.vision.projection_dim", DIM)
                .u32("clip.vision.image_size", V_IMAGE).u32("clip.vision.patch_size", V_PATCH)
                .f32("clip.vision.attention.layer_norm_epsilon", 1e-6f).f32s("clip.vision.image_mean", 0.5f, 0.5f, 0.5f)
                .f32s("clip.vision.image_std", 0.5f, 0.5f, 0.5f).bool("clip.use_silu", true)
                .u32("clip.vision.spatial_merge_size", 2).u32("clip.vision.n_wa_pattern", 2).u32("clip.vision.window_size", 8);
        w.tensor("v.patch_embd.weight", GgmlType.F32, new long[]{V_PATCH, V_PATCH, 3, V_E}, floats(r, V_E * 3 * V_PATCH * V_PATCH, 0.2));
        w.tensor("v.patch_embd.weight.1", GgmlType.F32, new long[]{V_PATCH, V_PATCH, 3, V_E}, floats(r, V_E * 3 * V_PATCH * V_PATCH, 0.2));
        for (int l = 0; l < V_LAYERS; l++) {
            String b = "v.blk." + l + ".";
            vec(w, r, b + "ln1.weight", V_E, 1.0, 0.1);
            vec(w, r, b + "ln2.weight", V_E, 1.0, 0.1);
            for (String n : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
                mat(w, r, b + n + ".weight", V_E, V_E, l == 0);
                vec(w, r, b + n + ".bias", V_E, 0, 0.1);
            }
            mat(w, r, b + "ffn_up.weight", V_FF, V_E, l == 0);
            vec(w, r, b + "ffn_up.bias", V_FF, 0, 0.1);
            mat(w, r, b + "ffn_gate.weight", V_FF, V_E, l == 0);
            vec(w, r, b + "ffn_gate.bias", V_FF, 0, 0.1);
            mat(w, r, b + "ffn_down.weight", V_E, V_FF, l == 0);
            vec(w, r, b + "ffn_down.bias", V_E, 0, 0.1);
        }
        vec(w, r, "v.post_ln.weight", V_E, 1.0, 0.1);
        mat(w, r, "mm.0.weight", 4 * V_E, 4 * V_E, true);
        vec(w, r, "mm.0.bias", 4 * V_E, 0, 0.1);
        mat(w, r, "mm.2.weight", DIM, 4 * V_E, true);
        vec(w, r, "mm.2.bias", DIM, 0, 0.1);
        w.write(p);
    }

    private static byte[] floats(Random r, int n, double std) {
        ByteBuffer b = ByteBuffer.allocate(4 * n).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            b.putFloat((float) (r.nextGaussian() * std));
        }
        return b.array();
    }

    /** A byte-level BPE tokenizer without merges (every byte a token) plus the Qwen control tokens. */
    private static Tokenizer tokenizer() {
        String[] tokens = new String[VOCAB];
        long[] types = new long[VOCAB];
        // GPT-2's byte <-> unicode table
        List<Integer> bs = new ArrayList<>();
        for (int i = 33; i <= 126; i++) bs.add(i);
        for (int i = 161; i <= 172; i++) bs.add(i);
        for (int i = 174; i <= 255; i++) bs.add(i);
        int[] map = new int[256];
        int n = 0;
        for (int b = 0; b < 256; b++) {
            map[b] = bs.contains(b) ? b : 256 + n++;
        }
        for (int b = 0; b < 256; b++) {
            tokens[b] = new String(Character.toChars(map[b]));
            types[b] = 1;
        }
        String[] special = {"<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|endoftext|>"};
        // ids 256 (bos) and 257 (eos) are im_start / im_end; the pad token is the last special
        for (int i = 0; i < special.length; i++) {
            tokens[256 + i] = special[i];
            types[256 + i] = 3;
        }
        for (int i = 256 + special.length; i < VOCAB; i++) {
            tokens[i] = "<unused" + i + ">";
            types[i] = 5;
        }
        return GgufTokenizer.build(new GgufTokenizer.Spec("gpt2", "default", tokens, types, null, new String[0], 256, 257, -1, false, true));
    }

    private static Rgb randomImage(long seed, int w, int h) {
        Random r = new Random(seed);
        byte[] d = new byte[3 * w * h];
        r.nextBytes(d);
        return new Rgb(w, h, d);
    }

    private static float maxDiff(float[] a, float[] b) {
        float m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }


    // ------------------------------------------------------------------ Qwen3-VL: interleaved M-RoPE and deepstack

    static final int Q3_SECTIONS_T = 6, Q3_SECTIONS_H = 5, Q3_SECTIONS_W = 5;

    /** A Qwen3-VL text model: qwen3 (q/k norms) with interleaved rotary sections and one deepstack layer. */
    private static void writeText3(Path p) throws Exception {
        Random r = new Random(15);
        GgufWriter w = new GgufWriter().str("general.architecture", "qwen3vl").str("general.name", "tiny-q3vl");
        String a = "qwen3vl.";
        w.u32(a + "block_count", LAYERS).u32(a + "embedding_length", DIM).u32(a + "feed_forward_length", FF)
                .u32(a + "attention.head_count", HEADS).u32(a + "attention.head_count_kv", KV)
                .u32(a + "context_length", 256).f32(a + "rope.freq_base", 10000f)
                .f32(a + "attention.layer_norm_rms_epsilon", 1e-6f).u32(a + "attention.key_length", HD)
                .u32(a + "attention.value_length", HD).u32(a + "n_deepstack_layers", 1)
                .i32s(a + "rope.dimension_sections", Q3_SECTIONS_T, Q3_SECTIONS_H, Q3_SECTIONS_W, 0)
                .u32("tokenizer.ggml.bos_token_id", 256).u32("tokenizer.ggml.eos_token_id", 257);
        mat(w, r, "token_embd.weight", VOCAB, DIM, false);
        vec(w, r, "output_norm.weight", DIM, 1.0, 0.1);
        for (int l = 0; l < LAYERS; l++) {
            String b = "blk." + l + ".";
            vec(w, r, b + "attn_norm.weight", DIM, 1.0, 0.1);
            mat(w, r, b + "attn_q.weight", HEADS * HD, DIM, false);
            mat(w, r, b + "attn_k.weight", KV * HD, DIM, false);
            mat(w, r, b + "attn_v.weight", KV * HD, DIM, false);
            vec(w, r, b + "attn_q_norm.weight", HD, 1.0, 0.1);
            vec(w, r, b + "attn_k_norm.weight", HD, 1.0, 0.1);
            mat(w, r, b + "attn_output.weight", DIM, HEADS * HD, false);
            vec(w, r, b + "ffn_norm.weight", DIM, 1.0, 0.1);
            mat(w, r, b + "ffn_gate.weight", FF, DIM, false);
            mat(w, r, b + "ffn_up.weight", FF, DIM, false);
            mat(w, r, b + "ffn_down.weight", DIM, FF, false);
        }
        w.write(p);
    }

    /**
     * A Qwen3-VL style projector: LayerNorm ViT with fused qkv, a learned 3 x 3 position grid resized to the picture,
     * GELU, no windows, 2 x 2 merger, and a deepstack merger after block 0; 16 x 16 pixels, 4-pixel patches.
     */
    private static void writeProjector3(Path p) throws Exception {
        Random r = new Random(19);
        GgufWriter w = new GgufWriter().str("general.architecture", "clip").bool("clip.has_vision_encoder", true)
                .str("clip.projector_type", "qwen3vl_merger").u32("clip.vision.embedding_length", V_E)
                .u32("clip.vision.feed_forward_length", V_FF).u32("clip.vision.block_count", V_LAYERS)
                .u32("clip.vision.attention.head_count", V_HEADS).u32("clip.vision.projection_dim", DIM)
                .u32("clip.vision.image_size", V_IMAGE).u32("clip.vision.patch_size", V_PATCH)
                .f32("clip.vision.attention.layer_norm_epsilon", 1e-6f).f32s("clip.vision.image_mean", 0.5f, 0.5f, 0.5f)
                .f32s("clip.vision.image_std", 0.5f, 0.5f, 0.5f).bool("clip.use_gelu", true)
                .u32("clip.vision.spatial_merge_size", 2);
        w.tensor("v.patch_embd.weight", GgmlType.F32, new long[]{V_PATCH, V_PATCH, 3, V_E}, floats(r, V_E * 3 * V_PATCH * V_PATCH, 0.2));
        w.tensor("v.patch_embd.weight.1", GgmlType.F32, new long[]{V_PATCH, V_PATCH, 3, V_E}, floats(r, V_E * 3 * V_PATCH * V_PATCH, 0.2));
        vec(w, r, "v.patch_embd.bias", V_E, 0, 0.1);
        mat(w, r, "v.position_embd.weight", 9, V_E, false);
        for (int l = 0; l < V_LAYERS; l++) {
            String b = "v.blk." + l + ".";
            vec(w, r, b + "ln1.weight", V_E, 1.0, 0.1);
            vec(w, r, b + "ln1.bias", V_E, 0, 0.1);
            vec(w, r, b + "ln2.weight", V_E, 1.0, 0.1);
            vec(w, r, b + "ln2.bias", V_E, 0, 0.1);
            mat(w, r, b + "attn_qkv.weight", 3 * V_E, V_E, l == 0);
            vec(w, r, b + "attn_qkv.bias", 3 * V_E, 0, 0.1);
            mat(w, r, b + "attn_out.weight", V_E, V_E, l == 0);
            vec(w, r, b + "attn_out.bias", V_E, 0, 0.1);
            mat(w, r, b + "ffn_up.weight", V_FF, V_E, l == 0);
            vec(w, r, b + "ffn_up.bias", V_FF, 0, 0.1);
            mat(w, r, b + "ffn_down.weight", V_E, V_FF, l == 0);
            vec(w, r, b + "ffn_down.bias", V_E, 0, 0.1);
        }
        String ds = "v.deepstack.0.";
        vec(w, r, ds + "norm.weight", 4 * V_E, 1.0, 0.1);
        vec(w, r, ds + "norm.bias", 4 * V_E, 0, 0.1);
        mat(w, r, ds + "fc1.weight", 4 * V_E, 4 * V_E, true);
        vec(w, r, ds + "fc1.bias", 4 * V_E, 0, 0.1);
        mat(w, r, ds + "fc2.weight", DIM, 4 * V_E, true);
        vec(w, r, ds + "fc2.bias", DIM, 0, 0.1);
        vec(w, r, "v.post_ln.weight", V_E, 1.0, 0.1);
        vec(w, r, "v.post_ln.bias", V_E, 0, 0.1);
        mat(w, r, "mm.0.weight", 4 * V_E, 4 * V_E, true);
        vec(w, r, "mm.0.bias", 4 * V_E, 0, 0.1);
        mat(w, r, "mm.2.weight", DIM, 4 * V_E, true);
        vec(w, r, "mm.2.bias", DIM, 0, 0.1);
        w.write(p);
    }

    private static void qwen3Vl(Path dir) throws Exception {
        Path text = dir.resolve("tiny-q3vl.gguf");
        Path proj = dir.resolve("mmproj-tiny-q3vl.gguf");
        writeText3(text);
        writeProjector3(proj);
        Tokenizer tok = tokenizer();
        ChatTemplate.Format fmt = ChatTemplate.forModelType("qwen3vl", tok);
        check(fmt.render(List.of(new ChatTemplate.Message("user", "hi")), true, false)
                        .equals("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"),
                "Qwen3-VL's chat format has no default system prompt and no thinking block");

        Rgb img = randomImage(3, 20, 12);
        Preprocess.Pixels px;
        VisionEncoder.Encoded cpu;
        float scale = 0;
        try (ClipModel m = ClipModel.open(proj); ClipCpu enc = new ClipCpu(m, 2)) {
            check(m.projector == ClipModel.Projector.QWEN3VL && m.layers.get(0).hasDeepstack() && !m.layers.get(1).hasDeepstack(),
                    "Qwen3-VL projector: deepstack merger found on block 0 only");
            px = Preprocess.preprocess(m, img, new Preprocess.Limits(0, 0)).tiles().getFirst();
            cpu = enc.encode(px);
            check(cpu.deepstack() != null && cpu.deepstack().length == 1 && cpu.deepstack()[0].length == cpu.embd().length
                    && cpu.embd().length == cpu.tokens() * DIM, "Qwen3-VL encoder: " + cpu.tokens() + " tokens, one deepstack feature per token");
            for (float v : cpu.deepstack()[0]) {
                scale = Math.max(scale, Math.abs(v));
            }
            check(scale > 0.01f && Float.isFinite(scale), "the deepstack features are not degenerate (max |x| = " + scale + ")");
        }
        boolean metal = Metal.available();
        if (metal) {
            try (ClipModel m = ClipModel.open(proj); ClipMetal enc = new ClipMetal(m, null)) {
                VisionEncoder.Encoded gpu = enc.encode(px);
                float d = maxDiff(gpu.embd(), cpu.embd());
                float dd = maxDiff(gpu.deepstack()[0], cpu.deepstack()[0]);
                check(d < 0.02f * Math.max(scale, 1f) && dd < 0.02f * Math.max(scale, 1f),
                        "the Metal Qwen3-VL encoder agrees with the CPU one (max diff " + d + ", deepstack " + dd + ")");
            }
        }

        try (GgufModel gm = GgufModel.open(text)) {
            Decoder arch = (Decoder) gm.architecture();
            RopeTable rt = arch.ropeTables().getFirst();
            check(rt.mrope().equals(List.of(Q3_SECTIONS_T, Q3_SECTIONS_H, Q3_SECTIONS_W)) && rt.mropeInterleaved(),
                    "qwen3vl GGUF: interleaved rotary sections " + rt.mrope());
            ModelWeights weights = gm.weights();
            try (VisionContext vc = VisionContext.open(proj, tok, arch, new VisionContext.Options("cpu", 2, new Preprocess.Limits(0, 0), 16));
                 Engine engine = new Engine(new CpuBackend(weights, 256, 16, 2, KvDtype.F32, false), tok, arch, Set.of(257), 0, fmt)) {
                engine.setVision(vc, false);
                List<ChatTemplate.Message> msgs = List.of(new ChatTemplate.Message("user", MediaChat.MARK + "ab"));
                MediaPrompt mp = MediaChat.render(engine, msgs, false, List.of(img));
                MediaPrompt.Span span = mp.spans().getFirst();
                check(span.deepstack() != null && span.deepstack().length == 1 && span.deepstack()[0].length == span.embeds().length,
                        "the picture span carries its deepstack features");
                RopePositions rp = mp.rope();
                boolean ext = rp.interleaved() && rp.e() != null;
                for (int j = 0; ext && j < span.length(); j++) {
                    ext &= rp.e()[span.start() + j] == 0;
                }
                for (int i = 0; ext && i < span.start(); i++) {
                    ext &= rp.e()[i] == i;
                }
                check(ext, "interleaved positions: the extra axis is 0 inside the picture and the position in the text");

                // deepstack is added to the hidden state: the logits differ from the plain embeddings' ones
                Backend be = engine.backend();
                int n = Math.min(span.length(), 8);
                float[] rows = Arrays.copyOf(span.embeds(), n * DIM);
                float[][] ds = {Arrays.copyOf(span.deepstack()[0], n * DIM)};
                float[] plain = new float[VOCAB];
                float[] with = new float[VOCAB];
                be.forwardEmbeds(rows, 0, n, 0, plain);
                be.forwardEmbeds(rows, 0, n, 0, ds, with);
                check(!Arrays.equals(plain, with), "deepstack features change the logits (CPU)");

                SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1L, 8, List.of());
                Engine.Result r1 = engine.generate(mp, greedy, null, () -> false);
                Engine.Result r1b = engine.generate(mp, greedy, null, () -> false);
                check(r1.tokens().length == 8 && Arrays.equals(r1.tokens(), r1b.tokens())
                        && r1b.stats().cachedTokens() == mp.length() - 1, "Qwen3-VL: deterministic answer, the picture is cached by its hash");
                if (metal) {
                    try (Engine me = new Engine(new MetalBackend(weights, 256, 16, (String) null), tok, arch, Set.of(257), 0, fmt)) {
                        me.setVision(vc, false);
                        float[] mwith = new float[VOCAB];
                        me.backend().forwardEmbeds(rows, 0, n, 0, ds, mwith);
                        float top = 0;
                        for (float v : with) {
                            top = Math.max(top, Math.abs(v));
                        }
                        check(maxDiff(with, mwith) < 0.01f * Math.max(top, 1f), "Metal: forwardEmbeds with deepstack agrees with the CPU (max diff "
                                + maxDiff(with, mwith) + " of " + top + ")");
                        float[] mplain = new float[VOCAB];
                        me.backend().forwardEmbeds(rows, 0, n, 0, mplain);
                        check(!Arrays.equals(mplain, mwith), "deepstack features change the logits (Metal)");
                        Engine.Result g1 = me.generate(mp, greedy, null, () -> false);
                        check(Arrays.equals(g1.tokens(), r1.tokens()), "Metal text backend: the same Qwen3-VL answer as the CPU backend");
                    }
                }
            }
            weights.close();
        }
    }

    private static void tinyModel(Path dir) throws Exception {
        Path text = dir.resolve("tiny-vl.gguf");
        Path proj = dir.resolve("mmproj-tiny-vl.gguf");
        writeText(text);
        writeProjector(proj);
        Tokenizer tok = tokenizer();
        check(tok.vocabSize() >= 262 && tok.tokenId("<|image_pad|>") == 260, "the test tokenizer has its control tokens");

        // --- the encoder: CPU, and the GPU encoder against it
        Rgb img = randomImage(1, 20, 12);
        Preprocess.Pixels px;
        VisionEncoder.Encoded cpu;
        try (ClipModel m = ClipModel.open(proj); ClipCpu enc = new ClipCpu(m, 2)) {
            check(m.projector == ClipModel.Projector.QWEN25VL && m.nLayer == V_LAYERS && m.nMerge == 2 && m.nWaPattern == 2, "projector hyper-parameters");
            Preprocess.Prepared pp = Preprocess.preprocess(m, img, new Preprocess.Limits(0, 0));
            px = pp.tiles().getFirst();
            check(px.width() % 8 == 0 && px.height() % 8 == 0, "the preprocessed image is a multiple of patch x merge: " + px.width() + "x" + px.height());
            cpu = enc.encode(px);
            check(cpu.tokens() == cpu.nx() * cpu.ny() && cpu.embd().length == cpu.tokens() * DIM, "encoder output shape " + cpu.nx() + "x" + cpu.ny());
            VisionEncoder.Encoded again = enc.encode(px);
            // (not bit for bit: the Vector API reduces lanes in another order in cold, interpreted code than in compiled code)
            check(maxDiff(cpu.embd(), again.embd()) < 1e-4f, "the CPU encoder repeats itself to float rounding (max diff " + maxDiff(cpu.embd(), again.embd()) + ")");
            float norm = 0;
            for (float v : cpu.embd()) {
                norm = Math.max(norm, Math.abs(v));
            }
            check(norm > 0.01f && Float.isFinite(norm), "the embeddings are not degenerate (max |x| = " + norm + ")");
        }
        boolean metal = Metal.available();
        if (metal) {
            try (ClipModel m = ClipModel.open(proj); ClipMetal enc = new ClipMetal(m, null)) {
                VisionEncoder.Encoded gpu = enc.encode(px);
                float scale = 0;
                for (float v : cpu.embd()) {
                    scale = Math.max(scale, Math.abs(v));
                }
                float d = maxDiff(gpu.embd(), cpu.embd());
                check(gpu.tokens() == cpu.tokens() && d < 0.02f * scale, "the Metal encoder agrees with the CPU one (max diff " + d + " of " + scale + ")");
            }
        }

        // --- the prompt
        try (GgufModel gm = GgufModel.open(text)) {
            Decoder arch = (Decoder) gm.architecture();
            check(!arch.ropeTables().getFirst().mrope().isEmpty() && arch.ropeTables().getFirst().mrope().equals(List.of(4, 6, 6)),
                    "the text model's rotary sections come from rope.dimension_sections");
            ModelWeights weights = gm.weights();
            try (VisionContext vc = VisionContext.open(proj, tok, arch, new VisionContext.Options("cpu", 2, new Preprocess.Limits(0, 0), 16));
                 Engine engine = new Engine(new CpuBackend(weights, 256, 16, 2, KvDtype.F32, false), tok, arch, Set.of(257))) {
                engine.setVision(vc, false);
                List<ChatTemplate.Message> msgs = List.of(new ChatTemplate.Message("user", MediaChat.MARK + "ab"));
                MediaPrompt mp = MediaChat.render(engine, msgs, false, List.of(img));
                MediaPrompt.Span span = mp.spans().getFirst();
                int imageTokens = cpu.tokens();
                check(mp.spans().size() == 1 && span.length() == imageTokens, "one span of " + imageTokens + " picture tokens");
                int pad = tok.tokenId("<|image_pad|>");
                boolean padsOk = true;
                for (int i = span.start(); i < span.end(); i++) {
                    padsOk &= mp.tokens()[i] == pad && mp.identity()[i] >= MediaPrompt.VIRTUAL_BASE;
                }
                check(padsOk && mp.tokens()[span.start() - 1] == tok.tokenId("<|vision_start|>") && mp.tokens()[span.end()] == tok.tokenId("<|vision_end|>"),
                        "the picture is framed by <|vision_start|> ... <|vision_end|>, filled with the pad token, identified by its hash");
                RopePositions rp = mp.rope();
                int s0 = span.start();
                int nx = cpu.nx();
                int ny = cpu.ny();
                boolean grid = rp != null;
                for (int j = 0; grid && j < span.length(); j++) {
                    grid &= rp.t()[s0 + j] == s0 && rp.h()[s0 + j] == s0 + j / nx && rp.w()[s0 + j] == s0 + j % nx;
                }
                check(grid, "picture tokens keep the time position and count rows and columns from the picture's start position");
                int after = span.end();
                int next = s0 + Math.max(nx, ny);
                check(rp.t()[after] == next && rp.h()[after] == next && rp.tailShift() == mp.length() - (next + (mp.length() - after)),
                        "the text after a picture continues from the largest position the picture used (" + nx + "x" + ny + " tokens: "
                                + span.length() + " slots, " + (next - s0) + " positions)");
                // wrong number of pictures
                try {
                    MediaChat.render(engine, msgs, false, List.of());
                    check(false, "marks without pictures are refused");
                } catch (IllegalArgumentException e) {
                    check(e.getMessage().contains("placeholders"), "marks without pictures are refused");
                }

                // --- generation: deterministic, cache keyed by the picture
                SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1L, 8, List.of());
                Engine.Result r1 = engine.generate(mp, greedy, null, () -> false);
                Engine.Result r1b = engine.generate(mp, greedy, null, () -> false);
                check(r1.tokens().length == 8 && Arrays.equals(r1.tokens(), r1b.tokens()), "the same picture twice gives the same answer");
                check(r1b.stats().cachedTokens() == mp.length() - 1, "the second request reuses the whole prompt but its last token: "
                        + r1b.stats().cachedTokens() + " of " + mp.length());
                MediaPrompt other = MediaChat.render(engine, msgs, false, List.of(randomImage(2, 20, 12)));
                Engine.Result r2 = engine.generate(other, greedy, null, () -> false);
                check(r2.stats().cachedTokens() == other.spans().getFirst().start(),
                        "another picture reuses the text before it and not a token of it: " + r2.stats().cachedTokens());
                Engine.Decoding lp = new Engine.Decoding(null, true, 0, false, null);
                SamplingParams one = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1L, 1, List.of());
                double la = engine.generate(mp, one, lp, 1, null, () -> false).getFirst().logprobs().getFirst().logprob();
                double lb = engine.generate(other, one, lp, 1, null, () -> false).getFirst().logprobs().getFirst().logprob();
                check(Math.abs(la - lb) > 1e-6, "another picture changes the next-token distribution (the model sees it): " + la + " vs " + lb);
                // text after the M-RoPE request: rotary rows are plain again
                List<ChatTemplate.Message> plain = List.of(new ChatTemplate.Message("user", "ab"));
                Engine.Result t1 = engine.generate(engine.renderChat(plain, false), greedy, null, () -> false);
                try (Engine fresh = new Engine(new CpuBackend(weights, 256, 16, 2, KvDtype.F32, false), tok, arch, Set.of(257))) {
                    Engine.Result t2 = fresh.generate(fresh.renderChat(plain, false), greedy, null, () -> false);
                    check(Arrays.equals(t1.tokens(), t2.tokens()) && t1.text().equals(t2.text()),
                            "a text request after a picture request equals one on a fresh engine (the rotary rows were restored)");
                }

                // --- embeddings as input equal token ids
                Backend be = engine.backend();
                int[] ids = {10, 20, 30, 40};
                float[] want = new float[VOCAB];
                float[] got = new float[VOCAB];
                be.forward(ids, 0, 4, 0, want);
                float[] rows = new float[4 * DIM];
                for (int i = 0; i < 4; i++) {
                    weights.embed.dequantizeRow(ids[i], rows, i * DIM);
                }
                be.forwardEmbeds(rows, 0, 4, 0, got);
                check(Arrays.equals(want, got), "forwardEmbeds with the embedding rows of some ids gives the logits of forward(ids)");

                // --- rotary positions: plain positions given explicitly change nothing; shifted ones do
                int n = 6;
                int[] seq = new int[n];
                for (int i = 0; i < n; i++) {
                    seq[i] = 3 + i;
                }
                float[] base = new float[VOCAB];
                be.forward(seq, 0, n, 0, base);
                int[] ident = new int[300];
                for (int i = 0; i < ident.length; i++) {
                    ident[i] = i;
                }
                be.setRopePositions(0, new RopePositions(ident, ident, ident, new int[]{4, 6, 6}, false, 0));
                float[] same = new float[VOCAB];
                be.forward(seq, 0, n, 0, same);
                check(Arrays.equals(base, same), "explicit identity positions give the plain logits, bit for bit");
                int[] shifted = new int[300];
                for (int i = 0; i < shifted.length; i++) {
                    shifted[i] = i < 3 ? i : i + 5;
                }
                be.setRopePositions(0, new RopePositions(shifted, shifted, shifted, new int[]{4, 6, 6}, false, -5));
                float[] moved = new float[VOCAB];
                be.forward(seq, 0, n, 0, moved);
                check(!Arrays.equals(base, moved) && be.ropePositionsCustom(), "shifted positions change the logits and are reported as custom");
                be.setRopePositions(0, null);
                float[] restored = new float[VOCAB];
                be.forward(seq, 0, n, 0, restored);
                check(Arrays.equals(base, restored) && !be.ropePositionsCustom(), "plain positions restored: the plain logits again");

                // --- the same on Metal
                if (metal) {
                    try (Engine me = new Engine(new MetalBackend(weights, 256, 16, (String) null), tok, arch, Set.of(257))) {
                        me.setVision(vc, false);
                        Engine.Result g1 = me.generate(mp, greedy, null, () -> false);
                        check(Arrays.equals(g1.tokens(), r1.tokens()), "Metal text backend: the same answer for the same picture as the CPU backend");
                        Engine.Result g2 = me.generate(mp, greedy, null, () -> false);
                        check(g2.stats().cachedTokens() == mp.length() - 1 && Arrays.equals(g2.tokens(), g1.tokens()),
                                "Metal: the picture is cached by its hash too");
                        Engine.Result gt = me.generate(me.renderChat(plain, false), greedy, null, () -> false);
                        check(Arrays.equals(gt.tokens(), t1.tokens()), "Metal: text after a picture request is as on a fresh engine");
                        float[] mw = new float[VOCAB];
                        me.backend().forward(ids, 0, 4, 0, mw);
                        float[] mg = new float[VOCAB];
                        me.backend().forwardEmbeds(rows, 0, 4, 0, mg);
                        check(Arrays.equals(mw, mg), "Metal: forwardEmbeds with embedding rows equals forward(ids)");
                    }
                }
            }
            weights.close();
        }
    }
}
