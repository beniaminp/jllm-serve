package org.robomiri.jllm;

import org.robomiri.jllm.cpu.Int8Kernels;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.DeepSeekRouter;
import org.robomiri.jllm.model.InterleavedRope;
import org.robomiri.jllm.model.LayerSpec;
import org.robomiri.jllm.model.MambaArch;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.Rope;
import org.robomiri.jllm.model.RopeTable;
import org.robomiri.jllm.model.Router;
import org.robomiri.jllm.model.Scaling;
import org.robomiri.jllm.safetensors.SafeTensors;
import org.robomiri.jllm.sampling.Sampler;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.StreamDecoder;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Dependency-free checks (no JUnit, by design). Run with:
 * {@code java --add-modules jdk.incubator.vector -cp target/classes:target/test-classes org.robomiri.jllm.SmokeTest}
 */
public final class SmokeTest {

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "models/Qwen_Qwen3-0.6B");
        json();
        Q2KCheck.run();
        Q6KCheck.run();
        tokenizer(dir);
        tokenizerErrors();
        bpeEquivalence(dir);
        streamDecoder();
        sampler();
        int8Kernels();
        safetensors(dir);
        ropeScaling();
        glmRope();
        yarnRope();
        deepseekRouting();
        modelConfigs();
        longRope();
        phi3Configs();
        GemmaChecks.run();
        SpeculationTest.run(dir);
        mamba();
        kvCacheAttention();
        fastMath();
        threadPool();
        prefixReuse(dir);
        persistCache(dir);
        mlxQuant();
        deepseekConfigs();
        System.out.println("all smoke tests passed");
    }

    /** Int8 activation path: exact integer lanes, bit-identical quantizer, small error against the float kernels. */
    static void int8Kernels() {
        int rows = 256;
        int cols = 1024;
        Random rnd = new Random(7);
        try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofShared()) {
            QTensor.RowSource src = (r, dst) -> {
                Random rr = new Random(r);
                for (int i = 0; i < cols; i++) {
                    dst[i] = (float) rr.nextGaussian() * 0.05f;
                }
            };
            QTensor w4 = QTensor.quantize(arena, QType.Q4_0, rows, cols, src);
            QTensor w8 = QTensor.quantize(arena, QType.Q8_0, rows, cols, src);
            float[] x = new float[2 * cols];
            for (int i = 0; i < x.length; i++) {
                x[i] = (float) rnd.nextGaussian() * (i % 97 == 0 ? 8f : 1f); // a few outliers, like real activations
            }
            short[] xq = new short[cols];
            float[] xs = new float[cols / 32];
            short[] xq2 = new short[cols];
            float[] xs2 = new float[cols / 32];
            Int8Kernels.quantize(x, cols, cols, xq, xs);
            Int8Kernels.quantizeScalar(x, cols, cols, xq2, xs2);
            check(Arrays.equals(xq, xq2) && Arrays.equals(xs, xs2), "int8 quantizer matches the scalar reference");
            int amax = 0;
            for (short v : xq) {
                amax = Math.max(amax, Math.abs(v));
            }
            check(amax == 127, "int8 quantizer uses the full range (max |q| = " + amax + ")");
            double err4 = 0, ref4 = 0, err8 = 0, ref8 = 0, kern = 0;
            int inexact = 0;
            for (int r = 0; r < rows; r++) {
                for (QTensor w : new QTensor[]{w4, w8}) {
                    long exact = 0;
                    for (int b = 0; b < cols / 32; b++) {
                        exact += Int8Kernels.blockDotScalar(w, r, b, xq, 0);
                    }
                    if (Int8Kernels.intDotRow(w, r, xq, 0) != exact) {
                        inexact++;
                    }
                    float i8 = Int8Kernels.dotRow(w, r, xq, xs, 0);
                    kern = Math.max(kern, Math.abs(i8 - Int8Kernels.dotRowScalar(w, r, xq, xs, 0)));
                    float[] wrow = new float[cols];
                    w.dequantizeRow(r, wrow, 0);
                    double f = 0;
                    for (int i = 0; i < cols; i++) {
                        f += (double) wrow[i] * x[cols + i];
                    }
                    if (w == w4) {
                        err4 += (i8 - f) * (i8 - f);
                        ref4 += f * f;
                    } else {
                        err8 += (i8 - f) * (i8 - f);
                        ref8 += f * f;
                    }
                }
            }
            double rel4 = Math.sqrt(err4 / ref4);
            double rel8 = Math.sqrt(err8 / ref8);
            check(inexact == 0, "int8 integer lanes exact on " + rows + " Q4_0 and Q8_0 rows (" + inexact + " mismatches)");
            check(kern < 1e-4, "int8 SIMD kernel matches the scalar scale arithmetic (max abs diff " + kern + ")");
            check(rel4 < 0.02 && rel8 < 0.02, String.format("int8 dot product error vs float: q4 %.3f %%, q8 %.3f %% (rms, relative)",
                    rel4 * 100, rel8 * 100));
        }
    }

    static void json() {
        Map<String, Object> m = Json.parseObject("{\"a\":[1,2.5,\"x\\u00e9\\n\",true,null],\"b\":{\"c\":-3}}");
        check(m.get("a") instanceof List<?> l && l.get(0).equals(1L) && l.get(1).equals(2.5) && l.get(2).equals("xé\n")
                && l.get(3).equals(Boolean.TRUE) && l.get(4) == null, "json parse");
        check(Json.num(Json.map(m, "b"), "c", 0) == -3, "json nested");
        String s = Json.write(Json.obj("k", "v\"\n", "n", 1, "l", List.of(1, 2)));
        check(s.equals("{\"k\":\"v\\\"\\n\",\"n\":1,\"l\":[1,2]}"), "json write: " + s);

        // nesting depth is bounded, so hostile input cannot overflow the stack
        int d = Json.MAX_DEPTH;
        check(Json.parse("[".repeat(d) + "]".repeat(d)) instanceof List<?>, "json nesting at the limit parses");
        check(Json.parse("{\"a\":".repeat(d - 1) + "{}" + "}".repeat(d - 1)) instanceof Map<?, ?>,
                "json object nesting at the limit parses");
        expectIae(() -> Json.parse("[".repeat(d + 1) + "]".repeat(d + 1)), "nesting deeper than",
                "json nesting one past the limit");
        expectIae(() -> Json.parse("{\"a\":".repeat(d) + "{}" + "}".repeat(d)), "nesting deeper than",
                "json object nesting one past the limit");
        expectIae(() -> Json.parse("[".repeat(20_000)), "nesting deeper than", "json 20000 open brackets");
        expectIae(() -> Json.parseObject("{\"x\":" + "[".repeat(100_000) + "}"), "nesting deeper than",
                "json 100000 open brackets inside an object");

        // strict escapes and numbers
        check(Json.parse("\"\\u00e9\\u0041\\uFFfd\"").equals("éA\ufffd"), "json unicode escape");
        for (String bad : new String[]{"\"\\u-1ab\"", "\"\\u+1ab\"", "\"\\u00zz\"", "\"\\u12\"", "\"\\u 123\""}) {
            expectIae(() -> Json.parse(bad), "bad unicode escape", "json rejects " + bad);
        }
        for (String bad : new String[]{"1.", "007", "-007", "1e5e5", "+1", "-", ".5", "1e", "1e+", "1.e5", "0x10",
                "--1", "1-2", "1+2", "[1.]", "{\"a\":01}"}) {
            expectIae(() -> Json.parse(bad), "JSON", "json rejects number " + bad);
        }
        check(Json.parse("0").equals(0L) && Json.parse("-0").equals(0L) && Json.parse("10").equals(10L)
                && Json.parse("-12").equals(-12L), "json integers");
        check(Json.parse("0.5").equals(0.5) && Json.parse("1e5").equals(1e5) && Json.parse("1E-2").equals(0.01)
                && Json.parse("-2.5e+3").equals(-2500.0) && Json.parse("0e0").equals(0.0), "json floats");
        check(Json.parse("1234567890123456789").equals(1234567890123456789L), "json 19-digit integer stays exact");
        check(Json.parse("9223372036854775807").equals(Long.MAX_VALUE)
                && Json.parse("-9223372036854775808").equals(Long.MIN_VALUE), "json long range stays exact");
        check(Json.parse("9223372036854775808").equals(9223372036854775808.0)
                && Json.parse("-9223372036854775809").equals(-9223372036854775809.0)
                && Json.parse("123456789012345678901234567890") instanceof Double, "json beyond long range -> double");
        check(Json.num(Json.parseObject("{\"seed\":1234567890123456789}"), "seed", -1) == 1234567890123456789L,
                "json 19-digit seed round-trips");

        // writer: primitive arrays, and no silent toString for anything else
        check(Json.write(new int[]{1, -2}).equals("[1,-2]") && Json.write(new int[0]).equals("[]"), "json write int[]");
        check(Json.write(new long[]{Long.MAX_VALUE, -1L}).equals("[9223372036854775807,-1]"), "json write long[]");
        check(Json.write(new float[]{1.5f, 2f, Float.NaN}).equals("[1.5,2.0,null]"), "json write float[]");
        check(Json.write(new double[]{0.25, Double.POSITIVE_INFINITY}).equals("[0.25,null]"), "json write double[]");
        check(Json.write(Json.obj("f", new float[]{1f}, "l", new long[]{2L})).equals("{\"f\":[1.0],\"l\":[2]}"),
                "json write arrays nested in objects");
        record Point(int x, int y) {
        }
        expectIae(() -> Json.write(new Point(1, 2)), "cannot serialize", "json write rejects a record");
        expectIae(() -> Json.write(new Object()), "cannot serialize", "json write rejects Object");
        expectIae(() -> Json.write(new String[]{"a"}), "cannot serialize", "json write rejects String[]");
        expectIae(() -> Json.write(Json.obj("k", java.nio.file.StandardOpenOption.READ)), "cannot serialize",
                "json write rejects an enum nested in an object");
    }

    static void tokenizer(Path dir) throws Exception {
        Tokenizer t = Tokenizer.load(dir.resolve("tokenizer.json"));
        // reference ids from HF tokenizers for Qwen2/3 vocab
        checkEq(t.encode("Hello world"), new int[]{9707, 1879}, "hello world");
        checkEq(t.encode("Hello, world!"), new int[]{9707, 11, 1879, 0}, "hello, world!");
        checkEq(t.encode("<|im_start|>user\nhi<|im_end|>\n"), new int[]{151644, 872, 198, 6023, 151645, 198}, "chatml");
        checkEq(t.encode("1234"), new int[]{16, 17, 18, 19}, "digits split individually");
        String[] samples = {"The quick brown fox", "  leading spaces", "naïve café — 日本語のテキスト 🚀", "x\n\ny",
                "def f(x):\n    return x*2", "I'm don't we've", "€100 ≈ $110"};
        for (String s : samples) {
            String back = t.decode(t.encode(s));
            check(back.equals(s), "roundtrip: " + s + " -> " + back);
        }
        check(t.tokenId("<|im_end|>") == 151645 && t.isSpecial(151645), "special tokens");
        String p = ChatTemplate.forModelType("qwen3").render(List.of(new ChatTemplate.Message("user", "hi")), true, false);
        check(p.equals("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"), "template");
    }

    /** Malformed tokenizer.json must fail with IllegalArgumentException and a message naming the problem. */
    static void tokenizerErrors() {
        expectIae(() -> Tokenizer.fromJson("[]"), "expected a JSON object", "non-object tokenizer.json");
        expectIae(() -> Tokenizer.fromJson("{}"), "no \"model\"", "missing model");
        expectIae(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"WordPiece\"}}"), "only BPE", "non-BPE model");
        expectIae(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\"}}"), "no \"vocab\"", "missing vocab");
        expectIae(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":[]}}"), "no \"vocab\"",
                "vocab not an object");
        expectIae(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":\"x\"}}}"), "invalid id",
                "vocab id not a number");
        expectIae(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":-1}}}"), "invalid id",
                "vocab id negative");
        expectIae(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[\"ab\"]}}"),
                "merge #0", "legacy merge without a space");
        expectIae(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[\"a b\",\"c \"]}}"),
                "merge #1", "legacy merge with empty right side");
        expectIae(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[[\"a\"]]}}"),
                "merge #0", "merge pair with one element");
        expectIae(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[[\"a\",1]]}}"),
                "merge #0", "merge pair with non-string");
        expectIae(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[7]}}"),
                "merge #0", "merge entry of wrong type");
        expectIae(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[\"a b\"]}}"),
                "byte-level symbol", "vocab without byte alphabet");
    }

    static void expectIae(Runnable r, String messagePart, String what) {
        try {
            r.run();
        } catch (IllegalArgumentException e) {
            check(e.getMessage() != null && e.getMessage().contains(messagePart),
                    what + ": " + e.getMessage());
            return;
        } catch (RuntimeException e) {
            throw new AssertionError(what + ": expected IllegalArgumentException, got " + e, e);
        }
        throw new AssertionError(what + ": expected IllegalArgumentException, nothing was thrown");
    }

    /**
     * The merge loop in {@link Tokenizer} is the O(n log n) linked-list + heap formulation. This checks it
     * against {@link ReferenceBpe}, a verbatim copy of the original rescan-every-pair implementation, on every
     * vocabulary entry, on random mixed strings and on long single-character runs (which also carry a time bound).
     */
    static void bpeEquivalence(Path dir) throws Exception {
        Tokenizer t = Tokenizer.load(dir.resolve("tokenizer.json"));
        ReferenceBpe ref = new ReferenceBpe(dir.resolve("tokenizer.json"));

        int vocabChecked = 0;
        for (String token : ref.vocabStrings()) {
            String text = new String(ref.decodeByteLevel(token), StandardCharsets.UTF_8);
            int[] a = t.encode(text, false);
            int[] b = ref.encode(text);
            if (!Arrays.equals(a, b)) {
                throw new AssertionError("bpe mismatch on vocab entry " + token + ": got " + Arrays.toString(a)
                        + " expected " + Arrays.toString(b));
            }
            vocabChecked++;
        }
        check(vocabChecked > 150_000, "bpe equivalence on " + vocabChecked + " vocab strings");

        Random rnd = new Random(20240601L);
        String[] pools = {
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "0123456789",
                "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
                " \n\t\r",
                "éñüßçøåÆŒΩπДжЖ日本語中文한국어العربيةעבריתไทय़—€£¥«»…é́​ﬁ",
        };
        int[] emoji = {0x1F680, 0x1F600, 0x1F44D, 0x1F1EF, 0x1F1F5, 0x2603, 0x1F9E0};
        for (int i = 0; i < 2000; i++) {
            int len = 1 + rnd.nextInt(200);
            StringBuilder sb = new StringBuilder();
            while (sb.length() < len) {
                int run = 1 + rnd.nextInt(8);
                int kind = rnd.nextInt(pools.length + 1);
                for (int k = 0; k < run && sb.length() < len; k++) {
                    if (kind == pools.length) {
                        sb.appendCodePoint(emoji[rnd.nextInt(emoji.length)]);
                    } else {
                        String pool = pools[kind];
                        sb.append(pool.charAt(rnd.nextInt(pool.length())));
                    }
                }
            }
            String s = sb.toString();
            int[] a = t.encode(s, false);
            int[] b = ref.encode(s);
            if (!Arrays.equals(a, b)) {
                throw new AssertionError("bpe mismatch on random string #" + i + " " + Json.write(s) + ": got "
                        + Arrays.toString(a) + " expected " + Arrays.toString(b));
            }
        }
        check(true, "bpe equivalence on 2000 random strings");

        String[] runs = {" ".repeat(20_000), "a".repeat(20_000), "\n".repeat(5_000)};
        String[] names = {"20000 spaces", "20000 'a'", "5000 newlines"};
        for (int i = 0; i < runs.length; i++) {
            long t0 = System.nanoTime();
            int[] a = t.encode(runs[i], false);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            int[] b = ref.encode(runs[i]);
            check(Arrays.equals(a, b), "bpe equivalence on " + names[i] + " (" + a.length + " tokens)");
            check(ms < 1000, "encode " + names[i] + " in " + ms + " ms");
        }

        // the BPE cache remembers short chunks only (<= 64 UTF-8 bytes), so long runs cannot fill the heap
        Tokenizer fresh = Tokenizer.load(dir.resolve("tokenizer.json"));
        Map<?, ?> cache = bpeCache(fresh);
        fresh.encode("hello", false);
        fresh.encode("b".repeat(64), false);
        fresh.encode("c".repeat(65), false);
        fresh.encode("é".repeat(40), false); // 40 chars but 80 UTF-8 bytes
        fresh.encode(" ".repeat(20_000), false);
        check(cache.containsKey("hello") && cache.containsKey("b".repeat(64)), "bpe cache keeps chunks <= 64 bytes");
        check(!cache.containsKey("c".repeat(65)) && !cache.containsKey("é".repeat(40))
                && !cache.containsKey(" ".repeat(20_000)), "bpe cache skips chunks > 64 bytes");
        check(cache.size() == 2, "bpe cache size " + cache.size());
    }

    private static Map<?, ?> bpeCache(Tokenizer t) throws Exception {
        java.lang.reflect.Field f = Tokenizer.class.getDeclaredField("cache");
        f.setAccessible(true);
        return (Map<?, ?>) f.get(t);
    }

    /** Verbatim copy of the original {@code Tokenizer} pipeline with the O(n^2) rescan-all-pairs merge loop. */
    static final class ReferenceBpe {
        private static final String DEFAULT_SPLIT =
                "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

        private final Map<String, Integer> vocab = new HashMap<>();
        private final Map<Long, Long> merges = new HashMap<>();
        private final char[] byteToChar = new char[256];
        private final int[] charToByte = new int[512];
        private final int[] byteCharIds = new int[256];
        private final Pattern splitPattern;
        private final boolean nfc;
        private final boolean ignoreMerges;

        ReferenceBpe(Path tokenizerJson) throws Exception {
            Map<String, Object> json = Json.parseObject(Files.readString(tokenizerJson));
            Arrays.fill(charToByte, -1);
            boolean[] direct = new boolean[256];
            for (int b = '!'; b <= '~'; b++) {
                direct[b] = true;
            }
            for (int b = 0xA1; b <= 0xAC; b++) {
                direct[b] = true;
            }
            for (int b = 0xAE; b <= 0xFF; b++) {
                direct[b] = true;
            }
            int extra = 0;
            for (int b = 0; b < 256; b++) {
                char c = direct[b] ? (char) b : (char) (256 + extra++);
                byteToChar[b] = c;
                charToByte[c] = b;
            }
            Map<String, Object> model = Json.map(json, "model");
            ignoreMerges = Json.bool(model, "ignore_merges", false);
            for (Map.Entry<String, Object> e : Json.map(model, "vocab").entrySet()) {
                vocab.put(e.getKey(), ((Number) e.getValue()).intValue());
            }
            for (int b = 0; b < 256; b++) {
                byteCharIds[b] = vocab.get(String.valueOf(byteToChar[b]));
            }
            int rank = 0;
            for (Object m : Json.list(model, "merges")) {
                String a;
                String b;
                if (m instanceof String s) {
                    int sp = s.indexOf(' ');
                    a = s.substring(0, sp);
                    b = s.substring(sp + 1);
                } else {
                    List<?> pair = (List<?>) m;
                    a = (String) pair.get(0);
                    b = (String) pair.get(1);
                }
                Integer ia = vocab.get(a);
                Integer ib = vocab.get(b);
                Integer merged = vocab.get(a + b);
                if (ia != null && ib != null && merged != null) {
                    merges.putIfAbsent(pairKey(ia, ib), ((long) rank << 32) | merged);
                }
                rank++;
            }
            Map<String, Object> normalizer = Json.map(json, "normalizer");
            nfc = normalizer != null && "NFC".equals(Json.str(normalizer, "type", null));
            String regex = null;
            for (Object c : Json.list(Json.map(json, "pre_tokenizer"), "pretokenizers")) {
                if ("Split".equals(Json.str(c, "type", null))) {
                    regex = Json.str(Json.map(c, "pattern"), "Regex", null);
                }
            }
            splitPattern = Pattern.compile(regex == null ? DEFAULT_SPLIT : regex, Pattern.UNICODE_CHARACTER_CLASS);
        }

        Iterable<String> vocabStrings() {
            return vocab.keySet();
        }

        int[] encode(String text) {
            if (nfc) {
                text = Normalizer.normalize(text, Normalizer.Form.NFC);
            }
            List<Integer> out = new ArrayList<>();
            Matcher m = splitPattern.matcher(text);
            int last = 0;
            while (m.find()) {
                if (m.start() > last) {
                    bpe(text.substring(last, m.start()), out);
                }
                if (m.end() > m.start()) {
                    bpe(m.group(), out);
                }
                last = m.end();
            }
            if (last < text.length()) {
                bpe(text.substring(last), out);
            }
            return out.stream().mapToInt(Integer::intValue).toArray();
        }

        private void bpe(String piece, List<Integer> out) {
            byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
            if (ignoreMerges) {
                Integer whole = vocab.get(encodeByteLevel(bytes));
                if (whole != null) {
                    out.add(whole);
                    return;
                }
            }
            int[] sym = new int[bytes.length];
            for (int i = 0; i < bytes.length; i++) {
                sym[i] = byteCharIds[bytes[i] & 0xFF];
            }
            int n = sym.length;
            while (n > 1) {
                long bestRank = Long.MAX_VALUE;
                int bestPos = -1;
                int bestId = -1;
                for (int i = 0; i + 1 < n; i++) {
                    Long v = merges.get(pairKey(sym[i], sym[i + 1]));
                    if (v != null && (v >>> 32) < bestRank) {
                        bestRank = v >>> 32;
                        bestPos = i;
                        bestId = (int) (long) v;
                    }
                }
                if (bestPos < 0) {
                    break;
                }
                sym[bestPos] = bestId;
                System.arraycopy(sym, bestPos + 2, sym, bestPos + 1, n - bestPos - 2);
                n--;
            }
            for (int i = 0; i < n; i++) {
                out.add(sym[i]);
            }
        }

        private static long pairKey(int a, int b) {
            return ((long) a << 32) | (b & 0xFFFFFFFFL);
        }

        private String encodeByteLevel(byte[] bytes) {
            char[] cs = new char[bytes.length];
            for (int i = 0; i < bytes.length; i++) {
                cs[i] = byteToChar[bytes[i] & 0xFF];
            }
            return new String(cs);
        }

        byte[] decodeByteLevel(String token) {
            byte[] out = new byte[token.length()];
            int n = 0;
            for (int i = 0; i < token.length(); i++) {
                char c = token.charAt(i);
                int b = c < charToByte.length ? charToByte[c] : -1;
                if (b >= 0) {
                    out[n++] = (byte) b;
                } else {
                    byte[] u = String.valueOf(c).getBytes(StandardCharsets.UTF_8);
                    out = Arrays.copyOf(out, out.length + u.length);
                    System.arraycopy(u, 0, out, n, u.length);
                    n += u.length;
                }
            }
            return n == out.length ? out : Arrays.copyOf(out, n);
        }
    }

    static void streamDecoder() {
        StreamDecoder d = new StreamDecoder();
        byte[] rocket = "🚀".getBytes(java.nio.charset.StandardCharsets.UTF_8); // 4 bytes
        check(d.push(new byte[]{rocket[0], rocket[1]}).isEmpty(), "held back partial");
        check(d.push(new byte[]{rocket[2], rocket[3]}).equals("🚀"), "completed");
        check(d.push("ab".getBytes()).equals("ab"), "ascii");
    }

    /**
     * Top-p without top-k: the set of tokens the sampler can return must be the nucleus a brute-force
     * reference computes over the full softmax (sort all probabilities, accumulate, include the crossing token).
     * Observed by drawing many samples with a fixed seed: anything outside the nucleus is a failure, and every
     * nucleus token with non-negligible mass must show up.
     */
    static void sampler() {
        // worked example from the review: p = (0.86, 0.05, rest spread thinly), top_p = 0.9 -> keep 2.
        // Normalizing over the surviving candidates alone gives 0.86 / 0.91 = 0.945 >= 0.9 and keeps only 1.
        float[] skewed = probLogits(1000, 0.86, 0.05);
        checkNucleus(skewed, 0.9f, 1f, 2000, "nucleus keeps the crossing token (0.86, 0.05, thin tail)");

        // six-token nucleus where the sixth is the crossing token: cum 0.30 0.50 0.65 0.75 0.83 0.90 vs p = 0.88
        float[] six = probLogits(1000, 0.30, 0.20, 0.15, 0.10, 0.08, 0.07);
        checkNucleus(six, 0.88f, 1f, 4000, "nucleus keeps six tokens (crossing token sixth)");

        // random logits with temperature: whatever the reference says, the sampler agrees
        Random rnd = new Random(7);
        float[] random = new float[4096];
        for (int i = 0; i < random.length; i++) {
            random[i] = (float) (rnd.nextGaussian() * 4);
        }
        checkNucleus(random, 0.9f, 0.7f, 20_000, "nucleus on random logits with T=0.7");
        checkNucleus(random, 0.5f, 1.3f, 20_000, "nucleus on random logits with T=1.3");
    }

    /** Logits whose softmax gives the listed probabilities to tokens 0..k-1 and spreads the rest uniformly. */
    static float[] probLogits(int vocab, double... head) {
        double used = 0;
        for (double p : head) {
            used += p;
        }
        float[] logits = new float[vocab];
        Arrays.fill(logits, (float) Math.log((1 - used) / (vocab - head.length)));
        for (int i = 0; i < head.length; i++) {
            logits[i] = (float) Math.log(head[i]);
        }
        return logits;
    }

    static void checkNucleus(float[] logits, float topP, float temperature, int draws, String what) {
        // brute-force reference over the full vocabulary
        int vocab = logits.length;
        double max = Double.NEGATIVE_INFINITY;
        for (float l : logits) {
            max = Math.max(max, l);
        }
        double[] prob = new double[vocab];
        double sum = 0;
        for (int i = 0; i < vocab; i++) {
            prob[i] = Math.exp((logits[i] - max) / temperature);
            sum += prob[i];
        }
        Integer[] order = new Integer[vocab];
        for (int i = 0; i < vocab; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(prob[b], prob[a]));
        java.util.Set<Integer> nucleus = new java.util.HashSet<>();
        double cum = 0;
        double nucleusMass = 0;
        for (int i : order) {
            nucleus.add(i);
            cum += prob[i] / sum;
            nucleusMass += prob[i];
            if (cum >= topP) {
                break;
            }
        }
        // sample
        SamplingParams params = new SamplingParams(temperature, 0, topP, 0f, 1f, 0f, 0f, 1234L, 1, List.of());
        Sampler s = new Sampler(params, vocab, new int[0]);
        java.util.Set<Integer> observed = new java.util.HashSet<>();
        for (int i = 0; i < draws; i++) {
            observed.add(s.sample(logits.clone()));
        }
        for (int t : observed) {
            if (!nucleus.contains(t)) {
                throw new AssertionError(what + ": sampled token " + t + " outside the nucleus of " + nucleus.size());
            }
        }
        for (int t : nucleus) {
            if (prob[t] / nucleusMass >= 0.01 && !observed.contains(t)) {
                throw new AssertionError(what + ": nucleus token " + t + " (p=" + prob[t] / nucleusMass
                        + " within the nucleus) never sampled in " + draws + " draws");
            }
        }
        check(true, what + ": nucleus of " + nucleus.size() + ", observed " + observed.size() + " distinct tokens");
    }

    static void safetensors(Path dir) throws Exception {
        try (SafeTensors st = SafeTensors.open(dir)) {
            SafeTensors.Tensor e = st.get("model.embed_tokens.weight");
            check(e.dtype() == SafeTensors.DType.BF16 && e.dim(0) == 151936 && e.dim(1) == 1024, "embed shape " + e);
            float[] row = new float[1024];
            e.toFloat(0, row, 0, 1024);
            boolean finite = true;
            for (float v : row) {
                finite &= Float.isFinite(v);
            }
            check(finite, "bf16 conversion finite");
            check(st.tensors().size() == 311, "tensor count " + st.tensors().size());
        }
    }

    /** llama3 RoPE scaling against values computed by hand from the transformers formula. */
    static void ropeScaling() {
        Scaling l3 = new Scaling.Llama3(8, 1, 4, 8192);
        // headDim 8, theta 1e4: inv_freq = 1, 0.1, 0.01, 0.001 with wavelengths 6.28, 62.8, 628, 6283.
        // Below 8192/4 = 2048 they are kept; 6283 lies in (2048, 8192): smooth = (8192/6283.185 - 1)/3 = 0.101266,
        // inv = (1 - smooth) * 0.001 / 8 + smooth * 0.001 = 2.13608e-4.
        double[] f = Rope.invFreq(8, 1e4, l3, 0);
        check(f[0] == 1.0 && Math.abs(f[1] - 0.1) < 1e-15 && Math.abs(f[2] - 0.01) < 1e-15, "llama3 keeps high frequencies");
        check(Math.abs(f[3] - 2.1360754402756864e-4) < 1e-12, "llama3 smoothed band: " + f[3]);
        // theta 1e6: inv_freq = 1, 0.0316, 0.001 (smoothed again), 3.1623e-5 (wavelen 198692 > 8192: / factor 8)
        double[] g = Rope.invFreq(8, 1e6, l3, 0);
        check(Math.abs(g[2] - 2.1360754402756864e-4) < 1e-12 && Math.abs(g[3] - 3.1622776601683795e-5 / 8) < 1e-15,
                "llama3 low frequencies divided by factor: " + g[3]);
        // headDim 64, theta 5e5 (Llama 3): first and last of 32 frequencies
        double[] h = Rope.invFreq(64, 5e5, l3, 0);
        check(h[0] == 1.0 && Math.abs(h[31] - 3.7673226901739635e-7) < 1e-18 && Math.abs(h[1] - 0.6636012376960885) < 1e-12,
                "llama3 headDim 64");
        check(Arrays.equals(Rope.invFreq(8, 1e4, Scaling.NONE, 0), new double[]{1.0, 0.1, 0.01, 0.001}) || Math.abs(
                Rope.invFreq(8, 1e4, Scaling.NONE, 0)[3] - 0.001) < 1e-15, "plain rope unchanged");
        Rope plain = new Rope(8, 1e4, 4);
        Rope scaled = new Rope(new RopeTable(8, 1e4, l3, RopeTable.Pairing.HALF), 4);
        check(plain.cos[8 + 3] != scaled.cos[8 + 3] && plain.cos[8] == scaled.cos[8], "scaled table differs only in low freqs");
    }

    /** GLM interleaved partial rotation on a hand-computed 6-dim head with rotaryDim 4. */
    static void glmRope() throws Exception {
        // pair 0 = (x0, x1) rotates by 90 degrees (cos 0, sin 1), pair 1 = (x2, x3) by 180 degrees (cos -1, sin 0);
        // x4, x5 are outside the rotary dims and stay put.
        // pair 0: (1, 2) -> (1*0 - 2*1, 2*0 + 1*1) = (-2, 1);  pair 1: (3, 4) -> (-3, -4)
        float[] v = {1, 2, 3, 4, 5, 6};
        float[] cos = {0f, -1f};
        float[] sin = {1f, 0f};
        InterleavedRope.rotate(v, 0, 4, cos, sin, 0);
        check(Arrays.equals(v, new float[]{-2, 1, -3, -4, 5, 6}), "glm interleaved rope: " + Arrays.toString(v));
        // the rotate-half convention would pair (x0, x2) and (x1, x3): (1,3)@90 -> (-3,1), (2,4)@180 -> (-2,-4)
        check(v[0] != -3 && v[1] != -2, "interleaved differs from rotate-half");
        // a second head behind the first with a table offset (position 1 of a 2-position table)
        float[] w = {9, 9, 9, 9, 9, 9, 1, 0, 0, 1};
        InterleavedRope.rotate(w, 6, 4, new float[]{1f, 1f, 0f, -1f}, new float[]{0f, 0f, 1f, 0f}, 2);
        check(Arrays.equals(w, new float[]{9, 9, 9, 9, 9, 9, 0, 1, -0f, -1}), "table offset and head offset: " + Arrays.toString(w));
        // tables cover rotaryDim only: theta^(-2i/rotaryDim) (rotaryDim 4 -> 1, 0.01), not theta^(-2i/headDim)
        Rope r = new Rope(4, 1e4, 3);
        check(r.cos.length == 3 * 2 && Math.abs(r.sin[2 * 2 + 1] - (float) Math.sin(0.02)) < 1e-7, "glm rope table width");
        Path tmp = Files.createTempDirectory("jllm-glm");
        try {
            Architecture edge = cfg(tmp, """
                    {"architectures":["GlmForCausalLM"],"partial_rotary_factor":1.0,"attention_bias":false,
                     "eos_token_id":[59246,59253,59255],"head_dim":128,"hidden_act":"silu","hidden_size":2048,
                     "intermediate_size":6144,"model_type":"glm","num_attention_heads":16,"num_hidden_layers":28,
                     "num_key_value_heads":4,"pad_token_id":59246,"rms_norm_eps":1e-05,"rope_theta":10000.0,
                     "tie_word_embeddings":true,"vocab_size":59264}""");
            check(interleaved(edge) && rope(edge).rotaryDim() == 128 && !gqa(edge, 0).bias() && !postNorms(edge)
                    && edge.core().tiedEmbeddings() && edge.core().eosTokenIds().equals(List.of(59246, 59253, 59255)),
                    "GLM-Edge config");
            Architecture chat = cfg(tmp, """
                    {"architectures":["GlmForCausalLM"],"attention_bias":true,"eos_token_id":[151329,151336,151338],
                     "head_dim":128,"hidden_size":4096,"intermediate_size":13696,"model_type":"glm",
                     "num_attention_heads":32,"num_hidden_layers":40,"num_key_value_heads":2,"rms_norm_eps":1.5625e-07,
                     "rope_theta":10000.0,"tie_word_embeddings":false,"vocab_size":151552}""");
            check(rope(chat).rotaryDim() == 64 && gqa(chat, 0).bias() && !postNorms(chat), "glm-4-9b-chat-hf: default partial_rotary_factor 0.5");
            Architecture g4 = cfg(tmp, """
                    {"architectures":["Glm4ForCausalLM"],"attention_bias":true,"head_dim":128,"hidden_size":4096,
                     "intermediate_size":13696,"model_type":"glm4","num_attention_heads":32,"num_hidden_layers":40,
                     "num_key_value_heads":2,"partial_rotary_factor":0.5,"rms_norm_eps":1e-05,"rope_theta":10000.0,
                     "vocab_size":151552}""");
            check(interleaved(g4) && postNorms(g4) && rope(g4).rotaryDim() == 64 && gqa(g4, 0).bias(), "GLM-4-0414 config");
            check(rope(cfg(tmp, """
                    {"model_type":"qwen3","hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":28,
                     "num_attention_heads":16,"head_dim":128,"vocab_size":151936}""")).rotaryDim() == 128, "non-GLM rotates the full head");
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    /** LongRoPE (Phi-3) frequencies, attention factor and partial rotation against hand-computed values. */
    static void longRope() {
        // rotDim 8, theta 1e4: theta^(2i/8) = 1, 10, 100, 1000
        double[] shortF = {1, 2, 4, 8};
        double[] longF = {1, 1.5, 2, 4};
        Scaling.LongRope sc = new Scaling.LongRope(32, 4096, shortF, longF, 0);
        double[] s = Rope.invFreq(8, 1e4, sc, 4096); // ctx <= original: short factors
        check(s[0] == 1.0 && Math.abs(s[1] - 1 / 20.0) < 1e-15 && Math.abs(s[2] - 1 / 400.0) < 1e-15
                && Math.abs(s[3] - 1 / 8000.0) < 1e-18, "longrope short factors: " + Arrays.toString(s));
        double[] g = Rope.invFreq(8, 1e4, sc, 4097);  // ctx > original: long factors
        check(g[0] == 1.0 && Math.abs(g[1] - 1 / 15.0) < 1e-15 && Math.abs(g[2] - 1 / 200.0) < 1e-15
                && Math.abs(g[3] - 1 / 4000.0) < 1e-18, "longrope long factors: " + Arrays.toString(g));
        // factor = 131072 / 4096 = 32: sqrt(1 + ln 32 / ln 4096) = sqrt(1 + 5/12) = sqrt(17/12) = 1.19023807
        check(Math.abs(Rope.attentionFactor(sc) - Math.sqrt(17.0 / 12.0)) < 1e-12
                && Math.abs(Rope.attentionFactor(sc) - 1.1902380714238083) < 1e-12, "longrope attention factor");
        check(Rope.attentionFactor(new Scaling.LongRope(1, 4096, shortF, longF, 0)) == 1.0,
                "attention factor is 1 for factor <= 1");
        check(Rope.attentionFactor(new Scaling.LongRope(32, 4096, shortF, longF, 1.5)) == 1.5,
                "attention_factor override");
        check(Rope.amplitude(Scaling.NONE) == 1.0 && Rope.amplitude(new Scaling.Llama3(8, 1, 4, 8192)) == 1.0,
                "other scalings have no attention factor");
        // table: headDim 8 but only the first 4 dims rotate -> 2 pairs per position, scaled by the attention factor
        Rope r = new Rope(new RopeTable(4, 1e4, sc, RopeTable.Pairing.HALF), 3);
        double amp = Math.sqrt(17.0 / 12.0);
        check(r.cos.length == 3 * 2 && r.sin.length == 3 * 2, "partial rope table size");
        check(Math.abs(r.cos[0] - amp) < 1e-6 && r.sin[0] == 0, "position 0: cos = amp, sin = 0");
        // frequencies use the rotated width: theta^(2i/4) = 1, 100, so pair 1 has inv = 1 / (2 * 100) = 0.005
        // (ctx 3 <= 4096: short factors); position 2 -> angle 0.01
        check(Math.abs(r.cos[2 * 2 + 1] - amp * Math.cos(0.01)) < 1e-6 && Math.abs(r.sin[2 * 2 + 1] - amp * Math.sin(0.01)) < 1e-6,
                "longrope table value at position 2");
        Rope plain = new Rope(8, 1e4, 3);
        check(plain.cos.length == 3 * 4, "full rotation by default");
    }

    static void phi3Configs() throws Exception {
        Path tmp = Files.createTempDirectory("jllm-phi");
        try {
            // Phi-3-mini-4k-instruct (Hub config): head_dim 96, MHA, window 2047, no scaling, BOS <s>
            Architecture p3 = cfg(tmp, """
                    {"architectures":["Phi3ForCausalLM"],"model_type":"phi3","hidden_size":3072,"intermediate_size":8192,
                     "num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":32,"vocab_size":32064,
                     "max_position_embeddings":4096,"original_max_position_embeddings":4096,"rms_norm_eps":1e-05,
                     "rope_scaling":null,"rope_theta":10000.0,"sliding_window":2047,"tie_word_embeddings":false,
                     "bos_token_id":1,"eos_token_id":32000,"attention_bias":false,"hidden_act":"silu"}""");
            check(gqa(p3, 0).headDim() == 96 && rope(p3).rotaryDim() == 96 && rope(p3).scaling() instanceof Scaling.None
                    && gqa(p3, 0).window() == 2047 && gqa(p3, 31).window() == 2047 && !gqa(p3, 0).qkNorm() && !gqa(p3, 0).bias()
                    && gqa(p3, 0).qDim() == 3072 && gqa(p3, 0).kvDim() == 3072, "phi-3-mini config");
            // Phi-4-mini-instruct: GQA 24/8, head_dim 128, partial_rotary_factor 0.75 (96 dims = 48 pairs), tied, BOS = <|endoftext|>
            String f48 = "1.0" + (",1.0").repeat(47);
            Architecture p4 = cfg(tmp, """
                    {"architectures":["Phi3ForCausalLM"],"model_type":"phi3","hidden_size":3072,"intermediate_size":8192,
                     "num_hidden_layers":32,"num_attention_heads":24,"num_key_value_heads":8,"vocab_size":200064,
                     "max_position_embeddings":131072,"original_max_position_embeddings":4096,"partial_rotary_factor":0.75,
                     "rope_scaling":{"type":"longrope","short_factor":[%s],"long_factor":[%s]},"rope_theta":10000.0,
                     "sliding_window":262144,"tie_word_embeddings":true,"bos_token_id":199999,"eos_token_id":199999}"""
                    .formatted(f48, f48));
            Scaling.LongRope p4s = (Scaling.LongRope) rope(p4).scaling();
            check(gqa(p4, 0).headDim() == 128 && rope(p4).rotaryDim() == 96
                    && p4s.shortFactor().length == 48 && p4s.originalMaxPositions() == 4096
                    && p4s.factor() == 32 && p4.core().tiedEmbeddings()
                    && gqa(p4, 5).window() == 262144 && gqa(p4, 0).qDim() == 3072 && gqa(p4, 0).kvDim() == 1024, "phi-4-mini config");
            check(Math.abs(Rope.attentionFactor(p4s) - 1.1902380714238083) < 1e-12, "phi-4-mini attention factor");
            // phi-4 (14B): 40/10 heads, head_dim 128, theta 250000, sliding_window null
            Architecture p14 = cfg(tmp, """
                    {"architectures":["Phi3ForCausalLM"],"model_type":"phi3","hidden_size":5120,"intermediate_size":17920,
                     "num_hidden_layers":40,"num_attention_heads":40,"num_key_value_heads":10,"vocab_size":100352,
                     "max_position_embeddings":16384,"rope_scaling":null,"rope_theta":250000,"sliding_window":null,
                     "tie_word_embeddings":false,"bos_token_id":100257,"eos_token_id":100265}""");
            check(gqa(p14, 0).headDim() == 128 && gqa(p14, 0).window() == 0 && rope(p14).theta() == 250000
                    && p14.core().eosTokenIds().equals(List.of(100265)), "phi-4 config");
            // legacy 'su' name, original_max_position_embeddings only at the top level, factor = 4096 * 32 / 4096
            Architecture su = cfg(tmp, """
                    {"model_type":"phi3","hidden_size":128,"intermediate_size":192,"num_hidden_layers":2,
                     "num_attention_heads":4,"vocab_size":100,"max_position_embeddings":131072,
                     "original_max_position_embeddings":4096,
                     "rope_scaling":{"type":"su","short_factor":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                                     "long_factor":[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]}}""");
            check(rope(su).scaling() instanceof Scaling.LongRope sl && sl.factor() == 32 && sl.longFactor()[3] == 2.0, "su is longrope");
            expectIae(() -> {
                try {
                    cfg(tmp, """
                            {"model_type":"phi3","hidden_size":128,"intermediate_size":192,"num_hidden_layers":2,
                             "num_attention_heads":4,"vocab_size":100,"partial_rotary_factor":0.75,
                             "rope_scaling":{"type":"longrope","short_factor":[1,1],"long_factor":[1,1]}}""");
                } catch (java.io.IOException e) {
                    throw new RuntimeException(e);
                }
            }, "short_factor", "longrope factor count must be rotary_dim/2");
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    static void modelConfigs() throws Exception {
        Path tmp = Files.createTempDirectory("jllm-cfg");
        try {
            Architecture ll = cfg(tmp, """
                    {"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":2048,"intermediate_size":8192,
                     "num_hidden_layers":16,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":64,"vocab_size":128256,
                     "hidden_act":"silu","rms_norm_eps":1e-5,"rope_theta":500000.0,"tie_word_embeddings":true,
                     "bos_token_id":128000,"eos_token_id":128009,
                     "rope_scaling":{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,
                                     "original_max_position_embeddings":8192,"rope_type":"llama3"}}""");
            check(ll.core().tiedEmbeddings() && gqa(ll, 0).headDim() == 64 && !gqa(ll, 0).bias() && !gqa(ll, 0).qkNorm()
                    && gqa(ll, 0).window() == 0 && rope(ll).scaling() instanceof Scaling.Llama3 l3 && l3.factor() == 32
                    && ll.core().eosTokenIds().equals(List.of(128009)), "llama 3.2 config");
            Architecture old = cfg(tmp, """
                    {"model_type":"llama","hidden_size":4096,"intermediate_size":14336,"num_hidden_layers":32,
                     "num_attention_heads":32,"vocab_size":128256,"rope_scaling":{"type":"llama3","factor":8.0,
                     "low_freq_factor":1.0,"high_freq_factor":4.0,"original_max_position_embeddings":8192}}""");
            check(gqa(old, 0).headDim() == 128 && gqa(old, 0).kvHeads() == 32 && rope(old).scaling() instanceof Scaling.Llama3 o3
                    && o3.factor() == 8, "legacy rope 'type' key, head_dim default");
            Architecture mi = cfg(tmp, """
                    {"architectures":["MistralForCausalLM"],"model_type":"mistral","hidden_size":4096,"intermediate_size":14336,
                     "num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":32768,
                     "sliding_window":4096,"bos_token_id":1,"eos_token_id":2}""");
            check(gqa(mi, 0).window() == 4096 && gqa(mi, 31).window() == 4096, "mistral window on every layer");
            Architecture mnull = cfg(tmp, """
                    {"model_type":"mistral","hidden_size":4096,"intermediate_size":14336,"num_hidden_layers":32,
                     "num_attention_heads":32,"vocab_size":32768,"sliding_window":null}""");
            check(gqa(mnull, 0).window() == 0, "mistral sliding_window null");
            Architecture q2 = cfg(tmp, """
                    {"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","hidden_size":1536,"intermediate_size":8960,
                     "num_hidden_layers":28,"num_attention_heads":12,"num_key_value_heads":2,"vocab_size":151936,
                     "sliding_window":32768,"use_sliding_window":false,"max_window_layers":21,"tie_word_embeddings":true,
                     "eos_token_id":[151645,151643]}""");
            check(gqa(q2, 0).bias() && !gqa(q2, 0).qkNorm() && gqa(q2, 27).window() == 0 && gqa(q2, 0).headDim() == 128
                    && q2.core().eosTokenIds().equals(List.of(151645, 151643)), "qwen2.5 config");
            Architecture q2w = cfg(tmp, """
                    {"model_type":"qwen2","hidden_size":1536,"intermediate_size":8960,"num_hidden_layers":28,
                     "num_attention_heads":12,"vocab_size":151936,"sliding_window":4096,"use_sliding_window":true,
                     "max_window_layers":21}""");
            check(gqa(q2w, 20).window() == 0 && gqa(q2w, 21).window() == 4096, "qwen2 window from max_window_layers");
            check(gqa(cfg(tmp, """
                    {"model_type":"qwen3","hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":28,
                     "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"vocab_size":151936}"""), 0).qkNorm(), "qwen3 qk-norm");
            expectIae(() -> {
                try {
                    cfg(tmp, "{\"model_type\":\"mistral\",\"hidden_size\":32,\"intermediate_size\":37,\"num_hidden_layers\":2,"
                            + "\"num_attention_heads\":4,\"vocab_size\":10,\"hidden_act\":\"gelu\"}");
                } catch (java.io.IOException e) {
                    throw new RuntimeException(e);
                }
            }, "hidden_act", "gelu rejected");
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    /** Mamba: config parsing, the selective-scan step and the causal conv with a known state (hand-computed). */
    static void mamba() throws Exception {
        Path tmp = Files.createTempDirectory("jllm-mamba-cfg");
        try {
            MambaArch m = (MambaArch) cfg(tmp, """
                    {"architectures":["MambaForCausalLM"],"model_type":"mamba","hidden_size":768,"state_size":16,
                     "conv_kernel":4,"expand":2,"intermediate_size":1536,"time_step_rank":48,"num_hidden_layers":24,
                     "layer_norm_epsilon":1e-5,"vocab_size":50280,"use_bias":false,"use_conv_bias":true,
                     "bos_token_id":0,"eos_token_id":0}""");
            check(m.core().dim() == 768 && m.core().layers() == 24 && m.core().vocabSize() == 50280 && m.core().tiedEmbeddings()
                    && m.dInner() == 1536 && m.dState() == 16 && m.dConv() == 4 && m.dtRank() == 48
                    && m.convBias() && m.core().eosTokenIds().equals(List.of(0)), "mamba-130m config");
            MambaArch auto = (MambaArch) cfg(tmp, """
                    {"model_type":"mamba","hidden_size":100,"state_size":8,"conv_kernel":3,"expand":2,
                     "time_step_rank":"auto","num_hidden_layers":2,"vocab_size":10}""");
            check(auto.dtRank() == 7 && auto.dInner() == 200, "mamba time_step_rank auto = ceil(d/16), expand");
            expectIae(() -> {
                try {
                    cfg(tmp, "{\"model_type\":\"mamba\",\"hidden_size\":64,\"num_hidden_layers\":1,\"vocab_size\":10,"
                            + "\"use_bias\":true}");
                } catch (java.io.IOException e) {
                    throw new RuntimeException(e);
                }
            }, "use_bias", "mamba use_bias rejected");
            expectIae(() -> {
                try {
                    cfg(tmp, "{\"model_type\":\"mamba2\",\"hidden_size\":64,\"num_hidden_layers\":1,\"vocab_size\":10}");
                } catch (java.io.IOException e) {
                    throw new RuntimeException(e);
                }
            }, "unsupported model", "mamba2 rejected");
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }

        // selective scan, 2 channels x 2 states: s = exp(dt*A)*s + dt*B*u ; y = sum(s*C)
        float[] bc = {1f, 2f, 3f, 4f};             // channel 0: B = (1, 2), C = (3, 4)
        float[] s0 = {1f, 0.5f};
        float y0 = org.robomiri.jllm.cpu.MambaCpuBackend.scanStep(0.5f, 2f, new float[]{-1f, -2f}, 0, s0, 0, bc, 0, 2);
        double e1 = Math.exp(-0.5) + 1.0;          // exp(0.5*-1)*1 + 0.5*1*2
        double e2 = 0.5 * Math.exp(-1.0) + 2.0;    // exp(0.5*-2)*0.5 + 0.5*2*2
        check(Math.abs(s0[0] - e1) < 1e-6 && Math.abs(s0[1] - e2) < 1e-6 && Math.abs(s0[0] - 1.60653066) < 1e-6
                && Math.abs(s0[1] - 2.18393972) < 1e-6, "scan step channel 0 state");
        check(Math.abs(y0 - 13.55535086) < 1e-4, "scan step channel 0 output " + y0 + " = 3*1.60653 + 4*2.18394");
        // channel 1 uses A = (0, -ln 2): exp(0) = 1, exp(-ln 2) = 1/2, so everything is exact
        float[] bc1 = {0.5f, 1f, 1f, 2f};
        float[] s1 = {1f, 1f};
        float y1 = org.robomiri.jllm.cpu.MambaCpuBackend.scanStep(1f, 2f, new float[]{0f, (float) -Math.log(2)}, 0,
                s1, 0, bc1, 0, 2);
        check(Math.abs(s1[0] - 2f) < 1e-6 && Math.abs(s1[1] - 2.5f) < 1e-6 && Math.abs(y1 - 7f) < 1e-5,
                "scan step channel 1: s = (2, 2.5), y = " + y1 + " (expected 7)");
        // a second token on channel 0 continues from the new state
        float y0b = org.robomiri.jllm.cpu.MambaCpuBackend.scanStep(0f, 5f, new float[]{-1f, -2f}, 0, s0, 0, bc, 0, 2);
        check(Math.abs(y0b - y0) < 1e-4 && Math.abs(s0[0] - e1) < 1e-6, "scan step with dt = 0 keeps the state");
        check(Math.abs(org.robomiri.jllm.cpu.MambaCpuBackend.softplus(0f) - Math.log(2)) < 1e-6
                && org.robomiri.jllm.cpu.MambaCpuBackend.softplus(25f) == 25f
                && Math.abs(org.robomiri.jllm.cpu.MambaCpuBackend.softplus(-30f)) < 1e-12, "softplus");

        // causal conv, d_conv = 4, state (oldest first) = (1, -1, 2), new input 3: 1*1 + 2*-1 + 3*2 + 4*3 + 0.5 = 17.5
        float[] st = {1f, -1f, 2f};
        float c1 = org.robomiri.jllm.cpu.MambaCpuBackend.convStep(new float[]{1f, 2f, 3f, 4f}, 0, 0.5f, st, 0, 4, 3f);
        check(Math.abs(c1 - 17.5f / (1f + (float) Math.exp(-17.5))) < 1e-5 && Arrays.equals(st, new float[]{-1f, 2f, 3f}),
                "conv step value " + c1 + " and shifted state " + Arrays.toString(st));
        // taps (0, 0, 0, 1) just pass the input through silu: silu(1) = 0.7310586; then (2, 3, 1) -> window (3, 1, ...)
        float[] st2 = {0f, 0f, 0f};
        float c2 = org.robomiri.jllm.cpu.MambaCpuBackend.convStep(new float[]{0f, 0f, 0f, 1f}, 0, 0f, st2, 0, 4, 1f);
        check(Math.abs(c2 - 0.7310586f) < 1e-6 && Arrays.equals(st2, new float[]{0f, 0f, 1f}), "conv step silu(1)");
        // oldest tap weighs the input from d_conv-1 steps ago: feed 1, 0, 0, 0 with taps (1, 0, 0, 0)
        float[] st3 = {0f, 0f, 0f};
        float[] outs = new float[4];
        float[] in = {1f, 0f, 0f, 0f};
        for (int t = 0; t < 4; t++) {
            outs[t] = org.robomiri.jllm.cpu.MambaCpuBackend.convStep(new float[]{1f, 0f, 0f, 0f}, 0, 0f, st3, 0, 4, in[t]);
        }
        check(Math.abs(outs[3] - 0.7310586f) < 1e-6 && outs[0] == 0f && outs[1] == 0f && outs[2] == 0f,
                "conv tap 0 sees the input d_conv-1 tokens back");
    }

    /**
     * The KV cache's tiled prefill attention against its per-token decode path (same kernels, so the same bits are
     * expected once both are compiled; the check tolerates the last bit because the interpreter's lane reduction
     * order differs from C2's) and against a double-precision reference, for f32 and f16 caches, with and without a
     * sliding window and soft-capping.
     */
    static void kvCacheAttention() throws Exception {
        kvCacheAttention(128); // head dim 32: the vector kernels
        kvCacheAttention(96);  // head dim 24: the one-key-at-a-time fallback where a head is not whole vectors
    }

    static void kvCacheAttention(int hidden) throws Exception {
        Path tmp = Files.createTempDirectory("jllm-kv");
        try {
            Attention.Gqa cfg = gqa(cfg(tmp, "{\"model_type\":\"llama\",\"hidden_size\":" + hidden + ",\"intermediate_size\":64,"
                    + "\"num_hidden_layers\":1,\"num_attention_heads\":4,\"num_key_value_heads\":2,\"vocab_size\":32}"), 0);
            int ctx = 40;
            int n = 23; // several tiles of the f32 and f16 kernels (at least two tasks per pool thread), plus remainders
            int hd = cfg.headDim();
            int qDim = cfg.qDim();
            int kvDim = cfg.kvDim();
            Random rnd = new Random(7);
            float[] k = new float[n * kvDim];
            float[] v = new float[n * kvDim];
            float[] q = new float[n * qDim];
            for (int i = 0; i < k.length; i++) {
                k[i] = rnd.nextFloat() * 2 - 1;
                v[i] = rnd.nextFloat() * 2 - 1;
            }
            for (int i = 0; i < q.length; i++) {
                q[i] = (rnd.nextFloat() * 2 - 1) * 3;
            }
            try (org.robomiri.jllm.cpu.ThreadPool pool = new org.robomiri.jllm.cpu.ThreadPool(3)) {
                for (org.robomiri.jllm.model.KvDtype dtype : org.robomiri.jllm.model.KvDtype.values()) {
                    org.robomiri.jllm.cpu.KvCache kv = org.robomiri.jllm.cpu.KvCache.create(dtype, 1, cfg, ctx);
                    kv.store(0, 0, k, v, n);
                    for (int window : new int[]{0, 6}) {
                        for (float softcap : new float[]{0f, 5f}) {
                            float scale = 0.2f;
                            float[] prefill = new float[n * qDim];
                            float[] decode = new float[n * qDim];
                            float[] one = new float[qDim];
                            float[] qt = new float[qDim];
                            for (int rep = 0; rep < 3; rep++) { // warm the kernels so both paths run compiled code
                                kv.attention(pool, 0, n, 0, q, prefill, window, scale, softcap);
                                for (int t = 0; t < n; t++) {
                                    System.arraycopy(q, t * qDim, qt, 0, qDim);
                                    kv.attention(pool, 0, 1, t, qt, one, window, scale, softcap);
                                    System.arraycopy(one, 0, decode, t * qDim, qDim);
                                }
                            }
                            // double-precision reference on the stored (possibly f16-rounded) rows
                            float[] ks = new float[n * kvDim];
                            float[] vs = new float[n * kvDim];
                            for (int i = 0; i < ks.length; i++) {
                                ks[i] = dtype == org.robomiri.jllm.model.KvDtype.F16
                                        ? Float.float16ToFloat(Float.floatToFloat16(k[i])) : k[i];
                                vs[i] = dtype == org.robomiri.jllm.model.KvDtype.F16
                                        ? Float.float16ToFloat(Float.floatToFloat16(v[i])) : v[i];
                            }
                            double maxRef = 0;
                            double maxPd = 0;
                            int exact = 0;
                            int group = cfg.heads() / cfg.kvHeads();
                            for (int t = 0; t < n; t++) {
                                int len = t + 1;
                                int first = window > 0 ? Math.max(0, len - window) : 0;
                                for (int h = 0; h < cfg.heads(); h++) {
                                    int ko = (h / group) * hd;
                                    double[] sc = new double[len];
                                    double max = Double.NEGATIVE_INFINITY;
                                    for (int s = first; s < len; s++) {
                                        double d = 0;
                                        for (int i = 0; i < hd; i++) {
                                            d += (double) q[t * qDim + h * hd + i] * ks[s * kvDim + ko + i];
                                        }
                                        d *= scale;
                                        if (softcap > 0) {
                                            d = Math.tanh(d / softcap) * softcap;
                                        }
                                        sc[s] = d;
                                        max = Math.max(max, d);
                                    }
                                    double sum = 0;
                                    for (int s = first; s < len; s++) {
                                        sc[s] = Math.exp(sc[s] - max);
                                        sum += sc[s];
                                    }
                                    for (int i = 0; i < hd; i++) {
                                        double o = 0;
                                        for (int s = first; s < len; s++) {
                                            o += sc[s] / sum * vs[s * kvDim + ko + i];
                                        }
                                        int idx = t * qDim + h * hd + i;
                                        maxRef = Math.max(maxRef, Math.abs(o - prefill[idx]));
                                        maxPd = Math.max(maxPd, Math.abs(prefill[idx] - decode[idx]));
                                        if (prefill[idx] == decode[idx]) {
                                            exact++;
                                        }
                                    }
                                }
                            }
                            String what = dtype + " hd=" + hd + " window=" + window + " softcap=" + softcap;
                            check(maxRef < 2e-6, "kv attention " + what + " vs double reference: max diff " + maxRef);
                            // same kernels, so bit-identical once both run C2 code; while one of them still runs
                            // the interpreter's sequential lane reduction a last-bit (1e-7) difference is expected
                            check(maxPd < 5e-7, "kv attention " + what + " tiled prefill vs decode: max diff " + maxPd
                                    + ", " + exact + "/" + (n * qDim) + " bit-identical");
                        }
                    }
                }
            }
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    /**
     * The opt-in polynomial exp / tanh of {@code -Djllm.cpu.fastmath} ({@code VectorKernels.expFast} / {@code tanhFast}):
     * relative error against {@code Math.exp} inside the clamp range, saturation outside it, absolute error of tanh,
     * and lane-wise results identical to the scalar twins.
     */
    static void fastMath() {
        double maxRelExp = 0;
        double maxAbsTanh = 0;
        boolean saturates = true;
        int n = 40001;
        float[] xs = new float[n];
        for (int i = 0; i < n; i++) {
            float x = -100f + i * (200f / (n - 1));
            xs[i] = x;
            float e = org.robomiri.jllm.cpu.VectorKernels.expFast(x);
            if (x >= -87f && x <= 88f) {
                maxRelExp = Math.max(maxRelExp, Math.abs(e - Math.exp(x)) / Math.exp(x));
            } else {
                saturates &= x < 0 ? e > 0 && e <= 2e-38f : e >= 1e38f && e < Float.POSITIVE_INFINITY;
            }
            maxAbsTanh = Math.max(maxAbsTanh, Math.abs(org.robomiri.jllm.cpu.VectorKernels.tanhFast(x) - Math.tanh(x)));
        }
        check(maxRelExp < 3e-7, "fastmath exp relative error " + maxRelExp + " on [-87, 88]");
        check(saturates, "fastmath exp saturates outside the clamp range");
        check(maxAbsTanh < 2e-6, "fastmath tanh absolute error " + maxAbsTanh);
        // lanes vs scalar: softcapFast runs full lanes plus a scalar tail
        float[] capped = xs.clone();
        org.robomiri.jllm.cpu.VectorKernels.softcapFast(capped, 3, n - 3, 30f);
        int lanesDiffer = 0;
        double maxCap = 0;
        for (int i = 3; i < n; i++) {
            if (capped[i] != org.robomiri.jllm.cpu.VectorKernels.tanhFast(xs[i] * (1f / 30f)) * 30f) {
                lanesDiffer++;
            }
            maxCap = Math.max(maxCap, Math.abs(capped[i] - Math.tanh(xs[i] / 30f) * 30f));
        }
        check(lanesDiffer == 0 && capped[0] == xs[0], "fastmath lanes match the scalar twin (" + lanesDiffer + " differ)");
        check(maxCap < 1e-4, "fastmath soft-cap(30) absolute error " + maxCap);
        float[] sm = {1f, -2f, 0.5f, 3f, -1f, 0.25f, 2f, -0.5f, 1.5f};
        float ref = 0;
        for (float v : sm) {
            ref += (float) Math.exp(v - 3f);
        }
        float got = org.robomiri.jllm.cpu.VectorKernels.softmaxExp(sm.clone(), 0, sm.length, 3f);
        check(Math.abs(got - ref) < 1e-6f, "softmaxExp sum " + got + " vs " + ref);
    }

    /**
     * The thread pool's reusable job: thousands of back-to-back jobs land every item exactly once (the epoch-tagged
     * task counter must never let a lagging worker run a stale body or steal a task of the next job), a throwing
     * task is rethrown on the caller with the remaining chunks skipped, and the pool keeps working afterwards.
     */
    static void threadPool() {
        try (org.robomiri.jllm.cpu.ThreadPool pool = new org.robomiri.jllm.cpu.ThreadPool(5)) {
            int items = 777;
            int jobs = 20000;
            int[] hits = new int[items];
            for (int j = 0; j < jobs; j++) {
                int add = j + 1;
                if (j % 2 == 0) {
                    pool.forRange(items, 7, (a, b) -> {
                        for (int i = a; i < b; i++) {
                            hits[i] += add;
                        }
                    });
                } else {
                    pool.forRangeFine(items, 3, (a, b) -> {
                        for (int i = a; i < b; i++) {
                            hits[i] += add;
                        }
                    });
                }
            }
            long expect = (long) jobs * (jobs + 1) / 2;
            boolean ok = true;
            for (int h : hits) {
                ok &= h == expect;
            }
            check(ok, "thread pool: " + jobs + " jobs, every item visited exactly once per job");
            java.util.concurrent.atomic.AtomicInteger ran = new java.util.concurrent.atomic.AtomicInteger();
            String msg = null;
            try {
                pool.forRangeFine(1000, 1, (a, b) -> {
                    if (a == 0) {
                        throw new IllegalStateException("chunk 0 failed");
                    }
                    ran.incrementAndGet();
                    try {
                        Thread.sleep(0, 200_000);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                });
            } catch (IllegalStateException e) {
                msg = e.getMessage();
            }
            check("chunk 0 failed".equals(msg), "thread pool rethrows the first exception on the caller: " + msg);
            check(ran.get() < 999, "thread pool skips chunks after the failure (" + ran.get() + " of 999 ran)");
            int[] again = new int[items];
            pool.forRange(items, 1, (a, b) -> {
                for (int i = a; i < b; i++) {
                    again[i]++;
                }
            });
            boolean fine = true;
            for (int h : again) {
                fine &= h == 1;
            }
            check(fine, "thread pool works after a failed job");
        }
    }

    /** A backend that records the (pos, n) of every forward() and always predicts token 0. */
    private static final class RecordingBackend implements org.robomiri.jllm.model.Backend {
        final boolean reuse;
        final List<int[]> calls = new ArrayList<>();

        RecordingBackend(boolean reuse) {
            this.reuse = reuse;
        }

        public String name() {
            return "recording";
        }

        public int contextLength() {
            return 64;
        }

        public int maxBatch() {
            return 16;
        }

        public boolean supportsPrefixReuse() {
            return reuse;
        }

        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            calls.add(new int[]{pos, n});
            if (logits != null) {
                Arrays.fill(logits, 0f);
                logits[0] = 1f;
            }
        }

        public void close() {
        }
    }

    /**
     * Engine prefix cache: a KV-cache backend resumes after the longest common prefix, a backend whose state cannot
     * be rewound (Mamba) only when the new prompt extends everything it has seen, otherwise it restarts at 0.
     */
    /**
     * {@link org.robomiri.jllm.engine.Engine#saveCache} / {@code loadCache} on the real model, for f32 and f16 caches:
     * a cache saved by one engine and loaded into a fresh one (another backend on the same weights) holds the same
     * tokens, saves back to an identical file, and continues a conversation exactly like the original engine; a wrong
     * fingerprint is refused.
     */
    static void persistCache(Path dir) throws Exception {
        Path tmp = Files.createTempDirectory("jllm-kvfile");
        try {
            for (String dtype : new String[]{"f32", "f16"}) {
                org.robomiri.jllm.config.Settings st = org.robomiri.jllm.config.Settings.load(new String[]{"--backend", "cpu",
                        "--model.dir", dir.toAbsolutePath().getParent().toString(), "--backend.ctx", "256", "--backend.kv-dtype", dtype, "--engine.speculative", "off"});
                try (org.robomiri.jllm.ModelLoader.Loaded l = org.robomiri.jllm.ModelLoader.load(st,
                        org.robomiri.jllm.ModelLoader.hub(st), dir.getFileName().toString());
                     org.robomiri.jllm.engine.Engine fresh = org.robomiri.jllm.ModelLoader.extraEngine(st, l)) {
                    org.robomiri.jllm.engine.Engine e = l.engine();
                    SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 6, List.of());
                    int[] p1 = e.tokenizer().encode("The capital of France is");
                    e.generate(p1, greedy, null, () -> false);
                    Path f = tmp.resolve("a.kv");
                    int saved = e.saveCache(f, "test-model");
                    check(saved > p1.length && Arrays.equals(org.robomiri.jllm.engine.CacheFile.readHeader(f).tokens(),
                            e.cachedTokens()), dtype + " cache saved with its tokens (" + saved + ")");
                    boolean refused = false;
                    try {
                        fresh.loadCache(f, "another-model");
                    } catch (java.io.IOException ex) {
                        refused = ex.getMessage().contains("another model");
                    }
                    check(refused && fresh.cachedTokens().length == 0, dtype + " cache of another model refused");
                    int loaded = fresh.loadCache(f, "test-model");
                    check(loaded == saved && Arrays.equals(fresh.cachedTokens(), e.cachedTokens()), dtype + " cache loaded: " + loaded);
                    Path g = tmp.resolve("b.kv");
                    fresh.saveCache(g, "test-model");
                    check(Arrays.equals(Files.readAllBytes(f), Files.readAllBytes(g)), dtype + " loaded cache saves back identically");
                    int[] cont = e.cachedTokens();
                    int[] more = e.tokenizer().encode(" And of Italy?");
                    int[] p2 = Arrays.copyOf(cont, cont.length + more.length);
                    System.arraycopy(more, 0, p2, cont.length, more.length);
                    org.robomiri.jllm.engine.Engine.Result c;
                    try (org.robomiri.jllm.engine.Engine third = org.robomiri.jllm.ModelLoader.extraEngine(st, l)) {
                        int adopted = third.adoptPrefix(e, p2); // a slot taking over another slot's prefix
                        check(adopted == cont.length && Arrays.equals(third.cachedTokens(), cont),
                                dtype + " prefix copied from another engine: " + adopted + " positions");
                        c = third.generate(p2, greedy, null, () -> false);
                    }
                    org.robomiri.jllm.engine.Engine.Result a = e.generate(p2, greedy, null, () -> false);
                    org.robomiri.jllm.engine.Engine.Result b = fresh.generate(p2, greedy, null, () -> false);
                    check(b.stats().cachedTokens() == saved && a.stats().cachedTokens() == saved
                                    && Arrays.equals(a.tokens(), b.tokens()),
                            dtype + " restored engine continues like the original: " + Arrays.toString(b.tokens()) + " vs "
                                    + Arrays.toString(a.tokens()) + ", reused " + b.stats().cachedTokens());
                    check(c.stats().cachedTokens() == saved && Arrays.equals(a.tokens(), c.tokens()),
                            dtype + " engine with a copied prefix continues like the original: " + Arrays.toString(c.tokens()));
                }
            }
        } finally {
            try (var files = Files.list(tmp)) {
                for (Path f : files.toList()) {
                    Files.deleteIfExists(f);
                }
            }
            Files.deleteIfExists(tmp);
        }
    }

    /**
     * MLX affine quantization ({@link org.robomiri.jllm.model.MlxQuant}): codes packed little-endian into uint32 words
     * here the way MLX packs them, f16 scales and biases per group; the dequantized view must give scale * q + bias
     * exactly, for 4 and 8 bits, and the group size from config.json must win over the shape ambiguity of an 8-bit
     * layer with groups of 64 (= the shapes of 4 bits with groups of 128).
     */
    static void mlxQuant() throws Exception {
        Path tmp = Files.createTempDirectory("jllm-mlx");
        try {
            Files.writeString(tmp.resolve("config.json"), "{\"quantization\": {\"group_size\": 64, \"bits\": 4}}");
            org.robomiri.jllm.model.MlxQuant mq = org.robomiri.jllm.model.MlxQuant.read(tmp);
            check(mq != null && mq.groupSize == 64 && mq.bits == 4, "MLX quantization read from config.json");
            Random rnd = new Random(3);
            for (int bits : new int[]{4, 8}) {
                int rows = 3;
                int cols = 256;
                int groups = cols / 64;
                java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofAuto();
                var packed = arena.allocate((long) rows * cols * bits / 8);
                var scales = arena.allocate((long) rows * groups * 2);
                var biases = arena.allocate((long) rows * groups * 2);
                int[][] q = new int[rows][cols];
                float[][] sc = new float[rows][groups];
                float[][] bi = new float[rows][groups];
                var le = java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN);
                var le16 = java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN);
                for (int r = 0; r < rows; r++) {
                    for (int g = 0; g < groups; g++) {
                        sc[r][g] = Float.float16ToFloat(Float.floatToFloat16(rnd.nextFloat() * 0.1f));
                        bi[r][g] = Float.float16ToFloat(Float.floatToFloat16(-rnd.nextFloat()));
                        scales.setAtIndex(le16, (long) r * groups + g, Float.floatToFloat16(sc[r][g]));
                        biases.setAtIndex(le16, (long) r * groups + g, Float.floatToFloat16(bi[r][g]));
                    }
                    int perWord = 32 / bits;
                    for (int c = 0; c < cols; c++) {
                        q[r][c] = rnd.nextInt(1 << bits);
                        long word = (long) r * (cols / perWord) + c / perWord;
                        int old = packed.getAtIndex(le, word);
                        packed.setAtIndex(le, word, old | (q[r][c] << ((c % perWord) * bits)));
                    }
                }
                var u32 = new org.robomiri.jllm.safetensors.SafeTensors.Tensor("w", org.robomiri.jllm.safetensors.SafeTensors.DType.U32,
                        new long[]{rows, (long) cols * bits / 32}, packed);
                var f16s = new org.robomiri.jllm.safetensors.SafeTensors.Tensor("s", org.robomiri.jllm.safetensors.SafeTensors.DType.F16,
                        new long[]{rows, groups}, scales);
                var f16b = new org.robomiri.jllm.safetensors.SafeTensors.Tensor("b", org.robomiri.jllm.safetensors.SafeTensors.DType.F16,
                        new long[]{rows, groups}, biases);
                var t = mq.dequantized("w", u32, f16s, f16b);
                float[] row = new float[cols];
                boolean exact = t.dim(0) == rows && t.dim(1) == cols;
                for (int r = 0; r < rows && exact; r++) {
                    t.toFloat((long) r * cols, row, 0, cols);
                    for (int c = 0; c < cols; c++) {
                        exact &= row[c] == sc[r][c / 64] * q[r][c] + bi[r][c / 64];
                    }
                }
                check(exact, "MLX " + bits + "-bit dequantization is scale * q + bias (" + t + ")");
            }
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    static void prefixReuse(Path dir) throws Exception {
        Path tmp = Files.createTempDirectory("jllm-engine");
        try {
            Architecture cfg = cfg(tmp, "{\"model_type\":\"mamba\",\"hidden_size\":64,\"num_hidden_layers\":1,"
                    + "\"vocab_size\":100}");
            Tokenizer tok = Tokenizer.load(dir.resolve("tokenizer.json"));
            SamplingParams two = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 2, List.of());
            for (boolean rewind : new boolean[]{true, false}) {
                RecordingBackend b = new RecordingBackend(rewind);
                org.robomiri.jllm.engine.Engine e = new org.robomiri.jllm.engine.Engine(b, tok, cfg, java.util.Set.of(99));
                e.generate(new int[]{5, 6, 7, 8}, two, null, () -> false); // feeds 5 6 7 8 and the generated 0
                check(b.calls.size() == 2 && b.calls.get(0)[0] == 0 && b.calls.get(0)[1] == 4 && b.calls.get(1)[0] == 4,
                        "first request prefills from 0");
                b.calls.clear();
                e.generate(new int[]{5, 6, 7, 8, 0, 9, 10}, two, null, () -> false);
                check(b.calls.get(0)[0] == 5 && b.calls.get(0)[1] == 2, (rewind ? "kv" : "mamba")
                        + " backend: a prompt extending the cached tokens resumes at 5");
                b.calls.clear();
                e.generate(new int[]{5, 6, 7, 8, 0, 9, 11, 12}, two, null, () -> false); // diverges at index 6
                int expect = rewind ? 6 : 0;
                check(b.calls.get(0)[0] == expect, (rewind ? "kv" : "mamba") + " backend: diverging prompt restarts at "
                        + expect + " (got " + b.calls.get(0)[0] + ")");
                b.calls.clear();
                e.generate(new int[]{5, 6, 7, 8, 0, 9, 11, 12}, two, null, () -> false); // identical resend
                check(b.calls.get(0)[0] == (rewind ? 7 : 0), (rewind ? "kv" : "mamba")
                        + " backend: resending the same prompt restarts at " + (rewind ? 7 : 0));
            }
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    /**
     * YaRN inverse frequencies (DeepSeek-V2-Lite: rope dim 64, theta 1e4, factor 40, original 4096, beta 32 / 1) against
     * hand-computed values. Correction dims: 64 ln(4096 / (32 * 2 pi)) / (2 ln 1e4) = 10.47, so low = floor = 10;
     * 64 ln(4096 / (1 * 2 pi)) / (2 ln 1e4) = 22.51, so high = ceil = 23. Pairs i <= 10 keep theta^(-i/32), pairs
     * i >= 23 are divided by 40, in between ramp = (i - 10) / 13 blends: i = 16 gives
     * 0.00025 * 6/13 + 0.01 * 7/13 = 0.0055.
     */
    static void yarnRope() {
        double[] f = Rope.yarnInvFreq(64, 1e4, 40, 4096, 32, 1);
        check(f.length == 32 && f[0] == 1.0, "yarn keeps the fastest pair");
        check(Math.abs(f[5] - Math.pow(1e4, -5.0 / 32)) < 1e-15 && Math.abs(f[10] - 0.056234132519034905) < 1e-15,
                "yarn extrapolates pairs up to the low correction dim: " + f[10]);
        check(Math.abs(f[16] - 0.0055) < 1e-15, "yarn ramp at pair 16: " + f[16]);
        check(Math.abs(f[22] - 0.0001778279410038922) < 1e-15, "yarn ramp at pair 22: " + f[22]);
        check(Math.abs(f[23] - 3.33380358040831e-05) < 1e-18 && Math.abs(f[23] - Math.pow(1e4, -23.0 / 32) / 40) < 1e-18
                && Math.abs(f[31] - Math.pow(1e4, -31.0 / 32) / 40) < 1e-18, "yarn interpolates the slow pairs (divided by 40)");
        // mscale = 0.1 * mscale * ln(factor) + 1
        check(Math.abs(Rope.yarnMscale(40, 0.707) - 1.2608037774058554) < 1e-12 && Rope.yarnMscale(1, 0.707) == 1.0
                && Math.abs(Rope.yarnMscale(40, 1.0) - 1.3688879454113936) < 1e-12, "yarn mscale");
        // the table: cos/sin are multiplied by mscale / mscale_all_dim (1.0 when they are equal)
        Scaling.Yarn same = new Scaling.Yarn(40, 4096, 32, 1, 0.707, 0.707);
        Rope r = new Rope(new RopeTable(64, 1e4, same, RopeTable.Pairing.INTERLEAVED), 4);
        check(r.cos[0] == 1f && r.sin[0] == 0f && Math.abs(r.cos[32 + 16] - (float) Math.cos(0.0055)) < 1e-6,
                "yarn table without mscale correction");
        Scaling.Yarn diff = new Scaling.Yarn(40, 4096, 32, 1, 1.0, 0.707);
        Rope r2 = new Rope(new RopeTable(64, 1e4, diff, RopeTable.Pairing.INTERLEAVED), 4);
        double m = 1.3688879454113936 / 1.2608037774058554;
        check(Math.abs(r2.cos[0] - m) < 1e-6 && Math.abs(r2.sin[32 + 16] - Math.sin(0.0055) * m) < 1e-6,
                "yarn table scaled by mscale / mscale_all_dim");
    }

    /** The three DeepSeek routing rules on hand-made score matrices. */
    static void deepseekRouting() {
        // softmax scores equal to the given probabilities (they sum to 1): logits = ln(p)
        double[] p = {0.30, 0.02, 0.02, 0.01, 0.25, 0.20, 0.15, 0.05};
        int[] idx = new int[3];
        float[] w = new float[3];
        // greedy: the three best experts overall
        route(p, ds("greedy", 1, 1, false, 1.0, false), null, 3, false, idx, w);
        check(Arrays.equals(idx, new int[]{0, 4, 5}) && near(w, 0.30f, 0.25f, 0.20f), "greedy top-3: " + Arrays.toString(idx));
        // group_limited_greedy: 2 groups of 4, keep the group with the best max (g0: 0.30 vs 0.25), top-3 inside it:
        // 0.30, then the 0.02 tie goes to the lower index
        route(p, ds("group_limited_greedy", 2, 1, false, 1.0, false), null, 3, false, idx, w);
        check(Arrays.equals(idx, new int[]{0, 1, 2}) && near(w, 0.30f, 0.02f, 0.02f),
                "group_limited_greedy stays inside the best group: " + Arrays.toString(idx) + Arrays.toString(w));
        // norm_topk_prob renormalizes to sum 1 (0.30 / 0.34 ...); V3 then multiplies by 2.5, V2 does not
        route(p, ds("group_limited_greedy", 2, 1, false, 2.5, false), null, 3, true, idx, w);
        check(near(w, 0.30f / 0.34f, 0.02f / 0.34f, 0.02f / 0.34f), "V2: routed_scaling_factor is skipped when renormalized");
        route(p, ds("group_limited_greedy", 2, 1, false, 2.5, true), null, 3, true, idx, w);
        check(near(w, 2.5f * 0.30f / 0.34f, 2.5f * 0.02f / 0.34f, 2.5f * 0.02f / 0.34f), "V3: renormalized weights times 2.5");
        route(p, ds("group_limited_greedy", 2, 1, false, 2.5, false), null, 3, false, idx, w);
        check(near(w, 0.75f, 0.05f, 0.05f), "V2: routed_scaling_factor applies without renormalization");
        // keeping both groups excludes nothing: same as greedy
        route(p, ds("group_limited_greedy", 2, 2, false, 1.0, false), null, 3, false, idx, w);
        check(Arrays.equals(idx, new int[]{0, 4, 5}), "group_limited_greedy with all groups kept equals greedy");

        // noaux_tc: sigmoid scores s = [.9 .1 | .6 .5 | .8 .7 | .3 .2], bias [-.5 0 | 0 0 | 0 0 | .45 .45]
        // corrected = [.4 .1 | .6 .5 | .8 .7 | .75 .65]; group scores (sum of top 2) 0.5 1.1 1.5 1.4 -> groups 2 and 3;
        // best two corrected inside them: expert 4 (.8), expert 6 (.75); weights are the uncorrected sigmoids .8 and .3
        double[] s = {0.9, 0.1, 0.6, 0.5, 0.8, 0.7, 0.3, 0.2};
        float[] bias = {-0.5f, 0, 0, 0, 0, 0, 0.45f, 0.45f};
        int[] idx2 = new int[2];
        float[] w2 = new float[2];
        Router.DeepSeek v3 = ds("noaux_tc", 4, 2, true, 2.5, true);
        DeepSeekRouter.select(logit(s), bias, v3, 2, true, idx2, w2, new float[8]);
        check(Arrays.equals(idx2, new int[]{4, 6}) && near(w2, 2.5f * 0.8f / 1.1f, 2.5f * 0.3f / 1.1f),
                "noaux_tc: selection on corrected scores, weights from the uncorrected: " + Arrays.toString(idx2) + Arrays.toString(w2));
        // without the bias the raw group sums are 1.0 1.1 1.5 0.5: groups 2 and 1, best two inside: experts 4 and 5
        DeepSeekRouter.select(logit(s), new float[8], v3, 2, true, idx2, w2, new float[8]);
        check(Arrays.equals(idx2, new int[]{4, 5}), "noaux_tc with zero bias: " + Arrays.toString(idx2));
        // noaux_tc ranks groups by the sum of their two best scores, group_limited_greedy by the maximum:
        // [.5 .5] (sum 1.0, max .5) against [.95 .01] (sum .96, max .95)
        double[] t = {0.5, 0.5, 0.95, 0.01};
        int[] i1 = new int[1];
        float[] w1 = new float[1];
        DeepSeekRouter.select(logit(t), new float[4], ds("noaux_tc", 2, 1, true, 1.0, true), 1, false, i1, w1, new float[4]);
        check(i1[0] == 0, "noaux_tc ranks groups by the sum of their two best scores: " + i1[0]);
        DeepSeekRouter.select(logit(t), null, ds("group_limited_greedy", 2, 1, true, 1.0, false), 1, false, i1, w1, new float[4]);
        check(i1[0] == 2, "group_limited_greedy ranks groups by their maximum: " + i1[0]);
    }

    private static Router.DeepSeek ds(String method, int nGroup, int topkGroup, boolean sigmoid, double scale, boolean scaleAlways) {
        return new Router.DeepSeek(sigmoid, Router.TopK.valueOf(method.toUpperCase()), nGroup, topkGroup, scale, scaleAlways);
    }

    /** Logits whose sigmoid is {@code s}. */
    private static float[] logit(double[] s) {
        float[] out = new float[s.length];
        for (int i = 0; i < s.length; i++) {
            out[i] = (float) Math.log(s[i] / (1 - s[i]));
        }
        return out;
    }

    private static void route(double[] probs, Router.DeepSeek d, float[] bias, int k, boolean norm, int[] idx, float[] w) {
        float[] logits = new float[probs.length];
        for (int i = 0; i < probs.length; i++) {
            logits[i] = (float) Math.log(probs[i]);
        }
        DeepSeekRouter.select(logits, bias, d, k, norm, idx, w, new float[probs.length]);
    }

    private static boolean near(float[] got, float... want) {
        for (int i = 0; i < want.length; i++) {
            if (Math.abs(got[i] - want[i]) > 1e-5f) {
                return false;
            }
        }
        return true;
    }

    /** The configs of DeepSeek-V2-Lite and DeepSeek-V3 as published (trimmed to the keys the engine reads). */
    static void deepseekConfigs() throws Exception {
        Path tmp = Files.createTempDirectory("jllm-cfg");
        try {
            Architecture v2 = cfg(tmp, """
                    {"architectures":["DeepseekV2ForCausalLM"],"model_type":"deepseek_v2","bos_token_id":100000,
                     "eos_token_id":100001,"first_k_dense_replace":1,"hidden_act":"silu","hidden_size":2048,
                     "intermediate_size":10944,"kv_lora_rank":512,"max_position_embeddings":163840,
                     "moe_intermediate_size":1408,"moe_layer_freq":1,"n_group":1,"n_routed_experts":64,"n_shared_experts":2,
                     "norm_topk_prob":false,"num_attention_heads":16,"num_experts_per_tok":6,"num_hidden_layers":27,
                     "num_key_value_heads":16,"q_lora_rank":null,"qk_nope_head_dim":128,"qk_rope_head_dim":64,
                     "rms_norm_eps":1e-06,"rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":40,"mscale":0.707,
                     "mscale_all_dim":0.707,"original_max_position_embeddings":4096,"type":"yarn"},"rope_theta":10000,
                     "routed_scaling_factor":1.0,"scoring_func":"softmax","tie_word_embeddings":false,"topk_group":1,
                     "topk_method":"greedy","v_head_dim":128,"vocab_size":102400}""");
            Attention.Mla d = mla(v2);
            Mlp.Moe e2 = moe(v2);
            Router.DeepSeek r2 = (Router.DeepSeek) e2.router();
            check(d.qkDim() == 192 && d.qDim() == 16 * 192 && d.heads() == 16 && d.qLoraRank() == 0 && d.vDim() == 128
                    && d.kvLoraRank() == 512 && e2.shared().hidden() == 2 * 1408 && e2.experts() == 64 && e2.topK() == 6
                    && e2.moeDim() == 1408 && !moeLayer(v2, 0) && moeLayer(v2, 1) && moeLayer(v2, 26) && !v2.core().tiedEmbeddings()
                    && v2.core().eosTokenIds().equals(List.of(100001)) && decoder(v2).maxMlpDim() == 10944
                    && !r2.sigmoid() && r2.topK() == Router.TopK.GREEDY && !r2.scaleAlways(), "deepseek-v2-lite config");
            // softmax scale = 192^-0.5 * (0.1 * 0.707 * ln 40 + 1)^2
            check(Math.abs(d.scale() - 0.07216878364870322 * 1.5896261651208736) < 1e-7
                    && rope(v2).scaling() instanceof Scaling.Yarn y2 && y2.factor() == 40 && rope(v2).rotaryDim() == 64
                    && rope(v2).pairing() == RopeTable.Pairing.INTERLEAVED,
                    "deepseek-v2-lite attention scale with the yarn mscale: " + d.scale());
            Architecture v3 = cfg(tmp, """
                    {"architectures":["DeepseekV3ForCausalLM"],"model_type":"deepseek_v3","bos_token_id":0,"eos_token_id":1,
                     "first_k_dense_replace":3,"hidden_size":7168,"intermediate_size":18432,"kv_lora_rank":512,
                     "moe_intermediate_size":2048,"moe_layer_freq":1,"n_group":8,"n_routed_experts":256,"n_shared_experts":1,
                     "norm_topk_prob":true,"num_attention_heads":128,"num_experts_per_tok":8,"num_hidden_layers":61,
                     "num_key_value_heads":128,"q_lora_rank":1536,"qk_nope_head_dim":128,"qk_rope_head_dim":64,
                     "rms_norm_eps":1e-06,"rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":40,"mscale":1.0,
                     "mscale_all_dim":1.0,"original_max_position_embeddings":4096,"type":"yarn"},"rope_theta":10000,
                     "routed_scaling_factor":2.5,"scoring_func":"sigmoid","tie_word_embeddings":false,"topk_group":4,
                     "topk_method":"noaux_tc","v_head_dim":128,"vocab_size":129280,
                     "quantization_config":{"quant_method":"fp8","weight_block_size":[128,128]}}""");
            Attention.Mla d3 = mla(v3);
            Router.DeepSeek r3 = (Router.DeepSeek) moe(v3).router();
            check(d3.qLoraRank() == 1536 && r3.sigmoid() && r3.topK() == Router.TopK.NOAUX_TC && r3.nGroup() == 8
                    && r3.topkGroup() == 4 && r3.routedScale() == 2.5 && r3.scaleAlways() && moe(v3).normTopK()
                    && !moeLayer(v3, 2) && moeLayer(v3, 3) && decoder(v3).maxMlpDim() == 18432
                    && Math.abs(d3.scale() - 0.07216878364870322 * 1.3688879454113936 * 1.3688879454113936) < 1e-7,
                    "deepseek-v3 config");
            // the V2 chat template of tokenizer_config.json
            ChatTemplate.Format f = ChatTemplate.forModelType("deepseek_v2");
            check(f.name().equals("deepseek-v2") && f.render(List.of(new ChatTemplate.Message("system", "Be brief."),
                            new ChatTemplate.Message("user", "Hi"), new ChatTemplate.Message("assistant", "Hello"),
                            new ChatTemplate.Message("user", "Bye")), true, false)
                            .equals("<｜begin▁of▁sentence｜>Be brief.\n\nUser: Hi\n\nAssistant: Hello<｜end▁of▁sentence｜>"
                                    + "User: Bye\n\nAssistant:"), "deepseek-v2 chat template");
            check(ChatTemplate.forModelType("deepseek_v3").name().equals("deepseek-r1"), "deepseek_v3 uses the V3/R1 template");
            // an FP8 checkpoint (V3 / R1 as published) is refused up front with an explanation
            Path fp8 = Files.createTempDirectory("jllm-fp8");
            try {
                Files.writeString(fp8.resolve("model.safetensors.index.json"),
                        "{\"weight_map\":{\"model.embed_tokens.weight\":\"a.safetensors\","
                                + "\"model.layers.0.self_attn.q_a_proj.weight\":\"a.safetensors\","
                                + "\"model.layers.0.self_attn.q_a_proj.weight_scale_inv\":\"a.safetensors\"}}");
                try {
                    org.robomiri.jllm.model.ModelConverter.convert(fp8, v3, org.robomiri.jllm.model.QType.Q4_0,
                            org.robomiri.jllm.model.ShardProvider.local(fp8));
                    check(false, "fp8 checkpoint must be refused");
                } catch (java.io.IOException e) {
                    check(e.getMessage().contains("FP8"), "fp8 checkpoint refused: " + e.getMessage());
                }
            } finally {
                Files.deleteIfExists(fp8.resolve("model.safetensors.index.json"));
                Files.deleteIfExists(fp8);
            }
            expectIae(() -> {
                try {
                    cfg(tmp, "{\"model_type\":\"deepseek_v2\",\"hidden_size\":64,\"intermediate_size\":64,\"num_hidden_layers\":2,"
                            + "\"num_attention_heads\":2,\"vocab_size\":10,\"kv_lora_rank\":32,\"qk_nope_head_dim\":32,"
                            + "\"qk_rope_head_dim\":32,\"v_head_dim\":32,\"topk_method\":\"magic\"}");
                } catch (java.io.IOException e) {
                    throw new RuntimeException(e);
                }
            }, "topk_method", "unknown topk_method rejected");
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    static Architecture cfg(Path dir, String json) throws java.io.IOException {
        Files.writeString(dir.resolve("config.json"), json);
        return Architecture.load(dir);
    }

    // ------------------------------------------------------------------ architecture helpers for the config checks

    static Decoder decoder(Architecture a) {
        return (Decoder) a;
    }

    /** The grouped-query attention of layer {@code l}. */
    static Attention.Gqa gqa(Architecture a, int l) {
        return (Attention.Gqa) decoder(a).layers().get(l).attn();
    }

    /** The latent attention (uniform across layers). */
    static Attention.Mla mla(Architecture a) {
        return (Attention.Mla) decoder(a).attention();
    }

    /** The global rotary table. */
    static RopeTable rope(Architecture a) {
        return decoder(a).ropeTables().getFirst();
    }

    static boolean interleaved(Architecture a) {
        return rope(a).pairing() == RopeTable.Pairing.INTERLEAVED;
    }

    static boolean postNorms(Architecture a) {
        return decoder(a).layers().getFirst().norms().post();
    }

    static boolean moeLayer(Architecture a, int l) {
        return decoder(a).layers().get(l).mlp() instanceof Mlp.Moe;
    }

    /** The first MoE layer's spec. */
    static Mlp.Moe moe(Architecture a) {
        for (LayerSpec s : decoder(a).layers()) {
            if (s.mlp() instanceof Mlp.Moe m) {
                return m;
            }
        }
        throw new IllegalStateException("no MoE layer in " + a);
    }

    static void checkEq(int[] a, int[] b, String what) {
        check(Arrays.equals(a, b), what + ": got " + Arrays.toString(a) + " expected " + Arrays.toString(b));
    }

    static void check(boolean ok, String what) {
        if (!ok) {
            throw new AssertionError(what);
        }
        System.out.println("ok  " + what);
    }
}
