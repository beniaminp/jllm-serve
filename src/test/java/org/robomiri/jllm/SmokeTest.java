package org.robomiri.jllm;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.safetensors.SafeTensors;
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
        tokenizer(dir);
        bpeEquivalence(dir);
        streamDecoder();
        safetensors(dir);
        System.out.println("all smoke tests passed");
    }

    static void json() {
        Map<String, Object> m = Json.parseObject("{\"a\":[1,2.5,\"x\\u00e9\\n\",true,null],\"b\":{\"c\":-3}}");
        check(m.get("a") instanceof List<?> l && l.get(0).equals(1L) && l.get(1).equals(2.5) && l.get(2).equals("xé\n")
                && l.get(3).equals(Boolean.TRUE) && l.get(4) == null, "json parse");
        check(Json.num(Json.map(m, "b"), "c", 0) == -3, "json nested");
        String s = Json.write(Json.obj("k", "v\"\n", "n", 1, "l", List.of(1, 2)));
        check(s.equals("{\"k\":\"v\\\"\\n\",\"n\":1,\"l\":[1,2]}"), "json write: " + s);
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
        String p = ChatTemplate.render(List.of(new ChatTemplate.Message("user", "hi")), true, false);
        check(p.equals("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"), "template");
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
