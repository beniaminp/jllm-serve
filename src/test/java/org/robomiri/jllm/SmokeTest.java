package org.robomiri.jllm;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.safetensors.SafeTensors;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.StreamDecoder;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Dependency-free checks (no JUnit, by design). Run with:
 * {@code java --add-modules jdk.incubator.vector -cp target/classes:target/test-classes org.robomiri.jllm.SmokeTest}
 */
public final class SmokeTest {

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "models/Qwen_Qwen3-0.6B");
        json();
        tokenizer(dir);
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
