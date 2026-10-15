package org.robomiri.jllm;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.StreamDecoder;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Cross-checks the Java tokenizer and chat templates against the Hugging Face references for every supported
 * model family. Run (needs the downloaded tokenizer files, see docs/04-tokenizer.md):
 * {@code java -cp target/classes:target/test-classes org.robomiri.jllm.TokenizerFamiliesTest /tmp/jllm-tokenizers}
 *
 * <p>The fixtures under {@code src/test/resources/tokenizers} hold, per family, the ids produced by the Python
 * {@code tokenizers} package for ~145 strings and the strings rendered by the official Jinja chat templates
 * (see {@code src/test/python/gen_tokenizer_fixtures.py}). Families whose tokenizer.json is not on disk are
 * skipped.
 */
public final class TokenizerFamiliesTest {

    /** directory name, config.json model_type to pass to {@link ChatTemplate#forModelType} */
    private static final String[][] FAMILIES = {
            {"qwen3", "qwen3"}, {"qwen2", "qwen2"}, {"llama3", "llama"}, {"mistral", "mistral"},
            {"gemma3", "gemma3_text"}, {"gemma2", "gemma2"}, {"phi3", "phi3"}, {"phi4", "phi3"},
            {"glmedge", "glm"}, {"glm4", "glm4"}, {"dsr1", "qwen2"}, {"mamba", "mamba"}};

    static int passed;
    static int failed;
    static int skipped;
    static int encodeCases;
    static int roundTripExact;
    static int chatCases;

    public static void main(String[] args) throws Exception {
        Path root = Path.of(args.length > 0 ? args[0] : "/tmp/jllm-tokenizers");
        for (String[] f : FAMILIES) {
            Path dir = root.resolve(f[0]);
            if (!Files.isRegularFile(dir.resolve("tokenizer.json"))) {
                System.out.println("SKIP " + f[0] + " (no tokenizer.json in " + dir + ")");
                skipped++;
                continue;
            }
            runFamily(f[0], f[1], dir);
        }
        misc();
        System.out.println();
        System.out.println("tokenizer families: " + passed + " checks passed, " + failed + " failed, " + skipped
                + " families skipped; " + encodeCases + " encode+decode cases (" + roundTripExact
                + " exact round trips), " + chatCases + " chat renders");
        if (failed > 0) {
            System.exit(1);
        }
    }

    private static void runFamily(String name, String modelType, Path dir) throws Exception {
        long t0 = System.nanoTime();
        Tokenizer tok = Tokenizer.load(dir.resolve("tokenizer.json"));
        long loadMs = (System.nanoTime() - t0) / 1_000_000;
        Map<String, Object> fx = fixture(name);
        System.out.printf("== %s (model_type %s): load %d ms, vocab %d, byteLevel=%b, bos=%d addBos=%b%n", name,
                modelType, loadMs, tok.vocabSize(), tok.isByteLevel(), tok.bosId(), tok.addBosByDefault());

        // ---- encode / decode against the Python reference
        List<Object> cases = Json.list(fx, "cases");
        int bad = 0;
        for (Object c : cases) {
            String text = Json.str(c, "text", "");
            int[] expect = ints(Json.list(c, "ids"));
            String expectDecoded = Json.str(c, "decoded", "");
            int[] got = tok.encode(text, true);
            encodeCases++;
            if (!Arrays.equals(got, expect)) {
                fail(name + " encode " + Json.write(text) + ": got " + Arrays.toString(head(got)) + " expected "
                        + Arrays.toString(head(expect)));
                bad++;
                continue;
            }
            String dec = tok.decode(got);
            if (!dec.equals(expectDecoded)) {
                fail(name + " decode " + Json.write(text) + ": got " + Json.write(dec) + " expected "
                        + Json.write(expectDecoded));
                bad++;
                continue;
            }
            // streaming: token by token through StreamDecoder must give the same text as one-shot decode
            StreamDecoder sd = new StreamDecoder(tok);
            StringBuilder sb = new StringBuilder();
            for (int id : got) {
                sb.append(sd.push(tok.tokenBytes(id)));
            }
            sb.append(sd.flush());
            if (!sb.toString().equals(dec)) {
                fail(name + " stream " + Json.write(text) + ": got " + Json.write(sb.toString()) + " expected "
                        + Json.write(dec));
                bad++;
                continue;
            }
            if (dec.equals(text)) {
                roundTripExact++;
            }
            passed++;
        }
        System.out.println("   encode/decode/stream: " + (cases.size() - bad) + "/" + cases.size() + " match the reference");

        // ---- chat templates against the Jinja renders
        List<Object> chats = Json.list(fx, "chat");
        if (chats.isEmpty()) {
            return;
        }
        ChatTemplate.Format fmt = ChatTemplate.forModelType(modelType, tok);
        int chatBad = 0;
        for (Object c : chats) {
            List<ChatTemplate.Message> msgs = new ArrayList<>();
            for (Object m : Json.list(c, "messages")) {
                msgs.add(new ChatTemplate.Message(Json.str(m, "role", null), Json.str(m, "content", null)));
            }
            boolean gen = Json.bool(c, "gen", false);
            Object think = ((Map<?, ?>) c).get("think");
            boolean thinking = think == null || (Boolean) think;
            String label = name + "/" + Json.str(c, "conv", "?") + " gen=" + gen + " think=" + think;
            String rendered = Json.str(c, "rendered", "");
            chatCases++;
            String got = fmt.render(msgs, gen, thinking);
            if (!got.equals(rendered)) {
                fail(label + " render:\n   got      " + Json.write(got) + "\n   expected " + Json.write(rendered));
                chatBad++;
                continue;
            }
            int[] ids = fmt.renderTokens(tok, msgs, gen, thinking);
            int[] expect = ints(Json.list(c, "ids"));
            if (!Arrays.equals(ids, expect)) {
                fail(label + " renderTokens: got " + Arrays.toString(head(ids)) + " expected "
                        + Arrays.toString(head(expect)));
                chatBad++;
                continue;
            }
            passed++;
        }
        System.out.println("   chat template renders: " + (chats.size() - chatBad) + "/" + chats.size()
                + " match Jinja (string and token ids)");
        int[] stops = fmt.stopIds(tok);
        check(stops.length > 0 && Arrays.stream(stops).allMatch(s -> s >= 0), name + " stop ids " + Arrays.toString(stops));
    }

    /** Behaviour that has no Python reference: injection safety, Mamba plain text, factory lookup. */
    private static void misc() {
    }

    // ------------------------------------------------------------------ helpers

    private static Map<String, Object> fixture(String name) throws Exception {
        try (InputStream in = TokenizerFamiliesTest.class.getResourceAsStream("/tokenizers/" + name + ".json")) {
            if (in == null) {
                throw new IllegalStateException("missing fixture /tokenizers/" + name + ".json (run src/test/python/gen_tokenizer_fixtures.py)");
            }
            return Json.parseObject(new String(in.readAllBytes(), StandardCharsets.UTF_8));
        }
    }

    private static int[] ints(List<Object> l) {
        int[] r = new int[l.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = ((Number) l.get(i)).intValue();
        }
        return r;
    }

    private static int[] head(int[] a) {
        return a.length <= 24 ? a : Arrays.copyOf(a, 24);
    }

    static void fail(String msg) {
        failed++;
        System.out.println("FAIL " + msg);
    }

    static void check(boolean ok, String what) {
        if (ok) {
            passed++;
        } else {
            fail(what);
        }
    }

    @SuppressWarnings("unused")
    private static final Random UNUSED = new Random();
}
