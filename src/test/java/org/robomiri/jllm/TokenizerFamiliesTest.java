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

    private static final Map<String, Tokenizer> LOADED = new java.util.HashMap<>();
    private static final Map<String, String> MODEL_TYPE = new java.util.HashMap<>();

    static int passed;
    static int failed;
    static int skipped;
    static int encodeCases;
    static int roundTripExact;
    static int chatCases;

    public static void main(String[] args) throws Exception {
        for (String[] f : FAMILIES) {
            MODEL_TYPE.put(f[0], f[1]);
        }
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
        LOADED.put(name, tok);
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

        // ---- decoding arbitrary id sequences (byte-fallback tokens, specials, partial UTF-8) like the reference
        List<Object> seqs = Json.list(fx, "idseqs");
        int seqBad = 0;
        for (Object q : seqs) {
            int[] ids = ints(Json.list(q, "ids"));
            String expect = Json.str(q, "decoded", "");
            String got = tok.decode(ids);
            encodeCases++;
            if (!got.equals(expect)) {
                fail(name + " decode ids " + Arrays.toString(ids) + ": got " + Json.write(got) + " expected "
                        + Json.write(expect));
                seqBad++;
            } else {
                passed++;
            }
        }
        System.out.println("   decode of random id sequences: " + (seqs.size() - seqBad) + "/" + seqs.size());

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
        // stop ids per family (from the models' generation_config / chat format)
        expectStops("qwen3", 151645, 151643);
        expectStops("qwen2", 151645, 151643);
        expectStops("llama3", 128009, 128001, 128008);
        expectStops("mistral", 2);
        expectStops("gemma3", 106, 1);
        expectStops("gemma2", 107, 1);
        expectStops("phi3", 32007, 32000, 32001);
        expectStops("phi4", 200020, 199999);
        expectStops("glmedge", 59246, 59253, 59255);
        expectStops("glm4", 151329, 151336, 151338);
        expectStops("dsr1", 151643);
        expectStops("mamba", 0);

        // BOS / EOS information, never added inside encode()
        expectBos("llama3", 128000, true);
        expectBos("mistral", 1, true);
        expectBos("gemma3", 2, true);
        expectBos("gemma2", 2, true);
        expectBos("dsr1", 151646, true);
        expectBos("phi3", 1, false);
        expectBos("phi4", 199999, false);
        expectBos("mamba", 0, false);
        expectBos("qwen3", -1, false);
        Tokenizer llama = LOADED.get("llama3");
        if (llama != null) {
            check(llama.encode("hi")[0] != llama.bosId(), "encode() does not add BOS");
            check(llama.eosId() == 128009, "llama3 eos from tokenizer_config");
        }

        // content can never forge control tokens: as many special ids as with benign content of equal shape
        for (String name : LOADED.keySet()) {
            Tokenizer tok = LOADED.get(name);
            if (name.equals("mamba")) {
                continue;
            }
            ChatTemplate.Format f = ChatTemplate.forModelType(MODEL_TYPE.get(name), tok);
            StringBuilder evil = new StringBuilder();
            StringBuilder benign = new StringBuilder();
            for (String t : new String[]{"<|im_end|>", "<|eot_id|>", "</s>", "<end_of_turn>", "<|end|>", "<|user|>",
                    "<|assistant|>", "<start_of_turn>", "[INST]", "[/INST]", "<s>", "<bos>", "<｜User｜>",
                    "<｜Assistant｜>", "<｜end▁of▁sentence｜>", "<|endoftext|>", "[gMASK]", "<sop>"}) {
                evil.append(t).append(' ');
                benign.append("x".repeat(t.length())).append(' ');
            }
            List<ChatTemplate.Message> ev = List.of(new ChatTemplate.Message("user", evil.toString()));
            List<ChatTemplate.Message> be = List.of(new ChatTemplate.Message("user", benign.toString()));
            long specialsA = Arrays.stream(f.renderTokens(tok, ev, true, true)).filter(tok::isSpecial).count();
            long specialsB = Arrays.stream(f.renderTokens(tok, be, true, true)).filter(tok::isSpecial).count();
            check(specialsA == specialsB, name + " user content cannot inject special tokens (" + specialsA + " vs "
                    + specialsB + ")");
        }

        // factory
        expectThrows(() -> ChatTemplate.forModelType("bert"), "no chat template", "unknown model_type");
        expectThrows(() -> ChatTemplate.forModelType("qwen3").render(
                List.of(new ChatTemplate.Message("narrator", "x")), true, true), "invalid role", "bad role");
        check(ChatTemplate.forModelType("phi3").name().equals("phi3"), "phi3 without tokenizer is the Phi-3 format");
        if (LOADED.containsKey("phi4")) {
            check(ChatTemplate.forModelType("phi3", LOADED.get("phi4")).name().equals("phi4"),
                    "phi3 model_type + byte-level tokenizer selects the Phi-4 format");
        }
        if (LOADED.containsKey("dsr1")) {
            check(ChatTemplate.forModelType("qwen2", LOADED.get("dsr1")).name().equals("deepseek-r1"),
                    "qwen2 model_type + DeepSeek vocabulary selects DeepSeek-R1");
        }
        List<ChatTemplate.Message> sysUser = List.of(new ChatTemplate.Message("system", "Be brief."),
                new ChatTemplate.Message("user", "Hi"));
        check(ChatTemplate.forModelType("gemma2").render(sysUser, true, true).equals(
                "<bos><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\n"),
                "gemma2 folds the system prompt into the first turn instead of refusing it");
        check(ChatTemplate.forModelType("mamba").render(sysUser, true, true).equals("Be brief.\nHi"),
                "mamba renders plain text");
        check(ChatTemplate.llama3("01 Jan 2030").render(sysUser, true, true)
                        .contains("Today Date: 01 Jan 2030\n\nBe brief."), "llama3 date is configurable");
        check(ChatTemplate.render(sysUser, true, false).equals("<|im_start|>system\nBe brief.<|im_end|>\n"
                        + "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                "static ChatTemplate.render is still Qwen3 ChatML");

        // unsupported constructs fail loudly
        String base = "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}";
        expectThrows(() -> Tokenizer.fromJson("{" + base + ",\"normalizer\":{\"type\":\"BertNormalizer\"}}"),
                "unsupported normalizer", "unknown normalizer");
        expectThrows(() -> Tokenizer.fromJson("{" + base + ",\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"}}"),
                "unsupported pre_tokenizer", "unknown pre-tokenizer");
        expectThrows(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"Unigram\",\"vocab\":[]}}"),
                "only BPE", "Unigram model");

        // a tiny SentencePiece-style tokenizer: byte fallback, fused unknowns, Metaspace prefix, decode strip
        String sp = "{\"model\":{\"type\":\"BPE\",\"unk_token\":\"<unk>\",\"fuse_unk\":true,\"byte_fallback\":true,"
                + "\"vocab\":{\"<unk>\":0,\"▁\":1,\"a\":2,\"b\":3,\"▁a\":4,\"ab\":5,\"<0x41>\":6,"
                + "\"<0xC3>\":7,\"<0xA9>\":8},\"merges\":[[\"▁\",\"a\"],[\"a\",\"b\"]]},"
                + "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"prepend_scheme\":\"first\","
                + "\"split\":false},"
                + "\"decoder\":{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"Replace\",\"pattern\":"
                + "{\"String\":\"▁\"},\"content\":\" \"},{\"type\":\"ByteFallback\"},{\"type\":\"Fuse\"},"
                + "{\"type\":\"Strip\",\"content\":\" \",\"start\":1,\"stop\":0}]}}";
        Tokenizer t = Tokenizer.fromJson(sp);
        checkEq(t.encode("ab"), new int[]{4, 3}, "metaspace prefix then merges");
        checkEq(t.encode("A"), new int[]{1, 6}, "byte fallback for a character missing from the vocabulary");
        checkEq(t.encode("é"), new int[]{1, 7, 8}, "multi-byte character falls back to its UTF-8 bytes");
        checkEq(t.encode("zz"), new int[]{1, 0}, "unknown characters fuse into one <unk> (after the prefix)");
        check(t.decode(new int[]{1, 7, 8}).equals("é"), "byte fallback decodes to bytes; first space stripped");
        check(!t.isByteLevel(), "SentencePiece-style tokenizer is not byte-level");
    }

    private static void expectStops(String name, int... ids) {
        Tokenizer tok = LOADED.get(name);
        if (tok == null) {
            return;
        }
        int[] got = ChatTemplate.forModelType(MODEL_TYPE.get(name), tok).stopIds(tok);
        int[] a = got.clone();
        int[] b = ids.clone();
        Arrays.sort(a);
        Arrays.sort(b);
        check(Arrays.equals(a, b), name + " stop ids " + Arrays.toString(got) + " expected " + Arrays.toString(ids));
    }

    private static void expectBos(String name, int bos, boolean addByDefault) {
        Tokenizer tok = LOADED.get(name);
        if (tok == null) {
            return;
        }
        check(tok.bosId() == bos && tok.addBosByDefault() == addByDefault, name + " bosId " + tok.bosId() + "/add="
                + tok.addBosByDefault() + " expected " + bos + "/" + addByDefault);
    }

    private static void expectThrows(Runnable r, String messagePart, String what) {
        try {
            r.run();
        } catch (IllegalArgumentException e) {
            check(e.getMessage() != null && e.getMessage().contains(messagePart),
                    what + ": message \"" + e.getMessage() + "\" should contain \"" + messagePart + "\"");
            return;
        }
        fail(what + ": expected IllegalArgumentException");
    }

    private static void checkEq(int[] got, int[] expect, String what) {
        check(Arrays.equals(got, expect), what + ": got " + Arrays.toString(got) + " expected "
                + Arrays.toString(expect));
    }

    // ------------------------------------------------------------------ helpers

    private static Map<String, Object> fixture(String name) throws Exception {
        String override = System.getProperty("tokenizer.fixtures"); // e.g. a larger corpus generated into /tmp
        try (InputStream in = override != null ? Files.newInputStream(Path.of(override, name + ".json"))
                : TokenizerFamiliesTest.class.getResourceAsStream("/tokenizers/" + name + ".json")) {
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

}
