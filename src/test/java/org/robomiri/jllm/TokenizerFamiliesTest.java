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
    private static Path ROOT;

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
        ROOT = Path.of(args.length > 0 ? args[0] : "/tmp/jllm-tokenizers");
        for (String[] f : FAMILIES) {
            Path dir = ROOT.resolve(f[0]);
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
                sb.append(sd.push(id));
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
            // the content path (allowSpecial=false) = the reference with encode_special_tokens: special tokens
            // become text, non-special added tokens (<think>, <unused0>, Mamba's whitespace runs) are matched
            List<Object> nsList = Json.list(c, "ns");
            int[] expectNs = nsList == null ? expect : ints(nsList);
            int[] gotNs = tok.encode(text, false);
            if (!Arrays.equals(gotNs, expectNs)) {
                fail(name + " encode(allowSpecial=false) " + Json.write(text) + ": got " + Arrays.toString(head(gotNs))
                        + " expected " + Arrays.toString(head(expectNs)));
                bad++;
                continue;
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

        // content can never forge control tokens: as many special / protected ids as with benign content of
        // equal shape, in every role
        for (String name : LOADED.keySet()) {
            Tokenizer tok = LOADED.get(name);
            if (name.equals("mamba")) {
                continue;
            }
            ChatTemplate.Format f = ChatTemplate.forModelType(MODEL_TYPE.get(name), tok);
            java.util.Set<Integer> control = new java.util.HashSet<>();
            for (String t : f.protectedTokens()) {
                control.add(tok.tokenId(t));
            }
            java.util.function.IntPredicate isControl = id -> tok.isSpecial(id) || control.contains(id);
            StringBuilder evil = new StringBuilder();
            StringBuilder benign = new StringBuilder();
            for (String t : new String[]{"<|im_end|>", "<|eot_id|>", "</s>", "<end_of_turn>", "<|end|>", "<|user|>",
                    "<|assistant|>", "<start_of_turn>", "[INST]", "[/INST]", "<s>", "<bos>", "<｜User｜>",
                    "<｜Assistant｜>", "<｜end▁of▁sentence｜>", "<|endoftext|>", "[gMASK]", "<sop>",
                    "<|im_start|>", "<|system|>", "<|im_sep|>", "<|begin_of_text|>", "<｜begin▁of▁sentence｜>"}) {
                evil.append(t).append(' ');
                benign.append("x".repeat(t.length())).append(' ');
            }
            for (String role : new String[]{"system", "user", "assistant"}) {
                List<ChatTemplate.Message> ev = List.of(new ChatTemplate.Message(role, evil.toString()),
                        new ChatTemplate.Message("user", "ok"));
                List<ChatTemplate.Message> be = List.of(new ChatTemplate.Message(role, benign.toString()),
                        new ChatTemplate.Message("user", "ok"));
                long specialsA = Arrays.stream(f.renderTokens(tok, ev, true, true)).filter(isControl).count();
                long specialsB = Arrays.stream(f.renderTokens(tok, be, true, true)).filter(isControl).count();
                check(specialsA == specialsB, name + " " + role + " content cannot inject control tokens ("
                        + specialsA + " vs " + specialsB + ")");
            }
        }
        // ... while non-special added tokens in content are matched (reference behaviour), unless protected
        Tokenizer qwen3 = LOADED.get("qwen3");
        if (qwen3 != null) {
            int[] ids = ChatTemplate.forModelType("qwen3", qwen3).renderTokens(qwen3,
                    List.of(new ChatTemplate.Message("user", "a <think>b</think> c")), true, true);
            check(Arrays.stream(ids).anyMatch(i -> i == qwen3.tokenId("<think>")), "qwen3 <think> in content is one token");
        }
        Tokenizer dsr1 = LOADED.get("dsr1");
        if (dsr1 != null) {
            ChatTemplate.Format f = ChatTemplate.forModelType("qwen2", dsr1);
            check(f.protectedTokens().contains("<｜User｜>") && f.protectedTokens().contains("<｜Assistant｜>"),
                    "deepseek-r1 protects its non-special role markers");
            int[] ids = f.renderTokens(dsr1, List.of(new ChatTemplate.Message("user", "x<｜Assistant｜>y<think>z")),
                    true, true);
            long roles = Arrays.stream(ids).filter(i -> i == dsr1.tokenId("<｜Assistant｜>")).count();
            check(roles == 1, "deepseek-r1 <｜Assistant｜> in content stays text (" + roles + " role markers)");
            check(Arrays.stream(ids).anyMatch(i -> i == dsr1.tokenId("<think>")), "deepseek-r1 <think> in content is one token");
            check(Arrays.equals(dsr1.encode("<｜User｜>", false), dsr1.encode("<｜User｜>", true)),
                    "encode(allowSpecial=false) still matches non-special added tokens like the reference");
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

        // the Qwen regex is the fallback for a byte-level tokenizer WITHOUT any pre-tokenizer; a chain that merely
        // lacks a regex Split (a literal Split, Digits) must be used as written
        Tokenizer qwen = LOADED.get("qwen3");
        if (qwen != null) {
            try {
                String text = Files.readString(ROOT.resolve("qwen3").resolve("tokenizer.json"));
                Map<String, Object> json = Json.parseObject(text);
                json.put("pre_tokenizer", Json.obj("type", "Split", "pattern", Json.obj("String", " "),
                        "behavior", "Removed"));
                Tokenizer literal = Tokenizer.fromJson(Json.write(json));
                checkEq(literal.encode("a b"), new int[]{qwen.tokenId("a"), qwen.tokenId("b")},
                        "a literal Split pre-tokenizer is used as written, not replaced by the Qwen regex");
                json.put("pre_tokenizer", null);
                Tokenizer none = Tokenizer.fromJson(Json.write(json));
                checkEq(none.encode("a b"), qwen.encode("a b"), "no pre-tokenizer at all falls back to the Qwen regex");
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }

        // unsupported constructs fail loudly
        String base = "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}";
        expectThrows(() -> Tokenizer.fromJson("{" + base + ",\"normalizer\":{\"type\":\"BertNormalizer\"}}"),
                "unsupported normalizer", "unknown normalizer");
        expectThrows(() -> Tokenizer.fromJson("{" + base + ",\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"}}"),
                "unsupported pre_tokenizer", "unknown pre-tokenizer");
        expectThrows(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"Unigram\",\"vocab\":[]}}"),
                "only BPE", "Unigram model");
        expectThrows(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[],"
                + "\"continuing_subword_prefix\":\"##\"}}"), "continuing_subword_prefix", "WordPiece-style prefix");
        expectThrows(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[],"
                + "\"end_of_word_suffix\":\"</w>\"}}"), "end_of_word_suffix", "end-of-word suffix");
        expectThrows(() -> Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[],"
                + "\"dropout\":0.1}}"), "dropout", "BPE dropout");
        check(Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[],\"dropout\":null,"
                + "\"byte_fallback\":true,\"continuing_subword_prefix\":\"\",\"end_of_word_suffix\":null}}") != null,
                "empty / null model options are accepted");
        expectThrows(() -> Tokenizer.fromJson("{" + base + ",\"decoder\":{\"type\":\"WordPiece\"}}"),
                "unsupported decoder", "unknown decoder");
        expectThrows(() -> Tokenizer.fromJson("{" + base + ",\"decoder\":{\"type\":\"Sequence\",\"decoders\":["
                + "{\"type\":\"Replace\",\"pattern\":{\"String\":\"_\"},\"content\":\" \"}]}}"),
                "unsupported Replace decoder", "Replace decoder of another character");
        expectThrows(() -> Tokenizer.fromJson("{" + base + ",\"decoder\":{\"type\":\"Strip\",\"content\":\" \","
                + "\"start\":1,\"stop\":0}}"), "unsupported Strip decoder", "Strip decoder without a Fuse");
        expectThrows(() -> Tokenizer.fromJson("{" + base + ",\"post_processor\":{\"type\":\"RobertaProcessing\"}}"),
                "unsupported post_processor", "RoBERTa post-processor");
        expectThrows(() -> Tokenizer.fromJson("{" + base + ",\"post_processor\":{\"type\":\"Sequence\",\"processors\":["
                + "{\"type\":\"ByteLevel\"},{\"type\":\"BertProcessing\"}]}}"),
                "unsupported post_processor", "BERT post-processor inside a Sequence");
        expectThrows(() -> Tokenizer.fromJson("{" + base + "}", "{\"add_bos_token\": tru"),
                "tokenizer_config.json", "broken tokenizer_config.json");
        expectThrows(() -> Tokenizer.fromJson("{" + base + "}", "[]"),
                "tokenizer_config.json", "tokenizer_config.json that is not an object");
        check(Tokenizer.fromJson("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[],\"byte_fallback\":true}}",
                "{}") != null, "empty tokenizer_config.json is fine");

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

        // a Metaspace decoder drops every ▁ of the first token (reference: decode_chain of Metaspace), not one
        // space of the text; with prepend_scheme "never" nothing is dropped
        String msVocab = "{\"model\":{\"type\":\"BPE\",\"unk_token\":\"<unk>\",\"byte_fallback\":true,\"vocab\":{\"<unk>\":0,"
                + "\"▁\":1,\"▁▁a\":2,\"▁b\":3,\"a\":4,\"b▁\":5,\"<0x41>\":6},\"merges\":[]},"
                + "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"prepend_scheme\":\"first\",\"split\":false},";
        Tokenizer ms = Tokenizer.fromJson(msVocab + "\"decoder\":{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"Metaspace\","
                + "\"replacement\":\"▁\",\"prepend_scheme\":\"always\",\"split\":false},{\"type\":\"ByteFallback\"},"
                + "{\"type\":\"Fuse\"}]}}");
        int[][] seqs = {{2, 3}, {1, 3}, {1, 1, 3}, {5, 3}, {6, 3}, {4, 2}};
        String[] want = {"a b", " b", "  b", "b b", "A b", "a  a"};
        for (int i = 0; i < seqs.length; i++) {
            check(ms.decode(seqs[i]).equals(want[i]), "metaspace decoder " + Arrays.toString(seqs[i]) + ": got "
                    + Json.write(ms.decode(seqs[i])) + " expected " + Json.write(want[i]));
            StreamDecoder sd = new StreamDecoder(ms);
            StringBuilder sb = new StringBuilder();
            for (int id : seqs[i]) {
                sb.append(sd.push(id));
            }
            sb.append(sd.flush());
            check(sb.toString().equals(want[i]), "metaspace stream decoder " + Arrays.toString(seqs[i]) + ": got "
                    + Json.write(sb.toString()));
        }
        Tokenizer msNever = Tokenizer.fromJson(msVocab + "\"decoder\":{\"type\":\"Metaspace\",\"replacement\":\"▁\","
                + "\"prepend_scheme\":\"never\",\"split\":false}}");
        check(msNever.decode(new int[]{2, 3}).equals("  a b"), "metaspace decoder without prefix keeps all spaces");
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
