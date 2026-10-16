package org.robomiri.jllm;

import org.robomiri.jllm.gguf.GgufFile;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The tokenizer built from GGUF metadata against two references.
 *
 * <ol>
 *   <li><b>llama.cpp's own tokenizer tests</b>: the vocabulary-only files {@code ggml-vocab-<name>.gguf} of
 *       llama.cpp's {@code models/} directory come with {@code .inp} (test strings separated by
 *       {@code \n__ggml_vocab_test__\n}) and {@code .out} (the ids llama.cpp produces, one line per string). Run with
 *       {@code --vocab DIR}: every string must give exactly the ids of the {@code .out} file.</li>
 *   <li><b>The Hugging Face tokenizer.json of the same model</b>: with {@code --gguf FILE --hf DIR --fixture FILE}
 *       the fixture strings ({@code src/test/resources/tokenizers/qwen3.json}: ids produced by the Python
 *       {@code tokenizers} package) and the chat renders must give the ids of {@code DIR/tokenizer.json} and of the
 *       fixture, and decoding must round-trip.</li>
 * </ol>
 */
public final class GgufTokenizerTest {

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
        Path vocab = null;
        Path gguf = null;
        Path hf = null;
        Path fixture = null;
        String modelType = "qwen3";
        for (int i = 0; i + 1 < args.length; i += 2) {
            switch (args[i]) {
                case "--vocab" -> vocab = Path.of(args[i + 1]);
                case "--gguf" -> gguf = Path.of(args[i + 1]);
                case "--hf" -> hf = Path.of(args[i + 1]);
                case "--fixture" -> fixture = Path.of(args[i + 1]);
                case "--model-type" -> modelType = args[i + 1];
                default -> throw new IllegalArgumentException(args[i]);
            }
        }
        if (vocab != null) {
            try (var s = Files.list(vocab)) {
                for (Path p : s.filter(x -> x.getFileName().toString().matches("ggml-vocab-.*\\.gguf")).sorted().toList()) {
                    llamaCppVocab(p);
                }
            }
        }
        if (gguf != null && hf != null && fixture != null) {
            againstHf(gguf, hf, fixture, modelType);
        }
        System.out.println("gguf tokenizer: " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    private static void llamaCppVocab(Path gguf) throws Exception {
        Path inp = gguf.resolveSibling(gguf.getFileName() + ".inp");
        Path out = gguf.resolveSibling(gguf.getFileName() + ".out");
        if (!Files.isRegularFile(inp) || !Files.isRegularFile(out)) {
            return;
        }
        String[] tests = Files.readString(inp, StandardCharsets.UTF_8).split("\n__ggml_vocab_test__\n", -1);
        List<String> lines = new ArrayList<>(Arrays.asList(Files.readString(out, StandardCharsets.UTF_8).split("\n", -1)));
        while (lines.size() > tests.length && lines.getLast().isEmpty()) {
            lines.removeLast();
        }
        Tokenizer tok;
        try (GgufFile f = GgufFile.open(gguf)) {
            tok = GgufModel.tokenizer(f);
        } catch (org.robomiri.jllm.tokenizer.GgufTokenizer.UnsupportedPreTokenizer e) {
            System.out.printf("%-34s skipped: %s%n", gguf.getFileName(), e.getMessage());
            return;
        }
        int ok = 0;
        for (int i = 0; i < tests.length && i < lines.size(); i++) {
            int[] want = Arrays.stream(lines.get(i).trim().split("\\s+")).filter(x -> !x.isEmpty()).mapToInt(Integer::parseInt).toArray();
            int[] got = tok.encode(tests[i], true);
            if (Arrays.equals(want, got)) {
                ok++;
            } else {
                check(false, gguf.getFileName() + " case " + i + " " + Json.write(tests[i]).substring(0, Math.min(60, Json.write(tests[i]).length()))
                        + ": want " + Arrays.toString(want) + " got " + Arrays.toString(got));
            }
        }
        check(ok == tests.length, gguf.getFileName() + ": " + ok + "/" + tests.length + " strings match llama.cpp");
        System.out.printf("%-34s %d/%d strings give llama.cpp's ids%n", gguf.getFileName(), ok, tests.length);
    }

    @SuppressWarnings("unchecked")
    private static void againstHf(Path gguf, Path hfDir, Path fixture, String modelType) throws Exception {
        Tokenizer hf = Tokenizer.load(hfDir.resolve("tokenizer.json"));
        Tokenizer tok;
        try (GgufFile f = GgufFile.open(gguf)) {
            tok = GgufModel.tokenizer(f);
        }
        Map<String, Object> fx = Json.parseObject(Files.readString(fixture));
        int n = 0;
        int same = 0;
        for (Object o : Json.list(fx, "cases")) {
            String text = Json.str(o, "text", "");
            int[] want = ((List<Object>) Json.list(o, "ids")).stream().mapToInt(x -> ((Number) x).intValue()).toArray();
            int[] got = tok.encode(text);
            n++;
            if (Arrays.equals(want, got)) {
                same++;
            } else {
                check(false, "case " + Json.write(text) + ": want " + Arrays.toString(want) + " got " + Arrays.toString(got));
            }
            check(Arrays.equals(hf.encode(text), got), "same ids as tokenizer.json for " + Json.write(text));
            check(tok.decode(got).equals(hf.decode(got)), "same decoding as tokenizer.json for " + Json.write(text));
        }
        for (Object o : Json.list(fx, "idseqs")) {
            int[] ids = ((List<Object>) Json.list(o, "ids")).stream().mapToInt(x -> ((Number) x).intValue()).toArray();
            check(tok.decode(ids).equals(Json.str(o, "decoded", "")), "decode of a random id sequence");
        }
        ChatTemplate.Format fmt = ChatTemplate.forModelType(modelType, tok);
        int chats = 0;
        for (Object o : Json.list(fx, "chat")) {
            List<ChatTemplate.Message> msgs = new ArrayList<>();
            for (Object m : Json.list(o, "messages")) {
                msgs.add(new ChatTemplate.Message(Json.str(m, "role", ""), Json.str(m, "content", "")));
            }
            boolean gen = Json.bool(o, "gen", true);
            boolean think = Json.bool(o, "think", false);
            int[] want = ((List<Object>) Json.list(o, "ids")).stream().mapToInt(x -> ((Number) x).intValue()).toArray();
            check(Arrays.equals(want, fmt.renderTokens(tok, msgs, gen, think)), "chat render " + Json.str(o, "conv", "?"));
            chats++;
        }
        check(tok.bosId() == hf.bosId() && tok.addBosByDefault() == hf.addBosByDefault(), "bos id / add_bos match tokenizer.json");
        check(tok.tokenId("<|im_end|>") == hf.tokenId("<|im_end|>") && tok.tokenId("<think>") == hf.tokenId("<think>"),
                "special token ids match");
        check(tok.isSpecial(tok.tokenId("<|im_end|>")) && !tok.isSpecial(tok.tokenId("<think>")),
                "<|im_end|> is special and <think> is not (as in tokenizer.json)");
        System.out.printf("%s: %d/%d fixture strings give the Hugging Face ids, %d chat renders checked%n",
                gguf.getFileName(), same, n, chats);
    }
}
