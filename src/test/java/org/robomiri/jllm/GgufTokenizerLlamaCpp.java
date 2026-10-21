package org.robomiri.jllm;

import org.robomiri.jllm.gguf.GgufFile;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.tokenizer.GgufTokenizer;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * The GGUF tokenizer against llama.cpp's own, on vocabularies written here (vocabulary-only GGUF files, no weights).
 *
 * <p>For every {@code tokenizer.ggml.pre} value the tokenizer supports a byte-level BPE vocabulary is generated: the 256
 * byte symbols, a merge for <i>every</i> pair of byte symbols (65 536 merges in a pseudo-random order, so wherever the
 * pre-tokenizer puts a boundary between two pieces the pair across it is a merge that did not happen, which makes the
 * boundary visible in the ids), a few merge chains (" the", "Hello"), a few tokens without merges (to see
 * {@code ignore_merges}) and a handful of control / user-defined tokens. SentencePiece vocabularies (Llama-2-, Gemma-
 * and Phi-3-like) have byte tokens, scored pieces and user-defined tokens. A corpus of hand-written strings
 * (contractions, digit runs, CJK, accents, emoji, whitespace of every kind, code, special tokens in the text) and
 * random strings over a Unicode alphabet is tokenized by {@code llama-tokenize} and by {@link GgufModel#tokenizer}.
 *
 * <pre>
 *   gen DIR                    write the vocabularies (DIR/synth-*.gguf) and DIR/corpus.txt
 *   check [DIR]                run llama-tokenize on every vocabulary (skips with a message when it is not installed);
 *                              prints PASS / FAIL per pre value, exits 1 on a failure
 *   record FILE.gz             store llama-tokenize's ids for the in-repo test (a subset of the corpus)
 *   verify FILE.gz             compare with the stored ids; no llama.cpp needed ({@link #verifyRecorded})
 * </pre>
 */
public final class GgufTokenizerLlamaCpp {

    static final String SEP = "<|sep|>";

    private GgufTokenizerLlamaCpp() {
    }

    // ------------------------------------------------------------------ byte-level alphabet

    private static final String[] BYTE_CHAR = new String[256];

    static {
        List<Integer> bs = new ArrayList<>();
        for (int b = '!'; b <= '~'; b++) {
            bs.add(b);
        }
        for (int b = 0xA1; b <= 0xAC; b++) {
            bs.add(b);
        }
        for (int b = 0xAE; b <= 0xFF; b++) {
            bs.add(b);
        }
        int n = 0;
        for (int b = 0; b < 256; b++) {
            BYTE_CHAR[b] = new String(Character.toChars(bs.contains(b) ? b : 256 + n++));
        }
    }

    /** The GPT-2 byte-level spelling of a text. */
    static String enc(String text) {
        StringBuilder sb = new StringBuilder();
        for (byte b : text.getBytes(StandardCharsets.UTF_8)) {
            sb.append(BYTE_CHAR[b & 0xFF]);
        }
        return sb.toString();
    }

    // ------------------------------------------------------------------ the byte-level BPE vocabulary

    /** Tokens, types and merges shared by every pre value (only {@code tokenizer.ggml.pre} differs). */
    record Bpe(List<String> tokens, List<Integer> types, List<String> merges, int endOfText) {
    }

    static Bpe bpeVocab(long seed) {
        List<String> tokens = new ArrayList<>();
        List<Integer> types = new ArrayList<>();
        Set<String> have = new HashSet<>();
        for (int b = 0; b < 256; b++) {
            tokens.add(BYTE_CHAR[b]);
            types.add(GgufTokenizer.NORMAL);
            have.add(BYTE_CHAR[b]);
        }
        List<String> merges = new ArrayList<>();
        Set<String> haveMerge = new HashSet<>();
        // chains: " the" = "Ġ t", "Ġt h", "Ġth e"
        for (String word : new String[]{" the", "the", " and", "ing", "tion", "Hello", " Hello", "12", "123", "1234", "\n\n", "  ", "   ",
                "'s", "'re", "...", "http", "://", "\n ", "def", " def", "return", "    return", "世界", "你好", "\u00e9", "caf\u00e9"}) {
            String e = enc(word);
            List<String> chars = new ArrayList<>();
            for (int i = 0; i < e.length(); ) {
                int cp = e.codePointAt(i);
                chars.add(new String(Character.toChars(cp)));
                i += Character.charCount(cp);
            }
            String acc = chars.get(0);
            for (int i = 1; i < chars.size(); i++) {
                String merge = acc + " " + chars.get(i);
                acc = acc + chars.get(i);
                if (haveMerge.add(merge)) {
                    merges.add(merge);
                }
                if (have.add(acc)) {
                    tokens.add(acc);
                    types.add(GgufTokenizer.NORMAL);
                }
            }
        }
        // tokens without merges: only a pre-tokenizer with ignore_merges can produce them
        for (String t : new String[]{" world", "\n\n\n", "    ", "foo", " foo", "12345", "\u4e2d", "\u4e2d\u6587", "\u00e9\u00e9", " \u00e9"}) {
            String e = enc(t);
            if (have.add(e)) {
                tokens.add(e);
                types.add(GgufTokenizer.NORMAL);
            }
        }
        // every pair of byte symbols, in a pseudo-random order
        List<int[]> pairs = new ArrayList<>(65536);
        for (int a = 0; a < 256; a++) {
            for (int b = 0; b < 256; b++) {
                pairs.add(new int[]{a, b});
            }
        }
        Collections.shuffle(pairs, new Random(seed));
        for (int[] p : pairs) {
            String merge = BYTE_CHAR[p[0]] + " " + BYTE_CHAR[p[1]];
            String joined = BYTE_CHAR[p[0]] + BYTE_CHAR[p[1]];
            if (!haveMerge.add(merge)) {
                continue;
            }
            merges.add(merge);
            if (have.add(joined)) {
                tokens.add(joined);
                types.add(GgufTokenizer.NORMAL);
            }
        }
        int eot = tokens.size();
        for (Object[] sp : new Object[][]{{"<|endoftext|>", GgufTokenizer.CONTROL}, {"<|im_start|>", GgufTokenizer.CONTROL},
                {"<|im_end|>", GgufTokenizer.CONTROL}, {"<think>", GgufTokenizer.USER_DEFINED},
                {"</think>", GgufTokenizer.USER_DEFINED}, {"<unk>", GgufTokenizer.UNKNOWN}, {SEP, GgufTokenizer.CONTROL},
                {"<|fim_prefix|>", GgufTokenizer.NORMAL}, {"</s>", GgufTokenizer.NORMAL}}) {
            tokens.add((String) sp[0]);
            types.add((Integer) sp[1]);
        }
        return new Bpe(tokens, types, merges, eot);
    }

    /** The merge order of the pairs; {@code -Dgguf.tok.seed=N} tries another one (a boundary hides behind some orders). */
    private static final Bpe BPE = bpeVocab(Long.getLong("gguf.tok.seed", 20240611L));

    /** A byte-level BPE vocabulary file with the given {@code tokenizer.ggml.pre} (null: the key is left out). */
    static Path writeBpe(Path dir, String label, String pre, Boolean addBos, String name) throws IOException {
        GgufWriter w = base("gpt2", name == null ? label : name);
        if (pre != null) {
            w.str("tokenizer.ggml.pre", pre);
        }
        w.strings("tokenizer.ggml.tokens", BPE.tokens().toArray(new String[0]));
        w.i32s("tokenizer.ggml.token_type", BPE.types().stream().mapToInt(Integer::intValue).toArray());
        w.strings("tokenizer.ggml.merges", BPE.merges().toArray(new String[0]));
        w.u32("tokenizer.ggml.bos_token_id", BPE.endOfText());
        w.u32("tokenizer.ggml.eos_token_id", BPE.endOfText());
        if (addBos != null) {
            w.bool("tokenizer.ggml.add_bos_token", addBos);
        }
        Path p = dir.resolve("synth-" + label + ".gguf");
        w.write(p);
        return p;
    }

    private static GgufWriter base(String model, String name) {
        return new GgufWriter().str("general.architecture", "llama").str("general.name", name)
                .u32("llama.block_count", 32).u32("llama.context_length", 4096).u32("llama.embedding_length", 4096)
                .u32("llama.feed_forward_length", 11008).u32("llama.attention.head_count", 32)
                .u32("llama.attention.head_count_kv", 32).f32("llama.attention.layer_norm_rms_epsilon", 1e-5f)
                .u32("llama.rope.dimension_count", 128).str("tokenizer.ggml.model", model);
    }

    // ------------------------------------------------------------------ SentencePiece vocabularies

    enum Spm {
        /** Llama-2-like: <unk>, <s>, </s>, byte tokens, distinct scores, a space prefix. */
        LLAMA("spm-llama", true, false, false),
        /** Equal scores in small steps, so many pairs tie. */
        TIES("spm-ties", true, false, false),
        /** Gemma-like: no space prefix, all scores 0, newline runs, user-defined turn tokens, control <bos>/<eos>/<pad>. */
        GEMMA("spm-gemma", false, true, false),
        /** Phi-3-like: <unk>/<s> control, </s> and the chat markers user-defined, the model name "Phi3". */
        PHI3("spm-phi3", true, false, true);

        final String label;
        final boolean spacePrefix, gemma, phi3;

        Spm(String label, boolean spacePrefix, boolean gemma, boolean phi3) {
            this.label = label;
            this.spacePrefix = spacePrefix;
            this.gemma = gemma;
            this.phi3 = phi3;
        }
    }

    static Path writeSpm(Path dir, Spm kind) throws IOException {
        return writeSpm(dir, kind, kind.label, null);
    }

    /** @param addBos the value of tokenizer.ggml.add_bos_token (null: the key is left out, llama.cpp defaults to true) */
    static Path writeSpm(Path dir, Spm kind, String fileLabel, Boolean addBos) throws IOException {
        List<String> tokens = new ArrayList<>();
        List<Integer> types = new ArrayList<>();
        List<Float> scores = new ArrayList<>();
        Random rnd = new Random(kind.ordinal() * 1000L + 5);
        int bos;
        int eos;
        int unk;
        if (kind.gemma) {
            for (Object[] t : new Object[][]{{"<pad>", 3}, {"<eos>", 3}, {"<bos>", 3}, {"<unk>", 2}}) {
                tokens.add((String) t[0]);
                types.add((Integer) t[1]);
                scores.add(0f);
            }
            bos = 2;
            eos = 1;
            unk = 3;
            for (String t : new String[]{"<start_of_turn>", "<end_of_turn>"}) {
                tokens.add(t);
                types.add(GgufTokenizer.USER_DEFINED);
                scores.add(0f);
            }
            for (int i = 1; i <= 6; i++) { // runs of newlines and of spaces, as Gemma defines them
                tokens.add("\n".repeat(i));
                types.add(GgufTokenizer.NORMAL);
                scores.add(0f);
                if (i >= 2) {
                    tokens.add("\u2581".repeat(i));
                    types.add(GgufTokenizer.NORMAL);
                    scores.add(0f);
                }
            }
        } else if (kind.phi3) {
            tokens.addAll(List.of("<unk>", "<s>", "</s>", "<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"));
            types.addAll(List.of(3, 3, 4, 3, 4, 4, 4, 4));
            for (int i = 0; i < 8; i++) {
                scores.add(-1000f);
            }
            bos = 1;
            eos = 3;
            unk = 0;
        } else {
            tokens.addAll(List.of("<unk>", "<s>", "</s>"));
            types.addAll(List.of(2, 3, 3));
            scores.addAll(List.of(0f, 0f, 0f));
            bos = 1;
            eos = 2;
            unk = 0;
        }
        Set<String> have = new HashSet<>(tokens);
        for (int b = 0; b < 256; b++) {
            String t = String.format("<0x%02X>", b);
            tokens.add(t);
            types.add(GgufTokenizer.BYTE);
            scores.add(0f);
            have.add(t);
        }
        int pieceStart = tokens.size();
        // single characters
        String singles = "abcdefghijklmnopqrstuvwxyz0123456789.,!?'\"-:;()/_<>|*=+\u2581ABCXYZ";
        List<String> chars = new ArrayList<>();
        for (int i = 0; i < singles.length(); i++) {
            chars.add(String.valueOf(singles.charAt(i)));
        }
        chars.add("\u00e9");
        chars.add("\u4e2d");
        if (!kind.gemma) {
            // Llama 2 has no "\n" token, only <0x0A>
        } else {
            chars.add("\n");
        }
        for (String c : chars) {
            if (have.add(c)) {
                tokens.add(c);
                types.add(GgufTokenizer.NORMAL);
                scores.add(score(kind, rnd, 0));
            }
        }
        // pairs, triples and longer pieces over a small alphabet, with scores that decide the merge order
        List<String> alpha = new ArrayList<>();
        for (char c = 'a'; c <= 'z'; c++) {
            alpha.add(String.valueOf(c));
        }
        alpha.addAll(List.of("\u2581", ".", ",", "1", "2", "3"));
        List<String> pieces = new ArrayList<>();
        for (String a : alpha) {
            for (String b : alpha) {
                pieces.add(a + b);
            }
        }
        List<String> next = new ArrayList<>(pieces);
        for (int round = 0; round < 3; round++) {
            List<String> longer = new ArrayList<>();
            for (int i = 0; i < 400; i++) {
                String base = next.get(rnd.nextInt(next.size()));
                String t = base + alpha.get(rnd.nextInt(alpha.size()));
                longer.add(rnd.nextBoolean() ? t : alpha.get(rnd.nextInt(alpha.size())) + base);
            }
            pieces.addAll(longer);
            next = longer;
        }
        for (String w : new String[]{"\u2581the", "\u2581and", "\u2581of", "ing", "tion", "\u2581hello", "\u2581world", "hello", "world",
                "\u2581a", "\u2581i", "\u2581it", "'s", "\u2581\u2581", "\u2581\u2581\u2581\u2581", "\u00e9\u00e9", "\u2581caf\u00e9", "<0x0A>x"}) {
            pieces.add(w);
            for (int cut = 1; cut < w.length(); cut++) {
                pieces.add(w.substring(0, cut));
            }
        }
        for (String p : pieces) {
            if (have.add(p)) {
                tokens.add(p);
                types.add(GgufTokenizer.NORMAL);
                scores.add(score(kind, rnd, p.length()));
            }
        }
        // the separator the comparison cuts the output at: a control token at the end of the vocabulary
        tokens.add(SEP);
        types.add(GgufTokenizer.CONTROL);
        scores.add(0f);
        GgufWriter w = base("llama", kind.phi3 ? "Phi3" : kind.label);
        w.str("tokenizer.ggml.pre", "default");
        w.strings("tokenizer.ggml.tokens", tokens.toArray(new String[0]));
        float[] sc = new float[scores.size()];
        for (int i = 0; i < sc.length; i++) {
            sc[i] = scores.get(i);
        }
        w.f32s("tokenizer.ggml.scores", sc);
        w.i32s("tokenizer.ggml.token_type", types.stream().mapToInt(Integer::intValue).toArray());
        w.u32("tokenizer.ggml.bos_token_id", bos);
        w.u32("tokenizer.ggml.eos_token_id", eos);
        w.u32("tokenizer.ggml.unknown_token_id", unk);
        if (addBos != null || fileLabel.equals(kind.label)) {
            w.bool("tokenizer.ggml.add_bos_token", addBos == null || addBos);
        }
        w.bool("tokenizer.ggml.add_eos_token", false);
        if (!kind.spacePrefix) {
            w.bool("tokenizer.ggml.add_space_prefix", false);
        }
        Path p = dir.resolve("synth-" + fileLabel + ".gguf");
        w.write(p);
        return p;
    }

    private static float score(Spm kind, Random rnd, int len) {
        return switch (kind) {
            case GEMMA -> 0f;
            case TIES -> -(float) rnd.nextInt(6);
            default -> -(float) (rnd.nextInt(1_000_000) / 1000.0 + 1);
        };
    }

    // ------------------------------------------------------------------ corpus

    /** Hand-written strings: what a tokenizer meets, and the corners of the regexes. */
    static List<String> handWritten() {
        List<String> c = new ArrayList<>(List.of(
                "Hello, world!", "Hello world", " Hello world", "  Hello  world  ", "hello", "HELLO", "Hello",
                "I'm sure it's the dog's, they're here; we've said I'd go, you'll see",
                "HE'S THEY'RE WE'VE I'M YOU'LL I'D DON'T", "He'S They'Re We'Ve I'M You'Ll I'D", "don't can't won't isn't",
                "'s 're 've 'm 'll 'd 't", "x's y're z've", "''s '''re", "rock 'n' roll", "o'clock", "\u2019s \u2019re it\u2019s",
                "1234567", "12345678901234567890", "3.14159", "007", "12 345 678", "1,000,000", "-42", "+3", "0x1F", "1e-5", "a1b22c333d4444",
                "\u0663\u0664\u0665", "x\u00b2+y\u00b3", "\u00bd \u00bc", "\u2460\u2461\u2462", "\u2167", "\uff11\uff12\uff13", "1\u00b22",
                "\u4f60\u597d\uff0c\u4e16\u754c\uff01", "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c", "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8", "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059\u3002",
                "\u4e2d\u6587English\u6df7\u5408123", "\u4e2d\u6587", " \u4e2d\u6587", "\u4e16\u754c\u4f60\u597d", "\u30ab\u30bf\u30ab\u30ca\u3072\u3089\u304c\u306a\u6f22\u5b57",
                "caf\u00e9 r\u00e9sum\u00e9 na\u00efve", "cafe\u0301", "A\u030a \u00c5", "\u00fcber stra\u00dfe", "\u0141\u00f3d\u017a", "\u0130stanbul",
                "\ud83d\ude00 \ud83c\udf89", "\ud83d\udc68\u200d\ud83d\udc69\u200d\ud83d\udc67", "\ud83c\uddfa\ud83c\uddf8", "\u2764\ufe0f", "\ud83d\udc4d\ud83c\udffd",
                "a  b", "a   b", "a    b", "  leading", "trailing  ", "tab\there", "\t\t", "   ", " ", "\t", "\n", "\n\n", "\n\n\n", "\n\n\n\n\n",
                "a\nb", "a \n b", "a\n b", "a \nb", "line1\r\nline2", "line1\r\n\r\nline2", "\r", "\r\n", "\r\n\r\n", "\n\r", " \n", "\n ", "  \n  ",
                "a\n\n\nb", "text\n\n  \n text", "a\u00a0b", "a\u3000b", "\u2003em space", "\u200bzero width", "a\u2028b", "a\u2029b",
                "a\u0085b", "\u00a0\u00a0", "a\u000bb\fc", "a\u001cb", "\u2000\u2001\u2002", " \u3000 ",
                "def f(x):\n    return x**2 + 1  # comment", "int main(void) { printf(\"%d\\n\", 1); }", "for (i=0;i<n;i++){a[i]=b[i]+c[i];}",
                "#include <stdio.h>", "https://example.com/path?q=1&r=2#frag", "user@example.com", "www.example.co.uk",
                "snake_case_name camelCaseName PascalCaseName HTTPServer XMLHttpRequest iPhone", "ALLCAPS lowercase MiXeD",
                "ThisIsATest", "aBc", "ABc", "AbC", "\u00c9COLE \u00e9cole", "foo.bar(baz)[0]->qux", "a+b=c; d*e/f-g", "!!!???...", "$$$", "```python\nprint('hi')\n```",
                "<div class=\"x\">&amp;</div>", "{\"key\": [1, 2.5, null, true]}", "C:\\Users\\name\\file.txt", "~/.bashrc", "100%", "50% off!",
                "(a)(b)[c]{d}", "a-b_c.d,e", "\u2014em\u2013dash\u201cquotes\u201d\u2018single\u2019", "\u2026ellipsis\u2026", "\u00bfQu\u00e9 tal? \u00a1Hola!",
                "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd",
                "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01", "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e", "\u0915\u094d\u0937\u093f", "\u0e01\u0e33",
                "\u20ac100 \u00a350 \u00a5200 \u00a9 \u2192 \u2211 \u221e", "a\u0301\u0308b", "\u0301", "\u0308a", "x\u200dy", "x\u200cy", "\u00ad", "a\u00adb", "\ufeffBOM",
                "\u0001\u0002", "\u007f", "a\u0000b".replace('\u0000', 'X'), "~~~ ``` ^^^ ||| <<< >>> $$ ++ == ",
                "<|endoftext|>", "a<|im_start|>user\nhi<|im_end|>\n", "<think>reasoning</think>", " <|im_end|> ", "<|im_start|><|im_start|>", "x<|endoftext|>y",
                "<unk>", "<|endoftext|", "<|end", "<s>", "</s>", "<|fim_prefix|>x", "<|im_end|>\n", "\n<|im_end|>", "  <think>  x  </think>  ",
                "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n", "<|user|>\nHi<|end|>\n<|assistant|>\n", "<|user|>  \n  Hi  <|end|>  x",
                "<s> hello </s>", "<s>hello</s><s>", "<0x0A>", "<0x41>b", "a<0x41>", "\n\n\n\n\n\n\n\n\n",
                "The quick brown fox jumps over the lazy dog.", "the the the the", " the  the   the", "worldworld world", "foo foo  foo",
                "A", "a", "", "?", "\u00e9", "\u4e2d"));
        return c;
    }

    private static final String[][] ALPHABETS = {
            // ASCII letters of both cases
            {"a", "b", "c", "e", "h", "i", "l", "n", "o", "r", "s", "t", "A", "B", "C", "E", "H", "S", "T", "X", "Z"},
            // digits
            {"0", "1", "2", "3", "5", "7", "9"},
            // whitespace
            {" ", " ", " ", " ", "\n", "\t", "\r", "\r\n", "\n\n", "  "},
            // ASCII punctuation and symbols
            {".", ",", "!", "?", "'", "\"", "-", "_", ":", ";", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "~", "`"},
            // contractions
            {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'RE", "'LL"},
            // non-ASCII letters, marks, digits, symbols, spaces, CJK
            {"\u00e9", "\u00f1", "\u00fc", "\u00df", "\u00c5", "\u0141", "\u0416", "\u03a9", "\u05d0", "\u0627", "\u0e01", "\u0915", "\u4e2d", "\u6587", "\u3042", "\u30ab", "\ud55c",
                    "\u0301", "\u0308", "\u200d", "\u0663", "\u00b2", "\u00bd", "\u2460", "\u0966", "\u20ac", "\u00a9", "\u2192", "\u2211", "\uff01", "\u3001", "\u3002", "\u201c", "\u2019", "\u2014",
                    "\u00a0", "\u3000", "\u2003", "\u2028", "\u200b", "\u00ad", "\ufeff", "\u0085", "\u0001", "\u007f", "\ud83d\ude00", "\ud83c\udf89", "\ud83d\udc4d", "\ud835\udc00", "\ud840\udc00"},
    };

    static List<String> randoms(int n, long seed) {
        Random r = new Random(seed);
        List<String> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            StringBuilder sb = new StringBuilder();
            int len = 1 + r.nextInt(i % 7 == 0 ? 40 : 12);
            // each string draws from two or three alphabets so that the transitions between classes are what varies
            int[] pick = {r.nextInt(ALPHABETS.length), r.nextInt(ALPHABETS.length), r.nextInt(ALPHABETS.length)};
            for (int k = 0; k < len; k++) {
                String[] a = ALPHABETS[pick[r.nextInt(r.nextInt(4) == 0 ? 3 : 2)]];
                sb.append(a[r.nextInt(a.length)]);
            }
            String s = sb.toString();
            if (!s.contains("<|")) {
                out.add(s);
            }
        }
        return out;
    }

    /** Every code point from {@code from} to {@code to} (step {@code step}) in four contexts: after a letter, between a space and a letter, before a digit, doubled. */
    static List<String> sweep(int from, int to, int step) throws IOException {
        List<String> out = new ArrayList<>();
        int[][] undefined = llamaUndefined();
        for (int cp = from; cp < to; cp += step) {
            if (cp >= 0xD800 && cp <= 0xDFFF || cp == 0) {
                continue;
            }
            // llama.cpp's Unicode tables are older than Java's: a character assigned since has no category there
            // (a known difference, see docs/16-gguf.md 16.12); where Java has none either the two agree and it is tested
            if (Character.getType(cp) != Character.UNASSIGNED && isUndefined(undefined, cp)) {
                continue;
            }
            String c = new String(Character.toChars(cp));
            out.add("a" + c);
            out.add(" " + c + "b");
            out.add(c + "1");
            out.add(c + c + " ");
        }
        return out;
    }

    /** The code points llama.cpp b11669 has no category for (from its unicode-data.cpp): src/test/resources/gguf/llama-cpp-undefined.txt. */
    static int[][] llamaUndefined() throws IOException {
        List<int[]> r = new ArrayList<>();
        try (var in = GgufTokenizerLlamaCpp.class.getResourceAsStream("/gguf/llama-cpp-undefined.txt")) {
            for (String line : new String(in.readAllBytes(), StandardCharsets.UTF_8).split("\n")) {
                if (!line.isBlank() && line.charAt(0) != '#') {
                    String[] ab = line.trim().split("-");
                    r.add(new int[]{Integer.parseInt(ab[0], 16), Integer.parseInt(ab[1], 16)});
                }
            }
        }
        return r.toArray(new int[0][]);
    }

    static boolean isUndefined(int[][] ranges, int cp) {
        int lo = 0;
        int hi = ranges.length - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (cp < ranges[mid][0]) {
                hi = mid - 1;
            } else if (cp > ranges[mid][1]) {
                lo = mid + 1;
            } else {
                return true;
            }
        }
        return false;
    }

    static List<String> corpus(int randoms) {
        List<String> c = new ArrayList<>(handWritten());
        c.addAll(randoms(randoms, 12345));
        return c;
    }

    // ------------------------------------------------------------------ running both tokenizers

    /** The ids of {@code strings}, as one text joined by the separator token, split at the separator id. */
    static List<int[]> split(int[] ids, int sepId) {
        List<int[]> out = new ArrayList<>();
        int from = 0;
        for (int i = 0; i < ids.length; i++) {
            if (ids[i] == sepId) {
                out.add(Arrays.copyOfRange(ids, from, i));
                from = i + 1;
            }
        }
        out.add(Arrays.copyOfRange(ids, from, ids.length));
        return out;
    }

    static String joined(List<String> strings) {
        return String.join(SEP, strings);
    }

    static Tokenizer load(Path gguf) throws IOException {
        try (GgufFile f = GgufFile.open(gguf)) {
            return GgufModel.tokenizer(f);
        }
    }

    static String findLlamaTokenize() {
        String env = System.getenv("LLAMA_TOKENIZE");
        if (env != null && !env.isEmpty() && Files.isExecutable(Path.of(env))) {
            return env;
        }
        String path = System.getenv("PATH");
        if (path != null) {
            for (String d : path.split(":")) {
                Path p = Path.of(d, "llama-tokenize");
                if (Files.isExecutable(p)) {
                    return p.toString();
                }
            }
        }
        return Files.isExecutable(Path.of("/opt/homebrew/bin/llama-tokenize")) ? "/opt/homebrew/bin/llama-tokenize" : null;
    }

    /** llama-tokenize's ids for the text file; null with a message on failure. */
    static int[] llamaIds(String exe, Path gguf, Path textFile, boolean bos) throws IOException, InterruptedException {
        List<String> cmd = new ArrayList<>(List.of(exe, "-m", gguf.toString(), "--ids", "--no-escape", "--log-disable", "-f", textFile.toString()));
        if (!bos) {
            cmd.add("--no-bos");
        }
        Process p = new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.DISCARD).start();
        String out;
        try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            char[] buf = new char[1 << 16];
            for (int n; (n = r.read(buf)) > 0; ) {
                sb.append(buf, 0, n);
            }
            out = sb.toString();
        }
        if (!p.waitFor(120, TimeUnit.SECONDS) || p.exitValue() != 0) {
            return null;
        }
        out = out.trim();
        if (!out.startsWith("[") || !out.endsWith("]")) {
            return null;
        }
        String body = out.substring(1, out.length() - 1).trim();
        if (body.isEmpty()) {
            return new int[0];
        }
        return Arrays.stream(body.split(",")).mapToInt(x -> Integer.parseInt(x.trim())).toArray();
    }

    /** One vocabulary to check. */
    record Case(String label, Path file) {
    }

    static String show(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            i += Character.charCount(cp);
            if (cp == '\n') {
                sb.append("\\n");
            } else if (cp == '\r') {
                sb.append("\\r");
            } else if (cp == '\t') {
                sb.append("\\t");
            } else if (cp < 32 || cp == 127 || (cp >= 0x80 && cp < 0xA1) || Character.isWhitespace(cp) && cp != ' ' || cp == 0xa0 || cp == 0x200b
                    || cp == 0xad || cp == 0xfeff || Character.getType(cp) == Character.FORMAT || Character.getType(cp) == Character.NON_SPACING_MARK) {
                sb.append(String.format("\\u{%04x}", cp));
            } else {
                sb.appendCodePoint(cp);
            }
        }
        return sb.append('"').toString();
    }

    /** All the vocabularies: every pre value (and the absent key), the SentencePiece kinds. */
    static List<Case> writeAll(Path dir) throws IOException {
        Files.createDirectories(dir);
        List<Case> cases = new ArrayList<>();
        cases.add(new Case("pre=(absent)", writeBpe(dir, "none", null, null, null)));
        for (String pre : GgufTokenizer.knownPre()) {
            cases.add(new Case("pre=" + pre, writeBpe(dir, "pre-" + pre.replaceAll("[^A-Za-z0-9_-]", "_"), pre, null, null)));
        }
        for (Spm s : Spm.values()) {
            cases.add(new Case(s.label, writeSpm(dir, s)));
        }
        return cases;
    }

    /** Strings per llama-tokenize call: its handling of special tokens is quadratic in their number. */
    static final int CHUNK = 1500;

    /** @return the number of strings that differ */
    static int compare(String exe, Case c, List<String> all, Path tmp, int showMax, StringBuilder log) throws Exception {
        Tokenizer tok = load(c.file());
        int sep = tok.tokenId(SEP);
        int bad = 0;
        for (int from = 0; from < all.size(); from += CHUNK) {
            List<String> strings = all.subList(from, Math.min(all.size(), from + CHUNK));
            Path text = tmp.resolve("corpus-" + c.file().getFileName() + ".txt");
            Files.writeString(text, joined(strings), StandardCharsets.UTF_8);
            int[] want = llamaIds(exe, c.file(), text, false);
            if (want == null) {
                log.append("   llama-tokenize failed on ").append(c.file().getFileName()).append('\n');
                bad += strings.size();
                continue;
            }
            List<int[]> w = split(want, sep);
            List<int[]> g = split(tok.encode(joined(strings), true), sep);
            if (w.size() != strings.size() || g.size() != strings.size()) {
                log.append("   separator count: llama.cpp ").append(w.size()).append(" ours ").append(g.size()).append(" expected ")
                        .append(strings.size()).append('\n');
            }
            for (int i = 0; i < strings.size(); i++) {
                int[] a = i < w.size() ? w.get(i) : new int[0];
                int[] b = i < g.size() ? g.get(i) : new int[0];
                if (!Arrays.equals(a, b)) {
                    if (bad++ < showMax) {
                        log.append("   ").append(show(strings.get(i))).append("\n      llama.cpp ").append(Arrays.toString(a))
                                .append("\n      ours      ").append(Arrays.toString(b)).append('\n');
                    }
                }
            }
        }
        return bad;
    }

    /**
     * The BOS rule: what llama-tokenize prints without {@code --no-bos} against {@code promptBos()} + our ids. The
     * vocabularies leave tokenizer.ggml.add_bos_token out where llama.cpp has a default for the pre-tokenizer.
     */
    static int checkBos(String exe, Path dir, StringBuilder log) throws Exception {
        record Bos(String label, String pre, Boolean addBos, Spm spm) {
        }
        List<Bos> cases = List.of(new Bos("bos: llama-bpe, key absent", "llama-bpe", null, null),
                new Bos("bos: llama-bpe, add_bos_token=false", "llama-bpe", false, null),
                new Bos("bos: tekken, key absent", "tekken", null, null),
                new Bos("bos: gpt-2, key absent", "gpt-2", null, null),
                new Bos("bos: gpt-2, add_bos_token=true", "gpt-2", true, null),
                new Bos("bos: qwen2, key absent", "qwen2", null, null),
                new Bos("bos: chatglm-bpe, key absent", "chatglm-bpe", null, null),
                new Bos("bos: glm4, add_bos_token=true", "glm4", true, null),
                new Bos("bos: (no pre), key absent", null, null, null),
                new Bos("bos: spm-llama, key absent", null, null, Spm.LLAMA),
                new Bos("bos: spm-llama, add_bos_token=false", null, false, Spm.LLAMA),
                new Bos("bos: spm-gemma, add_bos_token=true", null, true, Spm.GEMMA),
                new Bos("bos: spm-phi3, key absent", null, null, Spm.PHI3));
        int failed = 0;
        for (Bos b : cases) {
            String file = "bos" + Math.abs(b.label().hashCode());
            Path gguf = b.spm() != null ? writeSpm(dir, b.spm(), file, b.addBos()) : writeBpe(dir, file, b.pre(), b.addBos(), null);
            Tokenizer tok = load(gguf);
            int bad = 0;
            for (String text : new String[]{"Hello world", "<|endoftext|>Hello", " x\n", ""}) {
                Path t = dir.resolve("bos.txt");
                Files.writeString(t, text, StandardCharsets.UTF_8);
                int[] want = llamaIds(exe, gguf, t, true);
                int[] body = tok.encode(text, true);
                int bos = tok.promptBos();
                int[] got = new int[body.length + (bos >= 0 ? 1 : 0)];
                if (bos >= 0) {
                    got[0] = bos;
                }
                System.arraycopy(body, 0, got, bos >= 0 ? 1 : 0, body.length);
                if (!Arrays.equals(want, got)) {
                    bad++;
                    log.append("   ").append(b.label()).append(' ').append(show(text)).append("\n      llama.cpp ")
                            .append(Arrays.toString(want)).append("\n      ours      ").append(Arrays.toString(got)).append('\n');
                }
            }
            System.out.printf("%-34s %s%n", b.label(), bad == 0 ? "PASS" : "FAIL " + bad + " of 4");
            if (bad > 0) {
                failed++;
            }
            Files.deleteIfExists(gguf);
        }
        return failed;
    }

    // ------------------------------------------------------------------ recorded ids (no llama.cpp needed)

    /** Labels whose stored ids the in-repo test checks: one per distinct (regexes, flags) group, and the SentencePiece kinds. */
    static List<Case> recordedCases(Path dir) throws IOException {
        Map<String, Case> seen = new LinkedHashMap<>();
        List<Case> all = writeAll(dir);
        for (Case c : all) {
            String key = c.label().startsWith("pre=") || c.label().equals("pre=(absent)") ? groupKey(c.label().substring(4)) : c.label();
            seen.putIfAbsent(key, c);
        }
        return new ArrayList<>(seen.values());
    }

    private static String groupKey(String pre) {
        try {
            var m = Class.forName("org.robomiri.jllm.tokenizer.GgufPre").getDeclaredMethod("get", String.class);
            m.setAccessible(true);
            Object e = m.invoke(null, pre.equals("(absent)") ? null : pre);
            var rx = e.getClass().getDeclaredMethod("regexes");
            rx.setAccessible(true);
            var im = e.getClass().getDeclaredMethod("ignoreMerges");
            im.setAccessible(true);
            return Arrays.toString((String[]) rx.invoke(e)) + im.invoke(e);
        } catch (ReflectiveOperationException ex) {
            return pre;
        }
    }

    static final int RECORDED_RANDOMS = 120;

    static void record(Path dir, Path file) throws Exception {
        String exe = findLlamaTokenize();
        if (exe == null) {
            throw new IllegalStateException("llama-tokenize is not installed");
        }
        List<String> strings = corpus(RECORDED_RANDOMS);
        Path tmp = Files.createTempDirectory(dir, "rec");
        try (var out = new java.io.PrintWriter(new GZIPOutputStream(Files.newOutputStream(file)), false, StandardCharsets.UTF_8)) {
            for (Case c : recordedCases(dir)) {
                Tokenizer tok = load(c.file());
                Path text = tmp.resolve("t.txt");
                Files.writeString(text, joined(strings), StandardCharsets.UTF_8);
                int[] ids = llamaIds(exe, c.file(), text, false);
                List<int[]> parts = split(ids, tok.tokenId(SEP));
                out.print("@" + c.label() + "\n");
                for (int[] p : parts) {
                    out.print(Arrays.toString(p).replace("[", "").replace("]", "").replace(", ", " ") + "\n");
                }
            }
        }
    }

    /** Compares our ids with the stored ones; returns {passed, failed} string counts. */
    static int[] verifyRecorded(Path file, Path tmp, StringBuilder log) throws Exception {
        List<String> strings = corpus(RECORDED_RANDOMS);
        Map<String, List<int[]>> stored = new LinkedHashMap<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(file)), StandardCharsets.UTF_8))) {
            String line;
            List<int[]> cur = null;
            while ((line = r.readLine()) != null) {
                if (line.startsWith("@")) {
                    cur = new ArrayList<>();
                    stored.put(line.substring(1), cur);
                } else {
                    cur.add(line.isBlank() ? new int[0] : Arrays.stream(line.trim().split(" ")).mapToInt(Integer::parseInt).toArray());
                }
            }
        }
        int pass = 0;
        int fail = 0;
        for (Case c : recordedCases(tmp)) {
            List<int[]> want = stored.get(c.label());
            if (want == null) {
                log.append("   no stored ids for ").append(c.label()).append('\n');
                fail += strings.size();
                continue;
            }
            Tokenizer tok = load(c.file());
            List<int[]> got = split(tok.encode(joined(strings), true), tok.tokenId(SEP));
            int bad = 0;
            for (int i = 0; i < strings.size(); i++) {
                if (i < want.size() && i < got.size() && Arrays.equals(want.get(i), got.get(i))) {
                    pass++;
                } else {
                    fail++;
                    if (bad++ < 3) {
                        log.append("   ").append(c.label()).append(' ').append(show(strings.get(i))).append("\n      llama.cpp ")
                                .append(i < want.size() ? Arrays.toString(want.get(i)) : "?").append("\n      ours      ")
                                .append(i < got.size() ? Arrays.toString(got.get(i)) : "?").append('\n');
                    }
                }
            }
            Files.deleteIfExists(c.file());
        }
        return new int[]{pass, fail, stored.size()};
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) throws Exception {
        String cmd = args.length > 0 ? args[0] : "check";
        switch (cmd) {
            case "gen" -> {
                Path dir = Path.of(args[1]);
                List<Case> cases = writeAll(dir);
                Files.writeString(dir.resolve("corpus.txt"), joined(corpus(2000)), StandardCharsets.UTF_8);
                System.out.println(cases.size() + " vocabularies and corpus.txt written to " + dir);
            }
            case "record" -> {
                record(Files.createTempDirectory("gguf-tok-rec"), Path.of(args[1]));
                System.out.println("recorded " + args[1]);
            }
            case "verify" -> {
                StringBuilder log = new StringBuilder();
                int[] r = verifyRecorded(Path.of(args[1]), Files.createTempDirectory("gguf-tok-ver"), log);
                System.out.print(log);
                System.out.println(r[0] + " of " + (r[0] + r[1]) + " strings give the stored llama.cpp ids (" + r[2] + " vocabularies)");
                System.exit(r[1] == 0 ? 0 : 1);
            }
            case "stress" -> System.exit(stress(Path.of(args[1])));
            case "check" -> System.exit(check(args));
            default -> throw new IllegalArgumentException(cmd);
        }
    }

    /** Long runs of one kind of character through every pre-tokenizer: no StackOverflowError, no quadratic time. */
    static int stress(Path dir) throws Exception {
        String[] inputs = {"a".repeat(100_000), " ".repeat(100_000), "1".repeat(100_000), "ab ".repeat(30_000), "\n".repeat(100_000),
                "\u00e9".repeat(50_000), "\u4e2d".repeat(50_000), "'s".repeat(30_000), " \n".repeat(40_000), "Aa".repeat(50_000),
                ".".repeat(100_000), "a\u0301".repeat(40_000), "Hello, world! 123 ".repeat(8_000), "\ud83d\ude00".repeat(30_000),
                "a b".repeat(30_000) + " ".repeat(50_000) + "x"};
        int failed = 0;
        for (Case c : writeAll(dir)) {
            if (!c.label().startsWith("pre=") && !c.label().startsWith("spm")) {
                continue;
            }
            Tokenizer tok = load(c.file());
            long worst = 0;
            String worstIn = "";
            for (String in : inputs) {
                long t0 = System.nanoTime();
                try {
                    tok.encode(in, true);
                } catch (Throwable e) {
                    System.out.println(c.label() + " FAIL on " + show(in.substring(0, 6)) + "...(" + in.length() + "): " + e);
                    failed++;
                    continue;
                }
                long ms = (System.nanoTime() - t0) / 1_000_000;
                if (ms > worst) {
                    worst = ms;
                    worstIn = show(in.substring(0, Math.min(6, in.length())));
                }
            }
            System.out.printf("%-34s slowest %5d ms (%s...)%n", c.label(), worst, worstIn);
            Files.deleteIfExists(c.file());
        }
        return failed == 0 ? 0 : 1;
    }

    /** check [--dir DIR] [--randoms N] [--only SUBSTRING] [--show N]; returns the exit code. */
    static int check(String[] args) throws Exception {
        String exe = findLlamaTokenize();
        if (exe == null) {
            System.out.println("llama-tokenize is not installed (set LLAMA_TOKENIZE or put it on the PATH): skipped");
            return 0;
        }
        Path dir = null;
        int sweepStep = 61;
        int sweepFrom = 0x20;
        int sweepTo = 0x30000;
        int randoms = 1500;
        String only = null;
        int show = 4;
        for (int i = 1; i + 1 < args.length; i += 2) {
            switch (args[i]) {
                case "--dir" -> dir = Path.of(args[i + 1]);
                case "--randoms" -> randoms = Integer.parseInt(args[i + 1]);
                case "--only" -> only = args[i + 1];
                case "--show" -> show = Integer.parseInt(args[i + 1]);
                case "--sweep" -> sweepStep = Integer.parseInt(args[i + 1]);
                case "--sweep-from" -> sweepFrom = Integer.decode(args[i + 1]);
                case "--sweep-to" -> sweepTo = Integer.decode(args[i + 1]);
                default -> throw new IllegalArgumentException(args[i]);
            }
        }
        Path work = dir != null ? dir : Files.createTempDirectory("gguf-tok-check");
        List<String> strings = corpus(randoms);
        if (sweepStep > 0) {
            strings.addAll(sweep(sweepFrom, sweepTo, sweepStep));
        }
        System.out.println("llama-tokenize: " + exe);
        System.out.println(strings.size() + " strings per vocabulary (" + handWritten().size() + " hand-written, " + randoms + " random)");
        Map<String, int[]> results = new TreeMap<>();
        int failedVocabs = 0;
        for (Case c : writeAll(work)) {
            if (only != null && !c.label().contains(only)) {
                continue;
            }
            StringBuilder log = new StringBuilder();
            int bad = compare(exe, c, strings, work, show, log);
            results.put(c.label(), new int[]{strings.size() - bad, bad});
            System.out.printf("%-34s %s %d/%d%n", c.label(), bad == 0 ? "PASS" : "FAIL", strings.size() - bad, strings.size());
            if (bad > 0) {
                failedVocabs++;
                System.out.print(log);
            }
            Files.deleteIfExists(c.file());
        }
        System.out.println(results.size() - failedVocabs + " of " + results.size() + " vocabularies pass");
        if (only == null || "bos".contains(only)) {
            StringBuilder log = new StringBuilder();
            int bosFailed = checkBos(exe, work, log);
            System.out.print(log);
            failedVocabs += bosFailed;
        }
        return failedVocabs == 0 ? 0 : 1;
    }
}
