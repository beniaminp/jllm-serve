package org.robomiri.jllm.tokenizer;

import org.robomiri.jllm.json.Json;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Builds a {@link Tokenizer} from the tokenizer metadata of a GGUF file ({@code tokenizer.ggml.*}) by writing the
 * {@code tokenizer.json} the same tokenizer has in the Hugging Face ecosystem and loading that, so the encoder is
 * the one all other models use. The reference is llama.cpp's own tokenizer (build b11669): what it produces for a
 * text is what this class must produce, and {@code GgufTokenizerTest} / {@code scripts/gguf-tokenizer-check.sh}
 * compare the two.
 *
 * <ul>
 *   <li>{@code tokenizer.ggml.model = gpt2}: byte-level BPE. The tokens are the vocabulary, {@code merges} the merge
 *       list, {@code token_type} separates control / user-defined / unknown tokens (added tokens: special, resp. not)
 *       from the vocabulary; {@code tokenizer.ggml.pre} selects the pre-tokenizer ({@link GgufPre}: a list of
 *       regular expressions applied one after the other, plus {@code ignore_merges}). A missing pre is llama.cpp's
 *       "default" (GPT-NeoX); a value this build does not know is refused ({@link UnsupportedPreTokenizer}), so the
 *       caller can fetch the original tokenizer.json instead of tokenizing differently.</li>
 *   <li>{@code tokenizer.ggml.model = llama}: SentencePiece (Llama 2, Mistral, Gemma, Phi-3...). The file stores
 *       scores instead of merges; llama.cpp merges the adjacent pair whose concatenation is a token with the best
 *       score, leftmost first, and spells characters that are no token as {@code <0xXX>} bytes ({@link SpmGreedy}).
 *       Spaces become {@code U+2581}; a dummy prefix is added in front of the text and of the text after every
 *       special token when {@code add_space_prefix} (the default) is set.</li>
 *   <li>{@code tokenizer.ggml.model = t5}: SentencePiece unigram (T5-style encoders). Tokens and scores become the
 *       Unigram vocabulary (Viterbi segmentation), {@code tokenizer.ggml.precompiled_charsmap} the
 *       {@code Precompiled} normalizer, {@code remove_extra_whitespaces} a whitespace split and
 *       {@code add_space_prefix} the Metaspace prefix, as in the Hugging Face conversion of such models.</li>
 * </ul>
 *
 * <p>Special tokens in the text are found as llama.cpp finds them: the tokens of type CONTROL, USER_DEFINED and
 * UNKNOWN (plus those it promotes to control by name, {@link #CONTROL_BY_NAME}) are matched literally. For Phi-3
 * ({@code general.name} contains "phi-3" / "phi3") they swallow the whitespace after them (llama.cpp's
 * {@code LLAMA_TOKEN_ATTR_RSTRIP}), except {@code <unk>}, {@code <s>} and {@code <|endoftext|>}.
 */
public final class GgufTokenizer {

    /** {@code tokenizer.ggml.token_type} values. */
    public static final int NORMAL = 1, UNKNOWN = 2, CONTROL = 3, USER_DEFINED = 4, UNUSED = 5, BYTE = 6;

    /**
     * The {@code tokenizer.ggml.pre} values with a pre-tokenizer in this build (llama.cpp's
     * {@code LLAMA_VOCAB_PRE_TYPE_*}, see {@link GgufPre}); a file that names another one is refused.
     */
    public static Set<String> knownPre() {
        return GgufPre.names();
    }

    /**
     * Tokens llama.cpp promotes to control tokens whatever the file's {@code token_type} says (it logs "control-looking
     * token ... was not control-type" and overrides): the end-of-turn / end-of-generation markers and the
     * fill-in-the-middle markers, so that they are matched in the text as single tokens.
     */
    private static final Set<String> CONTROL_BY_NAME = Set.of(
            "<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>", "<EOT>", "_<EOT>",
            "[EOT]", "<｜end▁of▁sentence｜>", "<end_of_utterance>", "<|eom_id|>", "<|return|>",
            "<|call|>", "<|flush|>", "<|calls|>", "</s>", "[EOS]", "<eos>", "<turn|>", "<|tool_response>", "[e~[",
            "<|ifm|im_end|>",
            "<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<｜fim▁begin｜>", "<PRE>", "▁<PRE>",
            "<|code_prefix|>", "<|prefix|>", "<|plamo:fim_prefix|>",
            "<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<｜fim▁hole｜>", "<SUF>", "▁<SUF>",
            "<|code_suffix|>", "<|suffix|>", "<|plamo:fim_suffix|>",
            "<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<｜fim▁end｜>", "<MID>", "▁<MID>",
            "<|code_middle|>", "<|middle|>", "<|plamo:fim_middle|>",
            "<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]",
            "<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>",
            "<|file_sep|>", "<|plamo:file_separator|>");

    /** llama.cpp makes these plain user-defined tokens (matched in the text, not control) for gpt-oss. */
    private static final Set<String> USER_DEFINED_BY_NAME = Set.of("<|channel|>", "<|message|>", "<|start|>", "<|constrain|>");

    /** Thrown when the file names a pre-tokenizer this build has no regex for. */
    public static final class UnsupportedPreTokenizer extends IllegalArgumentException {
        public UnsupportedPreTokenizer(String message) {
            super(message);
        }
    }

    /**
     * The tokenizer metadata of a GGUF file.
     *
     * @param model    {@code tokenizer.ggml.model} ({@code gpt2}, {@code llama})
     * @param pre      {@code tokenizer.ggml.pre}, null if absent
     * @param tokens   {@code tokenizer.ggml.tokens}
     * @param types    {@code tokenizer.ggml.token_type}, null if absent (all normal)
     * @param scores   {@code tokenizer.ggml.scores}, null if absent
     * @param merges   {@code tokenizer.ggml.merges} ("left right" strings), null for SentencePiece
     * @param bosId    {@code tokenizer.ggml.bos_token_id}, -1 if absent
     * @param eosId    {@code tokenizer.ggml.eos_token_id}, -1 if absent
     * @param unkId    {@code tokenizer.ggml.unknown_token_id}, -1 if absent
     * @param addBos   {@code tokenizer.ggml.add_bos_token}, null if absent (then llama.cpp's default for the model:
     *                 SentencePiece and the Llama 3 / Tekken pre-tokenizers add one, other BPE models do not)
     * @param addSpacePrefix {@code tokenizer.ggml.add_space_prefix} (SentencePiece: the dummy prefix), default true
     * @param precompiledCharsmap {@code tokenizer.ggml.precompiled_charsmap} (unigram: the SentencePiece
     *                            normalization table), null if absent
     * @param removeExtraWhitespaces {@code tokenizer.ggml.remove_extra_whitespaces} (unigram), default false
     * @param generalName {@code general.name}, null if absent (llama.cpp keys Phi-3's whitespace stripping on it)
     * @param addEos   {@code tokenizer.ggml.add_eos_token}, null if absent
     */
    public record Spec(String model, String pre, String[] tokens, long[] types, double[] scores, String[] merges,
                       int bosId, int eosId, int unkId, Boolean addBos, boolean addSpacePrefix,
                       byte[] precompiledCharsmap, boolean removeExtraWhitespaces, String generalName, Boolean addEos) {

        /** A spec without the name and add_eos keys. */
        public Spec(String model, String pre, String[] tokens, long[] types, double[] scores, String[] merges,
                    int bosId, int eosId, int unkId, Boolean addBos, boolean addSpacePrefix,
                    byte[] precompiledCharsmap, boolean removeExtraWhitespaces) {
            this(model, pre, tokens, types, scores, merges, bosId, eosId, unkId, addBos, addSpacePrefix,
                    precompiledCharsmap, removeExtraWhitespaces, null, null);
        }

        /** A spec without the unigram-only keys. */
        public Spec(String model, String pre, String[] tokens, long[] types, double[] scores, String[] merges,
                    int bosId, int eosId, int unkId, Boolean addBos, boolean addSpacePrefix) {
            this(model, pre, tokens, types, scores, merges, bosId, eosId, unkId, addBos, addSpacePrefix, null, false);
        }
    }

    private GgufTokenizer() {
    }

    public static Tokenizer build(Spec s) {
        if (s.tokens() == null || s.tokens().length == 0) {
            throw new IllegalArgumentException("the GGUF file has no tokenizer.ggml.tokens");
        }
        return switch (s.model() == null ? "" : s.model()) {
            case "gpt2" -> byteLevel(s);
            case "llama" -> sentencePiece(s);
            case "t5" -> unigram(s);
            default -> throw new IllegalArgumentException("unsupported tokenizer.ggml.model '" + s.model()
                    + "' (supported: gpt2, llama, t5)");
        };
    }

    private static int type(Spec s, int id) {
        return s.types() == null || id >= s.types().length ? NORMAL : (int) s.types()[id];
    }

    private static String name(Spec s, int id) {
        return id >= 0 && id < s.tokens().length ? s.tokens()[id] : null;
    }

    private enum Kind { VOCAB, SPECIAL, USER }

    /** What llama.cpp makes of the token: found in the text as a control token, as a user-defined one, or not at all. */
    private static Kind kind(Spec s, int id) {
        String t = s.tokens()[id];
        Kind k = switch (type(s, id)) {
            case CONTROL, UNKNOWN -> Kind.SPECIAL;
            case USER_DEFINED -> Kind.USER;
            default -> Kind.VOCAB;
        };
        if (CONTROL_BY_NAME.contains(t)) {
            k = Kind.SPECIAL;
        }
        if (USER_DEFINED_BY_NAME.contains(t)) {
            k = Kind.USER;
        }
        return k;
    }

    /** Phi-3 files: the special tokens swallow the whitespace after them ({@code LLAMA_TOKEN_ATTR_RSTRIP}). */
    private static boolean rstrips(Spec s, int id, Kind k) {
        if (k == Kind.VOCAB || s.generalName() == null) {
            return false;
        }
        String n = s.generalName().toLowerCase(Locale.ROOT);
        if (!n.contains("phi-3") && !n.contains("phi3")) {
            return false;
        }
        String t = s.tokens()[id];
        return !t.equals("<unk>") && !t.equals("<s>") && !t.equals("<|endoftext|>");
    }

    /**
     * Whether the model adds a BOS to a prompt when the file does not say: SentencePiece does, byte-level BPE only for
     * the pre-tokenizers of the Llama 3 and Tekken families (llama.cpp's {@code add_bos = true} defaults).
     */
    private static boolean defaultAddBos(Spec s) {
        return switch (s.model() == null ? "" : s.model()) {
            case "llama" -> true;
            case "gpt2" -> {
                GgufPre.Entry e = GgufPre.get(s.pre());
                yield e != null && e.addBos();
            }
            default -> false;
        };
    }

    private static Map<String, Object> config(Spec s) {
        Map<String, Object> c = new LinkedHashMap<>();
        boolean spm = "llama".equals(s.model());
        int bos = s.bosId() >= 0 ? s.bosId() : spm ? 1 : -1;
        int eos = s.eosId() >= 0 ? s.eosId() : spm ? 2 : -1;
        c.put("bos_token", name(s, bos));
        c.put("eos_token", name(s, eos));
        c.put("add_bos_token", s.addBos() != null ? s.addBos() : defaultAddBos(s));
        return c;
    }

    private static Map<String, Object> addedToken(int id, String content, boolean special) {
        return addedToken(id, content, special, false);
    }

    private static Map<String, Object> addedToken(int id, String content, boolean special, boolean rstrip) {
        return Json.obj("id", (long) id, "content", content, "special", special, "lstrip", false, "rstrip", rstrip,
                "normalized", false, "single_word", false);
    }

    // ------------------------------------------------------------------ gpt2

    private static Tokenizer byteLevel(Spec s) {
        GgufPre.Entry pre = GgufPre.get(s.pre());
        if (pre == null) {
            throw new UnsupportedPreTokenizer("tokenizer.ggml.pre '" + s.pre() + "' has no pre-tokenizer in this build"
                    + " (known: " + String.join(", ", GgufPre.names()) + ")");
        }
        if (s.merges() == null) {
            throw new IllegalArgumentException("the GGUF file has no tokenizer.ggml.merges for its gpt2 tokenizer");
        }
        Map<String, Object> vocab = new LinkedHashMap<>();
        List<Object> added = new ArrayList<>();
        for (int id = 0; id < s.tokens().length; id++) {
            String t = s.tokens()[id];
            Kind k = kind(s, id);
            switch (k) {
                case SPECIAL -> added.add(addedToken(id, t, true, rstrips(s, id, k)));
                case USER -> added.add(addedToken(id, t, false, rstrips(s, id, k)));
                default -> vocab.putIfAbsent(t, (long) id);
            }
        }
        List<Object> merges = new ArrayList<>(s.merges().length);
        for (String m : s.merges()) {
            merges.add(m);
        }
        Map<String, Object> model = Json.obj("type", "BPE", "unk_token", null, "fuse_unk", false, "byte_fallback", false,
                "ignore_merges", pre.ignoreMerges(), "vocab", vocab, "merges", merges);
        // the regexes one after the other, each cutting the pieces the previous one left (llama.cpp: unicode_regex_split)
        List<Object> stages = new ArrayList<>();
        for (String regex : pre.javaRegexes()) {
            stages.add(Json.obj("type", "Split", "pattern", Json.obj("Regex", regex), "behavior", "Isolated", "invert", false));
        }
        stages.add(Json.obj("type", "ByteLevel", "add_prefix_space", false, "trim_offsets", false, "use_regex", false));
        Map<String, Object> json = Json.obj("model", model, "added_tokens", added, "strip_ascii_whitespace", true,
                "pre_tokenizer", Json.obj("type", "Sequence", "pretokenizers", stages),
                "decoder", Json.obj("type", "ByteLevel", "add_prefix_space", false, "trim_offsets", false, "use_regex", false));
        return Tokenizer.fromMaps(json, config(s));
    }

    // ------------------------------------------------------------------ unigram (t5)

    private static Tokenizer unigram(Spec s) {
        if (s.scores() == null || s.scores().length < s.tokens().length) {
            throw new IllegalArgumentException("the GGUF file has no tokenizer.ggml.scores for its unigram tokenizer");
        }
        int n = s.tokens().length;
        List<Object> vocab = new ArrayList<>(n);
        List<Object> added = new ArrayList<>();
        int unk = s.unkId();
        boolean bytes = false;
        for (int id = 0; id < n; id++) {
            String t = s.tokens()[id];
            vocab.add(List.of(t, s.scores()[id]));
            switch (type(s, id)) {
                case UNKNOWN -> {
                    added.add(addedToken(id, t, true));
                    if (unk < 0) {
                        unk = id;
                    }
                }
                case CONTROL -> added.add(addedToken(id, t, true));
                case USER_DEFINED -> added.add(addedToken(id, t, false));
                case BYTE -> bytes = true;
                default -> {
                }
            }
        }
        Map<String, Object> model = Json.obj("type", "Unigram", "unk_id", unk < 0 ? null : (long) unk,
                "byte_fallback", bytes, "vocab", vocab);
        // what the Hugging Face conversion of a T5-style SentencePiece model writes: the charsmap as a Precompiled
        // normalizer; extra whitespace removed by splitting on it; Metaspace with the dummy prefix
        String scheme = s.addSpacePrefix() ? "always" : "never";
        Map<String, Object> metaspace = Json.obj("type", "Metaspace", "replacement", "▁", "prepend_scheme", scheme,
                "split", true);
        Map<String, Object> json = Json.obj("model", model, "added_tokens", added,
                "pre_tokenizer", s.removeExtraWhitespaces()
                        ? Json.obj("type", "Sequence", "pretokenizers", List.of(Json.obj("type", "WhitespaceSplit"), metaspace))
                        : metaspace,
                "decoder", Json.obj("type", "Metaspace", "replacement", "▁", "prepend_scheme", scheme, "split", true));
        if (s.precompiledCharsmap() != null && s.precompiledCharsmap().length > 0) {
            json.put("normalizer", Json.obj("type", "Precompiled", "precompiled_charsmap",
                    java.util.Base64.getEncoder().encodeToString(s.precompiledCharsmap())));
        }
        return Tokenizer.fromMaps(json, config(s));
    }

    // ------------------------------------------------------------------ SentencePiece

    private static Tokenizer sentencePiece(Spec s) {
        int n = s.tokens().length;
        Map<String, Object> vocab = new LinkedHashMap<>();
        List<Object> added = new ArrayList<>();
        List<Object> allTokens = new ArrayList<>(n);
        List<Object> allScores = new ArrayList<>(n);
        String unk = null;
        Set<String> seen = new HashSet<>();
        for (int id = 0; id < n; id++) {
            String t = s.tokens()[id];
            allTokens.add(t);
            allScores.add(s.scores() != null && id < s.scores().length ? s.scores()[id] : 0.0);
            Kind k = kind(s, id);
            switch (k) {
                case SPECIAL -> {
                    added.add(addedToken(id, t, true, rstrips(s, id, k)));
                    if (type(s, id) == UNKNOWN) {
                        unk = t;
                    }
                }
                case USER -> added.add(addedToken(id, t, false, rstrips(s, id, k)));
                default -> {
                    vocab.putIfAbsent(t, (long) id);
                    seen.add(t);
                }
            }
        }
        if (unk == null && s.unkId() < 0 && n > 0 && type(s, 0) == UNKNOWN) {
            unk = s.tokens()[0];
        }
        Map<String, Object> model = Json.obj("type", "LlamaSpm", "unk_token", unk, "fuse_unk", true, "byte_fallback", true,
                "ignore_merges", false, "vocab", vocab, "spm_tokens", allTokens, "spm_scores", allScores);
        // SentencePiece's normalization as llama.cpp does it: a space in front of the text (and of the text after every
        // special token) when add_space_prefix, even if the text starts with a space itself, then every space becomes
        // U+2581 -- a normalizer, not Metaspace (which does not prepend when the text already starts with a space)
        List<Object> norms = new ArrayList<>();
        if (s.addSpacePrefix()) {
            norms.add(Json.obj("type", "Prepend", "prepend", "▁"));
        }
        norms.add(Json.obj("type", "Replace", "pattern", Json.obj("String", " "), "content", "▁"));
        Map<String, Object> json = Json.obj("model", model, "added_tokens", added, "strip_ascii_whitespace", true,
                "normalizer", Json.obj("type", "Sequence", "normalizers", norms),
                "decoder", Json.obj("type", "Sequence", "decoders", List.of(
                        Json.obj("type", "Replace", "pattern", Json.obj("String", "▁"), "content", " "),
                        Json.obj("type", "ByteFallback"),
                        Json.obj("type", "Fuse"),
                        Json.obj("type", "Strip", "content", " ", "start", s.addSpacePrefix() ? 1L : 0L, "stop", 0L))));
        return Tokenizer.fromMaps(json, config(s));
    }
}
