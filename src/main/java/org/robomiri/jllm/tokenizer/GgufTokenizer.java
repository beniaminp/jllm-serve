package org.robomiri.jllm.tokenizer;

import org.robomiri.jllm.json.Json;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds a {@link Tokenizer} from the tokenizer metadata of a GGUF file ({@code tokenizer.ggml.*}) by writing the
 * {@code tokenizer.json} the same tokenizer has in the Hugging Face ecosystem and loading that, so the encoder is
 * the one all other models use and the result is checked against it ({@code GgufTokenizerTest}).
 *
 * <ul>
 *   <li>{@code tokenizer.ggml.model = gpt2}: byte-level BPE. The tokens are the vocabulary, {@code merges} the merge
 *       list, {@code token_type} separates control / user-defined tokens (added tokens: special, resp. not) from
 *       the vocabulary; {@code tokenizer.ggml.pre} selects the pre-tokenizer regex ({@link #PRE}). Pre-tokenizers
 *       without a known regex are refused ({@link UnsupportedPreTokenizer}), so the caller can fetch the
 *       original tokenizer.json instead of tokenizing differently.</li>
 *   <li>{@code tokenizer.ggml.model = llama}: SentencePiece BPE (Llama 2, Mistral, TinyLlama...). The file stores
 *       scores instead of merges; as in {@code transformers}' conversion every split of a token into two
 *       vocabulary tokens becomes a merge, ranked by the score of the merged token (best first). Spaces become
 *       {@code U+2581}, a dummy prefix is added when {@code add_space_prefix} (the default) is set, unknown
 *       characters fall back to the {@code <0xXX>} byte tokens.</li>
 *   <li>{@code tokenizer.ggml.model = t5}: SentencePiece unigram (T5-style encoders). Tokens and scores become the
 *       Unigram vocabulary (Viterbi segmentation), {@code tokenizer.ggml.precompiled_charsmap} the
 *       {@code Precompiled} normalizer, {@code remove_extra_whitespaces} a whitespace split and
 *       {@code add_space_prefix} the Metaspace prefix, as in the Hugging Face conversion of such models.</li>
 * </ul>
 */
public final class GgufTokenizer {

    /** {@code tokenizer.ggml.token_type} values. */
    public static final int NORMAL = 1, UNKNOWN = 2, CONTROL = 3, USER_DEFINED = 4, UNUSED = 5, BYTE = 6;

    private static final String QWEN2 =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    private static final String LLAMA3 =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    private static final String GPT2 = PreTokenizers.GPT2_REGEX;

    /**
     * The {@code tokenizer.ggml.pre} values with a known regex (llama.cpp's {@code LLAMA_VOCAB_PRE_TYPE_*}); the value
     * is the Split regex of the Hugging Face tokenizer of the same model.
     */
    public static final Map<String, String> PRE = Map.of(
            "qwen2", QWEN2,
            "llama-bpe", LLAMA3,
            "llama3", LLAMA3,
            "llama-v3", LLAMA3,
            "gpt-2", GPT2,
            "default", GPT2);

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
     * @param addBos   {@code tokenizer.ggml.add_bos_token}, null if absent (then: whether a BOS exists)
     * @param addSpacePrefix {@code tokenizer.ggml.add_space_prefix} (SentencePiece: the dummy prefix), default true
     * @param precompiledCharsmap {@code tokenizer.ggml.precompiled_charsmap} (unigram: the SentencePiece
     *                            normalization table), null if absent
     * @param removeExtraWhitespaces {@code tokenizer.ggml.remove_extra_whitespaces} (unigram), default false
     */
    public record Spec(String model, String pre, String[] tokens, long[] types, double[] scores, String[] merges,
                       int bosId, int eosId, int unkId, Boolean addBos, boolean addSpacePrefix,
                       byte[] precompiledCharsmap, boolean removeExtraWhitespaces) {

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

    private static Map<String, Object> config(Spec s) {
        Map<String, Object> c = new LinkedHashMap<>();
        String bos = name(s, s.bosId());
        c.put("bos_token", bos);
        c.put("eos_token", name(s, s.eosId()));
        c.put("add_bos_token", s.addBos() != null ? s.addBos() : false);
        return c;
    }

    private static Map<String, Object> addedToken(int id, String content, boolean special) {
        return Json.obj("id", (long) id, "content", content, "special", special, "lstrip", false, "rstrip", false,
                "normalized", false, "single_word", false);
    }

    // ------------------------------------------------------------------ gpt2

    private static Tokenizer byteLevel(Spec s) {
        String regex = PRE.get(s.pre() == null ? "default" : s.pre());
        if (regex == null) {
            throw new UnsupportedPreTokenizer("tokenizer.ggml.pre '" + s.pre() + "' has no pre-tokenizer regex in this build"
                    + " (known: " + String.join(", ", new java.util.TreeSet<>(PRE.keySet())) + ")");
        }
        if (s.merges() == null) {
            throw new IllegalArgumentException("the GGUF file has no tokenizer.ggml.merges for its gpt2 tokenizer");
        }
        Map<String, Object> vocab = new LinkedHashMap<>();
        List<Object> added = new ArrayList<>();
        for (int id = 0; id < s.tokens().length; id++) {
            String t = s.tokens()[id];
            switch (type(s, id)) {
                case CONTROL -> added.add(addedToken(id, t, true));
                case USER_DEFINED -> added.add(addedToken(id, t, false));
                default -> vocab.putIfAbsent(t, (long) id);
            }
        }
        List<Object> merges = new ArrayList<>(s.merges().length);
        for (String m : s.merges()) {
            merges.add(m);
        }
        boolean llama3 = regex.equals(LLAMA3);
        Map<String, Object> model = Json.obj("type", "BPE", "unk_token", null, "fuse_unk", false, "byte_fallback", false,
                "ignore_merges", llama3, "vocab", vocab, "merges", merges);
        Map<String, Object> pre = Json.obj("type", "Sequence", "pretokenizers", List.of(
                Json.obj("type", "Split", "pattern", Json.obj("Regex", regex), "behavior", "Isolated", "invert", false),
                Json.obj("type", "ByteLevel", "add_prefix_space", false, "trim_offsets", false, "use_regex", false)));
        Map<String, Object> json = Json.obj("model", model, "added_tokens", added, "pre_tokenizer", pre,
                "decoder", Json.obj("type", "ByteLevel", "add_prefix_space", false, "trim_offsets", false, "use_regex", false));
        if ("qwen2".equals(s.pre())) {
            json.put("normalizer", Json.obj("type", "NFC")); // Qwen's tokenizer.json normalizes to NFC, the GGUF does not say so
        }
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
        if (s.scores() == null) {
            throw new IllegalArgumentException("the GGUF file has no tokenizer.ggml.scores for its SentencePiece tokenizer");
        }
        int n = s.tokens().length;
        Map<String, Integer> ids = new HashMap<>(n * 2);
        Map<String, Object> vocab = new LinkedHashMap<>();
        List<Object> added = new ArrayList<>();
        String unk = null;
        for (int id = 0; id < n; id++) {
            String t = s.tokens()[id];
            switch (type(s, id)) {
                case UNKNOWN -> {
                    added.add(addedToken(id, t, true));
                    unk = t;
                }
                case CONTROL -> added.add(addedToken(id, t, true));
                case USER_DEFINED -> added.add(addedToken(id, t, false));
                default -> {
                    vocab.putIfAbsent(t, (long) id);
                    ids.putIfAbsent(t, id);
                }
            }
        }
        // merges: every split of a token into two vocabulary tokens, best merged-token score first
        record Merge(String left, String right, double score, long order) {
        }
        List<Merge> all = new ArrayList<>();
        for (int id = 0; id < n; id++) {
            int ty = type(s, id);
            if (ty != NORMAL && ty != USER_DEFINED && ty != UNUSED) {
                continue;
            }
            String piece = s.tokens()[id];
            if (ty != NORMAL || piece.length() < 2) {
                continue;
            }
            List<Merge> local = new ArrayList<>();
            for (int cut = 1; cut < piece.length(); cut++) {
                if (Character.isLowSurrogate(piece.charAt(cut))) {
                    continue;
                }
                String l = piece.substring(0, cut);
                String r = piece.substring(cut);
                Integer il = ids.get(l);
                Integer ir = ids.get(r);
                if (il != null && ir != null) {
                    local.add(new Merge(l, r, s.scores()[id], ((long) il << 32) | ir));
                }
            }
            local.sort((a, b) -> Long.compare(a.order(), b.order()));
            all.addAll(local);
        }
        all.sort((a, b) -> Double.compare(b.score(), a.score())); // stable: ties keep the id order
        List<Object> merges = new ArrayList<>(all.size());
        for (Merge m : all) {
            merges.add(m.left() + " " + m.right());
        }
        Map<String, Object> model = Json.obj("type", "BPE", "unk_token", unk, "fuse_unk", true, "byte_fallback", true,
                "ignore_merges", false, "vocab", vocab, "merges", merges);
        // SentencePiece's normalization: the dummy prefix is always added (a text that starts with a space gets a
        // second one), then every space becomes U+2581 -- the normalizer of Llama 2's tokenizer.json, not Metaspace
        // (which does not prepend when the text already starts with a space)
        List<Object> norms = new ArrayList<>();
        if (s.addSpacePrefix()) {
            norms.add(Json.obj("type", "Prepend", "prepend", "▁"));
        }
        norms.add(Json.obj("type", "Replace", "pattern", Json.obj("String", " "), "content", "▁"));
        Map<String, Object> json = Json.obj("model", model, "added_tokens", added,
                "normalizer", Json.obj("type", "Sequence", "normalizers", norms),
                "decoder", Json.obj("type", "Sequence", "decoders", List.of(
                        Json.obj("type", "Replace", "pattern", Json.obj("String", "▁"), "content", " "),
                        Json.obj("type", "ByteFallback"),
                        Json.obj("type", "Fuse"),
                        Json.obj("type", "Strip", "content", " ", "start", s.addSpacePrefix() ? 1L : 0L, "stop", 0L))));
        return Tokenizer.fromMaps(json, config(s));
    }
}
