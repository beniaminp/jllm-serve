package org.robomiri.jllm.tokenizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * The {@code tokenizer.ggml.pre} values a byte-level BPE GGUF can name, with what llama.cpp does for each
 * ({@code llama_vocab::impl::load} and {@code llm_tokenizer_bpe}): the list of regular expressions the text is cut
 * with, one after the other (each one splits the pieces the previous one left, matches and the text between them
 * both kept: a Hugging Face {@code Sequence} of {@code Split} pre-tokenizers with the {@code Isolated} behavior), and
 * the flags {@code ignore_merges} (a piece that is a token as a whole is not merged) and the default of
 * {@code add_bos_token}.
 *
 * <p>The patterns are those of llama.cpp b11669, character for character ({@link #RAW}); {@link #javaRegex} applies
 * the few changes that make {@code java.util.regex} do what llama.cpp's C++ implementation does:
 * <ul>
 *   <li>llama.cpp's {@code \p{Lu}}, {@code \p{Ll}}, {@code \p{Lt}}, {@code \p{Lm}}, {@code \p{Lo}} are all "any
 *       letter" there ({@code k_ucat_enum}), so they become {@code \p{L}};</li>
 *   <li>a regex without a Unicode category runs on {@code std::wregex}, where every non-ASCII whitespace character
 *       was replaced by U+000B before matching: it is matched by {@code \s} but by no character class, so
 *       {@code &&[^\s]} is added to the non-ASCII classes of such patterns (DeepSeek's CJK ranges contain U+3000,
 *       U+2000...);</li>
 *   <li>{@code $} (ECMAScript: end of the piece, never "before a final line terminator") becomes {@code \z};</li>
 *   <li>the look-ahead groups {@code ((?=[\p{L}])([^a-z]))} of the GPT-4o / Tekken patterns, which would recurse once per
 *       letter in {@code java.util.regex}, become the same set as one class {@code [\p{L}&&[^a-z]]}.</li>
 * </ul>
 */
final class GgufPre {

    private GgufPre() {
    }

    /** One {@code tokenizer.ggml.pre} value: its regexes and flags. */
    record Entry(String name, String[] regexes, boolean ignoreMerges, boolean addBos) {
        /** The regexes as {@code java.util.regex} sources. */
        List<String> javaRegexes() {
            List<String> out = new ArrayList<>(regexes.length);
            for (String r : regexes) {
                out.add(javaRegex(r));
            }
            return out;
        }
    }

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_LLAMA3 (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] LLAMA3 = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_JAIS2 (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] JAIS2 = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] DEEPSEEK_LLM = {
            "[\r\n]",
            "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
            "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
            "\\s+$",
            "[一-龥ࠀ-一가-퟿]+",
            "\\p{N}+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] DEEPSEEK3_LLM = {
            "\\p{N}{1,3}",
            "[一-龥぀-ゟ゠-ヿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_SPARK2_5 (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] SPARK2_5 = {
            "\\p{N}{1,3}",
            "[一-龥぀-ゟ゠-ヿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+",
            "\\p{N}",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_YOUTU (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] YOUTU = {
            "[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] DEEPSEEK_CODER = {
            "[\r\n]",
            "\\s?\\p{L}+",
            "\\s?\\p{P}+",
            "[一-龥ࠀ-一가-퟿]+",
            "\\p{N}",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_FALCON (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] FALCON = {
            "[\\p{P}\\$\\+<=>\\^~\\|`]+",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            "[0-9][0-9][0-9]",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_STARCODER (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] STARCODER = {
            "\\p{N}",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_GPT2 (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] GPT2 = {
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_STABLELM2 (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] STABLELM2 = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35 (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] QWEN35 = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_PORO (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] PORO = {
            " ?[^(\\s|.,!?…。，、।۔،)]+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_VIKING (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] VIKING = {
            " ?[^(\\s|.,!?…。，、।۔،)]+",
            "\\p{N}",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_TEKKEN (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] TEKKEN = {
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_GPT4O (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] GPT4O = {
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_GRANITE_EMB_MULTI (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] GRANITE_EMB_MULTI = {
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_BAILINGMOE (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] BAILINGMOE = {
            "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_SEED_CODER (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] SEED_CODER = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_UFAKZEKA (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] UFAKZEKA = {
            "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_LAGUNA (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] LAGUNA = {
            "[^\\n]+|[\\n]+",
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_MINICPM5 (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] MINICPM5 = {
            "\\p{N}{1,3}",
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    /** llama.cpp LLAMA_VOCAB_PRE_TYPE_K2_HORIZON (llm_tokenizer_bpe::regex_exprs), copied verbatim. */
    static final String[] K2_HORIZON = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    /** llama.cpp's default pre-tokenizer: used when {@code tokenizer.ggml.pre} is absent or {@code "default"} (GPT-NeoX). */
    static final String[] DEFAULT = {
            "[\\p{P}\\$\\+<=>\\^~\\|]+",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            "\\p{N}+",
            "[0-9][0-9][0-9]",
    };

    /** The patterns exactly as llama.cpp has them, by {@code tokenizer.ggml.pre} value. */
    static final Map<String, Entry> RAW = new LinkedHashMap<>();

    private static void put(String[] regexes, boolean ignoreMerges, boolean addBos, String... names) {
        for (String n : names) {
            RAW.put(n, new Entry(n, regexes, ignoreMerges, addBos));
        }
    }

    static {
        // the Llama 3 family: ignore_merges; the BOS is added unless the file says otherwise
        put(LLAMA3, true, true, "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2",
                "jina-v5-nano");
        put(LLAMA3, false, false, "dbrx", "smaug-bpe", "chatglm-bpe");
        put(LLAMA3, true, false, "glm4", "glm5");
        put(JAIS2, false, false, "jais-2");
        put(DEEPSEEK_LLM, false, false, "deepseek-llm");
        put(DEEPSEEK_CODER, false, false, "deepseek-coder");
        put(DEEPSEEK3_LLM, false, false, "deepseek-v3", "hunyuan-dense", "hy_v4", "joyai-llm");
        put(SPARK2_5, false, false, "spark2_5");
        put(YOUTU, true, false, "youtu");
        put(FALCON, false, false, "falcon");
        put(GPT2, false, false, "mpt", "gpt-2", "phi-2", "jina-es", "jina-de", "gigachat", "jina-v2-es", "jina-v2-de",
                "a.x-4.0", "mellum", "modern-bert", "jina-v1-en", "jina-v2-code", "roberta-bpe", "olmo", "jais", "trillion",
                "granite-docling", "exaone4");
        put(STARCODER, false, false, "starcoder", "refact", "command-r", "smollm", "codeshell", "exaone", "minerva-7b",
                "mellum2");
        put(STABLELM2, false, false, "qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "stablelm2", "megrez", "hunyuan",
                "solar-open", "grok-2");
        put(QWEN35, false, false, "qwen35");
        put(PORO, false, false, "poro-chat", "bloom", "gpt3-finnish");
        put(VIKING, false, false, "viking");
        put(TEKKEN, true, true, "tekken");
        put(GPT4O, false, false, "gpt-4o", "llama4", "kanana2", "talkie", "minimax-m2");
        put(GRANITE_EMB_MULTI, true, false, "granite-embed-multi-97m");
        put(BAILINGMOE, false, false, "bailingmoe", "bailingmoe2", "llada-moe");
        put(SEED_CODER, false, false, "seed-coder");
        put(UFAKZEKA, false, false, "ufakzeka");
        put(LAGUNA, false, false, "laguna");
        put(MINICPM5, true, false, "minicpm5");
        put(K2_HORIZON, false, false, "k2-horizon");
    }

    /** The entry for a {@code tokenizer.ggml.pre} value; null/empty/"default" is llama.cpp's default; null if unknown. */
    static Entry get(String pre) {
        if (pre == null || pre.isEmpty() || pre.equals("default")) {
            return new Entry("default", DEFAULT, false, false);
        }
        return RAW.get(pre);
    }

    static Set<String> names() {
        Set<String> s = new TreeSet<>(RAW.keySet());
        s.add("default");
        return s;
    }

    private static final String[] CASED_CLASSES = {"\\p{Lu}", "\\p{Ll}", "\\p{Lt}", "\\p{Lm}", "\\p{Lo}"};

    /** The {@code java.util.regex} source that splits like llama.cpp's implementation of {@code raw}; see the class comment. */
    static String javaRegex(String raw) {
        String r = raw;
        for (String c : CASED_CLASSES) {
            r = r.replace(c, "\\p{L}");
        }
        boolean categories = raw.contains("\\p{");
        if (!categories) {
            r = excludeWhitespaceFromClasses(r);
        }
        if (r.endsWith("$") && !r.endsWith("\\$")) {
            r = r.substring(0, r.length() - 1) + "\\z";
        }
        r = r.replace("(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+", "[\\p{L}\\p{M}\\u200C\\u200D]+");
        r = r.replace("((?=[\\p{L}])([^a-z]))", "[\\p{L}&&[^a-z]]").replace("((?=[\\p{L}])([^A-Z]))", "[\\p{L}&&[^A-Z]]")
                .replace("((?=[\\p{L}\\p{M}])([^a-z]))", "[\\p{L}\\p{M}&&[^a-z]]")
                .replace("((?=[\\p{L}\\p{M}])([^A-Z]))", "[\\p{L}\\p{M}&&[^A-Z]]");
        return r;
    }

    /** Adds {@code &&[^\s]} to every positive character class that lists a non-ASCII character. */
    private static String excludeWhitespaceFromClasses(String r) {
        StringBuilder out = new StringBuilder(r.length() + 16);
        for (int i = 0; i < r.length(); i++) {
            char c = r.charAt(i);
            if (c == '\\') {
                out.append(c);
                if (i + 1 < r.length()) {
                    out.append(r.charAt(++i));
                }
            } else if (c == '[') {
                int j = i + 1;
                boolean negated = j < r.length() && r.charAt(j) == '^';
                boolean nonAscii = false;
                while (j < r.length() && r.charAt(j) != ']') {
                    if (r.charAt(j) == '\\') {
                        j++;
                    } else if (r.charAt(j) > 127) {
                        nonAscii = true;
                    }
                    j++;
                }
                out.append(r, i, j);
                if (nonAscii && !negated) {
                    out.append("&&[^\\s]");
                }
                i = j - 1;
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }
}
