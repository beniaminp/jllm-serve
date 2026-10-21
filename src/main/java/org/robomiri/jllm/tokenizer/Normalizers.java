package org.robomiri.jllm.tokenizer;

import org.robomiri.jllm.json.Json;

import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Builds the {@code normalizer} stage of a tokenizer.json into a composed string function. */
final class Normalizers {

    private Normalizers() {
    }

    /** @return the composed normalizer, or null when the tokenizer has none */
    static UnaryOperator<String> build(Map<String, Object> node) {
        if (node == null) {
            return null;
        }
        List<UnaryOperator<String>> ops = new ArrayList<>();
        collect(node, ops);
        if (ops.isEmpty()) {
            return null;
        }
        return s -> {
            for (UnaryOperator<String> op : ops) {
                s = op.apply(s);
            }
            return s;
        };
    }

    private static void collect(Map<String, Object> node, List<UnaryOperator<String>> ops) {
        String type = Json.str(node, "type", "");
        switch (type) {
            case "Sequence" -> {
                List<Object> children = Json.list(node, "normalizers");
                if (children != null) {
                    for (Object c : children) {
                        collect(Tokenizer.asMap(c), ops);
                    }
                }
            }
            case "NFC" -> ops.add(s -> Normalizer.normalize(s, Normalizer.Form.NFC));
            case "NFD" -> ops.add(s -> Normalizer.normalize(s, Normalizer.Form.NFD));
            case "NFKC" -> ops.add(s -> Normalizer.normalize(s, Normalizer.Form.NFKC));
            case "NFKD" -> ops.add(s -> Normalizer.normalize(s, Normalizer.Form.NFKD));
            case "Lowercase" -> ops.add(Normalizers::lowercase);
            case "StripAccents" -> ops.add(s -> removeMarks(s, false)); // all marks; the reference does not decompose first
            case "Nmt" -> ops.add(Normalizers::nmt);
            case "Precompiled" -> ops.add(Precompiled.of(Json.str(node, "precompiled_charsmap", null)));
            case "BertNormalizer" -> {
                boolean clean = Json.bool(node, "clean_text", true);
                boolean chinese = Json.bool(node, "handle_chinese_chars", true);
                boolean lower = Json.bool(node, "lowercase", true);
                // strip_accents: null means "follow lowercase"
                boolean strip = node.get("strip_accents") instanceof Boolean b ? b : lower;
                ops.add(s -> bert(s, clean, chinese, strip, lower));
            }
            case "Prepend" -> {
                String p = Json.str(node, "prepend", "");
                ops.add(s -> s.isEmpty() ? s : p + s); // HF prepends to non-empty text only
            }
            case "Strip" -> { // Rust str::trim_start / trim_end: the Unicode White_Space property
                boolean left = Json.bool(node, "strip_left", true);
                boolean right = Json.bool(node, "strip_right", true);
                ops.add(s -> {
                    int a = 0;
                    int b = s.length();
                    while (left && a < b && isWhiteSpace(s.charAt(a))) {
                        a++;
                    }
                    while (right && b > a && isWhiteSpace(s.charAt(b - 1))) {
                        b--;
                    }
                    return s.substring(a, b);
                });
            }
            case "Replace" -> {
                Map<String, Object> pat = Json.map(node, "pattern");
                String content = Json.str(node, "content", "");
                String literal = Json.str(pat, "String", null);
                if (literal != null) {
                    ops.add(s -> s.replace(literal, content));
                } else {
                    String regex = Json.str(pat, "Regex", null);
                    if (regex == null) {
                        throw new IllegalArgumentException("Replace normalizer has no String/Regex pattern");
                    }
                    // UNICODE_CHARACTER_CLASS like the pre-tokenizer regexes: the Rust regex crate's \s, \w, \d
                    // are Unicode-aware, Java's are ASCII-only by default
                    Pattern p = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
                    String repl = Matcher.quoteReplacement(content);
                    ops.add(s -> p.matcher(s).replaceAll(repl));
                }
            }
            default -> throw new IllegalArgumentException("unsupported normalizer type: " + type);
        }
    }

    /**
     * Hugging Face's {@code Lowercase}: every code point on its own through the full Unicode lowercase mapping
     * (Rust {@code char::to_lowercase}), so a final capital sigma becomes {@code σ}, not the context-dependent
     * {@code ς} of {@link String#toLowerCase}.
     */
    static String lowercase(String s) {
        int n = s.length();
        boolean ascii = true;
        for (int i = 0; i < n && ascii; i++) {
            ascii = s.charAt(i) < 0x80;
        }
        if (ascii) {
            return s.toLowerCase(Locale.ROOT);
        }
        StringBuilder sb = new StringBuilder(n);
        for (int i = 0; i < n; ) {
            int cp = s.codePointAt(i);
            int cl = Character.charCount(cp);
            if (cp < 0x80) {
                sb.append(Character.toLowerCase((char) cp));
            } else {
                sb.append(s.substring(i, i + cl).toLowerCase(Locale.ROOT));
            }
            i += cl;
        }
        return sb.toString();
    }

    /** Drops combining marks: all of General_Category M, or only Mn (nonspacing) when {@code nonspacingOnly}. */
    static String removeMarks(String s, boolean nonspacingOnly) {
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            int t = Character.getType(cp);
            boolean mark = t == Character.NON_SPACING_MARK
                    || (!nonspacingOnly && (t == Character.COMBINING_SPACING_MARK || t == Character.ENCLOSING_MARK));
            if (!mark) {
                sb.appendCodePoint(cp);
            }
            i += Character.charCount(cp);
        }
        return sb.toString();
    }

    /** SentencePiece's NMT normalization: drop some control characters, turn others and odd spaces into a space. */
    static String nmt(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if ((c >= 0x01 && c <= 0x08) || c == 0x0B || (c >= 0x0E && c <= 0x1F) || c == 0x7F || c == 0x8F
                    || c == 0x9F) {
                continue;
            }
            boolean space = c == 0x09 || c == 0x0A || c == 0x0C || c == 0x0D || c == 0x1680
                    || (c >= 0x200B && c <= 0x200F) || c == 0x2028 || c == 0x2029 || c == 0x2581 || c == 0xFEFF
                    || c == 0xFFFD;
            sb.append(space ? ' ' : c);
        }
        return sb.toString();
    }

    /**
     * BERT's normalizer, in the reference's order: clean text (drop NUL, U+FFFD and control / format /
     * private-use characters, turn every whitespace character into a space), put spaces around CJK ideographs,
     * strip accents (NFD, then drop nonspacing marks), lowercase.
     */
    static String bert(String s, boolean clean, boolean chinese, boolean stripAccents, boolean lowercase) {
        if (clean || chinese) {
            StringBuilder sb = new StringBuilder(s.length() + 8);
            for (int i = 0; i < s.length(); ) {
                int cp = s.codePointAt(i);
                i += Character.charCount(cp);
                if (clean) {
                    if (cp == 0 || cp == 0xFFFD || isBertControl(cp)) {
                        continue;
                    }
                    if (isWhiteSpace(cp)) { // includes tab, newline, carriage return
                        cp = ' ';
                    }
                }
                if (chinese && isCjk(cp)) {
                    sb.append(' ').appendCodePoint(cp).append(' ');
                } else {
                    sb.appendCodePoint(cp);
                }
            }
            s = sb.toString();
        }
        if (stripAccents) {
            s = removeMarks(Normalizer.normalize(s, Normalizer.Form.NFD), true);
        }
        return lowercase ? lowercase(s) : s;
    }

    /**
     * Control characters for BERT's clean-up: categories Cc, Cf and Co except tab, newline and carriage return
     * (which count as whitespace). Unassigned code points are kept, like the reference.
     */
    private static boolean isBertControl(int cp) {
        if (cp == '\t' || cp == '\n' || cp == '\r') {
            return false;
        }
        int t = Character.getType(cp);
        return t == Character.CONTROL || t == Character.FORMAT || t == Character.PRIVATE_USE
                || t == Character.SURROGATE;
    }

    /** The "Chinese character" ranges of BERT's normalizer (CJK Unified Ideographs and some of their extensions). */
    static boolean isCjk(int cp) {
        return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF)
                || (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F)
                || (cp >= 0x2B920 && cp <= 0x2CEAF) || (cp >= 0xF900 && cp <= 0xFAFF)
                || (cp >= 0x2F800 && cp <= 0x2FA1F);
    }

    /**
     * SentencePiece's {@code Precompiled} normalizer: the {@code precompiled_charsmap} of the {@code .model} file,
     * a darts-clone double-array trie over UTF-8 byte strings whose values point into a blob of NUL-terminated
     * replacement strings (for the usual {@code nmt_nfkc} rules: NFKC plus SentencePiece's whitespace and
     * control-character clean-up). Like the reference it is applied grapheme cluster by grapheme cluster: a
     * cluster shorter than 6 UTF-8 bytes is replaced as a whole when a prefix of it is in the trie (the shortest
     * such prefix decides the replacement), otherwise each code point is looked up on its own. Grapheme clusters
     * come from {@code java.util.regex}'s {@code \X}, whose Unicode version may differ slightly from the
     * reference's segmentation crate.
     *
     * <p>A {@code null} charsmap (which the reference refuses to load) falls back to {@code NFKC}, the bulk of
     * {@code nmt_nfkc}; that approximation differs on control characters and unusual spaces.
     */
    static final class Precompiled implements UnaryOperator<String> {
        private static final Pattern GRAPHEME = Pattern.compile("\\X");
        private final int[] trie;
        private final byte[] strings;

        private Precompiled(byte[] blob) {
            if (blob.length < 4) {
                throw new IllegalArgumentException("Precompiled normalizer: precompiled_charsmap is too short");
            }
            long trieBytes = (blob[0] & 0xFFL) | (blob[1] & 0xFFL) << 8 | (blob[2] & 0xFFL) << 16 | (blob[3] & 0xFFL) << 24;
            if (trieBytes == 0 || trieBytes % 4 != 0 || 4 + trieBytes > blob.length) {
                throw new IllegalArgumentException("Precompiled normalizer: cannot parse precompiled_charsmap");
            }
            trie = new int[(int) (trieBytes / 4)];
            for (int i = 0; i < trie.length; i++) {
                int o = 4 + 4 * i;
                trie[i] = (blob[o] & 0xFF) | (blob[o + 1] & 0xFF) << 8 | (blob[o + 2] & 0xFF) << 16
                        | (blob[o + 3] & 0xFF) << 24;
            }
            strings = Arrays.copyOfRange(blob, 4 + (int) trieBytes, blob.length);
        }

        static UnaryOperator<String> of(String base64) {
            if (base64 == null) {
                System.err.println("tokenizer: Precompiled normalizer without a charsmap, using NFKC instead");
                return s -> Normalizer.normalize(s, Normalizer.Form.NFKC);
            }
            try {
                return new Precompiled(Base64.getDecoder().decode(base64));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("Precompiled normalizer: " + e.getMessage(), e);
            }
        }

        /** The replacement of the shortest prefix of {@code key} found in the trie, or null. */
        private String transform(byte[] key) {
            int[] a = trie;
            int pos = offset(a[0]);
            for (byte b : key) {
                int c = b & 0xFF;
                if (c == 0) {
                    break;
                }
                pos ^= c;
                if (pos >= a.length) {
                    return null;
                }
                int unit = a[pos];
                if ((unit & 0x800000FF) != c) {
                    return null;
                }
                pos ^= offset(unit);
                if (((unit >>> 8) & 1) != 0) {
                    if (pos >= a.length) {
                        return null;
                    }
                    int start = a[pos] & 0x7FFFFFFF;
                    if (start > strings.length) {
                        return null;
                    }
                    int end = start;
                    while (end < strings.length && strings[end] != 0) {
                        end++;
                    }
                    return new String(strings, start, end - start, StandardCharsets.UTF_8);
                }
            }
            return null;
        }

        /** darts-clone unit offset: bits 10.. shifted left by 8 when bit 9 is set. */
        private static int offset(int unit) {
            return (unit >>> 10) << ((unit & (1 << 9)) >>> 6);
        }

        @Override
        public String apply(String s) {
            StringBuilder sb = new StringBuilder(s.length() + 8);
            Matcher m = GRAPHEME.matcher(s);
            while (m.find()) {
                String g = m.group();
                byte[] gb = g.getBytes(StandardCharsets.UTF_8);
                if (gb.length < 6) {
                    String norm = transform(gb);
                    if (norm != null) {
                        sb.append(norm);
                        continue;
                    }
                }
                for (int i = 0; i < g.length(); ) {
                    int cl = Character.charCount(g.codePointAt(i));
                    String part = g.substring(i, i + cl);
                    String norm = transform(part.getBytes(StandardCharsets.UTF_8));
                    sb.append(norm != null ? norm : part);
                    i += cl;
                }
            }
            return sb.toString();
        }
    }

    /**
     * Rust's {@code char::is_whitespace}: exactly the code points with the Unicode {@code White_Space} property.
     * Differs from {@link Character#isWhitespace} (which adds U+001C..U+001F and lacks the no-break spaces) and
     * from {@link Character#isSpaceChar} (which lacks the controls).
     */
    static boolean isWhiteSpace(int c) {
        if (c < 0x80) {
            return (c >= 0x09 && c <= 0x0D) || c == 0x20;
        }
        return c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028
                || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
    }
}
