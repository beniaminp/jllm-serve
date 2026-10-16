package org.robomiri.jllm.tokenizer;

import org.robomiri.jllm.json.Json;

import java.text.Normalizer;
import java.util.ArrayList;
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
            case "Lowercase" -> ops.add(s -> s.toLowerCase(Locale.ROOT));
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
