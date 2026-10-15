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
            case "Strip" -> {
                boolean left = Json.bool(node, "strip_left", true);
                boolean right = Json.bool(node, "strip_right", true);
                ops.add(s -> {
                    int a = 0;
                    int b = s.length();
                    while (left && a < b && Character.isWhitespace(s.charAt(a))) {
                        a++;
                    }
                    while (right && b > a && Character.isWhitespace(s.charAt(b - 1))) {
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
                    Pattern p = Pattern.compile(regex);
                    String repl = Matcher.quoteReplacement(content);
                    ops.add(s -> p.matcher(s).replaceAll(repl));
                }
            }
            default -> throw new IllegalArgumentException("unsupported normalizer type: " + type);
        }
    }
}
