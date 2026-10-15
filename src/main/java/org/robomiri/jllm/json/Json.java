package org.robomiri.jllm.json;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON parser and writer.
 *
 * <p>Objects map to {@link LinkedHashMap}, arrays to {@link ArrayList}, integral numbers to {@link Long},
 * other numbers to {@link Double}, plus {@link String}, {@link Boolean} and {@code null}.
 */
public final class Json {

    private Json() {
    }

    // ------------------------------------------------------------------ parsing

    public static Object parse(String text) {
        Parser p = new Parser(text);
        p.skipWs();
        Object v = p.value();
        p.skipWs();
        if (p.pos != text.length()) {
            throw p.error("trailing characters");
        }
        return v;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> parseObject(String text) {
        Object v = parse(text);
        if (!(v instanceof Map)) {
            throw new IllegalArgumentException("expected a JSON object");
        }
        return (Map<String, Object>) v;
    }

    private static final class Parser {
        final String s;
        int pos;

        Parser(String s) {
            this.s = s;
        }

        IllegalArgumentException error(String msg) {
            return new IllegalArgumentException("JSON " + msg + " at offset " + pos);
        }

        void skipWs() {
            while (pos < s.length()) {
                char c = s.charAt(pos);
                if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
                    pos++;
                } else {
                    break;
                }
            }
        }

        Object value() {
            if (pos >= s.length()) {
                throw error("unexpected end");
            }
            char c = s.charAt(pos);
            return switch (c) {
                case '{' -> object();
                case '[' -> array();
                case '"' -> string();
                case 't' -> literal("true", Boolean.TRUE);
                case 'f' -> literal("false", Boolean.FALSE);
                case 'n' -> literal("null", null);
                default -> {
                    if (c == '-' || (c >= '0' && c <= '9')) {
                        yield number();
                    }
                    throw error("unexpected character '" + c + "'");
                }
            };
        }

        Object literal(String word, Object v) {
            if (!s.startsWith(word, pos)) {
                throw error("invalid literal");
            }
            pos += word.length();
            return v;
        }

        Map<String, Object> object() {
            Map<String, Object> m = new LinkedHashMap<>();
            pos++; // {
            skipWs();
            if (peek() == '}') {
                pos++;
                return m;
            }
            while (true) {
                skipWs();
                if (peek() != '"') {
                    throw error("expected object key");
                }
                String key = string();
                skipWs();
                expect(':');
                skipWs();
                m.put(key, value());
                skipWs();
                char c = next();
                if (c == '}') {
                    return m;
                }
                if (c != ',') {
                    throw error("expected ',' or '}'");
                }
            }
        }

        List<Object> array() {
            List<Object> list = new ArrayList<>();
            pos++; // [
            skipWs();
            if (peek() == ']') {
                pos++;
                return list;
            }
            while (true) {
                skipWs();
                list.add(value());
                skipWs();
                char c = next();
                if (c == ']') {
                    return list;
                }
                if (c != ',') {
                    throw error("expected ',' or ']'");
                }
            }
        }

        String string() {
            pos++; // opening quote
            int start = pos;
            // fast path: no escapes
            while (pos < s.length()) {
                char c = s.charAt(pos);
                if (c == '"') {
                    String r = s.substring(start, pos);
                    pos++;
                    return r;
                }
                if (c == '\\') {
                    break;
                }
                pos++;
            }
            StringBuilder sb = new StringBuilder().append(s, start, pos);
            while (pos < s.length()) {
                char c = s.charAt(pos++);
                if (c == '"') {
                    return sb.toString();
                }
                if (c != '\\') {
                    sb.append(c);
                    continue;
                }
                char e = next();
                switch (e) {
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case '/' -> sb.append('/');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> {
                        if (pos + 4 > s.length()) {
                            throw error("bad unicode escape");
                        }
                        sb.append((char) Integer.parseInt(s, pos, pos + 4, 16));
                        pos += 4;
                    }
                    default -> throw error("bad escape '\\" + e + "'");
                }
            }
            throw error("unterminated string");
        }

        Object number() {
            int start = pos;
            boolean integral = true;
            if (peek() == '-') {
                pos++;
            }
            while (pos < s.length()) {
                char c = s.charAt(pos);
                if (c >= '0' && c <= '9') {
                    pos++;
                } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                    integral = false;
                    pos++;
                } else {
                    break;
                }
            }
            String num = s.substring(start, pos);
            if (integral && num.length() < 19) {
                return Long.parseLong(num);
            }
            return Double.parseDouble(num);
        }

        char peek() {
            if (pos >= s.length()) {
                throw error("unexpected end");
            }
            return s.charAt(pos);
        }

        char next() {
            char c = peek();
            pos++;
            return c;
        }

        void expect(char c) {
            if (next() != c) {
                throw error("expected '" + c + "'");
            }
        }
    }

    // ------------------------------------------------------------------ writing

    public static String write(Object value) {
        StringBuilder sb = new StringBuilder();
        write(value, sb);
        return sb.toString();
    }

    public static void write(Object v, StringBuilder sb) {
        switch (v) {
            case null -> sb.append("null");
            case String str -> writeString(str, sb);
            case Boolean b -> sb.append(b.booleanValue());
            case Double d when d.isNaN() || d.isInfinite() -> sb.append("null");
            case Float f when f.isNaN() || f.isInfinite() -> sb.append("null");
            case Number n -> sb.append(n);
            case Map<?, ?> m -> {
                sb.append('{');
                boolean first = true;
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    if (!first) {
                        sb.append(',');
                    }
                    first = false;
                    writeString(String.valueOf(e.getKey()), sb);
                    sb.append(':');
                    write(e.getValue(), sb);
                }
                sb.append('}');
            }
            case Iterable<?> it -> {
                sb.append('[');
                boolean first = true;
                for (Object o : it) {
                    if (!first) {
                        sb.append(',');
                    }
                    first = false;
                    write(o, sb);
                }
                sb.append(']');
            }
            case int[] arr -> {
                sb.append('[');
                for (int i = 0; i < arr.length; i++) {
                    if (i > 0) {
                        sb.append(',');
                    }
                    sb.append(arr[i]);
                }
                sb.append(']');
            }
            default -> writeString(v.toString(), sb);
        }
    }

    private static void writeString(String s, StringBuilder sb) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20 || c == 0x2028 || c == 0x2029) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        sb.append('"');
    }

    // ------------------------------------------------------------------ typed accessors

    @SuppressWarnings("unchecked")
    public static Map<String, Object> map(Object o, String key) {
        Object v = o instanceof Map<?, ?> m ? m.get(key) : null;
        return v instanceof Map ? (Map<String, Object>) v : null;
    }

    @SuppressWarnings("unchecked")
    public static List<Object> list(Object o, String key) {
        Object v = o instanceof Map<?, ?> m ? m.get(key) : null;
        return v instanceof List ? (List<Object>) v : null;
    }

    public static String str(Object o, String key, String def) {
        Object v = o instanceof Map<?, ?> m ? m.get(key) : null;
        return v instanceof String s ? s : def;
    }

    public static long num(Object o, String key, long def) {
        Object v = o instanceof Map<?, ?> m ? m.get(key) : null;
        return v instanceof Number n ? n.longValue() : def;
    }

    public static double dbl(Object o, String key, double def) {
        Object v = o instanceof Map<?, ?> m ? m.get(key) : null;
        return v instanceof Number n ? n.doubleValue() : def;
    }

    public static boolean bool(Object o, String key, boolean def) {
        Object v = o instanceof Map<?, ?> m ? m.get(key) : null;
        return v instanceof Boolean b ? b : def;
    }

    /** Builds an insertion-ordered map from alternating key/value arguments. */
    public static Map<String, Object> obj(Object... kv) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            m.put((String) kv[i], kv[i + 1]);
        }
        return m;
    }
}
