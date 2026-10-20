package org.robomiri.jllm.web.tools;

import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Just enough HTML handling to turn a page into readable text for a model. */
public final class Html {

    private static final Pattern DROP = Pattern.compile(
            "(?is)<(script|style|noscript|svg|template|iframe|head|nav|footer|form)\\b[^>]*>.*?</\\1\\s*>|<!--.*?-->");
    private static final Pattern BLOCK = Pattern.compile(
            "(?i)</?(p|div|br|li|ul|ol|h[1-6]|tr|table|section|article|main|header|blockquote|pre|hr|dd|dt)\\b[^>]*>");
    private static final Pattern TAG = Pattern.compile("(?s)<[^>]*>");
    private static final Pattern ENTITY = Pattern.compile("&(#x?[0-9A-Fa-f]+|[A-Za-z]+);");
    private static final Map<String, String> NAMED = Map.ofEntries(
            Map.entry("amp", "&"), Map.entry("lt", "<"), Map.entry("gt", ">"), Map.entry("quot", "\""),
            Map.entry("apos", "'"), Map.entry("nbsp", " "), Map.entry("ndash", "–"), Map.entry("mdash", "—"),
            Map.entry("hellip", "…"), Map.entry("laquo", "«"), Map.entry("raquo", "»"), Map.entry("lsquo", "‘"),
            Map.entry("rsquo", "’"), Map.entry("ldquo", "“"), Map.entry("rdquo", "”"), Map.entry("copy", "©"),
            Map.entry("reg", "®"), Map.entry("trade", "™"), Map.entry("deg", "°"), Map.entry("middot", "·"),
            Map.entry("times", "×"), Map.entry("euro", "€"), Map.entry("pound", "£"), Map.entry("bull", "•"));

    private Html() {
    }

    /** The page's {@code <title>}, or empty. */
    public static String title(String html) {
        Matcher m = Pattern.compile("(?is)<title[^>]*>(.*?)</title>").matcher(html);
        return m.find() ? clean(m.group(1)) : "";
    }

    /** Visible text, one paragraph per line; {@code <main>} or {@code <article>} when the page has one. */
    public static String toText(String html) {
        String body = html;
        for (String tag : new String[]{"main", "article"}) {
            Matcher m = Pattern.compile("(?is)<" + tag + "\\b[^>]*>(.*)</" + tag + "\\s*>").matcher(html);
            if (m.find() && m.group(1).length() > 500) {
                body = m.group(1);
                break;
            }
        }
        body = DROP.matcher(body).replaceAll(" ");
        body = BLOCK.matcher(body).replaceAll("\n");
        body = TAG.matcher(body).replaceAll("");
        body = decode(body);
        StringBuilder out = new StringBuilder();
        for (String line : body.split("\n")) {
            String t = line.replaceAll("[ \\t\\x0B\\f\\r\\u00a0]+", " ").strip();
            if (!t.isEmpty()) {
                out.append(t).append('\n');
            }
        }
        return out.toString().strip();
    }

    /** Inline HTML (a title, a snippet) to plain text. */
    public static String clean(String fragment) {
        return decode(TAG.matcher(fragment).replaceAll("")).replaceAll("\\s+", " ").strip();
    }

    public static String decode(String s) {
        Matcher m = ENTITY.matcher(s);
        StringBuilder sb = new StringBuilder();
        while (m.find()) {
            String e = m.group(1);
            String rep;
            if (e.startsWith("#")) {
                try {
                    int cp = e.startsWith("#x") || e.startsWith("#X") ? Integer.parseInt(e.substring(2), 16) : Integer.parseInt(e.substring(1));
                    rep = Character.isValidCodePoint(cp) ? new String(Character.toChars(cp)) : m.group();
                } catch (NumberFormatException ex) {
                    rep = m.group();
                }
            } else {
                rep = NAMED.getOrDefault(e.toLowerCase(Locale.ROOT), m.group());
            }
            m.appendReplacement(sb, Matcher.quoteReplacement(rep));
        }
        m.appendTail(sb);
        return sb.toString();
    }
}
