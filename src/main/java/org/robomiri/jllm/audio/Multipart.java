package org.robomiri.jllm.audio;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * A small {@code multipart/form-data} reader (RFC 7578) for the engine's own HTTP server: the audio endpoints take a
 * file and a few text fields. The web console gets the same parts from Spring instead.
 */
public final class Multipart {

    private Multipart() {
    }

    /** One part of the body; {@code filename} is null for a plain text field. */
    public record Part(String name, String filename, String contentType, byte[] data) {
        public String text() {
            return new String(data, StandardCharsets.UTF_8);
        }
    }

    /** True for a {@code Content-Type} header value of {@code multipart/form-data}. */
    public static boolean is(String contentType) {
        return contentType != null && contentType.toLowerCase(Locale.ROOT).startsWith("multipart/form-data");
    }

    /** @throws IllegalArgumentException for a missing boundary or a malformed body */
    public static List<Part> parse(String contentType, byte[] body) {
        String boundary = null;
        for (String p : contentType.split(";")) {
            String t = p.trim();
            if (t.toLowerCase(Locale.ROOT).startsWith("boundary=")) {
                boundary = t.substring(9).trim();
                if (boundary.length() >= 2 && boundary.startsWith("\"") && boundary.endsWith("\"")) {
                    boundary = boundary.substring(1, boundary.length() - 1);
                }
            }
        }
        if (boundary == null || boundary.isEmpty()) {
            throw new IllegalArgumentException("multipart body without a boundary");
        }
        byte[] delim = ("--" + boundary).getBytes(StandardCharsets.ISO_8859_1);
        List<Part> parts = new ArrayList<>();
        int pos = indexOf(body, delim, 0);
        if (pos < 0) {
            throw new IllegalArgumentException("multipart body: boundary not found");
        }
        while (true) {
            int start = pos + delim.length;
            if (start + 1 < body.length && body[start] == '-' && body[start + 1] == '-') {
                break;           // the closing delimiter
            }
            if (start + 1 >= body.length) {
                break;
            }
            if (body[start] == '\r' && body[start + 1] == '\n') {
                start += 2;
            } else if (body[start] == '\n') {
                start++;
            }
            int headersEnd = indexOf(body, new byte[]{'\r', '\n', '\r', '\n'}, start);
            int skip = 4;
            if (headersEnd < 0) {
                headersEnd = indexOf(body, new byte[]{'\n', '\n'}, start);
                skip = 2;
            }
            if (headersEnd < 0) {
                throw new IllegalArgumentException("multipart part without headers");
            }
            String headers = new String(body, start, headersEnd - start, StandardCharsets.UTF_8);
            int dataStart = headersEnd + skip;
            int next = indexOf(body, delim, dataStart);
            if (next < 0) {
                throw new IllegalArgumentException("multipart part is not terminated");
            }
            int dataEnd = next;
            if (dataEnd >= 2 && body[dataEnd - 2] == '\r' && body[dataEnd - 1] == '\n') {
                dataEnd -= 2;
            } else if (dataEnd >= 1 && body[dataEnd - 1] == '\n') {
                dataEnd--;
            }
            String name = null;
            String filename = null;
            String ctype = null;
            for (String line : headers.split("\r?\n")) {
                String l = line.toLowerCase(Locale.ROOT);
                if (l.startsWith("content-disposition:")) {
                    name = param(line, "name");
                    filename = param(line, "filename");
                } else if (l.startsWith("content-type:")) {
                    ctype = line.substring(13).trim();
                }
            }
            if (name != null) {
                byte[] data = new byte[dataEnd - dataStart];
                System.arraycopy(body, dataStart, data, 0, data.length);
                parts.add(new Part(name, filename, ctype, data));
            }
            pos = next;
        }
        return parts;
    }

    /** The first part called {@code name}, or null. */
    public static Part find(List<Part> parts, String name) {
        for (Part p : parts) {
            if (p.name().equals(name)) {
                return p;
            }
        }
        return null;
    }

    /** The text of the field {@code name}, or null. */
    public static String field(List<Part> parts, String name) {
        Part p = find(parts, name);
        return p == null ? null : p.text();
    }

    private static String param(String header, String key) {
        String k = key + "=\"";
        int i = header.toLowerCase(Locale.ROOT).indexOf(" " + k);
        if (i < 0) {
            i = header.toLowerCase(Locale.ROOT).indexOf(";" + k);
        }
        if (i < 0) {
            return null;
        }
        int s = i + 1 + k.length();
        StringBuilder sb = new StringBuilder();
        for (int j = s; j < header.length(); j++) {
            char c = header.charAt(j);
            if (c == '\\' && j + 1 < header.length()) {
                sb.append(header.charAt(++j));
            } else if (c == '"') {
                break;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private static int indexOf(byte[] a, byte[] pat, int from) {
        outer:
        for (int i = from; i <= a.length - pat.length; i++) {
            for (int j = 0; j < pat.length; j++) {
                if (a[i + j] != pat[j]) {
                    continue outer;
                }
            }
            return i;
        }
        return -1;
    }
}
