package org.robomiri.jllm.diffusion;

import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A small {@code multipart/form-data} reader for {@code POST /v1/images/edits} (the OpenAI SDKs send the image and the mask
 * as file parts next to text fields). Parts are kept in memory; the whole body is bounded by the server's body limit.
 */
public final class Multipart {

    /** One part: a text field ({@code filename == null}) or an uploaded file. */
    public record Part(String name, String filename, String contentType, byte[] data) {
        public String text() {
            return new String(data, StandardCharsets.UTF_8);
        }
    }

    private Multipart() {
    }

    /** The boundary of a {@code multipart/form-data; boundary=...} content type, or null if it is not one. */
    public static String boundary(String contentType) {
        if (contentType == null || !contentType.toLowerCase(java.util.Locale.ROOT).startsWith("multipart/form-data")) {
            return null;
        }
        for (String p : contentType.split(";")) {
            String t = p.trim();
            if (t.toLowerCase(java.util.Locale.ROOT).startsWith("boundary=")) {
                String b = t.substring(9).trim();
                if (b.length() >= 2 && b.startsWith("\"") && b.endsWith("\"")) {
                    b = b.substring(1, b.length() - 1);
                }
                return b;
            }
        }
        return null;
    }

    /** The parts by field name (a repeated name keeps the first part). */
    public static Map<String, Part> parse(byte[] body, String boundary) {
        byte[] delim = ("--" + boundary).getBytes(StandardCharsets.ISO_8859_1);
        Map<String, Part> out = new LinkedHashMap<>();
        int pos = indexOf(body, delim, 0);
        while (pos >= 0) {
            int start = pos + delim.length;
            if (start + 1 < body.length && body[start] == '-' && body[start + 1] == '-') {
                break; // closing delimiter
            }
            if (start + 1 < body.length && body[start] == '\r' && body[start + 1] == '\n') {
                start += 2;
            }
            int next = indexOf(body, delim, start);
            if (next < 0) {
                break;
            }
            int end = next;
            if (end >= 2 && body[end - 2] == '\r' && body[end - 1] == '\n') {
                end -= 2;
            }
            int headerEnd = indexOf(body, new byte[]{'\r', '\n', '\r', '\n'}, start);
            if (headerEnd >= 0 && headerEnd < end) {
                String headers = new String(body, start, headerEnd - start, StandardCharsets.UTF_8);
                String name = null;
                String filename = null;
                String type = null;
                for (String line : headers.split("\r\n")) {
                    String lower = line.toLowerCase(java.util.Locale.ROOT);
                    if (lower.startsWith("content-disposition:")) {
                        name = param(line, "name");
                        filename = param(line, "filename");
                    } else if (lower.startsWith("content-type:")) {
                        type = line.substring(13).trim();
                    }
                }
                if (name != null && !out.containsKey(name)) {
                    byte[] data = new byte[end - (headerEnd + 4)];
                    System.arraycopy(body, headerEnd + 4, data, 0, data.length);
                    out.put(name, new Part(name, filename, type, data));
                }
            }
            pos = next;
        }
        return out;
    }

    private static String param(String header, String key) {
        String k = key + "=";
        int i = 0;
        while (true) {
            i = header.indexOf(k, i);
            if (i < 0) {
                return null;
            }
            // a whole word: the previous character is a space or a semicolon (so "name=" does not match "filename=")
            if (i == 0 || header.charAt(i - 1) == ' ' || header.charAt(i - 1) == ';') {
                break;
            }
            i += k.length();
        }
        i += k.length();
        if (i < header.length() && header.charAt(i) == '"') {
            int j = header.indexOf('"', i + 1);
            return j < 0 ? null : header.substring(i + 1, j);
        }
        int j = header.indexOf(';', i);
        return (j < 0 ? header.substring(i) : header.substring(i, j)).trim();
    }

    private static int indexOf(byte[] a, byte[] b, int from) {
        outer:
        for (int i = from; i <= a.length - b.length; i++) {
            for (int j = 0; j < b.length; j++) {
                if (a[i + j] != b[j]) {
                    continue outer;
                }
            }
            return i;
        }
        return -1;
    }
}
