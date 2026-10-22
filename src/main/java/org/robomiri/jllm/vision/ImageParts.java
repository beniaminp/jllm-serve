package org.robomiri.jllm.vision;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The picture parts of an OpenAI chat request. A message's {@code content} may be an array of parts:
 * {@code {"type":"text","text":...}} and {@code {"type":"image_url","image_url":{"url":...}}} (a {@code data:} URL
 * with base64, or an {@code http(s)} URL the server fetches). {@link #extract} turns each message with parts into a
 * plain string content in which every picture is a {@link MediaChat#MARK}, in reading order, and returns the decoded
 * pictures; the rest of the request handling is unchanged.
 */
public final class ImageParts {

    /** Pictures one request may carry. */
    public static final int MAX_IMAGES = 16;

    /** A request error that is the client's to fix (the servers answer it with HTTP 400). */
    public static final class BadImage extends IllegalArgumentException {
        public BadImage(String message, Throwable cause) {
            super(message, cause);
        }

        public BadImage(String message) {
            super(message);
        }
    }

    /** @param messages the request's messages with picture parts replaced by marks; {@code images} in the order of the marks */
    public record Extracted(List<Object> messages, List<Rgb> images) {
    }

    private ImageParts() {
    }

    /** Whether any message has an {@code image_url} part (checked without decoding). */
    public static boolean hasImages(List<Object> messages) {
        for (Object m : messages) {
            if (m instanceof Map<?, ?> mm && mm.get("content") instanceof List<?> parts) {
                for (Object p : parts) {
                    if (p instanceof Map<?, ?> pm && ("image_url".equals(pm.get("type")) || "input_image".equals(pm.get("type")))) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * @param fetcher     downloads {@code http(s)} URLs; null refuses them
     * @param maxBytes    largest picture file accepted
     * @param maxPixels   largest decoded picture
     * @throws BadImage for an unreadable picture, a URL that cannot be fetched, an image in a message that is not the
     *                  user's, too many pictures, or an unknown content part type
     */
    public static Extracted extract(List<Object> messages, ImageFetcher fetcher, long maxBytes, long maxPixels) {
        List<Object> out = new ArrayList<>(messages.size());
        List<Rgb> images = new ArrayList<>();
        for (Object m : messages) {
            if (!(m instanceof Map<?, ?> mm)) {
                out.add(m);
                continue;
            }
            Object content = mm.get("content");
            if (content instanceof String s) {
                if (s.contains(MediaChat.MARK)) { // a client cannot forge a picture
                    Map<String, Object> copy = copy(mm);
                    copy.put("content", MediaChat.stripMarks(s));
                    out.add(copy);
                } else {
                    out.add(m);
                }
                continue;
            }
            if (!(content instanceof List<?> parts)) {
                out.add(m);
                continue;
            }
            StringBuilder sb = new StringBuilder();
            String role = String.valueOf(mm.get("role"));
            for (Object p : parts) {
                String type = p instanceof Map<?, ?> ? Json.str(p, "type", "text") : "?";
                switch (type) {
                    case "text" -> sb.append(MediaChat.stripMarks(Json.str(p, "text", "")));
                    case "image_url", "input_image" -> {
                        if (!role.equals("user")) {
                            throw new BadImage("images can only be sent in user messages (got one in a \"" + role + "\" message)");
                        }
                        if (images.size() >= MAX_IMAGES) {
                            throw new BadImage("at most " + MAX_IMAGES + " images per request");
                        }
                        images.add(load(url(p), fetcher, maxBytes, maxPixels));
                        sb.append(MediaChat.MARK);
                    }
                    default -> throw new BadImage("message content parts must be of type \"text\" or \"image_url\" (got \""
                            + type + "\"); audio and files are not supported");
                }
            }
            Map<String, Object> copy = copy(mm);
            copy.put("content", sb.toString());
            out.add(copy);
        }
        return new Extracted(out, images);
    }

    private static Map<String, Object> copy(Map<?, ?> m) {
        Map<String, Object> c = new LinkedHashMap<>();
        m.forEach((k, v) -> c.put(String.valueOf(k), v));
        return c;
    }

    /** {@code image_url} is a string or {@code {"url": ..., "detail": ...}}. */
    private static String url(Object part) {
        Object v = ((Map<?, ?>) part).get("image_url");
        if (v == null) {
            v = ((Map<?, ?>) part).get("url");
        }
        if (v instanceof Map<?, ?> vm) {
            v = vm.get("url");
        }
        if (!(v instanceof String s) || s.isBlank()) {
            throw new BadImage("an image_url part needs a \"url\"");
        }
        return s.strip();
    }

    /** Decodes a {@code data:} URL or fetches an {@code http(s)} one. */
    public static Rgb load(String url, ImageFetcher fetcher, long maxBytes, long maxPixels) {
        try {
            byte[] bytes;
            if (url.regionMatches(true, 0, "data:", 0, 5)) {
                bytes = ImageDecoder.dataUrlBytes(url);
            } else if (fetcher != null) {
                bytes = fetcher.fetch(url);
            } else {
                throw new BadImage("this server does not fetch image URLs; send the picture as a data: URL");
            }
            if (maxBytes > 0 && bytes.length > maxBytes) {
                throw new BadImage("the image is " + bytes.length + " bytes, the limit is " + maxBytes);
            }
            return ImageDecoder.decode(bytes, maxPixels > 0 ? maxPixels : ImageDecoder.MAX_PIXELS);
        } catch (IOException e) {
            throw new BadImage("cannot load the image" + (url.startsWith("data:") ? "" : " " + url) + ": " + e.getMessage(), e);
        }
    }
}
