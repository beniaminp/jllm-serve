package org.robomiri.jllm.web.chat;

import org.robomiri.jllm.vision.ImageDecoder;
import org.robomiri.jllm.vision.Rgb;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Pictures the console's user uploaded for the tools to look at ({@code POST /api/images}): a text-only chat model
 * cannot take a picture itself, but it can call {@code describe_image} with the id of an uploaded one and have a
 * vision model answer. Kept in memory, least recently used first out, bounded in bytes.
 */
@Service
public class ImageStore {

    private static final long MAX_BYTES = 128L << 20;

    private final Map<String, Rgb> images = new LinkedHashMap<>(16, 0.75f, true);
    private long bytes;

    /** Decodes and keeps a picture; the id is the first 12 hex digits of its pixel hash. */
    public synchronized String put(byte[] file, long maxPixels) throws IOException {
        Rgb img = ImageDecoder.decode(file, maxPixels);
        String id = "img-" + img.hash().substring(0, 12);
        if (images.put(id, img) == null) {
            bytes += img.data().length;
        }
        Iterator<Map.Entry<String, Rgb>> it = images.entrySet().iterator();
        while (bytes > MAX_BYTES && images.size() > 1 && it.hasNext()) {
            Map.Entry<String, Rgb> e = it.next();
            if (!e.getKey().equals(id)) {
                bytes -= e.getValue().data().length;
                it.remove();
            }
        }
        return id;
    }

    /** The picture with this id, or null (unknown, or evicted). */
    public synchronized Rgb get(String id) {
        return images.get(id);
    }
}
