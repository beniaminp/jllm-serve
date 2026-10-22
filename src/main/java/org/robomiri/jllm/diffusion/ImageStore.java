package org.robomiri.jllm.diffusion;

import java.security.SecureRandom;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The generated images a server keeps for a while so that {@code response_format: "url"} can point at them
 * ({@code GET /v1/images/files/<id>.png}), and so that the chat tools can show a picture inline: a bounded in-memory
 * LRU (by count and by bytes).
 */
public final class ImageStore {

    private final int maxImages;
    private final long maxBytes;
    private final SecureRandom random = new SecureRandom();
    private final LinkedHashMap<String, byte[]> images = new LinkedHashMap<>(16, 0.75f, true);
    private long bytes;

    public ImageStore(int maxImages, long maxBytes) {
        this.maxImages = maxImages;
        this.maxBytes = maxBytes;
    }

    /** Keeps a PNG and returns its id (an unguessable token: the URL is the capability to read it). */
    public synchronized String put(byte[] png) {
        byte[] id = new byte[12];
        random.nextBytes(id);
        String key = java.util.HexFormat.of().formatHex(id);
        images.put(key, png);
        bytes += png.length;
        while (images.size() > maxImages || (bytes > maxBytes && images.size() > 1)) {
            Map.Entry<String, byte[]> eldest = images.entrySet().iterator().next();
            bytes -= eldest.getValue().length;
            images.remove(eldest.getKey());
        }
        return key;
    }

    /** The PNG with this id, or null (unknown or evicted). */
    public synchronized byte[] get(String id) {
        return images.get(id);
    }

    public synchronized int size() {
        return images.size();
    }
}
