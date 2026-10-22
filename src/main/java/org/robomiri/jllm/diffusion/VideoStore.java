package org.robomiri.jllm.diffusion;

import java.security.SecureRandom;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The generated clips a server keeps for a while so that {@code response_format: "url"} can point at them
 * ({@code GET /v1/videos/files/<id>.<ext>}), and so that the chat tool can show a clip inline: a bounded in-memory LRU
 * (by count and by bytes).
 */
public final class VideoStore {

    /** A kept clip. */
    public record Clip(byte[] data, String ext, String mime) {
    }

    private final int maxClips;
    private final long maxBytes;
    private final SecureRandom random = new SecureRandom();
    private final LinkedHashMap<String, Clip> clips = new LinkedHashMap<>(16, 0.75f, true);
    private long bytes;

    public VideoStore(int maxClips, long maxBytes) {
        this.maxClips = maxClips;
        this.maxBytes = maxBytes;
    }

    /** Keeps a clip and returns its id (an unguessable token: the URL is the capability to read it). */
    public synchronized String put(byte[] data, VideoEncoder.Format format) {
        byte[] id = new byte[12];
        random.nextBytes(id);
        String key = java.util.HexFormat.of().formatHex(id);
        clips.put(key, new Clip(data, format.ext, format.mime));
        bytes += data.length;
        while (clips.size() > maxClips || (bytes > maxBytes && clips.size() > 1)) {
            Map.Entry<String, Clip> eldest = clips.entrySet().iterator().next();
            bytes -= eldest.getValue().data().length;
            clips.remove(eldest.getKey());
        }
        return key;
    }

    /** The clip with this id, or null (unknown or evicted). */
    public synchronized Clip get(String id) {
        return clips.get(id);
    }

    public synchronized int size() {
        return clips.size();
    }
}
