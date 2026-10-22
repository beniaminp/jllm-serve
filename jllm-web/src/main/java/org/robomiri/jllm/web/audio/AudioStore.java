package org.robomiri.jllm.web.audio;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.UUID;

/**
 * Short-lived audio clips held in memory so that a chat (and its tools) can refer to a recording or a generated speech
 * by an id: the browser uploads a recording and the model's {@code transcribe_audio} tool names it; the {@code speak}
 * tool stores its result and the chat shows a player for it. Least recently used clips are dropped beyond a size
 * budget and after a time-to-live.
 */
public final class AudioStore {

    /** A stored clip: a complete WAV file. */
    public record Clip(String id, byte[] wav, int sampleRate, double seconds, String label, long created) {
    }

    private static final long MAX_BYTES = 512L << 20;
    private static final long TTL_MS = 6L * 3600_000;

    private final Map<String, Clip> clips = new LinkedHashMap<>(16, 0.75f, true);
    private long bytes;

    public synchronized Clip put(byte[] wav, int sampleRate, double seconds, String label) {
        Clip c = new Clip(UUID.randomUUID().toString().replace("-", "").substring(0, 16), wav, sampleRate, seconds, label,
                System.currentTimeMillis());
        clips.put(c.id(), c);
        bytes += wav.length;
        evict();
        return c;
    }

    public synchronized Clip get(String id) {
        Clip c = clips.get(id);
        if (c == null) {
            throw new NoSuchElementException("no audio clip " + id + " (clips are kept for a few hours)");
        }
        return c;
    }

    public synchronized boolean remove(String id) {
        Clip c = clips.remove(id);
        if (c != null) {
            bytes -= c.wav().length;
        }
        return c != null;
    }

    public synchronized int size() {
        return clips.size();
    }

    private void evict() {
        long now = System.currentTimeMillis();
        for (Iterator<Clip> it = clips.values().iterator(); it.hasNext(); ) {
            Clip c = it.next();
            if (now - c.created() > TTL_MS || (bytes > MAX_BYTES && clips.size() > 1)) {
                bytes -= c.wav().length;
                it.remove();
            } else if (bytes <= MAX_BYTES) {
                break;
            }
        }
    }
}
