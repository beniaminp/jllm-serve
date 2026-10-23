package org.robomiri.jllm.kv;

import java.nio.ByteBuffer;

/**
 * A {@link KvSpill} that stores the blocks it is given through a {@link KvBlockCodec} (smaller rows on the SSD) and
 * decodes them on the way back. Put it between the warm tier and the file store.
 */
public final class CodecSpill implements KvSpill {

    private final KvSpill lower;
    private final KvBlockCodec codec;

    public CodecSpill(KvSpill lower, KvBlockCodec codec) {
        this.lower = lower;
        this.codec = codec;
    }

    @Override
    public boolean has(long pathHash) {
        return lower.has(pathHash);
    }

    @Override
    public boolean put(long pathHash, int[] tokens, ByteBuffer payload) {
        if (codec.identity()) {
            return lower.put(pathHash, tokens, payload);
        }
        ByteBuffer enc = ByteBuffer.allocateDirect(codec.encodedBytes());
        codec.encode(payload, enc);
        enc.flip();
        return lower.put(pathHash, tokens, enc);
    }

    @Override
    public boolean get(long pathHash, int[] tokens, ByteBuffer payload) {
        if (codec.identity()) {
            return lower.get(pathHash, tokens, payload);
        }
        ByteBuffer enc = ByteBuffer.allocateDirect(codec.encodedBytes());
        if (!lower.get(pathHash, tokens, enc)) {
            return false;
        }
        enc.flip();
        codec.decode(enc, payload);
        return true;
    }

    @Override
    public void prefetch(long pathHash, int[] tokens, int payloadBytes) {
        lower.prefetch(pathHash, tokens, codec.identity() ? payloadBytes : codec.encodedBytes());
    }

    @Override
    public void remove(long pathHash) {
        lower.remove(pathHash);
    }

    @Override
    public int files() {
        return lower.files();
    }

    @Override
    public long bytes() {
        return lower.bytes();
    }
}
