package org.robomiri.jllm.kv;

import java.nio.ByteBuffer;

/**
 * Where blocks go when the pool needs memory and they are not referenced by any sequence: a block store on disk.
 * A block is identified by the hash of its <em>path</em> (the tokens of every block from the root to it, see
 * {@link KvPool#pathHash}), so the pool can find a spilled block while walking a prompt without keeping a node for it,
 * and a store that survives a restart serves the next run of the same model (docs/24 §24.6).
 * Implementations must be thread safe.
 */
public interface KvSpill {

    /** Whether a block with this path hash is stored (a probable hit: {@link #get} verifies the tokens). */
    boolean has(long pathHash);

    /**
     * Stores the block {@code tokens} (the {@link KvPool#blockTokens()} tokens it holds) under {@code pathHash}.
     *
     * @return false when the store refuses (budget, I/O error): the block is then simply dropped
     */
    boolean put(long pathHash, int[] tokens, ByteBuffer payload);

    /**
     * Reads the block into {@code payload} (from its position, {@link KvStorage#spillBytes()} bytes) and marks it used.
     *
     * @return false when it is absent, unreadable, or was stored for other tokens (a hash collision)
     */
    boolean get(long pathHash, int[] tokens, ByteBuffer payload);

    /** Deletes the block, if stored. */
    void remove(long pathHash);

    /**
     * A hint that the block will be read soon (a prompt that will adopt it is being prepared): a store with several
     * tiers may start moving it to its fastest one in the background. Never blocks, never fails; the default does nothing.
     *
     * @param tokens       the block's tokens (the lower tiers compare them)
     * @param payloadBytes {@link KvStorage#spillBytes()}
     */
    default void prefetch(long pathHash, int[] tokens, int payloadBytes) {
    }

    /** Number of stored blocks. */
    int files();

    /** Bytes stored. */
    long bytes();
}
