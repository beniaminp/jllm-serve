package org.robomiri.jllm.kv;

import java.nio.ByteBuffer;

/**
 * The data side of a {@link KvPool}: where the keys and values of the blocks live. The pool decides which block id
 * holds what; the storage only knows how to make ids addressable, copy a block, and move a block's rows to and from
 * the canonical layout used for spilling (docs/24 §24.3).
 *
 * <p>One block is {@link #blockTokens()} consecutive positions of <em>every</em> layer, keys and values. The storage
 * chooses its own physical layout (the CPU backend: off-heap slabs with the keys transposed; the Metal backend: one
 * buffer per layer and kind, indexed by {@code block * blockTokens + row}); the canonical layout ({@link #save}) is
 * backend independent: per layer, per position of the block, the K row then the V row, exactly the rows
 * {@code Backend.saveState} writes for a saved cache, so spill files and saved conversations speak the same format.
 */
public interface KvStorage extends AutoCloseable {

    /** Positions per block. */
    int blockTokens();

    /** Bytes of one block in the canonical layout ({@link #save}). */
    int spillBytes();

    /** Bytes of memory one resident block occupies on the device (for statistics and budgets). */
    long residentBytes();

    /**
     * Makes the block ids {@code [0, blocks)} addressable. The pool calls it before it hands out an id at or above
     * the largest requested so far; a storage that reserves everything up front does nothing. The ids already
     * addressable keep their contents.
     */
    void grow(int blocks);

    /**
     * The codec that stores this storage's blocks in the on-disk format {@code f} (docs/26 §26.2.4), or null when the
     * storage cannot say (the block files then hold the canonical rows as they are).
     */
    default KvBlockCodec codec(KvBlockCodec.Format f) {
        return null;
    }

    /** Copies the first {@code rows} positions of every layer of block {@code src} into block {@code dst}. */
    void copy(int src, int dst, int rows);

    /** Writes {@link #spillBytes()} bytes at {@code dst}'s position (canonical layout) and advances it. */
    void save(int block, ByteBuffer dst);

    /** Reads what {@link #save} wrote into the block. */
    void load(int block, ByteBuffer src);

    @Override
    void close();
}
