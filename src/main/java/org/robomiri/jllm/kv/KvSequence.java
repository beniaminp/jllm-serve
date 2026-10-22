package org.robomiri.jllm.kv;

import java.util.Arrays;

/**
 * One slot's view of a {@link KvPool}: a <em>block table</em> mapping logical blocks of {@code blockTokens} positions to
 * physical block ids, plus the number of positions that hold valid rows. A backend owns exactly one. Mutating methods
 * are called by the thread that runs the slot (the engine's lock); they take the pool's monitor for everything that
 * touches shared state.
 *
 * <h3>The contract between a backend and its sequence</h3>
 * <ol>
 *   <li>Before a forward pass writes positions {@code [pos, pos + n)}, the backend calls {@link #beginWrite}. It
 *       releases the blocks wholly after {@code pos} (positions past the new write are invalid anyway: speculation
 *       rollback, a retried prompt), makes the block that contains {@code pos} private when someone else can see it
 *       (copy on write), and allocates the blocks up to {@code pos + n - 1}. After it every table entry
 *       {@code [0, ceil((pos + n) / B))} is a block this sequence may read, and those covering {@code [pos, pos + n)} are
 *       private, so kernels write straight into them.</li>
 *   <li>Kernels address position {@code p} at row {@code p % B} of the physical block {@code block(p / B)}. The table is
 *       the only indirection; {@link #table()} hands out the live array and a {@link TableSink} mirrors every change into
 *       a device buffer (the Metal backend), so a recorded replay needs nothing patched but positions.</li>
 *   <li>The engine calls {@link #match} / {@link #adopt} before prefilling a prompt (the tree may hold a longer prefix
 *       than this slot's own cache) and {@link #publish} when tokens are final (after a prefill, at the end of a
 *       request), which makes full blocks shareable and immutable.</li>
 * </ol>
 */
public final class KvSequence implements AutoCloseable {

    /** Told about every change of the table, so a device copy can follow. */
    @FunctionalInterface
    public interface TableSink {
        void set(int logical, int block);
    }

    private final KvPool pool;
    private final int B;
    private final int maxPositions;
    private int[] table;
    private int blocks;
    private int length;
    private boolean shareable = true;
    private volatile TableSink sink;

    KvSequence(KvPool pool, int maxPositions) {
        this.pool = pool;
        this.B = pool.blockTokens();
        this.maxPositions = maxPositions;
        this.table = new int[(maxPositions + B - 1) / B];
    }

    public KvPool pool() {
        return pool;
    }

    public int blockTokens() {
        return B;
    }

    /** Most positions the sequence can hold. */
    public int maxPositions() {
        return maxPositions;
    }

    /** Positions whose rows are valid (the end of the last {@link #beginWrite}). */
    public int length() {
        return length;
    }

    /** Logical blocks in the table. */
    public int blocks() {
        return blocks;
    }

    /** The physical block of logical block {@code i} ({@code i < blocks()}). */
    public int block(int i) {
        return table[i];
    }

    int entry(int i) {
        return table[i];
    }

    /** The live table (entries {@code [0, blocks())} are valid): read-only for callers. */
    public int[] table() {
        return table;
    }

    /** Mirrors table changes (including the entries already there) into a device buffer. */
    public synchronized void setSink(TableSink sink) {
        this.sink = sink;
        if (sink != null) {
            for (int i = 0; i < blocks; i++) {
                sink.set(i, table[i]);
            }
        }
    }

    /**
     * Whether this sequence takes part in sharing: when false {@link #adopt} finds nothing and {@link #publish} does
     * nothing. Off for a sequence whose rows do not depend on the tokens alone (pictures, custom rotary positions).
     */
    public boolean shareable() {
        return shareable;
    }

    public void setShareable(boolean on) {
        this.shareable = on;
    }

    // ------------------------------------------------------------------ writing

    /**
     * Prepares the table for a forward pass that writes positions {@code [pos, pos + n)} and declares the sequence's
     * valid positions to be {@code [0, pos + n)} afterwards.
     */
    public void beginWrite(int pos, int n) {
        if (pos < 0 || n < 0 || pos + n > maxPositions) {
            throw new IllegalArgumentException("positions [" + pos + ", " + (pos + n) + ") do not fit a sequence of " + maxPositions);
        }
        synchronized (pool) {
            int keep = (pos + B - 1) / B;
            while (blocks > keep) {
                pool.release(table[--blocks]);
            }
            int had = blocks;
            int need = (pos + n + B - 1) / B;
            while (blocks < need) {
                int b = pool.alloc();
                table[blocks] = b;
                notifySink(blocks, b);
                blocks++;
            }
            int j = pos / B;
            if (pos % B != 0 && j < had) { // the block that holds [.., pos) keeps those rows and takes the new ones
                int b = table[j];
                if (pool.shared(b)) {
                    int nb = pool.alloc();
                    pool.copyRows(b, nb, pos % B);
                    pool.release(b);
                    table[j] = nb;
                    notifySink(j, nb);
                }
            }
            length = pos + n;
        }
    }

    /** Forgets the positions {@code >= n}: whole blocks beyond are released. */
    public void truncate(int n) {
        synchronized (pool) {
            int keep = (n + B - 1) / B;
            while (blocks > keep) {
                pool.release(table[--blocks]);
            }
            length = Math.min(length, n);
        }
    }

    /** Releases every block. */
    public void clear() {
        truncate(0);
    }

    @Override
    public void close() {
        clear();
    }

    // ------------------------------------------------------------------ sharing

    /**
     * How many leading tokens of {@code tokens[0 .. max)} the pool's tree (and spill store) can supply. A snapshot:
     * eviction by another slot may change it, so {@link #adopt} may return less.
     */
    public int match(int[] tokens, int max) {
        return shareable ? pool.match(tokens, max).tokens() : 0;
    }

    /** {@link #match} with the part that has to come back from disk. */
    public KvPool.Match matchDetail(int[] tokens, int max) {
        return shareable ? pool.match(tokens, max) : new KvPool.Match(0, 0);
    }

    /**
     * Makes this sequence hold the rows of {@code tokens[0 .. n)} as the tree has them (shared blocks are retained, not
     * copied; a last partial block is copied). Blocks the sequence holds beyond are released.
     *
     * @return the number of leading positions now valid, at most {@code n}; 0 leaves the sequence untouched
     */
    public int adopt(int[] tokens, int n) {
        if (!shareable) {
            return 0;
        }
        return pool.adopt(this, tokens, Math.min(n, maxPositions));
    }

    /** Makes the full blocks holding {@code tokens[0 .. n)} (those below {@link #length}) shareable. */
    public void publish(int[] tokens, int n) {
        pool.publish(this, tokens, Math.min(n, length));
    }

    /** Takes over the first {@code n} positions of {@code o}'s blocks (retained: a copy is made on the first write). */
    public void shareFrom(KvSequence o, int n) {
        if (o.pool != pool) {
            throw new IllegalArgumentException("sequences of different pools cannot share blocks");
        }
        synchronized (pool) {
            if (o == this) {
                return;
            }
            int need = (n + B - 1) / B;
            for (int i = 0; i < need; i++) {
                int b = o.table[i];
                if (i < blocks && table[i] == b) {
                    continue;
                }
                pool.retain(b);
                put(i, b);
            }
            dropFrom(need);
            length = n;
        }
    }

    /**
     * A second sequence that holds the first {@code n} positions of this one's blocks (retained: a writer copies a shared
     * block first), so another thread can read them (a background save) while this sequence goes on. Close it when done.
     */
    public KvSequence snapshot(int n) {
        KvSequence copy = pool.newSequence(Math.max(n, 1));
        copy.shareFrom(this, n);
        return copy;
    }

    // ------------------------------------------------------------------ pausing a generation (docs/25)

    /**
     * What {@link #suspend} leaves of a paused sequence: the number of valid positions and the one block that cannot be
     * found again through the tree, the partly filled last one (its reference is kept).
     */
    public static final class Suspended {
        private final int positions;
        private final int fullBlocks;
        private final int tail;

        private Suspended(int positions, int fullBlocks, int tail) {
            this.positions = positions;
            this.fullBlocks = fullBlocks;
            this.tail = tail;
        }

        /** Positions that were valid. */
        public int positions() {
            return positions;
        }
    }

    /**
     * Whether {@link #suspend} can give the blocks back and {@link #resume} find them again: the sequence shares through
     * the tree (rows that are a function of the tokens alone) and the pool has sharing on.
     */
    public boolean suspendable() {
        return shareable && pool.sharing();
    }

    /**
     * Pauses the sequence at {@code n} valid positions (tokens {@code tokens[0 .. n)}): its full blocks are published and
     * released, so they are ordinary cached blocks that the pool may evict (and spill to disk) when it needs room, while
     * another conversation runs; only the last partly filled block, at most {@code B - 1} rows, is kept. The table is empty
     * afterwards; {@link #resume} brings the rows back.
     */
    public Suspended suspend(int[] tokens, int n) {
        synchronized (pool) {
            if (n > length) {
                throw new IllegalArgumentException("cannot suspend " + n + " positions of a sequence with " + length);
            }
            truncate(n);
            publish(tokens, n);
            int full = n / B;
            int tail = n % B != 0 ? table[full] : -1;
            for (int i = 0; i < full; i++) {
                pool.release(table[i]);
            }
            blocks = 0;
            length = 0;
            return new Suspended(n, full, tail);
        }
    }

    /**
     * Takes the rows of a {@link #suspend}ed sequence back: the full blocks come from the tree (reading spilled ones from
     * disk) and the kept tail block is put after them.
     *
     * @return the positions valid now: {@code s.positions()} when everything came back, else fewer (a block was evicted
     *         without a store, or the pool is full), and the caller computes the rest again from the tokens
     */
    public int resume(Suspended s, int[] tokens) {
        synchronized (pool) {
            int got = s.fullBlocks == 0 ? 0 : pool.adopt(this, tokens, s.fullBlocks * B);
            if (got == s.fullBlocks * B && s.tail >= 0) {
                put(s.fullBlocks, s.tail);
                length = s.positions;
                return length;
            }
            if (s.tail >= 0) {
                pool.release(s.tail);
            }
            return got;
        }
    }

    // ------------------------------------------------------------------ used by KvPool (under its monitor)

    /** Sets logical block {@code i} (an existing entry or the next one), releasing what was there. */
    void put(int i, int b) {
        if (i < blocks) {
            pool.release(table[i]);
        } else if (i == blocks) {
            blocks++;
        } else {
            throw new IllegalStateException("table gap at " + i + " (" + blocks + " blocks)");
        }
        table[i] = b;
        notifySink(i, b);
    }

    /** Releases the entries {@code >= n}. */
    void dropFrom(int n) {
        while (blocks > n) {
            pool.release(table[--blocks]);
        }
    }

    void setLength(int n) {
        length = n;
    }

    private void notifySink(int i, int b) {
        TableSink s = sink;
        if (s != null) {
            s.set(i, b);
        }
    }
}
