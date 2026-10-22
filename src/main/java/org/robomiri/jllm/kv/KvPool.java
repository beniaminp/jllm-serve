package org.robomiri.jllm.kv;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.TreeSet;

/**
 * A model's KV memory as a pool of fixed-size blocks, with a prefix tree over the tokens they hold (docs/24).
 *
 * <p>A <em>block</em> holds {@link #blockTokens()} consecutive positions of every layer. A {@link KvSequence} (one per
 * backend slot) is a <em>block table</em>: logical block {@code i} (positions {@code [i * B, (i + 1) * B)}) lives in
 * physical block {@code table[i]}. A block with <b>reference count</b> {@code refs} is in {@code refs} tables. A block
 * that holds a full block's worth of tokens can be <b>published</b> into the <b>prefix tree</b>, a trie over token ids
 * with one node per full block: the node reached by the tokens {@code t[0 .. (i + 1) * B)} owns the block whose
 * rows were computed from exactly those tokens (keys and values depend on the whole prefix, so a block is identified by
 * its path, not by its own tokens). Published blocks are immutable: a sequence that must write into one first makes a
 * private copy (<b>copy on write</b>), so identical prefixes are stored once across slots and conversations, and
 * branches (a retry, a subagent forking from one context) share everything up to where they diverge.
 *
 * <p>A block is in one of four states: <em>used</em> ({@code refs > 0}), <em>cached</em> ({@code refs == 0}, in the tree,
 * evictable), <em>free</em>, or never handed out. When a block is needed and none is free, the least recently used
 * evictable block <em>with no resident children</em> goes: it is written to the {@link KvSpill} store if there is one
 * (and found again by the hash of its path), else dropped. Ids are never renumbered, so a kernel may cache them
 * while a sequence holds the block.
 *
 * <p>Thread safety: every method of the pool and of its sequences' table changes runs under the pool's monitor, so
 * slots on different threads may share one pool; a sequence itself is used by one thread at a time (its engine's
 * lock). Reading or writing the rows of a block the sequence holds needs no lock: it is referenced (so not evicted) and
 * either private (only that sequence writes it) or immutable.
 */
public final class KvPool implements AutoCloseable {

    /** Counters for the console and the tests. */
    public record Stats(int blockTokens, int maxBlocks, int residentBlocks, int usedBlocks, int cachedBlocks,
                        int freeBlocks, long residentBytes, long evictions, long spills, long reloads, long copiesOnWrite,
                        long dedups, int spillFiles, long spillBytes) {
    }

    /** What {@link #match} found: {@code tokens} leading tokens in the tree, of which {@code spilled} are on disk. */
    public record Match(int tokens, int spilled) {
    }

    private static final long ROOT_HASH = 0x2545F4914F6CDD1DL;

    private final class Node {
        final Node parent;
        final int[] tokens;
        final long hash;
        final int depth;
        final long serial = ++serials;
        int block;
        long tick;
        int resChildren;
        HashMap<Integer, ArrayList<Node>> kids;

        Node(Node parent, int[] tokens, long hash, int block) {
            this.parent = parent;
            this.tokens = tokens;
            this.hash = hash;
            this.depth = parent == null ? 0 : parent.depth + 1;
            this.block = block;
        }

        Node find(int[] t, int off) {
            if (kids == null) {
                return null;
            }
            ArrayList<Node> l = kids.get(t[off]);
            if (l != null) {
                for (Node c : l) {
                    if (Arrays.equals(c.tokens, 0, B, t, off, off + B)) {
                        return c;
                    }
                }
            }
            return null;
        }

        void add(Node c) {
            if (kids == null) {
                kids = new HashMap<>();
            }
            kids.computeIfAbsent(c.tokens[0], k -> new ArrayList<>(1)).add(c);
            resChildren++;
        }

        void remove(Node c) {
            ArrayList<Node> l = kids.get(c.tokens[0]);
            l.remove(c);
            if (l.isEmpty()) {
                kids.remove(c.tokens[0]);
            }
            resChildren--;
        }
    }

    private final KvStorage storage;
    private final int B;
    private final int maxBlocks;
    private final int[] refs;
    private final Node[] owner;
    private int highWater;
    private int[] free = new int[64];
    private int freeCount;
    private int used;
    private int cached;
    private long tick;
    private long serials;
    private final Node root = new Node(null, new int[0], ROOT_HASH, -1);
    /** Cached blocks whose node has no resident child, oldest first (ties: deepest first). */
    private final TreeSet<Node> evictable = new TreeSet<>((a, b) -> a.tick != b.tick ? Long.compare(a.tick, b.tick)
            : a.depth != b.depth ? Integer.compare(b.depth, a.depth) : Long.compare(a.serial, b.serial));
    private volatile KvSpill spill;
    private volatile boolean sharing = true;
    private ByteBuffer spillBuf;
    private long evictions;
    private long spills;
    private long reloads;
    private long cows;
    private long dedups;
    private final String format;

    /**
     * @param storage   the block memory
     * @param maxBlocks most blocks that are ever resident at once
     * @param format    a description of the row layout (dtype, layers, widths): spill files of another format are not read
     */
    public KvPool(KvStorage storage, int maxBlocks, String format) {
        if (maxBlocks < 1) {
            throw new IllegalArgumentException("a KV pool needs at least one block");
        }
        this.storage = storage;
        this.B = storage.blockTokens();
        this.maxBlocks = maxBlocks;
        this.refs = new int[maxBlocks];
        this.owner = new Node[maxBlocks];
        this.format = format;
    }

    public int blockTokens() {
        return B;
    }

    public int maxBlocks() {
        return maxBlocks;
    }

    public KvStorage storage() {
        return storage;
    }

    /** The row layout this pool stores ({@link KvStorage} implementations describe dtype, layers and widths). */
    public String format() {
        return format;
    }

    /** A sequence (block table) for positions {@code [0, maxPositions)}. */
    public KvSequence newSequence(int maxPositions) {
        return new KvSequence(this, maxPositions);
    }

    /**
     * Asked for memory when every block is held by a sequence and none can be evicted: frees blocks held by sequences
     * nobody is using (an idle slot's conversation, which stays findable in the tree) and says whether it freed any.
     * Called with the pool's monitor held, so it must not wait for other locks (try-lock only).
     */
    @FunctionalInterface
    public interface Reclaimer {
        boolean reclaim();
    }

    private volatile Reclaimer reclaimer;

    /** Registers who can give blocks back under memory pressure ({@link Reclaimer}); null = nobody. */
    public void setReclaimer(Reclaimer r) {
        this.reclaimer = r;
    }

    /** Where evicted blocks go; null = they are dropped. */
    public void setSpill(KvSpill spill) {
        this.spill = spill;
    }

    public KvSpill spill() {
        return spill;
    }

    /** Whether sequences adopt blocks of the tree and publish their own ({@code backend.kv-prefix-share}). */
    public void setSharing(boolean on) {
        this.sharing = on;
    }

    public boolean sharing() {
        return sharing;
    }

    // ------------------------------------------------------------------ hashing

    /** The hash of the path that reaches the block holding {@code tokens[off .. off + n)} from the node with hash {@code parent}. */
    public static long pathHash(long parent, int[] tokens, int off, int n) {
        long h = parent;
        for (int i = off; i < off + n; i++) {
            h = (h ^ (tokens[i] + 0x9E3779B97F4A7C15L)) * 0xFF51AFD7ED558CCDL;
            h ^= h >>> 32;
        }
        return h * 0xC4CEB9FE1A85EC53L ^ (h >>> 29);
    }

    /** The hash of the root of the path (the empty prefix), for spill stores that want to derive file names. */
    public static long rootHash() {
        return ROOT_HASH;
    }

    // ------------------------------------------------------------------ blocks

    /** A fresh block with one reference (the caller's), private. Evicts the least recently used cached block if needed. */
    synchronized int alloc() {
        int b;
        for (;;) {
            if (freeCount > 0) {
                b = free[--freeCount];
            } else if (highWater < maxBlocks) {
                b = highWater++;
                storage.grow(highWater);
            } else {
                b = evictOne();
                if (b < 0) {
                    Reclaimer rc = reclaimer;
                    if (rc != null && rc.reclaim()) { // idle holders gave blocks back: start over (they may be free ones)
                        continue;
                    }
                    throw new KvPoolExhausted("the KV pool is full: all " + maxBlocks + " blocks of " + B + " positions are in"
                            + " use by running sequences; raise backend.kv-pool-tokens or lower the context / slots");
                }
            }
            break;
        }
        refs[b] = 1;
        owner[b] = null;
        used++;
        return b;
    }

    synchronized void retain(int b) {
        if (refs[b]++ == 0 && owner[b] != null) {
            cached--;
            used++;
            evictable.remove(owner[b]);
        }
    }

    synchronized void release(int b) {
        if (--refs[b] > 0) {
            return;
        }
        used--;
        Node n = owner[b];
        if (n != null) {
            cached++;
            if (n.resChildren == 0) {
                evictable.add(n);
            }
        } else {
            if (freeCount == free.length) {
                free = Arrays.copyOf(free, freeCount * 2);
            }
            free[freeCount++] = b;
        }
    }

    /** Whether writing into the block would be visible to someone else: referenced twice or published. */
    synchronized boolean shared(int b) {
        return refs[b] > 1 || owner[b] != null;
    }

    synchronized int refs(int b) {
        return refs[b];
    }

    synchronized void copyRows(int src, int dst, int rows) {
        storage.copy(src, dst, rows);
        cows++;
    }

    private ByteBuffer spillBuffer() {
        if (spillBuf == null) {
            spillBuf = ByteBuffer.allocateDirect(storage.spillBytes()).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        }
        spillBuf.clear();
        return spillBuf;
    }

    /** Takes the least recently used evictable block (writing it to the spill store), or -1 if none can go. */
    private int evictOne() {
        Node n = evictable.pollFirst();
        if (n == null) {
            return -1;
        }
        int b = n.block;
        KvSpill sp = spill;
        if (sp != null) {
            ByteBuffer buf = spillBuffer();
            storage.save(b, buf);
            buf.flip();
            if (sp.put(n.hash, n.tokens, buf)) {
                spills++;
            }
        }
        Node p = n.parent;
        p.remove(n);
        if (p != root && p.resChildren == 0 && refs[p.block] == 0) {
            evictable.add(p);
        }
        owner[b] = null;
        cached--;
        evictions++;
        return b;
    }

    /** Makes {@code c} a resident child of {@code parent}, which can no longer be evicted before it. */
    private void attach(Node parent, Node c) {
        if (parent != root) {
            evictable.remove(parent);
        }
        parent.add(c);
    }

    private void touch(Node n, long t) {
        if (n.tick == t) {
            return;
        }
        boolean in = evictable.remove(n);
        n.tick = t;
        if (in) {
            evictable.add(n);
        }
    }

    // ------------------------------------------------------------------ the tree

    /** The longest prefix of {@code tokens[0 .. max)} the tree (and the spill store) holds, to the token. */
    synchronized Match match(int[] tokens, int max) {
        if (!sharing || max <= 0) {
            return new Match(0, 0);
        }
        long t = ++tick;
        Node node = root;
        int i = 0;
        while (i + B <= max) {
            Node c = node.find(tokens, i);
            if (c == null) {
                break;
            }
            touch(c, t);
            node = c;
            i += B;
        }
        KvSpill sp = spill;
        int spilled = 0;
        if (sp != null) {
            long h = node.hash;
            while (i + B <= max) {
                long ph = pathHash(h, tokens, i, B);
                if (!sp.has(ph)) {
                    break;
                }
                h = ph;
                i += B;
                spilled += B;
            }
        }
        if (spilled == 0 && i < max) { // the rest may share the start of a resident block: copy on adopt
            int l = bestPartial(node, tokens, i, Math.min(B, max - i), null);
            i += l;
        }
        return new Match(i, spilled);
    }

    /** Length of the longest common start of {@code tokens[off .. off + want)} and a child of {@code node}; the child goes to {@code out[0]}. */
    private int bestPartial(Node node, int[] tokens, int off, int want, Node[] out) {
        if (node.kids == null || want <= 0) {
            return 0;
        }
        ArrayList<Node> l = node.kids.get(tokens[off]);
        if (l == null) {
            return 0;
        }
        int best = 0;
        for (Node c : l) {
            int k = Arrays.mismatch(c.tokens, 0, B, tokens, off, off + want);
            k = k < 0 ? want : k;
            if (k > best) {
                best = k;
                if (out != null) {
                    out[0] = c;
                }
            }
        }
        return best;
    }

    /**
     * Makes {@code s}'s table the path of {@code tokens[0 .. n)}: the blocks of the tree (retained; spilled ones are
     * read back into fresh blocks) and, for a last part that is less than a block or shares only the start of one, a
     * private copy of that block's first rows. The sequence's own blocks that already are the tree's stay as they are.
     *
     * @return how many leading tokens the table holds now (at most {@code n}, fewer if the pool or a spill file failed)
     */
    synchronized int adopt(KvSequence s, int[] tokens, int n) {
        if (!sharing || !s.shareable() || n <= 0) {
            return 0;
        }
        long t = ++tick;
        Node node = root;
        int i = 0;
        int full = n / B;
        while (i < full) {
            Node c = node.find(tokens, i * B);
            if (c == null) {
                break;
            }
            touch(c, t);
            place(s, i, c.block, true);
            node = c;
            i++;
        }
        // what the sequence holds past the shared part is replaced anyway: free it first, so that reading spilled blocks
        // back can evict it (a full pool whose only evictable blocks are this sequence's old tail would run out)
        s.dropFrom(i);
        KvSpill sp = spill;
        while (i < full && sp != null) {
            int off = i * B;
            long ph = pathHash(node.hash, tokens, off, B);
            if (!sp.has(ph)) {
                break;
            }
            int b;
            try {
                b = alloc();
            } catch (KvPoolExhausted e) {
                break;
            }
            ByteBuffer buf = spillBuffer();
            int[] tk = Arrays.copyOfRange(tokens, off, off + B);
            if (!sp.get(ph, tk, buf)) {
                sp.remove(ph);
                dropRaw(b);
                break;
            }
            buf.flip();
            storage.load(b, buf);
            reloads++;
            sp.remove(ph);
            Node c = new Node(node, tk, ph, b);
            c.tick = t;
            owner[b] = c;
            attach(node, c);
            place(s, i, b, false); // takes over alloc's reference
            node = c;
            i++;
        }
        int got = i * B;
        int want = Math.min(B, n - got);
        Node[] cand = new Node[1];
        int l = bestPartial(node, tokens, got, want, cand);
        if (l > 0) {
            Node c = cand[0];
            touch(c, t);
            retain(c.block); // alloc below may evict: keep the source
            try {
                int b = alloc();
                storage.copy(c.block, b, l);
                cows++;
                place(s, i, b, false);
                got += l;
                i++;
            } catch (KvPoolExhausted e) {
                // adopt what is whole
            } finally {
                release(c.block);
            }
        }
        s.dropFrom(i);
        s.setLength(got);
        return got;
    }

    /** Puts physical block {@code b} at logical {@code i} of {@code s}; {@code retain} when the caller holds no reference of its own. */
    private void place(KvSequence s, int i, int b, boolean retain) {
        if (i < s.blocks() && s.entry(i) == b) {
            if (!retain) {
                release(b); // the caller's reference duplicates the table's
            }
            return;
        }
        if (retain) {
            retain(b);
        }
        s.put(i, b);
    }

    /** Returns an allocated, never published block to the free list. */
    private void dropRaw(int b) {
        refs[b] = 0;
        used--;
        if (freeCount == free.length) {
            free = Arrays.copyOf(free, freeCount * 2);
        }
        free[freeCount++] = b;
    }

    /**
     * Publishes the full blocks of {@code s} that hold {@code tokens[0 .. n)}: each becomes (or already is) the node of
     * its path. When the tree already has a block for the path, the sequence switches to it and its own copy is freed.
     */
    synchronized void publish(KvSequence s, int[] tokens, int n) {
        if (!sharing || !s.shareable()) {
            return;
        }
        long t = ++tick;
        Node node = root;
        int full = Math.min(n / B, s.blocks());
        KvSpill sp = spill;
        for (int i = 0; i < full; i++) {
            int b = s.entry(i);
            Node c = node.find(tokens, i * B);
            if (c != null) {
                touch(c, t);
                if (c.block != b) {
                    place(s, i, c.block, true); // releases b
                    dedups++;
                }
            } else {
                if (owner[b] != null) {
                    return; // b is published under another path: the table does not match tokens
                }
                int[] tk = Arrays.copyOfRange(tokens, i * B, (i + 1) * B);
                long ph = pathHash(node.hash, tokens, i * B, B);
                c = new Node(node, tk, ph, b);
                c.tick = t;
                owner[b] = c;
                attach(node, c);
                if (sp != null) {
                    sp.remove(ph);
                }
            }
            node = c;
        }
    }

    // ------------------------------------------------------------------ statistics

    public synchronized Stats stats() {
        KvSpill sp = spill;
        return new Stats(B, maxBlocks, highWater - freeCount, used, cached, freeCount, (long) (highWater - freeCount) * storage.residentBytes(),
                evictions, spills, reloads, cows, dedups, sp == null ? 0 : sp.files(), sp == null ? 0 : sp.bytes());
    }

    /** Blocks in memory that hold data (referenced or cached). */
    public synchronized int residentBlocks() {
        return highWater - freeCount;
    }

    /** Drops every cached (unreferenced) block, spilling them if there is a store. For tests and for flushing to disk. */
    public synchronized int evictAllCached() {
        int n = 0;
        int b;
        while ((b = evictOne()) >= 0) {
            dropRaw0(b);
            n++;
        }
        return n;
    }

    private void dropRaw0(int b) {
        if (freeCount == free.length) {
            free = Arrays.copyOf(free, freeCount * 2);
        }
        free[freeCount++] = b;
    }

    @Override
    public synchronized void close() {
        storage.close();
    }
}
