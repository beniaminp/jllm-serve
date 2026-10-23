package org.robomiri.jllm;

import org.robomiri.jllm.kv.FileSpill;
import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.kv.KvPoolExhausted;
import org.robomiri.jllm.kv.KvSequence;
import org.robomiri.jllm.kv.KvStorage;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

/**
 * The block pool, the prefix tree, copy on write, eviction and the spill store, on a fake storage whose "rows" are the
 * rolling hash of the tokens up to their position (like a real key, a row depends on the whole prefix), so every
 * sharing mistake shows as a wrong row. No model is needed.
 */
public final class KvPoolTest {

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    /** One long per row; two layers' worth would add nothing here. */
    static final class FakeStorage implements KvStorage {
        final int B;
        final long[][] rows;
        int copies;

        FakeStorage(int blockTokens) {
            this(blockTokens, 4000);
        }

        FakeStorage(int blockTokens, int capacity) {
            this.B = blockTokens;
            rows = new long[capacity][blockTokens]; // addressable up front: the pool's grow() has nothing to do
        }

        @Override
        public int blockTokens() {
            return B;
        }

        @Override
        public int spillBytes() {
            return B * 8;
        }

        @Override
        public long residentBytes() {
            return B * 8L;
        }

        @Override
        public void grow(int blocks) {
            if (blocks > rows.length) {
                throw new IllegalStateException("fake storage too small");
            }
        }

        @Override
        public synchronized void copy(int src, int dst, int n) {
            System.arraycopy(rows[src], 0, rows[dst], 0, n);
            copies++;
        }

        @Override
        public synchronized void save(int block, ByteBuffer dst) {
            for (long v : rows[block]) {
                dst.putLong(v);
            }
        }

        @Override
        public synchronized void load(int block, ByteBuffer src) {
            for (int i = 0; i < B; i++) {
                rows[block][i] = src.getLong();
            }
        }

        @Override
        public void close() {
        }
    }

    /** Row value of position p of a token sequence: depends on every token up to p. */
    static long[] rowValues(int[] t) {
        long[] v = new long[t.length];
        long h = 1469598103934665603L;
        for (int i = 0; i < t.length; i++) {
            h = (h ^ (t[i] + 1)) * 1099511628211L + i;
            v[i] = h;
        }
        return v;
    }

    /** Writes the rows of {@code t[from, to)} the way a forward pass does. */
    static void fill(FakeStorage st, KvSequence s, int[] t, int from, int to) {
        s.beginWrite(from, to - from);
        long[] v = rowValues(t);
        int B = s.blockTokens();
        for (int p = from; p < to; p++) {
            st.rows[s.block(p / B)][p % B] = v[p];
        }
    }

    /** Whether positions [0, n) of the sequence hold the rows of t. */
    static boolean holds(FakeStorage st, KvSequence s, int[] t, int n) {
        long[] v = rowValues(t);
        int B = s.blockTokens();
        for (int p = 0; p < n; p++) {
            if (st.rows[s.block(p / B)][p % B] != v[p]) {
                return false;
            }
        }
        return true;
    }

    static int[] tokens(int seed, int n) {
        Random r = new Random(seed);
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = r.nextInt(1000);
        }
        return t;
    }

    static int[] concat(int[] a, int n, int[] b) {
        int[] r = Arrays.copyOf(a, n + b.length);
        System.arraycopy(b, 0, r, n, b.length);
        return r;
    }

    public static void main(String[] args) throws Exception {
        sharing();
        copyOnWrite();
        eviction();
        spill();
        concurrent();
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("kv pool test passed");
    }

    /** Two conversations with a shared long prefix store it once; branches keep their own tails. */
    static void sharing() {
        FakeStorage st = new FakeStorage(16);
        KvPool pool = new KvPool(st, 1000, "fake");
        int[] sys = tokens(1, 4000);
        KvSequence a = pool.newSequence(8192);
        int[] ta = concat(sys, sys.length, tokens(2, 100));
        fill(st, a, ta, 0, ta.length);
        a.publish(ta, ta.length);
        int blocksA = pool.residentBlocks();
        check(blocksA == (4100 + 15) / 16, "one conversation of 4100 tokens takes " + blocksA + " blocks");
        // a second conversation: same system prompt, other tail
        KvSequence b = pool.newSequence(8192);
        int[] tb = concat(sys, sys.length, tokens(3, 100));
        int m = b.match(tb, tb.length - 1);
        check(m >= 4000, "the second conversation finds the shared prefix in the tree (" + m + " tokens)");
        int got = b.adopt(tb, m);
        check(got == m && holds(st, b, tb, got), "adopting gives the shared rows (" + got + " tokens)");
        fill(st, b, tb, got, tb.length);
        b.publish(tb, tb.length);
        int blocksAB = pool.residentBlocks();
        int flat = 2 * blocksA;
        check(blocksAB < flat - 240, "two conversations share the 4000-token prefix: " + blocksAB + " blocks, flat would be " + flat);
        check(holds(st, a, ta, ta.length) && holds(st, b, tb, tb.length), "both conversations read their own rows");
        // a third arrives while the first is idle; the first's blocks are not duplicated either
        KvSequence c = pool.newSequence(8192);
        int[] tc = concat(sys, sys.length, tokens(4, 50));
        int got3 = c.adopt(tc, c.match(tc, tc.length - 1));
        fill(st, c, tc, got3, tc.length);
        c.publish(tc, tc.length);
        check(pool.residentBlocks() <= blocksAB + 5, "a third adds only its tail: " + pool.residentBlocks() + " blocks");
        // identical prompt computed twice independently: publishing the second switches it to the first's blocks
        FakeStorage st2 = new FakeStorage(16);
        KvPool p2 = new KvPool(st2, 100, "fake");
        int[] t = tokens(5, 160);
        KvSequence x = p2.newSequence(512);
        KvSequence y = p2.newSequence(512);
        fill(st2, x, t, 0, 160);
        fill(st2, y, t, 0, 160);
        x.publish(t, 160);
        int before = p2.residentBlocks();
        y.publish(t, 160);
        check(p2.residentBlocks() == before - 10 && p2.stats().dedups() == 10, "an identical prompt computed twice is stored once ("
                + before + " -> " + p2.residentBlocks() + " blocks)");
        check(holds(st2, y, t, 160), "the deduplicated sequence still reads the right rows");
    }

    /** Writing into a shared or published block copies it first; a fork in the middle of a block copies the part it shares. */
    static void copyOnWrite() {
        FakeStorage st = new FakeStorage(16);
        KvPool pool = new KvPool(st, 200, "fake");
        int[] t = tokens(10, 200);
        KvSequence a = pool.newSequence(1024);
        fill(st, a, t, 0, 200);
        a.publish(t, 200);
        // a retry: the same conversation, rewritten from position 40 (middle of block 2) with other tokens
        int[] t2 = concat(t, 40, tokens(11, 100));
        KvSequence b = pool.newSequence(1024);
        int m = b.match(t2, t2.length - 1);
        check(m == 40, "a fork at 40 shares 40 tokens (found " + m + ")");
        int got = b.adopt(t2, m);
        check(got == 40 && holds(st, b, t2, 40), "adopting copies the 8 shared rows of the block it forks in");
        fill(st, b, t2, 40, t2.length);
        check(holds(st, a, t, 200), "the original conversation is untouched by the fork");
        check(holds(st, b, t2, t2.length), "the fork reads its rows");
        // the owner rewinds into a published block: it copies, the tree keeps the old rows
        KvSequence c = pool.newSequence(1024);
        int g = c.adopt(t, c.match(t, 199));
        check(g >= 192, "a third adopts the original (" + g + ")");
        int[] t3 = concat(t, 70, tokens(12, 30));
        fill(st, a, t3, 70, 100); // a rewinds to 70 and continues differently
        check(holds(st, c, t, g), "a sequence that adopted the published blocks does not see a rewrite of another");
        check(holds(st, a, t3, 100), "the rewinding sequence reads its new rows");
        // truncating releases whole blocks
        int before = pool.stats().usedBlocks();
        a.truncate(16);
        check(pool.stats().usedBlocks() < before, "truncate releases blocks");
        // a snapshot keeps rows alive while the sequence changes
        KvSequence d = pool.newSequence(1024);
        int[] t4 = tokens(13, 50);
        fill(st, d, t4, 0, 50);
        KvSequence snap = d.snapshot(50);
        int[] t5 = concat(t4, 20, tokens(14, 20));
        fill(st, d, t5, 20, 40);
        long[] v = rowValues(t4);
        boolean ok = true;
        for (int p = 0; p < 50; p++) {
            ok &= st.rows[snap.block(p / 16)][p % 16] == v[p];
        }
        check(ok, "a snapshot still holds the rows after the sequence rewrote positions 20..40");
        snap.close();
        // sharing a partial last block between sequences (copyStateFrom) copies it on the first write
        KvSequence e = pool.newSequence(1024);
        e.shareFrom(d, 40);
        int copies = st.copies;
        int[] t6 = concat(t5, 40, tokens(15, 5));
        fill(st, e, t6, 40, 45);
        check(st.copies == copies + 1 && holds(st, d, t5, 40) && holds(st, e, t6, 45),
                "a shared partial block is copied by the writer, the other keeps its rows");
    }

    /** LRU eviction of unreferenced blocks, leaf first; a full pool of referenced blocks is an error. */
    static void eviction() {
        FakeStorage st = new FakeStorage(16);
        KvPool pool = new KvPool(st, 40, "fake");
        int[] sys = tokens(20, 160); // 10 blocks
        KvSequence a = pool.newSequence(1024);
        fill(st, a, sys, 0, 160);
        a.publish(sys, 160);
        a.clear();
        check(pool.stats().cachedBlocks() == 10 && pool.stats().usedBlocks() == 0, "a finished conversation stays cached: "
                + pool.stats().cachedBlocks() + " blocks");
        // the system prompt is still found
        KvSequence b = pool.newSequence(1024);
        check(b.match(sys, 159) >= 144, "a cached prefix is found after its sequence is gone");
        // fill the pool with other conversations: the old ones are evicted, least recently used first
        int[][] others = new int[6][];
        for (int i = 0; i < others.length; i++) {
            others[i] = tokens(30 + i, 160);
            KvSequence s = pool.newSequence(1024);
            fill(st, s, others[i], 0, 160);
            s.publish(others[i], 160);
            s.clear();
        }
        KvPool.Stats stt = pool.stats();
        check(stt.residentBlocks() <= 40 && stt.evictions() > 0, "the pool stays within its " + stt.maxBlocks() + " blocks (evictions "
                + stt.evictions() + ")");
        check(b.match(sys, 159) == 0 || b.match(sys, 159) < 144, "the oldest conversation was evicted");
        check(b.match(others[5], 159) >= 144, "the newest is still cached");
        // an evicted inner block cannot orphan its children: eviction went leaf first
        // all blocks referenced: exhaustion is an error, not a hang
        KvSequence[] hold = new KvSequence[4];
        boolean thrown = false;
        try {
            for (int i = 0; i < hold.length; i++) {
                hold[i] = pool.newSequence(1024);
                fill(st, hold[i], tokens(50 + i, 200), 0, 200); // 13 blocks each: 52 > 40
            }
        } catch (KvPoolExhausted e) {
            thrown = true;
        }
        check(thrown, "a pool whose blocks are all referenced says so");
        for (KvSequence s : hold) {
            if (s != null) {
                s.clear();
            }
        }
        check(pool.stats().usedBlocks() == 0, "everything is released afterwards");
        // an idle holder is asked for its blocks before the pool gives up
        FakeStorage st2 = new FakeStorage(16);
        KvPool p2 = new KvPool(st2, 20, "fake");
        KvSequence idle = p2.newSequence(1024);
        fill(st2, idle, tokens(60, 240), 0, 240); // 15 blocks, referenced by an idle slot
        KvSequence busy = p2.newSequence(1024);
        boolean starved = false;
        try {
            fill(st2, busy, tokens(61, 160), 0, 160); // needs 10, 5 are free
        } catch (KvPoolExhausted e) {
            starved = true;
        }
        busy.clear();
        p2.setReclaimer(() -> {
            boolean any = idle.blocks() > 0;
            idle.clear();
            return any;
        });
        fill(st2, busy, tokens(61, 160), 0, 160);
        check(starved && busy.length() == 160 && idle.blocks() == 0, "a full pool asks its reclaimer: the idle holder's blocks were freed");
    }

    /** Evicted blocks go to the spill store and come back, also after a restart (a new pool on the same directory). */
    static void spill() throws Exception {
        Path dir = Files.createTempDirectory("kvspill");
        try {
            FakeStorage st = new FakeStorage(16);
            KvPool pool = new KvPool(st, 30, "fake");
            FileSpill fs = new FileSpill(dir, "fake-model", 1L << 30);
            pool.setSpill(fs);
            int[] sys = tokens(60, 160);
            KvSequence a = pool.newSequence(1024);
            fill(st, a, sys, 0, 160);
            a.publish(sys, 160);
            a.clear();
            for (int i = 0; i < 4; i++) { // push it out
                int[] o = tokens(70 + i, 160);
                KvSequence s = pool.newSequence(1024);
                fill(st, s, o, 0, 160);
                s.publish(o, 160);
                s.clear();
            }
            check(pool.stats().spills() > 0 && fs.files() > 0, "evicted blocks were written to disk: " + fs.files() + " files, "
                    + fs.bytes() + " bytes");
            KvSequence b = pool.newSequence(1024);
            KvPool.Match mt = b.matchDetail(sys, 159);
            check(mt.tokens() >= 144 && mt.spilled() > 0, "the spilled prefix is found: " + mt.tokens() + " tokens, " + mt.spilled() + " on disk");
            int got = b.adopt(sys, mt.tokens());
            check(got == mt.tokens() && holds(st, b, sys, got), "reloading gives the same rows (" + got + " tokens)");
            check(pool.stats().reloads() > 0, "reloads counted: " + pool.stats().reloads());
            b.clear();
            pool.evictAllCached();
            pool.close();
            // a restart: new pool, same directory
            FakeStorage st2 = new FakeStorage(16);
            KvPool p2 = new KvPool(st2, 30, "fake");
            p2.setSpill(new FileSpill(dir, "fake-model", 1L << 30));
            KvSequence c = p2.newSequence(1024);
            int m2 = c.match(sys, 159);
            check(m2 >= 144, "after a restart the spilled blocks are found again (" + m2 + " tokens)");
            int g2 = c.adopt(sys, m2);
            check(g2 == m2 && holds(st2, c, sys, g2), "and give the same rows");
            // another model's fingerprint sees nothing
            KvPool p3 = new KvPool(new FakeStorage(16), 30, "fake");
            p3.setSpill(new FileSpill(dir, "other-model", 1L << 30));
            check(p3.newSequence(1024).match(sys, 159) == 0, "a store of another fingerprint is not read");
            // a full pool whose evictable blocks are the adopting sequence's own old tail: reading spilled blocks back must
            // free that tail first
            FakeStorage st4 = new FakeStorage(16);
            KvPool p4 = new KvPool(st4, 30, "fake");
            p4.setSpill(new FileSpill(dir, "tail-model", 1L << 30));
            int[] ta = tokens(90, 160);   // 10 blocks
            int[] tc = tokens(91, 320);   // 20 blocks
            int[] tx = tokens(92, 80);    // 5 blocks
            KvSequence sa = p4.newSequence(1024);
            fill(st4, sa, ta, 0, 160);
            sa.publish(ta, 160);
            sa.clear();
            KvSequence sc = p4.newSequence(1024);
            fill(st4, sc, tc, 0, 320);
            sc.publish(tc, 320);
            KvSequence sx = p4.newSequence(1024);
            fill(st4, sx, tx, 0, 80); // evicts the last 5 blocks of a
            sx.publish(tx, 80);
            KvPool.Match mt4 = sc.matchDetail(ta, 159);
            int got4 = sc.adopt(ta, mt4.tokens());
            check(mt4.spilled() > 0 && got4 == mt4.tokens() && holds(st4, sc, ta, got4), "a sequence holding its old conversation in a full pool"
                    + " adopts a prefix that is partly on disk (" + got4 + " of " + mt4.tokens() + " tokens, " + mt4.spilled() + " from disk)");
            // a budget deletes the oldest files
            FileSpill small = new FileSpill(dir, "small", 3 * (16 * 8 + 100));
            ByteBuffer payload = ByteBuffer.allocate(16 * 8).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < 10; i++) {
                payload.clear();
                small.put(i, new int[16], payload);
            }
            check(small.files() <= 3, "the spill budget keeps at most " + small.files() + " files");
        } finally {
            try (var walk = Files.walk(dir)) {
                walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }

    /** Several threads on one pool: every sequence reads its own rows afterwards, and the books balance. */
    static void concurrent() throws Exception {
        FakeStorage st = new FakeStorage(16);
        KvPool pool = new KvPool(st, 2000, "fake");
        int[] sys = tokens(80, 800);
        Thread[] th = new Thread[6];
        boolean[] ok = new boolean[th.length];
        for (int i = 0; i < th.length; i++) {
            int k = i;
            th[i] = new Thread(() -> {
                boolean good = true;
                for (int round = 0; round < 30; round++) {
                    int[] t = concat(sys, sys.length, tokens(100 + k * 100 + round, 200 + round));
                    KvSequence s = pool.newSequence(2048);
                    int m = s.adopt(t, s.match(t, t.length - 1));
                    fill(st, s, t, m, t.length);
                    s.publish(t, t.length);
                    good &= holds(st, s, t, t.length);
                    s.clear();
                }
                ok[k] = good;
            });
        }
        for (Thread t : th) {
            t.start();
        }
        for (Thread t : th) {
            t.join();
        }
        check(Arrays.equals(ok, new boolean[]{true, true, true, true, true, true}), "six threads sharing a prefix read their own rows");
        KvPool.Stats s = pool.stats();
        check(s.usedBlocks() == 0 && s.usedBlocks() + s.cachedBlocks() == s.residentBlocks(), "the books balance: used " + s.usedBlocks()
                + ", cached " + s.cachedBlocks() + ", free " + s.freeBlocks());
    }
}
