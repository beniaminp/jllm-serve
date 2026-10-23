package org.robomiri.jllm;

import org.robomiri.jllm.kv.CodecSpill;
import org.robomiri.jllm.kv.FileSpill;
import org.robomiri.jllm.kv.KvBlockCodec;
import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.kv.KvPoolExhausted;
import org.robomiri.jllm.kv.KvSequence;
import org.robomiri.jllm.kv.TieredSpill;
import org.robomiri.jllm.model.KvDtype;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * The tiers of docs/26 without a model: the warm (host memory) tier in front of the file store, the on-disk row
 * formats, a pool with a deliberately small hot cap that must push whole conversations down and bring them back
 * unchanged, and prefetch.
 */
public final class KvTierTest {

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    public static void main(String[] args) throws Exception {
        tieredSpill();
        codec();
        smallHotCap();
        exhausted();
        persistAndReload();
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("kv tier test passed");
    }

    private static ByteBuffer payload(long seed, int n) {
        ByteBuffer b = ByteBuffer.allocateDirect(n).order(ByteOrder.LITTLE_ENDIAN);
        Random r = new Random(seed);
        byte[] x = new byte[n];
        r.nextBytes(x);
        b.put(x).flip();
        return b;
    }

    private static boolean same(ByteBuffer a, ByteBuffer b) {
        return a.duplicate().rewind().equals(b.duplicate().rewind());
    }

    /** RAM budget of 4 payloads: older ones go to the files, every one stays readable, prefetch pulls one back. */
    static void tieredSpill() throws Exception {
        Path dir = Files.createTempDirectory("kvtier");
        FileSpill files = new FileSpill(dir, "t", 1L << 30);
        int n = 128 * 1024; // above the mapped-read threshold
        TieredSpill t = new TieredSpill(files, 4L * n);
        int[] tok = new int[16];
        for (int i = 0; i < 10; i++) {
            tok[0] = i;
            check(t.put(1000 + i, tok, payload(i, n)), "put " + i);
        }
        t.drain();
        TieredSpill.Stats st = t.stats();
        check(st.ramBlocks() <= 4 && st.demotions() >= 6, "the oldest payloads went down to the files: " + st);
        check(files.files() >= 6, "the file store holds " + files.files() + " payloads");
        boolean all = true;
        for (int i = 0; i < 10; i++) {
            tok[0] = i;
            ByteBuffer got = ByteBuffer.allocateDirect(n);
            all &= t.has(1000 + i) && t.get(1000 + i, tok.clone(), got) && same(got.flip(), payload(i, n));
        }
        check(all, "all ten payloads read back identical from whichever tier holds them");
        tok[0] = 0;
        ByteBuffer wrong = ByteBuffer.allocateDirect(n);
        int[] other = tok.clone();
        other[1] = 99;
        check(!t.get(1000, other, wrong), "a block asked for under other tokens is refused");
        // prefetch: a payload that only the files hold comes into memory
        t.remove(1005);
        t.put(1005, new int[]{5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, payload(5, n));
        for (int i = 20; i < 30; i++) {
            tok[0] = i;
            t.put(1000 + i, tok, payload(i, n)); // pushes 1005 down
        }
        t.drain();
        int[] t5 = new int[16];
        t5[0] = 5;
        long before = t.stats().prefetches();
        t.prefetch(1005, t5, n);
        t.drain();
        check(t.stats().prefetches() == before + 1, "prefetch brought a file payload into memory");
        ByteBuffer got = ByteBuffer.allocateDirect(n);
        long hits = t.stats().ramHits();
        check(t.get(1005, t5, got) && same(got.flip(), payload(5, n)) && t.stats().ramHits() == hits + 1
                && t.stats().prefetchHits() >= 1, "the prefetched payload is then served from memory");
        t.close();
        check(files.files() >= 10, "close flushed memory to the files (" + files.files() + ")");
        // a new process over the same directory finds them
        FileSpill again = new FileSpill(dir, "t", 1L << 30);
        TieredSpill t2 = new TieredSpill(again, 2L * n);
        tok[0] = 3;
        ByteBuffer g3 = ByteBuffer.allocateDirect(n);
        check(t2.has(1003) && t2.get(1003, tok.clone(), g3) && same(g3.flip(), payload(3, n)), "payloads survive a restart");
        t2.close();
        again.clear();
    }

    /** f16 and f32 pool rows through q8 and q4 files: sizes, and the error against the original. */
    static void codec() {
        int layers = 3;
        int B = 16;
        int kvHeads = 4;
        int headDim = 64;
        Random r = new Random(7);
        for (KvDtype dt : new KvDtype[]{KvDtype.F16, KvDtype.F32}) {
            for (KvBlockCodec.Format f : new KvBlockCodec.Format[]{KvBlockCodec.Format.Q8, KvBlockCodec.Format.Q4}) {
                KvBlockCodec c = new KvBlockCodec(f, dt, layers, B, kvHeads, headDim);
                int kvDim = kvHeads * headDim;
                float[] ref = new float[layers * B * 2 * kvDim];
                ByteBuffer src = ByteBuffer.allocateDirect(c.nativeBytes()).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < ref.length; i++) { // a few outlier channels, as real keys have
                    float v = (float) (r.nextGaussian() * ((i % kvDim) % headDim < 2 ? 8 : 1));
                    if (dt == KvDtype.F16) {
                        v = Float.float16ToFloat(Float.floatToFloat16(v));
                        src.putShort(Float.floatToFloat16(v));
                    } else {
                        src.putFloat(v);
                    }
                    ref[i] = v;
                }
                src.flip();
                ByteBuffer enc = ByteBuffer.allocateDirect(c.encodedBytes());
                c.encode(src, enc);
                check(!src.hasRemaining() && !enc.hasRemaining(), dt + " -> " + f + ": encode consumed and filled exactly");
                enc.flip();
                ByteBuffer dec = ByteBuffer.allocateDirect(c.nativeBytes()).order(ByteOrder.LITTLE_ENDIAN);
                c.decode(enc, dec);
                dec.flip();
                double num = 0;
                double den = 0;
                for (int i = 0; i < ref.length; i++) {
                    float v = dt == KvDtype.F16 ? Float.float16ToFloat(dec.getShort()) : dec.getFloat();
                    num += (v - ref[i]) * (v - ref[i]);
                    den += ref[i] * ref[i];
                }
                double rel = Math.sqrt(num / den);
                double ratio = (double) c.encodedBytes() / c.nativeBytes();
                double limit = f == KvBlockCodec.Format.Q8 ? 0.01 : 0.12;
                check(rel < limit, String.format("%s -> %s: relative error %.4f (limit %.2f), %.1f %% of the bytes", dt, f, rel,
                        limit, 100 * ratio));
            }
        }
        KvBlockCodec nat = new KvBlockCodec(KvBlockCodec.Format.Q4, KvDtype.Q8, layers, B, kvHeads, headDim);
        check(nat.identity(), "a q8 pool is stored as it is");
    }

    /** Twenty conversations of 160 tokens through a pool of 60 blocks: all but the running one are pushed down. */
    static void smallHotCap() throws Exception {
        Path dir = Files.createTempDirectory("kvtier");
        KvPoolTest.FakeStorage st = new KvPoolTest.FakeStorage(16, 400);
        KvPool pool = new KvPool(st, 400, "fake");
        pool.setHotCap(60);
        FileSpill files = new FileSpill(dir, "p", 1L << 30);
        TieredSpill warm = new TieredSpill(files, 40L * 16 * 8); // room for 40 blocks of 128 bytes
        pool.setSpill(warm);
        int convs = 20;
        int len = 160;
        int[][] toks = new int[convs][];
        for (int c = 0; c < convs; c++) {
            toks[c] = KvPoolTest.tokens(100 + c, len);
            KvSequence s = pool.newSequence(512);
            KvPoolTest.fill(st, s, toks[c], 0, len);
            s.publish(toks[c], len);
            s.close();
        }
        warm.drain();
        KvPool.Stats ps = pool.stats();
        check(ps.committedBlocks() <= 60, "the pool never committed more than its hot cap: " + ps.committedBlocks() + " of 60 blocks");
        check(ps.evictions() >= (convs * 10 - 60) && ps.spills() >= ps.evictions(), "the rest was evicted down: " + ps.evictions() + " blocks");
        TieredSpill.Stats ts = warm.stats();
        check(ts.demotions() > 0 && ts.ramBlocks() <= 40, "memory tier kept at most 40 blocks, the rest went to files: " + ts);
        // every conversation comes back bit for bit: the old ones from the files, the recent ones from memory
        boolean all = true;
        long reloadsBefore = ps.reloads();
        for (int c = 0; c < convs; c++) {
            KvSequence s = pool.newSequence(512);
            int m = s.match(toks[c], len - 1);
            int got = s.adopt(toks[c], m);
            all &= got == m && m >= len - 16 && KvPoolTest.holds(st, s, toks[c], got);
            s.close();
        }
        check(all, "all twenty conversations adopt their whole prefix with identical rows");
        check(pool.stats().reloads() > reloadsBefore, "blocks were reloaded from the tiers: " + (pool.stats().reloads() - reloadsBefore));
        // the same again with prefetch hints: the older ones are then found in memory
        warm.drain();
        pool.evictAllCached();
        warm.drain();
        KvSequence s = pool.newSequence(512);
        int m = s.match(toks[0], len - 1); // hints the blocks
        warm.drain();
        long ramHits = warm.stats().ramHits();
        int got = s.adopt(toks[0], m);
        check(got == m && KvPoolTest.holds(st, s, toks[0], got) && warm.stats().ramHits() - ramHits >= (m / 16) - 1,
                "after the prefetch hints the adoption reads from memory (" + (warm.stats().ramHits() - ramHits) + " blocks)");
        s.close();
        warm.close();
        files.clear();
    }

    /** With every block held by a running sequence the cap is a hard limit with a message that names the setting. */
    static void exhausted() {
        KvPoolTest.FakeStorage st = new KvPoolTest.FakeStorage(16, 100);
        KvPool pool = new KvPool(st, 100, "fake");
        pool.setHotCap(8);
        KvSequence s = pool.newSequence(1024);
        int[] t = KvPoolTest.tokens(1, 300);
        boolean thrown = false;
        String msg = "";
        try {
            KvPoolTest.fill(st, s, t, 0, 300);
        } catch (KvPoolExhausted e) {
            thrown = true;
            msg = e.getMessage();
        }
        check(thrown && msg.contains("backend.kv-hot-mb"), "a running sequence larger than the cap fails with a clear message");
        s.close();
        pool.setHotCap(100);
        KvSequence s2 = pool.newSequence(1024);
        KvPoolTest.fill(st, s2, t, 0, 300);
        check(pool.hotCap() == 100 && KvPoolTest.holds(st, s2, t, 300), "raising the cap lets the pool grow again");
    }

    /** A saved session is the blocks in the store: the tokens alone find them again, after a "restart", in a q8 file format. */
    static void persistAndReload() throws Exception {
        Path dir = Files.createTempDirectory("kvtier");
        KvPoolTest.FakeStorage st = new KvPoolTest.FakeStorage(16, 200);
        KvPool pool = new KvPool(st, 200, "fake");
        FileSpill files = new FileSpill(dir, "s", 1L << 30);
        pool.setSpill(new TieredSpill(files, 0));
        int[] t = KvPoolTest.tokens(9, 400);
        KvSequence s = pool.newSequence(512);
        KvPoolTest.fill(st, s, t, 0, 400);
        s.publish(t, 400);
        int written = pool.persist(t, 400);
        check(written == 25 && files.files() == 25, "persist wrote the 25 full blocks and kept them resident (" + written + ")");
        check(pool.persist(t, 400) == 0, "persisting again writes nothing");
        s.close();
        // a new pool over the same files
        KvPoolTest.FakeStorage st2 = new KvPoolTest.FakeStorage(16, 200);
        KvPool pool2 = new KvPool(st2, 200, "fake");
        pool2.setSpill(new TieredSpill(new FileSpill(dir, "s", 1L << 30), 0));
        KvSequence s2 = pool2.newSequence(512);
        int m = s2.match(t, 399);
        int got = s2.adopt(t, m);
        check(m >= 384 && got == m && KvPoolTest.holds(st2, s2, t, got), "a restarted pool adopts the saved session from the block files (" + got + " tokens)");
        new CodecSpill(files, new KvBlockCodec(KvBlockCodec.Format.Q8, KvDtype.F16, 1, 16, 1, 64)).files();
    }
}
