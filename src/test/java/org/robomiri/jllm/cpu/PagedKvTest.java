package org.robomiri.jllm.cpu;

import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.kv.KvSequence;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;

import java.io.ByteArrayOutputStream;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.util.Arrays;
import java.util.Random;

/**
 * The paged CPU cache against the flat one, bit for bit: the same stores, rollbacks and attention calls (prefill chunks,
 * decode steps, speculative verification chunks, sliding windows, soft-capping) must give identical outputs and the
 * same saved bytes, for every cache type, block size and head shape. Also the sharing paths: a sequence that adopts
 * another's blocks attends exactly as the cache that computed them, and a rewind into a shared block does not disturb
 * the other sequence. No model needed.
 */
public final class PagedKvTest {

    private static int failures;
    private static int checks;

    private static boolean warming;

    private static void check(boolean ok, String what) {
        if (warming) {
            return;
        }
        checks++;
        if (!ok) {
            failures++;
            note("FAIL " + what);
        }
    }

    /** Messages of the attempt in progress: printed only if the attempt is the last one (see {@link #scenario}). */
    private static final java.util.List<String> pending = new java.util.ArrayList<>();

    private static void note(String line) {
        pending.add(line);
    }

    public static void main(String[] args) throws Exception {
        try (ThreadPool pool = new ThreadPool(4); ThreadPool one = new ThreadPool(1)) {
            // warm the JIT first: a kernel's first, interpreted calls may round differently from its compiled ones (the flat
            // cache differs from itself there), which says nothing about paging
            warming = true;
            for (KvDtype dt : new KvDtype[]{KvDtype.F32, KvDtype.F16, KvDtype.Q8}) {
                for (int i = 0; i < 6; i++) {
                    scenario(pool, dt, 64, 2, 2, 16, 0, 0f);
                    scenario(pool, dt, 32, 2, 1, 16, 0, 0f);
                }
            }
            warming = false;
            for (KvDtype dt : new KvDtype[]{KvDtype.F32, KvDtype.F16, KvDtype.Q8}) {
                for (int hd : new int[]{32, 64, 128}) {
                    for (int[] shape : new int[][]{{2, 1}, {2, 2}, {8, 2}, {1, 4}, {16, 2}}) {
                        for (int block : new int[]{16, 32, 64}) {
                            for (int window : new int[]{0, 50}) {
                                for (float cap : new float[]{0f, 30f}) {
                                    if (window == 50 && cap == 30f && shape[0] != 2 && shape[0] != 16) {
                                        continue; // keep the matrix at a few hundred cases
                                    }
                                    scenario(pool, dt, hd, shape[0], shape[1], block, window, cap);
                                }
                            }
                        }
                    }
                }
                scenario(one, dt, 64, 2, 2, 16, 0, 0f); // one thread: the single-threaded kernel paths
            }
            for (int hd : new int[]{64, 128}) { // q8 with Hadamard-rotated keys
                for (int[] shape : new int[][]{{2, 2}, {8, 2}, {1, 4}}) {
                    for (int window : new int[]{0, 50}) {
                        scenario(pool, KvDtype.Q8, hd, shape[0], shape[1], 16, window, 0f, true);
                    }
                }
                scenario(pool, KvDtype.Q8, hd, 2, 2, 64, 0, 30f, true);
            }
            sharing(pool);
        }
        System.out.println(checks + " checks");
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("paged kv test passed");
    }

    private static float[] random(Random r, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (float) r.nextGaussian();
        }
        return a;
    }

    /** One run of stores and attention calls on the flat and the paged cache; every output must be identical. */
    static void scenario(ThreadPool pool, KvDtype dt, int hd, int kvHeads, int group, int block, int window, float cap) throws Exception {
        scenario(pool, dt, hd, kvHeads, group, block, window, cap, false);
    }

    static void scenario(ThreadPool pool, KvDtype dt, int hd, int kvHeads, int group, int block, int window, float cap,
                         boolean rot) throws Exception {
        // One run, from a cold JVM if need be: the lane sums of the kernels (dot, softmaxExp, the quantized dots) add in one
        // fixed order in every JIT tier (VectorKernels.laneSum, JitStableKernelsTest), so a repeat of the same call gives the
        // same bits whether the interpreter, C1 or C2 runs it.
        pending.clear();
        scenario0(pool, dt, hd, kvHeads, group, block, window, cap, rot);
        pending.forEach(System.out::println);
    }

    private static void scenario0(ThreadPool pool, KvDtype dt, int hd, int kvHeads, int group, int block, int window, float cap,
                                  boolean rot) throws Exception {
        String what = dt + (rot ? " rotated keys" : "") + " hd=" + hd + " kvHeads=" + kvHeads + " group=" + group + " block=" + block + " window=" + window
                + " cap=" + cap + " threads=" + pool.threads();
        int layers = 2;
        int ctx = 700;
        Attention.Gqa a = new Attention.Gqa(kvHeads * group, kvHeads, hd, window, 0, false, false, false, (float) (1 / Math.sqrt(hd)), cap);
        KvCache flat = KvCache.create(dt, layers, a, ctx, 64, 128, rot);
        CpuKvStorage st = new CpuKvStorage(dt, layers, a, block, rot);
        KvPool kp = new KvPool(st, 400, "t");
        KvSequence seq = kp.newSequence(ctx);
        PagedKv paged = new PagedKv(layers, a, ctx, st, seq);
        check(flat.format().equals(paged.format()), what + ": same format " + flat.format() + " / " + paged.format());
        Random r = new Random(hd * 31 + kvHeads * 7 + group + block);
        // (pos, n): prefill chunks, decode, a rewind into the middle of a block, a verification chunk, more decode
        int[][] ops = {{0, 64}, {64, 64}, {128, 3}, {131, 1}, {132, 1}, {133, 1}, {100, 20}, {120, 5}, {125, 1}, {126, 40}, {166, 1},
                {167, 1}, {40, 130}, {170, 1}};
        boolean same = true;
        int end = 0;
        for (int[] op : ops) {
            int pos = op[0];
            int n = op[1];
            float[] k = random(r, n * a.kvDim());
            float[] v = random(r, n * a.kvDim());
            float[] q = random(r, n * a.qDim());
            seq.beginWrite(pos, n);
            float[] af = new float[n * a.qDim()];
            float[] ap = new float[n * a.qDim()];
            for (int l = 0; l < layers; l++) {
                flat.store(l, pos, k, v, n);
                paged.store(l, pos, k, v, n);
                flat.attention(pool, l, n, pos, q, af, window, a.scale(), cap);
                paged.attention(pool, l, n, pos, q, ap, window, a.scale(), cap);
                float[] af2 = new float[n * a.qDim()];
                float[] ap2 = new float[n * a.qDim()];
                flat.attention(pool, l, n, pos, q, af2, window, a.scale(), cap);
                paged.attention(pool, l, n, pos, q, ap2, window, a.scale(), cap);
                if (!Arrays.equals(af, af2) || !Arrays.equals(ap, ap2)) {
                    note("  repeat differs: flat " + Arrays.equals(af, af2) + " paged " + Arrays.equals(ap, ap2));
                    same = false;
                }
                if (!Arrays.equals(af, ap)) {
                    int bad = Arrays.mismatch(af, ap);
                    note("  first difference: op (pos " + pos + ", n " + n + "), layer " + l + ", index " + bad + ": "
                            + af[bad] + " vs " + ap[bad]);
                    same = false;
                }
            }
            end = pos + n;
        }
        check(same, what + ": attention outputs identical");
        ByteArrayOutputStream fo = new ByteArrayOutputStream();
        ByteArrayOutputStream po = new ByteArrayOutputStream();
        flat.write(end, Channels.newChannel(fo));
        paged.write(end, Channels.newChannel(po));
        check(Arrays.equals(fo.toByteArray(), po.toByteArray()), what + ": saved bytes identical (" + po.size() + ")");
        // a saved flat cache loaded into a paged one attends the same
        KvPool kp2 = new KvPool(new CpuKvStorage(dt, layers, a, block, rot), 400, "t");
        KvSequence seq2 = kp2.newSequence(ctx);
        PagedKv loaded = new PagedKv(layers, a, ctx, (CpuKvStorage) kp2.storage(), seq2);
        ReadableByteChannel in = Channels.newChannel(new java.io.ByteArrayInputStream(fo.toByteArray()));
        seq2.beginWrite(0, end);
        loaded.read(end, in);
        float[] q = random(r, a.qDim());
        float[] af = new float[a.qDim()];
        float[] al = new float[a.qDim()];
        seq.beginWrite(end, 1);
        seq2.beginWrite(end, 1);
        float[] k = random(r, a.kvDim());
        float[] v = random(r, a.kvDim());
        flat.store(0, end, k, v, 1);
        loaded.store(0, end, k, v, 1);
        flat.attention(pool, 0, 1, end, q, af, window, a.scale(), cap);
        loaded.attention(pool, 0, 1, end, q, al, window, a.scale(), cap);
        check(Arrays.equals(af, al), what + ": a saved cache loaded into a paged one attends the same");
        kp.close();
        kp2.close();
    }

    /** Adoption, forks and rewinds through the tree on the real kernels. */
    static void sharing(ThreadPool pool) {
        int layers = 2;
        int ctx = 600;
        for (KvDtype dt : new KvDtype[]{KvDtype.F32, KvDtype.F16, KvDtype.Q8}) {
            Attention.Gqa a = new Attention.Gqa(4, 2, 64, 0, 0, false, false, false, 0.125f, 0f);
            CpuKvStorage st = new CpuKvStorage(dt, layers, a, 16);
            KvPool kp = new KvPool(st, 300, "t");
            KvSequence s1 = kp.newSequence(ctx);
            KvSequence s2 = kp.newSequence(ctx);
            PagedKv p1 = new PagedKv(layers, a, ctx, st, s1);
            PagedKv p2 = new PagedKv(layers, a, ctx, st, s2);
            KvCache flat = KvCache.create(dt, layers, a, ctx);
            Random r = new Random(5);
            int[] tokens = new int[300];
            for (int i = 0; i < tokens.length; i++) {
                tokens[i] = r.nextInt(500);
            }
            int n = 150;
            float[] k = random(r, n * a.kvDim());
            float[] v = random(r, n * a.kvDim());
            s1.beginWrite(0, n);
            for (int l = 0; l < layers; l++) {
                p1.store(l, 0, k, v, n);
                flat.store(l, 0, k, v, n);
            }
            s1.publish(tokens, n);
            // s2 takes over the prompt (a different tail follows) and attends the next token
            int found = s2.match(tokens, n);
            int got = s2.adopt(tokens, found);
            check(found == 144 && got == 144, dt + ": the second sequence adopts the 9 published blocks (" + found + ", " + got + ")");
            float[] q = random(r, a.qDim());
            // the tail of the prompt (6 rows) and the next token are written by s2 itself
            float[] kn = new float[7 * a.kvDim()];
            float[] vn = new float[7 * a.kvDim()];
            System.arraycopy(k, 144 * a.kvDim(), kn, 0, 6 * a.kvDim());
            System.arraycopy(v, 144 * a.kvDim(), vn, 0, 6 * a.kvDim());
            float[] extraK = random(r, a.kvDim());
            float[] extraV = random(r, a.kvDim());
            System.arraycopy(extraK, 0, kn, 6 * a.kvDim(), a.kvDim());
            System.arraycopy(extraV, 0, vn, 6 * a.kvDim(), a.kvDim());
            s2.beginWrite(144, 7);
            float[] ap = new float[a.qDim()];
            float[] af = new float[a.qDim()];
            for (int l = 0; l < layers; l++) {
                p2.store(l, 144, kn, vn, 7);
                flat.store(l, 150, extraK, extraV, 1);
                p2.attention(pool, l, 1, 150, q, ap, 0, 0.125f, 0f);
                flat.attention(pool, l, 1, 150, q, af, 0, 0.125f, 0f);
            }
            check(Arrays.equals(ap, af), dt + ": attention over adopted blocks equals the cache that computed them");
            check(kp.residentBlocks() <= (151 + 15) / 16 + 2, dt + ": the two sequences share the blocks (" + kp.residentBlocks() + ")");
            // s1 rewinds into a shared block and continues differently: s2 is not disturbed
            float[] k2 = random(r, 30 * a.kvDim());
            float[] v2 = random(r, 30 * a.kvDim());
            s1.beginWrite(70, 30);
            for (int l = 0; l < layers; l++) {
                p1.store(l, 70, k2, v2, 30);
            }
            float[] q2 = random(r, a.qDim());
            float[] kn2 = random(r, a.kvDim());
            float[] vn2 = random(r, a.kvDim());
            s2.beginWrite(151, 1);
            float[] ap2 = new float[a.qDim()];
            float[] af2 = new float[a.qDim()];
            for (int l = 0; l < layers; l++) {
                p2.store(l, 151, kn2, vn2, 1);
                flat.store(l, 151, kn2, vn2, 1);
                p2.attention(pool, l, 1, 151, q2, ap2, 0, 0.125f, 0f);
                flat.attention(pool, l, 1, 151, q2, af2, 0, 0.125f, 0f);
            }
            check(Arrays.equals(ap2, af2), dt + ": a rewind in the other sequence does not change what this one reads");
            kp.close();
        }
    }
}
