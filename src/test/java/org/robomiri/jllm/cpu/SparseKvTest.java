package org.robomiri.jllm.cpu;

import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.kv.KvSequence;
import org.robomiri.jllm.kv.KvSummary;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.KvQ4;
import org.robomiri.jllm.model.KvSparse;

import java.util.Arrays;
import java.util.Random;

/**
 * Sparse decode attention on the CPU (docs/27), no model needed. For every cache type (f32, f16, q8, q8 with rotated keys),
 * head width, group size and block size:
 * <ul>
 * <li>the stored key summary of a block is the per-channel minimum and maximum of the dequantized rows that are valid,
 *     rounded outwards to halves, also after a rewind into the block and after a copy-on-write copy;
 * <li>the bound {@code Σ max(q_i lo_i, q_i hi_i)} is never below the largest score of any key of the block;
 * <li>the blocks a sparse row selects are the sinks, the window, and the best candidates by that bound, and a planted
 *     "needle" key (a key along the query) is always found;
 * <li>the output equals a float64 attention over exactly the selected keys;
 * <li>a short chunk of rows, and a row below the budget, give what separate decode steps give.
 * </ul>
 */
public final class SparseKvTest {

    private static int failures;
    private static int checks;

    private static void check(boolean ok, String what) {
        checks++;
        if (!ok) {
            failures++;
            System.out.println("FAIL " + what);
        }
    }

    public static void main(String[] args) throws Exception {
        try (ThreadPool pool = new ThreadPool(4)) {
            summaries();
            for (KvDtype dt : new KvDtype[]{KvDtype.F32, KvDtype.F16, KvDtype.Q8}) {
                for (int hd : new int[]{64, 128}) {
                    for (int[] shape : new int[][]{{2, 1}, {2, 2}, {4, 2}, {1, 4}}) {
                        for (int block : new int[]{16, 32}) {
                            for (float cap : new float[]{0f, 30f}) {
                                scenario(pool, dt, hd, shape[0], shape[1], block, cap, false);
                            }
                        }
                    }
                }
            }
            for (int hd : new int[]{64, 128}) {
                scenario(pool, KvDtype.Q8, hd, 2, 2, 16, 0f, true);
                scenario(pool, KvDtype.Q8, hd, 4, 2, 32, 30f, true);
            }
            needle(pool);
            KvSparse.set(KvSparse.OFF);
            offLayout();
        }
        System.out.println(checks + " checks");
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("sparse kv test passed");
    }

    private static float[] random(Random r, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (float) r.nextGaussian();
        }
        return a;
    }

    /** The outward rounding of halves: containment, tightness, extremes. */
    private static void summaries() {
        Random r = new Random(5);
        boolean ok = true;
        for (int i = 0; i < 200000; i++) {
            float x = (float) (r.nextGaussian() * Math.pow(10, r.nextInt(12) - 8));
            float d = Float.float16ToFloat(KvSummary.down(x));
            float u = Float.float16ToFloat(KvSummary.up(x));
            // contained, and at most one half step wide: nothing between the half and x is a half
            ok &= d <= x && u >= x && d >= Math.nextDown(x) - Math.abs(x) * 1.0e-3f - 1e-7f && u <= Math.nextUp(x) + Math.abs(x) * 1.0e-3f + 1e-7f;
            if (!ok) {
                System.out.println("  x=" + x + " down=" + d + " up=" + u);
                break;
            }
        }
        check(ok, "outward rounding to halves contains x and is within one half step");
        check(Float.float16ToFloat(KvSummary.down(0f)) <= 0f && Float.float16ToFloat(KvSummary.up(0f)) >= 0f, "0 rounds to 0");
        check(Float.float16ToFloat(KvSummary.down(1e9f)) == 65504f && Float.float16ToFloat(KvSummary.up(-1e9f)) == -65504f, "clamped to the largest half");
        check(Float.float16ToFloat(KvSummary.down(1e-9f)) <= 1e-9f && Float.float16ToFloat(KvSummary.up(1e-9f)) >= 1e-9f
                && Float.float16ToFloat(KvSummary.down(-1e-9f)) <= -1e-9f && Float.float16ToFloat(KvSummary.up(-1e-9f)) >= -1e-9f, "tiny values round outwards");
        // the ranking: higher score first, the lower block on a tie
        float[] sc = {1f, 3f, 3f, 2f, -1f, 3f};
        check(Arrays.equals(KvSummary.top(sc, 10, 6, 2), new int[]{11, 12}), "top: ties go to the lower block");
        check(Arrays.equals(KvSummary.top(sc, 0, 6, 4), new int[]{1, 2, 3, 5}), "top: four best, ascending");
        check(Arrays.equals(KvSummary.top(sc, 0, 6, 9), new int[]{0, 1, 2, 3, 4, 5}), "top: k above the count takes all");
    }

    /** The sparse pool's blocks are bigger by the summary; the plain pool's are not. */
    private static void offLayout() {
        Attention.Gqa a = new Attention.Gqa(4, 2, 64, 0, 0, false, false, false, 0.125f, 0f);
        CpuKvStorage plain = new CpuKvStorage(KvDtype.F16, 2, a, 16, false, false);
        CpuKvStorage withSum = new CpuKvStorage(KvDtype.F16, 2, a, 16, false, true);
        long sum = 2L * 2 * 32 * 4; // kvHeads * (lo, hi) * pairs * 4 bytes
        check(plain.sumBytes == 0 && plain.layerBytes == plain.kBytes + plain.vBytes, "no summary region without the switch");
        check(withSum.sumBytes >= sum && withSum.layerBytes == plain.layerBytes + withSum.sumBytes, "summary region of " + withSum.sumBytes + " bytes per layer");
    }

    private static CpuKvStorage storage(KvDtype dt, int layers, Attention.Gqa a, int block, boolean rot) {
        return new CpuKvStorage(dt, layers, a, block, rot, true);
    }

    /** Stored summary of (layer, block, kv head) as floats: [lo[hd], hi[hd]]. */
    private static float[] stored(CpuKvStorage st, int layer, int block, int g) {
        float[] out = new float[2 * st.headDim];
        var seg = st.slab(block);
        long sb = st.sumBase(block, layer);
        for (int h = 0; h < 2; h++) {
            for (int i = 0; i < st.headDim; i++) {
                int word = seg.get(java.lang.foreign.ValueLayout.JAVA_INT, sb + (((long) g * 2 + h) * st.pairs + (i >> 1)) * 4);
                out[h * st.headDim + i] = Float.float16ToFloat(CpuKvStorage.half(word, (i & 1) != 0));
            }
        }
        return out;
    }

    /** The exact min / max of the first {@code rows} rows of the block, from the dequantized stored values. */
    private static float[] exact(CpuKvStorage st, int layer, int block, int g, int rows) {
        float[] out = new float[2 * st.headDim];
        for (int i = 0; i < st.headDim; i++) {
            float lo = Float.POSITIVE_INFINITY;
            float hi = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < rows; t++) {
                float x = st.keyValue(layer, block, g, i, t);
                lo = Math.min(lo, x);
                hi = Math.max(hi, x);
            }
            out[i] = lo;
            out[st.headDim + i] = hi;
        }
        return out;
    }

    private static void scenario(ThreadPool pool, KvDtype dt, int hd, int kvHeads, int group, int block, float cap, boolean rot) {
        String what = dt + (rot ? " rotated keys" : "") + " hd=" + hd + " kvHeads=" + kvHeads + " group=" + group + " block=" + block
                + " cap=" + cap;
        int layers = 2;
        int ctx = 900;
        KvSparse cfg = new KvSparse(true, 6, 16, 32, 1); // layer 0 dense, layer 1 sparse
        KvSparse.set(cfg);
        Attention.Gqa a = new Attention.Gqa(kvHeads * group, kvHeads, hd, 0, 0, false, false, false, (float) (1 / Math.sqrt(hd)), cap);
        CpuKvStorage st = storage(dt, layers, a, block, rot);
        KvPool kp = new KvPool(st, 200, "t");
        KvSequence seq = kp.newSequence(ctx);
        PagedKv kv = new PagedKv(layers, a, ctx, st, seq);
        kv.recordSelection = true;
        Random r = new Random(hd * 31 + kvHeads * 7 + group + block);
        int budget = cfg.budget(block);
        int layer = 1;
        // prefill 600 positions in chunks, then decode steps; keys with a few loud channels (as real keys have)
        int pos = 0;
        int[] chunks = {64, 64, 64, 64, 64, 64, 64, 64, 64, 56};
        for (int c : chunks) {
            seq.beginWrite(pos, c);
            float[] k = keys(r, c, a);
            float[] v = random(r, c * a.kvDim());
            for (int l = 0; l < layers; l++) {
                kv.store(l, pos, k, v, c);
            }
            pos += c;
        }
        int end = pos; // 600
        // 1. summaries are exact for every full block and for the partial last one
        boolean sumOk = true;
        boolean contained = true;
        int nb = (end + block - 1) / block;
        for (int j = 0; j < nb; j++) {
            int rows = Math.min(block, end - j * block);
            for (int g = 0; g < kvHeads; g++) {
                float[] got = stored(st, layer, seq.block(j), g);
                float[] ex = exact(st, layer, seq.block(j), g, rows);
                for (int i = 0; i < hd; i++) {
                    sumOk &= got[i] == Float.float16ToFloat(KvSummary.down(ex[i])) && got[hd + i] == Float.float16ToFloat(KvSummary.up(ex[hd + i]));
                    contained &= got[i] <= ex[i] && got[hd + i] >= ex[hd + i];
                }
            }
        }
        check(sumOk, what + ": summary = outward-rounded min / max of the valid rows");
        check(contained, what + ": summary contains every stored key");
        // the dense layer keeps no summary work: only sparse layers are updated (layer 0 is dense), checked by exactness above for layer 1
        // 2. decode steps: bound validity, selection, and the output against float64 over the selected keys
        boolean boundOk = true;
        boolean selOk = true;
        boolean sinkWin = true;
        double worst = 0;
        for (int step = 0; step < 6; step++) {
            int p = end + step;
            seq.beginWrite(p, 1);
            float[] k = keys(r, 1, a);
            float[] v = random(r, a.kvDim());
            for (int l = 0; l < layers; l++) {
                kv.store(l, p, k, v, 1);
            }
            float[] q = queries(r, a, step, kv, st, layer, seq, p, dt);
            float[] att = new float[a.qDim()];
            kv.attention(pool, layer, 1, p, q, att, 0, a.scale(), cap);
            int nBlocks = p / block + 1;
            check(nBlocks > budget, what + ": test context is above the budget");
            int sink = cfg.sinkBlocks(block);
            int win = cfg.windowBlocks(block);
            float[] qr = q;
            float scale = a.scale();
            if (rot) {
                qr = q.clone();
                for (int o = 0; o < qr.length; o += hd) {
                    KvQ4.rotate(qr, o, hd);
                }
                scale /= KvQ4.block(hd);
            }
            for (int g = 0; g < kvHeads; g++) {
                int[] sel = kv.lastSelection[g];
                // the bound of every candidate, in float64 from the stored summaries, against the true best score in the block
                double[] bound = new double[nBlocks];
                for (int j = sink; j < nBlocks - win; j++) {
                    float[] sm = stored(st, layer, seq.block(j), g);
                    double best = Double.NEGATIVE_INFINITY;
                    for (int rr = 0; rr < group; rr++) {
                        double b = 0;
                        double top = Double.NEGATIVE_INFINITY;
                        int qb = (g * group + rr) * hd;
                        for (int i = 0; i < hd; i++) {
                            b += Math.max((double) qr[qb + i] * sm[i], (double) qr[qb + i] * sm[hd + i]);
                        }
                        for (int t = 0; t < block; t++) {
                            double sc = 0;
                            for (int i = 0; i < hd; i++) {
                                sc += (double) qr[qb + i] * st.keyValue(layer, seq.block(j), g, i, t);
                            }
                            top = Math.max(top, sc);
                        }
                        boundOk &= b >= top - 1e-4 * (1 + Math.abs(top));
                        best = Math.max(best, b);
                    }
                    bound[j] = best;
                }
                // the selection: sinks, the window, and the candidates no worse than any candidate left out
                boolean sw = sel.length == sink + 6 + win;
                for (int j = 0; j < sink; j++) {
                    sw &= sel[j] == j;
                }
                for (int j = 0; j < win; j++) {
                    sw &= sel[sel.length - 1 - j] == nBlocks - 1 - j;
                }
                sinkWin &= sw;
                boolean[] in = new boolean[nBlocks];
                for (int j : sel) {
                    in[j] = true;
                }
                double minIn = Double.POSITIVE_INFINITY;
                double maxOut = Double.NEGATIVE_INFINITY;
                for (int j = sink; j < nBlocks - win; j++) {
                    if (in[j]) {
                        minIn = Math.min(minIn, bound[j]);
                    } else {
                        maxOut = Math.max(maxOut, bound[j]);
                    }
                }
                selOk &= minIn >= maxOut - 1e-4 * (1 + Math.abs(maxOut));
                // 3. the exact output over the selected keys
                for (int rr = 0; rr < group; rr++) {
                    int h = g * group + rr;
                    double[] ref = reference(st, layer, seq, g, qr, h * hd, hd, sel, block, p, scale, cap);
                    for (int i = 0; i < hd; i++) {
                        worst = Math.max(worst, Math.abs(ref[i] - att[h * hd + i]));
                    }
                }
            }
        }
        check(boundOk, what + ": bound >= the best score of every candidate block");
        check(sinkWin, what + ": selection = sinks + 6 blocks + window, ascending");
        check(selOk, what + ": the chosen candidates all bound at least as high as the rejected ones");
        check(worst < 2e-4, what + ": output equals float64 attention over the selected keys (max error " + worst + ")");

        // 4. a verification chunk of rows equals separate decode steps; rows below the budget are dense
        if (dt != KvDtype.Q8 || !rot) {
            chunkEqualsSteps(pool, dt, hd, kvHeads, group, block, cap, rot, cfg);
        }
        kp.close();
    }

    /** Keys with loud channels (a few channels 5x larger), so that the bound is not a random walk. */
    private static float[] keys(Random r, int n, Attention.Gqa a) {
        float[] k = random(r, n * a.kvDim());
        for (int t = 0; t < n; t++) {
            for (int g = 0; g < a.kvHeads(); g++) {
                for (int c = 0; c < 3; c++) {
                    k[t * a.kvDim() + g * a.headDim() + (c * 11 + g) % a.headDim()] *= 5f;
                }
            }
        }
        return k;
    }

    /** Queries; every other step one aims at an old key so that the selection has something to find. */
    private static float[] queries(Random r, Attention.Gqa a, int step, PagedKv kv, CpuKvStorage st, int layer, KvSequence seq, int p,
                                   KvDtype dt) {
        float[] q = random(r, a.qDim());
        if (step % 2 == 1) {
            int target = 77 + step * 13; // a position in the middle of the context
            for (int h = 0; h < a.heads(); h++) {
                int g = h / (a.heads() / a.kvHeads());
                for (int i = 0; i < a.headDim(); i++) {
                    q[h * a.headDim() + i] = 2f * st.keyValue(layer, seq.block(target / st.B), g, i, target % st.B) + 0.3f * q[h * a.headDim() + i];
                }
            }
        }
        return q;
    }

    /** Float64 attention of head {@code h} (query {@code q[qb ..]}) over the keys of the blocks {@code sel} at positions {@code <= p}. */
    private static double[] reference(CpuKvStorage st, int layer, KvSequence seq, int g, float[] q, int qb, int hd, int[] sel,
                                     int block, int p, float scale, float cap) {
        int n = 0;
        double[] s = new double[sel.length * block];
        int[] bl = new int[s.length];
        int[] ro = new int[s.length];
        for (int j : sel) {
            for (int t = 0; t < block; t++) {
                if (j * block + t <= p) {
                    double d = 0;
                    for (int i = 0; i < hd; i++) {
                        d += (double) q[qb + i] * st.keyValue(layer, seq.block(j), g, i, t);
                    }
                    d *= scale;
                    if (cap > 0) {
                        d = Math.tanh(d / cap) * cap;
                    }
                    s[n] = d;
                    bl[n] = seq.block(j);
                    ro[n] = t;
                    n++;
                }
            }
        }
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, s[i]);
        }
        double sum = 0;
        for (int i = 0; i < n; i++) {
            s[i] = Math.exp(s[i] - max);
            sum += s[i];
        }
        double[] out = new double[hd];
        for (int i = 0; i < n; i++) {
            for (int e = 0; e < hd; e++) {
                out[e] += s[i] / sum * st.valueValue(layer, bl[i], g, e, ro[i]);
            }
        }
        return out;
    }

    /** Rows of a short chunk attend as separate decode steps, around the point where the context passes the budget. */
    private static void chunkEqualsSteps(ThreadPool pool, KvDtype dt, int hd, int kvHeads, int group, int block, float cap, boolean rot,
                                         KvSparse cfg) {
        String what = dt + " hd=" + hd + " block=" + block + ": chunk rows";
        int layers = 2;
        int ctx = 600;
        Attention.Gqa a = new Attention.Gqa(kvHeads * group, kvHeads, hd, 0, 0, false, false, false, (float) (1 / Math.sqrt(hd)), cap);
        int budget = cfg.budget(block);
        int start = budget * block - 6; // rows straddle the first position that has more blocks than the budget
        CpuKvStorage st1 = storage(dt, layers, a, block, rot);
        CpuKvStorage st2 = storage(dt, layers, a, block, rot);
        KvPool k1 = new KvPool(st1, 100, "t");
        KvPool k2 = new KvPool(st2, 100, "t");
        KvSequence s1 = k1.newSequence(ctx);
        KvSequence s2 = k2.newSequence(ctx);
        PagedKv a1 = new PagedKv(layers, a, ctx, st1, s1);
        PagedKv a2 = new PagedKv(layers, a, ctx, st2, s2);
        Random r = new Random(9);
        float[] k = keys(r, start, a);
        float[] v = random(r, start * a.kvDim());
        s1.beginWrite(0, start);
        s2.beginWrite(0, start);
        for (int l = 0; l < layers; l++) {
            a1.store(l, 0, k, v, start);
            a2.store(l, 0, k, v, start);
        }
        int n = 12;
        float[] k2v = keys(r, n, a);
        float[] v2 = random(r, n * a.kvDim());
        float[] q = random(r, n * a.qDim());
        s1.beginWrite(start, n);
        for (int l = 0; l < layers; l++) {
            a1.store(l, start, k2v, v2, n);
        }
        float[] chunk = new float[n * a.qDim()];
        a1.attention(pool, 1, n, start, q, chunk, 0, a.scale(), cap);
        // the same rows one at a time, each stored just before it is read, as decode steps do
        float[] steps = new float[n * a.qDim()];
        for (int t = 0; t < n; t++) {
            s2.beginWrite(start + t, 1);
            for (int l = 0; l < layers; l++) {
                a2.store(l, start + t, Arrays.copyOfRange(k2v, t * a.kvDim(), (t + 1) * a.kvDim()),
                        Arrays.copyOfRange(v2, t * a.kvDim(), (t + 1) * a.kvDim()), 1);
            }
            float[] o = new float[a.qDim()];
            a2.attention(pool, 1, 1, start + t, Arrays.copyOfRange(q, t * a.qDim(), (t + 1) * a.qDim()), o, 0, a.scale(), cap);
            System.arraycopy(o, 0, steps, t * a.qDim(), a.qDim());
        }
        check(a1.sparseRows > 0 && a1.sparseRows < n, what + ": some rows are dense, some sparse (" + a1.sparseRows + " of " + n + ")");
        check(Arrays.equals(chunk, steps), what + ": a chunk of " + n + " rows equals " + n + " decode steps bit for bit");
        k1.close();
        k2.close();
    }

    /** A planted key along the query is found among 100+ blocks, for every cache type; the same query with the key absent is not required to find the block. */
    private static void needle(ThreadPool pool) {
        for (KvDtype dt : new KvDtype[]{KvDtype.F32, KvDtype.F16, KvDtype.Q8}) {
            for (boolean rot : dt == KvDtype.Q8 ? new boolean[]{false, true} : new boolean[]{false}) {
                int hd = 128;
                int block = 16;
                Attention.Gqa a = new Attention.Gqa(4, 2, hd, 0, 0, false, false, false, (float) (1 / Math.sqrt(hd)), 0f);
                KvSparse cfg = new KvSparse(true, 4, 16, 32, 0);
                KvSparse.set(cfg);
                int ctx = 3200;
                CpuKvStorage st = storage(dt, 1, a, block, rot);
                KvPool kp = new KvPool(st, 300, "t");
                KvSequence seq = kp.newSequence(ctx);
                PagedKv kv = new PagedKv(1, a, ctx, st, seq);
                kv.recordSelection = true;
                Random r = new Random(3);
                int found = 0;
                int trials = 20;
                float[] q = random(r, a.qDim());
                for (int trial = 0; trial < trials; trial++) {
                    int needleBlock = 20 + r.nextInt(150);
                    int needlePos = needleBlock * block + r.nextInt(block);
                    seq.beginWrite(0, 3000);
                    for (int pos = 0; pos < 3000; pos += 100) {
                        int c = Math.min(100, 3000 - pos);
                        float[] k = random(r, c * a.kvDim());
                        float[] v = random(r, c * a.kvDim());
                        if (needlePos >= pos && needlePos < pos + c) {
                            for (int g = 0; g < 2; g++) {
                                for (int i = 0; i < hd; i++) { // a key along the head's query
                                    k[(needlePos - pos) * a.kvDim() + g * hd + i] = 1.5f * q[(g * 2) * hd + i];
                                }
                            }
                        }
                        kv.store(0, pos, k, v, c);
                    }
                    float[] att = new float[a.qDim()];
                    seq.beginWrite(3000, 1);
                    kv.store(0, 3000, random(r, a.kvDim()), random(r, a.kvDim()), 1);
                    kv.attention(pool, 0, 1, 3000, q, att, 0, a.scale(), 0f);
                    boolean both = true;
                    for (int g = 0; g < 2; g++) {
                        both &= Arrays.binarySearch(kv.lastSelection[g], needleBlock) >= 0;
                    }
                    found += both ? 1 : 0;
                }
                check(found == trials, dt + (rot ? " rotated" : "") + ": the planted key's block is selected in " + found + " of " + trials);
                kp.close();
            }
        }
    }
}
