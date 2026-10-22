package org.robomiri.jllm.cpu;

import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.kv.KvSequence;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;

/**
 * A million positions through the paged CPU cache (docs/24 §24.7): the byte offsets of its blocks pass 2^32 (4.3 GB of
 * f32 keys and values over four layers of a small model), a position times the row width times the layers exceeds
 * 2^31, and attention of the last token over all 1 048 576 keys, and of a 64-token chunk at the end, equals a
 * float64 reference computed straight from the stored values. A flat cache cannot do this with {@code int} indexes (it
 * refuses a layer array of more than 2^31 elements); every offset of the paged one is a {@code long}.
 *
 * <p>usage: {@code LongContextTest [positions]} (default 1048576; needs ~5 GB of memory outside the Java heap)
 */
public final class LongContextTest {

    public static void main(String[] args) throws Exception {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 1 << 20;
        int layers = 4;
        int kvHeads = 4;
        int group = 2;
        int hd = 32;
        Attention.Gqa a = new Attention.Gqa(kvHeads * group, kvHeads, hd, 0, 0, false, false, false, (float) (1 / Math.sqrt(hd)), 0f);
        int kvDim = a.kvDim();
        CpuKvStorage st = new CpuKvStorage(KvDtype.F32, layers, a, 16);
        int blocks = (n + 15) / 16;
        long bytes = (long) blocks * st.blockBytes;
        System.out.printf("%d positions, %d blocks of 16, %.2f GB; position x kvDim x layers = %d (2^31 = %d)%n", n, blocks, bytes / 1e9,
                (long) n * kvDim * layers, 1L << 31);
        if (bytes <= (1L << 31) && n >= (1 << 20)) {
            throw new AssertionError("the test should pass 2^31 bytes");
        }
        KvPool pool = new KvPool(st, blocks + 2, "long");
        KvSequence seq = pool.newSequence(n);
        PagedKv kv = new PagedKv(layers, a, n, st, seq);
        long t0 = System.nanoTime();
        // row p: k[i] and v[i] are fixed functions of (position, layer, element), so the reference needs no copy
        float[] k = new float[64 * kvDim];
        float[] v = new float[64 * kvDim];
        for (int pos = 0; pos < n; pos += 64) {
            int m = Math.min(64, n - pos);
            seq.beginWrite(pos, m);
            for (int l = 0; l < layers; l++) {
                for (int t = 0; t < m; t++) {
                    for (int e = 0; e < kvDim; e++) {
                        k[t * kvDim + e] = key(pos + t, l, e);
                        v[t * kvDim + e] = val(pos + t, l, e);
                    }
                }
                kv.store(l, pos, k, v, m);
            }
        }
        System.out.printf("stored in %.1f s; resident blocks %d%n", (System.nanoTime() - t0) / 1e9, pool.residentBlocks());
        int failures = 0;
        try (ThreadPool tp = new ThreadPool(4)) {
            for (int layer : new int[]{0, layers - 1}) {
                // the last token (decode), and a 64-token chunk at the end (prefill tiles)
                for (int n1 : new int[]{1, 64}) {
                    int pos = n - n1;
                    float[] q = new float[n1 * a.qDim()];
                    for (int i = 0; i < q.length; i++) {
                        q[i] = (float) Math.sin(i * 0.37 + layer);
                    }
                    float[] out = new float[n1 * a.qDim()];
                    long t1 = System.nanoTime();
                    kv.attention(tp, layer, n1, pos, q, out, 0, a.scale(), 0f);
                    double ms = (System.nanoTime() - t1) / 1e6;
                    double worst = 0;
                    int[] rows = n1 == 1 ? new int[]{0} : new int[]{0, 31, 63};
                    for (int r : rows) {
                        for (int h = 0; h < a.heads(); h += 3) {
                            worst = Math.max(worst, reference(q, r * a.qDim() + h * hd, pos + r + 1, h / group, layer, a, out, r * a.qDim() + h * hd));
                        }
                    }
                    boolean ok = worst < 2e-5;
                    System.out.printf("%s layer %d, %d token(s) at %d: %.0f ms, max error against float64 %.2e%n", ok ? "ok  " : "FAIL",
                            layer, n1, pos, ms, worst);
                    if (!ok) {
                        failures++;
                    }
                }
            }
        }
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("long context test passed");
    }

    static float key(int pos, int layer, int e) {
        return ((pos * 31L + e * 17L + layer * 7L) % 257) / 128f - 1f;
    }

    static float val(int pos, int layer, int e) {
        return ((pos * 13L + e * 29L + layer * 3L) % 251) / 125f - 1f;
    }

    /** Max abs difference between {@code out[oo ..]} and the float64 attention of query {@code q[qo ..]} over keys [0, len). */
    private static double reference(float[] q, int qo, int len, int kvh, int layer, Attention.Gqa a, float[] out, int oo) {
        int hd = a.headDim();
        double[] s = new double[len];
        double max = Double.NEGATIVE_INFINITY;
        for (int p = 0; p < len; p++) {
            double d = 0;
            for (int i = 0; i < hd; i++) {
                d += (double) q[qo + i] * key(p, layer, kvh * hd + i);
            }
            s[p] = d * a.scale();
            max = Math.max(max, s[p]);
        }
        double sum = 0;
        for (int p = 0; p < len; p++) {
            s[p] = Math.exp(s[p] - max);
            sum += s[p];
        }
        double worst = 0;
        for (int i = 0; i < hd; i++) {
            double acc = 0;
            for (int p = 0; p < len; p++) {
                acc += s[p] * val(p, layer, kvh * hd + i);
            }
            worst = Math.max(worst, Math.abs(acc / sum - out[oo + i]));
        }
        return worst;
    }
}
