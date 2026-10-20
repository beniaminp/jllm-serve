package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;

import java.util.Arrays;

/**
 * The CPU backend's KV cache: per layer one K and one V matrix of {@code [ctx, kvDim]}, row-major, on the Java
 * heap, and the causal grouped-query attention over it. {@link F32} holds plain floats that the attention loop
 * reads in place. {@link F16} holds IEEE half floats in {@code short[]} (converted with
 * {@link Float#floatToFloat16} / {@link Float#float16ToFloat}, both JIT intrinsics) and expands each cached row
 * into thread-local scratch before the f32 SIMD kernels read it. The math is f32 in both cases; f16 only changes
 * what is stored between forward calls.
 *
 * <p><b>Tiling.</b> For {@code n > 1} the work is split into tiles of one kv head and up to {@link #MAX_TILE}
 * consecutive tokens (fewer when their score rows over the {@code pos + n} keys would exceed {@link #SCORE_BUDGET}), which serve {@code tile × group} query heads. Each cached K row is read (f32) or expanded
 * (f16) once per tile and dotted with all of the tile's query heads; the scores of every query head are kept for
 * all its keys, the softmax runs per head, and each V row is read / expanded once per tile and added to every
 * head's output. Per output element nothing changes against the per-(token, head) loop: the same {@code dot}
 * over {@code headDim}, max and sum over the keys in ascending order, and the same ascending {@code axpy}
 * accumulation, so the results are bit-identical; only the memory traffic drops by {@code tile × group}.
 * Decode ({@code n == 1}) on f32 keeps one task per query head (more tasks than kv heads); f16 decode is the
 * one-token tile, which converts each row once per kv head instead of once per query head.
 */
public abstract class KvCache {

    final int layers;
    final int ctx;
    final int kvDim;
    final int kvHeads;
    final int headDim;
    final int qDim;
    /** Query heads per kv head. */
    final int group;
    /**
     * Most tokens per prefill tile: each K / V row is read (and, f16, expanded) once per tile. Measured on a 6 K-token
     * prefill, 16 and 64 run equally fast; 16 needs a quarter of the scratch and leaves more tasks to balance.
     */
    static final int MAX_TILE = 16;
    /** Score floats per thread a prefill tile may use ({@code tile × group × (pos + n)}): 8 MB. */
    static final int SCORE_BUDGET = 1 << 21;
    /** Per thread, grown on demand: [tile × group × span scores][headDim K row][headDim V row][tile × group 1/sum]. */
    final ThreadLocal<float[]> scratch;

    /** @param a the head shape shared by every layer */
    KvCache(int layers, Attention.Gqa a, int ctx) {
        this.layers = layers;
        this.ctx = ctx;
        this.kvDim = a.kvDim();
        this.kvHeads = a.kvHeads();
        this.headDim = a.headDim();
        this.qDim = a.qDim();
        this.group = a.heads() / a.kvHeads();
        int decode = ctx + 2 * headDim + group; // f32 decode scores one head over the whole context
        scratch = ThreadLocal.withInitial(() -> new float[decode]);
    }

    /** This thread's scratch, at least {@code len} floats. */
    final float[] scratch(int len) {
        float[] a = scratch.get();
        if (a.length < len) {
            a = new float[len];
            scratch.set(a);
        }
        return a;
    }

    /** Heap bytes of the keys and values of {@code layers} layers for {@code ctx} positions. */
    public static long bytes(KvDtype dtype, int layers, Attention.Gqa a, int ctx) {
        return 2L * layers * ctx * a.kvDim() * (dtype == KvDtype.F16 ? 2 : 4);
    }

    /**
     * Refuses a cache that cannot fit, rather than letting the JVM spend its time in back-to-back full collections:
     * arrays this large live in the old generation, which with the parallel collector is only part of the heap.
     */
    private static void checkHeap(long need, KvDtype dtype, int ctx) {
        long free = oldGenFree();
        if (need > 0.92 * free) {
            System.gc(); // what looked used may be garbage
            free = oldGenFree();
        }
        if (need > 0.92 * free) {
            throw new IllegalStateException(String.format(java.util.Locale.ROOT, "the %s KV cache for a context of %d"
                    + " needs %.2f GB of Java heap, but only %.2f GB are free (heap -Xmx %.1f GB): lower backend.ctx,"
                    + " use backend.kv-dtype=f16, or give the JVM more heap (JLLM_XMX=16g)", dtype, ctx, need / 1e9,
                    free / 1e9, Runtime.getRuntime().maxMemory() / 1e9));
        }
    }

    /** Free space of the old generation (G1: the heap), or of the whole heap when the pools are unknown. */
    private static long oldGenFree() {
        for (java.lang.management.MemoryPoolMXBean p : java.lang.management.ManagementFactory.getMemoryPoolMXBeans()) {
            String n = p.getName();
            if (p.getType() == java.lang.management.MemoryType.HEAP && (n.contains("Old") || n.contains("Tenured"))) {
                long max = p.getUsage().getMax();
                if (max > 0) {
                    return max - p.getUsage().getUsed();
                }
            }
        }
        Runtime rt = Runtime.getRuntime();
        return rt.maxMemory() - (rt.totalMemory() - rt.freeMemory());
    }

    public static KvCache create(KvDtype dtype, int layers, Attention.Gqa a, int ctx) {
        long cacheLen = (long) ctx * a.kvDim();
        if (cacheLen > Integer.MAX_VALUE - 16) {
            throw new IllegalArgumentException("context " + ctx + " is too large for the CPU KV cache");
        }
        checkHeap(bytes(dtype, layers, a, ctx), dtype, ctx);
        return switch (dtype) {
            case F32 -> new F32(layers, a, ctx);
            case F16 -> new F16(layers, a, ctx);
        };
    }

    /**
     * Stores the {@code n} rows of {@code k} and {@code v} (each {@code [n, kvDim]} row-major, f32, already
     * normalized/rotated) at positions {@code [pos, pos + n)} of layer {@code layer}.
     */
    public abstract void store(int layer, int pos, float[] k, float[] v, int n);

    /**
     * Causal grouped-query attention for the tokens at positions {@code [pos, pos + n)}: reads the queries
     * {@code q[t * qDim + h * headDim ..]}, writes {@code att} in the same layout.
     *
     * @param window  sliding-window length in tokens, 0 = full causal attention
     * @param scale   multiplier of q.k
     * @param softcap {@code tanh(s / cap) * cap} on the scaled score, 0 = off
     */
    public abstract void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window,
                                   float scale, float softcap);

    /** The f32 K (or V) rows of {@code layer} when they can be read in place, null when they must be expanded. */
    abstract float[] rowsInPlace(int layer, boolean values);

    /** Expands the {@code headDim} values of kv head slice {@code ko} of row {@code s} into {@code dst[d ..]}. */
    abstract void expand(int layer, boolean values, int s, int ko, float[] dst, int d);

    private static final boolean FASTMATH = VectorKernels.FASTMATH;

    /**
     * Softmax numerator of the {@code n} scaled scores at {@code sc[o ..]} in place, returning their sum: the
     * max in ascending order, {@code exp(s - max)} and the sum in ascending order. Fast math
     * ({@link VectorKernels#FASTMATH}) applies the soft-cap here on lanes (the score pass left it out) and uses the
     * polynomial exp; otherwise the scalar {@code Math.exp} of the original loop.
     */
    static float softmax(float[] sc, int o, int n, float softcap) {
        if (softcap > 0 && FASTMATH) {
            VectorKernels.softcapFast(sc, o, n, softcap);
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int i = o; i < o + n; i++) {
            max = Math.max(max, sc[i]);
        }
        return VectorKernels.softmaxExp(sc, o, n, max);
    }

    /** Keys per block of the blocked kernel: their K or V rows (f32) take 64 KB at headDim 128. */
    static final int KEY_BLOCK = 128;

    /**
     * The tiled kernel described in the class comment, for every (kv head, token tile) pair of the chunk. The keys
     * are walked in blocks of {@link #KEY_BLOCK}: an f16 cache expands a block's K (then V) rows once into scratch,
     * the scores are computed four query heads at a time sharing each key load, and the outputs accumulate in
     * registers, four rows at a time, instead of going through memory once per key. Every output element still sees
     * the same operations in the same (ascending key) order as {@link VectorKernels#dot} / {@link VectorKernels#axpy}
     * one key at a time, so the results are unchanged.
     */
    final void attentionTiled(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window,
                              float scale, float softcap) {
        int hd = headDim;
        int fl = VectorKernels.FL;
        if (hd % fl != 0) {
            attentionTiledSimple(pool, layer, n, pos, q, att, window, scale, softcap);
            return;
        }
        // tiles are sized by the keys this chunk sees, not by the context: at a 32 K context a fixed budget left
        // one token per tile, so every K / V row was expanded and read once per token instead of once per tile
        int span = pos + n;
        // ... but at least two tasks per thread, so that wide machines with few kv heads stay busy
        int minTiles = (2 * pool.threads() + kvHeads - 1) / kvHeads;
        int tile = Math.max(1, Math.min(Math.min(MAX_TILE, (n + minTiles - 1) / minTiles),
                Math.min(n, SCORE_BUDGET / (group * span))));
        int tiles = (n + tile - 1) / tile;
        int blkOff = tile * group * span;              // one block of K or V rows, hd floats each
        int invOff = blkOff + KEY_BLOCK * hd;
        int scratchLen = invOff + tile * group;
        float[] kIn = rowsInPlace(layer, false);
        float[] vIn = rowsInPlace(layer, true);
        boolean capped = softcap > 0 && !FASTMATH;
        // later tiles see more keys: hand them out first so the tail of the job is made of the light ones
        pool.forRangeFine(kvHeads * tiles, 1, (i0, i1) -> {
            float[] sc = scratch(scratchLen);
            int[] rq = new int[tile * group];   // query offset of each row (token t, head h) of the tile
            int[] lo = new int[tile * group];   // its first visible key
            int[] hi = new int[tile * group];   // one past its last visible key
            for (int i = i0; i < i1; i++) {
                int g = i % kvHeads;
                int t0 = (tiles - 1 - i / kvHeads) * tile;
                int t1 = Math.min(n, t0 + tile);
                int ko = g * hd;
                int rows = (t1 - t0) * group;
                for (int t = t0; t < t1; t++) {
                    int len = pos + t + 1;
                    for (int h = 0; h < group; h++) {
                        int r = (t - t0) * group + h;
                        rq[r] = t * qDim + (g * group + h) * hd;
                        lo[r] = window > 0 ? Math.max(0, len - window) : 0;
                        hi[r] = len;
                    }
                }
                int sMin = lo[0];
                int sMax = hi[rows - 1];

                // scores: sc[r * span + s] for the visible keys of every row
                for (int b0 = sMin; b0 < sMax; b0 += KEY_BLOCK) {
                    int b1 = Math.min(sMax, b0 + KEY_BLOCK);
                    float[] kArr = kIn != null ? kIn : sc;
                    if (kIn == null) {
                        for (int s = b0; s < b1; s++) {
                            expand(layer, false, s, ko, sc, blkOff + (s - b0) * hd);
                        }
                    }
                    for (int s = b0; s < b1; s++) {
                        int koff = kIn != null ? s * kvDim + ko : blkOff + (s - b0) * hd;
                        int r = 0;
                        while (r < rows) {
                            if (s < lo[r] || s >= hi[r]) {
                                r++;
                                continue;
                            }
                            if (r + 3 < rows && s >= lo[r + 3] && s < hi[r + 3] && s >= lo[r + 1] && s < hi[r + 1]
                                    && s >= lo[r + 2] && s < hi[r + 2]) {
                                dot4(q, rq[r], rq[r + 1], rq[r + 2], rq[r + 3], kArr, koff, hd, sc, r * span + s, span,
                                        scale, softcap, capped);
                                r += 4;
                            } else {
                                float d = VectorKernels.dot(q, rq[r], kArr, koff, hd) * scale;
                                sc[r * span + s] = capped ? (float) Math.tanh(d / softcap) * softcap : d;
                                r++;
                            }
                        }
                    }
                }
                for (int r = 0; r < rows; r++) {
                    sc[invOff + r] = 1f / softmax(sc, r * span + lo[r], hi[r] - lo[r], softcap);
                    Arrays.fill(att, rq[r], rq[r] + hd, 0f);
                }

                // outputs: att[row] += p(row, s) * v[s], ascending s, accumulated in registers within a block
                for (int b0 = sMin; b0 < sMax; b0 += KEY_BLOCK) {
                    int b1 = Math.min(sMax, b0 + KEY_BLOCK);
                    float[] vArr = vIn != null ? vIn : sc;
                    int vBase;
                    int vStride;
                    if (vIn != null) {
                        vBase = b0 * kvDim + ko;
                        vStride = kvDim;
                    } else {
                        for (int s = b0; s < b1; s++) {
                            expand(layer, true, s, ko, sc, blkOff + (s - b0) * hd);
                        }
                        vBase = blkOff;
                        vStride = hd;
                    }
                    int r = 0;
                    for (; r + 3 < rows; r += 4) {
                        accumulate4(sc, span, invOff, r, lo, hi, b0, b1, vArr, vBase, vStride, att, rq, hd);
                    }
                    for (; r < rows; r++) {
                        accumulate1(sc, span, invOff, r, Math.max(lo[r], b0), Math.min(hi[r], b1), b0, vArr, vBase,
                                vStride, att, rq[r], hd);
                    }
                }
            }
        });
    }

    /**
     * Four rows' scores against one key, exactly as four {@link VectorKernels#dot} calls (two accumulators over
     * alternating vectors, then the scalar tail) but loading each key vector once.
     */
    private static void dot4(float[] q, int q0, int q1, int q2, int q3, float[] k, int ko, int hd, float[] sc, int so,
                             int stride, float scale, float softcap, boolean capped) {
        var sp = VectorKernels.F;
        int fl = VectorKernels.FL;
        jdk.incubator.vector.FloatVector a0 = jdk.incubator.vector.FloatVector.zero(sp), a1 = a0, b0 = a0, b1 = a0,
                c0 = a0, c1 = a0, d0 = a0, d1 = a0;
        int i = 0;
        int ub = hd - 2 * fl;
        for (; i <= ub; i += 2 * fl) {
            var k0 = jdk.incubator.vector.FloatVector.fromArray(sp, k, ko + i);
            var k1 = jdk.incubator.vector.FloatVector.fromArray(sp, k, ko + i + fl);
            a0 = jdk.incubator.vector.FloatVector.fromArray(sp, q, q0 + i).fma(k0, a0);
            a1 = jdk.incubator.vector.FloatVector.fromArray(sp, q, q0 + i + fl).fma(k1, a1);
            b0 = jdk.incubator.vector.FloatVector.fromArray(sp, q, q1 + i).fma(k0, b0);
            b1 = jdk.incubator.vector.FloatVector.fromArray(sp, q, q1 + i + fl).fma(k1, b1);
            c0 = jdk.incubator.vector.FloatVector.fromArray(sp, q, q2 + i).fma(k0, c0);
            c1 = jdk.incubator.vector.FloatVector.fromArray(sp, q, q2 + i + fl).fma(k1, c1);
            d0 = jdk.incubator.vector.FloatVector.fromArray(sp, q, q3 + i).fma(k0, d0);
            d1 = jdk.incubator.vector.FloatVector.fromArray(sp, q, q3 + i + fl).fma(k1, d1);
        }
        float sa = a0.add(a1).reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        float sb = b0.add(b1).reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        float scc = c0.add(c1).reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        float sd = d0.add(d1).reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        for (; i < hd; i++) {
            sa += q[q0 + i] * k[ko + i];
            sb += q[q1 + i] * k[ko + i];
            scc += q[q2 + i] * k[ko + i];
            sd += q[q3 + i] * k[ko + i];
        }
        sc[so] = cap(sa * scale, softcap, capped);
        sc[so + stride] = cap(sb * scale, softcap, capped);
        sc[so + 2 * stride] = cap(scc * scale, softcap, capped);
        sc[so + 3 * stride] = cap(sd * scale, softcap, capped);
    }

    private static float cap(float d, float softcap, boolean capped) {
        return capped ? (float) Math.tanh(d / softcap) * softcap : d;
    }

    /**
     * Rows {@code r .. r+3}: their keys of block {@code [b0, b1)} added to their outputs. The keys all four see run
     * with four independent accumulators per vector of the head; the others row by row, in ascending order.
     */
    private void accumulate4(float[] sc, int span, int invOff, int r, int[] lo, int[] hi, int b0, int b1, float[] v,
                             int vBase, int vStride, float[] att, int[] rq, int hd) {
        int a = Math.max(b0, Math.max(Math.max(lo[r], lo[r + 1]), Math.max(lo[r + 2], lo[r + 3])));
        int b = Math.min(b1, Math.min(Math.min(hi[r], hi[r + 1]), Math.min(hi[r + 2], hi[r + 3])));
        if (a >= b) { // nothing in common in this block (rare: near the diagonal or a window edge)
            for (int k = 0; k < 4; k++) {
                accumulate1(sc, span, invOff, r + k, Math.max(lo[r + k], b0), Math.min(hi[r + k], b1), b0, v, vBase,
                        vStride, att, rq[r + k], hd);
            }
            return;
        }
        for (int k = 0; k < 4; k++) { // keys before the common range
            accumulate1(sc, span, invOff, r + k, Math.max(lo[r + k], b0), a, b0, v, vBase, vStride, att, rq[r + k], hd);
        }
        var sp = VectorKernels.F;
        int fl = VectorKernels.FL;
        float i0 = sc[invOff + r];
        float i1 = sc[invOff + r + 1];
        float i2 = sc[invOff + r + 2];
        float i3 = sc[invOff + r + 3];
        int p0 = r * span;
        int p1 = p0 + span;
        int p2 = p1 + span;
        int p3 = p2 + span;
        for (int c = 0; c < hd; c += fl) {
            var x0 = jdk.incubator.vector.FloatVector.fromArray(sp, att, rq[r] + c);
            var x1 = jdk.incubator.vector.FloatVector.fromArray(sp, att, rq[r + 1] + c);
            var x2 = jdk.incubator.vector.FloatVector.fromArray(sp, att, rq[r + 2] + c);
            var x3 = jdk.incubator.vector.FloatVector.fromArray(sp, att, rq[r + 3] + c);
            for (int s = a; s < b; s++) {
                var vv = jdk.incubator.vector.FloatVector.fromArray(sp, v, vBase + (s - b0) * vStride + c);
                x0 = vv.fma(jdk.incubator.vector.FloatVector.broadcast(sp, sc[p0 + s] * i0), x0);
                x1 = vv.fma(jdk.incubator.vector.FloatVector.broadcast(sp, sc[p1 + s] * i1), x1);
                x2 = vv.fma(jdk.incubator.vector.FloatVector.broadcast(sp, sc[p2 + s] * i2), x2);
                x3 = vv.fma(jdk.incubator.vector.FloatVector.broadcast(sp, sc[p3 + s] * i3), x3);
            }
            x0.intoArray(att, rq[r] + c);
            x1.intoArray(att, rq[r + 1] + c);
            x2.intoArray(att, rq[r + 2] + c);
            x3.intoArray(att, rq[r + 3] + c);
        }
        for (int k = 0; k < 4; k++) { // keys after it
            accumulate1(sc, span, invOff, r + k, b, Math.min(hi[r + k], b1), b0, v, vBase, vStride, att, rq[r + k], hd);
        }
    }

    /** Row {@code r}: keys {@code [from, to)} (inside the block starting at {@code b0}) added to its output. */
    private static void accumulate1(float[] sc, int span, int invOff, int r, int from, int to, int b0, float[] v,
                                    int vBase, int vStride, float[] att, int qo, int hd) {
        if (from >= to) {
            return;
        }
        var sp = VectorKernels.F;
        int fl = VectorKernels.FL;
        float inv = sc[invOff + r];
        int p = r * span;
        for (int c = 0; c < hd; c += fl) {
            var x = jdk.incubator.vector.FloatVector.fromArray(sp, att, qo + c);
            for (int s = from; s < to; s++) {
                x = jdk.incubator.vector.FloatVector.fromArray(sp, v, vBase + (s - b0) * vStride + c)
                        .fma(jdk.incubator.vector.FloatVector.broadcast(sp, sc[p + s] * inv), x);
            }
            x.intoArray(att, qo + c);
        }
    }

    /** The tile loop one key at a time: used when {@code headDim} is not a multiple of the vector length. */
    private void attentionTiledSimple(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window,
                              float scale, float softcap) {
        int hd = headDim;
        // tiles are sized by the keys this chunk sees, not by the context: at a 32 K context a fixed budget left
        // one token per tile, so every K / V row was expanded and read once per token instead of once per tile
        int span = pos + n;
        int tile = Math.max(1, Math.min(MAX_TILE, Math.min(n, SCORE_BUDGET / (group * span))));
        int tiles = (n + tile - 1) / tile;
        int kOff = tile * group * span;
        int vOff = kOff + hd;
        int invOff = vOff + hd;
        int scratchLen = invOff + tile * group;
        float[] kIn = rowsInPlace(layer, false);
        float[] vIn = rowsInPlace(layer, true);
        // later tiles see more keys: hand them out first so the tail of the job is made of the light ones
        pool.forRangeFine(kvHeads * tiles, 1, (i0, i1) -> {
            float[] sc = scratch(scratchLen);
            float[] kArr = kIn != null ? kIn : sc;
            float[] vArr = vIn != null ? vIn : sc;
            for (int i = i0; i < i1; i++) {
                int g = i % kvHeads;
                int t0 = (tiles - 1 - i / kvHeads) * tile;
                int t1 = Math.min(n, t0 + tile);
                int ko = g * hd;
                int sMin = window > 0 ? Math.max(0, pos + t0 + 1 - window) : 0;
                int sMax = pos + t1;
                // scores: row (t - t0) * group + h holds the keys of query head g * group + h of token t
                for (int s = sMin; s < sMax; s++) {
                    int koff;
                    if (kIn != null) {
                        koff = s * kvDim + ko;
                    } else {
                        expand(layer, false, s, ko, sc, kOff);
                        koff = kOff;
                    }
                    for (int t = t0; t < t1; t++) {
                        int len = pos + t + 1;
                        int first = window > 0 ? Math.max(0, len - window) : 0;
                        if (s < first || s >= len) {
                            continue;
                        }
                        int qo = t * qDim + g * group * hd;
                        int row = (t - t0) * group;
                        for (int h = 0; h < group; h++) {
                            float d = VectorKernels.dot(q, qo + h * hd, kArr, koff, hd) * scale;
                            sc[(row + h) * span + s] = softcap > 0 && !FASTMATH ? (float) Math.tanh(d / softcap) * softcap : d;
                        }
                    }
                }
                for (int t = t0; t < t1; t++) {
                    int len = pos + t + 1;
                    int first = window > 0 ? Math.max(0, len - window) : 0;
                    int qo = t * qDim + g * group * hd;
                    for (int h = 0; h < group; h++) {
                        int row = (t - t0) * group + h;
                        int so = row * span;
                        sc[invOff + row] = 1f / softmax(sc, so + first, len - first, softcap);
                        Arrays.fill(att, qo + h * hd, qo + (h + 1) * hd, 0f);
                    }
                }
                for (int s = sMin; s < sMax; s++) {
                    int voff;
                    if (vIn != null) {
                        voff = s * kvDim + ko;
                    } else {
                        expand(layer, true, s, ko, sc, vOff);
                        voff = vOff;
                    }
                    for (int t = t0; t < t1; t++) {
                        int len = pos + t + 1;
                        int first = window > 0 ? Math.max(0, len - window) : 0;
                        if (s < first || s >= len) {
                            continue;
                        }
                        int qo = t * qDim + g * group * hd;
                        int row = (t - t0) * group;
                        for (int h = 0; h < group; h++) {
                            VectorKernels.axpy(sc[(row + h) * span + s] * sc[invOff + row + h], vArr, voff, att, qo + h * hd, hd);
                        }
                    }
                }
            }
        });
    }

    // ------------------------------------------------------------------ f32

    static final class F32 extends KvCache {
        final float[][] key;
        final float[][] val;

        F32(int layers, Attention.Gqa a, int ctx) {
            super(layers, a, ctx);
            key = new float[layers][ctx * kvDim];
            val = new float[layers][ctx * kvDim];
        }

        @Override
        public void store(int layer, int pos, float[] k, float[] v, int n) {
            System.arraycopy(k, 0, key[layer], pos * kvDim, n * kvDim);
            System.arraycopy(v, 0, val[layer], pos * kvDim, n * kvDim);
        }

        @Override
        float[] rowsInPlace(int layer, boolean values) {
            return values ? val[layer] : key[layer];
        }

        @Override
        void expand(int layer, boolean values, int s, int ko, float[] dst, int d) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window, float scale,
                       float softcap) {
            if (n > 1) {
                attentionTiled(pool, layer, n, pos, q, att, window, scale, softcap);
                return;
            }
            // decode: one task per query head, the finest split there is for a single token
            float[] kc = key[layer];
            float[] vc = val[layer];
            int hd = headDim;
            int nh = kvHeads * group;
            int len = pos + 1;
            int first = window > 0 ? Math.max(0, len - window) : 0;
            pool.forRangeFine(nh, 1, (h0, h1) -> {
                float[] sc = scratch.get();
                for (int h = h0; h < h1; h++) {
                    int ko = (h / group) * hd;
                    int qo = h * hd;
                    for (int s = first; s < len; s++) {
                        float d = VectorKernels.dot(q, qo, kc, s * kvDim + ko, hd) * scale;
                        sc[s] = softcap > 0 && !FASTMATH ? (float) Math.tanh(d / softcap) * softcap : d;
                    }
                    float inv = 1f / softmax(sc, first, len - first, softcap);
                    Arrays.fill(att, qo, qo + hd, 0f);
                    for (int s = first; s < len; s++) {
                        VectorKernels.axpy(sc[s] * inv, vc, s * kvDim + ko, att, qo, hd);
                    }
                }
            });
        }
    }

    // ------------------------------------------------------------------ f16

    static final class F16 extends KvCache {
        final short[][] key;
        final short[][] val;

        F16(int layers, Attention.Gqa a, int ctx) {
            super(layers, a, ctx);
            key = new short[layers][ctx * kvDim];
            val = new short[layers][ctx * kvDim];
        }

        @Override
        public void store(int layer, int pos, float[] k, float[] v, int n) {
            toHalf(k, 0, key[layer], pos * kvDim, n * kvDim);
            toHalf(v, 0, val[layer], pos * kvDim, n * kvDim);
        }

        @Override
        float[] rowsInPlace(int layer, boolean values) {
            return null;
        }

        @Override
        void expand(int layer, boolean values, int s, int ko, float[] dst, int d) {
            toFloat(values ? val[layer] : key[layer], s * kvDim + ko, dst, d, headDim);
        }

        @Override
        public void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window, float scale,
                       float softcap) {
            attentionTiled(pool, layer, n, pos, q, att, window, scale, softcap);
        }

        static void toHalf(float[] src, int so, short[] dst, int d, int n) {
            for (int i = 0; i < n; i++) {
                dst[d + i] = Float.floatToFloat16(src[so + i]);
            }
        }

        static void toFloat(short[] src, int so, float[] dst, int d, int n) {
            for (int i = 0; i < n; i++) {
                dst[d + i] = Float.float16ToFloat(src[so + i]);
            }
        }
    }
}
