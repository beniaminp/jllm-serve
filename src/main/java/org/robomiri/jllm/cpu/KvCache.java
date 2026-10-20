package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.util.Arrays;

/**
 * The CPU backend's KV cache, on the Java heap, and the causal grouped-query attention over it. Per layer and kv
 * head, K is stored <em>transposed</em> in chunks of {@link #CHUNK} keys ({@code [hd][CHUNK]} per chunk): a vector
 * load then holds one head element of {@code FL} keys, and a score vector for {@code FL} keys is {@code headDim} FMAs
 * with broadcast query elements, with no horizontal reduction per key. V is stored as rows ({@code [ctx, kvDim]}),
 * the shape the output accumulation wants. {@link F32} holds floats that the kernels read in place. {@link F16}
 * holds IEEE halves packed two to an {@code int}, which vector code converts with shifts and masks into thread-local
 * scratch, a block of keys at a time. The math is f32 in both cases; f16 only changes what is stored between
 * forward calls.
 *
 * <p><b>Tiling.</b> The work is split into tiles of one kv head and up to {@link #MAX_TILE} consecutive tokens
 * (fewer when their score rows over the {@code pos + n} keys would exceed {@link #SCORE_BUDGET}), which serve
 * {@code tile × group} query heads. Each block of K chunks is read (or expanded) once per tile and scored against
 * four query rows at a time; the softmax runs per row and normalizes the probabilities in place; each block of V
 * rows is read once per tile and added to four rows' outputs at a time, two vectors per row in registers. Every
 * score is the same chain of FMAs over the head elements, and every output element adds the keys in ascending
 * order, whichever kernel variant computes it, so prefill (tiles of many tokens) and decode (a one-token tile, or
 * {@link #attentionDecode} when there are fewer kv heads than threads) give bit-identical results. A head size that
 * is not whole vectors uses {@code attentionTiledSimple}, one key at a time over expanded rows.
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
     * Most tokens per prefill tile: each K / V row is read (and, f16, expanded) once per tile. With the f16 expansion
     * vectorized and the score kernel no longer reduction-bound, 64 beat 32 and 16 on a 6 K-token prefill (94 s, 98 s,
     * 101 s before the kernel changes).
     */
    static final int MAX_TILE = 64;
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

    /** The f32 transposed K chunks of {@code layer} (see {@link #keyChunk}) when they can be read in place, else null. */
    abstract float[] keysInPlace(int layer);

    /** The f32 V rows ({@code [ctx, kvDim]}) of {@code layer} when they can be read in place, else null. */
    abstract float[] valuesInPlace(int layer);

    /** Expands the K chunks {@code [c0, c1)} of kv head {@code g} to f32, {@code [hd][CHUNK]} each, into {@code dst[d ..]}. */
    abstract void expandKeys(int layer, int g, int c0, int c1, float[] dst, int d);

    /**
     * Expands elements {@code [i0, i1)} (whole vectors) of the V rows {@code [s0, s1)} ({@code s0} even) of kv head
     * {@code g} to f32, {@code i1 - i0} per row, into {@code dst[d ..]}.
     */
    abstract void expandValues(int layer, int g, int s0, int s1, int i0, int i1, float[] dst, int d);

    /** Expands the {@code headDim} values of kv head slice {@code ko} of row {@code s} into {@code dst[d ..]}. */
    abstract void expand(int layer, boolean values, int s, int ko, float[] dst, int d);

    /** Chunks per kv head: the context rounded up to whole chunks. */
    final int chunks() {
        return (ctx + CHUNK - 1) / CHUNK;
    }

    /** Offset of chunk {@code c} of kv head {@code g} in an f32 transposed K array: {@code [kvHeads][chunks][hd][CHUNK]}. */
    final int keyChunk(int g, int c) {
        return (g * chunks() + c) * headDim * CHUNK;
    }

    /** Layout of {@link #write}: the element type and the shape, so a file is only read into an equal cache. */
    public final String format() {
        return (this instanceof F16 ? "f16" : "f32") + " layers=" + layers + " kvDim=" + kvDim + " headDim=" + headDim;
    }

    /** Positions {@code [0, n)} of every layer copied from {@code o}, a cache of the same {@link #format}. */
    public final void copyFrom(KvCache o, int n) {
        if (!format().equals(o.format()) || o.ctx < n || ctx < n) {
            throw new IllegalArgumentException("cannot copy a " + o.format() + " cache into a " + format() + " one");
        }
        int chunksUsed = (n + CHUNK - 1) / CHUNK;
        for (int layer = 0; layer < layers; layer++) {
            copyLayer(o, layer, n, chunksUsed);
        }
    }

    abstract void copyLayer(KvCache o, int layer, int n, int chunksUsed);

    /** Rows per buffer of {@link #write} / {@link #read}. */
    private static final int IO_ROWS = 256;

    /**
     * Writes positions {@code [0, n)} of every layer: per layer, per position, the K row then the V row in natural
     * element order, little-endian, as f32 or f16 (the stored type, so nothing is rounded).
     */
    public final void write(int n, WritableByteChannel out) throws IOException {
        boolean half = this instanceof F16;
        int el = half ? 2 : 4;
        float[] row = new float[kvDim];
        ByteBuffer buf = ByteBuffer.allocate(IO_ROWS * 2 * kvDim * el).order(ByteOrder.LITTLE_ENDIAN);
        for (int layer = 0; layer < layers; layer++) {
            for (int s0 = 0; s0 < n; s0 += IO_ROWS) {
                buf.clear();
                for (int s = s0; s < Math.min(n, s0 + IO_ROWS); s++) {
                    for (int kind = 0; kind < 2; kind++) {
                        for (int g = 0; g < kvHeads; g++) {
                            expand(layer, kind == 1, s, g * headDim, row, g * headDim);
                        }
                        for (int i = 0; i < kvDim; i++) {
                            if (half) {
                                buf.putShort(Float.floatToFloat16(row[i]));
                            } else {
                                buf.putFloat(row[i]);
                            }
                        }
                    }
                }
                buf.flip();
                while (buf.hasRemaining()) {
                    out.write(buf);
                }
            }
        }
    }

    /** Reads what {@link #write} wrote for {@code n} positions into positions {@code [0, n)}. */
    public final void read(int n, ReadableByteChannel in) throws IOException {
        if (n > ctx) {
            throw new IOException("the saved cache holds " + n + " positions, the context is " + ctx);
        }
        boolean half = this instanceof F16;
        int el = half ? 2 : 4;
        float[] k = new float[IO_ROWS * kvDim];
        float[] v = new float[IO_ROWS * kvDim];
        ByteBuffer buf = ByteBuffer.allocate(IO_ROWS * 2 * kvDim * el).order(ByteOrder.LITTLE_ENDIAN);
        for (int layer = 0; layer < layers; layer++) {
            for (int s0 = 0; s0 < n; s0 += IO_ROWS) {
                int m = Math.min(n - s0, IO_ROWS);
                buf.clear().limit(m * 2 * kvDim * el);
                while (buf.hasRemaining()) {
                    if (in.read(buf) < 0) {
                        throw new EOFException("the saved cache file is truncated");
                    }
                }
                buf.flip();
                for (int r = 0; r < m; r++) {
                    for (int kind = 0; kind < 2; kind++) {
                        float[] dst = kind == 0 ? k : v;
                        for (int i = 0; i < kvDim; i++) {
                            dst[r * kvDim + i] = half ? Float.float16ToFloat(buf.getShort()) : buf.getFloat();
                        }
                    }
                }
                store(layer, s0, k, v, m);
            }
        }
    }

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

    /** Keys per block of the blocked kernel (a multiple of {@link #CHUNK}): their K or V rows (f32) take 64 KB at headDim 128. */
    static final int KEY_BLOCK = 128;

    /** Keys per chunk of the transposed K layout: one chunk holds {@code [headDim][CHUNK]} values. */
    static final int CHUNK = 32;

    /** Whether the vector kernels apply: whole vectors per head and per chunk. */
    final boolean vectorized() {
        int fl = VectorKernels.FL;
        return headDim % fl == 0 && CHUNK % (2 * fl) == 0;
    }

    /**
     * The tiled kernel described in the class comment, for every (kv head, token tile) pair of the chunk. The keys
     * are walked in blocks of {@link #KEY_BLOCK}. Scores come from the transposed K chunks: one vector holds the
     * scores of {@code FL} keys and is built by {@code headDim} FMAs with broadcast query elements, four query rows
     * at a time sharing each key load, with no horizontal reduction. Outputs accumulate in registers, four rows at a
     * time, over V rows in ascending key order. An f16 cache first expands a block's K chunks (then V rows) to f32
     * scratch with vector code. Decode is the one-token tile, so prefill and decode compute every score and output
     * with the same operations in the same order.
     */
    final void attentionTiled(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window,
                              float scale, float softcap) {
        if (!vectorized()) {
            attentionTiledSimple(pool, layer, n, pos, q, att, window, scale, softcap);
            return;
        }
        int hd = headDim;
        // tiles are sized by the keys this chunk sees, not by the context: at a 32 K context a fixed budget left
        // one token per tile, so every K / V row was expanded and read once per token instead of once per tile
        int span = roundUp(pos + n, CHUNK); // score row stride: whole chunks
        // ... but at least two tasks per thread, so that wide machines with few kv heads stay busy
        int minTiles = (2 * pool.threads() + kvHeads - 1) / kvHeads;
        int tile = Math.max(1, Math.min(Math.min(MAX_TILE, (n + minTiles - 1) / minTiles),
                Math.min(n, SCORE_BUDGET / (group * span))));
        int tiles = (n + tile - 1) / tile;
        int blkOff = tile * group * span;              // one block of K chunks or V rows, f32
        int scratchLen = blkOff + KEY_BLOCK * hd;
        float[] kIn = keysInPlace(layer);
        float[] vIn = valuesInPlace(layer);
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
                int sMin = lo[0] / CHUNK * CHUNK;
                int sMax = hi[rows - 1];

                // scores: sc[r * span + s] for the chunks holding visible keys of every row (others are ignored)
                for (int b0 = sMin; b0 < sMax; b0 += KEY_BLOCK) {
                    int b1 = Math.min(sMax, b0 + KEY_BLOCK);
                    float[] kArr;
                    int kBase;
                    if (kIn != null) {
                        kArr = kIn;
                        kBase = keyChunk(g, b0 / CHUNK);
                    } else {
                        expandKeys(layer, g, b0 / CHUNK, (b1 + CHUNK - 1) / CHUNK, sc, blkOff);
                        kArr = sc;
                        kBase = blkOff;
                    }
                    for (int c0 = b0; c0 < b1; c0 += CHUNK) {
                        int cb = kBase + (c0 - b0) * hd;
                        int c1 = c0 + CHUNK;
                        int r = 0;
                        while (r < rows) {
                            if (lo[r] >= c1 || hi[r] <= c0) {
                                r++;
                            } else if (r + 3 < rows && lo[r + 3] < c1 && hi[r + 3] > c0 && lo[r + 1] < c1 && hi[r + 1] > c0
                                    && lo[r + 2] < c1 && hi[r + 2] > c0) {
                                score4(kArr, cb, q, rq[r], rq[r + 1], rq[r + 2], rq[r + 3], hd, sc, r * span + c0, span, scale);
                                r += 4;
                            } else if (r + 1 < rows && lo[r + 1] < c1 && hi[r + 1] > c0) {
                                score2(kArr, cb, q, rq[r], rq[r + 1], hd, sc, r * span + c0, span, scale);
                                r += 2;
                            } else {
                                score1(kArr, cb, q, rq[r], hd, sc, r * span + c0, scale);
                                r++;
                            }
                        }
                    }
                }
                for (int r = 0; r < rows; r++) {
                    int o = r * span;
                    if (capped) {
                        for (int s = lo[r]; s < hi[r]; s++) {
                            sc[o + s] = (float) Math.tanh(sc[o + s] / softcap) * softcap;
                        }
                    }
                    float inv = 1f / softmax(sc, o + lo[r], hi[r] - lo[r], softcap);
                    VectorKernels.scale(sc, o + lo[r], hi[r] - lo[r], inv); // probabilities, used by every output vector
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
                        expandValues(layer, g, b0, b1, 0, hd, sc, blkOff);
                        vBase = blkOff;
                        vStride = hd;
                    }
                    int r = 0;
                    for (; r + 3 < rows; r += 4) {
                        accumulate4(sc, span, r, lo, hi, b0, b1, vArr, vBase, vStride, att, rq, hd);
                    }
                    for (; r < rows; r++) {
                        accumulate1(sc, span, r, Math.max(lo[r], b0), Math.min(hi[r], b1), b0, vArr, vBase, vStride,
                                att, rq[r], hd);
                    }
                }
            }
        });
    }

    /** Decode scores of every query head over the context, shared by the tasks of one {@link #attentionDecode}. */
    private float[] decodeScores = new float[0];

    /**
     * Decode ({@code n == 1}) in three parallel phases, so that a model with few kv heads still keeps every thread
     * busy: the scores by (kv head, range of key chunks), the softmax by query head, the outputs by (kv head, slice of
     * the head's elements). Each score and each output element sees the same operations in the same order as in
     * {@link #attentionTiled}, so decode and prefill still agree bit for bit.
     */
    final void attentionDecode(ThreadPool pool, int layer, int pos, float[] q, float[] att, int window, float scale,
                               float softcap) {
        int hd = headDim;
        int fl = VectorKernels.FL;
        int len = pos + 1;
        int first = window > 0 ? Math.max(0, len - window) : 0;
        int span = roundUp(len, CHUNK);
        int heads = kvHeads * group;
        if (decodeScores.length < heads * span) {
            decodeScores = new float[Math.max(heads * span, Math.min(heads * roundUp(ctx, CHUNK), 2 * heads * span))];
        }
        float[] sc = decodeScores;
        float[] kIn = keysInPlace(layer);
        float[] vIn = valuesInPlace(layer);
        int c0 = first / CHUNK;
        int c1 = span / CHUNK;
        int want = 2 * pool.threads();
        int keyParts = Math.max(1, Math.min((want + kvHeads - 1) / kvHeads, (c1 - c0) / 4)); // >= 4 chunks per task
        pool.forRangeFine(kvHeads * keyParts, 1, (i0, i1) -> {
            float[] buf = scratch(KEY_BLOCK * hd);
            for (int i = i0; i < i1; i++) {
                int g = i / keyParts;
                int part = i % keyParts;
                int pa = c0 + (int) ((long) (c1 - c0) * part / keyParts);
                int pb = c0 + (int) ((long) (c1 - c0) * (part + 1) / keyParts);
                for (int b0 = pa; b0 < pb; b0 += KEY_BLOCK / CHUNK) {
                    int b1 = Math.min(pb, b0 + KEY_BLOCK / CHUNK);
                    float[] kArr = kIn != null ? kIn : buf;
                    int kBase = 0;
                    if (kIn != null) {
                        kBase = keyChunk(g, b0);
                    } else {
                        expandKeys(layer, g, b0, b1, buf, 0);
                    }
                    for (int c = b0; c < b1; c++) {
                        int cb = kBase + (c - b0) * hd * CHUNK;
                        int r = 0;
                        int row = g * group;
                        for (; r + 3 < group; r += 4) {
                            score4(kArr, cb, q, (row + r) * hd, (row + r + 1) * hd, (row + r + 2) * hd, (row + r + 3) * hd,
                                    hd, sc, (row + r) * span + c * CHUNK, span, scale);
                        }
                        for (; r + 1 < group; r += 2) {
                            score2(kArr, cb, q, (row + r) * hd, (row + r + 1) * hd, hd, sc, (row + r) * span + c * CHUNK,
                                    span, scale);
                        }
                        for (; r < group; r++) {
                            score1(kArr, cb, q, (row + r) * hd, hd, sc, (row + r) * span + c * CHUNK, scale);
                        }
                    }
                }
            }
        });
        boolean capped = softcap > 0 && !FASTMATH;
        pool.forRangeFine(heads, 1, (h0, h1) -> {
            for (int h = h0; h < h1; h++) {
                int o = h * span;
                if (capped) {
                    for (int s = first; s < len; s++) {
                        sc[o + s] = (float) Math.tanh(sc[o + s] / softcap) * softcap;
                    }
                }
                float inv = 1f / softmax(sc, o + first, len - first, softcap);
                VectorKernels.scale(sc, o + first, len - first, inv);
                Arrays.fill(att, h * hd, (h + 1) * hd, 0f);
            }
        });
        // output slices of whole vectors, at least two per task so accumulate1 can keep several chains
        int slice = hd % (4 * fl) == 0 ? 4 * fl : hd % (2 * fl) == 0 ? 2 * fl : fl;
        int slices = hd / slice;
        int sMin = first / CHUNK * CHUNK;
        pool.forRangeFine(kvHeads * slices, 1, (i0, i1) -> {
            float[] buf = scratch(KEY_BLOCK * slice);
            for (int i = i0; i < i1; i++) {
                int g = i / slices;
                int e0 = (i % slices) * slice;
                for (int b0 = sMin; b0 < len; b0 += KEY_BLOCK) {
                    int b1 = Math.min(len, b0 + KEY_BLOCK);
                    float[] vArr;
                    int vBase;
                    int vStride;
                    if (vIn != null) {
                        vArr = vIn;
                        vBase = b0 * kvDim + g * hd + e0;
                        vStride = kvDim;
                    } else {
                        expandValues(layer, g, b0, b1, e0, e0 + slice, buf, 0);
                        vArr = buf;
                        vBase = 0;
                        vStride = slice;
                    }
                    for (int r = 0; r < group; r++) {
                        int h = g * group + r;
                        accumulate1(sc, span, h, Math.max(first, b0), b1, b0, vArr, vBase, vStride, att, h * hd + e0,
                                slice);
                    }
                }
            }
        });
    }

    static int roundUp(int x, int m) {
        return (x + m - 1) / m * m;
    }

    /**
     * Scores of four query rows against the {@link #CHUNK} keys of the transposed chunk at {@code k[kb ..]}
     * ({@code [hd][CHUNK]}): two vectors of keys at a time, one FMA per head element with the broadcast query element
     * (eight accumulators: each key and query load feeds several FMAs). Written scaled to {@code sc[so ..]},
     * {@code sc[so + stride ..]}, ...
     */
    private static void score4(float[] k, int kb, float[] q, int q0, int q1, int q2, int q3, int hd, float[] sc, int so,
                               int stride, float scale) {
        var sp = VectorKernels.F;
        int fl = VectorKernels.FL;
        for (int v = 0; v < CHUNK; v += 2 * fl) {
            FloatVector a0 = FloatVector.zero(sp);
            FloatVector a1 = a0;
            FloatVector a2 = a0;
            FloatVector a3 = a0;
            FloatVector b0 = a0;
            FloatVector b1 = a0;
            FloatVector b2 = a0;
            FloatVector b3 = a0;
            for (int i = 0; i < hd; i++) {
                FloatVector ka = FloatVector.fromArray(sp, k, kb + i * CHUNK + v);
                FloatVector kz = FloatVector.fromArray(sp, k, kb + i * CHUNK + v + fl);
                FloatVector x0 = FloatVector.broadcast(sp, q[q0 + i]);
                FloatVector x1 = FloatVector.broadcast(sp, q[q1 + i]);
                FloatVector x2 = FloatVector.broadcast(sp, q[q2 + i]);
                FloatVector x3 = FloatVector.broadcast(sp, q[q3 + i]);
                a0 = ka.fma(x0, a0);
                b0 = kz.fma(x0, b0);
                a1 = ka.fma(x1, a1);
                b1 = kz.fma(x1, b1);
                a2 = ka.fma(x2, a2);
                b2 = kz.fma(x2, b2);
                a3 = ka.fma(x3, a3);
                b3 = kz.fma(x3, b3);
            }
            a0.mul(scale).intoArray(sc, so + v);
            b0.mul(scale).intoArray(sc, so + v + fl);
            a1.mul(scale).intoArray(sc, so + stride + v);
            b1.mul(scale).intoArray(sc, so + stride + v + fl);
            a2.mul(scale).intoArray(sc, so + 2 * stride + v);
            b2.mul(scale).intoArray(sc, so + 2 * stride + v + fl);
            a3.mul(scale).intoArray(sc, so + 3 * stride + v);
            b3.mul(scale).intoArray(sc, so + 3 * stride + v + fl);
        }
    }

    /** {@link #score4} for two rows, with the same operations per score. */
    private static void score2(float[] k, int kb, float[] q, int q0, int q1, int hd, float[] sc, int so, int stride,
                               float scale) {
        var sp = VectorKernels.F;
        int fl = VectorKernels.FL;
        for (int v = 0; v < CHUNK; v += 2 * fl) {
            FloatVector a0 = FloatVector.zero(sp);
            FloatVector a1 = a0;
            FloatVector b0 = a0;
            FloatVector b1 = a0;
            for (int i = 0; i < hd; i++) {
                FloatVector ka = FloatVector.fromArray(sp, k, kb + i * CHUNK + v);
                FloatVector kz = FloatVector.fromArray(sp, k, kb + i * CHUNK + v + fl);
                FloatVector x0 = FloatVector.broadcast(sp, q[q0 + i]);
                FloatVector x1 = FloatVector.broadcast(sp, q[q1 + i]);
                a0 = ka.fma(x0, a0);
                b0 = kz.fma(x0, b0);
                a1 = ka.fma(x1, a1);
                b1 = kz.fma(x1, b1);
            }
            a0.mul(scale).intoArray(sc, so + v);
            b0.mul(scale).intoArray(sc, so + v + fl);
            a1.mul(scale).intoArray(sc, so + stride + v);
            b1.mul(scale).intoArray(sc, so + stride + v + fl);
        }
    }

    /** {@link #score4} for one row, with the same operations per score. */
    private static void score1(float[] k, int kb, float[] q, int q0, int hd, float[] sc, int so, float scale) {
        var sp = VectorKernels.F;
        int fl = VectorKernels.FL;
        for (int v = 0; v < CHUNK; v += 2 * fl) {
            FloatVector a0 = FloatVector.zero(sp);
            FloatVector b0 = a0;
            for (int i = 0; i < hd; i++) {
                FloatVector x0 = FloatVector.broadcast(sp, q[q0 + i]);
                a0 = FloatVector.fromArray(sp, k, kb + i * CHUNK + v).fma(x0, a0);
                b0 = FloatVector.fromArray(sp, k, kb + i * CHUNK + v + fl).fma(x0, b0);
            }
            a0.mul(scale).intoArray(sc, so + v);
            b0.mul(scale).intoArray(sc, so + v + fl);
        }
    }

    /**
     * Rows {@code r .. r+3}: their keys of block {@code [b0, b1)} added to their outputs ({@code sc} holds the
     * normalized probabilities). The keys all four see run with two output vectors per row in registers (eight
     * accumulators); the others row by row. Each output element sees the same FMAs in ascending key order either way.
     */
    private void accumulate4(float[] sc, int span, int r, int[] lo, int[] hi, int b0, int b1, float[] v,
                             int vBase, int vStride, float[] att, int[] rq, int hd) {
        int a = Math.max(b0, Math.max(Math.max(lo[r], lo[r + 1]), Math.max(lo[r + 2], lo[r + 3])));
        int b = Math.min(b1, Math.min(Math.min(hi[r], hi[r + 1]), Math.min(hi[r + 2], hi[r + 3])));
        if (a >= b) { // nothing in common in this block (rare: near the diagonal or a window edge)
            for (int k = 0; k < 4; k++) {
                accumulate1(sc, span, r + k, Math.max(lo[r + k], b0), Math.min(hi[r + k], b1), b0, v, vBase,
                        vStride, att, rq[r + k], hd);
            }
            return;
        }
        for (int k = 0; k < 4; k++) { // keys before the common range
            accumulate1(sc, span, r + k, Math.max(lo[r + k], b0), a, b0, v, vBase, vStride, att, rq[r + k], hd);
        }
        var sp = VectorKernels.F;
        int fl = VectorKernels.FL;
        int p0 = r * span;
        int p1 = p0 + span;
        int p2 = p1 + span;
        int p3 = p2 + span;
        int c = 0;
        for (; c + 2 * fl <= hd; c += 2 * fl) {
            var x0 = FloatVector.fromArray(sp, att, rq[r] + c);
            var y0 = FloatVector.fromArray(sp, att, rq[r] + c + fl);
            var x1 = FloatVector.fromArray(sp, att, rq[r + 1] + c);
            var y1 = FloatVector.fromArray(sp, att, rq[r + 1] + c + fl);
            var x2 = FloatVector.fromArray(sp, att, rq[r + 2] + c);
            var y2 = FloatVector.fromArray(sp, att, rq[r + 2] + c + fl);
            var x3 = FloatVector.fromArray(sp, att, rq[r + 3] + c);
            var y3 = FloatVector.fromArray(sp, att, rq[r + 3] + c + fl);
            for (int s = a; s < b; s++) {
                int vo = vBase + (s - b0) * vStride + c;
                var va = FloatVector.fromArray(sp, v, vo);
                var vz = FloatVector.fromArray(sp, v, vo + fl);
                var w0 = FloatVector.broadcast(sp, sc[p0 + s]);
                var w1 = FloatVector.broadcast(sp, sc[p1 + s]);
                var w2 = FloatVector.broadcast(sp, sc[p2 + s]);
                var w3 = FloatVector.broadcast(sp, sc[p3 + s]);
                x0 = va.fma(w0, x0);
                y0 = vz.fma(w0, y0);
                x1 = va.fma(w1, x1);
                y1 = vz.fma(w1, y1);
                x2 = va.fma(w2, x2);
                y2 = vz.fma(w2, y2);
                x3 = va.fma(w3, x3);
                y3 = vz.fma(w3, y3);
            }
            x0.intoArray(att, rq[r] + c);
            y0.intoArray(att, rq[r] + c + fl);
            x1.intoArray(att, rq[r + 1] + c);
            y1.intoArray(att, rq[r + 1] + c + fl);
            x2.intoArray(att, rq[r + 2] + c);
            y2.intoArray(att, rq[r + 2] + c + fl);
            x3.intoArray(att, rq[r + 3] + c);
            y3.intoArray(att, rq[r + 3] + c + fl);
        }
        for (; c < hd; c += fl) { // an odd number of vectors per head
            var x0 = FloatVector.fromArray(sp, att, rq[r] + c);
            var x1 = FloatVector.fromArray(sp, att, rq[r + 1] + c);
            var x2 = FloatVector.fromArray(sp, att, rq[r + 2] + c);
            var x3 = FloatVector.fromArray(sp, att, rq[r + 3] + c);
            for (int s = a; s < b; s++) {
                var vv = FloatVector.fromArray(sp, v, vBase + (s - b0) * vStride + c);
                x0 = vv.fma(FloatVector.broadcast(sp, sc[p0 + s]), x0);
                x1 = vv.fma(FloatVector.broadcast(sp, sc[p1 + s]), x1);
                x2 = vv.fma(FloatVector.broadcast(sp, sc[p2 + s]), x2);
                x3 = vv.fma(FloatVector.broadcast(sp, sc[p3 + s]), x3);
            }
            x0.intoArray(att, rq[r] + c);
            x1.intoArray(att, rq[r + 1] + c);
            x2.intoArray(att, rq[r + 2] + c);
            x3.intoArray(att, rq[r + 3] + c);
        }
        for (int k = 0; k < 4; k++) { // keys after it
            accumulate1(sc, span, r + k, b, Math.min(hi[r + k], b1), b0, v, vBase, vStride, att, rq[r + k], hd);
        }
    }

    /**
     * Row {@code r}: keys {@code [from, to)} (inside the block starting at {@code b0}) added to its output, up to four
     * output vectors at a time (independent chains; each element still adds the keys in ascending order).
     */
    private static void accumulate1(float[] sc, int span, int r, int from, int to, int b0, float[] v,
                                    int vBase, int vStride, float[] att, int qo, int hd) {
        if (from >= to) {
            return;
        }
        var sp = VectorKernels.F;
        int fl = VectorKernels.FL;
        int p = r * span;
        int c = 0;
        for (; c + 4 * fl <= hd; c += 4 * fl) {
            var x0 = FloatVector.fromArray(sp, att, qo + c);
            var x1 = FloatVector.fromArray(sp, att, qo + c + fl);
            var x2 = FloatVector.fromArray(sp, att, qo + c + 2 * fl);
            var x3 = FloatVector.fromArray(sp, att, qo + c + 3 * fl);
            for (int s = from; s < to; s++) {
                int vo = vBase + (s - b0) * vStride + c;
                var w = FloatVector.broadcast(sp, sc[p + s]);
                x0 = FloatVector.fromArray(sp, v, vo).fma(w, x0);
                x1 = FloatVector.fromArray(sp, v, vo + fl).fma(w, x1);
                x2 = FloatVector.fromArray(sp, v, vo + 2 * fl).fma(w, x2);
                x3 = FloatVector.fromArray(sp, v, vo + 3 * fl).fma(w, x3);
            }
            x0.intoArray(att, qo + c);
            x1.intoArray(att, qo + c + fl);
            x2.intoArray(att, qo + c + 2 * fl);
            x3.intoArray(att, qo + c + 3 * fl);
        }
        for (; c < hd; c += fl) {
            var x = FloatVector.fromArray(sp, att, qo + c);
            for (int s = from; s < to; s++) {
                x = FloatVector.fromArray(sp, v, vBase + (s - b0) * vStride + c)
                        .fma(FloatVector.broadcast(sp, sc[p + s]), x);
            }
            x.intoArray(att, qo + c);
        }
    }

    /** The tile loop one key at a time over expanded rows: used when {@code headDim} is not a multiple of the vector length. */
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
        // later tiles see more keys: hand them out first so the tail of the job is made of the light ones
        pool.forRangeFine(kvHeads * tiles, 1, (i0, i1) -> {
            float[] sc = scratch(scratchLen);
            for (int i = i0; i < i1; i++) {
                int g = i % kvHeads;
                int t0 = (tiles - 1 - i / kvHeads) * tile;
                int t1 = Math.min(n, t0 + tile);
                int ko = g * hd;
                int sMin = window > 0 ? Math.max(0, pos + t0 + 1 - window) : 0;
                int sMax = pos + t1;
                // scores: row (t - t0) * group + h holds the keys of query head g * group + h of token t
                for (int s = sMin; s < sMax; s++) {
                    expand(layer, false, s, ko, sc, kOff);
                    for (int t = t0; t < t1; t++) {
                        int len = pos + t + 1;
                        int first = window > 0 ? Math.max(0, len - window) : 0;
                        if (s < first || s >= len) {
                            continue;
                        }
                        int qo = t * qDim + g * group * hd;
                        int row = (t - t0) * group;
                        for (int h = 0; h < group; h++) {
                            float d = VectorKernels.dot(q, qo + h * hd, sc, kOff, hd) * scale;
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
                    expand(layer, true, s, ko, sc, vOff);
                    for (int t = t0; t < t1; t++) {
                        int len = pos + t + 1;
                        int first = window > 0 ? Math.max(0, len - window) : 0;
                        if (s < first || s >= len) {
                            continue;
                        }
                        int qo = t * qDim + g * group * hd;
                        int row = (t - t0) * group;
                        for (int h = 0; h < group; h++) {
                            VectorKernels.axpy(sc[(row + h) * span + s] * sc[invOff + row + h], sc, vOff, att, qo + h * hd, hd);
                        }
                    }
                }
            }
        });
    }

    // ------------------------------------------------------------------ f32

    /**
     * f32: K in transposed chunks, read in place by the score kernel; V rows {@code [ctx, kvDim]} read in place by the
     * output kernel.
     */
    static final class F32 extends KvCache {
        final float[][] key;
        final float[][] val;

        F32(int layers, Attention.Gqa a, int ctx) {
            super(layers, a, ctx);
            key = new float[layers][kvHeads * chunks() * headDim * CHUNK];
            val = new float[layers][ctx * kvDim];
        }

        @Override
        public void store(int layer, int pos, float[] k, float[] v, int n) {
            float[] kc = key[layer];
            for (int t = 0; t < n; t++) {
                int s = pos + t;
                for (int g = 0; g < kvHeads; g++) {
                    int base = keyChunk(g, s / CHUNK) + s % CHUNK;
                    int src = t * kvDim + g * headDim;
                    for (int i = 0; i < headDim; i++) {
                        kc[base + i * CHUNK] = k[src + i];
                    }
                }
            }
            System.arraycopy(v, 0, val[layer], pos * kvDim, n * kvDim);
        }

        @Override
        void copyLayer(KvCache o, int layer, int n, int chunksUsed) {
            F32 src = (F32) o;
            for (int g = 0; g < kvHeads; g++) {
                System.arraycopy(src.key[layer], src.keyChunk(g, 0), key[layer], keyChunk(g, 0), chunksUsed * headDim * CHUNK);
            }
            System.arraycopy(src.val[layer], 0, val[layer], 0, n * kvDim);
        }

        @Override
        float[] keysInPlace(int layer) {
            return key[layer];
        }

        @Override
        float[] valuesInPlace(int layer) {
            return val[layer];
        }

        @Override
        void expandKeys(int layer, int g, int c0, int c1, float[] dst, int d) {
            System.arraycopy(key[layer], keyChunk(g, c0), dst, d, (c1 - c0) * headDim * CHUNK);
        }

        @Override
        void expandValues(int layer, int g, int s0, int s1, int i0, int i1, float[] dst, int d) {
            int w = i1 - i0;
            for (int s = s0; s < s1; s++) {
                System.arraycopy(val[layer], s * kvDim + g * headDim + i0, dst, d + (s - s0) * w, w);
            }
        }

        @Override
        void expand(int layer, boolean values, int s, int ko, float[] dst, int d) {
            if (values) {
                System.arraycopy(val[layer], s * kvDim + ko, dst, d, headDim);
                return;
            }
            int base = keyChunk(ko / headDim, s / CHUNK) + s % CHUNK;
            for (int i = 0; i < headDim; i++) {
                dst[d + i] = key[layer][base + i * CHUNK];
            }
        }

        @Override
        public void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window, float scale,
                       float softcap) {
            // with as many kv heads as threads the tiled kernel is already balanced and keeps a head's work on one core
            if (n == 1 && vectorized() && kvHeads < pool.threads()) {
                attentionDecode(pool, layer, pos, q, att, window, scale, softcap);
            } else {
                attentionTiled(pool, layer, n, pos, q, att, window, scale, softcap);
            }
        }
    }

    // ------------------------------------------------------------------ f16

    /**
     * f16: IEEE halves packed two to an {@code int}, so that a vector of int lanes converts {@code 2 × FL} of them with
     * shifts and masks ({@link #lowHalf}, {@link #highHalf}; the Vector API has no half-float conversion). K chunks
     * hold head elements {@code 2p} (low) and {@code 2p+1} (high) of each key, {@code [kvHeads][chunks][hd/2][CHUNK]};
     * V holds keys {@code 2j} (low) and {@code 2j+1} (high) of each element, {@code [ctx/2][kvDim]}.
     */
    static final class F16 extends KvCache {
        private static final VectorSpecies<Integer> I = IntVector.SPECIES_PREFERRED;
        final int[][] key;
        final int[][] val;
        /** Element pairs per head (an odd {@code headDim} leaves the last high half zero). */
        private final int pairs;

        F16(int layers, Attention.Gqa a, int ctx) {
            super(layers, a, ctx);
            pairs = (headDim + 1) / 2;
            key = new int[layers][kvHeads * chunks() * pairs * CHUNK];
            val = new int[layers][(ctx + 1) / 2 * kvDim];
        }

        private int keyPairChunk(int g, int c) {
            return (g * chunks() + c) * pairs * CHUNK;
        }

        @Override
        public void store(int layer, int pos, float[] k, float[] v, int n) {
            int[] kc = key[layer];
            int[] vc = val[layer];
            for (int t = 0; t < n; t++) {
                int s = pos + t;
                for (int g = 0; g < kvHeads; g++) {
                    int base = keyPairChunk(g, s / CHUNK) + s % CHUNK;
                    int src = t * kvDim + g * headDim;
                    for (int i = 0; i < headDim; i++) {
                        put(kc, base + (i >> 1) * CHUNK, (i & 1) != 0, Float.floatToFloat16(k[src + i]));
                    }
                }
                int vb = (s >> 1) * kvDim;
                for (int i = 0; i < kvDim; i++) {
                    put(vc, vb + i, (s & 1) != 0, Float.floatToFloat16(v[t * kvDim + i]));
                }
            }
        }

        private static void put(int[] a, int i, boolean high, short h) {
            a[i] = high ? (a[i] & 0xFFFF) | (h << 16) : (a[i] & 0xFFFF0000) | (h & 0xFFFF);
        }

        private static short get(int[] a, int i, boolean high) {
            return (short) (high ? a[i] >>> 16 : a[i]);
        }

        @Override
        void copyLayer(KvCache o, int layer, int n, int chunksUsed) {
            F16 src = (F16) o;
            for (int g = 0; g < kvHeads; g++) {
                System.arraycopy(src.key[layer], src.keyPairChunk(g, 0), key[layer], keyPairChunk(g, 0), chunksUsed * pairs * CHUNK);
            }
            System.arraycopy(src.val[layer], 0, val[layer], 0, (n + 1) / 2 * kvDim);
        }

        @Override
        float[] keysInPlace(int layer) {
            return null;
        }

        @Override
        float[] valuesInPlace(int layer) {
            return null;
        }

        @Override
        void expandKeys(int layer, int g, int c0, int c1, float[] dst, int d) {
            int[] kc = key[layer];
            int fl = VectorKernels.FL;
            int hd = headDim;
            for (int c = c0; c < c1; c++) {
                int src = keyPairChunk(g, c);
                int o = d + (c - c0) * hd * CHUNK;
                for (int p = 0; p < pairs; p++) {
                    for (int v = 0; v < CHUNK; v += fl) {
                        IntVector x = IntVector.fromArray(I, kc, src + p * CHUNK + v);
                        lowHalf(x).intoArray(dst, o + 2 * p * CHUNK + v);
                        if (2 * p + 1 < hd) {
                            highHalf(x).intoArray(dst, o + (2 * p + 1) * CHUNK + v);
                        }
                    }
                }
            }
        }

        @Override
        void expandValues(int layer, int g, int s0, int s1, int i0, int i1, float[] dst, int d) {
            int[] vc = val[layer];
            int fl = VectorKernels.FL;
            int w = i1 - i0;
            for (int s = s0; s < s1; s += 2) {
                int src = (s >> 1) * kvDim + g * headDim;
                int o = d + (s - s0) * w - i0;
                for (int i = i0; i < i1; i += fl) {
                    IntVector x = IntVector.fromArray(I, vc, src + i);
                    lowHalf(x).intoArray(dst, o + i);
                    highHalf(x).intoArray(dst, o + w + i); // key s + 1: inside the scratch block even past s1
                }
            }
        }

        @Override
        void expand(int layer, boolean values, int s, int ko, float[] dst, int d) {
            if (values) {
                int base = (s >> 1) * kvDim + ko;
                for (int i = 0; i < headDim; i++) {
                    dst[d + i] = Float.float16ToFloat(get(val[layer], base + i, (s & 1) != 0));
                }
                return;
            }
            int base = keyPairChunk(ko / headDim, s / CHUNK) + s % CHUNK;
            for (int i = 0; i < headDim; i++) {
                dst[d + i] = Float.float16ToFloat(get(key[layer], base + (i >> 1) * CHUNK, (i & 1) != 0));
            }
        }

        @Override
        public void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window, float scale,
                       float softcap) {
            // with as many kv heads as threads the tiled kernel is already balanced and keeps a head's work on one core
            if (n == 1 && vectorized() && kvHeads < pool.threads()) {
                attentionDecode(pool, layer, pos, q, att, window, scale, softcap);
            } else {
                attentionTiled(pool, layer, n, pos, q, att, window, scale, softcap);
            }
        }

        private static final int EXP_MASK = 0x0F800000;    // a half's exponent after the shift to f32 position
        private static final int SIGN = 0x80000000;
        private static final int REBIAS = 112 << 23;       // f32 bias 127 - f16 bias 15
        private static final float SUB_MAGIC = 0x1p-14f;   // 2^-14: subnormals are renormalized by a float subtraction

        /**
         * The halves in the low 16 bits of each lane, as floats: exact for zeros, subnormals and normals (the
         * exponent is rebiased by an integer add; subnormals become {@code (2^-14 + m·2^-24) − 2^-14}, an exact float
         * subtraction with no denormal operand). Infinities and NaNs, which a KV cache does not hold, come out as
         * large finite values.
         */
        static FloatVector lowHalf(IntVector x) {
            return half(x.lanewise(VectorOperators.LSHL, 13), x.lanewise(VectorOperators.LSHL, 16));
        }

        /** {@link #lowHalf} for the halves in the high 16 bits of each lane. */
        static FloatVector highHalf(IntVector x) {
            return half(x.lanewise(VectorOperators.LSHR, 3), x);
        }

        /** {@code shifted}: the half's exponent and mantissa at f32 bits 13..27; {@code signed}: its sign at bit 31. */
        private static FloatVector half(IntVector shifted, IntVector signed) {
            IntVector em = shifted.and(0x0FFFE000);
            IntVector normal = em.add(REBIAS);
            FloatVector sub = em.add(REBIAS + (1 << 23)).reinterpretAsFloats().sub(SUB_MAGIC);
            IntVector bits = normal.blend(sub.reinterpretAsInts(), em.and(EXP_MASK).eq(0));
            return bits.or(signed.and(SIGN)).reinterpretAsFloats();
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
