package org.robomiri.jllm.cpu;

import org.robomiri.jllm.kv.KvSequence;
import org.robomiri.jllm.kv.KvSummary;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.KvSparse;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * The CPU attention over a block table (docs/24 §24.4): the same tile kernels as the flat cache ({@link KvCache}), reading
 * the keys and values of position {@code p} from the physical block {@code seq.block(p / B)} of a shared pool instead
 * of from a per-slot array.
 *
 * <p>The kernels ask for K "chunks" of {@link #CHUNK} keys in the transposed layout and V rows; here a chunk is gathered
 * from one or more blocks (a block holds {@code B} keys, 16 by default: two blocks per chunk) into the kernel's float
 * scratch with vector loads straight from the off-heap segment, converting f16 and q8 on the way exactly as the flat
 * cache does. The scratch contents are therefore identical to what the flat cache produces, and so are scores and
 * outputs: the paged cache is bit for bit the flat one, for every cache type, whatever the block size.
 *
 * <p>Writes ({@link #store}) go to rows the sequence owns ({@link KvSequence#beginWrite} made them private); positions
 * below the sequence's length that are shared are never written.
 */
final class PagedKv extends KvCache implements KvCache.RowCodec {

    private static final VectorSpecies<Float> F = VectorKernels.F;
    private static final VectorSpecies<Integer> I = IntVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> B16 = VectorKernels.B16;
    private static final java.nio.ByteOrder ORDER = CpuKvStorage.ORDER;

    final CpuKvStorage st;
    final KvSequence seq;
    private final KvDtype dtype;
    private final int B;
    private final int pairs;
    /** Sparse decode attention (docs/26); {@link KvSparse#OFF} unless the pool keeps key summaries. */
    private final KvSparse sparse;

    PagedKv(int layers, Attention.Gqa a, int ctx, CpuKvStorage st, KvSequence seq) {
        super(layers, a, ctx);
        this.st = st;
        this.seq = seq;
        this.dtype = st.dtype;
        this.B = st.B;
        this.pairs = st.pairs;
        this.sparse = st.summaries ? KvSparse.get() : KvSparse.OFF;
        this.direct = dtype == KvDtype.F32 && B % (2 * VectorKernels.FL) == 0 && !Boolean.getBoolean("jllm.cpu.pagedGather");
        if (B % VectorKernels.FL != 0 || B % 2 != 0) {
            throw new IllegalArgumentException("backend.kv-block " + B + " must be a multiple of the vector length " + VectorKernels.FL);
        }
    }

    @Override
    KvDtype storedType() {
        return dtype;
    }

    @Override
    boolean rotatedKeys() {
        return st.rotate;
    }

    /** The rotated queries of the current {@link #attention} call (q8 with rotated keys). */
    private float[] qRot = new float[0];

    @Override
    public void store(int layer, int pos, float[] k, float[] v, int n) {
        for (int t = 0; t < n; t++) {
            int s = pos + t;
            st.storeRow(layer, seq.block(s / B), s % B, k, v, t * kvDim);
        }
        if (sparse.layer(layer)) { // the touched blocks' summaries, from their valid rows (exact after a rewind too)
            int end = pos + n;
            for (int j = pos / B; j * B < end; j++) {
                st.updateSummary(layer, seq.block(j), Math.min(B, end - j * B));
            }
        }
    }

    @Override
    void copyLayer(KvCache o, int layer, int n, int chunksUsed) {
        throw new UnsupportedOperationException("a paged cache shares blocks (KvSequence.shareFrom) instead of copying rows");
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
        int fl = VectorKernels.FL;
        int hd = headDim;
        int mapped = seq.blocks();
        for (int c = c0; c < c1; c++) {
            int o = d + (c - c0) * hd * CHUNK;
            int sBase = c * CHUNK;
            int end = sBase + CHUNK;
            for (int s = sBase; s < end; ) {
                int j = s / B;
                if (j >= mapped) {
                    break; // keys past the sequence: their scores are never read
                }
                int r = s % B;
                int run = Math.min(B - r, end - s);
                int blk = seq.block(j);
                MemorySegment seg = st.slab(blk);
                long kb = st.base(blk, layer);
                int off = o + (s - sBase);
                switch (dtype) {
                    case F32 -> {
                        for (int i = 0; i < hd; i++) {
                            long a = kb + (((long) g * hd + i) * B + r) * 4;
                            for (int v = 0; v < run; v += fl) {
                                FloatVector.fromMemorySegment(F, seg, a + v * 4L, ORDER).intoArray(dst, off + i * CHUNK + v);
                            }
                        }
                    }
                    case F16 -> {
                        for (int p = 0; p < pairs; p++) {
                            long a = kb + (((long) g * pairs + p) * B + r) * 4;
                            for (int v = 0; v < run; v += fl) {
                                IntVector x = IntVector.fromMemorySegment(I, seg, a + v * 4L, ORDER);
                                F16.lowHalf(x).intoArray(dst, off + 2 * p * CHUNK + v);
                                if (2 * p + 1 < hd) {
                                    F16.highHalf(x).intoArray(dst, off + (2 * p + 1) * CHUNK + v);
                                }
                            }
                        }
                    }
                    default -> {
                        for (int v = 0; v < run; v += fl) {
                            FloatVector sc = FloatVector.fromMemorySegment(F, seg, kb + st.kScaleOff + ((long) g * B + r + v) * 4, ORDER);
                            for (int i = 0; i < hd; i++) {
                                widen(seg, kb + ((long) g * hd + i) * B + r + v).mul(sc).intoArray(dst, off + i * CHUNK + v);
                            }
                        }
                    }
                }
                s += run;
            }
        }
    }

    private static FloatVector widen(MemorySegment seg, long at) {
        return (FloatVector) ByteVector.fromMemorySegment(B16, seg, at, ORDER).castShape(VectorKernels.F, 0);
    }

    @Override
    void expandValues(int layer, int g, int s0, int s1, int i0, int i1, float[] dst, int d) {
        int fl = VectorKernels.FL;
        int w = i1 - i0;
        int hd = headDim;
        switch (dtype) {
            case F32 -> {
                for (int s = s0; s < s1; s++) {
                    int blk = seq.block(s / B);
                    MemorySegment seg = st.slab(blk);
                    long a = st.base(blk, layer) + st.kBytes + ((long) (s % B) * kvDim + g * hd + i0) * 4;
                    int o = d + (s - s0) * w;
                    for (int i = 0; i < w; i += fl) {
                        FloatVector.fromMemorySegment(F, seg, a + i * 4L, ORDER).intoArray(dst, o + i);
                    }
                }
            }
            case F16 -> {
                for (int s = s0; s < s1; s += 2) {
                    int blk = seq.block(s / B);
                    MemorySegment seg = st.slab(blk);
                    long a = st.base(blk, layer) + st.kBytes + ((long) ((s % B) >> 1) * kvDim + g * hd + i0) * 4;
                    int o = d + (s - s0) * w;
                    for (int i = 0; i < w; i += fl) {
                        IntVector x = IntVector.fromMemorySegment(I, seg, a + i * 4L, ORDER);
                        F16.lowHalf(x).intoArray(dst, o + i);
                        F16.highHalf(x).intoArray(dst, o + w + i); // key s + 1: inside the scratch block even past s1
                    }
                }
            }
            default -> {
                for (int s = s0; s < s1; s++) {
                    int blk = seq.block(s / B);
                    MemorySegment seg = st.slab(blk);
                    long vb = st.base(blk, layer) + st.kBytes;
                    float sc = seg.get(ValueLayout.JAVA_FLOAT, vb + st.vScaleOff + ((long) (s % B) * kvHeads + g) * 4);
                    long a = vb + (long) (s % B) * kvDim + g * hd + i0;
                    int o = d + (s - s0) * w;
                    for (int i = 0; i < w; i += fl) {
                        widen(seg, a + i).mul(sc).intoArray(dst, o + i);
                    }
                }
            }
        }
    }

    @Override
    void expand(int layer, boolean values, int s, int ko, float[] dst, int d) {
        int g = ko / headDim;
        int blk = seq.block(s / B);
        int t = s % B;
        for (int i = 0; i < headDim; i++) {
            dst[d + i] = values ? st.valueValue(layer, blk, g, i, t) : st.keyValue(layer, blk, g, i, t);
        }
    }

    @Override
    public void writeRow(int layer, boolean values, int s, ByteBuffer buf) {
        st.putRow(layer, seq.block(s / B), s % B, values, buf);
    }

    @Override
    public void readRow(int layer, boolean values, int s, ByteBuffer buf) {
        st.getRow(layer, seq.block(s / B), s % B, values, buf);
        if (!values && sparse.layer(layer)) { // a saved cache is read in ascending positions: grow the block's summary row by row
            st.mergeSummaryRow(layer, seq.block(s / B), s % B);
        }
    }

    /** Decode over f32 blocks without the gather ({@code -Djllm.cpu.pagedGather=true} turns it off, for comparisons). */
    private final boolean direct;

    @Override
    public void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window, float scale,
                          float softcap) {
        if (st.rotate) { // (Hq)·(Hk) = B q·k, and B is a power of two (as KvCache.Q8.attention)
            int len = n * qDim;
            if (qRot.length < len) {
                qRot = new float[len];
            }
            System.arraycopy(q, 0, qRot, 0, len);
            for (int o = 0; o < len; o += headDim) {
                org.robomiri.jllm.model.KvQ4.rotate(qRot, o, headDim);
            }
            q = qRot;
            scale /= org.robomiri.jllm.model.KvQ4.block(headDim);
        }
        if (sparse.layer(layer) && window == 0 && n < KvSparse.MAX_ROWS && vectorized() && (pos + n - 1) / B + 1 > sparse.budget(B)) {
            attentionSparseRows(pool, layer, n, pos, q, att, scale, softcap);
            return;
        }
        attentionDense(pool, layer, n, pos, q, att, window, scale, softcap);
    }

    /** {@link #attention} after the queries are rotated. */
    private void attentionDense(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window, float scale,
                                float softcap) {
        if (n == 1 && direct && vectorized()) {
            attentionDecodeDirect(pool, layer, pos, q, att, window, scale, softcap);
            return;
        }
        // the same choice as the flat caches, so that decode and prefill take the same kernels
        if (n == 1 && vectorized() && kvHeads < pool.threads()) {
            attentionDecode(pool, layer, pos, q, att, window, scale, softcap);
        } else {
            attentionTiled(pool, layer, n, pos, q, att, window, scale, softcap);
        }
    }

    // ------------------------------------------------------------------ decode straight from the blocks (f32)

    /** Decode scores of every query head over the context, shared by the tasks of one {@link #attentionDecodeDirect}. */
    private float[] blockScores = new float[0];

    /**
     * {@link KvCache#attentionDecode} for f32 blocks, reading the keys and values where they are instead of gathering
     * them into scratch first (the gather doubles the memory traffic of a decode step): the scores of a block's
     * {@code B} keys are {@code headDim} FMAs with broadcast query elements over the block's transposed
     * {@code [headDim][B]} keys ({@link #scores}), the outputs add the block's V rows in ascending key order
     * ({@link #accumulate}). Every score and every output element is the same chain of operations as the flat kernel's,
     * so the result is the same bit for bit; only the task split is by blocks.
     */
    private void attentionDecodeDirect(ThreadPool pool, int layer, int pos, float[] q, float[] att, int window, float scale,
                                       float softcap) {
        int hd = headDim;
        int fl = VectorKernels.FL;
        int len = pos + 1;
        int first = window > 0 ? Math.max(0, len - window) : 0;
        int heads = kvHeads * group;
        int span = roundUp(len, B);
        if (blockScores.length < (long) heads * span) {
            blockScores = new float[(int) Math.min(Integer.MAX_VALUE - 16, Math.max((long) heads * span, 2L * blockScores.length))];
        }
        float[] sc = blockScores;
        int blk0 = first / B;
        int blk1 = (len + B - 1) / B;
        int want = 2 * pool.threads();
        int keyParts = Math.max(1, Math.min((want + kvHeads - 1) / kvHeads, (blk1 - blk0) / 2)); // >= 2 blocks per task
        int mapped = seq.blocks();
        if (blk1 > mapped) {
            throw new IllegalStateException("position " + pos + " is past the sequence's blocks");
        }
        pool.forRangeFine(kvHeads * keyParts, 1, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int g = i / keyParts;
                int part = i % keyParts;
                int pa = blk0 + (int) ((long) (blk1 - blk0) * part / keyParts);
                int pb = blk0 + (int) ((long) (blk1 - blk0) * (part + 1) / keyParts);
                for (int blk = pa; blk < pb; blk++) {
                    int id = seq.block(blk);
                    MemorySegment seg = st.slab(id);
                    long kb = st.base(id, layer) + (long) g * hd * B * 4;
                    int row = g * group;
                    int r = 0;
                    for (; r + 3 < group; r += 4) {
                        scores4(seg, kb, q, (row + r) * hd, (row + r + 1) * hd, (row + r + 2) * hd, (row + r + 3) * hd, hd,
                                sc, (row + r) * span + blk * B, span, scale);
                    }
                    for (; r + 1 < group; r += 2) {
                        scores2(seg, kb, q, (row + r) * hd, (row + r + 1) * hd, hd, sc, (row + r) * span + blk * B, span, scale);
                    }
                    for (; r < group; r++) {
                        scores1(seg, kb, q, (row + r) * hd, hd, sc, (row + r) * span + blk * B, scale);
                    }
                }
            }
        });
        boolean capped = softcap > 0 && !VectorKernels.FASTMATH;
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
        int slice = hd % (4 * fl) == 0 ? 4 * fl : hd % (2 * fl) == 0 ? 2 * fl : fl;
        int slices = hd / slice;
        pool.forRangeFine(kvHeads * slices, 1, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int g = i / slices;
                int e0 = (i % slices) * slice;
                for (int blk = blk0; blk < blk1; blk++) {
                    int lo = Math.max(first, blk * B);
                    int hi = Math.min(len, (blk + 1) * B);
                    int id = seq.block(blk);
                    MemorySegment seg = st.slab(id);
                    long vb = st.base(id, layer) + st.kBytes + ((long) g * hd + e0) * 4; // key 0 of the block, element e0
                    for (int r = 0; r < group; r++) {
                        int h = g * group + r;
                        accumulate(sc, span, h, lo, hi, blk * B, seg, vb, kvDim, att, h * hd + e0, slice);
                    }
                }
            }
        });
    }

    /** Scores of four query rows against the {@code B} keys of the block at {@code k[kb ..]} ({@code [hd][B]} floats), as {@code KvCache.score4}. */
    private void scores4(MemorySegment k, long kb, float[] q, int q0, int q1, int q2, int q3, int hd, float[] sc, int so, int stride,
                         float scale) {
        int fl = VectorKernels.FL;
        for (int v = 0; v < B; v += 2 * fl) {
            FloatVector a0 = FloatVector.zero(F);
            FloatVector a1 = a0;
            FloatVector a2 = a0;
            FloatVector a3 = a0;
            FloatVector b0 = a0;
            FloatVector b1 = a0;
            FloatVector b2 = a0;
            FloatVector b3 = a0;
            for (int i = 0; i < hd; i++) {
                long at = kb + ((long) i * B + v) * 4;
                FloatVector ka = FloatVector.fromMemorySegment(F, k, at, ORDER);
                FloatVector kz = FloatVector.fromMemorySegment(F, k, at + fl * 4L, ORDER);
                FloatVector x0 = FloatVector.broadcast(F, q[q0 + i]);
                FloatVector x1 = FloatVector.broadcast(F, q[q1 + i]);
                FloatVector x2 = FloatVector.broadcast(F, q[q2 + i]);
                FloatVector x3 = FloatVector.broadcast(F, q[q3 + i]);
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

    private void scores2(MemorySegment k, long kb, float[] q, int q0, int q1, int hd, float[] sc, int so, int stride, float scale) {
        int fl = VectorKernels.FL;
        for (int v = 0; v < B; v += 2 * fl) {
            FloatVector a0 = FloatVector.zero(F);
            FloatVector a1 = a0;
            FloatVector b0 = a0;
            FloatVector b1 = a0;
            for (int i = 0; i < hd; i++) {
                long at = kb + ((long) i * B + v) * 4;
                FloatVector ka = FloatVector.fromMemorySegment(F, k, at, ORDER);
                FloatVector kz = FloatVector.fromMemorySegment(F, k, at + fl * 4L, ORDER);
                FloatVector x0 = FloatVector.broadcast(F, q[q0 + i]);
                FloatVector x1 = FloatVector.broadcast(F, q[q1 + i]);
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

    private void scores1(MemorySegment k, long kb, float[] q, int q0, int hd, float[] sc, int so, float scale) {
        int fl = VectorKernels.FL;
        for (int v = 0; v < B; v += 2 * fl) {
            FloatVector a0 = FloatVector.zero(F);
            FloatVector b0 = a0;
            for (int i = 0; i < hd; i++) {
                long at = kb + ((long) i * B + v) * 4;
                FloatVector x0 = FloatVector.broadcast(F, q[q0 + i]);
                a0 = FloatVector.fromMemorySegment(F, k, at, ORDER).fma(x0, a0);
                b0 = FloatVector.fromMemorySegment(F, k, at + fl * 4L, ORDER).fma(x0, b0);
            }
            a0.mul(scale).intoArray(sc, so + v);
            b0.mul(scale).intoArray(sc, so + v + fl);
        }
    }

    /**
     * Row {@code r}: keys {@code [from, to)} (inside the block starting at position {@code b0}) added to its output
     * slice {@code att[qo ..]} of {@code width} elements, up to four vectors at a time; the V rows of the block start at
     * byte {@code vb} of {@code v} with {@code vStride} floats between keys. As {@code KvCache.accumulate1}.
     */
    private void accumulate(float[] sc, int span, int r, int from, int to, int b0, MemorySegment v, long vb, int vStride,
                            float[] att, int qo, int width) {
        if (from >= to) {
            return;
        }
        int fl = VectorKernels.FL;
        int p = r * span;
        int c = 0;
        for (; c + 4 * fl <= width; c += 4 * fl) {
            var x0 = FloatVector.fromArray(F, att, qo + c);
            var x1 = FloatVector.fromArray(F, att, qo + c + fl);
            var x2 = FloatVector.fromArray(F, att, qo + c + 2 * fl);
            var x3 = FloatVector.fromArray(F, att, qo + c + 3 * fl);
            for (int s = from; s < to; s++) {
                long vo = vb + ((long) (s - b0) * vStride + c) * 4;
                var w = FloatVector.broadcast(F, sc[p + s]);
                x0 = FloatVector.fromMemorySegment(F, v, vo, ORDER).fma(w, x0);
                x1 = FloatVector.fromMemorySegment(F, v, vo + fl * 4L, ORDER).fma(w, x1);
                x2 = FloatVector.fromMemorySegment(F, v, vo + 2 * fl * 4L, ORDER).fma(w, x2);
                x3 = FloatVector.fromMemorySegment(F, v, vo + 3 * fl * 4L, ORDER).fma(w, x3);
            }
            x0.intoArray(att, qo + c);
            x1.intoArray(att, qo + c + fl);
            x2.intoArray(att, qo + c + 2 * fl);
            x3.intoArray(att, qo + c + 3 * fl);
        }
        for (; c < width; c += fl) {
            var x = FloatVector.fromArray(F, att, qo + c);
            for (int s = from; s < to; s++) {
                x = FloatVector.fromMemorySegment(F, v, vb + ((long) (s - b0) * vStride + c) * 4, ORDER)
                        .fma(FloatVector.broadcast(F, sc[p + s]), x);
            }
            x.intoArray(att, qo + c);
        }
    }

    // ------------------------------------------------------------------ sparse decode attention (docs/26)

    /** Set by tests and studies: {@link #lastSelection} receives the blocks each kv head attended in the last sparse row. */
    boolean recordSelection;
    int[][] lastSelection;
    /** Rows attended sparsely so far (a counter for tests and studies). */
    long sparseRows;

    /** What one kv head's task of a sparse row needs; reused between calls (one task per kv head, so no sharing). */
    private static final class SparseScratch {
        float[] bound = new float[0];  // the upper bound of each candidate block
        float[] qe = new float[0];     // the group's queries, even channels then odd channels: [group][2][pairs]
        float[] sc = new float[0];     // scores of the selected keys: [group][keys rounded up to B]
        float[] kt = new float[0];     // one block's keys, [headDim][B]
        float[] vt = new float[0];     // one block's values, [B][headDim]
    }

    private SparseScratch[] sparseScratch;

    /**
     * Rows {@code pos .. pos + n - 1} (a decode step, or a verification chunk) one at a time, each attending the blocks
     * its own position selects; a row whose context still fits the budget takes the dense kernel, so every row is what a
     * decode step at that position computes.
     */
    private void attentionSparseRows(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, float scale,
                                     float softcap) {
        for (int t = 0; t < n; t++) {
            int p = pos + t;
            if (p / B + 1 > sparse.budget(B)) {
                attentionSparseRow(pool, layer, p, q, t * qDim, att, t * qDim, scale, softcap);
            } else {
                float[] q1 = Arrays.copyOfRange(q, t * qDim, (t + 1) * qDim);
                float[] a1 = new float[qDim];
                attentionDense(pool, layer, 1, p, q1, a1, 0, scale, softcap);
                System.arraycopy(a1, 0, att, t * qDim, qDim);
            }
        }
    }

    private void attentionSparseRow(ThreadPool pool, int layer, int p, float[] q, int qo, float[] att, int ao, float scale,
                                    float softcap) {
        if (sparseScratch == null) {
            sparseScratch = new SparseScratch[kvHeads];
            for (int g = 0; g < kvHeads; g++) {
                sparseScratch[g] = new SparseScratch();
            }
        }
        if (recordSelection && lastSelection == null) {
            lastSelection = new int[kvHeads][];
        }
        sparseRows++;
        pool.forRangeFine(kvHeads, 1, (g0, g1) -> {
            for (int g = g0; g < g1; g++) {
                sparseHead(sparseScratch[g], g, layer, p, q, qo, att, ao, scale, softcap);
            }
        });
    }

    /**
     * Kv head {@code g} of the sparse row at position {@code p}: score the summaries of the candidate blocks (all but the
     * sinks and the window) with the group's queries, keep the best {@link KvSparse#blocks()}, and attend exactly over
     * sinks + those + window, which are ascending block numbers ending with the block that holds {@code p}.
     */
    private void sparseHead(SparseScratch s, int g, int layer, int p, float[] q, int qo, float[] att, int ao, float scale,
                            float softcap) {
        int hd = headDim;
        int nBlocks = p / B + 1;
        int sink = sparse.sinkBlocks(B);
        int win = sparse.windowBlocks(B);
        int cand1 = nBlocks - win;
        int m = cand1 - sink;
        int nSel = sink + sparse.blocks() + win;
        int keys = (nSel - 1) * B + (p % B + 1);
        int kpad = nSel * B;
        if (s.bound.length < m) {
            s.bound = new float[Math.max(m, 2 * s.bound.length)];
        }
        if (s.qe.length < group * 2 * pairs) {
            s.qe = new float[group * 2 * pairs];
        }
        if (s.sc.length < group * kpad) {
            s.sc = new float[group * kpad];
        }
        if (s.kt.length < hd * B) {
            s.kt = new float[hd * B];
            s.vt = new float[hd * B];
        }
        // 1. the group's queries split into even and odd channels, the order the summaries are packed in
        for (int r = 0; r < group; r++) {
            int qb = qo + (g * group + r) * hd;
            for (int pp = 0; pp < pairs; pp++) {
                s.qe[(r * 2) * pairs + pp] = q[qb + 2 * pp];
                s.qe[(r * 2 + 1) * pairs + pp] = 2 * pp + 1 < hd ? q[qb + 2 * pp + 1] : 0f;
            }
        }
        // 2. the bound of each candidate block: the largest over the group's heads
        for (int j = 0; j < m; j++) {
            int id = seq.block(sink + j);
            MemorySegment seg = st.slab(id);
            long sb = st.sumBase(id, layer) + (long) g * 2 * pairs * 4;
            float best = Float.NEGATIVE_INFINITY;
            for (int r = 0; r < group; r++) {
                best = Math.max(best, blockBound(seg, sb, s.qe, r * 2 * pairs));
            }
            s.bound[j] = best;
        }
        int[] top = KvSummary.top(s.bound, sink, m, sparse.blocks());
        int[] sel = new int[nSel];
        int c = 0;
        for (int j = 0; j < sink; j++) {
            sel[c++] = j;
        }
        for (int j : top) {
            sel[c++] = j;
        }
        for (int j = cand1; j < nBlocks; j++) {
            sel[c++] = j;
        }
        if (recordSelection) {
            lastSelection[g] = sel.clone();
        }
        // 3. exact attention over the selected keys: scores, softmax, values
        for (int mi = 0; mi < nSel; mi++) {
            gatherKeys(layer, g, seq.block(sel[mi]), s.kt);
            for (int r = 0; r < group; r++) {
                blockScores(s.kt, q, qo + (g * group + r) * hd, s.sc, r * kpad + mi * B, scale);
            }
        }
        boolean capped = softcap > 0 && !VectorKernels.FASTMATH;
        for (int r = 0; r < group; r++) {
            int o = r * kpad;
            if (capped) {
                for (int i = 0; i < keys; i++) {
                    s.sc[o + i] = (float) Math.tanh(s.sc[o + i] / softcap) * softcap;
                }
            }
            float inv = 1f / softmax(s.sc, o, keys, softcap);
            VectorKernels.scale(s.sc, o, keys, inv);
            Arrays.fill(att, ao + (g * group + r) * hd, ao + (g * group + r + 1) * hd, 0f);
        }
        for (int mi = 0; mi < nSel; mi++) {
            gatherValues(layer, g, seq.block(sel[mi]), s.vt);
            int rows = mi == nSel - 1 ? p % B + 1 : B;
            for (int r = 0; r < group; r++) {
                accumulateRows(s.sc, r * kpad + mi * B, s.vt, rows, att, ao + (g * group + r) * hd);
            }
        }
    }

    /** {@code Σ_i max(q_i lo_i, q_i hi_i)} for the summary at byte {@code sb} of {@code seg}, the query at {@code qe[qb ..]} (even channels, then odd). */
    private float blockBound(MemorySegment seg, long sb, float[] qe, int qb) {
        int fl = VectorKernels.FL;
        FloatVector acc = FloatVector.zero(F);
        int pp = 0;
        for (; pp + fl <= pairs; pp += fl) {
            IntVector lo = IntVector.fromMemorySegment(I, seg, sb + pp * 4L, ORDER);
            IntVector hi = IntVector.fromMemorySegment(I, seg, sb + (pairs + pp) * 4L, ORDER);
            FloatVector e = FloatVector.fromArray(F, qe, qb + pp);
            FloatVector o = FloatVector.fromArray(F, qe, qb + pairs + pp);
            FloatVector loE = F16.lowHalf(lo);
            FloatVector hiE = F16.lowHalf(hi);
            FloatVector loO = F16.highHalf(lo);
            FloatVector hiO = F16.highHalf(hi);
            acc = acc.add(e.mul(loE).max(e.mul(hiE))).add(o.mul(loO).max(o.mul(hiO)));
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; pp < pairs; pp++) {
            int lo = seg.get(ValueLayout.JAVA_INT, sb + pp * 4L);
            int hi = seg.get(ValueLayout.JAVA_INT, sb + (pairs + pp) * 4L);
            float e = qe[qb + pp];
            float o = qe[qb + pairs + pp];
            sum += Math.max(e * Float.float16ToFloat(CpuKvStorage.half(lo, false)), e * Float.float16ToFloat(CpuKvStorage.half(hi, false)));
            sum += Math.max(o * Float.float16ToFloat(CpuKvStorage.half(lo, true)), o * Float.float16ToFloat(CpuKvStorage.half(hi, true)));
        }
        return sum;
    }

    /** All {@code B} keys of kv head {@code g} of block {@code id} as f32, {@code [headDim][B]}, into {@code dst}. */
    private void gatherKeys(int layer, int g, int id, float[] dst) {
        int fl = VectorKernels.FL;
        int hd = headDim;
        MemorySegment seg = st.slab(id);
        long kb = st.base(id, layer);
        switch (dtype) {
            case F32 -> {
                for (int i = 0; i < hd; i++) {
                    long a = kb + ((long) g * hd + i) * B * 4;
                    for (int v = 0; v < B; v += fl) {
                        FloatVector.fromMemorySegment(F, seg, a + v * 4L, ORDER).intoArray(dst, i * B + v);
                    }
                }
            }
            case F16 -> {
                for (int p = 0; p < pairs; p++) {
                    long a = kb + ((long) g * pairs + p) * B * 4;
                    for (int v = 0; v < B; v += fl) {
                        IntVector x = IntVector.fromMemorySegment(I, seg, a + v * 4L, ORDER);
                        F16.lowHalf(x).intoArray(dst, 2 * p * B + v);
                        if (2 * p + 1 < hd) {
                            F16.highHalf(x).intoArray(dst, (2 * p + 1) * B + v);
                        }
                    }
                }
            }
            default -> {
                for (int v = 0; v < B; v += fl) {
                    FloatVector sc = FloatVector.fromMemorySegment(F, seg, kb + st.kScaleOff + ((long) g * B + v) * 4, ORDER);
                    for (int i = 0; i < hd; i++) {
                        widen(seg, kb + ((long) g * hd + i) * B + v).mul(sc).intoArray(dst, i * B + v);
                    }
                }
            }
        }
    }

    /** All {@code B} value rows of kv head {@code g} of block {@code id} as f32, {@code [B][headDim]}, into {@code dst}. */
    private void gatherValues(int layer, int g, int id, float[] dst) {
        int fl = VectorKernels.FL;
        int hd = headDim;
        MemorySegment seg = st.slab(id);
        long vb = st.base(id, layer) + st.kBytes;
        switch (dtype) {
            case F32 -> {
                for (int t = 0; t < B; t++) {
                    long a = vb + ((long) t * kvDim + g * hd) * 4;
                    for (int e = 0; e < hd; e += fl) {
                        FloatVector.fromMemorySegment(F, seg, a + e * 4L, ORDER).intoArray(dst, t * hd + e);
                    }
                }
            }
            case F16 -> {
                for (int t = 0; t < B; t += 2) {
                    long a = vb + ((long) (t >> 1) * kvDim + g * hd) * 4;
                    for (int e = 0; e < hd; e += fl) {
                        IntVector x = IntVector.fromMemorySegment(I, seg, a + e * 4L, ORDER);
                        F16.lowHalf(x).intoArray(dst, t * hd + e);
                        F16.highHalf(x).intoArray(dst, (t + 1) * hd + e);
                    }
                }
            }
            default -> {
                for (int t = 0; t < B; t++) {
                    float sc = seg.get(ValueLayout.JAVA_FLOAT, vb + st.vScaleOff + ((long) t * kvHeads + g) * 4);
                    long a = vb + (long) t * kvDim + g * hd;
                    for (int e = 0; e < hd; e += fl) {
                        widen(seg, a + e).mul(sc).intoArray(dst, t * hd + e);
                    }
                }
            }
        }
    }

    /** The scores of one query head against the {@code B} gathered keys {@code kt} ({@code [headDim][B]}): {@code scale · q·k} into {@code sc[so ..]}. */
    private void blockScores(float[] kt, float[] q, int qb, float[] sc, int so, float scale) {
        int fl = VectorKernels.FL;
        int hd = headDim;
        for (int v = 0; v < B; v += fl) {
            FloatVector a = FloatVector.zero(F);
            for (int i = 0; i < hd; i++) {
                a = FloatVector.fromArray(F, kt, i * B + v).fma(FloatVector.broadcast(F, q[qb + i]), a);
            }
            a.mul(scale).intoArray(sc, so + v);
        }
    }

    /** {@code att[ao ..] += Σ_t sc[so + t] · vt[t]} over the first {@code rows} gathered value rows, in ascending order. */
    private void accumulateRows(float[] sc, int so, float[] vt, int rows, float[] att, int ao) {
        int fl = VectorKernels.FL;
        int hd = headDim;
        int c = 0;
        for (; c + 4 * fl <= hd; c += 4 * fl) {
            FloatVector x0 = FloatVector.fromArray(F, att, ao + c);
            FloatVector x1 = FloatVector.fromArray(F, att, ao + c + fl);
            FloatVector x2 = FloatVector.fromArray(F, att, ao + c + 2 * fl);
            FloatVector x3 = FloatVector.fromArray(F, att, ao + c + 3 * fl);
            for (int t = 0; t < rows; t++) {
                FloatVector w = FloatVector.broadcast(F, sc[so + t]);
                int vo = t * hd + c;
                x0 = FloatVector.fromArray(F, vt, vo).fma(w, x0);
                x1 = FloatVector.fromArray(F, vt, vo + fl).fma(w, x1);
                x2 = FloatVector.fromArray(F, vt, vo + 2 * fl).fma(w, x2);
                x3 = FloatVector.fromArray(F, vt, vo + 3 * fl).fma(w, x3);
            }
            x0.intoArray(att, ao + c);
            x1.intoArray(att, ao + c + fl);
            x2.intoArray(att, ao + c + 2 * fl);
            x3.intoArray(att, ao + c + 3 * fl);
        }
        for (; c < hd; c += fl) {
            FloatVector x = FloatVector.fromArray(F, att, ao + c);
            for (int t = 0; t < rows; t++) {
                x = FloatVector.fromArray(F, vt, t * hd + c).fma(FloatVector.broadcast(F, sc[so + t]), x);
            }
            x.intoArray(att, ao + c);
        }
    }
}
