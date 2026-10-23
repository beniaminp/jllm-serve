package org.robomiri.jllm.cpu;

import org.robomiri.jllm.kv.KvSequence;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
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

    PagedKv(int layers, Attention.Gqa a, int ctx, CpuKvStorage st, KvSequence seq) {
        super(layers, a, ctx);
        this.st = st;
        this.seq = seq;
        this.dtype = st.dtype;
        this.B = st.B;
        this.pairs = st.pairs;
        this.direct = B % (2 * VectorKernels.FL) == 0 && !Boolean.getBoolean("jllm.cpu.pagedGather");
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
                    scoreBlock(st.slab(id), st.base(id, layer), g, q, sc, blk, span, scale);
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
                    long lb = st.base(id, layer);
                    for (int r = 0; r < group; r++) {
                        int h = g * group + r;
                        switch (dtype) {
                            case F32 -> accumulate(sc, span, h, lo, hi, blk * B, seg, lb + st.kBytes + ((long) g * hd + e0) * 4, kvDim,
                                    att, h * hd + e0, slice); // key 0 of the block, element e0
                            case F16 -> PagedKernels.accumulateF16(sc, span, h, lo, hi, blk * B, seg, lb + st.kBytes, kvDim, g, hd, e0,
                                    att, h * hd + e0, slice);
                            default -> PagedKernels.accumulateQ8(sc, span, h, lo, hi, blk * B, seg, lb + st.kBytes, st.vScaleOff, kvDim,
                                    kvHeads, g, hd, e0, att, h * hd + e0, slice);
                        }
                    }
                }
            }
        });
    }

    /** The scores of kv head {@code g}'s query rows against the {@code B} keys of block {@code blk} (layer starting at byte {@code lb}). */
    private void scoreBlock(MemorySegment seg, long lb, int g, float[] q, float[] sc, int blk, int span, float scale) {
        int hd = headDim;
        int row = g * group;
        int r = 0;
        int so = row * span + blk * B;
        switch (dtype) {
            case F32 -> {
                long kb = lb + (long) g * hd * B * 4;
                for (; r + 3 < group; r += 4) {
                    scores4(seg, kb, q, (row + r) * hd, (row + r + 1) * hd, (row + r + 2) * hd, (row + r + 3) * hd, hd, sc,
                            so + r * span, span, scale);
                }
                for (; r + 1 < group; r += 2) {
                    scores2(seg, kb, q, (row + r) * hd, (row + r + 1) * hd, hd, sc, so + r * span, span, scale);
                }
                for (; r < group; r++) {
                    scores1(seg, kb, q, (row + r) * hd, hd, sc, so + r * span, scale);
                }
            }
            case F16 -> {
                for (; r + 3 < group; r += 4) {
                    PagedKernels.scoresF16x4(seg, lb, g, pairs, B, q, (row + r) * hd, (row + r + 1) * hd, (row + r + 2) * hd,
                            (row + r + 3) * hd, sc, so + r * span, span, scale);
                }
                for (; r + 1 < group; r += 2) {
                    PagedKernels.scoresF16x2(seg, lb, g, pairs, B, q, (row + r) * hd, (row + r + 1) * hd, sc, so + r * span, span, scale);
                }
                for (; r < group; r++) {
                    PagedKernels.scoresF16x1(seg, lb, g, pairs, B, q, (row + r) * hd, sc, so + r * span, scale);
                }
            }
            default -> {
                for (; r + 3 < group; r += 4) {
                    PagedKernels.scoresQ8x4(seg, lb, st.kScaleOff, g, hd, B, q, (row + r) * hd, (row + r + 1) * hd, (row + r + 2) * hd,
                            (row + r + 3) * hd, sc, so + r * span, span, scale);
                }
                for (; r + 1 < group; r += 2) {
                    PagedKernels.scoresQ8x2(seg, lb, st.kScaleOff, g, hd, B, q, (row + r) * hd, (row + r + 1) * hd, sc, so + r * span,
                            span, scale);
                }
                for (; r < group; r++) {
                    PagedKernels.scoresQ8x1(seg, lb, st.kScaleOff, g, hd, B, q, (row + r) * hd, sc, so + r * span, scale);
                }
            }
        }
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
}
