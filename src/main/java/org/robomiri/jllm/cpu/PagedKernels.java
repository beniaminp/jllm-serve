package org.robomiri.jllm.cpu;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * The decode kernels of {@link PagedKv} for f16 and q8 blocks, which read the block memory in place and convert on the
 * fly instead of gathering a chunk into scratch first (docs/24 §24.2.6). Each score is the same chain of FMAs over the head
 * elements in ascending order, with the same converted operands, as the flat cache's expand-then-{@code score4}; each output
 * element adds the keys in ascending order as {@code accumulate1}; so the results are bit for bit the flat cache's.
 *
 * <p>f16 keys: a block holds {@code [kvHead][pairs][B]} ints, element {@code 2p} of key {@code t} in the low half of int
 * {@code (g * pairs + p) * B + t}, element {@code 2p + 1} in the high half. q8 keys: {@code [kvHead][hd][B]} bytes and the
 * scales {@code [kvHead][B]} floats. f16 values: {@code [B/2][kvDim]} ints (key {@code 2j} low, {@code 2j + 1} high); q8
 * values: {@code [B][kvDim]} bytes and {@code [B][kvHeads]} scales.
 */
final class PagedKernels {

    private static final VectorSpecies<Float> F = VectorKernels.F;
    private static final VectorSpecies<Integer> I = IntVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> B16 = VectorKernels.B16;
    private static final ByteOrder ORDER = CpuKvStorage.ORDER;

    private PagedKernels() {
    }

    private static FloatVector widen(MemorySegment seg, long at) {
        return (FloatVector) ByteVector.fromMemorySegment(B16, seg, at, ORDER).castShape(F, 0);
    }

    // ------------------------------------------------------------------ f16 keys

    /** Scores of four query rows against the {@code B} keys of kv head {@code g} of the block whose layer starts at byte {@code lb}. */
    static void scoresF16x4(MemorySegment k, long lb, int g, int pairs, int B, float[] q, int q0, int q1, int q2, int q3, float[] sc,
                            int so, int stride, float scale) {
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
            for (int p = 0; p < pairs; p++) {
                long at = lb + (((long) g * pairs + p) * B + v) * 4;
                IntVector xa = IntVector.fromMemorySegment(I, k, at, ORDER);
                IntVector xz = IntVector.fromMemorySegment(I, k, at + fl * 4L, ORDER);
                int i = 2 * p; // element 2p (low halves), then 2p + 1 (high), straight-line so the vectors stay unboxed
                FloatVector ka = KvCache.F16.lowHalf(xa);
                FloatVector kz = KvCache.F16.lowHalf(xz);
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
                ka = KvCache.F16.highHalf(xa);
                kz = KvCache.F16.highHalf(xz);
                x0 = FloatVector.broadcast(F, q[q0 + i + 1]);
                x1 = FloatVector.broadcast(F, q[q1 + i + 1]);
                x2 = FloatVector.broadcast(F, q[q2 + i + 1]);
                x3 = FloatVector.broadcast(F, q[q3 + i + 1]);
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

    static void scoresF16x2(MemorySegment k, long lb, int g, int pairs, int B, float[] q, int q0, int q1, float[] sc, int so,
                            int stride, float scale) {
        int fl = VectorKernels.FL;
        for (int v = 0; v < B; v += 2 * fl) {
            FloatVector a0 = FloatVector.zero(F);
            FloatVector a1 = a0;
            FloatVector b0 = a0;
            FloatVector b1 = a0;
            for (int p = 0; p < pairs; p++) {
                long at = lb + (((long) g * pairs + p) * B + v) * 4;
                IntVector xa = IntVector.fromMemorySegment(I, k, at, ORDER);
                IntVector xz = IntVector.fromMemorySegment(I, k, at + fl * 4L, ORDER);
                int i = 2 * p;
                FloatVector ka = KvCache.F16.lowHalf(xa);
                FloatVector kz = KvCache.F16.lowHalf(xz);
                FloatVector x0 = FloatVector.broadcast(F, q[q0 + i]);
                FloatVector x1 = FloatVector.broadcast(F, q[q1 + i]);
                a0 = ka.fma(x0, a0);
                b0 = kz.fma(x0, b0);
                a1 = ka.fma(x1, a1);
                b1 = kz.fma(x1, b1);
                ka = KvCache.F16.highHalf(xa);
                kz = KvCache.F16.highHalf(xz);
                x0 = FloatVector.broadcast(F, q[q0 + i + 1]);
                x1 = FloatVector.broadcast(F, q[q1 + i + 1]);
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

    static void scoresF16x1(MemorySegment k, long lb, int g, int pairs, int B, float[] q, int q0, float[] sc, int so, float scale) {
        int fl = VectorKernels.FL;
        for (int v = 0; v < B; v += 2 * fl) {
            FloatVector a0 = FloatVector.zero(F);
            FloatVector b0 = a0;
            for (int p = 0; p < pairs; p++) {
                long at = lb + (((long) g * pairs + p) * B + v) * 4;
                IntVector xa = IntVector.fromMemorySegment(I, k, at, ORDER);
                IntVector xz = IntVector.fromMemorySegment(I, k, at + fl * 4L, ORDER);
                int i = 2 * p;
                FloatVector x0 = FloatVector.broadcast(F, q[q0 + i]);
                a0 = KvCache.F16.lowHalf(xa).fma(x0, a0);
                b0 = KvCache.F16.lowHalf(xz).fma(x0, b0);
                x0 = FloatVector.broadcast(F, q[q0 + i + 1]);
                a0 = KvCache.F16.highHalf(xa).fma(x0, a0);
                b0 = KvCache.F16.highHalf(xz).fma(x0, b0);
            }
            a0.mul(scale).intoArray(sc, so + v);
            b0.mul(scale).intoArray(sc, so + v + fl);
        }
    }

    // ------------------------------------------------------------------ q8 keys

    static void scoresQ8x4(MemorySegment k, long lb, long scaleOff, int g, int hd, int B, float[] q, int q0, int q1, int q2, int q3,
                           float[] sc, int so, int stride, float scale) {
        int fl = VectorKernels.FL;
        for (int v = 0; v < B; v += 2 * fl) {
            FloatVector sa = FloatVector.fromMemorySegment(F, k, lb + scaleOff + ((long) g * B + v) * 4, ORDER);
            FloatVector sz = FloatVector.fromMemorySegment(F, k, lb + scaleOff + ((long) g * B + v + fl) * 4, ORDER);
            FloatVector a0 = FloatVector.zero(F);
            FloatVector a1 = a0;
            FloatVector a2 = a0;
            FloatVector a3 = a0;
            FloatVector b0 = a0;
            FloatVector b1 = a0;
            FloatVector b2 = a0;
            FloatVector b3 = a0;
            for (int i = 0; i < hd; i++) {
                long at = lb + ((long) g * hd + i) * B + v;
                FloatVector ka = widen(k, at).mul(sa);
                FloatVector kz = widen(k, at + fl).mul(sz);
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

    static void scoresQ8x2(MemorySegment k, long lb, long scaleOff, int g, int hd, int B, float[] q, int q0, int q1, float[] sc, int so,
                           int stride, float scale) {
        int fl = VectorKernels.FL;
        for (int v = 0; v < B; v += 2 * fl) {
            FloatVector sa = FloatVector.fromMemorySegment(F, k, lb + scaleOff + ((long) g * B + v) * 4, ORDER);
            FloatVector sz = FloatVector.fromMemorySegment(F, k, lb + scaleOff + ((long) g * B + v + fl) * 4, ORDER);
            FloatVector a0 = FloatVector.zero(F);
            FloatVector a1 = a0;
            FloatVector b0 = a0;
            FloatVector b1 = a0;
            for (int i = 0; i < hd; i++) {
                long at = lb + ((long) g * hd + i) * B + v;
                FloatVector ka = widen(k, at).mul(sa);
                FloatVector kz = widen(k, at + fl).mul(sz);
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

    static void scoresQ8x1(MemorySegment k, long lb, long scaleOff, int g, int hd, int B, float[] q, int q0, float[] sc, int so,
                           float scale) {
        int fl = VectorKernels.FL;
        for (int v = 0; v < B; v += 2 * fl) {
            FloatVector sa = FloatVector.fromMemorySegment(F, k, lb + scaleOff + ((long) g * B + v) * 4, ORDER);
            FloatVector sz = FloatVector.fromMemorySegment(F, k, lb + scaleOff + ((long) g * B + v + fl) * 4, ORDER);
            FloatVector a0 = FloatVector.zero(F);
            FloatVector b0 = a0;
            for (int i = 0; i < hd; i++) {
                long at = lb + ((long) g * hd + i) * B + v;
                FloatVector x0 = FloatVector.broadcast(F, q[q0 + i]);
                a0 = widen(k, at).mul(sa).fma(x0, a0);
                b0 = widen(k, at + fl).mul(sz).fma(x0, b0);
            }
            a0.mul(scale).intoArray(sc, so + v);
            b0.mul(scale).intoArray(sc, so + v + fl);
        }
    }

    // ------------------------------------------------------------------ values

    /**
     * Row {@code r}: keys {@code [from, to)} of the block starting at position {@code b0} added to the output slice
     * {@code att[qo ..]} of {@code width} elements (kv head {@code g}, elements from {@code e0}), f16 values: {@code vb} is the
     * byte offset of the block's V region, {@code kvDim} the row width. As {@code KvCache.accumulate1} over expanded halves.
     */
    static void accumulateF16(float[] sc, int span, int r, int from, int to, int b0, MemorySegment v, long vb, int kvDim, int g,
                              int hd, int e0, float[] att, int qo, int width) {
        if (from >= to) {
            return;
        }
        int fl = VectorKernels.FL;
        int p = r * span;
        long col = (long) g * hd + e0;
        int c = 0;
        for (; c + 4 * fl <= width; c += 4 * fl) {
            var x0 = FloatVector.fromArray(F, att, qo + c);
            var x1 = FloatVector.fromArray(F, att, qo + c + fl);
            var x2 = FloatVector.fromArray(F, att, qo + c + 2 * fl);
            var x3 = FloatVector.fromArray(F, att, qo + c + 3 * fl);
            for (int s = from; s < to; s++) {
                int row = s - b0;
                long vo = vb + ((long) (row >> 1) * kvDim + col + c) * 4;
                var w = FloatVector.broadcast(F, sc[p + s]);
                IntVector y0 = IntVector.fromMemorySegment(I, v, vo, ORDER);
                IntVector y1 = IntVector.fromMemorySegment(I, v, vo + fl * 4L, ORDER);
                IntVector y2 = IntVector.fromMemorySegment(I, v, vo + 2 * fl * 4L, ORDER);
                IntVector y3 = IntVector.fromMemorySegment(I, v, vo + 3 * fl * 4L, ORDER);
                if ((row & 1) != 0) {
                    x0 = KvCache.F16.highHalf(y0).fma(w, x0);
                    x1 = KvCache.F16.highHalf(y1).fma(w, x1);
                    x2 = KvCache.F16.highHalf(y2).fma(w, x2);
                    x3 = KvCache.F16.highHalf(y3).fma(w, x3);
                } else {
                    x0 = KvCache.F16.lowHalf(y0).fma(w, x0);
                    x1 = KvCache.F16.lowHalf(y1).fma(w, x1);
                    x2 = KvCache.F16.lowHalf(y2).fma(w, x2);
                    x3 = KvCache.F16.lowHalf(y3).fma(w, x3);
                }
            }
            x0.intoArray(att, qo + c);
            x1.intoArray(att, qo + c + fl);
            x2.intoArray(att, qo + c + 2 * fl);
            x3.intoArray(att, qo + c + 3 * fl);
        }
        for (; c < width; c += fl) {
            var x = FloatVector.fromArray(F, att, qo + c);
            for (int s = from; s < to; s++) {
                int row = s - b0;
                IntVector y = IntVector.fromMemorySegment(I, v, vb + ((long) (row >> 1) * kvDim + col + c) * 4, ORDER);
                var w = FloatVector.broadcast(F, sc[p + s]);
                if ((row & 1) != 0) {
                    x = KvCache.F16.highHalf(y).fma(w, x);
                } else {
                    x = KvCache.F16.lowHalf(y).fma(w, x);
                }
            }
            x.intoArray(att, qo + c);
        }
    }

    /** q8 values: {@code vb} is the byte offset of the block's V region (bytes {@code [B][kvDim]}), {@code scaleOff} its scales ({@code [B][kvHeads]}). */
    static void accumulateQ8(float[] sc, int span, int r, int from, int to, int b0, MemorySegment v, long vb, long scaleOff, int kvDim,
                             int kvHeads, int g, int hd, int e0, float[] att, int qo, int width) {
        if (from >= to) {
            return;
        }
        int fl = VectorKernels.FL;
        int p = r * span;
        long col = (long) g * hd + e0;
        int c = 0;
        for (; c + 4 * fl <= width; c += 4 * fl) {
            var x0 = FloatVector.fromArray(F, att, qo + c);
            var x1 = FloatVector.fromArray(F, att, qo + c + fl);
            var x2 = FloatVector.fromArray(F, att, qo + c + 2 * fl);
            var x3 = FloatVector.fromArray(F, att, qo + c + 3 * fl);
            for (int s = from; s < to; s++) {
                int row = s - b0;
                long vo = vb + (long) row * kvDim + col + c;
                float vs = v.get(ValueLayout.JAVA_FLOAT, vb + scaleOff + ((long) row * kvHeads + g) * 4);
                var w = FloatVector.broadcast(F, sc[p + s]);
                x0 = widen(v, vo).mul(vs).fma(w, x0);
                x1 = widen(v, vo + fl).mul(vs).fma(w, x1);
                x2 = widen(v, vo + 2L * fl).mul(vs).fma(w, x2);
                x3 = widen(v, vo + 3L * fl).mul(vs).fma(w, x3);
            }
            x0.intoArray(att, qo + c);
            x1.intoArray(att, qo + c + fl);
            x2.intoArray(att, qo + c + 2 * fl);
            x3.intoArray(att, qo + c + 3 * fl);
        }
        for (; c < width; c += fl) {
            var x = FloatVector.fromArray(F, att, qo + c);
            for (int s = from; s < to; s++) {
                int row = s - b0;
                float vs = v.get(ValueLayout.JAVA_FLOAT, vb + scaleOff + ((long) row * kvHeads + g) * 4);
                x = widen(v, vb + (long) row * kvDim + col + c).mul(vs).fma(FloatVector.broadcast(F, sc[p + s]), x);
            }
            x.intoArray(att, qo + c);
        }
    }
}
