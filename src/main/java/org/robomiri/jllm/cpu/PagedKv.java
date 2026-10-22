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
        if (B % VectorKernels.FL != 0 || B % 2 != 0) {
            throw new IllegalArgumentException("backend.kv-block " + B + " must be a multiple of the vector length " + VectorKernels.FL);
        }
    }

    @Override
    KvDtype storedType() {
        return dtype;
    }

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

    @Override
    public void attention(ThreadPool pool, int layer, int n, int pos, float[] q, float[] att, int window, float scale,
                          float softcap) {
        // the same choice as the flat caches, so that decode and prefill take the same kernels
        if (n == 1 && vectorized() && kvHeads < pool.threads()) {
            attentionDecode(pool, layer, pos, q, att, window, scale, softcap);
        } else {
            attentionTiled(pool, layer, n, pos, q, att, window, scale, softcap);
        }
    }
}
