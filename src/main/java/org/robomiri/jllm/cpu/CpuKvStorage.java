package org.robomiri.jllm.cpu;

import org.robomiri.jllm.kv.KvStorage;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * The CPU backend's block memory, off the Java heap: slabs of {@link MemorySegment}s allocated as the pool grows, each
 * holding {@code slabBlocks} blocks. A block holds {@code B} positions of every layer, so copying a block (copy on
 * write) is one memory copy:
 *
 * <pre>
 * slab:   [block][layer][K region | V region]          block = slab index (id &gt;&gt; shift), (id &amp; mask) within it
 * f32  K: float  [kvHead][headDim][B]      V: float  [B][kvDim]
 * f16  K: int    [kvHead][pairs][B]        V: int    [B/2][kvDim]       (two halves per int, as KvCache.F16)
 * q8   K: byte   [kvHead][headDim][B]      V: byte   [B][kvDim]
 *      then K scales float [kvHead][B]     V scales float [B][kvHead]
 * </pre>
 *
 * The keys keep the transposed layout of {@link KvCache} (a vector load holds one head element of consecutive keys),
 * with the block's {@code B} keys as the "chunk": the gather in {@link PagedKv} puts {@code CHUNK / B} blocks' worth side
 * by side into the tile kernels' scratch, so every score and output is computed by the same operations as with the flat
 * cache. All byte offsets are {@code long} (a 1M-token cache of a real model is far beyond 2^31 bytes).
 */
final class CpuKvStorage implements KvStorage {

    static final ByteOrder ORDER = ByteOrder.nativeOrder();

    final KvDtype dtype;
    final int layers;
    final int kvHeads;
    final int headDim;
    final int kvDim;
    final int B;
    /** Packed element pairs per head (f16). */
    final int pairs;
    final long kBytes;
    final long vBytes;
    /** Where the scales start inside the K / V region (q8). */
    final long kScaleOff;
    final long vScaleOff;
    final long layerBytes;
    final long blockBytes;
    private final int slabShift;
    private final int slabMask;
    private final long slabBytes;
    private volatile MemorySegment[] slabs = new MemorySegment[0];
    private final Arena arena = Arena.ofAuto();

    private static long round64(long x) {
        return (x + 63) & ~63L;
    }

    /** q8 with Hadamard-rotated keys ({@code backend.kv-q8-rotate}): {@link #storeRow} rotates a key head before it is quantized. */
    final boolean rotate;

    CpuKvStorage(KvDtype dtype, int layers, Attention.Gqa a, int blockTokens) {
        this(dtype, layers, a, blockTokens, false);
    }

    CpuKvStorage(KvDtype dtype, int layers, Attention.Gqa a, int blockTokens, boolean rotateQ8) {
        if (dtype == KvDtype.Q4) {
            throw new IllegalArgumentException("the paged CPU cache has no q4 blocks yet");
        }
        if (blockTokens < 16 || Integer.bitCount(blockTokens) != 1) {
            throw new IllegalArgumentException("backend.kv-block must be a power of two of at least 16, got " + blockTokens);
        }
        this.dtype = dtype;
        this.rotate = rotateQ8 && dtype == KvDtype.Q8 && org.robomiri.jllm.model.KvQ4.supports(a.headDim());
        this.layers = layers;
        this.kvHeads = a.kvHeads();
        this.headDim = a.headDim();
        this.kvDim = a.kvDim();
        this.B = blockTokens;
        this.pairs = (headDim + 1) / 2;
        switch (dtype) {
            case F32 -> {
                kBytes = round64((long) kvHeads * headDim * B * 4);
                vBytes = round64((long) B * kvDim * 4);
                kScaleOff = 0;
                vScaleOff = 0;
            }
            case F16 -> {
                kBytes = round64((long) kvHeads * pairs * B * 4);
                vBytes = round64((long) (B / 2) * kvDim * 4);
                kScaleOff = 0;
                vScaleOff = 0;
            }
            default -> { // Q8
                kScaleOff = round64((long) kvHeads * headDim * B);
                vScaleOff = round64((long) B * kvDim);
                kBytes = kScaleOff + round64((long) kvHeads * B * 4);
                vBytes = vScaleOff + round64((long) B * kvHeads * 4);
            }
        }
        layerBytes = kBytes + vBytes;
        blockBytes = layerBytes * layers;
        long perSlab = Math.max(1, Math.min(1024, (16L << 20) / blockBytes));
        int shift = 63 - Long.numberOfLeadingZeros(perSlab);
        this.slabShift = shift;
        this.slabMask = (1 << shift) - 1;
        this.slabBytes = ((long) 1 << shift) * blockBytes + 64; // padding: a 16-byte load at the last byte stays inside
    }

    @Override
    public int blockTokens() {
        return B;
    }

    @Override
    public int spillBytes() {
        long n = (long) layers * B * 2 * dtype.rowBytes(kvDim, kvHeads);
        if (n > Integer.MAX_VALUE) {
            throw new IllegalStateException("a block of " + n + " bytes is too large to spill; lower backend.kv-block");
        }
        return (int) n;
    }

    @Override
    public org.robomiri.jllm.kv.KvBlockCodec codec(org.robomiri.jllm.kv.KvBlockCodec.Format f) {
        return new org.robomiri.jllm.kv.KvBlockCodec(f, dtype, layers, B, kvHeads, headDim);
    }

    @Override
    public long residentBytes() {
        return blockBytes;
    }

    @Override
    public void grow(int blocks) {
        int need = ((blocks - 1) >> slabShift) + 1;
        if (slabs.length >= need) {
            return;
        }
        synchronized (this) {
            MemorySegment[] cur = slabs;
            if (cur.length >= need) {
                return;
            }
            MemorySegment[] next = java.util.Arrays.copyOf(cur, need);
            for (int i = cur.length; i < need; i++) {
                if (i >= hotSlabs && coldDir != null) {
                    next[i] = mapSlab();
                } else {
                    next[i] = arena.allocate(slabBytes, 64);
                    hotCount++;
                }
            }
            slabs = next;
        }
    }

    // ------------------------------------------------------------------ the SSD tier (docs/26 §26.2.3)

    private int hotSlabs = Integer.MAX_VALUE;
    private java.nio.file.Path coldDir;
    private int hotCount;
    private int coldCount;

    /**
     * Keeps at most {@code hotBytes} of slabs in the process's own memory: the slabs past that many are memory-mapped
     * files in {@code dir} (unlinked at once: they live as long as the mapping), so the operating system pages them between RAM and the
     * SSD. The blocks in them are read and written exactly like the others (same code, same bytes), so results do not
     * change. Call before the first block is handed out.
     */
    void setHotBudget(long hotBytes, java.nio.file.Path dir) {
        if (slabs.length > 0) {
            throw new IllegalStateException("the hot budget must be set before the pool grows");
        }
        this.hotSlabs = (int) Math.max(0, Math.min(Integer.MAX_VALUE, hotBytes / Math.max(1, slabBytes)));
        this.coldDir = dir;
    }

    /** Slabs in the process's memory and slabs that are mapped files (the SSD tier), for the console. */
    synchronized int[] slabCounts() {
        return new int[]{hotCount, coldCount};
    }

    private MemorySegment mapSlab() {
        try {
            java.nio.file.Files.createDirectories(coldDir);
            java.nio.file.Path f = java.nio.file.Files.createTempFile(coldDir, "kvslab", ".bin");
            try (java.nio.channels.FileChannel ch = java.nio.channels.FileChannel.open(f, java.nio.file.StandardOpenOption.READ,
                    java.nio.file.StandardOpenOption.WRITE)) {
                MemorySegment m = ch.map(java.nio.channels.FileChannel.MapMode.READ_WRITE, 0, slabBytes, arena);
                java.nio.file.Files.deleteIfExists(f); // unlinked while mapped: the space goes back with the mapping, even after a crash
                coldCount++;
                return m;
            }
        } catch (java.io.IOException e) {
            System.err.println("warning: no memory-mapped KV slab in " + coldDir + " (" + e.getMessage() + "), using memory");
            hotCount++;
            return arena.allocate(slabBytes, 64);
        }
    }

    /** The slab that holds {@code block}. */
    MemorySegment slab(int block) {
        return slabs[block >> slabShift];
    }

    /** Offset of {@code layer}'s K region of {@code block} inside {@link #slab}; the V region follows at {@link #kBytes}. */
    long base(int block, int layer) {
        return (long) (block & slabMask) * blockBytes + (long) layer * layerBytes;
    }

    @Override
    public void copy(int src, int dst, int rows) {
        MemorySegment.copy(slab(src), (long) (src & slabMask) * blockBytes, slab(dst), (long) (dst & slabMask) * blockBytes,
                blockBytes);
    }

    // ------------------------------------------------------------------ element access (scalar: stores, spills, saved caches)

    /** Offset of key element {@code i} of kv head {@code g} at row {@code t} of the K region (f32: float index, f16: int index, q8: byte). */
    long keyIndex(int g, int i, int t) {
        return switch (dtype) {
            case F32, Q8 -> ((long) g * headDim + i) * B + t;
            default -> ((long) g * pairs + (i >> 1)) * B + t;
        };
    }

    long valueIndex(int t, int e) {
        return switch (dtype) {
            case F16 -> (long) (t >> 1) * kvDim + e;
            default -> (long) t * kvDim + e;
        };
    }

    float keyValue(int layer, int block, int g, int i, int t) {
        MemorySegment seg = slab(block);
        long kb = base(block, layer);
        long ix = keyIndex(g, i, t);
        return switch (dtype) {
            case F32 -> seg.get(ValueLayout.JAVA_FLOAT, kb + ix * 4);
            case F16 -> Float.float16ToFloat(half(seg.get(ValueLayout.JAVA_INT, kb + ix * 4), (i & 1) != 0));
            default -> seg.get(ValueLayout.JAVA_BYTE, kb + ix) * seg.get(ValueLayout.JAVA_FLOAT, kb + kScaleOff + ((long) g * B + t) * 4);
        };
    }

    float valueValue(int layer, int block, int g, int i, int t) {
        MemorySegment seg = slab(block);
        long vb = base(block, layer) + kBytes;
        long ix = valueIndex(t, g * headDim + i);
        return switch (dtype) {
            case F32 -> seg.get(ValueLayout.JAVA_FLOAT, vb + ix * 4);
            case F16 -> Float.float16ToFloat(half(seg.get(ValueLayout.JAVA_INT, vb + ix * 4), (t & 1) != 0));
            default -> seg.get(ValueLayout.JAVA_BYTE, vb + ix) * seg.get(ValueLayout.JAVA_FLOAT, vb + vScaleOff + ((long) t * kvHeads + g) * 4);
        };
    }

    static short half(int word, boolean high) {
        return (short) (high ? word >>> 16 : word);
    }

    private static int withHalf(int word, boolean high, short h) {
        return high ? (word & 0xFFFF) | (h << 16) : (word & 0xFFFF0000) | (h & 0xFFFF);
    }

    /** Stores row {@code t} of block {@code block}, layer {@code layer}: the {@code kvDim} floats {@code k} / {@code v} from {@code src}. */
    void storeRow(int layer, int block, int t, float[] k, float[] v, int src) {
        MemorySegment seg = slab(block);
        long kb = base(block, layer);
        long vb = kb + kBytes;
        switch (dtype) {
            case F32 -> {
                for (int g = 0; g < kvHeads; g++) {
                    for (int i = 0; i < headDim; i++) {
                        seg.set(ValueLayout.JAVA_FLOAT, kb + (((long) g * headDim + i) * B + t) * 4, k[src + g * headDim + i]);
                    }
                }
                for (int e = 0; e < kvDim; e++) {
                    seg.set(ValueLayout.JAVA_FLOAT, vb + ((long) t * kvDim + e) * 4, v[src + e]);
                }
            }
            case F16 -> {
                for (int g = 0; g < kvHeads; g++) {
                    for (int i = 0; i < headDim; i++) {
                        long at = kb + (((long) g * pairs + (i >> 1)) * B + t) * 4;
                        seg.set(ValueLayout.JAVA_INT, at, withHalf(seg.get(ValueLayout.JAVA_INT, at), (i & 1) != 0,
                                Float.floatToFloat16(k[src + g * headDim + i])));
                    }
                }
                for (int e = 0; e < kvDim; e++) {
                    long at = vb + ((long) (t >> 1) * kvDim + e) * 4;
                    seg.set(ValueLayout.JAVA_INT, at, withHalf(seg.get(ValueLayout.JAVA_INT, at), (t & 1) != 0,
                            Float.floatToFloat16(v[src + e])));
                }
            }
            default -> {
                float[] row = rotate ? new float[headDim] : null;
                for (int g = 0; g < kvHeads; g++) {
                    int o = src + g * headDim;
                    float[] kx = k;
                    int ko = o;
                    if (rotate) { // the key row as the attention reads it: rotated, then quantized (as KvCache.Q8.store)
                        System.arraycopy(k, o, row, 0, headDim);
                        org.robomiri.jllm.model.KvQ4.rotate(row, 0, headDim);
                        kx = row;
                        ko = 0;
                    }
                    float ks = KvDtype.q8Scale(maxAbs(kx, ko, headDim));
                    seg.set(ValueLayout.JAVA_FLOAT, kb + kScaleOff + ((long) g * B + t) * 4, ks);
                    for (int i = 0; i < headDim; i++) {
                        seg.set(ValueLayout.JAVA_BYTE, kb + ((long) g * headDim + i) * B + t, KvDtype.q8(kx[ko + i], ks));
                    }
                    float vs = KvDtype.q8Scale(maxAbs(v, o, headDim));
                    seg.set(ValueLayout.JAVA_FLOAT, vb + vScaleOff + ((long) t * kvHeads + g) * 4, vs);
                    for (int i = 0; i < headDim; i++) {
                        seg.set(ValueLayout.JAVA_BYTE, vb + (long) t * kvDim + g * headDim + i, KvDtype.q8(v[o + i], vs));
                    }
                }
            }
        }
    }

    private static float maxAbs(float[] x, int o, int n) {
        float m = 0f;
        for (int i = 0; i < n; i++) {
            m = Math.max(m, Math.abs(x[o + i]));
        }
        return m;
    }

    /** Puts one row (K or V) of block position {@code t} in the saved-cache layout: values as stored, then (q8) the scales. */
    void putRow(int layer, int block, int t, boolean values, ByteBuffer buf) {
        MemorySegment seg = slab(block);
        long kb = base(block, layer);
        long rb = values ? kb + kBytes : kb;
        switch (dtype) {
            case F32 -> {
                for (int g = 0; g < kvHeads; g++) {
                    for (int i = 0; i < headDim; i++) {
                        long ix = values ? valueIndex(t, g * headDim + i) : keyIndex(g, i, t);
                        buf.putFloat(seg.get(ValueLayout.JAVA_FLOAT, rb + ix * 4));
                    }
                }
            }
            case F16 -> {
                for (int g = 0; g < kvHeads; g++) {
                    for (int i = 0; i < headDim; i++) {
                        long ix = values ? valueIndex(t, g * headDim + i) : keyIndex(g, i, t);
                        boolean high = values ? (t & 1) != 0 : (i & 1) != 0;
                        buf.putShort(half(seg.get(ValueLayout.JAVA_INT, rb + ix * 4), high));
                    }
                }
            }
            default -> {
                for (int g = 0; g < kvHeads; g++) {
                    for (int i = 0; i < headDim; i++) {
                        long ix = values ? valueIndex(t, g * headDim + i) : keyIndex(g, i, t);
                        buf.put(seg.get(ValueLayout.JAVA_BYTE, rb + ix));
                    }
                }
                for (int g = 0; g < kvHeads; g++) {
                    buf.putFloat(seg.get(ValueLayout.JAVA_FLOAT, values ? rb + vScaleOff + ((long) t * kvHeads + g) * 4
                            : rb + kScaleOff + ((long) g * B + t) * 4));
                }
            }
        }
    }

    /** Reads what {@link #putRow} wrote. */
    void getRow(int layer, int block, int t, boolean values, ByteBuffer buf) {
        MemorySegment seg = slab(block);
        long kb = base(block, layer);
        long rb = values ? kb + kBytes : kb;
        switch (dtype) {
            case F32 -> {
                for (int g = 0; g < kvHeads; g++) {
                    for (int i = 0; i < headDim; i++) {
                        long ix = values ? valueIndex(t, g * headDim + i) : keyIndex(g, i, t);
                        seg.set(ValueLayout.JAVA_FLOAT, rb + ix * 4, buf.getFloat());
                    }
                }
            }
            case F16 -> {
                for (int g = 0; g < kvHeads; g++) {
                    for (int i = 0; i < headDim; i++) {
                        long ix = values ? valueIndex(t, g * headDim + i) : keyIndex(g, i, t);
                        boolean high = values ? (t & 1) != 0 : (i & 1) != 0;
                        long at = rb + ix * 4;
                        seg.set(ValueLayout.JAVA_INT, at, withHalf(seg.get(ValueLayout.JAVA_INT, at), high, buf.getShort()));
                    }
                }
            }
            default -> {
                for (int g = 0; g < kvHeads; g++) {
                    for (int i = 0; i < headDim; i++) {
                        long ix = values ? valueIndex(t, g * headDim + i) : keyIndex(g, i, t);
                        seg.set(ValueLayout.JAVA_BYTE, rb + ix, buf.get());
                    }
                }
                for (int g = 0; g < kvHeads; g++) {
                    seg.set(ValueLayout.JAVA_FLOAT, values ? rb + vScaleOff + ((long) t * kvHeads + g) * 4
                            : rb + kScaleOff + ((long) g * B + t) * 4, buf.getFloat());
                }
            }
        }
    }

    @Override
    public void save(int block, ByteBuffer dst) {
        ByteBuffer b = dst.order(ByteOrder.LITTLE_ENDIAN);
        for (int layer = 0; layer < layers; layer++) {
            for (int t = 0; t < B; t++) {
                putRow(layer, block, t, false, b);
                putRow(layer, block, t, true, b);
            }
        }
    }

    @Override
    public void load(int block, ByteBuffer src) {
        ByteBuffer b = src.order(ByteOrder.LITTLE_ENDIAN);
        for (int layer = 0; layer < layers; layer++) {
            for (int t = 0; t < B; t++) {
                getRow(layer, block, t, false, b);
                getRow(layer, block, t, true, b);
            }
        }
    }

    @Override
    public void close() {
        slabs = new MemorySegment[0]; // the memory is freed when the arena is collected
    }
}
