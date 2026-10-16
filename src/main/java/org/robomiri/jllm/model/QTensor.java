package org.robomiri.jllm.model;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.stream.IntStream;

/**
 * A block-quantized 2-D weight matrix {@code [rows, cols]} in off-heap (or memory-mapped) memory.
 *
 * <p>Layout of {@link #data}: all quantized blocks row-major, then (64-byte aligned) all f32 block scales,
 * padded to a multiple of {@link #PAGE}. The segment is page aligned so the Metal backend can wrap it without
 * copying (Apple Silicon has unified memory), and so it can be stored verbatim in a {@link QuantFile}.
 *
 * <p>Several logical matrices can be stacked in one tensor (e.g. all experts of a MoE layer): rows are
 * contiguous, so sub-matrix {@code e} with {@code R} rows is simply rows {@code [e*R, (e+1)*R)}.
 */
public final class QTensor {

    /** Apple Silicon uses 16 KiB pages; 16 KiB alignment is also fine everywhere else. */
    public static final long PAGE = 16384;

    /** Supplies one row of the source matrix as floats. */
    @FunctionalInterface
    public interface RowSource {
        void row(int r, float[] dst);
    }

    public final QType type;
    public final int rows;
    public final int cols;
    public final MemorySegment data;
    public final long scalesOffset;

    private QTensor(QType type, int rows, int cols, MemorySegment data) {
        this.type = type;
        this.rows = rows;
        this.cols = cols;
        this.data = data;
        this.scalesOffset = scalesOffset(type, rows, cols);
    }

    public int blocksPerRow() {
        return blocksPerRow(type, cols);
    }

    /** Blocks per row; a Q2_K row that is not a multiple of 256 ends with a partial super-block. */
    private static int blocksPerRow(QType type, int cols) {
        return (cols + type.blockSize - 1) / type.blockSize;
    }

    /** Byte offset of row {@code r}'s first quantized block. */
    public long rowOffset(int r) {
        return (long) r * blocksPerRow() * type.blockBytes;
    }

    /** Byte offset of row {@code r}'s first scale. */
    public long scaleOffset(int r) {
        return scalesOffset + (long) r * blocksPerRow() * Float.BYTES;
    }

    public static long scalesOffset(QType type, int rows, int cols) {
        return align((long) rows * blocksPerRow(type, cols) * type.blockBytes, 64);
    }

    /** Total bytes of the segment holding a {@code rows × cols} tensor of {@code type}. */
    public static long byteSize(QType type, int rows, int cols) {
        return align(scalesOffset(type, rows, cols) + (long) rows * blocksPerRow(type, cols) * type.scaleBytes, PAGE);
    }

    /** Wraps an existing segment in the layout produced by {@link #quantize} (e.g. a memory-mapped file). */
    public static QTensor view(QType type, int rows, int cols, MemorySegment data) {
        if (data.byteSize() != byteSize(type, rows, cols)) {
            throw new IllegalArgumentException("segment size does not match " + rows + "x" + cols + " " + type);
        }
        return new QTensor(type, rows, cols, data);
    }

    /**
     * Wraps the bytes of a ggml tensor in place: {@code rows} rows of {@code cols / blockSize} contiguous blocks of
     * {@code type} (a {@link QType#gguf} type or {@link QType#Q2_K}, whose block layout is llama.cpp's), exactly
     * {@code rows * cols / blockSize * blockBytes} bytes, with no padding. The segment need not be page aligned.
     */
    public static QTensor mapped(QType type, int rows, int cols, MemorySegment data) {
        if (type.scaleBytes != 0 || cols % type.blockSize != 0) {
            throw new IllegalArgumentException(type + " cannot view a " + rows + "x" + cols + " ggml tensor");
        }
        long expect = (long) rows * (cols / type.blockSize) * type.blockBytes;
        if (data.byteSize() != expect) {
            throw new IllegalArgumentException("segment of " + data.byteSize() + " bytes does not match " + rows + "x"
                    + cols + " " + type + " (" + expect + ")");
        }
        return new QTensor(type, rows, cols, data);
    }

    /** True if this tensor is a view of a GGUF file's bytes (not the engine's page-padded layout). */
    public boolean isMapped() {
        return type.gguf;
    }

    /**
     * Per-thread scratch for {@link #quantize}: each row is quantized into on-heap arrays and copied out with
     * two bulk copies, instead of one bounds-checked segment access per quantized byte. The arrays are grown on
     * demand and kept, so quantizing a model's thousands of tensors allocates them once per worker thread.
     */
    private static final ThreadLocal<float[]> ROW_BUF = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<byte[]> Q_BUF = ThreadLocal.withInitial(() -> new byte[0]);
    private static final ThreadLocal<float[]> SCALE_BUF = ThreadLocal.withInitial(() -> new float[0]);

    private static float[] floats(ThreadLocal<float[]> tl, int n) {
        float[] a = tl.get();
        if (a.length < n) {
            a = new float[n];
            tl.set(a);
        }
        return a;
    }

    private static byte[] bytes(ThreadLocal<byte[]> tl, int n) {
        byte[] a = tl.get();
        if (a.length < n) {
            a = new byte[n];
            tl.set(a);
        }
        return a;
    }

    public static QTensor quantize(Arena arena, QType type, int rows, int cols, RowSource src) {
        if (type.gguf) {
            throw new IllegalArgumentException(type + " is a view of GGUF data; the engine does not quantize to it");
        }
        if (cols % QType.BLOCK != 0) {
            throw new IllegalArgumentException("cols " + cols + " is not a multiple of " + QType.BLOCK);
        }
        MemorySegment seg = arena.allocate(byteSize(type, rows, cols), PAGE);
        QTensor t = new QTensor(type, rows, cols, seg);
        int nb = t.blocksPerRow();
        int rowBytes = nb * type.blockBytes;
        IntStream.range(0, rows).parallel().forEach(r -> {
            float[] row = floats(ROW_BUF, cols);
            byte[] q = bytes(Q_BUF, rowBytes);
            float[] s = floats(SCALE_BUF, nb);
            src.row(r, row);
            if (type == QType.Q2_K) {
                for (int b = 0; b < nb; b++) {
                    int off = b * Q2K.BLOCK;
                    Q2K.quantizeBlock(row, off, Math.min(Q2K.BLOCK, cols - off), q, b * Q2K.BLOCK_BYTES);
                }
            } else if (type == QType.Q4_0) {
                for (int b = 0; b < nb; b++) {
                    s[b] = quantizeBlockQ4(row, b * QType.BLOCK, q, b * 16);
                }
            } else {
                for (int b = 0; b < nb; b++) {
                    s[b] = quantizeBlockQ8(row, b * QType.BLOCK, q, b * 32);
                }
            }
            MemorySegment.copy(q, 0, seg, ValueLayout.JAVA_BYTE, t.rowOffset(r), rowBytes);
            if (type.scaleBytes > 0) {
                MemorySegment.copy(s, 0, seg, ValueLayout.JAVA_FLOAT, t.scaleOffset(r), nb);
            }
        });
        return t;
    }

    /**
     * Same scheme as llama.cpp Q4_0: the largest-magnitude value maps to -8. Writes 16 bytes at
     * {@code q[qOff..]} and returns the block scale.
     */
    private static float quantizeBlockQ4(float[] x, int off, byte[] q, int qOff) {
        float amax = 0;
        float max = 0;
        for (int j = 0; j < QType.BLOCK; j++) {
            float v = x[off + j];
            if (Math.abs(v) > amax) {
                amax = Math.abs(v);
                max = v;
            }
        }
        float d = max / -8f;
        float id = d != 0 ? 1f / d : 0f;
        for (int j = 0; j < 16; j++) {
            int lo = Math.min(15, (int) (x[off + j] * id + 8.5f));
            int hi = Math.min(15, (int) (x[off + j + 16] * id + 8.5f));
            q[qOff + j] = (byte) (lo | (hi << 4));
        }
        return d;
    }

    /**
     * Writes 32 signed bytes at {@code q[qOff..]} and returns the block scale.
     *
     * <p>Same scheme as llama.cpp Q8_0 except for the rounding of exact halves: {@code Math.round} rounds them
     * up ({@code -2.5 -> -2}), C's {@code roundf} away from zero ({@code -2.5 -> -3}). It only affects weights
     * whose scaled value is exactly {@code k + 0.5} in float, so quality is unchanged, but Q8_0 bytes are not
     * guaranteed to match llama.cpp's; changing it would alter every existing cache.
     */
    private static float quantizeBlockQ8(float[] x, int off, byte[] q, int qOff) {
        float amax = 0;
        for (int j = 0; j < QType.BLOCK; j++) {
            amax = Math.max(amax, Math.abs(x[off + j]));
        }
        float d = amax / 127f;
        float id = d != 0 ? 1f / d : 0f;
        for (int j = 0; j < QType.BLOCK; j++) {
            q[qOff + j] = (byte) Math.round(x[off + j] * id);
        }
        return d;
    }

    /** Dequantizes row {@code r} into {@code dst[off .. off + cols)}. */
    public void dequantizeRow(int r, float[] dst, int off) {
        if (type == QType.Q2_K) {
            Q2K.dequantizeRow(data, rowOffset(r), cols, dst, off);
            return;
        }
        if (type.gguf) {
            long row = rowOffset(r);
            switch (type) {
                case Q4_0G -> Ggml.dequantQ4_0(data, row, cols, dst, off);
                case Q8_0G -> Ggml.dequantQ8_0(data, row, cols, dst, off);
                case Q4_K -> Ggml.dequantQ4_K(data, row, cols, dst, off);
                case Q5_K -> Ggml.dequantQ5_K(data, row, cols, dst, off);
                default -> throw new IllegalStateException(type.name());
            }
            return;
        }
        int nb = blocksPerRow();
        long q = rowOffset(r);
        long s = scaleOffset(r);
        for (int b = 0; b < nb; b++) {
            float d = data.get(ValueLayout.JAVA_FLOAT, s + (long) b * 4);
            int o = off + b * QType.BLOCK;
            if (type == QType.Q4_0) {
                long base = q + (long) b * 16;
                for (int j = 0; j < 16; j++) {
                    int v = data.get(ValueLayout.JAVA_BYTE, base + j) & 0xFF;
                    dst[o + j] = ((v & 0xF) - 8) * d;
                    dst[o + j + 16] = ((v >>> 4) - 8) * d;
                }
            } else {
                long base = q + (long) b * 32;
                for (int j = 0; j < QType.BLOCK; j++) {
                    dst[o + j] = data.get(ValueLayout.JAVA_BYTE, base + j) * d;
                }
            }
        }
    }

    public long byteSize() {
        return data.byteSize();
    }

    static long align(long v, long a) {
        return (v + a - 1) / a * a;
    }
}
