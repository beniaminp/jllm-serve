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
        return cols / QType.BLOCK;
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
        return align((long) rows * (cols / QType.BLOCK) * type.blockBytes, 64);
    }

    /** Total bytes of the segment holding a {@code rows × cols} tensor of {@code type}. */
    public static long byteSize(QType type, int rows, int cols) {
        return align(scalesOffset(type, rows, cols) + (long) rows * (cols / QType.BLOCK) * Float.BYTES, PAGE);
    }

    /** Wraps an existing segment in the layout produced by {@link #quantize} (e.g. a memory-mapped file). */
    public static QTensor view(QType type, int rows, int cols, MemorySegment data) {
        if (data.byteSize() != byteSize(type, rows, cols)) {
            throw new IllegalArgumentException("segment size does not match " + rows + "x" + cols + " " + type);
        }
        return new QTensor(type, rows, cols, data);
    }

    public static QTensor quantize(Arena arena, QType type, int rows, int cols, RowSource src) {
        if (cols % QType.BLOCK != 0) {
            throw new IllegalArgumentException("cols " + cols + " is not a multiple of " + QType.BLOCK);
        }
        MemorySegment seg = arena.allocate(byteSize(type, rows, cols), PAGE);
        QTensor t = new QTensor(type, rows, cols, seg);
        int nb = t.blocksPerRow();
        ThreadLocal<float[]> rowBuf = ThreadLocal.withInitial(() -> new float[cols]);
        IntStream.range(0, rows).parallel().forEach(r -> {
            float[] row = rowBuf.get();
            src.row(r, row);
            for (int b = 0; b < nb; b++) {
                if (type == QType.Q4_0) {
                    t.quantizeBlockQ4(r, b, row, b * QType.BLOCK);
                } else {
                    t.quantizeBlockQ8(r, b, row, b * QType.BLOCK);
                }
            }
        });
        return t;
    }

    /** Same scheme as llama.cpp Q4_0: the largest-magnitude value maps to -8. */
    private void quantizeBlockQ4(int r, int b, float[] x, int off) {
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
        long q = rowOffset(r) + (long) b * 16;
        for (int j = 0; j < 16; j++) {
            int lo = Math.min(15, (int) (x[off + j] * id + 8.5f));
            int hi = Math.min(15, (int) (x[off + j + 16] * id + 8.5f));
            data.set(ValueLayout.JAVA_BYTE, q + j, (byte) (lo | (hi << 4)));
        }
        data.set(ValueLayout.JAVA_FLOAT, scaleOffset(r) + (long) b * 4, d);
    }

    private void quantizeBlockQ8(int r, int b, float[] x, int off) {
        float amax = 0;
        for (int j = 0; j < QType.BLOCK; j++) {
            amax = Math.max(amax, Math.abs(x[off + j]));
        }
        float d = amax / 127f;
        float id = d != 0 ? 1f / d : 0f;
        long q = rowOffset(r) + (long) b * 32;
        for (int j = 0; j < QType.BLOCK; j++) {
            data.set(ValueLayout.JAVA_BYTE, q + j, (byte) Math.round(x[off + j] * id));
        }
        data.set(ValueLayout.JAVA_FLOAT, scaleOffset(r) + (long) b * 4, d);
    }

    /** Dequantizes row {@code r} into {@code dst[off .. off + cols)}. */
    public void dequantizeRow(int r, float[] dst, int off) {
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
