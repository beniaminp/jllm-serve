package org.robomiri.jllm.vision;

import org.robomiri.jllm.gguf.GgmlCodec;
import org.robomiri.jllm.gguf.GgmlType;
import org.robomiri.jllm.gguf.GgufFile;

import java.lang.foreign.MemorySegment;

/**
 * A tensor of a projector file seen as a row-major matrix {@code [rows][cols]} ({@code cols} = ggml's {@code ne[0]},
 * the contiguous dimension): F32, F16, BF16 or any of the block formats the loader can dequantize. The data stays in
 * the memory-mapped file; rows are converted to float on demand ({@link #rows}), so a 1.3 GB F16 projector does not
 * become 2.7 GB of floats.
 */
public final class Mat {

    public final String name;
    public final int rows;
    public final int cols;
    private final GgmlType type;
    private final MemorySegment seg;
    private final long rowBytes;

    Mat(GgufFile f, GgufFile.Tensor t) {
        this(f, t, Math.toIntExact(t.rows()), Math.toIntExact(t.cols()));
    }

    /** The tensor's elements viewed as {@code rows} rows of {@code cols} (a conv kernel {@code [out][c][ky][kx]} as a matrix). */
    Mat(GgufFile f, GgufFile.Tensor t, int rows, int cols) {
        if (!GgmlCodec.canDequantize(t.type())) {
            throw new IllegalArgumentException(t + ": ggml type " + t.type() + " is not supported in a vision projector");
        }
        if ((long) rows * cols != t.elements() || cols % t.type().blockSize != 0) {
            throw new IllegalArgumentException(t + " cannot be viewed as " + rows + " x " + cols);
        }
        this.name = t.name();
        this.cols = cols;
        this.rows = rows;
        this.type = t.type();
        this.seg = f.data(t);
        this.rowBytes = t.type().rowBytes(cols);
    }

    /** Rows {@code [r0, r0 + n)} as floats into {@code dst}, row stride {@code stride} (>= cols); the padding is zeroed. */
    public void rows(int r0, int n, float[] dst, int stride) {
        for (int i = 0; i < n; i++) {
            GgmlCodec.dequantizeRow(type, seg, (long) (r0 + i) * rowBytes, cols, dst, i * stride);
            if (stride > cols) {
                java.util.Arrays.fill(dst, i * stride + cols, (i + 1) * stride, 0f);
            }
        }
    }

    /** The whole matrix as floats ({@code rows * cols}). */
    public float[] toFloats() {
        float[] out = new float[Math.toIntExact((long) rows * cols)];
        rows(0, rows, out, cols);
        return out;
    }

    public GgmlType type() {
        return type;
    }

    /** The mapped bytes of the whole matrix (rows contiguous in the file's own type); for zero-copy use on the GPU. */
    public MemorySegment segment() {
        return seg;
    }

    @Override
    public String toString() {
        return name + " " + type + " [" + rows + " x " + cols + "]";
    }
}
