package org.robomiri.jllm.metal;

import org.robomiri.jllm.gguf.GgmlType;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

/**
 * Timings of the Metal matmul kernels of the GGUF block types on one random {@code rows × cols} matrix (best of 9
 * trials of 20 dispatches in one command buffer): the single-token decode kernel ({@code n = 1}) and the matrix-unit
 * prefill kernel ({@code n = 64}). Weights are streamed from a mapped-style segment, so the decode number is the
 * kernel's weight bandwidth. Run with the heavy lock; the GPU must be idle.
 *
 * <pre>java ... -Djllm.metal.lib=target/native/libjllm_metal.dylib org.robomiri.jllm.metal.GgufMetalBench [rows cols] TYPE...</pre>
 * (TYPE: a QType name, e.g. Q4_0G Q4_1 Q5_0 IQ4_NL IQ4_XS MXFP4 Q3_K Q4_K Q8_0G F16)
 */
public final class GgufMetalBench {

    private static GgmlType ggml(QType t) {
        return switch (t) {
            case Q4_0G -> GgmlType.Q4_0;
            case Q8_0G -> GgmlType.Q8_0;
            default -> GgmlType.valueOf(t.name());
        };
    }

    /** Random blocks with sane f16 scales (about +-0.1 weights). */
    private static byte[] blocks(Random r, QType t, long elements) {
        int nb = (int) (elements / t.blockSize);
        byte[] b = new byte[nb * t.blockBytes];
        r.nextBytes(b);
        ByteBuffer bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < nb; i++) {
            int o = i * t.blockBytes;
            switch (t) {
                case F16, BF16 -> {
                    for (int j = 0; j < 32; j++) {
                        float v = (float) (0.1 * r.nextGaussian());
                        bb.putShort(o + 2 * j, t == QType.F16 ? Float.floatToFloat16(v) : (short) (Float.floatToRawIntBits(v) >>> 16));
                    }
                }
                case MXFP4 -> b[o] = (byte) (116 + r.nextInt(5));
                case Q3_K -> bb.putShort(o + 108, Float.floatToFloat16(0.00004f + 0.00012f * r.nextFloat()));
                case Q6_K -> bb.putShort(o + 208, Float.floatToFloat16(0.00001f + 0.00004f * r.nextFloat()));
                case Q4_0G, Q5_0, Q8_0G, IQ4_NL -> bb.putShort(o, Float.floatToFloat16(0.001f + 0.004f * r.nextFloat()));
                default -> {
                    bb.putShort(o, Float.floatToFloat16(0.00004f + 0.00012f * r.nextFloat()));
                    bb.putShort(o + 2, Float.floatToFloat16(0.00004f + 0.00012f * r.nextFloat()));
                }
            }
        }
        return b;
    }

    public static void main(String[] args) {
        String lib = System.getProperty("jllm.metal.lib");
        if (!Metal.available(lib)) {
            System.out.println("skipped (Metal is not available)");
            return;
        }
        int a = 0;
        int rows = 8192;
        int cols = 4096;
        if (args.length >= 2 && args[0].matches("\\d+")) {
            rows = Integer.parseInt(args[0]);
            cols = Integer.parseInt(args[1]);
            a = 2;
        }
        Random rnd = new Random(1);
        float[] x = new float[64 * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 2 - 1;
        }
        try (Arena arena = Arena.ofShared(); MetalKernels k = new MetalKernels(Metal.create(lib), 1e-6f)) {
            Metal.Buffer in = k.upload(x);
            Metal.Buffer out = k.scratch(64L * rows * 4);
            for (int i = a; i < args.length; i++) {
                QType qt = QType.valueOf(args[i]);
                byte[] blk = blocks(rnd, qt, (long) rows * cols);
                MemorySegment seg = arena.allocate(blk.length + 16384, 16384).asSlice(96, blk.length);
                MemorySegment.copy(blk, 0, seg, ValueLayout.JAVA_BYTE, 0, blk.length);
                QTensor t = QTensor.mapped(qt, rows, cols, seg);
                Metal.Buffer w = k.wrap(seg);
                StringBuilder line = new StringBuilder(String.format("%-7s %dx%d %6.1f MB", qt, rows, cols, blk.length / 1e6));
                for (int n : new int[]{1, Integer.getInteger("n", 64)}) {
                    double best = Double.MAX_VALUE;
                    for (int trial = 0; trial < 9; trial++) {
                        k.begin();
                        long t0 = System.nanoTime();
                        for (int rep = 0; rep < 20; rep++) {
                            k.matmul(t, w, in, 0, out, 0, n, false, null, 0f);
                        }
                        k.commitAndWait();
                        best = Math.min(best, (System.nanoTime() - t0) / 20.0);
                    }
                    line.append(String.format("   n=%-2d %8.0f us (%5.1f GB/s, %5.2f ns per 1000 weights)", n, best / 1e3,
                            blk.length / best, best / ((double) rows * cols) * 1000));
                }
                // the fused families at n = 64: gate|up as a split pair (matmul_swiglu) and q|k|v (matmul3, three tensors of rows / 2 rows)
                int hr = rows / 2;
                long half = (long) hr * (cols / qt.blockSize) * qt.blockBytes;
                QTensor gate = QTensor.mapped(qt, hr, cols, seg.asSlice(0, half));
                QTensor up = QTensor.mapped(qt, hr, cols, seg.asSlice(half, half));
                QTensor gu = QTensor.split(gate, up, 0);
                Metal.Buffer wg = k.wrap(gate.data);
                Metal.Buffer sw = k.scratch(64L * hr * 4);
                Metal.Buffer o2 = k.scratch(64L * hr * 4);
                Metal.Buffer o3 = k.scratch(64L * hr * 4);
                double bs = Double.MAX_VALUE;
                double b3 = Double.MAX_VALUE;
                for (int trial = 0; trial < 9; trial++) {
                    k.begin();
                    long t0 = System.nanoTime();
                    for (int rep = 0; rep < 20; rep++) {
                        k.swiglu(gu, wg, in, null, sw, false, 64);
                    }
                    k.commitAndWait();
                    bs = Math.min(bs, (System.nanoTime() - t0) / 20.0);
                    k.begin();
                    t0 = System.nanoTime();
                    for (int rep = 0; rep < 20; rep++) {
                        k.matmul3(gate, wg, gate, wg, gate, wg, in, null, sw, o2, 0, o3, 0, false, null, null, null, 64);
                    }
                    k.commitAndWait();
                    b3 = Math.min(b3, (System.nanoTime() - t0) / 20.0);
                }
                line.append(String.format("   swiglu n=64 %7.0f us   matmul3 n=64 %7.0f us", bs / 1e3, b3 / 1e3));
                System.out.println(line);
            }
        }
    }
}
