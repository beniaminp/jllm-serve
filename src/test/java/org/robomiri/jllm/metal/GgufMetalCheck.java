package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

/**
 * The Metal kernels of the GGUF block types (Q4_0, Q8_0, Q4_K, Q5_K) against a double-precision reference built
 * from the {@code gguf.quants} fixtures (see {@link org.robomiri.jllm.cpu.GgufKernelsCheck}): every dispatch family
 * that has a per-type instantiation is exercised on weights at an odd byte offset of a native segment (so the
 * page-range wrapping and the buffer offsets are covered): the single-token and multi-token {@code matmul}, the
 * tiled prefill {@code matmul_pf}, {@code matmul3} (single-row and tiled), {@code matmul_swiglu} (single-row and
 * tiled) and the MoE {@code moe_up} / {@code moe_down}. Skipped when Metal is not available.
 */
public final class GgufMetalCheck {

    static int passed;
    static int failed;

    private static final int ROWS = 4;
    private static final int COLS = 512;

    static void check(boolean ok, String what) {
        if (ok) {
            passed++;
        } else {
            failed++;
            System.out.println("FAIL: " + what);
        }
    }

    private static byte[] resource(String name) throws IOException {
        try (InputStream in = GgufMetalCheck.class.getResourceAsStream("/gguf/" + name)) {
            if (in == null) {
                throw new IOException("missing test resource /gguf/" + name);
            }
            return in.readAllBytes();
        }
    }

    private static float[] floats(byte[] b) {
        float[] f = new float[b.length / 4];
        ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(f);
        return f;
    }

    private static float[] read(Metal.Buffer b, int n) {
        float[] out = new float[n];
        MemorySegment.copy(b.contents(), ValueLayout.JAVA_FLOAT, 0, out, 0, n);
        return out;
    }

    public static void main(String[] args) throws Exception {
        String lib = System.getProperty("jllm.metal.lib");
        if (!Metal.available(lib)) {
            System.out.println("gguf metal kernels: skipped (Metal is not available)");
            return;
        }
        try (Arena a = Arena.ofShared(); MetalKernels k = new MetalKernels(Metal.create(lib), 1e-6f)) {
            run(a, k, "Q4_0", QType.Q4_0G);
            run(a, k, "Q8_0", QType.Q8_0G);
            run(a, k, "Q4_K", QType.Q4_K);
            run(a, k, "Q5_K", QType.Q5_K);
            mixed(a, k);
        }
        System.out.println("gguf metal kernels: " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    private static void run(Arena a, MetalKernels k, String file, QType qt) throws IOException {
        byte[] blocks = resource(file + ".blocks");
        float[] ref = floats(resource(file + ".ref"));
        // 16 KiB pages plus an odd 32-byte-aligned offset: the tensor is a slice inside a larger mapping
        MemorySegment big = a.allocate(1 << 20, 16384).asSlice(0, 1 << 20);
        MemorySegment seg = big.asSlice(16384 + 96, blocks.length);
        MemorySegment.copy(blocks, 0, seg, ValueLayout.JAVA_BYTE, 0, blocks.length);
        QTensor t = QTensor.mapped(qt, ROWS, COLS, seg);
        Metal.Buffer w = k.wrap(seg);
        check(w.offset() == 96, file + ": slice offset inside the wrapped pages is " + w.offset());
        Random rnd = new Random(5);
        for (int n : new int[]{1, 3, 9}) { // decode, a few tokens, a prefill chunk (tiled kernels from 8 tokens)
            float[] x = new float[n * COLS];
            for (int i = 0; i < x.length; i++) {
                x[i] = rnd.nextFloat() * 2 - 1;
            }
            double[][] y = new double[n][ROWS];
            double[][] mag = new double[n][ROWS];
            for (int tok = 0; tok < n; tok++) {
                for (int r = 0; r < ROWS; r++) {
                    for (int c = 0; c < COLS; c++) {
                        double p = (double) ref[r * COLS + c] * x[tok * COLS + c];
                        y[tok][r] += p;
                        mag[tok][r] += Math.abs(p);
                    }
                }
            }
            Metal.Buffer in = k.upload(x);
            String tag = file + " n=" + n;

            Metal.Buffer out = k.scratch((long) n * ROWS * 4);
            k.begin();
            k.matmul(t, w, in, 0, out, 0, n, false, null, 0f);
            k.commitAndWait();
            compare(read(out, n * ROWS), y, mag, ROWS, tag + " matmul");

            if (n == 1) {
                Metal.Buffer nw = k.upload(ones(COLS));
                // fused rmsnorm variant: reference is the matmul of the normalised row
                double ss = 0;
                for (float v : x) {
                    ss += (double) v * v;
                }
                double scale = 1.0 / Math.sqrt(ss / COLS + 1e-6);
                double[][] yn = new double[1][ROWS];
                double[][] mn = new double[1][ROWS];
                for (int r = 0; r < ROWS; r++) {
                    for (int c = 0; c < COLS; c++) {
                        double p = (double) ref[r * COLS + c] * x[c] * scale;
                        yn[0][r] += p;
                        mn[0][r] += Math.abs(p);
                    }
                }
                Metal.Buffer outn = k.scratch(ROWS * 4L);
                k.begin();
                k.matmul(t, w, in, 0, outn, 0, 1, false, nw, 0f);
                k.commitAndWait();
                compare(read(outn, ROWS), yn, mn, ROWS, tag + " matmul+rmsnorm");
            }

            Metal.Buffer q = k.scratch((long) n * ROWS * 4);
            Metal.Buffer kk = k.scratch((long) n * ROWS * 4);
            Metal.Buffer v = k.scratch((long) n * ROWS * 4);
            k.begin();
            k.matmul3(t, w, t, w, t, w, in, null, q, kk, 0, v, 0, false, null, null, null, n);
            k.commitAndWait();
            compare(read(q, n * ROWS), y, mag, ROWS, tag + " matmul3 q");
            compare(read(kk, n * ROWS), y, mag, ROWS, tag + " matmul3 k");
            compare(read(v, n * ROWS), y, mag, ROWS, tag + " matmul3 v");

            // swiglu: hidden = 2, gate rows 0..1, up rows 2..3
            int hidden = ROWS / 2;
            double[][] ys = new double[n][hidden];
            double[][] ms = new double[n][hidden];
            for (int tok = 0; tok < n; tok++) {
                for (int r = 0; r < hidden; r++) {
                    double g = y[tok][r];
                    ys[tok][r] = g / (1 + Math.exp(-g)) * y[tok][hidden + r];
                    ms[tok][r] = Math.abs(ys[tok][r]) + 1e-3 * (mag[tok][r] + mag[tok][hidden + r]);
                }
            }
            Metal.Buffer sw = k.scratch((long) n * hidden * 4);
            k.begin();
            k.swiglu(t, w, in, null, sw, false, n);
            k.commitAndWait();
            compare(read(sw, n * hidden), ys, ms, hidden, tag + " swiglu");

            // MoE: one expert (assign = 0) whose stacked gate|up tensor is t: moe_up == swiglu of the same rows
            Metal.Buffer assign = k.scratch((long) n * 4);
            for (int i = 0; i < n; i++) {
                assign.contents().setAtIndex(ValueLayout.JAVA_INT, i, 0);
            }
            Metal.Buffer up = k.scratch((long) n * hidden * 4);
            k.begin();
            k.moeUp(t, w, in, assign, up, hidden, 1, n);
            k.commitAndWait();
            compare(read(up, n * hidden), ys, ms, hidden, tag + " moe_up");

            // moe_down with t as the [dim = 4, moeDim = 512] down matrix of expert 0 and weight 1: x += W h
            float[] zeros = new float[n * ROWS];
            Metal.Buffer xres = k.upload(zeros);
            float[] rwv = new float[n];
            java.util.Arrays.fill(rwv, 1f);
            Metal.Buffer rw = k.upload(rwv);
            k.begin();
            k.moeDown(t, w, in, assign, rw, xres, ROWS, 1, n);
            k.commitAndWait();
            compare(read(xres, n * ROWS), y, mag, ROWS, tag + " moe_down");
        }
    }

    /** q, k and v of different types (a Q4_K_M file has v in Q6_K in half the layers): one dispatch per projection. */
    private static void mixed(Arena a, MetalKernels k) throws IOException {
        String[] files = {"Q4_K", "Q8_0", "Q5_K"};
        QType[] types = {QType.Q4_K, QType.Q8_0G, QType.Q5_K};
        QTensor[] t = new QTensor[3];
        Metal.Buffer[] w = new Metal.Buffer[3];
        float[][] ref = new float[3][];
        for (int i = 0; i < 3; i++) {
            byte[] blocks = resource(files[i] + ".blocks");
            MemorySegment seg = a.allocate(blocks.length + 64, 32).asSlice(32, blocks.length);
            MemorySegment.copy(blocks, 0, seg, ValueLayout.JAVA_BYTE, 0, blocks.length);
            t[i] = QTensor.mapped(types[i], ROWS, COLS, seg);
            w[i] = k.wrap(seg);
            ref[i] = floats(resource(files[i] + ".ref"));
        }
        Random rnd = new Random(11);
        float[] bias = {0.5f, -1f, 2f, 0.25f};
        Metal.Buffer[] bb = {k.upload(bias), k.upload(bias), k.upload(bias)};
        for (int n : new int[]{1, 9}) {
            for (boolean withBias : new boolean[]{false, true}) {
                float[] x = new float[n * COLS];
                for (int i = 0; i < x.length; i++) {
                    x[i] = rnd.nextFloat() * 2 - 1;
                }
                Metal.Buffer in = k.upload(x);
                Metal.Buffer[] out = {k.scratch((long) n * ROWS * 4), k.scratch((long) n * ROWS * 4), k.scratch((long) n * ROWS * 4)};
                k.begin();
                k.matmul3(t[0], w[0], t[1], w[1], t[2], w[2], in, null, out[0], out[1], 0, out[2], 0, false,
                        withBias ? bb[0] : null, withBias ? bb[1] : null, withBias ? bb[2] : null, n);
                k.commitAndWait();
                for (int i = 0; i < 3; i++) {
                    double[][] y = new double[n][ROWS];
                    double[][] mag = new double[n][ROWS];
                    for (int tok = 0; tok < n; tok++) {
                        for (int r = 0; r < ROWS; r++) {
                            for (int c = 0; c < COLS; c++) {
                                double p = (double) ref[i][r * COLS + c] * x[tok * COLS + c];
                                y[tok][r] += p;
                                mag[tok][r] += Math.abs(p);
                            }
                            if (withBias) {
                                y[tok][r] += bias[r];
                            }
                        }
                    }
                    compare(read(out[i], n * ROWS), y, mag, ROWS, "mixed-type matmul3 n=" + n + " bias=" + withBias + " " + files[i]);
                }
            }
        }
    }

    private static float[] ones(int n) {
        float[] f = new float[n];
        java.util.Arrays.fill(f, 1f);
        return f;
    }

    private static void compare(float[] got, double[][] want, double[][] mag, int width, String what) {
        double worst = 0;
        for (int t = 0; t < want.length; t++) {
            for (int r = 0; r < width; r++) {
                worst = Math.max(worst, Math.abs(got[t * width + r] - want[t][r]) / (mag[t][r] + 1e-6));
            }
        }
        check(worst < 5e-5, what + ": max error relative to sum|w x| is " + worst);
    }
}
