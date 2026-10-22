package org.robomiri.jllm.vision;

import org.robomiri.jllm.metal.Metal;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Manual micro-benchmark of {@code vit_gemm}: {@code GemmBench <tokens m> <columns n> <k> [kernel] [reps]}. Prints the
 * GPU time per dispatch (reps dispatches in one command buffer) and the rate.
 */
public final class GemmBench {

    private GemmBench() {
    }

    public static void main(String[] a) {
        int m = Integer.parseInt(a[0]);
        int n = Integer.parseInt(a[1]);
        int k = Integer.parseInt(a[2]);
        String kernel = a.length > 3 ? a[3] : "vit_gemm";
        int reps = a.length > 4 ? Integer.parseInt(a[4]) : 20;
        Metal metal = Metal.create(System.getProperty("jllm.metal.lib"), "", "vision.metal");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment bufArgs = arena.allocate(ValueLayout.ADDRESS, 8);
            MemorySegment offArgs = arena.allocate(ValueLayout.JAVA_LONG, 8);
            MemorySegment params = arena.allocate(64);
            java.util.Random rnd = new java.util.Random(1);
            Metal.Buffer x = metal.newBuffer(4L * m * k, false);
            Metal.Buffer w = metal.newBuffer(2L * n * k, false);
            Metal.Buffer bias = metal.newBuffer(4L * n, false);
            Metal.Buffer out = metal.newBuffer(4L * m * n, false);
            for (long i = 0; i < (long) m * k; i++) {
                x.contents().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) rnd.nextGaussian());
            }
            for (long i = 0; i < (long) n * k; i++) {
                w.contents().setAtIndex(ValueLayout.JAVA_SHORT, i, Float.floatToFloat16((float) (rnd.nextGaussian() * 0.05)));
            }
            Metal.Buffer[] b = {x, w, bias, out};
            for (int i = 0; i < 4; i++) {
                bufArgs.setAtIndex(ValueLayout.ADDRESS, i, b[i].handle());
                offArgs.setAtIndex(ValueLayout.JAVA_LONG, i, b[i].offset());
            }
            int[] pv = {m, n, k, k, n, 0, 0, 1};
            for (int i = 0; i < pv.length; i++) {
                params.setAtIndex(ValueLayout.JAVA_INT, i, pv[i]);
            }
            int pipe = metal.pipeline(kernel);
            for (int round = 0; round < 3; round++) {
                metal.begin();
                long t0 = System.nanoTime();
                for (int r = 0; r < reps; r++) {
                    metal.dispatch(pipe, bufArgs, offArgs, 4, params, 32, 0, (n + 63) / 64, (m + 63) / 64, 1, 128, 1, 1);
                }
                metal.commitAndWait();
                double ms = (System.nanoTime() - t0) / 1e6 / reps;
                System.out.printf("%s m=%d n=%d k=%d: %.3f ms per dispatch, %.0f GFLOPS%n", kernel, m, n, k, ms, 2.0 * m * n * k / ms / 1e6);
            }
        }
        metal.close();
    }
}
