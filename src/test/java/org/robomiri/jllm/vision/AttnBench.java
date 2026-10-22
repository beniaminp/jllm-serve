package org.robomiri.jllm.vision;

import org.robomiri.jllm.metal.Metal;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Manual micro-benchmark of the vision attention kernels: {@code AttnBench <kernel> <tokens> <heads> <dh> [reps]}, e.g.
 * {@code vit_attention_mma_d64 475 16 64}. Prints the GPU time per dispatch (reps dispatches in one command buffer).
 */
public final class AttnBench {

    private AttnBench() {
    }

    public static void main(String[] a) {
        String kernel = a[0];
        int t = Integer.parseInt(a[1]);
        int heads = Integer.parseInt(a[2]);
        int dh = Integer.parseInt(a[3]);
        int reps = a.length > 4 ? Integer.parseInt(a[4]) : 24;
        Metal metal = Metal.create(System.getProperty("jllm.metal.lib"), "", "vision.metal");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment bufArgs = arena.allocate(ValueLayout.ADDRESS, 8);
            MemorySegment offArgs = arena.allocate(ValueLayout.JAVA_LONG, 8);
            MemorySegment params = arena.allocate(64);
            long n = (long) t * heads * dh;
            java.util.Random rnd = new java.util.Random(1);
            Metal.Buffer[] b = new Metal.Buffer[6];
            for (int i = 0; i < 4; i++) {
                b[i] = metal.newBuffer(4 * n, false);
                for (long j = 0; j < n; j++) {
                    b[i].contents().setAtIndex(ValueLayout.JAVA_FLOAT, j, (float) (rnd.nextGaussian() * Double.parseDouble(System.getProperty("scale", "1"))));
                }
            }
            b[4] = metal.newBuffer(4L * t, false);
            b[5] = metal.newBuffer(4L * t, false);
            for (int i = 0; i < t; i++) {
                b[4].contents().setAtIndex(ValueLayout.JAVA_INT, i, 0);
                b[5].contents().setAtIndex(ValueLayout.JAVA_INT, i, t);
            }
            for (int i = 0; i < 6; i++) {
                bufArgs.setAtIndex(ValueLayout.ADDRESS, i, b[i].handle());
                offArgs.setAtIndex(ValueLayout.JAVA_LONG, i, b[i].offset());
            }
            params.setAtIndex(ValueLayout.JAVA_INT, 0, t);
            params.setAtIndex(ValueLayout.JAVA_INT, 1, heads);
            params.setAtIndex(ValueLayout.JAVA_INT, 2, dh);
            params.setAtIndex(ValueLayout.JAVA_FLOAT, 3, (float) (1 / Math.sqrt(dh)));
            int pipe = metal.pipeline(kernel);
            boolean mma = kernel.contains("mma");
            int rows = 32;
            for (int round = 0; round < 3; round++) {
                metal.begin();
                long t0 = System.nanoTime();
                for (int r = 0; r < reps; r++) {
                    if (Boolean.getBoolean("sync") && r > 0) {
                        metal.commitAndWait();
                        metal.begin();
                    }
                    if (mma) {
                        metal.dispatch(pipe, bufArgs, offArgs, 6, params, 16, 0, heads, (t + rows - 1) / rows, 1, 128, 1, 1);
                    } else {
                        metal.dispatch(pipe, bufArgs, offArgs, 6, params, 16, 0, (t + 15) / 16, heads, 1, 128, 1, 1);
                    }
                }
                metal.commitAndWait();
                double ms = (System.nanoTime() - t0) / 1e6 / reps;
                double gflop = 4.0 * t * t * dh * heads / 1e9;
                System.out.printf("%s t=%d heads=%d dh=%d: %.3f ms per dispatch, %.0f GFLOPS%n", kernel, t, heads, dh, ms, gflop / ms * 1000);
            }
        }
        metal.close();
    }
}
