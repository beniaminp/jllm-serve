package org.robomiri.jllm.diffusion;

/**
 * Times the big GPU kernels alone (run under the machine's heavy lock): a linear layer, a 3x3 convolution, flash attention
 * and group norm at the sizes of a Stable Diffusion UNet at 512x512, reporting GFLOP/s or GB/s.
 * {@code KernelBench [f16|f32] [iterations]}
 */
public final class KernelBench {

    public static void main(String[] a) {
        String precision = a.length > 0 ? a[0] : "f16";
        int iters = a.length > 1 ? Integer.parseInt(a[1]) : 20;
        String only = a.length > 2 ? a[2] : "linear,conv,attention,groupnorm"; // sections to run
        try (Devices dev = Devices.open("metal", precision, false)) {
            Ops ops = dev.main();
            java.util.Random rnd = new java.util.Random(1);
            // linear: rows x K times K x N
            for (int[] s : only.contains("linear") ? new int[][]{{4096, 1280, 1280}, {4096, 320, 320}, {1024, 5120, 1280}, {16384, 320, 1280}} : new int[0][]) {
                Tn x = ops.upload(rand(rnd, s[0] * s[2]), s[0], s[2]);
                Tn w = ops.upload(rand(rnd, s[1] * s[2]), s[1], s[2]);
                double sec = time(ops, iters, () -> ops.free(ops.linear(x, w, null, null, Ops.Act.NONE)));
                System.out.printf("linear %5d x %4d x %4d: %7.2f ms  %7.1f GFLOP/s%n", s[0], s[2], s[1], sec * 1e3, 2.0 * s[0] * s[1] * s[2] / sec / 1e9);
            }
            // convolution 3x3: [1,h,w,c] with [c,3,3,c]
            for (int[] s : !only.contains("conv") ? new int[0][] : new int[][]{{64, 64, 320, 320}, {32, 32, 640, 640}, {16, 16, 1280, 1280}, {128, 128, 512, 512}, {256, 256, 256, 256}, {512, 512, 128, 128}}) {
                Tn x = ops.upload(rand(rnd, s[0] * s[1] * s[2]), 1, s[0], s[1], s[2]);
                Tn w = ops.upload(rand(rnd, s[3] * 9 * s[2]), s[3], 3, 3, s[2]);
                double sec = time(ops, Math.max(3, iters / 4), () -> ops.free(ops.conv2d(x, w, null, 1, 1, 1, 1, 1, null, null)));
                System.out.printf("conv3x3 %3dx%3d c=%4d->%4d: %7.2f ms  %7.1f GFLOP/s%n", s[0], s[1], s[2], s[3], sec * 1e3,
                        2.0 * s[0] * s[1] * s[2] * s[3] * 9 / sec / 1e9);
            }
            // attention
            for (int[] s : !only.contains("attention") ? new int[0][] : new int[][]{{4096, 4096, 5, 64}, {1024, 1024, 10, 64}, {4096, 77, 5, 64}, {4096, 4096, 8, 40}, {4096, 32, 5, 64}, {4096, 64, 5, 64}, {4096, 128, 5, 64}, {4096, 512, 5, 64}, {256, 77, 5, 64}}) {
                int dm = s[2] * s[3];
                Tn q = ops.upload(rand(rnd, s[0] * dm), 1, s[0], dm);
                Tn k = ops.upload(rand(rnd, s[1] * dm), 1, s[1], dm);
                Tn v = ops.upload(rand(rnd, s[1] * dm), 1, s[1], dm);
                double sec = time(ops, Math.max(3, iters / 4), () -> ops.free(ops.attention(q, k, v, s[2], false)));
                System.out.printf("attention lq=%4d lk=%4d heads=%2d hd=%2d: %7.2f ms  %7.1f GFLOP/s%n", s[0], s[1], s[2], s[3], sec * 1e3,
                        4.0 * s[0] * s[1] * dm / sec / 1e9);
            }
            // group norm (memory bound)
            for (int[] s : !only.contains("groupnorm") ? new int[0][] : new int[][]{{64, 64, 320}, {256, 256, 256}, {512, 512, 128}}) {
                Tn x = ops.upload(rand(rnd, s[0] * s[1] * s[2]), 1, s[0], s[1], s[2]);
                Tn g = ops.weight(vecTensor(s[2], rnd), Ops.Kind.VECTOR);
                double sec = time(ops, Math.max(3, iters / 4), () -> ops.free(ops.groupNorm(x, g, g, 32, 1e-5f, true)));
                double bytes = (ops.half() ? 2.0 : 4.0) * s[0] * s[1] * s[2] * 2;
                System.out.printf("groupnorm %3dx%3d c=%4d: %7.2f ms  %6.1f GB/s%n", s[0], s[1], s[2], sec * 1e3, bytes / sec / 1e9);
            }
        }
    }

    private static org.robomiri.jllm.safetensors.SafeTensors.Tensor vecTensor(int n, java.util.Random rnd) {
        java.lang.foreign.MemorySegment seg = java.lang.foreign.Arena.ofAuto().allocate(n * 4L);
        for (int i = 0; i < n; i++) {
            seg.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, 1f);
        }
        return new org.robomiri.jllm.safetensors.SafeTensors.Tensor("g", org.robomiri.jllm.safetensors.SafeTensors.DType.F32, new long[]{n}, seg);
    }

    private static float[] rand(java.util.Random r, int n) {
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            f[i] = (float) (r.nextGaussian() * 0.1);
        }
        return f;
    }

    private static double time(Ops ops, int iters, Runnable op) {
        op.run();
        op.run();
        ops.sync();
        long t0 = System.nanoTime();
        for (int i = 0; i < iters; i++) {
            op.run();
        }
        ops.sync();
        return (System.nanoTime() - t0) / 1e9 / iters;
    }
}
