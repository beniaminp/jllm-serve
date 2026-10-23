package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.Random;

/**
 * Microbenchmark of the dense matmul kernels at the Qwen3-0.6B shapes: ms per dispatch for n = 1 .. 16 tokens
 * (32 distinct weight tensors per shape, so the weights stream from memory).
 * {@code java ... org.robomiri.jllm.metal.MatmulBench <lib|-> [rounds] [shape]}.
 */
public final class MatmulBench {

    public static void main(String[] args) {
        Metal metal = Metal.create(args[0].equals("-") ? null : args[0], "");
        MetalKernels k = new MetalKernels(metal, 1e-6f);
        int[] ns = {1, 2, 3, 4, 6, 8, 9, 12, 16};
        int rounds = args.length > 1 ? Integer.parseInt(args[1]) : 15;
        try (Arena arena = Arena.ofShared()) {
            Metal.Buffer in = k.upload(new float[16 * 3072]);
            Metal.Buffer out = k.scratch(16L * 151936 * 4);
            Metal.Buffer q = k.scratch(16L * 2048 * 4);
            Metal.Buffer kk = k.scratch(16L * 1024 * 4);
            Metal.Buffer vv = k.scratch(16L * 1024 * 4);
            String only = args.length > 2 ? args[2] : "";
            if (only.isEmpty() || only.equals("down")) {
                run("down 1024x3072 q4", k, QType.Q4_0, 1024, 3072, 32, arena, ns, rounds, (t, buf, n) ->
                        k.matmul(t, buf, in, 0, out, 0, n, true, null, 0f));
            }
            if (only.isEmpty() || only.equals("swiglu")) {
                run("swiglu 6144x1024 q4", k, QType.Q4_0, 6144, 1024, 32, arena, ns, rounds, (t, buf, n) ->
                        k.swiglu(t, buf, in, null, out, false, n));
            }
            if (only.isEmpty() || only.equals("o")) {
                run("o_proj 1024x2048 q4", k, QType.Q4_0, 1024, 2048, 32, arena, ns, rounds, (t, buf, n) ->
                        k.matmul(t, buf, in, 0, out, 0, n, true, null, 0f));
            }
            if (only.isEmpty() || only.equals("qkv")) {
                run("qkv 2048+1024+1024 x1024 q4", k, QType.Q4_0, 4096, 1024, 32, arena, ns, rounds, (t, buf, n) ->
                        k.matmul3(t, buf, t, buf, t, buf, in, null, q, kk, 0, vv, 0, false, null, null, null, n));
            }
            if (only.isEmpty() || only.equals("head")) {
                run("lm_head 151936x1024 q8", k, QType.Q8_0, 151936, 1024, 1, arena, ns, rounds, (t, buf, n) ->
                        k.matmul(t, buf, in, 0, out, 0, n, false, null, 0f));
            }
        }
        k.close();
    }

    interface Op {
        void run(QTensor t, Metal.Buffer buf, int n);
    }

    private static void run(String label, MetalKernels k, QType type, int rows, int cols, int copies, Arena arena,
                            int[] ns, int rounds, Op op) {
        QTensor[] ts = new QTensor[copies];
        Metal.Buffer[] bufs = new Metal.Buffer[copies];
        for (int c = 0; c < copies; c++) {
            Random rnd = new Random(c);
            ts[c] = QTensor.quantize(arena, type, rows, cols, (r, dst) -> {
                for (int i = 0; i < dst.length; i++) {
                    dst[i] = (float) rnd.nextGaussian() * 0.05f;
                }
            });
            bufs[c] = k.wrap(ts[c].data);
        }
        double[] best = new double[ns.length];
        Arrays.fill(best, Double.MAX_VALUE);
        for (int r = -2; r < rounds; r++) {
            for (int i = 0; i < ns.length; i++) {
                k.begin();
                long t0 = System.nanoTime();
                for (int c = 0; c < copies; c++) {
                    op.run(ts[c], bufs[c], ns[i]);
                }
                k.commitAndWait();
                double ms = (System.nanoTime() - t0) / 1e6 / copies;
                if (r >= 0) {
                    best[i] = Math.min(best[i], ms);
                }
            }
        }
        StringBuilder sb = new StringBuilder(String.format("%-30s", label));
        for (int i = 0; i < ns.length; i++) {
            sb.append(String.format(" n%d=%.3f", ns[i], best[i]));
        }
        System.out.println(sb);
    }
}
