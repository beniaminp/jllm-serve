package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.Random;

/**
 * Timings behind docs/07 §7.16: one {@code rows × cols} matrix times one activation, per type,
 *
 * <ul>
 *   <li>single thread: Java int8 kernels row by row, the native library in one call (at every level this CPU runs),
 *       and the float kernels ({@link VectorKernels#dotRow});</li>
 *   <li>{@link CpuMatmul#mul} for one token on {@code threads} threads: native int8, Java int8, float.</li>
 * </ul>
 *
 * Prints the best and the median of the rounds (other processes make single runs noisy). Run one type per process:
 * the Java kernels of several formats in one JVM share profiled Vector API call sites, and C2 code compiled for one
 * can be thrown away for another (docs/07 §7.10), which would make the Java numbers look worse than they are.
 *
 * <pre>java ... org.robomiri.jllm.cpu.NativeCpuBench TYPE [rows] [cols] [threads] [rounds]</pre>
 */
public final class NativeCpuBench {

    public static void main(String[] args) {
        QType type = QType.valueOf(args[0]);
        int rows = args.length > 1 ? Integer.parseInt(args[1]) : 2048;
        int cols = args.length > 2 ? Integer.parseInt(args[2]) : 1024;
        int threads = args.length > 3 ? Integer.parseInt(args[3]) : Runtime.getRuntime().availableProcessors();
        int rounds = args.length > 4 ? Integer.parseInt(args[4]) : 60;
        NativeCpu n = NativeCpu.load(NativeCpu.locate(null));
        int detected = n.detectedLevel();
        Random rnd = new Random(3);
        float[] x = new float[cols];
        for (int i = 0; i < cols; i++) {
            x[i] = (float) rnd.nextGaussian();
        }
        short[] xq = new short[cols];
        float[] xs = new float[cols / QType.BLOCK];
        int[] sums = new int[xs.length];
        Int8Kernels.quantize(x, 0, cols, xq, xs);
        KQuantKernels.blockSums(xq, cols, sums);
        NativeCpu.Activation act = new NativeCpu.Activation();
        n.quantize(x, 0, cols, act);
        float[] out = new float[rows];
        try (Arena arena = Arena.ofShared(); ThreadPool pool = new ThreadPool(threads)) {
            {
                QType t = type;
                QTensor w = NativeCpu.randomTensor(arena, t, rows, cols, rnd, false, false);
                boolean k = KQuantKernels.supports(w);
                boolean g = GgufInt8Kernels.supports(w);
                boolean int8 = k || g || t == QType.Q4_0 || t == QType.Q8_0 || t == QType.Q4_0G || t == QType.Q8_0G;
                StringBuilder line = new StringBuilder(String.format("%-6s %dx%d 1 thread  ", t, rows, cols));
                if (int8) {
                    line.append(time("java-int8", rounds, () -> {
                        for (int r = 0; r < rows; r++) {
                            out[r] = k ? KQuantKernels.dotRow(w, r, xq, xs, sums, 0) : g ? GgufInt8Kernels.dotRow(w, r, xq, xs, sums, 0)
                                    : Int8Kernels.dotRow(w, r, xq, xs, 0);
                        }
                    }));
                }
                for (int level = 0; level <= detected; level++) {
                    if (n.setLevel(level) == level && n.hasKernelFor(t)) {
                        line.append(time(n.name(level), rounds, () -> n.dotRows(w, 0, rows, act, 0, out, 0, false)));
                    }
                }
                n.setLevel(detected);
                line.append(time("float", rounds, () -> {
                    for (int r = 0; r < rows; r++) {
                        out[r] = VectorKernels.dotRow(w, r, x, 0);
                    }
                }));
                System.out.println(line);
            }
            {
                QType t = type;
                QTensor w = NativeCpu.randomTensor(arena, t, rows, cols, rnd, false, false);
                CpuMatmul nat = new CpuMatmul(pool, true, false, 4 * cols, n);
                CpuMatmul java = new CpuMatmul(pool, true, false, 4 * cols, null);
                CpuMatmul f32 = new CpuMatmul(pool, false, false, 4 * cols, null);
                System.out.println(String.format("%-6s %dx%d %d threads", t, rows, cols, threads)
                        + time("native-int8", rounds, () -> nat.mul(w, x, out, 0, 1, false))
                        + time("java-int8", rounds, () -> java.mul(w, x, out, 0, 1, false))
                        + time("float", rounds, () -> f32.mul(w, x, out, 0, 1, false)));
            }
        }
    }

    private static String time(String name, int rounds, Runnable r) {
        long until = System.nanoTime() + 3_000_000_000L; // warm-up: C2 must have compiled the Java kernels
        for (int i = 0; i < 20 || System.nanoTime() < until; i++) {
            r.run();
        }
        long[] t = new long[rounds];
        for (int i = 0; i < rounds; i++) {
            long a = System.nanoTime();
            r.run();
            t[i] = System.nanoTime() - a;
        }
        Arrays.sort(t); // us: best / median
        return String.format("  %s %.0f / %.0f", name, t[0] / 1e3, t[rounds / 2] / 1e3);
    }
}
