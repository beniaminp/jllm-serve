package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Ggml;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The native int8 kernels ({@code libjllm_cpu}, {@link NativeCpu}) against the Java ones ({@link Int8Kernels},
 * {@link KQuantKernels}), at every kernel level this CPU supports (scalar, and AVX2 / AVX-VNNI / AVX512-VNNI or NEON /
 * dotprod):
 *
 * <ul>
 *   <li>the activation quantizer: bit for bit (values, scales, block sums);</li>
 *   <li>every type (Q4_0, Q8_0, Q4_0G, Q8_0G, Q4_K, Q5_K, Q6_K) with unit scales, where every intermediate value is an
 *       integer below 2^24 so both sides must agree exactly (full-range quants and activations; and random sub-block
 *       scales with small activations);</li>
 *   <li>random scales: within 3e-5 of {@code Σ|w_i x_i|} (the float sums are added in another order);</li>
 *   <li>odd row counts, columns that are not a multiple of 256 (the 32-block formats), activation offsets (MoE slots),
 *       accumulation into the output, split tensors (fused gate | up, per-expert interleave) across their part
 *       boundaries, and the {@link CpuMatmul} decode path with and without the library.</li>
 * </ul>
 *
 * Skipped (passes) when the library is not built ({@code scripts/build-native-cpu.sh}); the Maven test phase builds it
 * first when a C compiler is present.
 */
public final class NativeCpuCheck {

    static int passed;
    static int failed;

    static void check(String failure, String what) {
        if (failure == null) {
            passed++;
        } else {
            failed++;
            System.out.println("FAIL: " + what + ": " + failure);
        }
    }

    public static void main(String[] args) {
        Path lib = NativeCpu.locate(args.length > 0 ? args[0] : null);
        if (lib == null) {
            System.out.println("native cpu kernels: SKIPPED (libjllm_cpu not built; scripts/build-native-cpu.sh)");
            return;
        }
        NativeCpu n = NativeCpu.load(lib);
        int detected = n.detectedLevel();
        List<String> levels = new ArrayList<>();
        for (int level = 0; level <= detected; level++) {
            if (n.setLevel(level) != level) {
                continue; // not a level of this architecture
            }
            levels.add(n.name(level));
            runLevel(n, n.name(level));
        }
        n.setLevel(detected);
        check(n.selfCheck(), "load-time self-check");
        matmul(n);
        fewRows(n);
        System.out.println("native cpu kernels (" + lib + ", levels " + String.join(", ", levels) + "; detected "
                + n.name(detected) + "): " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    private static final QType[] TYPES = {QType.Q4_0, QType.Q8_0, QType.Q4_0G, QType.Q8_0G, QType.Q4_K, QType.Q5_K, QType.Q6_K,
            QType.Q4_1, QType.Q5_0, QType.Q5_1, QType.IQ4_NL, QType.MXFP4, QType.IQ4_XS, QType.Q3_K};

    /** The Java int8 kernel for {@code w}: what the native one must agree with. */
    private static float javaDot(QTensor w, int r, short[] xq, float[] xs, int[] sums, int xo) {
        return KQuantKernels.supports(w) ? KQuantKernels.dotRow(w, r, xq, xs, sums, xo)
                : GgufInt8Kernels.supports(w) ? GgufInt8Kernels.dotRow(w, r, xq, xs, sums, xo)
                : Int8Kernels.dotRow(w, r, xq, xs, xo);
    }

    private static void runLevel(NativeCpu n, String level) {
        Random rnd = new Random(7);
        for (int len : new int[]{32, 96, 256, 1056, 4096}) {
            check(n.checkQuantize(rnd, len), level + " quantize " + len);
        }
        try (Arena arena = Arena.ofConfined()) {
            for (QType t : TYPES) {
                if (NativeCpu.typeId(t) >= NativeCpu.T_Q4_1 && !n.hasKernelFor(t)) {
                    continue; // the formats added later have no kernels at the x86-64 levels
                }
                boolean k = t.blockSize == Ggml.QK_K;
                int[] exactCols = k ? new int[]{256} : new int[]{32, 96, 160, 256};
                for (int cols : exactCols) {
                    for (int rows : new int[]{1, 3, 7}) {
                        // full-range activations, unit scales; then small activations with random sub-block scales
                        check(exact(n, arena, rnd, t, rows, cols, false), level + " " + t + " exact " + rows + "x" + cols);
                        check(exact(n, arena, rnd, t, rows, cols, true), level + " " + t + " exact-small " + rows + "x" + cols);
                    }
                }
                int[] cols = k ? new int[]{256, 768, 2048} : new int[]{32, 96, 1056, 2048, 3072};
                for (int c : cols) {
                    for (int rows : new int[]{1, 5, 13}) {
                        check(n.checkType(arena, rnd, t, rows, c, false), level + " " + t + " " + rows + "x" + c);
                    }
                }
                check(offsetsAndAccumulate(n, arena, rnd, t), level + " " + t + " offsets / accumulate");
                if (t.gguf) {
                    check(split(n, arena, rnd, t, 0), level + " " + t + " split gate|up");
                    check(split(n, arena, rnd, t, 6), level + " " + t + " split per expert");
                }
            }
        }
    }

    private static String exact(NativeCpu n, Arena arena, Random rnd, QType t, int rows, int cols, boolean small) {
        QTensor w = NativeCpu.randomTensor(arena, t, rows, cols, rnd, true, small);
        byte[] xq = new byte[cols];
        float[] xs = new float[cols / QType.BLOCK];
        int lim = small ? 7 : 127;
        for (int i = 0; i < cols; i++) {
            xq[i] = (byte) (rnd.nextInt(2 * lim + 1) - lim);
        }
        java.util.Arrays.fill(xs, 1f);
        return n.compare(w, xq, xs, true, rows);
    }

    /** Rows {@code [2, 9)} with the activation from block 3 on, accumulated into an output with values, at offset 5. */
    private static String offsetsAndAccumulate(NativeCpu n, Arena arena, Random rnd, QType t) {
        int cols = t.blockSize == Ggml.QK_K ? 512 : 160;
        int rows = 11;
        QTensor w = NativeCpu.randomTensor(arena, t, rows, cols, rnd, false, false);
        int xBlock = 3;
        float[] x = new float[cols + xBlock * QType.BLOCK];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) rnd.nextGaussian();
        }
        NativeCpu.Activation a = new NativeCpu.Activation();
        n.quantize(x, 0, x.length, a);
        short[] xq = new short[x.length];
        float[] xs = new float[x.length / QType.BLOCK];
        Int8Kernels.quantize(x, 0, x.length, xq, xs);
        int[] sums = new int[xs.length];
        KQuantKernels.blockSums(xq, x.length, sums);
        float[] out = new float[20];
        for (int i = 0; i < out.length; i++) {
            out[i] = i * 0.25f;
        }
        float[] base = out.clone();
        n.dotRows(w, 2, 9, a, xBlock, out, 5, true);
        for (int i = 0; i < out.length; i++) {
            int r = i - 5 + 2;
            if (r < 2 || r >= 9) {
                if (out[i] != base[i]) {
                    return "output " + i + " outside the rows changed";
                }
                continue;
            }
            int xo = xBlock * QType.BLOCK;
            float java = javaDot(w, r, xq, xs, sums, xo);
            float want = base[i] + java;
            if (!(Math.abs(out[i] - want) <= 1e-4f * (1 + Math.abs(want)))) {
                return "row " + r + ": " + out[i] + " vs " + want;
            }
        }
        return null;
    }

    /** A split tensor of two {@code rows}-row views: native over ranges crossing the part boundaries vs Java per row. */
    private static String split(NativeCpu n, Arena arena, Random rnd, QType t, int interleave) {
        int cols = t.blockSize == Ggml.QK_K ? 256 : 96;
        int rows = 24; // 4 experts of 6 rows (interleave 6) or two halves of 24
        QTensor a = NativeCpu.randomTensor(arena, t, rows, cols, rnd, false, false);
        QTensor b = NativeCpu.randomTensor(arena, t, rows, cols, rnd, false, false);
        QTensor w = QTensor.split(a, b, interleave);
        float[] x = new float[cols];
        for (int i = 0; i < cols; i++) {
            x[i] = (float) rnd.nextGaussian();
        }
        NativeCpu.Activation act = new NativeCpu.Activation();
        n.quantize(x, 0, cols, act);
        short[] xq = new short[cols];
        float[] xs = new float[cols / QType.BLOCK];
        Int8Kernels.quantize(x, 0, cols, xq, xs);
        int[] sums = new int[xs.length];
        KQuantKernels.blockSums(xq, cols, sums);
        float[] out = new float[w.rows];
        int[][] ranges = {{0, w.rows}, {3, 31}, {5, 7}, {23, 25}, {11, 12}};
        for (int[] rg : ranges) {
            java.util.Arrays.fill(out, Float.NaN);
            n.dotRows(w, rg[0], rg[1], act, 0, out, rg[0], false);
            for (int r = rg[0]; r < rg[1]; r++) {
                float java = javaDot(w, r, xq, xs, sums, 0);
                if (!(Math.abs(out[r] - java) <= 1e-4f * (1 + Math.abs(java)))) {
                    return "rows " + rg[0] + ".." + rg[1] + ", row " + r + ": " + out[r] + " vs " + java;
                }
            }
        }
        return null;
    }

    /** {@link CpuMatmul#mul} for one token, native vs Java kernels, on a pool of 3 threads (odd row counts). */
    private static void matmul(NativeCpu n) {
        Random rnd = new Random(11);
        try (Arena arena = Arena.ofShared(); ThreadPool pool = new ThreadPool(3)) {
            for (QType t : TYPES) {
                if (NativeCpu.typeId(t) >= NativeCpu.T_Q4_1 && !n.hasKernelFor(t)) {
                    continue;
                }
                int cols = t.blockSize == Ggml.QK_K ? 1024 : 1056;
                for (int rows : new int[]{17, 333}) {
                    QTensor w = NativeCpu.randomTensor(arena, t, rows, cols, rnd, false, false);
                    float[] x = new float[cols];
                    for (int i = 0; i < cols; i++) {
                        x[i] = (float) rnd.nextGaussian();
                    }
                    CpuMatmul nat = new CpuMatmul(pool, true, false, 4 * cols, n);
                    CpuMatmul java = new CpuMatmul(pool, true, false, 4 * cols, null);
                    String fail = nat.nativeInt8(w) ? null : "the native path is not taken";
                    float[] a = new float[rows];
                    float[] b = new float[rows];
                    nat.mul(w, x, a, 0, 1, false);
                    java.mul(w, x, b, 0, 1, false);
                    for (int r = 0; r < rows && fail == null; r++) {
                        if (!(Math.abs(a[r] - b[r]) <= 1e-4f * (1 + Math.abs(b[r])))) {
                            fail = "row " + r + ": " + a[r] + " vs " + b[r];
                        }
                    }
                    check(fail, "CpuMatmul " + t + " " + rows + "x" + cols);
                }
            }
        }
    }

    /**
     * {@link CpuMatmul#mul} for 2 and 3 tokens with {@code backend.cpu.batch-int8-rows} 3 (the native single-token kernel
     * per token, weight rows in cache-sized chunks): every token's output equals its own one-token {@code mul} bit for bit,
     * with and without accumulating, on rows that are not a multiple of the chunk; with the switch at 0 the float
     * multi-row kernels run instead (within float rounding of the same).
     */
    private static void fewRows(NativeCpu n) {
        Random rnd = new Random(12);
        int saved = CpuMatmul.fewRows();
        try (Arena arena = Arena.ofShared(); ThreadPool pool = new ThreadPool(3)) {
            for (QType t : TYPES) {
                if (NativeCpu.typeId(t) >= NativeCpu.T_Q4_1 && !n.hasKernelFor(t)) {
                    continue;
                }
                int cols = t.blockSize == Ggml.QK_K ? 1024 : 1056;
                int rows = 333;
                QTensor w = NativeCpu.randomTensor(arena, t, rows, cols, rnd, false, false);
                CpuMatmul mm = new CpuMatmul(pool, true, false, 4 * cols, n);
                for (int tokens = 2; tokens <= 3; tokens++) {
                    float[] x = new float[tokens * cols];
                    for (int i = 0; i < x.length; i++) {
                        x[i] = (float) rnd.nextGaussian();
                    }
                    for (boolean acc : new boolean[]{false, true}) {
                        float[] base = new float[tokens * rows];
                        for (int i = 0; i < base.length; i++) {
                            base[i] = (float) rnd.nextGaussian();
                        }
                        float[] lone = base.clone();
                        for (int k = 0; k < tokens; k++) {
                            float[] xk = java.util.Arrays.copyOfRange(x, k * cols, (k + 1) * cols);
                            float[] ok = java.util.Arrays.copyOfRange(base, k * rows, (k + 1) * rows);
                            mm.mul(w, xk, ok, 0, 1, acc);
                            System.arraycopy(ok, 0, lone, k * rows, rows);
                        }
                        CpuMatmul.configureFewRows(3);
                        float[] few = base.clone();
                        mm.mul(w, x, few, 0, tokens, acc);
                        CpuMatmul.configureFewRows(0);
                        float[] tiles = base.clone();
                        mm.mul(w, x, tiles, 0, tokens, acc);
                        String fail = null;
                        for (int i = 0; i < lone.length && fail == null; i++) {
                            if (Float.floatToRawIntBits(few[i]) != Float.floatToRawIntBits(lone[i])) {
                                fail = "token " + i / rows + " row " + i % rows + ": " + few[i] + " vs " + lone[i] + " alone";
                            }
                        }
                        check(fail, "CpuMatmul " + t + " " + tokens + " tokens" + (acc ? " +=" : "")
                                + ": the few-rows path is each token's one-token result bit for bit");
                        double worst = 0;
                        double scale = 0;
                        for (int i = 0; i < lone.length; i++) {
                            worst = Math.max(worst, Math.abs(tiles[i] - lone[i]));
                            scale = Math.max(scale, Math.abs(lone[i]));
                        }
                        // the float tiles see unrounded activations: int8 rounding apart (~0.4 % per element), the same sums
                        check(worst <= 0.02 * scale ? null : "difference " + worst + " of outputs up to " + scale,
                                "CpuMatmul " + t + " " + tokens
                                + " tokens" + (acc ? " +=" : "") + ": with the switch off the float tiles agree");
                    }
                }
            }
        } finally {
            CpuMatmul.configureFewRows(saved);
        }
    }
}
