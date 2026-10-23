package org.robomiri.jllm;

import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.model.QType;

import java.nio.file.Path;
import java.util.Arrays;

/**
 * Small-batch forward passes on the Metal backend. {@code exact <lib> <modelDir>}: {@code forwardAll} over n tokens
 * must give bit-for-bit the logits of n single-token steps (n = 2, 4, 8, 16, and the sizes in between).
 * {@code bench <lib> <modelDir> [rounds] [n...]}: the cost of a pass of n tokens relative to the first n (minimum
 * over interleaved rounds). Without a GPU the check is skipped.
 */
public final class MetalPassCheck {

    private static final int PREFIX = 48;

    public static void main(String[] args) throws Exception {
        String mode = args[0];
        String lib = args[1].equals("-") ? null : args[1];
        if (!Metal.available(lib)) {
            System.out.println("note: Metal unavailable, MetalPassCheck skipped");
            return;
        }
        Path dir = Path.of(args[2]);
        Architecture cfg = Architecture.load(dir);
        try (ModelWeights w = ModelWeights.load(dir, cfg, QType.Q4_0, ShardProvider.local(dir));
             Backend b = new MetalBackend(w, 256, 16, lib)) {
            if (mode.equals("exact")) {
                exact(b, cfg.core().vocabSize());
            } else {
                int rounds = args.length > 3 ? Integer.parseInt(args[3]) : 30;
                int[] ns = args.length > 4 ? Arrays.stream(args, 4, args.length).mapToInt(Integer::parseInt).toArray()
                        : new int[]{1, 2, 3, 5, 9, 16};
                bench(b, cfg.core().vocabSize(), rounds, ns);
            }
        }
    }

    private static int[] tokens(int n, int vocab, int seed) {
        int[] t = new int[n];
        long s = seed * 2654435761L + 12345;
        for (int i = 0; i < n; i++) {
            s = s * 6364136223846793005L + 1442695040888963407L;
            t[i] = (int) ((s >>> 33) % Math.min(vocab, 20000));
        }
        return t;
    }

    private static void prefill(Backend b, int[] prefix, int vocab) {
        for (int p = 0; p < prefix.length; p += 16) {
            b.forward(prefix, p, Math.min(16, prefix.length - p), p, new float[vocab]);
        }
    }

    static void exact(Backend b, int vocab) {
        int[] prefix = tokens(PREFIX, vocab, 1);
        int failures = 0;
        for (int n : new int[]{2, 3, 4, 5, 7, 8, 9, 12, 16}) {
            int[] t = tokens(n, vocab, 100 + n);
            prefill(b, prefix, vocab);
            float[][] single = new float[n][vocab];
            for (int i = 0; i < n; i++) {
                b.forward(t, i, 1, PREFIX + i, single[i]);
            }
            prefill(b, prefix, vocab);
            float[] all = new float[n * vocab];
            b.forwardAll(t, 0, n, PREFIX, all, vocab);
            int bad = 0;
            for (int i = 0; i < n; i++) {
                for (int v = 0; v < vocab; v++) {
                    if (Float.floatToRawIntBits(single[i][v]) != Float.floatToRawIntBits(all[i * vocab + v])) {
                        bad++;
                    }
                }
            }
            System.out.println((bad == 0 ? "ok  " : "FAIL") + " forwardAll(" + n + ") == " + n + " single steps bit-for-bit"
                    + (bad == 0 ? "" : " (" + bad + " logits differ)"));
            failures += bad == 0 ? 0 : 1;
        }
        if (failures > 0) {
            throw new AssertionError(failures + " batch sizes are not bit-identical to single steps");
        }
    }

    static void bench(Backend b, int vocab, int rounds, int[] ns) {
        int[] prefix = tokens(PREFIX, vocab, 1);
        prefill(b, prefix, vocab);
        double[] best = new double[ns.length];
        Arrays.fill(best, Double.MAX_VALUE);
        float[] out = new float[16 * vocab];
        for (int r = -2; r < rounds; r++) {
            for (int k = 0; k < ns.length; k++) {
                int n = ns[k];
                int[] t = tokens(n, vocab, 7 + n);
                long t0 = System.nanoTime();
                b.forwardAll(t, 0, n, PREFIX, out, vocab);
                double ms = (System.nanoTime() - t0) / 1e6;
                if (r >= 0) {
                    best[k] = Math.min(best[k], ms);
                }
            }
        }
        System.out.printf("pass cost (min of %d rounds, forwardAll, ms and x of n=%d):%n", rounds, ns[0]);
        for (int k = 0; k < ns.length; k++) {
            System.out.printf("  n=%2d  %7.2f ms  %.2fx%n", ns[k], best[k], best[k] / best[0]);
        }
    }
}
