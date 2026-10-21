package org.robomiri.jllm.metal;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Which experts of an MoE model stay in CPU memory when the GPU cannot keep all of them ({@code backend.metal.cpu-moe}),
 * and the routing statistics that choice is made from.
 *
 * <p>Routers are far from uniform: in a layer of 128 experts a few dozen take most of the tokens. Moving the
 * <em>rarely chosen</em> experts to the CPU frees as many bytes as moving whole layers, while the CPU computes only the
 * few slots that pick them. But every layer with at least one CPU expert costs a GPU → CPU → GPU round trip per step,
 * whether or not a token picks one of them. {@link #plan} weighs both with a cost per token:
 * <pre>
 *     cost = Σ_{layers with a CPU expert} SYNC_US  +  Σ_{CPU experts (l, e)} p(l, e) · expertBytes(l) / CPU_BYTES_PER_US
 * </pre>
 * where {@code p(l, e)} is the expected number of times a token picks expert e of layer l (Σ_e p = topK), from the
 * counts of earlier runs plus one (so a layer without counts is uniform). It frees at least the bytes asked for,
 * greedily taking the move with the lowest cost per byte freed: either the next coldest expert of a layer that already
 * has CPU experts, or opening a new layer with its k coldest experts (the round trip spread over them). With uniform
 * counts this fills one layer completely before opening the next, i.e. whole layers; skewed counts move the cold
 * experts of a few layers. The two constants are estimates (Apple Silicon CPU bandwidth, a command-buffer round
 * trip), not measurements; only their ratio matters.
 *
 * <p>The counts come from the GPU ({@code moe_remap} adds up every layer's choices while experts are split) and are
 * kept in {@code ~/.cache/jllm/expert-usage/<model key>.txt}, one line per MoE layer, summed over runs (halved when a
 * layer passes {@link #MAX_COUNT} picks, so recent use weighs more).
 */
final class ExpertPlacement {

    /** Estimated cost of one GPU → CPU → GPU round trip, microseconds ({@code -Djllm.metal.cpuMoeSyncUs} to tune). */
    static final double SYNC_US = Double.parseDouble(System.getProperty("jllm.metal.cpuMoeSyncUs", "100"));
    /** Estimated CPU memory bandwidth, bytes per microsecond (60 GB/s). */
    static final double CPU_BYTES_PER_US = 60e3;
    static final long MAX_COUNT = 1L << 30;

    private static final Object FILE_LOCK = new Object();

    private ExpertPlacement() {
    }

    /**
     * The experts to leave on the CPU: {@code cold[l][e]}, null for a dense layer.
     *
     * @param need        bytes the GPU must give up (nothing moves when {@code need <= 0})
     * @param expertBytes bytes of one expert of layer l (gate, up and down), 0 for a dense layer
     * @param experts     experts of layer l
     * @param topK        experts per token of layer l
     * @param counts      how often each expert was picked (null, or a null row, for none)
     */
    static boolean[][] plan(long need, long[] expertBytes, int[] experts, int[] topK, long[][] counts) {
        int layers = expertBytes.length;
        boolean[][] cold = new boolean[layers][];
        double[][] cost = new double[layers][];   // per expert, ascending in order[l]
        int[][] order = new int[layers][];
        int[] moved = new int[layers];
        for (int l = 0; l < layers; l++) {
            if (expertBytes[l] <= 0) {
                continue;
            }
            int e = experts[l];
            cold[l] = new boolean[e];
            long[] c = counts != null && counts[l] != null && counts[l].length == e ? counts[l] : new long[e];
            double total = 0;
            for (long v : c) {
                total += v + 1;
            }
            double[] p = new double[e];
            Integer[] idx = new Integer[e];
            for (int i = 0; i < e; i++) {
                p[i] = (c[i] + 1) / total * topK[l];
                idx[i] = i;
            }
            Arrays.sort(idx, (a, b) -> a.equals(b) ? 0 : p[a] != p[b] ? Double.compare(p[a], p[b]) : Integer.compare(a, b));
            order[l] = new int[e];
            cost[l] = new double[e];
            for (int i = 0; i < e; i++) {
                order[l][i] = idx[i];
                cost[l][i] = p[idx[i]] * expertBytes[l] / CPU_BYTES_PER_US;
            }
        }
        long freed = 0;
        while (freed < need) {
            int bestLayer = -1;
            int bestK = 0;
            double bestPerByte = Double.MAX_VALUE;
            for (int l = 0; l < layers; l++) {
                if (cold[l] == null || moved[l] == experts[l]) {
                    continue;
                }
                if (moved[l] > 0) {
                    double perByte = cost[l][moved[l]] / expertBytes[l];
                    if (perByte < bestPerByte) {
                        bestPerByte = perByte;
                        bestLayer = l;
                        bestK = 1;
                    }
                    continue;
                }
                // opening the layer: the round trip shared by its k coldest experts, k at most what is still needed
                int maxK = (int) Math.min(experts[l], (need - freed + expertBytes[l] - 1) / expertBytes[l]);
                double sum = SYNC_US;
                for (int k = 1; k <= maxK; k++) {
                    sum += cost[l][k - 1];
                    double perByte = sum / ((double) k * expertBytes[l]);
                    if (perByte < bestPerByte) {
                        bestPerByte = perByte;
                        bestLayer = l;
                        bestK = k;
                    }
                }
            }
            if (bestLayer < 0) {
                break; // everything is on the CPU already
            }
            for (int k = 0; k < bestK; k++) {
                cold[bestLayer][order[bestLayer][moved[bestLayer]++]] = true;
                freed += expertBytes[bestLayer];
            }
        }
        return cold;
    }

    /** Every expert of the last {@code n} MoE layers on the CPU. */
    static boolean[][] lastLayers(int n, long[] expertBytes, int[] experts) {
        boolean[][] cold = new boolean[expertBytes.length][];
        for (int l = expertBytes.length - 1; l >= 0; l--) {
            if (expertBytes[l] > 0) {
                cold[l] = new boolean[experts[l]];
                if (n > 0) {
                    Arrays.fill(cold[l], true);
                    n--;
                }
            }
        }
        return cold;
    }

    /** {@code ~/.cache/jllm/expert-usage/<key>.txt}, the key derived from the model's description and size. */
    static Path statsFile(String modelKey) {
        String dir = System.getProperty("jllm.metal.expertStatsDir");
        Path base = dir != null ? Path.of(dir) : Path.of(System.getProperty("user.home"), ".cache", "jllm", "expert-usage");
        return base.resolve(Integer.toHexString(modelKey.hashCode()) + "-" + Integer.toHexString(modelKey.length()) + ".txt");
    }

    /** The saved counts (one row per layer, null for a dense layer or none saved), or null without a file. */
    static long[][] load(Path file, int layers) {
        synchronized (FILE_LOCK) {
            try {
                if (!Files.isRegularFile(file)) {
                    return null;
                }
                long[][] c = new long[layers][];
                for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
                    int colon = line.indexOf(':');
                    if (line.startsWith("#") || colon < 0) {
                        continue;
                    }
                    int l = Integer.parseInt(line.substring(0, colon).trim());
                    String[] v = line.substring(colon + 1).trim().split(",");
                    if (l >= 0 && l < layers) {
                        c[l] = new long[v.length];
                        for (int i = 0; i < v.length; i++) {
                            c[l][i] = Long.parseLong(v[i].trim());
                        }
                    }
                }
                return c;
            } catch (IOException | RuntimeException e) {
                System.err.println("warning: ignoring unreadable expert statistics " + file + ": " + e.getMessage());
                return null;
            }
        }
    }

    /** Adds {@code delta} to the saved counts (halving a layer's that pass MAX_COUNT) and writes the file atomically. */
    static void add(Path file, String modelKey, long[][] delta) {
        synchronized (FILE_LOCK) {
            long[][] c = load(file, delta.length);
            if (c == null) {
                c = new long[delta.length][];
            }
            List<String> lines = new ArrayList<>();
            lines.add("# jllm expert usage: <layer>: picks per expert, summed over runs; " + modelKey);
            for (int l = 0; l < delta.length; l++) {
                if (delta[l] == null) {
                    continue;
                }
                long[] row = c[l] != null && c[l].length == delta[l].length ? c[l] : new long[delta[l].length];
                long sum = 0;
                for (int i = 0; i < row.length; i++) {
                    row[i] += delta[l][i];
                    sum += row[i];
                }
                if (sum > MAX_COUNT) {
                    for (int i = 0; i < row.length; i++) {
                        row[i] /= 2;
                    }
                }
                StringBuilder sb = new StringBuilder().append(l).append(':');
                for (int i = 0; i < row.length; i++) {
                    sb.append(i == 0 ? "" : ",").append(row[i]);
                }
                lines.add(sb.toString());
            }
            try {
                Files.createDirectories(file.getParent());
                Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
                Files.write(tmp, lines, StandardCharsets.UTF_8);
                Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException e) {
                System.err.println("warning: could not save expert statistics " + file + ": " + e.getMessage());
            }
        }
    }
}
