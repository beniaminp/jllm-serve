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
 * counts of earlier runs plus one. It frees at least the bytes asked for, greedily taking the move with the lowest
 * cost per byte freed: either the next coldest expert of a layer that already has CPU experts, or opening a new layer
 * with its k coldest experts (the round trip spread over them; k at most what is still needed). A layer without
 * counts is uniform and is only ever moved whole: splitting it would cost a packed copy and a remap per step for no
 * saving, since no expert of it is known to be colder than another. So without statistics the plan is whole layers;
 * skewed counts move the cold experts of a few layers. The two constants are estimates (Apple Silicon CPU bandwidth, a command-buffer round
 * trip), not measurements; only their ratio matters.
 *
 * <p>The counts come from the GPU (the router kernels add up every layer's choices while experts are split) and are
 * kept in {@code ~/.cache/jllm/expert-usage/<model key>.txt}, one line per MoE layer, summed over runs (halved when a
 * layer passes {@link #MAX_COUNT} picks, so recent use weighs more).
 */
final class ExpertPlacement {

    /** Estimated cost of one GPU → CPU → GPU round trip, microseconds ({@code -Djllm.metal.cpuMoeSyncUs} to tune). */
    static final double SYNC_US = nonNegativeProperty("jllm.metal.cpuMoeSyncUs", 100);
    /** Estimated CPU memory bandwidth, bytes per microsecond (60 GB/s). */
    static final double CPU_BYTES_PER_US = 60e3;
    static final long MAX_COUNT = 1L << 30;

    private static final Object FILE_LOCK = new Object();

    /** One daemon thread for the periodic saves, so the decode thread never waits for the file or its lock. */
    private static final java.util.concurrent.ExecutorService SAVER = java.util.concurrent.Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "jllm-expert-stats");
        t.setDaemon(true);
        return t;
    });

    /** A numeric system property, or {@code def} (with a warning) when it is missing, malformed or not positive. */
    static double positiveProperty(String name, double def) {
        double v = numberProperty(name, def);
        if (v != def && !(v > 0)) {
            System.err.println("warning: -D" + name + " must be positive; ignored");
            return def;
        }
        return v;
    }

    private static double nonNegativeProperty(String name, double def) {
        double v = numberProperty(name, def);
        if (!(v >= 0)) {
            System.err.println("warning: -D" + name + " must not be negative; using " + def);
            return def;
        }
        return v;
    }

    private static double numberProperty(String name, double def) {
        String s = System.getProperty(name);
        if (s == null) {
            return def;
        }
        try {
            return Double.parseDouble(s.trim());
        } catch (NumberFormatException e) {
            System.err.println("warning: -D" + name + "=" + s + " is not a number; using " + def);
            return def;
        }
    }

    private ExpertPlacement() {
    }

    /**
     * The share of a layer's picks each of its {@code experts} gets: {@code (count + 1) / Σ (count + 1)}, uniform
     * without counts (a null row or one of another length).
     */
    static double[] shares(long[] counts, int experts) {
        long[] c = counts != null && counts.length == experts ? counts : new long[experts];
        double total = 0;
        for (long v : c) {
            total += v + 1;
        }
        double[] p = new double[experts];
        for (int i = 0; i < experts; i++) {
            p[i] = (c[i] + 1) / total;
        }
        return p;
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
            double[] p = shares(counts != null ? counts[l] : null, e);
            Integer[] idx = new Integer[e];
            for (int i = 0; i < e; i++) {
                p[i] *= topK[l];
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
                // opening the layer: the round trip shared by its k coldest experts, k at most what is still needed;
                // a layer without counts only whole
                boolean known = counts != null && counts[l] != null && counts[l].length == experts[l];
                int maxK = (int) Math.min(experts[l], (need - freed + expertBytes[l] - 1) / expertBytes[l]);
                double sum = SYNC_US;
                int kMin = known ? 1 : experts[l];
                int kMax = known ? maxK : experts[l];
                for (int k = 1; k <= kMax; k++) {
                    sum += cost[l][k - 1];
                    if (k < kMin) {
                        continue;
                    }
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

    /** The planner's cost per token, in microseconds, of the placement {@code cold} under the usage {@code counts}. */
    static double cost(boolean[][] cold, long[] expertBytes, int[] experts, int[] topK, long[][] counts) {
        double us = 0;
        for (int l = 0; l < cold.length; l++) {
            if (cold[l] == null || expertBytes[l] <= 0) {
                continue;
            }
            double[] p = shares(counts != null ? counts[l] : null, experts[l]);
            boolean any = false;
            for (int e = 0; e < experts[l]; e++) {
                if (cold[l][e]) {
                    any = true;
                    us += p[e] * topK[l] * expertBytes[l] / CPU_BYTES_PER_US;
                }
            }
            us += any ? SYNC_US : 0;
        }
        return us;
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

    private static final String HEADER = "# jllm expert usage: <layer>: picks per expert, summed over runs; ";

    /**
     * The saved counts (one row per layer, null for a dense layer or none saved), or null without a file or when the
     * file belongs to another model (its header names the model key; the file name is only a hash of it).
     */
    static long[][] load(Path file, int layers, String modelKey) {
        synchronized (FILE_LOCK) {
            try {
                if (!Files.isRegularFile(file)) {
                    return null;
                }
                List<String> all = Files.readAllLines(file, StandardCharsets.UTF_8);
                if (all.isEmpty() || !all.get(0).equals(HEADER + modelKey)) {
                    return null;
                }
                long[][] c = new long[layers][];
                for (String line : all) {
                    int colon = line.indexOf(':');
                    if (line.startsWith("#") || colon < 0) {
                        continue;
                    }
                    try { // a damaged line costs that layer's history, not the file's
                        int l = Integer.parseInt(line.substring(0, colon).trim());
                        String[] v = line.substring(colon + 1).trim().split(",");
                        long[] row = new long[v.length];
                        for (int i = 0; i < v.length; i++) {
                            row[i] = Math.max(0, Long.parseLong(v[i].trim()));
                        }
                        if (l >= 0 && l < layers) {
                            c[l] = row;
                        }
                    } catch (NumberFormatException e) {
                        System.err.println("warning: skipping a damaged line of " + file);
                    }
                }
                return c;
            } catch (IOException | RuntimeException e) {
                System.err.println("warning: ignoring unreadable expert statistics " + file + ": " + e.getMessage());
                return null;
            }
        }
    }

    /** {@link #add} on the background saver thread. */
    static void addLater(Path file, String modelKey, long[][] delta) {
        SAVER.execute(() -> add(file, modelKey, delta));
    }

    /** Adds {@code delta} to the saved counts (halving a layer's that pass MAX_COUNT) and writes the file atomically. */
    static void add(Path file, String modelKey, long[][] delta) {
        // FILE_LOCK orders this process's backends, the lock file other processes on the same model (a server and a
        // CLI run): each reads, adds and replaces the file whole
        synchronized (FILE_LOCK) {
            try {
                Files.createDirectories(file.getParent());
            } catch (IOException e) {
                System.err.println("warning: could not save expert statistics " + file + ": " + e.getMessage());
                return;
            }
            try (java.nio.channels.FileChannel lockFile = java.nio.channels.FileChannel.open(
                    file.resolveSibling(file.getFileName() + ".lock"), java.nio.file.StandardOpenOption.CREATE,
                    java.nio.file.StandardOpenOption.WRITE);
                 java.nio.channels.FileLock ignored = lockFile.lock()) {
                addLocked(file, modelKey, delta);
            } catch (IOException e) {
                System.err.println("warning: could not save expert statistics " + file + ": " + e.getMessage());
            }
        }
    }

    private static void addLocked(Path file, String modelKey, long[][] delta) throws IOException {
        {
            long[][] c = load(file, delta.length, modelKey);
            if (c == null) {
                c = new long[delta.length][];
            }
            List<String> lines = new ArrayList<>();
            lines.add(HEADER + modelKey);
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
            Path tmp = Files.createTempFile(file.getParent(), file.getFileName().toString(), ".tmp");
            try {
                Files.write(tmp, lines, StandardCharsets.UTF_8);
                Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } finally {
                Files.deleteIfExists(tmp);
            }
        }
    }
}
