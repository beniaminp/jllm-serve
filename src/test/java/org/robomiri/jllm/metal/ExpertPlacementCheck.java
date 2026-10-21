package org.robomiri.jllm.metal;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** {@link ExpertPlacement#plan} on small cases with known answers and on random usage (its invariants). */
public final class ExpertPlacementCheck {

    private ExpertPlacementCheck() {
    }

    private static int cold(boolean[] row) {
        int n = 0;
        for (boolean b : row) {
            n += b ? 1 : 0;
        }
        return n;
    }

    /** @return descriptions of the failed checks */
    public static List<String> run() {
        List<String> bad = new ArrayList<>();
        long[] eb = {0, 100, 100, 100, 100}; // layer 0 dense
        int[] ex = {0, 8, 8, 8, 8};
        int[] topK = {0, 2, 2, 2, 2};

        boolean[][] none = ExpertPlacement.plan(0, eb, ex, topK, null);
        for (int l = 1; l < 5; l++) {
            if (cold(none[l]) != 0) {
                bad.add("need 0 moved experts of layer " + l);
            }
        }
        if (none[0] != null) {
            bad.add("a dense layer got a placement");
        }

        // no statistics: uniform, so one layer is filled before another is opened (whole layers, fewest round trips)
        boolean[][] u = ExpertPlacement.plan(250, eb, ex, topK, null);
        int touched = 0;
        int moved = 0;
        for (int l = 1; l < 5; l++) {
            touched += cold(u[l]) > 0 ? 1 : 0;
            moved += cold(u[l]);
        }
        if (touched != 1 || moved != 3) {
            bad.add("uniform, 250 bytes: expected 3 experts of one layer, got " + moved + " in " + touched + " layers");
        }
        boolean[][] u2 = ExpertPlacement.plan(1000, eb, ex, topK, null);
        int full = 0;
        moved = 0;
        for (int l = 1; l < 5; l++) {
            full += cold(u2[l]) == 8 ? 1 : 0;
            moved += cold(u2[l]);
        }
        if (full != 1 || moved != 10) {
            bad.add("uniform, 1000 bytes: expected one whole layer + 2 experts, got " + full + " whole, " + moved + " experts");
        }

        // skewed: layer 1's experts 6 and 7 are never picked, the other layers are uniform -> those two go
        long[][] counts = new long[5][];
        counts[1] = new long[]{1000, 1000, 1000, 1000, 1000, 1000, 0, 0};
        for (int l = 2; l < 5; l++) {
            counts[l] = new long[]{100, 100, 100, 100, 100, 100, 100, 100};
        }
        boolean[][] s = ExpertPlacement.plan(200, eb, ex, topK, counts);
        if (!s[1][6] || !s[1][7] || cold(s[1]) != 2 || cold(s[2]) + cold(s[3]) + cold(s[4]) != 0) {
            bad.add("skewed: expected experts 6 and 7 of layer 1 only");
        }

        // random usage: frees at least what is asked (or everything), and within a layer only the least used go
        Random r = new Random(3);
        for (int trial = 0; trial < 200; trial++) {
            int layers = 1 + r.nextInt(6);
            long[] b = new long[layers];
            int[] e = new int[layers];
            int[] k = new int[layers];
            long[][] c = new long[layers][];
            long total = 0;
            for (int l = 0; l < layers; l++) {
                if (r.nextInt(5) == 0) {
                    continue; // dense
                }
                e[l] = 2 + r.nextInt(30);
                k[l] = 1 + r.nextInt(e[l]);
                b[l] = 1 + r.nextInt(1000);
                total += b[l] * e[l];
                if (r.nextBoolean()) {
                    c[l] = new long[e[l]];
                    for (int i = 0; i < e[l]; i++) {
                        c[l][i] = r.nextInt(3) == 0 ? 0 : r.nextInt(100000);
                    }
                }
            }
            long need = (long) (r.nextDouble() * 1.2 * total);
            boolean[][] p = ExpertPlacement.plan(need, b, e, k, c);
            long freed = 0;
            for (int l = 0; l < layers; l++) {
                if (b[l] == 0) {
                    continue;
                }
                for (int i = 0; i < e[l]; i++) {
                    freed += p[l][i] ? b[l] : 0;
                    for (int j = 0; j < e[l] && c[l] != null; j++) {
                        if (p[l][i] && !p[l][j] && c[l][i] > c[l][j]) {
                            bad.add("trial " + trial + ": layer " + l + " moved expert " + i + " (" + c[l][i]
                                    + " picks) but kept " + j + " (" + c[l][j] + ")");
                        }
                    }
                }
            }
            if (freed < Math.min(need, total)) {
                bad.add("trial " + trial + ": freed " + freed + " of " + need + " needed");
            }
        }
        return bad;
    }
}
