package org.robomiri.jllm.kv;

/**
 * The arithmetic of block summaries for sparse decode attention (docs/26), shared by the CPU and Metal backends and by
 * the tests: the per-channel minimum and maximum of a block's keys, stored as IEEE halves rounded <em>outwards</em> (the
 * minimum down, the maximum up) so that the stored box always contains the real keys, the upper bound of
 * {@code q·k} built on it, and the total order in which blocks are ranked.
 */
public final class KvSummary {

    /** The largest finite half; summary values are clamped to {@code [-HALF_MAX, HALF_MAX]} (a key beyond it is not a real key). */
    public static final float HALF_MAX = 65504f;

    private KvSummary() {
    }

    /** The largest half that is {@code <= x} (x clamped to the finite halves; NaN counts as 0). */
    public static short down(float x) {
        x = clamp(x);
        short h = Float.floatToFloat16(x);
        if (Float.float16ToFloat(h) > x) { // rounded up to the nearest half: one step towards -infinity
            int b = h & 0xFFFF;
            h = (short) (b == 0 ? 0x8001 : (b & 0x8000) == 0 ? b - 1 : b + 1);
        }
        return h;
    }

    /** The smallest half that is {@code >= x} (x clamped to the finite halves; NaN counts as 0). */
    public static short up(float x) {
        x = clamp(x);
        short h = Float.floatToFloat16(x);
        if (Float.float16ToFloat(h) < x) { // rounded down: one step towards +infinity
            int b = h & 0xFFFF;
            h = (short) (b == 0x8000 || b == 0 ? 1 : (b & 0x8000) == 0 ? b + 1 : b - 1);
        }
        return h;
    }

    private static float clamp(float x) {
        return x != x ? 0f : Math.max(-HALF_MAX, Math.min(HALF_MAX, x));
    }

    /**
     * The bound of Quest (Tang et al. 2024): for every key {@code k} of a block with per-channel minima {@code lo} and
     * maxima {@code hi}, {@code q·k = Σ q_i k_i <= Σ max(q_i lo_i, q_i hi_i)}, because each {@code k_i} lies in
     * {@code [lo_i, hi_i]} and {@code q_i k_i} is linear in it, so its largest value is at one of the two ends.
     */
    public static float bound(float[] q, int qo, float[] lo, float[] hi, int o, int n) {
        float s = 0f;
        for (int i = 0; i < n; i++) {
            float a = q[qo + i] * lo[o + i];
            float b = q[qo + i] * hi[o + i];
            s += Math.max(a, b);
        }
        return s;
    }

    /** A total order on (score, block) pairs that agrees with the float order of the scores and prefers the lower block on a tie. */
    public static long rank(float score, int block) {
        int bits = Float.floatToRawIntBits(score);
        int key = bits >= 0 ? bits : bits ^ 0x7FFFFFFF; // monotone in the float value (-0 sorts just below +0)
        return ((long) key << 32) | (0xFFFFFFFFL - block);
    }

    /**
     * The {@code k} best of the {@code m} candidates by {@link #rank} (candidate {@code i} is block {@code base + i}),
     * as ascending block numbers: the blocks a decode step attends besides the sinks and the window. When {@code m <= k}
     * all of them.
     */
    public static int[] top(float[] scores, int base, int m, int k) {
        int take = Math.min(k, m);
        long[] r = new long[m];
        for (int i = 0; i < m; i++) {
            r[i] = rank(scores[i], base + i);
        }
        if (take < m) {
            select(r, 0, m - 1, m - take); // r[m - take ..] are the take largest
        }
        int[] out = new int[take];
        for (int i = 0; i < take; i++) {
            out[i] = (int) (0xFFFFFFFFL - (r[m - take + i] & 0xFFFFFFFFL));
        }
        java.util.Arrays.sort(out);
        return out;
    }

    /** Quickselect: afterwards {@code a[kth]} is the element that sorted order puts there, smaller ones before it, larger ones after. */
    private static void select(long[] a, int lo, int hi, int kth) {
        while (lo < hi) {
            long pivot = a[(lo + hi) >>> 1];
            int i = lo;
            int j = hi;
            while (i <= j) {
                while (a[i] < pivot) {
                    i++;
                }
                while (a[j] > pivot) {
                    j--;
                }
                if (i <= j) {
                    long t = a[i];
                    a[i] = a[j];
                    a[j] = t;
                    i++;
                    j--;
                }
            }
            if (kth <= j) {
                hi = j;
            } else if (kth >= i) {
                lo = i;
            } else {
                return;
            }
        }
    }
}
