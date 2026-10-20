package org.robomiri.jllm.engine;

/**
 * Where each n-gram ({@code min <= n <= max} tokens) of a token history last ended, for prompt-lookup speculation:
 * an open-addressing hash table from (n, the n tokens) to the index of the n-gram's last token. Only the most recent
 * occurrence is kept; a hit is checked against the history, so hash collisions only cost a miss.
 *
 * <p>When the history is cut back to a common prefix with the next request ({@link #truncate}), entries pointing
 * past the cut may have overwritten older occurrences that are still valid, so the index is rebuilt over the kept
 * prefix (a few milliseconds for tens of thousands of tokens); a history that is only extended keeps its index.
 */
final class NgramIndex {

    static final int MAX_N = 16;

    private final int min;
    private final int max;
    private long[] keys = new long[1 << 12];
    private int[] ends = new int[1 << 12];
    private int used;
    private int indexed = -1; // the last end added

    NgramIndex(int min, int max) {
        this.min = min;
        this.max = max;
    }

    private static long hash(int[] h, int end, int n) {
        long x = 0x9E3779B97F4A7C15L * (n + 1);
        for (int i = end - n + 1; i <= end; i++) {
            x = (x ^ (h[i] + 1)) * 0xBF58476D1CE4E5B9L;
            x ^= x >>> 31;
        }
        return x == 0 ? 1 : x; // 0 marks an empty slot
    }

    /** Indexes the n-grams that end at {@code end} (all tokens up to it are in {@code h}). */
    void add(int[] h, int end) {
        indexed = Math.max(indexed, end);
        for (int n = min; n <= max && n <= end + 1; n++) {
            put(hash(h, end, n), end);
        }
    }

    /**
     * The end of the latest earlier occurrence of the n-gram {@code h[end-n+1 .. end]} (before {@code end - n + 1}
     * is not required: overlapping repeats such as "a a a" are fine), or -1.
     */
    int find(int[] h, int end, int n) {
        if (end - n + 1 < 0) {
            return -1;
        }
        long k = hash(h, end, n);
        int mask = keys.length - 1;
        for (int i = (int) (k ^ (k >>> 32)) & mask; keys[i] != 0; i = (i + 1) & mask) {
            if (keys[i] == k) {
                int e = ends[i];
                if (e >= end) {
                    return -1;
                }
                for (int j = 0; j < n; j++) {
                    if (e - j < 0 || h[e - j] != h[end - j]) {
                        return -1; // a collision
                    }
                }
                return e;
            }
        }
        return -1;
    }

    /** The history now has {@code len} tokens, a prefix of what was indexed. */
    void truncate(int[] h, int len) {
        if (len - 2 >= indexed) {
            return; // nothing past the cut was indexed
        }
        java.util.Arrays.fill(keys, 0);
        used = 0;
        indexed = -1;
        for (int end = 0; end <= len - 2; end++) { // as onToken does: the n-grams ending one token back
            add(h, end);
        }
    }

    private void put(long k, int end) {
        if (2 * (used + 1) > keys.length) {
            grow();
        }
        int mask = keys.length - 1;
        int i = (int) (k ^ (k >>> 32)) & mask;
        while (keys[i] != 0 && keys[i] != k) {
            i = (i + 1) & mask;
        }
        if (keys[i] == 0) {
            used++;
        }
        keys[i] = k;
        ends[i] = end;
    }

    private void grow() {
        long[] ok = keys;
        int[] oe = ends;
        keys = new long[ok.length * 2];
        ends = new int[ok.length * 2];
        used = 0;
        for (int i = 0; i < ok.length; i++) {
            if (ok[i] != 0) {
                put(ok[i], oe[i]);
            }
        }
    }
}
