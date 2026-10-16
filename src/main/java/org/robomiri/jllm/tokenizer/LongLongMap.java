package org.robomiri.jllm.tokenizer;

/**
 * Open-addressed {@code long -> long} map without boxing, for the merge table keyed by {@code (left << 32) | right}.
 * {@code Long.hashCode} folds such a key to {@code left ^ right}, which collides massively (every pair with the
 * same xor shares a bin, and {@code HashMap} then treeifies a large share of its bins); the slot index here comes
 * from a 64-bit multiplicative mix of the whole key instead. Values are non-negative; {@link #MISSING} marks an
 * empty slot. Linear probing, load factor at most 1/2, grows by doubling.
 */
final class LongLongMap {

    static final long MISSING = -1L;

    private long[] keys;
    private long[] values;
    private int mask;
    private int size;

    LongLongMap(int expected) {
        int cap = 16;
        while (cap < expected * 2) {
            cap <<= 1;
        }
        allocate(cap);
    }

    private void allocate(int cap) {
        keys = new long[cap];
        values = new long[cap];
        java.util.Arrays.fill(values, MISSING);
        mask = cap - 1;
    }

    private static int slot(long key, int mask) {
        long h = key * 0x9E3779B97F4A7C15L;
        return (int) (h ^ (h >>> 32)) & mask;
    }

    /** @return the value for {@code key}, or {@link #MISSING} */
    long get(long key) {
        long[] ks = keys;
        long[] vs = values;
        int m = mask;
        int i = slot(key, m);
        while (true) {
            long v = vs[i];
            if (v == MISSING || ks[i] == key) {
                return v;
            }
            i = (i + 1) & m;
        }
    }

    /** Stores {@code value} (which must be non-negative) unless {@code key} is already present. */
    void putIfAbsent(long key, long value) {
        if (value < 0) {
            throw new IllegalArgumentException("negative value " + value);
        }
        if ((size + 1) * 2 > keys.length) {
            grow();
        }
        int i = slot(key, mask);
        while (values[i] != MISSING) {
            if (keys[i] == key) {
                return;
            }
            i = (i + 1) & mask;
        }
        keys[i] = key;
        values[i] = value;
        size++;
    }

    int size() {
        return size;
    }

    private void grow() {
        long[] ok = keys;
        long[] ov = values;
        allocate(ok.length * 2);
        for (int i = 0; i < ok.length; i++) {
            if (ov[i] != MISSING) {
                int j = slot(ok[i], mask);
                while (values[j] != MISSING) {
                    j = (j + 1) & mask;
                }
                keys[j] = ok[i];
                values[j] = ov[i];
            }
        }
    }
}
