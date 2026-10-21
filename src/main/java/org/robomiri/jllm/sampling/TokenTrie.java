package org.robomiri.jllm.sampling;

import java.util.Arrays;
import java.util.Map;
import java.util.WeakHashMap;

/**
 * The vocabulary as a byte trie: one node per distinct token prefix, the tokens hanging off the node of their last
 * byte. Walking it alongside a byte automaton ({@link #allowed}) computes the set of allowed tokens while testing
 * every shared prefix once and skipping whole subtrees at the first byte the automaton refuses: in a structural JSON
 * position (after a key, between values) only the few subtrees under the handful of allowed first bytes are visited,
 * instead of 150 000 tokens fed one by one.
 *
 * <p>Layout: first-child / next-sibling arrays (children in byte order), built from the tokens sorted by their bytes.
 * Special tokens (null in the token table) are not in the trie.
 */
final class TokenTrie {

    /** A byte automaton the trie can be walked with. */
    interface ByteState {
        /** An independent copy. */
        ByteState copy();

        /** Feeds one byte (0..255); false when it is not allowed (the state is then undefined). */
        boolean feed(int b);
    }

    private static final Map<byte[][], TokenTrie> CACHE = new WeakHashMap<>();

    private int[] first; // first child, -1 = none
    private int[] next;  // next sibling, -1 = none
    private byte[] edge; // the byte leading into the node
    private int[] tokStart; // tokens ending at the node: sorted[tokStart .. tokStart + tokCount)
    private int[] tokCount;
    private int nodes;
    private final int[] sorted;

    /** The trie of a token table ({@link JsonConstraint#tokenTable}), built once per table. */
    static TokenTrie of(byte[][] table) {
        synchronized (CACHE) {
            return CACHE.computeIfAbsent(table, TokenTrie::new);
        }
    }

    private TokenTrie(byte[][] table) {
        Integer[] ids = new Integer[table.length];
        int n = 0;
        for (int i = 0; i < table.length; i++) {
            if (table[i] != null) {
                ids[n++] = i;
            }
        }
        Integer[] order = Arrays.copyOf(ids, n);
        Arrays.sort(order, (a, b) -> Arrays.compareUnsigned(table[a], table[b]));
        sorted = new int[n];
        int cap = 1 << 16;
        first = new int[cap];
        next = new int[cap];
        edge = new byte[cap];
        tokStart = new int[cap];
        tokCount = new int[cap];
        nodes = 1;
        first[0] = -1;
        next[0] = -1;
        int[] path = new int[256]; // path[d] = node at depth d of the previous token
        int[] last = new int[256]; // last child created under path[d]
        Arrays.fill(last, -1);
        byte[] prev = new byte[0];
        int depthPrev = 0;
        for (int k = 0; k < n; k++) {
            int id = order[k];
            sorted[k] = id;
            byte[] b = table[id];
            int common = Arrays.mismatch(prev, b);
            if (common < 0) {
                common = prev.length; // identical bytes: the same node
            }
            if (b.length >= path.length) {
                path = Arrays.copyOf(path, b.length * 2);
                int old = last.length;
                last = Arrays.copyOf(last, b.length * 2);
                Arrays.fill(last, old, last.length, -1);
            }
            for (int d = common + 1; d <= depthPrev; d++) {
                last[d] = -1; // the children of the abandoned path start over
            }
            for (int d = common; d < b.length; d++) {
                int node = newNode(b[d]);
                int parent = path[d];
                if (last[d] < 0) {
                    first[parent] = node;
                } else {
                    next[last[d]] = node;
                }
                last[d] = node;
                last[d + 1] = -1;
                path[d + 1] = node;
            }
            int at = path[b.length];
            if (tokCount[at]++ == 0) {
                tokStart[at] = k;
            }
            prev = b;
            depthPrev = b.length;
        }
    }

    private int newNode(byte b) {
        if (nodes == first.length) {
            int cap = nodes * 2;
            first = Arrays.copyOf(first, cap);
            next = Arrays.copyOf(next, cap);
            edge = Arrays.copyOf(edge, cap);
            tokStart = Arrays.copyOf(tokStart, cap);
            tokCount = Arrays.copyOf(tokCount, cap);
        }
        int n = nodes++;
        first[n] = -1;
        next[n] = -1;
        edge[n] = b;
        return n;
    }

    int nodeCount() {
        return nodes;
    }

    /** Whether some token is longer than {@code bytes} and starts with them (the model could extend them). */
    boolean extendable(byte[] bytes) {
        int node = 0;
        for (byte b : bytes) {
            int c = first[node];
            while (c >= 0 && edge[c] != b) {
                c = next[c];
            }
            if (c < 0) {
                return false;
            }
            node = c;
        }
        return first[node] >= 0;
    }

    /**
     * Marks in {@code mask} every token whose bytes {@code start} accepts. {@code start} is consumed (copies are
     * made only where the walk branches).
     *
     * @return the number of trie nodes visited
     */
    int allowed(ByteState start, long[] mask) {
        return walk(start, 0, mask);
    }

    private int walk(ByteState s, int node, long[] mask) {
        int visited = 0;
        for (int c = first[node]; c >= 0; c = next[c]) {
            ByteState t = next[c] < 0 ? s : s.copy(); // the last child may use up the parent's state
            visited++;
            if (!t.feed(edge[c] & 0xff)) {
                continue;
            }
            for (int k = tokStart[c], e = k + tokCount[c]; k < e; k++) {
                int id = sorted[k];
                mask[id >>> 6] |= 1L << id;
            }
            if (first[c] >= 0) {
                visited += walk(t, c, mask);
            }
        }
        return visited;
    }
}
