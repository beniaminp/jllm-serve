package org.robomiri.jllm.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.IntPredicate;

/**
 * Leftmost-longest matcher for the added tokens of a tokenizer: a trie over UTF-16 code units, indexed by first
 * character so that the common case (a character that starts no added token) costs one array read. Replaces a
 * regex alternation over every added token, which scales with the number of tokens (6 415 for Gemma 3) at every
 * position of the text.
 */
final class AddedTokenTrie {

    /** Characters that start at least one added token (UTF-16 units; supplementary characters go via surrogates). */
    private final boolean[] starts = new boolean[Character.MAX_VALUE + 1];
    /** Per node: outgoing edge labels (ascending) and target nodes; {@code term[node]} is the token id or -1. */
    private final char[][] keys;
    private final int[][] next;
    private final int[] term;

    private AddedTokenTrie(char[][] keys, int[][] next, int[] term) {
        this.keys = keys;
        this.next = next;
        this.term = term;
        for (char c : keys[0]) {
            starts[c] = true;
        }
    }

    /** @return the trie, or null when there is no non-empty token to match */
    static AddedTokenTrie build(Map<String, Integer> tokens) {
        List<TreeMap<Character, Integer>> children = new ArrayList<>();
        List<Integer> ids = new ArrayList<>();
        children.add(new TreeMap<>());
        ids.add(-1);
        for (Map.Entry<String, Integer> e : tokens.entrySet()) {
            String s = e.getKey();
            if (s.isEmpty()) {
                continue;
            }
            int node = 0;
            for (int i = 0; i < s.length(); i++) {
                Integer child = children.get(node).get(s.charAt(i));
                if (child == null) {
                    child = children.size();
                    children.get(node).put(s.charAt(i), child);
                    children.add(new TreeMap<>());
                    ids.add(-1);
                }
                node = child;
            }
            ids.set(node, e.getValue());
        }
        if (children.get(0).isEmpty()) {
            return null;
        }
        int n = children.size();
        char[][] keys = new char[n][];
        int[][] next = new int[n][];
        int[] term = new int[n];
        for (int i = 0; i < n; i++) {
            TreeMap<Character, Integer> ch = children.get(i);
            keys[i] = new char[ch.size()];
            next[i] = new int[ch.size()];
            int k = 0;
            for (Map.Entry<Character, Integer> e : ch.entrySet()) {
                keys[i][k] = e.getKey();
                next[i][k++] = e.getValue();
            }
            term[i] = ids.get(i);
        }
        return new AddedTokenTrie(keys, next, term);
    }

    /** True when some added token starts with {@code c} (cheap pre-check for the scan loop). */
    boolean mayStart(char c) {
        return starts[c];
    }

    /**
     * Longest added token starting at {@code from} whose id satisfies {@code allowed}.
     *
     * @return {@code (length << 32) | id}, or -1 when none matches
     */
    long longest(CharSequence s, int from, IntPredicate allowed) {
        int node = 0;
        long best = -1;
        int len = s.length();
        for (int i = from; i < len; i++) {
            node = child(node, s.charAt(i));
            if (node < 0) {
                break;
            }
            int id = term[node];
            if (id >= 0 && allowed.test(id)) {
                best = ((long) (i - from + 1) << 32) | id;
            }
        }
        return best;
    }

    private int child(int node, char c) {
        char[] k = keys[node];
        int[] nx = next[node];
        int n = k.length;
        if (n <= 8) {
            for (int i = 0; i < n; i++) {
                if (k[i] == c) {
                    return nx[i];
                }
            }
            return -1;
        }
        int lo = 0;
        int hi = n - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (k[mid] < c) {
                lo = mid + 1;
            } else if (k[mid] > c) {
                hi = mid - 1;
            } else {
                return nx[mid];
            }
        }
        return -1;
    }
}
