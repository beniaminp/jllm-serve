package org.robomiri.jllm.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;

/**
 * The SentencePiece unigram model ({@code "model": {"type": "Unigram"}}): every vocabulary piece has a score, the
 * log-probability of the piece, and a word is segmented into the pieces whose scores have the largest sum
 * (the most probable segmentation), found by a Viterbi pass over the lattice of all pieces that occur in the
 * word. This reproduces Hugging Face's {@code Unigram::encode_optimized} exactly, including its tie-breaking
 * (the first best path found wins; ends are relaxed in order of increasing start, shorter pieces first), the
 * score of an unknown character ({@code min score - 10}) and the fusing of runs of unknown characters into one
 * {@code unk} (always on for this model).
 */
final class Unigram {

    /** Hugging Face's {@code K_UNK_PENALTY}: an unknown character scores this much below the worst piece. */
    private static final double UNK_PENALTY = 10.0;

    /** Child links of the piece trie: key = (node << 16 | UTF-16 char), value = child node. Node 0 is the root. */
    private final LongLongMap children;
    /** Token id ending at each trie node, or -1. */
    private int[] terminal;
    private final double[] scores;
    private final double unkScore;
    private final int unkId;

    /**
     * @param pieces the vocabulary in id order
     * @param scores the log-probability of each piece
     * @param ids    piece -> id as the tokenizer resolves it (for duplicate pieces the last one, like the reference)
     */
    Unigram(String[] pieces, double[] scores, Map<String, Integer> ids, int unkId) {
        this.scores = scores;
        this.unkId = unkId;
        double min = Double.MAX_VALUE;
        for (double s : scores) {
            min = Math.min(min, s);
        }
        unkScore = min - UNK_PENALTY;
        children = new LongLongMap(pieces.length * 4);
        terminal = new int[Math.max(16, pieces.length * 2)];
        Arrays.fill(terminal, -1);
        int nodes = 1;
        for (String p : pieces) {
            if (p == null || p.isEmpty()) {
                continue;
            }
            int node = 0;
            for (int i = 0; i < p.length(); i++) {
                long key = ((long) node << 16) | p.charAt(i);
                long child = children.get(key);
                if (child == LongLongMap.MISSING) {
                    child = nodes++;
                    children.putIfAbsent(key, child);
                    if (nodes > terminal.length) {
                        int old = terminal.length;
                        terminal = Arrays.copyOf(terminal, old * 2);
                        Arrays.fill(terminal, old, terminal.length, -1);
                    }
                }
                node = (int) child;
            }
            terminal[node] = ids.get(p);
        }
    }

    /**
     * Viterbi segmentation of one pre-tokenized piece.
     *
     * @return the pieces of the best path as (start, end, id) triples flattened, in order; runs of unknown
     *         characters are fused into one triple with id {@code unkId}
     */
    int[] segment(String s) {
        int n = s.length();
        // best[e]: score of the best path covering s[0, e); from[e]: start of its last piece; id[e]: that piece
        double[] best = new double[n + 1];
        int[] from = new int[n + 1];
        int[] id = new int[n + 1];
        Arrays.fill(from, -1);
        for (int start = 0; start < n; ) {
            int cl = Character.charCount(s.codePointAt(start));
            double base = best[start];
            boolean single = false;
            int node = 0;
            for (int i = start; i < n; i++) {
                long child = children.get(((long) node << 16) | s.charAt(i));
                if (child == LongLongMap.MISSING) {
                    break;
                }
                node = (int) child;
                int t = terminal[node];
                if (t < 0) {
                    continue;
                }
                int end = i + 1;
                double cand = scores[t] + base;
                if (from[end] < 0 || cand > best[end]) {
                    best[end] = cand;
                    from[end] = start;
                    id[end] = t;
                }
                if (end - start == cl) {
                    single = true;
                }
            }
            if (!single) { // no piece covers this character: it is unknown
                int end = start + cl;
                double cand = unkScore + base;
                if (from[end] < 0 || cand > best[end]) {
                    best[end] = cand;
                    from[end] = start;
                    id[end] = unkId;
                }
            }
            start += cl;
        }
        // backtrack, fusing adjacent unknowns (the reference's fuse_unk is always on for Unigram)
        int[] out = new int[3 * n];
        int k = out.length;
        int end = n;
        while (end > 0) {
            int start = from[end];
            int t = id[end];
            if (unkId >= 0 && t == unkId && k < out.length && out[k + 2] == unkId && out[k] == end) {
                out[k] = start; // extend the unknown run that follows
            } else {
                out[--k] = t;
                out[--k] = end;
                out[--k] = start;
            }
            end = start;
        }
        return Arrays.copyOfRange(out, k, out.length);
    }

    /**
     * Ids of one pre-tokenized piece: the Viterbi pieces; a fused unknown run is looked up as a string (it may
     * happen to be a piece), else spelled as {@code <0xXX>} byte tokens when byte fallback is on and all of them
     * exist, else it is {@code unkId}. Without an unknown token the run is dropped (the reference raises an error).
     */
    int[] encode(String s, Map<String, Integer> vocab, boolean byteFallback, int[] byteTokenIds) {
        int[] seg = segment(s);
        Tokenizer.IntList out = new Tokenizer.IntList(seg.length / 3 + 1);
        for (int i = 0; i < seg.length; i += 3) {
            int t = seg[i + 2];
            if (t != unkId) {
                out.add(t);
                continue;
            }
            String text = s.substring(seg[i], seg[i + 1]);
            Integer known = vocab.get(text);
            if (known != null) {
                out.add(known);
                continue;
            }
            if (byteFallback) {
                byte[] bs = text.getBytes(StandardCharsets.UTF_8);
                boolean all = true;
                for (byte b : bs) {
                    all &= byteTokenIds[b & 0xFF] >= 0;
                }
                if (all) {
                    for (byte b : bs) {
                        out.add(byteTokenIds[b & 0xFF]);
                    }
                    continue;
                }
            }
            if (unkId >= 0) {
                out.add(unkId);
            }
        }
        return out.toArray();
    }
}
