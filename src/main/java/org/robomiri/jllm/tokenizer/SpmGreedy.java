package org.robomiri.jllm.tokenizer;

import org.robomiri.jllm.json.Json;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * llama.cpp's SentencePiece segmentation ({@code llm_tokenizer_spm}), used for GGUF files with
 * {@code tokenizer.ggml.model = llama}: the text is cut into characters, then the adjacent pair whose concatenation
 * is a token of the vocabulary (of any type) and has the highest score is merged, over and over (equal scores: the
 * leftmost pair first); a character that is no token becomes its {@code <0xXX>} byte tokens. This is not the same
 * as BPE merges ranked by score: tokens of any type take part, and two pairs with equal scores merge left to right
 * instead of in the order of the merge list.
 */
final class SpmGreedy {

    private final Map<String, Integer> ids;
    private final double[] scores;
    /** {@code <0xXX>} (or, failing that, the one-byte string) token of each byte, -1 if the vocabulary has neither. */
    private final int[] byteIds = new int[256];

    SpmGreedy(String[] tokens, double[] scores) {
        this.scores = scores;
        ids = new HashMap<>(tokens.length * 2);
        for (int i = 0; i < tokens.length; i++) {
            ids.put(tokens[i], i); // a repeated text resolves to its last id, like token_to_id[word] = i
        }
        for (int b = 0; b < 256; b++) {
            Integer id = ids.get(String.format("<0x%02X>", b));
            if (id == null && b < 128) {
                id = ids.get(String.valueOf((char) b));
            }
            byteIds[b] = id == null ? -1 : id;
        }
    }

    /** From the {@code model} object GgufTokenizer writes: {@code spm_tokens} (all tokens by id) and {@code spm_scores}. */
    static SpmGreedy fromJson(Map<String, Object> model) {
        List<Object> t = Json.list(model, "spm_tokens");
        List<Object> sc = Json.list(model, "spm_scores");
        if (t == null || sc == null || sc.size() < t.size()) {
            throw new IllegalArgumentException("LlamaSpm model needs spm_tokens and spm_scores");
        }
        String[] tokens = new String[t.size()];
        double[] scores = new double[t.size()];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = (String) t.get(i);
            scores[i] = ((Number) sc.get(i)).doubleValue();
        }
        return new SpmGreedy(tokens, scores);
    }

    private record Bigram(double score, int left, int right, int size) {
    }

    int[] encode(String text) {
        int n = text.codePointCount(0, text.length());
        int[] start = new int[n];
        int[] len = new int[n];
        int[] prev = new int[n];
        int[] next = new int[n];
        for (int i = 0, off = 0; i < n; i++) {
            int cl = Character.charCount(text.codePointAt(off));
            start[i] = off;
            len[i] = cl;
            off += cl;
            prev[i] = i - 1;
            next[i] = i + 1 < n ? i + 1 : -1;
        }
        PriorityQueue<Bigram> queue = new PriorityQueue<>((a, b) -> a.score != b.score
                ? Double.compare(b.score, a.score) : Integer.compare(a.left, b.left));
        for (int i = 1; i < n; i++) {
            offer(queue, text, start, len, i - 1, i);
        }
        while (!queue.isEmpty()) {
            Bigram b = queue.poll();
            if (len[b.left] == 0 || len[b.right] == 0 || len[b.left] + len[b.right] != b.size) {
                continue; // one of the symbols was merged in the meantime
            }
            len[b.left] += len[b.right];
            len[b.right] = 0;
            next[b.left] = next[b.right];
            if (next[b.right] >= 0) {
                prev[next[b.right]] = b.left;
            }
            if (prev[b.left] >= 0) {
                offer(queue, text, start, len, prev[b.left], b.left);
            }
            if (next[b.left] >= 0) {
                offer(queue, text, start, len, b.left, next[b.left]);
            }
        }
        int[] out = new int[text.length() + 8];
        int k = 0;
        for (int i = n == 0 ? -1 : 0; i != -1; i = next[i]) {
            String piece = text.substring(start[i], start[i] + len[i]);
            Integer id = ids.get(piece);
            if (id != null) {
                out = grow(out, k + 1);
                out[k++] = id;
                continue;
            }
            for (byte by : piece.getBytes(StandardCharsets.UTF_8)) { // no token: its bytes
                int bid = byteIds[by & 0xFF];
                if (bid >= 0) {
                    out = grow(out, k + 1);
                    out[k++] = bid;
                }
            }
        }
        return java.util.Arrays.copyOf(out, k);
    }

    private static int[] grow(int[] a, int need) {
        return need <= a.length ? a : java.util.Arrays.copyOf(a, Math.max(need, a.length * 2));
    }

    private void offer(PriorityQueue<Bigram> queue, String text, int[] start, int[] len, int left, int right) {
        Integer id = ids.get(text.substring(start[left], start[right] + len[right]));
        if (id != null) {
            queue.add(new Bigram(scores[id], left, right, len[left] + len[right]));
        }
    }
}
