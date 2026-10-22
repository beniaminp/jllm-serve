package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The SentencePiece <em>Unigram</em> tokenizer of T5 / UMT5 (Wan's text encoder), read from a Hugging Face {@code tokenizer.json}:
 * the text's runs of spaces are collapsed ({@code Replace " {2,}" → " "}), every word gets the marker {@code ▁} in front of it
 * (the {@code Metaspace} pre-tokenizer with {@code prepend_scheme = always}, one piece per word), each word is cut into vocabulary
 * pieces by the segmentation of highest total log-probability (Viterbi over the piece scores), characters that no piece covers
 * become the unknown token (consecutive ones fused), and the end-of-sequence token {@code </s>} follows.
 *
 * <p>This is the tokenizer.json of {@code google/umt5-xxl} as shipped with Wan: no Unicode normalisation table (the older
 * {@code spiece.model} has one; the fast tokenizer file does not). The special tokens such as {@code <extra_id_0>} are not looked for
 * inside the text.
 */
public final class UnigramTokenizer {

    private static final char MARK = '▁';

    private final Map<String, int[]> ids = new HashMap<>(); // piece -> {id}
    private final Map<String, Double> scores = new HashMap<>();
    private final int unk;
    private final int eos;
    private final int maxPiece;
    private final double unkScore;

    public UnigramTokenizer(Path tokenizerJson) throws IOException {
        Map<String, Object> root = Json.parseObject(Files.readString(tokenizerJson, StandardCharsets.UTF_8));
        Map<String, Object> model = Json.map(root, "model");
        if (!"Unigram".equals(Json.str(model, "type", ""))) {
            throw new IOException(tokenizerJson + " is not a Unigram tokenizer");
        }
        List<Object> vocab = Json.list(model, "vocab");
        double min = 0;
        int max = 1;
        for (int i = 0; i < vocab.size(); i++) {
            List<?> e = (List<?>) vocab.get(i);
            String piece = (String) e.get(0);
            double score = ((Number) e.get(1)).doubleValue();
            ids.putIfAbsent(piece, new int[]{i});
            scores.putIfAbsent(piece, score);
            if (i >= 4) { // the special tokens have score 0 and are never produced by segmentation
                min = Math.min(min, score);
            }
            max = Math.max(max, piece.codePointCount(0, piece.length()));
        }
        unk = (int) Json.num(model, "unk_id", 3);
        eos = 1;
        maxPiece = max;
        unkScore = min - 10.0;
    }

    public int eos() {
        return eos;
    }

    /** Token ids of {@code text} followed by {@code </s>}. */
    public int[] encode(String text) {
        String t = text.replaceAll(" {2,}", " ");
        List<Integer> out = new ArrayList<>();
        if (t.isEmpty()) {
            out.add(eos);
            return new int[]{eos};
        }
        // Metaspace with split: every word is "▁" + the characters up to the next space
        StringBuilder word = new StringBuilder();
        List<String> words = new ArrayList<>();
        String s = t.replace(' ', MARK);
        if (s.charAt(0) != MARK) { // prepend_scheme "always" adds the marker unless the text already starts with a space
            s = MARK + s;
        }
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            if (cp == MARK && word.length() > 0) {
                words.add(word.toString());
                word.setLength(0);
            }
            word.appendCodePoint(cp);
            i += Character.charCount(cp);
        }
        if (word.length() > 0) {
            words.add(word.toString());
        }
        for (String w : words) {
            segment(w, out);
        }
        out.add(eos);
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Viterbi segmentation of one word. */
    private void segment(String w, List<Integer> out) {
        int[] cps = w.codePoints().toArray();
        int n = cps.length;
        int[] offs = new int[n + 1]; // char offsets of the code points
        for (int i = 0; i < n; i++) {
            offs[i + 1] = offs[i] + Character.charCount(cps[i]);
        }
        double[] best = new double[n + 1];
        int[] from = new int[n + 1];
        int[] piece = new int[n + 1]; // id of the piece ending here, -1 for an unknown character
        java.util.Arrays.fill(best, Double.NEGATIVE_INFINITY);
        best[0] = 0;
        for (int b = 0; b < n; b++) {
            if (best[b] == Double.NEGATIVE_INFINITY) {
                continue;
            }
            boolean any = false;
            for (int len = 1; len <= Math.min(maxPiece, n - b); len++) {
                String p = w.substring(offs[b], offs[b + len]);
                int[] id = ids.get(p);
                if (id == null) {
                    continue;
                }
                any |= len == 1;
                double sc = best[b] + scores.get(p);
                if (sc > best[b + len]) {
                    best[b + len] = sc;
                    from[b + len] = b;
                    piece[b + len] = id[0];
                }
            }
            if (!any) { // no piece covers this character: an unknown token for it
                double sc = best[b] + unkScore;
                if (sc > best[b + 1]) {
                    best[b + 1] = sc;
                    from[b + 1] = b;
                    piece[b + 1] = -1;
                }
            }
        }
        List<Integer> rev = new ArrayList<>();
        for (int e = n; e > 0; e = from[e]) {
            rev.add(piece[e]);
        }
        boolean lastUnk = false;
        for (int i = rev.size() - 1; i >= 0; i--) {
            int id = rev.get(i);
            if (id < 0) {
                if (!lastUnk) {
                    out.add(unk);
                }
                lastUnk = true;
            } else {
                out.add(id);
                lastUnk = false;
            }
        }
    }
}
