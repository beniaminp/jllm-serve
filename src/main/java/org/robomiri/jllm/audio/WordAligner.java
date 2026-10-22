package org.robomiri.jllm.audio;

import org.robomiri.jllm.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Word timestamps from Whisper's cross-attention, the method of OpenAI's {@code whisper.timing}: a few decoder heads (the
 * checkpoint's {@code alignment_heads}) attend to the audio position a token is about to be written for. Their attention
 * over the window's frames is renormalised over the audio that is there, standardised across tokens, median filtered
 * (width 7) along time and averaged over the heads; dynamic time warping finds the monotonic path through that
 * token-by-time matrix, and the time at which the path enters a token's row is that token's start. Words are runs of tokens
 * (a new word starts at a token with a leading space or at punctuation); punctuation is merged into the neighbouring
 * word; over-long words at pauses and segment edges are trimmed as in OpenAI's {@code add_word_timestamps}.
 */
public final class WordAligner {

    private WordAligner() {
    }

    /** A word with its tokens and times in seconds from the window start (the caller adds the window's offset). */
    public static final class Timing {
        public String word;
        public int[] tokens;
        public double start;
        public double end;

        Timing(String word, int[] tokens, double start, double end) {
            this.word = word;
            this.tokens = tokens;
            this.start = start;
            this.end = end;
        }
    }

    static final int MEDIAN_WIDTH = 7;
    static final double TOKENS_PER_SECOND = 50.0;     // one encoder position = 20 ms
    private static final Set<String> CJK = Set.of("zh", "ja", "th", "lo", "my", "yue");
    private static final String PREPEND = "\"'“¿([{-";
    private static final String APPEND = "\"'.。,，!！?？:：”)]}、";

    /**
     * @param weights  per alignment head: the cross-attention probabilities [rows][positions] of all {@code rows} tokens
     *                 of {@code [sot sequence..., notimestamps, text..., eot]}
     * @param rows     number of tokens fed
     * @param positions encoder positions per row in {@code weights}
     * @param used     encoder positions that hold audio (the window's frames / 2)
     * @param skip     tokens before the first text token's predictor row: the sot sequence length
     * @param text     the text tokens
     * @param language the language code (decides how words are split)
     */
    public static List<Timing> align(float[][] weights, int rows, int positions, int used, int skip, int[] text, String language,
                                     Tokenizer tok, int eot) {
        int n = text.length;
        if (n == 0 || used < 2) {
            return List.of();
        }
        int h = weights.length;
        int cols = used;
        // renormalise over the used positions, then standardise over the token axis per column, then median filter
        float[][] filtered = new float[h][rows * cols];
        for (int k = 0; k < h; k++) {
            float[] w = weights[k];
            float[] m = filtered[k];
            for (int r = 0; r < rows; r++) {
                double sum = 0;
                for (int c = 0; c < cols; c++) {
                    sum += w[r * positions + c];
                }
                double inv = sum > 0 ? 1.0 / sum : 0;
                for (int c = 0; c < cols; c++) {
                    m[r * cols + c] = (float) (w[r * positions + c] * inv);
                }
            }
            for (int c = 0; c < cols; c++) {
                double mean = 0;
                for (int r = 0; r < rows; r++) {
                    mean += m[r * cols + c];
                }
                mean /= rows;
                double var = 0;
                for (int r = 0; r < rows; r++) {
                    double d = m[r * cols + c] - mean;
                    var += d * d;
                }
                double std = Math.sqrt(var / rows);
                for (int r = 0; r < rows; r++) {
                    m[r * cols + c] = std > 0 ? (float) ((m[r * cols + c] - mean) / std) : 0f;
                }
            }
            medianFilter(m, rows, cols);
        }
        // matrix rows skip .. skip + n (the n text tokens' predictors and the eot's), mean over heads
        int nr = n + 1;
        double[][] cost = new double[nr][cols];
        for (int r = 0; r < nr; r++) {
            for (int c = 0; c < cols; c++) {
                double s = 0;
                for (int k = 0; k < h; k++) {
                    s += filtered[k][(skip + r) * cols + c];
                }
                cost[r][c] = -s / h;
            }
        }
        int[][] path = dtw(cost);
        int[] textIdx = path[0];
        int[] timeIdx = path[1];
        // time of each row's first appearance on the path
        double[] jump = new double[nr];
        Arrays.fill(jump, -1);
        int last = -1;
        for (int i = 0; i < textIdx.length; i++) {
            if (textIdx[i] != last) {
                jump[textIdx[i]] = timeIdx[i] / TOKENS_PER_SECOND;
                last = textIdx[i];
            }
        }
        // words
        int[] withEot = Arrays.copyOf(text, n + 1);
        withEot[n] = eot;
        List<int[]> wordTokens = new ArrayList<>();
        List<String> words = new ArrayList<>();
        splitWords(withEot, language, tok, eot, words, wordTokens);
        if (words.size() <= 1) {
            return List.of();
        }
        List<Timing> out = new ArrayList<>();
        int boundary = 0;
        for (int w = 0; w < words.size() - 1; w++) {
            int next = boundary + wordTokens.get(w).length;
            out.add(new Timing(words.get(w), wordTokens.get(w), jump[boundary], jump[next]));
            boundary = next;
        }
        return out;
    }

    /** The window's words with OpenAI's duration clean-up and punctuation merging; times relative to the window start. */
    public static void tidy(List<Timing> alignment) {
        double[] durations = alignment.stream().mapToDouble(t -> t.end - t.start).filter(d -> d != 0).sorted().toArray();
        double median = 0;
        if (durations.length > 0) {
            median = durations.length % 2 == 1 ? durations[durations.length / 2]
                    : (durations[durations.length / 2 - 1] + durations[durations.length / 2]) / 2;
        }
        median = Math.min(0.7, median);
        double max = median * 2;
        if (durations.length > 0) {
            String marks = ".。!！?？";
            for (int i = 1; i < alignment.size(); i++) {
                Timing t = alignment.get(i);
                if (t.end - t.start > max) {
                    if (marks.contains(t.word) && !t.word.isEmpty()) {
                        t.end = t.start + max;
                    } else if (!alignment.get(i - 1).word.isEmpty() && marks.contains(alignment.get(i - 1).word)) {
                        t.start = t.end - max;
                    }
                }
            }
        }
        mergePunctuations(alignment);
    }

    static void mergePunctuations(List<Timing> a) {
        int i = a.size() - 2;
        int j = a.size() - 1;
        while (i >= 0) {
            Timing prev = a.get(i);
            Timing fol = a.get(j);
            if (prev.word.startsWith(" ") && PREPEND.contains(prev.word.strip()) && !prev.word.strip().isEmpty()) {
                fol.word = prev.word + fol.word;
                fol.tokens = concat(prev.tokens, fol.tokens);
                prev.word = "";
                prev.tokens = new int[0];
            } else {
                j = i;
            }
            i--;
        }
        i = 0;
        j = 1;
        while (j < a.size()) {
            Timing prev = a.get(i);
            Timing fol = a.get(j);
            if (!prev.word.endsWith(" ") && !fol.word.isEmpty() && APPEND.contains(fol.word)) {
                prev.word = prev.word + fol.word;
                prev.tokens = concat(prev.tokens, fol.tokens);
                fol.word = "";
                fol.tokens = new int[0];
            } else {
                i = j;
            }
            j++;
        }
    }

    private static int[] concat(int[] a, int[] b) {
        int[] r = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }

    // ------------------------------------------------------------------ words

    static void splitWords(int[] tokens, String language, Tokenizer tok, int eot, List<String> words, List<int[]> wordTokens) {
        List<String> subwords = new ArrayList<>();
        List<int[]> subTokens = new ArrayList<>();
        splitOnUnicode(tokens, tok, subwords, subTokens);
        if (language != null && CJK.contains(language)) {
            words.addAll(subwords);
            wordTokens.addAll(subTokens);
            return;
        }
        for (int i = 0; i < subwords.size(); i++) {
            String sub = subwords.get(i);
            int[] st = subTokens.get(i);
            boolean special = st[0] >= eot;
            boolean withSpace = sub.startsWith(" ");
            String stripped = sub.strip();
            boolean punctuation = !stripped.isEmpty() && "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".contains(stripped)
                    && stripped.length() == 1 || isAllAsciiPunct(stripped);
            if (special || withSpace || punctuation || words.isEmpty()) {
                words.add(sub);
                wordTokens.add(st);
            } else {
                int last = words.size() - 1;
                words.set(last, words.get(last) + sub);
                wordTokens.set(last, concat(wordTokens.get(last), st));
            }
        }
    }

    private static boolean isAllAsciiPunct(String s) {
        // Python: subword.strip() in string.punctuation  (a substring test: "", "!", "?!", ... are all contained)
        return "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".contains(s);
    }

    /** Groups tokens so that every group decodes to complete characters (a multi-byte character split over tokens stays together). */
    static void splitOnUnicode(int[] tokens, Tokenizer tok, List<String> words, List<int[]> wordTokens) {
        String full = tok.decode(tokens);
        List<Integer> cur = new ArrayList<>();
        int offset = 0;
        for (int t : tokens) {
            cur.add(t);
            String decoded = tok.decode(cur.stream().mapToInt(Integer::intValue).toArray());
            int bad = decoded.indexOf('�');
            if (bad < 0 || (offset + bad < full.length() && full.charAt(offset + bad) == '�')) {
                words.add(decoded);
                wordTokens.add(cur.stream().mapToInt(Integer::intValue).toArray());
                cur.clear();
                offset += decoded.length();
            }
        }
        if (!cur.isEmpty()) {
            words.add(tok.decode(cur.stream().mapToInt(Integer::intValue).toArray()));
            wordTokens.add(cur.stream().mapToInt(Integer::intValue).toArray());
        }
    }

    // ------------------------------------------------------------------ numerics

    /** Median over a window of 7 along the columns with reflect padding, in place for every row. */
    static void medianFilter(float[] m, int rows, int cols) {
        int pad = MEDIAN_WIDTH / 2;
        float[] padded = new float[cols + 2 * pad];
        float[] win = new float[MEDIAN_WIDTH];
        for (int r = 0; r < rows; r++) {
            int o = r * cols;
            for (int i = 0; i < pad; i++) {
                padded[i] = m[o + Math.min(cols - 1, pad - i)];                  // reflect: x[pad], ..., x[1]
                padded[pad + cols + i] = m[o + Math.max(0, cols - 2 - i)];       // x[n-2], x[n-3], ...
            }
            System.arraycopy(m, o, padded, pad, cols);
            for (int c = 0; c < cols; c++) {
                System.arraycopy(padded, c, win, 0, MEDIAN_WIDTH);
                Arrays.sort(win);
                m[o + c] = win[pad];
            }
        }
    }

    /** Dynamic time warping over a cost matrix [n][m]: the path's (row, column) pairs as two arrays, as OpenAI's {@code dtw_cpu}. */
    static int[][] dtw(double[][] x) {
        int n = x.length;
        int m = x[0].length;
        double[][] cost = new double[n + 1][m + 1];
        for (double[] row : cost) {
            Arrays.fill(row, Double.POSITIVE_INFINITY);
        }
        cost[0][0] = 0;
        byte[][] trace = new byte[n + 1][m + 1];
        for (int j = 1; j <= m; j++) {
            for (int i = 1; i <= n; i++) {
                double c0 = cost[i - 1][j - 1];
                double c1 = cost[i - 1][j];
                double c2 = cost[i][j - 1];
                double c;
                byte t;
                if (c0 < c1 && c0 < c2) {
                    c = c0;
                    t = 0;
                } else if (c1 < c0 && c1 < c2) {
                    c = c1;
                    t = 1;
                } else {
                    c = c2;
                    t = 2;
                }
                cost[i][j] = x[i - 1][j - 1] + c;
                trace[i][j] = t;
            }
        }
        for (int j = 0; j <= m; j++) {
            trace[0][j] = 2;
        }
        for (int i = 0; i <= n; i++) {
            trace[i][0] = 1;
        }
        int i = n;
        int j = m;
        List<int[]> path = new ArrayList<>();
        while (i > 0 || j > 0) {
            path.add(new int[]{i - 1, j - 1});
            switch (trace[i][j]) {
                case 0 -> {
                    i--;
                    j--;
                }
                case 1 -> i--;
                default -> j--;
            }
        }
        int len = path.size();
        int[] a = new int[len];
        int[] b = new int[len];
        for (int k = 0; k < len; k++) {
            int[] p = path.get(len - 1 - k);
            a[k] = p[0];
            b[k] = p[1];
        }
        return new int[][]{a, b};
    }
}
