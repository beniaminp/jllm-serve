package org.robomiri.jllm.sampling;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Token sampler: penalties, temperature, top-k, top-p (nucleus), min-p, then a categorical draw.
 * One instance per generation request (it tracks token counts for the penalties).
 */
public final class Sampler {

    private final SamplingParams p;
    private final int vocab;
    private final SplittableRandom rng;
    private final boolean[] seen;
    private final int[] generatedCounts;

    // scratch
    private final int[] idx;
    private final float[] val;
    /** Full-vocabulary softmax denominator computed by the last {@link #nucleusCandidates} call. */
    private double nucleusSum;

    /**
     * @param vocab number of valid token ids; logits beyond it (padding rows of the embedding) are never sampled
     */
    public Sampler(SamplingParams p, int vocab, int[] prompt) {
        this.p = p;
        this.vocab = vocab;
        this.rng = p.seed() >= 0 ? new SplittableRandom(p.seed()) : new SplittableRandom();
        this.seen = new boolean[vocab];
        this.generatedCounts = new int[vocab];
        for (int t : prompt) {
            if (t >= 0 && t < vocab) {
                seen[t] = true;
            }
        }
        int cap = p.topK() > 0 ? Math.min(p.topK(), vocab) : vocab;
        idx = new int[cap];
        val = new float[cap];
    }

    /** Records a generated token so the penalties see it. */
    public void accept(int token) {
        if (token >= 0 && token < vocab) {
            seen[token] = true;
            generatedCounts[token]++;
        }
    }

    /** Samples the next token. {@code logits} is modified in place. */
    public int sample(float[] logits) {
        applyPenalties(logits);
        if (p.temperature() == 0f || p.topK() == 1) {
            return argmax(logits);
        }
        float invT = 1f / p.temperature();
        int n;
        boolean sorted = true;
        boolean nucleus = false;
        if (p.topK() > 0 && p.topK() < vocab) {
            n = topK(logits, p.topK());
        } else if (p.topP() < 1f) {
            n = nucleusCandidates(logits, invT);
            nucleus = true;
        } else { // plain temperature sampling over the whole vocabulary, no need to sort
            n = vocab;
            for (int i = 0; i < n; i++) {
                idx[i] = i;
                val[i] = logits[i];
            }
            sorted = false;
        }
        // softmax over candidates
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, val[i]);
        }
        double sum = 0;
        for (int i = 0; i < n; i++) {
            float e = (float) Math.exp((val[i] - max) * invT);
            val[i] = e;
            sum += e;
        }
        if (nucleus) {
            // Top-p is defined over the whole distribution. The candidates are only the tokens that can be in
            // the nucleus; normalizing by their mass alone would inflate every cumulative probability and cut
            // the nucleus early. Whenever any candidate survives the bound the most likely token does too, so the
            // max used in nucleusCandidates() equals the one above and the two denominators are comparable.
            sum = nucleusSum;
        }
        // top-p on the sorted candidates
        int keep = n;
        if (sorted && p.topP() < 1f) {
            double cum = 0;
            for (int i = 0; i < n; i++) {
                cum += val[i] / sum;
                if (cum >= p.topP()) {
                    keep = i + 1;
                    break;
                }
            }
        }
        // min-p relative to the most likely token (whose unnormalized weight is exp(0) = 1)
        if (p.minP() > 0f) {
            float threshold = p.minP();
            for (int i = 0; i < keep; i++) {
                if (val[i] < threshold) {
                    val[i] = 0f;
                }
            }
        }
        double total = 0;
        for (int i = 0; i < keep; i++) {
            total += val[i];
        }
        double r = rng.nextDouble() * total;
        double acc = 0;
        for (int i = 0; i < keep; i++) {
            acc += val[i];
            if (r < acc) {
                return idx[i];
            }
        }
        return argmax(logits);
    }

    private void applyPenalties(float[] logits) {
        float rep = p.repetitionPenalty();
        float pres = p.presencePenalty();
        float freq = p.frequencyPenalty();
        boolean any = rep != 1f || pres != 0f || freq != 0f;
        if (!any) {
            return;
        }
        for (int t = 0; t < vocab; t++) {
            if (!seen[t]) {
                continue;
            }
            if (rep != 1f) {
                logits[t] = logits[t] > 0 ? logits[t] / rep : logits[t] * rep;
            }
            int cnt = generatedCounts[t];
            if (cnt > 0) {
                logits[t] -= pres + freq * cnt;
            }
        }
    }

    private int argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < vocab; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }

    /** Selects the k largest logits into idx/val, sorted descending. Uses a size-k min-heap. */
    private int topK(float[] logits, int k) {
        int n = 0;
        for (int i = 0; i < vocab; i++) {
            float v = logits[i];
            if (n < k) {
                val[n] = v;
                idx[n] = i;
                siftUp(n++);
            } else if (v > val[0]) {
                val[0] = v;
                idx[0] = i;
                siftDown(0, n);
            }
        }
        // heap-sort ascending min-heap into descending order
        for (int end = n - 1; end > 0; end--) {
            swap(0, end);
            siftDown(0, end);
        }
        return n;
    }

    /**
     * Without top-k, only tokens whose probability can matter for top-p are candidates: anything below
     * (1 - topP) / (V - 1) can never be inside the nucleus. This avoids sorting the whole vocabulary.
     */
    private int nucleusCandidates(float[] logits, float invT) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; i++) {
            max = Math.max(max, logits[i]);
        }
        double sum = 0;
        for (int i = 0; i < vocab; i++) {
            sum += Math.exp((logits[i] - max) * invT);
        }
        nucleusSum = sum;
        double cutoff = p.topP() < 1f ? (1.0 - p.topP()) / (vocab - 1) : 0.0;
        int n = 0;
        for (int i = 0; i < vocab; i++) {
            if (Math.exp((logits[i] - max) * invT) / sum >= cutoff) {
                idx[n] = i;
                val[n] = logits[i];
                n++;
            }
        }
        sortDescending(n);
        return n;
    }

    private void sortDescending(int n) {
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Float.compare(val[b], val[a]));
        int[] i2 = new int[n];
        float[] v2 = new float[n];
        for (int i = 0; i < n; i++) {
            i2[i] = idx[order[i]];
            v2[i] = val[order[i]];
        }
        System.arraycopy(i2, 0, idx, 0, n);
        System.arraycopy(v2, 0, val, 0, n);
    }

    private void siftUp(int i) {
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (val[i] >= val[parent]) {
                return;
            }
            swap(i, parent);
            i = parent;
        }
    }

    private void siftDown(int i, int n) {
        while (true) {
            int l = 2 * i + 1;
            int r = l + 1;
            int m = i;
            if (l < n && val[l] < val[m]) {
                m = l;
            }
            if (r < n && val[r] < val[m]) {
                m = r;
            }
            if (m == i) {
                return;
            }
            swap(i, m);
            i = m;
        }
    }

    private void swap(int a, int b) {
        float tv = val[a];
        val[a] = val[b];
        val[b] = tv;
        int ti = idx[a];
        idx[a] = idx[b];
        idx[b] = ti;
    }
}
