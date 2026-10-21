package org.robomiri.jllm.sampling;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.IntPredicate;

/**
 * Token sampler: penalties, logit bias, temperature, top-k, top-p (nucleus), min-p, then a categorical draw.
 * One instance per generation request (it tracks token counts for the penalties).
 */
public final class Sampler {

    private final SamplingParams p;
    private final int vocab;
    private final SplittableRandom rng;
    private final boolean[] seen;
    private final int[] generatedCounts;
    private final float[] bias; // OpenAI logit_bias, added to the logits after the penalties; null = none

    // scratch
    private final int[] idx;
    private final float[] val;
    /** Full-vocabulary softmax denominator computed by the last {@link #nucleusCandidates} call. */
    private double nucleusSum;

    /**
     * @param vocab number of valid token ids; logits beyond it (padding rows of the embedding) are never sampled
     */
    public Sampler(SamplingParams p, int vocab, int[] prompt) {
        this(p, vocab, prompt, null);
    }

    /**
     * @param bias added to the logits before sampling (after the penalties), indexed by token id; null for none.
     *             OpenAI's {@code logit_bias}: -100 effectively bans a token, +100 all but forces it
     */
    public Sampler(SamplingParams p, int vocab, int[] prompt, float[] bias) {
        this.p = p;
        this.bias = bias;
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

    /**
     * Greedy without penalties: the next token is exactly the argmax of the logits (lowest index among equal maxima),
     * which a device can compute itself (pipelined decoding).
     */
    public boolean greedyPlain() {
        return (p.temperature() == 0f || p.topK() == 1) && p.repetitionPenalty() == 1f && p.presencePenalty() == 0f
                && p.frequencyPenalty() == 0f && bias == null;
    }

    /** Samples the next token. {@code logits} is modified in place. */
    public int sample(float[] logits) {
        if (!prepare(logits)) {
            return argmax(logits);
        }
        return draw(logits);
    }

    /** How many of the most likely tokens a constrained step considers (and checks) before widening its search. */
    static final int CONSTRAINED_POOL = 256;
    private int[] cIdx;
    private float[] cVal;

    /**
     * Samples the next token among those {@code allowed} accepts (constrained decoding), or returns -1 when none
     * is. Greedy first tests the argmax: when the model's own choice is allowed (most steps) that is the answer and
     * nothing else is checked. Otherwise the constraint is checked lazily: the {@value #CONSTRAINED_POOL} most likely
     * tokens (after penalties and bias) are tested, most likely first; only if none of them is allowed is the whole
     * vocabulary scanned (the constraint then computes its full allowed set once, {@link ByteConstraint}). Greedy
     * takes the first allowed token. Sampling draws from the allowed tokens among those candidates, renormalized:
     * top-k keeps the k most likely allowed tokens, top-p and min-p apply to the renormalized distribution, so the
     * tail beyond the candidates (negligible next to them) is left out. {@code logits} is modified in place.
     */
    public int sample(float[] logits, IntPredicate allowed) {
        if (allowed == null) {
            return sample(logits);
        }
        applyPenalties(logits);
        return masked(logits, allowed, -1);
    }

    /**
     * {@link #verify(float[], int)} under a constraint: the result is distributed like
     * {@link #sample(float[], IntPredicate)} and equals {@code draft} (an allowed token) exactly when the draft is
     * accepted. Greedy: the masked argmax, whatever the draft.
     */
    public int verify(float[] logits, int draft, IntPredicate allowed) {
        if (allowed == null) {
            return verify(logits, draft);
        }
        applyPenalties(logits);
        return masked(logits, allowed, draft);
    }

    /**
     * Sampling where nearly everything is allowed ({@link TokenConstraint#relaxed}, free text between tool calls):
     * a plain sample, kept when {@code allowed} accepts it; otherwise a masked sample ({@link #sample(float[],
     * IntPredicate)}). The result is distributed exactly as the plain distribution restricted to the allowed tokens
     * whenever the masked fallback covers them (p(x) + (1 − A) p(x) / A = p(x) / A for an allowed x, A the allowed
     * mass), and free text is the unconstrained text token for token.
     */
    public int sampleRelaxed(float[] logits, IntPredicate allowed) {
        applyPenalties(logits);
        int t = build(logits) ? draw(logits) : argmax(logits);
        return allowed.test(t) ? t : masked(logits, allowed, -1);
    }

    /** {@link #verify(float[], int)} where nearly everything is allowed; the fallback as in {@link #sampleRelaxed}. */
    public int verifyRelaxed(float[] logits, int draft, IntPredicate allowed) {
        applyPenalties(logits);
        int t;
        if (!build(logits)) {
            t = argmax(logits);
        } else {
            t = -1;
            for (int i = 0; i < keep && t < 0; i++) {
                if (idx[i] == draft) {
                    if (val[i] > 0 && rng.nextDouble() * total < val[i]) {
                        t = draft;
                    } else {
                        total -= val[i];
                        val[i] = 0f;
                    }
                    break;
                }
            }
            if (t < 0) {
                t = draw(logits);
            }
        }
        return allowed.test(t) ? t : masked(logits, allowed, -1);
    }

    /**
     * The next token under {@code c} (null = unconstrained): an end-of-sequence token ({@code isStop}) is allowed
     * exactly when the constrained output may end. Relaxed states sample as {@link #sampleRelaxed}.
     *
     * @return the token, or -1 when the constraint allows none
     */
    public int sample(float[] logits, TokenConstraint c, IntPredicate isStop) {
        if (c == null) {
            return sample(logits);
        }
        IntPredicate allowed = id -> isStop.test(id) ? c.canEnd() : c.allows(id);
        return c.relaxed() ? sampleRelaxed(logits, allowed) : sample(logits, allowed);
    }

    /** {@link #verify(float[], int)} under {@code c} (null = unconstrained), like {@link #sample(float[], TokenConstraint, IntPredicate)}. */
    public int verify(float[] logits, int draft, TokenConstraint c, IntPredicate isStop) {
        if (c == null) {
            return verify(logits, draft);
        }
        IntPredicate allowed = id -> isStop.test(id) ? c.canEnd() : c.allows(id);
        return c.relaxed() ? verifyRelaxed(logits, draft, allowed) : verify(logits, draft, allowed);
    }

    /**
     * The masked choice (penalties already applied): candidates, then greedy pick or a draw; with {@code draft >= 0}
     * the speculative accept / resample rule over the masked distribution.
     */
    private int masked(float[] logits, IntPredicate allowed, int draft) {
        boolean greedy = p.temperature() == 0f || p.topK() == 1;
        if (greedy) {
            int a = argmax(logits);
            if (allowed.test(a)) {
                return a;
            }
        }
        int want = greedy ? 1 : p.topK() > 0 ? Math.min(p.topK(), CONSTRAINED_POOL) : CONSTRAINED_POOL;
        int pool = Math.min(vocab, CONSTRAINED_POOL);
        if (cIdx == null) {
            cIdx = new int[pool];
            cVal = new float[pool];
        }
        int m = topK(logits, pool, cIdx, cVal, null);
        int n = 0;
        for (int i = 0; i < m && n < want; i++) {
            if (allowed.test(cIdx[i])) {
                cIdx[n] = cIdx[i];
                cVal[n] = cVal[i];
                n++;
            }
        }
        if (n == 0 && m < vocab) { // nothing likely is allowed: search everything
            n = topK(logits, want, cIdx, cVal, allowed);
        }
        if (n == 0) {
            return -1;
        }
        if (greedy || n == 1) {
            return cIdx[0];
        }
        float invT = 1f / p.temperature();
        double sum = 0;
        for (int i = 0; i < n; i++) {
            float e = (float) Math.exp((cVal[i] - cVal[0]) * invT);
            cVal[i] = e;
            sum += e;
        }
        int kept = n;
        if (p.topP() < 1f) {
            double cum = 0;
            for (int i = 0; i < n; i++) {
                cum += cVal[i] / sum;
                if (cum >= p.topP()) {
                    kept = i + 1;
                    break;
                }
            }
        }
        double t = 0;
        for (int i = 0; i < kept; i++) {
            if (cVal[i] < p.minP()) { // relative to the most likely allowed token (weight 1)
                cVal[i] = 0f;
            }
            t += cVal[i];
        }
        if (draft >= 0) {
            for (int i = 0; i < kept; i++) {
                if (cIdx[i] == draft) {
                    if (cVal[i] > 0 && rng.nextDouble() * t < cVal[i]) {
                        return draft;
                    }
                    t -= cVal[i];
                    cVal[i] = 0f;
                    break;
                }
            }
        }
        double r = rng.nextDouble() * t;
        double acc = 0;
        for (int i = 0; i < kept; i++) {
            acc += cVal[i];
            if (r < acc) {
                return cIdx[i];
            }
        }
        for (int i = 0; i < kept; i++) { // rounding: the last token with weight
            if (cVal[kept - 1 - i] > 0) {
                return cIdx[kept - 1 - i];
            }
        }
        return cIdx[0];
    }

    /**
     * Speculative-sampling step for a deterministic draft token (a prompt-lookup match or a greedy draft model:
     * its proposal distribution is one-hot, q(draft) = 1). Greedy: returns the target's argmax, which equals
     * {@code draft} exactly when the draft is right. Sampling: accepts {@code draft} with probability
     * min(1, p(draft) / q(draft)) = p(draft), where p is the distribution {@link #sample} would draw from
     * (penalties, temperature, top-k, top-p, min-p applied); otherwise draws from norm(max(0, p - q)), i.e. p
     * with the draft's mass removed. The returned token is then distributed exactly as a plain sample would be,
     * and it equals {@code draft} iff the draft was accepted. {@code logits} is modified in place.
     */
    public int verify(float[] logits, int draft) {
        if (!prepare(logits)) {
            return argmax(logits);
        }
        for (int i = 0; i < keep; i++) {
            if (idx[i] == draft) {
                if (val[i] > 0 && rng.nextDouble() * total < val[i]) {
                    return draft;
                }
                total -= val[i];
                val[i] = 0f;
                break;
            }
        }
        return draw(logits);
    }

    /** Candidate count and total weight of the distribution {@link #prepare} built. */
    private int keep;
    private double total;

    /**
     * Applies the penalties and builds the sampling distribution into {@code idx}/{@code val} (weights, not yet
     * normalized; {@code keep} candidates, {@code total} their sum).
     *
     * @return false when decoding is greedy (the distribution is the argmax)
     */
    private boolean prepare(float[] logits) {
        applyPenalties(logits);
        return build(logits);
    }

    /** {@link #prepare} without the penalties (already applied). */
    private boolean build(float[] logits) {
        if (p.temperature() == 0f || p.topK() == 1) {
            return false;
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
        int kept = n;
        if (sorted && p.topP() < 1f) {
            double cum = 0;
            for (int i = 0; i < n; i++) {
                cum += val[i] / sum;
                if (cum >= p.topP()) {
                    kept = i + 1;
                    break;
                }
            }
        }
        // min-p relative to the most likely token (whose unnormalized weight is exp(0) = 1)
        if (p.minP() > 0f) {
            float threshold = p.minP();
            for (int i = 0; i < kept; i++) {
                if (val[i] < threshold) {
                    val[i] = 0f;
                }
            }
        }
        double t = 0;
        for (int i = 0; i < kept; i++) {
            t += val[i];
        }
        keep = kept;
        total = t;
        return true;
    }

    /** One categorical draw from the distribution {@link #prepare} built. */
    private int draw(float[] logits) {
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
        for (int t = 0; any && t < vocab; t++) {
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
        if (bias != null) {
            for (int t = 0; t < vocab; t++) {
                logits[t] += bias[t];
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
        return topK(logits, k, idx, val, null);
    }

    /**
     * Selects the {@code k} largest logits among the tokens {@code allowed} accepts (all when null) into
     * {@code ix}/{@code v}, sorted descending.
     */
    private int topK(float[] logits, int k, int[] ix, float[] v, IntPredicate allowed) {
        int n = 0;
        for (int i = 0; i < vocab; i++) {
            float x = logits[i];
            if (n < k) {
                if (allowed != null && !allowed.test(i)) {
                    continue;
                }
                v[n] = x;
                ix[n] = i;
                siftUp(ix, v, n++);
            } else if (x > v[0] && (allowed == null || allowed.test(i))) {
                v[0] = x;
                ix[0] = i;
                siftDown(ix, v, 0, n);
            }
        }
        // heap-sort ascending min-heap into descending order
        for (int end = n - 1; end > 0; end--) {
            swap(ix, v, 0, end);
            siftDown(ix, v, 0, end);
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

    private static void siftUp(int[] ix, float[] v, int i) {
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (v[i] >= v[parent]) {
                return;
            }
            swap(ix, v, i, parent);
            i = parent;
        }
    }

    private static void siftDown(int[] ix, float[] v, int i, int n) {
        while (true) {
            int l = 2 * i + 1;
            int r = l + 1;
            int m = i;
            if (l < n && v[l] < v[m]) {
                m = l;
            }
            if (r < n && v[r] < v[m]) {
                m = r;
            }
            if (m == i) {
                return;
            }
            swap(ix, v, i, m);
            i = m;
        }
    }

    private static void swap(int[] ix, float[] v, int a, int b) {
        float tv = v[a];
        v[a] = v[b];
        v[b] = tv;
        int ti = ix[a];
        ix[a] = ix[b];
        ix[b] = ti;
    }
}
