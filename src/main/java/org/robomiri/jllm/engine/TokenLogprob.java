package org.robomiri.jllm.engine;

/**
 * The log-probability of one token under the model, plus the most likely alternatives at that position (OpenAI
 * {@code logprobs}). Computed from the raw logits the model produced, before penalties, logit bias, temperature,
 * truncation or a constraint: {@code logprob(t) = z_t - log(sum_j exp(z_j))}, the log-softmax of the logits
 * {@code z}. This is the model's own distribution, the same for every sampling setting, so the values are
 * comparable between requests (and summing them scores a sequence, as {@code best_of} does).
 *
 * @param token       the token id
 * @param logprob     its natural-log probability (at most 0)
 * @param topTokens   the most likely token ids at this position, most likely first (may be empty)
 * @param topLogprobs their log-probabilities
 */
public record TokenLogprob(int token, float logprob, int[] topTokens, float[] topLogprobs) {

    /**
     * @param logits raw logits of the position (only the first {@code vocab} entries count)
     * @param token  the token whose log-probability is wanted
     * @param top    how many alternatives to report (0 = none)
     */
    public static TokenLogprob of(float[] logits, int vocab, int token, int top) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; i++) {
            max = Math.max(max, logits[i]);
        }
        double sum = 0;
        for (int i = 0; i < vocab; i++) {
            sum += Math.exp(logits[i] - max);
        }
        double lse = max + Math.log(sum);
        top = Math.min(top, vocab);
        int[] ids = new int[top];
        float[] vals = new float[top];
        int n = 0;
        for (int i = 0; i < vocab && top > 0; i++) { // insertion into a short sorted list; most tokens fail the first test
            float x = logits[i];
            if (n == top && x <= vals[n - 1]) {
                continue;
            }
            int j = n < top ? n++ : n - 1;
            while (j > 0 && vals[j - 1] < x) {
                vals[j] = vals[j - 1];
                ids[j] = ids[j - 1];
                j--;
            }
            vals[j] = x;
            ids[j] = i;
        }
        float[] lp = new float[top];
        for (int i = 0; i < top; i++) {
            lp[i] = (float) (vals[i] - lse);
        }
        return new TokenLogprob(token, (float) (logits[token] - lse), ids, lp);
    }
}
