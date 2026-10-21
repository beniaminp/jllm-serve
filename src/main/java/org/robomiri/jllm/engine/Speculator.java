package org.robomiri.jllm.engine;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.Sampler;

import java.util.Arrays;
import java.util.Locale;
import java.util.Set;

/**
 * Speculative decoding: proposes a few tokens that probably follow the text so far, verifies them with one
 * multi-token target pass ({@link Backend#forwardAll}) and keeps the longest prefix the target agrees with,
 * so several tokens are produced per pass over the weights.
 *
 * <p>Drafts come from prompt lookup (the longest suffix n-gram of the text found earlier in prompt + output; no
 * second model) or from a small {@link DraftModel} that shares the tokenizer. Prompt lookup is tuned for coding
 * agents, whose output keeps copying what is already in the context (lines of a file the agent read, the shape of
 * earlier tool calls): every n-gram of the history is kept in a hash index ({@link NgramIndex}), so an occurrence
 * any distance back is found in constant time, the longest match wins, and the number of drafted tokens adapts to
 * the acceptance rate (doubling while whole drafts are accepted, falling back to what was accepted otherwise; for a draft
 * model, once measured, the length that {@link CostModel} finds fastest, or none). Both propose a deterministic
 * sequence, so the proposal distribution is one-hot and the acceptance rule of speculative sampling reduces to
 * {@link Sampler#verify}: greedy decoding accepts a draft token iff it is the target's argmax (the output is
 * the non-speculative output), sampled decoding accepts it with probability p(token) and otherwise draws from p
 * with that token's mass removed (the output distribution is exactly the target's). The sampler's penalties see
 * the accepted tokens in order.
 *
 * <p>Rejected positions need no rollback: the KV cache positions after the accepted prefix are overwritten by
 * the next pass. Not usable with a backend whose state cannot be rewound ({@link Backend#supportsPrefixReuse}).
 */
public final class Speculator implements AutoCloseable {

    public enum Mode {
        OFF, NGRAM, DRAFT;

        public static Mode parse(String s) {
            return switch (s.toLowerCase(Locale.ROOT)) {
                case "off", "false", "no", "0" -> OFF;
                case "ngram", "prompt-lookup" -> NGRAM;
                case "draft" -> DRAFT;
                default -> throw new IllegalArgumentException("engine.speculative must be off, ngram or draft, got '" + s + "'");
            };
        }
    }

    /**
     * @param k        most tokens proposed per pass (prompt lookup starts lower and adapts up to it)
     * @param ngramMax longest suffix n-gram looked up (prompt lookup)
     * @param ngramMin shortest suffix n-gram that may propose
     * @param window   how many of the most recent tokens may be copied from (prompt lookup), 0 = all of them
     */
    public record Config(Mode mode, int k, int ngramMax, int ngramMin, int window) {
        public Config {
            if (k < 1 || ngramMax < 1 || ngramMin < 1 || ngramMin > ngramMax || window < 0 || ngramMax > NgramIndex.MAX_N) {
                throw new IllegalArgumentException("invalid speculation settings: " + mode + " k=" + k + " ngram "
                        + ngramMin + ".." + ngramMax + " window=" + window);
            }
        }
    }

    private final Config cfg;
    private final Backend target;
    private final DraftModel draft; // null for prompt lookup
    private final Set<Integer> stopIds;
    private final int stride;       // row stride of the verification logits (the target's vocabulary rows)
    private final int vocab;        // valid token ids
    private final int maxDrafts;

    private int[] hist = new int[256];
    private int len;
    private final float[] rows;
    private final float[] row;
    private final int[] feed;
    private final int[] drafts;

    // result of the last successful step()
    private final int[] ready;
    private int readyCount;
    private int preAccepted;
    private int newPos;
    private int misses;   // consecutive passes in which no draft token was accepted
    private int cooldown; // plain steps still to take before drafting again
    private long draftedTotal;
    private long acceptedTotal;
    private int kNow;     // adaptive draft length (prompt lookup)
    private final NgramIndex index;

    /**
     * @param stride    logits row width of the target (its model vocabulary size)
     * @param vocab     number of valid token ids (the tokenizer's vocabulary)
     * @param draft     the draft model for {@link Mode#DRAFT}, else null
     * @param stopIds   end-of-sequence ids: a verification pass stops sampling once one is accepted
     */
    public Speculator(Config cfg, Backend target, DraftModel draft, int stride, int vocab, Set<Integer> stopIds) {
        if (cfg.mode() == Mode.OFF) {
            throw new IllegalArgumentException("speculation is off");
        }
        if (cfg.mode() == Mode.DRAFT && draft == null) {
            throw new IllegalArgumentException("engine.speculative=draft needs a draft model");
        }
        this.cfg = cfg;
        this.target = target;
        this.draft = draft;
        this.stride = stride;
        this.vocab = vocab;
        this.stopIds = stopIds;
        this.maxDrafts = Math.max(1, Math.min(cfg.k(), target.maxBatch() - 1));
        this.rows = new float[(maxDrafts + 1) * stride];
        this.row = new float[stride];
        this.feed = new int[maxDrafts + 1];
        this.drafts = new int[maxDrafts];
        this.ready = new int[maxDrafts + 1];
        this.index = cfg.mode() == Mode.NGRAM ? new NgramIndex(cfg.ngramMin(), cfg.ngramMax()) : null;
    }

    /** Whether the backend can be used for speculation at all (a recurrent state cannot be rolled back). */
    public static boolean supported(Backend b) {
        return b.supportsPrefixReuse() && b.maxBatch() >= 2;
    }

    /** Starts a request: the history is the (prepared) prompt. */
    public void begin(int[] prompt) {
        // an agent's next request repeats the previous one and adds to it: keep the index of the common prefix
        int keep = 0;
        int n = Math.min(len, prompt.length);
        while (keep < n && hist[keep] == prompt[keep]) {
            keep++;
        }
        if (index != null) {
            index.truncate(hist, keep);
        }
        len = keep;
        misses = 0;
        cooldown = 0;
        kNow = Math.min(4, maxDrafts);
        for (int i = keep; i < prompt.length; i++) {
            onToken(prompt[i]);
        }
    }

    /** Appends a token of the output to the history. */
    public void onToken(int tok) {
        if (len == hist.length) {
            hist = Arrays.copyOf(hist, len * 2);
        }
        hist[len++] = tok;
        if (index != null && len >= 2) {
            index.add(hist, len - 2); // the n-grams ending one token back: never the suffix being looked up
        }
    }

    /**
     * Tries to produce more than one token with a single target pass. Call after {@code tok} was added to the
     * history ({@link #onToken}) and accepted by the sampler, before it is fed to the model.
     *
     * @param pos       sequence position {@code tok} goes to (positions {@code [0, pos)} hold valid K/V)
     * @param ctxLimit  context length of the target
     * @param room      most tokens that may still be produced after {@code tok}
     * @return false when nothing was drafted (the caller runs a normal single-token step); else
     *         {@link #ready}, {@link #preAccepted} and {@link #newPos} describe the outcome
     */
    public boolean step(int tok, int pos, int ctxLimit, int room, Sampler sampler) {
        return step(tok, pos, ctxLimit, room, sampler, null);
    }

    /** @param beforeVerify run right before the verification pass, when there is one (drains a pipelined backend) */
    public boolean step(int tok, int pos, int ctxLimit, int room, Sampler sampler, Runnable beforeVerify) {
        // a draft model: the length the measured costs and acceptance rate make best (-1: not measured yet, then the
        // doubling rule below; 0: speculation does not pay at all right now)
        int best = cfg.mode() == Mode.DRAFT && adaptive ? cost.bestK(maxDrafts) : -1;
        int want = Math.min(Math.min(best > 0 ? best : kNow, room - 1), ctxLimit - pos - 1);
        if (want < 1) {
            return false;
        }
        if (cooldown > 0) { // recent passes accepted nothing: plain steps for a while
            cooldown--;
            return false;
        }
        if (best == 0 && ++unprofitable % PROBE_EVERY != 0) {
            cooldown = PLAIN_STEPS - 1; // plain steps (which also re-measure their cost), then look again
            return false;
        }                               // (every PROBE_EVERY-th time a pass is tried anyway: the text may have changed)
        long t0 = System.nanoTime();
        int m = cfg.mode() == Mode.NGRAM ? lookup(want) : draft.propose(hist, len, want, drafts, vocab, stopIds);
        if (draft != null && draft.lastSteps() > 0) {
            cost.draft(System.nanoTime() - t0, draft.lastSteps());
        }
        if (m < 1) {
            return false;
        }
        feed[0] = tok;
        System.arraycopy(drafts, 0, feed, 1, m);
        if (beforeVerify != null) {
            beforeVerify.run();
        }
        long t1 = System.nanoTime();
        target.forwardAll(feed, 0, m + 1, pos, rows, stride);
        cost.verify(m + 1, System.nanoTime() - t1);
        int acc = 0;
        int examined = 0;
        readyCount = 0;
        for (int j = 0; j <= m; j++) {
            System.arraycopy(rows, j * stride, row, 0, stride);
            int s = j < m ? sampler.verify(row, drafts[j]) : sampler.sample(row);
            ready[readyCount++] = s;
            examined += j < m ? 1 : 0;
            if (j < m && s == drafts[j]) {
                acc++;
                if (stopIds.contains(s)) {
                    break; // the stream ends here: no point in sampling further rows
                }
                sampler.accept(s);
                continue;
            }
            break;
        }
        cost.acceptance(acc, examined);
        // whole draft accepted: try longer ones next time; else as many as this text supported
        kNow = acc == m ? Math.min(maxDrafts, Math.max(kNow, 2 * m)) : Math.max(1, Math.min(kNow, acc + 1));
        if (acc > 0) {
            misses = 0;
        } else { // a pass that gained nothing costs more than a plain step: back off exponentially (1, 3, 7, 15 steps)
            cooldown = (1 << Math.min(++misses, 4)) - 1;
        }
        preAccepted = acc;
        newPos = pos + 1 + acc;
        draftedTotal += m;
        acceptedTotal += acc;
        return true;
    }

    /**
     * The time of a plain single-token step of the target (the engine reports the steps it takes without a draft), the
     * baseline a speculative pass has to beat.
     */
    public void plainStep(long nanos) {
        cost.plain(nanos);
    }

    /** {@code -Djllm.engine.adaptiveDraft=false}: a draft model's length follows the doubling rule only. */
    private final boolean adaptive = !"false".equals(System.getProperty("jllm.engine.adaptiveDraft"));
    /** Plain steps taken when the cost model says speculation does not pay, before it is asked again. */
    static final int PLAIN_STEPS = 16;
    /** One in this many "does not pay" decisions runs a pass anyway, so a change in acceptance is noticed. */
    static final int PROBE_EVERY = 4;

    private final CostModel cost = new CostModel();
    private int unprofitable;

    /** The draft length the cost model would choose now, -1 before it has measurements (for tests and logs). */
    public int modelDraftLength() {
        return cost.bestK(maxDrafts);
    }

    /**
     * Chooses a draft model's draft length from measured costs. With per-token acceptance probability α (drafts are
     * accepted left to right until the first miss), a pass with k drafts yields on average
     * {@code E(k) = 1 + α + α² + … + α^k} tokens (the accepted prefix plus the target's own token) and costs
     * {@code k · d + V(k + 1)}: k draft steps of d each and one verification pass over k + 1 positions, modelled as
     * {@code V(n) = a + b (n − 1)}. The best k maximizes {@code E(k) / (k d + a + b k)}; when even that is below the
     * plain decoding rate {@code 1 / p} (p a plain step), speculation is paused. α, d, p, a and b are exponentially
     * weighted (recent passes count most): α from the accepted and examined positions of every pass, d from the draft's
     * time per step, p from the plain steps the engine reports, a and b by least squares over the verification passes
     * (the pass lengths vary with the doubling rule and with early stops; with too little spread, a is the plain step
     * and b what remains).
     */
    static final class CostModel {
        private static final double DECAY = 0.9;
        private static final double PRIOR_B = 0.36; // a pass's extra cost per position as a share of a, before data
        private double acc = 1.2;                   // prior: alpha 0.6 over two positions
        private double exam = 2.0;
        private double draftNs = -1;
        private double plainNs = -1;
        private double s0, s1, s2, sy, s1y;          // weighted sums of x = n - 1 and y = V(n)
        private int passes;

        void acceptance(int accepted, int examined) {
            acc = DECAY * acc + accepted;
            exam = DECAY * exam + examined;
        }

        void draft(long nanos, int steps) {
            double v = nanos / (double) steps;
            draftNs = draftNs < 0 ? v : 0.8 * draftNs + 0.2 * v;
        }

        void plain(long nanos) {
            plainNs = plainNs < 0 ? nanos : 0.8 * plainNs + 0.2 * nanos;
        }

        void verify(int n, long nanos) {
            double x = n - 1;
            s0 = DECAY * s0 + 1;
            s1 = DECAY * s1 + x;
            s2 = DECAY * s2 + x * x;
            sy = DECAY * sy + nanos;
            s1y = DECAY * s1y + x * nanos;
            passes++;
        }

        double alpha() {
            return Math.min(0.99, Math.max(0.0, acc / exam));
        }

        /** {a, b} of V(n) = a + b (n - 1). */
        double[] fit() {
            double mx = s1 / s0;
            double my = sy / s0;
            double var = s2 / s0 - mx * mx;
            double a;
            double b;
            if (var > 0.25) {
                b = (s1y / s0 - mx * my) / var;
                a = my - b * mx;
            } else if (plainNs > 0) {
                a = Math.min(plainNs, my);
                b = mx > 0 ? (my - a) / mx : PRIOR_B * a;
            } else {
                a = my / (1 + PRIOR_B * mx);
                b = PRIOR_B * a;
            }
            if (!(a > 0)) {
                a = Math.max(my / (1 + mx), 1);
            }
            return new double[]{a, Math.max(0, b)};
        }

        /** The best draft length in [1, max], 0 when no length beats plain decoding, -1 before there is data. */
        int bestK(int max) {
            if (passes < 2 || draftNs < 0) {
                return -1;
            }
            double[] ab = fit();
            double alpha = alpha();
            double plain = plainNs > 0 ? plainNs : ab[0];
            double bestRate = 1.0 / plain;
            int best = 0;
            double e = 1;
            double ak = 1;
            for (int k = 1; k <= max; k++) {
                ak *= alpha;
                e += ak;
                double rate = e / (k * draftNs + ab[0] + ab[1] * k);
                if (rate > bestRate) {
                    bestRate = rate;
                    best = k;
                }
            }
            return best;
        }
    }

    /** Tokens decided by the last {@link #step}: the accepted drafts, then one token the target chose itself. */
    public int[] ready() {
        return ready;
    }

    public int readyCount() {
        return readyCount;
    }

    /** How many leading {@link #ready} tokens the sampler has already accepted (penalties); the rest it has not. */
    public int preAccepted() {
        return preAccepted;
    }

    /** Position of the last {@link #ready} token (its K/V is not computed yet): the engine's next feed position. */
    public int newPos() {
        return newPos;
    }

    public long drafted() {
        return draftedTotal;
    }

    public long accepted() {
        return acceptedTotal;
    }

    /**
     * Prompt lookup: the most recent earlier occurrence of the longest (at most {@code ngramMax}, at least
     * {@code ngramMin}) suffix of the history; proposes the tokens that followed it, into {@link #drafts}.
     */
    private int lookup(int want) {
        int min = cfg.window() > 0 ? len - 1 - cfg.window() : -1;
        for (int n = Math.min(cfg.ngramMax(), len - 1); n >= cfg.ngramMin(); n--) {
            int e = index.find(hist, len - 1, n);
            if (e >= 0 && e > min) {
                int m = Math.min(want, len - 1 - e);
                System.arraycopy(hist, e + 1, drafts, 0, m);
                return m;
            }
        }
        return 0;
    }

    @Override
    public void close() {
        if (draft != null) {
            draft.close();
        }
    }
}
