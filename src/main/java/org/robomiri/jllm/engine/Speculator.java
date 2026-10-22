package org.robomiri.jllm.engine;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.Sampler;
import org.robomiri.jllm.sampling.TokenConstraint;

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
 * <p>A third source is the model's own multi-token-prediction head ({@link Mode#MTP}, Qwen3.5 GGUF): after each
 * sampled token the head proposes the token after it from the main model's last hidden state, and the target verifies
 * it in the same two-position pass that computes the next step. A hybrid's recurrent state cannot be rewound by
 * overwriting, so that mode checkpoints it before the pass and, when the draft is rejected, puts it back and replays
 * the accepted tokens ({@link Backend#checkpointState}).
 *
 * <p>Rejected positions need no rollback: the KV cache positions after the accepted prefix are overwritten by
 * the next pass. Not usable with a backend whose state cannot be rewound ({@link Backend#supportsPrefixReuse}), except
 * for the MTP mode on a backend that can checkpoint it.
 */
public final class Speculator implements AutoCloseable {

    public enum Mode {
        OFF, NGRAM, DRAFT, MTP;

        public static Mode parse(String s) {
            return switch (s.toLowerCase(Locale.ROOT)) {
                case "off", "false", "no", "0" -> OFF;
                case "ngram", "prompt-lookup" -> NGRAM;
                case "draft" -> DRAFT;
                case "mtp" -> MTP;
                default -> throw new IllegalArgumentException("engine.speculative must be off, ngram, draft or mtp, got '" + s + "'");
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
    private final TokenConstraint[] forks;

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
        if (cfg.mode() == Mode.MTP && !supported(target, Mode.MTP)) {
            throw new IllegalArgumentException("engine.speculative=mtp needs a backend with a multi-token-prediction head");
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
        this.forks = new TokenConstraint[maxDrafts + 1];
        this.index = cfg.mode() == Mode.NGRAM ? new NgramIndex(cfg.ngramMin(), cfg.ngramMax()) : null;
        this.gated = cfg.mode() == Mode.NGRAM && adaptive && !"false".equals(System.getProperty("jllm.engine.adaptiveNgram"));
        if (gated) {
            cost.draft(0, 1); // a lookup costs nothing next to a pass: the model only weighs verification against plain steps
        }
        this.mtpLogits = cfg.mode() == Mode.MTP ? new float[stride] : null;
        if (cfg.mode() == Mode.MTP) {
            target.enableMtp(true);
        }
    }

    private final float[] mtpLogits;

    /** Whether the backend can be used for speculation at all (a recurrent state cannot be rolled back). */
    public static boolean supported(Backend b) {
        return b.supportsPrefixReuse() && b.maxBatch() >= 2;
    }

    /** {@link #supported(Backend)} for the mode: the MTP mode instead needs the head and a state it can roll back. */
    public static boolean supported(Backend b, Mode mode) {
        return mode == Mode.MTP ? b.hasMtp() && b.supportsRollback() && b.maxBatch() >= 2 : supported(b);
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
        if (gated) {
            cost.forgetAcceptance(); // a new request may copy where the last one did not (and the other way round)
        }
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
        return step(tok, pos, ctxLimit, room, sampler, beforeVerify, null);
    }

    /**
     * @param con the output's constraint after {@code tok}, or null. The draft is cut at its first token the
     *            constraint does not allow (checked on copies, so the caller's constraint is not touched), and each
     *            row is verified with the sampler's constrained rule against the constraint as it would stand at that
     *            position: greedy output is exactly the non-speculative constrained output
     */
    public boolean step(int tok, int pos, int ctxLimit, int room, Sampler sampler, Runnable beforeVerify,
                        TokenConstraint con) {
        // a draft model: the length the measured costs and acceptance rate make best (-1: not measured yet, then the
        // doubling rule below; 0: speculation does not pay at all right now)
        int best = cfg.mode() == Mode.DRAFT && adaptive || gated ? cost.bestK(maxDrafts) : -1;
        // (prompt lookup uses the model only to pause: the length is what the text supports, see kNow)
        int want = Math.min(Math.min(best > 0 && !gated ? best : kNow, room - 1), ctxLimit - pos - 1);
        if (want < 1) {
            return false;
        }
        if (cooldown > 0) { // recent passes accepted nothing: plain steps for a while
            cooldown--;
            return false;
        }
        if (best == 0 && ++unprofitable % PROBE_EVERY != 0 && !strongMatch()) {
            cooldown = PLAIN_STEPS - 1; // plain steps (which also re-measure their cost), then look again
            return false;
        }                               // (every PROBE_EVERY-th time a pass is tried anyway: the text may have changed)
        long t0 = System.nanoTime();
        int m = cfg.mode() == Mode.NGRAM ? lookup(want)
                : cfg.mode() == Mode.MTP ? mtpPropose(tok, want) : draft.propose(hist, len, want, drafts, vocab, stopIds);
        if (draft != null && draft.lastSteps() > 0) {
            cost.draft(System.nanoTime() - t0, draft.lastSteps());
        }
        TokenConstraint[] at = null; // at[j]: the constraint after tok, d1..dj
        if (con != null && m >= 1) {
            at = forks;
            at[0] = con.copy();
            int ok = 0;
            while (ok < m) {
                int d = drafts[ok];
                boolean stop = stopIds.contains(d);
                if (stop ? !at[ok].canEnd() : !at[ok].allows(d)) {
                    break;
                }
                ok++;
                if (stop) {
                    break; // nothing is verified after an end-of-sequence draft
                }
                at[ok] = at[ok - 1].copy();
                at[ok].accept(d);
                if (at[ok].mustEnd()) {
                    break; // the output is complete: nothing may follow
                }
            }
            m = ok;
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
        boolean mtp = cfg.mode() == Mode.MTP;
        if (mtp) {
            target.checkpointState(); // the recurrent state goes through every fed token: kept to undo the rejected one
        }
        target.forwardAll(feed, 0, m + 1, pos, rows, stride);
        long verifyNanos = System.nanoTime() - t1;
        cost.verify(m + 1, verifyNanos);
        int acc = 0;
        int examined = 0;
        readyCount = 0;
        for (int j = 0; j <= m; j++) {
            System.arraycopy(rows, j * stride, row, 0, stride);
            int s;
            if (at == null) {
                s = j < m ? sampler.verify(row, drafts[j]) : sampler.sample(row);
            } else {
                s = j < m ? sampler.verify(row, drafts[j], at[j], stopIds::contains)
                        : sampler.sample(row, at[j], stopIds::contains);
            }
            ready[readyCount++] = s;
            examined += j < m ? 1 : 0;
            if (s < 0) {
                break; // the constraint allows nothing (the engine ends the output there)
            }
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
        if (mtp && acc < m) { // the state ran through the rejected draft: back to after the accepted tokens
            target.rollbackState(acc + 1);
        }
        cost.acceptance(acc, examined);
        if (DEBUG) {
            System.err.printf("spec %s: pass of %d took %.1f ms, accepted %d; %s%n", cfg.mode(), m + 1, verifyNanos / 1e6, acc, cost.describe(maxDrafts));
        }
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

    /**
     * Prompt lookup under the cost model ({@code -Djllm.engine.adaptiveNgram=false} turns it off): when the measured
     * acceptance of this request's passes and the measured cost of a verification pass say that no draft beats plain
     * steps, lookups pause as a draft model's do. Costs a mixed workload nothing where text is copied, and spares the
     * backends whose multi-token pass is slow (the CPU, mixture-of-experts models) the failed passes of free text.
     */
    private final boolean gated;
    /** {@code -Djllm.engine.adaptiveDraft=false}: a draft model's length follows the doubling rule only. */
    private final boolean adaptive = !"false".equals(System.getProperty("jllm.engine.adaptiveDraft"));
    /** Plain steps taken when the cost model says speculation does not pay, before it is asked again. */
    static final int PLAIN_STEPS = 16;
    /** One in this many "does not pay" decisions runs a pass anyway, so a change in acceptance is noticed. */
    static final int PROBE_EVERY = 4;

    private static final boolean DEBUG = Boolean.getBoolean("jllm.engine.specDebug");
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

        /** Back to the prior acceptance (α 0.6): the next request is judged on its own passes. */
        void forgetAcceptance() {
            acc = 1.2;
            exam = 2.0;
        }

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

        String describe(int max) {
            if (passes < 1) {
                return "no data";
            }
            double[] ab = fit();
            return String.format(java.util.Locale.ROOT, "plain %.1f ms, a %.1f b %.1f ms, alpha %.2f, best k %d", plainNs / 1e6, ab[0] / 1e6,
                    ab[1] / 1e6, alpha(), bestK(max));
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

    /** The head's guess for the token after {@code tok} (argmax over the valid ids), into {@link #drafts}; 0 when it cannot. */
    private int mtpPropose(int tok, int want) {
        if (want < 1 || !target.mtpReady()) {
            return 0;
        }
        target.mtpDraft(tok, mtpLogits);
        int lim = Math.min(vocab, mtpLogits.length);
        int best = 0;
        for (int i = 1; i < lim; i++) {
            if (mtpLogits[i] > mtpLogits[best]) {
                best = i;
            }
        }
        drafts[0] = best;
        return 1;
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
     * Prompt lookup under the gate: whether the longest suffix ({@code engine.ngram-max} tokens) already occurred earlier.
     * Eight tokens repeating is strong evidence that the text is being copied (an edit of a file in the prompt), far
     * stronger than the prior acceptance a new request starts from, so a paused lookup tries a pass.
     */
    private boolean strongMatch() {
        int n = cfg.ngramMax();
        if (!gated || len - 1 < n) {
            return false;
        }
        int e = index.find(hist, len - 1, n);
        return e >= 0 && (cfg.window() <= 0 || e > len - 1 - cfg.window());
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
