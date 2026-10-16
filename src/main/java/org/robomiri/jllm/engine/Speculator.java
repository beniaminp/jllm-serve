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
 * <p>Drafts come from prompt lookup (the longest recent n-gram of the suffix found earlier in prompt + output; no
 * second model) or from a small {@link DraftModel} that shares the tokenizer. Both propose a deterministic
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
     * @param k        most tokens proposed per pass
     * @param ngramMax longest suffix n-gram looked up (prompt lookup)
     * @param ngramMin shortest suffix n-gram that may propose
     * @param window   how many of the most recent tokens are searched (prompt lookup)
     */
    public record Config(Mode mode, int k, int ngramMax, int ngramMin, int window) {
        public Config {
            if (k < 1 || ngramMax < 1 || ngramMin < 1 || ngramMin > ngramMax || window < 1) {
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
    private long draftedTotal;
    private long acceptedTotal;

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
    }

    /** Whether the backend can be used for speculation at all (a recurrent state cannot be rolled back). */
    public static boolean supported(Backend b) {
        return b.supportsPrefixReuse() && b.maxBatch() >= 2;
    }

    /** Starts a request: the history is the (prepared) prompt. */
    public void begin(int[] prompt) {
        len = 0;
        for (int t : prompt) {
            onToken(t);
        }
    }

    /** Appends a token of the output to the history. */
    public void onToken(int tok) {
        if (len == hist.length) {
            hist = Arrays.copyOf(hist, len * 2);
        }
        hist[len++] = tok;
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
        int want = Math.min(Math.min(maxDrafts, room - 1), ctxLimit - pos - 1);
        if (want < 1) {
            return false;
        }
        int m = cfg.mode() == Mode.NGRAM ? lookup(want) : draft.propose(hist, len, want, drafts, vocab, stopIds);
        if (m < 1) {
            return false;
        }
        feed[0] = tok;
        System.arraycopy(drafts, 0, feed, 1, m);
        target.forwardAll(feed, 0, m + 1, pos, rows, stride);
        int acc = 0;
        readyCount = 0;
        for (int j = 0; j <= m; j++) {
            System.arraycopy(rows, j * stride, row, 0, stride);
            int s = j < m ? sampler.verify(row, drafts[j]) : sampler.sample(row);
            ready[readyCount++] = s;
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
        preAccepted = acc;
        newPos = pos + 1 + acc;
        draftedTotal += m;
        acceptedTotal += acc;
        return true;
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
        int last = hist[len - 1];
        int bestLen = 0;
        int bestEnd = -1;
        int lo = Math.max(cfg.ngramMin() - 1, len - 1 - cfg.window());
        for (int e = len - 2; e >= lo; e--) {
            if (hist[e] != last) {
                continue;
            }
            int m = 1;
            while (m < cfg.ngramMax() && e - m >= 0 && hist[e - m] == hist[len - 1 - m]) {
                m++;
            }
            if (m > bestLen) {
                bestLen = m;
                bestEnd = e;
                if (m == cfg.ngramMax()) {
                    break;
                }
            }
        }
        if (bestLen < cfg.ngramMin()) {
            return 0;
        }
        int m = Math.min(want, len - 1 - bestEnd);
        System.arraycopy(hist, bestEnd + 1, drafts, 0, m);
        return m;
    }

    @Override
    public void close() {
        if (draft != null) {
            draft.close();
        }
    }
}
