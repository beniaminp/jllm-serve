package org.robomiri.jllm.model;

import java.util.Arrays;

/**
 * The bookkeeping of a multi-token-prediction head ({@link Qwen3NextMtp}) that follows a backend's forward passes: which
 * rows of the head have been computed, with which hidden state and token, so that the head's own KV cache holds, for
 * every position {@code t} already passed, the row {@code (h_t, x_{t+1})} (the main model's final hidden state after
 * {@code x_t} and the token that followed) and nothing else.
 *
 * <p>A pass that feeds tokens {@code x_pos .. x_{pos+n-1}} yields the hidden states {@code h_pos .. h_{pos+n-1}}. The
 * pairs it completes are {@code (h_t, x_{t+1})} for {@code t = pos .. pos+n-2}, and {@code (h_{pos-1}, x_pos)} for the
 * position before it, whose hidden state was <em>pending</em> since the previous pass (the last position's token is only
 * known after sampling). {@link #afterPass} runs those rows; the last hidden state becomes the new pending one. A draft
 * ({@link #draft}) runs the pending row early, with the token just sampled, and returns the head's logits: the guess for
 * the token after it; the next pass then skips that row.
 *
 * <p>Rows of a verification pass that are later rejected need no undoing: the next pass starts at the first rejected
 * position and recomputes the row of the position before it from the saved hidden state (the pending one, kept by
 * {@link #snapshot} / {@link #restore} together with the recurrent state of the main model); the rows after it are
 * overwritten position by position.
 */
public final class MtpFeeder {

    /** What the backend does for the head: a batch of rows at consecutive positions. */
    public interface Head {
        /**
         * Runs the head on {@code n} rows: row {@code i} is the main model's normed hidden state {@code hidden[i * dim ..)}
         * and the token that followed it {@code tokens[i]}, at position {@code pos + i}; writes the head's K/V for them.
         *
         * @param logits null, or the array that receives the head's logits of the <em>last</em> row
         */
        void rows(int[] tokens, float[] hidden, int n, int pos, float[] logits);
    }

    private final int dim;
    private final Head head;
    private boolean enabled;
    private boolean valid;
    private float[] pending;
    private int pendingPos = -1;
    private int doneToken = -1; // the pending position's row was run with this token already (a draft)
    private float[] rowsH = new float[0];
    private int[] rowsT = new int[0];
    private float[] savedPending;
    private int savedPendingPos;
    private int savedDoneToken;
    private boolean savedValid;

    public MtpFeeder(int dim, Head head) {
        this.dim = dim;
        this.head = head;
        this.pending = new float[dim];
        this.savedPending = new float[dim];
    }

    public void enable(boolean on) {
        enabled = on;
        if (!on) {
            valid = false;
        }
    }

    public boolean enabled() {
        return enabled;
    }

    /** True when the head holds every row up to the last pass and a draft can be asked for. */
    public boolean ready(int statePos) {
        return enabled && valid && pendingPos >= 0 && pendingPos == statePos - 1;
    }

    /** The head's K/V no longer match the main state (it was loaded or copied): drafts are off until a prompt starts again. */
    public void invalidate() {
        valid = false;
    }

    /**
     * Called after a main pass of {@code n} tokens {@code tokens[off ..)} at {@code pos} with their normed hidden states
     * {@code hidden[0 .. n * dim)}.
     */
    public void afterPass(int[] tokens, int off, int n, int pos, float[] hidden) {
        if (!enabled) {
            return;
        }
        if (pos == 0) {
            valid = true;
            pendingPos = -1;
            doneToken = -1;
        } else if (!valid || pendingPos != pos - 1) {
            valid = false;
            return;
        }
        boolean withPending = pos > 0 && !(doneToken == tokens[off]);
        int rows = (withPending ? 1 : 0) + n - 1;
        if (rows > 0) {
            if (rowsH.length < rows * dim) {
                rowsH = new float[rows * dim];
                rowsT = new int[rows];
            }
            int r = 0;
            if (withPending) {
                System.arraycopy(pending, 0, rowsH, 0, dim);
                rowsT[r++] = tokens[off];
            }
            for (int i = 0; i < n - 1; i++) {
                System.arraycopy(hidden, i * dim, rowsH, r * dim, dim);
                rowsT[r++] = tokens[off + 1 + i];
            }
            head.rows(rowsT, rowsH, rows, withPending ? pos - 1 : pos, null);
        }
        System.arraycopy(hidden, (n - 1) * dim, pending, 0, dim);
        pendingPos = pos + n - 1;
        doneToken = -1;
    }

    /**
     * Runs the head's pending row with {@code token} (the token just sampled after the last processed position) and
     * returns, in {@code logits}, its guess for the token after it.
     */
    public void draft(int token, float[] logits) {
        head.rows(new int[]{token}, pending, 1, pendingPos, logits);
        doneToken = token;
    }

    /**
     * After a pass of which only the first {@code kept} tokens count (the rest was a rejected draft): the last kept
     * position's hidden state is the pending one again, as if the pass had been that short. The head's rows computed for
     * positions past it are overwritten by the next pass.
     */
    public void truncate(int pos, int kept, float[] hidden) {
        System.arraycopy(hidden, (kept - 1) * dim, pending, 0, dim);
        pendingPos = pos + kept - 1;
        doneToken = -1;
    }

    /** Remembers the pending hidden state (before a verification pass that may be rolled back). */
    public void snapshot() {
        System.arraycopy(pending, 0, savedPending, 0, dim);
        savedPendingPos = pendingPos;
        savedDoneToken = doneToken;
        savedValid = valid;
    }

    public void restore() {
        System.arraycopy(savedPending, 0, pending, 0, dim);
        pendingPos = savedPendingPos;
        doneToken = savedDoneToken;
        valid = savedValid;
    }

    @Override
    public String toString() {
        return "mtp(enabled=" + enabled + " valid=" + valid + " pending@" + pendingPos + " done=" + doneToken + ")";
    }

    /** For tests. */
    int[] rowTokens() {
        return Arrays.copyOf(rowsT, rowsT.length);
    }
}
