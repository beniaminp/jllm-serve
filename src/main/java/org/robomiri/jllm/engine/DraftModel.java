package org.robomiri.jllm.engine;

import org.robomiri.jllm.model.Backend;

import java.util.Arrays;
import java.util.Set;

/**
 * A small model of the same tokenizer that proposes tokens greedily. It has its own backend and KV cache; between
 * calls it remembers which tokens that cache holds and re-feeds only what differs from the new history (the tokens
 * the target accepted or replaced since the last proposal). On Metal its decode steps are pipelined (see
 * {@link #propose}).
 */
public final class DraftModel implements AutoCloseable {

    private final Backend backend;
    private final int vocabSize; // logits row width
    private final float[] logits;
    private int[] fed = new int[256]; // tokens whose K/V the backend holds, positions [0, fedLen)
    private int fedLen;
    private final int[] one = new int[1];
    private final Runnable afterClose;

    /** @param afterClose runs after the backend is closed (releases the draft's weights); may be null */
    public DraftModel(Backend backend, int vocabSize, Runnable afterClose) {
        this.backend = backend;
        this.afterClose = afterClose;
        this.vocabSize = vocabSize;
        this.logits = new float[vocabSize];
    }

    public Backend backend() {
        return backend;
    }

    /** Draft decode steps the last {@link #propose} ran (the speculator divides its time by them). */
    int lastSteps() {
        return lastSteps;
    }

    private int lastSteps;

    /**
     * Greedily proposes up to {@code want} tokens that follow {@code hist[0..len)}, into {@code out}.
     *
     * <p>The cache is "rolled back" by bookkeeping alone: positions are overwritten in place, so only the history after
     * the longest common prefix of what the cache holds ({@code fed}) and the new history is fed again (the drafts the
     * target accepted stay; a rejected draft's position is simply overwritten by the correction), whatever the cache's
     * element type (f32, f16 or q8 rows are each written whole for their position).
     *
     * <p>On a backend with a {@link Backend.GreedyPipe} (Metal) the draft's decode steps are pipelined: the last
     * history token and then each draft are fed by steps that take their token from the argmax the GPU computed in the
     * step before; the next step is always queued before the CPU reads the previous one's token, so the GPU runs the
     * steps back to back and only the argmax ids (4 bytes, not the logits) come back. Otherwise each step's logits come
     * back and the CPU picks the argmax.
     *
     * @param vocab ids at or above this are never proposed
     * @return the number proposed (0 when the draft's context is full); a stop token ends the proposal
     */
    int propose(int[] hist, int len, int want, int[] out, int vocab, Set<Integer> stopIds) {
        int ctx = backend.contextLength();
        want = Math.min(want, ctx - len - 1);
        lastSteps = 0;
        if (want < 1) {
            return 0;
        }
        int common = 0;
        int max = Math.min(fedLen, len - 1); // the last history token is always re-fed: its logits are needed
        while (common < max && fed[common] == hist[common]) {
            common++;
        }
        if (fed.length < len + want) {
            fed = Arrays.copyOf(fed, Math.max(fed.length * 2, len + want));
        }
        int lim = Math.min(vocab, vocabSize);
        Backend.GreedyPipe pipe = PIPELINE ? backend.greedyPipe(lim) : null;
        if (pipe != null) {
            int pos = common;
            while (pos < len - 1) { // the history before its last token: plain chunks, no logits needed
                int n = Math.min(backend.maxBatch(), len - 1 - pos);
                backend.forward(hist, pos, n, pos, null);
                pos += n;
            }
            System.arraycopy(hist, 0, fed, 0, len);
            fedLen = len;
            pipe.submit(hist[len - 1], len - 1);
            lastSteps = 1;
            int m = 0;
            for (int j = 0; j < want; j++) {
                if (j + 1 < want) {
                    pipe.submit(-1, len + j); // queued before step j is read: it feeds the draft step j chooses
                    lastSteps++;
                }
                int t = pipe.next();
                out[m++] = t;
                if (j + 1 < want) {
                    fed[fedLen++] = t; // fed by step j + 1, which is in flight
                }
                if (stopIds.contains(t)) {
                    pipe.drain(); // the step feeding the stop token finishes; nothing is queued after it
                    break;
                }
            }
            return m;
        }
        int pos = common;
        while (pos < len) {
            int n = Math.min(backend.maxBatch(), len - pos);
            backend.forward(hist, pos, n, pos, pos + n == len ? logits : null);
            pos += n;
        }
        System.arraycopy(hist, 0, fed, 0, len);
        fedLen = len;
        int m = 0;
        while (true) {
            int best = 0;
            for (int i = 1; i < lim; i++) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            out[m++] = best;
            lastSteps++;
            if (m == want || stopIds.contains(best)) {
                break;
            }
            one[0] = best;
            backend.forward(one, 0, 1, fedLen, logits);
            fed[fedLen++] = best;
        }
        return m;
    }

    /** {@code -Djllm.engine.draftPipeline=false}: the draft returns every step's logits to the CPU (for comparisons). */
    static final boolean PIPELINE = !"false".equals(System.getProperty("jllm.engine.draftPipeline"));

    @Override
    public void close() {
        try {
            backend.close();
        } finally {
            if (afterClose != null) {
                afterClose.run();
            }
        }
    }
}
