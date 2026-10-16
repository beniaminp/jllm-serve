package org.robomiri.jllm.engine;

import org.robomiri.jllm.model.Backend;

import java.util.Arrays;
import java.util.Set;

/**
 * A small model of the same tokenizer that proposes tokens greedily. It has its own backend and KV cache; between
 * calls it remembers which tokens that cache holds and re-feeds only what differs from the new history (the tokens
 * the target accepted or replaced since the last proposal).
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

    /**
     * Greedily proposes up to {@code want} tokens that follow {@code hist[0..len)}, into {@code out}.
     *
     * @param vocab ids at or above this are never proposed
     * @return the number proposed (0 when the draft's context is full); a stop token ends the proposal
     */
    int propose(int[] hist, int len, int want, int[] out, int vocab, Set<Integer> stopIds) {
        int ctx = backend.contextLength();
        want = Math.min(want, ctx - len - 1);
        if (want < 1) {
            return 0;
        }
        int common = 0;
        int max = Math.min(fedLen, len - 1); // the last history token is always re-fed: its logits are needed
        while (common < max && fed[common] == hist[common]) {
            common++;
        }
        int pos = common;
        while (pos < len) {
            int n = Math.min(backend.maxBatch(), len - pos);
            backend.forward(hist, pos, n, pos, pos + n == len ? logits : null);
            pos += n;
        }
        if (fed.length < len + want) {
            fed = Arrays.copyOf(fed, Math.max(fed.length * 2, len + want));
        }
        System.arraycopy(hist, 0, fed, 0, len);
        fedLen = len;
        int lim = Math.min(vocab, vocabSize);
        int m = 0;
        while (true) {
            int best = 0;
            for (int i = 1; i < lim; i++) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            out[m++] = best;
            if (m == want || stopIds.contains(best)) {
                break;
            }
            one[0] = best;
            backend.forward(one, 0, 1, fedLen, logits);
            fed[fedLen++] = best;
        }
        return m;
    }

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
