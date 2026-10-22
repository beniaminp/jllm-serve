package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;

/** Placeholder until the Metal backend of the hybrids is written. */
public final class HybridMetalBackend implements Backend {
    public HybridMetalBackend(ModelWeights w, int ctx, int maxBatch, String lib, KvDtype kv) {
        throw new UnsupportedOperationException("not yet");
    }

    @Override
    public String name() {
        return null;
    }

    @Override
    public int contextLength() {
        return 0;
    }

    @Override
    public int maxBatch() {
        return 0;
    }

    @Override
    public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
    }

    @Override
    public void close() {
    }
}
