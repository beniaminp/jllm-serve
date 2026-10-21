package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;

/** Placeholder while the CPU side is validated. */
public final class Qwen3NextMetalBackend implements Backend {

    public Qwen3NextMetalBackend(ModelWeights w, int ctx, int maxBatch, String metalLib, KvDtype kvDtype) {
        throw new UnsupportedOperationException("not yet");
    }

    @Override
    public String name() {
        return "metal qwen3-next";
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
