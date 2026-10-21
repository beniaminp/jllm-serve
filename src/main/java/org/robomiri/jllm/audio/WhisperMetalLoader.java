package org.robomiri.jllm.audio;

import java.io.IOException;

/** Creates the Metal Whisper net when the machine has a GPU and the native library, else returns null. */
final class WhisperMetalLoader {

    private WhisperMetalLoader() {
    }

    /** {@code -Djllm.audio.metalExact=true}: float weight tiles for the linear layers too (the checkpoint's precision). */
    static final boolean EXACT = Boolean.getBoolean("jllm.audio.metalExact");

    static WhisperNet tryCreate(WhisperWeights w, int threads) {
        try {
            GpuCtx g = GpuCtx.create(System.getProperty("jllm.metal.lib"));
            if (g == null) {
                return null;
            }
            try {
                return new WhisperMetal(w, g, threads, EXACT);
            } catch (RuntimeException | Error e) {
                g.close();
                throw e;
            }
        } catch (IOException e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
    }
}
