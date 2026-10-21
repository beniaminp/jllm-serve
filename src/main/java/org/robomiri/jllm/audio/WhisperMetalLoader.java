package org.robomiri.jllm.audio;

import java.io.IOException;

/** Creates the Metal Whisper net when the machine has a GPU and the native library, else returns null. */
final class WhisperMetalLoader {

    private WhisperMetalLoader() {
    }

    /** {@code -Djllm.audio.metalExact=true}: float weight tiles for the linear layers too (the checkpoint's precision). */
    static boolean exact() {
        return Boolean.getBoolean("jllm.audio.metalExact");
    }

    /**
     * @param decoder "on", "off" or "auto" (the decoder runs on the GPU when the model is at least 768 wide: for small
     *                models a step costs more in dispatches than the CPU needs to compute it)
     */
    static WhisperNet tryCreate(WhisperWeights w, int threads, String decoder) {
        try {
            GpuCtx g = GpuCtx.create(System.getProperty("jllm.metal.lib"));
            if (g == null) {
                return null;
            }
            try {
                decoder = System.getProperty("jllm.audio.metalDecoder", decoder);   // tests and experiments
                boolean gpuDecoder = decoder.equals("on") || (decoder.equals("auto") && w.cfg.dModel() >= 768);
                return new WhisperMetal(w, g, threads, exact(), gpuDecoder);
            } catch (RuntimeException | Error e) {
                g.close();
                throw e;
            }
        } catch (IOException e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
    }
}
