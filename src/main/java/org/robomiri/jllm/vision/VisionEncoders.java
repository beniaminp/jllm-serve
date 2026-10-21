package org.robomiri.jllm.vision;

/** Chooses the device a projector runs on. */
final class VisionEncoders {

    private VisionEncoders() {
    }

    static VisionEncoder create(ClipModel clip, VisionContext.Options o) {
        int threads = o.threads() > 0 ? o.threads() : Runtime.getRuntime().availableProcessors();
        return new ClipCpu(clip, threads);
    }
}
