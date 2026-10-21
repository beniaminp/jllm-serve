package org.robomiri.jllm.vision;

/** Chooses the device a projector runs on. */
final class VisionEncoders {

    private VisionEncoders() {
    }

    /**
     * {@code metal} runs the encoder on the GPU and fails if there is none; {@code auto} takes the GPU when it can and
     * falls back to the CPU (also when a projector type has no GPU path yet); {@code cpu} never uses the GPU.
     */
    static VisionEncoder create(ClipModel clip, VisionContext.Options o) {
        String b = o.backend() == null ? "cpu" : o.backend();
        if (b.equals("metal") || b.equals("auto")) {
            boolean gpu = ClipMetal.available(o.metalLib());
            if (gpu && !System.getProperty("jllm.vision.noMetal", "false").equals("true")) {
                try {
                    return new ClipMetal(clip, o.metalLib());
                } catch (RuntimeException e) {
                    if (b.equals("metal")) {
                        throw e;
                    }
                    System.err.println("note: the vision encoder cannot use the GPU (" + e.getMessage() + "), using the CPU");
                }
            } else if (b.equals("metal")) {
                throw new IllegalArgumentException("vision.backend=metal, but the Metal library is not available");
            }
        }
        int threads = o.threads() > 0 ? o.threads() : Runtime.getRuntime().availableProcessors();
        return new ClipCpu(clip, threads);
    }
}
