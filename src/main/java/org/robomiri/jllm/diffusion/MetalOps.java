package org.robomiri.jllm.diffusion;

/** placeholder until the Metal implementation lands */
final class MetalOps {
    static boolean available() {
        return false;
    }

    static Devices open(String precision, boolean xl) {
        throw new IllegalStateException("not yet");
    }
}
