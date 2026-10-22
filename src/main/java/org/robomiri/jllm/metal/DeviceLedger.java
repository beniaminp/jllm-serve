package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.ModelWeights;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

/**
 * Every open Metal backend of the process, attention and Mamba models alike. Metal's working set is the device's, so
 * a model planning its GPU memory ({@code backend.metal.cpu-moe}) or warning about it must count what the others
 * hold: their own buffers (KV caches, states, activations), and once per weights the part of them on the GPU.
 */
final class DeviceLedger {

    /** What one backend keeps resident on the GPU. */
    interface Resident {
        ModelWeights weights();

        /** Bytes of the weights given to the GPU (all of them, unless experts are left on the CPU). */
        long gpuWeightBytes();

        /** Bytes of the backend's own buffers. */
        long ownBytes();
    }

    private static final Set<Resident> LIVE = Collections.newSetFromMap(new IdentityHashMap<>());

    private DeviceLedger() {
    }

    static void add(Resident r) {
        synchronized (LIVE) {
            LIVE.add(r);
        }
    }

    static void remove(Resident r) {
        synchronized (LIVE) {
            LIVE.remove(r);
        }
    }

    /** GPU-resident bytes of every open backend: their buffers, and once per weights their GPU share. */
    static long total() {
        synchronized (LIVE) {
            long b = 0;
            Set<ModelWeights> counted = Collections.newSetFromMap(new IdentityHashMap<>());
            for (Resident o : LIVE) {
                b += o.ownBytes();
                if (counted.add(o.weights())) {
                    b += o.gpuWeightBytes();
                }
            }
            return b;
        }
    }

    /** GPU-resident bytes of every open backend but {@code self}: their buffers, and once per weights their GPU share. */
    static long others(Resident self) {
        synchronized (LIVE) {
            long b = 0;
            Set<ModelWeights> counted = Collections.newSetFromMap(new IdentityHashMap<>());
            for (Resident o : LIVE) {
                if (o == self) {
                    continue;
                }
                b += o.ownBytes();
                if (o.weights() != self.weights() && counted.add(o.weights())) {
                    b += o.gpuWeightBytes();
                }
            }
            return b;
        }
    }
}
