package org.robomiri.jllm.metal;

/**
 * What the Metal device can hold, for the code that sizes things before a backend exists (the default context length,
 * {@code backend.ctx=auto}): the device's recommended working set and what the open backends already keep resident
 * ({@link DeviceLedger}).
 */
public final class MetalMemory {

    private static volatile long workingSet = -1;

    private MetalMemory() {
    }

    /**
     * Metal's {@code recommendedMaxWorkingSetSize} (or {@code -Djllm.metal.workingSetGb}, as the backend reads it), in
     * bytes; 0 when it cannot be asked. Asked once per process with a context that compiles one empty kernel, so it
     * costs a few milliseconds.
     */
    public static long workingSet(String explicitLib) {
        String gb = System.getProperty("jllm.metal.workingSetGb");
        if (gb != null) {
            try {
                double v = Double.parseDouble(gb);
                if (v > 0) {
                    return (long) (v * 1e9);
                }
            } catch (NumberFormatException e) {
                // fall through to the device
            }
        }
        long ws = workingSet;
        if (ws < 0) {
            try {
                Metal m = Metal.createFromSource(explicitLib, "kernel void jm_probe(device float* a [[buffer(0)]],"
                        + " uint i [[thread_position_in_grid]]) { a[i] = 0.0f; }");
                try {
                    ws = m.recommendedWorkingSet();
                } finally {
                    m.close();
                }
            } catch (RuntimeException | LinkageError e) {
                ws = 0;
            }
            workingSet = ws;
        }
        return ws;
    }

    /** GPU-resident bytes of every open Metal backend of this process (their buffers and their share of the weights). */
    public static long residentBytes() {
        return DeviceLedger.total();
    }
}
