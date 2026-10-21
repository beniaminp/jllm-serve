package org.robomiri.jllm.metal;

/**
 * Which mixture-of-experts layers of a Metal model keep their routed experts in CPU memory and run them on the CPU
 * ({@code backend.metal.cpu-moe}). The experts are almost all of an MoE model's bytes, and Metal will only keep
 * about {@code recommendedMaxWorkingSetSize} (roughly two thirds to three quarters of RAM) resident for the GPU; a
 * model above that pages on every token. Leaving some layers' experts out of the GPU's buffers keeps the GPU's share
 * resident, while those experts are read by the CPU from the memory-mapped file like any other file data.
 *
 * @param layers          {@link #AUTO} (just enough layers to fit the working set, 0 when the model fits), or a count
 *                        of MoE layers (the last ones), {@link Integer#MAX_VALUE} for all
 * @param threads         CPU threads for the experts
 * @param int8Activations integer dot products for single tokens, as {@code backend.cpu.int8-activations}
 */
public record MoeSplit(int layers, int threads, boolean int8Activations) {

    public static final int AUTO = -1;

    /** Every expert on the GPU. */
    public static final MoeSplit OFF = new MoeSplit(0, 1, false);

    public MoeSplit {
        if (layers < AUTO || threads < 1) {
            throw new IllegalArgumentException("bad MoE split " + layers + " layers, " + threads + " threads");
        }
    }

    /** {@code auto}, {@code off} (= 0), {@code all} or a number of layers. */
    public static int parseLayers(String s) {
        return switch (s.trim().toLowerCase(java.util.Locale.ROOT)) {
            case "auto" -> AUTO;
            case "off" -> 0;
            case "all" -> Integer.MAX_VALUE;
            default -> {
                int n;
                try {
                    n = Integer.parseInt(s.trim());
                } catch (NumberFormatException e) {
                    n = -1;
                }
                if (n < 0) {
                    throw new IllegalArgumentException("must be auto, off, all or a number of layers, got '" + s + "'");
                }
                yield n;
            }
        };
    }
}
