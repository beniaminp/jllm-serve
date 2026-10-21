package org.robomiri.jllm.metal;

/**
 * Which mixture-of-experts layers of a Metal model keep their routed experts in CPU memory and run them on the CPU
 * ({@code backend.metal.cpu-moe}). The experts are almost all of an MoE model's bytes, and Metal will only keep
 * about {@code recommendedMaxWorkingSetSize} (roughly two thirds to three quarters of RAM) resident for the GPU; a
 * model above that pages on every token. Leaving some experts out of the GPU's buffers keeps the GPU's share
 * resident, while those experts are read by the CPU from the memory-mapped file like any other file data.
 *
 * @param layers          {@link #AUTO} (just enough experts to fit the working set, none when the model fits: the
 *                        least used ones by the usage of earlier runs, see {@link ExpertPlacement}), or a count of
 *                        whole MoE layers (the last ones), {@link Integer#MAX_VALUE} for all
 * @param threads         CPU threads for the experts
 * @param int8Activations integer dot products for single tokens, as {@code backend.cpu.int8-activations}
 * @param cold            an explicit placement instead of {@code layers}: {@code cold[l][e]} for the experts of layer l
 *                        left on the CPU (null rows for none), e.g. to test split layers; normally null
 * @param backends        how many backends will run on these weights (a server's slots): {@link #AUTO} budgets for
 *                        all of their caches, since the later ones reuse the first one's placement
 */
public record MoeSplit(int layers, int threads, boolean int8Activations, boolean[][] cold, int backends) {

    public static final int AUTO = -1;

    /** Every expert on the GPU. */
    public static final MoeSplit OFF = new MoeSplit(0, 1, false);

    public MoeSplit(int layers, int threads, boolean int8Activations) {
        this(layers, threads, int8Activations, null, 1);
    }

    /** These experts on the CPU, the rest on the GPU. */
    public static MoeSplit experts(boolean[][] cold, int threads, boolean int8Activations) {
        return new MoeSplit(0, threads, int8Activations, cold, 1);
    }

    public MoeSplit {
        if (layers < AUTO || threads < 1 || backends < 1) {
            throw new IllegalArgumentException("bad MoE split " + layers + " layers, " + threads + " threads, " + backends
                    + " backends");
        }
    }

    /** What the placement depends on (backends sharing weights reuse a placement made for the same config). */
    String config() {
        return layers + "|" + java.util.Arrays.deepToString(cold);
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
