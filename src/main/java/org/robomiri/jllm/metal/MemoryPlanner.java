package org.robomiri.jllm.metal;

import java.util.ArrayList;
import java.util.List;

/**
 * One budget for the GPU's share of memory (docs/26 §26.2.6): the weights (with the experts of a mixture-of-experts
 * model that stay on the CPU, {@link ExpertPlacement}), the KV blocks, and everything else (activations, scratch, other
 * models). Two things can give the KV room, and the planner picks the cheaper by their cost per generated token:
 * <ul>
 *   <li><b>experts to the CPU</b> ({@link ExpertPlacement#plan}): the coldest experts first; the cost is the planner's own
 *       estimate (round trips of the layers that have CPU experts, plus the bytes of the picks that land on the CPU);
 *       independent of the context length;</li>
 *   <li><b>streaming the KV</b> (the cache lives in host memory and one layer at a time is copied to the GPU, {@link StreamedKv}):
 *       the GPU then needs only the two windows, but every token copies the whole cache through the CPU, so the cost
 *       grows with the context.</li>
 * </ul>
 * Experts are far cheaper per byte at long contexts (a token reads a few of them, but all of the cache), so the plan keeps
 * the KV resident and moves experts until even the last expert would cost more than streaming does.
 */
final class MemoryPlanner {

    /** Estimated host-to-GPU copy bandwidth of the streamed KV, bytes per microsecond (30 GB/s; {@link #stageBandwidth}). */
    static final double STAGE_BYTES_PER_US = ExpertPlacement.positiveProperty("jllm.metal.stageBytesPerUs", 30e3);

    private MemoryPlanner() {
    }

    enum Mode {RESIDENT, STREAMED}

    /**
     * @param budget           bytes the GPU may hold (Metal's recommended working set)
     * @param fixed            bytes on the GPU that are neither experts nor KV: the other weights, activations and scratch, other models
     * @param expertBytes      bytes of one expert of layer l (0 for a dense layer)
     * @param experts          experts of layer l
     * @param topK             experts per token of layer l
     * @param usage            picks counted per expert (null rows for none)
     * @param kvPerPosition    KV bytes of one position, all layers
     * @param kvPositions      positions the resident KV has to hold
     * @param canStream        whether a slot may stream its KV instead
     * @param windowBytes      GPU bytes the streaming windows take (all slots)
     * @param layers           decoder layers (one command buffer each when streaming)
     * @param typicalPositions context length the streaming cost is figured at
     * @param stageBytesPerUs  measured or estimated copy bandwidth of the streaming
     */
    record Inputs(long budget, long fixed, long[] expertBytes, int[] experts, int[] topK, long[][] usage, long kvPerPosition,
                  long kvPositions, boolean canStream, long windowBytes, int layers, long typicalPositions,
                  double stageBytesPerUs) {
    }

    /**
     * @param cold             the experts left on the CPU ({@code cold[l][e]}, null rows for dense layers)
     * @param expertBytesOnCpu bytes of those experts
     * @param kvGpuBytes       KV bytes on the GPU in this plan (the resident cache, or the windows)
     * @param gpuBytes         everything the plan puts on the GPU
     * @param fits             whether {@code gpuBytes} is within the budget
     * @param expertCostUs     the planner's cost per token of the CPU experts
     * @param streamCostUs     the cost per token of streaming the KV at {@code typicalPositions} (0 when resident)
     */
    record Plan(Mode mode, boolean[][] cold, long expertBytesOnCpu, long kvGpuBytes, long gpuBytes, boolean fits,
                double expertCostUs, double streamCostUs, String why) {
        double costUs() {
            return expertCostUs + streamCostUs;
        }
    }

    static long totalExpertBytes(Inputs in) {
        long t = 0;
        for (int l = 0; l < in.expertBytes().length; l++) {
            t += in.expertBytes()[l] * in.experts()[l];
        }
        return t;
    }

    private static long bytesOf(boolean[][] cold, long[] expertBytes) {
        long b = 0;
        for (int l = 0; l < cold.length; l++) {
            for (int e = 0; cold[l] != null && e < cold[l].length; e++) {
                b += cold[l][e] ? expertBytes[l] : 0;
            }
        }
        return b;
    }

    /** The cheaper of keeping {@code kvPositions} of KV resident and streaming it, under {@code in.budget()}. */
    static Plan plan(Inputs in) {
        long expertTotal = totalExpertBytes(in);
        long kvResident = in.kvPerPosition() * in.kvPositions();
        Plan resident = candidate(in, Mode.RESIDENT, kvResident, expertTotal, 0);
        if (!in.canStream() || in.kvPerPosition() == 0) {
            return resident;
        }
        double perToken = in.layers() * ExpertPlacement.SYNC_US
                + (double) in.typicalPositions() * in.kvPerPosition() / Math.max(1e-9, in.stageBytesPerUs());
        Plan streamed = candidate(in, Mode.STREAMED, in.windowBytes(), expertTotal, perToken);
        if (resident.fits() && (!streamed.fits() || resident.costUs() <= streamed.costUs())) {
            return resident;
        }
        if (streamed.fits()) {
            return streamed;
        }
        // neither fits: the one that overshoots less
        return resident.gpuBytes() <= streamed.gpuBytes() ? resident : streamed;
    }

    private static Plan candidate(Inputs in, Mode mode, long kvGpu, long expertTotal, double streamCost) {
        long need = in.fixed() + expertTotal + kvGpu - in.budget();
        boolean[][] cold = ExpertPlacement.plan(need, in.expertBytes(), in.experts(), in.topK(), in.usage());
        long moved = bytesOf(cold, in.expertBytes());
        long gpu = in.fixed() + expertTotal - moved + kvGpu;
        double ec = ExpertPlacement.cost(cold, in.expertBytes(), in.experts(), in.topK(), in.usage());
        String why = need <= 0 ? "everything fits" : String.format("%.1f GB over the budget: %.1f GB of experts to the CPU",
                need / 1e9, moved / 1e9);
        return new Plan(mode, cold, moved, kvGpu, gpu, gpu <= in.budget(), ec, streamCost, why);
    }

    /**
     * Stops the plan from changing back and forth: a new plan replaces the current one only when the current one no longer
     * fits, or when it is cheaper by at least {@code minGain} (a fraction of the current cost) and {@code minIntervalNanos}
     * have passed since the last change.
     */
    static final class Hysteresis {
        private final double minGain;
        private final long minIntervalNanos;
        private long lastChange;
        private boolean changed;

        Hysteresis(double minGain, long minIntervalNanos) {
            this.minGain = minGain;
            this.minIntervalNanos = minIntervalNanos;
        }

        boolean accept(boolean currentFits, double currentCostUs, double newCostUs, long nowNanos) {
            boolean ok = !currentFits
                    || ((!changed || nowNanos - lastChange >= minIntervalNanos) && newCostUs < (1 - minGain) * currentCostUs);
            if (ok) {
                lastChange = nowNanos;
                changed = true;
            }
            return ok;
        }
    }

    /**
     * The hot cap, in blocks, a pool should get next: it grows by {@code growth} (at least to hold what is committed with
     * the same headroom) once the committed blocks pass {@code high} of the cap, never beyond {@code full}; unchanged otherwise.
     */
    static int nextCap(int cap, int committed, int full, double high, double growth) {
        if (cap >= full || committed < high * cap) {
            return cap;
        }
        long grown = (long) Math.ceil(Math.max(cap * growth, committed * growth));
        return (int) Math.min(full, Math.max(cap + 1, grown));
    }

    /** One line per component, for the load log. */
    static List<String> describe(Inputs in, Plan p) {
        List<String> l = new ArrayList<>();
        l.add(String.format("budget %.1f GB: fixed %.1f GB, experts %.1f GB (%.1f GB on the CPU), KV %.1f GB on the GPU (%s)",
                in.budget() / 1e9, in.fixed() / 1e9, totalExpertBytes(in) / 1e9, p.expertBytesOnCpu() / 1e9, p.kvGpuBytes() / 1e9,
                p.mode().name().toLowerCase()));
        l.add(String.format("cost per token: experts %.0f us + streaming %.0f us; %s%s", p.expertCostUs(), p.streamCostUs(), p.why(),
                p.fits() ? "" : "; DOES NOT FIT"));
        return l;
    }
}
