package org.robomiri.jllm.metal;

/** {@link MemoryPlanner}: the trade between experts and KV, streaming, the hysteresis and the growth steps. No model, no GPU. */
public final class MemoryPlannerTest {

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    private static final long GB = 1_000_000_000L;

    /** 48 MoE layers of 128 experts (top 8), 6.2 MB per expert: 38 GB of experts, like a 30B-A3B at f16. */
    private static MemoryPlanner.Inputs inputs(long budget, long fixed, long kvPositions, boolean canStream, long typical) {
        int layers = 48;
        long[] eb = new long[layers];
        int[] ex = new int[layers];
        int[] top = new int[layers];
        for (int l = 0; l < layers; l++) {
            eb[l] = 3_100_000L;
            ex[l] = 128;
            top[l] = 8;
        }
        return new MemoryPlanner.Inputs(budget, fixed, eb, ex, top, null, 98_304L, kvPositions, canStream, 2 * 2 * 32768L * 1024L,
                layers, typical, MemoryPlanner.STAGE_BYTES_PER_US);
    }

    private static int coldExperts(MemoryPlanner.Plan p) {
        int n = 0;
        for (boolean[] row : p.cold()) {
            for (int e = 0; row != null && e < row.length; e++) {
                n += row[e] ? 1 : 0;
            }
        }
        return n;
    }

    public static void main(String[] args) {
        long experts = 48L * 128 * 3_100_000L; // 19 GB
        // everything fits: nothing moves
        MemoryPlanner.Plan fits = MemoryPlanner.plan(inputs(30 * GB, 2 * GB, 8192, true, 8192));
        check(fits.mode() == MemoryPlanner.Mode.RESIDENT && coldExperts(fits) == 0 && fits.fits(), "a model that fits moves nothing: " + fits.why());

        // the KV grows: more experts leave the GPU, never fewer
        int last = -1;
        boolean monotone = true;
        for (long pos : new long[]{4096, 16384, 65536, 131072}) {
            MemoryPlanner.Plan p = MemoryPlanner.plan(inputs(22 * GB, 2 * GB, pos, false, pos));
            int c = coldExperts(p);
            monotone &= c >= last && p.mode() == MemoryPlanner.Mode.RESIDENT;
            last = c;
            System.out.printf("     %6d positions of KV (%.1f GB): %d experts of %d on the CPU, fits %s, cost %.0f us%n", pos,
                    pos * 98_304L / 1e9, c, 48 * 128, p.fits(), p.costUs());
        }
        check(monotone, "as the context grows, experts leave the GPU one after the other and the KV stays resident");

        // a context that cannot fit even with every expert on the CPU: only streaming fits
        MemoryPlanner.Plan huge = MemoryPlanner.plan(inputs(22 * GB, 2 * GB, 1_000_000, true, 1_000_000));
        MemoryPlanner.Plan hugeNoStream = MemoryPlanner.plan(inputs(22 * GB, 2 * GB, 1_000_000, false, 1_000_000));
        check(huge.mode() == MemoryPlanner.Mode.STREAMED && huge.fits() && !hugeNoStream.fits(),
                "1M positions (98 GB): resident cannot fit, streaming does (" + huge.why() + ")");

        // at a short typical context streaming is cheap, resident wins only when its expert cost is lower
        MemoryPlanner.Plan mid = MemoryPlanner.plan(inputs(22 * GB, 2 * GB, 40000, true, 40000));
        MemoryPlanner.Plan midRes = MemoryPlanner.plan(inputs(22 * GB, 2 * GB, 40000, false, 40000));
        check(mid.costUs() <= midRes.costUs() + 1e-6, "the plan with streaming allowed is never more expensive than without");

        // usage statistics: the hot experts stay, so the same bytes cost less than with uniform use
        long[][] usage = new long[48][];
        for (int l = 0; l < 48; l++) {
            usage[l] = new long[128];
            for (int e = 0; e < 128; e++) {
                usage[l][e] = e < 16 ? 100000 : 10; // 16 hot experts take almost every pick
            }
        }
        MemoryPlanner.Inputs base = inputs(22 * GB, 2 * GB, 60000, false, 60000);
        MemoryPlanner.Inputs withUsage = new MemoryPlanner.Inputs(base.budget(), base.fixed(), base.expertBytes(), base.experts(),
                base.topK(), usage, base.kvPerPosition(), base.kvPositions(), false, base.windowBytes(), base.layers(),
                base.typicalPositions(), base.stageBytesPerUs());
        MemoryPlanner.Plan uni = MemoryPlanner.plan(base);
        MemoryPlanner.Plan skew = MemoryPlanner.plan(withUsage);
        boolean hotKept = true;
        for (boolean[] row : skew.cold()) {
            for (int e = 0; row != null && e < 16; e++) {
                hotKept &= !row[e];
            }
        }
        check(hotKept && skew.expertCostUs() < uni.expertCostUs(), String.format(
                "with usage statistics the hot experts stay on the GPU and the cost falls (%.0f -> %.0f us)", uni.expertCostUs(), skew.expertCostUs()));

        // hysteresis
        MemoryPlanner.Hysteresis h = new MemoryPlanner.Hysteresis(0.2, 60_000_000_000L);
        long t = 0;
        check(!h.accept(true, 100, 95, t), "a 5 % gain does not move anything");
        check(h.accept(true, 100, 70, t), "a 30 % gain does (the first change needs no wait)");
        check(!h.accept(true, 100, 50, t + 1_000_000_000L), "a second change within the minimum interval waits");
        check(h.accept(true, 100, 50, t + 61_000_000_000L), "and happens after it");
        check(h.accept(false, 100, 99, t + 62_000_000_000L), "a plan that no longer fits is replaced at once");
        // flapping inputs (KV demand oscillating around a threshold) produce few changes
        MemoryPlanner.Hysteresis g = new MemoryPlanner.Hysteresis(0.2, 60_000_000_000L);
        int changes = 0;
        double cost = 100;
        for (int i = 0; i < 1000; i++) {
            double next = cost * (i % 2 == 0 ? 0.9 : 1.1); // +-10 %: inside the dead band
            if (g.accept(true, cost, next, i * 1_000_000_000L)) {
                changes++;
                cost = next;
            }
        }
        check(changes == 0, "demand flapping by 10 % causes no re-plan (" + changes + " changes in 1000 checks)");

        // cap growth: geometric, with headroom, bounded
        check(MemoryPlanner.nextCap(1000, 500, 100000, 0.85, 1.5) == 1000, "no growth below the high-water fraction");
        check(MemoryPlanner.nextCap(1000, 900, 100000, 0.85, 1.5) == 1500, "growth by half at 85 % committed");
        check(MemoryPlanner.nextCap(1000, 1000, 1200, 0.85, 1.5) == 1200, "bounded by the full pool");
        check(MemoryPlanner.nextCap(1200, 1200, 1200, 0.85, 1.5) == 1200, "no growth once full");
        System.out.println("     experts total " + experts / 1e9 + " GB");
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("memory planner test passed");
    }
}
