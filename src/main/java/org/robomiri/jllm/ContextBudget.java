package org.robomiri.jllm;

import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.MambaArch;
import org.robomiri.jllm.model.Qwen3NextArch;

import java.util.Locale;

/**
 * {@code backend.ctx=auto} (docs/10 §10.13): the context length a model gets when nobody chose one, sized from the
 * memory that is free for its KV caches.
 *
 * <p>The size is {@code min(cap, max(max_position_embeddings, 4096), floor(budget / (slots * bytesPerToken)))}, rounded
 * down to a multiple of 1024 (256 below 4096), with a floor of 1024. {@code budget} is {@link #SHARE} of what is free:
 * on Metal the device's recommended working set minus what the open backends and this model's weights hold (and a
 * reserve for activations); on the CPU backends the free Java heap, which holds their flat KV caches, or, with the
 * paged cache ({@code backend.kv-paged}, docs/24), whose blocks are off-heap, the physical memory alone. Both are capped by a
 * share of the physical memory. A mixture-of-experts model whose experts may go to the CPU
 * ({@code backend.metal.cpu-moe}) may also push up to {@link #MOE_SPILL} of the working set of its experts out for
 * the cache. The planner is a pure function ({@link #plan}) so it is tested without a device; {@link ModelLoader}
 * gathers the numbers.
 */
public final class ContextBudget {

    /** The context of every earlier version, and the least {@code auto} gives a model that has room for it. */
    public static final int FALLBACK = 4096;
    /** The least {@code auto} ever returns (a model whose weights leave almost nothing). */
    public static final int FLOOR = 1024;
    /** Default of {@code backend.ctx-max}: what agents need (32k - 128k) without a runaway for 1M-token models. */
    public static final int CAP = 131072;
    /** The part of the free memory the KV caches may take; the rest is activations, scratch, the OS and other programs. */
    public static final double SHARE = Double.parseDouble(System.getProperty("jllm.ctx.share", "0.6"));
    /** Metal, mixture-of-experts: working-set fraction of experts that may move to the CPU to make room for the cache. */
    public static final double MOE_SPILL = 0.10;
    /** Of the physical memory, what weights and KV caches together may fill. */
    static final double RAM_SHARE = 0.75;

    private ContextBudget() {
    }

    /**
     * The decision and the numbers behind it.
     *
     * @param ctx        tokens of context per slot
     * @param perToken   KV bytes of one position
     * @param budget     bytes the KV caches of all slots may take
     * @param limitedBy  {@code max_position_embeddings}, {@code backend.ctx-max} or {@code memory}
     * @param basis      the memory arithmetic, for the log
     */
    public record Plan(int ctx, long perToken, long budget, String limitedBy, String basis) {
        /** KV bytes of one slot at this context. */
        public long kvBytes() {
            return perToken * ctx;
        }

        public String describe() {
            return String.format(Locale.ROOT, "%d tokens (backend.ctx=auto: KV %.0f KB/token, %s; limited by %s)", ctx,
                    perToken / 1e3, basis, limitedBy);
        }
    }

    /** KV bytes per position of the whole model ({@code kv}: the cache element type); 0 for a model without a KV cache. */
    public static long bytesPerToken(Architecture arch, KvDtype kv) {
        return switch (arch) {
            case Decoder d -> switch (d.attention()) {
                case Attention.Gqa g -> 2L * d.core().layers() * kv.rowBytes(g.kvDim(), g.kvHeads());
                case Attention.Mla m -> (long) d.core().layers()
                        * (kv == KvDtype.Q4 ? KvDtype.Q8 : kv).rowBytes(m.kvLoraRank() + m.rope(), 1);
            };
            case Qwen3NextArch q -> 2L * q.fullLayers() * kv.rowBytes(q.attention().kvDim(), q.attention().kvHeads());
            case MambaArch m -> 0L;
            case org.robomiri.jllm.model.HybridArch h -> h.attention() == null ? 0L
                    : 2L * h.attentionLayers() * kv.rowBytes(h.attention().kvDim(), h.attention().kvHeads());
        };
    }

    /**
     * @param perToken     KV bytes of one position ({@link #bytesPerToken})
     * @param maxPositions the model's trained context ({@code max_position_embeddings}), 0 if unknown
     * @param cap          {@code backend.ctx-max}
     * @param budget       bytes for the KV caches of all slots together
     * @param slots        engines that will each hold a cache of this context
     */
    public static Plan plan(long perToken, int maxPositions, int cap, long budget, int slots, String basis) {
        int byModel = Math.max(maxPositions, FALLBACK);
        int want = Math.min(cap, byModel);
        String limit = cap < byModel ? "backend.ctx-max" : maxPositions >= FALLBACK ? "max_position_embeddings" : "the 4096 minimum";
        long fit = perToken <= 0 ? Long.MAX_VALUE : Math.max(0, budget) / Math.max(1, slots) / perToken;
        int ctx = want;
        if (fit < want) {
            ctx = (int) fit;
            limit = "memory";
        }
        ctx = ctx >= FALLBACK ? ctx / 1024 * 1024 : Math.max(FLOOR, ctx / 256 * 256);
        return new Plan(Math.min(ctx, want), perToken, Math.max(0, budget), limit, basis);
    }

    /**
     * The plan for {@code arch} on the backend {@code s} resolves to: Metal's working set (minus what the open
     * backends, this model's weights and the activations hold) or the free Java heap, each also bounded by the physical
     * memory.
     *
     * @param slots engines that will each get a cache of the chosen size
     * @param draft a draft model joins the target on the same device with a cache of the same length: half the budget
     */
    static Plan forModel(org.robomiri.jllm.config.Settings s, Architecture arch, org.robomiri.jllm.model.ModelWeights w,
                         int slots, boolean draft) {
        KvDtype kv = KvDtype.parse(s.get("backend.kv-dtype"));
        long per = bytesPerToken(arch, kv);
        if (s.getBool("backend.kv-sparse") && arch instanceof Decoder d && d.attention() instanceof Attention.Gqa g) {
            // sparse decode attention (docs/26): every block also keeps a min / max key summary of halves per layer and kv head
            per += (long) d.core().layers() * g.kvHeads() * 2 * g.headDim() * 2 / s.getInt("backend.kv-block");
        }
        long weights = w.byteSize();
        long ram = physicalMemory();
        long budget;
        String basis;
        long ws = ModelLoader.resolveBackend(s).equals("metal") ? org.robomiri.jllm.metal.MetalMemory.workingSet(s.opt("backend.metal.lib")) : 0;
        if (tierManaged(s, arch, kv)) {
            // tiered KV storage (docs/26): the cache is not bounded by the GPU's share. A slot that does not fit the hot tier keeps its
            // rows in host memory and streams them (backend.kv-stream), the planner (backend.memory-plan) grows the pool and moves
            // experts as contexts grow, and what the pool pushes out goes to host memory and the SSD. So the context is bounded
            // by the physical memory, as for the off-heap paged cache, and, when streaming is off, by the hot budget.
            budget = ram > 0 ? (long) (SHARE * Math.max(0, RAM_SHARE * ram - weights)) : Long.MAX_VALUE / 4;
            basis = String.format(Locale.ROOT, "tiered KV storage (docs/26): physical memory %.1f GB - weights %.1f GB, %.0f%% for the cache",
                    ram / 1e9, weights / 1e9, 100 * SHARE);
            long hot = s.getLong("backend.kv-hot-mb") * 1_000_000L;
            if (s.get("backend.kv-stream").trim().equalsIgnoreCase("off") && hot > 0 && hot < budget) {
                budget = hot;
                basis += String.format(Locale.ROOT, "; backend.kv-stream=off keeps it within backend.kv-hot-mb (%.1f GB)", hot / 1e9);
            }
        } else if (ws > 0) {
            long others = org.robomiri.jllm.metal.MetalMemory.residentBytes();
            long reserve = Math.max(512L << 20, ws / 32);
            boolean moe = arch instanceof Decoder d && d.anyMoe() || arch instanceof Qwen3NextArch q && q.anyMoe();
            long spill = moe && !s.get("backend.metal.cpu-moe").trim().equalsIgnoreCase("off") ? (long) (MOE_SPILL * ws) : 0;
            long free = Math.max(0, ws - others - weights - reserve + spill);
            budget = (long) (SHARE * free);
            basis = String.format(Locale.ROOT, "Metal working set %.1f GB - weights %.1f - other models %.1f - reserve %.1f%s = %.1f GB free, %.0f%% for the cache",
                    ws / 1e9, weights / 1e9, others / 1e9, reserve / 1e9, spill > 0 ? String.format(Locale.ROOT, " + %.1f experts may move to the CPU", spill / 1e9) : "",
                    free / 1e9, 100 * SHARE);
            if (ram > 0) {
                long byRam = (long) (SHARE * Math.max(0, RAM_SHARE * ram - weights));
                if (byRam < budget) { // the weights are memory-mapped: they and the cache must also fit in RAM
                    budget = byRam;
                    basis += String.format(Locale.ROOT, "; physical memory %.1f GB leaves %.1f GB", ram / 1e9, byRam / 1e9);
                }
            }
        } else if (ram > 0 && pagedOffHeap(s, arch, kv, w, slots)) {
            // the paged KV cache (backend.kv-paged, docs/24) keeps the CPU blocks in off-heap MemorySegment slabs: the Java heap
            // does not bound the cache, only the physical memory does (the pool never takes more than the flat caches would)
            budget = (long) (SHARE * Math.max(0, RAM_SHARE * ram - weights));
            basis = String.format(Locale.ROOT, "paged KV cache, off the Java heap: physical memory %.1f GB - weights %.1f GB, %.0f%% for the cache",
                    ram / 1e9, weights / 1e9, 100 * SHARE);
        } else {
            long heap = heapFree();
            budget = (long) (SHARE * heap);
            basis = String.format(Locale.ROOT, "free Java heap %.1f GB, %.0f%% for the cache", heap / 1e9, 100 * SHARE);
            if (ram > 0) {
                long byRam = (long) (SHARE * Math.max(0, RAM_SHARE * ram - weights));
                if (byRam < budget) {
                    budget = byRam;
                    basis += String.format(Locale.ROOT, "; physical memory %.1f GB leaves %.1f GB", ram / 1e9, byRam / 1e9);
                }
            }
        }
        if (draft) {
            budget /= 2;
            basis += "; half for the draft model's cache";
        }
        if (slots > 1) {
            basis += "; " + slots + " slots";
        }
        return plan(per, arch.core().maxPositions(), s.getInt("backend.ctx-max"), budget, slots, basis);
    }

    /**
     * Whether the cache is tiered ({@code backend.kv-tier}, docs/26) in a way that frees the context from the GPU's share: streaming
     * is allowed, or the planner owns the budget, or a hot budget is given. (Tiered with none of these, the pool is the whole GPU
     * share and the Metal arithmetic above applies.)
     */
    static boolean tierManaged(org.robomiri.jllm.config.Settings s, Architecture arch, KvDtype kv) {
        return s.getBool("backend.kv-tier") && arch instanceof Decoder d && d.attention() instanceof Attention.Gqa && kv != KvDtype.Q4
                && (!s.get("backend.kv-stream").trim().equalsIgnoreCase("off") || s.getBool("backend.memory-plan")
                || s.getLong("backend.kv-hot-mb") > 0);
    }

    /** Whether this model's CPU KV cache is the paged, off-heap pool, with {@code backend.kv-paged} set to true: an attention model with f32 / f16 / q8. */
    static boolean pagedOffHeap(org.robomiri.jllm.config.Settings s, Architecture arch, KvDtype kv) {
        return (s.get("backend.kv-paged").trim().equalsIgnoreCase("true") || s.getBool("backend.kv-sparse")) && pageable(arch, kv);
    }

    /** The same, with {@code auto} decided as {@link ModelLoader#paged} does for a model of {@code slots} slots. */
    static boolean pagedOffHeap(org.robomiri.jllm.config.Settings s, Architecture arch, KvDtype kv,
                                org.robomiri.jllm.model.ModelWeights w, int slots) {
        return pageable(arch, kv) && (ModelLoader.paged(s, w, slots) || s.getBool("backend.kv-sparse"));
    }

    private static boolean pageable(Architecture arch, KvDtype kv) {
        return arch instanceof Decoder d && d.attention() instanceof Attention.Gqa && kv != KvDtype.Q4;
    }

    /** Free space of the Java heap's old generation (where the CPU KV caches live), as {@code KvCache.checkHeap} sees it. */
    static long heapFree() {
        for (java.lang.management.MemoryPoolMXBean p : java.lang.management.ManagementFactory.getMemoryPoolMXBeans()) {
            String n = p.getName();
            if (p.getType() == java.lang.management.MemoryType.HEAP && (n.contains("Old") || n.contains("Tenured"))) {
                long max = p.getUsage().getMax();
                if (max > 0) {
                    return max - p.getUsage().getUsed();
                }
            }
        }
        Runtime rt = Runtime.getRuntime();
        return rt.maxMemory() - (rt.totalMemory() - rt.freeMemory());
    }

    /** Physical memory of the machine (the container's limit inside one), 0 if unknown. */
    static long physicalMemory() {
        if (java.lang.management.ManagementFactory.getOperatingSystemMXBean() instanceof com.sun.management.OperatingSystemMXBean os) {
            return os.getTotalMemorySize();
        }
        return 0;
    }
}
