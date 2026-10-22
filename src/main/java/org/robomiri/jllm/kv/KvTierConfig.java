package org.robomiri.jllm.kv;

import java.nio.file.Path;

/**
 * The tiered KV storage settings a backend or pool is built with ({@code backend.kv-tier} and its companions, docs/26).
 *
 * @param hotBytes   KV bytes the fast tier may hold: the Metal pool's committed blocks (the GPU's share of the working
 *                   set), or the CPU pool's slabs in the process's own memory; 0 = no limit of its own (the planner or the
 *                   pool's capacity decide)
 * @param hostBytes  host memory for blocks pushed out of the hot pool, before they go to the SSD (0 = straight to the SSD)
 * @param diskFormat how the block files store rows ({@link KvBlockCodec.Format})
 * @param prefetch   bring blocks a prompt is about to adopt closer ahead of time
 * @param plan       the memory planner is on ({@code backend.memory-plan}): it sizes the hot tier and moves experts as contexts grow
 * @param stream     a slot whose whole context does not fit the hot budget keeps its KV in host memory and streams one layer
 *                   at a time to the GPU
 * @param dir        where the SSD tier lives (block files, memory-mapped slabs)
 */
public record KvTierConfig(long hotBytes, long hostBytes, KvBlockCodec.Format diskFormat, boolean prefetch, boolean plan, Stream stream,
                           Path dir) {

    /** Whether a slot may stream its KV layer by layer. */
    public enum Stream {
        /** Only when the slot's context cannot fit the hot budget. */
        AUTO,
        /** Never: a slot that does not fit fails when the pool is full. */
        OFF,
        /** Always (tests and measurements). */
        FORCE;

        public static Stream parse(String s) {
            return switch (s.trim().toLowerCase()) {
                case "auto" -> AUTO;
                case "off", "false" -> OFF;
                case "force", "on", "true" -> FORCE;
                default -> throw new IllegalArgumentException("must be auto, off or force, got '" + s + "'");
            };
        }
    }

    public KvTierConfig {
        if (hotBytes < 0 || hostBytes < 0) {
            throw new IllegalArgumentException("tier budgets must not be negative");
        }
        if (diskFormat == null) {
            diskFormat = KvBlockCodec.Format.NATIVE;
        }
        if (stream == null) {
            stream = Stream.AUTO;
        }
    }

    public KvTierConfig withHotBytes(long bytes) {
        return new KvTierConfig(bytes, hostBytes, diskFormat, prefetch, plan, stream, dir);
    }

    public KvTierConfig withStream(Stream s) {
        return new KvTierConfig(hotBytes, hostBytes, diskFormat, prefetch, plan, s, dir);
    }
}
