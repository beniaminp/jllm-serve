package org.robomiri.jllm.kv;

import java.util.LinkedHashMap;
import java.util.Map;

/** JSON-ready descriptions of a pool and its tiers for the console and the memory endpoint (docs/26 §26.2.7). */
public final class KvReports {

    private KvReports() {
    }

    /** The pool's blocks by tier: hot (in the storage), warm (host memory), cold (files), and the traffic between them. */
    public static Map<String, Object> pool(KvPool pool) {
        Map<String, Object> out = new LinkedHashMap<>();
        KvPool.Stats s = pool.stats();
        long blockBytes = pool.storage().residentBytes();
        Map<String, Object> kv = new LinkedHashMap<>();
        kv.put("mode", "paged");
        kv.put("blockTokens", s.blockTokens());
        kv.put("blockBytes", blockBytes);
        Map<String, Object> hot = new LinkedHashMap<>();
        hot.put("capBlocks", s.hotCap());
        hot.put("capBytes", (long) s.hotCap() * blockBytes);
        hot.put("committedBlocks", s.committedBlocks());
        hot.put("committedBytes", s.committedBytes());
        hot.put("usedBlocks", s.usedBlocks());
        hot.put("cachedBlocks", s.cachedBlocks());
        hot.put("freeBlocks", s.freeBlocks());
        hot.put("maxBlocks", s.maxBlocks());
        kv.put("hot", hot);
        Map<String, Object> warm = new LinkedHashMap<>();
        Map<String, Object> cold = new LinkedHashMap<>();
        KvSpill sp = pool.spill();
        if (sp instanceof TieredSpill t) {
            TieredSpill.Stats ts = t.stats();
            warm.put("blocks", ts.ramBlocks());
            warm.put("bytes", ts.ramBytes());
            warm.put("budgetBytes", ts.ramBudget());
            warm.put("hits", ts.ramHits());
            warm.put("prefetches", ts.prefetches());
            warm.put("prefetchHits", ts.prefetchHits());
            cold.put("demotions", ts.demotions());
            cold.put("dropped", ts.dropped());
            cold.put("hits", ts.lowerHits());
            cold.put("files", Math.max(0, s.spillFiles() - ts.ramBlocks()));
            cold.put("bytes", Math.max(0, s.spillBytes() - ts.ramBytes()));
        } else if (sp != null) {
            cold.put("files", s.spillFiles());
            cold.put("bytes", s.spillBytes());
        }
        kv.put("warm", warm);
        kv.put("cold", cold);
        Map<String, Object> traffic = new LinkedHashMap<>();
        traffic.put("evictions", s.evictions());
        traffic.put("spills", s.spills());
        traffic.put("reloads", s.reloads());
        traffic.put("prefetchHints", s.prefetchHints());
        traffic.put("copiesOnWrite", s.copiesOnWrite());
        traffic.put("dedups", s.dedups());
        kv.put("traffic", traffic);
        out.put("kv", kv);
        return out;
    }
}
