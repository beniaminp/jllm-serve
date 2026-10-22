package org.robomiri.jllm.kv;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.function.Supplier;

/**
 * The pool of KV blocks of each loaded model: the first backend that runs on a model's weights creates it, the other
 * slots of the same model find it here (that is what lets them share blocks). Keyed weakly by the weights object, so a
 * pool goes away with its model.
 */
public final class KvPools {

    private static final Map<Object, Map<String, KvPool>> POOLS = Collections.synchronizedMap(new WeakHashMap<>());

    private KvPools() {
    }

    /**
     * The pool of {@code owner} (the model's weights) for {@code key} (backend, cache type, block size, context), made by
     * {@code create} on first use; {@code create} may return null (the model cannot be paged), which is remembered.
     */
    public static KvPool get(Object owner, String key, Supplier<KvPool> create) {
        synchronized (POOLS) {
            Map<String, KvPool> m = POOLS.computeIfAbsent(owner, o -> new HashMap<>());
            if (!m.containsKey(key)) {
                m.put(key, create.get());
            }
            return m.get(key);
        }
    }

    /** Closes and forgets the pools of {@code owner} (the model is being unloaded; no sequence of it is in use any more). */
    public static void release(Object owner) {
        Map<String, KvPool> m = owner == null ? null : POOLS.remove(owner);
        if (m != null) {
            for (KvPool p : m.values()) {
                if (p != null) {
                    p.close();
                }
            }
        }
    }

    /** The pool {@link #get} would return, without creating one. */
    public static KvPool peek(Object owner, String key) {
        Map<String, KvPool> m = POOLS.get(owner);
        return m == null ? null : m.get(key);
    }
}
