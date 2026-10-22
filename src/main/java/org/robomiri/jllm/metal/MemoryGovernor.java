package org.robomiri.jllm.metal;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The memory planner's running state for one model (docs/26 §26.2.6): when it last looked, how soon it may look again after a
 * refusal, the {@link MemoryPlanner.Hysteresis} that keeps expert moves from flapping, and the last decisions for the console.
 * Shared by the slots of a model (it hangs on the pool's storage), or private to a streaming slot.
 */
final class MemoryGovernor {

    /** Looks at the pool at most this often. */
    static final long CHECK_NANOS = Long.getLong("jllm.metal.planCheckMs", 2000) * 1_000_000L;
    /** After a refusal, the same growth is not tried again for this long. */
    static final long RETRY_NANOS = Long.getLong("jllm.metal.planRetryMs", 30_000) * 1_000_000L;

    final MemoryPlanner.Hysteresis hysteresis = new MemoryPlanner.Hysteresis(0.2, 30_000_000_000L);
    private final ArrayDeque<Map<String, Object>> history = new ArrayDeque<>();
    private long lastCheck;
    private long retryAt;
    private volatile List<String> plan = List.of();

    /** Whether it is time to look at the pool again. */
    synchronized boolean due(long now) {
        if (now - lastCheck < CHECK_NANOS || now < retryAt) {
            return false;
        }
        lastCheck = now;
        return true;
    }

    /** Do not try again before the retry interval has passed. */
    synchronized void backOff(long now) {
        retryAt = now + RETRY_NANOS;
    }

    synchronized void record(String kind, String detail) {
        Map<String, Object> d = new LinkedHashMap<>();
        d.put("atMillis", System.currentTimeMillis());
        d.put("kind", kind);
        d.put("detail", detail);
        history.addLast(d);
        while (history.size() > 40) {
            history.removeFirst();
        }
        System.err.println("Metal memory plan: " + kind + ": " + detail);
    }

    synchronized List<Map<String, Object>> history() {
        return new ArrayList<>(history);
    }

    void setPlan(List<String> lines) {
        this.plan = List.copyOf(lines);
    }

    List<String> plan() {
        return plan;
    }
}
