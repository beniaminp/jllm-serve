package org.robomiri.jllm.web.models;

import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.engine.Admission;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.SlotPool.Slot;
import org.robomiri.jllm.engine.SlotPool;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * A model in memory, ready to serve. Requests hold a {@link Lease} while they use the engine, so that unloading
 * waits for them (after asking them to stop) before the backend's buffers and the mapped weights are released.
 *
 * <p>A model may have several <em>slots</em>: engines with their own KV cache on the same mapped weights. Each
 * request goes to the slot that already holds the longest prefix of its prompt, so a coding agent's long
 * conversation stays cached while its short side requests (titles, summaries) run in another slot, and two
 * requests can run at the same time.
 */
public final class LoadedModel {

    /** Thrown when a request arrives for a model that is being unloaded. */
    public static final class UnavailableException extends RuntimeException {
        UnavailableException(String message) {
            super(message);
        }
    }

    public interface Lease extends AutoCloseable {
        @Override
        void close();
    }

    /**
     * @param settings the load-time settings of the model (backend, context, quantization, speculation)
     * @param slots    the KV cache slots; {@code busy} of them are running or reserved by a request
     */
    public record View(String id, String source, String description, String backend, int contextLength,
                       String quant, long weightBytes, String template, long loadedAt, double loadSeconds,
                       long requests, long generatedTokens, double lastTokensPerSecond, int active,
                       Map<String, String> settings, int slots, int busy, boolean vision, String projector, String kind) {
    }

    private final String id;
    private final String source;
    private final KvStore store; // null: no saved caches
    private final ModelLoader.Loaded loaded;
    private final SlotPool pool;
    private final Map<String, String> settings;
    private final long loadedAt = System.currentTimeMillis();
    private final double loadSeconds;
    private final ReentrantReadWriteLock guard = new ReentrantReadWriteLock();
    private final AtomicLong requests = new AtomicLong();
    private final AtomicLong generated = new AtomicLong();
    private final AtomicInteger active = new AtomicInteger();
    private volatile double lastTokPerSec;
    private volatile boolean closing;

    /** @param extra the engines of slots 1.. (slot 0 is {@code loaded}'s) */
    LoadedModel(String id, String source, ModelLoader.Loaded loaded, List<Engine> extra, Map<String, String> settings,
                double loadSeconds, KvStore store) {
        this.id = id;
        this.source = source;
        this.store = store;
        this.loaded = loaded;
        List<Engine> engines = new ArrayList<>();
        engines.add(loaded.engine());
        engines.addAll(extra);
        this.settings = Map.copyOf(settings);
        this.loadSeconds = loadSeconds;
        // the slot choice, the mixed scheduler's queue and the paged housekeeping are the engine's (SlotPool, also used by
        // ./jllm serve); a conversation saved to disk that covers more of a prompt is loaded into the chosen slot
        this.pool = new SlotPool(engines, mixedConfig(this.settings), store == null ? null : (slot, prompt) -> store.restore(this, slot, prompt), id);
    }

    /** The mixed scheduler's queue settings among the load-time settings. */
    private static Admission.Config mixedConfig(Map<String, String> st) {
        return new Admission.Config(Integer.parseInt(st.getOrDefault("engine.max-running", "0")),
                Boolean.parseBoolean(st.getOrDefault("engine.preempt", "true")),
                Integer.parseInt(st.getOrDefault("engine.session-max-running", "0")),
                Long.parseLong(st.getOrDefault("engine.age-promote-ms", "10000")) * 1_000_000L);
    }

    /** How long a request waits for a generation it pauses to reach its next checkpoint. */
    static final long PAUSE_WAIT_MS = SlotPool.PAUSE_WAIT_MS;

    /** The queue of the mixed scheduler, or null. */
    public Admission<Slot> admission() {
        return pool.admission();
    }

    public String id() {
        return id;
    }

    public String source() {
        return source;
    }

    /** The first slot's engine: tokenizer, template and context are the same for all slots. */
    public Engine engine() {
        return loaded.engine();
    }

    /** The load-time settings (those of {@link View#settings}). */
    public Map<String, String> settings() {
        return settings;
    }

    List<Slot> slots() {
        return pool.slots();
    }

    /** The slot to run {@code prompt} (prepared ids) on: {@link SlotPool#pick(int[])}. */
    public Slot pick(int[] prompt) {
        return pick(prompt, null, () -> false);
    }

    /**
     * {@link SlotPool#pick(int[], String, java.util.function.BooleanSupplier)} for the request of {@code session}; a request
     * still waiting in the mixed scheduler's queue when unloading starts leaves it.
     */
    public Slot pick(int[] prompt, String session, java.util.function.BooleanSupplier gone) {
        return pool.pick(prompt, session, () -> closing || gone.getAsBoolean());
    }

    /** Copying a prefix from another slot must save at least this many tokens of prefill. */
    static final int MIN_COPY = SlotPool.MIN_COPY;

    /** Idle slots of a paged model give their blocks back: {@link SlotPool#enablePagedHousekeeping}. */
    public void enablePagedHousekeeping(long idleSeconds) {
        pool.enablePagedHousekeeping(idleSeconds);
    }

    /** {@link SlotPool#releaseIdle}. */
    boolean releaseIdle(long minIdleMs) {
        return pool.releaseIdle(minIdleMs);
    }

    public int slotCount() {
        return pool.size();
    }

    /** Slots a request is running on or has reserved right now. */
    public int busySlots() {
        return pool.busy();
    }

    /** True once unloading started: running generations should stop. */
    public boolean closing() {
        return closing;
    }

    /** Marks the start of a request on this model. */
    public Lease lease() {
        if (closing) {
            throw new UnavailableException("model " + id + " is being unloaded");
        }
        guard.readLock().lock();
        if (closing) {
            guard.readLock().unlock();
            throw new UnavailableException("model " + id + " is being unloaded");
        }
        requests.incrementAndGet();
        active.incrementAndGet();
        return () -> {
            active.decrementAndGet();
            guard.readLock().unlock();
        };
    }

    /** Records a finished request on {@code slot}; an expensive prefill gets its cache saved to disk. */
    public void record(Slot slot, Engine.Stats stats) {
        record(stats);
        if (store != null) {
            store.afterRequest(this, slot, stats);
        }
    }

    public void record(Engine.Stats stats) {
        generated.addAndGet(stats.generatedTokens());
        if (stats.generatedTokens() > 1) {
            lastTokPerSec = stats.tokPerSec();
        }
    }

    /** Stops new requests, waits for the running ones (which see {@link #closing()}) and frees the model. */
    boolean close(long timeoutMs) throws InterruptedException {
        closing = true;
        pool.stop();
        if (!guard.writeLock().tryLock(timeoutMs, TimeUnit.MILLISECONDS)) {
            return false;
        }
        try {
            if (store != null) {
                try {
                    store.saveAll(this);
                } catch (RuntimeException e) {
                    System.err.println("kv-cache: saving on unload failed: " + e);
                }
            }
            for (int i = 1; i < pool.size(); i++) {
                pool.slots().get(i).engine().close();
            }
            loaded.close();
        } finally {
            guard.writeLock().unlock();
        }
        return true;
    }

    public View view() {
        var w = loaded.weights();
        var e = loaded.engine();
        return new View(id, source, w.arch.toString(), e.backend().name(), e.backend().contextLength(),
                String.valueOf(w.quant), w.byteSize(), e.template().name(), loadedAt, loadSeconds, requests.get(),
                generated.get(), Math.round(lastTokPerSec * 10) / 10.0, active.get(), settings, pool.size(),
                pool.busy(), e.vision() != null,
                e.vision() == null ? null : e.vision().describe(), "text");
    }
}
