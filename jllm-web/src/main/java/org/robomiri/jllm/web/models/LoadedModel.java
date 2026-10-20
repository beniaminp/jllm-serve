package org.robomiri.jllm.web.models;

import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.engine.Engine;

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
                       Map<String, String> settings, int slots, int busy) {
    }

    /** One engine (KV cache) of the model, reserved for a request until closed. */
    public static final class Slot implements AutoCloseable {
        private final Engine engine;
        private final int index;
        private final AtomicInteger users = new AtomicInteger();

        Slot(Engine engine, int index) {
            this.engine = engine;
            this.index = index;
        }

        public Engine engine() {
            return engine;
        }

        public int index() {
            return index;
        }

        @Override
        public void close() {
            users.decrementAndGet();
        }
    }

    private final String id;
    private final String source;
    private final ModelLoader.Loaded loaded;
    private final List<Slot> slots = new ArrayList<>();
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
                double loadSeconds) {
        this.id = id;
        this.source = source;
        this.loaded = loaded;
        slots.add(new Slot(loaded.engine(), 0));
        for (Engine e : extra) {
            slots.add(new Slot(e, slots.size()));
        }
        this.settings = Map.copyOf(settings);
        this.loadSeconds = loadSeconds;
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

    /**
     * Reserves the slot to run {@code prompt} (prepared ids) on: an idle slot with the longest cached prefix, unless
     * a busy slot holds a much longer one (waiting for it beats prefilling a long conversation again).
     */
    public synchronized Slot pick(int[] prompt) {
        Slot bestIdle = null;
        Slot bestAny = null;
        int idleLen = -1;
        int anyLen = -1;
        for (Slot s : slots) {
            int len = s.engine.cachedPrefix(prompt);
            if (s.users.get() == 0 && len > idleLen) {
                bestIdle = s;
                idleLen = len;
            }
            if (len > anyLen || (len == anyLen && bestAny != null && s.users.get() < bestAny.users.get())) {
                bestAny = s;
                anyLen = len;
            }
        }
        Slot s = bestIdle != null && (anyLen - idleLen < 1024) ? bestIdle : bestAny;
        s.users.incrementAndGet();
        return s;
    }

    public int slotCount() {
        return slots.size();
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

    public void record(Engine.Stats stats) {
        generated.addAndGet(stats.generatedTokens());
        if (stats.generatedTokens() > 1) {
            lastTokPerSec = stats.tokPerSec();
        }
    }

    /** Stops new requests, waits for the running ones (which see {@link #closing()}) and frees the model. */
    boolean close(long timeoutMs) throws InterruptedException {
        closing = true;
        if (!guard.writeLock().tryLock(timeoutMs, TimeUnit.MILLISECONDS)) {
            return false;
        }
        try {
            for (int i = 1; i < slots.size(); i++) {
                slots.get(i).engine.close();
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
                generated.get(), Math.round(lastTokPerSec * 10) / 10.0, active.get(), settings, slots.size(),
                (int) slots.stream().filter(s -> s.users.get() > 0).count());
    }
}
