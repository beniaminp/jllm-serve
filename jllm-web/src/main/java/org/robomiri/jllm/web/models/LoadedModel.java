package org.robomiri.jllm.web.models;

import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.engine.Engine;

import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * A model in memory, ready to serve. Requests hold a {@link Lease} while they use the engine, so that unloading
 * waits for them (after asking them to stop) before the backend's buffers and the mapped weights are released.
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
     */
    public record View(String id, String source, String description, String backend, int contextLength,
                       String quant, long weightBytes, String template, long loadedAt, double loadSeconds,
                       long requests, long generatedTokens, double lastTokensPerSecond, int active,
                       Map<String, String> settings) {
    }

    private final String id;
    private final String source;
    private final ModelLoader.Loaded loaded;
    private final Map<String, String> settings;
    private final long loadedAt = System.currentTimeMillis();
    private final double loadSeconds;
    private final ReentrantReadWriteLock guard = new ReentrantReadWriteLock();
    private final AtomicLong requests = new AtomicLong();
    private final AtomicLong generated = new AtomicLong();
    private final AtomicInteger active = new AtomicInteger();
    private volatile double lastTokPerSec;
    private volatile boolean closing;

    LoadedModel(String id, String source, ModelLoader.Loaded loaded, Map<String, String> settings, double loadSeconds) {
        this.id = id;
        this.source = source;
        this.loaded = loaded;
        this.settings = Map.copyOf(settings);
        this.loadSeconds = loadSeconds;
    }

    public String id() {
        return id;
    }

    public String source() {
        return source;
    }

    public Engine engine() {
        return loaded.engine();
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
                generated.get(), Math.round(lastTokPerSec * 10) / 10.0, active.get(), settings);
    }
}
