package org.robomiri.jllm.web.images;

import org.robomiri.jllm.diffusion.ImagePipeline;
import org.robomiri.jllm.diffusion.ImageStore;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * The image models in memory (Stable Diffusion family pipelines of the engine's {@code diffusion} package) and the pictures
 * generated recently. They sit beside the text models of {@code ModelManager}, which loads them through its jobs; the
 * Models page lists both with their kind, and {@code /v1/images/*} and the chat tools use them.
 */
@Service
public class ImageService implements DisposableBean {

    /** A loaded pipeline. Requests hold a {@link Lease} so that unloading waits for the running generation. */
    public static final class Loaded {
        private final String id;
        private final String source;
        private final ImagePipeline pipeline;
        private final Map<String, String> settings;
        private final long loadedAt = System.currentTimeMillis();
        private final double loadSeconds;
        private final ReentrantReadWriteLock guard = new ReentrantReadWriteLock();
        private final AtomicLong requests = new AtomicLong();
        private final AtomicLong images = new AtomicLong();
        private final AtomicInteger active = new AtomicInteger();
        private volatile boolean closing;

        Loaded(String id, String source, ImagePipeline pipeline, Map<String, String> settings, double loadSeconds) {
            this.id = id;
            this.source = source;
            this.pipeline = pipeline;
            this.settings = Map.copyOf(settings);
            this.loadSeconds = loadSeconds;
        }

        public String id() {
            return id;
        }

        public String source() {
            return source;
        }

        public ImagePipeline pipeline() {
            return pipeline;
        }

        public boolean closing() {
            return closing;
        }

        /** Marks the start of a request; throws when the model is being unloaded. */
        public Lease lease() {
            if (closing) {
                throw new IllegalStateException("model " + id + " is being unloaded");
            }
            guard.readLock().lock();
            if (closing) {
                guard.readLock().unlock();
                throw new IllegalStateException("model " + id + " is being unloaded");
            }
            requests.incrementAndGet();
            active.incrementAndGet();
            return () -> {
                active.decrementAndGet();
                guard.readLock().unlock();
            };
        }

        public void countImages(int n) {
            images.addAndGet(n);
        }

        boolean close(long timeoutMs) throws InterruptedException {
            closing = true;
            if (!guard.writeLock().tryLock(timeoutMs, TimeUnit.MILLISECONDS)) {
                return false;
            }
            try {
                pipeline.close();
            } finally {
                guard.writeLock().unlock();
            }
            return true;
        }

        public View view() {
            ImagePipeline p = pipeline;
            return new View(id, source, p.description(), p.description().contains("Metal") ? "metal" : "cpu", 0,
                    p.description().contains("float16") ? "f16" : "f32", p.weightBytes(), "", loadedAt, loadSeconds, requests.get(),
                    images.get(), 0.0, active.get(), settings, 1, active.get() > 0 ? 1 : 0, "image",
                    Map.of("steps", p.defaults.steps(), "guidance", p.defaults.guidance(), "size", p.defaults.size(),
                            "scheduler", p.defaults.scheduler(), "distilled", p.defaults.distilled()),
                    p.sizeMultiple(), p.hasInpaintUnet(), p.memoryBytes());
        }
    }

    public interface Lease extends AutoCloseable {
        @Override
        void close();
    }

    /**
     * The fields of the text models' view (so the Models page lists both in one table) plus the image specifics. {@code contextLength},
     * {@code template} and the token counters are unused for images ({@code generatedTokens} counts images).
     */
    public record View(String id, String source, String description, String backend, int contextLength, String quant,
                       long weightBytes, String template, long loadedAt, double loadSeconds, long requests, long generatedTokens,
                       double lastTokensPerSecond, int active, Map<String, String> settings, int slots, int busy, String kind,
                       Map<String, Object> defaults, int sizeMultiple, boolean inpaintUnet, long memoryBytes) {
    }

    private final Map<String, Loaded> loaded = new LinkedHashMap<>();
    private final ImageStore store = new ImageStore(128, 512L << 20);

    public ImageStore store() {
        return store;
    }

    public synchronized List<Loaded> loaded() {
        return new ArrayList<>(loaded.values());
    }

    public synchronized boolean has(String id) {
        return loaded.containsKey(id);
    }

    public synchronized void register(String id, String source, ImagePipeline pipeline, Map<String, String> settings, double loadSeconds) {
        loaded.put(id, new Loaded(id, source, pipeline, settings, loadSeconds));
    }

    /**
     * The model a request names: by id; when the id is unknown or missing and exactly one image model is loaded, that one.
     *
     * @throws NoSuchElementException when nothing fits
     */
    public synchronized Loaded resolve(String id) {
        if (id != null && loaded.containsKey(id)) {
            return loaded.get(id);
        }
        if (loaded.size() == 1) {
            return loaded.values().iterator().next();
        }
        if (loaded.isEmpty()) {
            throw new NoSuchElementException("no image model is loaded: load one in the web console (Models tab), e.g. stabilityai/sd-turbo");
        }
        throw new NoSuchElementException("image model '" + id + "' is not loaded; loaded: " + String.join(", ", loaded.keySet()));
    }

    public void unload(String id) throws InterruptedException {
        Loaded m;
        synchronized (this) {
            m = loaded.get(id);
            if (m == null) {
                throw new NoSuchElementException("model '" + id + "' is not loaded");
            }
        }
        if (!m.close(120_000)) {
            throw new IllegalStateException("model " + id + " is still busy after 120 s; try again");
        }
        synchronized (this) {
            loaded.remove(id);
        }
        System.err.println("Unloaded " + id);
    }

    @Override
    public void destroy() {
        for (Loaded m : loaded()) {
            try {
                m.close(5_000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }
}
