package org.robomiri.jllm.web.video;

import org.robomiri.jllm.diffusion.VideoApi;
import org.robomiri.jllm.diffusion.VideoEncoder;
import org.robomiri.jllm.diffusion.VideoPipeline;
import org.robomiri.jllm.diffusion.VideoStore;
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
 * The video models in memory (AnimateDiff pipelines of the engine's {@code diffusion} package) and the clips generated recently.
 * They sit beside the text and image models of {@code ModelManager}, which loads them through its jobs; the Models page lists
 * all with their kind, and {@code /v1/videos/*} and the chat tool use them.
 */
@Service
public class VideoService implements DisposableBean {

    /** A loaded pipeline. Requests hold a {@link Lease} so that unloading waits for the running generation. */
    public static final class Loaded {
        private final String id;
        private final String source;
        private final String baseSource;
        private final VideoPipeline pipeline;
        private final Map<String, String> settings;
        private final long loadedAt = System.currentTimeMillis();
        private final double loadSeconds;
        private final ReentrantReadWriteLock guard = new ReentrantReadWriteLock();
        private final AtomicLong requests = new AtomicLong();
        private final AtomicLong videos = new AtomicLong();
        private final AtomicInteger active = new AtomicInteger();
        private volatile boolean closing;

        Loaded(String id, String source, String baseSource, VideoPipeline pipeline, Map<String, String> settings, double loadSeconds) {
            this.id = id;
            this.source = source;
            this.baseSource = baseSource;
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

        public VideoPipeline pipeline() {
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

        public void countVideos(int n) {
            videos.addAndGet(n);
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
            VideoPipeline p = pipeline;
            return new View(id, source, p.description() + " (base " + baseSource + ")", p.description().contains("Metal") ? "metal" : "cpu", 0,
                    p.description().contains("float16") ? "f16" : "f32", p.weightBytes(), "", loadedAt, loadSeconds, requests.get(),
                    videos.get(), 0.0, active.get(), settings, 1, active.get() > 0 ? 1 : 0, "video", p.info(), p.sizeMultiple(), false,
                    p.memoryBytes());
        }
    }

    public interface Lease extends AutoCloseable {
        @Override
        void close();
    }

    /**
     * The fields of the text and image models' view (so the Models page lists all in one table) plus the video specifics.
     * {@code contextLength}, {@code template} and the token counters are unused ({@code generatedTokens} counts clips).
     */
    public record View(String id, String source, String description, String backend, int contextLength, String quant,
                       long weightBytes, String template, long loadedAt, double loadSeconds, long requests, long generatedTokens,
                       double lastTokensPerSecond, int active, Map<String, String> settings, int slots, int busy, String kind,
                       Map<String, Object> defaults, int sizeMultiple, boolean inpaintUnet, long memoryBytes) {
    }

    private final Map<String, Loaded> loaded = new LinkedHashMap<>();
    private final VideoStore store = new VideoStore(32, 512L << 20);
    private final Map<String, java.util.Optional<String>> ffmpegs = new java.util.concurrent.ConcurrentHashMap<>();

    public VideoStore store() {
        return store;
    }

    /** The ffmpeg and format settings as the API wants them; each ffmpeg setting is resolved once (running it to see that it works). */
    public VideoApi.Config config(String ffmpegSetting, String formatSetting) {
        String exe = ffmpegs.computeIfAbsent(ffmpegSetting, k -> java.util.Optional.ofNullable(VideoEncoder.resolveFfmpeg(k))).orElse(null);
        return new VideoApi.Config(exe, formatSetting);
    }

    public synchronized List<Loaded> loaded() {
        return new ArrayList<>(loaded.values());
    }

    public synchronized boolean has(String id) {
        return loaded.containsKey(id);
    }

    public synchronized void register(String id, String source, String baseSource, VideoPipeline pipeline, Map<String, String> settings,
                                      double loadSeconds) {
        loaded.put(id, new Loaded(id, source, baseSource, pipeline, settings, loadSeconds));
    }

    /**
     * The model a request names: by id; when the id is unknown or missing and exactly one video model is loaded, that one.
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
            throw new NoSuchElementException("no video model is loaded: load one in the web console (Models tab), e.g. guoyww/animatediff-motion-adapter-v1-5-2");
        }
        throw new NoSuchElementException("video model '" + id + "' is not loaded; loaded: " + String.join(", ", loaded.keySet()));
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
