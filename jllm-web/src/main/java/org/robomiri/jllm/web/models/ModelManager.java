package org.robomiri.jllm.web.models;

import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.QuantFile;
import org.robomiri.jllm.diffusion.DiffusersRepo;
import org.robomiri.jllm.diffusion.ImageModels;
import org.robomiri.jllm.diffusion.ImagePipeline;
import org.robomiri.jllm.diffusion.VideoModels;
import org.robomiri.jllm.diffusion.VideoPipeline;
import org.robomiri.jllm.web.WebProperties;
import org.robomiri.jllm.web.images.ImageService;
import org.robomiri.jllm.web.settings.SettingsService;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Stream;

/**
 * Downloads models from the Hugging Face Hub, loads them into memory with per-model settings and unloads them.
 * Downloads run two at a time; loads one at a time (each can take most of the machine's memory and cores).
 */
@Service
public class ModelManager implements DisposableBean {

    /**
     * @param source   a Hugging Face link, {@code owner/name}, {@code owner/name:QUANT} or a local {@code .gguf} path
     * @param id       the name clients use in the {@code model} field; empty = derived from the source
     * @param settings engine settings for this model only, e.g. {@code backend.ctx}, {@code backend}, {@code model.quant}
     * @param slots    KV cache slots (1 to 8, default 1): each keeps one conversation's prefix cached; see {@link LoadedModel}
     */
    public record LoadRequest(String source, String id, String revision, String ggufFile, String ggufQuant,
                              Map<String, String> settings, Integer slots) {
        public LoadRequest(String source, String id, String revision, String ggufFile, String ggufQuant,
                           Map<String, String> settings) {
            this(source, id, revision, ggufFile, ggufQuant, settings, null);
        }
    }

    public record DownloadRequest(String source, String revision, String ggufFile, String ggufQuant,
                                  Map<String, String> settings) {
    }

    /**
     * A model on disk.
     *
     * @param source    what to put into a {@link LoadRequest} to load it
     * @param ggufFile  the GGUF file within the repository directory, or null for a safetensors checkpoint
     * @param caches    the quantized caches converted from a safetensors checkpoint
     * @param loadedAs  ids of the loaded models that use it
     * @param modelDir  the models folder it was found in: loading it passes this as {@code model.dir}, so the model
     *                  is read where it is instead of being downloaded into the default folder
     */
    public record LocalModel(String name, String format, String source, String revision, String ggufFile,
                             String path, long bytes, List<String> caches, List<String> loadedAs, String modelDir,
                             List<String> mmprojs) {
    }

    /** What a link resolves to on the Hub: the repository's files and, for a GGUF repository, the model files. */
    public record RepoInfo(String repo, String revision, String file, String quant, String defaultId, boolean gguf,
                           List<HuggingFaceHub.RepoFile> files, List<HuggingFaceHub.RepoFile> ggufFiles,
                           boolean diffusers, List<HuggingFaceHub.RepoFile> imageFiles, boolean video) {
    }

    /** Where a request points, after the link and the explicit fields are combined. */
    private record Target(String model, String revision, String file, String quant, boolean local, String id) {
    }

    private static final String MARKER = ".jllm-web.json";

    private final SettingsService settings;
    private final WebProperties props;
    private final KvStore kvStore;
    private final ImageService images;
    private final Map<String, LoadedModel> loaded = new LinkedHashMap<>();
    private final Map<String, Job> jobs = new LinkedHashMap<>();
    private final ExecutorService downloads = Executors.newFixedThreadPool(2, Thread.ofPlatform().name("jllm-download-", 1).daemon().factory());
    private final ExecutorService loads = Executors.newSingleThreadExecutor(Thread.ofPlatform().name("jllm-load").daemon().factory());
    private final AtomicLong jobCounter = new AtomicLong();

    public ModelManager(SettingsService settings, WebProperties props, KvStore kvStore, ImageService images) {
        this.settings = settings;
        this.props = props;
        this.kvStore = kvStore;
        this.images = images;
    }

    // --- video (an AnimateDiff motion adapter is held by VideoService; it is used with a Stable Diffusion 1.x base model)
    @org.springframework.beans.factory.annotation.Autowired(required = false)
    private org.robomiri.jllm.web.video.VideoService video;

    // --- audio (speech-to-text / text-to-speech models are held by AudioService; a downloaded Whisper or OuteTTS
    // checkpoint is recognised after the download and loaded there instead of as a text model)
    @org.springframework.beans.factory.annotation.Autowired(required = false)
    private org.robomiri.jllm.web.audio.AudioService audio;

    @EventListener(ApplicationReadyEvent.class)
    public void autoload() {
        for (String spec : props.autoload()) {
            if (!spec.isBlank()) {
                load(new LoadRequest(spec.trim(), null, null, null, null, Map.of()));
            }
        }
    }

    // ------------------------------------------------------------------ loaded models

    public synchronized List<LoadedModel> loaded() {
        return new ArrayList<>(loaded.values());
    }

    /**
     * The model a request names: by id; when the id is unknown or missing and exactly one model is loaded, that one
     * (clients send whatever model name they were configured with).
     *
     * @throws NoSuchElementException when nothing fits
     */
    public synchronized LoadedModel resolve(String id) {
        if (id != null) {
            LoadedModel m = loaded.get(id);
            if (m != null) {
                return m;
            }
        }
        if (loaded.size() == 1) {
            return loaded.values().iterator().next();
        }
        if (loaded.isEmpty()) {
            throw new NoSuchElementException("no model is loaded: load one in the web console (Models tab) or with POST /api/models/load");
        }
        throw new NoSuchElementException("model '" + id + "' is not loaded; loaded: " + String.join(", ", loaded.keySet()));
    }

    /** Tests: serves {@code engine} (typically on a fake backend) as model {@code id}, without weights or saved caches. */
    synchronized void registerForTest(String id, org.robomiri.jllm.engine.Engine engine) {
        loaded.put(id, new LoadedModel(id, "test", new ModelLoader.Loaded(null, engine), List.of(), Map.of(), 0, null));
    }

    /** Tests: removes a model {@link #registerForTest} added (its engine is closed by the caller). */
    synchronized void removeForTest(String id) {
        loaded.remove(id);
    }

    /** Unloads a model: running requests on it are cancelled first. */
    public void unload(String id) throws InterruptedException {
        if (images.has(id)) {
            images.unload(id);
            return;
        }
        if (video != null && video.has(id)) {
            video.unload(id);
            return;
        }
        if (audio != null && audio.has(id)) {
            audio.unload(id);
            System.err.println("Unloaded " + id);
            return;
        }
        LoadedModel m;
        synchronized (this) {
            m = loaded.get(id);
            if (m == null) {
                throw new NoSuchElementException("model '" + id + "' is not loaded");
            }
        }
        if (!m.close(60_000)) {
            throw new IllegalStateException("model " + id + " is still busy after 60 s; try again");
        }
        synchronized (this) {
            loaded.remove(id);
        }
        System.err.println("Unloaded " + id);
    }

    // ------------------------------------------------------------------ jobs

    public synchronized List<Job.View> jobs() {
        List<Job.View> out = new ArrayList<>();
        for (Job j : jobs.values()) {
            out.add(j.view());
        }
        return out.reversed();
    }

    public synchronized void cancel(String jobId) {
        Job j = jobs.get(jobId);
        if (j == null) {
            throw new NoSuchElementException("no job " + jobId);
        }
        if (!j.finished() && j.future != null) {
            j.future.cancel(true);
            if (j.state == Job.State.QUEUED) {
                j.finish(Job.State.CANCELLED, null);
            } else {
                j.phase = "cancelling";
            }
        }
    }

    public synchronized void clearFinishedJobs() {
        jobs.values().removeIf(Job::finished);
    }

    /** Starts downloading a model without loading it. */
    public Job.View download(DownloadRequest r) {
        Target t = target(r.source(), r.revision(), r.ggufFile(), r.ggufQuant(), null);
        if (t.local()) {
            throw new IllegalArgumentException("a local file needs no download: load it");
        }
        Map<String, String> extra = settingsFor(t, r.settings());
        locate(t, extra);
        Settings s = settings.with(extra); // validates before the job is queued
        if (isVideoModel(t, s)) {
            Job videoJob = newJob(Job.Kind.DOWNLOAD, t.id(), r.source());
            submit(downloads, videoJob, () -> {
                videoJob.phase = "downloading";
                downloadVideoModel(videoJob, s, t);
            });
            return videoJob.view();
        }
        if (isImageModel(t, s)) {
            Job imageJob = newJob(Job.Kind.DOWNLOAD, t.id(), r.source());
            submit(downloads, imageJob, () -> {
                imageJob.phase = "downloading";
                downloadImageModel(imageJob, s, t);
            });
            return imageJob.view();
        }
        Job job = newJob(Job.Kind.DOWNLOAD, t.id(), r.source());
        submit(downloads, job, () -> {
            HuggingFaceHub hub = ModelLoader.hub(s);
            planProgress(job, s, hub, t);
            job.phase = "downloading";
            ModelLoader.download(s, hub, t.model());
            writeMarker(s, t);
        });
        return job.view();
    }

    /** Starts loading a model (downloading what is missing first). */
    public Job.View load(LoadRequest r) {
        Target t = target(r.source(), r.revision(), r.ggufFile(), r.ggufQuant(), r.id());
        Map<String, String> extra = settingsFor(t, r.settings());
        locate(t, extra);
        Settings s = settings.with(extra);
        if (isVideoModel(t, s)) {
            return loadVideoModel(t, s);
        }
        if (isImageModel(t, s)) {
            return loadImageModel(t, s, extra);
        }
        int slots = r.slots() == null ? 1 : r.slots();
        if (slots < 1 || slots > 8) {
            throw new IllegalArgumentException("slots must be 1 to 8");
        }
        synchronized (this) {
            if (loaded.containsKey(t.id()) || images.has(t.id()) || (video != null && video.has(t.id()))) {
                throw new IllegalStateException("a model named '" + t.id() + "' is already loaded: unload it first or choose another id");
            }
            for (Job j : jobs.values()) {
                if (j.kind == Job.Kind.LOAD && !j.finished() && j.model.equals(t.id())) {
                    throw new IllegalStateException("'" + t.id() + "' is already being loaded (job " + j.id + ")");
                }
            }
        }
        Job job = newJob(Job.Kind.LOAD, t.id(), r.source());
        submit(loads, job, () -> {
            HuggingFaceHub hub = ModelLoader.hub(s);
            long t0 = System.nanoTime();
            // low-disk downloads shard by shard while converting; a checkpoint whose quantized cache is here needs
            // nothing from the Hub (its shards may have been deleted after the conversion)
            if (!t.local() && !s.getBool("model.low-disk") && !cacheReady(s, t)) {
                planProgress(job, s, hub, t);
                job.phase = "downloading";
                ModelLoader.download(s, hub, t.model());
                writeMarker(s, t);
            }
            checkInterrupted();
            job.phase = "loading";
            if (audio != null && !t.local() && audio.isAudio(repoDir(s, t.model()))) { // audio model: AudioService holds it
                System.err.println("Loading " + t.id() + " (" + t.model() + ") as an audio model");
                org.robomiri.jllm.audio.AudioModels.Entry a = audio.load(s, t.id(), t.model(), repoDir(s, t.model()));
                writeMarker(s, t);
                System.err.println("Loaded " + t.id() + ": " + a.kind().label() + ", " + a.backend());
                return;
            }
            System.err.println("Loading " + t.id() + " (" + t.model() + ")");
            ModelLoader.Loaded l = ModelLoader.load(s, hub, t.model(), slots); // budgets for every slot's cache
            if (Thread.currentThread().isInterrupted()) {
                l.close();
                throw new CancellationException();
            }
            if (!t.local()) {
                writeMarker(s, t);
            }
            List<org.robomiri.jllm.engine.Engine> extraSlots = new ArrayList<>();
            try {
                for (int i = 1; i < slots; i++) {
                    job.phase = "creating slot " + (i + 1) + " of " + slots;
                    extraSlots.add(ModelLoader.extraEngine(s, l));
                }
            } catch (Throwable e) {
                extraSlots.forEach(org.robomiri.jllm.engine.Engine::close);
                l.close();
                throw e;
            }
            Map<String, String> snapshot = new LinkedHashMap<>();
            for (String k : s.keys()) {
                if (SettingsService.scope(k) == SettingsService.Scope.LOAD && !s.get(k).isEmpty()) {
                    snapshot.put(k, s.get(k));
                }
            }
            snapshot.put("slots", String.valueOf(slots));
            LoadedModel m = new LoadedModel(t.id(), t.model(), l, extraSlots, snapshot, (System.nanoTime() - t0) / 1e9,
                    kvStore);
            synchronized (this) {
                loaded.put(t.id(), m);
            }
            System.err.println("Loaded " + t.id() + ": " + l.engine().backend().name() + ", ctx="
                    + l.engine().backend().contextLength() + (slots > 1 ? ", " + slots + " slots" : ""));
        });
        return job.view();
    }

    // ------------------------------------------------------------------ video models (AnimateDiff motion adapters on a Stable Diffusion 1.x base)

    /** A motion adapter: a local folder whose config.json is a MotionAdapter's, or a Hub repository with such a config.json. */
    private boolean isVideoModel(Target t, Settings s) {
        if (t.local() || video == null) {
            return false;
        }
        Path dir = repoDir(s, t.model());
        if (VideoModels.isAdapterDir(dir)) {
            return true;
        }
        if (Files.isDirectory(dir)) {
            return false; // a model already on disk that is not an adapter
        }
        try {
            HuggingFaceHub hub = ModelLoader.hub(s);
            List<HuggingFaceHub.RepoFile> listing = hub.listFiles(t.model(), s.get("model.revision"));
            if (listing.stream().noneMatch(f -> f.name().equals("config.json")) || listing.stream().anyMatch(f -> f.name().equals("model_index.json"))) {
                return false;
            }
            Path tmp = Files.createTempDirectory("jllm-adapter");
            try {
                hub.ensure(t.model(), s.get("model.revision"), tmp, "config.json");
                return VideoModels.isAdapterDir(tmp);
            } finally {
                deleteTree(tmp);
            }
        } catch (Exception e) {
            return false; // the Hub is unreachable: the normal path reports it
        }
    }

    private static void deleteTree(Path dir) {
        try (Stream<Path> st = Files.walk(dir)) {
            for (Path x : st.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(x);
            }
        } catch (IOException e) {
            // a temporary folder: left for the system to clean up
        }
    }

    /** Fetches the adapter (small) and then its base model (a diffusers repository), with a progress estimate for each. */
    private void downloadVideoModel(Job job, Settings s, Target t) throws Exception {
        HuggingFaceHub hub = ModelLoader.hub(s);
        Path dir = repoDir(s, t.model());
        boolean adapterReady = VideoModels.isAdapterDir(dir) && (Files.isRegularFile(dir.resolve("diffusion_pytorch_model.fp16.safetensors"))
                || Files.isRegularFile(dir.resolve("diffusion_pytorch_model.safetensors")));
        if (!adapterReady) {
            Map<String, Long> sizes = new LinkedHashMap<>();
            for (HuggingFaceHub.RepoFile f : hub.listFiles(t.model(), s.get("model.revision"))) {
                if (f.name().equals("config.json") || f.name().equals("diffusion_pytorch_model.fp16.safetensors")) {
                    sizes.put(f.name(), f.size());
                }
            }
            job.expect(dir, sizes);
            VideoModels.resolveAdapter(s, hub, t.model());
        }
        checkInterrupted();
        String base = s.get("video.base-model");
        Path baseDir = ImageModels.folder(s, base); // a Hub repository under model.dir, or a local diffusers folder
        if (!(DiffusersRepo.isDiffusersDir(baseDir) && DiffusersRepo.missingWeights(baseDir).isEmpty())) {
            List<HuggingFaceHub.RepoFile> listing = hub.listFiles(base, s.get("model.revision"));
            Files.createDirectories(baseDir);
            hub.ensure(base, s.get("model.revision"), baseDir, "model_index.json");
            Map<String, Object> index = Json.parseObject(Files.readString(baseDir.resolve("model_index.json")));
            Map<String, Long> sizes = new LinkedHashMap<>();
            DiffusersRepo.select(listing, index).forEach((n, f) -> sizes.put(n, f.size()));
            job.expect(baseDir, sizes);
            ImageModels.resolve(s, hub, base);
        }
        writeMarker(s, t);
    }

    private Job.View loadVideoModel(Target t, Settings s) {
        synchronized (this) {
            if (loaded.containsKey(t.id()) || images.has(t.id()) || video.has(t.id())) {
                throw new IllegalStateException("a model named '" + t.id() + "' is already loaded: unload it first or choose another id");
            }
            for (Job j : jobs.values()) {
                if (j.kind == Job.Kind.LOAD && !j.finished() && j.model.equals(t.id())) {
                    throw new IllegalStateException("'" + t.id() + "' is already being loaded (job " + j.id + ")");
                }
            }
        }
        if (!s.getBool("video.enabled")) {
            throw new IllegalStateException("video generation is switched off (video.enabled=false)");
        }
        Job job = newJob(Job.Kind.LOAD, t.id(), t.model());
        submit(loads, job, () -> {
            long t0 = System.nanoTime();
            job.phase = "downloading";
            downloadVideoModel(job, s, t);
            checkInterrupted();
            job.phase = "loading";
            System.err.println("Loading video model " + t.id() + " (" + t.model() + " on " + s.get("video.base-model") + ")");
            VideoPipeline p = VideoModels.load(s, ModelLoader.hub(s), t.model());
            if (Thread.currentThread().isInterrupted()) {
                p.close();
                throw new CancellationException();
            }
            Map<String, String> snapshot = new LinkedHashMap<>();
            for (String k : s.keys()) {
                if (k.startsWith("video.") && !s.get(k).isEmpty() && SettingsService.scope(k) == SettingsService.Scope.LOAD) {
                    snapshot.put(k, s.get(k));
                }
            }
            video.register(t.id(), t.model(), s.get("video.base-model"), p, snapshot, (System.nanoTime() - t0) / 1e9);
            System.err.println("Loaded video model " + t.id() + ": " + p.description());
        });
        return job.view();
    }

    // ------------------------------------------------------------------ image models (Stable Diffusion family, diffusers layout)

    /** A diffusers pipeline: a local folder with a model_index.json, or a Hub repository that has one. */
    private boolean isImageModel(Target t, Settings s) {
        if (t.local()) {
            return false;
        }
        if (DiffusersRepo.isDiffusersDir(repoDir(s, t.model()))) {
            return true;
        }
        if (Files.isDirectory(repoDir(s, t.model()))) {
            return false; // a text model already on disk
        }
        try {
            return DiffusersRepo.isDiffusers(ModelLoader.hub(s).listFiles(t.model(), s.get("model.revision")));
        } catch (Exception e) {
            return false; // the Hub is unreachable: the normal path reports it
        }
    }

    private void downloadImageModel(Job job, Settings s, Target t) throws Exception {
        HuggingFaceHub hub = ModelLoader.hub(s);
        Path dir = repoDir(s, t.model());
        if (DiffusersRepo.isDiffusersDir(dir) && DiffusersRepo.missingWeights(dir).isEmpty()) {
            return;
        }
        List<HuggingFaceHub.RepoFile> listing = hub.listFiles(t.model(), s.get("model.revision"));
        hub.ensure(t.model(), s.get("model.revision"), dir, "model_index.json");
        Map<String, Object> index = Json.parseObject(Files.readString(dir.resolve("model_index.json")));
        Map<String, Long> sizes = new LinkedHashMap<>();
        DiffusersRepo.select(listing, index).forEach((n, f) -> sizes.put(n, f.size()));
        job.expect(dir, sizes);
        DiffusersRepo.download(hub, t.model(), s.get("model.revision"), dir);
        writeMarker(s, t);
    }

    private Job.View loadImageModel(Target t, Settings s, Map<String, String> extra) {
        synchronized (this) {
            if (loaded.containsKey(t.id()) || images.has(t.id())) {
                throw new IllegalStateException("a model named '" + t.id() + "' is already loaded: unload it first or choose another id");
            }
            for (Job j : jobs.values()) {
                if (j.kind == Job.Kind.LOAD && !j.finished() && j.model.equals(t.id())) {
                    throw new IllegalStateException("'" + t.id() + "' is already being loaded (job " + j.id + ")");
                }
            }
        }
        Job job = newJob(Job.Kind.LOAD, t.id(), t.model());
        submit(loads, job, () -> {
            long t0 = System.nanoTime();
            Path dir = repoDir(s, t.model());
            if (!DiffusersRepo.isDiffusersDir(dir) || !DiffusersRepo.missingWeights(dir).isEmpty()) {
                job.phase = "downloading";
                downloadImageModel(job, s, t);
            }
            checkInterrupted();
            job.phase = "loading";
            System.err.println("Loading image model " + t.id() + " (" + t.model() + ")");
            ImagePipeline p = ImageModels.load(s, ModelLoader.hub(s), t.model());
            if (Thread.currentThread().isInterrupted()) {
                p.close();
                throw new CancellationException();
            }
            Map<String, String> snapshot = new LinkedHashMap<>();
            for (String k : s.keys()) {
                if (k.startsWith("image.") && !s.get(k).isEmpty() && SettingsService.scope(k) == SettingsService.Scope.LOAD) {
                    snapshot.put(k, s.get(k));
                }
            }
            images.register(t.id(), t.model(), p, snapshot, (System.nanoTime() - t0) / 1e9);
            System.err.println("Loaded image model " + t.id() + ": " + p.description());
        });
        return job.view();
    }

    private synchronized Job newJob(Job.Kind kind, String model, String source) {
        Job job = new Job("job-" + jobCounter.incrementAndGet(), kind, model, source);
        jobs.put(job.id(), job);
        while (jobs.size() > 100) { // forget the oldest finished ones
            String oldest = jobs.values().stream().filter(Job::finished).map(Job::id).findFirst().orElse(null);
            if (oldest == null) {
                break;
            }
            jobs.remove(oldest);
        }
        return job;
    }

    private interface Work {
        void run() throws Exception;
    }

    private void submit(ExecutorService pool, Job job, Work work) {
        job.future = pool.submit(() -> {
            if (job.finished()) { // cancelled while queued
                return;
            }
            job.state = Job.State.RUNNING;
            job.phase = "starting";
            try {
                work.run();
                job.finish(Job.State.DONE, null);
            } catch (CancellationException | InterruptedException e) {
                job.finish(Job.State.CANCELLED, null);
            } catch (OutOfMemoryError e) {
                long max = Runtime.getRuntime().maxMemory();
                String msg = String.format("out of memory (Java heap %.1f GB): the KV caches (context length x slots) do"
                        + " not fit. Lower the context length or the slots, use KV cache f16, or restart with a larger"
                        + " heap, e.g. JLLM_XMX=16g ./jllm-web/run.sh", max / 1e9);
                job.finish(Job.State.FAILED, msg);
                System.err.println(job.kind.name().toLowerCase(Locale.ROOT) + " of " + job.model + " failed: " + msg);
            } catch (Throwable e) {
                if (Thread.currentThread().isInterrupted()) {
                    job.finish(Job.State.CANCELLED, null);
                } else {
                    String msg = e.getMessage() != null ? e.getMessage() : e.toString();
                    job.finish(Job.State.FAILED, msg);
                    System.err.println(job.kind.name().toLowerCase(Locale.ROOT) + " of " + job.model + " failed: " + msg);
                    if (!(e instanceof IOException) && !(e instanceof IllegalArgumentException)) {
                        e.printStackTrace();
                    }
                }
            }
        });
    }

    private static void checkInterrupted() {
        if (Thread.currentThread().isInterrupted()) {
            throw new CancellationException();
        }
    }

    private Target target(String source, String revision, String file, String quant, String id) {
        HfLink link = HfLink.parse(source);
        String rev = blank(revision) ? link.revision() : revision.trim();
        String f = blank(file) ? link.file() : file.trim();
        String q = blank(quant) ? link.quant() : quant.trim();
        if (f != null) {
            q = null; // an exact file wins over a label
        }
        HfLink effective = new HfLink(link.repo(), rev, f, q, link.local());
        String name = blank(id) ? effective.defaultId() : id.trim();
        return new Target(link.repo(), rev, f, q, link.local(), name);
    }

    private static Map<String, String> settingsFor(Target t, Map<String, String> requested) {
        Map<String, String> extra = new LinkedHashMap<>();
        if (requested != null) {
            requested.forEach((k, v) -> {
                if (k.equals("model.repo") || k.equals("model.revision") || k.equals("model.gguf-file")
                        || k.equals("model.gguf-quant")) {
                    throw new IllegalArgumentException(k + " is given by the source, revision and GGUF fields");
                }
                if (v != null && !v.isBlank()) {
                    extra.put(k, v.trim());
                }
            });
        }
        extra.put("model.repo", t.model());
        extra.put("model.revision", t.revision() == null ? "main" : t.revision());
        extra.put("model.gguf-file", t.file() == null ? "" : t.file());
        extra.put("model.gguf-quant", t.quant() == null ? "" : t.quant());
        return extra;
    }

    private static boolean blank(String s) {
        return s == null || s.isBlank();
    }

    private static Path repoDir(Settings s, String repo) {
        return Path.of(s.get("model.dir")).resolve(repo.replace('/', '_'));
    }

    /**
     * A request that names no {@code model.dir} reads the model from the models folder that already holds it
     * ({@code model.dir} first, then {@code jllm.web.model-dirs}), so a model on disk is never downloaded again
     * because it sits in another folder than the default one. A folder where the model is usable (the quantized
     * cache, every weight file, or the GGUF file) wins over one with only its metadata (what an interrupted
     * download leaves); only when no folder has it does the download go to {@code model.dir}.
     */
    private void locate(Target t, Map<String, String> extra) {
        if (t.local() || extra.containsKey("model.dir")) {
            return;
        }
        Settings s = settings.with(extra);
        String name = t.model().replace('/', '_');
        Path partial = null;
        for (Path root : modelRoots()) {
            Path dir = root.resolve(name);
            try {
                boolean st = Files.isRegularFile(dir.resolve("config.json"));
                boolean diffusers = Files.isRegularFile(dir.resolve("model_index.json")) // an image model: usable as it is
                        || (VideoModels.isAdapterDir(dir) && (Files.isRegularFile(dir.resolve("diffusion_pytorch_model.fp16.safetensors"))
                        || Files.isRegularFile(dir.resolve("diffusion_pytorch_model.safetensors")))); // a video motion adapter
                List<String> gguf = HuggingFaceHub.localGgufNames(dir);
                if (!st && !diffusers && gguf.isEmpty()) {
                    continue;
                }
                boolean usable = diffusers || (t.file() != null ? Files.isRegularFile(dir.resolve(t.file()))
                        : !gguf.isEmpty() && (t.quant() != null || !st)
                        || st && (QuantFile.exists(dir, QType.parse(s.get("model.quant")), QType.parseHead(s.get("model.head-quant")))
                                  || allWeights(dir)));
                if (usable) {
                    extra.put("model.dir", root.toString());
                    return;
                }
                if (partial == null) {
                    partial = root;
                }
            } catch (IOException | RuntimeException e) {
                // unreadable: look further
            }
        }
        if (partial != null) {
            extra.put("model.dir", partial.toString()); // finish the download where it started
        }
    }

    /** Every weight file the checkpoint's index (or the single-file name) lists is on disk. */
    private static boolean allWeights(Path dir) throws IOException {
        List<String> names = HuggingFaceHub.weightFileNames(dir);
        if (names.isEmpty()) {
            return false;
        }
        for (String n : names) {
            if (!Files.isRegularFile(dir.resolve(n))) {
                return false;
            }
        }
        return true;
    }

    /** A safetensors checkpoint whose quantized cache for the requested format is already converted. */
    private static boolean cacheReady(Settings s, Target t) {
        if (t.file() != null || t.quant() != null) {
            return false;
        }
        Path dir = repoDir(s, t.model());
        try {
            return Files.isRegularFile(dir.resolve("config.json")) && Files.isRegularFile(dir.resolve("tokenizer.json"))
                    && QuantFile.exists(dir, QType.parse(s.get("model.quant")), QType.parseHead(s.get("model.head-quant")));
        } catch (RuntimeException e) {
            return false; // an unknown format: let the loader report it
        }
    }

    /**
     * The folders models are listed from: {@code model.dir} first, then {@code jllm.web.model-dirs}; only existing
     * ones, each once (by real path).
     */
    public List<Path> modelRoots() {
        List<String> names = new ArrayList<>();
        names.add(settings.current().get("model.dir"));
        names.addAll(props.modelDirs());
        LinkedHashSet<Path> seen = new LinkedHashSet<>();
        List<Path> out = new ArrayList<>();
        for (String n : names) {
            if (n == null || n.isBlank()) {
                continue;
            }
            Path p = Path.of(n.trim()).toAbsolutePath().normalize();
            if (!Files.isDirectory(p)) {
                continue;
            }
            try {
                if (seen.add(p.toRealPath())) {
                    out.add(p);
                }
            } catch (IOException e) {
                // unreadable: skip
            }
        }
        return out;
    }

    /** Works out which files the job will fetch, the same way {@link ModelLoader#resolve} decides, for the progress bar. */
    private static void planProgress(Job job, Settings s, HuggingFaceHub hub, Target t) {
        try {
            Path dir = repoDir(s, t.model());
            boolean forced = t.file() != null || t.quant() != null;
            if (!forced && Files.isRegularFile(dir.resolve("config.json"))) {
                // a checkpoint that is (partly) here: its shards are what the index lists
                Map<String, Long> sizes = new LinkedHashMap<>();
                Map<String, Long> byName = new LinkedHashMap<>();
                for (HuggingFaceHub.RepoFile f : hub.listFiles(t.model(), s.get("model.revision"))) {
                    byName.put(f.name(), f.size());
                }
                for (String n : HuggingFaceHub.weightFileNames(dir)) {
                    sizes.put(n, byName.getOrDefault(n, -1L));
                }
                job.expect(dir, sizes);
                return;
            }
            List<HuggingFaceHub.RepoFile> listing = hub.listFiles(t.model(), s.get("model.revision"));
            Map<String, Long> sizes = new LinkedHashMap<>();
            if (forced || HuggingFaceHub.isGgufRepo(listing)) {
                List<String> names = listing.stream().map(HuggingFaceHub.RepoFile::name).toList();
                var sel = HuggingFaceHub.selectGguf(names, t.file(), t.quant());
                for (HuggingFaceHub.RepoFile f : listing) {
                    if (sel.getValue().contains(f.name())) {
                        sizes.put(f.name(), f.size());
                    }
                }
            } else {
                for (HuggingFaceHub.RepoFile f : listing) {
                    if (HuggingFaceHub.isMetadataFile(f.name()) || f.name().endsWith(".safetensors")) {
                        sizes.put(f.name(), f.size());
                    }
                }
            }
            job.expect(dir, sizes);
        } catch (Exception e) {
            // no progress bar; the download itself reports the problem
        }
    }

    /** Remembers which repository and revision a model directory holds (its name loses the owner/name split). */
    private static void writeMarker(Settings s, Target t) {
        try {
            Path dir = repoDir(s, t.model());
            if (Files.isDirectory(dir)) {
                Files.writeString(dir.resolve(MARKER), Json.write(Json.obj("repo", t.model(), "revision",
                        t.revision() == null ? "main" : t.revision())));
            }
        } catch (IOException e) {
            System.err.println("note: cannot write " + MARKER + ": " + e.getMessage());
        }
    }

    // ------------------------------------------------------------------ hub and disk

    public RepoInfo inspect(String source, String revision) throws IOException, InterruptedException {
        Target t = target(source, revision, null, null, null);
        if (t.local()) {
            throw new IllegalArgumentException("a local file: nothing to look up on the Hub");
        }
        HfLink link = HfLink.parse(source);
        Settings s = settings.current();
        String rev = t.revision() == null ? "main" : t.revision();
        List<HuggingFaceHub.RepoFile> files = ModelLoader.hub(s).listFiles(t.model(), rev);
        boolean diffusers = DiffusersRepo.isDiffusers(files);
        List<HuggingFaceHub.RepoFile> imageFiles = List.of();
        boolean adapter = false;
        if (!diffusers && video != null && files.stream().anyMatch(f -> f.name().equals("config.json"))) { // an AnimateDiff motion adapter?
            try {
                Path tmp = Files.createTempDirectory("jllm-adapter");
                try {
                    ModelLoader.hub(s).ensure(t.model(), rev, tmp, "config.json");
                    adapter = VideoModels.isAdapterDir(tmp);
                } finally {
                    deleteTree(tmp);
                }
            } catch (IOException | RuntimeException e) {
                adapter = false;
            }
            if (adapter) {
                boolean half = files.stream().anyMatch(f -> f.name().equals("diffusion_pytorch_model.fp16.safetensors"));
                imageFiles = files.stream().filter(f -> f.name().equals("config.json")
                        || f.name().equals(half ? "diffusion_pytorch_model.fp16.safetensors" : "diffusion_pytorch_model.safetensors")).toList();
            }
        }
        if (diffusers) {
            try {
                Path tmp = Files.createTempDirectory("jllm-index");
                try {
                    ModelLoader.hub(s).ensure(t.model(), rev, tmp, "model_index.json");
                    imageFiles = new ArrayList<>(DiffusersRepo.select(files,
                            Json.parseObject(Files.readString(tmp.resolve("model_index.json")))).values());
                } finally {
                    try (Stream<Path> st = Files.walk(tmp)) {
                        for (Path x : st.sorted(Comparator.reverseOrder()).toList()) {
                            Files.deleteIfExists(x);
                        }
                    }
                }
            } catch (IOException | RuntimeException e) {
                imageFiles = List.of();
            }
        }
        boolean gguf = !diffusers && !adapter && (link.file() != null || link.quant() != null || HuggingFaceHub.isGgufRepo(files));
        List<HuggingFaceHub.RepoFile> ggufFiles = files.stream()
                .filter(f -> HuggingFaceHub.isGgufModelFile(f.name()))
                .filter(f -> !f.name().matches(".*-0000[2-9]-of-.*|.*-000[1-9][0-9]-of-.*")) // split parts after the first
                .toList();
        return new RepoInfo(t.model(), rev, link.file(), link.quant(), t.id(), gguf, files, ggufFiles, diffusers, imageFiles, adapter);
    }

    /** The models in {@code model.dir} and {@code jllm.web.model-dirs}: safetensors checkpoints and GGUF files. */
    public List<LocalModel> localModels() throws IOException {
        Map<String, List<String>> users = new LinkedHashMap<>();
        for (LoadedModel m : loaded()) {
            users.computeIfAbsent(m.source(), k -> new ArrayList<>()).add(m.id());
        }
        for (ImageService.Loaded m : images.loaded()) {
            users.computeIfAbsent(m.source(), k -> new ArrayList<>()).add(m.id());
        }
        if (video != null) {
            for (var m : video.loaded()) {
                users.computeIfAbsent(m.source(), k -> new ArrayList<>()).add(m.id());
            }
        }
        if (audio != null) {
            for (var a : audio.loaded()) {
                users.computeIfAbsent(a.source(), k -> new ArrayList<>()).add(a.id());
            }
        }
        List<LocalModel> out = new ArrayList<>();
        for (Path root : modelRoots()) {
            listModels(root, users, out);
        }
        return out;
    }

    private void listModels(Path root, Map<String, List<String>> users, List<LocalModel> out) throws IOException {
        String rootName = root.toString();
        List<Path> entries;
        try (Stream<Path> st = Files.list(root)) {
            entries = st.sorted().toList();
        }
        for (Path p : entries) {
            String name = p.getFileName().toString();
            if (Files.isRegularFile(p) && isFirstGguf(name)) {
                String abs = p.toString();
                out.add(new LocalModel(name, "gguf", abs, null, null, abs, Files.size(p), List.of(),
                        users.getOrDefault(abs, List.of()), rootName, List.of()));
                continue;
            }
            if (!Files.isDirectory(p) || name.equals("tokenizers") || name.startsWith(".")) {
                continue;
            }
            String[] repoRev = repoOf(p);
            String repo = repoRev[0];
            List<String> loadedAs = users.getOrDefault(repo, List.of());
            if (DiffusersRepo.isDiffusersDir(p)) {
                out.add(new LocalModel(repo, "diffusers", repo, repoRev[1], null, p.toString(), dirSize(p), List.of(), loadedAs, rootName, List.of()));
                continue;
            }
            if (VideoModels.isAdapterDir(p)) {
                out.add(new LocalModel(repo, "video-adapter", repo, repoRev[1], null, p.toString(), dirSize(p), List.of(), loadedAs, rootName, List.of()));
                continue;
            }
            if (Files.isRegularFile(p.resolve("config.json"))) {
                List<String> caches = new ArrayList<>();
                try (Stream<Path> st = Files.list(p)) {
                    st.map(x -> x.getFileName().toString()).filter(n -> n.startsWith("jllm-") && n.endsWith(".bin"))
                            .sorted().forEach(caches::add);
                }
                out.add(new LocalModel(repo, "safetensors", repo, repoRev[1], null, p.toString(), dirSize(p), caches, loadedAs,
                        rootName, List.of()));
            }
            List<String> projectors = new ArrayList<>(); // multimodal projectors in the folder: a vision model uses one
            try (Stream<Path> st = Files.list(p)) {
                st.map(x -> x.getFileName().toString()).filter(n -> org.robomiri.jllm.vision.Mmproj.isMmproj(n) && !n.endsWith(".incomplete"))
                        .sorted().forEach(projectors::add);
            }
            for (String g : HuggingFaceHub.localGgufNames(p)) {
                if (!isFirstGguf(Path.of(g).getFileName().toString())) {
                    continue;
                }
                long bytes = 0;
                for (Path part : ggufParts(p.resolve(g))) {
                    bytes += Files.size(part);
                }
                out.add(new LocalModel(repo + " / " + g, "gguf", repo, repoRev[1], g, p.resolve(g).toString(), bytes,
                        List.of(), loadedAs, rootName, projectors));
            }
        }
    }

    /**
     * Deletes a model from disk: a whole checkpoint directory (with its quantized caches), or one GGUF file (with its
     * split parts).
     */
    public void deleteLocal(String path) throws IOException {
        Path p = Path.of(path).toAbsolutePath().normalize();
        List<Path> roots = modelRoots();
        if (roots.stream().noneMatch(root -> p.startsWith(root) && !p.equals(root))) {
            throw new IllegalArgumentException("only models inside " + roots + " can be deleted");
        }
        for (LocalModel m : localModels()) {
            if (Path.of(m.path()).toAbsolutePath().normalize().equals(p) && !m.loadedAs().isEmpty()) {
                throw new IllegalStateException(m.name() + " is loaded as " + String.join(", ", m.loadedAs()) + ": unload it first");
            }
        }
        if (Files.isDirectory(p)) {
            try (Stream<Path> st = Files.walk(p)) {
                for (Path x : st.sorted(Comparator.reverseOrder()).toList()) {
                    Files.delete(x);
                }
            }
        } else if (Files.isRegularFile(p) && p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".gguf")) {
            for (Path part : ggufParts(p)) {
                Files.deleteIfExists(part);
            }
        } else {
            throw new NoSuchElementException("no model at " + p);
        }
        System.err.println("Deleted " + p);
    }

    private static boolean isFirstGguf(String name) {
        return HuggingFaceHub.isGgufModelFile(name) && !name.endsWith(".incomplete")
                && !name.matches("(?i).*-(\\d{5})-of-\\d{5}\\.gguf") || name.matches("(?i).*-00001-of-\\d{5}\\.gguf");
    }

    /** A GGUF file and, for {@code x-00001-of-0000N.gguf}, its other parts. */
    private static List<Path> ggufParts(Path first) {
        String n = first.getFileName().toString();
        var m = java.util.regex.Pattern.compile("(?i)(.*)-00001-of-(\\d{5})\\.gguf").matcher(n);
        if (!m.matches()) {
            return List.of(first);
        }
        int count = Integer.parseInt(m.group(2));
        List<Path> parts = new ArrayList<>();
        for (int i = 1; i <= count; i++) {
            Path part = first.resolveSibling(String.format("%s-%05d-of-%s.gguf", m.group(1), i, m.group(2)));
            if (Files.isRegularFile(part)) {
                parts.add(part);
            }
        }
        return parts;
    }

    /**
     * The repository and revision a model directory came from: the web console's marker, else the identity the
     * quantized cache recorded, else a guess from the directory name (the first '_' was the '/').
     */
    private static String[] repoOf(Path dir) {
        for (Path f : new Path[]{dir.resolve(MARKER)}) {
            String[] r = readRepo(f);
            if (r != null) {
                return r;
            }
        }
        try (Stream<Path> st = Files.list(dir)) {
            for (Path f : st.filter(x -> {
                String n = x.getFileName().toString();
                return n.startsWith("jllm-") && n.endsWith(".json") && !n.contains("partial");
            }).sorted().toList()) {
                String[] r = readRepo(f);
                if (r != null) {
                    return r;
                }
            }
        } catch (IOException ignored) {
            // fall through to the guess
        }
        String name = dir.getFileName().toString();
        int u = name.indexOf('_');
        return new String[]{u > 0 ? name.substring(0, u) + "/" + name.substring(u + 1) : name, null};
    }

    private static String[] readRepo(Path f) {
        if (!Files.isRegularFile(f)) {
            return null;
        }
        try {
            Map<String, Object> o = Json.parseObject(Files.readString(f));
            String repo = Json.str(o, "repo", null);
            if (repo == null || !repo.contains("/")) {
                return null;
            }
            return new String[]{repo, Json.str(o, "revision", null)};
        } catch (IOException | RuntimeException e) {
            return null;
        }
    }

    private static long dirSize(Path dir) throws IOException {
        try (Stream<Path> st = Files.walk(dir)) {
            return st.filter(Files::isRegularFile).mapToLong(p -> {
                try {
                    return Files.size(p);
                } catch (IOException e) {
                    return 0;
                }
            }).sum();
        }
    }

    @Override
    public void destroy() {
        downloads.shutdownNow();
        loads.shutdownNow();
        for (LoadedModel m : loaded()) {
            try {
                m.close(5_000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }
}
