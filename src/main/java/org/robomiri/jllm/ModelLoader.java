package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.cpu.MambaCpuBackend;
import org.robomiri.jllm.cpu.Qwen3NextCpuBackend;
import org.robomiri.jllm.engine.DraftModel;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.Speculator;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MambaMetalBackend;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.metal.MoeSplit;
import org.robomiri.jllm.metal.Qwen3NextMetalBackend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.MambaArch;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.Qwen3NextArch;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.tokenizer.GgufTokenizer;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Resolves, downloads and loads a model as the settings describe it: a Hugging Face safetensors checkpoint
 * (quantized once into a cache), a GGUF repository or a local {@code .gguf} file, on the configured backend, with
 * speculative decoding when enabled. Used by the command line ({@link Main}) and by embedders such as the web
 * module, which build a {@link Settings} per model they load.
 */
public final class ModelLoader {

    private ModelLoader() {
    }

    /** A Hub client configured from {@code hub.endpoint}, {@code hub.token} and {@code hub.timeout-ms}. */
    public static HuggingFaceHub hub(Settings s) {
        return new HuggingFaceHub(s.opt("hub.endpoint"), s.opt("hub.token"), s.getLong("hub.timeout-ms"));
    }

    /**
     * The loaded model. Close in this order: engine (the backend releases its buffers) and then weights (the
     * memory-mapped quantized cache): {@code try (ModelWeights w = l.weights(); Engine e = l.engine())}.
     */
    public record Loaded(ModelWeights weights, Engine engine) implements AutoCloseable {
        @Override
        public void close() {
            try {
                engine.close();
            } finally {
                try {
                    org.robomiri.jllm.kv.KvPools.release(weights); // the model's pool of KV blocks (its buffers) goes with it
                } finally {
                    weights.close();
                }
            }
        }
    }

    /** Where the model comes from: a GGUF file (used as it is), or a Hugging Face safetensors checkpoint in {@code dir}. */
    public record Source(Path gguf, Path dir) {
    }

    /**
     * Downloads {@code model} (the selection of {@code model.gguf-file} / {@code model.gguf-quant} for a GGUF
     * repository) without loading it.
     *
     * @return the GGUF file, or the directory of the safetensors checkpoint
     */
    public static Path download(Settings s, HuggingFaceHub hub, String model) throws Exception {
        Source src = resolve(s, hub, model, s.opt("model.gguf-file"), s.opt("model.gguf-quant"));
        if (src.gguf() == null) {
            return hub.download(model, s.get("model.revision"), src.dir());
        }
        System.err.println(src.gguf());
        try { // a vision-language repository: its multimodal projector comes with the model (vision.auto-download)
            org.robomiri.jllm.vision.Mmproj.locate(s, hub, model, src.gguf());
        } catch (IOException e) {
            System.err.println("note: " + e.getMessage());
        }
        return src.gguf();
    }

    /**
     * Decides how {@code model} is loaded, downloading what is missing. A name ending in {@code .gguf} is a local
     * file. A repository is a GGUF one when {@code model.gguf-file} / {@code model.gguf-quant} is set, or when it
     * has {@code .gguf} files and no safetensors checkpoint; a model directory that already has a
     * {@code config.json} is a safetensors one without asking the Hub. Only the chosen GGUF file (and its split
     * parts) is downloaded.
     */
    public static Source resolve(Settings s, HuggingFaceHub hub, String model, String file, String quant) throws Exception {
        if (model.toLowerCase(java.util.Locale.ROOT).endsWith(".gguf")) {
            Path local = Path.of(model).toAbsolutePath();
            if (!Files.isRegularFile(local)) {
                throw new HuggingFaceHub.GgufSelectionException("no such GGUF file: " + local);
            }
            return new Source(local, local.getParent());
        }
        Path dir = Path.of(s.get("model.dir")).resolve(model.replace('/', '_'));
        String revision = s.get("model.revision");
        boolean forced = file != null || quant != null;
        boolean safetensors = Files.isRegularFile(dir.resolve("config.json"));
        if (!forced && safetensors) {
            return new Source(null, dir);
        }
        IOException localProblem = null;
        List<String> local = HuggingFaceHub.localGgufNames(dir);
        if (!local.isEmpty()) {
            try {
                var sel = HuggingFaceHub.selectGguf(local, file, quant);
                if (sel.getValue().stream().allMatch(n -> Files.isRegularFile(dir.resolve(n)))) {
                    return new Source(dir.resolve(sel.getKey()), dir);
                }
            } catch (IOException e) {
                localProblem = e; // none or several fit: the repository's listing may settle it
            }
        }
        try {
            if (!forced && !HuggingFaceHub.isGgufRepo(hub.listFiles(model, revision))) {
                return new Source(null, dir);
            }
            return new Source(hub.downloadGguf(model, revision, dir, file, quant), dir);
        } catch (IOException e) {
            if (localProblem != null && !(e instanceof HuggingFaceHub.HttpStatusException)) {
                throw new IOException(localProblem.getMessage() + " (the Hub could not be asked: " + e.getMessage() + ")", e);
            }
            throw e;
        }
    }

    /** A model ready to run: architecture, tokenizer and weights, from a safetensors checkpoint or a GGUF file. */
    public record Prepared(Architecture arch, Tokenizer tok, ModelWeights weights, Path dir, Path gguf) {
        public Prepared(Architecture arch, Tokenizer tok, ModelWeights weights, Path dir) {
            this(arch, tok, weights, dir, null);
        }
    }

    public static Loaded load(Settings s, HuggingFaceHub hub, String model) throws Exception {
        return load(s, hub, model, 1);
    }

    /**
     * {@code slots}: how many engines will run on these weights ({@link #extraEngine} adds the others), so the Metal
     * backend's expert placement budgets GPU memory for all of their KV caches.
     */
    public static Loaded load(Settings s, HuggingFaceHub hub, String model, int slots) throws Exception {
        Prepared p = prepare(s, hub, model, s.opt("model.gguf-file"), s.opt("model.gguf-quant"));
        Architecture arch = p.arch();
        Tokenizer tok = p.tok();
        ModelWeights weights = p.weights();
        int ctx = s.getInt("backend.ctx");
        int batch = batch(s, arch);
        Speculator.Config spec = speculation(s, arch, weights);
        // the draft model first, on the same device: it joins DeviceLedger before the target's backend plans its GPU
        // memory (backend.metal.cpu-moe), so the target's budget leaves room for the draft's weights and caches
        DraftModel draft = spec.mode() == Speculator.Mode.DRAFT ? loadDraft(s, hub, tok, arch, weights) : null;
        Backend backend;
        try {
            backend = batched(s, weights, newBackend(s, arch, weights, slots, true));
        } catch (Exception e) {
            if (draft != null) {
                draft.close();
            }
            weights.close();
            throw e;
        }
        System.err.println("Backend: " + backend.name() + ", ctx=" + ctx + ", batch=" + batch + ", kv=" + KvDtype.parse(s.get("backend.kv-dtype"))
                + (KvDtype.parse(s.get("backend.kv-dtype")) == KvDtype.Q4 ? " recent=" + s.getInt("backend.kv-q4-recent") : "")
                + (KvDtype.parse(s.get("backend.kv-dtype")) == KvDtype.Q8 && s.getBool("backend.kv-q8-rotate") ? " rotated keys" : ""));
        try {
            Engine engine = new Engine(backend, tok, arch, eosIds(p.dir(), tok, arch), s.getLong("server.queue-timeout-ms"),
                    chatFormat(p, arch, tok));
            constraints(s, engine);
            long promptCacheMb = s.getLong("engine.prompt-cache-mb");
            if (promptCacheMb > 0) {
                engine.setPromptCache(new org.robomiri.jllm.tokenizer.PromptCache(tok, promptCacheMb << 20));
            }
            if (spec.mode() != Speculator.Mode.OFF) {
                DraftModel d = draft;
                draft = null; // the engine owns (and closes) it from here on
                if (engine.enableSpeculation(spec, d)) {
                    System.err.println("Speculative decoding: " + spec.mode().name().toLowerCase()
                            + (spec.mode() == Speculator.Mode.MTP ? ", the model's own multi-token-prediction head, one draft token per pass"
                            : ", up to " + spec.k() + " tokens per pass") + (d != null ? ", draft " + s.get("engine.draft-model") : ""));
                }
            }
            attachVision(s, hub, model, p, engine);
            return new Loaded(weights, engine);
        } catch (Exception e) { // e.g. a family without a chat template, or a bad generation_config.json
            if (draft != null) {
                draft.close();
            }
            backend.close();
            weights.close();
            throw e;
        }
    }

    /**
     * Gives {@code engine} eyes when its GGUF model has a multimodal projector next to it ({@code model.mmproj}, or an
     * {@code mmproj*.gguf} in the same folder / repository; docs/20). A projector that was named explicitly and does not
     * work is an error; one found automatically that does not fit the model is only reported.
     */
    private static void attachVision(Settings s, HuggingFaceHub hub, String model, Prepared p, Engine engine) throws Exception {
        if (p.gguf() == null || !s.getBool("vision.enabled")) {
            return;
        }
        Path mm = org.robomiri.jllm.vision.Mmproj.locate(s, hub, model, p.gguf());
        if (mm == null) {
            return;
        }
        try {
            org.robomiri.jllm.vision.VisionContext vc = org.robomiri.jllm.vision.VisionContext.open(mm, p.tok(), p.arch(),
                    visionOptions(s));
            engine.setVision(vc, true);
            System.err.println("Vision: " + vc.describe());
        } catch (IOException | RuntimeException e) {
            if (s.opt("model.mmproj") != null) {
                throw e;
            }
            System.err.println("warning: the projector " + mm.getFileName() + " next to the model cannot be used, loading it as"
                    + " a text model: " + e.getMessage());
        }
    }

    /** The vision encoder options of the settings. */
    public static org.robomiri.jllm.vision.VisionContext.Options visionOptions(Settings s) {
        String backend = s.get("vision.backend");
        if (backend.equals("auto")) {
            backend = s.get("backend");
        }
        return new org.robomiri.jllm.vision.VisionContext.Options(backend, s.getInt("backend.threads"),
                new org.robomiri.jllm.vision.Preprocess.Limits(s.getInt("vision.image-min-tokens"), s.getInt("vision.image-max-tokens")),
                s.getInt("vision.cache-mb"), s.opt("backend.metal.lib"));
    }

    /** Gemma 3: the tokens of one picture (docs/20.12); {@code backend.batch} at its default is raised to this. */
    static final int GEMMA3_PICTURE_BATCH = 256;

    /** Prefill chunk of {@code backend.batch=auto} for mixture-of-experts models. */
    static final int MOE_BATCH = 256;
    /** Prefill chunk of {@code backend.batch=auto} for dense decoders on Metal (measured, docs/11 §11.1). */
    static final int METAL_DENSE_BATCH = Integer.getInteger("jllm.metal.autoBatch", 256);
    /** Prefill chunk of {@code backend.batch=auto} everywhere else (the CPU backends, Mamba, dense Qwen3-Next). */
    static final int DEFAULT_BATCH = 64;

    /** True if {@code backend.batch} holds a number (not {@code auto}). */
    public static boolean explicitBatch(Settings s) {
        return !"auto".equalsIgnoreCase(s.get("backend.batch").trim());
    }

    /**
     * Tokens per prefill chunk: {@code backend.batch} when it is a number; with {@code auto} (the default) a
     * mixture-of-experts model takes {@link #MOE_BATCH}, a dense decoder on Metal {@link #METAL_DENSE_BATCH}, anything
     * else {@link #DEFAULT_BATCH}. A MoE chunk reads each chosen expert's weights once for all its tokens (the
     * grouped kernels on Metal, the dequantized row groups on the CPU), and with 128 experts and top-8 a 64-token
     * chunk gives an expert only ~4 tokens: 256 tokens measured 1.6x faster per token for the CPU experts (docs/11).
     * A dense chunk of 64 tokens leaves the GPU under-filled (a 1024-row projection is 16 x 2 MMA threadgroups).
     */
    static int batch(Settings s, Architecture arch) {
        if (explicitBatch(s)) {
            return s.getInt("backend.batch");
        }
        boolean moe = arch instanceof org.robomiri.jllm.model.Decoder d && d.anyMoe()
                || arch instanceof Qwen3NextArch q && q.anyMoe();
        int auto = moe ? MOE_BATCH
                : arch instanceof org.robomiri.jllm.model.Decoder && resolveBackend(s).equals("metal") ? METAL_DENSE_BATCH : DEFAULT_BATCH;
        if (arch.core().modelType().startsWith("gemma3")) {
            return Math.max(auto, GEMMA3_PICTURE_BATCH); // a picture's 256 tokens attend to each other and go through in one pass
        }
        return auto;
    }

    /** {@code backend}: cpu or metal, where auto means metal when the Metal library is available. */
    static String resolveBackend(Settings s) {
        String name = s.get("backend");
        return name.equals("auto") ? (Metal.available(s.opt("backend.metal.lib")) ? "metal" : "cpu") : name;
    }

    /**
     * The speculation settings for this model: prompt lookup is the default, except for mixture-of-experts models,
     * where a verification pass reads the experts of every drafted position and was measured not to pay
     * (docs/11 §11.6), unless {@code engine.speculative} was set explicitly.
     */
    static Speculator.Config speculation(Settings s, Architecture arch, ModelWeights weights) {
        Speculator.Config spec = speculation(s, arch);
        if (weights.mtp != null && "default".equals(s.source("engine.speculative"))) {
            // a model that carries its own multi-token-prediction head drafts with it unless told otherwise (a hybrid
            // cannot use prompt lookup: its state cannot be rewound by overwriting)
            return new Speculator.Config(Speculator.Mode.MTP, spec.k(), spec.ngramMax(), spec.ngramMin(), spec.window());
        }
        return spec;
    }

    static Speculator.Config speculation(Settings s, Architecture arch) {
        Speculator.Config spec = s.speculation();
        boolean moe = arch instanceof org.robomiri.jllm.model.Decoder d && d.anyMoe();
        if (moe && spec.mode() == Speculator.Mode.NGRAM && "default".equals(s.source("engine.speculative"))) {
            return new Speculator.Config(Speculator.Mode.OFF, spec.k(), spec.ngramMax(), spec.ngramMin(), spec.window());
        }
        return spec;
    }

    /**
     * The constrained decoding settings (docs/17 §17.7): {@code engine.jump-forward} for this engine and
     * {@code engine.constraint-masks} for every constraint of the process.
     */
    static void constraints(Settings s, Engine engine) {
        engine.setJumpForward(s.getBool("engine.jump-forward"));
        org.robomiri.jllm.sampling.JsonConstraint.setFastMasks(s.getBool("engine.constraint-masks"));
    }

    /**
     * The pool of KV blocks the CPU slots of {@code weights} share when {@code backend.kv-paged} is on (docs/24), made
     * by the first slot; null when the flat cache is wanted or the model cannot be paged.
     */
    private static org.robomiri.jllm.kv.KvPool kvPool(Settings s, ModelWeights weights, KvDtype kvDtype, int slots,
                                                      boolean metal) {
        if (!s.getBool("backend.kv-paged")) {
            return null;
        }
        int ctx = s.getInt("backend.ctx");
        int block = s.getInt("backend.kv-block");
        long tokens = s.getLong("backend.kv-pool-tokens");
        String key = (metal ? "metal|" : "cpu|") + kvDtype + (kvDtype == KvDtype.Q8 && KvDtype.q8Rotate() ? "r" : "") + "|" + block + "|" + ctx + "|" + tokens;
        org.robomiri.jllm.kv.KvPool pool = org.robomiri.jllm.kv.KvPools.get(weights, key,
                () -> metal ? org.robomiri.jllm.metal.MetalBackend.newKvPool(weights, ctx, slots, kvDtype, block, tokens,
                        s.opt("backend.metal.lib")) : CpuBackend.newKvPool(weights, ctx, slots, kvDtype, block, tokens));
        if (pool == null) {
            System.err.println("note: backend.kv-paged is not available for this model on " + (metal ? "Metal" : "the CPU") + " ("
                    + (kvDtype == KvDtype.Q4 ? "q4 cache" : "latent attention") + "), using the flat KV cache");
        } else {
            pool.setSharing(s.getBool("backend.kv-prefix-share"));
        }
        return pool;
    }

    /** The batch group of each loaded model's weights (engine.batch-decode); weak: gone with the weights. */
    private static final java.util.Map<ModelWeights, org.robomiri.jllm.engine.BatchGroup> GROUPS =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    /** {@code backend} as a member of its weights' batch group, or as it is when batching is off. */
    private static Backend batched(Settings s, ModelWeights weights, Backend backend) {
        org.robomiri.jllm.engine.BatchGroup g = group(s, weights);
        return g == null ? backend : g.join(backend);
    }

    /** The batch group of {@code weights}, or null with engine.batch-decode off. */
    private static org.robomiri.jllm.engine.BatchGroup group(Settings s, ModelWeights weights) {
        return s.getBool("engine.batch-decode")
                ? GROUPS.computeIfAbsent(weights, w -> new org.robomiri.jllm.engine.BatchGroup()) : null;
    }

    /**
     * Another engine on the weights of {@code l}: its own backend and KV cache, the same tokenizer, template, stop
     * tokens and queue timeout. Lets a server keep several conversations' prefixes cached at once (memory: one KV
     * cache each; the memory-mapped weights are shared). Prompt-lookup speculation carries over; a draft model
     * does not (it would need its own copy).
     */
    /** The chat format of the model; a GGUF of a family whose model_type is shared with another one asks by its own name. */
    private static org.robomiri.jllm.tokenizer.ChatTemplate.Format chatFormat(Prepared p, Architecture arch, Tokenizer tok) {
        if (p.gguf() != null) {
            try (org.robomiri.jllm.gguf.GgufFile f = org.robomiri.jllm.gguf.GgufFile.open(p.gguf())) {
                String name = f.str("general.architecture", "");
                if (name.equals("qwen3vl")) {
                    return org.robomiri.jllm.tokenizer.ChatTemplate.forModelType(name, tok);
                }
            } catch (IOException | RuntimeException e) {
                // fall through to the model type's format
            }
        }
        return org.robomiri.jllm.tokenizer.ChatTemplate.forModelType(arch.core().modelType(), tok);
    }

    public static Engine extraEngine(Settings s, Loaded l) throws Exception {
        Engine first = l.engine();
        Backend backend = batched(s, l.weights(), newBackend(s, l.weights().arch, l.weights(), 1, true));
        Engine e = new Engine(backend, first.tokenizer(), l.weights().arch, first.stopTokenIds(), first.queueTimeoutMs(),
                first.template());
        constraints(s, e);
        Speculator.Config spec = speculation(s, l.weights().arch, l.weights());
        if (spec.mode() == Speculator.Mode.NGRAM || spec.mode() == Speculator.Mode.MTP) {
            e.enableSpeculation(spec, null);
        }
        e.setVision(first.vision(), false); // the encoder is shared; the first engine closes it
        return e;
    }

    /**
     * Resolves {@code model} (a Hugging Face repository, a local {@code .gguf} file, or a GGUF repository narrowed by
     * {@code file} / {@code quant}), downloads what is missing and loads its architecture, tokenizer and weights.
     * Used for the target model and for a draft model alike.
     */
    public static Prepared prepare(Settings s, HuggingFaceHub hub, String model, String file, String quant) throws Exception {
        Source src = resolve(s, hub, model, file, quant);
        if (src.gguf() != null) {
            return prepareGguf(s, hub, src);
        }
        Path dir = src.dir();
        String revision = s.get("model.revision");
        if (!Files.isRegularFile(dir.resolve("config.json"))) {
            hub.downloadMetadata(model, revision, dir);
        }
        Architecture arch = Architecture.load(dir);
        System.err.println("Model: " + arch);
        Tokenizer tok = Tokenizer.load(dir.resolve("tokenizer.json"));
        QType q = QType.parse(s.get("model.quant"));
        QType head = QType.parseHead(s.get("model.head-quant"));
        ShardProvider shards = shardProvider(hub, model, revision, dir, s.getBool("model.low-disk"));
        ModelWeights weights = ModelWeights.load(dir, arch, q, head, shards, model + "@" + revision);
        return new Prepared(arch, tok, weights, dir);
    }

    /** A GGUF file: architecture, tokenizer and weights all come from it, nothing is converted or cached. */
    private static Prepared prepareGguf(Settings s, HuggingFaceHub hub, Source src) throws Exception {
        long t0 = System.nanoTime();
        long tokNanos;
        GgufModel gm = GgufModel.open(src.gguf());
        ModelWeights weights;
        Architecture arch;
        Tokenizer tok;
        try {
            arch = gm.architecture();
            System.err.println("GGUF: " + gm.describe());
            System.err.println("Model: " + arch);
            String warn = gm.chatTemplateWarning();
            if (warn != null) {
                System.err.println("warning: " + warn);
            }
            tok = ggufTokenizer(s, hub, gm);
            tokNanos = System.nanoTime() - t0;
            String backendName = s.get("backend");
            boolean cpu = backendName.equals("cpu") || backendName.equals("auto") && !Metal.available(s.opt("backend.metal.lib"));
            java.util.Set<org.robomiri.jllm.gguf.GgmlType> requantized = gm.forBackend(cpu);
            if (!requantized.isEmpty()) {
                System.err.println("GGUF on the CPU backend: " + requantized + " matrices are requantized to Q8_0 at load (their decode is"
                        + " faster on the Q8_0 kernels than in place; -Djllm.gguf.requantize=none reads them in place)");
            }
            weights = gm.weights();
        } catch (Exception e) {
            gm.close();
            throw e;
        }
        System.err.printf("Mapped %s weights (GGUF, %s, %.0f MB) in %.2fs (tokenizer %.2fs)%n", arch.core().architecture(),
                weights.quant, weights.byteSize() / 1e6, (System.nanoTime() - t0) / 1e9, tokNanos / 1e9);
        return new Prepared(arch, tok, weights, null, src.gguf());
    }

    /** The tokenizer built from the file; when its pre-tokenizer is unknown, the tokenizer.json of the original repository. */
    private static Tokenizer ggufTokenizer(Settings s, HuggingFaceHub hub, GgufModel gm) throws Exception {
        try {
            return gm.tokenizer();
        } catch (GgufTokenizer.UnsupportedPreTokenizer e) {
            String repo = gm.originalRepo();
            if (repo == null) {
                throw new IOException(e.getMessage() + "; and the file names no original repository (general.base_model.0.repo_url)"
                        + " to take tokenizer.json from", e);
            }
            System.err.println("note: " + e.getMessage() + "; using the tokenizer.json of " + repo);
            Path tdir = Path.of(s.get("model.dir")).resolve("tokenizers").resolve(repo.replace('/', '_'));
            Files.createDirectories(tdir);
            hub.ensure(repo, "main", tdir, "tokenizer.json");
            try {
                hub.ensure(repo, "main", tdir, "tokenizer_config.json");
            } catch (IOException ignored) {
                // optional: it only supplies BOS / EOS names
            }
            return Tokenizer.load(tdir.resolve("tokenizer.json"));
        }
    }

    private static ShardProvider shardProvider(HuggingFaceHub hub, String model, String revision, Path dir, boolean lowDisk) {
        return new ShardProvider() {
            @Override
            public Path ensure(String name) throws IOException {
                Path p = dir.resolve(name);
                if (Files.isRegularFile(p)) {
                    return p;
                }
                try {
                    return hub.ensure(model, revision, dir, name);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("interrupted");
                }
            }

            @Override
            public String commitSha() {
                try {
                    return hub.commitSha(model, revision);
                } catch (IOException | InterruptedException e) {
                    return null; // offline: the cache simply records no commit
                }
            }

            @Override
            public void release(String name) throws IOException {
                if (lowDisk) {
                    Files.deleteIfExists(dir.resolve(name));
                    System.err.println("\ndeleted " + name + " (model.low-disk)");
                }
            }
        };
    }

    public static Backend createBackend(Settings s, Architecture arch, ModelWeights weights) throws Exception {
        try {
            return newBackend(s, arch, weights, 1, false);
        } catch (Exception e) {
            weights.close();
            throw e;
        }
    }

    /** A backend on {@code weights}, which stay open whatever happens (they may be shared). */
    /** {@code member}: the backend will join its weights' batch group (and share the group's CPU threads). */
    private static Backend newBackend(Settings s, Architecture arch, ModelWeights weights, int slots, boolean member)
            throws Exception {
        int ctx = s.getInt("backend.ctx");
        int batch = batch(s, arch);
        String backendName = resolveBackend(s);
        int threads = s.getInt("backend.threads");
        // before any CPU matmul exists (and before the int8 auto probe, which times the native kernels when present)
        // (-Djllm.cpu.native, as the Docker image's JLLM_CPU_NATIVE sets it, stands in for an unset backend.cpu.native)
        String nativeMode = s.get("backend.cpu.native");
        if ("default".equals(s.source("backend.cpu.native")) && System.getProperty("jllm.cpu.native") != null) {
            nativeMode = System.getProperty("jllm.cpu.native");
        }
        org.robomiri.jllm.cpu.NativeCpu.configure(nativeMode, s.opt("backend.cpu.native-lib"));
        KvDtype kvDtype = KvDtype.parse(s.get("backend.kv-dtype"));
        KvDtype.setQ4Recent(s.getInt("backend.kv-q4-recent"));
        KvDtype.setQ8Rotate(s.getBool("backend.kv-q8-rotate"));
        boolean mamba = arch instanceof MambaArch;
        if (member && s.getBool("backend.kv-paged") && (mamba || arch instanceof Qwen3NextArch)) {
            System.err.println("note: backend.kv-paged does not apply to " + (mamba ? "Mamba" : "Qwen3-Next / Qwen3.5")
                    + ": a recurrent state is a function of every earlier token and cannot be shared per block, so the flat cache is used");
        }
        if (mamba && kvDtype != KvDtype.F32) { // a recurrent state: there is no KV cache to store as f16
            System.err.println("note: backend.kv-dtype=" + kvDtype + " is ignored for a Mamba model (it has no KV cache)");
        }
        if (arch instanceof Qwen3NextArch) { // hybrid: DeltaNet states plus a KV cache for the full-attention layers
            return switch (backendName) {
                case "cpu" -> new Qwen3NextCpuBackend(weights, ctx, batch,
                        threads > 0 ? threads : Runtime.getRuntime().availableProcessors(), kvDtype,
                        hybridInt8(s, weights, threads));
                case "metal" -> new Qwen3NextMetalBackend(weights, ctx, batch, s.opt("backend.metal.lib"), kvDtype);
                default -> throw new IllegalArgumentException("unknown backend '" + backendName + "' (cpu|metal|auto)");
            };
        }
        return switch (backendName) {
            case "cpu" -> mamba
                    ? new MambaCpuBackend(weights, ctx, batch,
                            threads > 0 ? threads : Runtime.getRuntime().availableProcessors(),
                            int8(s, weights, threads))
                    : member && group(s, weights) != null // a group's slots share one pool: they never compute at once
                    ? new CpuBackend(weights, ctx, batch, group(s, weights).cpuPool(
                            threads > 0 ? threads : Runtime.getRuntime().availableProcessors()), kvDtype,
                            int8(s, weights, threads), member ? kvPool(s, weights, kvDtype, slots, false) : null)
                    : new CpuBackend(weights, ctx, batch,
                            threads > 0 ? threads : Runtime.getRuntime().availableProcessors(), kvDtype,
                            int8(s, weights, threads), member ? kvPool(s, weights, kvDtype, slots, false) : null);
            case "metal" -> mamba
                    ? new MambaMetalBackend(weights, ctx, batch, s.opt("backend.metal.lib"))
                    : new MetalBackend(weights, ctx, batch, s.opt("backend.metal.lib"), kvDtype,
                            new MoeSplit(MoeSplit.parseLayers(s.get("backend.metal.cpu-moe")),
                                    threads > 0 ? threads : Runtime.getRuntime().availableProcessors(),
                                    int8(s, weights, threads), null, Math.max(1, slots)),
                            member ? kvPool(s, weights, kvDtype, slots, true) : null);
            default -> throw new IllegalArgumentException("unknown backend '" + backendName + "' (cpu|metal|auto)");
        };
    }

    /**
     * Loads the draft model of {@code engine.draft-model} (a safetensors repository, a GGUF repository with a single
     * file, or a local {@code .gguf}) on the same backend type; it must share the target's tokenizer.
     */
    private static DraftModel loadDraft(Settings s, HuggingFaceHub hub, Tokenizer targetTok, Architecture targetArch,
                                        ModelWeights targetWeights) throws Exception {
        String repo = s.get("engine.draft-model");
        Prepared p;
        try {
            p = prepare(s, hub, repo, null, null);
        } catch (Exception e) {
            targetWeights.close();
            throw e;
        }
        Architecture arch = p.arch();
        ModelWeights weights = p.weights();
        try {
            if (arch instanceof MambaArch || targetArch instanceof MambaArch
                    || arch instanceof Qwen3NextArch || targetArch instanceof Qwen3NextArch) {
                throw new IllegalArgumentException("engine.draft-model: speculation needs attention (KV cache) models");
            }
            System.err.println("Draft model: " + arch);
            Tokenizer tok = p.tok();
            String probe = "Hello, world! The quick brown fox \u00e9\u4e2d 123\n\tdef f(x): return x";
            if (tok.vocabSize() != targetTok.vocabSize() || !java.util.Arrays.equals(tok.encode(probe), targetTok.encode(probe))) {
                throw new IllegalArgumentException("engine.draft-model " + repo + " does not share the target's tokenizer");
            }
        } catch (RuntimeException e) {
            weights.close();
            targetWeights.close();
            throw e;
        }
        Backend backend;
        try {
            backend = createBackend(s, arch, weights); // closes the draft's weights if it fails
        } catch (Exception e) {
            targetWeights.close();
            throw e;
        }
        System.err.println("Draft backend: " + backend.name());
        return new DraftModel(backend, arch.core().vocabSize(), weights::close);
    }

    private static Set<Integer> eosIds(Path dir, Tokenizer tok, Architecture arch) throws Exception {
        Set<Integer> ids = new HashSet<>(arch.core().eosTokenIds());
        Path gen = dir == null ? null : dir.resolve("generation_config.json");
        if (gen != null && Files.isRegularFile(gen)) {
            Object eos = Json.parseObject(Files.readString(gen)).get("eos_token_id");
            if (eos instanceof Number n) {
                ids.add(n.intValue());
            } else if (eos instanceof List<?> l) {
                for (Object o : l) {
                    if (!(o instanceof Number n)) {
                        throw new IllegalArgumentException(gen + ": eos_token_id must be an integer or a list of"
                                + " integers, found " + Json.write(o));
                    }
                    ids.add(n.intValue());
                }
            } else if (eos != null) {
                throw new IllegalArgumentException(gen + ": eos_token_id must be an integer or a list of integers,"
                        + " found " + Json.write(eos));
            }
        }
        // Family stop tokens: ChatML (Qwen), Llama 3 end-of-turn / end-of-text, Mistral / Llama 2 </s>.
        // the family template's end-of-turn ids are added by Engine; these cover base models without a template
        for (String name : new String[]{"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "</s>"}) {
            int id = tok.tokenId(name);
            if (id >= 0) {
                ids.add(id);
            }
        }
        // no fallback to token 0: config.json's eos_token_id is already in the core's eosTokenIds, and the chat
        // template adds its own end-of-turn ids in Engine; a model without any simply stops at max_tokens
        return ids;
    }

    /**
     * {@code backend.cpu.int8-activations} for a Gated DeltaNet hybrid: {@code auto} is the measurement of the dense
     * backends when it can be taken on these weights, else the float path (which is what the checks compare with).
     */
    private static boolean hybridInt8(Settings s, org.robomiri.jllm.model.ModelWeights weights, int threads) {
        try {
            return int8(s, weights, threads);
        } catch (IllegalArgumentException e) {
            throw e; // a malformed setting
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** {@code backend.cpu.int8-activations} for {@code weights} ({@code auto} measures both paths once per model). */
    private static boolean int8(Settings s, org.robomiri.jllm.model.ModelWeights weights, int threads) {
        return org.robomiri.jllm.cpu.CpuBackend.int8Activations(s.get("backend.cpu.int8-activations"), weights,
                threads > 0 ? threads : Runtime.getRuntime().availableProcessors());
    }
}
