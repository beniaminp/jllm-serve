package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.cpu.MambaCpuBackend;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MambaMetalBackend;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.MambaArch;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.server.ApiServer;
import org.robomiri.jllm.server.HttpServer;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.GgufTokenizer;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Command line entry point. All options live in {@code jllm.properties} (bundled defaults), may be overridden
 * by an external file ({@code --config}, {@code $JLLM_CONFIG} or {@code ./jllm.properties}) and by flags.
 *
 * <pre>
 *   jllm download [--model Qwen/Qwen3-0.6B]
 *   jllm chat     [--backend cpu|metal] [--quant q4|q8] [--ctx 4096] [--think]
 *   jllm run      --prompt "..." [--max-tokens 256] [--temp 0.6]
 *   jllm serve    [--port 8080] [--host 127.0.0.1]
 *   jllm bench
 *   jllm config   (prints the effective settings)
 * </pre>
 */
public final class Main {

    private Main() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || args[0].equals("-h") || args[0].equals("--help")) {
            usage();
            return;
        }
        String cmd = args[0];
        Settings s;
        try {
            s = Settings.load(args).validate(); // every typed value is checked here, before the weights load
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.exit(2);
            return;
        }
        String model = s.get("model.repo");
        HuggingFaceHub hub = new HuggingFaceHub(s.opt("hub.endpoint"), s.opt("hub.token"), s.getLong("hub.timeout-ms"));

        try {
            switch (cmd) {
                case "config" -> System.out.print(s.describe());
                case "download" -> download(s, hub, model);
                case "chat" -> chat(load(s, hub, model), s);
                case "run" -> run(load(s, hub, model), s);
                case "serve" -> serve(load(s, hub, model), model, s);
                case "bench" -> bench(load(s, hub, model), s);
                default -> {
                    System.err.println("unknown command: " + cmd);
                    usage();
                    System.exit(2);
                }
            }
        } catch (HuggingFaceHub.GgufSelectionException e) {
            System.err.println(e.getMessage()); // a usage problem, not a failure: the message lists the choices
            System.exit(2);
        }
    }

    private static void usage() {
        System.err.println("""
                usage: jllm <command> [options]
                  download   fetch the model from the Hugging Face Hub
                  chat       interactive chat in the terminal
                  run        single prompt completion (--prompt "...")
                  serve      OpenAI-compatible HTTP server
                  bench      measure prefill/decode throughput
                  config     print the effective configuration and where each value comes from
                options (defaults in src/main/resources/jllm.properties; override with --config <file> or flags):
                  --model <repo|file.gguf>  --revision <rev>  --dir <path>      --quant q4|q8|q2k   --low-disk
                  --gguf-file <name>    --gguf-quant <label> (a GGUF repo: pick the file; the file is used as it is)
                  --backend cpu|metal   --ctx <n>            --batch <n>         --threads <n>
                  --temp <t> --top-k <k> --top-p <p> --min-p <p> --rep-penalty <r> --seed <s> --max-tokens <n>
                  --think               --system "..."       --host <h>          --port <p>
                  --prompt "..."        --raw (run: no chat template)
                  any property key directly, e.g. --server.port 9000 or --bench.rounds=5""");
    }

    /**
     * The loaded model. Close in this order: engine (the backend releases its buffers) and then weights (the
     * memory-mapped quantized cache): {@code try (ModelWeights w = l.weights(); Engine e = l.engine())}.
     */
    private record Loaded(ModelWeights weights, Engine engine) {
    }

    /** Where the model comes from: a GGUF file (used as it is), or a Hugging Face safetensors checkpoint in {@code dir}. */
    private record Source(Path gguf, Path dir) {
    }

    private static void download(Settings s, HuggingFaceHub hub, String model) throws Exception {
        Source src = resolve(s, hub, model);
        if (src.gguf() == null) {
            hub.download(model, s.get("model.revision"), src.dir());
        } else {
            System.err.println(src.gguf());
        }
    }

    /**
     * Decides how {@code model} is loaded, downloading what is missing. A name ending in {@code .gguf} is a local
     * file. A repository is a GGUF one when {@code model.gguf-file} / {@code model.gguf-quant} is set, or when it
     * has {@code .gguf} files and no safetensors checkpoint; a model directory that already has a
     * {@code config.json} is a safetensors one without asking the Hub. Only the chosen GGUF file (and its split
     * parts) is downloaded.
     */
    private static Source resolve(Settings s, HuggingFaceHub hub, String model) throws Exception {
        if (model.toLowerCase(java.util.Locale.ROOT).endsWith(".gguf")) {
            Path file = Path.of(model).toAbsolutePath();
            if (!Files.isRegularFile(file)) {
                throw new IOException("no such GGUF file: " + file);
            }
            return new Source(file, file.getParent());
        }
        Path dir = Path.of(s.get("model.dir")).resolve(model.replace('/', '_'));
        String revision = s.get("model.revision");
        String file = s.opt("model.gguf-file");
        String quant = s.opt("model.gguf-quant");
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

    private static Loaded load(Settings s, HuggingFaceHub hub, String model) throws Exception {
        Source src = resolve(s, hub, model);
        if (src.gguf() != null) {
            return loadGguf(s, hub, src);
        }
        Path dir = src.dir();
        String revision = s.get("model.revision");
        if (!Files.isRegularFile(dir.resolve("config.json"))) {
            hub.downloadMetadata(model, revision, dir);
        }
        Architecture arch = Architecture.load(dir);
        System.err.println("Model: " + arch);
        Tokenizer tok = Tokenizer.load(dir.resolve("tokenizer.json"));
        QType quant = QType.parse(s.get("model.quant"));
        boolean lowDisk = s.getBool("model.low-disk");
        ShardProvider shards = new ShardProvider() {
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
        ModelWeights weights = ModelWeights.load(dir, arch, quant, shards, model + "@" + revision);
        return start(s, arch, tok, weights, dir);
    }

    /** A GGUF file: architecture, tokenizer and weights all come from it, nothing is converted or cached. */
    private static Loaded loadGguf(Settings s, HuggingFaceHub hub, Source src) throws Exception {
        long t0 = System.nanoTime();
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
            weights = gm.weights();
        } catch (Exception e) {
            gm.close();
            throw e;
        }
        System.err.printf("Mapped %s weights (GGUF, %s, %.0f MB) in %.2fs%n", arch.core().architecture(), weights.quant,
                weights.byteSize() / 1e6, (System.nanoTime() - t0) / 1e9);
        return start(s, arch, tok, weights, null);
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

    /** Backend and engine over loaded weights; {@code dir} (null for a GGUF file) may hold a generation_config.json. */
    private static Loaded start(Settings s, Architecture arch, Tokenizer tok, ModelWeights weights, Path dir) throws Exception {
        int ctx = s.getInt("backend.ctx");
        int batch = s.getInt("backend.batch");
        String backendName = s.get("backend");
        if (backendName.equals("auto")) {
            backendName = Metal.available(s.opt("backend.metal.lib")) ? "metal" : "cpu";
        }
        int threads = s.getInt("backend.threads");
        KvDtype kvDtype = KvDtype.parse(s.get("backend.kv-dtype"));
        boolean mamba = arch instanceof MambaArch;
        if (mamba && kvDtype != KvDtype.F32) { // a recurrent state: there is no KV cache to store as f16
            System.err.println("note: backend.kv-dtype=" + kvDtype + " is ignored for a Mamba model (it has no KV cache)");
        }
        Backend backend;
        try {
            backend = switch (backendName) {
                case "cpu" -> mamba
                        ? new MambaCpuBackend(weights, ctx, batch,
                                threads > 0 ? threads : Runtime.getRuntime().availableProcessors(),
                                s.getBool("backend.cpu.int8-activations"))
                        : new CpuBackend(weights, ctx, batch,
                                threads > 0 ? threads : Runtime.getRuntime().availableProcessors(), kvDtype,
                                s.getBool("backend.cpu.int8-activations"));
                case "metal" -> mamba
                        ? new MambaMetalBackend(weights, ctx, batch, s.opt("backend.metal.lib"))
                        : new MetalBackend(weights, ctx, batch, s.opt("backend.metal.lib"), kvDtype);
                default -> throw new IllegalArgumentException("unknown backend '" + backendName + "' (cpu|metal|auto)");
            };
        } catch (Exception e) {
            weights.close();
            throw e;
        }
        System.err.println("Backend: " + backend.name() + ", ctx=" + ctx + ", batch=" + batch + ", kv=" + kvDtype);
        try {
            return new Loaded(weights, new Engine(backend, tok, arch, eosIds(dir, tok, arch), s.getLong("server.queue-timeout-ms")));
        } catch (Exception e) { // e.g. a family without a chat template, or a bad generation_config.json
            backend.close();
            weights.close();
            throw e;
        }
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

    private static void run(Loaded l, Settings s) {
        try (ModelWeights weights = l.weights(); Engine engine = l.engine()) {
            String text = s.has("prompt") ? s.get("prompt") : "Hello";
            int[] prompt = s.has("raw") && s.getBool("raw")
                    ? engine.tokenizer().encode(text)
                    : engine.renderChat(List.of(new ChatTemplate.Message("user", text)), s.getBool("chat.thinking"));
            Engine.Result r = engine.generate(prompt, s.sampling(), out -> {
                System.out.print(out);
                System.out.flush();
            }, () -> false);
            System.out.println();
            System.err.println(statsLine(r.stats()) + ", finish=" + r.finishReason());
        }
    }

    private static void chat(Loaded l, Settings s) throws Exception {
        try (ModelWeights weights = l.weights(); Engine engine = l.engine()) {
            BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
            List<ChatTemplate.Message> history = new ArrayList<>();
            String system = s.opt("chat.system");
            if (system != null) {
                history.add(new ChatTemplate.Message("system", system));
            }
            boolean think = s.getBool("chat.thinking");
            SamplingParams params = s.sampling();
            System.err.println("Type a message, /clear to reset, /quit to exit.");
            while (true) {
                System.out.print("\n> ");
                System.out.flush();
                String line = in.readLine();
                if (line == null || line.trim().equals("/quit")) {
                    break;
                }
                if (line.trim().equals("/clear")) {
                    history.clear();
                    continue;
                }
                if (line.isBlank()) {
                    continue;
                }
                history.add(new ChatTemplate.Message("user", line));
                int[] prompt = engine.renderChat(history, think);
                Engine.Result r = engine.generate(prompt, params, out -> {
                    System.out.print(out);
                    System.out.flush();
                }, () -> false);
                System.out.println();
                System.err.println("[" + statsLine(r.stats()) + "]");
                history.add(new ChatTemplate.Message("assistant", r.text()));
            }
        }
    }

    private static void serve(Loaded l, String model, Settings s) throws Exception {
        int port = s.getInt("server.port");
        String host = s.get("server.host");
        HttpServer.Options http = new HttpServer.Options(
                s.getInt("server.read-timeout-ms"),
                s.getInt("server.write-timeout-ms"),
                s.getInt("server.max-connections"),
                s.getLong("server.max-body-bytes"),
                s.opt("server.cors-origin"));
        ApiServer.Options options = new ApiServer.Options(s.sampling(), s.getBool("chat.thinking"),
                s.getLong("engine.max-prompt-bytes"), s.getLong("server.request-timeout-ms"), http);
        try (ModelWeights weights = l.weights(); Engine engine = l.engine();
             ApiServer server = new ApiServer(engine, model, host, port, options)) {
            System.err.println("Listening on http://" + host + ":" + server.port() + "  (chat UI at /, API at /v1/chat/completions)");
            server.serve();
        }
    }

    private static void bench(Loaded l, Settings s) {
        try (ModelWeights weights = l.weights(); Engine engine = l.engine()) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 40; i++) {
                sb.append("The quick brown fox jumps over the lazy dog. ");
            }
            int[] prompt = engine.renderChat(List.of(new ChatTemplate.Message("user", sb.toString())), false);
            SamplingParams p = s.sampling().withMaxTokens(s.getInt("bench.max-tokens"));
            engine.generate(prompt, p, null, () -> false); // warm-up
            for (int i = 0; i < s.getInt("bench.rounds"); i++) {
                engine.generate("warm " + i, p.withMaxTokens(1), null); // evict prefix cache
                Engine.Result r = engine.generate(prompt, p, null, () -> false);
                Engine.Stats st = r.stats();
                System.out.printf("prefill %d tok in %.0f ms (%.1f tok/s), decode %d tok in %.0f ms (%.1f tok/s)%n",
                        st.promptTokens() - st.cachedTokens(), st.prefillMs(),
                        (st.promptTokens() - st.cachedTokens()) / (st.prefillMs() / 1000), st.generatedTokens(),
                        st.decodeMs(), st.tokPerSec());
            }
        }
    }

    private static String statsLine(Engine.Stats s) {
        return String.format("%d prompt tok (%d cached), prefill %.0f ms, %d generated in %.0f ms = %.1f tok/s",
                s.promptTokens(), s.cachedTokens(), s.prefillMs(), s.generatedTokens(), s.decodeMs(), s.tokPerSec());
    }
}
