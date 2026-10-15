package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.cpu.MambaCpuBackend;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MambaMetalBackend;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.server.ApiServer;
import org.robomiri.jllm.server.HttpServer;
import org.robomiri.jllm.tokenizer.ChatTemplate;
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
            s = Settings.load(args);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.exit(2);
            return;
        }
        String model = s.get("model.repo");
        Path dir = Path.of(s.get("model.dir")).resolve(model.replace('/', '_'));
        HuggingFaceHub hub = new HuggingFaceHub(s.opt("hub.endpoint"), s.opt("hub.token"), s.getLong("hub.timeout-ms"));

        switch (cmd) {
            case "config" -> System.out.print(s.describe());
            case "download" -> hub.download(model, s.get("model.revision"), dir);
            case "chat" -> chat(load(s, hub, model, dir), s);
            case "run" -> run(load(s, hub, model, dir), s);
            case "serve" -> serve(load(s, hub, model, dir), model, s);
            case "bench" -> bench(load(s, hub, model, dir), s);
            default -> {
                System.err.println("unknown command: " + cmd);
                usage();
                System.exit(2);
            }
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
                  --model <repo>        --revision <rev>     --dir <path>        --quant q4|q8|q2k   --low-disk
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

    private static Loaded load(Settings s, HuggingFaceHub hub, String model, Path dir) throws Exception {
        String revision = s.get("model.revision");
        if (!Files.isRegularFile(dir.resolve("config.json"))) {
            hub.downloadMetadata(model, revision, dir);
        }
        ModelConfig cfg = ModelConfig.load(dir);
        System.err.println("Model: " + cfg);
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
            public void release(String name) throws IOException {
                if (lowDisk) {
                    Files.deleteIfExists(dir.resolve(name));
                    System.err.println("\ndeleted " + name + " (model.low-disk)");
                }
            }
        };
        ModelWeights weights = ModelWeights.load(dir, cfg, quant, shards);
        int ctx = s.getInt("backend.ctx");
        int batch = s.getInt("backend.batch");
        String backendName = s.get("backend");
        if (backendName.equals("auto")) {
            backendName = Metal.available(s.opt("backend.metal.lib")) ? "metal" : "cpu";
        }
        int threads = s.getInt("backend.threads");
        KvDtype kvDtype = KvDtype.parse(s.get("backend.kv-dtype"));
        Backend backend;
        try {
            backend = switch (backendName) {
                case "cpu" -> weights.cfg.isMamba()
                        ? new MambaCpuBackend(weights, ctx, batch,
                                threads > 0 ? threads : Runtime.getRuntime().availableProcessors())
                        : new CpuBackend(weights, ctx, batch,
                                threads > 0 ? threads : Runtime.getRuntime().availableProcessors(), kvDtype);
                case "metal" -> weights.cfg.isMamba()
                        ? new MambaMetalBackend(weights, ctx, batch, s.opt("backend.metal.lib"))
                        : new MetalBackend(weights, ctx, batch, s.opt("backend.metal.lib"), kvDtype);
                default -> throw new IllegalArgumentException("unknown backend '" + backendName + "' (cpu|metal|auto)");
            };
        } catch (Exception e) {
            weights.close();
            throw e;
        }
        System.err.println("Backend: " + backend.name() + ", ctx=" + ctx + ", batch=" + batch + ", kv=" + kvDtype);
        return new Loaded(weights, new Engine(backend, tok, cfg, eosIds(dir, tok, cfg), s.getLong("server.queue-timeout-ms")));
    }

    private static Set<Integer> eosIds(Path dir, Tokenizer tok, ModelConfig cfg) throws Exception {
        Set<Integer> ids = new HashSet<>(cfg.eosTokenIds());
        Path gen = dir.resolve("generation_config.json");
        if (Files.isRegularFile(gen)) {
            Object eos = Json.parseObject(Files.readString(gen)).get("eos_token_id");
            if (eos instanceof Number n) {
                ids.add(n.intValue());
            } else if (eos instanceof List<?> l) {
                for (Object o : l) {
                    ids.add(((Number) o).intValue());
                }
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
        if (ids.isEmpty()) {
            ids.add((int) Json.num(Json.parseObject(Files.readString(dir.resolve("config.json"))), "eos_token_id", 0));
        }
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
