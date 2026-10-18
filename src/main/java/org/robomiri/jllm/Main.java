package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.hub.HuggingFaceHub;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.server.ApiServer;
import org.robomiri.jllm.server.HttpServer;
import org.robomiri.jllm.tokenizer.ChatTemplate;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

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
        HuggingFaceHub hub = ModelLoader.hub(s);

        try {
            switch (cmd) {
                case "config" -> System.out.print(s.describe());
                case "download" -> ModelLoader.download(s, hub, model);
                case "chat" -> chat(ModelLoader.load(s, hub, model), s);
                case "run" -> run(ModelLoader.load(s, hub, model), s);
                case "serve" -> serve(ModelLoader.load(s, hub, model), model, s);
                case "bench" -> bench(ModelLoader.load(s, hub, model), s);
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
                  --model <repo|file.gguf>  --revision <rev>  --dir <path>  --quant q4|q8|q2k|q6k  --head-quant q8|q6k  --low-disk
                  --gguf-file <name>    --gguf-quant <label> (a GGUF repo: pick the file; the file is used as it is)
                  --backend cpu|metal   --ctx <n>            --batch <n>         --threads <n>
                  --temp <t> --top-k <k> --top-p <p> --min-p <p> --rep-penalty <r> --seed <s> --max-tokens <n>
                  --think               --system "..."       --host <h>          --port <p>
                  --prompt "..."        --raw (run: no chat template)
                  any property key directly, e.g. --server.port 9000 or --bench.rounds=5""");
    }


    private static void run(ModelLoader.Loaded l, Settings s) {
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

    private static void chat(ModelLoader.Loaded l, Settings s) throws Exception {
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

    private static void serve(ModelLoader.Loaded l, String model, Settings s) throws Exception {
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

    private static void bench(ModelLoader.Loaded l, Settings s) {
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
        String spec = s.draftedTokens() == 0 ? "" : String.format(", speculation accepted %d of %d drafted (%.0f%%)",
                s.acceptedTokens(), s.draftedTokens(), 100.0 * s.acceptedTokens() / s.draftedTokens());
        return String.format("%d prompt tok (%d cached), prefill %.0f ms, %d generated in %.0f ms = %.1f tok/s%s",
                s.promptTokens(), s.cachedTokens(), s.prefillMs(), s.generatedTokens(), s.decodeMs(), s.tokPerSec(), spec);
    }
}
