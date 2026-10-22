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
                case "serve" -> serve(ModelLoader.load(s, hub, model), model, s, hub);
                case "transcribe" -> org.robomiri.jllm.audio.AudioCli.transcribe(s, hub, model, cli(s, "audio"), cli(s, "language"),
                        "true".equals(cli(s, "translate")), cli(s, "format") == null ? "text" : cli(s, "format"),
                        "true".equals(cli(s, "words")));
                case "speak" -> org.robomiri.jllm.audio.AudioCli.speak(s, hub, model, cli(s, "prompt"), cli(s, "voice"),
                        cli(s, "speed") == null ? 1.0 : Double.parseDouble(cli(s, "speed")), cli(s, "out"));
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
            if (s.has("image")) {
                runWithImages(engine, s, text);
                return;
            }
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

    /** {@code run --image a.png,b.jpg --prompt "..."}: the pictures come first in the message, in the order given. */
    private static void runWithImages(Engine engine, Settings s, String text) {
        List<org.robomiri.jllm.vision.Rgb> images = new ArrayList<>();
        StringBuilder content = new StringBuilder();
        for (String f : s.get("image").split(",")) {
            if (f.isBlank()) {
                continue;
            }
            try {
                images.add(org.robomiri.jllm.vision.ImageDecoder.decode(java.nio.file.Files.readAllBytes(java.nio.file.Path.of(f.trim())),
                        s.getLong("vision.max-image-pixels")));
            } catch (java.io.IOException e) {
                throw new IllegalArgumentException("--image " + f + ": " + e.getMessage(), e);
            }
            content.append(org.robomiri.jllm.vision.MediaChat.MARK);
        }
        content.append(org.robomiri.jllm.vision.MediaChat.stripMarks(text));
        long t0 = System.nanoTime();
        org.robomiri.jllm.engine.MediaPrompt mp = org.robomiri.jllm.vision.MediaChat.render(engine,
                List.of(new ChatTemplate.Message("user", content.toString())), s.getBool("chat.thinking"), images);
        if (Boolean.getBoolean("jllm.vision.debug")) { // the prompt ids with each picture as <image n>
            StringBuilder sb = new StringBuilder();
            int[] tk = mp.tokens();
            for (int i = 0; i < tk.length; ) {
                org.robomiri.jllm.engine.MediaPrompt.Span sp = mp.spanAt(i);
                if (sp != null && sp.start() == i) {
                    sb.append(" <image ").append(sp.length()).append(">");
                    i += sp.length();
                } else {
                    sb.append(' ').append(tk[i++]);
                }
            }
            System.err.println("prompt:" + sb);
        }
        double encodeMs = engine.vision().lastEncodeMs();
        System.err.printf("%d image(s) -> %d prompt tokens, %.0f ms to encode (%d new)%n", images.size(), mp.length(), encodeMs,
                engine.vision().lastEncodedImages());
        Engine.Result r = engine.generate(mp, s.sampling(), Engine.Decoding.NONE, 1, new Engine.ChoiceSink() {
            @Override
            public void text(int index, String t) {
                System.out.print(t);
                System.out.flush();
            }
        }, () -> false).getFirst();
        System.out.println();
        System.err.println(statsLine(r.stats()) + ", finish=" + r.finishReason() + ", total " + (System.nanoTime() - t0) / 1_000_000 + " ms");
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

    /** A flag only some commands take (--audio, --voice, ...), or null. */
    private static String cli(Settings s, String key) {
        return s.has(key) ? s.get(key) : null;
    }

    private static void serve(ModelLoader.Loaded l, String model, Settings s, HuggingFaceHub hub) throws Exception {
        int port = s.getInt("server.port");
        String host = s.get("server.host");
        org.robomiri.jllm.audio.AudioSettings as = org.robomiri.jllm.audio.AudioSettings.of(s);
        boolean audio = !as.sttModel().isEmpty() || !as.ttsModel().isEmpty();
        HttpServer.Options http = new HttpServer.Options(
                s.getInt("server.read-timeout-ms"),
                s.getInt("server.write-timeout-ms"),
                s.getInt("server.max-connections"),
                audio ? Math.max(s.getLong("server.max-body-bytes"), as.maxUploadBytes()) : s.getLong("server.max-body-bytes"),
                s.opt("server.cors-origin"));
        ApiServer.Options options = new ApiServer.Options(s.sampling(), s.getBool("chat.thinking"),
                s.getLong("engine.max-prompt-bytes"), s.getLong("server.request-timeout-ms"), http);
        try (ModelWeights weights = l.weights(); Engine engine = l.engine();
             org.robomiri.jllm.audio.AudioModels audioModels = new org.robomiri.jllm.audio.AudioModels();
             ApiServer server = new ApiServer(engine, model, host, port, options).imageLimits(
                     s.getLong("vision.max-image-bytes"), s.getLong("vision.max-image-pixels"), s.getBool("vision.allow-private-urls"))) {
            if (audio) {
                for (String repo : new String[]{as.sttModel(), as.ttsModel()}) {
                    if (!repo.isEmpty()) {
                        System.err.println("Audio model " + org.robomiri.jllm.audio.AudioCli.load(s, hub, audioModels, repo).description());
                    }
                }
                server.setAudio(new org.robomiri.jllm.server.AudioRoutes(audioModels,
                        new org.robomiri.jllm.audio.AudioDecoder(as.ffmpeg(), as.maxSeconds()), as));
            }
            System.err.println("Listening on http://" + host + ":" + server.port() + "  (chat UI at /, API at /v1/chat/completions)");
            server.serve();
        }
    }

    private static void bench(ModelLoader.Loaded l, Settings s) {
        try (ModelWeights weights = l.weights(); Engine engine = l.engine()) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < s.getInt("bench.prompt-repeat"); i++) {
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
