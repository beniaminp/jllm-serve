package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.server.ApiServer;
import org.robomiri.jllm.server.HttpServer;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Socket-level checks of {@link HttpServer} / {@link ApiServer} against a fake backend (instant, deterministic
 * tokens) and the real Qwen3 tokenizer. Dependency-free like {@link SmokeTest}; run with:
 * {@code java --add-modules jdk.incubator.vector -cp target/classes:target/test-classes org.robomiri.jllm.ServerTest}
 */
public final class ServerTest {

    /** Timeouts of the server under test (ms); every slow-client check is measured against these. */
    static final int READ_TIMEOUT = 1000;
    static final int WRITE_TIMEOUT = 1000;
    static final int QUEUE_TIMEOUT = 15000; // generous: a loaded machine fills the stalled client's socket buffers slowly

    static Path modelDir;

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "models/Qwen_Qwen3-0.6B");
        modelDir = dir;
        settings();
        jsonConstraintUnit();
        Tokenizer tok = Tokenizer.load(dir.resolve("tokenizer.json"));
        try (Fixture f = new Fixture(tok, "qwen3", 1 << 20)) {
            basics(f);
            badRequests(f);
            logprobs(f);
            logitBias(f);
            multipleChoices(f);
            bestOf(f);
            echoAndSuffix(f);
            jsonOutput(f);
            keepAlive(f);
            stalledStreamClient(f);
            disconnectedClient(f, false);
            disconnectedClient(f, true);
            slowRequest(f);
            midStreamFailure(f);
        }
        // a tokenizer that adds BOS (as Llama/Mistral tokenizer configs do): the BOS rule is the tokenizer's, not the family's
        Tokenizer tokBos = Tokenizer.fromJson(Files.readString(dir.resolve("tokenizer.json")),
                "{\"add_bos_token\": true, \"bos_token\": \"<|im_start|>\"}");
        try (Fixture f = new Fixture(tokBos, "llama", 64)) {
            bosContextCheck(f);
        }
        System.out.println("all server tests passed");
    }

    // ------------------------------------------------------------------ checks

    /** L9: every typed setting is validated up front, with a message naming the key. */
    static void settings() throws Exception {
        Settings ok = Settings.load(new String[]{"serve", "--ctx", "2048", "--port=9000"}).validate();
        check(ok.getInt("backend.ctx") == 2048 && ok.getInt("server.port") == 9000, "settings: aliases and --key=value");
        for (String[] bad : new String[][]{{"--ctx", "abc"}, {"--ctx", "0"}, {"--backend", "gpu"}, {"--quant", "q3"},
                {"--backend.kv-dtype", "f64"}, {"--hub.timeout-ms", "0"}, {"--top-p", "2"}, {"--temp", "x"},
                {"--server.port", "70000"}, {"--batch", "0"}, {"--batch", "big"}}) {
            String[] args = {"serve", bad[0], bad[1]};
            String key = Settings.ALIASES.getOrDefault(bad[0].substring(2), bad[0].substring(2));
            try {
                Settings.load(args).validate();
                check(false, "settings: " + bad[0] + " " + bad[1] + " rejected");
            } catch (IllegalArgumentException e) {
                check(e.getMessage().contains(key.startsWith("sampling.") ? "sampling." : key),
                        "settings: " + bad[0] + " " + bad[1] + " rejected up front: " + e.getMessage());
            }
        }
        // backend.batch: auto (the default, resolved per model by ModelLoader.batch) or a positive number
        check(Settings.load(new String[]{"serve"}).validate().get("backend.batch").equals("auto")
                        && !ModelLoader.explicitBatch(Settings.load(new String[]{"serve"}))
                        && ModelLoader.explicitBatch(Settings.load(new String[]{"serve", "--batch", "128"}).validate()),
                "settings: backend.batch is auto by default and a number when given");
        // L10: --key=value for values that start with --, boolean flags, strict booleans, CLI-only keys in a file
        Settings s = Settings.load(new String[]{"run", "--system=--weird", "--raw", "--prompt", "hi", "--think", "no"}).validate();
        check(s.get("chat.system").equals("--weird") && s.getBool("raw") && s.get("prompt").equals("hi") && !s.getBool("chat.thinking"),
                "settings: --key=--value, bare --raw before another flag, --think no");
        s = Settings.load(new String[]{"run", "--raw", "0", "--low-disk", "FALSE", "--think"}).validate();
        check(!s.getBool("raw") && !s.getBool("model.low-disk") && s.getBool("chat.thinking"),
                "settings: boolean flags consume a following boolean literal (0, FALSE) or stand alone");
        for (String[] args : new String[][]{{"serve", "--chat.thinking", "flase"}, {"serve", "--chat.thinking=flase"},
                {"run", "--raw", "tell me a story"}}) {
            try {
                Settings.load(args).validate();
                check(false, "settings: " + String.join(" ", args) + " rejected");
            } catch (IllegalArgumentException e) {
                check(e.getMessage().contains(args[args.length - 1].replace("--chat.thinking=", "")),
                        "settings: " + String.join(" ", args) + " rejected: " + e.getMessage());
            }
        }
        Path file = Files.createTempFile("jllm-test", ".properties");
        try {
            Files.writeString(file, "prompt=hello\n");
            try {
                Settings.load(new String[]{"run", "--config", file.toString()});
                check(false, "settings: prompt= in the properties file rejected");
            } catch (IllegalArgumentException e) {
                check(e.getMessage().contains("prompt"), "settings: prompt= in the properties file rejected: " + e.getMessage());
            }
            Files.writeString(file, "backend.ctx=512\n");
            check(Settings.load(new String[]{"run", "--config", file.toString()}).validate().getInt("backend.ctx") == 512,
                    "settings: the external file is still applied");
        } finally {
            Files.deleteIfExists(file);
        }
    }

    static void basics(Fixture f) throws Exception {
        Resp r = f.get("/health");
        check(r.status == 200 && r.body.contains("\"ok\""), "GET /health -> 200");
        r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5,\"temperature\":0}");
        Map<String, Object> j = Json.parseObject(r.body);
        check(r.status == 200 && Json.num(Json.map(j, "usage"), "completion_tokens", -1) == 5,
                "non-stream chat: 200 with 5 completion tokens");
        // L18: an empty prompt becomes the tokenizer's EOS token (deterministically)
        r = f.post("/v1/completions", "{\"prompt\":\"\",\"max_tokens\":2,\"temperature\":0}");
        int eos = f.tok.eosId() >= 0 ? f.tok.eosId() : f.tok.tokenId("<|im_end|>");
        check(r.status == 200 && Json.num(Json.map(Json.parseObject(r.body), "usage"), "prompt_tokens", -1) == 1
                && f.backend.firstToken == eos, "empty prompt: one EOS token (" + f.backend.firstToken + ") is fed");
        r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3,\"temperature\":0,\"stream\":true}");
        check(r.status == 200 && r.headers.getOrDefault("transfer-encoding", "").equals("chunked")
                && r.body.endsWith("data: [DONE]\n\n") && r.body.contains("\"finish_reason\":\"length\""),
                "stream chat: chunked SSE ending in [DONE]");
    }

    /** Request validation: each of these must be a 400 with a JSON error, never a 500 or a dropped connection. */
    static void badRequests(Fixture f) throws Exception {
        String msgs = "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]";
        Resp r = f.post("/v1/chat/completions", "{" + msgs + ",\"temperature\":1e400}");
        check(r.status == 400 && r.body.contains("finite"), "temperature 1e400 (infinite): 400");
        r = f.get("/nope");
        check(r.status == 404 && r.headers.get("content-type").equals("application/json") && r.body.contains("\"message\""),
                "unknown route: 404 JSON error");
        // L16: HttpServer's own fallback for a handler that does not answer uses the same JSON error shape
        try (HttpServer bare = new HttpServer("127.0.0.1", 0, (q, s) -> { })) {
            Thread.ofPlatform().daemon().start(bare::serve);
            try (Socket s = new Socket("127.0.0.1", bare.port())) {
                s.getOutputStream().write(request("GET", "/x", null));
                s.getOutputStream().flush();
                r = readResponse(s.getInputStream());
                check(r.status == 404 && r.headers.get("content-type").equals("application/json") && r.body.contains("not found: /x"),
                        "handler without a response: 404 JSON error");
            }
        }
        // L13: integer fields are range-checked before the int cast
        for (String topK : new String[]{"10000000000", "1.5", "-1", "\"5\""}) {
            r = f.post("/v1/chat/completions", "{" + msgs + ",\"top_k\":" + topK + "}");
            check(r.status == 400 && r.body.contains("top_k"), "top_k " + topK + ": 400");
        }
        // L3: a chunk size that overflows when added to what was already received
        String chunked = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                + "Transfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n7fffffffffffffff\r\n";
        r = f.exchange(chunked.getBytes(StandardCharsets.ISO_8859_1));
        check(r.status == 413, "chunk size 0x7fffffffffffffff after 5 bytes: 413 (got " + r.status + ")");
        // L4: one header line beyond the 64 KB limit
        r = f.exchange(request("GET", "/health", null, "X-Long: " + "y".repeat(70 * 1024)));
        check(r.status == 431, "70 KB header line: 431 (got " + r.status + ")");
        // L5: ambiguous framing
        r = f.exchange(request("POST", "/v1/chat/completions", "{}", "Content-Type: application/json", "Content-Length: 2"));
        check(r.status == 400 && r.body.contains("duplicate"), "two Content-Length headers: 400 (got " + r.status + ")");
        r = f.exchange(("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n"
                + "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n").getBytes(StandardCharsets.ISO_8859_1));
        check(r.status == 400 && r.body.contains("duplicate"), "two Transfer-Encoding headers: 400 (got " + r.status + ")");
        // L8: unsupported request features
        r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"},"
                + "{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}]}]}");
        check(r.status == 400 && r.body.contains("cannot see images"), "image content part without a vision projector: 400 (got " + r.status + ")");
        r = f.post("/v1/chat/completions", "{" + msgs + ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}");
        check(r.status == 400 && r.body.contains("not supported"), "tools: 400 (the web console's job)");
        // the OpenAI extras: invalid values are a 400 naming the field
        String[][] bad = {
                {"/v1/completions", "\"logit_bias\":{\"x\":5}", "logit_bias"},
                {"/v1/completions", "\"logit_bias\":{\"1\":101}", "logit_bias"},
                {"/v1/completions", "\"logit_bias\":{\"999999999\":1}", "logit_bias"},
                {"/v1/completions", "\"best_of\":2,\"stream\":true", "best_of"},
                {"/v1/completions", "\"best_of\":2,\"n\":3", "best_of"},
                {"/v1/completions", "\"n\":17", "'n'"},
                {"/v1/completions", "\"n\":0", "'n'"},
                {"/v1/completions", "\"logprobs\":21", "logprobs"},
                {"/v1/completions", "\"echo\":true,\"suffix\":\"x\"", "echo"},
                {"/v1/chat/completions", "\"echo\":true", "echo"},
                {"/v1/chat/completions", "\"top_logprobs\":3", "top_logprobs"},
                {"/v1/chat/completions", "\"logprobs\":true,\"top_logprobs\":21", "top_logprobs"},
                {"/v1/chat/completions", "\"response_format\":{\"type\":\"xml\"}", "response_format"},
                {"/v1/chat/completions", "\"response_format\":{\"type\":\"json_schema\"}", "json_schema"},
                {"/v1/chat/completions", "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"schema\":"
                        + "{\"type\":\"string\",\"pattern\":\"a+\"}}}", "'pattern'"},
                {"/v1/chat/completions", "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"schema\":"
                        + "{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"string\"}]}}}", "anyOf"},
        };
        for (String[] b : bad) {
            String prompt = b[0].contains("chat") ? msgs : "\"prompt\":\"hi\"";
            r = f.post(b[0], "{" + prompt + ",\"max_tokens\":2," + b[1] + "}");
            check(r.status == 400 && r.body.contains(b[2]), b[0] + " " + b[1] + ": 400 naming " + b[2] + " (" + r.body + ")");
        }
    }

    // ------------------------------------------------------------------ OpenAI extras

    /** The log-probability the fake backend's default logits give a token: 1 for {@code token}, 0 elsewhere. */
    static double fakeLogprob(Fixture f, boolean favoured) {
        int v = f.tok.vocabSize();
        double lse = Math.log(Math.E + (v - 1));
        return (favoured ? 1 : 0) - lse;
    }

    static boolean near(Object x, double want) {
        return x instanceof Number n && Math.abs(n.doubleValue() - want) < 1e-4;
    }

    /** logprobs are the log-softmax of the raw logits, in the chat and the completions shape, streamed too. */
    static void logprobs(Fixture f) throws Exception {
        String hello = org.robomiri.jllm.server.OpenAiFeatures.tokenText(f.tok, f.backend.token);
        Resp r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"lp\"}],\"max_tokens\":3,"
                + "\"temperature\":0,\"logprobs\":true,\"top_logprobs\":2}");
        Map<String, Object> choice = (Map<String, Object>) Json.list(Json.parseObject(r.body), "choices").getFirst();
        List<Object> content = Json.list(Json.map(choice, "logprobs"), "content");
        Map<String, Object> first = (Map<String, Object>) content.getFirst();
        List<Object> top = Json.list(first, "top_logprobs");
        check(r.status == 200 && content.size() == 3 && first.get("token").equals(hello)
                        && near(first.get("logprob"), fakeLogprob(f, true)) && top.size() == 2
                        && near(((Map<?, ?>) top.getFirst()).get("logprob"), fakeLogprob(f, true))
                        && near(((Map<?, ?>) top.get(1)).get("logprob"), fakeLogprob(f, false))
                        && Json.list(first, "bytes").size() == hello.getBytes(StandardCharsets.UTF_8).length,
                "chat logprobs: log-softmax of the logits (" + first.get("logprob") + "), top 2, bytes");
        // completions: logprobs: k gives tokens / token_logprobs / top_logprobs / text_offset
        r = f.post("/v1/completions", "{\"prompt\":\"lp\",\"max_tokens\":3,\"temperature\":0,\"logprobs\":1}");
        choice = (Map<String, Object>) Json.list(Json.parseObject(r.body), "choices").getFirst();
        Map<String, Object> lp = Json.map(choice, "logprobs");
        check(r.status == 200 && Json.list(lp, "tokens").equals(List.of(hello, hello, hello))
                        && near(Json.list(lp, "token_logprobs").get(2), fakeLogprob(f, true))
                        && Json.list(lp, "text_offset").equals(List.of(0L, (long) hello.length(), 2L * hello.length()))
                        && ((Map<?, ?>) Json.list(lp, "top_logprobs").getFirst()).containsKey(hello),
                "completion logprobs: tokens, token_logprobs, top_logprobs, text_offset");
        // streamed: every token's entry arrives in some chunk
        r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"lp\"}],\"max_tokens\":4,"
                + "\"temperature\":0,\"logprobs\":true,\"stream\":true}");
        int entries = 0;
        for (Map<String, Object> ev : events(r.body)) {
            for (Object c : Json.list(ev, "choices")) {
                Map<String, Object> l = Json.map(c, "logprobs");
                if (l != null) {
                    entries += Json.list(l, "content").size();
                }
            }
        }
        check(r.status == 200 && entries == 4 && r.body.endsWith("data: [DONE]\n\n"), "streamed logprobs: 4 entries over the chunks");
    }

    /** logit_bias is added to the logits: +50 makes a token win, -100 bans the favourite. */
    static void logitBias(Fixture f) throws Exception {
        int other = f.tok.encode(" world")[0];
        Resp r = f.post("/v1/completions", "{\"prompt\":\"bias\",\"max_tokens\":3,\"temperature\":0,\"logit_bias\":{\""
                + other + "\":50}}");
        String text = (String) ((Map<?, ?>) Json.list(Json.parseObject(r.body), "choices").getFirst()).get("text");
        check(r.status == 200 && text.equals(" world world world"), "logit_bias +50: the biased token wins (" + text + ")");
        r = f.post("/v1/completions", "{\"prompt\":\"bias\",\"max_tokens\":2,\"temperature\":0.8,\"seed\":3,\"logit_bias\":{\""
                + f.backend.token + "\":-100}}");
        text = (String) ((Map<?, ?>) Json.list(Json.parseObject(r.body), "choices").getFirst()).get("text");
        check(r.status == 200 && !text.contains("hello"), "logit_bias -100: the favourite is never sampled");
    }

    /** n choices: indexed, independently sampled (seed + i), the prompt prefilled once; streamed with indices. */
    static void multipleChoices(Fixture f) throws Exception {
        String body = "{\"prompt\":\"many choices please\",\"max_tokens\":4,\"temperature\":1,\"top_k\":0,\"top_p\":1,"
                + "\"seed\":7,\"n\":3}";
        int fed0 = f.backend.fed.get();
        Resp r = f.post("/v1/completions", body);
        int fed = f.backend.fed.get() - fed0;
        Map<String, Object> j = Json.parseObject(r.body);
        List<Object> choices = Json.list(j, "choices");
        long promptTokens = Json.num(Json.map(j, "usage"), "prompt_tokens", -1);
        Set<String> texts = new java.util.HashSet<>();
        for (int i = 0; i < choices.size(); i++) {
            texts.add((String) ((Map<?, ?>) choices.get(i)).get("text"));
            check(Json.num(choices.get(i), "index", -1) == i, "n=3: choice " + i + " has index " + i);
        }
        check(r.status == 200 && choices.size() == 3 && texts.size() == 3
                        && Json.num(Json.map(j, "usage"), "completion_tokens", -1) == 12,
                "n=3: three different samples, usage counts all 12 completion tokens");
        // choice 0 prefills the prompt, choices 1 and 2 recompute only its last token
        check(fed == promptTokens + 3 * 4 - 1, "n=3: prompt prefilled once (" + fed + " tokens fed for a "
                + promptTokens + "-token prompt)");
        Resp again = f.post("/v1/completions", body);
        check(again.status == 200 && Json.list(Json.parseObject(again.body), "choices").equals(choices),
                "n=3 with a seed: the same choices again");
        // choice 0 of n equals the single seeded request
        Resp one = f.post("/v1/completions", body.replace(",\"n\":3", ""));
        check(((Map<?, ?>) Json.list(Json.parseObject(one.body), "choices").getFirst()).get("text")
                .equals(((Map<?, ?>) choices.getFirst()).get("text")), "n=3: choice 0 is the single request's sample");
        // streaming: each choice has its role chunk, deltas and one finish chunk, all tagged with its index
        r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,"
                + "\"temperature\":0,\"n\":2,\"stream\":true,\"stream_options\":{\"include_usage\":true}}");
        int[] roles = new int[2];
        int[] finishes = new int[2];
        StringBuilder[] content = {new StringBuilder(), new StringBuilder()};
        long completion = -1;
        for (Map<String, Object> ev : events(r.body)) {
            for (Object c : Json.list(ev, "choices")) {
                int i = (int) Json.num(c, "index", -1);
                Map<String, Object> delta = Json.map(c, "delta");
                if (delta.containsKey("role")) {
                    roles[i]++;
                }
                if (delta.get("content") instanceof String piece) {
                    content[i].append(piece);
                }
                if (((Map<?, ?>) c).get("finish_reason") != null) {
                    finishes[i]++;
                }
            }
            if (Json.map(ev, "usage") != null) {
                completion = Json.num(Json.map(ev, "usage"), "completion_tokens", -1);
            }
        }
        check(r.status == 200 && roles[0] == 1 && roles[1] == 1 && finishes[0] == 1 && finishes[1] == 1
                        && content[0].toString().equals(" hello hello") && content[1].toString().equals(" hello hello")
                        && completion == 4,
                "stream n=2: per-index role, content and finish chunks; usage sums both choices");
    }

    /** best_of: the n candidates with the highest cumulative logprob are returned. */
    static void bestOf(Fixture f) throws Exception {
        f.backend.logitsFn = Fixture::noisyLogits;
        try {
            String base = "{\"prompt\":\"pick the best\",\"max_tokens\":3,\"temperature\":1,\"top_k\":0,\"seed\":11";
            Resp all = f.post("/v1/completions", base + ",\"n\":4,\"logprobs\":0}");
            List<Object> choices = Json.list(Json.parseObject(all.body), "choices");
            String best = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (Object c : choices) {
                double sum = 0;
                for (Object x : Json.list(Json.map(c, "logprobs"), "token_logprobs")) {
                    sum += ((Number) x).doubleValue();
                }
                if (sum > bestScore) {
                    bestScore = sum;
                    best = (String) ((Map<?, ?>) c).get("text");
                }
            }
            Resp r = f.post("/v1/completions", base + ",\"best_of\":4}");
            Map<String, Object> j = Json.parseObject(r.body);
            List<Object> got = Json.list(j, "choices");
            check(r.status == 200 && got.size() == 1 && ((Map<?, ?>) got.getFirst()).get("text").equals(best)
                            && !((Map<?, ?>) got.getFirst()).containsKey("logprobs")
                            && Json.num(Json.map(j, "usage"), "completion_tokens", -1) == 12,
                    "best_of=4: the most likely of the 4 candidates (" + best + "), usage counts all 12 tokens");
        } finally {
            f.backend.logitsFn = null;
        }
    }

    /** echo prepends the prompt (with its logprobs); suffix builds a fill-in-the-middle prompt. */
    static void echoAndSuffix(Fixture f) throws Exception {
        String prompt = "echo this";
        int n = f.tok.encode(prompt).length;
        Resp r = f.post("/v1/completions", "{\"prompt\":\"" + prompt + "\",\"max_tokens\":2,\"temperature\":0,\"echo\":true,\"logprobs\":1}");
        Map<String, Object> c = (Map<String, Object>) Json.list(Json.parseObject(r.body), "choices").getFirst();
        Map<String, Object> lp = Json.map(c, "logprobs");
        List<Object> tl = Json.list(lp, "token_logprobs");
        List<Object> offsets = Json.list(lp, "text_offset");
        check(r.status == 200 && c.get("text").equals(prompt + " hello hello") && tl.size() == n + 2 && tl.getFirst() == null
                        && near(tl.get(1), f.backend.token == f.tok.encode(prompt)[1] ? fakeLogprob(f, true) : fakeLogprob(f, false))
                        && ((Number) offsets.get(n)).intValue() == prompt.length(),
                "echo + logprobs: prompt text and its token logprobs (first null), then the completion");
        r = f.post("/v1/completions", "{\"prompt\":\"" + prompt + "\",\"max_tokens\":0,\"echo\":true,\"logprobs\":0}");
        c = (Map<String, Object>) Json.list(Json.parseObject(r.body), "choices").getFirst();
        check(r.status == 200 && c.get("text").equals(prompt) && Json.list(Json.map(c, "logprobs"), "tokens").size() == n,
                "echo with max_tokens 0: scores the prompt only");
        r = f.post("/v1/completions", "{\"prompt\":\"" + prompt + "\",\"max_tokens\":2,\"temperature\":0,\"echo\":true,\"stream\":true}");
        StringBuilder text = new StringBuilder();
        for (Map<String, Object> ev : events(r.body)) {
            for (Object ch : Json.list(ev, "choices")) {
                text.append(((Map<?, ?>) ch).get("text"));
            }
        }
        check(r.status == 200 && text.toString().equals(prompt + " hello hello"), "streamed echo: the prompt comes first");
        // suffix: <|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>
        r = f.post("/v1/completions", "{\"prompt\":\"def f(\",\"suffix\":\"\\n    return x\",\"max_tokens\":1,\"temperature\":0}");
        check(r.status == 200 && f.backend.firstToken == f.tok.tokenId("<|fim_prefix|>"), "suffix: a fill-in-the-middle prompt");
        Map<String, Object> json = Json.parseObject(Files.readString(modelDir.resolve("tokenizer.json")));
        Json.list(json, "added_tokens").removeIf(t -> Json.str(t, "content", "").startsWith("<|fim_"));
        Tokenizer plain = Tokenizer.fromMaps(json, Map.of()); // the same vocabulary without the FIM tokens
        try {
            org.robomiri.jllm.server.OpenAiFeatures.fimPrompt(plain, "a", "b");
            check(false, "suffix without FIM tokens rejected");
        } catch (IllegalArgumentException e) {
            check(e.getMessage().contains("fim_prefix"), "suffix without FIM tokens: " + e.getMessage());
        }
    }

    /** response_format: the output is valid JSON matching the schema, also with logits that favour other text. */
    static void jsonOutput(Fixture f) throws Exception {
        f.backend.logitsFn = Fixture::noisyLogits;
        try {
            String schema = "{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"},\"n\":{\"type\":\"integer\"},"
                    + "\"color\":{\"enum\":[\"red\",\"green\"]}},\"required\":[\"ok\",\"n\",\"color\"],\"additionalProperties\":false}";
            String body = "{\"messages\":[{\"role\":\"user\",\"content\":\"json please\"}],\"max_tokens\":300,\"temperature\":0,"
                    + "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"x\",\"schema\":" + schema + "}}}";
            long t0 = System.nanoTime();
            Resp r = f.post("/v1/chat/completions", body);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            Map<String, Object> c = (Map<String, Object>) Json.list(Json.parseObject(r.body), "choices").getFirst();
            String content = (String) Json.map(c, "message").get("content");
            Map<String, Object> obj = Json.parseObject(content);
            check(r.status == 200 && "stop".equals(c.get("finish_reason")) && obj.keySet().equals(Set.of("ok", "n", "color"))
                            && obj.get("ok") instanceof Boolean && obj.get("n") instanceof Long
                            && List.of("red", "green").contains(obj.get("color")),
                    "json_schema: " + Json.write(obj) + " (" + ms + " ms)");
            // streamed: the same text
            r = f.post("/v1/chat/completions", body.replace("\"temperature\":0,", "\"temperature\":0,\"stream\":true,"));
            StringBuilder streamed = new StringBuilder();
            for (Map<String, Object> ev : events(r.body)) {
                for (Object ch : Json.list(ev, "choices")) {
                    if (Json.map(ch, "delta").get("content") instanceof String piece) {
                        streamed.append(piece);
                    }
                }
            }
            check(streamed.toString().equals(content), "json_schema streamed: the same output");
            // json_object with sampling, n=2: each output is an object (or a valid prefix of one when cut by max_tokens)
            r = f.post("/v1/completions", "{\"prompt\":\"obj\",\"max_tokens\":40,\"temperature\":0.7,\"seed\":5,\"n\":2,"
                    + "\"response_format\":{\"type\":\"json_object\"}}");
            for (Object ch : Json.list(Json.parseObject(r.body), "choices")) {
                String text = (String) ((Map<?, ?>) ch).get("text");
                boolean complete = "stop".equals(((Map<?, ?>) ch).get("finish_reason"));
                org.robomiri.jllm.sampling.JsonConstraint jc = new org.robomiri.jllm.sampling.JsonConstraint(
                        org.robomiri.jllm.sampling.JsonConstraint.Schema.anyObject(), new byte[0][]);
                check(r.status == 200 && text.strip().startsWith("{") && jc.allowsText(text)
                                && (!complete || Json.parse(text) instanceof Map),
                        "json_object: " + (complete ? "a complete object" : "a valid prefix") + ": " + text.strip());
            }
        } finally {
            f.backend.logitsFn = null;
        }
        // with the fake's default logits (" hello" first, everything else tied), greedy picks the first valid token
        Resp r = f.post("/v1/completions", "{\"prompt\":\"bool\",\"max_tokens\":10,\"temperature\":0,"
                + "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"schema\":{\"type\":\"boolean\"}}}}");
        Map<String, Object> c = (Map<String, Object>) Json.list(Json.parseObject(r.body), "choices").getFirst();
        check(r.status == 200 && List.of("true", "false").contains(((String) c.get("text")).strip()) && "stop".equals(c.get("finish_reason")),
                "json_schema boolean: " + c.get("text"));
    }

    /** The JSON automaton on its own: what each schema keyword admits. */
    static void jsonConstraintUnit() {
        String schema = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\",\"maxLength\":3},"
                + "\"age\":{\"type\":\"integer\"},\"tags\":{\"type\":\"array\",\"items\":{\"enum\":[\"a\",\"b\"]},\"maxItems\":2},"
                + "\"next\":{\"anyOf\":[{\"$ref\":\"#\"},{\"type\":\"null\"}]}},\"required\":[\"name\"],\"additionalProperties\":false}";
        org.robomiri.jllm.sampling.JsonConstraint.Schema s = org.robomiri.jllm.sampling.JsonConstraint.Schema.compile(Json.parseObject(schema));
        String[][] cases = {
                {"{\"name\":\"bob\",\"age\":42,\"tags\":[\"a\",\"b\"],\"next\":null}", "true"},
                {"{\"name\":\"x\",\"next\":{\"name\":\"y\",\"next\":null}}", "true"},
                {" {\n  \"name\" : \"\\u00e9\\n\"\n}", "true"},
                {"{\"name\":\"bobby\"}", "false"},      // maxLength
                {"{\"age\":1}", "false"},                // required name missing at '}'
                {"{\"age\":1.5", "false"},               // integer
                {"{\"name\":\"x\",\"other\":1}", "false"}, // additionalProperties false
                {"{\"name\":\"x\",\"name\":\"y\"}", "false"}, // duplicate key
                {"{\"tags\":[\"c\"]}", "false"},        // enum
                {"{\"tags\":[\"a\",\"a\",\"a\"]}", "false"}, // maxItems
                {"{\"next\":5}", "false"},               // anyOf: object or null
                {"{\"name\":\"\\x\"}", "false"},       // bad escape
                {"{\"name\":\"x\"} ", "false"},         // nothing after the value
                {"{\"name\":\"x\",}", "false"},         // trailing comma
                {"{\"name\":\"a\tb\"}", "false"},        // raw control character
        };
        for (String[] c : cases) {
            org.robomiri.jllm.sampling.JsonConstraint jc = new org.robomiri.jllm.sampling.JsonConstraint(s, new byte[0][]);
            boolean ok = c[1].equals("true") ? jc.completesWith(c[0]) : jc.allowsText(c[0]);
            check(ok == c[1].equals("true"), "json schema automaton: " + c[0].replace("\n", "\\n") + " -> " + ok);
        }
        org.robomiri.jllm.sampling.JsonConstraint num = new org.robomiri.jllm.sampling.JsonConstraint(
                org.robomiri.jllm.sampling.JsonConstraint.Schema.compile(Json.parseObject("{\"type\":\"number\"}")), new byte[0][]);
        check(num.completesWith("-12.5e+3") && num.completesWith("0") && !num.allowsText("012") && num.allowsText("1.")
                        && !num.completesWith("1.") && !num.allowsText("1.5.") && !num.allowsText("--1"),
                "json schema automaton: numbers");
        String deep = "{\"type\":\"array\",\"items\":{\"type\":\"array\"},\"minItems\":1}";
        org.robomiri.jllm.sampling.JsonConstraint arr = new org.robomiri.jllm.sampling.JsonConstraint(
                org.robomiri.jllm.sampling.JsonConstraint.Schema.compile(Json.parseObject(deep)), new byte[0][]);
        check(arr.completesWith("[[1,[2]],[]]") && !arr.allowsText("[]") && !arr.allowsText("[1]")
                        && !arr.allowsText("[" + " ".repeat(65) + "[]]") && !arr.allowsText("[\n\n[]]")
                        && arr.completesWith("[\n  [\n    1\n  ]\n]"),
                "json schema automaton: minItems, nested items, whitespace cap (one newline per gap)");
    }

    /** The data events of an SSE body, parsed (the [DONE] marker skipped). */
    static List<Map<String, Object>> events(String body) {
        List<Map<String, Object>> out = new java.util.ArrayList<>();
        for (String e : body.split("\n\n")) {
            if (e.startsWith("data: {")) {
                out.add(Json.parseObject(e.substring(6)));
            }
        }
        return out;
    }

    /** Two requests on one connection (the second must not be disturbed by the first's bookkeeping). */
    static void keepAlive(Fixture f) throws Exception {
        try (Socket s = f.connect()) {
            OutputStream out = s.getOutputStream();
            String body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"temperature\":0}";
            out.write(request("POST", "/v1/chat/completions", body, "Content-Type: application/json"));
            out.flush();
            Resp a = readResponse(s.getInputStream());
            out.write(request("GET", "/health", null));
            out.flush();
            Resp b = readResponse(s.getInputStream());
            check(a.status == 200 && b.status == 200 && a.headers.get("connection").equals("keep-alive"),
                    "keep-alive: second request on the same connection is served");
        }
    }

    /**
     * H1: a streaming client that stops reading must not hold the engine. The stream is queued for at most
     * the write timeout, then the connection is dropped and the generation cancelled; a second client is served
     * well within the engine's queue timeout.
     */
    static void stalledStreamClient(Fixture f) throws Exception {
        int before = f.backend.forwards.get();
        Socket stalled = new Socket();
        stalled.setReceiveBufferSize(8 * 1024); // small window: the server's buffers fill up sooner
        stalled.connect(new InetSocketAddress("127.0.0.1", f.server.port()));
        String body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1000000,"
                + "\"temperature\":0,\"stream\":true}";
        stalled.getOutputStream().write(request("POST", "/v1/chat/completions", body, "Content-Type: application/json"));
        stalled.getOutputStream().flush();
        InputStream in = stalled.getInputStream();
        readHeaders(in); // the stream has started; from here on the client never reads again
        long t0 = System.nanoTime();
        Resp r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"next\"}],\"max_tokens\":1,\"temperature\":0}");
        long ms = (System.nanoTime() - t0) / 1_000_000;
        check(r.status == 200, "stalled stream client: the next request is served (status " + r.status + ")");
        check(ms < QUEUE_TIMEOUT, "stalled stream client: served in " + ms + " ms (< engine queue timeout)");
        int afterFirst = f.backend.forwards.get();
        Thread.sleep(200);
        check(f.backend.forwards.get() == afterFirst, "stalled stream client: its generation was cancelled");
        check(afterFirst - before < 1_000_000, "stalled stream client: generation did not run to max_tokens");
        boolean closedByServer;
        try {
            byte[] buf = new byte[1 << 16];
            while (in.read(buf) >= 0) { // drain whatever was queued before the drop
                // keep reading
            }
            closedByServer = true;
        } catch (IOException e) {
            closedByServer = true; // reset by peer is also a close
        }
        check(closedByServer, "stalled stream client: server closed the connection");
        stalled.close();
    }

    /**
     * M1: a client that closes its connection mid-generation is noticed by the connection probe, for a
     * non-streaming request (nothing is written until the end) as well as a streaming one (before any write
     * fails), and the engine stops generating within a few tokens.
     */
    static void disconnectedClient(Fixture f, boolean stream) throws Exception {
        String what = (stream ? "stream" : "non-stream") + " client disconnect";
        f.backend.delayMs = 2; // ~2 ms per token: 100k tokens would take minutes
        try {
            Socket s = f.connect();
            String body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":100000,"
                    + "\"temperature\":0,\"stream\":" + stream + "}";
            s.getOutputStream().write(request("POST", "/v1/chat/completions", body, "Content-Type: application/json"));
            s.getOutputStream().flush();
            if (stream) {
                readHeaders(s.getInputStream());
            } else {
                Thread.sleep(150); // let the generation start
            }
            int atClose = f.backend.forwards.get();
            s.close();
            long t0 = System.nanoTime();
            Resp r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"next\"}],\"max_tokens\":1,\"temperature\":0}");
            long ms = (System.nanoTime() - t0) / 1_000_000;
            check(r.status == 200 && ms < 1500, what + ": the next request is served in " + ms + " ms");
            int settled = f.backend.forwards.get();
            Thread.sleep(200);
            check(f.backend.forwards.get() == settled, what + ": generation stopped");
            check(settled - atClose < 50, what + ": stopped within a few tokens (" + (settled - atClose) + " forwards after the close)");
        } finally {
            f.backend.delayMs = 0;
        }
    }

    /**
     * M2: a client that keeps a request alive by sending one header byte at a time (each within the per-read
     * timeout) gets a 408 once the request as a whole has taken longer than the read timeout; and a keep-alive
     * connection that goes quiet after a response is closed silently.
     */
    static void slowRequest(Fixture f) throws Exception {
        try (Socket s = f.connect()) {
            OutputStream out = s.getOutputStream();
            out.write("GET /health HTTP/1.1\r\nHost: x\r\nX-Slow: ".getBytes(StandardCharsets.ISO_8859_1));
            out.flush();
            long t0 = System.nanoTime();
            try {
                for (int i = 0; i < 10; i++) { // 3 s of trickling, one byte every 300 ms < READ_TIMEOUT
                    Thread.sleep(300);
                    out.write('x');
                    out.flush();
                }
            } catch (IOException e) {
                // the server has already answered and closed: expected
            }
            Resp r = readHeaders(s.getInputStream());
            long ms = (System.nanoTime() - t0) / 1_000_000;
            check(r.status == 408, "slow request: 408 after " + ms + " ms (per-read timeout never fired)");
        }
        try (Socket s = f.connect()) {
            s.getOutputStream().write(request("GET", "/health", null));
            s.getOutputStream().flush();
            check(readResponse(s.getInputStream()).status == 200, "idle keep-alive: first request served");
            long t0 = System.nanoTime();
            int c = s.getInputStream().read(); // nothing sent: the server closes after the idle timeout
            long ms = (System.nanoTime() - t0) / 1_000_000;
            check(c == -1 && ms >= READ_TIMEOUT - 50 && ms < 3 * READ_TIMEOUT,
                    "idle keep-alive: closed silently after " + ms + " ms");
        }
    }

    /** M4: a backend failure after the stream has started ends it with an in-band error event and no [DONE]. */
    static void midStreamFailure(Fixture f) throws Exception {
        f.backend.failAfter = f.backend.forwards.get() + 4; // prompt, then a few tokens, then the failure
        System.err.println("(a stack trace of the injected failure follows: expected)");
        try {
            Resp r = f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":50,\"temperature\":0,\"stream\":true}");
            String[] events = r.body.split("\n\n");
            String last = events[events.length - 1];
            check(r.status == 200 && last.startsWith("data: {\"error\":") && last.contains("\"server_error\"")
                    && !r.body.contains("[DONE]") && r.body.contains("\"content\":\" hello\""),
                    "mid-stream failure: tokens, then an error event, then a proper end without [DONE]");
        } finally {
            f.backend.failAfter = -1;
        }
        check(f.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"temperature\":0}").status == 200,
                "mid-stream failure: the server keeps serving afterwards");
    }

    /**
     * M3: for a family with a prompt BOS token (here a llama config, ctx 64) the context check must count the
     * BOS the engine prepends: a raw prompt of ctx-1 tokens is a 400, not a 500 from the engine.
     */
    static void bosContextCheck(Fixture f) throws Exception {
        int ctx = f.backend.contextLength();
        Resp r = f.post("/v1/completions", "{\"prompt\":" + Json.write(promptOf(f.tok, ctx - 1)) + ",\"max_tokens\":1,\"temperature\":0}");
        check(r.status == 400 && r.body.contains("context_length_exceeded"),
                "prompt of ctx-1 tokens plus BOS: 400 context_length_exceeded (got " + r.status + ")");
        r = f.post("/v1/completions", "{\"prompt\":" + Json.write(promptOf(f.tok, ctx - 2)) + ",\"max_tokens\":5,\"temperature\":0}");
        Map<String, Object> usage = Json.map(Json.parseObject(r.body), "usage");
        check(r.status == 200 && Json.num(usage, "prompt_tokens", -1) == ctx - 1 && Json.num(usage, "completion_tokens", -1) == 1,
                "prompt of ctx-2 tokens plus BOS: served, one token of room");
    }

    /** Text that the tokenizer encodes to exactly {@code n} tokens. */
    static String promptOf(Tokenizer tok, int n) {
        StringBuilder sb = new StringBuilder("x");
        while (tok.encode(sb.toString()).length < n) {
            sb.append(" x");
        }
        check(tok.encode(sb.toString()).length == n, "built a " + n + "-token prompt");
        return sb.toString();
    }

    // ------------------------------------------------------------------ fixture: engine on a fake backend

    /** A backend that produces the same token every step, instantly (or after {@code delayMs}). */
    static final class FakeBackend implements Backend {
        final int ctx;
        final int token;
        final AtomicInteger forwards = new AtomicInteger();
        final AtomicInteger fed = new AtomicInteger(); // tokens run through forward(), prefill and decode
        /** Fills the logits for position {@code pos}, instead of the default (" hello" first); null = default. */
        volatile java.util.function.ObjIntConsumer<float[]> logitsFn;
        volatile long delayMs;
        volatile int failAfter = -1; // throw from forward() once this many calls happened (-1 = never)
        volatile int firstToken = -1; // the token at position 0 of the most recent prefill

        FakeBackend(int ctx, int token) {
            this.ctx = ctx;
            this.token = token;
        }

        @Override
        public String name() {
            return "fake";
        }

        @Override
        public int contextLength() {
            return ctx;
        }

        @Override
        public int maxBatch() {
            return 64;
        }

        @Override
        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            int calls = forwards.incrementAndGet();
            fed.addAndGet(n);
            if (pos == 0) {
                firstToken = tokens[off];
            }
            if (failAfter >= 0 && calls > failAfter) {
                throw new IllegalStateException("fake backend failure");
            }
            if (delayMs > 0) {
                try {
                    Thread.sleep(delayMs);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            if (logits != null) {
                var fn = logitsFn;
                if (fn != null) {
                    fn.accept(logits, pos + n - 1);
                } else {
                    Arrays.fill(logits, 0f);
                    logits[token] = 1f;
                }
            }
        }

        @Override
        public void close() {
        }
    }

    static final class Fixture implements AutoCloseable {
        final Tokenizer tok;
        final FakeBackend backend;
        final Engine engine;
        final ApiServer server;
        final Thread serveThread;
        final Path tmp;

        Fixture(Tokenizer tok, String modelType, int ctx) throws Exception {
            this.tok = tok;
            tmp = Files.createTempDirectory("jllm-server-test");
            String bos = modelType.equals("qwen3") ? "" : ",\"bos_token_id\":" + tok.tokenId("<|im_start|>");
            Files.writeString(tmp.resolve("config.json"), "{\"model_type\":\"" + modelType + "\",\"hidden_size\":64,"
                    + "\"num_hidden_layers\":1,\"num_attention_heads\":2,\"intermediate_size\":128,\"vocab_size\":" + tok.vocabSize()
                    + ",\"eos_token_id\":" + tok.tokenId("<|im_end|>") + bos + "}");
            Architecture cfg = Architecture.load(tmp);
            backend = new FakeBackend(ctx, tok.encode(" hello")[0]);
            engine = new Engine(backend, tok, cfg, Set.of(tok.tokenId("<|im_end|>")), QUEUE_TIMEOUT,
                    ChatTemplate.forModelType("qwen3"));
            HttpServer.Options http = new HttpServer.Options(READ_TIMEOUT, WRITE_TIMEOUT, 16, 64 * 1024, null);
            server = new ApiServer(engine, "test-model", "127.0.0.1", 0,
                    new ApiServer.Options(SamplingParams.defaults(), false, 0, 0, http));
            serveThread = Thread.ofPlatform().daemon().start(server::serve);
        }

        /** Deterministic pseudo-random logits in [0, 8) per position: a "model" with opinions that are not JSON. */
        static void noisyLogits(float[] logits, int pos) {
            long h = 0x9E3779B97F4A7C15L * (pos + 1);
            for (int i = 0; i < logits.length; i++) {
                long x = h + i * 0xBF58476D1CE4E5B9L;
                x = (x ^ (x >>> 31)) * 0x94D049BB133111EBL;
                x ^= x >>> 29;
                logits[i] = (x >>> 40) * (8f / (1 << 24));
            }
        }

        Socket connect() throws IOException {
            Socket s = new Socket("127.0.0.1", server.port());
            s.setSoTimeout(30_000);
            return s;
        }

        Resp get(String path) throws IOException {
            return exchange(request("GET", path, null));
        }

        Resp post(String path, String json) throws IOException {
            return exchange(request("POST", path, json, "Content-Type: application/json"));
        }

        Resp exchange(byte[] raw) throws IOException {
            try (Socket s = connect()) {
                s.getOutputStream().write(raw);
                s.getOutputStream().flush();
                return readResponse(s.getInputStream());
            }
        }

        @Override
        public void close() throws Exception {
            server.close();
            engine.close();
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    // ------------------------------------------------------------------ a minimal HTTP/1.1 client

    record Resp(int status, Map<String, String> headers, String body) {
    }

    static byte[] request(String method, String path, String body, String... headers) {
        StringBuilder sb = new StringBuilder();
        sb.append(method).append(' ').append(path).append(" HTTP/1.1\r\nHost: localhost\r\n");
        for (String h : headers) {
            sb.append(h).append("\r\n");
        }
        byte[] b = body == null ? new byte[0] : body.getBytes(StandardCharsets.UTF_8);
        if (body != null) {
            sb.append("Content-Length: ").append(b.length).append("\r\n");
        }
        sb.append("\r\n");
        byte[] head = sb.toString().getBytes(StandardCharsets.ISO_8859_1);
        byte[] all = Arrays.copyOf(head, head.length + b.length);
        System.arraycopy(b, 0, all, head.length, b.length);
        return all;
    }

    /** Reads the status line and headers; returns the status with the lower-cased header names. */
    static Resp readHeaders(InputStream in) throws IOException {
        String status = readLine(in);
        if (status == null) {
            throw new IOException("connection closed before the status line");
        }
        int code = Integer.parseInt(status.split(" ")[1]);
        Map<String, String> headers = new LinkedHashMap<>();
        String line;
        while ((line = readLine(in)) != null && !line.isEmpty()) {
            int c = line.indexOf(':');
            headers.put(line.substring(0, c).trim().toLowerCase(), line.substring(c + 1).trim());
        }
        return new Resp(code, headers, "");
    }

    static Resp readResponse(InputStream in) throws IOException {
        Resp h = readHeaders(in);
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        if ("chunked".equals(h.headers.get("transfer-encoding"))) {
            while (true) {
                String size = readLine(in);
                int n = Integer.parseInt(size.trim(), 16);
                if (n == 0) {
                    readLine(in);
                    break;
                }
                body.write(in.readNBytes(n));
                readLine(in);
            }
        } else {
            int n = Integer.parseInt(h.headers.getOrDefault("content-length", "0"));
            body.write(in.readNBytes(n));
        }
        return new Resp(h.status, h.headers, body.toString(StandardCharsets.UTF_8));
    }

    static String readLine(InputStream in) throws IOException {
        StringBuilder sb = new StringBuilder();
        int c;
        while ((c = in.read()) >= 0) {
            if (c == '\n') {
                int n = sb.length();
                if (n > 0 && sb.charAt(n - 1) == '\r') {
                    sb.setLength(n - 1);
                }
                return sb.toString();
            }
            sb.append((char) c);
        }
        return sb.isEmpty() ? null : sb.toString();
    }

    static void check(boolean ok, String what) {
        if (!ok) {
            throw new AssertionError(what);
        }
        System.out.println("ok  " + what);
    }
}
