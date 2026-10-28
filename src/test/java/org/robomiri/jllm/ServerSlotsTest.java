package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.BatchGroup;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.SlotPool;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.server.ApiServer;
import org.robomiri.jllm.server.HttpServer;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * {@code ./jllm serve} with {@code server.slots} (docs/10 §10.9c) on the real model, CPU backend, end to end over HTTP: three
 * users sending chat completions at the same time get the greedy answers each gets alone, their requests are in flight
 * together and their decode steps share forward passes, a system prompt one user's request computed is found in the paged
 * cache by another user's request on another slot, and {@code /health} reports the slots; a user who finds every slot busy
 * gets the 503 of a busy server after {@code server.queue-timeout-ms}.
 *
 * <p>usage: ServerSlotsTest &lt;model dir&gt; slots|mixed|bench... (models/Qwen_Qwen3-0.6B). {@code slots}: the slots
 * scheduler on the fast float path (batched steps take the multi-row kernels, the arg max is the lone step's as in
 * BatchDecodeTest). {@code mixed}: {@code engine.scheduler=mixed}, run with {@code -Djllm.cpu.deterministic=true} (a mixed
 * pass also carries prompt chunks, whose multi-row rounding can move a near-tie on the CPU float path, docs/25 §25.2.9).
 * {@code bench [--setting value ...]} (last): generated tokens per second and time to the first token of 1, 2, 4 and 8
 * concurrent users on 4 slots against 1 and 4 users on one slot.
 */
public final class ServerSlotsTest {

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        for (int i = 1; i < args.length; i++) {
            if (args[i].equals("bench")) {
                bench(dir, java.util.Arrays.copyOfRange(args, i + 1, args.length)); // the rest: settings for the bench
                break;
            }
            run(dir, args[i], 3);
            overload(dir, args[i]);
        }
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("server slots test passed");
    }

    /** The test's settings: float activations, so a batched step is the single-token arithmetic up to summation order. */
    static Settings settings(Path dir, String scheduler, int slots, String... extra) throws java.io.IOException {
        List<String> a = new ArrayList<>(List.of("--backend", "cpu", "--model.dir", dir.toAbsolutePath().getParent().toString(),
                "--backend.ctx", "2048", "--engine.speculative", "off", "--backend.cpu.int8-activations", "false",
                "--engine.scheduler", scheduler, "--server.slots", String.valueOf(slots)));
        a.addAll(List.of(extra));
        return Settings.load(a.toArray(new String[0])).validate();
    }

    /** A server like Main.serve builds it: the loaded engine, server.slots - 1 extra ones, one SlotPool. */
    record Served(ModelLoader.Loaded loaded, List<Engine> extra, SlotPool pool, ApiServer server) implements AutoCloseable {
        static Served start(Path dir, Settings st) throws Exception {
            int n = st.getInt("server.slots");
            ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString(), n);
            List<Engine> engines = new ArrayList<>(List.of(l.engine()));
            List<Engine> extra = new ArrayList<>();
            for (int i = 1; i < n; i++) {
                Engine e = ModelLoader.extraEngine(st, l);
                extra.add(e);
                engines.add(e);
            }
            SlotPool pool = new SlotPool(engines, SlotPool.mixedConfig(st), null, "test");
            ApiServer server = ApiServer.onSlots(pool, "test", "127.0.0.1", 0, new ApiServer.Options(st.sampling(), false, 0, 0,
                    HttpServer.Options.defaults()));
            Thread.ofPlatform().daemon().start(server::serve);
            return new Served(l, extra, pool, server);
        }

        @Override
        public void close() throws Exception {
            server.close();
            pool.stop();
            for (Engine e : extra) {
                e.close();
            }
            loaded.close();
        }
    }

    static final HttpClient HTTP = HttpClient.newHttpClient();

    /** A greedy chat completion: the response object, and the wall-clock interval of the request. */
    record Answer(Map<String, Object> body, long startNanos, long endNanos) {
        @SuppressWarnings("unchecked")
        String text() {
            List<Object> choices = Json.list(body, "choices");
            return Json.str(Json.map((Map<String, Object>) choices.getFirst(), "message"), "content", "");
        }

        long cached() {
            return Json.num(Json.map(body, "usage"), "cached_tokens", -1);
        }
    }

    static Answer chat(int port, String system, String user, int maxTokens) throws Exception {
        List<Object> messages = new ArrayList<>();
        if (system != null) {
            messages.add(Json.obj("role", "system", "content", system));
        }
        messages.add(Json.obj("role", "user", "content", user + " /no_think"));
        String body = Json.write(Json.obj("model", "test", "messages", messages, "temperature", 0, "max_tokens", maxTokens));
        long t0 = System.nanoTime();
        HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + "/v1/chat/completions"))
                .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(body)).build(),
                HttpResponse.BodyHandlers.ofString());
        long t1 = System.nanoTime();
        if (r.statusCode() != 200) {
            throw new IllegalStateException("HTTP " + r.statusCode() + ": " + r.body());
        }
        return new Answer(Json.parseObject(r.body()), t0, t1);
    }

    static long[] statsOf(Engine e) {
        Backend b = e.backend();
        return b instanceof BatchGroup.Member m ? m.group().stats() : new long[2];
    }

    static final String[] QUESTIONS = {"What is the capital of France?", "Write a haiku about autumn leaves.",
            "Explain what a hash map is in one sentence."};

    static void run(Path dir, String scheduler, int slots) throws Exception {
        System.out.println("# scheduler=" + scheduler + ", " + slots + " slots");
        try (Served sv = Served.start(dir, settings(dir, scheduler, slots))) {
            int port = sv.server().port();
            Engine first = sv.pool().first();
            check(first.backend() instanceof BatchGroup.Member, scheduler + ": the slots' backends are members of a batch group");
            check(first.backend().kvSequence() != null, scheduler + ": " + slots + " slots page their KV cache (backend.kv-paged=auto)");
            check(scheduler.equals("mixed") == (sv.pool().admission() != null), scheduler + ": the mixed scheduler's queue exists"
                    + " exactly under engine.scheduler=mixed");
            Map<String, Object> health = Json.parseObject(HTTP.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port
                    + "/health")).build(), HttpResponse.BodyHandlers.ofString()).body());
            check(Json.num(health, "slots", 0) == slots && Json.num(health, "busy_slots", -1) == 0,
                    scheduler + ": /health reports " + slots + " slots, none busy: " + health);

            // alone: one user after another
            String[] alone = new String[QUESTIONS.length];
            for (int i = 0; i < QUESTIONS.length; i++) {
                alone[i] = chat(port, null, QUESTIONS[i], 24).text();
            }
            // together: three users at once
            long[] before = statsOf(first);
            ExecutorService ex = Executors.newFixedThreadPool(QUESTIONS.length);
            CountDownLatch go = new CountDownLatch(1);
            List<Future<Answer>> fs = new ArrayList<>();
            for (String q : QUESTIONS) {
                fs.add(ex.submit(() -> {
                    go.await();
                    return chat(port, null, q, 24);
                }));
            }
            go.countDown();
            Answer[] together = new Answer[QUESTIONS.length];
            for (int i = 0; i < together.length; i++) {
                together[i] = fs.get(i).get();
            }
            ex.shutdown();
            long[] after = statsOf(first);
            for (int i = 0; i < QUESTIONS.length; i++) {
                check(alone[i].equals(together[i].text()), scheduler + ": user " + i + " gets the answer it gets alone: "
                        + abbreviate(together[i].text()));
            }
            long lastStart = 0;
            long firstEnd = Long.MAX_VALUE;
            for (Answer a : together) {
                lastStart = Math.max(lastStart, a.startNanos());
                firstEnd = Math.min(firstEnd, a.endNanos());
            }
            check(lastStart < firstEnd, scheduler + ": the three requests were in flight at the same time");
            check(after[1] - before[1] > 0, scheduler + ": decode steps of different users shared forward passes ("
                    + (after[1] - before[1]) + " steps in " + (after[0] - before[0]) + " passes)");

            // a long system prompt: the second user's request finds the first one's blocks, on whichever slot it runs
            StringBuilder sys = new StringBuilder("You are a careful assistant. Follow these rules.\n");
            for (int i = 1; i <= 40; i++) {
                sys.append("Rule ").append(i).append(": answer briefly and never invent facts about item ").append(i * 7).append(".\n");
            }
            Answer a1 = chat(port, sys.toString(), "Name a primary color.", 8);
            Answer a2 = chat(port, sys.toString(), "Name a planet.", 8);
            long sysTokens = Json.num(Json.map(a1.body(), "usage"), "prompt_tokens", 0);
            check(a2.cached() > sysTokens / 2, scheduler + ": the second user reuses the shared system prompt from the paged cache ("
                    + a2.cached() + " of " + Json.num(Json.map(a2.body(), "usage"), "prompt_tokens", 0) + " prompt tokens cached)");
            check(Json.num(Json.map(Json.map(a2.body(), "usage"), "prompt_tokens_details"), "cached_tokens", -1) == a2.cached(),
                    scheduler + ": usage.prompt_tokens_details.cached_tokens (where OpenAI clients look) reports the same count");
        }
    }

    /**
     * Two slots busy with long answers and a third user: it waits {@code server.queue-timeout-ms} (300 ms here) at the
     * slot's engine (slots scheduler) or in the model's queue (mixed) and gets the 503 of a busy server.
     */
    static void overload(Path dir, String scheduler) throws Exception {
        try (Served sv = Served.start(dir, settings(dir, scheduler, 2, "--server.queue-timeout-ms", "300"))) {
            int port = sv.server().port();
            ExecutorService ex = Executors.newFixedThreadPool(2);
            List<Future<Answer>> busy = new ArrayList<>();
            for (int i = 0; i < 2; i++) {
                int k = i;
                busy.add(ex.submit(() -> chat(port, null, "Tell me a long story about robot number " + (k + 1) + ".", 96)));
            }
            long until = System.nanoTime() + 30_000_000_000L;
            while (sv.pool().busy() < 2 && System.nanoTime() < until) {
                Thread.sleep(5);
            }
            Thread.sleep(200); // both are generating
            String body = Json.write(Json.obj("model", "test", "messages", List.of(Json.obj("role", "user", "content",
                    "Hello /no_think")), "max_tokens", 4));
            long t0 = System.nanoTime();
            HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + "/v1/chat/completions"))
                    .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(body)).build(),
                    HttpResponse.BodyHandlers.ofString());
            double waited = (System.nanoTime() - t0) / 1e6;
            check(r.statusCode() == 503 && r.body().contains("overloaded") && waited >= 250,
                    scheduler + ": a third user while both slots are busy gets the 503 after the queue timeout (HTTP "
                            + r.statusCode() + " after " + Math.round(waited) + " ms)");
            for (Future<Answer> f : busy) {
                check(f.get().text().length() > 20, scheduler + ": the busy users' answers are unaffected");
            }
            ex.shutdown();
        }
    }

    /** One streamed request of the bench: when it started, when its first token and its end arrived, its token count. */
    record Streamed(long startNanos, long firstNanos, long endNanos, long tokens) {
    }

    /** A streamed greedy chat completion (stream_options.include_usage), timed at its first content chunk. */
    static Streamed stream(int port, String user, int maxTokens) throws Exception {
        String body = Json.write(Json.obj("model", "test", "messages", List.of(Json.obj("role", "user", "content",
                user + " /no_think")), "temperature", 0, "max_tokens", maxTokens, "stream", true,
                "stream_options", Json.obj("include_usage", true)));
        long t0 = System.nanoTime();
        HttpResponse<java.util.stream.Stream<String>> r = HTTP.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port
                        + "/v1/chat/completions")).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofLines());
        long first = 0;
        long tokens = 0;
        for (String line : (Iterable<String>) r.body()::iterator) {
            if (!line.startsWith("data: {")) {
                continue;
            }
            Map<String, Object> ev = Json.parseObject(line.substring(6));
            List<Object> choices = Json.list(ev, "choices");
            if (first == 0 && choices != null && !choices.isEmpty()
                    && !Json.str(Json.map(choices.getFirst(), "delta"), "content", "").isEmpty()) {
                first = System.nanoTime();
            }
            if (Json.map(ev, "usage") != null) {
                tokens = Json.num(Json.map(ev, "usage"), "completion_tokens", 0);
            }
        }
        return new Streamed(t0, first, System.nanoTime(), tokens);
    }

    /**
     * {@code users} concurrent users on {@code slots} slots, each streaming 64 tokens of a different story (int8 activations
     * as a default CPU run; prompt lookup off so every token is a decode step): generated tokens per second of all users
     * together, and the mean and worst time to the first token.
     */
    static double[] throughput(Path dir, int slots, int users, String[] extra) throws Exception {
        String[] all = java.util.Arrays.copyOf(new String[]{"--backend.cpu.int8-activations", "auto"}, 2 + extra.length);
        System.arraycopy(extra, 0, all, 2, extra.length);
        try (Served sv = Served.start(dir, settings(dir, "slots", slots, all))) {
            int port = sv.server().port();
            chat(port, null, "Warm up.", 8);
            ExecutorService ex = Executors.newFixedThreadPool(users);
            CountDownLatch go = new CountDownLatch(1);
            List<Future<Streamed>> fs = new ArrayList<>();
            for (int u = 0; u < users; u++) {
                int k = u;
                fs.add(ex.submit(() -> {
                    go.await();
                    return stream(port, "Tell me a long story about robot number " + (k + 1) + ".", 64);
                }));
            }
            long t0 = System.nanoTime();
            go.countDown();
            long tokens = 0;
            double ttftSum = 0;
            double ttftMax = 0;
            for (Future<Streamed> f : fs) {
                Streamed st = f.get();
                tokens += st.tokens();
                double ttft = (st.firstNanos() - st.startNanos()) / 1e9;
                ttftSum += ttft;
                ttftMax = Math.max(ttftMax, ttft);
            }
            double s = (System.nanoTime() - t0) / 1e9;
            ex.shutdown();
            return new double[]{tokens / s, ttftSum / users, ttftMax};
        }
    }

    static void bench(Path dir, String[] extra) throws Exception {
        System.out.println("\n| slots | concurrent users | generated tok/s, all users together | mean / worst time to first token |");
        System.out.println("|---|---|---|---|");
        for (int[] c : new int[][]{{1, 1}, {1, 4}, {4, 1}, {4, 2}, {4, 4}, {4, 8}}) {
            double[] r = throughput(dir, c[0], c[1], extra);
            System.out.printf(java.util.Locale.ROOT, "| %d | %d | %.1f | %.2f s / %.2f s |%n", c[0], c[1], r[0], r[1], r[2]);
        }
    }

    private static String abbreviate(String s) {
        String one = s.replace('\n', ' ');
        return "[" + (one.length() > 50 ? one.substring(0, 50) + "…" : one) + "]";
    }
}
