package org.robomiri.jllm.web.models;

import org.junit.jupiter.api.Test;
import org.robomiri.jllm.ModelLoader;
import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.BatchGroup;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.Priority;
import org.robomiri.jllm.engine.SlotPool;
import org.robomiri.jllm.sampling.SamplingParams;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A simulated agent workload against a loaded model, for comparing {@code engine.scheduler=slots} with {@code mixed}
 * (docs/25). Only runs with {@code -Dbench.model=<model dir>}:
 *
 * <pre>
 *   cd jllm-web && mvn -o test -Dtest=SchedulerBench -Dbench.model=Qwen/Qwen3-0.6B -Dbench.args="--backend metal ..."
 * </pre>
 *
 * <p>{@code agents} clients each run {@code turns} turns of a coding-agent conversation: every turn resends the whole
 * conversation (a system prompt of {@code system} tokens shared by all agents, the agent's own earlier messages and answers,
 * and {@code fresh} new tokens of "file contents"), the prompt is ~20x longer than the {@code gen} tokens generated, and the
 * agent thinks for {@code think} ms before the next turn. {@code chats} short chats (a 40-token prompt, {@code chatGen}
 * tokens) run next to them, back to back, until the agents are done. All are greedy; the model, slots and scheduler come
 * from {@code bench.args}. Per request: time to first token (from the request's arrival, so the wait for a slot counts)
 * and the gaps between tokens; per run: the wall time and the tokens per second.
 */
class SchedulerBench {

    private record Sample(String kind, double ttftMs, double[] gapsMs, int prompt, int cached, int generated) {
    }

    @Test
    void run() throws Exception {
        String model = System.getProperty("bench.model");
        assumeTrue(model != null, "-Dbench.model not given: skipped");
        String[] extra = System.getProperty("bench.args", "").isBlank() ? new String[0] : System.getProperty("bench.args").trim().split("\\s+");
        int slots = Integer.getInteger("bench.slots", 4);
        int agents = Integer.getInteger("bench.agents", 6);
        int turns = Integer.getInteger("bench.turns", 4);
        int chats = Integer.getInteger("bench.chats", 2);
        int system = Integer.getInteger("bench.system", 2000);
        int fresh = Integer.getInteger("bench.fresh", 300);
        int gen = Integer.getInteger("bench.gen", 64);
        int chatGen = Integer.getInteger("bench.chatGen", 96);
        int think = Integer.getInteger("bench.think", 200);
        boolean agentsBackground = Boolean.getBoolean("bench.agentsBackground");
        String label = System.getProperty("bench.label", "run");

        // bench.model is the repository id (Qwen/Qwen3-0.6B), as `--model` of the launcher; the folder is bench.dir/Qwen_Qwen3-0.6B
        Path models = Path.of(System.getProperty("bench.dir", "../models")).toAbsolutePath();
        List<String> args = new ArrayList<>(List.of("--model.dir", models.toString()));
        args.addAll(Arrays.asList(extra));
        Settings st = Settings.load(args.toArray(new String[0]));
        try (ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), model, slots)) {
            List<Engine> more = new ArrayList<>();
            for (int i = 1; i < slots; i++) {
                more.add(ModelLoader.extraEngine(st, l));
            }
            Map<String, String> snap = new LinkedHashMap<>();
            for (String k : new String[]{"engine.scheduler", "engine.max-running", "engine.preempt", "engine.session-max-running",
                    "engine.age-promote-ms"}) {
                snap.put(k, st.get(k));
            }
            LoadedModel lm = new LoadedModel("bench", "bench", l, more, snap, 0, null);
            boolean mixed = lm.admission() != null;
            Engine e0 = l.engine();
            System.out.printf("== %s: %s, %d slots, scheduler %s%s, %s%n", label, e0.backend().name(), slots,
                    mixed ? "mixed" : "slots", e0.backend().kvSequence() != null ? " (paged cache)" : " (flat cache)", Arrays.toString(extra));
            int[] sys = e0.tokenizer().encode(text("You are a careful coding agent. Follow the rules below exactly. ", 0, system));
            sys = Arrays.copyOf(sys, system);
            SamplingParams p = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, gen, List.of());
            SamplingParams pc = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, chatGen, List.of());

            // warm-up: compile the kernels, touch the weights (not measured)
            try (SlotPool.Slot s = lm.pick(sys)) {
                s.engine().generate(Arrays.copyOf(sys, 200), new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 8, List.of()), null, () -> false);
            }

            List<Sample> samples = Collections.synchronizedList(new ArrayList<>());
            AtomicBoolean agentsDone = new AtomicBoolean();
            CountDownLatch go = new CountDownLatch(1);
            List<Thread> threads = new ArrayList<>();
            int[] systemPrompt = sys;
            CountDownLatch agentsLeft = new CountDownLatch(agents);
            for (int a = 0; a < agents; a++) {
                int id = a;
                Thread t = new Thread(() -> {
                    try {
                        go.await();
                        int[] conv = systemPrompt;
                        for (int turn = 0; turn < turns; turn++) {
                            int[] next = Arrays.copyOf(e0.tokenizer().encode(text("Agent " + id + " turn " + turn + " file contents: ",
                                    id * 1000 + turn, fresh)), fresh);
                            int[] prompt = concat(conv, next);
                            Result r = request("agent", lm, prompt, p, agentsBackground ? Priority.BACKGROUND : Priority.INTERACTIVE, "agent" + id);
                            samples.add(r.sample());
                            conv = concat(prompt, r.tokens()); // the next turn resends everything, answer included
                            Thread.sleep(think);
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    } finally {
                        agentsLeft.countDown();
                    }
                }, "agent-" + a);
                threads.add(t);
            }
            for (int c = 0; c < chats; c++) {
                int id = c;
                Thread t = new Thread(() -> {
                    try {
                        go.await();
                        int n = 0;
                        while (!agentsDone.get()) {
                            int[] prompt = Arrays.copyOf(e0.tokenizer().encode(text("Chat " + id + " question " + n++ + ": ", id * 77 + n, 60)), 40);
                            samples.add(request("chat", lm, prompt, pc, Priority.INTERACTIVE, "chat" + id).sample());
                            Thread.sleep(150);
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }, "chat-" + c);
                threads.add(t);
            }
            threads.forEach(Thread::start);
            long t0 = System.nanoTime();
            go.countDown();
            agentsLeft.await();
            double wall = (System.nanoTime() - t0) / 1e9;
            agentsDone.set(true);
            for (Thread t : threads) {
                t.join();
            }
            double total = (System.nanoTime() - t0) / 1e9;

            long gener = samples.stream().mapToLong(Sample::generated).sum();
            long prompts = samples.stream().mapToLong(s -> s.prompt() - s.cached()).sum();
            long cached = samples.stream().mapToLong(Sample::cached).sum();
            System.out.printf("agents finished in %.1f s (all done %.1f s); %d requests; generated %d tokens (%.1f tok/s over the agents' wall time),"
                    + " prefilled %d prompt tokens, %d reused from the cache%n", wall, total, samples.size(), gener, gener / wall, prompts, cached);
            for (String kind : new String[]{"agent", "chat"}) {
                List<Sample> ss = samples.stream().filter(s -> s.kind().equals(kind)).toList();
                if (ss.isEmpty()) {
                    continue;
                }
                double[] ttft = ss.stream().mapToDouble(Sample::ttftMs).sorted().toArray();
                double[] gaps = ss.stream().flatMapToDouble(s -> Arrays.stream(s.gapsMs())).sorted().toArray();
                System.out.printf("  %-5s %3d requests: TTFT p50 %7.0f ms  p95 %7.0f ms  | inter-token p50 %6.1f ms  p95 %6.1f ms  p99 %6.1f ms  | %.1f tok/s per stream%n",
                        kind, ss.size(), pct(ttft, 50), pct(ttft, 95), pct(gaps, 50), pct(gaps, 95), pct(gaps, 99),
                        1000.0 / Math.max(1e-9, pct(gaps, 50)));
            }
            if (e0.backend() instanceof BatchGroup.Member m && m.group().mixed()) {
                long[] ms = m.group().mixedStats();
                System.out.printf("  mixed passes with a chunk and riders: %d, rider rows %d, chunks run in parts %d%n", ms[0], ms[1], ms[2]);
            }
            if (lm.admission() != null) {
                long[] a = lm.admission().stats();
                System.out.printf("  admission: %d grants, %d background slots paused, %d resumed, %d dropped%n", a[0], a[1], a[2], a[3]);
            }
            for (Engine e : more) {
                e.close();
            }
        }
    }

    private record Result(Sample sample, int[] tokens) {
    }

    private static Result request(String kind, LoadedModel lm, int[] prompt, SamplingParams p, Priority prio, String session)
            throws Exception {
        long t0 = System.nanoTime();
        List<Double> stamps = new ArrayList<>();
        try (Priority.Scope ignored = Priority.enter(prio); SlotPool.Slot slot = lm.pick(prompt, session, () -> false)) {
            Engine.Result r = slot.engine().generate(prompt, p, piece -> stamps.add((System.nanoTime() - t0) / 1e6), () -> false);
            double ttft = stamps.isEmpty() ? (System.nanoTime() - t0) / 1e6 : stamps.getFirst();
            double[] gaps = new double[Math.max(0, stamps.size() - 1)];
            for (int i = 1; i < stamps.size(); i++) {
                gaps[i - 1] = stamps.get(i) - stamps.get(i - 1);
            }
            return new Result(new Sample(kind, ttft, gaps, r.stats().promptTokens(), r.stats().cachedTokens(), r.tokens().length), r.tokens());
        }
    }

    private static double pct(double[] sorted, int p) {
        if (sorted.length == 0) {
            return 0;
        }
        return sorted[Math.min(sorted.length - 1, (int) Math.ceil(p / 100.0 * sorted.length) - 1)];
    }

    private static int[] concat(int[] a, int[] b) {
        int[] c = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }

    /** About {@code tokens} tokens of deterministic, varied text for {@code seed}. */
    private static String text(String lead, int seed, int tokens) {
        StringBuilder b = new StringBuilder(lead);
        java.util.Random r = new java.util.Random(seed * 7919L + 13);
        String[] words = {"function", "return", "value", "buffer", "index", "config", "request", "handler", "module", "string",
                "number", "update", "parse", "render", "cache", "token", "stream", "client", "server", "layer", "block", "queue"};
        while (b.length() < tokens * 6) {
            b.append(words[r.nextInt(words.length)]).append(r.nextInt(100)).append(r.nextBoolean() ? "; " : ", ");
        }
        return b.toString();
    }
}
