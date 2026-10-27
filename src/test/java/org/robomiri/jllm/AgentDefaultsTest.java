package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.BatchGroup;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.Priority;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.server.OpenAiFeatures;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The defaults for agent serving (docs/10 §10.13, §10.14): {@code backend.ctx=auto} sized from memory
 * ({@link ContextBudget}, a pure function) and the request priority ({@link Priority}): its parsing, the OpenAI field and
 * header, and the order in which {@link BatchGroup} gives the device to waiting requests.
 *
 * <p>usage: AgentDefaultsTest &lt;model dir&gt; (only the config.json is read)
 */
public final class AgentDefaultsTest {

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    public static void main(String[] args) throws Exception {
        context(Path.of(args[0]));
        priorityParsing();
        deviceOrder();
        if (failures > 0) {
            System.out.println(failures + " check(s) failed");
            System.exit(1);
        }
        System.out.println("agent defaults passed");
    }

    // ------------------------------------------------------------------ backend.ctx=auto

    private static void context(Path dir) throws Exception {
        Architecture arch = Architecture.load(dir);
        long per = ContextBudget.bytesPerToken(arch, KvDtype.F32);
        // Qwen3-0.6B: 28 layers x (keys + values) x 8 kv heads x 128 x 4 bytes
        check(per == 2L * 28 * 8 * 128 * 4, "bytes per token of Qwen3-0.6B at f32: " + per);
        check(ContextBudget.bytesPerToken(arch, KvDtype.F16) * 2 == per, "f16 halves the cache");
        check(ContextBudget.bytesPerToken(arch, KvDtype.Q8) < per / 3, "q8 is a quarter and a bit");

        long gb = 1_000_000_000L;
        ContextBudget.Plan roomy = ContextBudget.plan(per, 40960, 131072, 12 * gb, 1, "test");
        check(roomy.ctx() == 40960 && roomy.limitedBy().equals("max_position_embeddings"), "plenty of memory: the trained length, " + roomy.ctx());
        ContextBudget.Plan tight = ContextBudget.plan(per, 40960, 131072, 2 * gb, 1, "test");
        check(tight.ctx() == 8192 && tight.limitedBy().equals("memory"), "2 GB at 229 KB/token: 8192, " + tight.ctx());
        ContextBudget.Plan two = ContextBudget.plan(per, 40960, 131072, 2 * gb, 2, "test");
        check(two.ctx() == 4096, "two slots share the budget: " + two.ctx());
        ContextBudget.Plan starved = ContextBudget.plan(per, 40960, 131072, 100_000_000L, 1, "test");
        check(starved.ctx() == ContextBudget.FLOOR, "a model with no room still loads at the floor: " + starved.ctx());
        ContextBudget.Plan none = ContextBudget.plan(per, 40960, 131072, -5, 1, "test");
        check(none.ctx() == ContextBudget.FLOOR, "a negative budget is zero: " + none.ctx());
        ContextBudget.Plan tiny = ContextBudget.plan(per / 100, 128, 131072, 12 * gb, 1, "test");
        check(tiny.ctx() == ContextBudget.FALLBACK, "a model trained on 128 positions keeps the old 4096: " + tiny.ctx());
        ContextBudget.Plan huge = ContextBudget.plan(per, 262144, 131072, 100 * gb, 1, "test");
        check(huge.ctx() == 131072 && huge.limitedBy().equals("backend.ctx-max"), "capped by backend.ctx-max: " + huge.ctx());
        ContextBudget.Plan capped = ContextBudget.plan(per, 262144, 32768, 100 * gb, 1, "test");
        check(capped.ctx() == 32768, "a lower backend.ctx-max: " + capped.ctx());
        ContextBudget.Plan state = ContextBudget.plan(0, 0, 131072, 0, 1, "test");
        check(state.ctx() == ContextBudget.FALLBACK, "a model without a KV cache: " + state.ctx());
        ContextBudget.Plan odd = ContextBudget.plan(per, 40960, 131072, 3_000_000_000L, 1, "test");
        check(odd.ctx() % 1024 == 0 && odd.ctx() * per <= 3_000_000_000L, "rounded down to a multiple of 1024 that fits: " + odd.ctx());

        Settings paged = Settings.load(new String[]{"--backend.kv-paged", "true"});
        check(ContextBudget.pagedOffHeap(paged, arch, KvDtype.F32) && ContextBudget.pagedOffHeap(paged, arch, KvDtype.Q8),
                "paged KV cache: the CPU budget is off-heap (physical memory), not the Java heap");
        check(!ContextBudget.pagedOffHeap(paged, arch, KvDtype.Q4) && !ContextBudget.pagedOffHeap(Settings.load(new String[]{}), arch, KvDtype.F32),
                "a q4 cache, or paging off, stays on the heap budget");
        Settings mixed = Settings.load(new String[]{"--backend.kv-paged", "false", "--engine.scheduler", "mixed"});
        check(ContextBudget.pagedOffHeap(mixed, arch, KvDtype.F32, null, 1), // kv-paged=false never looks at the weights
                "the mixed scheduler pages the cache even with backend.kv-paged=false: the CPU budget is off-heap");
        // the paged CPU pool's slabs come from an automatic arena, which counts against the JVM's direct-memory limit
        // (this JVM sets no -XX:MaxDirectMemorySize, so the limit is the heap's maximum)
        long before = ContextBudget.directFree();
        java.lang.foreign.MemorySegment slab = java.lang.foreign.Arena.ofAuto().allocate(64L << 20, 64);
        long after = ContextBudget.directFree();
        check(before > 0 && before <= Runtime.getRuntime().maxMemory() && before - after >= 64L << 20,
                String.format("direct memory: %.0f MB free, %.0f MB after a 64 MB slab of an automatic arena", before / 1e6, after / 1e6));
        java.lang.ref.Reference.reachabilityFence(slab);

        Settings s = Settings.load(new String[]{});
        check(s.get("backend.ctx").equals("auto") && !ModelLoader.explicitCtx(s), "backend.ctx is auto by default");
        Settings n = Settings.load(new String[]{"--ctx", "2048"});
        check(ModelLoader.explicitCtx(n) && n.getInt("backend.ctx") == 2048, "a number overrides");
        check(rejects("--ctx", "lots"), "backend.ctx=lots is rejected");
        check(rejects("--backend.ctx-max", "100"), "backend.ctx-max below 1024 is rejected");
        check(!rejects("--ctx", "AUTO"), "backend.ctx=AUTO is accepted");
    }

    private static boolean rejects(String flag, String value) throws Exception {
        try {
            Settings.load(new String[]{flag, value}).validate();
            return false;
        } catch (IllegalArgumentException e) {
            return true;
        }
    }

    // ------------------------------------------------------------------ priority

    private static void priorityParsing() {
        check(Priority.current() == Priority.INTERACTIVE, "a thread is interactive by default");
        try (Priority.Scope a = Priority.enter(Priority.BACKGROUND)) {
            check(Priority.current() == Priority.BACKGROUND, "inside a scope");
            try (Priority.Scope b = Priority.enter(Priority.INTERACTIVE)) {
                check(Priority.current() == Priority.INTERACTIVE, "nested scope");
            }
            check(Priority.current() == Priority.BACKGROUND, "the outer scope is restored");
        }
        check(Priority.current() == Priority.INTERACTIVE, "back to the default");
        check(Priority.parse(" Background ") == Priority.BACKGROUND && Priority.parse(null) == Priority.INTERACTIVE
                && Priority.parse("") == Priority.INTERACTIVE, "parse");
        boolean bad = false;
        try {
            Priority.parse("urgent");
        } catch (IllegalArgumentException e) {
            bad = true;
        }
        check(bad, "an unknown priority is an error");

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("messages", List.of());
        check(OpenAiFeatures.parse(body, true, 100).priority() == Priority.INTERACTIVE, "no field: interactive");
        body.put("priority", "background");
        OpenAiFeatures.Options o = OpenAiFeatures.parse(body, true, 100);
        check(o.priority() == Priority.BACKGROUND && o.decoding(null).priority() == Priority.BACKGROUND,
                "\"priority\": \"background\" reaches the engine options");
        Map<String, Object> viaHeader = new LinkedHashMap<>();
        OpenAiFeatures.headerPriority(viaHeader, "background");
        check(OpenAiFeatures.parse(viaHeader, true, 100).priority() == Priority.BACKGROUND, "the X-Jllm-Priority header");
        OpenAiFeatures.headerPriority(body, "interactive");
        check(OpenAiFeatures.parse(body, true, 100).priority() == Priority.BACKGROUND, "the body wins over the header");
        body.put("priority", "soon");
        boolean rejected = false;
        try {
            OpenAiFeatures.parse(body, true, 100);
        } catch (IllegalArgumentException e) {
            rejected = true;
        }
        check(rejected, "a bad priority field is a 400");
        body.put("priority", 3L);
        rejected = false;
        try {
            OpenAiFeatures.parse(body, true, 100);
        } catch (IllegalArgumentException e) {
            rejected = true;
        }
        check(rejected, "a non-string priority is a 400");
        Engine.Decoding d = Engine.Decoding.NONE.withPriority(Priority.BACKGROUND).withConstraint(null);
        check(d.priority() == Priority.BACKGROUND, "Decoding keeps the priority when the constraint changes");
    }

    // ------------------------------------------------------------------ the device queue

    /** A backend whose passes take a while; {@code log} gets the name when a pass of it runs. */
    private static final class Slow implements Backend {
        final String name;
        final long millis;
        final List<String> log;

        Slow(String name, long millis, List<String> log) {
            this.name = name;
            this.millis = millis;
            this.log = log;
        }

        public String name() {
            return name;
        }

        public int contextLength() {
            return 256;
        }

        public int maxBatch() {
            return 64;
        }

        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            log.add(name + ":start");
            try {
                Thread.sleep(millis);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            log.add(name + ":end");
        }

        public void close() {
        }
    }

    private static void deviceOrder() throws Exception {
        for (boolean ordered : new boolean[]{true, false}) {
            List<String> log = Collections.synchronizedList(new ArrayList<>());
            BatchGroup g = new BatchGroup(ordered);
            Backend running = g.join(new Slow("running", 300, log));
            Backend background = g.join(new Slow("background", 20, log));
            Backend interactive = g.join(new Slow("interactive", 20, log));
            int[] t = new int[8];
            Thread a = new Thread(() -> {
                try (Priority.Scope s = Priority.enter(Priority.BACKGROUND)) {
                    running.forward(t, 0, 8, 0, null);
                }
            });
            Thread b = new Thread(() -> {
                try (Priority.Scope s = Priority.enter(Priority.BACKGROUND)) {
                    background.forward(t, 0, 8, 0, null);
                }
            });
            Thread c = new Thread(() -> {
                try (Priority.Scope s = Priority.enter(Priority.INTERACTIVE)) {
                    interactive.forward(t, 0, 8, 0, null);
                }
            });
            a.start();
            Thread.sleep(80);
            b.start(); // waits first ...
            Thread.sleep(80);
            c.start(); // ... the interactive request arrives later
            a.join();
            b.join();
            c.join();
            int ib = log.indexOf("background:start");
            int ic = log.indexOf("interactive:start");
            if (ordered) {
                check(ic < ib, "the interactive request takes the device before the background one that waited first: " + log);
            } else {
                check(ib < ic, "engine.priority-order=false: first come, first served: " + log);
            }
        }
    }
}
