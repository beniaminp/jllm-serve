package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.cpu.KvCache;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.sampling.SamplingParams;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Measures the paged KV cache against the flat one on a real model (docs/24 §24.9). Run through the heavy-job lock.
 *
 * <pre>
 *   PagedKvBench &lt;model dir&gt; [setting value ...]        e.g.  --backend metal --backend.kv-dtype f16
 *   1. memory:   N conversations sharing a 4 096-token system prompt (slots = N, ctx 8 192)
 *   2. TTFT:     a conversation resent after another one took its slot
 *   3. speed:    prefill and decode tok/s of one conversation (2 000-token prompt, 64 generated), median of 3
 * </pre>
 */
public final class PagedKvBench {

    static final int SYSTEM = 4096;
    static final int CTX = 8192;

    private static String[] base(Path dir, boolean paged, String... extra) {
        List<String> a = new ArrayList<>(List.of("--model.dir", dir.toAbsolutePath().getParent().toString(), "--backend.ctx", "" + CTX,
                "--engine.speculative", "off", "--backend.kv-paged", "" + paged));
        a.addAll(List.of(extra));
        return a.toArray(new String[0]);
    }

    private static ModelLoader.Loaded load(Path dir, boolean paged, int slots, String[] extra) throws Exception {
        Settings st = Settings.load(base(dir, paged, extra));
        return ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString(), slots);
    }

    private static Engine extra(Path dir, boolean paged, ModelLoader.Loaded l, String[] extra) throws Exception {
        return ModelLoader.extraEngine(Settings.load(base(dir, paged, extra)), l);
    }

    private static int[] text(Engine e, int from, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = from; sb.length() < n * 7; i++) {
            sb.append("entry ").append(i).append(" of the list, ");
        }
        return Arrays.copyOf(e.tokenizer().encode(sb.toString()), n);
    }

    private static int[] concat(int[] a, int[] b) {
        int[] r = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }

    private static final SamplingParams GREEDY16 = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 16, List.of());

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        String[] extra = Arrays.copyOfRange(args, 1, args.length);
        KvDtype dt = KvDtype.F32;
        for (int i = 0; i + 1 < extra.length; i++) {
            if (extra[i].equals("--backend.kv-dtype")) {
                dt = KvDtype.parse(extra[i + 1]);
            }
        }
        System.out.println("settings: " + String.join(" ", extra));
        memory(dir, extra, dt);
        ttft(dir, extra);
        speed(dir, extra);
    }

    /** N conversations with a shared 4 096-token system prompt and 64 own tokens each. */
    static void memory(Path dir, String[] extra, KvDtype dt) throws Exception {
        System.out.println("\n| conversations | flat slots, allocated (MB) | flat, positions in use (MB) | paged pool (MB) | paged / flat in use | prefill of the n-th (ms) |");
        System.out.println("|---|---|---|---|---|---|");
        for (int n : new int[]{1, 2, 4, 8}) {
            try (ModelLoader.Loaded l = load(dir, true, n, extra)) {
                List<Engine> engines = new ArrayList<>();
                engines.add(l.engine());
                for (int i = 1; i < n; i++) {
                    engines.add(extra(dir, true, l, extra));
                }
                int[] system = text(l.engine(), 0, SYSTEM);
                double lastPrefill = 0;
                int used = 0;
                for (int i = 0; i < n; i++) {
                    int[] prompt = concat(system, text(l.engine(), 100_000 * (i + 1), 64));
                    Engine.Result r = engines.get(i).generate(prompt, GREEDY16, null, () -> false);
                    lastPrefill = r.stats().prefillMs();
                    used += prompt.length + r.tokens().length;
                }
                KvPool pool = engines.get(0).backend().kvSequence().pool();
                double paged = pool.stats().residentBytes() / 1e6;
                var arch = l.weights().decoder();
                Attention.Gqa g = (Attention.Gqa) arch.attention();
                long perPos = KvCache.bytes(dt, arch.core().layers(), g, 1);
                double alloc = n * (double) KvCache.bytes(dt, arch.core().layers(), g, CTX) / 1e6;
                double inUse = used * (double) perPos / 1e6;
                System.out.printf("| %d | %.0f | %.0f | %.0f | %.2f | %.0f |%n", n, alloc, inUse, paged, paged / inUse, lastPrefill);
                for (int i = 1; i < n; i++) {
                    engines.get(i).close();
                }
            }
        }
    }

    /** A conversation resent after another one took its slot: flat prefills it (or copies a shared prefix), paged adopts. */
    static void ttft(Path dir, String[] extra) throws Exception {
        System.out.println("\n| resent conversation (4 160 tokens) | prefill before the first token (ms) | tokens reused |");
        System.out.println("|---|---|---|");
        for (boolean paged : new boolean[]{false, true}) {
            try (ModelLoader.Loaded l = load(dir, paged, 2, extra)) {
                Engine e0 = l.engine();
                Engine e1 = extra(dir, paged, l, extra);
                int[] system = text(e0, 0, SYSTEM);
                int[] a = concat(system, text(e0, 200_000, 64));
                int[] other = text(e0, 900_000, SYSTEM + 64);
                e0.generate(a, GREEDY16, null, () -> false);                       // A in slot 0
                Engine.Result same = e0.generate(concat(a, text(e0, 7, 8)), GREEDY16, null, () -> false); // its next turn, same slot
                e0.generate(other, GREEDY16, null, () -> false);                   // another conversation takes slot 0
                Engine.Result back;
                if (paged) {
                    back = e1.generate(a, GREEDY16, null, () -> false);            // slot 1: the blocks come from the tree
                } else {
                    back = e1.generate(a, GREEDY16, null, () -> false);            // slot 1 has nothing: full prefill
                }
                System.out.printf("| %s: next turn in the same slot | %.0f | %d |%n", paged ? "paged" : "flat", same.stats().prefillMs(), same.stats().cachedTokens());
                System.out.printf("| %s: resent after its slot was taken | %.0f | %d |%n", paged ? "paged" : "flat", back.stats().prefillMs(), back.stats().cachedTokens());
                e1.close();
            }
        }
    }

    /** One conversation: prefill and decode speed of a 2 000-token prompt and 64 generated tokens, median of 3. */
    static void speed(Path dir, String[] extra) throws Exception {
        System.out.println("\n| KV cache | prefill tok/s | decode tok/s |");
        System.out.println("|---|---|---|");
        SamplingParams gen = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 64, List.of());
        for (boolean paged : new boolean[]{false, true}) {
            try (ModelLoader.Loaded l = load(dir, paged, 1, extra)) {
                Engine e = l.engine();
                e.generate(text(e, 5, 300), GREEDY16, null, () -> false); // warm up
                double[] pre = new double[3];
                double[] dec = new double[3];
                for (int r = 0; r < 3; r++) {
                    int[] prompt = text(e, 1000 * (r + 3), 2000);
                    Engine.Result res = e.generate(prompt, gen, null, () -> false);
                    pre[r] = prompt.length / (res.stats().prefillMs() / 1000);
                    dec[r] = res.stats().tokPerSec();
                }
                Arrays.sort(pre);
                Arrays.sort(dec);
                System.out.printf("| %s | %.0f | %.1f |%n", paged ? "paged" : "flat", pre[1], dec[1]);
            }
        }
    }
}
