package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.kv.FileSpill;
import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.kv.KvSequence;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.metal.MoeSplit;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.Architecture;

import java.io.ByteArrayOutputStream;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The paged KV cache (docs/24) against the flat one, bit for bit, for one model of {@link ArchSmoke}, on the CPU backend
 * and, when a GPU (or the emulator) is there, on Metal; every cache type. The same scripted calls on a flat and a paged
 * backend must give identical logits:
 * <ul>
 *   <li>prefill chunks, decode steps (Metal: recorded and replayed), a rewind into the middle of a block and a
 *       verification-shaped chunk with the logits of every row (speculation rollback);</li>
 *   <li>sharing: a second backend adopts the first one's published blocks and continues, a third forks in the middle
 *       of a block (copy on write), and a rewind of the first does not disturb the others;</li>
 *   <li>eviction: a pool too small for four prompts spills the oldest to a directory and a backend reloads it;</li>
 *   <li>three backends decoding in one batched pass (batched rows);</li>
 *   <li>saved caches: the bytes of {@code saveState} (and of a snapshot) equal the flat cache's, in both directions.</li>
 * </ul>
 * Models whose cache is not paged (Mamba, Qwen3-Next, DeepSeek's latent attention) are only checked to refuse politely.
 */
final class PagedKvChecks {

    private static final int CTX = 128;
    private static final int BATCH = 64;

    private final ModelWeights w;
    private final String lib;
    private final boolean metal;
    private final KvDtype dt;
    private final Decoder arch;
    private final int vocab;
    private final String name;
    private int failures;
    private final int[] tokens;

    private PagedKvChecks(ModelWeights w, String lib, boolean metal, KvDtype dt, Architecture cfg) {
        this.w = w;
        this.lib = lib;
        this.metal = metal;
        this.dt = dt;
        this.arch = (Decoder) cfg;
        this.vocab = cfg.core().vocabSize();
        this.name = (metal ? "metal" : "cpu") + " " + dt + (dt == KvDtype.Q8 && KvDtype.q8Rotate() ? " rotated keys" : "");
        this.tokens = new int[CTX];
        for (int i = 0; i < CTX; i++) {
            tokens[i] = (i * 37 + 11) % Math.min(vocab, 600);
        }
    }

    /** Runs the checks on the CPU backend (and Metal when {@code lib} is usable) for the cache types; returns the failures. */
    static int run(ModelWeights w, String lib, boolean withMetal, Architecture cfg, boolean allTypes) throws Exception {
        if (!(cfg instanceof Decoder d)) {
            return 0;
        }
        if (d.attention() instanceof Attention.Mla) {
            System.out.println("  paged kv: not for a latent cache (the flat cache is used)");
            return 0;
        }
        int failures = 0;
        KvDtype[] types = allTypes ? new KvDtype[]{KvDtype.F32, KvDtype.F16, KvDtype.Q8} : new KvDtype[]{KvDtype.F32, KvDtype.Q8};
        for (KvDtype dt : types) {
            failures += new PagedKvChecks(w, lib, false, dt, cfg).all();
            if (withMetal && d.attention().headDim() % 32 == 0) {
                failures += new PagedKvChecks(w, lib, true, dt, cfg).all();
            }
        }
        if (d.attention() instanceof Attention.Gqa g && org.robomiri.jllm.model.KvQ4.supports(g.headDim())) {
            KvDtype.setQ8Rotate(true); // q8 with Hadamard-rotated keys (backend.kv-q8-rotate, the shipped default)
            try {
                failures += new PagedKvChecks(w, lib, false, KvDtype.Q8, cfg).all();
                if (withMetal && g.headDim() % 32 == 0) {
                    failures += new PagedKvChecks(w, lib, true, KvDtype.Q8, cfg).all();
                }
            } finally {
                KvDtype.setQ8Rotate(false);
            }
        }
        return failures;
    }

    private void check(boolean ok, String what) {
        System.out.println((ok ? "  ok   paged " : "  FAIL paged ") + name + ": " + what);
        if (!ok) {
            failures++;
        }
    }

    private KvPool pool(int slots, long poolTokens) {
        KvPool p = metal ? MetalBackend.newKvPool(w, CTX, slots, dt, 16, poolTokens, lib)
                : CpuBackend.newKvPool(w, CTX, slots, dt, 16, poolTokens);
        if (p == null) {
            throw new IllegalStateException("no pool");
        }
        return p;
    }

    private Backend flat() {
        return metal ? new MetalBackend(w, CTX, BATCH, lib, dt, MoeSplit.OFF) : new CpuBackend(w, CTX, BATCH, 4, dt, false);
    }

    private Backend paged(KvPool pool) {
        return metal ? new MetalBackend(w, CTX, BATCH, lib, dt, MoeSplit.OFF, pool) : new CpuBackend(w, CTX, BATCH, 4, dt, false, pool);
    }

    private int all() throws Exception {
        try {
            script();
            sharing();
            eviction();
            batched();
            savedCaches();
        } catch (RuntimeException e) {
            e.printStackTrace(System.out);
            check(false, "threw " + e);
        }
        return failures;
    }

    /** {pos, n}: prefill chunks, decode, a rewind into a block, a verification chunk, a rewind to a block boundary. */
    private static final int[][] OPS = {{0, 20}, {20, 13}, {33, 1}, {34, 1}, {35, 1}, {36, 1}, {30, 4}, {34, 6}, {40, 1}, {41, 1},
            {16, 17}, {33, 1}, {34, 1}, {35, 1}, {36, 1}, {37, 1}};

    private float[] step(Backend b, int pos, int n, boolean all) {
        if (all) {
            float[] out = new float[n * vocab];
            b.forwardAll(tokens, pos, n, pos, out, vocab);
            return out;
        }
        float[] out = new float[vocab];
        b.forward(tokens, pos, n, pos, out);
        return out;
    }

    /**
     * The scripted passes, repeated until they agree (at most three times). A kernel's first calls (interpreted, or
     * compiled at a lower tier) can round differently from its later ones, so even the flat cache can differ from itself
     * on a repeat of the same call while the JVM is cold; a paging error fails every attempt.
     */
    private void script() {
        int before = failures;
        for (int attempt = 1; attempt <= 3; attempt++) {
            failures = before;
            if (script1(attempt == 3)) {
                if (attempt > 1) {
                    System.out.println("  (agreed on attempt " + attempt + ": the first one ran while the kernels were cold)");
                }
                return;
            }
        }
    }

    private boolean script1(boolean report) {
        try (Backend f = flat(); KvPool pool = pool(1, 0); Backend p = paged(pool)) {
            if (report) {
                check(f.stateFormat() != null && f.stateFormat().equals(p.stateFormat()), "the same saved-cache format");
            }
            boolean same = true;
            int i = 0;
            for (int[] op : OPS) {
                boolean all = op[1] == 6; // the verification-shaped chunk returns every row
                same &= Arrays.equals(step(f, op[0], op[1], all), step(p, op[0], op[1], all));
                i++;
                if (!same) {
                    if (report) {
                        check(false, "logits differ at op " + i + " (pos " + op[0] + ", n " + op[1] + ")");
                    }
                    return false;
                }
            }
            check(true, "prefill, decode, rewind into a block, verification chunk: logits identical (" + OPS.length + " passes)");
            return true;
        }
    }

    /** The published prefix of one backend serves another; a fork in the middle of a block copies; a rewind disturbs nobody. */
    private void sharing() {
        try (KvPool pool = pool(3, 0); Backend a = paged(pool); Backend b = paged(pool); Backend c = paged(pool);
             Backend fa = flat(); Backend fb = flat(); Backend fc = flat()) {
            // a computes a 46-token prompt in two chunks and publishes it
            step(a, 0, 20, false);
            step(a, 20, 26, false);
            a.kvSequence().publish(tokens, 46);
            for (Backend f : List.of(fa, fb, fc)) {
                step(f, 0, 20, false);
                step(f, 20, 26, false);
            }
            // b adopts the two full blocks and runs the rest; its flat twin rewinds to 32 and runs the same
            KvSequence sb = b.kvSequence();
            int found = sb.match(tokens, 45);
            int got = sb.adopt(tokens, found);
            check(found == 32 && got == 32, "the second backend adopts the two published blocks (" + found + ", " + got + ")");
            check(Arrays.equals(step(b, 32, 14, false), step(fb, 32, 14, false)), "adopted rows give the flat logits");
            // c forks in the middle of block 1 (token 21) with other tokens
            int[] other = tokens.clone();
            for (int t = 21; t < CTX; t++) {
                other[t] = (tokens[t] + 101) % Math.min(vocab, 600);
            }
            KvSequence sc = c.kvSequence();
            int fm = sc.match(other, 40);
            int fg = sc.adopt(other, fm);
            check(fm == 21 && fg == 21, "a fork at 21 shares 21 tokens (" + fm + ", " + fg + ")");
            float[] lc = new float[vocab];
            c.forward(other, 21, 10, 21, lc);
            float[] lfc = new float[vocab];
            fc.forward(other, 21, 10, 21, lfc);
            check(Arrays.equals(lc, lfc), "the fork (copy on write of the shared block) gives the flat logits");
            // a rewinds into block 1 and continues: b and c keep what they read
            float[] la = new float[vocab];
            a.forward(other, 25, 6, 25, la);
            float[] lfa = new float[vocab];
            fa.forward(other, 25, 6, 25, lfa);
            check(Arrays.equals(la, lfa), "a rewind into a shared block gives the flat logits");
            check(Arrays.equals(step(b, 46, 1, false), step(fb, 46, 1, false))
                    && Arrays.equals(step(c, 31, 1, false), step(fc, 31, 1, false)), "and the other sequences read their rows undisturbed");
            KvPool.Stats s = pool.stats();
            check(s.residentBlocks() < 3 * 4, "three sequences of ~46 tokens take " + s.residentBlocks() + " blocks, not 9");
        }
    }

    /** Four prompts through a pool of six blocks with a spill directory: the first comes back from disk. */
    private void eviction() throws Exception {
        Path dir = Files.createTempDirectory("kvspill");
        try (KvPool pool = pool(1, 96); Backend fa = flat()) {
            pool.setSpill(new FileSpill(dir, "arch-smoke " + name, 1L << 30));
            int[] p1 = tokens;
            try (Backend a = paged(pool)) {
                step(a, 0, 20, false);
                step(a, 20, 28, false);
                a.kvSequence().publish(p1, 48);
            }
            step(fa, 0, 20, false);
            step(fa, 20, 28, false);
            for (int k = 1; k <= 3; k++) { // other prompts push it out
                int[] pk = new int[CTX];
                for (int t = 0; t < CTX; t++) {
                    pk[t] = (tokens[t] * (k + 2) + 17 * k) % Math.min(vocab, 600);
                }
                try (Backend d = paged(pool)) {
                    float[] out = new float[vocab];
                    d.forward(pk, 0, 16, 0, out);
                    d.forward(pk, 16, 16, 16, out);
                    d.forward(pk, 32, 16, 32, out);
                    d.kvSequence().publish(pk, 48);
                }
            }
            KvPool.Stats s = pool.stats();
            check(s.evictions() > 0 && s.spills() > 0, "blocks were evicted (" + s.evictions() + ") and spilled (" + s.spillFiles() + " files)");
            try (Backend b = paged(pool)) {
                KvSequence sb = b.kvSequence();
                int found = sb.match(p1, 47);
                int got = sb.adopt(p1, found);
                check(found == 32 && got == 32 && pool.stats().reloads() > 0, "the first prompt's two blocks come back from disk ("
                        + got + " tokens, " + pool.stats().reloads() + " reloaded)");
                float[] lb = new float[vocab];
                b.forward(p1, 32, 16, 32, lb);
                float[] lf = new float[vocab];
                fa.forward(p1, 32, 16, 32, lf);
                check(Arrays.equals(lb, lf), "reloaded rows give the flat logits");
            }
        } finally {
            try (var walk = Files.walk(dir)) {
                walk.sorted(java.util.Comparator.reverseOrder()).forEach(x -> x.toFile().delete());
            }
        }
    }

    /** Three backends decode one token each in one batched pass: paged equals flat. */
    private void batched() {
        try (KvPool pool = pool(3, 0)) {
            List<Backend> paged = new ArrayList<>();
            List<Backend> flats = new ArrayList<>();
            try {
                int[] pos = {23, 31, 17};
                for (int i = 0; i < 3; i++) {
                    paged.add(paged(pool));
                    flats.add(flat());
                    int[] t = new int[CTX];
                    for (int j = 0; j < CTX; j++) {
                        t[j] = (tokens[j] + 13 * i) % Math.min(vocab, 600);
                    }
                    float[] out = new float[vocab];
                    paged.get(i).forward(t, 0, pos[i], 0, out);
                    flats.get(i).forward(t, 0, pos[i], 0, out);
                }
                int[] next = {5, 77, 301};
                float[][] lp = new float[3][vocab];
                float[][] lf = new float[3][vocab];
                boolean okP = paged.get(0).forwardBatch(paged.toArray(new Backend[0]), next, pos, lp);
                boolean okF = flats.get(0).forwardBatch(flats.toArray(new Backend[0]), next, pos, lf);
                check(okP == okF, "batched rows: both ran a batched pass (" + okP + ")");
                if (okP && okF) {
                    check(Arrays.equals(lp[0], lf[0]) && Arrays.equals(lp[1], lf[1]) && Arrays.equals(lp[2], lf[2]),
                            "three rows in one pass: logits identical");
                    // and the next lone steps read the rows the pass wrote
                    float[] a1 = new float[vocab];
                    float[] a2 = new float[vocab];
                    paged.get(1).forward(next, 1, 1, pos[1] + 1, a1);
                    flats.get(1).forward(next, 1, 1, pos[1] + 1, a2);
                    check(Arrays.equals(a1, a2), "a lone step after the batched pass reads the rows it wrote");
                }
            } finally {
                for (Backend b : paged) {
                    b.close();
                }
                for (Backend b : flats) {
                    b.close();
                }
            }
        }
    }

    private static byte[] saved(Backend b, int n) throws Exception {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        b.saveState(n, Channels.newChannel(o));
        return o.toByteArray();
    }

    /** saveState / snapshotState of a paged backend are the flat bytes; each loads into the other kind. */
    private void savedCaches() throws Exception {
        try (KvPool pool = pool(2, 0); Backend f = flat(); Backend p = paged(pool); Backend p2 = paged(pool); Backend f2 = flat()) {
            step(f, 0, 20, false);
            step(f, 20, 13, false);
            step(p, 0, 20, false);
            step(p, 20, 13, false);
            byte[] fb = saved(f, 33);
            byte[] pb = saved(p, 33);
            check(Arrays.equals(fb, pb), "saved bytes identical (" + pb.length + ")");
            Backend.StateSnapshot snap = p.snapshotState(33);
            step(p, 30, 8, false); // the paged backend moves on (rewrites rows 30..37 of a snapshotted block)
            ByteArrayOutputStream o = new ByteArrayOutputStream();
            if (snap != null) {
                snap.write(Channels.newChannel(o));
                snap.close();
            }
            check(snap == null ? false : Arrays.equals(fb, o.toByteArray()), "a snapshot taken before the rewrite still writes the flat bytes");
            // flat bytes into a paged backend, paged bytes into a flat one: the next steps agree
            p2.loadState(33, Channels.newChannel(new java.io.ByteArrayInputStream(fb)));
            f2.loadState(33, Channels.newChannel(new java.io.ByteArrayInputStream(pb)));
            float[] a = new float[vocab];
            float[] b = new float[vocab];
            p2.forward(tokens, 33, 1, 33, a);
            f2.forward(tokens, 33, 1, 33, b);
            float[] c = new float[vocab];
            Backend g = flat();
            try {
                step(g, 0, 20, false);
                step(g, 20, 13, false);
                g.forward(tokens, 33, 1, 33, c);
            } finally {
                g.close();
            }
            check(Arrays.equals(a, c) && Arrays.equals(b, c), "a cache loaded across kinds continues like the original");
            // copyStateFrom: blocks are shared, not copied
            Backend p3 = paged(pool);
            try {
                p3.copyStateFrom(p2, 33);
                float[] d = new float[vocab];
                p3.forward(tokens, 33, 1, 33, d);
                check(Arrays.equals(d, c), "copyStateFrom shares the blocks and continues like the original");
            } finally {
                p3.close();
            }
        }
    }
}
