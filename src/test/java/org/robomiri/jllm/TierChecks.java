package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.kv.FileSpill;
import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.kv.KvSequence;
import org.robomiri.jllm.kv.KvTier;
import org.robomiri.jllm.kv.KvTierConfig;
import org.robomiri.jllm.kv.TieredSpill;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.metal.MoeSplit;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;

import java.io.ByteArrayOutputStream;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Tiered KV storage (docs/26) against all-resident storage, bit for bit, for one model of {@link ArchSmoke}:
 * <ul>
 *   <li><b>Pool tiers</b> (CPU and Metal, every cache type): a paged pool with a hot cap of five blocks and a host tier of two
 *       blocks in front of the block files; prompts that do not fit are pushed down (memory, then files) and a backend adopts
 *       them back (prefetched) and continues with the logits of the flat cache. On the CPU the slabs beyond the first are
 *       memory-mapped files and the same scripted passes must give the flat logits.</li>
 *   <li><b>Streamed KV</b> (Metal): a slot that keeps its rows on the host and runs layer by layer through two windows gives
 *       the flat cache's logits for prefill, decode, a rewind and a verification chunk, writes the same saved bytes, loads
 *       the flat bytes, and does the same when some layers' rows are memory-mapped files.</li>
 * </ul>
 */
final class TierChecks {

    private static final int CTX = 128;
    private static final int BATCH = 64;

    private final ModelWeights w;
    private final String lib;
    private final boolean metal;
    private final KvDtype dt;
    private final int vocab;
    private final String name;
    private int failures;
    private final int[] tokens;
    private Path dir;

    private TierChecks(ModelWeights w, String lib, boolean metal, KvDtype dt, Architecture cfg) {
        this.w = w;
        this.lib = lib;
        this.metal = metal;
        this.dt = dt;
        this.vocab = cfg.core().vocabSize();
        this.name = (metal ? "metal" : "cpu") + " " + dt + (dt == KvDtype.Q8 && KvDtype.q8Rotate() ? " rotated keys" : "");
        this.tokens = new int[CTX];
        for (int i = 0; i < CTX; i++) {
            tokens[i] = (i * 37 + 11) % Math.min(vocab, 600);
        }
    }

    static int run(ModelWeights w, String lib, boolean withMetal, Architecture cfg, boolean allTypes) throws Exception {
        if (!(cfg instanceof Decoder d) || !(d.attention() instanceof Attention.Gqa g)) {
            return 0;
        }
        int failures = 0;
        KvDtype[] types = allTypes ? new KvDtype[]{KvDtype.F32, KvDtype.F16, KvDtype.Q8} : new KvDtype[]{KvDtype.F32, KvDtype.Q8};
        for (KvDtype dt : types) {
            failures += new TierChecks(w, lib, false, dt, cfg).all();
            if (withMetal && g.headDim() % 32 == 0) {
                failures += new TierChecks(w, lib, true, dt, cfg).all();
            }
        }
        if (org.robomiri.jllm.model.KvQ4.supports(g.headDim())) {
            KvDtype.setQ8Rotate(true);
            try {
                failures += new TierChecks(w, lib, false, KvDtype.Q8, cfg).all();
                if (withMetal && g.headDim() % 32 == 0) {
                    failures += new TierChecks(w, lib, true, KvDtype.Q8, cfg).all();
                }
            } finally {
                KvDtype.setQ8Rotate(false);
            }
        }
        return failures;
    }

    private void check(boolean ok, String what) {
        String line = (ok ? "  ok   tier " : "  FAIL tier ") + name + ": " + what + "\n";
        if (held != null) {
            held.append(line);
        } else {
            System.out.print(line);
        }
        if (!ok) {
            failures++;
        }
    }

    private int all() throws Exception {
        dir = Files.createTempDirectory("kvtiers");
        try {
            poolTiers();
            if (!metal) {
                mappedSlabs();
            } else {
                streamed(0);
                streamed(1);
            }
        } catch (RuntimeException e) {
            e.printStackTrace(System.out);
            check(false, "threw " + e);
        } finally {
            try (var walk = Files.walk(dir)) {
                walk.sorted(java.util.Comparator.reverseOrder()).forEach(x -> x.toFile().delete());
            }
        }
        return failures;
    }

    private KvTierConfig cfg(long hot, long host) {
        return new KvTierConfig(hot, host, org.robomiri.jllm.kv.KvBlockCodec.Format.NATIVE, true, KvTierConfig.Stream.OFF, dir);
    }

    private KvPool pool(int slots, long poolTokens, KvTierConfig tier) {
        KvPool p = metal ? MetalBackend.newKvPool(w, CTX, slots, dt, 16, poolTokens, lib, tier)
                : CpuBackend.newKvPool(w, CTX, slots, dt, 16, poolTokens, KvDtype.q8Rotate(), tier);
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

    /** {pos, n}: prefill chunks, decode, a rewind into a block, a verification chunk, a rewind to a block boundary. */
    private static final int[][] OPS = {{0, 20}, {20, 13}, {33, 1}, {34, 1}, {35, 1}, {36, 1}, {30, 4}, {34, 6}, {40, 1}, {41, 1},
            {16, 17}, {33, 1}, {34, 1}, {35, 1}, {36, 1}, {37, 1}};

    /**
     * A pool that may commit five blocks in front of a two-block host tier: a 48-token prompt is published, other prompts push
     * it down to the files, and a new backend adopts it back and goes on with the flat logits.
     */
    private void poolTiers() throws Exception {
        for (int attempt = 1; attempt <= 3; attempt++) {
            int before = failures;
            held = new StringBuilder();
            poolTiers1(true);
            boolean ok = failures == before;
            String text = held.toString();
            held = null;
            if (ok || attempt == 3) {
                System.out.print(text);
                return;
            }
            failures = before;
        }
    }

    /** While set, {@link #check} keeps its lines here (a pass that may be repeated while the kernels are cold). */
    private StringBuilder held;

    private boolean poolTiers1(boolean report) throws Exception {
        Path files = Files.createTempDirectory(dir, "blocks");
        try (KvPool pool = pool(1, 0, cfg(0, 0)); Backend fa = flat()) {
            pool.setHotCap(5);
            long blockBytes = pool.storage().spillBytes();
            TieredSpill t = KvTier.attach(pool, cfg(0, 2 * blockBytes), "arch-smoke " + name, files, 1L << 30);
            int[] p1 = tokens;
            try (Backend a = paged(pool)) {
                step(a, 0, 20, false);
                step(a, 20, 28, false);
                a.kvSequence().publish(p1, 48);
            }
            step(fa, 0, 20, false);
            step(fa, 20, 28, false);
            for (int k = 1; k <= 3; k++) { // other prompts push it out of the five blocks, through the two-block tier
                int[] pk = new int[CTX];
                for (int i = 0; i < CTX; i++) {
                    pk[i] = (tokens[i] * (k + 2) + 17 * k) % Math.min(vocab, 600);
                }
                try (Backend d = paged(pool)) {
                    float[] out = new float[vocab];
                    d.forward(pk, 0, 16, 0, out);
                    d.forward(pk, 16, 16, 16, out);
                    d.forward(pk, 32, 16, 32, out);
                    d.kvSequence().publish(pk, 48);
                }
            }
            t.drain();
            KvPool.Stats s = pool.stats();
            boolean tiers = s.committedBlocks() <= 5 && s.evictions() > 0 && t.stats().demotions() > 0;
            if (report) {
                check(tiers, "five hot blocks held: " + s.committedBlocks() + " committed, " + s.evictions() + " evicted, "
                        + t.stats().demotions() + " went on to files");
            } else if (!tiers) {
                return false;
            }
            try (Backend b = paged(pool)) {
                KvSequence sb = b.kvSequence();
                int found = sb.match(p1, 47);
                t.drain(); // the prefetch hints
                int got = sb.adopt(p1, found);
                float[] lb = new float[vocab];
                b.forward(p1, 32, 16, 32, lb);
                float[] lf = new float[vocab];
                fa.forward(p1, 32, 16, 32, lf);
                boolean ok = found == 32 && got == 32 && Arrays.equals(lb, lf);
                if (report) {
                    check(ok, "the first prompt comes back through the tiers (" + got + " tokens, " + pool.stats().reloads()
                            + " blocks reloaded, " + t.stats().ramHits() + " from memory) with the flat logits");
                } else if (!ok) {
                    return false;
                }
            }
        }
        return true;
    }

    /** CPU: the slabs past the first are memory-mapped files; the scripted passes give the flat logits. */
    private void mappedSlabs() {
        for (int attempt = 1; attempt <= 3; attempt++) {
            int before = failures;
            try (Backend f = flat(); KvPool pool = pool(1, 0, cfg(1, 0)); Backend p = paged(pool)) {
                boolean same = true;
                for (int[] op : OPS) {
                    boolean all = op[1] == 6;
                    same &= Arrays.equals(step(f, op[0], op[1], all), step(p, op[0], op[1], all));
                }
                int[] counts = CpuBackend.slabCounts(pool);
                boolean mapped = counts != null && counts[1] > 0;
                if ((same && mapped) || attempt == 3) {
                    check(mapped, "slabs past the hot budget are memory-mapped files (" + (counts == null ? "?" : counts[0] + " in memory, "
                            + counts[1] + " mapped") + ")");
                    check(same, "prefill, decode, rewind, verification chunk over mapped slabs: logits identical (" + OPS.length + " passes)");
                    return;
                }
            }
            failures = before;
        }
    }

    private Backend streaming(long hostBytes) {
        KvTierConfig t = new KvTierConfig(1, hostBytes, org.robomiri.jllm.kv.KvBlockCodec.Format.NATIVE, true, KvTierConfig.Stream.FORCE, dir);
        return new MetalBackend(w, CTX, BATCH, lib, dt, MoeSplit.OFF, null, t);
    }

    private static byte[] saved(Backend b, int n) throws Exception {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        b.saveState(n, Channels.newChannel(o));
        return o.toByteArray();
    }

    /** Metal: a streaming slot against the flat cache; {@code mappedLayers} > 0 puts the rows of all but one layer in mapped files. */
    private void streamed(int mappedLayers) throws Exception {
        String kind = mappedLayers == 0 ? "host rows" : "rows of most layers in mapped files";
        long oneLayer = 2L * CTX * dt.rowBytes(((Attention.Gqa) ((Decoder) w.decoder()).attention()).kvDim(),
                ((Attention.Gqa) ((Decoder) w.decoder()).attention()).kvHeads());
        for (int attempt = 1; attempt <= 3; attempt++) {
            int before = failures;
            try (Backend f = flat(); Backend s = streaming(mappedLayers == 0 ? 0 : oneLayer)) {
                boolean same = true;
                int i = 0;
                for (int[] op : OPS) {
                    boolean all = op[1] == 6;
                    same &= Arrays.equals(step(f, op[0], op[1], all), step(s, op[0], op[1], all));
                    if (!same) {
                        break;
                    }
                    i++;
                }
                if (same) {
                    byte[] fb = saved(f, 38);
                    byte[] sb = saved(s, 38);
                    boolean bytes = Arrays.equals(fb, sb);
                    if (!bytes && attempt < 3) {
                        failures = before;
                        continue;
                    }
                    check(true, "streamed (" + kind + "): prefill, decode, rewind, verification chunk: logits identical (" + OPS.length + " passes)");
                    check(bytes, "streamed (" + kind + "): saved bytes identical (" + sb.length + ")");
                    // flat bytes into a streaming slot, then on: same as the original continuing
                    try (Backend g = streaming(0); Backend h = flat()) {
                        g.loadState(38, Channels.newChannel(new java.io.ByteArrayInputStream(fb)));
                        float[] a = new float[vocab];
                        g.forward(tokens, 38, 1, 38, a);
                        float[] c = step(f, 38, 1, false);
                        check(Arrays.equals(a, c), "streamed (" + kind + "): a loaded cache continues like the original");
                        h.copyStateFrom(s, 38);
                        float[] d = new float[vocab];
                        h.forward(tokens, 38, 1, 38, d);
                        check(Arrays.equals(d, c), "a flat slot copied from the streamed one continues like the original");
                    }
                    return;
                }
                if (attempt == 3) {
                    check(false, "streamed (" + kind + "): logits differ at op " + (i + 1));
                    return;
                }
            }
            failures = before;
        }
    }
}
