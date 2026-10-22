package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.metal.MoeSplit;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Mixed passes ({@link Backend#forwardMixed}, docs/25) against the same conversations run alone, for one model of
 * {@link ArchSmoke}, on the CPU backend and, when a GPU (or the emulator) is there, on Metal; every cache type of the paged
 * cache. Three conversations run scripts of passes: a prompt in two chunks (the second with logits), a short prompt, and a
 * prompt followed by a three-token verification run (every row's logits) and decode steps. Alone, each backend runs its
 * script on its own; together, pass k carries the k-th step of every conversation, the longest segment first, with the
 * lender of the pass rotating (once a backend that owns no segment). Every segment's logits must equal the alone run's to
 * the rounding of the multi-row kernels (the bound of the batched-decode checks: 1e-4 of the largest logit), and
 * its arg max exactly; the number of bit-identical segments is reported.
 */
final class MixedPassChecks {

    private static final int CTX = 128;
    private static final int BATCH = 96;
    private static final int LEN = 80;

    /** One step of a conversation: {@code n} tokens from {@code pos}; {@code all}: every row's logits, else the last row's. */
    private record Op(int pos, int n, boolean logits, boolean all) {
    }

    private final ModelWeights w;
    private final String lib;
    private final boolean metal;
    private final KvDtype dt;
    private final MoeSplit split;
    private final String name;
    private final int vocab;
    private final int[][] tokens = new int[3][LEN];
    private int failures;

    private MixedPassChecks(ModelWeights w, String lib, boolean metal, KvDtype dt, MoeSplit split, String label, Architecture cfg) {
        this.w = w;
        this.lib = lib;
        this.metal = metal;
        this.dt = dt;
        this.split = split;
        this.vocab = cfg.core().vocabSize();
        this.name = (metal ? "metal " : "cpu ") + label;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < LEN; j++) {
                tokens[i][j] = (j * 37 + 11 * (i + 1) + i * j * 5) % Math.min(vocab, 600);
            }
        }
    }

    static int run(ModelWeights w, String lib, boolean withMetal, Architecture cfg, boolean allTypes) throws Exception {
        if (!(cfg instanceof Decoder d) || !(d.attention() instanceof Attention.Gqa g)) {
            return 0;
        }
        int failures = 0;
        KvDtype[] types = allTypes ? new KvDtype[]{KvDtype.F32, KvDtype.F16, KvDtype.Q8} : new KvDtype[]{KvDtype.F32, KvDtype.Q8};
        for (KvDtype dt : types) {
            failures += new MixedPassChecks(w, lib, false, dt, MoeSplit.OFF, dt.toString(), cfg).all();
            if (withMetal && g.headDim() % 32 == 0) {
                failures += new MixedPassChecks(w, lib, true, dt, MoeSplit.OFF, dt.toString(), cfg).all();
            }
        }
        if (org.robomiri.jllm.model.KvQ4.supports(g.headDim())) {
            KvDtype.setQ8Rotate(true); // q8 with Hadamard-rotated keys (backend.kv-q8-rotate, the shipped default)
            try {
                failures += new MixedPassChecks(w, lib, false, KvDtype.Q8, MoeSplit.OFF, "Q8 rotated keys", cfg).all();
                if (withMetal && g.headDim() % 32 == 0) {
                    failures += new MixedPassChecks(w, lib, true, KvDtype.Q8, MoeSplit.OFF, "Q8 rotated keys", cfg).all();
                }
            } finally {
                KvDtype.setQ8Rotate(false);
            }
        }
        if (withMetal && g.headDim() % 32 == 0 && d.anyMoe()) {
            failures += new MixedPassChecks(w, lib, true, KvDtype.F32, new MoeSplit(1, 4, false), "F32, experts of 1 layer on the cpu", cfg).all();
        }
        return failures;
    }

    private KvPool pool(int slots) {
        KvPool p = metal ? MetalBackend.newKvPool(w, CTX, slots, dt, 16, 0, lib) : CpuBackend.newKvPool(w, CTX, slots, dt, 16, 0);
        if (p == null) {
            throw new IllegalStateException("no pool");
        }
        return p;
    }

    private Backend backend(KvPool pool) {
        return metal ? new MetalBackend(w, CTX, BATCH, lib, dt, split, pool) : new CpuBackend(w, CTX, BATCH, 4, dt, false, pool);
    }

    private static final Op[][] SCRIPT = {
            {new Op(0, 32, false, false), new Op(32, 18, true, false), new Op(50, 1, true, false), new Op(51, 1, true, false),
                    new Op(52, 1, true, false), new Op(53, 1, true, false), new Op(54, 1, true, false), new Op(55, 1, true, false)},
            {new Op(0, 12, true, false), new Op(12, 1, true, false), new Op(13, 1, true, false), new Op(14, 1, true, false),
                    new Op(15, 1, true, false), new Op(16, 1, true, false), new Op(17, 1, true, false), new Op(18, 1, true, false)},
            {new Op(0, 9, true, false), new Op(9, 3, true, true), new Op(12, 1, true, false), new Op(13, 1, true, false),
                    new Op(14, 1, true, false), new Op(15, 1, true, false), new Op(16, 1, true, false), new Op(17, 1, true, false)}};

    /** The same conversations with other chunk boundaries. */
    private static final Op[][] ALT = {
            {new Op(0, 50, true, false), new Op(50, 1, true, false), new Op(51, 1, true, false), new Op(52, 1, true, false),
                    new Op(53, 1, true, false), new Op(54, 1, true, false), new Op(55, 1, true, false)},
            SCRIPT[1],
            {new Op(0, 9, true, false), new Op(9, 1, true, false), new Op(10, 1, true, false), new Op(11, 1, true, false),
                    new Op(12, 1, true, false), new Op(13, 1, true, false), new Op(14, 1, true, false), new Op(15, 1, true, false),
                    new Op(16, 1, true, false), new Op(17, 1, true, false)}};

    private float[][][] runAlone(Op[][] script) {
        float[][][] out = new float[3][][];
        try (KvPool pool = pool(3)) {
            for (int i = 0; i < 3; i++) {
                out[i] = new float[script[i].length][];
                try (Backend b = backend(pool)) {
                    for (int s = 0; s < script[i].length; s++) {
                        Op op = script[i][s];
                        float[] o = op.logits() ? new float[(op.all() ? op.n() : 1) * vocab] : null;
                        if (op.all()) {
                            b.forwardAll(tokens[i], op.pos(), op.n(), op.pos(), o, vocab);
                        } else {
                            b.forward(tokens[i], op.pos(), op.n(), op.pos(), o);
                        }
                        out[i][s] = o;
                    }
                }
            }
        }
        return out;
    }

    /** Every logits row by the conversation and the position of the token it follows (conversation * 1000 + position). */
    private java.util.Map<Integer, float[]> rowsByPosition(Op[][] script, float[][][] results) {
        java.util.Map<Integer, float[]> m = new java.util.HashMap<>();
        for (int i = 0; i < 3; i++) {
            for (int s = 0; s < script[i].length; s++) {
                Op op = script[i][s];
                float[] r = results[i][s];
                if (r == null) {
                    continue;
                }
                if (op.all()) {
                    for (int t = 0; t < op.n(); t++) {
                        m.put(i * 1000 + op.pos() + t, Arrays.copyOfRange(r, t * vocab, (t + 1) * vocab));
                    }
                } else {
                    m.put(i * 1000 + op.pos() + op.n() - 1, r);
                }
            }
        }
        return m;
    }

    private int all() {
        try {
            // alone: each conversation's script on its own backend of its own pool; and again with other chunk boundaries
            // (the prompt of conversation 0 in one chunk, the verification run of conversation 2 as three decode steps): how far
            // two legitimate ways of running the same conversation are from each other, whatever the batch around them
            float[][][] alone = runAlone(SCRIPT);
            float[][][] alt = runAlone(ALT);
            // together
            float[][][] together = new float[3][SCRIPT[0].length][];
            int mixedPasses = 0;
            try (KvPool pool = pool(5)) {
                Backend[] b = new Backend[4]; // three owners and one idle lender
                try {
                    for (int i = 0; i < 4; i++) {
                        b[i] = backend(pool);
                    }
                    for (int s = 0; s < SCRIPT[0].length; s++) {
                        List<Integer> convs = new ArrayList<>(List.of(0, 1, 2));
                        final int step = s;
                        convs.sort(Comparator.comparingInt((Integer i) -> -SCRIPT[i][step].n())); // the longest segment first
                        Backend.Segment[] segs = new Backend.Segment[3];
                        for (int k = 0; k < 3; k++) {
                            int i = convs.get(k);
                            Op op = SCRIPT[i][s];
                            float[] out = op.logits() ? new float[(op.all() ? op.n() : 1) * vocab] : null;
                            together[i][s] = out;
                            segs[k] = new Backend.Segment(b[i], tokens[i], op.pos(), op.n(), op.pos(), out, op.all() ? vocab : 0);
                        }
                        Backend lender = b[s == 3 ? 3 : convs.get(s % 3)]; // the pass at step 3 is led by a backend that owns no segment
                        boolean ok = lender.forwardMixed(segs);
                        if (!ok) {
                            check(false, "the backend refused a mixed pass at step " + s);
                            return failures;
                        }
                        mixedPasses++;
                    }
                } finally {
                    for (Backend x : b) {
                        if (x != null) {
                            x.close();
                        }
                    }
                }
            }
            java.util.Map<Integer, float[]> pa = rowsByPosition(SCRIPT, alone);
            java.util.Map<Integer, float[]> pt = rowsByPosition(SCRIPT, together);
            java.util.Map<Integer, float[]> pb = rowsByPosition(ALT, alt);
            float diff = 0;
            float scale = 0;
            float base = 0;
            int exact = 0;
            boolean argmax = true;
            for (var e : pa.entrySet()) {
                float[] x = pa.get(e.getKey());
                float[] y = pt.get(e.getKey());
                exact += Arrays.equals(x, y) ? 1 : 0;
                diff = Math.max(diff, maxDiff(x, y));
                scale = Math.max(scale, maxAbs(x));
                argmax &= argmax(x, 0) == argmax(y, 0);
                float[] z = pb.get(e.getKey());
                if (z != null) {
                    base = Math.max(base, maxDiff(x, z));
                }
            }
            // The bound is that of the batched-decode checks (1e-4 of the largest logit). A q8 cache can exceed it: a different
            // kernel for a different number of rows rounds a key or value a hair differently and an int8 code flips (the
            // documented hazard of the q8 and q4 caches), which any other cut of the same conversation does too; there the
            // bound is twice how far the other chunking of the same conversation is from the first.
            float bound = 1e-4f * scale + 1e-5f;
            boolean ok = argmax && (diff <= bound || (dt != KvDtype.F32 && dt != KvDtype.F16 && diff <= 2 * base));
            check(ok, String.format("%d mixed passes of a chunk, decode rows and a verification run: arg max %s, max|alone-mixed| = %.1e"
                    + " (bound %.1e; alone cut differently: %.1e), %d of %d rows bit-identical", mixedPasses,
                    argmax ? "same" : "DIFFERENT", diff, bound, base, exact, pa.size()));
        } catch (RuntimeException e) {
            e.printStackTrace(System.out);
            check(false, "threw " + e);
        }
        return failures;
    }

    private void check(boolean ok, String what) {
        System.out.println((ok ? "  ok   mixed " : "  FAIL mixed ") + name + ": " + what);
        if (!ok) {
            failures++;
        }
    }

    private static float maxDiff(float[] a, float[] b) {
        float m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    private static float maxAbs(float[] a) {
        float m = 0;
        for (float v : a) {
            m = Math.max(m, Math.abs(v));
        }
        return m;
    }

    private int argmax(float[] a, int from) {
        int best = from;
        for (int i = from; i < from + vocab; i++) {
            if (a[i] > a[best]) {
                best = i;
            }
        }
        return best - from;
    }
}
