package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.AttnWeights;
import org.robomiri.jllm.model.DecoderLayer;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.LayerWeights;
import org.robomiri.jllm.model.MambaLayer;
import org.robomiri.jllm.model.MlpWeights;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;

/**
 * Start-up self-check of the Metal backend's newer kernels on the device it runs on: each candidate is run next to the
 * established kernel it replaces on synthetic data of the model's shapes, and is used only if its output agrees
 * (relative error below 1e-3 of the output's magnitude) and, on a real GPU, if it is faster. A kernel that this
 * driver or chip gets wrong (or slow) therefore never reaches a model's output, and the choice adapts to the chip (an
 * M1 and an M4 differ in which tile shapes pay). The decisions are cached per device, kernel source and model shape
 * in {@code ~/.cache/jllm/metal-tuning.properties}, so the check runs once (a few hundred milliseconds).
 *
 * <p>Candidates: {@code attention_fa} for prefill chunks (instead of {@code attention_t}) and flash decoding
 * ({@code attention_split} + {@code attention_combine}) for long decode steps. {@code -Djllm.metal.flashPrefill=
 * true|false} and {@code -Djllm.metal.flashDecode=true|false} force a choice; on the CPU emulator (no meaningful
 * timings) correctness alone decides.
 */
final class MetalTuner {

    /** What the check decided, and a line for the log. */
    record Decision(boolean flashPrefill, boolean flashDecode, String note) {
    }

    private static final Path CACHE = Path.of(System.getProperty("user.home"), ".cache", "jllm", "metal-tuning.properties");

    private MetalTuner() {
    }

    /**
     * Also sets {@link MetalKernels#flashPrefillKernel}: of the two flash prefill forms ({@code attention_fa}, and
     * {@code attention_fb} with K and V staged together) the faster of those that agree with {@code attention_t} (on the
     * emulator, where timings mean nothing, {@code attention_fb} when it agrees, so the newer kernel is the one exercised).
     * {@code -Djllm.metal.flashPrefillKernel=fa|fb} forces the form.
     */
    static Decision decide(MetalKernels k, Attention.Gqa g, KvDtype kv, String attnKernel, int attnGroup) {
        String forcedPrefill = MetalKernels.FLASH_PREFILL;
        String forcedDecode = MetalKernels.FLASH_DECODE_MODE;
        String forcedForm = System.getProperty("jllm.metal.flashPrefillKernel", "auto");
        boolean prefillFits = MetalKernels.flashPrefillFits(g.headDim());
        if (!"auto".equals(forcedPrefill) && !"auto".equals(forcedDecode)) {
            k.flashPrefillKernel = "fb".equals(forcedForm) ? "attention_fb" : "attention_fa";
            return new Decision(Boolean.parseBoolean(forcedPrefill) && prefillFits, Boolean.parseBoolean(forcedDecode),
                    "forced");
        }
        String key = (k.metal.deviceName() + "|" + k.metal.sourceHash + "|hd" + g.headDim() + "|g"
                + g.heads() / g.kvHeads() + "|" + kv).replace(' ', '_').replace('=', '_').replace(':', '_');
        Properties cache = load();
        String cached = cache.getProperty(key);
        boolean[] got;
        boolean fb;
        String note;
        if (cached != null) {
            got = new boolean[]{cached.contains("flashPrefill=1"), cached.contains("flashDecode=1")};
            fb = cached.contains("fb=1");
            note = "cached " + cached;
        } else {
            StringBuilder sb = new StringBuilder();
            Result fa = prefillFits ? check(k, g, kv, attnKernel, attnGroup, true, "attention_fa", sb) : Result.NO;
            Result fbr = prefillFits ? check(k, g, kv, attnKernel, attnGroup, true, "attention_fb", sb) : Result.NO;
            got = new boolean[]{fa.use() || fbr.use(), check(k, g, kv, attnKernel, attnGroup, false, null, sb).use()};
            fb = fbr.use() && (!fa.use() || Metal.EMULATOR || fbr.ms() < fa.ms());
            note = sb.toString().trim();
            cache.setProperty(key, "flashPrefill=" + (got[0] ? 1 : 0) + ",flashDecode=" + (got[1] ? 1 : 0)
                    + ",fb=" + (fb ? 1 : 0));
            store(cache);
        }
        if (!"auto".equals(forcedForm)) {
            fb = "fb".equals(forcedForm);
        }
        k.flashPrefillKernel = fb ? "attention_fb" : "attention_fa";
        boolean prefill = "auto".equals(forcedPrefill) ? got[0] : Boolean.parseBoolean(forcedPrefill) && prefillFits;
        boolean decode = "auto".equals(forcedDecode) ? got[1] : Boolean.parseBoolean(forcedDecode);
        return new Decision(prefill, decode, note + (prefill ? " [" + k.flashPrefillKernel + "]" : ""));
    }

    /** A candidate's verdict: use it (agrees, and faster than the reference on a GPU), and its best time in ms. */
    private record Result(boolean use, double ms) {
        static final Result NO = new Result(false, Double.MAX_VALUE);
    }

    /**
     * Runs candidate and reference on the same synthetic data. {@code faKernel}: the flash prefill form to check
     * ({@code attention_fa} / {@code attention_fb}; null for flash decoding).
     */
    private static Result check(MetalKernels k, Attention.Gqa g, KvDtype kv, String attnKernel, int attnGroup, boolean prefill,
                                String faKernel, StringBuilder log) {
        boolean emu = Metal.EMULATOR;
        int hd = g.headDim();
        int heads = g.heads();
        int kvHeads = g.kvHeads();
        int kvDim = g.kvDim();
        int qDim = g.qDim();
        int keys = emu ? (prefill ? 96 : 600) : 2048;
        int n = prefill ? (emu ? 40 : 128) : 1;
        int pos = keys - n;
        Random rnd = new Random(11);
        Metal.Buffer q = k.scratch((long) n * qDim * 4);
        int stride = kv.rowStride(kvDim, kvHeads); // cache row stride in elements (q8: bytes, with the scales)
        Metal.Buffer kc = k.scratch((long) keys * kv.rowBytes(kvDim, kvHeads));
        Metal.Buffer vc = k.scratch((long) keys * kv.rowBytes(kvDim, kvHeads));
        fill(q.contents(), 1, KvDtype.F32, 1, n * qDim, rnd);
        fill(kc.contents(), keys, kv, kvHeads, kvDim, rnd);
        fill(vc.contents(), keys, kv, kvHeads, kvDim, rnd);
        Metal.Buffer ref = k.scratch((long) n * qDim * 4);
        Metal.Buffer cand = k.scratch((long) n * qDim * 4);
        Metal.Buffer scores = k.scratch((long) Math.min(MetalConstants.ATT_TILE, n) * heads * keys * 4);
        Metal.Buffer part = prefill ? null : k.scratch((long) heads * MetalKernels.FLASH_SPLITS * (hd + 2) * 4);
        float scale = (float) (1 / Math.sqrt(hd));
        Runnable reference = () -> k.attention(attnKernel, attnGroup, q, kc, vc, ref, scores, heads, kvHeads, hd, hd, pos, keys,
                scale, qDim, stride, stride, qDim, 0, 0f, n);
        Runnable candidate = prefill
                ? () -> k.flashPrefill(kv, hd, q, kc, vc, cand, heads, kvHeads, pos, n, scale, qDim, stride, qDim, 0, 0f)
                : () -> k.flashAttention(kv, q, kc, vc, part, scores, cand, heads, kvHeads, hd, pos, keys, scale, qDim,
                stride, 0, 0f);
        String what = prefill ? "flash prefill " + faKernel : "flash decode";
        String keep = k.flashPrefillKernel;
        if (prefill) {
            k.flashPrefillKernel = faKernel;
        }
        try {
            double tRef = time(k, reference, emu ? 1 : 5);
            double tCand = time(k, candidate, emu ? 1 : 5);
            float max = 0;
            float diff = 0;
            for (int i = 0; i < n * qDim; i++) {
                float a = ref.contents().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float b = cand.contents().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                max = Math.max(max, Math.abs(a));
                diff = Float.isNaN(b) ? Float.POSITIVE_INFINITY : Math.max(diff, Math.abs(a - b));
            }
            boolean correct = diff <= 1e-3f * max + 1e-5f;
            boolean faster = emu || tCand < 0.95 * tRef;
            log.append(String.format(java.util.Locale.ROOT, " %s: %s, max diff %.1e%s;", what, correct ? "agrees" : "DISAGREES",
                    diff, emu ? "" : String.format(java.util.Locale.ROOT, ", %.2f vs %.2f ms", tCand, tRef)));
            return new Result(correct && faster, tCand);
        } catch (RuntimeException e) { // a kernel the device cannot compile or run
            log.append(" ").append(what).append(": failed (").append(e.getMessage()).append(");");
            return Result.NO;
        } finally {
            k.flashPrefillKernel = keep;
        }
    }

    /**
     * Whether the decode attention {@code attention_dec} runs on this device: it must agree with {@code attention_t}
     * (relative error below 1e-3 of the output's magnitude, on synthetic data of the model's head shape) and be at least
     * 3 % faster over a long context (2048 keys) and not slower over a short one (400). On the emulator (no meaningful
     * timings) agreement alone decides. {@code -Djllm.metal.decodeAttention=true|false} forces it (an explicit
     * {@code -Djllm.metal.flashDecode=true} turns it off: the user asked for the other split kernel). Cached per device,
     * kernel source, head shape, cache type and chunk like the other decisions.
     */
    static boolean decodeAttention(MetalKernels k, Attention.Gqa g, KvDtype kv, String attnKernel, int attnGroup) {
        String forced = MetalKernels.DEC_ATTN_MODE;
        if (!"auto".equals(forced)) {
            return Boolean.parseBoolean(forced);
        }
        if ("true".equals(MetalKernels.FLASH_DECODE_MODE)) {
            return false;
        }
        String key = (k.metal.deviceName() + "|" + k.metal.sourceHash + "|decattn|hd" + g.headDim() + "|" + g.heads() + "x"
                + g.kvHeads() + "|" + kv + "|c" + k.decChunk).replace(' ', '_').replace('=', '_').replace(':', '_');
        Properties cache = load();
        String cached = cache.getProperty(key);
        if (cached != null) {
            return cached.equals("1");
        }
        boolean use = false;
        StringBuilder log = new StringBuilder();
        boolean emu = Metal.EMULATOR;
        int hd = g.headDim();
        int heads = g.heads();
        int kvHeads = g.kvHeads();
        int kvDim = g.kvDim();
        int qDim = g.qDim();
        int keys = emu ? 600 : 2048;
        try {
            Random rnd = new Random(13);
            Metal.Buffer q = k.scratch((long) qDim * 4);
            int stride = kv.rowStride(kvDim, kvHeads);
            Metal.Buffer kc = k.scratch((long) keys * kv.rowBytes(kvDim, kvHeads));
            Metal.Buffer vc = k.scratch((long) keys * kv.rowBytes(kvDim, kvHeads));
            fill(q.contents(), 1, KvDtype.F32, 1, qDim, rnd);
            fill(kc.contents(), keys, kv, kvHeads, kvDim, rnd);
            fill(vc.contents(), keys, kv, kvHeads, kvDim, rnd);
            Metal.Buffer ref = k.scratch((long) qDim * 4);
            Metal.Buffer cand = k.scratch((long) qDim * 4);
            Metal.Buffer scores = k.scratch((long) heads * keys * 4);
            Metal.Buffer part = k.scratch((long) heads * MetalConstants.FLASH_MAX_SPLITS * (hd + 2) * 4);
            float scale = (float) (1 / Math.sqrt(hd));
            double[] tRef = new double[2];
            double[] tCand = new double[2];
            float max = 0;
            float diff = 0;
            for (int pass = 0; pass < 2; pass++) { // pass 0: the long context, 1: a short one (400 keys)
                int pos = pass == 0 ? keys - 1 : Math.min(keys, emu ? 100 : 400) - 1;
                Runnable reference = () -> k.attention(attnKernel, attnGroup, q, kc, vc, ref, scores, heads, kvHeads, hd, hd,
                        pos, keys, scale, qDim, stride, stride, qDim, 0, 0f, 1);
                Runnable candidate = () -> k.decodeAttention(kv, q, 0, kc, vc, part, cand, 0, heads, kvHeads, hd, pos, keys,
                        scale, stride, 0, 0f);
                int reps = emu ? 1 : 20;
                tRef[pass] = time(k, () -> {
                    for (int i = 0; i < reps; i++) {
                        reference.run();
                    }
                }, emu ? 1 : 5) / reps;
                tCand[pass] = time(k, () -> {
                    for (int i = 0; i < reps; i++) {
                        candidate.run();
                    }
                }, emu ? 1 : 5) / reps;
                for (int i = 0; i < qDim; i++) {
                    float a = ref.contents().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    float b = cand.contents().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    max = Math.max(max, Math.abs(a));
                    diff = Float.isNaN(b) ? Float.POSITIVE_INFINITY : Math.max(diff, Math.abs(a - b));
                }
            }
            boolean correct = diff <= 1e-3f * max + 1e-5f;
            use = correct && (emu || (tCand[0] < 0.97 * tRef[0] && tCand[1] <= 1.03 * tRef[1]));
            log.append(String.format(java.util.Locale.ROOT, "decode attention %s, max diff %.1e%s", correct ? "agrees" : "DISAGREES",
                    diff, emu ? "" : String.format(java.util.Locale.ROOT, ", %.3f vs %.3f ms (%d keys), %.3f vs %.3f ms (400 keys)",
                            tCand[0], tRef[0], keys, tCand[1], tRef[1])));
        } catch (RuntimeException e) { // a kernel the device cannot compile or run
            log.append("decode attention failed (").append(e.getMessage()).append(")");
        }
        System.err.println("Metal decode attention: " + (use ? "on" : "off") + " (" + log + ")");
        cache.setProperty(key, use ? "1" : "0");
        store(cache);
        return use;
    }

    /**
     * Threads per threadgroup for the plain single-token matmul ({@link MetalKernels#decodeThreads}): 64, 128 and 256
     * timed on the model's own matrix {@code t} (an output projection: the matrices this kernel serves). The results
     * are bit-identical for every size (checked), so only speed decides; the emulator keeps the default. Cached per
     * device and matrix shape like the other decisions; {@code -Djllm.metal.decodeThreads=N} forces it.
     */
    static int decodeThreads(MetalKernels k, org.robomiri.jllm.model.QTensor t, Metal.Buffer buf) {
        Integer forced = Integer.getInteger("jllm.metal.decodeThreads");
        if (forced != null && (forced == 64 || forced == 128 || forced == 256)) {
            return forced;
        }
        if (Metal.EMULATOR) {
            return MetalConstants.MM_THREADS;
        }
        String key = (k.metal.deviceName() + "|" + k.metal.sourceHash + "|mv" + t.type + "|" + t.rows + "x" + t.cols)
                .replace(' ', '_').replace('=', '_').replace(':', '_');
        Properties cache = load();
        String cached = cache.getProperty(key);
        if (cached != null) {
            return Integer.parseInt(cached);
        }
        Metal.Buffer x = k.scratch((long) t.cols * 4);
        fill(x.contents(), 1, KvDtype.F32, 1, t.cols, new Random(5));
        Metal.Buffer ref = k.scratch((long) t.rows * 4);
        Metal.Buffer out = k.scratch((long) t.rows * 4);
        int best = MetalConstants.MM_THREADS;
        double bestMs = Double.MAX_VALUE;
        StringBuilder log = new StringBuilder();
        try {
            for (int threads : new int[]{128, 64, 256}) { // the default first: the reference for the comparison
                k.decodeThreads = threads;
                Metal.Buffer dst = threads == MetalConstants.MM_THREADS ? ref : out;
                Runnable r = () -> k.matmul(t, buf, x, 0, dst, 0, 1, false, null, 0f);
                double ms = time(k, () -> {
                    for (int i = 0; i < 20; i++) {
                        r.run();
                    }
                }, 5);
                boolean same = dst == ref || dst.contents().mismatch(ref.contents()) == -1;
                log.append(String.format(java.util.Locale.ROOT, " %d: %.3f ms%s;", threads, ms / 20, same ? "" : " DIFFERS"));
                if (same && ms < 0.97 * bestMs) {
                    bestMs = ms;
                    best = threads;
                }
            }
        } catch (RuntimeException e) {
            best = MetalConstants.MM_THREADS;
            log.append(" failed (").append(e.getMessage()).append(")");
        } finally {
            k.decodeThreads = MetalConstants.MM_THREADS;
        }
        System.err.println("Metal decode matmul threadgroup: " + best + " threads (" + log.toString().trim() + ")");
        cache.setProperty(key, Integer.toString(best));
        store(cache);
        return best;
    }

    // ---------------------------------------------------------------- half-weight MMA prefill matmuls

    /**
     * One prefill matrix shape to decide: the family, its tensors (matmul3: q, k, v; MoE: the stacked experts) and the
     * {@link MetalKernels#hmmaKey keys} the decision is stored under (a matmul3 whose three tensors differ in type is
     * dispatched as three parts, one key each).
     */
    private record Probe(MetalKernels.MmaFamily f, QTensor[] t, int rows, List<String> keys) {
    }

    /**
     * For every distinct prefill matrix shape of the model (the q/k/v projection, o_proj, gate/up, down, the experts'
     * gate/up and down, MLA's and Mamba's projections), runs the f32 {@code *_mma} kernel and the half-weight
     * {@code *_hmma} kernel on the model's own weights with random activations of a prefill chunk, and keeps the
     * half-weight one when it agrees ({@code max |hmma - mma| <= 1e-3 max |mma| + 1e-5}) and is at least 3 % faster. On
     * the emulator (no meaningful timings) the agreement is still checked and logged but the more accurate f32 form is
     * kept, unlike the attention choices, whose candidates lose no accuracy. Decisions go to {@link MetalKernels#hmmaChoice} and are cached per device, kernel source and shape like
     * the others. Does nothing when {@code -Djllm.metal.hmma} forces a form or the MMA prefill is off.
     */
    static void prefillMatmuls(MetalKernels k, ModelWeights w) {
        if (!MetalKernels.MMA_PREFILL || !"auto".equals(MetalKernels.HMMA)) {
            return;
        }
        Properties cache = load();
        StringBuilder log = new StringBuilder();
        int on = 0;
        int total = 0;
        boolean changed = false;
        for (Probe pr : probes(w)) {
            String ck = (k.metal.deviceName() + "|" + k.metal.sourceHash + "|hmma|" + pr.keys().getFirst())
                    .replace(' ', '_').replace('=', '_').replace(':', '_');
            String cached = cache.getProperty(ck);
            boolean use;
            if (cached != null) {
                use = cached.equals("1");
            } else {
                use = checkHmma(k, pr, log);
                cache.setProperty(ck, use ? "1" : "0");
                changed = true;
            }
            for (String key : pr.keys()) {
                k.hmmaChoice.put(key, use);
            }
            on += use ? 1 : 0;
            total++;
        }
        if (changed) {
            store(cache);
        }
        if (total > 0) {
            System.err.println("Metal prefill matmuls: half-weight MMA for " + on + " of " + total + " shapes"
                    + (changed ? " (" + log.toString().trim() + ")" : " (cached)"));
        }
    }

    /** The distinct prefill matmul shapes of the model's layers. */
    private static List<Probe> probes(ModelWeights w) {
        Map<String, Probe> m = new LinkedHashMap<>();
        for (LayerWeights lw : w.layers) {
            switch (lw) {
                case DecoderLayer L -> {
                    switch (L.attn()) {
                        case AttnWeights.Gqa g -> {
                            QTensor[] qkv = {g.wq(), g.wk(), g.wv()};
                            if (g.wq().type == g.wk().type && g.wk().type == g.wv().type) {
                                int rows = g.wq().rows + g.wk().rows + g.wv().rows;
                                add(m, MetalKernels.MmaFamily.MATMUL3, qkv, rows,
                                        List.of(MetalKernels.hmmaKey(MetalKernels.MmaFamily.MATMUL3, g.wq().type, rows, g.wq().cols)));
                            } else {
                                List<String> keys = new ArrayList<>();
                                for (QTensor t : qkv) {
                                    keys.add(MetalKernels.hmmaKey(MetalKernels.MmaFamily.MATMUL3, t.type, t.rows, t.cols));
                                }
                                add(m, MetalKernels.MmaFamily.MATMUL3, qkv, 0, keys);
                            }
                        }
                        case AttnWeights.Mla a -> {
                            switch (a.query()) {
                                case AttnWeights.Mla.Proj p -> matmul(m, p.w());
                                case AttnWeights.Mla.Lora q -> {
                                    matmul(m, q.a());
                                    matmul(m, q.b());
                                }
                            }
                            matmul(m, a.kvA());
                            matmul(m, a.kPe());
                        }
                    }
                    matmul(m, L.wo());
                    switch (L.mlp()) {
                        case MlpWeights.Dense d -> dense(m, d);
                        case MlpWeights.Moe e -> {
                            int dim = w.arch.core().dim();
                            int moeDim = e.expDown().cols;
                            add(m, MetalKernels.MmaFamily.MOE_UP, new QTensor[]{e.expGateUp()}, moeDim, List.of(
                                    MetalKernels.hmmaKey(MetalKernels.MmaFamily.MOE_UP, e.expGateUp().type, moeDim, e.expGateUp().cols)));
                            add(m, MetalKernels.MmaFamily.MOE_DOWN, new QTensor[]{e.expDown()}, dim, List.of(
                                    MetalKernels.hmmaKey(MetalKernels.MmaFamily.MOE_DOWN, e.expDown().type, dim, e.expDown().cols)));
                            if (e.shared() != null) {
                                dense(m, e.shared());
                            }
                        }
                    }
                }
                case MambaLayer ml -> {
                    matmul(m, ml.inProj());
                    matmul(m, ml.xProj());
                    matmul(m, ml.outProj());
                }
                case org.robomiri.jllm.model.Qwen3NextLayer q -> { // its own backend (Qwen3NextMetalBackend) does not use the tuned prefill matmuls
                }
            }
        }
        return new ArrayList<>(m.values());
    }

    private static void dense(Map<String, Probe> m, MlpWeights.Dense d) {
        int hidden = d.gateUp().rows / 2;
        add(m, MetalKernels.MmaFamily.SWIGLU, new QTensor[]{d.gateUp()}, hidden,
                List.of(MetalKernels.hmmaKey(MetalKernels.MmaFamily.SWIGLU, d.gateUp().type, hidden, d.gateUp().cols)));
        matmul(m, d.down());
    }

    private static void matmul(Map<String, Probe> m, QTensor t) {
        add(m, MetalKernels.MmaFamily.MATMUL, new QTensor[]{t}, t.rows,
                List.of(MetalKernels.hmmaKey(MetalKernels.MmaFamily.MATMUL, t.type, t.rows, t.cols)));
    }

    private static void add(Map<String, Probe> m, MetalKernels.MmaFamily f, QTensor[] t, int rows, List<String> keys) {
        m.putIfAbsent(String.join(",", keys), new Probe(f, t, rows, keys));
    }

    /** Both forms on the probe's weights; true if the half-weight one agrees and (on a GPU) is faster. */
    private static boolean checkHmma(MetalKernels k, Probe pr, StringBuilder log) {
        boolean emu = Metal.EMULATOR;
        int n = emu ? 37 : 128; // tokens (MoE: slots); 37 = one full and one partial tile
        Random rnd = new Random(7);
        List<Metal.Buffer> bufs = new ArrayList<>();
        QTensor t0 = pr.t()[0];
        String what = pr.f() + " " + t0.type + " " + pr.rows() + "x" + t0.cols;
        try {
            Metal.Buffer[] w = new Metal.Buffer[pr.t().length];
            for (int i = 0; i < w.length; i++) {
                w[i] = k.wrap(pr.t()[i].data);
                bufs.add(w[i]);
            }
            int cols = t0.cols;
            Metal.Buffer x = scratch(k, bufs, (long) n * cols * 4);
            fill(x.contents(), 1, KvDtype.F32, 1, n * cols, rnd);
            Runnable run;
            Metal.Buffer[] outs;
            switch (pr.f()) {
                case MATMUL -> {
                    outs = new Metal.Buffer[]{scratch(k, bufs, (long) n * t0.rows * 4)};
                    run = () -> k.matmul(t0, w[0], x, 0, outs[0], 0, n, false, null, 0f);
                }
                case SWIGLU -> {
                    outs = new Metal.Buffer[]{scratch(k, bufs, (long) n * pr.rows() * 4)};
                    run = () -> k.swiglu(t0, w[0], x, null, outs[0], false, n);
                }
                case MATMUL3 -> {
                    QTensor[] t = pr.t();
                    outs = new Metal.Buffer[]{scratch(k, bufs, (long) n * t[0].rows * 4),
                            scratch(k, bufs, (long) n * t[1].rows * 4), scratch(k, bufs, (long) n * t[2].rows * 4)};
                    run = () -> k.matmul3(t[0], w[0], t[1], w[1], t[2], w[2], x, null, outs[0], outs[1], 0, outs[2], 0, false,
                            null, null, null, n);
                }
                default -> { // MOE_UP / MOE_DOWN: n slots (topK 1) in tiles of 16 over up to 8 experts spread over the layer
                    int perExpert = pr.f() == MetalKernels.MmaFamily.MOE_UP ? 2 * pr.rows() : pr.rows();
                    int experts = t0.rows / perExpert;
                    int nTiles = (n + 15) / 16;
                    Metal.Buffer order = scratch(k, bufs, (long) n * 4);
                    Metal.Buffer tiles = scratch(k, bufs, (long) (1 + 3 * nTiles) * 4);
                    for (int s = 0; s < n; s++) {
                        order.contents().setAtIndex(ValueLayout.JAVA_INT, s, s);
                    }
                    tiles.contents().setAtIndex(ValueLayout.JAVA_INT, 0, nTiles);
                    for (int i = 0; i < nTiles; i++) {
                        tiles.contents().setAtIndex(ValueLayout.JAVA_INT, 1 + 3 * i, (int) ((long) (i % 8) * experts / Math.min(8, nTiles)) % experts);
                        tiles.contents().setAtIndex(ValueLayout.JAVA_INT, 2 + 3 * i, 16 * i);
                        tiles.contents().setAtIndex(ValueLayout.JAVA_INT, 3 + 3 * i, Math.min(16, n - 16 * i));
                    }
                    if (pr.f() == MetalKernels.MmaFamily.MOE_UP) {
                        outs = new Metal.Buffer[]{scratch(k, bufs, (long) n * pr.rows() * 4)};
                        run = () -> k.moeUpTiles(t0, w[0], x, order, tiles, outs[0], pr.rows(), 1, nTiles);
                    } else {
                        Metal.Buffer rw = scratch(k, bufs, (long) n * 4);
                        fill(rw.contents(), 1, KvDtype.F32, 1, n, rnd);
                        outs = new Metal.Buffer[]{scratch(k, bufs, (long) n * pr.rows() * 4)};
                        run = () -> k.moeDownTiles(t0, w[0], x, order, tiles, rw, outs[0], pr.rows(), 1, nTiles);
                    }
                }
            }
            int reps = emu ? 1 : 4;
            Runnable many = () -> {
                for (int i = 0; i < reps; i++) {
                    run.run();
                }
            };
            k.hmmaForce = false;
            double tOld = time(k, many, emu ? 1 : 5);
            float[][] ref = new float[outs.length][];
            for (int i = 0; i < outs.length; i++) {
                ref[i] = outs[i].contents().toArray(ValueLayout.JAVA_FLOAT);
            }
            k.hmmaForce = true;
            double tNew = time(k, many, emu ? 1 : 5);
            float max = 0;
            float diff = 0;
            for (int i = 0; i < outs.length; i++) {
                float[] got = outs[i].contents().toArray(ValueLayout.JAVA_FLOAT);
                for (int j = 0; j < got.length; j++) {
                    max = Math.max(max, Math.abs(ref[i][j]));
                    diff = Float.isNaN(got[j]) ? Float.POSITIVE_INFINITY : Math.max(diff, Math.abs(ref[i][j] - got[j]));
                }
            }
            boolean correct = diff <= 1e-3f * max + 1e-5f;
            // The half-weight form trades ~3e-4 relative accuracy for speed, so without a timing to show the speed
            // (the emulator) the f32 form stays: -Djllm.metal.hmma=true runs the half-weight kernels there.
            boolean faster = !emu && tNew < 0.97 * tOld;
            log.append(String.format(java.util.Locale.ROOT, " %s: %s, max diff %.1e%s;", what, correct ? "agrees" : "DISAGREES",
                    diff, emu ? " (emulator: no timing, f32 kept)"
                            : String.format(java.util.Locale.ROOT, ", %.3f vs %.3f ms", tNew / reps, tOld / reps)));
            return correct && faster;
        } catch (RuntimeException e) { // a kernel the device cannot compile or run
            log.append(" ").append(what).append(": failed (").append(e.getMessage()).append(");");
            return false;
        } finally {
            k.hmmaForce = null;
            for (Metal.Buffer b : bufs) {
                k.metal.release(b);
            }
        }
    }

    private static Metal.Buffer scratch(MetalKernels k, List<Metal.Buffer> bufs, long bytes) {
        Metal.Buffer b = k.scratch(bytes);
        bufs.add(b);
        return b;
    }

    /** Best of {@code reps} timed runs, milliseconds. */
    private static double time(MetalKernels k, Runnable r, int reps) {
        double best = Double.MAX_VALUE;
        for (int i = 0; i < reps; i++) {
            k.begin();
            long t0 = System.nanoTime();
            r.run();
            k.commitAndWait();
            best = Math.min(best, (System.nanoTime() - t0) / 1e6);
        }
        return best;
    }

    /** {@code rows} rows of {@code width} random values in [-1, 1) as a cache of type {@code kv} stores them. */
    private static void fill(MemorySegment seg, int rows, KvDtype kv, int heads, int width, Random rnd) {
        long rb = kv.rowBytes(width, heads);
        for (int r = 0; r < rows; r++) {
            for (int i = 0; i < width; i++) {
                float v = rnd.nextFloat() * 2 - 1;
                long o = r * rb;
                switch (kv) {
                    case F32 -> seg.set(ValueLayout.JAVA_FLOAT, o + 4L * i, v);
                    case F16 -> seg.set(ValueLayout.JAVA_SHORT, o + 2L * i, Float.floatToFloat16(v));
                    case Q8 -> seg.set(ValueLayout.JAVA_BYTE, o + i, (byte) Math.round(v * 127));
                }
            }
            for (int h = 0; kv == KvDtype.Q8 && h < heads; h++) { // per-head scales between 0.5 / 127 and 1.5 / 127
                seg.set(ValueLayout.JAVA_FLOAT_UNALIGNED, r * rb + width + 4L * h, (0.5f + rnd.nextFloat()) / 127);
            }
        }
    }

    private static Properties load() {
        Properties p = new Properties();
        if (Files.isRegularFile(CACHE)) {
            try (InputStream in = Files.newInputStream(CACHE)) {
                p.load(in);
            } catch (IOException ignored) {
                // start over
            }
        }
        return p;
    }

    private static void store(Properties p) {
        try {
            Files.createDirectories(CACHE.getParent());
            try (OutputStream out = Files.newOutputStream(CACHE)) {
                p.store(out, "jllm Metal kernel self-check: device|kernel source|head dim|group|cache type");
            }
        } catch (IOException ignored) {
            // the check simply runs again next time
        }
    }
}
