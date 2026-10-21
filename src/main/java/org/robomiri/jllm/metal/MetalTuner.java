package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.KvDtype;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
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

    static Decision decide(MetalKernels k, Attention.Gqa g, KvDtype kv, String attnKernel, int attnGroup) {
        String forcedPrefill = MetalKernels.FLASH_PREFILL;
        String forcedDecode = MetalKernels.FLASH_DECODE_MODE;
        boolean prefillFits = MetalKernels.flashPrefillFits(g.headDim());
        if (!"auto".equals(forcedPrefill) && !"auto".equals(forcedDecode)) {
            return new Decision(Boolean.parseBoolean(forcedPrefill) && prefillFits, Boolean.parseBoolean(forcedDecode),
                    "forced");
        }
        String key = (k.metal.deviceName() + "|" + k.metal.sourceHash + "|hd" + g.headDim() + "|g"
                + g.heads() / g.kvHeads() + "|" + kv).replace(' ', '_').replace('=', '_').replace(':', '_');
        Properties cache = load();
        String cached = cache.getProperty(key);
        boolean[] got;
        String note;
        if (cached != null) {
            got = new boolean[]{cached.contains("flashPrefill=1"), cached.contains("flashDecode=1")};
            note = "cached " + cached;
        } else {
            StringBuilder sb = new StringBuilder();
            got = new boolean[]{prefillFits && check(k, g, kv, attnKernel, attnGroup, true, sb),
                    check(k, g, kv, attnKernel, attnGroup, false, sb)};
            note = sb.toString().trim();
            cache.setProperty(key, "flashPrefill=" + (got[0] ? 1 : 0) + ",flashDecode=" + (got[1] ? 1 : 0));
            store(cache);
        }
        boolean prefill = "auto".equals(forcedPrefill) ? got[0] : Boolean.parseBoolean(forcedPrefill) && prefillFits;
        boolean decode = "auto".equals(forcedDecode) ? got[1] : Boolean.parseBoolean(forcedDecode);
        return new Decision(prefill, decode, note);
    }

    /** Runs candidate and reference on the same synthetic data; true if the candidate is to be used. */
    private static boolean check(MetalKernels k, Attention.Gqa g, KvDtype kv, String attnKernel, int attnGroup, boolean prefill,
                                 StringBuilder log) {
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
        String what = prefill ? "flash prefill" : "flash decode";
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
            return correct && faster;
        } catch (RuntimeException e) { // a kernel the device cannot compile or run
            log.append(" ").append(what).append(": failed (").append(e.getMessage()).append(");");
            return false;
        }
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
