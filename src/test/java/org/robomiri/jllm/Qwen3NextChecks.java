package org.robomiri.jllm;

import org.robomiri.jllm.cpu.Qwen3NextCpuBackend;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.Qwen3NextMetalBackend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.Qwen3NextArch;
import org.robomiri.jllm.model.ShardProvider;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Qwen3-Next checks on tiny random checkpoints ({@code make-tiny-model.py qwen3next*}), with Q8_0 weights:
 * <ol>
 *   <li>the last-token logits of the 12-token ArchSmoke prompt and the 8 greedy tokens after it match
 *       {@code scripts/qwen3next-ref.py} (an independent float64 numpy reference written after transformers'
 *       modeling_qwen3_next.py, run with {@code REF_Q8=1} so its matrices are the engine's Q8_0 values): CPU, and Metal
 *       when available;</li>
 *   <li>the chunked delta rule equals the token loop: a 150-token prompt in one call with chunks of 64 (two full
 *       chunks and a partial one) and of 7, the token loop over the same call, and 150 single-token calls give the same
 *       logits to float rounding;</li>
 *   <li>the saved state (KV rows of the full-attention layers + conv and recurrent states) loads and copies exactly:
 *       continuing after a load or a copy gives bit-identical logits; a state cannot be saved at another length;
 *       the CPU's file loads into the Metal backend, for f32 and f16 KV caches.</li>
 * </ol>
 * usage: {@code Qwen3NextChecks <metal-lib|-> <model-dir>...}; each dir needs {@code ref-logits.json}
 * ({@code REF_Q8=1 python3 scripts/qwen3next-ref.py <dir>}). Exits with status 1 on any failure.
 */
public final class Qwen3NextChecks {

    private static final int[] PROMPT = {1, 7, 42, 99, 13, 5, 250, 8, 77, 31, 600, 9};
    private static final int LONG = 150;

    public static void main(String[] args) throws Exception {
        String lib = args[0].equals("-") ? null : args[0];
        boolean metal = Metal.available(lib);
        int failures = 0;
        for (int i = 1; i < args.length; i++) {
            failures += check(Path.of(args[i]), lib, metal);
        }
        if (failures > 0) {
            System.out.println("FAILED: " + failures);
            System.exit(1);
        }
        System.out.println("qwen3-next checks passed");
    }

    private interface Factory {
        Backend open(ModelWeights w, int ctx, int batch, KvDtype kv);
    }

    static int check(Path dir, String lib, boolean metal) throws Exception {
        Qwen3NextArch cfg = (Qwen3NextArch) Architecture.load(dir);
        var ref = Json.parseObject(Files.readString(dir.resolve("ref-logits.json")));
        if (!Json.bool(ref, "q8", false)) {
            throw new IllegalStateException(dir + "/ref-logits.json was not written with REF_Q8=1");
        }
        float[] expect = floats(Json.list(ref, "logits"));
        int[] greedy = Json.list(ref, "greedy").stream().mapToInt(o -> ((Number) o).intValue()).toArray();
        String name = dir.getFileName().toString();
        System.out.println(name + "  " + cfg);
        int bad = 0;
        try (ModelWeights w = ModelWeights.load(dir, cfg, QType.Q8_0, ShardProvider.local(dir))) {
            Factory cpu = (ww, ctx, batch, kv) -> new Qwen3NextCpuBackend(ww, ctx, batch, 4, kv);
            Factory gpu = (ww, ctx, batch, kv) -> new Qwen3NextMetalBackend(ww, ctx, batch, lib, kv);
            bad += reference(name + " cpu", w, cpu, expect, greedy);
            bad += chunking(name, w);
            bad += states(name + " cpu", w, cpu, null);
            if (metal) {
                bad += reference(name + " metal", w, gpu, expect, greedy);
                bad += metalChunks(name, w, gpu);
                bad += states(name + " metal", w, gpu, cpu);
            }
        }
        return bad;
    }

    /** Last-token logits within 1e-3 of the logit range of the reference, and the reference's greedy tokens. */
    private static int reference(String label, ModelWeights w, Factory f, float[] expect, int[] greedy) {
        int vocab = w.arch.core().vocabSize();
        float[] logits = new float[vocab];
        int[] got = new int[greedy.length];
        try (Backend b = f.open(w, 64, 16, KvDtype.F32)) {
            b.forward(PROMPT, 0, PROMPT.length, 0, logits);
            float rel = maxDiff(expect, logits) / maxAbs(expect);
            int pos = PROMPT.length;
            float[] step = logits.clone();
            for (int i = 0; i < got.length; i++) {
                got[i] = argmax(step);
                if (i + 1 < got.length) {
                    b.forward(new int[]{got[i]}, 0, 1, pos++, step);
                }
            }
            boolean ok = rel <= 1e-3f && Arrays.equals(got, greedy);
            System.out.printf("  %s vs reference: max|ref-engine| / max|ref| = %.2e, greedy %s %s -> %s%n", label, rel,
                    Arrays.toString(got), Arrays.equals(got, greedy) ? "== reference" : "!= reference " + Arrays.toString(greedy),
                    ok ? "ok" : "MISMATCH");
            return ok ? 0 : 1;
        }
    }

    private static int[] longPrompt(int vocab) {
        Random r = new Random(42);
        int[] t = new int[LONG];
        for (int i = 0; i < t.length; i++) {
            t[i] = r.nextInt(vocab);
        }
        return t;
    }

    /** CPU: chunked (64 / 7) vs token loop in one call vs single-token calls over a 150-token prompt. */
    private static int chunking(String name, ModelWeights w) {
        int vocab = w.arch.core().vocabSize();
        int[] t = longPrompt(vocab);
        float[] c64 = new float[vocab];
        float[] c7 = new float[vocab];
        float[] loop = new float[vocab];
        float[] single = new float[vocab];
        try (Qwen3NextCpuBackend b = new Qwen3NextCpuBackend(w, 256, LONG, 4, KvDtype.F32)) {
            b.setRecurrence(true, 64);
            b.forward(t, 0, LONG, 0, c64);
            b.setRecurrence(true, 7);
            b.forward(t, 0, LONG, 0, c7);
            b.setRecurrence(false, 1);
            b.forward(t, 0, LONG, 0, loop);
            for (int i = 0; i < LONG; i++) {
                b.forward(t, i, 1, i, i == LONG - 1 ? single : null);
            }
        }
        float s = maxAbs(loop);
        float d64 = maxDiff(loop, c64) / s;
        float d7 = maxDiff(loop, c7) / s;
        float d1 = maxDiff(loop, single) / s;
        boolean ok = d64 <= 1e-4f && d7 <= 1e-4f && d1 <= 1e-4f;
        System.out.printf("  %s cpu %d tokens, relative to the token loop: chunks of 64 %.1e, chunks of 7 %.1e, one call per"
                + " token %.1e -> %s%n", name, LONG, d64, d7, d1, ok ? "ok" : "CHUNKING MISMATCH");
        return ok ? 0 : 1;
    }

    /** Metal: the 150-token prompt in one call, in chunks of 16 and token by token. */
    private static int metalChunks(String name, ModelWeights w, Factory f) {
        int vocab = w.arch.core().vocabSize();
        int[] t = longPrompt(vocab);
        float[] one = new float[vocab];
        float[] sixteen = new float[vocab];
        float[] single = new float[vocab];
        float[] cpu = new float[vocab];
        try (Backend b = f.open(w, 256, LONG, KvDtype.F32)) {
            b.forward(t, 0, LONG, 0, one);
            for (int i = 0; i < LONG; i += 16) {
                int n = Math.min(16, LONG - i);
                b.forward(t, i, n, i, i + n == LONG ? sixteen : null);
            }
            for (int i = 0; i < LONG; i++) {
                b.forward(t, i, 1, i, i == LONG - 1 ? single : null);
            }
        }
        try (Backend b = new Qwen3NextCpuBackend(w, 256, LONG, 4, KvDtype.F32)) {
            b.forward(t, 0, LONG, 0, cpu);
        }
        float s = maxAbs(one);
        float d16 = maxDiff(one, sixteen) / s;
        float d1 = maxDiff(one, single) / s;
        float dc = maxDiff(one, cpu) / s;
        boolean ok = d16 <= 1e-4f && d1 <= 1e-4f && dc <= 1e-3f;
        System.out.printf("  %s metal %d tokens: chunks of 16 %.1e, token by token %.1e, vs cpu %.1e -> %s%n", name, LONG,
                d16, d1, dc, ok ? "ok" : "MISMATCH");
        return ok ? 0 : 1;
    }

    /**
     * Saved, loaded and copied states continue with bit-identical logits; saving at a length the state does not hold
     * is refused; with {@code cpu} given (the Metal run), the CPU backend's file has the same format and loads here.
     */
    private static int states(String label, ModelWeights w, Factory f, Factory cpu) throws Exception {
        int vocab = w.arch.core().vocabSize();
        int n = PROMPT.length;
        int[] next = {PROMPT[3], PROMPT[5]};
        int bad = 0;
        for (KvDtype dt : KvDtype.values()) {
            float[] want = new float[vocab];
            float[] got = new float[vocab];
            try (Backend a = f.open(w, 64, 16, dt); Backend b = f.open(w, 64, 16, dt); Backend c = f.open(w, 64, 16, dt)) {
                a.forward(PROMPT, 0, n, 0, null);
                var bytes = new ByteArrayOutputStream();
                a.saveState(n, Channels.newChannel(bytes));
                boolean refused = false;
                try {
                    a.saveState(n - 1, Channels.newChannel(new ByteArrayOutputStream()));
                } catch (IllegalStateException e) {
                    refused = true;
                }
                b.loadState(n, Channels.newChannel(new ByteArrayInputStream(bytes.toByteArray())));
                c.copyStateFrom(a, n);
                a.forward(next, 0, 2, n, want);
                b.forward(next, 0, 2, n, got);
                boolean loaded = Arrays.equals(want, got);
                c.forward(next, 0, 2, n, got);
                boolean copied = Arrays.equals(want, got);
                String cross = "";
                boolean crossOk = true;
                if (cpu != null) {
                    try (Backend k = cpu.open(w, 64, 16, dt)) {
                        k.forward(PROMPT, 0, n, 0, null);
                        var cb = new ByteArrayOutputStream();
                        k.saveState(n, Channels.newChannel(cb));
                        boolean same = k.stateFormat().equals(b.stateFormat()) && cb.size() == bytes.size();
                        b.loadState(n, Channels.newChannel(new ByteArrayInputStream(cb.toByteArray())));
                        b.forward(next, 0, 2, n, got);
                        float d = maxDiff(want, got) / maxAbs(want);
                        crossOk = same && d <= 1e-3f;
                        cross = String.format(", cpu file in metal %s (%.1e)", crossOk ? "ok" : "MISMATCH", d);
                    }
                }
                boolean ok = loaded && copied && refused && crossOk;
                System.out.printf("  %s state, kv %s (%d bytes): saved+loaded %s, copied %s, wrong length refused=%b%s -> %s%n",
                        label, dt, bytes.size(), loaded ? "identical" : "DIFFERENT", copied ? "identical" : "DIFFERENT",
                        refused, cross, ok ? "ok" : "FAILED");
                bad += ok ? 0 : 1;
            }
        }
        return bad;
    }

    private static float[] floats(List<Object> l) {
        float[] a = new float[l.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = ((Number) l.get(i)).floatValue();
        }
        return a;
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    static float maxDiff(float[] a, float[] b) {
        float d = 0;
        for (int i = 0; i < a.length; i++) {
            d = Math.max(d, Math.abs(a[i] - b[i]));
        }
        return d;
    }

    static float maxAbs(float[] a) {
        float m = 0;
        for (float v : a) {
            m = Math.max(m, Math.abs(v));
        }
        return m;
    }
}
