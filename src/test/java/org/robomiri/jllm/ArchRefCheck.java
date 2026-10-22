package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.cpu.HybridCpuBackend;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.metal.HybridMetalBackend;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.HybridArch;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.QuantFile;
import org.robomiri.jllm.model.ShardProvider;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The engine against the transformers reference of {@code scripts/hf-arch-ref.py} on a tiny random model of one of the
 * long-context families (gpt-oss, Llama 4), driven by {@code scripts/hf-arch-check.sh}.
 *
 * <p>usage: {@code ArchRefCheck <model-dir> <metal-lib | -> [quant]} with the reference in {@code <model-dir>/ref}. The
 * reference ran the whole sequence (prompt plus its 12 greedy ids) with the weights rounded to Q8_0 (so the engine at q8
 * has the same weights). For the CPU backend and, with a library, the Metal backend it checks, on that sequence:
 * the logits of every position against the reference (largest difference relative to the largest logit), the argmax of
 * every position, the free-running greedy ids from the prompt, and that cutting the prompt differently (the whole prompt
 * in one chunk, chunks of 7, one token at a time) gives the same last-position logits, which exercises the sliding
 * window, the chunks and the sinks across tile and chunk boundaries. Exits 1 on any failure.
 */
public final class ArchRefCheck {

    private static final float TOL_CPU = 5e-5f;
    private static final float TOL_GPU = 1e-4f;

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        String lib = args.length > 1 && !args[1].equals("-") ? args[1] : null;
        QType quant = QType.parse(args.length > 2 ? args[2] : "q8");
        Path refDir = dir.resolve("ref");
        Map<String, Object> ref = Json.parseObject(Files.readString(refDir.resolve("ref.json")));
        int[] prompt = ints(Json.list(ref, "prompt"));
        int[] ids = ints(Json.list(ref, "ids"));
        int vocab = (int) Json.num(ref, "vocab", 0);
        float[] want = floats(refDir.resolve("logits.f32"), ids.length * vocab);
        Architecture arch = Architecture.load(dir);
        System.out.println(dir.getFileName() + ": " + arch);
        int failures = 0;
        try (ModelWeights w = ModelWeights.load(dir, arch, quant, QuantFile.DEFAULT_HEAD, ShardProvider.local(dir), null)) {
            int ctx = 64;
            try (Backend b = arch instanceof HybridArch ? new HybridCpuBackend(w, ctx, 16, 4, KvDtype.F32)
                    : new CpuBackend(w, ctx, 16, 4, KvDtype.F32, false)) {
                failures += check("cpu", b, prompt, ids, want, vocab, TOL_CPU);
            }
            if (lib != null) {
                try (Backend b = arch instanceof HybridArch ? new HybridMetalBackend(w, ctx, 16, lib, KvDtype.F32)
                        : new MetalBackend(w, ctx, 16, lib)) {
                    failures += check("metal", b, prompt, ids, want, vocab, TOL_GPU);
                }
                try (Backend b = arch instanceof HybridArch ? new HybridMetalBackend(w, ctx, 16, lib, KvDtype.F16)
                        : new MetalBackend(w, ctx, 16, lib, KvDtype.F16)) {
                    failures += check("metal f16 cache", b, prompt, ids, want, vocab, 1e-2f);
                }
            }
        }
        System.out.println(failures == 0 ? "arch ref check passed" : "arch ref check FAILED: " + failures);
        System.exit(failures == 0 ? 0 : 1);
    }

    private static int check(String label, Backend b, int[] prompt, int[] ids, float[] want, int vocab, float tol) {
        int failures = 0;
        int n = ids.length;
        float[] got = new float[n * vocab];
        int batch = b.maxBatch();
        for (int off = 0; off < n; off += batch) {
            int m = Math.min(batch, n - off);
            float[] part = new float[m * vocab];
            b.forwardAll(ids, off, m, off, part, vocab);
            System.arraycopy(part, 0, got, off * vocab, m * vocab);
        }
        float scale = 0;
        float err = 0;
        int argmaxWrong = 0;
        for (int i = 0; i < n; i++) {
            if (argmax(got, i * vocab, vocab) != argmax(want, i * vocab, vocab)) {
                argmaxWrong++;
            }
        }
        for (int i = 0; i < got.length; i++) {
            scale = Math.max(scale, Math.abs(want[i]));
            err = Math.max(err, Math.abs(got[i] - want[i]));
        }
        if (Boolean.getBoolean("archref.verbose")) {
            for (int i = 0; i < n; i++) {
                float e = 0;
                for (int j = 0; j < vocab; j++) {
                    e = Math.max(e, Math.abs(got[i * vocab + j] - want[i * vocab + j]));
                }
                System.out.printf("    pos %d err %.3g%n", i, e);
            }
        }
        boolean ok = err / scale <= tol && argmaxWrong == 0;
        System.out.printf("  %-16s %d positions: max |logit - reference| %.3g (%.2g of max |logit| %.3g), argmax wrong at %d  %s%n",
                label, n, err, err / scale, scale, argmaxWrong, ok ? "ok" : "FAIL");
        failures += ok ? 0 : 1;

        // free-running greedy decoding from the prompt
        int[] seq = Arrays.copyOf(prompt, ids.length);
        float[] logits = new float[vocab];
        for (int pos = 0; pos < prompt.length; pos += b.maxBatch()) {
            b.forward(prompt, pos, Math.min(b.maxBatch(), prompt.length - pos), pos, logits);
        }
        boolean same = true;
        for (int s = prompt.length; s < ids.length; s++) {
            int next = argmax(logits, 0, vocab);
            seq[s] = next;
            same &= next == ids[s];
            b.forward(new int[]{next}, 0, 1, s, logits);
        }
        System.out.printf("  %-16s greedy %d ids from the prompt equal the reference's: %s%n", label, ids.length - prompt.length,
                same ? "ok" : "FAIL " + Arrays.toString(Arrays.copyOfRange(seq, prompt.length, seq.length)));
        failures += same ? 0 : 1;

        // chunking invariance: the last-position logits after the whole prompt cut in different ways
        float[] whole = lastLogits(b, prompt, prompt.length, vocab);
        for (int chunk : new int[]{7, 1}) {
            float[] cut = lastLogits(b, prompt, chunk, vocab);
            float d = 0;
            for (int i = 0; i < vocab; i++) {
                d = Math.max(d, Math.abs(cut[i] - whole[i]));
            }
            boolean eq = d / scale <= (chunk == 1 ? 2 * tol : tol);
            System.out.printf("  %-16s prompt in chunks of %d vs in one: max diff %.3g  %s%n", label, chunk, d, eq ? "ok" : "FAIL");
            failures += eq ? 0 : 1;
        }
        return failures;
    }

    private static float[] lastLogits(Backend b, int[] prompt, int chunk, int vocab) {
        float[] logits = new float[vocab];
        chunk = Math.min(chunk, b.maxBatch());
        for (int pos = 0; pos < prompt.length; pos += chunk) {
            int n = Math.min(chunk, prompt.length - pos);
            b.forward(prompt, pos, n, pos, logits);
        }
        return logits;
    }

    private static int argmax(float[] v, int off, int n) {
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (v[off + i] > v[off + best]) {
                best = i;
            }
        }
        return best;
    }

    private static int[] ints(List<Object> l) {
        int[] out = new int[l.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) l.get(i)).intValue();
        }
        return out;
    }

    private static float[] floats(Path p, int n) throws Exception {
        ByteBuffer bb = ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
        if (bb.remaining() != 4 * n) {
            throw new IllegalStateException(p + " has " + bb.remaining() / 4 + " floats, expected " + n);
        }
        float[] out = new float[n];
        bb.asFloatBuffer().get(out);
        return out;
    }
}
