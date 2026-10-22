package org.robomiri.jllm;

import org.robomiri.jllm.cpu.Qwen3NextCpuBackend;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.metal.Qwen3NextMetalBackend;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.Qwen3NextArch;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.file.Path;
import java.util.Arrays;

/**
 * Where the time of a Qwen3.5 decode step with multi-token prediction goes (run under the heavy lock): the median of
 * a plain single-token step, a two-token {@code forwardAll} (the verification pass), the head's draft, and the passes
 * with the head on (their extra rows), each over {@code reps} repetitions after a prompt.
 *
 * <pre>GgufProfile model.gguf metal|cpu [reps]</pre>
 */
public final class GgufProfile {

    public static void main(String[] args) throws Exception {
        int reps = args.length > 2 ? Integer.parseInt(args[2]) : 15;
        try (GgufModel g = GgufModel.open(Path.of(args[0]))) {
            Tokenizer tok = GgufModel.tokenizer(g.file());
            Qwen3NextArch arch = (Qwen3NextArch) g.architecture();
            int vocab = arch.core().vocabSize();
            int[] prompt = tok.encode("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n"
                    + "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n");
            try (ModelWeights w = g.weights();
                 Backend b = args[1].equals("metal")
                         ? new Qwen3NextMetalBackend(w, 1024, 512, System.getProperty("jllm.metal.lib"), KvDtype.F32)
                         : new Qwen3NextCpuBackend(w, 1024, 512, Runtime.getRuntime().availableProcessors(), KvDtype.F32)) {
                float[] logits = new float[vocab];
                float[] rows = new float[2 * vocab];
                for (int n : new int[]{8, 32, 128, 512}) { // prefill speed (median of 3 repetitions), head off
                    int[] ids = new int[n];
                    for (int i = 0; i < n; i++) {
                        ids[i] = prompt[i % prompt.length];
                    }
                    long[] t = new long[3];
                    for (int r = 0; r < 3; r++) {
                        long t0 = System.nanoTime();
                        b.forward(ids, 0, n, 0, logits);
                        t[r] = System.nanoTime() - t0;
                    }
                    System.out.printf("prefill %d tokens: %.0f ms = %.1f tok/s%n", n, median(t), n / (median(t) / 1e3));
                }
                for (boolean mtp : new boolean[]{false, true}) {
                    if (mtp) {
                        b.enableMtp(true);
                    }
                    b.forward(prompt, 0, prompt.length, 0, logits);
                    int pos = prompt.length;
                    int next = argmax(logits);
                    // warm up
                    for (int i = 0; i < 3; i++) {
                        b.forward(new int[]{next}, 0, 1, pos++, logits);
                        next = argmax(logits);
                    }
                    long[] plain = new long[reps];
                    for (int i = 0; i < reps; i++) {
                        long t = System.nanoTime();
                        b.forward(new int[]{next}, 0, 1, pos++, logits);
                        plain[i] = System.nanoTime() - t;
                        next = argmax(logits);
                    }
                    long[] two = new long[reps];
                    for (int i = 0; i < reps; i++) {
                        if (b.supportsRollback()) {
                            b.checkpointState();
                        }
                        long t = System.nanoTime();
                        b.forwardAll(new int[]{next, 7}, 0, 2, pos, rows, vocab);
                        two[i] = System.nanoTime() - t;
                        if (b.supportsRollback()) {
                            b.rollbackState(1);
                        }
                        pos += 1;
                        System.arraycopy(rows, 0, logits, 0, vocab);
                        next = argmax(logits);
                    }
                    String label = mtp ? "head on " : "head off";
                    System.out.printf("%s: plain step %.1f ms, two-token pass (+ checkpoint, rollback) %.1f ms%n", label,
                            median(plain), median(two));
                    if (mtp) {
                        long[] draft = new long[reps];
                        float[] head = new float[vocab];
                        for (int i = 0; i < reps; i++) {
                            long t = System.nanoTime();
                            b.mtpDraft(next, head);
                            draft[i] = System.nanoTime() - t;
                        }
                        System.out.printf("head draft (one row + logits): %.1f ms%n", median(draft));
                    }
                }
            }
        }
    }

    private static double median(long[] ns) {
        long[] s = ns.clone();
        Arrays.sort(s);
        return s[s.length / 2] / 1e6;
    }

    private static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[best]) {
                best = i;
            }
        }
        return best;
    }
}
