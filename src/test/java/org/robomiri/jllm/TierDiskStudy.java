package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.kv.KvPool;
import org.robomiri.jllm.kv.KvSequence;
import org.robomiri.jllm.model.Backend;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The accuracy of the on-disk block formats (docs/26 §26.5): a 700-token conversation of Qwen3-0.6B is computed, every block
 * is pushed to the files in {@code native} / {@code q8} / {@code q4}, a fresh slot adopts it back, and the logits of the next
 * token and the greedy continuation are compared with the flat cache that never left memory.
 *
 * <p>usage: TierDiskStudy &lt;model dir&gt; [f16|f32]
 */
public final class TierDiskStudy {

    private static ModelLoader.Loaded load(Path dir, String dt, String... extra) throws Exception {
        List<String> a = new ArrayList<>(List.of("--backend", "cpu", "--model.dir", dir.toAbsolutePath().getParent().toString(),
                "--backend.ctx", "1024", "--engine.speculative", "off", "--backend.cpu.int8-activations", "false",
                "--backend.threads", "4", "--backend.kv-dtype", dt));
        a.addAll(List.of(extra));
        Settings st = Settings.load(a.toArray(new String[0]));
        return ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString(), 1);
    }

    private static int[] text(Engine e, int from, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = from; sb.length() < n * 8; i++) {
            sb.append("item ").append(i).append(" of the list, ");
        }
        return Arrays.copyOf(e.tokenizer().encode(sb.toString()), n);
    }

    private static double[] softmax(float[] l) {
        double m = Double.NEGATIVE_INFINITY;
        for (float v : l) {
            m = Math.max(m, v);
        }
        double s = 0;
        double[] p = new double[l.length];
        for (int i = 0; i < l.length; i++) {
            p[i] = Math.exp(l[i] - m);
            s += p[i];
        }
        for (int i = 0; i < l.length; i++) {
            p[i] /= s;
        }
        return p;
    }

    /** 48 greedy tokens after the logits {@code l} of position {@code n - 1}. */
    private static int[] greedy(Backend b, float[] l, int n, int vocab) {
        float[] cur = l.clone();
        int[] out = new int[48];
        int[] one = new int[1];
        for (int i = 0; i < out.length; i++) {
            int best = 0;
            for (int j = 1; j < vocab; j++) {
                if (cur[j] > cur[best]) {
                    best = j;
                }
            }
            out[i] = best;
            one[0] = best;
            b.forward(one, 0, 1, n + i, cur);
        }
        return out;
    }

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        String dt = args.length > 1 ? args[1] : "f16";
        int n = 700;
        System.out.println("format | bytes per block | vs native | max |dlogit| | KL(flat||tier) | greedy tokens equal of 48 | first top-1 diff");
        try (ModelLoader.Loaded flat = load(dir, dt)) {
            int[] ids = text(flat.engine(), 0, n);
            int vocab = flat.weights().arch.core().vocabSize();
            float[] ref = new float[vocab];
            Backend fb = flat.engine().backend();
            for (int pos = 0; pos < n; pos += 64) {
                fb.forward(ids, pos, Math.min(64, n - pos), pos, ref);
            }
            int[] expect = greedy(fb, ref, n, vocab);
            long nativeBytes = 0;
            for (String fmt : new String[]{"native", "q8", "q4"}) {
                Path tmp = Files.createTempDirectory("tierdisk");
                try (ModelLoader.Loaded t = load(dir, dt, "--backend.kv-tier", "true", "--backend.kv-host-mb", "0", "--backend.kv-disk-format", fmt,
                        "--backend.kv-tier-dir", tmp.toString(), "--backend.kv-stream", "off")) {
                    Engine te = t.engine();
                    Backend tb = te.backend();
                    float[] tmpl = new float[vocab];
                    for (int pos = 0; pos < n; pos += 64) {
                        tb.forward(ids, pos, Math.min(64, n - pos), pos, tmpl);
                    }
                    tb.kvSequence().publish(ids, n);
                    tb.kvSequence().clear(); // the blocks are cached now, not held
                    KvPool pool = tb.kvSequence().pool();
                    pool.evictAllCached(); // every block goes to the files in the format
                    KvPool.Stats s = pool.stats();
                    long perBlock = s.spillFiles() > 0 ? s.spillBytes() / s.spillFiles() : 0;
                    if (fmt.equals("native")) {
                        nativeBytes = perBlock;
                    }
                    // the logits of the next token after adopting the prefix from the files
                    KvSequence seq = tb.kvSequence();
                    int found = seq.match(ids, n - 1);
                    int got = seq.adopt(ids, found);
                    float[] out = new float[vocab];
                    tb.forward(ids, got, n - got, got, out);
                    double maxd = 0;
                    for (int i = 0; i < vocab; i++) {
                        maxd = Math.max(maxd, Math.abs(out[i] - ref[i]));
                    }
                    double[] p = softmax(ref);
                    double[] q = softmax(out);
                    double kl = 0;
                    for (int i = 0; i < vocab; i++) {
                        if (p[i] > 1e-12) {
                            kl += p[i] * Math.log(p[i] / Math.max(q[i], 1e-300));
                        }
                    }
                    int[] mine = greedy(tb, out, n, vocab);
                    int same = 0;
                    int first = -1;
                    for (int i = 0; i < expect.length; i++) {
                        if (mine[i] == expect[i]) {
                            same++;
                        } else if (first < 0) {
                            first = i;
                        }
                    }
                    System.out.printf("%s | %d | %.2f | %.4f | %.6f | %d of %d | %s  (adopted %d tokens)%n", fmt, perBlock,
                            nativeBytes > 0 ? (double) perBlock / nativeBytes : 1.0, maxd, kl, same, expect.length,
                            first < 0 ? "none" : String.valueOf(first), got);
                } finally {
                    try (var walk = Files.walk(tmp)) {
                        walk.sorted(java.util.Comparator.reverseOrder()).forEach(x -> x.toFile().delete());
                    }
                }
            }
        }
    }
}
