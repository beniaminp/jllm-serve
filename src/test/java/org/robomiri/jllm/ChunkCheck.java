package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.model.Backend;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Does the prefill chunk size change the logits? Prefills the same text (up to 1 500 tokens) on a Metal backend in
 * chunks of several sizes and compares the last row's logits bit for bit with the first run's (docs/08 §8.26).
 * Chunks whose last piece is shorter than {@code MMA_MIN_TOKENS} (32) leave the matrix-unit kernels for the scalar
 * tiled ones and differ in the last bits; every other size agrees exactly.
 *
 * <pre>java ... org.robomiri.jllm.ChunkCheck models/Qwen_Qwen3-0.6B docs/08-metal-backend.md [settings...]</pre>
 */
public final class ChunkCheck {

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]).toAbsolutePath();
        String text = Files.readString(Path.of(args[1]));
        String[] base = {"--backend", "metal", "--model.dir", dir.getParent().toString(), "--backend.ctx", "2048",
                "--engine.speculative", "off", "--batch", "512"};
        String[] extra = Arrays.copyOfRange(args, 2, args.length);
        String[] all = Arrays.copyOf(base, base.length + extra.length);
        System.arraycopy(extra, 0, all, base.length, extra.length);
        Settings st = Settings.load(all);
        try (ModelLoader.Loaded l = ModelLoader.load(st, ModelLoader.hub(st), dir.getFileName().toString())) {
            Backend b = l.engine().backend();
            int[] ids = l.engine().tokenizer().encode(text);
            int n = Math.min(ids.length, 1500);
            int[] prompt = Arrays.copyOf(ids, n);
            int vocab = l.weights().lmHead.rows;
            System.out.println(b.name() + ": " + n + " tokens, maxBatch " + b.maxBatch());
            float[] ref = null;
            boolean ok = true;
            for (int chunk : new int[]{512, 256, 128, 100, 64}) {
                float[] logits = new float[vocab];
                for (int off = 0; off < n; off += chunk) {
                    int m = Math.min(chunk, n - off);
                    b.forward(prompt, off, m, off, off + m == n ? logits : null);
                }
                if (ref == null) {
                    ref = logits;
                }
                int differ = 0;
                double worst = 0;
                for (int i = 0; i < vocab; i++) {
                    if (Float.floatToIntBits(logits[i]) != Float.floatToIntBits(ref[i])) {
                        differ++;
                        worst = Math.max(worst, Math.abs(logits[i] - ref[i]));
                    }
                }
                int tail = n % chunk == 0 ? chunk : n % chunk;
                boolean expectExact = tail >= 32;
                ok &= differ == 0 || !expectExact;
                System.out.printf("chunk %3d (last piece %3d tokens): %d of %d logits differ from chunk 512, max |d| %.2e%s%n",
                        chunk, tail, differ, vocab, worst, differ > 0 && expectExact ? "  UNEXPECTED" : "");
            }
            System.out.println(ok ? "chunk check passed" : "chunk check FAILED");
            if (!ok) {
                System.exit(1);
            }
        }
    }
}
