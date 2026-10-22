package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.KvSparse;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * The accuracy study of the quantized KV caches (docs/07 §7.13): every cache type against the f16 cache of the same
 * model, on the same weights.
 *
 * <p>{@code accuracy}: for each prompt, f16 greedily generates {@code steps} tokens (the reference); a variant is
 * then measured two ways. Free-running: it generates its own greedy tokens (first position where they differ from the
 * reference, and the fraction of positions that are identical). Teacher-forced: it is fed the reference's tokens and
 * its logits at every step are compared with the reference's (mean absolute difference over the vocabulary, mean
 * absolute difference over the reference's 20 best tokens, the largest difference, KL(reference || variant), and how
 * often the arg max is the same). Then a fixed text is run through the prefill path in chunks and the log-probabilities
 * of its tokens compared (perplexity and the per-token difference).
 *
 * <p>{@code needle}: a pass key hidden at several depths of a filler text of the given lengths; the question is
 * answered greedily; the table counts the answers that contain the key.
 *
 * <p>usage: KvStudy accuracy|needle model=&lt;dir&gt; backend=cpu|metal variants=f32,f16,q8,q4:0,q4:128 [steps=300]
 * [ctx=8192] [text=&lt;file&gt;] [ppl=2048] [lens=4096,8192] [trials=3] [lib=&lt;metal lib&gt;]. A variant {@code q4:R} has the recent
 * window R. Whether keys are rotated is decided by the JVM flag {@code -Djllm.kv.q4.rotate=false}.
 *
 * <p>Sparse decode attention (docs/26): a variant {@code f16+p} is the paged cache with dense attention, and
 * {@code f16+sK[:sink[:window[:denseLayers]]]} the paged cache with sparse decode attention choosing K blocks (sink and
 * window in tokens, defaults 16 and 256; 2 dense layers). {@code needle} takes {@code tail=N}: the last N tokens of the
 * prompt (the question) are fed one at a time, as decode steps, so that they too attend sparsely (default 0: the prompt is
 * prefilled in chunks, as the engine does, and only the generated tokens are sparse).
 *
 * <p>{@code decodeppl}: teacher-forced decode perplexity, {@code lens=} contexts, {@code text=} the document (its first
 * {@code lens} tokens), {@code tail=N} tokens at the end (default 512) scored one decode step at a time after a chunked
 * prefill of the rest; per variant the mean negative log-probability, the perplexity, the mean absolute difference and
 * the mean KL against the first variant, and how often the arg max is the same.
 */
public final class KvStudy {

    private static final String[] PROMPTS = {
            "Write a Python function that merges overlapping intervals in a list of [start, end] pairs, with a docstring "
                    + "and three unit tests.",
            "Write an essay of about 300 words on how the printing press changed Europe.",
            "@DOC@\n\nSummarize the main points of the document above in one paragraph."};
    private static final String[] PROMPT_NAMES = {"code", "prose", "long document"};

    public static void main(String[] a) throws Exception {
        String mode = a[0];
        java.util.Map<String, String> o = new java.util.HashMap<>();
        for (int i = 1; i < a.length; i++) {
            int eq = a[i].indexOf('=');
            o.put(a[i].substring(0, eq), a[i].substring(eq + 1));
        }
        Path dir = Path.of(o.getOrDefault("model", "models/Qwen_Qwen3-0.6B"));
        String backend = o.getOrDefault("backend", "cpu");
        int ctx = Integer.parseInt(o.getOrDefault("ctx", "8192"));
        int batch = Integer.parseInt(o.getOrDefault("batch", "64"));
        String[] variants = o.getOrDefault("variants", "f32,f16,q8,q4:0,q4:128").split(",");
        String[] st = {"--backend", backend, "--model.dir", dir.toAbsolutePath().getParent().toString(),
                "--backend.ctx", "256", "--engine.speculative", "off", "--engine.batch-decode", "false",
                "--backend.kv-dtype", "f16", "--backend.cpu.int8-activations", "false"};
        Settings s = Settings.load(st);
        System.out.println("# KvStudy " + mode + " backend=" + backend + " rotate=" + org.robomiri.jllm.model.KvQ4.ROTATE
                + " ctx=" + ctx + " model=" + dir.getFileName());
        try (ModelLoader.Loaded l = ModelLoader.load(s, ModelLoader.hub(s), dir.getFileName().toString().replaceFirst("_", "/"))) {
            Study study = new Study(l, backend, o.get("lib"), ctx, batch);
            study.tail = mode.equals("needle") ? Integer.parseInt(o.getOrDefault("tail", "0")) : 0;
            if (mode.equals("decodeppl")) {
                String[] lens = o.getOrDefault("lens", "8192").split(",");
                for (String len : lens) {
                    study.decodePpl(variants, Integer.parseInt(len), o.get("text"), Integer.parseInt(o.getOrDefault("tail", "512")));
                }
            } else if (mode.equals("accuracy")) {
                study.accuracy(variants, Integer.parseInt(o.getOrDefault("steps", "300")), o.get("text"),
                        Integer.parseInt(o.getOrDefault("ppl", "2048")), o.get("doc"),
                        Integer.parseInt(o.getOrDefault("doctokens", "3000")), o.getOrDefault("only", ""));
            } else {
                String[] lens = o.getOrDefault("lens", "4096,8192").split(",");
                int[] ls = new int[lens.length];
                for (int i = 0; i < ls.length; i++) {
                    ls[i] = Integer.parseInt(lens[i]);
                }
                study.needle(variants, ls, Integer.parseInt(o.getOrDefault("trials", "3")),
                        Integer.parseInt(o.getOrDefault("depths", "3")));
            }
        }
    }

    private static final class Study {
        final ModelLoader.Loaded l;
        final Engine engine;
        final ModelWeights w;
        final Tokenizer tok;
        final String backendName;
        final String lib;
        final int ctx;
        final int batch;
        final int vocab;
        final Set<Integer> ban;
        /** needle: the last {@code tail} prompt tokens are fed one at a time (decode steps) instead of in the last chunk. */
        int tail;

        Study(ModelLoader.Loaded l, String backendName, String lib, int ctx, int batch) {
            this.l = l;
            this.engine = l.engine();
            this.w = l.weights();
            this.tok = engine.tokenizer();
            this.backendName = backendName;
            this.lib = lib;
            this.ctx = ctx;
            this.batch = batch;
            this.vocab = w.arch.core().vocabSize();
            this.ban = engine.stopTokenIds();
        }

        /** A backend of the variant ({@code f32}, {@code f16}, {@code q8}, {@code q8r} (q8 with rotated keys), {@code q4:R}); f16 is the loader's own. */
        Backend open(String variant) {
            String[] plus = variant.split("\\+");
            if (plus.length > 1) {
                return openPaged(plus[0], plus[1]);
            }
            String[] p = variant.split(":");
            boolean rot = p[0].equals("q8r"); // q8 with Hadamard-rotated keys (backend.kv-q8-rotate)
            KvDtype dt = KvDtype.parse(rot ? "q8" : p[0]);
            KvDtype.setQ8Rotate(rot);
            if (dt == KvDtype.Q4) {
                KvDtype.setQ4Recent(p.length > 1 ? Integer.parseInt(p[1]) : KvDtype.Q4_RECENT_DEFAULT);
            }
            if (backendName.equals("metal")) {
                return new MetalBackend(w, ctx, batch, lib, dt);
            }
            return new CpuBackend(w, ctx, batch, Runtime.getRuntime().availableProcessors(), dt, false);
        }

        private org.robomiri.jllm.kv.KvPool lastPool;

        /** A paged backend ({@code mode}: {@code p} dense, {@code sK[:sink[:window[:dense]]]} sparse); the previous variant's pool is closed. */
        Backend openPaged(String base, String mode) {
            if (lastPool != null) {
                lastPool.close();
                lastPool = null;
            }
            boolean rot = base.equals("q8r");
            KvDtype dt = KvDtype.parse(rot ? "q8" : base);
            KvDtype.setQ8Rotate(rot);
            if (mode.startsWith("s")) {
                String[] a = mode.substring(1).split(":");
                KvSparse.set(new KvSparse(true, Integer.parseInt(a[0]), a.length > 1 ? Integer.parseInt(a[1]) : 16,
                        a.length > 2 ? Integer.parseInt(a[2]) : 256, a.length > 3 ? Integer.parseInt(a[3]) : 2));
            } else {
                KvSparse.set(KvSparse.OFF);
            }
            try {
                if (backendName.equals("metal")) {
                    lastPool = MetalBackend.newKvPool(w, ctx, 1, dt, 16, 0, lib);
                    return new MetalBackend(w, ctx, batch, lib, dt, null, lastPool);
                }
                lastPool = CpuBackend.newKvPool(w, ctx, 1, dt, 16, 0, rot);
                return new CpuBackend(w, ctx, batch, Runtime.getRuntime().availableProcessors(), dt, false, lastPool);
            } finally {
                KvSparse.set(KvSparse.OFF);
            }
        }

        int[] chat(String user) {
            return engine.renderChat(List.of(new ChatTemplate.Message("user", user + " /no_think")), false);
        }

        int argmax(float[] x) {
            int best = -1;
            for (int i = 0; i < x.length; i++) {
                if (!ban.contains(i) && (best < 0 || x[i] > x[best])) {
                    best = i;
                }
            }
            return best;
        }

        /** Prefills {@code prompt} from position 0; the logits of its last token go to {@code logits}. */
        void prefill(Backend b, int[] prompt, float[] logits) {
            int chunked = prompt.length - Math.min(tail, prompt.length - 1);
            for (int off = 0; off < chunked; off += b.maxBatch()) {
                int n = Math.min(b.maxBatch(), chunked - off);
                b.forward(prompt, off, n, off, off + n == prompt.length ? logits : null);
            }
            for (int i = chunked; i < prompt.length; i++) {
                b.forward(prompt, i, 1, i, i == prompt.length - 1 ? logits : null);
            }
        }

        /** Teacher-forced decode perplexity: see the class comment ({@code decodeppl}). */
        void decodePpl(String[] variants, int len, String textFile, int tailTokens) throws Exception {
            int[] all = tok.encode(Files.readString(Path.of(textFile)));
            if (all.length < len) {
                throw new IllegalArgumentException(textFile + " has only " + all.length + " tokens, " + len + " wanted");
            }
            int[] text = java.util.Arrays.copyOf(all, len);
            int pre = len - tailTokens;
            float[][] ref = new float[tailTokens][];
            System.out.printf("%n# decode perplexity, context %d, the last %d tokens scored one decode step at a time%n", len, tailTokens);
            System.out.println("| cache | mean log-prob | perplexity | mean log-prob difference vs the first | mean abs logit difference"
                    + " | mean KL (nats) | arg max agreement |");
            System.out.println("|---|---|---|---|---|---|---|");
            double refMean = 0;
            for (int vi = 0; vi < variants.length; vi++) {
                float[] logits = new float[vocab];
                double sum = 0;
                double dAbs = 0;
                double dKl = 0;
                int agree = 0;
                try (Backend b = open(variants[vi])) {
                    int saved = tail;
                    tail = 0;
                    int[] head = java.util.Arrays.copyOf(text, pre);
                    prefill(b, head, logits);
                    tail = saved;
                    for (int t = 0; t < tailTokens; t++) {
                        sum += logSoftmax(logits, 0, text[pre + t]);
                        if (vi == 0) {
                            ref[t] = logits.clone();
                        } else {
                            double[] d = diff(ref[t], logits);
                            dAbs += d[0];
                            dKl += d[3];
                            agree += argmax(ref[t]) == argmax(logits) ? 1 : 0;
                        }
                        if (t + 1 < tailTokens) {
                            b.forward(text, pre + t, 1, pre + t, logits);
                        }
                    }
                }
                double mean = sum / tailTokens;
                if (vi == 0) {
                    refMean = mean;
                }
                if (vi == 0) {
                    System.out.printf(Locale.ROOT, "| %s | %.4f | %.4f | reference | | | |%n", variants[vi], mean, Math.exp(-mean));
                } else {
                    System.out.printf(Locale.ROOT, "| %s | %.4f | %.4f | %+.5f | %.5f | %.5f | %.1f%% |%n", variants[vi], mean,
                            Math.exp(-mean), mean - refMean, dAbs / tailTokens, dKl / tailTokens, 100.0 * agree / tailTokens);
                }
                System.out.flush();
            }
        }

        /** Greedy tokens (stop tokens excluded); {@code rec}, when given, receives the logits each was chosen from. */
        int[] generate(Backend b, int[] prompt, int steps, float[][] rec) {
            float[] logits = new float[vocab];
            prefill(b, prompt, logits);
            int[] out = new int[steps];
            for (int t = 0; t < steps; t++) {
                if (rec != null) {
                    System.arraycopy(logits, 0, rec[t], 0, vocab);
                }
                out[t] = argmax(logits);
                if (t + 1 < steps) {
                    b.forward(out, t, 1, prompt.length + t, logits);
                }
            }
            return out;
        }

        /** The variant's logits at each step when fed the reference tokens. */
        void forced(Backend b, int[] prompt, int[] ref, float[][] rec) {
            float[] logits = new float[vocab];
            prefill(b, prompt, logits);
            for (int t = 0; t < ref.length; t++) {
                System.arraycopy(logits, 0, rec[t], 0, vocab);
                if (t + 1 < ref.length) {
                    b.forward(ref, t, 1, prompt.length + t, logits);
                }
            }
        }

        double[] logProbs(Backend b, int[] text) {
            double[] lp = new double[text.length - 1];
            float[] out = new float[b.maxBatch() * vocab];
            for (int off = 0; off < text.length - 1; off += b.maxBatch()) {
                int n = Math.min(b.maxBatch(), text.length - 1 - off);
                b.forwardAll(text, off, n, off, out, vocab);
                for (int i = 0; i < n; i++) {
                    lp[off + i] = logSoftmax(out, i * vocab, text[off + i + 1]);
                }
            }
            return lp;
        }

        double logSoftmax(float[] x, int o, int target) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < vocab; i++) {
                max = Math.max(max, x[o + i]);
            }
            double sum = 0;
            for (int i = 0; i < vocab; i++) {
                sum += Math.exp(x[o + i] - max);
            }
            return x[o + target] - max - Math.log(sum);
        }

        void accuracy(String[] variants, int steps, String textFile, int pplTokens, String docFile, int docTokens,
                      String only) throws Exception {
            String doc = docFile != null ? Files.readString(Path.of(docFile)) : "";
            int[] dtoks = tok.encode(doc);
            if (dtoks.length > docTokens) {
                doc = tok.decode(java.util.Arrays.copyOf(dtoks, docTokens));
            }
            List<String> rows = new ArrayList<>();
            System.out.println("\n| prompt | cache | first divergence | identical tokens | mean abs logit diff | top-20 mean abs diff"
                    + " | max abs logit diff | mean KL (nats) | teacher-forced arg max agreement |");
            System.out.println("|---|---|---|---|---|---|---|---|---|");
            for (int pi = 0; pi < PROMPTS.length && !only.equals("none"); pi++) {
                if (!only.isEmpty() && !only.contains(Integer.toString(pi))) {
                    continue;
                }
                if (pi == 2 && docFile == null) {
                    continue;
                }
                int[] prompt = chat(PROMPTS[pi].replace("@DOC@", doc));
                float[][] refLogits = new float[steps][vocab];
                int[] ref;
                try (Backend b = open("f16")) {
                    ref = generate(b, prompt, steps, refLogits);
                }
                System.out.printf("# %s: %d prompt tokens, reference f16: %s%n", PROMPT_NAMES[pi], prompt.length,
                        tok.decode(java.util.Arrays.copyOf(ref, 40)).replace('\n', ' '));
                float[][] rec = new float[steps][vocab];
                for (String v : variants) {
                    if (v.equals("f16")) {
                        continue;
                    }
                    int[] got;
                    try (Backend b = open(v)) {
                        got = generate(b, prompt, steps, null);
                    }
                    try (Backend b = open(v)) {
                        forced(b, prompt, ref, rec);
                    }
                    int first = steps;
                    int same = 0;
                    for (int t = 0; t < steps; t++) {
                        if (got[t] == ref[t]) {
                            same++;
                        } else if (first == steps) {
                            first = t;
                        }
                    }
                    double mean = 0;
                    double top = 0;
                    double max = 0;
                    double kl = 0;
                    int agree = 0;
                    for (int t = 0; t < steps; t++) {
                        double[] d = diff(refLogits[t], rec[t]);
                        mean += d[0];
                        top += d[1];
                        max = Math.max(max, d[2]);
                        kl += d[3];
                        agree += argmax(refLogits[t]) == argmax(rec[t]) ? 1 : 0;
                    }
                    System.out.printf(Locale.ROOT, "| %s | %s | %s | %.1f%% (%d/%d) | %.4f | %.4f | %.3f | %.5f | %.1f%% |%n",
                            PROMPT_NAMES[pi], v, first == steps ? "none in " + steps : Integer.toString(first),
                            100.0 * same / steps, same, steps, mean / steps, top / steps, max, kl / steps,
                            100.0 * agree / steps);
                    System.out.flush();
                }
            }
            if (textFile != null) {
                int[] text = tok.encode(Files.readString(Path.of(textFile)));
                text = java.util.Arrays.copyOf(text, Math.min(text.length, pplTokens));
                double[] refLp;
                try (Backend b = open("f16")) {
                    refLp = logProbs(b, text);
                }
                System.out.printf(Locale.ROOT, "%n# perplexity-like check on %d tokens of %s (prefill chunks of %d)%n",
                        text.length, textFile, batch);
                System.out.println("| cache | mean log-prob | perplexity | mean log-prob difference vs f16 | mean abs per-token difference | "
                        + "largest per-token difference |");
                System.out.println("|---|---|---|---|---|---|");
                System.out.printf(Locale.ROOT, "| f16 | %.4f | %.4f | 0 | 0 | 0 |%n", mean(refLp), Math.exp(-mean(refLp)));
                for (String v : variants) {
                    if (v.equals("f16")) {
                        continue;
                    }
                    double[] lp;
                    try (Backend b = open(v)) {
                        lp = logProbs(b, text);
                    }
                    double dm = 0;
                    double da = 0;
                    double dx = 0;
                    for (int i = 0; i < lp.length; i++) {
                        double d = lp[i] - refLp[i];
                        dm += d;
                        da += Math.abs(d);
                        dx = Math.max(dx, Math.abs(d));
                    }
                    System.out.printf(Locale.ROOT, "| %s | %.4f | %.4f | %+.5f | %.5f | %.4f |%n", v, mean(lp),
                            Math.exp(-mean(lp)), dm / lp.length, da / lp.length, dx);
                    System.out.flush();
                }
            }
        }

        static double mean(double[] x) {
            double s = 0;
            for (double v : x) {
                s += v;
            }
            return s / x.length;
        }

        /** [mean |a - b| over the vocabulary, mean over a's 20 best, max, KL(softmax a || softmax b)]. */
        double[] diff(float[] a, float[] b) {
            double sum = 0;
            double mx = 0;
            float ma = Float.NEGATIVE_INFINITY;
            float mb = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < vocab; i++) {
                double d = Math.abs(a[i] - b[i]);
                sum += d;
                mx = Math.max(mx, d);
                ma = Math.max(ma, a[i]);
                mb = Math.max(mb, b[i]);
            }
            double za = 0;
            double zb = 0;
            for (int i = 0; i < vocab; i++) {
                za += Math.exp(a[i] - ma);
                zb += Math.exp(b[i] - mb);
            }
            double kl = 0;
            for (int i = 0; i < vocab; i++) {
                double pa = Math.exp(a[i] - ma) / za;
                if (pa > 1e-12) {
                    kl += pa * ((a[i] - ma - Math.log(za)) - (b[i] - mb - Math.log(zb)));
                }
            }
            int[] top = top(a, 20);
            double ts = 0;
            for (int i : top) {
                ts += Math.abs(a[i] - b[i]);
            }
            return new double[]{sum / vocab, ts / top.length, mx, kl};
        }

        static int[] top(float[] x, int k) {
            int[] idx = new int[k];
            float[] val = new float[k];
            java.util.Arrays.fill(val, Float.NEGATIVE_INFINITY);
            for (int i = 0; i < x.length; i++) {
                if (x[i] > val[k - 1]) {
                    int j = k - 1;
                    while (j > 0 && val[j - 1] < x[i]) {
                        val[j] = val[j - 1];
                        idx[j] = idx[j - 1];
                        j--;
                    }
                    val[j] = x[i];
                    idx[j] = i;
                }
            }
            return idx;
        }

        // ------------------------------------------------------------------ needle in a haystack

        static final String[] ADJ = {"quiet", "old", "bright", "heavy", "narrow", "green", "distant", "cold", "wooden", "busy"};
        static final String[] NOUN = {"harbor", "library", "garden", "bridge", "market", "tower", "valley", "workshop", "station",
                "orchard", "lighthouse", "bakery"};
        static final String[] VERB = {"stayed open", "was repainted", "stood empty", "was visited", "was measured",
                "was forgotten", "was cleaned", "kept its secret"};
        static final String[] WHEN = {"until spring", "through the winter", "after the storm", "before the festival",
                "all of last year", "for three long days"};

        String filler(Random r, int sentences) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < sentences; i++) {
                sb.append("Note ").append(r.nextInt(100000)).append(": the ").append(ADJ[r.nextInt(ADJ.length)]).append(' ')
                        .append(NOUN[r.nextInt(NOUN.length)]).append(' ').append(VERB[r.nextInt(VERB.length)]).append(' ')
                        .append(WHEN[r.nextInt(WHEN.length)]).append(".\n");
            }
            return sb.toString();
        }

        void needle(String[] variants, int[] lens, int trials, int depths) {
            System.out.println("\n| context (tokens) | cache | found the key | by depth (start .. end) |");
            System.out.println("|---|---|---|---|");
            Random r0 = new Random(7);
            int per = tok.encode(filler(new Random(1), 200)).length / 200;
            for (int len : lens) {
                // the same haystacks for every variant
                List<int[]> prompts = new ArrayList<>();
                List<String> keys = new ArrayList<>();
                List<Integer> depthIdx = new ArrayList<>();
                for (int d = 0; d < depths; d++) {
                    for (int t = 0; t < trials; t++) {
                        String key = Integer.toString(10000 + r0.nextInt(90000));
                        double depth = depths == 1 ? 0.5 : (double) d / (depths - 1);
                        int sentences = Math.max(10, (len - 120) / per);
                        int at = (int) (depth * sentences);
                        Random fr = new Random(1000 + len + d * 31 + t);
                        String text = filler(fr, at) + "The pass key is " + key + ". Remember it. " + key + " is the pass key.\n"
                                + filler(fr, sentences - at);
                        int[] p = chat(text + "\nWhat is the pass key mentioned in the text above? Answer with only the number.");
                        prompts.add(p);
                        keys.add(key);
                        depthIdx.add(d);
                    }
                }
                for (String v : variants) {
                    int found = 0;
                    int[] byDepth = new int[depths];
                    int promptLen = 0;
                    try (Backend b = open(v)) {
                        for (int i = 0; i < prompts.size(); i++) {
                            int[] p = prompts.get(i);
                            promptLen = p.length;
                            if (p.length + 24 > ctx) {
                                throw new IllegalStateException("prompt of " + p.length + " tokens does not fit ctx " + ctx);
                            }
                            long t0 = System.nanoTime();
                            int[] out = generate(b, p, 14, null);
                            String ans = tok.decode(out);
                            System.out.printf(Locale.ROOT, "# %s len %d trial %d (depth %d): %s in %.1f s%n", v, len, i,
                                    depthIdx.get(i), ans.contains(keys.get(i)) ? "found" : "missed", (System.nanoTime() - t0) / 1e9);
                            System.out.flush();
                            if (ans.contains(keys.get(i))) {
                                found++;
                                byDepth[depthIdx.get(i)]++;
                            }
                        }
                    }
                    StringBuilder bd = new StringBuilder();
                    for (int d = 0; d < depths; d++) {
                        bd.append(d > 0 ? " / " : "").append(byDepth[d]).append('/').append(trials);
                    }
                    System.out.printf(Locale.ROOT, "| %d (%d) | %s | %d/%d | %s |%n", len, promptLen, v, found, prompts.size(), bd);
                    System.out.flush();
                }
            }
        }
    }
}
