package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.cpu.MambaCpuBackend;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.MambaArch;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.QuantFile;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.tokenizer.ChatTemplate;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Compares the engine (CPU backend) with the Hugging Face transformers reference written by
 * {@code scripts/real_ref.py} on a real published checkpoint; driven by {@code scripts/real-weights-check.sh}.
 *
 * <p>usage: {@code RealWeightsCheck <model-dir> <ref-dir> [quant]} (quant: q8 by default). For every prompt of
 * {@code <ref-dir>/ref.json} it checks
 * <ul>
 *   <li>the tokenizer: the engine's ids for the prompt (the family {@link ChatTemplate} for a chat prompt, the
 *       tokenizer's BOS rule plus {@link Tokenizer#encode} for raw text) and the rendered template text equal the
 *       reference's {@code apply_chat_template} / {@code tokenizer(text)}; plus the tokenizer probes;</li>
 *   <li>free-running greedy decoding from the reference's prompt ids: how many of the 16 ids equal the reference's
 *       before the first divergence, and the reference's logit margin there (its top logit minus its logit of the
 *       engine's token: a small margin is a near tie that quantization may flip);</li>
 *   <li>teacher-forced: one pass over the prompt plus the reference's 16 ids, at each step the engine's argmax
 *       against the reference's token, the top-20 overlap and the Pearson correlation of the full logit rows;</li>
 *   <li>when the reference also wrote the Q8_0-rounded pass ({@code <prompt>.q8.f32}): the same teacher-forced rows
 *       against it, the Pearson correlation and the largest |engine - reference| relative to the largest |logit|.
 *       That reference has the engine's weights exactly, so only float rounding of the activations remains: this
 *       separates quantization noise (first comparison) from a wrong computation (this one).</li>
 * </ul>
 * Prints a table and writes {@code <ref-dir>/result-<quant>.json}; {@code -Dreal.prompts=raw_text,raw_code} runs only
 * those prompts (and adds them to the file name), to split a slow model over several runs. Run with {@code -Djllm.cpu.fastmath=false}; the
 * int8-activation kernels are off, so the matmuls see float activations.
 */
public final class RealWeightsCheck {

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        Path refDir = Path.of(args[1]);
        QType quant = QType.parse(args.length > 2 ? args[2] : "q8");
        String sel = System.getProperty("real.prompts");
        List<String> only = sel == null || sel.isBlank() ? null : List.of(sel.split(","));
        Map<String, Object> ref = Json.parseObject(Files.readString(refDir.resolve("ref.json")));
        Architecture arch = Architecture.load(dir);
        Tokenizer tok = Tokenizer.load(dir.resolve("tokenizer.json"));
        ChatTemplate.Format fmt = null;
        try {
            fmt = ChatTemplate.forModelType(arch.core().modelType(), tok);
        } catch (IllegalArgumentException e) {
            System.out.println("no engine chat template: " + e.getMessage());
        }
        int threads = Runtime.getRuntime().availableProcessors();
        int vocab = arch.core().vocabSize();

        List<Object> results = new ArrayList<>();
        int probeOk = 0;
        List<Object> probes = Json.list(ref, "probes");
        for (Object o : probes) {
            String text = Json.str(o, "text", "");
            int[] want = ints(Json.list(o, "ids"));
            int[] got = tok.encode(text);
            if (Arrays.equals(want, got)) {
                probeOk++;
            } else {
                System.out.println("probe MISMATCH " + Json.write(text) + "\n  ref " + Arrays.toString(want)
                        + "\n  eng " + Arrays.toString(got));
            }
        }

        try (ModelWeights w = ModelWeights.load(dir, arch, quant, QuantFile.DEFAULT_HEAD, shards(dir), null);
             Backend b = arch instanceof MambaArch
                     ? new MambaCpuBackend(w, 1024, 64, threads, false)
                     : new CpuBackend(w, 1024, 64, threads, KvDtype.F32, false)) {
            System.out.printf("%-12s %5s %5s %5s %8s %10s %9s %8s %8s %9s %7s %6s%n", "prompt", "ids", "tmpl", "greedy",
                    "1st-div", "ref-margin", "forced", "top20", "pearson", "q8-pearson", "q8-err", "time");
            for (Object o : Json.list(ref, "prompts")) {
                long t0 = System.nanoTime();
                String name = Json.str(o, "name", "?");
                if (only != null && !only.contains(name)) {
                    continue;
                }
                String kind = Json.str(o, "kind", "?");
                if (Json.str(o, "error", null) != null) {
                    System.out.printf("%-12s reference template refused it: %s%n", name, Json.str(o, "error", ""));
                    results.add(Json.obj("name", name, "skipped", Json.str(o, "error", "")));
                    continue;
                }
                int[] ids = ints(Json.list(o, "ids"));
                int[] gen = ints(Json.list(o, "gen"));
                int steps = gen.length;
                int refVocab = (int) Json.num(o, "vocab", vocab);
                float[] refLogits = readF32(refDir.resolve(name + ".f32"), steps * refVocab);
                int n = Math.min(vocab, refVocab);

                // tokenizer and template
                Boolean idsMatch;
                Boolean tmplMatch = null;
                if (kind.equals("chat")) {
                    if (fmt == null) {
                        idsMatch = null;
                    } else {
                        List<ChatTemplate.Message> msgs = new ArrayList<>();
                        for (Object m : Json.list(o, "messages")) {
                            msgs.add(new ChatTemplate.Message(Json.str(m, "role", ""), Json.str(m, "content", "")));
                        }
                        String text = fmt.render(msgs, true, true);
                        tmplMatch = text.equals(Json.str(o, "text", ""));
                        if (!tmplMatch) {
                            System.out.println(name + " template MISMATCH\n  ref " + Json.write(Json.str(o, "text", ""))
                                    + "\n  eng " + Json.write(text));
                        }
                        int[] eng = fmt.renderTokens(tok, msgs, true, true);
                        idsMatch = Arrays.equals(eng, ids);
                        if (!idsMatch) {
                            System.out.println(name + " ids MISMATCH\n  ref " + Arrays.toString(ids) + "\n  eng "
                                    + Arrays.toString(eng));
                        }
                    }
                } else {
                    int[] enc = tok.encode(Json.str(o, "text", ""));
                    int bos = tok.promptBos();
                    int[] eng = bos >= 0 ? concat(new int[]{bos}, enc) : enc;
                    idsMatch = Arrays.equals(eng, ids);
                    if (!idsMatch) {
                        System.out.println(name + " ids MISMATCH\n  ref " + Arrays.toString(ids) + "\n  eng "
                                + Arrays.toString(eng));
                    }
                }

                // free-running greedy from the reference's prompt ids
                float[] logits = new float[vocab];
                prefill(b, ids, logits);
                int[] mine = new int[steps];
                int pos = ids.length;
                for (int s = 0; s < steps; s++) {
                    mine[s] = argmax(logits, n);
                    if (s + 1 < steps) {
                        b.forward(new int[]{mine[s]}, 0, 1, pos++, logits);
                    }
                }
                int div = 0;
                while (div < steps && mine[div] == gen[div]) {
                    div++;
                }
                double refMargin = Double.NaN;
                if (div < steps) {
                    int o0 = div * refVocab;
                    refMargin = refLogits[o0 + gen[div]] - refLogits[o0 + mine[div]];
                }

                // teacher-forced over the reference's ids
                int[] all = concat(ids, Arrays.copyOf(gen, steps - 1));
                float[] rows = forwardAll(b, all, vocab, ids.length - 1);
                int forced = 0;
                double overlap = 0;
                double pearson = 0;
                double minPearson = 1;
                Path q8Path = refDir.resolve(name + ".q8.f32");
                float[] q8Logits = quant == QType.Q8_0 && Files.isRegularFile(q8Path)
                        ? readF32(q8Path, steps * refVocab) : null;
                double q8Pearson = 0;
                double q8Err = 0;
                int q8Argmax = 0;
                for (int s = 0; s < steps; s++) {
                    float[] e = Arrays.copyOfRange(rows, s * vocab, s * vocab + n);
                    float[] r = Arrays.copyOfRange(refLogits, s * refVocab, s * refVocab + n);
                    if (argmax(e, n) == gen[s]) {
                        forced++;
                    }
                    overlap += overlap(topk(e, 20), topk(r, 20)) / 20.0;
                    double p = pearson(e, r);
                    pearson += p;
                    minPearson = Math.min(minPearson, p);
                    if (q8Logits != null) {
                        float[] q = Arrays.copyOfRange(q8Logits, s * refVocab, s * refVocab + n);
                        q8Pearson += pearson(e, q);
                        q8Err = Math.max(q8Err, maxAbsDiff(e, q) / maxAbs(q));
                        q8Argmax += argmax(e, n) == argmax(q, n) ? 1 : 0;
                    }
                }
                q8Pearson /= steps;
                overlap /= steps;
                pearson /= steps;
                System.out.printf("%-12s %5s %5s %3d/%d %8s %10s %6d/%d %8.3f %8.5f %9s %7s %5.0fs%n", name,
                        yn(idsMatch), yn(tmplMatch), div, steps, div < steps ? String.valueOf(div) : "-",
                        div < steps ? String.format("%.3f", refMargin) : "-", forced, steps, overlap, pearson,
                        q8Logits == null ? "-" : String.format("%.6f", q8Pearson),
                        q8Logits == null ? "-" : String.format("%.2f%%", 100 * q8Err), (System.nanoTime() - t0) / 1e9);
                if (div < steps) {
                    System.out.println("  ref " + Arrays.toString(gen) + "\n  eng " + Arrays.toString(mine));
                }
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("name", name);
                r.put("kind", kind);
                r.put("ids_match", idsMatch);
                r.put("template_match", tmplMatch);
                r.put("greedy_prefix", div);
                r.put("steps", steps);
                r.put("first_divergence", div < steps ? div : null);
                r.put("ref_margin", div < steps ? round(refMargin) : null);
                r.put("forced_match", forced);
                r.put("top20_overlap", round(overlap));
                r.put("pearson_mean", Math.round(pearson * 1e7) / 1e7);
                r.put("pearson_min", Math.round(minPearson * 1e7) / 1e7);
                if (q8Logits != null) {
                    r.put("q8ref_pearson_mean", Math.round(q8Pearson * 1e7) / 1e7);
                    r.put("q8ref_max_rel_err", round(q8Err));
                    r.put("q8ref_argmax_match", q8Argmax);
                }
                r.put("engine_gen", toList(mine));
                results.add(r);
            }
        }
        System.out.println("tokenizer probes: " + probeOk + "/" + probes.size());
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("model_type", arch.core().modelType());
        out.put("quant", quant.name());
        out.put("reference_dtype", Json.str(ref, "dtype", "?"));
        out.put("transformers", Json.str(ref, "transformers", "?"));
        out.put("probes_match", probeOk);
        out.put("probes", probes.size());
        out.put("prompts", results);
        String suffix = only == null ? "" : "-" + String.join("+", only);
        Files.writeString(refDir.resolve("result-" + quant.name().toLowerCase() + suffix + ".json"), Json.write(out));
    }

    /**
     * The checkpoint's shards; with {@code -Dreal.lowDisk=true} each is deleted once converted (the reference has
     * already run), so a model fits where checkpoint and cache together would not.
     */
    private static ShardProvider shards(Path dir) {
        ShardProvider local = ShardProvider.local(dir);
        if (!Boolean.getBoolean("real.lowDisk")) {
            return local;
        }
        return new ShardProvider() {
            @Override
            public Path ensure(String fileName) throws java.io.IOException {
                return local.ensure(fileName);
            }

            @Override
            public void release(String fileName) throws java.io.IOException {
                Files.deleteIfExists(dir.resolve(ShardProvider.checkName(dir, fileName)));
            }
        };
    }

    private static void prefill(Backend b, int[] ids, float[] logits) {
        int max = b.maxBatch();
        for (int off = 0; off < ids.length; off += max) {
            int c = Math.min(max, ids.length - off);
            b.forward(ids, off, c, off, logits);
        }
    }

    /**
     * Logits of positions {@code from .. all.length - 1}, one {@code vocab} row each (the positions before
     * {@code from} are run with {@link Backend#forward}, which skips their LM head).
     */
    private static float[] forwardAll(Backend b, int[] all, int vocab, int from) {
        int max = b.maxBatch();
        float[] out = new float[(all.length - from) * vocab];
        float[] last = new float[vocab];
        for (int off = 0; off < from; off += max) {
            b.forward(all, off, Math.min(max, from - off), off, last);
        }
        for (int off = from; off < all.length; off += max) {
            int c = Math.min(max, all.length - off);
            float[] chunk = new float[c * vocab];
            b.forwardAll(all, off, c, off, chunk, vocab);
            System.arraycopy(chunk, 0, out, (off - from) * vocab, c * vocab);
        }
        return out;
    }

    private static int argmax(float[] a, int n) {
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (a[i] > a[best]) {
                best = i;
            }
        }
        return best;
    }

    private static double maxAbsDiff(float[] a, float[] b) {
        double m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    private static double maxAbs(float[] a) {
        double m = 0;
        for (float x : a) {
            m = Math.max(m, Math.abs(x));
        }
        return m;
    }

    private static int[] topk(float[] a, int k) {
        int[] r = new int[k]; // indices of the k largest, in descending order (insertion into a short list)
        int size = 0;
        for (int i = 0; i < a.length; i++) {
            if (size == k && a[i] <= a[r[k - 1]]) {
                continue;
            }
            int j = size < k ? size++ : k - 1;
            while (j > 0 && a[r[j - 1]] < a[i]) {
                r[j] = r[j - 1];
                j--;
            }
            r[j] = i;
        }
        return r;
    }

    private static int overlap(int[] a, int[] b) {
        Set<Integer> s = new HashSet<>();
        for (int x : a) {
            s.add(x);
        }
        int c = 0;
        for (int x : b) {
            c += s.contains(x) ? 1 : 0;
        }
        return c;
    }

    private static double pearson(float[] a, float[] b) {
        double ma = 0;
        double mb = 0;
        for (int i = 0; i < a.length; i++) {
            ma += a[i];
            mb += b[i];
        }
        ma /= a.length;
        mb /= b.length;
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < a.length; i++) {
            double x = a[i] - ma;
            double y = b[i] - mb;
            sab += x * y;
            saa += x * x;
            sbb += y * y;
        }
        return sab / Math.sqrt(saa * sbb);
    }

    private static float[] readF32(Path p, int count) throws Exception {
        ByteBuffer bb = ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
        if (bb.remaining() != count * 4) {
            throw new IllegalStateException(p + ": " + bb.remaining() + " bytes, expected " + count * 4);
        }
        float[] f = new float[count];
        bb.asFloatBuffer().get(f);
        return f;
    }

    private static int[] ints(List<Object> l) {
        int[] r = new int[l.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = ((Number) l.get(i)).intValue();
        }
        return r;
    }

    private static List<Object> toList(int[] a) {
        List<Object> l = new ArrayList<>();
        for (int x : a) {
            l.add((long) x);
        }
        return l;
    }

    private static int[] concat(int[] a, int[] b) {
        int[] r = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }

    private static double round(double x) {
        return Math.round(x * 1e4) / 1e4;
    }

    private static String yn(Boolean b) {
        return b == null ? "-" : b ? "yes" : "NO";
    }
}
