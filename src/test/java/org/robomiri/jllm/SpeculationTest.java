package org.robomiri.jllm;

import org.robomiri.jllm.engine.DraftModel;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.Speculator;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.Sampler;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.robomiri.jllm.SmokeTest.check;

/**
 * Speculative decoding against a fake backend whose logits depend on the whole cached history (so a stale or
 * wrongly rolled-back KV position changes the output): greedy output must equal the non-speculative output, the
 * K/V bookkeeping must match, and sampled output must follow the target's distribution.
 */
final class SpeculationTest {

    static final int VOCAB = 40;

    /**
     * Token {@code t} at position {@code p} makes the next token near g = f(t[p], t[p-1], p): a bump of
     * logits around g, so sampling has a few likely tokens. The state is the token array itself.
     */
    static class FakeBackend implements Backend {
        final int[] kv = new int[256];
        int forwardCalls;
        int tokensFed;
        final boolean allRows;

        FakeBackend(boolean allRows) {
            this.allRows = allRows;
        }

        public String name() {
            return "fake";
        }

        public int contextLength() {
            return 256;
        }

        public int maxBatch() {
            return 16;
        }

        void logitsAt(int p, float[] out, int base) {
            int g = (kv[p] * 7 + 3 + (p % 23 == 0 ? p : 0)) % 20 + 8;
            for (int t = 0; t < VOCAB; t++) {
                out[base + t] = -Math.abs(t - g) * 1.1f - (t == 3 ? 0.4f : 0f);
            }
        }

        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            forwardCalls++;
            tokensFed += n;
            for (int i = 0; i < n; i++) {
                kv[pos + i] = tokens[off + i];
            }
            if (logits != null) {
                logitsAt(pos + n - 1, logits, 0);
            }
        }

        @Override
        public void forwardAll(int[] tokens, int off, int n, int pos, float[] out, int stride) {
            if (!allRows) {
                Backend.super.forwardAll(tokens, off, n, pos, out, stride);
                return;
            }
            forwardCalls++;
            tokensFed += n;
            for (int i = 0; i < n; i++) {
                kv[pos + i] = tokens[off + i];
            }
            for (int i = 0; i < n; i++) {
                logitsAt(pos + i, out, i * stride);
            }
        }

        public void close() {
        }
    }

    static void run(Path dir) throws Exception {
        samplerVerify();
        engine(dir);
    }

    private static SamplingParams greedy(int max) {
        return new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, max, List.of());
    }

    private static Engine engine(FakeBackend b, Architecture cfg, Tokenizer tok, Speculator.Config spec) {
        Engine e = new Engine(b, tok, cfg, Set.of(39));
        if (spec != null) {
            check(e.enableSpeculation(spec, null), "speculation enabled");
        }
        return e;
    }

    /** The accept / resample rule of {@link Sampler#verify}: the result is distributed as a plain sample. */
    static void samplerVerify() {
        int v = 6;
        float[] base = {2.0f, 1.0f, 0.5f, 0f, -1f, 1.5f};
        double[] p = new double[v];
        double z = 0;
        for (int i = 0; i < v; i++) {
            z += Math.exp(base[i]);
        }
        for (int i = 0; i < v; i++) {
            p[i] = Math.exp(base[i]) / z;
        }
        int n = 200_000;
        for (int draft : new int[]{0, 3, 5}) { // the likeliest, an unlikely, a middling token
            SamplingParams sp = new SamplingParams(1f, 0, 1f, 0f, 1f, 0f, 0f, 77, 8, List.of());
            Sampler s = new Sampler(sp, v, new int[0]);
            int[] count = new int[v];
            int accepted = 0;
            for (int i = 0; i < n; i++) {
                int t = s.verify(base.clone(), draft);
                count[t]++;
                accepted += t == draft ? 1 : 0;
            }
            double worst = 0;
            for (int i = 0; i < v; i++) {
                worst = Math.max(worst, Math.abs(count[i] / (double) n - p[i]));
            }
            check(worst < 0.006, "verify(draft " + draft + ") follows the target distribution (max deviation " + worst + ")");
            // a draft token is kept with exactly probability p(draft) when it is the only way to produce it... here
            // the residual excludes it, so P(result == draft) = p(draft)
            check(Math.abs(accepted / (double) n - p[draft]) < 0.006, "draft " + draft + " is accepted with probability p = "
                    + String.format("%.3f vs %.3f", accepted / (double) n, p[draft]));
        }
        // top-k / min-p: a draft outside the truncated distribution is always rejected, results stay inside it
        SamplingParams cut = new SamplingParams(1f, 3, 1f, 0f, 1f, 0f, 0f, 5, 8, List.of());
        Sampler s = new Sampler(cut, v, new int[0]);
        boolean ok = true;
        for (int i = 0; i < 2000; i++) {
            int t = s.verify(base.clone(), 4);
            ok &= t == 0 || t == 1 || t == 5;
        }
        check(ok, "a draft outside the top-k set is rejected and the result stays in the set");
        // greedy: accepted iff the draft is the argmax
        Sampler g = new Sampler(greedy(8), v, new int[0]);
        check(g.verify(base.clone(), 0) == 0 && g.verify(base.clone(), 1) == 0, "greedy verify returns the argmax");
    }

    static void engine(Path dir) throws Exception {
        Path tmp = Files.createTempDirectory("jllm-spec");
        try {
            Architecture cfg = SmokeTest.cfg(tmp, "{\"model_type\":\"mamba\",\"hidden_size\":64,\"num_hidden_layers\":1,"
                    + "\"vocab_size\":" + VOCAB + "}");
            Tokenizer tok = Tokenizer.load(dir.resolve("tokenizer.json"));
            Speculator.Config ng = new Speculator.Config(Speculator.Mode.NGRAM, 5, 3, 1, 4096);
            int[][] prompts = {{5, 6, 7, 8, 5, 6, 7, 8, 5, 6}, {1}, {12, 13, 12, 13, 12, 13, 14, 12, 13}, {20, 9, 2, 33, 4, 4, 4, 4}};
            for (boolean allRows : new boolean[]{true, false}) {
                int drafted = 0;
                int accepted = 0;
                int equalRuns = 0;
                for (int[] prompt : prompts) {
                    FakeBackend plainB = new FakeBackend(allRows);
                    Engine plain = engine(plainB, cfg, tok, null);
                    FakeBackend specB = new FakeBackend(allRows);
                    Engine spec = engine(specB, cfg, tok, ng);
                    Engine.Result a = plain.generate(prompt, greedy(60), null, () -> false);
                    Engine.Result b = spec.generate(prompt, greedy(60), null, () -> false);
                    check(Arrays.equals(a.tokens(), b.tokens()) && a.finishReason().equals(b.finishReason()),
                            "greedy ngram output equals plain output (prompt " + prompt.length + " tokens, "
                                    + a.tokens().length + " generated, " + a.finishReason() + ")");
                    equalRuns++;
                    drafted += b.stats().draftedTokens();
                    accepted += b.stats().acceptedTokens();
                    check(!allRows || specB.forwardCalls == plainB.forwardCalls - b.stats().acceptedTokens(),
                            "each accepted draft token saves one pass (" + specB.forwardCalls + " vs " + plainB.forwardCalls
                                    + ", accepted " + b.stats().acceptedTokens() + ")");
                    // the cached tokens are exactly the accepted ones: a follow-up request that extends the output
                    // reuses all of them and still produces what a fresh engine would
                    int[] ext = new int[prompt.length + b.tokens().length + 2];
                    System.arraycopy(prompt, 0, ext, 0, prompt.length);
                    System.arraycopy(b.tokens(), 0, ext, prompt.length, b.tokens().length);
                    ext[ext.length - 2] = 7;
                    ext[ext.length - 1] = 8;
                    Engine.Result c = spec.generate(ext, greedy(30), null, () -> false);
                    Engine fresh = engine(new FakeBackend(allRows), cfg, tok, null);
                    Engine.Result d = fresh.generate(ext, greedy(30), null, () -> false);
                    check(Arrays.equals(c.tokens(), d.tokens()), "a follow-up request on the cached tokens matches a fresh run (cached "
                            + c.stats().cachedTokens() + " of " + ext.length + ")");
                    check(c.stats().cachedTokens() >= prompt.length, "the follow-up reuses at least the prompt");
                }
                check(drafted > 0 && accepted > 0 && accepted <= drafted, "drafts were proposed and some accepted ("
                        + accepted + "/" + drafted + ")");
                check(equalRuns == prompts.length, "all prompts compared (" + (allRows ? "forwardAll" : "default forwardAll") + ")");
            }

            // stop at EOS / max_tokens in the middle of a verified chunk
            for (int max : new int[]{1, 2, 3, 7, 20}) {
                FakeBackend pb = new FakeBackend(true);
                FakeBackend sb = new FakeBackend(true);
                int[] prompt = {5, 6, 7, 8, 5, 6, 7, 8, 5, 6};
                Engine.Result a = engine(pb, cfg, tok, null).generate(prompt, greedy(max), null, () -> false);
                Engine.Result b = engine(sb, cfg, tok, ng).generate(prompt, greedy(max), null, () -> false);
                check(Arrays.equals(a.tokens(), b.tokens()) && b.tokens().length == a.tokens().length,
                        "max_tokens " + max + " respected under speculation");
            }

            // penalties see the accepted tokens in order: same output with a repetition + frequency penalty
            SamplingParams pen = new SamplingParams(0f, 0, 1f, 0f, 1.3f, 0.2f, 0.4f, 1, 50, List.of());
            for (int[] prompt : prompts) {
                Engine.Result a = engine(new FakeBackend(true), cfg, tok, null).generate(prompt, pen, null, () -> false);
                Engine.Result b = engine(new FakeBackend(true), cfg, tok, ng).generate(prompt, pen, null, () -> false);
                check(Arrays.equals(a.tokens(), b.tokens()), "greedy with penalties: identical output (" + a.tokens().length + " tokens)");
            }

            // sampled: the joint distribution of the first three tokens is the same with and without speculation
            int runs = 20_000;
            int[] cycle = engine(new FakeBackend(true), cfg, tok, null).generate(new int[]{1}, greedy(40), null, () -> false).tokens();
            int[] prompt = new int[cycle.length + 1];
            prompt[0] = 1;
            System.arraycopy(cycle, 0, prompt, 1, cycle.length);
            Map<Integer, Integer> hp = new HashMap<>();
            Map<Integer, Integer> hq = new HashMap<>(); // plain again, other seeds: the sampling noise to compare against
            Map<Integer, Integer> hs = new HashMap<>();
            Engine ep = engine(new FakeBackend(true), cfg, tok, null);
            Engine es = engine(new FakeBackend(true), cfg, tok, ng);
            long acceptedTotal = 0;
            long draftedTotal = 0;
            for (int i = 0; i < runs; i++) {
                float temp = 0.45f;
                Engine.Result a = ep.generate(prompt, new SamplingParams(temp, 0, 1f, 0f, 1f, 0f, 0f, 1000 + i, 3, List.of()), null, () -> false);
                Engine.Result q = ep.generate(prompt, new SamplingParams(temp, 0, 1f, 0f, 1f, 0f, 0f, 900_000 + i, 3, List.of()), null, () -> false);
                Engine.Result b = es.generate(prompt, new SamplingParams(temp, 0, 1f, 0f, 1f, 0f, 0f, 500_000 + i, 3, List.of()), null, () -> false);
                hp.merge(key(a.tokens()), 1, Integer::sum);
                hq.merge(key(q.tokens()), 1, Integer::sum);
                hs.merge(key(b.tokens()), 1, Integer::sum);
                acceptedTotal += b.stats().acceptedTokens();
                draftedTotal += b.stats().draftedTokens();
            }
            double tv = totalVariation(hp, hs, runs);
            double noise = totalVariation(hp, hq, runs);
            check(draftedTotal > runs / 2 && acceptedTotal > 0, "sampled runs speculated (" + acceptedTotal + "/" + draftedTotal + ")");
            check(tv < noise * 1.35 + 0.01, "sampled 3-token distribution with speculation matches the target's (total variation "
                    + String.format("%.4f", tv) + " vs " + String.format("%.4f", noise) + " between two plain runs)");

            // a backend that cannot roll back keeps speculation off
            FakeBackend recurrent = new FakeBackend(true) {
                @Override
                public boolean supportsPrefixReuse() {
                    return false;
                }
            };
            Engine er = new Engine(recurrent, tok, cfg, Set.of(39));
            check(!er.enableSpeculation(ng, null), "speculation is refused for a backend that cannot roll back");

            // a draft model: a second fake backend that agrees with the target except at some positions
            DraftModel dm = new DraftModel(new FakeBackend(true) {
                @Override
                void logitsAt(int p, float[] out, int base) {
                    super.logitsAt(p, out, base);
                    if (p % 5 == 4) { // disagrees here
                        out[base + (kv[p] % VOCAB)] += 50f;
                    }
                }
            }, VOCAB, null);
            Speculator.Config dc = new Speculator.Config(Speculator.Mode.DRAFT, 4, 3, 1, 4096);
            for (int[] pr : prompts) {
                FakeBackend tb = new FakeBackend(true);
                Engine ed = new Engine(tb, tok, cfg, Set.of(39));
                ed.enableSpeculation(dc, dm);
                Engine.Result a = engine(new FakeBackend(true), cfg, tok, null).generate(pr, greedy(40), null, () -> false);
                Engine.Result b = ed.generate(pr, greedy(40), null, () -> false);
                check(Arrays.equals(a.tokens(), b.tokens()), "greedy draft-model output equals plain output ("
                        + b.stats().acceptedTokens() + "/" + b.stats().draftedTokens() + " accepted)");
            }
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    private static double totalVariation(Map<Integer, Integer> a, Map<Integer, Integer> b, int runs) {
        Set<Integer> keys = new java.util.HashSet<>(a.keySet());
        keys.addAll(b.keySet());
        double tv = 0;
        for (int k : keys) {
            tv += Math.abs(a.getOrDefault(k, 0) - b.getOrDefault(k, 0)) / (double) runs;
        }
        return tv / 2;
    }

    private static int key(int[] t) {
        int k = 0;
        for (int x : t) {
            k = k * 61 + x;
        }
        return k;
    }
}
