package org.robomiri.jllm;

import org.robomiri.jllm.config.Settings;
import org.robomiri.jllm.engine.DraftModel;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.Speculator;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.sampling.SamplingParams;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Draft-model speculation ({@code engine.speculative=draft}) on the real model, CPU backend: Qwen3-0.6B drafts for
 * itself. Greedy output with speculation must equal the output without it, token for token, (a) with the draft loaded
 * by {@link ModelLoader} from the settings (every draft accepted: the multi-token verification of
 * {@code -Djllm.cpu.deterministic=true} computes what the single steps compute), and (b) with a draft that is
 * deliberately wrong at every third position (its best token is hidden), so the target rejects drafts and both caches
 * continue from the accepted prefix: the target's positions after it are overwritten by the next pass, the draft
 * re-feeds the history after the longest prefix its cache shares with the new one.
 *
 * <p>usage: DraftSpecTest &lt;model dir&gt; (models/Qwen_Qwen3-0.6B; run with -Djllm.cpu.deterministic=true)
 */
public final class DraftSpecTest {

    private static int failures;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) {
            failures++;
        }
    }

    public static void main(String[] args) throws Exception {
        run(Path.of(args[0]));
        if (failures > 0) {
            System.out.println(failures + " FAILED");
            System.exit(1);
        }
        System.out.println("draft speculation test passed");
    }

    /** Hides the best token of every third position's logits: the draft then proposes its second choice there. */
    private static final class Misleading implements Backend {
        private final Backend inner;

        Misleading(Backend inner) {
            this.inner = inner;
        }

        @Override
        public String name() {
            return "misleading " + inner.name();
        }

        @Override
        public int contextLength() {
            return inner.contextLength();
        }

        @Override
        public int maxBatch() {
            return inner.maxBatch();
        }

        @Override
        public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
            inner.forward(tokens, off, n, pos, logits);
            if (logits != null && (pos + n - 1) % 3 == 1) {
                int best = 0;
                for (int i = 1; i < logits.length; i++) {
                    if (logits[i] > logits[best]) {
                        best = i;
                    }
                }
                logits[best] = Float.NEGATIVE_INFINITY;
            }
        }

        @Override
        public void close() {
            inner.close();
        }
    }

    private static void run(Path dir) throws Exception {
        String[] common = {"--backend", "cpu", "--model.dir", dir.toAbsolutePath().getParent().toString(),
                "--backend.ctx", "512", "--backend.cpu.int8-activations", "false", "--engine.batch-decode", "false"};
        Settings plainSt = Settings.load(concat(common, "--engine.speculative", "off"));
        Settings draftSt = Settings.load(concat(common, "--engine.speculative", "draft", "--engine.draft-model",
                dir.getFileName().toString(), "--engine.draft-tokens", "4"));
        String[] prompts = {"Write a haiku about autumn leaves.", "def fibonacci(n):", "List three prime numbers and explain why."};
        SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 40, List.of());
        String name = dir.getFileName().toString();
        int[][] plain = new int[prompts.length][];
        try (ModelLoader.Loaded l = ModelLoader.load(plainSt, ModelLoader.hub(plainSt), name)) {
            Engine e = l.engine();
            for (int i = 0; i < prompts.length; i++) {
                plain[i] = e.generate(chat(e, prompts[i]), greedy, null, () -> false).tokens();
            }
            // (b) a misleading draft on the same weights
            DraftModel misleading = new DraftModel(new Misleading(ModelLoader.createBackend(plainSt, l.weights().arch,
                    l.weights())), l.weights().arch.core().vocabSize(), null);
            System.setProperty("jllm.engine.adaptiveDraft", "false"); // a pass on every step: as many rejections as possible
            try (Engine spec = new Engine(ModelLoader.createBackend(plainSt, l.weights().arch, l.weights()), e.tokenizer(),
                    l.weights().arch, e.stopTokenIds(), 0, e.template())) {
                check(spec.enableSpeculation(new Speculator.Config(Speculator.Mode.DRAFT, 4, 8, 3, 0), misleading),
                        "speculation enabled with a misleading draft");
                for (int i = 0; i < prompts.length; i++) {
                    Engine.Result r = spec.generate(chat(spec, prompts[i]), greedy, null, () -> false);
                    check(Arrays.equals(plain[i], r.tokens()), "misleading draft, prompt " + i + ": output equals plain greedy ("
                            + r.tokens().length + " tokens, " + r.stats().acceptedTokens() + " of " + r.stats().draftedTokens()
                            + " drafts accepted)");
                    check(r.stats().draftedTokens() > r.stats().acceptedTokens(), "  some drafts were rejected");
                }
            } finally {
                System.clearProperty("jllm.engine.adaptiveDraft");
            }
        }
        // (a) the draft as the settings load it (ModelLoader: draft first, then the target)
        try (ModelLoader.Loaded l = ModelLoader.load(draftSt, ModelLoader.hub(draftSt), name)) {
            Engine e = l.engine();
            for (int i = 0; i < prompts.length; i++) {
                Engine.Result r = e.generate(chat(e, prompts[i]), greedy, null, () -> false);
                check(Arrays.equals(plain[i], r.tokens()) && r.stats().draftedTokens() > 0, "self draft, prompt " + i
                        + ": output equals plain greedy (" + r.tokens().length + " tokens, " + r.stats().acceptedTokens()
                        + " of " + r.stats().draftedTokens() + " drafts accepted)");
            }
        }
    }

    private static String[] concat(String[] a, String... b) {
        String[] r = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }

    private static int[] chat(Engine e, String user) {
        return e.renderChat(List.of(new org.robomiri.jllm.tokenizer.ChatTemplate.Message("user", user + " /no_think")),
                false);
    }
}
