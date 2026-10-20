package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.engine.Engine;
import org.robomiri.jllm.engine.Speculator;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.QuantFile;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.sampling.SamplingParams;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Whole-engine greedy generation on the Metal backend, where plain decode steps are pipelined (the GPU feeds its own
 * argmax to the next step), against the CPU backend: the same tokens with speculation off, and with prompt-lookup
 * speculation on (verification passes drain the pipeline in between). Tiny random models, any tokenizer.
 *
 * <p>usage: {@code EnginePipeTest <metal-lib> <tokenizer-dir> <model-dir>...} (with -Djllm.metal.emulator=true on a
 * machine without an Apple GPU).
 */
public final class EnginePipeTest {

    public static void main(String[] args) throws Exception {
        String lib = args[0];
        if (!Metal.available(lib)) {
            System.out.println("Metal unavailable: nothing to compare");
            return;
        }
        Tokenizer tok = Tokenizer.load(Path.of(args[1]).resolve("tokenizer.json"));
        // a prompt that repeats itself, so prompt lookup finds drafts
        int[] prompt = {11, 12, 13, 14, 15, 16, 17, 11, 12, 13, 14, 15, 16, 17, 11, 12, 13};
        SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, 24, List.of());
        int failures = 0;
        for (int i = 2; i < args.length; i++) {
            Path dir = Path.of(args[i]);
            Architecture cfg = Architecture.load(dir);
            try (ModelWeights w = ModelWeights.load(dir, cfg, QType.Q8_0, QuantFile.DEFAULT_HEAD, ShardProvider.local(dir), null)) {
                for (boolean spec : new boolean[]{false, true}) {
                    int[] cpu;
                    int[] gpu;
                    try (Engine e = new Engine(new CpuBackend(w, 128, 16, 4, KvDtype.F32, false), tok, cfg, Set.of(-1))) {
                        enable(e, spec);
                        cpu = e.generate(prompt, greedy, null, () -> false).tokens();
                    }
                    long drafted;
                    try (Engine e = new Engine(new MetalBackend(w, 128, 16, lib), tok, cfg, Set.of(-1))) {
                        enable(e, spec);
                        Engine.Result r = e.generate(prompt, greedy, null, () -> false);
                        gpu = r.tokens();
                        drafted = r.stats().draftedTokens();
                        // a second request on the cache the pipelined steps left behind
                        int[] p2 = Arrays.copyOf(prompt, prompt.length + 3);
                        p2[prompt.length] = gpu[0];
                        p2[prompt.length + 1] = gpu[1];
                        p2[prompt.length + 2] = 5;
                        int[] gpu2 = e.generate(p2, greedy, null, () -> false).tokens();
                        try (Engine c = new Engine(new CpuBackend(w, 128, 16, 4, KvDtype.F32, false), tok, cfg, Set.of(-1))) {
                            enable(c, spec);
                            int[] cpu2 = c.generate(p2, greedy, null, () -> false).tokens();
                            boolean ok2 = Arrays.equals(cpu2, gpu2);
                            System.out.println("  follow-up request " + (ok2 ? "matches" : "DIFFERS: " + Arrays.toString(gpu2)
                                    + " vs " + Arrays.toString(cpu2)));
                            failures += ok2 ? 0 : 1;
                        }
                    }
                    boolean ok = Arrays.equals(cpu, gpu);
                    System.out.println(dir.getFileName() + (spec ? " speculation on (" + drafted + " drafted)" : " speculation off")
                            + ": " + (ok ? "metal (pipelined) == cpu" : "MISMATCH\n  cpu   " + Arrays.toString(cpu)
                            + "\n  metal " + Arrays.toString(gpu)));
                    failures += ok ? 0 : 1;
                }
            }
        }
        if (failures > 0) {
            System.out.println("FAILED: " + failures);
            System.exit(1);
        }
        System.out.println("engine pipe test passed");
    }

    private static void enable(Engine e, boolean spec) {
        if (spec) {
            e.enableSpeculation(new Speculator.Config(Speculator.Mode.NGRAM, 8, 4, 2, 0), null);
        }
    }
}
