package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.ShardProvider;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Checks the DeepSeek forward pass against {@code scripts/deepseek-ref.py}, an independent float64 reference
 * written after modeling_deepseek.py: the last-token logits of the 12-token ArchSmoke prompt, computed from the
 * checkpoint with every matrix rounded to Q8_0 exactly as the converter does ({@code REF_Q8=1}), must be matched
 * by the Q8_0 engine (CPU and, when available, Metal) to float rounding, so a wrong rope, scale or routing rule
 * fails (without the rounding the two differ by 1-6 % of the logit range, which is quantization noise).
 *
 * <p>usage: {@code DeepSeekRef <metal-lib|-> <model-dir>...}; each dir needs {@code ref-logits.json}
 * ({@code REF_Q8=1 python3 scripts/deepseek-ref.py <dir>}).
 */
public final class DeepSeekRef {

    private static final int[] PROMPT = {1, 7, 42, 99, 13, 5, 250, 8, 77, 31, 600, 9};

    public static void main(String[] args) throws Exception {
        String lib = args[0].equals("-") ? null : args[0];
        boolean metal = Metal.available(lib);
        int failures = 0;
        for (int i = 1; i < args.length; i++) {
            Path dir = Path.of(args[i]);
            Architecture cfg = Architecture.load(dir);
            List<?> ref = Json.list(Json.parseObject(Files.readString(dir.resolve("ref-logits.json"))), "logits");
            float[] expect = new float[ref.size()];
            for (int k = 0; k < expect.length; k++) {
                expect[k] = ((Number) ref.get(k)).floatValue();
            }
            try (ModelWeights w = ModelWeights.load(dir, cfg, QType.Q8_0, ShardProvider.local(dir))) {
                float[] cpu = new float[cfg.core().vocabSize()];
                try (Backend b = new CpuBackend(w, 64, 16, 4, KvDtype.F32, false)) {
                    b.forward(PROMPT, 0, PROMPT.length, 0, cpu);
                }
                failures += compare(dir.getFileName() + " cpu", expect, cpu);
                if (metal) {
                    float[] gpu = new float[cfg.core().vocabSize()];
                    try (Backend b = new MetalBackend(w, 64, 16, lib)) {
                        b.forward(PROMPT, 0, PROMPT.length, 0, gpu);
                    }
                    failures += compare(dir.getFileName() + " metal", expect, gpu);
                }
            }
        }
        if (failures > 0) {
            System.out.println("FAILED: " + failures);
            System.exit(1);
        }
        System.out.println("deepseek reference check passed");
    }

    private static int compare(String label, float[] ref, float[] got) {
        double scale = 0;
        double diff = 0;
        int ra = 0;
        int ga = 0;
        for (int i = 0; i < ref.length; i++) {
            scale = Math.max(scale, Math.abs(ref[i]));
            diff = Math.max(diff, Math.abs(ref[i] - got[i]));
            ra = ref[i] > ref[ra] ? i : ra;
            ga = got[i] > got[ga] ? i : ga;
        }
        double rel = diff / scale;
        System.out.printf("%s: max|ref-engine| / max|ref| = %.2e (argmax ref %d, engine %d)%n", label, rel, ra, ga);
        if (rel > 2e-3) {
            System.out.println("  MISMATCH vs reference");
            return 1;
        }
        return 0;
    }
}
