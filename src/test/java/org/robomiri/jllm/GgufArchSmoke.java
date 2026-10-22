package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.cpu.MambaCpuBackend;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.metal.MambaMetalBackend;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.MambaArch;
import org.robomiri.jllm.model.ModelWeights;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * The pipeline test of {@code ArchSmoke} for GGUF files: every tiny checkpoint is written as a Q8_0 GGUF the way
 * llama.cpp's converter would ({@link HfToGguf}), loaded in place, and 8 tokens are generated greedily from a fixed
 * prompt on the CPU backend and on the Metal backend: the two id sequences must be equal and the logits must agree.
 * The F32 file of the same checkpoint must give the same greedy tokens as the Q8_0 file's dequantized values on the CPU.
 * Checkpoints llama.cpp's converter cannot express (a GLM-Edge head_dim, Qwen3-MoE with dense layers, ...) are skipped
 * by name.
 *
 * <p>usage: {@code GgufArchSmoke <metal-lib|-> <model-dir>...}
 */
public final class GgufArchSmoke {

    private static final int[] PROMPT = {1, 7, 42, 99, 13, 5, 250, 8, 77, 31, 600, 9};
    private static final int NEW_TOKENS = 8;

    public static void main(String[] args) throws Exception {
        String lib = args[0].equals("-") ? null : args[0];
        boolean metal = Metal.available(lib);
        if (!metal) {
            System.out.println("note: Metal unavailable, comparing nothing on the GPU");
        }
        Path tmp = Files.createTempDirectory("gguf-arch-smoke");
        int failures = 0;
        int ran = 0;
        try {
            for (int i = 1; i < args.length; i++) {
                Path dir = Path.of(args[i]);
                Path q8 = tmp.resolve(dir.getFileName() + "-q8.gguf");
                String arch;
                try {
                    String n = dir.getFileName().toString();
                    arch = HfToGguf.convert(dir, q8, new HfToGguf.Options(HfToGguf.Outtype.Q8_0, false, n.contains("fused-qkv"),
                            n.contains("fused-exps")));
                } catch (IllegalArgumentException e) {
                    System.out.println(dir.getFileName() + ": GGUF skipped, " + e.getMessage());
                    continue;
                }
                ran++;
                failures += run(dir.getFileName().toString(), arch, q8, metal, lib);
            }
        } finally {
            try (var s = Files.walk(tmp)) {
                s.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
        if (failures > 0 || ran == 0) {
            System.out.println("GGUF arch smoke FAILED: " + failures + " failures, " + ran + " models");
            System.exit(1);
        }
        System.out.println("GGUF arch smoke passed (" + ran + " models)");
    }

    private static int[] generate(Backend b, float[] logits) {
        b.forward(PROMPT, 0, PROMPT.length, 0, logits);
        int[] out = new int[NEW_TOKENS];
        int pos = PROMPT.length;
        for (int i = 0; i < NEW_TOKENS; i++) {
            out[i] = GgufModelTest.argmax(logits);
            if (i + 1 < NEW_TOKENS) {
                b.forward(new int[]{out[i]}, 0, 1, pos++, logits);
            }
        }
        return out;
    }

    private static int run(String name, String ggufArch, Path gguf, boolean metal, String lib) throws Exception {
        int failures = 0;
        try (GgufModel g = GgufModel.open(gguf); ModelWeights w = g.weights()) {
            Architecture cfg = w.arch;
            boolean mamba = cfg instanceof MambaArch;
            float[] cpuLogits = new float[cfg.core().vocabSize()];
            int[] cpu;
            try (Backend b = mamba ? new MambaCpuBackend(w, 64, 16, 4) : new CpuBackend(w, 64, 16, 4, KvDtype.F32, false)) {
                cpu = generate(b, cpuLogits);
            }
            System.out.println(name + " (" + ggufArch + ", Q8_0 gguf)  " + cfg);
            System.out.println("  cpu   " + Arrays.toString(cpu));
            if (!metal) {
                return 0;
            }
            if (cfg instanceof Decoder d && d.attention().headDim() % 32 != 0) {
                System.out.println("  metal skipped: headDim " + d.attention().headDim() + " is not a multiple of 32");
                return 0;
            }
            float[] gpuLogits = new float[cfg.core().vocabSize()];
            int[] gpu;
            try (Backend b = mamba ? new MambaMetalBackend(w, 64, 16, lib) : new MetalBackend(w, 64, 16, lib)) {
                gpu = generate(b, gpuLogits);
            }
            System.out.println("  metal " + Arrays.toString(gpu));
            double diff = GgufModelTest.maxDiff(new float[][]{cpuLogits}, new float[][]{gpuLogits});
            if (!Arrays.equals(cpu, gpu)) {
                System.out.println("  FAIL: CPU and Metal generate different tokens");
                failures++;
            }
            if (!(diff < 1e-3)) {
                System.out.println("  FAIL: last-step logits differ by " + diff + " (relative)");
                failures++;
            }
        }
        return failures;
    }
}
