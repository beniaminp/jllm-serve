package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Pipeline test for the model families on tiny random checkpoints: for each model directory and quantization
 * the checkpoint is converted, loaded, and 8 tokens are generated greedily (from a fixed 12-token prompt, so
 * no tokenizer is needed) on the CPU backend and on the Metal backend; the two id sequences must be equal.
 *
 * <p>usage: {@code ArchSmoke <metal-lib|-> <model-dir>...}. Exits with status 1 on any mismatch.
 */
public final class ArchSmoke {

    private static final int[] PROMPT = {1, 7, 42, 99, 13, 5, 250, 8, 77, 31, 600, 9};
    private static final int NEW_TOKENS = 8;

    public static void main(String[] args) throws Exception {
        String lib = args[0].equals("-") ? null : args[0];
        boolean metal = Metal.available(lib);
        if (!metal) {
            System.out.println("note: Metal unavailable, comparing nothing on the GPU");
        }
        List<QType> quants = new ArrayList<>(List.of(QType.Q4_0, QType.Q8_0));
        try {
            quants.add(QType.valueOf("Q2_K"));
        } catch (IllegalArgumentException e) {
            // Q2_K lives on another branch
        }
        int failures = 0;
        for (int i = 1; i < args.length; i++) {
            Path dir = Path.of(args[i]);
            ModelConfig cfg = ModelConfig.load(dir);
            if (cfg.dim() % QType.BLOCK != 0 || cfg.hiddenDim() % QType.BLOCK != 0 || cfg.qDim() % QType.BLOCK != 0) {
                System.out.println(dir.getFileName() + ": skipped, dims (" + cfg.dim() + ", " + cfg.hiddenDim() + ", "
                        + cfg.qDim() + ") are not multiples of the " + QType.BLOCK + "-wide quantization block");
                continue;
            }
            for (QType q : quants) {
                failures += run(dir, cfg, q, metal, lib);
            }
            if (cfg.modelType().equals("phi3")) {
                failures += checkFusedSplit(dir, cfg);
            }
        }
        if (failures > 0) {
            System.out.println("FAILED: " + failures);
            System.exit(1);
        }
        System.out.println("arch smoke passed");
    }

    /**
     * The converter must turn Phi-3's fused {@code qkv_proj} / {@code gate_up_proj} into the q/k/v cache tensors and
     * {@code mlp.gate_up}: every dequantized row (Q8_0) equals the matching row of the fused checkpoint tensor
     * within the quantization step, with the row ranges q, then k, then v / gate, then up.
     */
    private static int checkFusedSplit(Path dir, ModelConfig cfg) throws Exception {
        int bad = 0;
        try (ModelWeights w = ModelWeights.load(dir, cfg, QType.Q8_0, ShardProvider.local(dir));
             SafeTensors st = SafeTensors.open(dir.resolve("model.safetensors"))) {
            for (int l = 0; l < cfg.layers(); l++) {
                String p = "model.layers." + l + ".";
                SafeTensors.Tensor qkv = st.get(p + "self_attn.qkv_proj.weight");
                SafeTensors.Tensor gu = st.get(p + "mlp.gate_up_proj.weight");
                ModelWeights.Layer L = w.layers[l];
                bad += compareRows(qkv, 0, L.wq, p + "q");
                bad += compareRows(qkv, cfg.qDim(), L.wk, p + "k");
                bad += compareRows(qkv, cfg.qDim() + cfg.kvDim(), L.wv, p + "v");
                bad += compareRows(gu, 0, L.wGateUp, p + "gate_up");
            }
        }
        System.out.println(dir.getFileName() + " fused qkv / gate_up split: " + (bad == 0 ? "ok" : bad + " BAD ROWS"));
        return bad == 0 ? 0 : 1;
    }

    private static int compareRows(SafeTensors.Tensor fused, int rowStart, QTensor part, String what) {
        int cols = part.cols;
        float[] ref = new float[cols];
        float[] got = new float[cols];
        int bad = 0;
        for (int r = 0; r < part.rows; r++) {
            fused.toFloat((long) (rowStart + r) * cols, ref, 0, cols);
            part.dequantizeRow(r, got, 0);
            float max = 0;
            for (float v : ref) {
                max = Math.max(max, Math.abs(v));
            }
            for (int i = 0; i < cols; i++) {
                if (Math.abs(ref[i] - got[i]) > max / 127f * 0.51f + 1e-7f) {
                    System.out.println("  " + what + " row " + r + " col " + i + ": " + ref[i] + " vs " + got[i]);
                    bad++;
                    break;
                }
            }
        }
        return bad;
    }

    private static int run(Path dir, ModelConfig cfg, QType quant, boolean metal, String lib) throws Exception {
        String label = dir.getFileName() + " " + quant;
        try (ModelWeights w = ModelWeights.load(dir, cfg, quant, ShardProvider.local(dir))) {
            int[] cpu;
            float[] cpuLogits = new float[cfg.vocabSize()];
            try (Backend b = new CpuBackend(w, 64, 16, 4)) {
                cpu = generate(b, cpuLogits);
            }
            System.out.println(label + "  " + cfg);
            System.out.println("  cpu   " + Arrays.toString(cpu));
            if (!metal) {
                return 0;
            }
            if (cfg.headDim() % 32 != 0) {
                System.out.println("  metal skipped: headDim " + cfg.headDim() + " is not a multiple of 32");
                return 0;
            }
            int[] gpu;
            float[] gpuLogits = new float[cfg.vocabSize()];
            try (Backend b = new MetalBackend(w, 64, 16, lib)) {
                gpu = generate(b, gpuLogits);
            }
            System.out.println("  metal " + Arrays.toString(gpu));
            // the argmax of a random model can be degenerate, so also compare the full final logits
            float scale = 0;
            float diff = 0;
            for (int i = 0; i < cpuLogits.length; i++) {
                scale = Math.max(scale, Math.abs(cpuLogits[i]));
                diff = Math.max(diff, Math.abs(cpuLogits[i] - gpuLogits[i]));
            }
            System.out.printf("  final logits: max|cpu-metal| = %.2e (max|logit| = %.2f)%n", diff, scale);
            if (!(diff <= 1e-3f * scale + 1e-4f)) {
                System.out.println("  LOGITS MISMATCH");
                return 1;
            }
            if (!Arrays.equals(cpu, gpu)) {
                System.out.println("  MISMATCH");
                return 1;
            }
            System.out.println("  cpu == metal");
            return 0;
        }
    }

    private static int[] generate(Backend b, float[] logits) {
        int vocab = logits.length;
        b.forward(PROMPT, 0, PROMPT.length, 0, logits);
        int[] out = new int[NEW_TOKENS];
        int pos = PROMPT.length;
        for (int i = 0; i < NEW_TOKENS; i++) {
            int best = 0;
            for (int t = 1; t < vocab; t++) {
                if (logits[t] > logits[best]) {
                    best = t;
                }
            }
            out[i] = best;
            if (i + 1 < NEW_TOKENS) {
                b.forward(new int[]{best}, 0, 1, pos++, logits);
            }
        }
        return out;
    }
}
