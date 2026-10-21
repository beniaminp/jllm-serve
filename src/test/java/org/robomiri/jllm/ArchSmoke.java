package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.cpu.MambaCpuBackend;
import org.robomiri.jllm.metal.MambaMetalBackend;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.metal.MoeSplit;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.AttnWeights;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.LayerSpec;
import org.robomiri.jllm.model.MambaArch;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.QuantFile;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.nio.file.Path;
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
        List<QType> quants = List.of(QType.Q4_0, QType.Q8_0, QType.Q2_K, QType.Q6_K);
        int failures = 0;
        for (int i = 1; i < args.length; i++) {
            Path dir = Path.of(args[i]);
            Architecture cfg = Architecture.load(dir);
            if (!quantizable(cfg)) {
                System.out.println(dir.getFileName() + ": skipped, a width is not a multiple of the " + QType.BLOCK
                        + "-wide quantization block");
                continue;
            }
            for (QType q : quants) {
                failures += run(dir, cfg, q, QuantFile.DEFAULT_HEAD, metal, lib);
            }
            failures += run(dir, cfg, QType.Q4_0, QType.Q6_K, metal, lib);
            if (cfg.core().modelType().equals("phi3")) {
                failures += checkFusedSplit(dir, cfg);
            }
        }
        if (failures > 0) {
            System.out.println("FAILED: " + failures);
            System.exit(1);
        }
        System.out.println("arch smoke passed");
    }

    /** Every quantized width (the residual stream, the MLPs, the query projection / mixer) is a multiple of the block. */
    private static boolean quantizable(Architecture a) {
        int b = QType.BLOCK;
        if (a.core().dim() % b != 0) {
            return false;
        }
        return switch (a) {
            case MambaArch m -> m.dInner() % b == 0;
            case Decoder d -> {
                boolean ok = d.attention().qDim() % b == 0;
                for (LayerSpec s : d.layers()) {
                    ok &= !(s.mlp() instanceof Mlp.Dense dense) || dense.hidden() % b == 0;
                }
                yield ok;
            }
        };
    }

    /**
     * The converter must turn Phi-3's fused {@code qkv_proj} / {@code gate_up_proj} into the q/k/v cache tensors and
     * {@code mlp.gate_up}: every dequantized row (Q8_0) equals the matching row of the fused checkpoint tensor
     * within the quantization step, with the row ranges q, then k, then v / gate, then up.
     */
    private static int checkFusedSplit(Path dir, Architecture cfg) throws Exception {
        int bad = 0;
        Attention.Gqa g = (Attention.Gqa) ((Decoder) cfg).attention();
        try (ModelWeights w = ModelWeights.load(dir, cfg, QType.Q8_0, ShardProvider.local(dir));
             SafeTensors st = SafeTensors.open(dir.resolve("model.safetensors"))) {
            for (int l = 0; l < cfg.core().layers(); l++) {
                String p = "model.layers." + l + ".";
                SafeTensors.Tensor qkv = st.get(p + "self_attn.qkv_proj.weight");
                SafeTensors.Tensor gu = st.get(p + "mlp.gate_up_proj.weight");
                AttnWeights.Gqa L = w.decoderLayer(l).gqa();
                bad += compareRows(qkv, 0, L.wq(), p + "q");
                bad += compareRows(qkv, g.qDim(), L.wk(), p + "k");
                bad += compareRows(qkv, g.qDim() + g.kvDim(), L.wv(), p + "v");
                bad += compareRows(gu, 0, w.decoderLayer(l).dense().gateUp(), p + "gate_up");
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

    private static int run(Path dir, Architecture cfg, QType quant, QType head, boolean metal, String lib) throws Exception {
        String label = dir.getFileName() + " " + quant + (head == QuantFile.DEFAULT_HEAD ? "" : " head " + head);
        int failures = 0;
        boolean mamba = cfg instanceof MambaArch;
        try (ModelWeights w = ModelWeights.load(dir, cfg, quant, head, ShardProvider.local(dir), null)) {
            int[] cpu;
            float[] cpuLogits = new float[cfg.core().vocabSize()];
            try (Backend b = mamba ? new MambaCpuBackend(w, 64, 16, 4) : new CpuBackend(w, 64, 16, 4, KvDtype.F32, false)) {
                cpu = generate(b, cpuLogits);
                failures += mamba ? stateChecks(b, cfg, "cpu") : 0;
            }
            System.out.println(label + "  " + cfg);
            System.out.println("  cpu   " + Arrays.toString(cpu));
            if (!metal) {
                return failures;
            }
            if (cfg instanceof Decoder d && d.attention().headDim() % 32 != 0) {
                System.out.println("  metal skipped: headDim " + d.attention().headDim() + " is not a multiple of 32");
                return failures;
            }
            int[] gpu;
            float[] gpuLogits = new float[cfg.core().vocabSize()];
            try (Backend b = mamba ? new MambaMetalBackend(w, 64, 16, lib) : new MetalBackend(w, 64, 16, lib)) {
                gpu = generate(b, gpuLogits);
                failures += mamba ? stateChecks(b, cfg, "metal") : 0;
            }
            System.out.println("  metal " + Arrays.toString(gpu));
            if (!mamba) {
                failures += pipeCheck(w, lib, cfg, gpu, label);
            }
            if (cfg instanceof Decoder d && d.anyMoe()) {
                failures += moeSplitCheck(w, lib, cfg, cpu, cpuLogits);
            }
            if (!mamba && quant == QType.Q4_0 && head == QuantFile.DEFAULT_HEAD && ((Decoder) cfg).attention() instanceof Attention.Gqa) {
                for (KvDtype dt : KvDtype.values()) {
                    failures += kvStateChecks(w, lib, cfg, dt);
                }
            }
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
                return failures + 1;
            }
            if (!Arrays.equals(cpu, gpu)) {
                System.out.println("  MISMATCH");
                return failures + 1;
            }
            System.out.println("  cpu == metal");
            return failures;
        }
    }

    /**
     * Experts on the CPU (backend.metal.cpu-moe): with the routed experts of one MoE layer, and of all of them, run by
     * the CPU between the GPU's layers, the generation and the final logits match the CPU backend within the usual
     * CPU / Metal tolerance (prefill takes the batched expert path, decode the single-token one; DeepSeek's shared
     * experts run on the GPU at the same time).
     */
    private static int moeSplitCheck(ModelWeights w, String lib, Architecture cfg, int[] cpu, float[] cpuLogits) throws Exception {
        int bad = 0;
        for (int layers : new int[]{1, Integer.MAX_VALUE}) {
            float[] logits = new float[cfg.core().vocabSize()];
            int[] got;
            try (Backend b = new MetalBackend(w, 64, 16, lib, KvDtype.F32, new MoeSplit(layers, 4, false))) {
                got = generate(b, logits);
            }
            float scale = 0;
            float diff = 0;
            for (int i = 0; i < logits.length; i++) {
                scale = Math.max(scale, Math.abs(cpuLogits[i]));
                diff = Math.max(diff, Math.abs(cpuLogits[i] - logits[i]));
            }
            boolean ok = Arrays.equals(cpu, got) && diff <= 1e-3f * scale + 1e-4f;
            System.out.printf("  experts on cpu (%s layers): %s max|cpu-split| = %.2e -> %s%n",
                    layers == Integer.MAX_VALUE ? "all" : layers, Arrays.toString(got), diff, ok ? "ok" : "SPLIT MISMATCH");
            bad += ok ? 0 : 1;
        }
        return bad;
    }

    /**
     * Mamba keeps a recurrent state instead of a KV cache, so the way the prompt is cut into forward() calls must not
     * change the result: one chunk, 5 + 7, and token by token give the same logits, and pos == 0 starts afresh
     * (here after the state was left in some other position). Continuing at a wrong position is refused.
     */
    private static int stateChecks(Backend b, Architecture cfg, String name) {
        float[] one = new float[cfg.core().vocabSize()];
        float[] split = new float[cfg.core().vocabSize()];
        float[] single = new float[cfg.core().vocabSize()];
        b.forward(PROMPT, 0, PROMPT.length, 0, one);
        b.forward(PROMPT, 0, 5, 0, null);
        b.forward(PROMPT, 5, PROMPT.length - 5, 5, split);
        for (int i = 0; i < PROMPT.length; i++) {
            b.forward(PROMPT, i, 1, i, i == PROMPT.length - 1 ? single : null);
        }
        float scale = 0;
        float d1 = 0;
        float d2 = 0;
        for (int i = 0; i < one.length; i++) {
            scale = Math.max(scale, Math.abs(one[i]));
            d1 = Math.max(d1, Math.abs(one[i] - split[i]));
            d2 = Math.max(d2, Math.abs(one[i] - single[i]));
        }
        boolean refused = false;
        try {
            b.forward(PROMPT, 0, 1, PROMPT.length + 3, null);
        } catch (IllegalStateException e) {
            refused = true;
        }
        boolean ok = d1 <= 1e-4f * scale + 1e-5f && d2 <= 1e-4f * scale + 1e-5f && refused && !b.supportsPrefixReuse();
        System.out.printf("  %s state: one chunk vs 5+7 %.1e, vs token by token %.1e, wrong pos refused=%b -> %s%n", name, d1,
                d2, refused, ok ? "ok" : "FAILED");
        return ok ? 0 : 1;
    }

    /**
     * Saved caches and prefix copies on the Metal backend: a cache saved by one backend and loaded into another, or
     * copied from it, continues with exactly the logits of the original; a cache saved by the CPU backend has the
     * same layout and loads into the Metal backend (the continuation then agrees within the usual CPU / Metal
     * tolerance).
     */
    private static int kvStateChecks(ModelWeights w, String lib, Architecture cfg, KvDtype dt) throws Exception {
        int vocab = cfg.core().vocabSize();
        int n = PROMPT.length;
        int[] next = {PROMPT[3]};
        float[] want = new float[vocab];
        float[] got = new float[vocab];
        int bad = 0;
        try (MetalBackend a = new MetalBackend(w, 64, 16, lib, dt); MetalBackend b = new MetalBackend(w, 64, 16, lib, dt);
             MetalBackend c = new MetalBackend(w, 64, 16, lib, dt); CpuBackend cpu = new CpuBackend(w, 64, 16, 4, dt, false)) {
            a.forward(PROMPT, 0, n, 0, null);
            var bytes = new java.io.ByteArrayOutputStream();
            a.saveState(n, java.nio.channels.Channels.newChannel(bytes));
            b.loadState(n, java.nio.channels.Channels.newChannel(new java.io.ByteArrayInputStream(bytes.toByteArray())));
            c.copyStateFrom(a, n);
            a.forward(next, 0, 1, n, want);
            b.forward(next, 0, 1, n, got);
            boolean loaded = Arrays.equals(want, got);
            c.forward(next, 0, 1, n, got);
            boolean copied = Arrays.equals(want, got);
            // the CPU backend's file in the Metal backend
            cpu.forward(PROMPT, 0, n, 0, null);
            var cpuBytes = new java.io.ByteArrayOutputStream();
            cpu.saveState(n, java.nio.channels.Channels.newChannel(cpuBytes));
            boolean sameFormat = cpu.stateFormat().equals(a.stateFormat()) && cpuBytes.size() == bytes.size();
            b.loadState(n, java.nio.channels.Channels.newChannel(new java.io.ByteArrayInputStream(cpuBytes.toByteArray())));
            b.forward(next, 0, 1, n, got);
            float scale = 0;
            float diff = 0;
            for (int i = 0; i < vocab; i++) {
                scale = Math.max(scale, Math.abs(want[i]));
                diff = Math.max(diff, Math.abs(want[i] - got[i]));
            }
            boolean crossOk = diff <= 1e-3f * scale + 1e-4f;
            System.out.printf("  kv %s: saved+loaded %s, copied %s, cpu file in metal %s (max diff %.1e)%n", dt,
                    loaded ? "identical" : "DIFFERENT", copied ? "identical" : "DIFFERENT",
                    sameFormat && crossOk ? "ok" : "MISMATCH", diff);
            bad += (loaded ? 0 : 1) + (copied ? 0 : 1) + (sameFormat && crossOk ? 0 : 1);
        }
        return bad;
    }

    /**
     * Pipelined greedy decoding (Backend.GreedyPipe: the GPU feeds its own argmax to the next step) gives the tokens of
     * plain forward + argmax on the same backend.
     */
    private static int pipeCheck(ModelWeights w, String lib, Architecture cfg, int[] want, String label) {
        float[] logits = new float[cfg.core().vocabSize()];
        try (MetalBackend b = new MetalBackend(w, 64, 16, lib)) {
            Backend.GreedyPipe pipe = b.greedyPipe(cfg.core().vocabSize());
            if (pipe == null) {
                return 0;
            }
            b.forward(PROMPT, 0, PROMPT.length, 0, logits);
            int[] got = new int[NEW_TOKENS];
            int best = 0;
            for (int t = 1; t < logits.length; t++) {
                if (logits[t] > logits[best]) {
                    best = t;
                }
            }
            got[0] = best;
            int pos = PROMPT.length;
            pipe.submit(best, pos);
            for (int i = 1; i < NEW_TOKENS; i++) {
                if (i + 1 < NEW_TOKENS) {
                    pipe.submit(-1, pos + i); // queued before the token it feeds is read
                }
                got[i] = pipe.next();
            }
            pipe.drain();
            boolean ok = Arrays.equals(got, want);
            System.out.println("  pipe  " + Arrays.toString(got) + (ok ? "" : "  PIPELINE MISMATCH"));
            return ok ? 0 : 1;
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
