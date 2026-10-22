package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.cpu.MambaCpuBackend;
import org.robomiri.jllm.cpu.Qwen3NextCpuBackend;
import org.robomiri.jllm.engine.Speculator;
import org.robomiri.jllm.metal.MambaMetalBackend;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.metal.MoeSplit;
import org.robomiri.jllm.metal.Qwen3NextMetalBackend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.AttnWeights;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.KvQ4;
import org.robomiri.jllm.model.LayerSpec;
import org.robomiri.jllm.model.MambaArch;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.QuantFile;
import org.robomiri.jllm.model.Qwen3NextArch;
import org.robomiri.jllm.model.ShardProvider;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Pipeline test for the model families on tiny random checkpoints: for each model directory and quantization
 * the checkpoint is converted, loaded, and 8 tokens are generated greedily (from a fixed 12-token prompt, so
 * no tokenizer is needed) on the CPU backend and on the Metal backend; the two id sequences must be equal. Every
 * family is also run with the q8 KV cache on both backends (see {@code q8Check}).
 *
 * <p>usage: {@code ArchSmoke <metal-lib|-> <model-dir>...}. Exits with status 1 on any mismatch.
 */
public final class ArchSmoke {

    private static final int[] PROMPT = {1, 7, 42, 99, 13, 5, 250, 8, 77, 31, 600, 9};
    private static final int NEW_TOKENS = 8;
    /**
     * -Djllm.smoke.q4=false leaves out the q4-cache comparisons. Only the forced-MMA pass of arch-smoke.sh sets it: with
     * MMA prefill from 2 rows a batched pass and a lone step take different matmul kernels, and the q4 cache turns
     * their rounding-level difference into a flipped 4-bit code (4e-3 on Phi3Mha); the default pass checks every q4 case.
     */
    private static final boolean Q4 = !"false".equals(System.getProperty("jllm.smoke.q4"));

    public static void main(String[] args) throws Exception {
        String lib = args[0].equals("-") ? null : args[0];
        boolean metal = Metal.available(lib);
        if (!metal) {
            System.out.println("note: Metal unavailable, comparing nothing on the GPU");
        }
        // -Djllm.smoke.quants=Q4_0,Q8_0 limits the run to those formats (and drops the Q6_K-head variant unless Q6_K is listed)
        String only = System.getProperty("jllm.smoke.quants");
        List<QType> quants = List.of(QType.Q4_0, QType.Q8_0, QType.Q2_K, QType.Q6_K).stream()
                .filter(q -> only == null || List.of(only.split(",")).contains(q.name())).toList();
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
            if (only == null || List.of(only.split(",")).contains("Q6_K")) {
                failures += run(dir, cfg, QType.Q4_0, QType.Q6_K, metal, lib);
            }
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
            case Qwen3NextArch q -> {
                boolean ok = q.attention().qDim() % b == 0 && q.deltaNet().valueWidth() % b == 0;
                for (Mlp p : q.mlps()) {
                    ok &= switch (p) {
                        case Mlp.Dense dense -> dense.hidden() % b == 0;
                        case Mlp.Moe moe -> moe.moeDim() % b == 0 && moe.shared().hidden() % b == 0;
                    };
                }
                yield ok;
            }
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
        boolean hybrid = cfg instanceof Qwen3NextArch; // Qwen3-Next: recurrent DeltaNet states plus a KV cache
        boolean recurrent = mamba || hybrid;
        try (ModelWeights w = ModelWeights.load(dir, cfg, quant, head, ShardProvider.local(dir), null)) {
            int[] cpu;
            float[] cpuLogits = new float[cfg.core().vocabSize()];
            try (Backend b = mamba ? new MambaCpuBackend(w, 64, 16, 4)
                    : hybrid ? new Qwen3NextCpuBackend(w, 64, 16, 4, KvDtype.F32)
                    : new CpuBackend(w, 64, 16, 4, KvDtype.F32, false)) {
                cpu = generate(b, cpuLogits);
                failures += recurrent ? stateChecks(b, cfg, "cpu") : 0;
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
            try (Backend b = mamba ? new MambaMetalBackend(w, 64, 16, lib)
                    : hybrid ? new Qwen3NextMetalBackend(w, 64, 16, lib, KvDtype.F32)
                    : new MetalBackend(w, 64, 16, lib)) {
                gpu = generate(b, gpuLogits);
                failures += recurrent ? stateChecks(b, cfg, "metal") : 0;
            }
            System.out.println("  metal " + Arrays.toString(gpu));
            failures += concurrentCheck(w, lib, cfg, quant == QType.Q4_0 && head == QuantFile.DEFAULT_HEAD);
            if (!recurrent) {
                failures += pipeCheck(w, lib, cfg, gpu, label);
            }
            if (hybrid && quant == QType.Q4_0 && head == QuantFile.DEFAULT_HEAD) {
                failures += hybridBatchCheck(w, lib);
            } else if (!mamba && quant == QType.Q4_0 && head == QuantFile.DEFAULT_HEAD) {
                failures += batchCheck(w, lib, cfg);
                failures += draftCheck(w, dir, lib, cfg);
            }
            if (cfg instanceof Decoder d && d.anyMoe()) {
                failures += moeSplitCheck(w, lib, cfg, cpu, cpuLogits);
            }
            if (!mamba && cfg instanceof Decoder && quant == QType.Q4_0 && head == QuantFile.DEFAULT_HEAD) {
                failures += q8Check(w, lib, cfg, cpu, KvDtype.Q8, 0);
                if (Q4) {
                    failures += q8Check(w, lib, cfg, cpu, KvDtype.Q4, 0);
                    failures += q8Check(w, lib, cfg, cpu, KvDtype.Q4, 4);
                }
                // f32 and q8 caches for every family; the f16 cache (its own mix of staging and cache rows) on the
                // first dense and the first MoE family, to keep the emulator run's time in bounds
                failures += replayCheck(w, lib, cfg, KvDtype.F32);
                failures += replayCheck(w, lib, cfg, KvDtype.Q8);
                failures += replayBatchCheck(w, lib, cfg, KvDtype.F32);
                // With the matrix kernels forced on from 2 tokens (the second pass) a batched pass of 3 rows and a lone step take
                // different matmul kernels, and the int8 cache can flip a rounding step between them (up to 1e-2 on a tiny
                // random Phi3Mha, whichever single-token kernel runs); the first pass, where they take the same ones, checks it.
                if (Integer.getInteger("jllm.metal.mmaMinTokens", 32) >= 8) {
                    failures += replayBatchCheck(w, lib, cfg, KvDtype.Q8);
                }
                // q4: without a window, and with one short enough that decode steps cross from the ring to the q4 rows
                for (int recent : Q4 ? new int[]{0, 6} : new int[0]) {
                    KvDtype.setQ4Recent(recent);
                    try {
                        failures += replayCheck(w, lib, cfg, KvDtype.Q4);
                    } finally {
                        KvDtype.setQ4Recent(KvDtype.Q4_RECENT_DEFAULT);
                    }
                }
                if (Q4) {
                    failures += replayBatchCheck(w, lib, cfg, KvDtype.Q4);
                }
                if (replayF16.add(((Decoder) cfg).anyMoe() ? "moe" : "dense")) {
                    failures += replayCheck(w, lib, cfg, KvDtype.F16);
                }
            }
            // q8 with Hadamard-rotated keys (backend.kv-q8-rotate), where the attention is standard with whole groups of 32:
            // CPU against Metal, decode replay (alone and batched), saved / copied caches; the batched decode and the draft
            // speculation of the rotated cache are cases of batchCheck / draftCheck above
            if (!mamba && !hybrid && cfg instanceof Decoder rd && rd.attention() instanceof Attention.Gqa rg
                    && KvQ4.supports(rg.headDim()) && quant == QType.Q4_0 && head == QuantFile.DEFAULT_HEAD) {
                System.out.println("  --- q8 with rotated keys ---");
                KvDtype.setQ8Rotate(true);
                try {
                    failures += q8Check(w, lib, cfg, cpu, KvDtype.Q8, 0);
                    failures += replayCheck(w, lib, cfg, KvDtype.Q8);
                    if (Q4) { // (the forced-MMA pass leaves it out, like q4's: a rotated int8 code flips between a batched pass and a lone step)
                        failures += replayBatchCheck(w, lib, cfg, KvDtype.Q8);
                    }
                    failures += kvStateChecks(w, lib, cfg, KvDtype.Q8);
                    failures += replayStateCheck(w, lib, cfg, KvDtype.Q8);
                } finally {
                    KvDtype.setQ8Rotate(false);
                }
            }
            if (!mamba && cfg instanceof Decoder dec && quant == QType.Q4_0 && head == QuantFile.DEFAULT_HEAD) { // standard and latent (MLA) caches alike
                boolean latent = dec.attention() instanceof Attention.Mla;
                for (KvDtype dt : KvDtype.values()) {
                    if (!(latent && (dt == KvDtype.Q8 || dt == KvDtype.Q4))) { // a q8 latent cache is not saved (q4 is served as q8 there)
                        failures += kvStateChecks(w, lib, cfg, dt);
                        failures += replayStateCheck(w, lib, cfg, dt);
                    }
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
     * Experts on the CPU (backend.metal.cpu-moe): with the routed experts of one MoE layer, of all of them, and split
     * inside layers (every other expert on the CPU in every MoE layer; all but one in the first), the generation and
     * the final logits match the CPU backend within the usual CPU / Metal tolerance (decode takes the per-slot
     * kernels and the single-token CPU path); a 20-token chunk takes the expert-grouped kernels and the CPU's batched
     * path; and the usage counts the split backend saves hold topK picks per processed token in every MoE layer.
     */
    private static int moeSplitCheck(ModelWeights w, String lib, Architecture cfg, int[] cpu, float[] cpuLogits) throws Exception {
        Decoder d = (Decoder) cfg;
        int layers = d.core().layers();
        boolean[][] alternate = new boolean[layers][];
        boolean[][] mostlyCold = new boolean[layers][];
        int topK = 0;
        boolean first = true;
        for (int l = 0; l < layers; l++) {
            if (d.layers().get(l).mlp() instanceof Mlp.Moe m) {
                topK = m.topK();
                alternate[l] = new boolean[m.experts()];
                for (int e = 0; e < m.experts(); e += 2) {
                    alternate[l][e] = true;
                }
                if (first) {
                    mostlyCold[l] = new boolean[m.experts()];
                    java.util.Arrays.fill(mostlyCold[l], 1, m.experts(), true);
                    first = false;
                }
            }
        }
        java.nio.file.Path stats = java.nio.file.Files.createTempDirectory("jllm-expert-stats");
        System.setProperty("jllm.metal.expertStatsDir", stats.toString());
        int bad = 0;
        Object[][] cases = {
                {"1 layer", new MoeSplit(1, 4, false)},
                {"all layers", new MoeSplit(Integer.MAX_VALUE, 4, false)},
                {"alternate experts", MoeSplit.experts(alternate, 4, false)},
                {"all but one expert of a layer", MoeSplit.experts(mostlyCold, 4, false)},
        };
        int[] chunk = new int[20];
        for (int i = 0; i < chunk.length; i++) {
            chunk[i] = PROMPT[i % PROMPT.length] + 3 * i;
        }
        float[] cpuChunk = new float[cfg.core().vocabSize()];
        try (Backend b = new CpuBackend(w, 64, 32, 4, KvDtype.F32, false)) {
            b.forward(chunk, 0, chunk.length, 0, cpuChunk);
        }
        for (Object[] c : cases) {
            float[] logits = new float[cfg.core().vocabSize()];
            float[] chunkLogits = new float[cfg.core().vocabSize()];
            int[] got;
            try (Backend b = new MetalBackend(w, 64, 32, lib, KvDtype.F32, (MoeSplit) c[1])) {
                got = generate(b, logits);
                b.forward(chunk, 0, chunk.length, 0, chunkLogits);
            }
            float diff = maxDiff(cpuLogits, logits);
            float chunkDiff = maxDiff(cpuChunk, chunkLogits);
            boolean ok = Arrays.equals(cpu, got) && diff <= 1e-3f * maxAbs(cpuLogits) + 1e-4f
                    && chunkDiff <= 1e-3f * maxAbs(cpuChunk) + 1e-4f;
            System.out.printf("  experts on cpu (%s): %s max|cpu-split| = %.2e, 20-token chunk %.2e -> %s%n", c[0],
                    Arrays.toString(got), diff, chunkDiff, ok ? "ok" : "SPLIT MISMATCH");
            bad += ok ? 0 : 1;
        }
        // the four backends above processed the same tokens: per MoE layer the saved picks are a multiple of topK, the
        // same in every layer
        long[] sums = new long[layers];
        try (var files = java.nio.file.Files.list(stats)) {
            for (java.nio.file.Path f : files.toList()) {
                for (String line : java.nio.file.Files.readAllLines(f)) {
                    if (!line.startsWith("#") && line.contains(":")) {
                        int l = Integer.parseInt(line.substring(0, line.indexOf(':')));
                        for (String v : line.substring(line.indexOf(':') + 1).split(",")) {
                            sums[l] += Long.parseLong(v.trim());
                        }
                    }
                }
                java.nio.file.Files.delete(f);
            }
        }
        java.nio.file.Files.delete(stats);
        System.clearProperty("jllm.metal.expertStatsDir");
        // two backends on the same weights at once (a server's slots) share one placement and its packed experts; each
        // gives the same result, and closing one leaves the other working
        try (Backend a = new MetalBackend(w, 64, 32, lib, KvDtype.F32, MoeSplit.experts(alternate, 4, false))) {
            float[] la = new float[cfg.core().vocabSize()];
            float[] lb = new float[cfg.core().vocabSize()];
            int[] ga;
            try (Backend b = new MetalBackend(w, 64, 32, lib, KvDtype.F32, MoeSplit.experts(alternate, 4, false))) {
                ga = generate(a, la);
                int[] gb = generate(b, lb);
                boolean same = Arrays.equals(ga, gb) && Arrays.equals(la, lb);
                System.out.println("  two slots sharing a placement: " + (same ? "identical" : "DIFFERENT"));
                bad += same ? 0 : 1;
            }
            int[] again = generate(a, la);
            boolean ok = Arrays.equals(ga, again);
            System.out.println("  first slot after the second closed: " + (ok ? "ok" : "BROKEN"));
            bad += ok ? 0 : 1;
        }
        bad += rebalanceCheck(w, lib, cfg, cpu, cpuLogits);
        long expect = -1;
        boolean statsOk = true;
        for (int l = 0; l < layers; l++) {
            if (alternate[l] != null) {
                expect = expect < 0 ? sums[l] : expect;
                statsOk &= sums[l] == expect && sums[l] > 0 && sums[l] % topK == 0;
            }
        }
        System.out.println("  expert usage saved: " + expect + " picks per MoE layer -> " + (statsOk ? "ok" : "STATS WRONG"));
        return bad + (statsOk ? 0 : 1);
    }

    /**
     * Re-placement while running: under backend.metal.cpu-moe=auto with a GPU budget about half the experts short and
     * no statistics, the first plan moves whole layers; after some tokens have been counted, rebalanceExperts re-plans
     * from them (arch-smoke.sh sets the round-trip estimate to 0, so splitting layers is preferred) and moves experts,
     * and the generation is still the CPU backend's.
     */
    private static int rebalanceCheck(ModelWeights w, String lib, Architecture cfg, int[] cpu, float[] cpuLogits)
            throws Exception {
        Decoder d = (Decoder) cfg;
        long experts = 0;
        for (int l = 0; l < d.core().layers(); l++) {
            if (d.layers().get(l).mlp() instanceof Mlp.Moe) {
                experts += w.decoderLayer(l).moe().expGateUp().data.byteSize() + w.decoderLayer(l).moe().expDown().data.byteSize();
            }
        }
        long full;
        try (MetalBackend b = new MetalBackend(w, 64, 32, lib, KvDtype.F32, MoeSplit.OFF)) {
            full = b.gpuResidentBytes();
        }
        java.nio.file.Path stats = java.nio.file.Files.createTempDirectory("jllm-expert-stats");
        System.setProperty("jllm.metal.expertStatsDir", stats.toString());
        System.setProperty("jllm.metal.workingSetGb", Double.toString((full - experts / 2) / 1e9));
        int bad = 0;
        try (MetalBackend b = new MetalBackend(w, 64, 32, lib, KvDtype.F32, new MoeSplit(MoeSplit.AUTO, 4, false))) {
            float[] logits = new float[cfg.core().vocabSize()];
            generate(b, logits);
            generate(b, logits);
            int moved = b.rebalanceExperts();
            int[] got = generate(b, logits);
            float diff = maxDiff(cpuLogits, logits);
            boolean ok = moved > 0 && Arrays.equals(cpu, got) && diff <= 1e-3f * maxAbs(cpuLogits) + 1e-4f;
            System.out.printf("  experts re-placed while running: %d layers moved, %s max|cpu-split| = %.2e -> %s%n", moved,
                    Arrays.toString(got), diff, ok ? "ok" : "REBALANCE FAILED");
            bad += ok ? 0 : 1;
            // Decode replay x re-placement: while experts are on the CPU a decode step is not replayed (it waits for the
            // CPU halfway). With room for every expert the next re-placement moves the rest to the GPU, which makes
            // the backend replayable (and drops recordings that bound the old expert buffers): the steps after it must be
            // recorded, replayed, and still the CPU backend's.
            System.setProperty("jllm.metal.workingSetGb", Double.toString((full + 100_000_000L) / 1e9));
            boolean replayedBefore = b.replaying();
            int moved2 = b.rebalanceExperts();
            long r0 = b.replayedSteps();
            float[] l2 = new float[cfg.core().vocabSize()];
            int[] first = generate(b, l2);  // records the decode shapes
            int[] second = generate(b, l2); // replays them
            long replayed = b.replayedSteps() - r0;
            float diff2 = maxDiff(cpuLogits, l2);
            boolean ok2 = !replayedBefore && moved2 > 0 && b.replaying() && replayed > 0 && Arrays.equals(cpu, first)
                    && Arrays.equals(cpu, second) && diff2 <= 1e-3f * maxAbs(cpuLogits) + 1e-4f;
            System.out.printf("  experts re-placed onto the GPU, then replayed: %d layers moved, %d steps replayed, %s max|cpu-split| = %.2e -> %s%n",
                    moved2, replayed, Arrays.toString(second), diff2, ok2 ? "ok" : "REPLAY x REBALANCE FAILED");
            bad += ok2 ? 0 : 1;
        } finally {
            System.clearProperty("jllm.metal.workingSetGb");
            System.clearProperty("jllm.metal.expertStatsDir");
            try (var files = java.nio.file.Files.list(stats)) {
                for (java.nio.file.Path f : files.toList()) {
                    java.nio.file.Files.delete(f);
                }
            }
            java.nio.file.Files.delete(stats);
        }
        return bad;
    }

    private static float maxDiff(float[] a, float[] b) {
        float d = 0;
        for (int i = 0; i < a.length; i++) {
            d = Math.max(d, Math.abs(a[i] - b[i]));
        }
        return d;
    }

    private static float maxAbs(float[] a) {
        float m = 0;
        for (float v : a) {
            m = Math.max(m, Math.abs(v));
        }
        return m;
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
        KvDtype.setQ4Recent(3); // q4: the saved window holds the last 3 of the prompt's rows, the others are 4-bit only
        try {
            return kvStateChecks0(w, lib, cfg, dt);
        } finally {
            KvDtype.setQ4Recent(KvDtype.Q4_RECENT_DEFAULT);
        }
    }

    private static int kvStateChecks0(ModelWeights w, String lib, Architecture cfg, KvDtype dt) throws Exception {
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
            var resaved = new java.io.ByteArrayOutputStream();
            b.saveState(n, java.nio.channels.Channels.newChannel(resaved));
            boolean sameBytes = Arrays.equals(resaved.toByteArray(), cpuBytes.toByteArray()); // rows (and window) taken in unchanged
            b.forward(next, 0, 1, n, got);
            float scale = 0;
            float diff = 0;
            for (int i = 0; i < vocab; i++) {
                scale = Math.max(scale, Math.abs(want[i]));
                diff = Math.max(diff, Math.abs(want[i] - got[i]));
            }
            // The file's bytes must come back unchanged (sameBytes). Metal's own continuation (want) is compared with the
            // CPU file's: both prefilled the prompt in their own float arithmetic, so a q4 value on a rounding edge of the
            // 4-bit grid lands one step (1/7.5 of its group's largest value) apart in the two caches, and a tiny random
            // model turns that into a logit difference a q8 edge flip (1/127) does not reach: 10x the tolerance for q4.
            // q8 with rotated keys: the same kind of edge flip as q4's, 10x the tolerance (see q8Check)
            boolean crossOk = sameBytes && diff <= (dt == KvDtype.Q4 || dt == KvDtype.Q8 && KvDtype.q8Rotate() ? 1e-2f : 1e-3f) * scale + 1e-4f;
            System.out.printf("  kv %s: saved+loaded %s, copied %s, cpu file in metal %s (max diff %.1e, bytes %s)%n", a.stateFormat(),
                    loaded ? "identical" : "DIFFERENT", copied ? "identical" : "DIFFERENT",
                    sameFormat && crossOk ? "ok" : "MISMATCH", diff, sameBytes ? "unchanged" : "CHANGED");
            bad += (loaded ? 0 : 1) + (copied ? 0 : 1) + (sameFormat && crossOk ? 0 : 1);
            if (dt == KvDtype.Q4) { // the first n - 2 positions of a longer state: the ring keeps only the window below them
                int m = n - 2;
                b.loadState(m, n, java.nio.channels.Channels.newChannel(new java.io.ByteArrayInputStream(cpuBytes.toByteArray())));
                cpu.loadState(m, n, java.nio.channels.Channels.newChannel(new java.io.ByteArrayInputStream(cpuBytes.toByteArray())));
                var bOut = new java.io.ByteArrayOutputStream();
                var cOut = new java.io.ByteArrayOutputStream();
                b.saveState(m, java.nio.channels.Channels.newChannel(bOut));
                cpu.saveState(m, java.nio.channels.Channels.newChannel(cOut));
                boolean sameWindow = Arrays.equals(bOut.toByteArray(), cOut.toByteArray()); // rows and the part of the window kept
                b.forward(next, 0, 1, m, got);
                float[] cpuLogits = new float[vocab];
                cpu.forward(next, 0, 1, m, cpuLogits);
                float sc = 0;
                float df = 0;
                for (int i = 0; i < vocab; i++) {
                    sc = Math.max(sc, Math.abs(cpuLogits[i]));
                    df = Math.max(df, Math.abs(cpuLogits[i] - got[i]));
                }
                boolean prefixOk = sameWindow && df <= 1e-2f * sc + 1e-4f; // same file in both: only the new row's rounding differs
                System.out.printf("  kv %s: first %d of %d saved positions, saved again %s on both, cpu vs metal max diff %.1e -> %s%n",
                        a.stateFormat(), m, n, sameWindow ? "the same bytes" : "DIFFERENT bytes", df, prefixOk ? "ok" : "MISMATCH");
                bad += prefixOk ? 0 : 1;
            }
        }
        return bad;
    }

    /**
     * Decode replay x saved caches: a slot that has been decoding with replay (so it holds recordings) gets another
     * conversation's cache, loaded from a saved file and then copied from a live backend; its next decode steps, flash
     * decoding forced on so they cross shapes, must be bit-identical to those of a backend with replay off that got the
     * same cache, and must really be replayed (the recordings bind the cache buffers and a load writes their contents,
     * so they must stay valid). The two ways of restoring the same cache must also continue identically.
     */
    private static int replayStateCheck(ModelWeights w, String lib, Architecture cfg, KvDtype dt) throws Exception {
        int vocab = cfg.core().vocabSize();
        int[] other = {5, 900, 17, 3, 44, 61, 2, 8, 19, 400, 3, 77, 100};
        int n = other.length;
        int steps = 14;
        int[] toks = new int[steps];
        for (int j = 0; j < steps; j++) {
            toks[j] = (PROMPT[j % PROMPT.length] * 7 + j) % vocab;
        }
        try (MetalBackend on = new MetalBackend(w, 64, 16, lib, dt); MetalBackend ref = new MetalBackend(w, 64, 16, lib, dt);
             MetalBackend src = new MetalBackend(w, 64, 16, lib, dt)) {
            on.replay(true);
            ref.replay(false);
            src.replay(false);
            on.flashDecodeThresholds(PROMPT.length + 6, 4);
            ref.flashDecodeThresholds(PROMPT.length + 6, 4);
            // a first conversation on both: `on` records its steps
            continueSteps(on, PROMPT, steps, toks, vocab);
            continueSteps(ref, PROMPT, steps, toks, vocab);
            long recorded = on.recordedSteps();
            src.forward(other, 0, n, 0, null);
            var bytes = new java.io.ByteArrayOutputStream();
            src.saveState(n, java.nio.channels.Channels.newChannel(bytes));
            on.loadState(n, java.nio.channels.Channels.newChannel(new java.io.ByteArrayInputStream(bytes.toByteArray())));
            ref.loadState(n, java.nio.channels.Channels.newChannel(new java.io.ByteArrayInputStream(bytes.toByteArray())));
            long before = on.replayedSteps();
            float[][] loadedOn = tail(on, n, steps, toks, vocab);
            float[][] loadedRef = tail(ref, n, steps, toks, vocab);
            long replayedAfterLoad = on.replayedSteps() - before;
            on.copyStateFrom(src, n);
            ref.copyStateFrom(src, n);
            before = on.replayedSteps();
            float[][] copiedOn = tail(on, n, steps, toks, vocab);
            float[][] copiedRef = tail(ref, n, steps, toks, vocab);
            long replayedAfterCopy = on.replayedSteps() - before;
            boolean loaded = Arrays.deepEquals(loadedOn, loadedRef);
            boolean copied = Arrays.deepEquals(copiedOn, copiedRef);
            boolean same = Arrays.deepEquals(loadedOn, copiedOn);
            boolean ok = loaded && copied && same && recorded > 0 && replayedAfterLoad > 0 && replayedAfterCopy > 0;
            System.out.printf("  replay x saved cache %s: after a load %s (%d steps replayed), after a copy %s (%d replayed), load == copy %s -> %s%n",
                    dt, loaded ? "bit-identical" : "DIFFERENT", replayedAfterLoad, copied ? "bit-identical" : "DIFFERENT",
                    replayedAfterCopy, same ? "yes" : "NO", ok ? "ok" : "REPLAY x STATE MISMATCH");
            return ok ? 0 : 1;
        }
    }

    /** Prefill {@code prompt}, then {@code steps} decode steps teacher-forced with {@code toks}. */
    private static void continueSteps(MetalBackend b, int[] prompt, int steps, int[] toks, int vocab) {
        b.forward(prompt, 0, prompt.length, 0, new float[vocab]);
        tail(b, prompt.length, steps, toks, vocab);
    }

    /** {@code steps} decode steps from position {@code n} (the cache holds n positions), teacher-forced; their logits. */
    private static float[][] tail(MetalBackend b, int n, int steps, int[] toks, int vocab) {
        float[][] out = new float[steps][vocab];
        for (int j = 0; j < steps; j++) {
            b.forward(new int[]{toks[j]}, 0, 1, n + j, out[j]);
        }
        return out;
    }

    /**
     * The q8 KV cache (int8 with a scale per row of a kv head; DeepSeek: per latent row), or the q4 one (Hadamard-rotated
     * K, 4-bit groups of 32; DeepSeek: q8) with a recent window of {@code recent} positions, on both backends, teacher-forced
     * through the prompt followed by the f32 CPU run's tokens: once as one 20-token chunk (the prefill kernels, flash
     * prefill where the tuner enabled it) and once as the prompt then token by token (decode). CPU q8 and Metal q8 must
     * agree within the usual CPU / Metal tolerance at every step. The distance of the q8 logits from the f32 cache's
     * (CPU) is printed next to the f16 cache's: how much a random model amplifies a perturbation varies a lot (the
     * tiny DeepSeeks turn the f16 cache's 0.2-0.4% of the largest logit into 13-25% for q8, the dense families stay
     * near 1%), so q8 is bounded relative to f16, by 200 times its distance (observed: 20-45 times for the dense
     * families, up to 115 for the DeepSeeks) plus 10^-3 of the largest logit: a broken dequantization is caught, the
     * amplification is not. q4's step is about 16 times q8's (1/8 of a group's largest value against 1/254 of a row's,
     * for typical values), so its bound is 16 times as wide.
     */
    private static int q8Check(ModelWeights w, String lib, Architecture cfg, int[] tokens, KvDtype dt, int recent)
            throws Exception {
        int[] seq = Arrays.copyOf(PROMPT, PROMPT.length + tokens.length);
        System.arraycopy(tokens, 0, seq, PROMPT.length, tokens.length);
        float[][] f32;
        float[][] f16;
        float[][] cpu;
        float[][] gpu;
        try (Backend b = new CpuBackend(w, 64, 32, 4, KvDtype.F32, false)) {
            f32 = forced(b, seq, cfg.core().vocabSize());
        }
        try (Backend b = new CpuBackend(w, 64, 32, 4, KvDtype.F16, false)) {
            f16 = forced(b, seq, cfg.core().vocabSize());
        }
        KvDtype.setQ4Recent(recent);
        try {
            try (Backend b = new CpuBackend(w, 64, 32, 4, dt, false)) {
                cpu = forced(b, seq, cfg.core().vocabSize());
            }
            try (Backend b = new MetalBackend(w, 64, 32, lib, dt)) {
                gpu = forced(b, seq, cfg.core().vocabSize());
            }
        } finally {
            KvDtype.setQ4Recent(KvDtype.Q4_RECENT_DEFAULT);
        }
        float scale = 0;
        float diff = 0;
        float drift = 0;
        float drift16 = 0;
        for (int r = 0; r < cpu.length; r++) {
            for (int i = 0; i < cpu[r].length; i++) {
                scale = Math.max(scale, Math.abs(f32[r][i]));
                diff = Math.max(diff, Math.abs(cpu[r][i] - gpu[r][i]));
                drift = Math.max(drift, Math.abs(cpu[r][i] - f32[r][i]));
                drift16 = Math.max(drift16, Math.abs(f16[r][i] - f32[r][i]));
            }
        }
        // q8 with rotated keys: the CPU and the Metal prefill compute k in float with rounding differences of ~1e-7, and
        // an int8 code of the rotated row that sits on a rounding edge (one code in ~6000) lands one step apart: a flip
        // in layer 0 of a tiny random model moves later layers' k and v codes too (counted on the Qwen2 checkpoint: one
        // flip in layer 0, 71 of 6144 codes in all, with plain q8 none) and the logits by up to 1.5e-3 of the largest
        // logit. The kernels themselves agree bit for bit on equal inputs (kernel_tests.py), so this is q4's case: 10x.
        boolean agree = diff <= (dt == KvDtype.Q8 && KvDtype.q8Rotate() ? 1e-2f : 1e-3f) * scale + 1e-4f;
        boolean close = drift <= (dt == KvDtype.Q4 ? 3200 : 200) * drift16 + 1e-3f * scale;
        String what = dt + (dt == KvDtype.Q4 ? " recent=" + recent : dt == KvDtype.Q8 && KvDtype.q8Rotate() ? " rotated keys" : "");
        System.out.printf("  kv %s: max|cpu-metal| = %.2e%s, max|%s-f32| = %.2e%s (f16: %.2e, ratio %.0f, max|logit| = %.2f)%n",
                what, diff, agree ? "" : " MISMATCH", dt, drift, close ? "" : " TOO FAR", drift16, drift / drift16, scale);
        return (agree ? 0 : 1) + (close ? 0 : 1);
    }

    private static final int REPLAY_STEPS = 40;
    private static final java.util.Set<String> replayF16 = new java.util.HashSet<>();

    // ------------------------------------------------------------------ the concurrent compute encoder (docs/08 8.26)

    private static final int CONC_CHUNK = 40; // >= 32 tokens: the simdgroup-matrix prefill kernels too

    /** Concurrent encoder on or off for a backend of any Metal family. */
    private static void concurrent(Backend b, boolean on) {
        if (b instanceof MetalBackend m) {
            m.concurrent(on);
        } else if (b instanceof MambaMetalBackend m) {
            m.concurrent(on);
        } else if (b instanceof Qwen3NextMetalBackend m) {
            m.concurrent(on);
        }
    }

    private static long barriers(Backend b) {
        return b instanceof MetalBackend m ? m.barriers() : b instanceof MambaMetalBackend m ? m.barriers()
                : b instanceof Qwen3NextMetalBackend m ? m.barriers() : 0;
    }

    /**
     * Every kind of pass in a row, recording each logits vector: a 40-token prefill chunk (the simdgroup-matrix kernels), 10
     * decode steps (replay, the pipelined path is {@link #concurrentPipe}), a 7-token chunk (the row kernels), and a
     * 4-token {@code forwardAll} (draft verification's multi-token pass).
     */
    private static List<float[]> concurrentScenario(Backend b, int vocab) {
        List<float[]> out = new ArrayList<>();
        int[] tok = new int[CONC_CHUNK];
        for (int i = 0; i < tok.length; i++) {
            tok[i] = (i * 37 + 11) % vocab;
        }
        float[] l = new float[vocab];
        b.forward(tok, 0, tok.length, 0, l);
        out.add(l.clone());
        int pos = tok.length;
        for (int i = 0; i < 10; i++) {
            b.forward(new int[]{argmax(l)}, 0, 1, pos++, l);
            out.add(l.clone());
        }
        int[] chunk = new int[7];
        for (int i = 0; i < chunk.length; i++) {
            chunk[i] = (i * 53 + 5) % vocab;
        }
        b.forward(chunk, 0, chunk.length, pos, l);
        out.add(l.clone());
        pos += chunk.length;
        float[] all = new float[4 * vocab];
        b.forwardAll(new int[]{3, 77, 150, 21}, 0, 4, pos, all, vocab);
        out.add(all);
        return out;
    }

    private interface BackendMaker {
        Backend make() throws Exception;
    }

    /**
     * The concurrent compute encoder against the serial one on a backend of the same weights: the logits of every pass of
     * {@link #concurrentScenario} must be bit-identical (the encoder only lets independent dispatches overlap, so any
     * difference is a missing barrier). Races are intermittent, so the concurrent run is repeated {@code runs} times.
     * With {@code full} (Q4_0): the f16 and q8 caches, experts split between CPU and GPU, pipelined greedy decoding and
     * batched decoding of three slots too.
     */
    private static int concurrentCheck(ModelWeights w, String lib, Architecture cfg, boolean full) throws Exception {
        if (!Metal.available(lib)) {
            return 0;
        }
        int vocab = cfg.core().vocabSize();
        boolean mamba = cfg instanceof MambaArch;
        boolean hybrid = cfg instanceof Qwen3NextArch;
        int bad = 0;
        List<Object[]> cases = new ArrayList<>(); // label, maker
        for (KvDtype dt : mamba ? new KvDtype[]{KvDtype.F32} : full ? KvDtype.values() : new KvDtype[]{KvDtype.F32}) {
            if (hybrid && dt == KvDtype.Q8) {
                continue; // no q8 cache for the hybrid
            }
            BackendMaker mk = () -> mamba ? new MambaMetalBackend(w, 64, CONC_CHUNK, lib)
                    : hybrid ? new Qwen3NextMetalBackend(w, 64, CONC_CHUNK, lib, dt) : new MetalBackend(w, 64, CONC_CHUNK, lib, dt);
            cases.add(new Object[]{"kv " + dt, mk});
        }
        java.nio.file.Path stats = null;
        if (full && cfg instanceof Decoder d && d.anyMoe() && !hybrid) {
            // experts on the CPU: the GPU's command buffer is submitted and waited for halfway through the layer, whole
            // layers, and a layer split between the two (every other expert on the CPU)
            boolean[][] alternate = new boolean[d.core().layers()][];
            for (int l = 0; l < alternate.length; l++) {
                if (d.layers().get(l).mlp() instanceof Mlp.Moe m) {
                    alternate[l] = new boolean[m.experts()];
                    for (int e = 0; e < m.experts(); e += 2) {
                        alternate[l][e] = true;
                    }
                }
            }
            stats = Files.createTempDirectory("jllm-expert-stats");
            System.setProperty("jllm.metal.expertStatsDir", stats.toString());
            Object[][] splits = {{"experts of 1 layer on the cpu", new MoeSplit(1, 4, false)},
                    {"experts of all layers on the cpu", new MoeSplit(Integer.MAX_VALUE, 4, false)},
                    {"every other expert on the cpu", MoeSplit.experts(alternate, 4, false)}};
            for (Object[] s : splits) {
                cases.add(new Object[]{"kv F32, " + s[0],
                        (BackendMaker) () -> new MetalBackend(w, 64, CONC_CHUNK, lib, KvDtype.F32, (MoeSplit) s[1])});
            }
        }
        try {
            bad += concurrentCases(cases, vocab);
        } finally {
            if (stats != null) {
                try (var files = Files.list(stats)) {
                    for (Path f : files.toList()) {
                        Files.delete(f);
                    }
                }
                Files.delete(stats);
                System.clearProperty("jllm.metal.expertStatsDir");
            }
        }
        if (full && !mamba && !hybrid) {
            bad += concurrentPipe(w, lib, cfg);
            bad += concurrentBatch(w, lib, cfg);
        }
        return bad;
    }

    private static int concurrentCases(List<Object[]> cases, int vocab) throws Exception {
        int bad = 0;
        for (Object[] c : cases) {
            BackendMaker mk = (BackendMaker) c[1];
            List<float[]> want;
            try (Backend b = mk.make()) {
                concurrent(b, false);
                want = concurrentScenario(b, vocab);
            }
            int runs = 3;
            long barriers = 0;
            boolean same = true;
            for (int r = 0; r < runs; r++) {
                try (Backend b = mk.make()) {
                    concurrent(b, true);
                    List<float[]> got = concurrentScenario(b, vocab);
                    barriers = barriers(b);
                    for (int i = 0; i < want.size(); i++) {
                        if (!Arrays.equals(want.get(i), got.get(i))) {
                            System.out.println("  concurrent encoder, " + c[0] + ": pass " + i + " differs (run " + r + ")");
                            same = false;
                            break;
                        }
                    }
                }
            }
            System.out.println("  concurrent encoder, " + c[0] + ": " + want.size() + " passes x " + runs
                    + " runs, logits " + (same ? "bit-identical to serial" : "DIFFER") + " (" + barriers + " barriers)"
                    + (same ? "" : "  CONCURRENT MISMATCH"));
            bad += same ? 0 : 1;
        }
        return bad;
    }

    /** Pipelined greedy decoding (token ring, replay) with the concurrent encoder gives the serial encoder's tokens. */
    private static int concurrentPipe(ModelWeights w, String lib, Architecture cfg) {
        int vocab = cfg.core().vocabSize();
        int[][] ids = new int[2][];
        for (int mode = 0; mode < 2; mode++) {
            try (MetalBackend b = new MetalBackend(w, 64, 16, lib)) {
                b.concurrent(mode == 1);
                Backend.GreedyPipe pipe = b.greedyPipe(vocab);
                if (pipe == null) {
                    return 0;
                }
                float[] l = new float[vocab];
                b.forward(PROMPT, 0, PROMPT.length, 0, l);
                int n = 20;
                ids[mode] = new int[n];
                pipe.submit(argmax(l), PROMPT.length);
                for (int i = 0; i < n; i++) {
                    if (i + 1 < n) {
                        pipe.submit(-1, PROMPT.length + i + 1);
                    }
                    ids[mode][i] = pipe.next();
                }
                pipe.drain();
            }
        }
        boolean ok = Arrays.equals(ids[0], ids[1]);
        System.out.println("  concurrent encoder, pipelined greedy decoding: tokens " + (ok ? "same" : "DIFFERENT")
                + (ok ? "" : "  CONCURRENT MISMATCH"));
        return ok ? 0 : 1;
    }

    /**
     * Batched decoding of three slots (per-row dispatches in independent groups) with the concurrent encoder: every row's
     * logits in every shared pass are bit-identical to the serial encoder's, for the f32, f16 and q8 caches.
     */
    private static int concurrentBatch(ModelWeights w, String lib, Architecture cfg) throws Exception {
        int vocab = cfg.core().vocabSize();
        int[][] prompts = {PROMPT, Arrays.copyOf(PROMPT, 7), {5, 900, 17, 3, 44, 61, 2, 8, 19, 400}};
        int bad = 0;
        for (KvDtype dt : KvDtype.values()) {
            List<float[]> want = null;
            boolean same = true;
            long barriers = 0;
            for (int mode = 0; mode < 4; mode++) { // serial once, then the concurrent encoder three times
                List<float[]> got = new ArrayList<>();
                MetalBackend[] b = new MetalBackend[3];
                try {
                    for (int i = 0; i < 3; i++) {
                        b[i] = new MetalBackend(w, 64, 16, lib, dt);
                        b[i].concurrent(mode > 0);
                    }
                    int[] tok = new int[3];
                    float[][] l = new float[3][vocab];
                    for (int i = 0; i < 3; i++) {
                        b[i].forward(prompts[i], 0, prompts[i].length, 0, l[i]);
                        tok[i] = argmax(l[i]);
                    }
                    for (int s = 1; s < NEW_TOKENS; s++) {
                        int lead = s % 3;
                        Backend[] members = new Backend[3];
                        int[] tokens = new int[3];
                        int[] positions = new int[3];
                        float[][] out = new float[3][];
                        for (int r = 0; r < 3; r++) {
                            int i = (lead + r) % 3;
                            members[r] = b[i];
                            tokens[r] = tok[i];
                            positions[r] = prompts[i].length + s - 1;
                            out[r] = l[i];
                        }
                        if (!b[lead].forwardBatch(members, tokens, positions, out)) {
                            System.out.println("  concurrent encoder, batched decode kv " + dt + ": no batched pass  BATCH FAILED");
                            return bad + 1;
                        }
                        for (int i = 0; i < 3; i++) {
                            tok[i] = argmax(l[i]);
                            got.add(l[i].clone());
                        }
                    }
                    barriers = b[0].barriers();
                } finally {
                    for (MetalBackend x : b) {
                        if (x != null) {
                            x.close();
                        }
                    }
                }
                if (want == null) {
                    want = got;
                } else {
                    for (int i = 0; i < want.size() && same; i++) {
                        same &= Arrays.equals(want.get(i), got.get(i));
                    }
                }
            }
            System.out.println("  concurrent encoder, batched decode, 3 slots, kv " + dt + ": logits "
                    + (same ? "bit-identical to serial" : "DIFFER") + " (" + barriers + " barriers on the first slot)"
                    + (same ? "" : "  CONCURRENT MISMATCH"));
            bad += same ? 0 : 1;
        }
        return bad;
    }

    /**
     * Decode replay (a single-token step recorded once per shape, then re-run with its position patched in) against
     * encoding every step dispatch by dispatch, on two backends fed the same tokens: the logits of all
     * {@code REPLAY_STEPS} decode steps after the prompt must be bit-identical, and so must the token ids of as many
     * pipelined greedy steps (Backend.GreedyPipe, where the token ring slots are patched too). Flash decoding is forced
     * on from 8 positions after the prompt with 4 keys per split, so the steps cross from the single-pass attention to
     * the split kernels and through a new split count every 4 positions (each a new recording) within the run. Also
     * prints the dispatches and the Java -> native calls per decode token with and without replay.
     */
    private static int replayCheck(ModelWeights w, String lib, Architecture cfg, KvDtype dt) {
        int vocab = cfg.core().vocabSize();
        int[] tokens = new int[REPLAY_STEPS];
        boolean same = true;
        long[] calls = new long[2];
        long[] disp = new long[2];
        long recorded;
        long replayed;
        try (MetalBackend on = new MetalBackend(w, 64, 16, lib, dt); MetalBackend off = new MetalBackend(w, 64, 16, lib, dt)) {
            on.replay(true);
            off.replay(false);
            if (!on.replaying()) {
                System.out.println("  replay " + dt + ": not available (native library without jm_run)");
                return 1;
            }
            for (MetalBackend b : new MetalBackend[]{on, off}) {
                b.flashDecodeThresholds(PROMPT.length + 8, 4);
            }
            float[] lOn = new float[vocab];
            float[] lOff = new float[vocab];
            on.forward(PROMPT, 0, PROMPT.length, 0, lOn);
            off.forward(PROMPT, 0, PROMPT.length, 0, lOff);
            same &= Arrays.equals(lOn, lOff);
            long[] c0 = {on.downcalls(), off.downcalls()};
            long[] d0 = {on.dispatches(), off.dispatches()};
            int firstBad = -1;
            for (int i = 0; i < REPLAY_STEPS; i++) {
                int best = argmax(lOn);
                tokens[i] = best;
                on.forward(new int[]{best}, 0, 1, PROMPT.length + i, lOn);
                off.forward(new int[]{best}, 0, 1, PROMPT.length + i, lOff);
                if (!Arrays.equals(lOn, lOff) && firstBad < 0) {
                    firstBad = i;
                }
            }
            same &= firstBad < 0;
            calls[0] = on.downcalls() - c0[0];
            calls[1] = off.downcalls() - c0[1];
            disp[0] = on.dispatches() - d0[0];
            disp[1] = off.dispatches() - d0[1];
            recorded = on.recordedSteps();
            replayed = on.replayedSteps();
            // pipelined greedy decoding with replay, from the same prompt (the GPU feeds its own argmax forward): step i's
            // token must be the argmax the decode loop above saw after step i (the same on both backends)
            int[] ids = null;
            Backend.GreedyPipe pipe = on.greedyPipe(vocab);
            if (pipe != null) {
                on.forward(PROMPT, 0, PROMPT.length, 0, lOn);
                ids = new int[REPLAY_STEPS];
                pipe.submit(argmax(lOn), PROMPT.length);
                for (int i = 0; i < REPLAY_STEPS; i++) {
                    if (i + 1 < REPLAY_STEPS) {
                        pipe.submit(-1, PROMPT.length + i + 1);
                    }
                    ids[i] = pipe.next();
                }
                pipe.drain();
            }
            boolean pipeSame = ids == null || Arrays.equals(ids, 0, REPLAY_STEPS - 1, tokens, 1, REPLAY_STEPS);
            String msg = String.format("  replay %s: %d decode steps %s (%d recorded, %d replayed), pipelined %s; per token "
                            + "%.1f dispatches, %.1f native calls with replay vs %.1f / %.1f without", dt, REPLAY_STEPS,
                    same ? "bit-identical" : "DIFFER from step " + firstBad, recorded, replayed,
                    ids == null ? "n/a" : pipeSame ? "the same tokens" : "DIFFERENT tokens", disp[0] / (double) REPLAY_STEPS,
                    calls[0] / (double) REPLAY_STEPS, disp[1] / (double) REPLAY_STEPS, calls[1] / (double) REPLAY_STEPS);
            boolean ok = same && pipeSame && replayed > 0 && calls[0] < calls[1];
            System.out.println(msg + (ok ? "" : "  REPLAY MISMATCH"));
            return ok ? 0 : 1;
        }
    }

    /**
     * Decode replay x batched decoding: three slots with replay on take single-token steps (recorded, then replayed with
     * the position patched in) and batched passes ({@link MetalBackend#forwardBatch}: other kernels, other buffers, the
     * rows written into each slot's cache) in turn, in a pattern that leaves a slot out of a batch now and then, with
     * flash decoding forced on so the single steps keep changing shape. Three reference slots on the same weights
     * decode the same tokens alone with replay off. Every step's logits must agree to rounding (a slot's recorded step
     * must not see a stale binding or a cache row the batched pass wrote differently), the single steps' logits are
     * counted as bit-identical, and the tokens are the same.
     */
    private static int replayBatchCheck(ModelWeights w, String lib, Architecture cfg, KvDtype dt) {
        int vocab = cfg.core().vocabSize();
        int[][] prompts = {PROMPT, Arrays.copyOf(PROMPT, 7), {5, 900, 17, 3, 44, 61, 2, 8, 19, 400}};
        int steps = 24;
        MetalBackend[] on = new MetalBackend[3];
        MetalBackend[] ref = new MetalBackend[3];
        try {
            for (int i = 0; i < 3; i++) {
                on[i] = new MetalBackend(w, 64, 16, lib, dt);
                ref[i] = new MetalBackend(w, 64, 16, lib, dt);
                on[i].replay(true);
                ref[i].replay(false);
                // the same thresholds in every slot, as in a server: a batched pass decides flash or not for all its rows
                // by the leader's
                on[i].flashDecodeThresholds(prompts[1].length + 6, 4);
                ref[i].flashDecodeThresholds(prompts[1].length + 6, 4);
            }
            if (!on[0].replaying()) {
                System.out.println("  replay x batch " + dt + ": replay not available");
                return 1;
            }
            float[][] lOn = new float[3][vocab];
            float[][] lRef = new float[3][vocab];
            int[] tok = new int[3];
            for (int i = 0; i < 3; i++) {
                on[i].forward(prompts[i], 0, prompts[i].length, 0, lOn[i]);
                ref[i].forward(prompts[i], 0, prompts[i].length, 0, lRef[i]);
                tok[i] = argmax(lRef[i]);
            }
            boolean batched = true;
            boolean tokens = true;
            int batchedSteps = 0;
            int soloExact = 0;
            int soloSteps = 0;
            float diff = 0;
            float scale = 0;
            for (int s = 0; s < steps; s++) {
                int[] pos = {prompts[0].length + s, prompts[1].length + s, prompts[2].length + s};
                for (int i = 0; i < 3; i++) {
                    ref[i].forward(new int[]{tok[i]}, 0, 1, pos[i], lRef[i]);
                }
                int mode = s % 4; // 0: all alone, 1 and 3: all three batched, 2: two batched and one alone
                int lead = s % 3;
                boolean[] inBatch = new boolean[3];
                if (mode == 1 || mode == 3) {
                    Arrays.fill(inBatch, true);
                } else if (mode == 2) {
                    Arrays.fill(inBatch, true);
                    inBatch[(lead + 1) % 3] = false;
                }
                List<Integer> rows = new ArrayList<>();
                for (int r = 0; r < 3; r++) {
                    if (inBatch[(lead + r) % 3]) {
                        rows.add((lead + r) % 3);
                    }
                }
                for (int i = 0; i < 3; i++) {
                    if (!inBatch[i]) {
                        on[i].forward(new int[]{tok[i]}, 0, 1, pos[i], lOn[i]);
                    }
                }
                if (!rows.isEmpty()) {
                    Backend[] members = new Backend[rows.size()];
                    int[] tokens1 = new int[rows.size()];
                    int[] positions = new int[rows.size()];
                    float[][] out = new float[rows.size()][];
                    for (int r = 0; r < rows.size(); r++) {
                        int i = rows.get(r);
                        members[r] = on[i];
                        tokens1[r] = tok[i];
                        positions[r] = pos[i];
                        out[r] = lOn[i];
                    }
                    batched &= on[rows.get(0)].forwardBatch(members, tokens1, positions, out);
                    batchedSteps++;
                }
                for (int i = 0; i < 3; i++) {
                    float d = maxDiff(lOn[i], lRef[i]);
                    diff = Math.max(diff, d);
                    scale = Math.max(scale, maxAbs(lRef[i]));
                    tokens &= argmax(lOn[i]) == argmax(lRef[i]);
                    if (!inBatch[i]) {
                        soloSteps++;
                        soloExact += d == 0 ? 1 : 0;
                    }
                    tok[i] = argmax(lRef[i]);
                }
            }
            long replayed = on[0].replayedSteps() + on[1].replayedSteps() + on[2].replayedSteps();
            // 1e-4 of the largest logit: the batched rows' rounding (docs/08 8.24), the same bound batchCheck uses
            boolean ok = batched && tokens && replayed > 0 && diff <= 1e-4f * scale + 1e-5f;
            System.out.printf("  replay x batch %s: %d steps (%d batched passes, %d single steps %d bit-identical, %d replayed),"
                            + " max|replay+batch-alone| = %.1e, tokens %s -> %s%n", dt, steps, batchedSteps, soloSteps, soloExact,
                    replayed, diff, tokens ? "same" : "DIFFERENT", ok ? "ok" : "REPLAY x BATCH MISMATCH");
            return ok ? 0 : 1;
        } finally {
            for (int i = 0; i < 3; i++) {
                if (on[i] != null) {
                    on[i].close();
                }
                if (ref[i] != null) {
                    ref[i].close();
                }
            }
        }
    }

    private static int argmax(float[] l) {
        int best = 0;
        for (int t = 1; t < l.length; t++) {
            if (l[t] > l[best]) {
                best = t;
            }
        }
        return best;
    }

    /** The logits of {@code seq}'s last token as one chunk, then those of every token after the prompt fed one at a time. */
    private static float[][] forced(Backend b, int[] seq, int vocab) {
        float[][] out = new float[1 + seq.length - PROMPT.length][vocab];
        b.forward(seq, 0, seq.length, 0, out[0]);
        b.forward(PROMPT, 0, PROMPT.length, 0, out[1]);
        for (int i = PROMPT.length; i + 1 < seq.length; i++) {
            b.forward(seq, i, 1, i, out[2 + i - PROMPT.length]);
        }
        return out;
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

    /**
     * Batched decoding across slots ({@link MetalBackend#forwardBatch}): three backends on the same weights, each with its
     * own prompt (other lengths, so other positions), decode greedily alone and then together in shared passes, the
     * leader rotating so every backend lends its buffers in turn, with one step where a slot decodes on its own between
     * batched ones. Every row must give the slot's own tokens, and its logits must match the slot's own decode step to
     * rounding; for the f32, f16 and q8 caches, and for MoE models also with experts on the CPU. The f32 case (and the
     * CPU-experts one) also runs through a {@link org.robomiri.jllm.engine.BatchGroup} with three threads (the engine's
     * path): same tokens, steps in shared passes.
     */
    private static int batchCheck(ModelWeights w, String lib, Architecture cfg) throws Exception {
        int vocab = cfg.core().vocabSize();
        int[][] prompts = {PROMPT, Arrays.copyOf(PROMPT, 7), {5, 900, 17, 3, 44, 61, 2, 8, 19, 400}};
        List<Object[]> cases = new ArrayList<>();
        for (KvDtype dt : KvDtype.values()) {
            if (dt == KvDtype.Q4 && !Q4) {
                continue;
            }
            cases.add(new Object[]{dt.toString(), dt, MoeSplit.OFF});
        }
        // (not in the forced-MMA pass: a batched pass and a lone step take different matmul kernels there, and a rotated int8 code flips)
        if (Q4 && ((Decoder) cfg).attention() instanceof Attention.Gqa rg && KvQ4.supports(rg.headDim())) {
            cases.add(new Object[]{"Q8, rotated keys", KvDtype.Q8, MoeSplit.OFF}); // backend.kv-q8-rotate: rows rotate their own q
        }
        if (((Decoder) cfg).anyMoe()) {
            cases.add(new Object[]{"F32, experts of 1 layer on the cpu", KvDtype.F32, new MoeSplit(1, 4, false)});
        }
        Path stats = Files.createTempDirectory("jllm-expert-stats");
        System.setProperty("jllm.metal.expertStatsDir", stats.toString());
        int bad = 0;
        try {
            for (Object[] c : cases) {
                KvDtype dt = (KvDtype) c[1];
                MoeSplit split = (MoeSplit) c[2];
                KvDtype.setQ8Rotate(c[0].toString().contains("rotated"));
                MetalBackend[] b = new MetalBackend[3];
                try {
                    for (int i = 0; i < 3; i++) {
                        b[i] = new MetalBackend(w, 64, 16, lib, dt, split);
                    }
                    float[][][] alone = new float[3][NEW_TOKENS][vocab];
                    int[][] aloneTok = new int[3][];
                    for (int i = 0; i < 3; i++) {
                        aloneTok[i] = decode(b[i], prompts[i], alone[i], vocab);
                    }
                    float[][][] tog = new float[3][NEW_TOKENS][vocab];
                    int[][] togTok = new int[3][NEW_TOKENS];
                    for (int i = 0; i < 3; i++) {
                        b[i].forward(prompts[i], 0, prompts[i].length, 0, tog[i][0]);
                        togTok[i][0] = argmax(tog[i][0]);
                    }
                    boolean batchedAll = true;
                    for (int s = 1; s < NEW_TOKENS; s++) {
                        int lead = s % 3;
                        int solo = s == 3 ? (lead + 1) % 3 : -1; // decodes on its own this step
                        List<Integer> rows = new ArrayList<>();
                        for (int r = 0; r < 3; r++) {
                            int i = (lead + r) % 3;
                            if (i != solo) {
                                rows.add(i);
                            }
                        }
                        Backend[] members = new Backend[rows.size()];
                        int[] tokens = new int[rows.size()];
                        int[] positions = new int[rows.size()];
                        float[][] out = new float[rows.size()][];
                        for (int r = 0; r < rows.size(); r++) {
                            int i = rows.get(r);
                            members[r] = b[i];
                            tokens[r] = togTok[i][s - 1];
                            positions[r] = prompts[i].length + s - 1;
                            out[r] = tog[i][s];
                        }
                        if (solo >= 0) {
                            b[solo].forward(new int[]{togTok[solo][s - 1]}, 0, 1, prompts[solo].length + s - 1, tog[solo][s]);
                        }
                        batchedAll &= b[lead].forwardBatch(members, tokens, positions, out);
                        for (int i = 0; i < 3; i++) {
                            togTok[i][s] = argmax(tog[i][s]);
                        }
                    }
                    float diff = 0;
                    float scale = 0;
                    boolean exact = true;
                    boolean same = batchedAll;
                    for (int i = 0; i < 3; i++) {
                        same &= Arrays.equals(aloneTok[i], togTok[i]);
                        for (int s = 0; s < NEW_TOKENS; s++) {
                            diff = Math.max(diff, maxDiff(alone[i][s], tog[i][s]));
                            scale = Math.max(scale, maxAbs(alone[i][s]));
                            exact &= Arrays.equals(alone[i][s], tog[i][s]);
                        }
                    }
                    boolean ok = same && diff <= 1e-4f * scale + 1e-5f;
                    System.out.printf("  batched decode, 3 slots, kv %s: tokens %s, max|alone-batched| = %.1e%s -> %s%n",
                            c[0], same ? "same" : "DIFFERENT", diff, exact ? " (bit-identical)" : "",
                            ok ? "ok" : "BATCH MISMATCH");
                    bad += ok ? 0 : 1;
                    if (dt == KvDtype.F32) {
                        bad += groupCheck(b, prompts, aloneTok, vocab, true);
                    }
                } finally {
                    for (MetalBackend x : b) {
                        if (x != null) {
                            x.close();
                        }
                    }
                }
            }
        } finally {
            KvDtype.setQ8Rotate(false);
            try (var files = Files.list(stats)) {
                for (Path f : files.toList()) {
                    Files.delete(f);
                }
            }
            Files.delete(stats);
            System.clearProperty("jllm.metal.expertStatsDir");
        }
        return bad;
    }

    /**
     * Qwen3-Next (Gated DeltaNet states plus a KV cache for the full-attention layers) is not batched: its Metal
     * backend has no {@code forwardBatch}, so a batch group runs the queued steps one after another. Verified here:
     * three slots in one group, three threads decoding at once, give the tokens each gives alone, and no step runs in
     * a shared pass (a pass that claimed to batch them would be unchecked code). Draft and prompt-lookup speculation
     * are off for it as well: a recurrent state cannot roll back ({@code Speculator.supported}), and
     * {@code ModelLoader.loadDraft} refuses it as a draft or a target.
     */
    private static int hybridBatchCheck(ModelWeights w, String lib) throws Exception {
        int vocab = w.arch.core().vocabSize();
        int[][] prompts = {PROMPT, Arrays.copyOf(PROMPT, 7), {5, 900, 17, 3, 44, 61, 2, 8, 19, 400}};
        Qwen3NextMetalBackend[] b = new Qwen3NextMetalBackend[3];
        int bad = 0;
        try {
            for (int i = 0; i < 3; i++) {
                b[i] = new Qwen3NextMetalBackend(w, 64, 16, lib, KvDtype.F32);
            }
            int[][] alone = new int[3][];
            for (int i = 0; i < 3; i++) {
                alone[i] = decode(b[i], prompts[i], null, vocab);
            }
            for (int i = 0; i < 3; i++) { // the same slots again, from scratch: a group must reproduce the tokens
                b[i].close();
                b[i] = new Qwen3NextMetalBackend(w, 64, 16, lib, KvDtype.F32);
            }
            boolean spec = Speculator.supported(b[0]);
            System.out.println("  qwen3-next speculation supported: " + spec + (spec ? " -> SPECULATION ON FOR A RECURRENT STATE" : " -> ok"));
            bad += spec ? 1 : 0;
            bad += groupCheck(b, prompts, alone, vocab, false);
        } finally {
            for (Qwen3NextMetalBackend x : b) {
                if (x != null) {
                    x.close();
                }
            }
        }
        return bad;
    }

    /**
     * The slots of {@code b} in a batch group, three threads decoding at once: each gets its own tokens of alone.
     * With {@code batchable}, steps must also have run in shared passes; without it (a backend that cannot batch),
     * none may have.
     */
    private static int groupCheck(Backend[] b, int[][] prompts, int[][] alone, int vocab, boolean batchable) throws Exception {
        org.robomiri.jllm.engine.BatchGroup g = new org.robomiri.jllm.engine.BatchGroup();
        Backend[] m = new Backend[b.length];
        for (int i = 0; i < b.length; i++) {
            m[i] = g.join(b[i]);
        }
        ExecutorService ex = Executors.newFixedThreadPool(b.length);
        // The threads meet before every decode step. Without it the outcome depends on the GPU's speed: the thread
        // whose prefill ends first decodes all its steps alone while the others still wait for the device lock for their
        // prefills (a leader only waits for slots that decoded within the last 200 ms), and on a fast GPU no step is ever
        // shared. A server's slots behave like this (each samples a token between steps and comes right back), and it
        // keeps what is verified: that steps really run in shared passes and give the tokens they give alone.
        CyclicBarrier meet = new CyclicBarrier(b.length);
        try {
            List<Future<int[]>> fs = new ArrayList<>();
            for (int i = 0; i < b.length; i++) {
                int k = i;
                fs.add(ex.submit(() -> decode(m[k], prompts[k], null, vocab, meet)));
            }
            boolean same = true;
            for (int i = 0; i < b.length; i++) {
                same &= Arrays.equals(fs.get(i).get(), alone[i]);
            }
            long[] st = g.stats();
            boolean ok = same && (batchable ? st[1] > 0 : st[1] == 0);
            System.out.printf("  batch group, 3 threads: tokens %s, %d decode steps in shared passes (%d passes) -> %s%n",
                    same ? "same" : "DIFFERENT", st[1], st[0], ok ? "ok" : "GROUP MISMATCH");
            return ok ? 0 : 1;
        } finally {
            ex.shutdown();
        }
    }

    /** Greedy: prefill {@code prompt}, then NEW_TOKENS - 1 decode steps; step s's logits to stepLogits[s] if not null. */
    private static int[] decode(Backend b, int[] prompt, float[][] stepLogits, int vocab) {
        return decode(b, prompt, stepLogits, vocab, null);
    }

    /** As above; the threads sharing {@code meet} start every decode step together. */
    private static int[] decode(Backend b, int[] prompt, float[][] stepLogits, int vocab, CyclicBarrier meet) {
        float[] logits = stepLogits != null ? stepLogits[0] : new float[vocab];
        b.forward(prompt, 0, prompt.length, 0, logits);
        int[] out = new int[NEW_TOKENS];
        int pos = prompt.length;
        for (int s = 0; s < NEW_TOKENS; s++) {
            if (s > 0) {
                if (stepLogits != null) {
                    logits = stepLogits[s];
                }
                if (meet != null) {
                    try {
                        meet.await(60, TimeUnit.SECONDS);
                    } catch (Exception e) {
                        throw new IllegalStateException("the decoding threads did not meet", e);
                    }
                }
                b.forward(new int[]{out[s - 1]}, 0, 1, pos++, logits);
            }
            out[s] = argmax(logits);
        }
        return out;
    }

    /** {@code -Djllm.smoke.tokenizer} (default: Qwen3-0.6B's, as the tests use): the engine needs one to stream text. */
    private static org.robomiri.jllm.tokenizer.Tokenizer tokenizer() throws Exception {
        Path p = Path.of(System.getProperty("jllm.smoke.tokenizer", "models/Qwen_Qwen3-0.6B/tokenizer.json"));
        return Files.isRegularFile(p) ? org.robomiri.jllm.tokenizer.Tokenizer.load(p) : null;
    }

    /**
     * Draft-model speculation on the Metal backend, through the engine (pipelined greedy target, pipelined draft, one
     * multi-token verification pass, both caches rolled back by position): the greedy output with a draft equals the
     * output without speculation, for f32, f16 and q8 caches, with three drafts: the same weights (every draft is
     * accepted), the same checkpoint at Q2_K (some are rejected) and, with {@code -Djllm.smoke.draftPair=<target dir
     * name>=<draft dir>}, another random model of the same shapes (most are rejected).
     */
    private static int draftCheck(ModelWeights w, Path dir, String lib, Architecture cfg) throws Exception {
        org.robomiri.jllm.tokenizer.Tokenizer tok = tokenizer();
        if (tok == null) {
            System.out.println("  draft speculation skipped: no tokenizer (-Djllm.smoke.tokenizer)");
            return 0;
        }
        int vocab = cfg.core().vocabSize();
        int max = 32; // within ctx 64 with the prompt and a full draft
        var greedy = new org.robomiri.jllm.sampling.SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, max, List.of());
        var fmt = org.robomiri.jllm.tokenizer.ChatTemplate.forModelType("qwen3", tok);
        List<Object[]> drafts = new ArrayList<>();
        drafts.add(new Object[]{"self", w});
        ModelWeights q2 = ModelWeights.load(dir, cfg, QType.Q2_K, QuantFile.DEFAULT_HEAD, ShardProvider.local(dir), null);
        drafts.add(new Object[]{"Q2_K", q2});
        ModelWeights other = null;
        String pair = System.getProperty("jllm.smoke.draftPair", "");
        if (pair.startsWith(dir.getFileName() + "=")) {
            Path od = Path.of(pair.substring(pair.indexOf('=') + 1));
            other = ModelWeights.load(od, Architecture.load(od), QType.Q4_0, QuantFile.DEFAULT_HEAD, ShardProvider.local(od), null);
            drafts.add(new Object[]{od.getFileName().toString(), other});
        }
        int bad = 0;
        try {
            int rotRound = ((Decoder) cfg).attention() instanceof Attention.Gqa rg && KvQ4.supports(rg.headDim()) ? 1 : 0;
            for (int round = 0; round < KvDtype.values().length + rotRound; round++) {
                // the last round: q8 with rotated keys (backend.kv-q8-rotate); the rolled-back positions are re-stored rotated
                KvDtype dt = round < KvDtype.values().length ? KvDtype.values()[round] : KvDtype.Q8;
                KvDtype.setQ8Rotate(round >= KvDtype.values().length);
                int[] plain;
                try (org.robomiri.jllm.engine.Engine e = new org.robomiri.jllm.engine.Engine(new MetalBackend(w, 64, 16, lib, dt),
                        tok, cfg, Set.of(), 0, fmt)) {
                    plain = e.generate(PROMPT, greedy, null, () -> false).tokens();
                }
                StringBuilder line = new StringBuilder();
                boolean ok = true;
                for (Object[] d : drafts) {
                    if (dt != KvDtype.F32 && d[1] == w) {
                        continue; // the f16 / q8 caches: the drafts that get rejected, which is what rolls back
                    }
                    // replay is on by default: the draft's pipelined steps are recorded once and replayed (docs/08 8.22), and
                    // the output must still be the plain run's; the drafting for itself is also run with replay off
                    // on both backends, to show that the two agree rather than both being wrong the same way
                    for (boolean replay : d[1] == w && dt == KvDtype.F32 ? new boolean[]{true, false} : new boolean[]{true}) {
                    MetalBackend draftBackend = new MetalBackend((ModelWeights) d[1], 64, 16, lib, dt);
                    MetalBackend targetBackend = new MetalBackend(w, 64, 16, lib, dt);
                    draftBackend.replay(replay);
                    targetBackend.replay(replay);
                    var dm = new org.robomiri.jllm.engine.DraftModel(draftBackend, vocab, null);
                    try (org.robomiri.jllm.engine.Engine e = new org.robomiri.jllm.engine.Engine(targetBackend, tok, cfg, Set.of(), 0, fmt)) {
                        // a pass on every step (the emulator's timings would make the cost model pause speculation)
                        System.setProperty("jllm.engine.adaptiveDraft", "false");
                        try {
                            e.enableSpeculation(new org.robomiri.jllm.engine.Speculator.Config(
                                    org.robomiri.jllm.engine.Speculator.Mode.DRAFT, 4, 8, 3, 0), dm);
                        } finally {
                            System.clearProperty("jllm.engine.adaptiveDraft");
                        }
                        var r = e.generate(PROMPT, greedy, null, () -> false);
                        boolean same = Arrays.equals(plain, r.tokens());
                        boolean replayed = !replay || draftBackend.replayedSteps() > 0;
                        ok &= same && r.stats().draftedTokens() > 0 && replayed;
                        line.append(String.format(", %s draft%s %d/%d accepted%s%s", d[0], replay ? "" : " (replay off)",
                                r.stats().acceptedTokens(), r.stats().draftedTokens(), same ? "" : " OUTPUT DIFFERS",
                                replayed ? "" : " DRAFT STEPS NOT REPLAYED"));
                    }
                    }
                }
                System.out.printf("  draft speculation, kv %s%s: %d tokens%s -> %s%n", dt, KvDtype.q8Rotate() ? " rotated keys" : "", plain.length, line,
                        ok ? "ok" : "SPECULATION MISMATCH");
                bad += ok ? 0 : 1;
            }
        } finally {
            KvDtype.setQ8Rotate(false);
            q2.close();
            if (other != null) {
                other.close();
            }
        }
        return bad;
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
