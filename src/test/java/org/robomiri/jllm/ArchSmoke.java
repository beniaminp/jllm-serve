package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.cpu.MambaCpuBackend;
import org.robomiri.jllm.cpu.Qwen3NextCpuBackend;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
            if (!recurrent) {
                failures += pipeCheck(w, lib, cfg, gpu, label);
            }
            if (!mamba && quant == QType.Q4_0 && head == QuantFile.DEFAULT_HEAD) {
                failures += batchCheck(w, lib, cfg);
                failures += draftCheck(w, dir, lib, cfg);
            }
            if (cfg instanceof Decoder d && d.anyMoe()) {
                failures += moeSplitCheck(w, lib, cfg, cpu, cpuLogits);
            }
            if (!mamba && cfg instanceof Decoder && quant == QType.Q4_0 && head == QuantFile.DEFAULT_HEAD) {
                failures += q8Check(w, lib, cfg, cpu);
                // f32 and q8 caches for every family; the f16 cache (its own mix of staging and cache rows) on the
                // first dense and the first MoE family, to keep the emulator run's time in bounds
                failures += replayCheck(w, lib, cfg, KvDtype.F32);
                failures += replayCheck(w, lib, cfg, KvDtype.Q8);
                if (replayF16.add(((Decoder) cfg).anyMoe() ? "moe" : "dense")) {
                    failures += replayCheck(w, lib, cfg, KvDtype.F16);
                }
            }
            if (!mamba && cfg instanceof Decoder dec && quant == QType.Q4_0 && head == QuantFile.DEFAULT_HEAD) { // standard and latent (MLA) caches alike
                boolean latent = dec.attention() instanceof Attention.Mla;
                for (KvDtype dt : KvDtype.values()) {
                    if (!(latent && dt == KvDtype.Q8)) { // a q8 latent cache is not saved
                        failures += kvStateChecks(w, lib, cfg, dt);
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
     * The q8 KV cache (int8 with a scale per row of a kv head; DeepSeek: per latent row) on both backends, teacher-forced
     * through the prompt followed by the f32 CPU run's tokens: once as one 20-token chunk (the prefill kernels, flash
     * prefill where the tuner enabled it) and once as the prompt then token by token (decode). CPU q8 and Metal q8 must
     * agree within the usual CPU / Metal tolerance at every step. The distance of the q8 logits from the f32 cache's
     * (CPU) is printed next to the f16 cache's: how much a random model amplifies a perturbation varies a lot (the
     * tiny DeepSeeks turn the f16 cache's 0.2-0.4% of the largest logit into 13-25% for q8, the dense families stay
     * near 1%), so q8 is bounded relative to f16, by 200 times its distance (observed: 20-45 times for the dense
     * families, up to 115 for the DeepSeeks) plus 10^-3 of the largest logit: a broken dequantization is caught, the
     * amplification is not.
     */
    private static int q8Check(ModelWeights w, String lib, Architecture cfg, int[] tokens) throws Exception {
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
        try (Backend b = new CpuBackend(w, 64, 32, 4, KvDtype.Q8, false)) {
            cpu = forced(b, seq, cfg.core().vocabSize());
        }
        try (Backend b = new MetalBackend(w, 64, 32, lib, KvDtype.Q8)) {
            gpu = forced(b, seq, cfg.core().vocabSize());
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
        boolean agree = diff <= 1e-3f * scale + 1e-4f;
        boolean close = drift <= 200 * drift16 + 1e-3f * scale;
        System.out.printf("  kv q8: max|cpu-metal| = %.2e%s, max|q8-f32| = %.2e%s (f16: %.2e, max|logit| = %.2f)%n", diff,
                agree ? "" : " MISMATCH", drift, close ? "" : " TOO FAR", drift16, scale);
        return (agree ? 0 : 1) + (close ? 0 : 1);
    }

    private static final int REPLAY_STEPS = 40;
    private static final java.util.Set<String> replayF16 = new java.util.HashSet<>();

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
            cases.add(new Object[]{dt.toString(), dt, MoeSplit.OFF});
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
                        bad += groupCheck(b, prompts, aloneTok, vocab);
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

    /** The slots of {@code b} in a batch group, three threads decoding at once: each gets its own tokens of alone. */
    private static int groupCheck(MetalBackend[] b, int[][] prompts, int[][] alone, int vocab) throws Exception {
        org.robomiri.jllm.engine.BatchGroup g = new org.robomiri.jllm.engine.BatchGroup();
        Backend[] m = new Backend[b.length];
        for (int i = 0; i < b.length; i++) {
            m[i] = g.join(b[i]);
        }
        ExecutorService ex = Executors.newFixedThreadPool(b.length);
        try {
            List<Future<int[]>> fs = new ArrayList<>();
            for (int i = 0; i < b.length; i++) {
                int k = i;
                fs.add(ex.submit(() -> decode(m[k], prompts[k], null, vocab)));
            }
            boolean same = true;
            for (int i = 0; i < b.length; i++) {
                same &= Arrays.equals(fs.get(i).get(), alone[i]);
            }
            long[] st = g.stats();
            boolean ok = same && st[1] > 0;
            System.out.printf("  batch group, 3 threads: tokens %s, %d decode steps in shared passes (%d passes) -> %s%n",
                    same ? "same" : "DIFFERENT", st[1], st[0], ok ? "ok" : "GROUP MISMATCH");
            return ok ? 0 : 1;
        } finally {
            ex.shutdown();
        }
    }

    /** Greedy: prefill {@code prompt}, then NEW_TOKENS - 1 decode steps; step s's logits to stepLogits[s] if not null. */
    private static int[] decode(Backend b, int[] prompt, float[][] stepLogits, int vocab) {
        float[] logits = stepLogits != null ? stepLogits[0] : new float[vocab];
        b.forward(prompt, 0, prompt.length, 0, logits);
        int[] out = new int[NEW_TOKENS];
        int pos = prompt.length;
        for (int s = 0; s < NEW_TOKENS; s++) {
            if (s > 0) {
                if (stepLogits != null) {
                    logits = stepLogits[s];
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
            for (KvDtype dt : KvDtype.values()) {
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
                    var dm = new org.robomiri.jllm.engine.DraftModel(new MetalBackend((ModelWeights) d[1], 64, 16, lib, dt), vocab,
                            null);
                    try (org.robomiri.jllm.engine.Engine e = new org.robomiri.jllm.engine.Engine(new MetalBackend(w, 64, 16, lib, dt),
                            tok, cfg, Set.of(), 0, fmt)) {
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
                        ok &= same && r.stats().draftedTokens() > 0;
                        line.append(String.format(", %s draft %d/%d accepted%s", d[0], r.stats().acceptedTokens(),
                                r.stats().draftedTokens(), same ? "" : " OUTPUT DIFFERS"));
                    }
                }
                System.out.printf("  draft speculation, kv %s: %d tokens%s -> %s%n", dt, plain.length, line,
                        ok ? "ok" : "SPECULATION MISMATCH");
                bad += ok ? 0 : 1;
            }
        } finally {
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
