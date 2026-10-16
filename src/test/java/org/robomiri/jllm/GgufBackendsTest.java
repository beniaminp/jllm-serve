package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.gguf.GgmlType;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.MetalBackend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * The CPU and Metal backends on tiny random GGUF models whose tensors are random blocks of the ggml formats read in
 * place (Q4_0, Q8_0, Q4_K, Q5_K) -- every dispatch of the forward pass (fused q/k/v, split gate|up, MoE experts, o_proj,
 * down, LM head) on every family, with all matrices of one format and with the formats mixed per tensor (which makes
 * the q/k/v dispatch and the gate|up pair mixed too) -- must give the same logits within float rounding. Skipped
 * when Metal is not available.
 */
public final class GgufBackendsTest {

    static final int DIM = 256, HEADS = 8, KV = 4, HD = 32, FF = 512, LAYERS = 2, VOCAB = 256, EXPERTS = 4, TOPK = 2, MOE_DIM = 256;
    static int passed;
    static int failed;

    static void check(boolean ok, String what) {
        if (ok) {
            passed++;
        } else {
            failed++;
            System.out.println("FAIL: " + what);
        }
    }

    private static final GgmlType[] TYPES = {GgmlType.Q4_0, GgmlType.Q8_0, GgmlType.Q4_K, GgmlType.Q5_K};

    /** Random valid blocks: random quant bytes, f16 scales small enough that the weights are about +-0.1. */
    static byte[] blocks(Random r, GgmlType t, long elements) {
        int nb = (int) (elements / t.blockSize);
        byte[] b = new byte[nb * t.typeSize];
        r.nextBytes(b);
        ByteBuffer bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < nb; i++) {
            int o = i * t.typeSize;
            switch (t) {
                case Q4_0 -> bb.putShort(o, Float.floatToFloat16(0.004f + 0.012f * r.nextFloat()));
                case Q8_0 -> bb.putShort(o, Float.floatToFloat16(0.0004f + 0.0012f * r.nextFloat()));
                default -> {
                    bb.putShort(o, Float.floatToFloat16(0.00004f + 0.00012f * r.nextFloat()));
                    bb.putShort(o + 2, Float.floatToFloat16(0.00004f + 0.00012f * r.nextFloat()));
                }
            }
        }
        return b;
    }

    static GgmlType pick(String mode, String name, Random r) {
        if (mode.equals("mixed")) {
            return TYPES[Math.floorMod(name.hashCode() + r.nextInt(2), TYPES.length)];
        }
        return GgmlType.valueOf(mode);
    }

    static void matrix(GgufWriter w, Random r, String mode, String name, long... dims) {
        long n = 1;
        for (long d : dims) {
            n *= d;
        }
        GgmlType t = pick(mode, name, r);
        w.tensor(name, t, dims, blocks(r, t, n));
    }

    static void vec(GgufWriter w, Random r, String name, int n, double base, double std) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) (base + std * r.nextGaussian());
        }
        w.f32Tensor(name, new long[]{n}, v);
    }

    static void write(Path p, String family, String mode, long seed) throws Exception {
        Random r = new Random(seed);
        String arch = family.equals("qwen3_moe") ? "qwen3moe" : family;
        boolean moe = family.equals("qwen3_moe");
        boolean qwen3 = family.startsWith("qwen3");
        boolean bias = family.equals("qwen2");
        boolean tied = !family.equals("llama");
        GgufWriter w = new GgufWriter().str("general.architecture", arch);
        String a = arch + ".";
        w.u32(a + "block_count", LAYERS).u32(a + "embedding_length", DIM).u32(a + "feed_forward_length", FF)
                .u32(a + "attention.head_count", HEADS).u32(a + "attention.head_count_kv", KV).u32(a + "context_length", 512)
                .f32(a + "rope.freq_base", 10000f).f32(a + "attention.layer_norm_rms_epsilon", 1e-5f)
                .u32(a + "attention.key_length", HD).u32(a + "rope.dimension_count", HD)
                .u32("tokenizer.ggml.bos_token_id", 1).u32("tokenizer.ggml.eos_token_id", 2);
        if (moe) {
            w.u32(a + "expert_count", EXPERTS).u32(a + "expert_used_count", TOPK).u32(a + "expert_feed_forward_length", MOE_DIM);
        }
        matrix(w, r, mode, "token_embd.weight", DIM, VOCAB);
        vec(w, r, "output_norm.weight", DIM, 1.0, 0.1);
        if (!tied) {
            matrix(w, r, mode, "output.weight", DIM, VOCAB);
        }
        for (int l = 0; l < LAYERS; l++) {
            String b = "blk." + l + ".";
            vec(w, r, b + "attn_norm.weight", DIM, 1.0, 0.1);
            matrix(w, r, mode, b + "attn_q.weight", DIM, HEADS * HD);
            matrix(w, r, mode, b + "attn_k.weight", DIM, KV * HD);
            matrix(w, r, mode, b + "attn_v.weight", DIM, KV * HD);
            matrix(w, r, mode, b + "attn_output.weight", HEADS * HD, DIM);
            if (bias) {
                vec(w, r, b + "attn_q.bias", HEADS * HD, 0, 0.1);
                vec(w, r, b + "attn_k.bias", KV * HD, 0, 0.1);
                vec(w, r, b + "attn_v.bias", KV * HD, 0, 0.1);
            }
            if (qwen3) {
                vec(w, r, b + "attn_q_norm.weight", HD, 1.0, 0.1);
                vec(w, r, b + "attn_k_norm.weight", HD, 1.0, 0.1);
            }
            vec(w, r, b + "ffn_norm.weight", DIM, 1.0, 0.1);
            if (moe) {
                float[] router = new float[EXPERTS * DIM];
                for (int i = 0; i < router.length; i++) {
                    router[i] = (float) (0.1 * r.nextGaussian());
                }
                w.f32Tensor(b + "ffn_gate_inp.weight", new long[]{DIM, EXPERTS}, router);
                matrix(w, r, mode, b + "ffn_gate_exps.weight", DIM, MOE_DIM, EXPERTS);
                matrix(w, r, mode, b + "ffn_up_exps.weight", DIM, MOE_DIM, EXPERTS);
                matrix(w, r, mode, b + "ffn_down_exps.weight", MOE_DIM, DIM, EXPERTS);
            } else {
                matrix(w, r, mode, b + "ffn_gate.weight", DIM, FF);
                matrix(w, r, mode, b + "ffn_up.weight", DIM, FF);
                matrix(w, r, mode, b + "ffn_down.weight", FF, DIM);
            }
        }
        w.write(p);
    }

    public static void main(String[] args) throws Exception {
        String lib = System.getProperty("jllm.metal.lib");
        if (!Metal.available(lib)) {
            System.out.println("gguf backends: skipped (Metal is not available)");
            return;
        }
        Path dir = Files.createTempDirectory("gguf-backends");
        try {
            int[] prompt = {1, 5, 17, 33, 2, 40, 9, 77, 120, 3, 200, 14};
            for (String family : new String[]{"llama", "qwen2", "qwen3", "qwen3_moe"}) {
                for (String mode : new String[]{"Q4_0", "Q8_0", "Q4_K", "Q5_K", "mixed"}) {
                    Path p = dir.resolve(family + "-" + mode + ".gguf");
                    write(p, family, mode, 100 + family.hashCode() + mode.hashCode());
                    float[][] cpu;
                    float[][] gpu;
                    try (GgufModel g = GgufModel.open(p); ModelWeights w = g.weights();
                         CpuBackend b = new CpuBackend(w, 64, 8, 4, KvDtype.F32, false)) {
                        cpu = run(b, prompt);
                    }
                    try (GgufModel g = GgufModel.open(p); ModelWeights w = g.weights();
                         MetalBackend b = new MetalBackend(w, 64, 8, lib, KvDtype.F32)) {
                        gpu = run(b, prompt);
                    }
                    double diff = GgufModelTest.maxDiff(cpu, gpu);
                    boolean sameTokens = true;
                    for (int i = 0; i < cpu.length; i++) {
                        sameTokens &= GgufModelTest.argmax(cpu[i]) == GgufModelTest.argmax(gpu[i]);
                    }
                    double mag = 0;
                    for (float[] row : cpu) {
                        for (float v : row) {
                            mag = Math.max(mag, Math.abs(v));
                        }
                    }
                    check(!Double.isNaN(diff) && mag > 0 && mag < 1e6, family + "/" + mode + ": logits are finite (max |logit| " + mag + ")");
                    check(diff < 1e-3, family + "/" + mode + ": CPU vs Metal max logit diff " + diff);
                    check(sameTokens, family + "/" + mode + ": same greedy tokens");
                    System.out.printf("%-9s %-6s CPU vs Metal: max logit diff %.2e (max |logit| %.2f)%n", family, mode, diff, mag);
                }
            }
        } finally {
            try (var s = Files.walk(dir)) {
                s.sorted(java.util.Comparator.reverseOrder()).forEach(q -> q.toFile().delete());
            }
        }
        System.out.println("gguf backends: " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    /** Prefill of the prompt in chunks of up to 8 tokens, then 4 greedy decode steps; logits after each call. */
    static float[][] run(org.robomiri.jllm.model.Backend b, int[] prompt) {
        float[][] out = new float[3 + 4][VOCAB];
        int pos = 0;
        int call = 0;
        for (int off = 0; off < prompt.length; off += 8) {
            int n = Math.min(8, prompt.length - off);
            b.forward(prompt, off, n, pos, out[call++]);
            pos += n;
        }
        // the first decode token is the argmax of the last prefill logits
        for (int i = 0; i < 4; i++) {
            int next = GgufModelTest.argmax(out[call - 1]);
            b.forward(new int[]{next}, 0, 1, pos++, out[call++]);
        }
        return java.util.Arrays.copyOf(out, call);
    }
}
