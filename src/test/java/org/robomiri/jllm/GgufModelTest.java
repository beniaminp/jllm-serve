package org.robomiri.jllm;

import org.robomiri.jllm.cpu.CpuBackend;
import org.robomiri.jllm.gguf.GgmlType;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.ShardProvider;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * GGUF loading end to end on tiny random models (no download, CPU backend). For each family the same random
 * weights are written twice: as a Hugging Face checkpoint (config.json + safetensors) that the converter turns into
 * a Q8_0 cache, and as a GGUF file with llama.cpp's tensor names, dimension order and, for the {@code llama}
 * architecture, its permutation of the q and k rows (reimplemented here from convert_hf_to_gguf.py's
 * {@code LlamaModel.permute}). The logits of the two loads must agree:
 * <ul>
 *   <li>GGUF with F32 tensors: the engine quantizes them to Q8_0 exactly as the converter does, so the only
 *       difference is the RoPE pairing (interleaved for the permuted llama rows): logits equal to ~1e-5;</li>
 *   <li>GGUF with Q8_0 blocks (f16 scales): read in place by the kernels (dense gate/up as a split view, experts as
 *       a split interleaved pair), compared with the F32 file of the same dequantized values.</li>
 * </ul>
 * Also checks the Architecture resolved from the metadata (same description as the safetensors config) and that
 * <em>not</em> undoing the permutation is what breaks llama (the same file loaded as qwen2 differs).
 */
public final class GgufModelTest {

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

    // ------------------------------------------------------------------ tiny model description

    static final int DIM = 64, HEADS = 4, KV = 2, HD = 16, FF = 96, LAYERS = 2, VOCAB = 96, EXPERTS = 4, TOPK = 2, MOE_DIM = 32;

    record Tiny(String family, Map<String, Object> config, Map<String, float[]> hf, Map<String, long[]> shapes) {
    }

    private static float[] rnd(Random r, int n, double std, double base) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (float) (base + r.nextGaussian() * std);
        }
        return a;
    }

    /** YaRN of the {@code -yarn} variants: factor 4 over a pretrained length of 64, betas 16 / 2, attention factor 1.2. */
    static final Map<String, Object> YARN = Json.obj("rope_type", "yarn", "factor", 4.0, "original_max_position_embeddings", 64L,
            "beta_fast", 16.0, "beta_slow", 2.0, "attention_factor", 1.2);

    static Tiny tiny(String variant, long seed) {
        Random r = new Random(seed);
        boolean yarn = variant.endsWith("-yarn");
        String family = yarn ? variant.substring(0, variant.length() - 5) : variant;
        boolean moe = family.equals("qwen3_moe");
        boolean qwen3 = family.startsWith("qwen3");
        boolean bias = family.equals("qwen2");
        Map<String, Object> c = Json.obj("model_type", family, "hidden_size", (long) DIM, "intermediate_size", (long) FF,
                "num_hidden_layers", (long) LAYERS, "num_attention_heads", (long) HEADS, "num_key_value_heads", (long) KV,
                "vocab_size", (long) VOCAB, "max_position_embeddings", 256L, "rms_norm_eps", 1e-5,
                "rope_theta", 500000.0, "hidden_act", "silu", "tie_word_embeddings", !family.equals("llama"),
                "bos_token_id", 1L, "eos_token_id", 2L);
        if (qwen3) {
            c.put("head_dim", (long) HD);
        }
        if (yarn) {
            c.put("rope_scaling", YARN);
        }
        if (moe) {
            c.put("num_experts", (long) EXPERTS);
            c.put("num_experts_per_tok", (long) TOPK);
            c.put("moe_intermediate_size", (long) MOE_DIM);
            c.put("norm_topk_prob", true);
            c.put("decoder_sparse_step", 1L);
        }
        Map<String, float[]> t = new LinkedHashMap<>();
        Map<String, long[]> sh = new LinkedHashMap<>();
        java.util.function.BiConsumer<String, long[]> mat = (name, shape) -> {
            int n = 1;
            for (long d : shape) {
                n *= (int) d;
            }
            t.put(name, rnd(r, n, 0.08, 0));
            sh.put(name, shape);
        };
        java.util.function.BiConsumer<String, long[]> vec = (name, shape) -> {
            t.put(name, rnd(r, (int) shape[0], 0.1, 1.0));
            sh.put(name, shape);
        };
        mat.accept("model.embed_tokens.weight", new long[]{VOCAB, DIM});
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            vec.accept(p + "input_layernorm.weight", new long[]{DIM});
            mat.accept(p + "self_attn.q_proj.weight", new long[]{HEADS * HD, DIM});
            mat.accept(p + "self_attn.k_proj.weight", new long[]{KV * HD, DIM});
            mat.accept(p + "self_attn.v_proj.weight", new long[]{KV * HD, DIM});
            mat.accept(p + "self_attn.o_proj.weight", new long[]{DIM, HEADS * HD});
            if (bias) {
                for (String n : new String[]{"q", "k", "v"}) {
                    int rows = n.equals("q") ? HEADS * HD : KV * HD;
                    t.put(p + "self_attn." + n + "_proj.bias", rnd(r, rows, 0.1, 0));
                    sh.put(p + "self_attn." + n + "_proj.bias", new long[]{rows});
                }
            }
            if (qwen3) {
                vec.accept(p + "self_attn.q_norm.weight", new long[]{HD});
                vec.accept(p + "self_attn.k_norm.weight", new long[]{HD});
            }
            vec.accept(p + "post_attention_layernorm.weight", new long[]{DIM});
            if (moe) {
                mat.accept(p + "mlp.gate.weight", new long[]{EXPERTS, DIM});
                for (int e = 0; e < EXPERTS; e++) {
                    mat.accept(p + "mlp.experts." + e + ".gate_proj.weight", new long[]{MOE_DIM, DIM});
                    mat.accept(p + "mlp.experts." + e + ".up_proj.weight", new long[]{MOE_DIM, DIM});
                    mat.accept(p + "mlp.experts." + e + ".down_proj.weight", new long[]{DIM, MOE_DIM});
                }
            } else {
                mat.accept(p + "mlp.gate_proj.weight", new long[]{FF, DIM});
                mat.accept(p + "mlp.up_proj.weight", new long[]{FF, DIM});
                mat.accept(p + "mlp.down_proj.weight", new long[]{DIM, FF});
            }
        }
        vec.accept("model.norm.weight", new long[]{DIM});
        if (!family.equals("qwen2") && !qwen3) {
            mat.accept("lm_head.weight", new long[]{VOCAB, DIM});
        }
        return new Tiny(family, c, t, sh);
    }

    // ------------------------------------------------------------------ writers

    static void writeSafetensors(Path p, Map<String, float[]> t, Map<String, long[]> shapes) throws IOException {
        Map<String, Object> header = new LinkedHashMap<>();
        long off = 0;
        for (Map.Entry<String, float[]> e : t.entrySet()) {
            long len = 4L * e.getValue().length;
            List<Object> dims = new ArrayList<>();
            for (long d : shapes.get(e.getKey())) {
                dims.add(d);
            }
            header.put(e.getKey(), Json.obj("dtype", "F32", "shape", dims, "data_offsets", List.of(off, off + len)));
            off += len;
        }
        byte[] h = Json.write(header).getBytes(StandardCharsets.UTF_8);
        ByteBuffer b = ByteBuffer.allocate((int) (8 + h.length + off)).order(ByteOrder.LITTLE_ENDIAN);
        b.putLong(h.length).put(h);
        for (float[] a : t.values()) {
            for (float v : a) {
                b.putFloat(v);
            }
        }
        Files.write(p, b.array());
    }

    /** llama.cpp's {@code LlamaModel.permute}: rows of each head go from (2, half) order to (half, 2) order. */
    static float[] permute(float[] w, int rows, int cols, int nHead) {
        int hd = rows / nHead;
        int half = hd / 2;
        float[] out = new float[w.length];
        for (int h = 0; h < nHead; h++) {
            for (int j = 0; j < half; j++) {
                for (int c = 0; c < 2; c++) {
                    System.arraycopy(w, (h * hd + c * half + j) * cols, out, (h * hd + 2 * j + c) * cols, cols);
                }
            }
        }
        return out;
    }

    static byte[] q8Blocks(float[] w) {
        int nb = w.length / 32;
        ByteBuffer b = ByteBuffer.allocate(nb * 34).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < nb; i++) {
            float amax = 0;
            for (int j = 0; j < 32; j++) {
                amax = Math.max(amax, Math.abs(w[i * 32 + j]));
            }
            short d16 = Float.floatToFloat16(amax / 127f);
            float d = Float.float16ToFloat(d16);
            b.putShort(d16);
            for (int j = 0; j < 32; j++) {
                b.put((byte) Math.max(-127, Math.min(127, Math.round(d == 0 ? 0 : w[i * 32 + j] / d))));
            }
        }
        return b.array();
    }

    static float[] dequantQ8(byte[] blocks) {
        ByteBuffer b = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[blocks.length / 34 * 32];
        for (int i = 0; i < blocks.length / 34; i++) {
            float d = Float.float16ToFloat(b.getShort(i * 34));
            for (int j = 0; j < 32; j++) {
                out[i * 32 + j] = b.get(i * 34 + 2 + j) * d;
            }
        }
        return out;
    }

    private static String gname(String hf) {
        String[][] m = {{"model.embed_tokens.weight", "token_embd.weight"}, {"lm_head.weight", "output.weight"},
                {"model.norm.weight", "output_norm.weight"}};
        for (String[] x : m) {
            if (hf.equals(x[0])) {
                return x[1];
            }
        }
        java.util.regex.Matcher mt = java.util.regex.Pattern.compile("model\\.layers\\.(\\d+)\\.(.*)").matcher(hf);
        if (!mt.matches()) {
            throw new IllegalArgumentException(hf);
        }
        String b = "blk." + mt.group(1) + ".";
        return b + switch (mt.group(2)) {
            case "input_layernorm.weight" -> "attn_norm.weight";
            case "post_attention_layernorm.weight" -> "ffn_norm.weight";
            case "self_attn.q_norm.weight" -> "attn_q_norm.weight";
            case "self_attn.k_norm.weight" -> "attn_k_norm.weight";
            case "self_attn.q_proj.weight" -> "attn_q.weight";
            case "self_attn.k_proj.weight" -> "attn_k.weight";
            case "self_attn.v_proj.weight" -> "attn_v.weight";
            case "self_attn.q_proj.bias" -> "attn_q.bias";
            case "self_attn.k_proj.bias" -> "attn_k.bias";
            case "self_attn.v_proj.bias" -> "attn_v.bias";
            case "self_attn.o_proj.weight" -> "attn_output.weight";
            case "mlp.gate_proj.weight" -> "ffn_gate.weight";
            case "mlp.up_proj.weight" -> "ffn_up.weight";
            case "mlp.down_proj.weight" -> "ffn_down.weight";
            case "mlp.gate.weight" -> "ffn_gate_inp.weight";
            default -> throw new IllegalArgumentException(hf);
        };
    }

    /**
     * Writes the GGUF file for {@code t}. {@code q8}: matrices as Q8_0 blocks (norms / biases / router stay F32);
     * otherwise everything F32. {@code permuteQk}: llama.cpp's llama-architecture permutation of q and k.
     */
    static void writeGguf(Path p, Tiny t, boolean q8, boolean permuteQk, String arch, Map<String, float[]> effective,
                          Map<String, long[]> dimsOut) throws IOException {
        GgufWriter w = new GgufWriter().str("general.architecture", arch).str("general.name", "tiny-" + t.family());
        String a = arch + ".";
        w.u32(a + "block_count", LAYERS).u32(a + "embedding_length", DIM).u32(a + "feed_forward_length", FF)
                .u32(a + "attention.head_count", HEADS).u32(a + "attention.head_count_kv", KV)
                .u32(a + "context_length", 256).f32(a + "rope.freq_base", 500000f)
                .f32(a + "attention.layer_norm_rms_epsilon", 1e-5f).u32(a + "attention.key_length", HD)
                .u32(a + "rope.dimension_count", HD);
        w.u32("tokenizer.ggml.bos_token_id", 1).u32("tokenizer.ggml.eos_token_id", 2);
        yarnMeta(w, a, t);
        boolean moe = t.family().equals("qwen3_moe");
        if (moe) {
            w.u32(a + "expert_count", EXPERTS).u32(a + "expert_used_count", TOPK).u32(a + "expert_feed_forward_length", MOE_DIM);
        }
        Map<String, float[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, float[]> e : t.hf().entrySet()) {
            String hf = e.getKey();
            if (hf.contains(".mlp.experts.")) {
                continue; // stacked below
            }
            float[] data = e.getValue();
            long[] shape = t.shapes().get(hf);
            if (permuteQk && hf.endsWith("self_attn.q_proj.weight")) {
                data = permute(data, (int) shape[0], (int) shape[1], HEADS);
            } else if (permuteQk && hf.endsWith("self_attn.k_proj.weight")) {
                data = permute(data, (int) shape[0], (int) shape[1], KV);
            }
            addTensor(w, out, dimsOut, gname(hf), shape, data, q8 && shape.length == 2 && !hf.endsWith("mlp.gate.weight"));
        }
        if (moe) {
            for (int l = 0; l < LAYERS; l++) {
                String p0 = "model.layers." + l + ".mlp.experts.";
                for (String[] part : new String[][]{{"gate_proj", "ffn_gate_exps"}, {"up_proj", "ffn_up_exps"}, {"down_proj", "ffn_down_exps"}}) {
                    boolean down = part[0].equals("down_proj");
                    int rows = down ? DIM : MOE_DIM;
                    int cols = down ? MOE_DIM : DIM;
                    float[] all = new float[EXPERTS * rows * cols];
                    for (int e = 0; e < EXPERTS; e++) {
                        System.arraycopy(t.hf().get(p0 + e + "." + part[0] + ".weight"), 0, all, e * rows * cols, rows * cols);
                    }
                    addTensor(w, out, dimsOut, "blk." + l + "." + part[1] + ".weight", new long[]{EXPERTS, rows, cols}, all, q8);
                }
            }
        }
        w.write(p);
        effective.putAll(out);
    }

    /**
     * The keys convert_hf_to_gguf.py writes for a yarn {@code rope_scaling}: type, factor, original context and the
     * betas; {@code attention_factor} becomes ggml's {@code attn_factor}, the multiplier of its own
     * {@code 0.1 ln(factor) + 1}.
     */
    private static void yarnMeta(GgufWriter w, String a, Tiny t) {
        if (t.config().get("rope_scaling") instanceof Map<?, ?> rs) {
            double factor = ((Number) rs.get("factor")).doubleValue();
            w.str(a + "rope.scaling.type", "yarn").f32(a + "rope.scaling.factor", (float) factor)
                    .u32(a + "rope.scaling.original_context_length", ((Number) rs.get("original_max_position_embeddings")).intValue())
                    .f32(a + "rope.scaling.yarn_beta_fast", ((Number) rs.get("beta_fast")).floatValue())
                    .f32(a + "rope.scaling.yarn_beta_slow", ((Number) rs.get("beta_slow")).floatValue())
                    .f32(a + "rope.scaling.attn_factor", (float) (((Number) rs.get("attention_factor")).doubleValue()
                            / (0.1 * Math.log(factor) + 1)));
        }
    }

    /** Adds a tensor with ggml dims (innermost first); with {@code q8} its values become the dequantized Q8_0 ones in {@code effective}. */
    private static void addTensor(GgufWriter w, Map<String, float[]> effective, Map<String, long[]> dimsOut, String name,
                                  long[] shape, float[] data, boolean q8) {
        long[] dims = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            dims[i] = shape[shape.length - 1 - i];
        }
        dimsOut.put(name, dims);
        if (q8) {
            byte[] blocks = q8Blocks(data);
            effective.put(name, dequantQ8(blocks));
            w.tensor(name, GgmlType.Q8_0, dims, blocks);
        } else {
            effective.put(name, data);
            w.f32Tensor(name, dims, data);
        }
    }

    // ------------------------------------------------------------------ running

    static float[][] logits(ModelWeights w, int[] prompt, int decode) {
        try (CpuBackend b = new CpuBackend(w, 64, 8, 2, KvDtype.F32, false)) {
            float[][] out = new float[1 + decode][VOCAB];
            b.forward(prompt, 0, prompt.length, 0, out[0]);
            int pos = prompt.length;
            for (int i = 0; i < decode; i++) {
                int next = argmax(out[i]);
                b.forward(new int[]{next}, 0, 1, pos++, out[i + 1]);
            }
            return out;
        }
    }

    static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[best]) {
                best = i;
            }
        }
        return best;
    }

    static double maxDiff(float[][] a, float[][] b) {
        double worst = 0;
        double mag = 1e-9;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                worst = Math.max(worst, Math.abs(a[i][j] - b[i][j]));
                mag = Math.max(mag, Math.abs(a[i][j]));
            }
        }
        return worst / mag;
    }

    public static void main(String[] args) throws Exception {
        Path dir = Files.createTempDirectory("gguf-model-test");
        try {
            for (String family : new String[]{"llama", "qwen2", "qwen3", "qwen3_moe", "qwen3-yarn", "llama-yarn"}) {
                family(dir, family);
            }
            malformed(dir);
            metadata(dir);
        } finally {
            try (var s = Files.walk(dir)) {
                s.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
        System.out.println("gguf models: " + passed + " checks passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    private static void family(Path root, String variant) throws Exception {
        Tiny t = tiny(variant, 42 + variant.hashCode());
        String family = t.family();
        Path hfDir = Files.createDirectories(root.resolve("hf-" + variant));
        Files.writeString(hfDir.resolve("config.json"), Json.write(t.config()));
        writeSafetensors(hfDir.resolve("model.safetensors"), t.hf(), t.shapes());
        Architecture hfArch = Architecture.load(hfDir);
        String arch = family.equals("qwen3_moe") ? "qwen3moe" : family;
        boolean llama = family.equals("llama");
        int[] prompt = {1, 5, 17, 33, 2, 40, 9, 77};

        Path f32 = root.resolve(variant + "-f32.gguf");
        Path q8 = root.resolve(variant + "-q8.gguf");
        Map<String, float[]> effF32 = new LinkedHashMap<>();
        Map<String, float[]> effQ8 = new LinkedHashMap<>();
        Map<String, long[]> dims = new LinkedHashMap<>();
        writeGguf(f32, t, false, llama, arch, effF32, dims);
        writeGguf(q8, t, true, llama, arch, effQ8, new LinkedHashMap<>());

        try (ModelWeights hf = ModelWeights.load(hfDir, hfArch, QType.Q8_0, ShardProvider.local(hfDir))) {
            float[][] ref = logits(hf, prompt, 4);
            try (GgufModel g = GgufModel.open(f32)) {
                String d = g.architecture().core().description();
                String want = hfArch.core().description();
                check(llama ? d.startsWith(want) && d.contains("rope=interleaved/" + HD) : d.equals(want),
                        family + ": description '" + d + "' vs '" + want + "'");
                check(g.architecture().core().modelType().equals(hfArch.core().modelType()), family + ": model type");
                if (t.config().containsKey("rope_scaling")) {
                    // the GGUF metadata must give the scaling the config gives: the same frequencies and amplitude
                    var gs = ((org.robomiri.jllm.model.Decoder) g.architecture()).ropeTables().getFirst().scaling();
                    var hs = ((org.robomiri.jllm.model.Decoder) hfArch).ropeTables().getFirst().scaling();
                    double[] gf = org.robomiri.jllm.model.Rope.invFreq(HD, 500000, gs, 256);
                    double[] hfF = org.robomiri.jllm.model.Rope.invFreq(HD, 500000, hs, 256);
                    double ga = org.robomiri.jllm.model.Rope.amplitude(gs);
                    double ha = org.robomiri.jllm.model.Rope.amplitude(hs);
                    check(java.util.Arrays.equals(gf, hfF) && Math.abs(ga - ha) < 1e-6 && Math.abs(ha - 1.2) < 1e-12
                            && d.contains("rope=yarnx4.0"), variant + ": GGUF yarn " + gs + " vs config " + hs);
                    check(gf[0] == 1.0 && gf[HD / 2 - 1] == 1.0 / (4 * Math.pow(500000, (HD - 2.0) / HD)),
                            variant + ": fastest pair kept, slowest pair interpolated: " + java.util.Arrays.toString(gf));
                }
                var pairing = ((org.robomiri.jllm.model.Decoder) g.architecture()).ropeTables().getFirst().pairing();
                check(pairing == (llama ? org.robomiri.jllm.model.RopeTable.Pairing.INTERLEAVED
                        : org.robomiri.jllm.model.RopeTable.Pairing.HALF), family + ": rope pairing " + pairing);
                try (ModelWeights w = g.weights()) {
                    double diff = maxDiff(ref, logits(w, prompt, 4));
                    check(diff < 2e-5, family + ": GGUF F32 (requantized to Q8_0) vs converted safetensors: max logit diff " + diff);
                    System.out.printf("%-9s F32 gguf vs safetensors Q8_0: max logit diff %.2e (relative to max |logit|)%n", family, diff);
                }
            }
            // Q8_0 blocks read in place vs the F32 file holding the same dequantized values (requantized: same blocks)
            Path deq = root.resolve(variant + "-deq.gguf");
            writeDequantized(deq, t, effQ8, dims, arch);
            try (GgufModel g = GgufModel.open(q8); GgufModel gd = GgufModel.open(deq)) {
                float[][] a;
                float[][] b;
                try (ModelWeights w = g.weights()) {
                    a = logits(w, prompt, 4);
                }
                try (ModelWeights w = gd.weights()) {
                    b = logits(w, prompt, 4);
                }
                double diff = maxDiff(b, a);
                check(diff < 2e-5, family + ": Q8_0 blocks in place vs the same values requantized: max logit diff " + diff);
                System.out.printf("%-9s Q8_0 gguf in place vs requantized: max logit diff %.2e%n", family, diff);
                double vs = maxDiff(ref, a);
                check(vs < 0.05, family + ": Q8_0 gguf vs converted safetensors within quantization noise: " + vs);
                System.out.printf("%-9s Q8_0 gguf vs safetensors Q8_0: max logit diff %.2e (quantization noise)%n", family, vs);
            }
            if (llama) {
                // negative control: the permuted q/k rows with the half-pairing RoPE of the Hugging Face layout are wrong
                org.robomiri.jllm.gguf.GgufFile gf = org.robomiri.jllm.gguf.GgufFile.open(f32);
                org.robomiri.jllm.gguf.GgufWeights src = new org.robomiri.jllm.gguf.GgufWeights(gf);
                try (ModelWeights w = ModelWeights.of(hfArch, QType.Q8_0, src, () -> {
                    src.close();
                    gf.close();
                }, gf.byteSize())) {
                    double diff = maxDiff(ref, logits(w, prompt, 4));
                    check(diff > 0.05, "llama: without the interleaved RoPE the permuted rows give different logits: " + diff);
                    System.out.printf("llama     permuted rows with half-pairing RoPE (wrong): max logit diff %.2e%n", diff);
                }
            }
        }
    }

    /** A GGUF with F32 tensors holding {@code effective} (llama permutation already applied in it). */
    private static void writeDequantized(Path p, Tiny t, Map<String, float[]> effective, Map<String, long[]> dims,
                                         String arch) throws IOException {
        GgufWriter w = new GgufWriter().str("general.architecture", arch);
        String a = arch + ".";
        w.u32(a + "block_count", LAYERS).u32(a + "embedding_length", DIM).u32(a + "feed_forward_length", FF)
                .u32(a + "attention.head_count", HEADS).u32(a + "attention.head_count_kv", KV)
                .u32(a + "context_length", 256).f32(a + "rope.freq_base", 500000f)
                .f32(a + "attention.layer_norm_rms_epsilon", 1e-5f).u32(a + "attention.key_length", HD)
                .u32(a + "rope.dimension_count", HD).u32("tokenizer.ggml.bos_token_id", 1).u32("tokenizer.ggml.eos_token_id", 2);
        yarnMeta(w, a, t);
        if (t.family().equals("qwen3_moe")) {
            w.u32(a + "expert_count", EXPERTS).u32(a + "expert_used_count", TOPK).u32(a + "expert_feed_forward_length", MOE_DIM);
        }
        dims.forEach((name, shape) -> w.f32Tensor(name, shape, effective.get(name)));
        w.write(p);
    }

    private static void malformed(Path root) throws Exception {
        // unsupported architecture and a missing tensor give clear IOExceptions
        Path p = root.resolve("olmo2.gguf");
        new GgufWriter().str("general.architecture", "olmo2").write(p);
        expectFail(p, "unsupported GGUF architecture 'olmo2'");
        p = root.resolve("nemotron-moe.gguf");
        new GgufWriter().str("general.architecture", "nemotron_h_moe").write(p);
        expectFail(p, "mixture-of-experts layers are not implemented");
        // the hybrids: Jamba's attention layers must be every n-th layer, Granite 4's experts need the shared expert
        p = root.resolve("jamba-irregular.gguf");
        new GgufWriter().str("general.architecture", "jamba").u32("jamba.block_count", 6).u32("jamba.embedding_length", 64)
                .u32("jamba.attention.head_count", 4).i32s("jamba.attention.head_count_kv", 0, 2, 2, 0, 0, 2)
                .f32Tensor("token_embd.weight", new long[]{64, 96}, new float[64 * 96]).write(p);
        expectFail(p, "are not every n-th layer");
        p = root.resolve("granite-noshared.gguf");
        new GgufWriter().str("general.architecture", "granitehybrid").u32("granitehybrid.block_count", 2)
                .u32("granitehybrid.embedding_length", 64).u32("granitehybrid.attention.head_count", 4)
                .i32s("granitehybrid.attention.head_count_kv", 0, 2).u32("granitehybrid.feed_forward_length", 32)
                .u32("granitehybrid.expert_count", 4).u32("granitehybrid.expert_used_count", 2)
                .f32Tensor("token_embd.weight", new long[]{64, 96}, new float[64 * 96]).write(p);
        expectFail(p, "without a shared expert");
        p = root.resolve("notensors.gguf");
        new GgufWriter().str("general.architecture", "llama").u32("llama.block_count", 1).u32("llama.embedding_length", 64)
                .u32("llama.attention.head_count", 4).u32("llama.feed_forward_length", 96)
                .f32Tensor("token_embd.weight", new long[]{64, 96}, new float[64 * 96]).write(p);
        try (GgufModel g = GgufModel.open(p)) {
            g.weights().close();
            check(false, "missing layer tensors: expected an IOException");
        } catch (IOException e) {
            check(e.getMessage().contains("blk.0."), "missing layer tensors: " + e.getMessage());
        }
    }

    /** A llama GGUF of 8 heads x 64 dims, 1 layer, only the embedding as a tensor (enough to resolve the architecture). */
    private static GgufWriter llamaMeta(String... name) {
        GgufWriter w = new GgufWriter().str("general.architecture", "llama");
        if (name.length > 0) {
            w.str("general.name", name[0]);
        }
        return w.u32("llama.block_count", 1).u32("llama.embedding_length", 512).u32("llama.feed_forward_length", 1024)
                .u32("llama.attention.head_count", 8).u32("llama.attention.head_count_kv", 2).u32("llama.context_length", 4096)
                .f32("llama.rope.freq_base", 500000f).f32("llama.attention.layer_norm_rms_epsilon", 1e-5f)
                .f32Tensor("token_embd.weight", new long[]{512, 96}, new float[512 * 96]);
    }

    private static org.robomiri.jllm.model.Decoder decoder(Path root, String file, GgufWriter w) throws IOException {
        Path p = root.resolve(file);
        w.write(p);
        try (GgufModel g = GgufModel.open(p)) {
            return (org.robomiri.jllm.model.Decoder) g.architecture();
        }
    }

    /** The metadata -> Architecture mapping beyond the families above: rope scaling, mistral, partial rotary, rejections. */
    private static void metadata(Path root) throws Exception {
        // Llama 3.x: llama.cpp stores the scaled frequencies as rope_freqs.weight, computed by the converter's formula
        int hd = 64;
        double base = 500000;
        float[] freqs = new float[hd / 2];
        for (int i = 0; i < freqs.length; i++) {
            double f = 1.0 / Math.pow(base, (2.0 * i) / hd);
            double wavelen = 2 * Math.PI / f;
            double low = 8192.0 / 1.0;
            double high = 8192.0 / 4.0;
            if (wavelen < high) {
                freqs[i] = 1;
            } else if (wavelen > low) {
                freqs[i] = 8;
            } else {
                double smooth = (8192.0 / wavelen - 1.0) / (4.0 - 1.0);
                freqs[i] = (float) (1 / ((1 - smooth) / 8 + smooth));
            }
        }
        var d = decoder(root, "l3.gguf", llamaMeta().f32Tensor("rope_freqs.weight", new long[]{hd / 2}, freqs));
        var sc = d.ropeTables().getFirst().scaling();
        check(sc instanceof org.robomiri.jllm.model.Scaling.Llama3 l && l.factor() == 8 && l.lowFreqFactor() == 1
                && l.highFreqFactor() == 4 && l.originalMaxPositions() == 8192, "rope_freqs.weight of Llama 3.1 is recognised: " + sc);
        check(d.core().description().contains("rope=llama3x8.0") && d.core().description().contains("rope=interleaved/64"),
                "llama3 description: " + d.core().description());
        check(((org.robomiri.jllm.model.Attention.Gqa) d.attention()).heads() == 8 && ((org.robomiri.jllm.model.Attention.Gqa) d.attention()).kvHeads() == 2
                && ((org.robomiri.jllm.model.Attention.Gqa) d.attention()).headDim() == 64,
                "head dim defaults to dim / heads");
        float[] other = freqs.clone();
        other[3] *= 1.5f;
        var dl = decoder(root, "lr.gguf", llamaMeta().f32Tensor("rope_freqs.weight", new long[]{hd / 2}, other));
        sc = dl.ropeTables().getFirst().scaling();
        check(sc instanceof org.robomiri.jllm.model.Scaling.LongRope, "another rope_freqs table becomes a longrope table: " + sc.getClass().getSimpleName());
        double[] inv = org.robomiri.jllm.model.Rope.invFreq(dl.ropeTables().getFirst(), 4096);
        double want = 1.0 / (other[3] * Math.pow(base, 6.0 / hd));
        check(Math.abs(inv[3] - want) < 1e-12 * want && org.robomiri.jllm.model.Rope.amplitude(sc) == 1.0,
                "the table divides the frequencies, as ggml's freq_factors do, with no amplitude change");
        sc = decoder(root, "lin.gguf", llamaMeta().str("llama.rope.scaling.type", "linear").f32("llama.rope.scaling.factor", 4f))
                .ropeTables().getFirst().scaling();
        check(sc instanceof org.robomiri.jllm.model.Scaling.Linear l && l.factor() == 4, "linear rope scaling: " + sc);
        check(decoder(root, "mis.gguf", llamaMeta("Mistral-7B-Instruct")).core().modelType().equals("mistral"), "mistral by name");
        d = decoder(root, "sw.gguf", llamaMeta().u32("llama.attention.sliding_window", 4096));
        check(d.core().modelType().equals("mistral") && d.core().description().contains("window=4096"), "sliding window: " + d.core().description());
        d = decoder(root, "rot.gguf", llamaMeta().u32("llama.rope.dimension_count", 32).u32("llama.attention.key_length", 64));
        check(d.core().description().contains("rope=interleaved/32"), "partial rotary: " + d.core().description());
        check(d.core().vocabSize() == 96 && d.core().tiedEmbeddings(), "vocab from token_embd rows, tied without output.weight");
        for (String bad : new String[]{"unknown-scaling"}) {
            Path p = root.resolve("bad.gguf");
            llamaMeta().str("llama.rope.scaling.type", bad).write(p);
            expectFail(p, "unsupported rope scaling type '" + bad + "'");
        }
        Path p = root.resolve("perlayer.gguf");
        new GgufWriter().str("general.architecture", "llama").u32("llama.block_count", 2).u32("llama.embedding_length", 512)
                .i32s("llama.attention.head_count", 8, 8).f32Tensor("token_embd.weight", new long[]{512, 96}, new float[512 * 96]).write(p);
        expectFail(p, "per-layer array");
        p = root.resolve("noembd.gguf");
        new GgufWriter().str("general.architecture", "llama").u32("llama.block_count", 1).write(p);
        expectFail(p, "no llama.embedding_length");
        p = root.resolve("noarch.gguf");
        new GgufWriter().u32("x", 1).write(p);
        expectFail(p, "general.architecture");
    }

    private static void expectFail(Path p, String contains) {
        try (GgufModel g = GgufModel.open(p)) {
            check(false, contains + ": expected an IOException");
        } catch (IOException e) {
            check(e.getMessage().contains(contains), "message '" + e.getMessage() + "' should contain '" + contains + "'");
        }
    }
}
