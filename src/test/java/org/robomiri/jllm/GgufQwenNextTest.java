package org.robomiri.jllm;

import org.robomiri.jllm.cpu.Qwen3NextCpuBackend;
import org.robomiri.jllm.engine.Speculator;
import org.robomiri.jllm.sampling.Sampler;
import org.robomiri.jllm.sampling.SamplingParams;
import java.util.Set;
import org.robomiri.jllm.gguf.GgmlType;
import org.robomiri.jllm.gguf.GgufModel;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.metal.Qwen3NextMetalBackend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QType;
import org.robomiri.jllm.model.Qwen3NextArch;
import org.robomiri.jllm.model.ShardProvider;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * The Gated DeltaNet hybrids as GGUF ({@code qwen3next}: Qwen3-Next, {@code qwen35}: Qwen3.5) on tiny random models,
 * against the same weights loaded from a Hugging Face checkpoint through the converter. The GGUF tensors are written
 * the way llama.cpp's convert_hf_to_gguf.py writes them (reimplemented here from {@code Qwen3NextModel} and
 * {@code _LinearAttentionVReorderBase}): {@code in_proj_qkvz} flattened to [q | k | v] and z, {@code A_log} to
 * {@code -exp}, the conv kernel squeezed, the zero-centred norms with the 1 folded in, the experts stacked, and for
 * Qwen3.5 additionally the value heads reordered from grouped to tiled (rows of v, z, b, a, the conv's v channels,
 * A, dt_bias and the <em>columns</em> of out_proj). Then:
 * <ul>
 *   <li>F32 tensors (the engine requantizes them to Q8_0 exactly like the converter): logits equal to ~1e-5, CPU;</li>
 *   <li>Q8_0 blocks read in place (the q rows are gathered into q_proj / q_gate as copies of blocks) against the F32
 *       file holding the same dequantized values: logits equal;</li>
 *   <li>CPU against Metal on the GGUF model, when Metal is available.</li>
 * </ul>
 */
public final class GgufQwenNextTest {

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

    static final int DIM = 128, HEADS = 4, KV = 2, HD = 64, LAYERS = 4, VOCAB = 96, FF = 192, NK = 2, NV = 4, DK = 32, DV = 32,
            KC = 4, EXPERTS = 4, TOPK = 2, MOE_DIM = 32, SHARED = 64, ROT = 16, INTERVAL = 2;
    static final int CONV_DIM = 2 * NK * DK + NV * DV;

    /** A tiny Hugging Face Qwen3-Next checkpoint: grouped per-key-head in_proj_qkvz / in_proj_ba, q_proj = [query | gate] per head. */
    record Tiny(boolean moe, Map<String, Object> config, Map<String, float[]> hf, Map<String, long[]> shapes) {
    }

    private static float[] rnd(Random r, int n, double std, double base) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (float) (base + r.nextGaussian() * std);
        }
        return a;
    }

    static boolean full(int l) {
        return (l + 1) % INTERVAL == 0;
    }

    static Tiny tiny(boolean moe, long seed) {
        Random r = new Random(seed);
        Map<String, Object> c = Json.obj("architectures", List.of("Qwen3NextForCausalLM"), "model_type", "qwen3_next",
                "hidden_size", (long) DIM, "intermediate_size", (long) FF, "num_hidden_layers", (long) LAYERS,
                "num_attention_heads", (long) HEADS, "num_key_value_heads", (long) KV, "head_dim", (long) HD,
                "vocab_size", (long) VOCAB, "max_position_embeddings", 512L, "rms_norm_eps", 1e-6, "rope_theta", 1000000.0,
                "partial_rotary_factor", ROT / (double) HD, "hidden_act", "silu", "attention_bias", false,
                "tie_word_embeddings", false, "full_attention_interval", (long) INTERVAL, "linear_conv_kernel_dim", (long) KC,
                "linear_key_head_dim", (long) DK, "linear_value_head_dim", (long) DV, "linear_num_key_heads", (long) NK,
                "linear_num_value_heads", (long) NV, "bos_token_id", 1L, "eos_token_id", 2L);
        if (moe) {
            c.put("num_experts", (long) EXPERTS);
            c.put("num_experts_per_tok", (long) TOPK);
            c.put("moe_intermediate_size", (long) MOE_DIM);
            c.put("shared_expert_intermediate_size", (long) SHARED);
            c.put("norm_topk_prob", true);
            c.put("decoder_sparse_step", 1L);
            c.put("mlp_only_layers", List.of(0L));
        } else {
            c.put("num_experts", 0L);
        }
        Map<String, float[]> t = new LinkedHashMap<>();
        Map<String, long[]> sh = new LinkedHashMap<>();
        java.util.function.BiConsumer<String, long[]> mat = (name, shape) -> {
            int n = 1;
            for (long d : shape) {
                n *= (int) d;
            }
            t.put(name, rnd(r, n, name.contains("in_proj_ba") || name.contains("mlp.gate.") ? 0.3 : 0.1, 0));
            sh.put(name, shape);
        };
        java.util.function.BiConsumer<String, long[]> vec = (name, shape) -> {
            t.put(name, rnd(r, (int) shape[0], 0.1, name.endsWith("linear_attn.norm.weight") ? 1.0 : 0.0));
            sh.put(name, shape);
        };
        mat.accept("model.embed_tokens.weight", new long[]{VOCAB, DIM});
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            vec.accept(p + "input_layernorm.weight", new long[]{DIM});
            vec.accept(p + "post_attention_layernorm.weight", new long[]{DIM});
            if (full(l)) {
                String a = p + "self_attn.";
                mat.accept(a + "q_proj.weight", new long[]{HEADS * HD * 2, DIM});
                mat.accept(a + "k_proj.weight", new long[]{KV * HD, DIM});
                mat.accept(a + "v_proj.weight", new long[]{KV * HD, DIM});
                mat.accept(a + "o_proj.weight", new long[]{DIM, HEADS * HD});
                vec.accept(a + "q_norm.weight", new long[]{HD});
                vec.accept(a + "k_norm.weight", new long[]{HD});
            } else {
                String a = p + "linear_attn.";
                mat.accept(a + "in_proj_qkvz.weight", new long[]{2 * NK * DK + 2 * NV * DV, DIM});
                mat.accept(a + "in_proj_ba.weight", new long[]{2 * NV, DIM});
                t.put(a + "conv1d.weight", rnd(r, CONV_DIM * KC, 0.4, 0));
                sh.put(a + "conv1d.weight", new long[]{CONV_DIM, 1, KC});
                t.put(a + "dt_bias", rnd(r, NV, 0.8, 0));
                sh.put(a + "dt_bias", new long[]{NV});
                float[] aLog = new float[NV];
                for (int i = 0; i < NV; i++) {
                    aLog[i] = (float) Math.log(0.1 + 7.9 * r.nextDouble());
                }
                t.put(a + "A_log", aLog);
                sh.put(a + "A_log", new long[]{NV});
                vec.accept(a + "norm.weight", new long[]{DV});
                mat.accept(a + "out_proj.weight", new long[]{DIM, NV * DV});
            }
            if (moe && l != 0) {
                mat.accept(p + "mlp.gate.weight", new long[]{EXPERTS, DIM});
                for (int e = 0; e < EXPERTS; e++) {
                    mlp(mat, p + "mlp.experts." + e + ".", MOE_DIM);
                }
                mlp(mat, p + "mlp.shared_expert.", SHARED);
                mat.accept(p + "mlp.shared_expert_gate.weight", new long[]{1, DIM});
            } else {
                mlp(mat, p + "mlp.", FF);
            }
        }
        vec.accept("model.norm.weight", new long[]{DIM});
        mat.accept("lm_head.weight", new long[]{VOCAB, DIM});
        return new Tiny(moe, c, t, sh);
    }

    private static void mlp(java.util.function.BiConsumer<String, long[]> mat, String p, int inter) {
        mat.accept(p + "gate_proj.weight", new long[]{inter, DIM});
        mat.accept(p + "up_proj.weight", new long[]{inter, DIM});
        mat.accept(p + "down_proj.weight", new long[]{DIM, inter});
    }

    // ------------------------------------------------------------------ llama.cpp's converter

    /** Rows {@code src[i]} of a {@code [rows, cols]} matrix. */
    static float[] rows(float[] w, int cols, int[] src) {
        float[] out = new float[src.length * cols];
        for (int i = 0; i < src.length; i++) {
            System.arraycopy(w, src[i] * cols, out, i * cols, cols);
        }
        return out;
    }

    static int[] range(int from, int n) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = from + i;
        }
        return a;
    }

    /** {@code _reorder_v_heads}: unit {@code p = i * NK + g} (tiled) takes unit {@code g * r + i} (grouped); {@code unit} elements each. */
    static int[] tile(int units, int unit, int offset) {
        int r = NV / NK;
        int[] src = new int[units * unit];
        for (int p = 0; p < units; p++) {
            int g = p % NK;
            int i = p / NK;
            for (int k = 0; k < unit; k++) {
                src[p * unit + k] = offset + (g * r + i) * unit + k;
            }
        }
        return src;
    }

    /** The tensors of the GGUF file by llama.cpp's names, as {name -> {ggml dims, floats}}; matrices are flagged for Q8_0 by name. */
    static Map<String, Object[]> convert(Tiny t, boolean qwen35) {
        Map<String, Object[]> out = new LinkedHashMap<>();
        int r = NV / NK;
        for (Map.Entry<String, float[]> e : t.hf().entrySet()) {
            String hf = e.getKey();
            float[] data = e.getValue();
            long[] shape = t.shapes().get(hf);
            java.util.regex.Matcher m = java.util.regex.Pattern.compile("model\\.layers\\.(\\d+)\\.(.*)").matcher(hf);
            if (!m.matches()) {
                String name = switch (hf) {
                    case "model.embed_tokens.weight" -> "token_embd.weight";
                    case "lm_head.weight" -> "output.weight";
                    case "model.norm.weight" -> "output_norm.weight";
                    default -> throw new IllegalArgumentException(hf);
                };
                if (hf.equals("model.norm.weight")) {
                    data = plusOne(data);
                }
                put(out, name, shape, data);
                continue;
            }
            String b = "blk." + m.group(1) + ".";
            String rest = m.group(2);
            switch (rest) {
                case "input_layernorm.weight" -> put(out, b + "attn_norm.weight", shape, plusOne(data));
                case "post_attention_layernorm.weight" -> put(out, b + "post_attention_norm.weight", shape, plusOne(data));
                case "self_attn.q_norm.weight" -> put(out, b + "attn_q_norm.weight", shape, plusOne(data));
                case "self_attn.k_norm.weight" -> put(out, b + "attn_k_norm.weight", shape, plusOne(data));
                case "self_attn.q_proj.weight" -> put(out, b + "attn_q.weight", shape, data);
                case "self_attn.k_proj.weight" -> put(out, b + "attn_k.weight", shape, data);
                case "self_attn.v_proj.weight" -> put(out, b + "attn_v.weight", shape, data);
                case "self_attn.o_proj.weight" -> put(out, b + "attn_output.weight", shape, data);
                case "linear_attn.in_proj_qkvz.weight" -> {
                    // [q k v z] per key head -> [q (all) | k | v] and z, each part in head order
                    int group = 2 * DK + 2 * r * DV;
                    int[] q = new int[NK * DK];
                    int[] k = new int[NK * DK];
                    int[] v = new int[NV * DV];
                    int[] z = new int[NV * DV];
                    for (int g = 0; g < NK; g++) {
                        for (int i = 0; i < DK; i++) {
                            q[g * DK + i] = g * group + i;
                            k[g * DK + i] = g * group + DK + i;
                        }
                        for (int i = 0; i < r * DV; i++) {
                            v[g * r * DV + i] = g * group + 2 * DK + i;
                            z[g * r * DV + i] = g * group + 2 * DK + r * DV + i;
                        }
                    }
                    float[] qkv = rows(data, DIM, concat(q, k, v));
                    float[] zz = rows(data, DIM, z);
                    if (qwen35) {
                        qkv = rows(qkv, DIM, concat(range(0, 2 * NK * DK), tile(NV, DV, 2 * NK * DK)));
                        zz = rows(zz, DIM, tile(NV, DV, 0));
                    }
                    put(out, b + "attn_qkv.weight", new long[]{CONV_DIM, DIM}, qkv);
                    put(out, b + "attn_gate.weight", new long[]{NV * DV, DIM}, zz);
                }
                case "linear_attn.in_proj_ba.weight" -> {
                    if (qwen35) { // Qwen3.5: separate beta and alpha, value heads tiled
                        int[] bb = new int[NV];
                        int[] aa = new int[NV];
                        for (int g = 0; g < NK; g++) {
                            for (int j = 0; j < r; j++) {
                                bb[g * r + j] = g * 2 * r + j;
                                aa[g * r + j] = g * 2 * r + r + j;
                            }
                        }
                        put(out, b + "ssm_beta.weight", new long[]{NV, DIM}, rows(rows(data, DIM, bb), DIM, tile(NV, 1, 0)));
                        put(out, b + "ssm_alpha.weight", new long[]{NV, DIM}, rows(rows(data, DIM, aa), DIM, tile(NV, 1, 0)));
                    } else {
                        put(out, b + "ssm_ba.weight", shape, data);
                    }
                }
                case "linear_attn.conv1d.weight" -> {
                    float[] c = data; // [convDim, 1, K] -> [convDim, K]
                    if (qwen35) {
                        c = rows(c, KC, concat(range(0, 2 * NK * DK), tile(NV, DV, 2 * NK * DK)));
                    }
                    put(out, b + "ssm_conv1d.weight", new long[]{CONV_DIM, KC}, c);
                }
                case "linear_attn.A_log" -> {
                    float[] a = new float[data.length];
                    for (int i = 0; i < a.length; i++) {
                        a[i] = (float) -Math.exp(data[i]);
                    }
                    put(out, b + "ssm_a", shape, qwen35 ? rows(a, 1, tile(NV, 1, 0)) : a);
                }
                case "linear_attn.dt_bias" -> put(out, b + "ssm_dt.bias", shape, qwen35 ? rows(data, 1, tile(NV, 1, 0)) : data);
                case "linear_attn.norm.weight" -> put(out, b + "ssm_norm.weight", shape, data); // not zero-centred: no + 1
                case "linear_attn.out_proj.weight" -> {
                    float[] o = data; // [DIM, NV * DV]: columns in head blocks
                    if (qwen35) {
                        int[] cols = tile(NV, DV, 0);
                        float[] n = new float[o.length];
                        for (int row = 0; row < DIM; row++) {
                            for (int c = 0; c < cols.length; c++) {
                                n[row * NV * DV + c] = o[row * NV * DV + cols[c]];
                            }
                        }
                        o = n;
                    }
                    put(out, b + "ssm_out.weight", shape, o);
                }
                case "mlp.gate_proj.weight" -> put(out, b + "ffn_gate.weight", shape, data);
                case "mlp.up_proj.weight" -> put(out, b + "ffn_up.weight", shape, data);
                case "mlp.down_proj.weight" -> put(out, b + "ffn_down.weight", shape, data);
                case "mlp.gate.weight" -> put(out, b + "ffn_gate_inp.weight", shape, data);
                case "mlp.shared_expert.gate_proj.weight" -> put(out, b + "ffn_gate_shexp.weight", shape, data);
                case "mlp.shared_expert.up_proj.weight" -> put(out, b + "ffn_up_shexp.weight", shape, data);
                case "mlp.shared_expert.down_proj.weight" -> put(out, b + "ffn_down_shexp.weight", shape, data);
                case "mlp.shared_expert_gate.weight" -> put(out, b + "ffn_gate_inp_shexp.weight", shape, data);
                default -> {
                    if (!rest.startsWith("mlp.experts.")) {
                        throw new IllegalArgumentException(hf);
                    }
                }
            }
        }
        if (qwen35) { // the multi-token-prediction block (never part of the Hugging Face tiny model): random, its own seed
            Random mr = new Random(99);
            String b = "blk." + LAYERS + ".";
            put(out, b + "nextn.eh_proj.weight", new long[]{DIM, 2 * DIM}, rnd(mr, DIM * 2 * DIM, 0.08, 0));
            put(out, b + "nextn.enorm.weight", new long[]{DIM}, rnd(mr, DIM, 0.1, 1));
            put(out, b + "nextn.hnorm.weight", new long[]{DIM}, rnd(mr, DIM, 0.1, 1));
            put(out, b + "nextn.shared_head_norm.weight", new long[]{DIM}, rnd(mr, DIM, 0.1, 1));
            put(out, b + "attn_norm.weight", new long[]{DIM}, rnd(mr, DIM, 0.1, 1));
            put(out, b + "post_attention_norm.weight", new long[]{DIM}, rnd(mr, DIM, 0.1, 1));
            put(out, b + "attn_q.weight", new long[]{HEADS * HD * 2, DIM}, rnd(mr, HEADS * HD * 2 * DIM, 0.1, 0));
            put(out, b + "attn_k.weight", new long[]{KV * HD, DIM}, rnd(mr, KV * HD * DIM, 0.1, 0));
            put(out, b + "attn_v.weight", new long[]{KV * HD, DIM}, rnd(mr, KV * HD * DIM, 0.1, 0));
            put(out, b + "attn_output.weight", new long[]{DIM, HEADS * HD}, rnd(mr, DIM * HEADS * HD, 0.1, 0));
            put(out, b + "attn_q_norm.weight", new long[]{HD}, rnd(mr, HD, 0.1, 1));
            put(out, b + "attn_k_norm.weight", new long[]{HD}, rnd(mr, HD, 0.1, 1));
            put(out, b + "ffn_gate.weight", new long[]{FF, DIM}, rnd(mr, FF * DIM, 0.1, 0));
            put(out, b + "ffn_up.weight", new long[]{FF, DIM}, rnd(mr, FF * DIM, 0.1, 0));
            put(out, b + "ffn_down.weight", new long[]{DIM, FF}, rnd(mr, DIM * FF, 0.1, 0));
        }
        if (t.moe()) {
            for (int l = 1; l < LAYERS; l++) {
                String p0 = "model.layers." + l + ".mlp.experts.";
                for (String[] part : new String[][]{{"gate_proj", "ffn_gate_exps"}, {"up_proj", "ffn_up_exps"}, {"down_proj", "ffn_down_exps"}}) {
                    boolean down = part[0].equals("down_proj");
                    int rows = down ? DIM : MOE_DIM;
                    int cols = down ? MOE_DIM : DIM;
                    float[] all = new float[EXPERTS * rows * cols];
                    for (int e = 0; e < EXPERTS; e++) {
                        System.arraycopy(t.hf().get(p0 + e + "." + part[0] + ".weight"), 0, all, e * rows * cols, rows * cols);
                    }
                    put(out, "blk." + l + "." + part[1] + ".weight", new long[]{EXPERTS, rows, cols}, all);
                }
            }
        }
        return out;
    }

    static int[] concat(int[]... parts) {
        int n = 0;
        for (int[] p : parts) {
            n += p.length;
        }
        int[] out = new int[n];
        int o = 0;
        for (int[] p : parts) {
            System.arraycopy(p, 0, out, o, p.length);
            o += p.length;
        }
        return out;
    }

    static float[] plusOne(float[] w) {
        float[] o = new float[w.length];
        for (int i = 0; i < o.length; i++) {
            o[i] = w[i] + 1f;
        }
        return o;
    }

    /** Stores a tensor with its PyTorch shape; {@link #write} reverses it into ggml's dims. */
    private static void put(Map<String, Object[]> out, String name, long[] shape, float[] data) {
        out.put(name, new Object[]{shape, data});
    }

    /** Matrices that stay float in a real file: norms, vectors, the conv, the router and the small b / a / gate tensors. */
    static boolean quantizable(String name, long[] shape) {
        return (shape.length == 2 || shape.length == 3) && !name.contains("ssm_conv1d") && !name.contains("ffn_gate_inp") && !name.contains("ssm_beta")
                && !name.contains("ssm_alpha") && !name.contains("ssm_ba");
    }

    /**
     * Writes the GGUF file. {@code q8}: quantizable matrices as Q8_0 blocks, their dequantized values stored in
     * {@code effective} (so an F32 file of {@code effective} holds the same weights).
     */
    static GgufWriter meta(boolean qwen35) {
        String arch = qwen35 ? "qwen35" : "qwen3next";
        String a = arch + ".";
        GgufWriter w = new GgufWriter().str("general.architecture", arch).str("general.name", "tiny-" + arch);
        w.u32(a + "block_count", LAYERS + (qwen35 ? 1 : 0)).u32(a + "embedding_length", DIM).u32(a + "feed_forward_length", FF)
                .u32(a + "attention.head_count", HEADS).u32(a + "attention.head_count_kv", KV).u32(a + "context_length", 512)
                .f32(a + "rope.freq_base", 1000000f).f32(a + "attention.layer_norm_rms_epsilon", 1e-6f)
                .u32(a + "attention.key_length", HD).u32(a + "attention.value_length", HD).u32(a + "rope.dimension_count", ROT)
                .u32(a + "ssm.conv_kernel", KC).u32(a + "ssm.state_size", DK).u32(a + "ssm.group_count", NK)
                .u32(a + "ssm.time_step_rank", NV).u32(a + "ssm.inner_size", NV * DV).u32(a + "full_attention_interval", INTERVAL)
                .u32("tokenizer.ggml.bos_token_id", 1).u32("tokenizer.ggml.eos_token_id", 2);
        if (qwen35) {
            w.u32(a + "nextn_predict_layers", 1).i32s(a + "rope.dimension_sections", 3, 3, 2, 0);
        } else {
            w.u32(a + "expert_count", EXPERTS).u32(a + "expert_used_count", TOPK).u32(a + "expert_feed_forward_length", MOE_DIM)
                    .u32(a + "expert_shared_feed_forward_length", SHARED);
        }
        return w;
    }

    /**
     * Writes the GGUF file. {@code q8}: quantizable matrices as Q8_0 blocks; {@code effective} receives the values
     * the file holds (dequantized for those), so an F32 file of {@code effective} holds the same weights
     * ({@code useEffective}).
     */
    static void write(Path p, Tiny t, boolean qwen35, boolean q8, Map<String, float[]> effective, boolean useEffective)
            throws IOException {
        GgufWriter w = meta(qwen35);
        for (Map.Entry<String, Object[]> e : convert(t, qwen35).entrySet()) {
            String name = e.getKey();
            long[] shape = (long[]) e.getValue()[0];
            float[] data = useEffective ? effective.get(name) : (float[]) e.getValue()[1];
            long[] dims = new long[shape.length];
            for (int i = 0; i < shape.length; i++) {
                dims[i] = shape[shape.length - 1 - i];
            }
            if (q8 && quantizable(name, shape)) {
                byte[] blocks = GgufModelTest.q8Blocks(data);
                effective.put(name, GgufModelTest.dequantQ8(blocks));
                w.tensor(name, GgmlType.Q8_0, dims, blocks);
            } else {
                effective.putIfAbsent(name, data);
                w.f32Tensor(name, dims, data);
            }
        }
        w.write(p);
    }

    // ------------------------------------------------------------------ running

    interface Factory {
        Backend open(ModelWeights w);
    }

    static float[][] logits(ModelWeights w, Factory f, int[] prompt, int decode) {
        try (Backend b = f.open(w)) {
            float[][] out = new float[1 + decode][VOCAB];
            b.forward(prompt, 0, prompt.length, 0, out[0]);
            int pos = prompt.length;
            for (int i = 0; i < decode; i++) {
                int next = GgufModelTest.argmax(out[i]);
                b.forward(new int[]{next}, 0, 1, pos++, out[i + 1]);
            }
            return out;
        }
    }

    static boolean sameTokens(float[][] a, float[][] b) {
        for (int i = 0; i < a.length; i++) {
            if (GgufModelTest.argmax(a[i]) != GgufModelTest.argmax(b[i])) {
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) throws Exception {
        // the CPU backend token by token for every call: verification passes then compute what single steps compute
        System.setProperty("jllm.cpu.deterministic", "true");
        String lib = System.getProperty("jllm.metal.lib");
        boolean metal = Metal.available(lib);
        Path dir = Files.createTempDirectory("gguf-qwen-next");
        try {
            variant(dir, false, true, metal, lib);  // qwen3next: grouped value heads, fused ssm_ba, MoE with a shared expert
            variant(dir, true, false, metal, lib);  // qwen35: tiled value heads, separate beta / alpha, dense FFN, an MTP block
            malformed(dir);
        } finally {
            try (var s = Files.walk(dir)) {
                s.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
        System.out.println("gguf qwen-next: " + passed + " checks passed, " + failed + " failed" + (metal ? "" : " (Metal not available)"));
        if (failed > 0) {
            System.exit(1);
        }
    }

    private static void variant(Path root, boolean qwen35, boolean moe, boolean metal, String lib) throws Exception {
        String label = qwen35 ? "qwen35" : "qwen3next";
        Tiny t = tiny(moe, 7 + label.hashCode());
        Path hfDir = Files.createDirectories(root.resolve("hf-" + label));
        Files.writeString(hfDir.resolve("config.json"), Json.write(t.config()));
        GgufModelTest.writeSafetensors(hfDir.resolve("model.safetensors"), t.hf(), t.shapes());
        Architecture hfArch = Architecture.load(hfDir);
        int[] prompt = {1, 5, 17, 33, 2, 40, 9, 77, 12, 60, 3, 21};
        Factory cpu = w -> new Qwen3NextCpuBackend(w, 64, 16, 2, KvDtype.F32);
        Factory gpu = w -> new Qwen3NextMetalBackend(w, 64, 16, lib, KvDtype.F32);

        Path f32 = root.resolve(label + "-f32.gguf");
        Path q8 = root.resolve(label + "-q8.gguf");
        write(f32, t, qwen35, false, new LinkedHashMap<>(), false);
        Map<String, float[]> eff = new LinkedHashMap<>();
        write(q8, t, qwen35, true, eff, false);
        Path deq = root.resolve(label + "-deq.gguf");
        write(deq, t, qwen35, false, eff, true);

        try (ModelWeights hf = ModelWeights.load(hfDir, hfArch, QType.Q8_0, ShardProvider.local(hfDir))) {
            float[][] ref = logits(hf, cpu, prompt, 4);
            try (GgufModel g = GgufModel.open(f32)) {
                String d = g.architecture().core().description();
                String want = hfArch.core().description();
                check((qwen35 ? d.replace(" tiled", "") : d).equals(want), label + ": description '" + d + "' vs '" + want + "'");
                Qwen3NextArch na = (Qwen3NextArch) g.architecture();
                check(na.deltaNet().tiled() == qwen35, label + ": tiled flag " + na.deltaNet().tiled());
                check(na.core().layers() == LAYERS, label + ": the MTP block is not a layer: " + na.core().layers());
                try (ModelWeights w = g.weights()) {
                    float[][] got = logits(w, cpu, prompt, 4);
                    double diff = GgufModelTest.maxDiff(ref, got);
                    check(diff < 2e-5 && sameTokens(ref, got), label + ": GGUF F32 vs converted safetensors (CPU): max logit diff " + diff);
                    System.out.printf("%-9s F32 gguf vs safetensors, cpu: max logit diff %.2e%n", label, diff);
                }
            }
            float[][] fromDeq;
            try (GgufModel g = GgufModel.open(deq); ModelWeights w = g.weights()) {
                fromDeq = logits(w, cpu, prompt, 4);
            }
            float[][] q8cpu;
            try (GgufModel g = GgufModel.open(q8); ModelWeights w = g.weights()) {
                q8cpu = logits(w, cpu, prompt, 4);
                double diff = GgufModelTest.maxDiff(fromDeq, q8cpu);
                check(diff < 2e-5 && sameTokens(fromDeq, q8cpu), label + ": Q8_0 blocks in place vs the same values requantized: " + diff);
                System.out.printf("%-9s Q8_0 gguf in place vs requantized: max logit diff %.2e%n", label, diff);
                double vs = GgufModelTest.maxDiff(ref, q8cpu);
                check(vs < 0.1, label + ": Q8_0 gguf vs converted safetensors within quantization noise: " + vs);
                System.out.printf("%-9s Q8_0 gguf vs safetensors Q8_0: max logit diff %.2e (quantization noise)%n", label, vs);
            }
            if (qwen35) {
                mtp(q8, prompt, metal ? gpu : null);
            }
            if (metal) {
                try (GgufModel g = GgufModel.open(q8); ModelWeights w = g.weights()) {
                    float[][] gpuOut = logits(w, gpu, prompt, 4);
                    double diff = GgufModelTest.maxDiff(q8cpu, gpuOut);
                    check(diff < 2e-3 && sameTokens(q8cpu, gpuOut), label + ": Metal vs CPU on the GGUF model: " + diff);
                    System.out.printf("%-9s gguf cpu vs metal: max logit diff %.2e%n", label, diff);
                }
            }
        }
    }

    // ------------------------------------------------------------------ multi-token prediction

    /** Plain greedy decoding: {@code n} tokens after {@code prompt}. */
    static int[] plain(Backend b, int[] prompt, int n) {
        float[] logits = new float[VOCAB];
        b.forward(prompt, 0, prompt.length, 0, logits);
        int[] out = new int[n];
        int pos = prompt.length;
        for (int i = 0; i < n; i++) {
            out[i] = GgufModelTest.argmax(logits);
            if (i + 1 < n) {
                b.forward(new int[]{out[i]}, 0, 1, pos++, logits);
            }
        }
        return out;
    }

    /** Greedy decoding through the speculator, the way {@code Engine} drives it. */
    static int[] speculative(Backend b, Speculator sp, int[] prompt, int n) {
        SamplingParams greedy = new SamplingParams(0f, 0, 1f, 0f, 1f, 0f, 0f, 1, n, java.util.List.of());
        Sampler sampler = new Sampler(greedy, VOCAB, prompt);
        float[] logits = new float[VOCAB];
        b.forward(prompt, 0, prompt.length, 0, logits);
        sp.begin(prompt);
        int pos = prompt.length;
        int[] out = new int[n];
        int size = 0;
        int[] ready = null;
        int rIdx = 0;
        int rLen = 0;
        int rPre = 0;
        while (size < n) {
            boolean queued = rIdx < rLen;
            int tok = queued ? ready[rIdx] : sampler.sample(logits);
            boolean pre = queued && rIdx < rPre;
            if (queued) {
                rIdx++;
            }
            if (!pre) {
                sampler.accept(tok);
            }
            out[size++] = tok;
            sp.onToken(tok);
            if (size < n && rIdx >= rLen) {
                if (sp.step(tok, pos, b.contextLength(), n - size, sampler)) {
                    ready = sp.ready();
                    rIdx = 0;
                    rLen = sp.readyCount();
                    rPre = sp.preAccepted();
                    pos = sp.newPos();
                } else {
                    b.forward(new int[]{tok}, 0, 1, pos++, logits);
                }
            }
        }
        return out;
    }

    /** A backend whose head is an oracle: right about two drafts out of three, so the accept and the rollback paths both run. */
    private static final class Oracle implements Backend {
        final Backend inner;
        final int[] truth;       // the plain greedy continuation
        final int promptLen;
        int pos;
        int ckPos;
        int asked;

        Oracle(Backend inner, int[] truth, int promptLen) {
            this.inner = inner;
            this.truth = truth;
            this.promptLen = promptLen;
        }

        public String name() {
            return "oracle(" + inner.name() + ")";
        }

        public int contextLength() {
            return inner.contextLength();
        }

        public int maxBatch() {
            return inner.maxBatch();
        }

        public void forward(int[] t, int off, int n, int p, float[] logits) {
            inner.forward(t, off, n, p, logits);
            pos = p + n;
        }

        public void forwardAll(int[] t, int off, int n, int p, float[] out, int stride) {
            inner.forwardAll(t, off, n, p, out, stride);
            pos = p + n;
        }

        public boolean supportsPrefixReuse() {
            return false;
        }

        public boolean hasMtp() {
            return true;
        }

        public void enableMtp(boolean on) {
            inner.enableMtp(on);
        }

        public boolean mtpReady() {
            return inner.mtpReady();
        }

        public void mtpDraft(int token, float[] logits) {
            int idx = pos - promptLen; // the token is fed at pos: out[idx]; the guess is out[idx + 1]
            asked++;
            java.util.Arrays.fill(logits, -1f);
            int guess = idx + 1 < truth.length ? truth[idx + 1] : 0;
            if (asked % 3 == 0) {
                guess = (guess + 1) % VOCAB;
            }
            logits[guess] = 1f;
        }

        public boolean supportsRollback() {
            return true;
        }

        public void checkpointState() {
            inner.checkpointState();
            ckPos = pos;
        }

        public void rollbackState() {
            inner.rollbackState();
            pos = ckPos;
        }

        public void close() {
        }
    }

    /**
     * The multi-token-prediction head of a tiny qwen35 file: (1) the head's guess depends only on the history, not on how
     * the history was fed (one pass, chunks, single tokens); (2) greedy decoding through the speculator equals plain
     * decoding, with an oracle head (accept and reject paths both taken) and with the real head.
     */
    private static void mtp(Path file, int[] prompt, Factory gpu) throws Exception {
        int n = 40;
        float[][] guess = new float[3][VOCAB];
        int[][] feeds = {{prompt.length}, {5, 4, 3}, {1}};
        for (int mode = 0; mode < 3; mode++) {
            try (GgufModel g = GgufModel.open(file); ModelWeights w = g.weights();
                 Qwen3NextCpuBackend b = new Qwen3NextCpuBackend(w, 128, 16, 2, KvDtype.F32)) {
                check(w.mtp != null && b.hasMtp(), "qwen35: the multi-token-prediction head is loaded");
                b.enableMtp(true);
                int pos = 0;
                int ci = 0;
                float[] sink = new float[VOCAB];
                while (pos < prompt.length) {
                    int step = Math.min(feeds[mode][Math.min(ci++, feeds[mode].length - 1)], prompt.length - pos);
                    b.forward(prompt, pos, step, pos, sink);
                    pos += step;
                }
                check(b.mtpReady(), "qwen35: the head follows the state after the prompt (feed " + mode + ")");
                b.mtpDraft(17, guess[mode]);
            }
        }
        double d1 = GgufModelTest.maxDiff(new float[][]{guess[0]}, new float[][]{guess[1]});
        double d2 = GgufModelTest.maxDiff(new float[][]{guess[0]}, new float[][]{guess[2]});
        check(d1 < 1e-5 && d2 < 1e-5, "qwen35: the head's logits do not depend on how the prompt was chunked: " + d1 + ", " + d2);
        System.out.printf("qwen35    mtp head logits, whole prompt vs chunks of 5/4/3 vs single tokens: %.2e, %.2e%n", d1, d2);
        float[] other = new float[VOCAB];
        try (GgufModel g = GgufModel.open(file); ModelWeights w = g.weights();
             Qwen3NextCpuBackend b = new Qwen3NextCpuBackend(w, 128, 16, 2, KvDtype.F32)) {
            b.enableMtp(true);
            b.forward(prompt, 0, prompt.length, 0, new float[VOCAB]);
            b.mtpDraft(18, other);
        }
        check(GgufModelTest.maxDiff(new float[][]{guess[0]}, new float[][]{other}) > 1e-3,
                "qwen35: the head's guess depends on the token it is given");

        if (gpu != null) { // the GPU head: the same guess as the CPU's, chunked or not
            float[][] g2 = new float[2][VOCAB];
            for (int mode = 0; mode < 2; mode++) {
                try (GgufModel g = GgufModel.open(file); ModelWeights w = g.weights(); Backend b = gpu.open(w)) {
                    b.enableMtp(true);
                    int pos = 0;
                    while (pos < prompt.length) {
                        int step = Math.min(mode == 0 ? prompt.length : 4, prompt.length - pos);
                        b.forward(prompt, pos, step, pos, new float[VOCAB]);
                        pos += step;
                    }
                    check(b.mtpReady(), "qwen35 metal: the head follows the state after the prompt");
                    b.mtpDraft(17, g2[mode]);
                }
            }
            double dc = GgufModelTest.maxDiff(new float[][]{guess[0]}, new float[][]{g2[0]});
            double dk = GgufModelTest.maxDiff(new float[][]{g2[0]}, new float[][]{g2[1]});
            check(dc < 2e-3 && dk < 2e-3, "qwen35 metal: head logits vs cpu " + dc + ", chunked vs whole " + dk);
            System.out.printf("qwen35    mtp head logits, metal vs cpu: %.2e, metal chunked vs whole: %.2e%n", dc, dk);
        }
        int[] truth;
        try (GgufModel g = GgufModel.open(file); ModelWeights w = g.weights();
             Qwen3NextCpuBackend b = new Qwen3NextCpuBackend(w, 128, 16, 2, KvDtype.F32)) {
            truth = plain(b, prompt, n);
        }
        Speculator.Config cfg = new Speculator.Config(Speculator.Mode.MTP, 1, 4, 1, 0);
        try (GgufModel g = GgufModel.open(file); ModelWeights w = g.weights();
             Qwen3NextCpuBackend inner = new Qwen3NextCpuBackend(w, 128, 16, 2, KvDtype.F32)) {
            Oracle o = new Oracle(inner, truth, prompt.length);
            try (Speculator sp = new Speculator(cfg, o, null, VOCAB, VOCAB, Set.of())) {
                int[] got = speculative(o, sp, prompt, n);
                check(java.util.Arrays.equals(got, truth), "qwen35: speculative output with an oracle head equals plain output: "
                        + java.util.Arrays.toString(got) + " vs " + java.util.Arrays.toString(truth));
                check(sp.accepted() > 0 && sp.accepted() < sp.drafted(), "qwen35: the oracle run both accepted and rejected drafts: "
                        + sp.accepted() + " / " + sp.drafted());
                System.out.printf("qwen35    mtp speculation with an oracle head: %d of %d drafts accepted, output == plain%n",
                        sp.accepted(), sp.drafted());
            }
        }
        try (GgufModel g = GgufModel.open(file); ModelWeights w = g.weights();
             Qwen3NextCpuBackend b = new Qwen3NextCpuBackend(w, 128, 16, 2, KvDtype.F32)) {
            try (Speculator sp = new Speculator(cfg, b, null, VOCAB, VOCAB, Set.of())) {
                int[] got = speculative(b, sp, prompt, n);
                check(java.util.Arrays.equals(got, truth), "qwen35: speculative output with the real (random) head equals plain output: "
                        + java.util.Arrays.toString(got) + " vs " + java.util.Arrays.toString(truth));
                System.out.printf("qwen35    mtp speculation with the real head: %d of %d drafts accepted, output == plain%n",
                        sp.accepted(), sp.drafted());
            }
        }
        if (gpu != null) {
            try (GgufModel g = GgufModel.open(file); ModelWeights w = g.weights(); Backend b = gpu.open(w)) {
                int[] gtruth = plain(b, prompt, n);
                try (GgufModel g2 = GgufModel.open(file); ModelWeights w2 = g2.weights(); Backend inner = gpu.open(w2)) {
                    Oracle o = new Oracle(inner, gtruth, prompt.length);
                    try (Speculator sp = new Speculator(cfg, o, null, VOCAB, VOCAB, Set.of())) {
                        int[] got = speculative(o, sp, prompt, n);
                        check(java.util.Arrays.equals(got, gtruth), "qwen35 metal: speculative output with an oracle head equals plain output: "
                                + java.util.Arrays.toString(got) + " vs " + java.util.Arrays.toString(gtruth));
                        check(sp.accepted() > 0 && sp.accepted() < sp.drafted(), "qwen35 metal: accepted and rejected drafts: "
                                + sp.accepted() + " / " + sp.drafted());
                        System.out.printf("qwen35    metal mtp speculation with an oracle head: %d of %d drafts accepted, output == plain%n",
                                sp.accepted(), sp.drafted());
                    }
                }
                b.enableMtp(true);
                try (GgufModel g3 = GgufModel.open(file); ModelWeights w3 = g3.weights(); Backend real = gpu.open(w3);
                     Speculator sp = new Speculator(cfg, real, null, VOCAB, VOCAB, Set.of())) {
                    int[] got = speculative(real, sp, prompt, n);
                    check(java.util.Arrays.equals(got, gtruth), "qwen35 metal: speculative output with the real head equals plain output");
                    System.out.printf("qwen35    metal mtp speculation with the real head: %d of %d drafts accepted, output == plain%n",
                            sp.accepted(), sp.drafted());
                }
            }
        }
    }

    private static void malformed(Path root) throws Exception {
        // a value-head width that differs from the key-head width is refused with the reason
        Path p = root.resolve("bad.gguf");
        new GgufWriter().str("general.architecture", "qwen35").u32("qwen35.block_count", 4).u32("qwen35.embedding_length", DIM)
                .u32("qwen35.attention.head_count", HEADS).u32("qwen35.ssm.conv_kernel", KC).u32("qwen35.ssm.state_size", DK)
                .u32("qwen35.ssm.group_count", NK).u32("qwen35.ssm.time_step_rank", NV).u32("qwen35.ssm.inner_size", 1000)
                .f32Tensor("token_embd.weight", new long[]{DIM, VOCAB}, new float[DIM * VOCAB]).write(p);
        try (GgufModel g = GgufModel.open(p)) {
            check(false, "inconsistent ssm.inner_size: expected an IOException");
        } catch (IOException e) {
            check(e.getMessage().contains("ssm.inner_size"), "inconsistent ssm.inner_size: " + e.getMessage());
        }
    }
}
