package org.robomiri.jllm.gguf;

import org.robomiri.jllm.gguf.GgufArchitecture.Family;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The metadata mapping of the architectures beyond the Llama / Qwen ones of {@link GgufArchitecture}: for each,
 * the Hugging Face {@code config.json} keys that llama.cpp's converter ({@code convert_hf_to_gguf.py},
 * {@code conversion/*.py}) wrote the GGUF keys from, read back, with the defaults llama.cpp's loader
 * ({@code src/models/*.cpp}) applies where a key is absent. One method per family; see docs/16-gguf.md (16.12).
 */
final class GgufFamilies {

    private GgufFamilies() {
    }

    // ------------------------------------------------------------------ Gemma 2 / Gemma 3

    /**
     * {@code gemma2} / {@code gemma3}: GeLU-tanh, sandwich norms (the file's norms already hold {@code 1 + w}, as the
     * converter adds the 1), the embedding scaled by {@code sqrt(dim)} in float (llama.cpp's {@code ggml_scale}), the
     * attention scale {@code 1/sqrt(head_dim)} (llama.cpp ignores {@code query_pre_attn_scalar}; only the 27B models,
     * 46 / 62 layers, use {@code 1/sqrt(dim/heads)}), the local layers from {@code attention.sliding_window} and its
     * pattern (default every 2nd layer for Gemma 2, 5 of 6 for Gemma 3; or a per-layer bool array), the local rope
     * base {@code rope.freq_base_swa}, and the optional soft-caps.
     */
    static void gemma(GgufFile g, Family family, String p, Map<String, Object> c, long dim, long heads, long headDim, long layers) {
        boolean g3 = family == Family.GEMMA3;
        boolean g4 = family.gemma4();
        c.put("hidden_act", "gelu_pytorch_tanh");
        c.put("hidden_activation", "gelu_pytorch_tanh");
        c.put("torch_dtype", "float32"); // llama.cpp scales by sqrtf(n_embd) with no rounding to the checkpoint's dtype
        c.put("query_pre_attn_scalar", (double) (layers == (g3 ? 62 : 46) ? dim / heads : headDim));
        long window = g.num(p + "attention.sliding_window", 0);
        List<Object> types = new ArrayList<>();
        Object pattern = g.get(p + "attention.sliding_window_pattern");
        long n = g.num(p + "attention.sliding_window_pattern", g3 || g4 ? 6 : 2);
        for (int l = 0; l < layers; l++) {
            boolean local;
            if (window <= 0) {
                local = false; // no window: every layer is a global one with the global rope base
            } else if (pattern instanceof boolean[] b) {
                local = l < b.length && b[l];
            } else {
                local = n == 0 || (l % n) < (n - 1);
            }
            types.add(local ? "sliding_attention" : "full_attention");
        }
        if (window > 0) {
            c.put("sliding_window", window);
        }
        c.put("layer_types", types);
        if (g3) {
            c.put("rope_local_base_freq", g.dbl(p + "rope.freq_base_swa", 10000.0));
        }
        if (g4) {
            // Gemma 4: the full layers use proportional RoPE over the whole (wide) head; the sliding layers a plain
            // table at the local base. llama.cpp writes the proportional part as a rope_freqs table whose factors
            // beyond the rotated pairs are 1e30 (a zero frequency); its finite prefix gives partial_rotary_factor.
            double theta = g.dbl(p + "rope.freq_base", 1_000_000.0);
            double localTheta = g.dbl(p + "rope.freq_base_swa", 10_000.0);
            long fullHeadDim = g.num(p + "attention.key_length", headDim);
            c.put("global_head_dim", fullHeadDim);
            c.putIfAbsent("num_global_key_value_heads", 1L);
            double partial = proportionalFactor(g, fullHeadDim);
            Map<String, Object> full = new LinkedHashMap<>();
            full.put("rope_type", "proportional");
            full.put("rope_theta", theta);
            full.put("partial_rotary_factor", partial);
            Map<String, Object> local = new LinkedHashMap<>();
            local.put("rope_type", "default");
            local.put("rope_theta", localTheta);
            Map<String, Object> ropes = new LinkedHashMap<>();
            ropes.put("full_attention", full);
            ropes.put("sliding_attention", local);
            c.put("rope_parameters", ropes);
            // K = V when the full-attention layers have no attn_v tensor (the 12B); keep the projection otherwise
            boolean kEqV = true;
            for (int l = 0; l < types.size(); l++) {
                if (!"sliding_attention".equals(types.get(l)) && g.hasTensor("blk." + l + ".attn_v.weight")) {
                    kEqV = false;
                    break;
                }
            }
            c.put("attention_k_eq_v", kEqV);
        }
        for (String k : new String[]{"attn_logit_softcapping", "final_logit_softcapping"}) {
            if (g.get(p + k) instanceof Number s) {
                c.put(k, s.doubleValue());
            } else if (g3) {
                c.put(k, null); // llama.cpp's loader switches Gemma 3's caps off when the keys are absent
            }
        }
    }

    /**
     * The rotated fraction of a Gemma 4 full-attention head from its {@code rope_freqs.weight}: the factors before the
     *1e30 tail (llama.cpp's zero frequency) divided by {@code headDim / 2}; 0.25 when the file has no table.
     */
    private static double proportionalFactor(GgufFile g, long headDim) {
        if (!g.hasTensor("rope_freqs.weight") || headDim <= 0) {
            return 0.25;
        }
        try {
            List<Object> f = table(g, g.tensor("rope_freqs.weight"));
            int active = 0;
            for (Object v : f) {
                if (((Number) v).doubleValue() < 1e20) {
                    active++;
                }
            }
            double p = 2.0 * active / headDim;
            return p > 0 && p <= 1 ? p : 0.25;
        } catch (IOException | RuntimeException e) {
            return 0.25; // an unusual table: the released 12B's 0.25
        }
    }

    // ------------------------------------------------------------------ Phi-3

    /** {@code phi3}: the sliding window of the file (llama.cpp ignores it, the engine applies it as transformers does). */
    static void phi3(GgufFile g, String p, Map<String, Object> c) {
        long window = g.num(p + "attention.sliding_window", 0);
        if (window > 0) {
            c.put("sliding_window", window);
        }
    }

    /**
     * LongRoPE from {@code rope_factors_long.weight} / {@code rope_factors_short.weight} (Phi-3 128k, Phi-4-mini): the two
     * per-pair divisor tables, {@code rope.scaling.original_context_length} as the switch point (the long table when the
     * context exceeds it), {@code rope.scaling.attn_factor} (1 when absent, as in llama.cpp) as the cos/sin amplitude.
     * Returns null when the file has no such tables.
     */
    static Map<String, Object> longRope(GgufFile g, String p) throws IOException {
        if (!g.hasTensor("rope_factors_long.weight") || !g.hasTensor("rope_factors_short.weight")) {
            return null;
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("rope_type", "longrope");
        out.put("long_factor", table(g, g.tensor("rope_factors_long.weight")));
        out.put("short_factor", table(g, g.tensor("rope_factors_short.weight")));
        long ctx = g.num(p + "context_length", 4096);
        long orig = g.num(p + "rope.scaling.original_context_length", ctx);
        out.put("original_max_position_embeddings", Math.max(orig, 2));
        out.put("factor", (double) ctx / Math.max(orig, 1));
        out.put("attention_factor", g.dbl(p + "rope.scaling.attn_factor", 1.0));
        return out;
    }

    static List<Object> table(GgufFile g, GgufFile.Tensor t) throws IOException {
        if (t.type() != GgmlType.F32 || t.dims().length != 1) {
            throw new IOException(t.name() + " must be an F32 vector, got " + t);
        }
        float[] row = new float[(int) t.cols()];
        MemorySegment seg = g.data(t);
        GgmlCodec.dequantizeRow(GgmlType.F32, seg, 0, row.length, row, 0);
        List<Object> l = new ArrayList<>(row.length);
        for (float v : row) {
            l.add((double) v);
        }
        return l;
    }

    // ------------------------------------------------------------------ Mamba

    /**
     * {@code mamba} (Mamba v1, not the classic-with-B/C-norm {@code falcon_mamba} variant): the state-space sizes from
     * {@code ssm.*}. The converter writes {@code head_count 0}, {@code feed_forward_length 0} and a 2^20 context.
     */
    static Map<String, Object> mamba(GgufFile g) throws IOException {
        String p = "mamba.";
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("model_type", "mamba");
        long dim = GgufArchitecture.need(g, p + "embedding_length");
        c.put("hidden_size", dim);
        c.put("num_hidden_layers", GgufArchitecture.need(g, p + "block_count"));
        GgufFile.Tensor embed = g.tensors().get("token_embd.weight");
        if (embed == null) {
            throw new IOException("GGUF has no token_embd.weight");
        }
        c.put("vocab_size", embed.rows());
        long inner = GgufArchitecture.need(g, p + "ssm.inner_size");
        c.put("intermediate_size", inner);
        c.put("expand", inner / dim);
        c.put("state_size", GgufArchitecture.need(g, p + "ssm.state_size"));
        c.put("conv_kernel", GgufArchitecture.need(g, p + "ssm.conv_kernel"));
        c.put("time_step_rank", GgufArchitecture.need(g, p + "ssm.time_step_rank"));
        c.put("layer_norm_epsilon", g.dbl(p + "attention.layer_norm_rms_epsilon", 1e-5));
        c.put("use_bias", false);
        c.put("use_conv_bias", g.hasTensor("blk.0.ssm_conv1d.bias"));
        c.put("hidden_act", "silu");
        c.put("tie_word_embeddings", !g.hasTensor("output.weight"));
        c.put("max_position_embeddings", g.num(p + "context_length", 1 << 20));
        List<Object> eos = new ArrayList<>();
        for (String k : new String[]{"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"}) {
            if (g.get(k) instanceof Number e && !eos.contains(e.longValue())) {
                eos.add(e.longValue());
            }
        }
        c.put("eos_token_id", eos);
        if (g.bool(p + "ssm.dt_b_c_rms", false)) {
            throw new IOException("Falcon-Mamba (mamba.ssm.dt_b_c_rms: RMS norm on dt, B and C) is not supported");
        }
        return c;
    }

    // ------------------------------------------------------------------ DeepSeek-V2 / V3 (deepseek2)

    /**
     * {@code deepseek2}: Multi-head Latent Attention with the sizes of {@code attention.kv_lora_rank},
     * {@code q_lora_rank}, {@code key_length_mla} / {@code value_length_mla} (the K and V widths after "decompression":
     * {@code nope + rope} and {@code v}; older files only have {@code key_length} / {@code value_length} with those
     * meanings and an unsplit {@code attn_kv_b}), the dense prefix {@code leading_dense_block_count}, routed experts
     * with the gating function (sigmoid when 2) and the group-limited selection of {@code expert_group_*}, shared
     * experts and the YaRN of the rotary slice, whose {@code yarn_log_multiplier} is {@code 0.1 * mscale_all_dim}.
     */
    static void deepseek2(GgufFile g, String p, Map<String, Object> c) throws IOException {
        c.remove("head_dim");
        c.remove("partial_rotary_factor");
        long rope = GgufArchitecture.need(g, p + "rope.dimension_count");
        long kvLora = GgufArchitecture.need(g, p + "attention.kv_lora_rank");
        long qk;
        long v;
        if (g.get(p + "attention.key_length_mla") instanceof Number k && g.get(p + "attention.value_length_mla") instanceof Number vl) {
            qk = k.longValue();
            v = vl.longValue();
        } else {
            qk = GgufArchitecture.need(g, p + "attention.key_length");
            v = GgufArchitecture.need(g, p + "attention.value_length");
        }
        if (qk <= rope) {
            throw new IOException(p + "attention.key_length_mla " + qk + " must exceed rope.dimension_count " + rope);
        }
        c.put("kv_lora_rank", kvLora);
        c.put("qk_rope_head_dim", rope);
        c.put("qk_nope_head_dim", qk - rope);
        c.put("v_head_dim", v);
        c.put("q_lora_rank", g.num(p + "attention.q_lora_rank", 0) > 0 ? (Object) g.num(p + "attention.q_lora_rank", 0) : null);
        long dense = g.num(p + "leading_dense_block_count", 0);
        c.put("first_k_dense_replace", dense);
        c.put("moe_layer_freq", 1L);
        long experts = g.num(p + "expert_count", 0);
        long layers = ((Number) c.get("num_hidden_layers")).longValue();
        if (experts > 0 && dense < layers) {
            c.put("n_routed_experts", experts);
            c.put("num_experts_per_tok", GgufArchitecture.need(g, p + "expert_used_count"));
            c.put("moe_intermediate_size", GgufArchitecture.need(g, p + "expert_feed_forward_length"));
            c.put("n_shared_experts", g.num(p + "expert_shared_count", 0));
            long gating = g.num(p + "expert_gating_func", 1);
            if (gating != 1 && gating != 2) {
                throw new IOException(p + "expert_gating_func " + gating + " is not supported (1 = softmax, 2 = sigmoid)");
            }
            c.put("scoring_func", gating == 2 ? "sigmoid" : "softmax");
            c.put("routed_scaling_factor", g.dbl(p + "expert_weights_scale", 1.0));
            c.put("norm_topk_prob", g.bool(p + "expert_weights_norm", false));
            long groups = g.num(p + "expert_group_count", 1);
            c.put("n_group", Math.max(groups, 1));
            c.put("topk_group", g.num(p + "expert_group_used_count", 1));
            boolean bias = g.hasTensor("blk." + dense + ".exp_probs_b.bias");
            // the file does not say how groups are scored: the selection bias belongs to DeepSeek-V3's noaux_tc
            // (group score = sum of its two best); without it V2's group_limited_greedy (group score = its best)
            c.put("topk_method", bias ? "noaux_tc" : groups > 1 ? "group_limited_greedy" : "greedy");
            // V3's router normalizes the selected weights and then scales them; V2's scales only when it does not normalize.
            // llama.cpp always does the first (normalize if asked, then scale), so a file asking for both is V3's rule
            boolean normThenScale = c.get("norm_topk_prob") == Boolean.TRUE && ((Number) c.get("routed_scaling_factor")).doubleValue() != 1.0;
            c.put("model_type", gating == 2 || bias || normThenScale ? "deepseek_v3" : "deepseek_v2");
        } else {
            c.put("model_type", "deepseek_v2");
            c.put("first_k_dense_replace", layers);
        }
        c.put("tie_word_embeddings", !g.hasTensor("output.weight"));
        c.put("attention_bias", false);
    }

    /**
     * DeepSeek's YaRN: llama.cpp needs {@code mscale == mscale_all_dim} (what DeepSeek-V2 / V3 ship), so the cos / sin
     * amplitude is 1 and only the softmax scale changes by {@code yarn_get_mscale(factor, mscale_all_dim)^2}.
     */
    static void deepseekYarn(GgufFile g, String p, Map<String, Object> rs) {
        double logMul = g.dbl(p + "rope.scaling.yarn_log_multiplier", 0);
        if (logMul != 0) {
            double mscaleAllDim = logMul / 0.1;
            rs.put("mscale", mscaleAllDim);
            rs.put("mscale_all_dim", mscaleAllDim);
        }
    }
}
