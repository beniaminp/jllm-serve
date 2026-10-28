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

    // ------------------------------------------------------------------ gpt-oss

    /**
     * {@code gpt-oss}: sliding-window and full attention layers alternating (llama.cpp's pattern of 2, the even layers
     * sliding, unless the file has an {@code attention.sliding_window_pattern}), a learned sink logit per head
     * ({@code attn_sinks}), biases on q / k / v / o, on the router and on the experts, the top-k-then-softmax router, YaRN.
     * The expert width is {@code expert_feed_forward_length}. The clamped SwiGLU's limit and alpha are not in the file:
     * llama.cpp's graph uses gpt-oss's 7.0 and 1.702 ({@code LLM_FFN_SWIGLU_OAI_MOE}), and so does this mapping.
     */
    static void gptOss(GgufFile g, String p, Map<String, Object> c, long layers) throws IOException {
        c.put("model_type", "gpt_oss");
        c.put("num_local_experts", GgufArchitecture.need(g, p + "expert_count"));
        c.put("num_experts_per_tok", GgufArchitecture.need(g, p + "expert_used_count"));
        c.put("intermediate_size", GgufArchitecture.need(g, p + "expert_feed_forward_length"));
        c.put("sliding_window", GgufArchitecture.need(g, p + "attention.sliding_window"));
        c.put("layer_types", windowTypes(g, p, layers, 2, "sliding_attention", "full_attention"));
        c.put("swiglu_limit", 7.0);
        c.put("swiglu_alpha", 1.702);
        if (!c.containsKey("rope_scaling")) {
            // the file says no scaling; the family's own default (YaRN x32) must not apply
            Map<String, Object> none = new LinkedHashMap<>();
            none.put("rope_type", "default");
            c.put("rope_scaling", none);
        }
    }

    /**
     * The layer types of llama.cpp's {@code load_swa_pattern(n)}: the file's {@code attention.sliding_window_pattern}
     * (a bool per layer) when it has one, else layer {@code l} uses the window when {@code l % n < n - 1}.
     */
    static List<Object> windowTypes(GgufFile g, String p, long layers, int n, String windowed, String full) {
        Object pattern = g.get(p + "attention.sliding_window_pattern");
        long every = pattern instanceof Number num ? num.longValue() : n;
        List<Object> types = new ArrayList<>();
        for (int l = 0; l < layers; l++) {
            boolean local = pattern instanceof boolean[] b ? l < b.length && b[l] : every == 0 || l % every < every - 1;
            types.add(local ? windowed : full);
        }
        return types;
    }

    // ------------------------------------------------------------------ Llama 4

    /**
     * {@code llama4} (text): llama.cpp writes none of the attention extras and hard-codes them in its loader, as
     * Scout and Maverick ship them: chunked attention of 8192 positions on three layers of four (the fourth is a
     * NoPE layer with global attention, {@code n_no_rope_layer_step} 4), the attention temperature of the NoPE layers
     * (floor 8192, scale 0.1), and the L2 norm of q and k on the rotated layers unless the model has 128 experts
     * (Maverick). A file with {@code attention.sliding_window} 0 (MobileLLM-style) has neither chunks nor NoPE layers
     * before the last. The dense width is {@code feed_forward_length}, the experts' (and the shared expert's)
     * {@code expert_feed_forward_length}; MoE layers every {@code interleave_moe_layer_step}. The q / k rows are in the
     * Hugging Face order (the converter does not permute them for llama4), which is the family's pairing already.
     */
    static void llama4(GgufFile g, String p, Map<String, Object> c, long layers) throws IOException {
        c.put("model_type", "llama4_text");
        long experts = GgufArchitecture.need(g, p + "expert_count");
        c.put("num_local_experts", experts);
        c.put("num_experts_per_tok", GgufArchitecture.need(g, p + "expert_used_count"));
        c.put("intermediate_size_mlp", c.get("intermediate_size"));
        c.put("intermediate_size", GgufArchitecture.need(g, p + "expert_feed_forward_length"));
        c.put("interleave_moe_layer_step", GgufArchitecture.need(g, p + "interleave_moe_layer_step"));
        boolean noChunks = g.get(p + "attention.sliding_window") instanceof Number w && w.longValue() == 0;
        List<Object> types = new ArrayList<>();
        if (noChunks) {
            c.put("attention_chunk_size", null);
            c.put("no_rope_layer_interval", layers);
            c.put("attn_temperature_tuning", false);
            for (int l = 0; l < layers; l++) {
                types.add("full_attention");
            }
        } else {
            c.put("attention_chunk_size", 8192L);
            c.put("no_rope_layer_interval", 4L);
            c.put("attn_temperature_tuning", true);
            c.put("floor_scale", 8192L);
            c.put("attn_scale", 0.1);
            types = windowTypes(g, p, layers, 4, "chunked_attention", "full_attention");
        }
        c.put("layer_types", types);
        c.put("use_qk_norm", experts != 128);
    }

    // ------------------------------------------------------------------ attention / state-space hybrids

    /** The keys every hybrid shares: sizes, vocabulary, tokens, and the attention heads of the attention layers. */
    private static Map<String, Object> hybridBase(GgufFile g, String p, String modelType) throws IOException {
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("model_type", modelType);
        c.put("hidden_size", GgufArchitecture.need(g, p + "embedding_length"));
        c.put("num_hidden_layers", GgufArchitecture.need(g, p + "block_count"));
        GgufFile.Tensor embed = g.tensors().get("token_embd.weight");
        if (embed == null) {
            throw new IOException("GGUF has no token_embd.weight");
        }
        c.put("vocab_size", embed.rows());
        c.put("max_position_embeddings", g.num(p + "context_length", 1 << 20));
        c.put("tie_word_embeddings", !g.hasTensor("output.weight"));
        if (g.get("tokenizer.ggml.bos_token_id") instanceof Number b) {
            c.put("bos_token_id", b.longValue());
        }
        List<Object> eos = new ArrayList<>();
        for (String k : new String[]{"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"}) {
            if (g.get(k) instanceof Number e && !eos.contains(e.longValue())) {
                eos.add(e.longValue());
            }
        }
        c.put("eos_token_id", eos);
        c.put("hidden_act", "silu");
        long[] kv = perLayer(g, p + "attention.head_count_kv", ((Number) c.get("num_hidden_layers")).intValue());
        long kvHeads = 0;
        for (long v : kv) {
            kvHeads = Math.max(kvHeads, v);
        }
        if (kvHeads > 0) {
            c.put("num_attention_heads", GgufArchitecture.scalar(g, p + "attention.head_count"));
            c.put("num_key_value_heads", kvHeads);
        }
        if (g.get(p + "attention.key_length") instanceof Number kl) {
            c.put("head_dim", kl.longValue());
        }
        c.put("attention_bias", g.hasTensor(firstAttention(kv) + "attn_q.bias"));
        return c;
    }

    /** A per-layer integer key: its array, or its one value for every layer (0 when absent). */
    static long[] perLayer(GgufFile g, String key, int layers) throws IOException {
        long[] out = new long[layers];
        Object v = g.get(key);
        if (v instanceof long[] a) {
            if (a.length < layers) {
                throw new IOException(key + " has " + a.length + " entries for " + layers + " layers");
            }
            System.arraycopy(a, 0, out, 0, layers);
        } else if (v instanceof Number n) {
            java.util.Arrays.fill(out, n.longValue());
        }
        return out;
    }

    private static String firstAttention(long[] kv) {
        for (int l = 0; l < kv.length; l++) {
            if (kv[l] > 0) {
                return "blk." + l + ".";
            }
        }
        return "blk.0.";
    }

    /** The Mamba-2 sizes of Granite 4 / Nemotron-H: {@code {d_inner, heads, head_dim, groups, state, conv_kernel}}. */
    private static long[] mamba2(GgufFile g, String p) throws IOException {
        long inner = GgufArchitecture.need(g, p + "ssm.inner_size");
        long heads = GgufArchitecture.need(g, p + "ssm.time_step_rank"); // llama.cpp keeps the Mamba-2 head count here
        if (inner % heads != 0) {
            throw new IOException(p + "ssm.inner_size " + inner + " is not a multiple of ssm.time_step_rank " + heads);
        }
        return new long[]{inner, heads, inner / heads, GgufArchitecture.need(g, p + "ssm.group_count"),
                GgufArchitecture.need(g, p + "ssm.state_size"), GgufArchitecture.need(g, p + "ssm.conv_kernel")};
    }

    /**
     * {@code granitehybrid} (Granite 4): a layer is a Mamba-2 one when its {@code attention.head_count_kv} is 0, an
     * attention one otherwise; each has the MoE of {@code expert_count} experts of {@code feed_forward_length} plus
     * the shared expert of {@code expert_shared_feed_forward_length} (or, without experts, a dense MLP of
     * {@code feed_forward_length}); the muP multipliers ({@code attention.scale}, absent meaning llama.cpp's
     * {@code 1/sqrt(head_dim)}, {@code embedding_scale}, {@code residual_scale}, {@code logit_scale}); RoPE only when
     * {@code rope.scaling.finetuned} is true (its default), on q / k rows the converter permuted. The gated norm of
     * the Mamba-2 output is per group, as llama.cpp computes it ({@link org.robomiri.jllm.model.Architecture#GGUF_GROUPED_SSM_NORM}).
     */
    static Map<String, Object> graniteHybrid(GgufFile g) throws IOException {
        String p = "granitehybrid.";
        Map<String, Object> c = hybridBase(g, p, "granitemoehybrid");
        int layers = ((Number) c.get("num_hidden_layers")).intValue();
        long dim = ((Number) c.get("hidden_size")).longValue();
        long[] kv = perLayer(g, p + "attention.head_count_kv", layers);
        List<Object> types = new ArrayList<>();
        for (long v : kv) {
            types.add(v == 0 ? "mamba" : "attention");
        }
        c.put("layer_types", types);
        c.put("rms_norm_eps", g.dbl(p + "attention.layer_norm_rms_epsilon", 1e-5));
        long experts = g.num(p + "expert_count", 0);
        long ff = GgufArchitecture.need(g, p + "feed_forward_length");
        c.put("num_local_experts", experts);
        c.put("intermediate_size", ff);
        if (experts > 0) {
            c.put("num_experts_per_tok", GgufArchitecture.need(g, p + "expert_used_count"));
            if (g.num(p + "expert_shared_feed_forward_length", 0) <= 0) {
                throw new IOException("Granite 4 without a shared expert (" + p + "expert_shared_feed_forward_length) is not supported");
            }
            c.put("shared_intermediate_size", g.num(p + "expert_shared_feed_forward_length", 0));
        } else {
            c.put("shared_intermediate_size", ff);
        }
        long heads = g.num(p + "attention.head_count", 0);
        if (c.containsKey("num_attention_heads")) {
            long headDim = c.get("head_dim") instanceof Number h ? h.longValue() : dim / Math.max(1, heads);
            c.put("attention_multiplier", g.dbl(p + "attention.scale", 0) != 0 ? g.dbl(p + "attention.scale", 0)
                    : 1.0 / Math.sqrt(headDim));
        }
        c.put("embedding_multiplier", nonZero(g.dbl(p + "embedding_scale", 0)));
        c.put("residual_multiplier", nonZero(g.dbl(p + "residual_scale", 0)));
        c.put("logits_scaling", nonZero(g.dbl(p + "logit_scale", 0)));
        boolean rope = g.bool(p + "rope.scaling.finetuned", true);
        c.put("position_embedding_type", rope ? "rope" : "nope");
        if (rope) {
            c.put("rope_theta", g.dbl(p + "rope.freq_base", 10000.0));
            c.put(org.robomiri.jllm.model.Architecture.GGUF_INTERLEAVED_ROPE, true);
        }
        if (layers > countNonZero(kv)) {
            long[] m = mamba2(g, p);
            c.put("mamba_n_heads", m[1]);
            c.put("mamba_d_head", m[2]);
            c.put("mamba_expand", m[0] / dim);
            c.put("mamba_n_groups", m[3]);
            c.put("mamba_d_state", m[4]);
            c.put("mamba_d_conv", m[5]);
            c.put(org.robomiri.jllm.model.Architecture.GGUF_GROUPED_SSM_NORM, true);
        }
        return c;
    }

    private static double nonZero(double v) {
        return v == 0 ? 1.0 : v; // llama.cpp: 0 (absent) means no scaling
    }

    private static int countNonZero(long[] a) {
        int n = 0;
        for (long v : a) {
            n += v != 0 ? 1 : 0;
        }
        return n;
    }

    /**
     * {@code nemotron_h}: per layer a Mamba-2 mixer ({@code attention.head_count_kv} and {@code feed_forward_length}
     * both 0), an attention mixer ({@code feed_forward_length} 0) or a ReLU-squared MLP of that width, written back as
     * the {@code hybrid_override_pattern} ("M", "*", "-"); one norm per layer with {@code attention.layer_norm_epsilon};
     * attention without positions. llama.cpp's scan does not clamp the step size, so {@code time_step_min} is 0 here.
     */
    static Map<String, Object> nemotronH(GgufFile g) throws IOException {
        String p = "nemotron_h.";
        Map<String, Object> c = hybridBase(g, p, "nemotron_h");
        int layers = ((Number) c.get("num_hidden_layers")).intValue();
        if (g.num(p + "expert_count", 0) > 0) {
            throw new IOException("Nemotron-H mixture-of-experts layers are not supported");
        }
        long[] kv = perLayer(g, p + "attention.head_count_kv", layers);
        long[] ff = perLayer(g, p + "feed_forward_length", layers);
        StringBuilder pattern = new StringBuilder();
        long width = 0;
        for (int l = 0; l < layers; l++) {
            pattern.append(ff[l] > 0 ? '-' : kv[l] > 0 ? '*' : 'M');
            if (ff[l] > 0) {
                if (width != 0 && width != ff[l]) {
                    throw new IOException(p + "feed_forward_length differs between MLP layers (" + width + ", " + ff[l] + ")");
                }
                width = ff[l];
            }
        }
        c.put("hybrid_override_pattern", pattern.toString());
        c.put("intermediate_size", width);
        double eps = g.dbl(p + "attention.layer_norm_epsilon", g.dbl(p + "attention.layer_norm_rms_epsilon", 1e-5));
        c.put("layer_norm_epsilon", eps);
        c.put("mamba_hidden_act", "silu");
        c.put("mlp_hidden_act", "relu2");
        if (pattern.indexOf("M") >= 0) {
            long[] m = mamba2(g, p);
            c.put("mamba_num_heads", m[1]);
            c.put("mamba_head_dim", m[2]);
            c.put("n_groups", m[3]);
            c.put("ssm_state_size", m[4]);
            c.put("conv_kernel", m[5]);
            c.put("time_step_min", 0.0);
        }
        return c;
    }

    /**
     * {@code jamba}: an attention layer where {@code attention.head_count_kv} is non-zero, a Mamba-1 one (with the
     * norms of dt, B and C) otherwise; a MoE where the layer has {@code ffn_gate_inp}, a SwiGLU MLP otherwise. The
     * family's config describes both by a period and an offset, which are recovered from the layers (and checked).
     */
    static Map<String, Object> jamba(GgufFile g) throws IOException {
        String p = "jamba.";
        Map<String, Object> c = hybridBase(g, p, "jamba");
        int layers = ((Number) c.get("num_hidden_layers")).intValue();
        long dim = ((Number) c.get("hidden_size")).longValue();
        long[] kv = perLayer(g, p + "attention.head_count_kv", layers);
        boolean[] attn = new boolean[layers];
        boolean[] moe = new boolean[layers];
        for (int l = 0; l < layers; l++) {
            attn[l] = kv[l] > 0;
            moe[l] = g.hasTensor("blk." + l + ".ffn_gate_inp.weight");
        }
        long[] a = periodic(attn, "attention");
        long[] e = periodic(moe, "expert");
        c.put("attn_layer_period", a[0]);
        c.put("attn_layer_offset", a[1]);
        c.put("expert_layer_period", e[0]);
        c.put("expert_layer_offset", e[1]);
        c.put("num_experts", g.num(p + "expert_count", 0));
        c.put("num_experts_per_tok", g.num(p + "expert_used_count", 0));
        c.put("intermediate_size", GgufArchitecture.need(g, p + "feed_forward_length"));
        c.put("rms_norm_eps", g.dbl(p + "attention.layer_norm_rms_epsilon", 1e-6));
        if (countNonZero(kv) < layers) {
            long inner = GgufArchitecture.need(g, p + "ssm.inner_size");
            if (inner % dim != 0) {
                throw new IOException(p + "ssm.inner_size " + inner + " is not a multiple of embedding_length " + dim);
            }
            c.put("mamba_expand", inner / dim);
            c.put("mamba_d_state", GgufArchitecture.need(g, p + "ssm.state_size"));
            c.put("mamba_d_conv", GgufArchitecture.need(g, p + "ssm.conv_kernel"));
            c.put("mamba_dt_rank", GgufArchitecture.need(g, p + "ssm.time_step_rank"));
        }
        return c;
    }

    /** {@code {period, offset}} with {@code mask[l] == (l % period == offset)} for every layer, or an error. */
    static long[] periodic(boolean[] mask, String what) throws IOException {
        List<Integer> on = new ArrayList<>();
        for (int l = 0; l < mask.length; l++) {
            if (mask[l]) {
                on.add(l);
            }
        }
        long period = on.size() >= 2 ? on.get(1) - on.get(0) : mask.length + 1L;
        long offset = on.isEmpty() ? mask.length : on.get(0) % period;
        for (int l = 0; l < mask.length; l++) {
            if (mask[l] != (l % period == offset)) {
                throw new IOException("the " + what + " layers of this Jamba GGUF (" + on + ") are not every n-th layer");
            }
        }
        return new long[]{period, offset};
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
