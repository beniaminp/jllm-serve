package org.robomiri.jllm.model;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Architecture.Core;
import org.robomiri.jllm.model.LayerSpec.Norms;
import org.robomiri.jllm.model.Mlp.Activation;
import org.robomiri.jllm.model.RopeTable.Pairing;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntPredicate;

/**
 * The parsers of the supported {@code model_type}s: one small method per family that holds that family's
 * defaults and switches, over a shared {@link Draft} that reads the keys every decoder has (with the defaults
 * of transformers' config classes) and resolves the per-layer {@link LayerSpec}s. Adding a family that fits
 * the existing vocabulary is a new method here plus a line in {@link #SUPPORTED} (docs/14.12).
 */
final class Families {

    private Families() {
    }

    /** {@code model_type} values this engine implements, with the architecture class each corresponds to. */
    static final Map<String, String> SUPPORTED = Map.ofEntries(
            Map.entry("llama", "LlamaForCausalLM"),
            Map.entry("mistral", "MistralForCausalLM"),
            Map.entry("qwen2", "Qwen2ForCausalLM"),
            Map.entry("qwen3", "Qwen3ForCausalLM"),
            Map.entry("qwen3_moe", "Qwen3MoeForCausalLM"),
            Map.entry("glm", "GlmForCausalLM"),
            Map.entry("glm4", "Glm4ForCausalLM"),
            Map.entry("phi3", "Phi3ForCausalLM"),
            Map.entry("gemma2", "Gemma2ForCausalLM"),
            Map.entry("gemma3_text", "Gemma3ForCausalLM"),
            Map.entry("mamba", "MambaForCausalLM"),
            Map.entry("deepseek_v2", "DeepseekV2ForCausalLM"),
            Map.entry("deepseek_v3", "DeepseekV3ForCausalLM"),
            Map.entry("qwen3_next", "Qwen3NextForCausalLM"),
            Map.entry("gpt_oss", "GptOssForCausalLM"),
            Map.entry("llama4_text", "Llama4ForCausalLM"));

    /** A multimodal wrapper whose {@code text_config} is a supported decoder (vision tensors are never read). */
    private static final Map<String, String> WRAPPERS = Map.of("gemma3_text", "Gemma3ForConditionalGeneration",
            "llama4_text", "Llama4ForConditionalGeneration");

    static Architecture parse(Map<String, Object> c) {
        List<Object> archs = Json.list(c, "architectures");
        String declaredArch = archs != null && !archs.isEmpty() ? String.valueOf(archs.getFirst()) : null;
        Map<String, Object> text = Json.map(c, "text_config");
        if (text != null) {
            // Multimodal wrapper (Gemma 3 4B+): the decoder's hyper-parameters live in text_config; the
            // wrapper-level ids and dtype fill in what the text config leaves out.
            Map<String, Object> merged = new LinkedHashMap<>(text);
            for (String k : new String[]{"bos_token_id", "eos_token_id", "torch_dtype"}) {
                if (merged.get(k) == null && c.get(k) != null) {
                    merged.put(k, c.get(k));
                }
            }
            if (!(merged.get("model_type") instanceof String) || "gemma3".equals(merged.get("model_type"))) {
                merged.put("model_type", "gemma3_text");
            }
            c = merged;
        } else if ("gemma3".equals(c.get("model_type"))) {
            c = new LinkedHashMap<>(c);
            c.put("model_type", "gemma3_text");
        }
        String modelType = Json.str(c, "model_type", null);
        if (modelType == null && declaredArch != null) {
            modelType = SUPPORTED.entrySet().stream().filter(e -> e.getValue().equals(declaredArch))
                    .map(Map.Entry::getKey).findFirst().orElse(null);
        }
        if (modelType == null || !SUPPORTED.containsKey(modelType)
                || (declaredArch != null && !SUPPORTED.get(modelType).equals(declaredArch)
                && !declaredArch.equals(WRAPPERS.get(modelType)))) {
            throw new IllegalArgumentException("unsupported model: model_type=" + modelType + ", architectures="
                    + archs + "; supported: " + SUPPORTED);
        }
        String arch = declaredArch != null ? declaredArch : SUPPORTED.get(modelType);
        return switch (modelType) {
            case "mamba" -> mamba(c, arch);
            case "qwen3_next" -> Qwen3NextArch.parse(c, arch);
            case "llama" -> llama(new Draft(c, modelType, arch));
            case "mistral" -> mistral(new Draft(c, modelType, arch));
            case "qwen2" -> qwen2(new Draft(c, modelType, arch));
            case "qwen3", "qwen3_moe" -> qwen3(new Draft(c, modelType, arch));
            case "glm" -> glm(new Draft(c, modelType, arch), false);
            case "glm4" -> glm(new Draft(c, modelType, arch), true);
            case "phi3" -> phi3(new Draft(c, modelType, arch));
            case "gemma2" -> gemma(new Draft(c, modelType, arch), false);
            case "gemma3_text" -> gemma(new Draft(c, modelType, arch), true);
            case "deepseek_v2" -> deepseek(new Draft(c, modelType, arch), false);
            case "deepseek_v3" -> deepseek(new Draft(c, modelType, arch), true);
            case "gpt_oss" -> gptOss(new Draft(c, modelType, arch));
            case "llama4_text" -> llama4(new Draft(c, modelType, arch));
            default -> throw new IllegalStateException(modelType); // SUPPORTED lists exactly the cases above
        };
    }

    // ------------------------------------------------------------------ families

    /**
     * Llama 3.x: the reference decoder, every default of {@link Draft} is Llama's (rotate-half RoPE, llama3 scaling
     * read generically); {@code attention_bias} biases q, k, v and o_proj, as in transformers' LlamaAttention.
     */
    private static Decoder llama(Draft d) {
        d.bias = Draft.Bias.CONFIG_QKVO;
        return ggufRope(d).build();
    }

    /** Mistral: {@code sliding_window} applies to every layer. */
    private static Decoder mistral(Draft d) {
        d.window = Draft.Window.ALL;
        return ggufRope(d).build();
    }

    private static Draft ggufRope(Draft d) {
        if (Json.bool(d.c, Architecture.GGUF_INTERLEAVED_ROPE, false)) {
            d.pairing = Pairing.INTERLEAVED;
        }
        return d;
    }

    /** Qwen2 / 2.5: q/k/v biases (the config omits the key); the window on layers >= {@code max_window_layers} only with {@code use_sliding_window}. */
    private static Decoder qwen2(Draft d) {
        d.bias = Draft.Bias.ALWAYS;
        d.window = Draft.Window.FROM_MAX_WINDOW_LAYERS;
        return d.build();
    }

    /**
     * Qwen3 dense and MoE: per-head q/k RMSNorm; the window rule of Qwen2; no biases unless {@code attention_bias}
     * (then q, k, v and o_proj); MoE from {@code num_experts}.
     */
    private static Decoder qwen3(Draft d) {
        d.bias = Draft.Bias.CONFIG_QKVO;
        d.qkNorm = true;
        d.window = Draft.Window.FROM_MAX_WINDOW_LAYERS;
        return d.build();
    }

    /**
     * GLM (GLM-Edge, glm-4-9b-chat-hf) and GLM-4-0414: interleaved RoPE over {@code partial_rotary_factor}
     * (0.5 by default) of each head, {@code attention_bias} defaulting to true (q/k/v only), {@code rms_norm_eps}
     * 1.5625e-7; {@code glm4} normalizes the attention and MLP outputs before the residual add.
     */
    private static Decoder glm(Draft d, boolean glm4) {
        d.bias = Draft.Bias.CONFIG_DEFAULT_TRUE;
        d.pairing = Pairing.INTERLEAVED;
        d.partialRotaryDefault = 0.5;
        d.normEpsDefault = 1.5625e-7;
        if (glm4) {
            d.norms = Norms.SANDWICH_GLM4;
        }
        return d.build();
    }

    /** Phi-3 / 3.5 / 4-mini / phi-4: eps 1e-5, the window on every layer (262144 in 3.5 / 4-mini: never reached); longrope and the partial rotary are read generically. */
    private static Decoder phi3(Draft d) {
        d.normEpsDefault = 1e-5;
        d.window = Draft.Window.ALL;
        return d.build();
    }

    /**
     * Gemma 2 and Gemma 3 (text): GeLU-tanh MLP, norm gains {@code 1 + w}, four norms per layer, embedding scaled
     * by {@code bf16(sqrt(dim))}, attention scale {@code query_pre_attn_scalar^-0.5}, optional logit soft-caps
     * (Gemma 2: 50 / 30 by default; Gemma 3: off), tied head by default, and per-layer local (sliding-window)
     * / global attention; Gemma 3 adds QK-norm, a 1e6 global base and a second table (base
     * {@code rope_local_base_freq}, 1e4) for the local layers. See transformers' modeling_gemma2.py / modeling_gemma3.py.
     */
    private static Decoder gemma(Draft d, boolean g3) {
        Map<String, Object> c = d.c;
        // Gemma names the key hidden_activation (hidden_act is a stale duplicate in some configs)
        String ga = Json.str(c, "hidden_activation", Json.str(c, "hidden_act", "gelu_pytorch_tanh"));
        if (!ga.equals("gelu_pytorch_tanh") && !ga.equals("gelu_new") && !ga.equals("gelu_tanh")) {
            throw new IllegalArgumentException("unsupported hidden_activation '" + ga + "': Gemma needs gelu_pytorch_tanh");
        }
        d.gemma = true;
        d.bias = Draft.Bias.CONFIG_QKVO; // Gemma2/3Attention: q, k, v and o_proj all follow attention_bias
        d.act = Activation.GELU_TANH;
        d.norms = Norms.SANDWICH_GEMMA;
        d.normPlusOne = true;
        d.tiedDefault = true; // Gemma configs omit the key; the class default is true
        d.qkNorm = g3;
        d.thetaDefault = g3 ? 1e6 : 1e4; // Gemma3TextConfig 1e6 (local layers 1e4), Gemma2Config 1e4
        d.window = Draft.Window.PER_LAYER;
        d.local = new boolean[d.layers];
        List<Object> types = Json.list(c, "layer_types");
        if (types != null && types.size() < d.layers) {
            throw new IllegalArgumentException("config.json: layer_types has " + types.size() + " entries for "
                    + d.layers + " layers");
        }
        int pattern = (int) Json.num(c, "sliding_window_pattern", 6);
        for (int l = 0; l < d.layers; l++) {
            if (types != null) {
                Object t = types.get(l);
                if (!"sliding_attention".equals(t) && !"full_attention".equals(t)) {
                    throw new IllegalArgumentException("config.json: unsupported layer_types[" + l + "] = " + t
                            + " (sliding_attention, full_attention)");
                }
                d.local[l] = "sliding_attention".equals(t);
            } else if (g3) {
                d.local[l] = pattern > 0 && (l + 1) % pattern != 0;
            } else {
                d.local[l] = l % 2 == 0;
            }
        }
        // transformers computes the normalizer as a tensor in the checkpoint dtype: sqrt(dim) rounded to bf16/f16
        String dtype = Json.str(c, "torch_dtype", Json.str(c, "dtype", "bfloat16"));
        float embedScale = (float) Math.sqrt(d.dim);
        if (dtype.equals("bfloat16")) {
            embedScale = Float.intBitsToFloat(roundBf16(Float.floatToIntBits(embedScale)));
        } else if (dtype.equals("float16")) {
            embedScale = Float.float16ToFloat(Float.floatToFloat16(embedScale));
        }
        d.embedScale = embedScale;
        // Gemma2Config defaults the caps to 50 / 30; Gemma3TextConfig to None (off). An explicit null is off.
        d.attnSoftcap = (float) softcap(c, "attn_logit_softcapping", g3 ? 0 : 50);
        d.finalSoftcap = (float) softcap(c, "final_logit_softcapping", g3 ? 0 : 30);
        d.queryScalar = Json.dbl(c, "query_pre_attn_scalar", 256);
        d.localTheta = g3 ? Json.dbl(c, "rope_local_base_freq", Json.dbl(d.localRope, "rope_theta", 1e4)) : 0;
        return d.build();
    }

    /**
     * DeepSeek-V2 / V3: Multi-head Latent Attention, a dense prefix of {@code first_k_dense_replace} layers then
     * routed + shared experts every {@code moe_layer_freq} layers, the DeepSeek routing rule (V3: sigmoid scores,
     * {@code noaux_tc} over 8 groups keeping 4, {@code routed_scaling_factor} 2.5, {@code norm_topk_prob} on;
     * V2: softmax, greedy), YaRN on the rotary slice.
     */
    private static Decoder deepseek(Draft d, boolean v3) {
        Map<String, Object> c = d.c;
        int qLora = (int) Json.num(c, "q_lora_rank", 0);
        int kvLora = (int) Json.num(c, "kv_lora_rank", 0);
        int nope = (int) Json.num(c, "qk_nope_head_dim", 0);
        int rope = (int) Json.num(c, "qk_rope_head_dim", 0);
        int vDim = (int) Json.num(c, "v_head_dim", 0);
        if (kvLora <= 0 || nope <= 0 || rope <= 0 || rope % 2 != 0 || vDim <= 0 || qLora < 0) {
            throw new IllegalArgumentException("invalid DeepSeek MLA config: q_lora_rank=" + qLora + " kv_lora_rank="
                    + kvLora + " qk_nope_head_dim=" + nope + " qk_rope_head_dim=" + rope + " v_head_dim=" + vDim);
        }
        String scoring = Json.str(c, "scoring_func", v3 ? "sigmoid" : "softmax");
        if (!scoring.equals("softmax") && !scoring.equals("sigmoid")) {
            throw new IllegalArgumentException("unsupported scoring_func '" + scoring + "' (softmax, sigmoid)");
        }
        String method = Json.str(c, "topk_method", v3 ? "noaux_tc" : "greedy");
        Router.TopK topK = null;
        for (Router.TopK t : Router.TopK.values()) {
            if (t.key().equals(method)) {
                topK = t;
            }
        }
        if (topK == null) {
            throw new IllegalArgumentException("unsupported topk_method '" + method + "' (supported: greedy, group_limited_greedy, noaux_tc)");
        }
        int nGroup = (int) Json.num(c, "n_group", v3 ? 8 : 1);
        int topkGroup = (int) Json.num(c, "topk_group", v3 ? 4 : 1);
        int nRouted = (int) Json.num(c, "n_routed_experts", 0);
        int perTok = (int) Json.num(c, "num_experts_per_tok", 0);
        if (topK != Router.TopK.GREEDY) {
            if (nGroup <= 0 || nGroup > 64 || nRouted % nGroup != 0 || topkGroup <= 0 || topkGroup > nGroup
                    || perTok > topkGroup * (nRouted / nGroup)) {
                throw new IllegalArgumentException("invalid group routing: n_group=" + nGroup + " topk_group="
                        + topkGroup + " n_routed_experts=" + nRouted + " num_experts_per_tok=" + perTok);
            }
        }
        int moeFreq = (int) Json.num(c, "moe_layer_freq", 1);
        if (moeFreq <= 0) {
            throw new IllegalArgumentException("moe_layer_freq must be positive, got " + moeFreq);
        }
        Object shared = c.get("n_shared_experts");
        int nShared = shared instanceof Number n ? n.intValue() : 0;

        Scaling.Yarn yarn = null;
        double scale = 1.0 / Math.sqrt(nope + rope);
        Map<String, Object> rs = d.scalingMap;
        if (rs != null) {
            String type = Json.str(rs, "type", Json.str(rs, "rope_type", "default"));
            if (type.equals("yarn")) {
                double factor = Json.dbl(rs, "factor", 1.0);
                int orig = (int) Json.num(rs, "original_max_position_embeddings", 4096);
                if (factor <= 0 || orig <= 0) {
                    throw new IllegalArgumentException("invalid yarn rope_scaling: " + rs);
                }
                yarn = new Scaling.Yarn(factor, orig, Json.dbl(rs, "beta_fast", 32), Json.dbl(rs, "beta_slow", 1),
                        Json.dbl(rs, "mscale", 1), Json.dbl(rs, "mscale_all_dim", 0));
                if (yarn.mscaleAllDim() != 0) {
                    double m = Rope.yarnMscale(factor, yarn.mscaleAllDim());
                    scale *= m * m;
                }
            } else if (!type.equals("default")) {
                throw new IllegalArgumentException("unsupported DeepSeek rope_scaling type '" + type
                        + "' (supported: yarn, none)");
            }
        }
        d.mla = new Attention.Mla(d.heads, qLora, kvLora, nope, rope, vDim, 0, (float) scale);
        d.headDim = d.mla.qkDim(); // per-head query/key width; MLA has no grouped KV heads
        d.kvHeads = d.heads;
        d.yarn = yarn;
        d.router = new Router.DeepSeek(scoring.equals("sigmoid"), topK, nGroup, topkGroup,
                Json.dbl(c, "routed_scaling_factor", v3 ? 2.5 : 1.0), v3);
        d.expertsKey = "n_routed_experts";
        d.normTopKDefault = v3; // DeepseekV3Config defaults to true
        d.nShared = nShared;
        d.firstKDense = (int) Json.num(c, "first_k_dense_replace", 0);
        int first = d.firstKDense;
        d.moeLayer = l -> l >= first && l % moeFreq == 0;
        return d.build();
    }

    /**
     * gpt-oss (transformers' modeling_gpt_oss.py): alternating sliding-window ({@code sliding_window}, 128) and full
     * attention layers ({@code layer_types}, else sliding on the even layers), a learned sink logit per head in every
     * softmax, biases on q, k, v and o_proj ({@code attention_bias} defaults to true), YaRN with {@code truncate} false,
     * and in every layer a MoE of {@code num_local_experts} experts of width {@code intermediate_size} with a biased
     * router (top-k logits, then a softmax over the selected ones) and the clamped SwiGLU of
     * {@code swiglu_limit} / {@code swiglu_alpha}, with biases in the experts.
     */
    private static Decoder gptOss(Draft d) {
        Map<String, Object> c = d.c;
        d.bias = Draft.Bias.CONFIG_QKVO_DEFAULT_TRUE;
        d.thetaDefault = 150000.0;
        d.normEpsDefault = 1e-5;
        d.expertsKey = "num_local_experts";
        d.moeDimOverride = (int) Json.num(c, "intermediate_size", 0);
        d.sinks = true;
        d.window = Draft.Window.PER_LAYER;
        d.local = new boolean[d.layers];
        List<Object> types = Json.list(c, "layer_types");
        if (types != null && types.size() < d.layers) {
            throw new IllegalArgumentException("config.json: layer_types has " + types.size() + " entries for "
                    + d.layers + " layers");
        }
        for (int l = 0; l < d.layers; l++) {
            if (types == null) {
                d.local[l] = l % 2 == 0;
                continue;
            }
            Object t = types.get(l);
            if (!"sliding_attention".equals(t) && !"full_attention".equals(t)) {
                throw new IllegalArgumentException("config.json: unsupported layer_types[" + l + "] = " + t
                        + " (sliding_attention, full_attention)");
            }
            d.local[l] = "sliding_attention".equals(t);
        }
        d.slidingDefault = 128; // GptOssConfig's default
        Map<String, Object> yarn = new LinkedHashMap<>();
        yarn.put("rope_type", "yarn");
        yarn.put("factor", 32.0);
        yarn.put("beta_fast", 32.0);
        yarn.put("beta_slow", 1.0);
        yarn.put("truncate", Boolean.FALSE);
        yarn.put("original_max_position_embeddings", 4096L);
        d.scalingDefault = yarn;
        d.router = new Router.TopLogits(false, true);
        d.expertSpec = new Mlp.ExpertSpec((float) Json.dbl(c, "swiglu_limit", 7.0), (float) Json.dbl(c, "swiglu_alpha", 1.702),
                true, false, true);
        return d.build();
    }

    /**
     * Llama 4 text (Scout, Maverick; transformers' modeling_llama4.py): rotate-pairs RoPE (adjacent pairs, llama3-scaled)
     * on the layers {@code no_rope_layers} marks with 1, which also attend chunk-wise ({@code attention_chunk_size}, the
     * {@code layer_types} "chunked_attention"); the others (every 4th) are NoPE layers with global attention whose
     * queries get the position-dependent temperature of {@code attn_temperature_tuning}; the rotated layers L2-normalize
     * q and k after the rotation ({@code use_qk_norm}). Layers in {@code moe_layers} (else every
     * {@code interleave_moe_layer_step}-th) route to {@code num_local_experts} experts of width {@code intermediate_size}
     * by the sigmoid of the top-k router logits applied to the expert's input, plus one shared expert; the other
     * layers are dense with {@code intermediate_size_mlp}.
     */
    private static Decoder llama4(Draft d) {
        Map<String, Object> c = d.c;
        d.bias = Draft.Bias.CONFIG_QKVO;
        d.thetaDefault = 500000.0;
        d.normEpsDefault = 1e-5;
        d.pairing = Pairing.INTERLEAVED;
        d.expertsKey = "num_local_experts";
        d.moeDimOverride = (int) Json.num(c, "intermediate_size", 0);
        d.hiddenDim = (int) Json.num(c, "intermediate_size_mlp", 0);
        d.nShared = 1;
        d.router = new Router.TopLogits(true, false);
        d.expertSpec = new Mlp.ExpertSpec(0f, 0f, false, true, false);
        d.noPe = new boolean[d.layers];
        List<Object> noRope = Json.list(c, "no_rope_layers");
        int interval = (int) Json.num(c, "no_rope_layer_interval", 4);
        if (noRope != null && noRope.size() < d.layers) {
            throw new IllegalArgumentException("config.json: no_rope_layers has " + noRope.size() + " entries for "
                    + d.layers + " layers");
        }
        for (int l = 0; l < d.layers; l++) {
            boolean rope = noRope != null ? ((Number) noRope.get(l)).intValue() != 0 : (l + 1) % Math.max(1, interval) != 0;
            d.noPe[l] = !rope;
        }
        long chunk = c.get("attention_chunk_size") == null && c.containsKey("attention_chunk_size") ? 0
                : Json.num(c, "attention_chunk_size", 8192);
        d.chunked = new boolean[d.layers];
        List<Object> types = Json.list(c, "layer_types");
        for (int l = 0; l < d.layers; l++) {
            if (types != null && l < types.size()) {
                Object t = types.get(l);
                if (!"chunked_attention".equals(t) && !"full_attention".equals(t)) {
                    throw new IllegalArgumentException("config.json: unsupported layer_types[" + l + "] = " + t
                            + " (chunked_attention, full_attention)");
                }
                d.chunked[l] = "chunked_attention".equals(t);
            } else {
                d.chunked[l] = !d.noPe[l];
            }
        }
        d.chunkSize = (int) Math.max(0, Math.min(chunk, Integer.MAX_VALUE));
        d.l2Norm = Json.bool(c, "use_qk_norm", true);
        if (Json.bool(c, "attn_temperature_tuning", true)) {
            d.tempFloor = (int) Json.num(c, "floor_scale", 8192);
            d.tempScale = (float) Json.dbl(c, "attn_scale", 0.1);
        }
        int step = (int) Json.num(c, "interleave_moe_layer_step", 1);
        List<Object> moeLayers = Json.list(c, "moe_layers");
        Set<Integer> moeSet = new java.util.HashSet<>();
        if (moeLayers != null) {
            for (Object o : moeLayers) {
                moeSet.add(((Number) o).intValue());
            }
        } else {
            for (int l = Math.max(1, step) - 1; l < d.layers; l += Math.max(1, step)) {
                moeSet.add(l);
            }
        }
        d.moeLayer = moeSet::contains;
        return d.build();
    }

    /**
     * Mamba (v1) sizes: {@code state_size}, {@code conv_kernel}, {@code time_step_rank} (resolved from "auto" =
     * ceil(d_model/16)), {@code d_inner} ({@code intermediate_size} or {@code expand * hidden_size}) and the two
     * optional biases. The LM head is tied to the embeddings unless the config says otherwise (the released
     * checkpoints have no lm_head).
     */
    private static MambaArch mamba(Map<String, Object> c, String arch) {
        int dim = requirePositive(c, "hidden_size");
        int layers = requirePositive(c, "num_hidden_layers");
        int vocab = requirePositive(c, "vocab_size");
        int dState = (int) Json.num(c, "state_size", 16);
        int dConv = (int) Json.num(c, "conv_kernel", 4);
        int expand = (int) Json.num(c, "expand", 2);
        int dInner = (int) Json.num(c, "intermediate_size", (long) expand * dim);
        Object rank = c.get("time_step_rank");
        int dtRank = rank instanceof Number n ? n.intValue() : (dim + 15) / 16; // "auto"
        if (dState <= 0 || dState > 64 || dConv < 1 || dConv > 8 || dInner <= 0 || dtRank <= 0) {
            throw new IllegalArgumentException("invalid mamba config: state_size=" + dState + " (1..64), conv_kernel="
                    + dConv + " (1..8), d_inner=" + dInner + ", time_step_rank=" + dtRank);
        }
        String act = Json.str(c, "hidden_act", "silu");
        if (!act.equals("silu")) {
            throw new IllegalArgumentException("unsupported hidden_act '" + act + "': only silu is implemented");
        }
        if (Json.bool(c, "use_bias", false)) {
            throw new IllegalArgumentException("mamba models with in_proj/out_proj biases (use_bias=true) are not "
                    + "supported; no released checkpoint uses them");
        }
        String description = String.format("%s dim=%d dInner=%d layers=%d dState=%d dConv=%d dtRank=%d vocab=%d", arch,
                dim, dInner, layers, dState, dConv, dtRank, vocab);
        Core core = new Core(arch, "mamba", dim, layers, vocab, (float) Json.dbl(c, "layer_norm_epsilon", 1e-5),
                Json.bool(c, "tie_word_embeddings", true), -1, intList(c, "eos_token_id"),
                (int) Json.num(c, "max_position_embeddings", 1 << 20), 1f, 0f, false, description);
        return new MambaArch(core, dState, dConv, dtRank, dInner, Json.bool(c, "use_conv_bias", true));
    }

    // ------------------------------------------------------------------ the shared decoder parser

    /**
     * The decoder keys every family reads and the switches a family sets before {@link #build} reads the rest
     * with the family's defaults, validates, and resolves the layer specs and the canonical description. The
     * field defaults are Llama's.
     */
    private static final class Draft {
        /** Which layers get {@code sliding_window}. */
        enum Window {
            NONE,
            ALL,
            /** Layers {@code >= max_window_layers}, only when {@code use_sliding_window} (Qwen2 / Qwen3). */
            FROM_MAX_WINDOW_LAYERS,
            /** The layers flagged in {@link Draft#local} (Gemma). */
            PER_LAYER
        }

        /** How {@code attention_bias} is read. */
        enum Bias {
            /**
             * Rejected when true: the family's transformers class does not read it (Mistral, Phi-3), or it would
             * bias projections this engine does not model (DeepSeek's q_a / kv_a).
             */
            REJECT,
            /** q/k/v biases always present, whatever the config says (Qwen2); no o_proj bias. */
            ALWAYS,
            /** q/k/v biases from the config value, true when absent (GLM); no o_proj bias. */
            CONFIG_DEFAULT_TRUE,
            /** The config value, false when absent, biases q, k, v and o_proj (Llama, Qwen3, Gemma). */
            CONFIG_QKVO,
            /** As {@link #CONFIG_QKVO} but true when absent (gpt-oss). */
            CONFIG_QKVO_DEFAULT_TRUE
        }

        final Map<String, Object> c;
        final String modelType;
        final String architecture;
        /** transformers >= 5 nests the rope keys in rope_parameters (keyed by layer type for Gemma 3). */
        final Map<String, Object> globalRope;
        final Map<String, Object> localRope;
        /** The scaling map ({@code rope_scaling}, or the rope_parameters entry when it names a type), null for plain RoPE. */
        final Map<String, Object> scalingMap;
        final int dim;
        final int layers;
        final int vocab;
        int hiddenDim;
        final int heads;
        int kvHeads;
        int headDim;

        // family switches
        Bias bias = Bias.REJECT;
        boolean qkNorm;
        Norms norms = Norms.PRE;
        Activation act = Activation.SILU;
        Pairing pairing = Pairing.HALF;
        double partialRotaryDefault = 1.0;
        double thetaDefault = 1e4;
        double normEpsDefault = 1e-6;
        Window window = Window.NONE;
        boolean[] local;
        boolean tiedDefault;
        boolean normPlusOne;
        boolean gemma;
        float embedScale = 1f;
        float attnSoftcap;
        float finalSoftcap;
        /** Attention scale {@code queryScalar^-0.5} instead of {@code headDim^-0.5} (Gemma), NaN = headDim. */
        double queryScalar = Double.NaN;
        /** Base of a second rotary table for the {@link #local} layers (Gemma 3), 0 = one table. */
        double localTheta;
        String expertsKey = "num_experts";
        boolean normTopKDefault;
        Router router = Router.SOFTMAX;
        /** Which layers route to experts, given that the model has experts; null = the Qwen rule. */
        IntPredicate moeLayer;
        int nShared;
        int firstKDense;
        Attention.Mla mla;
        Scaling.Yarn yarn;
        /** gpt-oss / Llama 4: the learned attention sinks, the experts' arithmetic, and where the width of an expert lives. */
        boolean sinks;
        Mlp.ExpertSpec expertSpec = Mlp.ExpertSpec.PLAIN;
        int moeDimOverride;
        /** {@code sliding_window} when the config omits it (gpt-oss: 128). */
        long slidingDefault;
        /** The rope scaling a config without one gets (gpt-oss's YaRN), null = plain RoPE. */
        Map<String, Object> scalingDefault;
        /** Llama 4: chunked attention layers and their chunk, NoPE layers, L2 q/k norm, temperature tuning. */
        int chunkSize;
        boolean[] chunked;
        boolean[] noPe;
        boolean l2Norm;
        int tempFloor;
        float tempScale;

        Draft(Map<String, Object> c, String modelType, String architecture) {
            this.c = c;
            this.modelType = modelType;
            this.architecture = architecture;
            dim = requirePositive(c, "hidden_size");
            layers = requirePositive(c, "num_hidden_layers");
            vocab = requirePositive(c, "vocab_size");
            heads = requirePositive(c, "num_attention_heads");
            kvHeads = (int) Json.num(c, "num_key_value_heads", heads);
            headDim = (int) Json.num(c, "head_dim", dim / heads);
            hiddenDim = (int) Json.num(c, "intermediate_size", 0);
            // transformers >= 5 nests rope_theta / partial_rotary_factor and the scaling keys in rope_parameters
            // (keyed by layer type for Gemma 3); configs written by transformers 4 use the top-level keys and
            // rope_scaling. The legacy keys win when both are present.
            Map<String, Object> ropeParams = Json.map(c, "rope_parameters");
            Map<String, Object> fullRope = Json.map(ropeParams, "full_attention");
            globalRope = fullRope != null ? fullRope : ropeParams;
            localRope = Json.map(ropeParams, "sliding_attention");
            Map<String, Object> rs = Json.map(c, "rope_scaling");
            if (rs == null && globalRope != null && (globalRope.containsKey("rope_type") || globalRope.containsKey("type"))) {
                rs = globalRope;
            }
            scalingMap = rs;
        }

        Decoder build() {
            if (kvHeads <= 0 || heads % kvHeads != 0) {
                throw new IllegalArgumentException("num_key_value_heads=" + kvHeads + " must be positive and divide "
                        + "num_attention_heads=" + heads);
            }
            if (headDim <= 0) {
                throw new IllegalArgumentException("head_dim must be positive, got " + headDim);
            }
            if (act == Activation.SILU) {
                String a = Json.str(c, "hidden_act", "silu");
                if (!a.equals("silu")) {
                    throw new IllegalArgumentException("unsupported hidden_act '" + a + "': only silu (SwiGLU) is implemented");
                }
            }
            if (Json.bool(c, "mlp_bias", false)) {
                throw new IllegalArgumentException("models with an MLP bias (mlp_bias=true) are not supported");
            }
            boolean attnBias = switch (bias) {
                case ALWAYS -> true;
                case CONFIG_DEFAULT_TRUE -> Json.bool(c, "attention_bias", true);
                case CONFIG_QKVO -> Json.bool(c, "attention_bias", false);
                case CONFIG_QKVO_DEFAULT_TRUE -> Json.bool(c, "attention_bias", true);
                case REJECT -> {
                    if (Json.bool(c, "attention_bias", false)) {
                        throw new IllegalArgumentException("attention_bias=true is not supported for " + modelType
                                + " (supported: llama, qwen2, qwen3, glm, gemma)");
                    }
                    yield false;
                }
            };
            boolean outBias = attnBias && (bias == Bias.CONFIG_QKVO || bias == Bias.CONFIG_QKVO_DEFAULT_TRUE);
            double partialRotary = Json.dbl(c, "partial_rotary_factor",
                    Json.dbl(globalRope, "partial_rotary_factor", partialRotaryDefault));
            if (partialRotary <= 0 || partialRotary > 1 || (int) (headDim * partialRotary) < 2) {
                throw new IllegalArgumentException("invalid partial_rotary_factor " + partialRotary);
            }
            int rotaryDim = (int) (headDim * partialRotary) & ~1;
            double ropeTheta = Json.dbl(c, "rope_theta", Json.dbl(globalRope, "rope_theta", thetaDefault));
            int sw = 0;
            int windowFrom = 0;
            long swRaw = Json.num(c, "sliding_window", slidingDefault);
            if (swRaw > 0) {
                switch (window) {
                    case NONE -> {
                    }
                    case ALL, PER_LAYER -> sw = (int) Math.min(swRaw, Integer.MAX_VALUE);
                    case FROM_MAX_WINDOW_LAYERS -> {
                        if (Json.bool(c, "use_sliding_window", false)) {
                            sw = (int) Math.min(swRaw, Integer.MAX_VALUE);
                            windowFrom = (int) Json.num(c, "max_window_layers", layers);
                        }
                    }
                }
            }
            Map<String, Object> scalingMap = this.scalingMap != null ? this.scalingMap : scalingDefault;
            Scaling scaling = mla != null ? Scaling.NONE : parseRopeScaling(scalingMap, c, rotaryDim);
            List<Integer> eos = intList(c, "eos_token_id");
            Set<Integer> mlpOnlyLayers = Set.copyOf(intList(c, "mlp_only_layers"));
            double normEps = Json.dbl(c, "rms_norm_eps", normEpsDefault);
            int numExperts = (int) Json.num(c, expertsKey, 0);
            int topK = (int) Json.num(c, "num_experts_per_tok", 0);
            int moeDim = moeDimOverride > 0 ? moeDimOverride : (int) Json.num(c, "moe_intermediate_size", 0);
            if (numExperts > 0 && (topK <= 0 || topK > numExperts || moeDim <= 0)) {
                throw new IllegalArgumentException("invalid MoE config: num_experts=" + numExperts
                        + " num_experts_per_tok=" + topK + " moe_intermediate_size=" + moeDim);
            }
            boolean normTopK = Json.bool(c, "norm_topk_prob", normTopKDefault);
            int sparseStep = (int) Json.num(c, "decoder_sparse_step", 1);
            // same rule as transformers' Qwen3MoeDecoderLayer unless the family set its own
            IntPredicate rule = moeLayer != null ? moeLayer
                    : l -> !mlpOnlyLayers.contains(l) && (l + 1) % Math.max(1, sparseStep) == 0;
            boolean[] isMoe = new boolean[layers];
            boolean anyDense = false;
            for (int l = 0; l < layers; l++) {
                isMoe[l] = numExperts > 0 && rule.test(l);
                anyDense |= !isMoe[l];
            }
            if (anyDense && hiddenDim <= 0) {
                throw new IllegalArgumentException("intermediate_size must be positive for a model with dense MLP layers");
            }

            List<RopeTable> tables = new ArrayList<>(2);
            tables.add(mla != null ? new RopeTable(mla.rope(), ropeTheta, yarn != null ? yarn : Scaling.NONE, Pairing.INTERLEAVED)
                    : new RopeTable(rotaryDim, ropeTheta, scaling, pairing, mropeSections(scalingMap),
                    scalingMap != null && Json.bool(scalingMap, "mrope_interleaved", false)));
            boolean anyLocal = false;
            if (localTheta > 0 && local != null) {
                for (boolean b : local) {
                    anyLocal |= b;
                }
            }
            if (anyLocal) {
                tables.add(new RopeTable(rotaryDim, localTheta, Scaling.NONE, pairing));
            }
            int noPeTable = -1;
            for (int l = 0; noPe != null && l < layers; l++) {
                if (noPe[l] && noPeTable < 0) {
                    noPeTable = tables.size(); // an identity table: the NoPE layers go through the same rotary kernels
                    tables.add(new RopeTable(rotaryDim, ropeTheta, Scaling.IDENTITY, pairing));
                }
            }
            float scale = (float) (1.0 / Math.sqrt(Double.isNaN(queryScalar) ? headDim : queryScalar));
            Mlp.Dense shared = nShared > 0 ? new Mlp.Dense(nShared * moeDim, Activation.SILU) : null;
            List<LayerSpec> specs = new ArrayList<>(layers);
            for (int l = 0; l < layers; l++) {
                boolean windowed = sw > 0 && (window == Window.PER_LAYER ? local[l] : l >= windowFrom);
                boolean nope = noPe != null && noPe[l];
                Attention attn = mla != null ? mla
                        : new Attention.Gqa(heads, kvHeads, headDim, windowed ? sw : 0,
                        nope ? noPeTable : anyLocal && local[l] ? 1 : 0, qkNorm, attnBias, outBias, scale, attnSoftcap,
                        sinks, chunked != null && chunked[l] ? chunkSize : 0, l2Norm && !nope,
                        nope ? tempFloor : 0, nope ? tempScale : 0f);
                Mlp mlp = isMoe[l] ? new Mlp.Moe(numExperts, topK, moeDim, normTopK, router, shared, expertSpec)
                        : new Mlp.Dense(hiddenDim, act);
                specs.add(new LayerSpec(attn, mlp, norms));
            }

            // The canonical description. Features absent from the original Qwen3-only engine are appended only
            // when set, so the description (also stored in the quantized cache) of an existing model never changes.
            String moe = numExperts > 0 ? String.format(" experts=%d/%d moeDim=%d", topK, numExperts, moeDim) : "";
            String mlaS = "";
            if (mla != null) {
                Router.DeepSeek r = (Router.DeepSeek) router;
                mlaS = String.format(" mla(q=%d kv=%d nope=%d rope=%d v=%d) shared=%d dense=%d %s", mla.qLoraRank(),
                        mla.kvLoraRank(), mla.nope(), mla.rope(), mla.vDim(), nShared, firstKDense, r.topK().key())
                        + (yarn != null ? " yarnx" + yarn.factor() : "");
            }
            String extra = mlaS + (attnBias ? " bias" : "") + (outBias ? " obias" : "") + (sw > 0 ? " window=" + sw
                    + (windowFrom > 0 ? "@" + windowFrom : "") : "")
                    + (!(scaling instanceof Scaling.None) ? " rope=" + scalingKey(scaling) + "x" + scalingFactor(scaling) : "")
                    + (pairing == Pairing.INTERLEAVED ? " rope=interleaved/" + rotaryDim : "")
                    + (pairing != Pairing.INTERLEAVED && rotaryDim != headDim ? " rotary=" + rotaryDim : "")
                    + (norms.post() ? " postnorms" : "")
                    + (gemma ? " gemma(softcap=" + attnSoftcap + "/" + finalSoftcap + " qscalar=" + queryScalar
                    + " localRope=" + localTheta + " layers=" + localLayerString() + ")" : "")
                    + (sinks ? " sinks layers=" + localLayerString() : "")
                    + (expertSpec.special() ? String.format(" experts(%s%s%s)", expertSpec.clamped()
                    ? "clamp" + expertSpec.clampLimit() + "/" + expertSpec.alpha() + " " : "",
                    expertSpec.bias() ? "bias " : "", expertSpec.weightInput() ? "inputweight" : "") : "")
                    + (router instanceof Router.TopLogits t ? " router=toplogits" + (t.sigmoid() ? "/sigmoid" : "/softmax")
                    + (t.bias() ? "/bias" : "") : "")
                    + (chunked != null ? " chunk=" + chunkSize + " layers=" + layerString(chunked) + " nope="
                    + layerString(noPe) + (l2Norm ? " l2" : "") + (tempFloor > 0 ? " temp=" + tempFloor + "/" + tempScale : "") : "")
                    + (scaling instanceof Scaling.Yarn y && !y.truncate() ? " yarn-notrunc" : "");
            String description = String.format("%s dim=%d ffn=%d layers=%d heads=%d/%d headDim=%d vocab=%d theta=%.0f%s%s",
                    architecture, dim, hiddenDim, layers, heads, kvHeads, headDim, vocab, ropeTheta, moe, extra);

            Core core = new Core(architecture, modelType, dim, layers, vocab, (float) normEps,
                    Json.bool(c, "tie_word_embeddings", tiedDefault), (int) Json.num(c, "bos_token_id", -1), eos,
                    (int) Json.num(c, "max_position_embeddings", 4096), embedScale, finalSoftcap, normPlusOne, description);
            return new Decoder(core, specs, tables);
        }

        private String localLayerString() {
            return layerString(local);
        }

        private static String layerString(boolean[] flags) {
            StringBuilder sb = new StringBuilder();
            for (boolean b : flags) {
                sb.append(b ? 'L' : 'G');
            }
            return sb.toString();
        }
    }

    // ------------------------------------------------------------------ helpers

    /** The config.json name of a scaling type (the description prints it). */
    private static String scalingKey(Scaling s) {
        return switch (s) {
            case Scaling.None n -> "default";
            case Scaling.Identity i -> "none";
            case Scaling.Linear l -> "linear";
            case Scaling.Llama3 l -> "llama3";
            case Scaling.LongRope l -> "longrope";
            case Scaling.Yarn y -> "yarn";
        };
    }

    private static double scalingFactor(Scaling s) {
        return switch (s) {
            case Scaling.None n -> 1.0;
            case Scaling.Identity i -> 1.0;
            case Scaling.Linear l -> l.factor();
            case Scaling.Llama3 l -> l.factor();
            case Scaling.LongRope l -> l.factor();
            case Scaling.Yarn y -> y.factor();
        };
    }

    /** {@code mrope_section} of a Qwen-VL rope_scaling (three numbers; a trailing unused 4th section is dropped), else empty. */
    static List<Integer> mropeSections(Map<String, Object> rs) {
        if (rs == null || !(rs.get("mrope_section") instanceof List<?> l) || l.size() < 3) {
            return List.of();
        }
        List<Integer> out = new ArrayList<>(3);
        for (int i = 0; i < 3; i++) {
            if (!(l.get(i) instanceof Number n)) {
                throw new IllegalArgumentException("invalid mrope_section " + l);
            }
            out.add(n.intValue());
        }
        for (int i = 3; i < l.size(); i++) {
            if (!(l.get(i) instanceof Number n) || n.intValue() != 0) {
                throw new IllegalArgumentException("mrope_section with a non-zero 4th section is not supported: " + l);
            }
        }
        return out;
    }

    /**
     * @param rs the scaling map ({@code rope_scaling}, or the {@code rope_parameters} entry), null for plain RoPE
     * @param c  the config, for the position limits longrope derives its factor from
     */
    static Scaling parseRopeScaling(Map<String, Object> rs, Map<String, Object> c, int rotaryDim) {
        if (rs == null) {
            return Scaling.NONE;
        }
        String type = Json.str(rs, "rope_type", Json.str(rs, "type", "default"));
        double factor = Json.dbl(rs, "factor", 1.0);
        switch (type) {
            case "default", "mrope": // Qwen-VL's multimodal RoPE is plain RoPE for text (the sections: mropeSections)
                return Scaling.NONE;
            case "llama3": {
                double low = Json.dbl(rs, "low_freq_factor", 1.0);
                double high = Json.dbl(rs, "high_freq_factor", 4.0);
                int orig = (int) Json.num(rs, "original_max_position_embeddings", 8192);
                if (factor <= 0 || low <= 0 || high < low || orig <= 0) {
                    throw new IllegalArgumentException("invalid llama3 rope_scaling: " + rs);
                }
                return new Scaling.Llama3(factor, low, high, orig);
            }
            case "longrope", "su": {
                int maxPos = (int) Json.num(c, "max_position_embeddings", 4096);
                int orig = (int) Json.num(rs, "original_max_position_embeddings",
                        Json.num(c, "original_max_position_embeddings", maxPos));
                double[] shortF = factors(rs, "short_factor", rotaryDim / 2);
                double[] longF = factors(rs, "long_factor", rotaryDim / 2);
                // Precedence as in transformers' _compute_longrope_parameters: an explicit rope_scaling.factor
                // wins; without one the factor is max_position_embeddings / original_max_position_embeddings,
                // the latter taken from rope_scaling first and the top level second (Phi-3 puts it there). The
                // factor only feeds the attention_factor default (Rope.attentionFactor), not the frequencies.
                double f = rs.containsKey("factor") ? factor : (double) maxPos / orig;
                if (orig <= 1 || f <= 0) {
                    throw new IllegalArgumentException("invalid longrope rope_scaling: " + rs);
                }
                return new Scaling.LongRope(f, orig, shortF, longF, Json.dbl(rs, "attention_factor", 0));
            }
            case "linear":
                if (factor <= 0) {
                    throw new IllegalArgumentException("invalid linear rope_scaling: " + rs);
                }
                return new Scaling.Linear(factor);
            case "yarn":
                return yarn(rs, c);
            default:
                throw new IllegalArgumentException("unsupported rope_scaling type '" + type
                        + "' (supported: default, linear, llama3, longrope, yarn)");
        }
    }

    /**
     * A {@code yarn} scaling map (Qwen2 / Qwen3 long context and the like): as transformers'
     * {@code _compute_yarn_parameters}, with {@code beta_fast} 32 and {@code beta_slow} 1 by default, the
     * pretrained length from {@code original_max_position_embeddings} (else {@code max_position_embeddings}) and
     * the amplitude {@code attention_factor}, else {@code mscale / mscale_all_dim} when both are set, else
     * {@code 0.1 ln(factor) + 1}. One deliberate difference: transformers replaces {@code factor} by
     * {@code max_position_embeddings / original_max_position_embeddings} when the latter is given; here the
     * {@code factor} of the map is used, as llama.cpp and vLLM do (and as a GGUF stores it), so a config that
     * enables YaRN without raising {@code max_position_embeddings} still gets the factor it asked for.
     */
    private static Scaling.Yarn yarn(Map<String, Object> rs, Map<String, Object> c) {
        double factor = Json.dbl(rs, "factor", 1.0);
        int orig = (int) Json.num(rs, "original_max_position_embeddings", Json.num(c, "max_position_embeddings", 0));
        double betaFast = Json.dbl(rs, "beta_fast", 32);
        double betaSlow = Json.dbl(rs, "beta_slow", 1);
        double attention = Json.dbl(rs, "attention_factor", 0);
        if (!(factor > 0) || orig <= 0 || !(betaFast > betaSlow) || !(betaSlow > 0) || attention < 0) {
            throw new IllegalArgumentException("invalid yarn rope_scaling: " + rs);
        }
        boolean truncate = Json.bool(rs, "truncate", true);
        double mscale = Json.dbl(rs, "mscale", 0);
        double mscaleAllDim = Json.dbl(rs, "mscale_all_dim", 0);
        boolean both = mscale != 0 && mscaleAllDim != 0;
        return new Scaling.Yarn(factor, orig, betaFast, betaSlow, both ? mscale : 1, both ? mscaleAllDim : 0, attention, truncate);
    }

    private static double[] factors(Map<String, Object> rs, String key, int n) {
        List<Object> l = Json.list(rs, key);
        if (l == null || l.size() != n) {
            throw new IllegalArgumentException("rope_scaling." + key + " must have " + n + " entries (rotary_dim/2), got "
                    + (l == null ? "nothing" : String.valueOf(l.size())));
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = ((Number) l.get(i)).doubleValue();
            if (!(out[i] > 0)) {
                throw new IllegalArgumentException("rope_scaling." + key + "[" + i + "] must be positive");
            }
        }
        return out;
    }

    /** A soft-capping value: {@code def} when the key is absent, 0 (off) when it is null, else the number. */
    private static double softcap(Map<String, Object> c, String key, double def) {
        return c.containsKey(key) ? Json.dbl(c, key, 0) : def;
    }

    /**
     * {@code key} as a list of ints: absent or null gives the empty list, a single int a one-entry list; a list
     * with anything but integral numbers (or any other value) is rejected with a clear message.
     */
    static List<Integer> intList(Map<String, Object> c, String key) {
        Object v = c.get(key);
        if (v == null) {
            return List.of();
        }
        if (v instanceof Number n) {
            return List.of(n.intValue());
        }
        if (v instanceof List<?> l) {
            List<Integer> out = new ArrayList<>(l.size());
            for (Object o : l) {
                if (!(o instanceof Number n) || n.doubleValue() != n.longValue()) {
                    throw new IllegalArgumentException("config.json: " + key + " must be an int or a list of ints, got " + v);
                }
                out.add(n.intValue());
            }
            return List.copyOf(out);
        }
        throw new IllegalArgumentException("config.json: " + key + " must be an int or a list of ints, got " + v);
    }

    /** Round-to-nearest-even of a float's bits to the upper 16 bits (bfloat16). */
    static int roundBf16(int bits) {
        int lsb = (bits >>> 16) & 1;
        return (bits + 0x7FFF + lsb) & 0xFFFF0000;
    }

    static int requirePositive(Map<String, Object> c, String key) {
        long v = Json.num(c, key, 0);
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("config.json: " + key + " must be a positive integer, got "
                    + (c.containsKey(key) ? c.get(key) : "nothing"));
        }
        return (int) v;
    }
}
