package org.robomiri.jllm.model;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Hyper-parameters from a Hugging Face {@code config.json}: Llama ({@code llama}), Mistral ({@code mistral}),
 * Qwen2 ({@code qwen2}), Qwen3 dense ({@code qwen3}) and Mixture-of-Experts ({@code qwen3_moe}), Gemma 2
 * ({@code gemma2}) and Gemma 3 text ({@code gemma3_text}, also inside the {@code gemma3} multimodal wrapper) decoders, and
 * DeepSeek-V2 / V3 ({@code deepseek_v2}, {@code deepseek_v3}; see {@link DeepSeekConfig}).
 *
 * @param modelType     the {@code model_type} string
 * @param attentionBias q/k/v projections have a bias (Qwen2)
 * @param qkNorm        per-head RMSNorm on queries and keys (Qwen3)
 * @param slidingWindow attention window in tokens, 0 for full causal attention
 * @param windowFromLayer first layer that uses {@code slidingWindow} (Mistral 0, Qwen2 {@code max_window_layers})
 * @param ropeScaling   RoPE frequency scaling, null for plain RoPE
 * @param bosTokenId    {@code bos_token_id}, -1 if absent
 * @param eosTokenIds   {@code eos_token_id} (an int or a list in config.json)
 * @param numExperts    0 for dense models
 * @param expertsPerTok experts activated per token (top-k of the router)
 * @param moeDim        intermediate size of each expert
 * @param normTopK      renormalize the top-k router weights to sum to 1
 * @param mlpOnlyLayers layers that use the dense MLP even in a MoE model
 * @param partialRotaryFactor fraction of each head that is rotated by RoPE (GLM: 0.5 or 1.0; others 1.0)
 * @param gemma         Gemma 2 / 3 specifics, null for every other family
 * @param mamba         state-space parameters of a Mamba model (then {@code hiddenDim} is {@code d_inner}, attention
 *                      fields are placeholders), null for transformers
 * @param deepseek      DeepSeek-V2/V3 attention (MLA) and router parameters, null for every other family
 */
public record ModelConfig(
        String architecture,
        String modelType,
        boolean attentionBias,
        boolean qkNorm,
        int slidingWindow,
        int windowFromLayer,
        RopeScaling ropeScaling,
        int bosTokenId,
        List<Integer> eosTokenIds,
        int dim,
        int hiddenDim,
        int layers,
        int heads,
        int kvHeads,
        int headDim,
        int vocabSize,
        int maxPositions,
        float normEps,
        double ropeTheta,
        boolean tiedEmbeddings,
        int numExperts,
        int expertsPerTok,
        int moeDim,
        boolean normTopK,
        int sparseStep,
        Set<Integer> mlpOnlyLayers,
        double partialRotaryFactor,
        Gemma gemma,
        MambaParams mamba,
        DeepSeekConfig deepseek) {

    /**
     * What distinguishes Gemma 2 / Gemma 3 (text) from the Llama-style decoder: GeLU(tanh) MLP, norm gain
     * {@code 1 + w}, four norms per layer, embedding scaled by {@code sqrt(dim)}, attention scale
     * {@code query_pre_attn_scalar^-0.5}, optional logit soft-capping and per-layer local/global attention with
     * a different RoPE base for the local layers.
     *
     * @param attnSoftcap     {@code attn_logit_softcapping}, 0 = off (Gemma 3)
     * @param finalSoftcap    {@code final_logit_softcapping}, 0 = off (Gemma 3)
     * @param queryScalar     {@code query_pre_attn_scalar}
     * @param local           per layer: true = sliding-window attention (and the local RoPE table if there is one)
     * @param localRopeTheta  {@code rope_local_base_freq} (Gemma 3), 0 = local layers share the global RoPE table
     * @param embedScale      factor applied to every embedding row (sqrt(dim), rounded to the checkpoint dtype)
     */
    public record Gemma(float attnSoftcap, float finalSoftcap, double queryScalar, boolean[] local,
                        double localRopeTheta, float embedScale) {
    }

    public boolean isGemma() {
        return gemma != null;
    }

    /** MLP activation: {@code silu} (SwiGLU) or {@code gelu_tanh} (Gemma's GeGLU). */
    public String activation() {
        return gemma != null ? "gelu_tanh" : "silu";
    }

    /** Multiplier of q.k before the softmax: {@code 1/sqrt(head_dim)}, or {@code query_pre_attn_scalar^-0.5}. */
    public float attnScale() {
        return gemma != null ? (float) (1.0 / Math.sqrt(gemma.queryScalar())) : (float) (1.0 / Math.sqrt(headDim));
    }

    /** Attention-logit soft cap ({@code tanh(s/cap)*cap}), 0 = none. */
    public float attnSoftcap() {
        return gemma != null ? gemma.attnSoftcap() : 0f;
    }

    /** Final-logit soft cap, 0 = none. */
    public float finalSoftcap() {
        return gemma != null ? gemma.finalSoftcap() : 0f;
    }

    /** True if layer {@code l} uses the local RoPE table (Gemma 3 sliding layers; its theta differs from the global one). */
    public boolean usesLocalRope(int l) {
        return gemma != null && gemma.localRopeTheta() > 0 && gemma.local()[l];
    }

    /** Scale applied to the dequantized embedding of every token (1 except for Gemma). */
    public float embedScale() {
        return gemma != null ? gemma.embedScale() : 1f;
    }

    /** Gemma stores norm gains as {@code w} and applies {@code 1 + w}; the converter folds the 1 in. */
    public boolean normPlusOne() {
        return gemma != null;
    }

    /**
     * Mamba (v1) mixer sizes from config.json: {@code state_size}, {@code conv_kernel}, {@code time_step_rank}
     * (resolved from "auto" = ceil(d_model/16)), {@code d_inner} ({@code intermediate_size} or
     * {@code expand * hidden_size}) and the two optional biases.
     */
    public record MambaParams(int dState, int dConv, int dtRank, int dInner, boolean useBias, boolean convBias) {
    }

    public boolean isMamba() {
        return mamba != null;
    }

    /**
     * RoPE scaling from {@code rope_scaling}. {@code type} is {@code llama3}, {@code linear} or {@code longrope}
     * (Phi-3; the legacy name {@code su} is normalized). LongRoPE uses {@code shortFactor} / {@code longFactor}
     * (one divisor per rotary pair) and {@code attentionFactor} (0 = derive it from {@code factor}); there
     * {@code factor} is {@code rope_scaling.factor} or else {@code max_position_embeddings / original}.
     */
    public record RopeScaling(String type, double factor, double lowFreqFactor, double highFreqFactor,
                              int originalMaxPositions, double[] shortFactor, double[] longFactor,
                              double attentionFactor) {
        public RopeScaling(String type, double factor, double lowFreqFactor, double highFreqFactor,
                           int originalMaxPositions) {
            this(type, factor, lowFreqFactor, highFreqFactor, originalMaxPositions, null, null, 0);
        }
    }

    /** Attention window of layer {@code l} in tokens; 0 means the full causal context. */
    public int windowFor(int l) {
        if (gemma != null) {
            return slidingWindow > 0 && gemma.local()[l] ? slidingWindow : 0;
        }
        return slidingWindow > 0 && l >= windowFromLayer ? slidingWindow : 0;
    }

    public int qDim() {
        return heads * headDim;
    }

    public int kvDim() {
        return kvHeads * headDim;
    }

    /** GLM family (glm, glm4): RoPE rotates the interleaved pairs (2i, 2i+1) instead of (i, i + headDim/2). */
    public boolean ropeInterleaved() {
        return modelType.equals("glm") || modelType.equals("glm4");
    }

    /** Leading dimensions of each head that RoPE rotates: {@code headDim * partial_rotary_factor}, even. */
    public int rotaryDim() {
        return (int) (headDim * partialRotaryFactor) & ~1;
    }

    /** Rotate-half RoPE on a prefix of each head only (Phi-4-mini, {@code partial_rotary_factor} < 1). */
    public boolean ropePartialHalf() {
        return !ropeInterleaved() && rotaryDim() != headDim;
    }

    /** Glm4 and Gemma: RMSNorms on the attention and MLP outputs, before they are added to the residual stream. */
    public boolean postNorms() {
        return modelType.equals("glm4") || gemma != null;
    }

    public boolean isMoe() {
        return numExperts > 0;
    }

    /** Same rule as transformers' Qwen3MoeDecoderLayer; DeepSeek: {@code l >= first_k_dense_replace && l % moe_layer_freq == 0}. */
    public boolean isMoeLayer(int l) {
        if (deepseek != null) {
            return numExperts > 0 && l >= deepseek.firstKDense() && l % deepseek.moeLayerFreq() == 0;
        }
        return numExperts > 0 && !mlpOnlyLayers.contains(l) && (l + 1) % Math.max(1, sparseStep) == 0;
    }

    /** Width of the widest dense SwiGLU MLP: the dense layers' {@code intermediate_size} or the shared experts'. */
    public int maxMlpDim() {
        return deepseek == null ? hiddenDim : Math.max(hiddenDim, deepseek.nShared() * moeDim);
    }

    /** {@code model_type} values this engine implements, with the architecture class each corresponds to. */
    private static final Map<String, String> SUPPORTED = Map.ofEntries(
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
            Map.entry("deepseek_v3", "DeepseekV3ForCausalLM"));

    /** A multimodal wrapper whose {@code text_config} is a supported decoder (vision tensors are never read). */
    private static final Map<String, String> WRAPPERS = Map.of("gemma3_text", "Gemma3ForConditionalGeneration");

    public static ModelConfig load(Path dir) throws IOException {
        Map<String, Object> c;
        try {
            c = Json.parseObject(Files.readString(dir.resolve("config.json")));
        } catch (RuntimeException e) {
            throw new IOException(dir.resolve("config.json") + ": " + e.getMessage(), e);
        }
        List<Object> archs = Json.list(c, "architectures");
        String declaredArch = archs != null && !archs.isEmpty() ? String.valueOf(archs.getFirst()) : null;
        Map<String, Object> text = Json.map(c, "text_config");
        if (text != null) {
            // Multimodal wrapper (Gemma 3 4B+): the decoder's hyper-parameters live in text_config; the
            // wrapper-level ids and dtype fill in what the text config leaves out.
            Map<String, Object> merged = new java.util.LinkedHashMap<>(text);
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
            c = new java.util.LinkedHashMap<>(c);
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
        if (modelType.equals("mamba")) {
            return loadMamba(c, arch);
        }
        int dim = requirePositive(c, "hidden_size");
        int layers = requirePositive(c, "num_hidden_layers");
        int vocab = requirePositive(c, "vocab_size");
        int heads = requirePositive(c, "num_attention_heads");
        int kvHeads = (int) Json.num(c, "num_key_value_heads", heads);
        int headDim = (int) Json.num(c, "head_dim", dim / heads);
        boolean isDeepseek = modelType.startsWith("deepseek");
        DeepSeekConfig ds = null;
        if (isDeepseek) {
            ds = DeepSeekConfig.parse(c, modelType.equals("deepseek_v3"));
            headDim = ds.qkDim();   // per-head query/key width; MLA has no grouped KV heads
            kvHeads = heads;
        }
        if (kvHeads <= 0 || heads % kvHeads != 0) {
            throw new IllegalArgumentException("num_key_value_heads=" + kvHeads + " must be positive and divide "
                    + "num_attention_heads=" + heads);
        }
        if (headDim <= 0) {
            throw new IllegalArgumentException("head_dim must be positive, got " + headDim);
        }
        boolean gemma = modelType.startsWith("gemma");
        if (gemma) {
            // Gemma names the key hidden_activation (hidden_act is a stale duplicate in some configs)
            String ga = Json.str(c, "hidden_activation", Json.str(c, "hidden_act", "gelu_pytorch_tanh"));
            if (!ga.equals("gelu_pytorch_tanh") && !ga.equals("gelu_new") && !ga.equals("gelu_tanh")) {
                throw new IllegalArgumentException("unsupported hidden_activation '" + ga
                        + "': Gemma needs gelu_pytorch_tanh");
            }
        } else {
            String act = Json.str(c, "hidden_act", "silu");
            if (!act.equals("silu")) {
                throw new IllegalArgumentException("unsupported hidden_act '" + act + "': only silu (SwiGLU) is implemented");
            }
        }
        if (Json.bool(c, "mlp_bias", false)) {
            throw new IllegalArgumentException("models with an MLP bias (mlp_bias=true) are not supported");
        }
        boolean qwen2 = modelType.equals("qwen2");
        boolean glm = modelType.equals("glm") || modelType.equals("glm4");
        boolean phi3 = modelType.equals("phi3");
        // GlmConfig / Glm4Config default attention_bias to true (q/k/v only; o_proj never has a bias there)
        boolean bias = qwen2 || (glm && Json.bool(c, "attention_bias", true));
        if (!qwen2 && !glm && Json.bool(c, "attention_bias", false)) {
            // transformers also adds a bias to o_proj in that case, which this engine does not implement
            throw new IllegalArgumentException("attention_bias=true is only supported for qwen2 and glm (q/k/v bias)");
        }
        // newer transformers nest rope_theta / partial_rotary_factor in rope_parameters; Hub configs use top level
        Map<String, Object> ropeParams = Json.map(c, "rope_parameters");
        double partialRotary = Json.dbl(c, "partial_rotary_factor",
                ropeParams != null ? Json.dbl(ropeParams, "partial_rotary_factor", glm ? 0.5 : 1.0) : glm ? 0.5 : 1.0);
        if (partialRotary <= 0 || partialRotary > 1 || (int) (headDim * partialRotary) < 2) {
            throw new IllegalArgumentException("invalid partial_rotary_factor " + partialRotary);
        }
        double ropeTheta = Json.dbl(c, "rope_theta",
                ropeParams != null ? Json.dbl(ropeParams, "rope_theta", 10000.0) : 10000.0);
        int window = 0;
        int windowFrom = 0;
        long sw = Json.num(c, "sliding_window", 0);
        if (sw > 0) {
            if (modelType.equals("mistral") || gemma) {
                window = (int) Math.min(sw, Integer.MAX_VALUE);
            } else if (phi3) {
                // all layers; Phi-3.5 / Phi-4-mini set 262144, which is never reached
                window = (int) Math.min(sw, Integer.MAX_VALUE);
            } else if (qwen2 && Json.bool(c, "use_sliding_window", false)) {
                window = (int) Math.min(sw, Integer.MAX_VALUE);
                windowFrom = (int) Json.num(c, "max_window_layers", layers);
            }
        }
        double localTheta = Json.dbl(c, "rope_local_base_freq", 0);
        Map<String, Object> ropeSource = c;
        Map<String, Object> rp = Json.map(c, "rope_parameters");
        if (gemma && rp != null) {
            // transformers >= 5: rope_parameters, either flat or keyed by layer type
            Map<String, Object> full = Json.map(rp, "full_attention");
            Map<String, Object> slide = Json.map(rp, "sliding_attention");
            Map<String, Object> g = full != null ? full : rp;
            ropeTheta = Json.dbl(g, "rope_theta", ropeTheta);
            if (slide != null) {
                localTheta = Json.dbl(slide, "rope_theta", localTheta);
            }
            ropeSource = Json.obj("rope_scaling", g.containsKey("rope_type") ? g : null);
        }
        ModelConfig.RopeScaling scaling = isDeepseek ? null // DeepSeek: yarn lives in ds
                : parseRopeScaling(ropeSource, (int) (headDim * partialRotary) & ~1);
        List<Integer> eos = new java.util.ArrayList<>();
        Object eosRaw = c.get("eos_token_id");
        if (eosRaw instanceof Number n) {
            eos.add(n.intValue());
        } else if (eosRaw instanceof List<?> l) {
            for (Object o : l) {
                eos.add(((Number) o).intValue());
            }
        }
        List<Object> mlpOnly = Json.list(c, "mlp_only_layers");
        Set<Integer> mlpOnlyLayers = mlpOnly == null ? Set.of()
                : mlpOnly.stream().map(o -> ((Number) o).intValue()).collect(Collectors.toUnmodifiableSet());
        int numExperts = (int) Json.num(c, isDeepseek ? "n_routed_experts" : "num_experts", 0);
        int expertsPerTok = (int) Json.num(c, "num_experts_per_tok", 0);
        int moeDim = (int) Json.num(c, "moe_intermediate_size", 0);
        if (numExperts > 0 && (expertsPerTok <= 0 || expertsPerTok > numExperts || moeDim <= 0)) {
            throw new IllegalArgumentException("invalid MoE config: num_experts=" + numExperts
                    + " num_experts_per_tok=" + expertsPerTok + " moe_intermediate_size=" + moeDim);
        }
        ModelConfig cfg = new ModelConfig(
                arch,
                modelType,
                bias,
                modelType.startsWith("qwen3") || modelType.equals("gemma3_text"),
                window,
                windowFrom,
                scaling,
                (int) Json.num(c, "bos_token_id", -1),
                List.copyOf(eos),
                dim,
                (int) Json.num(c, "intermediate_size", 0),
                layers,
                heads,
                kvHeads,
                headDim,
                vocab,
                (int) Json.num(c, "max_position_embeddings", 4096),
                (float) Json.dbl(c, "rms_norm_eps", 1e-6),
                ropeTheta,
                Json.bool(c, "tie_word_embeddings", gemma), // Gemma configs omit the key; the class default is true
                numExperts,
                expertsPerTok,
                moeDim,
                Json.bool(c, "norm_topk_prob", false),
                (int) Json.num(c, "decoder_sparse_step", 1),
                mlpOnlyLayers,
                partialRotary,
                gemma ? gemmaOptions(c, modelType, layers, dim, localTheta) : null,
                null,
                ds);
        boolean anyDense = false;
        for (int l = 0; l < layers; l++) {
            anyDense |= !cfg.isMoeLayer(l);
        }
        if (anyDense && cfg.hiddenDim() <= 0) {
            throw new IllegalArgumentException("intermediate_size must be positive for a model with dense MLP layers");
        }
        return cfg;
    }

    /** Gemma 2/3 settings; see transformers' modeling_gemma2.py / modeling_gemma3.py. */
    private static Gemma gemmaOptions(Map<String, Object> c, String modelType, int layers, int dim, double localTheta) {
        boolean g3 = modelType.equals("gemma3_text");
        boolean[] local = new boolean[layers];
        List<Object> types = Json.list(c, "layer_types");
        int pattern = (int) Json.num(c, "sliding_window_pattern", 6);
        for (int l = 0; l < layers; l++) {
            if (types != null && l < types.size()) {
                local[l] = "sliding_attention".equals(types.get(l));
            } else if (g3) {
                local[l] = pattern > 0 && (l + 1) % pattern != 0;
            } else {
                local[l] = l % 2 == 0;
            }
        }
        // transformers computes the normalizer as a tensor in the checkpoint dtype: sqrt(dim) rounded to bf16/f16
        String dtype = Json.str(c, "torch_dtype", Json.str(c, "dtype", "bfloat16"));
        float embedScale = (float) Math.sqrt(dim);
        if (dtype.equals("bfloat16")) {
            embedScale = Float.intBitsToFloat(roundBf16(Float.floatToIntBits(embedScale)));
        } else if (dtype.equals("float16")) {
            embedScale = Float.float16ToFloat(Float.floatToFloat16(embedScale));
        }
        return new Gemma((float) Json.dbl(c, "attn_logit_softcapping", 0), (float) Json.dbl(c, "final_logit_softcapping", 0),
                Json.dbl(c, "query_pre_attn_scalar", 256), local, g3 ? localTheta : 0, embedScale);
    }

    /** Round-to-nearest-even of a float's bits to the upper 16 bits (bfloat16). */
    public static int roundBf16(int bits) {
        int lsb = (bits >>> 16) & 1;
        return (bits + 0x7FFF + lsb) & 0xFFFF0000;
    }

    /**
     * Mamba config. Attention fields are placeholders (1 head of size d_model) so generic checks hold; the
     * LM head is tied to the embeddings unless the config says otherwise (the released checkpoints have no lm_head).
     */
    private static ModelConfig loadMamba(Map<String, Object> c, String arch) {
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
        List<Integer> eos = new java.util.ArrayList<>();
        Object eosRaw = c.get("eos_token_id");
        if (eosRaw instanceof Number n) {
            eos.add(n.intValue());
        } else if (eosRaw instanceof List<?> l) {
            for (Object o : l) {
                eos.add(((Number) o).intValue());
            }
        }
        MambaParams mp = new MambaParams(dState, dConv, dtRank, dInner, false, Json.bool(c, "use_conv_bias", true));
        return new ModelConfig(arch, "mamba", false, false, 0, 0, null, -1, List.copyOf(eos), dim, dInner, layers,
                1, 1, dim, vocab, (int) Json.num(c, "max_position_embeddings", 1 << 20),
                (float) Json.dbl(c, "layer_norm_epsilon", 1e-5), 0, Json.bool(c, "tie_word_embeddings", true), 0, 0, 0, false, 1, Set.of(), 1.0, null, mp, null);
    }

    private static ModelConfig.RopeScaling parseRopeScaling(Map<String, Object> c, int rotaryDim) {
        Map<String, Object> rs = Json.map(c, "rope_scaling");
        if (rs == null) {
            return null;
        }
        String type = Json.str(rs, "rope_type", Json.str(rs, "type", "default"));
        double factor = Json.dbl(rs, "factor", 1.0);
        switch (type) {
            case "default":
                return null;
            case "llama3": {
                double low = Json.dbl(rs, "low_freq_factor", 1.0);
                double high = Json.dbl(rs, "high_freq_factor", 4.0);
                int orig = (int) Json.num(rs, "original_max_position_embeddings", 8192);
                if (factor <= 0 || low <= 0 || high <= low || orig <= 0) {
                    throw new IllegalArgumentException("invalid llama3 rope_scaling: " + rs);
                }
                return new ModelConfig.RopeScaling(type, factor, low, high, orig);
            }
            case "longrope", "su": {
                int maxPos = (int) Json.num(c, "max_position_embeddings", 4096);
                int orig = (int) Json.num(rs, "original_max_position_embeddings",
                        Json.num(c, "original_max_position_embeddings", maxPos));
                double[] shortF = factors(rs, "short_factor", rotaryDim / 2);
                double[] longF = factors(rs, "long_factor", rotaryDim / 2);
                double f = rs.containsKey("factor") ? factor : (double) maxPos / orig;
                if (orig <= 1 || f <= 0) {
                    throw new IllegalArgumentException("invalid longrope rope_scaling: " + rs);
                }
                return new ModelConfig.RopeScaling("longrope", f, 1, 4, orig, shortF, longF,
                        Json.dbl(rs, "attention_factor", 0));
            }
            case "linear":
                if (factor <= 0) {
                    throw new IllegalArgumentException("invalid linear rope_scaling: " + rs);
                }
                return new ModelConfig.RopeScaling(type, factor, 1, 4, 0);
            default:
                throw new IllegalArgumentException("unsupported rope_scaling type '" + type
                        + "' (supported: default, linear, llama3, longrope)");
        }
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

    private String localLayerString() {
        StringBuilder sb = new StringBuilder();
        for (boolean b : gemma.local()) {
            sb.append(b ? 'L' : 'G');
        }
        return sb.toString();
    }

    private static int requirePositive(Map<String, Object> c, String key) {
        long v = Json.num(c, key, 0);
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("config.json: " + key + " must be a positive integer, got "
                    + (c.containsKey(key) ? c.get(key) : "nothing"));
        }
        return (int) v;
    }

    @Override
    public String toString() {
        if (isMamba()) {
            return String.format("%s dim=%d dInner=%d layers=%d dState=%d dConv=%d dtRank=%d vocab=%d", architecture,
                    dim, mamba.dInner(), layers, mamba.dState(), mamba.dConv(), mamba.dtRank(), vocabSize);
        }
        String moe = isMoe() ? String.format(" experts=%d/%d moeDim=%d", expertsPerTok, numExperts, moeDim) : "";
        // Features absent from the original Qwen3-only engine are appended only when set, so the description
        // (also stored in the quantized cache) of an existing Qwen3 model does not change.
        String mla = deepseek == null ? "" : String.format(" mla(q=%d kv=%d nope=%d rope=%d v=%d) shared=%d dense=%d %s",
                deepseek.qLoraRank(), deepseek.kvLoraRank(), deepseek.qkNopeDim(), deepseek.qkRopeDim(),
                deepseek.vHeadDim(), deepseek.nShared(), deepseek.firstKDense(), deepseek.topkMethod())
                + (deepseek.yarn() != null ? " yarnx" + deepseek.yarn().factor() : "");
        String extra = mla + (attentionBias ? " bias" : "") + (slidingWindow > 0 ? " window=" + slidingWindow
                + (windowFromLayer > 0 ? "@" + windowFromLayer : "") : "")
                + (ropeScaling != null ? " rope=" + ropeScaling.type() + "x" + ropeScaling.factor() : "")
                + (ropeInterleaved() ? " rope=interleaved/" + rotaryDim() : "")
                + (!ropeInterleaved() && rotaryDim() != headDim ? " rotary=" + rotaryDim() : "") + (postNorms() ? " postnorms" : "")
                + (gemma != null ? " gemma(softcap=" + gemma.attnSoftcap() + "/" + gemma.finalSoftcap() + " qscalar="
                + gemma.queryScalar() + " localRope=" + gemma.localRopeTheta() + " layers=" + localLayerString() + ")" : "");
        return String.format("%s dim=%d ffn=%d layers=%d heads=%d/%d headDim=%d vocab=%d theta=%.0f%s%s",
                architecture, dim, hiddenDim, layers, heads, kvHeads, headDim, vocabSize, ropeTheta, moe, extra);
    }
}
