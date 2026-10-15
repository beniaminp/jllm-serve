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
 * Qwen2 ({@code qwen2}), Qwen3 dense ({@code qwen3}) and Mixture-of-Experts ({@code qwen3_moe}) decoders.
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
        double partialRotaryFactor) {

    /** RoPE scaling from {@code rope_scaling}. {@code type} is {@code llama3} or {@code linear}. */
    public record RopeScaling(String type, double factor, double lowFreqFactor, double highFreqFactor,
                              int originalMaxPositions) {
    }

    /** Attention window of layer {@code l} in tokens; 0 means the full causal context. */
    public int windowFor(int l) {
        return slidingWindow > 0 && l >= windowFromLayer ? slidingWindow : 0;
    }

    /**
     * The token id to put in front of every prompt, or -1 when the family does not use one: Llama and Mistral
     * were trained with {@code <s>} / {@code <|begin_of_text|>} at the start of each sequence, Qwen was not.
     */
    public int promptBosId() {
        return modelType.equals("llama") || modelType.equals("mistral") ? bosTokenId : -1;
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

    /** Glm4: RMSNorms on the attention and MLP outputs, before they are added to the residual stream. */
    public boolean postNorms() {
        return modelType.equals("glm4");
    }

    public boolean isMoe() {
        return numExperts > 0;
    }

    /** Same rule as transformers' Qwen3MoeDecoderLayer. */
    public boolean isMoeLayer(int l) {
        return numExperts > 0 && !mlpOnlyLayers.contains(l) && (l + 1) % Math.max(1, sparseStep) == 0;
    }

    /** {@code model_type} values this engine implements, with the architecture class each corresponds to. */
    private static final Map<String, String> SUPPORTED = Map.ofEntries(
            Map.entry("llama", "LlamaForCausalLM"),
            Map.entry("mistral", "MistralForCausalLM"),
            Map.entry("qwen2", "Qwen2ForCausalLM"),
            Map.entry("qwen3", "Qwen3ForCausalLM"),
            Map.entry("qwen3_moe", "Qwen3MoeForCausalLM"),
            Map.entry("glm", "GlmForCausalLM"),
            Map.entry("glm4", "Glm4ForCausalLM"));

    public static ModelConfig load(Path dir) throws IOException {
        Map<String, Object> c;
        try {
            c = Json.parseObject(Files.readString(dir.resolve("config.json")));
        } catch (RuntimeException e) {
            throw new IOException(dir.resolve("config.json") + ": " + e.getMessage(), e);
        }
        List<Object> archs = Json.list(c, "architectures");
        String declaredArch = archs != null && !archs.isEmpty() ? String.valueOf(archs.getFirst()) : null;
        String modelType = Json.str(c, "model_type", null);
        if (modelType == null && declaredArch != null) {
            modelType = SUPPORTED.entrySet().stream().filter(e -> e.getValue().equals(declaredArch))
                    .map(Map.Entry::getKey).findFirst().orElse(null);
        }
        if (modelType == null || !SUPPORTED.containsKey(modelType)
                || (declaredArch != null && !SUPPORTED.get(modelType).equals(declaredArch))) {
            throw new IllegalArgumentException("unsupported model: model_type=" + modelType + ", architectures="
                    + archs + "; supported: " + SUPPORTED);
        }
        String arch = declaredArch != null ? declaredArch : SUPPORTED.get(modelType);
        int dim = requirePositive(c, "hidden_size");
        int layers = requirePositive(c, "num_hidden_layers");
        int vocab = requirePositive(c, "vocab_size");
        int heads = requirePositive(c, "num_attention_heads");
        int kvHeads = (int) Json.num(c, "num_key_value_heads", heads);
        int headDim = (int) Json.num(c, "head_dim", dim / heads);
        if (kvHeads <= 0 || heads % kvHeads != 0) {
            throw new IllegalArgumentException("num_key_value_heads=" + kvHeads + " must be positive and divide "
                    + "num_attention_heads=" + heads);
        }
        if (headDim <= 0) {
            throw new IllegalArgumentException("head_dim must be positive, got " + headDim);
        }
        String act = Json.str(c, "hidden_act", "silu");
        if (!act.equals("silu")) {
            throw new IllegalArgumentException("unsupported hidden_act '" + act + "': only silu (SwiGLU) is implemented");
        }
        if (Json.bool(c, "mlp_bias", false)) {
            throw new IllegalArgumentException("models with an MLP bias (mlp_bias=true) are not supported");
        }
        boolean qwen2 = modelType.equals("qwen2");
        boolean glm = modelType.equals("glm") || modelType.equals("glm4");
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
            if (modelType.equals("mistral")) {
                window = (int) Math.min(sw, Integer.MAX_VALUE);
            } else if (qwen2 && Json.bool(c, "use_sliding_window", false)) {
                window = (int) Math.min(sw, Integer.MAX_VALUE);
                windowFrom = (int) Json.num(c, "max_window_layers", layers);
            }
        }
        ModelConfig.RopeScaling scaling = parseRopeScaling(c);
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
        int numExperts = (int) Json.num(c, "num_experts", 0);
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
                modelType.startsWith("qwen3"),
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
                Json.bool(c, "tie_word_embeddings", false),
                numExperts,
                expertsPerTok,
                moeDim,
                Json.bool(c, "norm_topk_prob", false),
                (int) Json.num(c, "decoder_sparse_step", 1),
                mlpOnlyLayers,
                partialRotary);
        boolean anyDense = false;
        for (int l = 0; l < layers; l++) {
            anyDense |= !cfg.isMoeLayer(l);
        }
        if (anyDense && cfg.hiddenDim() <= 0) {
            throw new IllegalArgumentException("intermediate_size must be positive for a model with dense MLP layers");
        }
        return cfg;
    }

    private static ModelConfig.RopeScaling parseRopeScaling(Map<String, Object> c) {
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
            case "linear":
                if (factor <= 0) {
                    throw new IllegalArgumentException("invalid linear rope_scaling: " + rs);
                }
                return new ModelConfig.RopeScaling(type, factor, 1, 4, 0);
            default:
                throw new IllegalArgumentException("unsupported rope_scaling type '" + type
                        + "' (supported: default, linear, llama3)");
        }
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
        String moe = isMoe() ? String.format(" experts=%d/%d moeDim=%d", expertsPerTok, numExperts, moeDim) : "";
        // Features absent from the original Qwen3-only engine are appended only when set, so the description
        // (also stored in the quantized cache) of an existing Qwen3 model does not change.
        String extra = (attentionBias ? " bias" : "") + (slidingWindow > 0 ? " window=" + slidingWindow
                + (windowFromLayer > 0 ? "@" + windowFromLayer : "") : "")
                + (ropeScaling != null ? " rope=" + ropeScaling.type() + "x" + ropeScaling.factor() : "")
                + (ropeInterleaved() ? " rope=interleaved/" + rotaryDim() : "") + (postNorms() ? " postnorms" : "");
        return String.format("%s dim=%d ffn=%d layers=%d heads=%d/%d headDim=%d vocab=%d theta=%.0f%s%s",
                architecture, dim, hiddenDim, layers, heads, kvHeads, headDim, vocabSize, ropeTheta, moe, extra);
    }
}
