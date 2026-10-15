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
 * Hyper-parameters from a Hugging Face {@code config.json}: Qwen3 dense ({@code qwen3}) and Mixture-of-Experts
 * ({@code qwen3_moe}) decoders.
 *
 * @param numExperts    0 for dense models
 * @param expertsPerTok experts activated per token (top-k of the router)
 * @param moeDim        intermediate size of each expert
 * @param normTopK      renormalize the top-k router weights to sum to 1
 * @param mlpOnlyLayers layers that use the dense MLP even in a MoE model
 */
public record ModelConfig(
        String architecture,
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
        Set<Integer> mlpOnlyLayers) {

    public int qDim() {
        return heads * headDim;
    }

    public int kvDim() {
        return kvHeads * headDim;
    }

    public boolean isMoe() {
        return numExperts > 0;
    }

    /** Same rule as transformers' Qwen3MoeDecoderLayer. */
    public boolean isMoeLayer(int l) {
        return numExperts > 0 && !mlpOnlyLayers.contains(l) && (l + 1) % Math.max(1, sparseStep) == 0;
    }

    /** {@code model_type} values this engine implements, with the architecture class each corresponds to. */
    private static final Map<String, String> SUPPORTED = Map.of(
            "qwen3", "Qwen3ForCausalLM",
            "qwen3_moe", "Qwen3MoeForCausalLM");

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
        if (Json.map(c, "rope_scaling") != null) {
            System.err.println("warning: rope_scaling is not supported and will be ignored");
        }
        if (Json.bool(c, "attention_bias", false)) {
            throw new IllegalArgumentException("models with attention bias are not supported");
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
                dim,
                (int) Json.num(c, "intermediate_size", 0),
                layers,
                heads,
                kvHeads,
                headDim,
                vocab,
                (int) Json.num(c, "max_position_embeddings", 4096),
                (float) Json.dbl(c, "rms_norm_eps", 1e-6),
                Json.dbl(c, "rope_theta", 10000.0),
                Json.bool(c, "tie_word_embeddings", false),
                numExperts,
                expertsPerTok,
                moeDim,
                Json.bool(c, "norm_topk_prob", false),
                (int) Json.num(c, "decoder_sparse_step", 1),
                mlpOnlyLayers);
        boolean anyDense = false;
        for (int l = 0; l < layers; l++) {
            anyDense |= !cfg.isMoeLayer(l);
        }
        if (anyDense && cfg.hiddenDim() <= 0) {
            throw new IllegalArgumentException("intermediate_size must be positive for a model with dense MLP layers");
        }
        return cfg;
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
        return String.format("%s dim=%d ffn=%d layers=%d heads=%d/%d headDim=%d vocab=%d theta=%.0f%s",
                architecture, dim, hiddenDim, layers, heads, kvHeads, headDim, vocabSize, ropeTheta, moe);
    }
}
