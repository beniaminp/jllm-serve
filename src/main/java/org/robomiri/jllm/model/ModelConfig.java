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

    public static ModelConfig load(Path dir) throws IOException {
        Map<String, Object> c = Json.parseObject(Files.readString(dir.resolve("config.json")));
        List<Object> archs = Json.list(c, "architectures");
        String arch = archs != null && !archs.isEmpty() ? String.valueOf(archs.getFirst()) : Json.str(c, "model_type", "unknown");
        int dim = (int) Json.num(c, "hidden_size", 0);
        int heads = (int) Json.num(c, "num_attention_heads", 0);
        if (Json.map(c, "rope_scaling") != null) {
            System.err.println("warning: rope_scaling is not supported and will be ignored");
        }
        if (Json.bool(c, "attention_bias", false)) {
            throw new IllegalArgumentException("models with attention bias are not supported");
        }
        List<Object> mlpOnly = Json.list(c, "mlp_only_layers");
        Set<Integer> mlpOnlyLayers = mlpOnly == null ? Set.of()
                : mlpOnly.stream().map(o -> ((Number) o).intValue()).collect(Collectors.toUnmodifiableSet());
        return new ModelConfig(
                arch,
                dim,
                (int) Json.num(c, "intermediate_size", 0),
                (int) Json.num(c, "num_hidden_layers", 0),
                heads,
                (int) Json.num(c, "num_key_value_heads", heads),
                (int) Json.num(c, "head_dim", dim / Math.max(1, heads)),
                (int) Json.num(c, "vocab_size", 0),
                (int) Json.num(c, "max_position_embeddings", 4096),
                (float) Json.dbl(c, "rms_norm_eps", 1e-6),
                Json.dbl(c, "rope_theta", 10000.0),
                Json.bool(c, "tie_word_embeddings", false),
                (int) Json.num(c, "num_experts", 0),
                (int) Json.num(c, "num_experts_per_tok", 0),
                (int) Json.num(c, "moe_intermediate_size", 0),
                Json.bool(c, "norm_topk_prob", false),
                (int) Json.num(c, "decoder_sparse_step", 1),
                mlpOnlyLayers);
    }

    @Override
    public String toString() {
        String moe = isMoe() ? String.format(" experts=%d/%d moeDim=%d", expertsPerTok, numExperts, moeDim) : "";
        return String.format("%s dim=%d ffn=%d layers=%d heads=%d/%d headDim=%d vocab=%d theta=%.0f%s",
                architecture, dim, hiddenDim, layers, heads, kvHeads, headDim, vocabSize, ropeTheta, moe);
    }
}
