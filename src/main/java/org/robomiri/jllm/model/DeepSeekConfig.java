package org.robomiri.jllm.model;

import org.robomiri.jllm.json.Json;

import java.util.Map;
import java.util.Set;

/**
 * DeepSeek-V2 / V3 specifics of a {@code config.json}: Multi-head Latent Attention (MLA) shapes, the routed +
 * shared expert layout and the router rule, YaRN rotary scaling.
 *
 * <p>In the enclosing {@link ModelConfig} {@code headDim} is the per-head query/key width
 * {@code qkNopeDim + qkRopeDim}, {@code kvHeads == heads}, {@code numExperts} is {@code n_routed_experts} and
 * {@code moeDim} is {@code moe_intermediate_size}.
 *
 * @param qLoraRank     {@code q_lora_rank}, 0 when the query is a single {@code q_proj} (V2-Lite)
 * @param kvLoraRank    width of the compressed key/value latent
 * @param qkNopeDim     per-head key/query width that carries no position
 * @param qkRopeDim     per-head query width (and shared key width) that is rotated
 * @param vHeadDim      per-head value width
 * @param firstKDense   leading layers that use the dense MLP ({@code first_k_dense_replace})
 * @param moeLayerFreq  {@code moe_layer_freq}: layer l is MoE when {@code l >= firstKDense && l % freq == 0}
 * @param nShared       number of shared experts (0 = none); they form one dense MLP of {@code nShared * moeDim}
 * @param nGroup        expert groups for the group-limited rules
 * @param topkGroup     groups kept by the group-limited rules
 * @param sigmoid       {@code scoring_func}: sigmoid (V3) instead of softmax (V2)
 * @param topkMethod    {@code greedy}, {@code group_limited_greedy} or {@code noaux_tc}
 * @param routedScale   {@code routed_scaling_factor}
 * @param scaleAlways   V3 multiplies the top-k weights by {@code routedScale} also after renormalizing; the V2 code
 *                      only does so when {@code norm_topk_prob} is off
 * @param yarn          YaRN parameters, null when {@code rope_scaling} is absent
 * @param softmaxScale  attention score scale: {@code (nope+rope)^-0.5}, times {@code mscale^2} for YaRN
 */
public record DeepSeekConfig(
        int qLoraRank,
        int kvLoraRank,
        int qkNopeDim,
        int qkRopeDim,
        int vHeadDim,
        int firstKDense,
        int moeLayerFreq,
        int nShared,
        int nGroup,
        int topkGroup,
        boolean sigmoid,
        String topkMethod,
        double routedScale,
        boolean scaleAlways,
        Yarn yarn,
        double softmaxScale) {

    /** {@code rope_scaling} of type yarn. {@code mscale} / {@code mscaleAllDim} as in the DeepSeek code. */
    public record Yarn(double factor, int originalMaxPositions, double betaFast, double betaSlow, double mscale,
                       double mscaleAllDim) {
    }

    public static final Set<String> TOPK_METHODS = Set.of("greedy", "group_limited_greedy", "noaux_tc");

    /** Per-head query/key width. */
    public int qkDim() {
        return qkNopeDim + qkRopeDim;
    }

    public boolean hasGroups() {
        return !topkMethod.equals("greedy");
    }

    /** Parses and validates the DeepSeek keys; {@code v3} is true for {@code model_type: deepseek_v3}. */
    static DeepSeekConfig parse(Map<String, Object> c, boolean v3) {
        int qLora = (int) Json.num(c, "q_lora_rank", 0);
        int kvLora = (int) Json.num(c, "kv_lora_rank", 0);
        int nope = (int) Json.num(c, "qk_nope_head_dim", 0);
        int rope = (int) Json.num(c, "qk_rope_head_dim", 0);
        int vDim = (int) Json.num(c, "v_head_dim", 0);
        if (kvLora <= 0 || nope <= 0 || rope <= 0 || rope % 2 != 0 || vDim <= 0 || qLora < 0) {
            throw new IllegalArgumentException("invalid DeepSeek MLA config: q_lora_rank=" + qLora + " kv_lora_rank="
                    + kvLora + " qk_nope_head_dim=" + nope + " qk_rope_head_dim=" + rope + " v_head_dim=" + vDim);
        }
        String scoring = Json.str(c, "scoring_func", "softmax");
        if (!scoring.equals("softmax") && !scoring.equals("sigmoid")) {
            throw new IllegalArgumentException("unsupported scoring_func '" + scoring + "' (softmax, sigmoid)");
        }
        String method = Json.str(c, "topk_method", "greedy");
        if (!TOPK_METHODS.contains(method)) {
            throw new IllegalArgumentException("unsupported topk_method '" + method + "' (supported: " + TOPK_METHODS + ")");
        }
        int nGroup = (int) Json.num(c, "n_group", 1);
        int topkGroup = (int) Json.num(c, "topk_group", 1);
        int nRouted = (int) Json.num(c, "n_routed_experts", 0);
        int topK = (int) Json.num(c, "num_experts_per_tok", 0);
        if (!method.equals("greedy")) {
            if (nGroup <= 0 || nGroup > 64 || nRouted % nGroup != 0 || topkGroup <= 0 || topkGroup > nGroup
                    || topK > topkGroup * (nRouted / nGroup)) {
                throw new IllegalArgumentException("invalid group routing: n_group=" + nGroup + " topk_group="
                        + topkGroup + " n_routed_experts=" + nRouted + " num_experts_per_tok=" + topK);
            }
        }
        int moeFreq = (int) Json.num(c, "moe_layer_freq", 1);
        if (moeFreq <= 0) {
            throw new IllegalArgumentException("moe_layer_freq must be positive, got " + moeFreq);
        }
        Object shared = c.get("n_shared_experts");
        int nShared = shared instanceof Number n ? n.intValue() : 0;

        Yarn yarn = null;
        double scale = 1.0 / Math.sqrt(nope + rope);
        Map<String, Object> rs = Json.map(c, "rope_scaling");
        if (rs != null) {
            String type = Json.str(rs, "type", Json.str(rs, "rope_type", "default"));
            if (type.equals("yarn")) {
                double factor = Json.dbl(rs, "factor", 1.0);
                int orig = (int) Json.num(rs, "original_max_position_embeddings", 4096);
                if (factor <= 0 || orig <= 0) {
                    throw new IllegalArgumentException("invalid yarn rope_scaling: " + rs);
                }
                yarn = new Yarn(factor, orig, Json.dbl(rs, "beta_fast", 32), Json.dbl(rs, "beta_slow", 1),
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
        return new DeepSeekConfig(qLora, kvLora, nope, rope, vDim, (int) Json.num(c, "first_k_dense_replace", 0),
                moeFreq, nShared, nGroup, topkGroup, scoring.equals("sigmoid"), method,
                Json.dbl(c, "routed_scaling_factor", 1.0), v3, yarn, scale);
    }
}
