package org.robomiri.jllm.model;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.model.Architecture.Core;
import org.robomiri.jllm.model.HybridArch.Layer;
import org.robomiri.jllm.model.HybridArch.Mixer;
import org.robomiri.jllm.model.Mlp.Activation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The parsers of the attention / state-space hybrids ({@link HybridArch}): Granite 4 ({@code granitemoehybrid}),
 * Nemotron-H ({@code nemotron_h}) and Jamba ({@code jamba}), each following the defaults and rules of its transformers
 * config and modeling file.
 */
final class HybridParsers {

    private HybridParsers() {
    }

    private static float f(Map<String, Object> c, String key, double def) {
        return (float) Json.dbl(c, key, def);
    }

    /** A layer list entry: mamba / linear_attention, attention / full_attention, mlp, moe. */
    private static char kind(Object t) {
        return switch (String.valueOf(t)) {
            case "mamba", "linear_attention", "M" -> 'M';
            case "attention", "full_attention", "*" -> 'A';
            case "mlp", "-" -> 'F';
            case "moe", "E" -> 'E';
            default -> throw new IllegalArgumentException("config.json: unsupported layer type '" + t
                    + "' (mamba, attention, mlp)");
        };
    }

    private static Attention.Gqa attention(Map<String, Object> c, int dim, double scale) {
        int heads = Families.requirePositive(c, "num_attention_heads");
        int kv = (int) Json.num(c, "num_key_value_heads", heads);
        int headDim = (int) Json.num(c, "head_dim", Json.num(c, "attention_head_dim", dim / heads));
        if (kv <= 0 || heads % kv != 0 || headDim <= 0) {
            throw new IllegalArgumentException("invalid attention shape: heads=" + heads + " kv_heads=" + kv + " head_dim="
                    + headDim);
        }
        if (Json.bool(c, "attention_bias", false)) {
            throw new IllegalArgumentException("attention_bias=true is not supported for the hybrid families");
        }
        return new Attention.Gqa(heads, kv, headDim, 0, 0, false, false, false,
                (float) (Double.isNaN(scale) ? 1.0 / Math.sqrt(headDim) : scale), 0f);
    }

    private static RopeTable identityRope(Attention.Gqa a) {
        return new RopeTable(a.headDim() & ~1, 10000.0, Scaling.IDENTITY, RopeTable.Pairing.HALF);
    }

    private static HybridArch build(String arch, String modelType, Map<String, Object> c, int dim, int vocab, double eps,
                                    List<Layer> layers, Attention.Gqa attn, RopeTable rope, HybridArch.Mamba2 m2,
                                    HybridArch.Mamba1 m1, float embedScale, float logitsDivisor, String extra) {
        StringBuilder pattern = new StringBuilder();
        for (Layer l : layers) {
            pattern.append(switch (l.mixer()) {
                case ATTENTION -> 'A';
                case MAMBA2 -> 'M';
                case MAMBA1 -> 'J';
                case NONE -> '-';
            }).append(l.ffn() == null ? "" : l.ffn() instanceof Mlp.Moe ? "e" : "f");
        }
        String description = String.format("%s dim=%d layers=%d vocab=%d pattern=%s%s%s%s%s", arch, dim, layers.size(), vocab,
                pattern,
                attn == null ? "" : String.format(" attn(%d/%d x %d scale=%g%s)", attn.heads(), attn.kvHeads(), attn.headDim(),
                        attn.scale(), rope.scaling() instanceof Scaling.Identity ? " nope" : " rope"),
                m2 == null ? "" : " mamba2(" + m2 + ")", m1 == null ? "" : " mamba1(" + m1 + ")", extra);
        Core core = new Core(arch, modelType, dim, layers.size(), vocab, (float) eps, Json.bool(c, "tie_word_embeddings", false),
                (int) Json.num(c, "bos_token_id", -1), Families.intList(c, "eos_token_id"),
                (int) Json.num(c, "max_position_embeddings", 4096), embedScale, 0f, false, description);
        return new HybridArch(core, layers, attn, rope, m2, m1, logitsDivisor);
    }

    // ------------------------------------------------------------------ Granite 4

    /**
     * {@code granitemoehybrid}: {@code layer_types} of mamba / attention; every layer has the shared MLP
     * ({@code shared_intermediate_size}) and, with {@code num_local_experts > 0}, a MoE of experts of width
     * {@code intermediate_size} (router: top-k of the logits, then a softmax over them); the multipliers of the
     * muP parametrization; attention without RoPE unless {@code position_embedding_type} is "rope".
     */
    static HybridArch granite(Map<String, Object> c, String arch) {
        int dim = Families.requirePositive(c, "hidden_size");
        int n = Families.requirePositive(c, "num_hidden_layers");
        int vocab = Families.requirePositive(c, "vocab_size");
        if (!Json.str(c, "hidden_act", "silu").equals("silu")) {
            throw new IllegalArgumentException("unsupported hidden_act '" + Json.str(c, "hidden_act", "silu") + "': only silu");
        }
        if (Json.bool(c, "mamba_proj_bias", false) || Json.bool(c, "attention_bias", false)) {
            throw new IllegalArgumentException("Granite 4 with mamba_proj_bias / attention_bias is not supported");
        }
        List<Object> types = Json.list(c, "layer_types");
        if (types == null) {
            types = Json.list(c, "layers_block_type");
        }
        boolean[] isAttn = new boolean[n];
        for (int l = 0; l < n; l++) {
            isAttn[l] = types != null && l < types.size() && kind(types.get(l)) == 'A';
        }
        if (types != null && types.size() < n) {
            throw new IllegalArgumentException("config.json: layer_types has " + types.size() + " entries for " + n + " layers");
        }
        int experts = (int) Json.num(c, "num_local_experts", 8);
        int topK = (int) Json.num(c, "num_experts_per_tok", 2);
        int moeDim = (int) Json.num(c, "intermediate_size", 0);
        int shared = (int) Json.num(c, "shared_intermediate_size", 1024);
        if (experts > 0 && (topK <= 0 || topK > experts || moeDim <= 0)) {
            throw new IllegalArgumentException("invalid MoE config: num_local_experts=" + experts + " num_experts_per_tok="
                    + topK + " intermediate_size=" + moeDim);
        }
        Mlp ffn = experts > 0 ? new Mlp.Moe(experts, topK, moeDim, false, new Router.TopLogits(false, false),
                new Mlp.Dense(shared, Activation.SILU)) : new Mlp.Dense(shared, Activation.SILU);
        float res = f(c, "residual_multiplier", 1.0);
        List<Layer> layers = new ArrayList<>(n);
        for (int l = 0; l < n; l++) {
            layers.add(new Layer(isAttn[l] ? Mixer.ATTENTION : Mixer.MAMBA2, ffn, res));
        }
        boolean anyAttn = false;
        for (boolean b : isAttn) {
            anyAttn |= b;
        }
        Attention.Gqa attn = anyAttn ? attention(c, dim, Json.dbl(c, "attention_multiplier", 1.0)) : null;
        RopeTable rope = null;
        if (attn != null) {
            if ("rope".equals(Json.str(c, "position_embedding_type", null))) {
                Map<String, Object> rp = Json.map(c, "rope_parameters");
                double theta = Json.dbl(c, "rope_theta", Json.dbl(rp, "rope_theta", 10000));
                Map<String, Object> rs = Json.map(c, "rope_scaling");
                if (rs == null && rp != null && (rp.containsKey("rope_type") || rp.containsKey("type"))) {
                    rs = rp;
                }
                rope = new RopeTable(attn.headDim() & ~1, theta, Families.parseRopeScaling(rs, c, attn.headDim() & ~1),
                        RopeTable.Pairing.HALF);
            } else {
                rope = identityRope(attn);
            }
        }
        int heads = (int) Json.num(c, "mamba_n_heads", 128);
        int expand = (int) Json.num(c, "mamba_expand", 2);
        Object dh = c.get("mamba_d_head");
        int headDim = dh instanceof Number num ? num.intValue() : expand * dim / Math.max(1, heads);
        int groups = (int) Json.num(c, "mamba_n_groups", 1);
        int state = (int) Json.num(c, "mamba_d_state", 256);
        int conv = (int) Json.num(c, "mamba_d_conv", 4);
        List<Object> limit = Json.list(c, "time_step_limit");
        float dtMin = limit != null && limit.size() == 2 && limit.get(0) instanceof Number a ? a.floatValue() : 0f;
        float dtMax = limit != null && limit.size() == 2 && limit.get(1) instanceof Number b ? b.floatValue() : Float.POSITIVE_INFINITY;
        double eps = Json.dbl(c, "rms_norm_eps", 1e-6);
        HybridArch.Mamba2 m2 = null;
        if (countFalse(isAttn) > 0) {
            if (heads * headDim != expand * dim || groups <= 0 || heads % groups != 0 || state <= 0 || conv < 1 || conv > 8) {
                throw new IllegalArgumentException("invalid mamba config: mamba_n_heads=" + heads + " mamba_d_head=" + headDim
                        + " expand=" + expand + " hidden_size=" + dim + " n_groups=" + groups + " d_state=" + state
                        + " d_conv=" + conv);
            }
            m2 = new HybridArch.Mamba2(heads, headDim, groups, state, conv, 1, dtMin, dtMax, (float) eps);
        }
        return build(arch, "granitemoehybrid", c, dim, vocab, eps, layers, attn, rope, m2, null, f(c, "embedding_multiplier", 1.0),
                f(c, "logits_scaling", 1.0), String.format(" mult(res=%g)%s", res,
                        experts > 0 ? String.format(" experts=%d/%d moeDim=%d shared=%d", topK, experts, moeDim, shared)
                                : " shared=" + shared));
    }

    private static int countFalse(boolean[] a) {
        int n = 0;
        for (boolean b : a) {
            n += b ? 0 : 1;
        }
        return n;
    }

    // ------------------------------------------------------------------ Nemotron-H

    /**
     * {@code nemotron_h}: {@code hybrid_override_pattern} ("M" Mamba-2, "*" attention, "-" a ReLU-squared MLP; layers hold
     * one mixer or one MLP, with a single norm), Mamba-2 with {@code n_groups} groups and a group-wise gated norm, the step
     * size clamped below at {@code time_step_min}, attention without positional rotation.
     */
    static HybridArch nemotron(Map<String, Object> c, String arch) {
        int dim = Families.requirePositive(c, "hidden_size");
        int vocab = Families.requirePositive(c, "vocab_size");
        List<Object> types = Json.list(c, "layers_block_type");
        if (types == null) {
            types = Json.list(c, "layer_types");
        }
        String pattern = Json.str(c, "hybrid_override_pattern", null);
        List<Character> kinds = new ArrayList<>();
        if (pattern != null) {
            for (char ch : pattern.toCharArray()) {
                kinds.add(kind(String.valueOf(ch)));
            }
        } else if (types != null) {
            for (Object t : types) {
                kinds.add(kind(t));
            }
        } else {
            throw new IllegalArgumentException("config.json: nemotron_h needs hybrid_override_pattern");
        }
        int n = kinds.size();
        if (c.containsKey("num_hidden_layers") && Json.num(c, "num_hidden_layers", n) != n) {
            throw new IllegalArgumentException("config.json: num_hidden_layers does not match hybrid_override_pattern");
        }
        if (kinds.contains('E')) {
            throw new IllegalArgumentException("Nemotron-H mixture-of-experts layers (\"E\") are not supported");
        }
        if (Json.bool(c, "use_bias", false) || Json.bool(c, "mlp_bias", false) || Json.bool(c, "attention_bias", false)) {
            throw new IllegalArgumentException("Nemotron-H with biases (use_bias / mlp_bias / attention_bias) is not supported");
        }
        if (!Json.str(c, "mamba_hidden_act", "silu").equals("silu") || !Json.str(c, "mlp_hidden_act", "relu2").equals("relu2")) {
            throw new IllegalArgumentException("Nemotron-H needs mamba_hidden_act silu and mlp_hidden_act relu2");
        }
        double eps = Json.dbl(c, "layer_norm_epsilon", Json.dbl(c, "rms_norm_eps", 1e-5));
        int ffnDim = (int) Json.num(c, "intermediate_size", 0);
        List<Layer> layers = new ArrayList<>(n);
        boolean anyAttn = false;
        boolean anyMamba = false;
        for (char k : kinds) {
            switch (k) {
                case 'M' -> {
                    layers.add(new Layer(Mixer.MAMBA2, null, 1f));
                    anyMamba = true;
                }
                case 'A' -> {
                    layers.add(new Layer(Mixer.ATTENTION, null, 1f));
                    anyAttn = true;
                }
                default -> {
                    if (ffnDim <= 0) {
                        throw new IllegalArgumentException("intermediate_size must be positive for the MLP layers");
                    }
                    layers.add(new Layer(Mixer.NONE, new Mlp.Dense(ffnDim, Activation.RELU2), 1f));
                }
            }
        }
        Attention.Gqa attn = anyAttn ? attention(c, dim, Double.NaN) : null;
        HybridArch.Mamba2 m2 = null;
        if (anyMamba) {
            int heads = Families.requirePositive(c, "mamba_num_heads");
            int headDim = Families.requirePositive(c, "mamba_head_dim");
            int groups = (int) Json.num(c, "n_groups", 8);
            int state = (int) Json.num(c, "ssm_state_size", 128);
            int conv = (int) Json.num(c, "conv_kernel", 4);
            if (groups <= 0 || heads % groups != 0 || (heads * headDim) % groups != 0 || state <= 0 || conv < 1 || conv > 8) {
                throw new IllegalArgumentException("invalid mamba config: mamba_num_heads=" + heads + " mamba_head_dim="
                        + headDim + " n_groups=" + groups + " ssm_state_size=" + state + " conv_kernel=" + conv);
            }
            // transformers: time_step_limit = (time_step_min, inf), whatever the config's time_step_limit says
            m2 = new HybridArch.Mamba2(heads, headDim, groups, state, conv, groups, f(c, "time_step_min", 0.001),
                    Float.POSITIVE_INFINITY, (float) eps);
        }
        return build(arch, "nemotron_h", c, dim, vocab, eps, layers, attn, attn == null ? null : identityRope(attn), m2, null, 1f,
                1f, ffnDim > 0 ? " relu2=" + ffnDim : "");
    }

    // ------------------------------------------------------------------ Jamba

    /**
     * {@code jamba}: an attention layer where {@code i % attn_layer_period == attn_layer_offset}, a Mamba-1 layer otherwise;
     * the feed-forward block is a MoE ({@code num_experts} of {@code intermediate_size}, softmax then top-k, not
     * renormalized) where {@code i % expert_layer_period == expert_layer_offset}, else a SwiGLU MLP. Attention has no
     * positional rotation; the Mamba blocks normalize dt, B and C.
     */
    static HybridArch jamba(Map<String, Object> c, String arch) {
        int dim = Families.requirePositive(c, "hidden_size");
        int n = Families.requirePositive(c, "num_hidden_layers");
        int vocab = Families.requirePositive(c, "vocab_size");
        if (!Json.str(c, "hidden_act", "silu").equals("silu")) {
            throw new IllegalArgumentException("unsupported hidden_act '" + Json.str(c, "hidden_act", "silu") + "': only silu");
        }
        if (Json.bool(c, "mamba_proj_bias", false) || Json.bool(c, "attention_bias", false)) {
            throw new IllegalArgumentException("Jamba with mamba_proj_bias / attention_bias is not supported");
        }
        int attnPeriod = (int) Json.num(c, "attn_layer_period", 8);
        int attnOffset = (int) Json.num(c, "attn_layer_offset", 4);
        int expPeriod = (int) Json.num(c, "expert_layer_period", 2);
        int expOffset = (int) Json.num(c, "expert_layer_offset", 1);
        int experts = (int) Json.num(c, "num_experts", Json.num(c, "num_local_experts", 16));
        int topK = (int) Json.num(c, "num_experts_per_tok", 2);
        int ffn = (int) Json.num(c, "intermediate_size", 14336);
        if (attnPeriod <= 0 || expPeriod <= 0 || ffn <= 0 || (experts > 1 && (topK <= 0 || topK > experts))) {
            throw new IllegalArgumentException("invalid Jamba config: attn_layer_period=" + attnPeriod + " expert_layer_period="
                    + expPeriod + " intermediate_size=" + ffn + " num_experts=" + experts + " num_experts_per_tok=" + topK);
        }
        double eps = Json.dbl(c, "rms_norm_eps", 1e-6);
        List<Layer> layers = new ArrayList<>(n);
        boolean anyAttn = false;
        boolean anyMamba = false;
        for (int l = 0; l < n; l++) {
            boolean attn = l % attnPeriod == attnOffset;
            boolean moe = experts > 1 && l % expPeriod == expOffset;
            layers.add(new Layer(attn ? Mixer.ATTENTION : Mixer.MAMBA1, moe
                    ? new Mlp.Moe(experts, topK, ffn, false, Router.SOFTMAX, null) : new Mlp.Dense(ffn, Activation.SILU), 1f));
            anyAttn |= attn;
            anyMamba |= !attn;
        }
        Attention.Gqa attn = anyAttn ? attention(c, dim, Double.NaN) : null;
        HybridArch.Mamba1 m1 = null;
        if (anyMamba) {
            int state = (int) Json.num(c, "mamba_d_state", 16);
            int conv = (int) Json.num(c, "mamba_d_conv", 4);
            int di = (int) Json.num(c, "mamba_expand", 2) * dim;
            Object rank = c.get("mamba_dt_rank");
            int dtRank = rank instanceof Number num ? num.intValue() : (dim + 15) / 16;
            if (state <= 0 || conv < 1 || conv > 8 || dtRank <= 0) {
                throw new IllegalArgumentException("invalid mamba config: d_state=" + state + " d_conv=" + conv + " dt_rank=" + dtRank);
            }
            m1 = new HybridArch.Mamba1(di, state, conv, dtRank, (float) eps);
        }
        return build(arch, "jamba", c, dim, vocab, eps, layers, attn, attn == null ? null : identityRope(attn), null, m1, 1f, 1f,
                String.format(" ffn=%d experts=%d/%d", ffn, topK, experts));
    }
}
