package org.robomiri.jllm.model;

import org.robomiri.jllm.json.Json;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Qwen3-Next ({@code Qwen3NextForCausalLM}, e.g. Qwen3-Next-80B-A3B): a hybrid of linear and full attention
 * (docs/14.14). Every layer is {@code x += mixer(norm(x)); x += mlp(norm(x))}; the mixer is a <b>Gated DeltaNet</b>
 * (linear attention with a fixed-size recurrent state per head, a short causal conv and an output-gated norm) in
 * most layers and <b>gated full attention</b> (an output gate {@code sigmoid(gate) * attn}, per-head q/k norm,
 * RoPE on a {@code partial_rotary_factor} slice of each head) in every {@code full_attention_interval}-th layer.
 * The MLP is a dense SwiGLU or a high-sparsity mixture of experts with one shared expert whose output is scaled by
 * {@code sigmoid(shared_expert_gate . x)}. Only the full-attention layers hold a KV cache.
 *
 * <p>Every RMSNorm except the DeltaNet's gated one is zero-centred ({@code x̂ · (1 + w)}); the converter folds the 1 in,
 * so the backends see plain gains.
 *
 * @param fullAttention per layer: true = gated full attention, false = Gated DeltaNet
 * @param attention     the full-attention layers' shape: q/k norm on, no bias, scale {@code head_dim^-0.5}; q and the
 *                      output gate come from one {@code q_proj} that the converter splits
 * @param rope          the rotary table of the full-attention layers (rotate-half over {@code rotaryDim} dims)
 * @param deltaNet      the linear-attention layers' shape
 * @param mlps          per layer: {@link Mlp.Dense}, or {@link Mlp.Moe} whose {@code shared} is the gated shared expert
 */
public record Qwen3NextArch(Core core, List<Boolean> fullAttention, Attention.Gqa attention, RopeTable rope,
                            DeltaNet deltaNet, List<Mlp> mlps) implements Architecture {

    public Qwen3NextArch {
        fullAttention = List.copyOf(fullAttention);
        mlps = List.copyOf(mlps);
        if (fullAttention.size() != core.layers() || mlps.size() != core.layers()) {
            throw new IllegalArgumentException("layer lists do not match " + core.layers() + " layers");
        }
    }

    /**
     * The Gated DeltaNet of the linear-attention layers.
     *
     * @param keyHeads   {@code linear_num_key_heads}: heads of q and k
     * @param valueHeads {@code linear_num_value_heads}: heads of v, of the recurrent state and of the output; a
     *                   multiple of keyHeads (each q/k head serves {@code valueHeads / keyHeads} value heads)
     * @param keyDim     {@code linear_key_head_dim}
     * @param valueDim   {@code linear_value_head_dim}
     * @param convKernel {@code linear_conv_kernel_dim}: taps of the depthwise causal conv over [q | k | v]
     */
    public record DeltaNet(int keyHeads, int valueHeads, int keyDim, int valueDim, int convKernel) {

        /** Width of all q (or k) heads of a token. */
        public int keyWidth() {
            return keyHeads * keyDim;
        }

        /** Width of all v heads (also of the output and of the gate z). */
        public int valueWidth() {
            return valueHeads * valueDim;
        }

        /** Channels of the causal conv: q, k and v. */
        public int convDim() {
            return 2 * keyWidth() + valueWidth();
        }

        /** Rows of the fused input projection the converter writes: [q | k | v | z | b | a]. */
        public int projDim() {
            return convDim() + valueWidth() + 2 * valueHeads;
        }

        /** Value heads per key head. */
        public int group() {
            return valueHeads / keyHeads;
        }

        /** Floats of one layer's recurrent state: {@code valueHeads x valueDim x keyDim}. */
        public int stateSize() {
            return valueHeads * valueDim * keyDim;
        }

        /** Floats of one layer's conv state: the last {@code convKernel - 1} inputs of every channel. */
        public int convStateSize() {
            return convDim() * (convKernel - 1);
        }
    }

    public boolean isFull(int layer) {
        return fullAttention.get(layer);
    }

    /** Number of full-attention layers (the ones with a KV cache). */
    public int fullLayers() {
        int n = 0;
        for (boolean f : fullAttention) {
            n += f ? 1 : 0;
        }
        return n;
    }

    /** Number of Gated DeltaNet layers. */
    public int linearLayers() {
        return core.layers() - fullLayers();
    }

    /** Index of layer {@code l} among the layers of its own kind (its slot in the KV cache or in the recurrent states). */
    public int kindIndex(int l) {
        boolean full = isFull(l);
        int n = 0;
        for (int i = 0; i < l; i++) {
            n += isFull(i) == full ? 1 : 0;
        }
        return n;
    }

    public boolean anyMoe() {
        return maxExperts() > 0;
    }

    public int maxExperts() {
        int m = 0;
        for (Mlp p : mlps) {
            if (p instanceof Mlp.Moe moe) {
                m = Math.max(m, moe.experts());
            }
        }
        return m;
    }

    public int maxTopK() {
        int m = 0;
        for (Mlp p : mlps) {
            if (p instanceof Mlp.Moe moe) {
                m = Math.max(m, moe.topK());
            }
        }
        return m;
    }

    public int maxMoeDim() {
        int m = 0;
        for (Mlp p : mlps) {
            if (p instanceof Mlp.Moe moe) {
                m = Math.max(m, moe.moeDim());
            }
        }
        return m;
    }

    /** Width of the widest dense SwiGLU: a dense layer's or the shared expert's. */
    public int maxMlpDim() {
        int m = 0;
        for (Mlp p : mlps) {
            m = Math.max(m, switch (p) {
                case Mlp.Dense d -> d.hidden();
                case Mlp.Moe moe -> moe.shared() == null ? 0 : moe.shared().hidden();
            });
        }
        return m;
    }

    @Override
    public String toString() {
        return core.description();
    }

    // ------------------------------------------------------------------ config.json

    /**
     * Resolves a {@code qwen3_next} config with the defaults of transformers' {@code Qwen3NextConfig}: head_dim 256,
     * partial_rotary_factor 0.25, layer_types from {@code full_attention_interval} (4: layers 3, 7, ... are full
     * attention), the linear_* sizes 16 x 128 keys / 32 x 128 values / conv 4, and the MoE rule of Qwen3-MoE
     * (layer l routes unless in mlp_only_layers or (l + 1) % decoder_sparse_step != 0, given num_experts > 0) with a
     * shared expert of {@code shared_expert_intermediate_size}.
     */
    static Qwen3NextArch parse(Map<String, Object> c, String arch) {
        int dim = Families.requirePositive(c, "hidden_size");
        int layers = Families.requirePositive(c, "num_hidden_layers");
        int vocab = Families.requirePositive(c, "vocab_size");
        int heads = Families.requirePositive(c, "num_attention_heads");
        int kvHeads = (int) Json.num(c, "num_key_value_heads", 2);
        int headDim = (int) Json.num(c, "head_dim", 256);
        if (kvHeads <= 0 || heads % kvHeads != 0 || headDim <= 0) {
            throw new IllegalArgumentException("invalid attention shape: heads=" + heads + " kv_heads=" + kvHeads
                    + " head_dim=" + headDim);
        }
        String act = Json.str(c, "hidden_act", "silu");
        if (!act.equals("silu")) {
            throw new IllegalArgumentException("unsupported hidden_act '" + act + "': only silu is implemented");
        }
        if (Json.bool(c, "attention_bias", false)) {
            throw new IllegalArgumentException("qwen3_next with attention_bias=true is not supported");
        }
        // rope: transformers 4 writes the top-level keys, transformers 5 nests them in rope_parameters
        Map<String, Object> rp = Json.map(c, "rope_parameters");
        double theta = Json.dbl(c, "rope_theta", Json.dbl(rp, "rope_theta", 10000));
        double partial = Json.dbl(c, "partial_rotary_factor", Json.dbl(rp, "partial_rotary_factor", 0.25));
        int rotaryDim = (int) (headDim * partial) & ~1;
        if (!(partial > 0) || partial > 1 || rotaryDim < 2) {
            throw new IllegalArgumentException("invalid partial_rotary_factor " + partial);
        }
        Map<String, Object> rs = Json.map(c, "rope_scaling");
        if (rs == null && rp != null && (rp.containsKey("rope_type") || rp.containsKey("type"))) {
            rs = rp;
        }
        Scaling scaling = Families.parseRopeScaling(rs, c, rotaryDim);
        RopeTable rope = new RopeTable(rotaryDim, theta, scaling, RopeTable.Pairing.HALF);
        Attention.Gqa attn = new Attention.Gqa(heads, kvHeads, headDim, 0, 0, true, false, false,
                (float) (1.0 / Math.sqrt(headDim)), 0f);

        DeltaNet dn = new DeltaNet((int) Json.num(c, "linear_num_key_heads", 16), (int) Json.num(c, "linear_num_value_heads", 32),
                (int) Json.num(c, "linear_key_head_dim", 128), (int) Json.num(c, "linear_value_head_dim", 128),
                (int) Json.num(c, "linear_conv_kernel_dim", 4));
        if (dn.keyHeads() <= 0 || dn.valueHeads() <= 0 || dn.valueHeads() % dn.keyHeads() != 0 || dn.keyDim() <= 0
                || dn.valueDim() <= 0 || dn.convKernel() < 1 || dn.convKernel() > 8) {
            throw new IllegalArgumentException("invalid linear attention config: " + dn
                    + " (value heads a multiple of key heads, conv kernel 1..8)");
        }

        List<Boolean> full = new ArrayList<>(layers);
        List<Object> types = Json.list(c, "layer_types");
        if (types != null) {
            if (types.size() < layers) {
                throw new IllegalArgumentException("config.json: layer_types has " + types.size() + " entries for "
                        + layers + " layers");
            }
            for (int l = 0; l < layers; l++) {
                Object t = types.get(l);
                if (!"linear_attention".equals(t) && !"full_attention".equals(t)) {
                    throw new IllegalArgumentException("config.json: unsupported layer_types[" + l + "] = " + t
                            + " (linear_attention, full_attention)");
                }
                full.add("full_attention".equals(t));
            }
        } else {
            int interval = (int) Json.num(c, "full_attention_interval", 4);
            if (interval <= 0) {
                throw new IllegalArgumentException("full_attention_interval must be positive, got " + interval);
            }
            for (int l = 0; l < layers; l++) {
                full.add((l + 1) % interval == 0);
            }
        }

        int experts = (int) Json.num(c, "num_experts", 512);
        int topK = (int) Json.num(c, "num_experts_per_tok", 10);
        int moeDim = (int) Json.num(c, "moe_intermediate_size", 512);
        int sharedDim = (int) Json.num(c, "shared_expert_intermediate_size", 512);
        int hidden = (int) Json.num(c, "intermediate_size", 5632);
        boolean normTopK = Json.bool(c, "norm_topk_prob", true);
        int sparseStep = (int) Json.num(c, "decoder_sparse_step", 1);
        Set<Integer> mlpOnly = Set.copyOf(Families.intList(c, "mlp_only_layers"));
        if (experts > 0 && (topK <= 0 || topK > experts || moeDim <= 0 || sharedDim <= 0 || sparseStep <= 0)) {
            throw new IllegalArgumentException("invalid MoE config: num_experts=" + experts + " num_experts_per_tok=" + topK
                    + " moe_intermediate_size=" + moeDim + " shared_expert_intermediate_size=" + sharedDim
                    + " decoder_sparse_step=" + sparseStep);
        }
        List<Mlp> mlps = new ArrayList<>(layers);
        StringBuilder kinds = new StringBuilder();
        for (int l = 0; l < layers; l++) {
            boolean moe = experts > 0 && !mlpOnly.contains(l) && (l + 1) % sparseStep == 0;
            if (!moe && hidden <= 0) {
                throw new IllegalArgumentException("intermediate_size must be positive for a dense MLP layer");
            }
            mlps.add(moe ? new Mlp.Moe(experts, topK, moeDim, normTopK, Router.SOFTMAX,
                    new Mlp.Dense(sharedDim, Mlp.Activation.SILU)) : new Mlp.Dense(hidden, Mlp.Activation.SILU));
            kinds.append(full.get(l) ? (moe ? 'A' : 'a') : (moe ? 'L' : 'l'));
        }

        String description = String.format("%s dim=%d ffn=%d layers=%d heads=%d/%d headDim=%d rotary=%d vocab=%d theta=%.0f"
                        + " gdn(k=%dx%d v=%dx%d conv=%d) kinds=%s%s%s", arch, dim, hidden, layers, heads, kvHeads, headDim,
                rotaryDim, vocab, theta, dn.keyHeads(), dn.keyDim(), dn.valueHeads(), dn.valueDim(), dn.convKernel(), kinds,
                experts > 0 ? String.format(" experts=%d/%d moeDim=%d shared=%d%s", topK, experts, moeDim, sharedDim,
                        normTopK ? "" : " unnormalized") : "",
                scaling instanceof Scaling.None ? "" : " scaled-rope");
        Core core = new Core(arch, "qwen3_next", dim, layers, vocab, (float) Json.dbl(c, "rms_norm_eps", 1e-6),
                Json.bool(c, "tie_word_embeddings", false), (int) Json.num(c, "bos_token_id", -1),
                Families.intList(c, "eos_token_id"), (int) Json.num(c, "max_position_embeddings", 32768), 1f, 0f, false,
                description);
        return new Qwen3NextArch(core, full, attn, rope, dn, mlps);
    }
}
