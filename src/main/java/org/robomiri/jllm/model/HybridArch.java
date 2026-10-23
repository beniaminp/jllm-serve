package org.robomiri.jllm.model;

import java.util.List;

/**
 * The hybrids of attention and state-space layers (docs/14.22): IBM Granite 4 ({@code GraniteMoeHybridForCausalLM}),
 * NVIDIA Nemotron-H ({@code NemotronHForCausalLM}) and AI21 Jamba ({@code JambaForCausalLM}). Every layer is
 * {@code x += r * mixer(norm1(x))} followed by {@code x += r * ffn(norm2(x))}, where the mixer is grouped-query attention
 * (no rotary embedding in these three; Granite can have one), a Mamba-2 state-space block or a Mamba-1 block (Jamba),
 * and the feed-forward block is a gated SwiGLU, a plain ReLU-squared MLP (Nemotron-H), a mixture of experts (Granite 4:
 * top-k logits then a softmax, plus the shared MLP every layer has; Jamba: softmax then top-k) or absent. A Nemotron-H
 * layer has either a mixer or a feed-forward block, never both, and one norm. Only the attention layers hold a KV cache;
 * the state-space layers keep a fixed-size recurrent state ({@link Mamba2}, {@link Mamba1}).
 *
 * <p>Granite's multipliers: the embedding rows are scaled ({@code core.embedScale}), the residual branches by
 * {@link Layer#residual}, the attention scores by {@code attention_multiplier} (the {@link Attention.Gqa#scale}), the
 * logits divided by {@code logits_scaling} ({@link #logitsDivisor}).
 *
 * @param attention the attention layers' shape (null if there are none); no biases, no q/k norm
 * @param rope      the rotary table of the attention layers: identity (no positional rotation) unless the config has one
 * @param mamba2    the Mamba-2 block's shape (null if no layer has one)
 * @param mamba1    the Mamba-1 block's shape (null if no layer has one)
 */
public record HybridArch(Core core, List<Layer> layers, Attention.Gqa attention, RopeTable rope, Mamba2 mamba2,
                         Mamba1 mamba1, float logitsDivisor) implements Architecture {

    /** What mixes the tokens of a layer. */
    public enum Mixer {
        ATTENTION, MAMBA2, MAMBA1,
        /** Nemotron-H's feed-forward-only layers: the layer is just {@link Layer#ffn}. */
        NONE
    }

    /**
     * @param ffn      the feed-forward block, or null (a Nemotron-H mixer layer); its norm is {@code post_attention_layernorm}
     *                 ({@code input_layernorm} when the layer has no mixer)
     * @param residual multiplier of both branches (Granite's {@code residual_multiplier}, else 1)
     */
    public record Layer(Mixer mixer, Mlp ffn, float residual) {
    }

    /**
     * A Mamba-2 (SSD) block. {@code in_proj} gives {@code [z (dInner) | x (dInner) | B (groups*state) | C (groups*state)
     * | dt (heads)]}; the conv covers {@code [x | B | C]} ({@link #convDim}); each of the {@code heads} heads has a
     * state {@code headDim x state} and reads group {@code h / (heads / groups)}'s B and C.
     *
     * @param normGroups the gated RMSNorm covers {@code dInner / normGroups} channels at a time (Nemotron-H: its
     *                   n_groups; Granite 4: one group, the whole width)
     * @param dtMin      the step size {@code softplus(dt + bias)} is clamped to {@code [dtMin, dtMax]}
     */
    public record Mamba2(int heads, int headDim, int groups, int state, int convKernel, int normGroups, float dtMin,
                         float dtMax, float normEps) {

        public int dInner() {
            return heads * headDim;
        }

        public int convDim() {
            return dInner() + 2 * groups * state;
        }

        public int projDim() {
            return dInner() + convDim() + heads;
        }

        /** Floats of the conv state: the last {@code convKernel - 1} inputs of every conv channel. */
        public int convStateSize() {
            return convDim() * (convKernel - 1);
        }

        /** Floats of the SSM state: {@code heads x headDim x state}. */
        public int ssmStateSize() {
            return heads * headDim * state;
        }
    }

    /**
     * A Mamba-1 block (docs/15) with Jamba's three extra RMS norms on the {@code dt} (dtRank), {@code B} and {@code C}
     * (state) outputs of {@code x_proj}.
     */
    public record Mamba1(int dInner, int state, int convKernel, int dtRank, float normEps) {

        public int convStateSize() {
            return dInner * (convKernel - 1);
        }

        public int ssmStateSize() {
            return dInner * state;
        }
    }

    public HybridArch {
        layers = List.copyOf(layers);
        if (layers.size() != core.layers()) {
            throw new IllegalArgumentException(layers.size() + " layers for " + core.layers());
        }
    }

    /** Number of attention layers (the ones with a KV cache). */
    public int attentionLayers() {
        return count(Mixer.ATTENTION);
    }

    public int count(Mixer m) {
        int n = 0;
        for (Layer l : layers) {
            n += l.mixer() == m ? 1 : 0;
        }
        return n;
    }

    /** Index of layer {@code l} among the layers of its own mixer kind (its slot in the KV cache or in the states). */
    public int kindIndex(int l) {
        Mixer m = layers.get(l).mixer();
        int n = 0;
        for (int i = 0; i < l; i++) {
            n += layers.get(i).mixer() == m ? 1 : 0;
        }
        return n;
    }

    public boolean anyMoe() {
        return maxExperts() > 0;
    }

    public int maxExperts() {
        int m = 0;
        for (Layer l : layers) {
            if (l.ffn() instanceof Mlp.Moe moe) {
                m = Math.max(m, moe.experts());
            }
        }
        return m;
    }

    public int maxTopK() {
        int m = 0;
        for (Layer l : layers) {
            if (l.ffn() instanceof Mlp.Moe moe) {
                m = Math.max(m, moe.topK());
            }
        }
        return m;
    }

    public int maxMoeDim() {
        int m = 0;
        for (Layer l : layers) {
            if (l.ffn() instanceof Mlp.Moe moe) {
                m = Math.max(m, moe.moeDim());
            }
        }
        return m;
    }

    /** Width of the widest dense block: a dense feed-forward layer's or a MoE layer's shared MLP. */
    public int maxMlpDim() {
        int m = 0;
        for (Layer l : layers) {
            m = Math.max(m, switch (l.ffn()) {
                case null -> 0;
                case Mlp.Dense d -> d.hidden();
                case Mlp.Moe moe -> moe.shared() == null ? 0 : moe.shared().hidden();
            });
        }
        return m;
    }

    /** Width of the widest mixer output (the input of its output projection). */
    public int maxMixerWidth() {
        int m = attention != null ? attention.outDim() : 0;
        if (mamba2 != null) {
            m = Math.max(m, mamba2.dInner());
        }
        if (mamba1 != null) {
            m = Math.max(m, mamba1.dInner());
        }
        return m;
    }

    @Override
    public String toString() {
        return core.description();
    }
}
