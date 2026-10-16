package org.robomiri.jllm.model;

import java.util.List;

/**
 * A decoder-only transformer: embedding, {@code layers} × (RMSNorm → attention → residual → RMSNorm → MLP →
 * residual), final RMSNorm, LM head. Each layer is a {@link LayerSpec}; the rotary tables the attention layers
 * refer to (by index) are listed once.
 *
 * <p>Every layer has the same attention variant and head shape (a backend sizes its caches from it); window,
 * rotary table, scale and the MLP may differ per layer.
 */
public record Decoder(Core core, List<LayerSpec> layers, List<RopeTable> ropeTables) implements Architecture {

    public Decoder {
        layers = List.copyOf(layers);
        ropeTables = List.copyOf(ropeTables);
        if (layers.size() != core.layers()) {
            throw new IllegalArgumentException(layers.size() + " layer specs for " + core.layers() + " layers");
        }
        if (layers.isEmpty()) {
            throw new IllegalArgumentException("a decoder needs at least one layer");
        }
        Attention first = layers.getFirst().attn();
        for (LayerSpec s : layers) {
            Attention a = s.attn();
            if (a.ropeTable() < 0 || a.ropeTable() >= ropeTables.size()) {
                throw new IllegalArgumentException("rope table " + a.ropeTable() + " of " + ropeTables.size());
            }
            if (!a.sameShape(first)) {
                throw new IllegalArgumentException("attention shapes differ between layers: " + first + " vs " + a);
            }
        }
    }

    /** The attention of layer 0: its variant and head shape are those of every layer. */
    public Attention attention() {
        return layers.getFirst().attn();
    }

    /** Width of the widest dense GLU (the dense layers' hidden size or the shared experts'), 0 if there is none. */
    public int maxMlpDim() {
        int m = 0;
        for (LayerSpec s : layers) {
            m = Math.max(m, switch (s.mlp()) {
                case Mlp.Dense d -> d.hidden();
                case Mlp.Moe moe -> moe.shared() == null ? 0 : moe.shared().hidden();
            });
        }
        return m;
    }

    public boolean anyMoe() {
        return maxExperts() > 0;
    }

    /** Routed experts of the widest MoE layer, 0 for a dense model. */
    public int maxExperts() {
        int m = 0;
        for (LayerSpec s : layers) {
            if (s.mlp() instanceof Mlp.Moe moe) {
                m = Math.max(m, moe.experts());
            }
        }
        return m;
    }

    public int maxTopK() {
        int m = 0;
        for (LayerSpec s : layers) {
            if (s.mlp() instanceof Mlp.Moe moe) {
                m = Math.max(m, moe.topK());
            }
        }
        return m;
    }

    public int maxMoeDim() {
        int m = 0;
        for (LayerSpec s : layers) {
            if (s.mlp() instanceof Mlp.Moe moe) {
                m = Math.max(m, moe.moeDim());
            }
        }
        return m;
    }

    /** True if any layer uses rotary table {@code t}. */
    public boolean usesRopeTable(int t) {
        for (LayerSpec s : layers) {
            if (s.attn().ropeTable() == t) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return core.description();
    }
}
