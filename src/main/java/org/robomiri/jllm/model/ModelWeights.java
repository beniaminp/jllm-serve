package org.robomiri.jllm.model;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Quantized weights of a Llama / Mistral / Qwen2 / Qwen3 (dense or MoE) decoder, memory-mapped from a {@link QuantFile}.
 *
 * <p>Linear layers use the requested format (Q4_0 by default); the embedding / LM head always uses Q8_0 because
 * small models are sensitive to output-layer precision. Gate and up projections are fused into one matrix; in MoE
 * layers all experts are stacked into one gate/up tensor and one down tensor.
 */
public final class ModelWeights implements AutoCloseable {

    public static final class Layer {
        /** Mamba block weights; for a Mamba model this is the only field set. */
        public MambaLayer mamba;
        public float[] attnNorm;
        public float[] ffnNorm;
        /** Glm4: RMSNorm of the attention / MLP output before the residual add; null for other models. */
        public float[] postAttnNorm;
        public float[] postMlpNorm;
        /** Per-head RMSNorm on queries/keys (Qwen3); null for models without it. */
        public float[] qNorm;
        public float[] kNorm;
        /** q/k/v projection biases (Qwen2); null for models without them. */
        public float[] bq;
        public float[] bk;
        public float[] bv;
        public QTensor wq;
        public QTensor wk;
        public QTensor wv;
        public QTensor wo;
        /** Dense MLP: rows {@code [0, hiddenDim)} are gate_proj, rows {@code [hiddenDim, 2*hiddenDim)} are up_proj. */
        public QTensor wGateUp;
        public QTensor wDown;
        /** MoE: true when this layer routes tokens to experts instead of using {@link #wGateUp}. */
        public boolean moe;
        /** MoE router {@code [numExperts × dim]}, f32. */
        public float[] router;
        /** MoE: expert {@code e} gate/up is rows {@code [e*2*moeDim, (e+1)*2*moeDim)} (gate first, then up). */
        public QTensor expGateUp;
        /** MoE: expert {@code e} down is rows {@code [e*dim, (e+1)*dim)}, cols = moeDim. */
        public QTensor expDown;
        /** DeepSeek: MLA projections, shared experts and router bias (wq/wk/wv are null); null for other families. */
        public MlaWeights mla;
    }

    public final ModelConfig cfg;
    public final QType quant;
    public final QTensor embed;
    public final QTensor lmHead;
    public final Layer[] layers;
    public final float[] finalNorm;
    private final QuantFile file;

    private ModelWeights(ModelConfig cfg, QType quant, QuantFile file) {
        this.cfg = cfg;
        this.quant = quant;
        this.file = file;
        embed = file.tensor(cfg.isMamba() ? "backbone.embeddings.weight" : "model.embed_tokens.weight");
        lmHead = file.has("lm_head.weight") ? file.tensor("lm_head.weight") : embed;
        // The engine sizes its logits from vocabSize and backends write lmHead.rows outputs, so both must agree.
        if (embed.cols != cfg.dim() || embed.rows != cfg.vocabSize()
                || lmHead.cols != cfg.dim() || lmHead.rows != cfg.vocabSize()) {
            throw new IllegalArgumentException("embedding " + embed.rows + "x" + embed.cols + " / lm_head "
                    + lmHead.rows + "x" + lmHead.cols + " do not match config " + cfg + STALE_HINT);
        }
        layers = new Layer[cfg.layers()];
        if (cfg.isMamba()) {
            for (int l = 0; l < cfg.layers(); l++) {
                layers[l] = new Layer();
                layers[l].mamba = MambaLayer.load(file, "backbone.layers." + l + ".", cfg);
            }
            finalNorm = file.floats("backbone.norm_f.weight");
            if (finalNorm.length != cfg.dim()) {
                throw new IllegalArgumentException("backbone.norm_f.weight has " + finalNorm.length
                        + " entries, expected " + cfg.dim() + STALE_HINT);
            }
            return;
        }
        for (int l = 0; l < cfg.layers(); l++) {
            String p = "model.layers." + l + ".";
            Layer L = new Layer();
            L.attnNorm = file.floats(p + "input_layernorm.weight");
            if (cfg.isGemma()) {
                // Gemma: post_attention_layernorm is applied to the attention output; the MLP input norm is
                // pre_feedforward_layernorm. All four hold 1 + w (folded in by the converter).
                L.ffnNorm = file.floats(p + "pre_feedforward_layernorm.weight");
                L.postAttnNorm = file.floats(p + "post_attention_layernorm.weight");
                L.postMlpNorm = file.floats(p + "post_feedforward_layernorm.weight");
            } else {
                L.ffnNorm = file.floats(p + "post_attention_layernorm.weight");
                if (cfg.postNorms()) {
                    L.postAttnNorm = file.floats(p + "post_self_attn_layernorm.weight");
                    L.postMlpNorm = file.floats(p + "post_mlp_layernorm.weight");
                }
            }
            L.qNorm = file.has(p + "self_attn.q_norm.weight") ? file.floats(p + "self_attn.q_norm.weight") : null;
            L.kNorm = file.has(p + "self_attn.k_norm.weight") ? file.floats(p + "self_attn.k_norm.weight") : null;
            if (cfg.deepseek() != null) {
                L.mla = loadMla(file, cfg, l, p);
            } else {
                L.wq = file.tensor(p + "self_attn.q_proj.weight");
                L.wk = file.tensor(p + "self_attn.k_proj.weight");
                L.wv = file.tensor(p + "self_attn.v_proj.weight");
            }
            L.wo = file.tensor(p + "self_attn.o_proj.weight");
            if (cfg.attentionBias()) {
                L.bq = file.floats(p + "self_attn.q_proj.bias");
                L.bk = file.floats(p + "self_attn.k_proj.bias");
                L.bv = file.floats(p + "self_attn.v_proj.bias");
            }
            L.moe = file.has(p + "mlp.router");
            if (L.moe) {
                L.router = file.floats(p + "mlp.router");
                L.expGateUp = file.tensor(p + "mlp.experts.gate_up");
                L.expDown = file.tensor(p + "mlp.experts.down");
            } else {
                L.wGateUp = file.tensor(p + "mlp.gate_up");
                L.wDown = file.tensor(p + "mlp.down_proj.weight");
            }
            checkShapes(cfg, L, l);
            layers[l] = L;
        }
        finalNorm = file.floats("model.norm.weight");
        if (finalNorm.length != cfg.dim()) {
            throw new IllegalArgumentException("model.norm.weight has " + finalNorm.length + " entries, expected "
                    + cfg.dim() + STALE_HINT);
        }
    }

    private static MlaWeights loadMla(QuantFile file, ModelConfig cfg, int l, String p) {
        DeepSeekConfig d = cfg.deepseek();
        MlaWeights m = new MlaWeights();
        String a = p + "self_attn.";
        if (d.qLoraRank() > 0) {
            m.qA = file.tensor(a + "q_a_proj.weight");
            m.qALayernorm = file.floats(a + "q_a_layernorm.weight");
            m.qB = file.tensor(a + "q_b_proj.weight");
        } else {
            m.qProj = file.tensor(a + "q_proj.weight");
        }
        m.kvA = file.tensor(a + "kv_a_latent.weight");
        m.kPe = file.tensor(a + "k_pe_proj.weight");
        m.kvALayernorm = file.floats(a + "kv_a_layernorm.weight");
        m.kvB = file.tensor(a + "kv_b_proj.weight");
        if (cfg.isMoeLayer(l)) {
            if (d.nShared() > 0) {
                m.sharedGateUp = file.tensor(p + "mlp.shared.gate_up");
                m.sharedDown = file.tensor(p + "mlp.shared.down");
            }
            if (file.has(p + "mlp.router_bias")) {
                m.routerBias = file.floats(p + "mlp.router_bias");
            }
        }
        return m;
    }

    static final String STALE_HINT = " (stale cache)";

    /**
     * Opens the quantized cache in {@code dir}, converting the Hugging Face checkpoint first if there is none.
     *
     * <p>A cache that exists but cannot be used — written by another {@link QuantFile#LAYOUT_VERSION}, an
     * unparseable or corrupt index, or tensors whose shapes do not match {@code cfg} — is deleted and converted
     * again. The second attempt must succeed: a shape mismatch after a fresh conversion is a real inconsistency
     * between the checkpoint and its config and is reported as such.
     */
    public static ModelWeights load(Path dir, ModelConfig cfg, QType quant, ShardProvider shards) throws IOException {
        boolean converted = false;
        if (!QuantFile.exists(dir, quant)) {
            ModelConverter.convert(dir, cfg, quant, shards);
            converted = true;
        }
        while (true) {
            long t0 = System.nanoTime();
            String stale = null;
            try {
                QuantFile file = QuantFile.open(dir, quant);
                try {
                    ModelWeights w = new ModelWeights(cfg, quant, file);
                    System.err.printf("Mapped %s weights (%s, %.0f MB) in %.2fs%n", cfg.architecture(), quant,
                            file.byteSize() / 1e6, (System.nanoTime() - t0) / 1e9);
                    return w;
                } catch (IllegalArgumentException e) {
                    // a tensor is missing or has the wrong shape: the cache was written for another checkpoint or config
                    file.close();
                    stale = e.getMessage();
                } catch (RuntimeException e) {
                    file.close();
                    throw e;
                }
            } catch (QuantFile.StaleCacheException e) {
                stale = e.getMessage();
            }
            if (converted) {
                throw new IOException("the freshly converted cache " + QuantFile.binPath(dir, quant)
                        + " does not match the config: " + stale);
            }
            System.err.println("Reconverting " + QuantFile.binPath(dir, quant).getFileName() + ": " + stale);
            QuantFile.delete(dir, quant);
            ModelConverter.convert(dir, cfg, quant, shards);
            converted = true;
        }
    }

    /** Every shape a backend relies on: a mismatch here would otherwise surface as a silent out-of-bounds read. */
    private static void checkShapes(ModelConfig c, Layer L, int l) {
        if (c.deepseek() != null) {
            checkMlaShapes(c, L, l);
            return;
        }
        boolean ok = L.attnNorm.length == c.dim() && L.ffnNorm.length == c.dim()
                && L.wq.rows == c.qDim() && L.wq.cols == c.dim()
                && L.wk.rows == c.kvDim() && L.wk.cols == c.dim()
                && L.wv.rows == c.kvDim() && L.wv.cols == c.dim()
                && L.wo.rows == c.dim() && L.wo.cols == c.qDim()
                && (c.attentionBias() == (L.bq != null))
                && (L.bq == null || (L.bq.length == c.qDim() && L.bk.length == c.kvDim() && L.bv.length == c.kvDim()))
                && (c.postNorms() == (L.postAttnNorm != null) && c.postNorms() == (L.postMlpNorm != null))
                && (L.postAttnNorm == null || (L.postAttnNorm.length == c.dim() && L.postMlpNorm.length == c.dim()))
                && (c.qkNorm() == (L.qNorm != null) && c.qkNorm() == (L.kNorm != null))
                && (L.qNorm == null || L.qNorm.length == c.headDim())
                && (L.kNorm == null || L.kNorm.length == c.headDim());
        if (L.moe) {
            ok &= c.isMoeLayer(l)
                    && L.router.length == c.numExperts() * c.dim()
                    && L.expGateUp.rows == c.numExperts() * 2 * c.moeDim() && L.expGateUp.cols == c.dim()
                    && L.expDown.rows == c.numExperts() * c.dim() && L.expDown.cols == c.moeDim();
        } else {
            ok &= !c.isMoeLayer(l)
                    && L.wGateUp.rows == 2 * c.hiddenDim() && L.wGateUp.cols == c.dim()
                    && L.wDown.rows == c.dim() && L.wDown.cols == c.hiddenDim();
        }
        if (!ok) {
            throw new IllegalArgumentException("layer " + l + " weight shapes do not match config " + c + STALE_HINT);
        }
    }

    /** Shapes of a DeepSeek layer (MLA projections, dense / routed / shared MLP). */
    private static void checkMlaShapes(ModelConfig c, Layer L, int l) {
        DeepSeekConfig d = c.deepseek();
        MlaWeights m = L.mla;
        int h = c.heads();
        boolean ok = L.attnNorm.length == c.dim() && L.ffnNorm.length == c.dim()
                && (d.qLoraRank() > 0
                    ? m.qA.rows == d.qLoraRank() && m.qA.cols == c.dim() && m.qALayernorm.length == d.qLoraRank()
                        && m.qB.rows == h * d.qkDim() && m.qB.cols == d.qLoraRank()
                    : m.qProj.rows == h * d.qkDim() && m.qProj.cols == c.dim())
                && m.kvA.rows == d.kvLoraRank() && m.kvA.cols == c.dim()
                && m.kPe.rows == d.qkRopeDim() && m.kPe.cols == c.dim()
                && m.kvALayernorm.length == d.kvLoraRank()
                && m.kvB.rows == h * (d.qkNopeDim() + d.vHeadDim()) && m.kvB.cols == d.kvLoraRank()
                && L.wo.rows == c.dim() && L.wo.cols == h * d.vHeadDim();
        if (L.moe) {
            int sh = d.nShared() * c.moeDim();
            ok &= c.isMoeLayer(l)
                    && L.router.length == c.numExperts() * c.dim()
                    && L.expGateUp.rows == c.numExperts() * 2 * c.moeDim() && L.expGateUp.cols == c.dim()
                    && L.expDown.rows == c.numExperts() * c.dim() && L.expDown.cols == c.moeDim()
                    && (sh == 0 ? m.sharedGateUp == null
                        : m.sharedGateUp != null && m.sharedGateUp.rows == 2 * sh && m.sharedGateUp.cols == c.dim()
                            && m.sharedDown.rows == c.dim() && m.sharedDown.cols == sh)
                    && (d.topkMethod().equals("noaux_tc")
                        ? m.routerBias != null && m.routerBias.length == c.numExperts() : m.routerBias == null);
        } else {
            ok &= !c.isMoeLayer(l)
                    && L.wGateUp.rows == 2 * c.hiddenDim() && L.wGateUp.cols == c.dim()
                    && L.wDown.rows == c.dim() && L.wDown.cols == c.hiddenDim();
        }
        if (!ok) {
            throw new IllegalArgumentException("layer " + l + " weight shapes do not match config " + c + STALE_HINT);
        }
    }

    public long byteSize() {
        return file.byteSize();
    }

    @Override
    public void close() {
        file.close();
    }
}
