package org.robomiri.jllm.model;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Quantized weights of a Qwen3 (dense or MoE) decoder, memory-mapped from a {@link QuantFile}.
 *
 * <p>Linear layers use the requested format (Q4_0 by default); the embedding / LM head always uses Q8_0 because
 * small models are sensitive to output-layer precision. Gate and up projections are fused into one matrix; in MoE
 * layers all experts are stacked into one gate/up tensor and one down tensor.
 */
public final class ModelWeights implements AutoCloseable {

    public static final class Layer {
        public float[] attnNorm;
        public float[] ffnNorm;
        /** Per-head RMSNorm on queries/keys (Qwen3); null for models without it. */
        public float[] qNorm;
        public float[] kNorm;
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
        embed = file.tensor("model.embed_tokens.weight");
        lmHead = file.has("lm_head.weight") ? file.tensor("lm_head.weight") : embed;
        // The engine sizes its logits from vocabSize and backends write lmHead.rows outputs, so both must agree.
        if (embed.cols != cfg.dim() || embed.rows != cfg.vocabSize()
                || lmHead.cols != cfg.dim() || lmHead.rows != cfg.vocabSize()) {
            throw new IllegalArgumentException("embedding " + embed.rows + "x" + embed.cols + " / lm_head "
                    + lmHead.rows + "x" + lmHead.cols + " do not match config " + cfg + STALE_HINT);
        }
        layers = new Layer[cfg.layers()];
        for (int l = 0; l < cfg.layers(); l++) {
            String p = "model.layers." + l + ".";
            Layer L = new Layer();
            L.attnNorm = file.floats(p + "input_layernorm.weight");
            L.ffnNorm = file.floats(p + "post_attention_layernorm.weight");
            L.qNorm = file.has(p + "self_attn.q_norm.weight") ? file.floats(p + "self_attn.q_norm.weight") : null;
            L.kNorm = file.has(p + "self_attn.k_norm.weight") ? file.floats(p + "self_attn.k_norm.weight") : null;
            L.wq = file.tensor(p + "self_attn.q_proj.weight");
            L.wk = file.tensor(p + "self_attn.k_proj.weight");
            L.wv = file.tensor(p + "self_attn.v_proj.weight");
            L.wo = file.tensor(p + "self_attn.o_proj.weight");
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

    private static final String STALE_HINT = " (stale jllm-*.bin cache? delete it to reconvert)";

    /**
     * Opens the quantized cache in {@code dir}, converting the Hugging Face checkpoint first if there is none.
     */
    public static ModelWeights load(Path dir, ModelConfig cfg, QType quant, ShardProvider shards) throws IOException {
        if (!QuantFile.exists(dir, quant)) {
            ModelConverter.convert(dir, cfg, quant, shards);
        }
        long t0 = System.nanoTime();
        QuantFile file = QuantFile.open(dir, quant);
        try {
            ModelWeights w = new ModelWeights(cfg, quant, file);
            System.err.printf("Mapped %s weights (%s, %.0f MB) in %.2fs%n", cfg.architecture(), quant,
                    file.byteSize() / 1e6, (System.nanoTime() - t0) / 1e9);
            return w;
        } catch (RuntimeException e) {
            file.close();
            throw e;
        }
    }

    /** Every shape a backend relies on: a mismatch here would otherwise surface as a silent out-of-bounds read. */
    private static void checkShapes(ModelConfig c, Layer L, int l) {
        boolean ok = L.attnNorm.length == c.dim() && L.ffnNorm.length == c.dim()
                && L.wq.rows == c.qDim() && L.wq.cols == c.dim()
                && L.wk.rows == c.kvDim() && L.wk.cols == c.dim()
                && L.wv.rows == c.kvDim() && L.wv.cols == c.dim()
                && L.wo.rows == c.dim() && L.wo.cols == c.qDim()
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

    public long byteSize() {
        return file.byteSize();
    }

    @Override
    public void close() {
        file.close();
    }
}
