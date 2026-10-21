package org.robomiri.jllm.model;

/**
 * The multi-token-prediction ("nextn") head of Qwen3.5 (llama.cpp: {@code qwen35.nextn_predict_layers = 1}, tensors
 * {@code blk.<layers>.nextn.*} and that block's own attention and MLP). It predicts the token <em>after</em> the one the
 * main model has just chosen, from the main model's last hidden state and the embedding of the chosen token, so the
 * main model can verify its guess together with the next position in one pass (speculative decoding, docs/10):
 * <pre>
 *   x = eh_proj · [ rmsnorm(embedding(token), enorm) ; rmsnorm(h, hnorm) ]        [2·dim] → [dim]
 *   x += gated_attention(rmsnorm(x, attn_norm))      the block's own KV cache, RoPE at the position of h
 *   x += swiglu(rmsnorm(x, post_attention_norm))
 *   logits = head · rmsnorm(x, shared_head_norm)
 * </pre>
 * where {@code h} is the main model's hidden state <em>after</em> its final norm at the position before the token,
 * and head / norm / embedding default to the main model's.
 *
 * @param ehProj   {@code [dim, 2·dim]}: columns {@code [0, dim)} read the normed token embedding, the rest the normed hidden
 * @param enorm    gain for the token embedding {@code [dim]}
 * @param hnorm    gain for the hidden state {@code [dim]}
 * @param headNorm gain before the head {@code [dim]}; the main model's final norm when the file has none
 * @param head     the output head ({@code nextn.shared_head_head}); the main model's when the file has none
 * @param embed    the embedding table ({@code nextn.embed_tokens}); the main model's when the file has none
 * @param layer    the block: gated full attention and the model's MLP
 */
public record Qwen3NextMtp(QTensor ehProj, float[] enorm, float[] hnorm, float[] headNorm, QTensor head, QTensor embed,
                           Qwen3NextLayer layer) {

    static Qwen3NextMtp load(TensorSource f, String p, Qwen3NextArch arch, QTensor mainEmbed, QTensor mainHead,
                             float[] mainNorm) {
        int dim = arch.core().dim();
        QTensor eh = f.tensor(p + "nextn.eh_proj");
        float[] enorm = f.floats(p + "nextn.enorm");
        float[] hnorm = f.floats(p + "nextn.hnorm");
        QTensor embed = f.has(p + "nextn.embed_tokens") ? f.tensor(p + "nextn.embed_tokens") : mainEmbed;
        QTensor head = f.has(p + "nextn.shared_head_head") ? f.tensor(p + "nextn.shared_head_head") : mainHead;
        float[] norm = f.has(p + "nextn.shared_head_norm") ? f.floats(p + "nextn.shared_head_norm") : mainNorm;
        if (eh.rows != dim || eh.cols != 2 * dim || enorm.length != dim || hnorm.length != dim || norm.length != dim
                || embed.cols != dim || head.cols != dim || head.rows != arch.core().vocabSize()) {
            throw new IllegalArgumentException("the multi-token-prediction head's shapes do not match config " + arch
                    + ModelWeights.STALE_HINT);
        }
        return new Qwen3NextMtp(eh, enorm, hnorm, norm, head, embed, Qwen3NextLayer.loadMtp(f, p, arch));
    }
}
