package org.robomiri.jllm.audio;

import org.robomiri.jllm.json.Json;

import java.util.Map;

/**
 * The hyper-parameters of a Hugging Face Whisper checkpoint ({@code config.json}).
 *
 * @param dModel        hidden size of encoder and decoder
 * @param encLayers     encoder layers
 * @param decLayers     decoder layers (4 for large-v3-turbo)
 * @param encHeads      attention heads of the encoder
 * @param decHeads      attention heads of the decoder
 * @param encFfn        encoder feed-forward width
 * @param decFfn        decoder feed-forward width
 * @param nMels         mel bins (80, or 128 for large-v3)
 * @param vocab         vocabulary size
 * @param sourcePos     encoder positions after the stride-2 convolution (1500 for a 30 s window)
 * @param targetPos     decoder positions (448)
 * @param scaleEmbedding whether token embeddings are scaled by sqrt(dModel) (never for the released checkpoints)
 */
public record WhisperConfig(int dModel, int encLayers, int decLayers, int encHeads, int decHeads, int encFfn, int decFfn,
                            int nMels, int vocab, int sourcePos, int targetPos, boolean scaleEmbedding) {

    public static WhisperConfig parse(Map<String, Object> c) {
        int d = (int) Json.num(c, "d_model", 0);
        if (d <= 0) {
            throw new IllegalArgumentException("config.json has no d_model: not a Whisper checkpoint");
        }
        return new WhisperConfig(d,
                (int) Json.num(c, "encoder_layers", 0), (int) Json.num(c, "decoder_layers", 0),
                (int) Json.num(c, "encoder_attention_heads", 0), (int) Json.num(c, "decoder_attention_heads", 0),
                (int) Json.num(c, "encoder_ffn_dim", 4L * d), (int) Json.num(c, "decoder_ffn_dim", 4L * d),
                (int) Json.num(c, "num_mel_bins", 80), (int) Json.num(c, "vocab_size", 51865),
                (int) Json.num(c, "max_source_positions", 1500), (int) Json.num(c, "max_target_positions", 448),
                Json.bool(c, "scale_embedding", false));
    }

    public int encHeadDim() {
        return dModel / encHeads;
    }

    public int decHeadDim() {
        return dModel / decHeads;
    }

    /** Mel frames of one window: twice the encoder positions (3000 for the released models). */
    public int windowFrames() {
        return 2 * sourcePos;
    }

    /** Whether the vocabulary is the multilingual one (51865 / 51866 tokens), with language and task tokens. */
    public boolean multilingual() {
        return vocab >= 51865;
    }
}
