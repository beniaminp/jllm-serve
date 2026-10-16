package org.robomiri.jllm.model;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * What a checkpoint computes, resolved once from its Hugging Face {@code config.json}: the model-wide
 * {@link Core} plus a typed description of every layer. {@link Decoder} is every attention model (Llama, Mistral,
 * Qwen2, Qwen3 dense and MoE, GLM / GLM-4, Phi-3, Gemma 2 / 3, DeepSeek-V2 / V3), {@link MambaArch} the Mamba
 * state-space model.
 *
 * <p>A family's switches (q/k/v bias, QK-norm, sliding window, a second RoPE base, post-norms, which layers route
 * to experts, latent attention, the routing rule, ...) are read by that family's parser in {@link Families} and
 * written into the per-layer {@link LayerSpec}s, so the loader, the converter and the backends only ever look
 * at the typed vocabulary below and never at the {@code model_type}.
 */
public sealed interface Architecture permits Decoder, MambaArch {

    /**
     * Config key written by the GGUF loader and never found in a Hugging Face config: the file's q/k rows were
     * permuted by llama.cpp's converter for its interleaved RoPE (llama architecture), so the rotary table pairs
     * {@code 2i} with {@code 2i + 1} instead of {@code i} with {@code i + rotaryDim/2} (docs/16-gguf.md).
     */
    String GGUF_INTERLEAVED_ROPE = "jllm_gguf_interleaved_rope";

    Core core();

    /**
     * What every model has.
     *
     * @param architecture   the architecture class ({@code Qwen3ForCausalLM}, ...), for messages
     * @param modelType      the {@code model_type} string: selects the chat template, nothing else
     * @param dim            {@code hidden_size}, the width of the residual stream
     * @param layers         {@code num_hidden_layers}
     * @param vocabSize      rows of the embedding and of the LM head
     * @param normEps        epsilon of every RMSNorm
     * @param tiedEmbeddings the LM head is the embedding matrix
     * @param bosTokenId     {@code bos_token_id}, -1 if absent
     * @param eosTokenIds    {@code eos_token_id} (an int or a list in config.json)
     * @param maxPositions   {@code max_position_embeddings}, the trained context length
     * @param embedScale     multiplier of every dequantized embedding row (Gemma: {@code bf16(sqrt(dim))}; 1 elsewhere)
     * @param finalSoftcap   {@code tanh(l / cap) * cap} on the logits, 0 = off
     * @param normPlusOne    norm gains are stored as {@code w} and applied as {@code 1 + w} (Gemma; the converter
     *                       folds the 1 into every norm it writes, so the backends never see this flag)
     * @param description    the canonical one-line description of the model; it is recorded in the quantized
     *                       cache index and compared on resume, so for a given config it must never change
     */
    record Core(String architecture, String modelType, int dim, int layers, int vocabSize, float normEps,
                boolean tiedEmbeddings, int bosTokenId, List<Integer> eosTokenIds, int maxPositions,
                float embedScale, float finalSoftcap, boolean normPlusOne, String description) {
        public Core {
            eosTokenIds = List.copyOf(eosTokenIds);
        }
    }

    /** Parses {@code dir/config.json}; see {@link #parse}. */
    static Architecture load(Path dir) throws IOException {
        Map<String, Object> c;
        try {
            c = Json.parseObject(Files.readString(dir.resolve("config.json")));
        } catch (RuntimeException e) {
            throw new IOException(dir.resolve("config.json") + ": " + e.getMessage(), e);
        }
        return parse(c);
    }

    /**
     * Resolves a parsed {@code config.json} with the parser of its {@code model_type} (the {@code text_config}
     * of a multimodal wrapper is unwrapped first).
     *
     * @throws IllegalArgumentException for an unsupported family or an invalid or unsupported configuration
     */
    static Architecture parse(Map<String, Object> config) {
        return Families.parse(config);
    }
}
