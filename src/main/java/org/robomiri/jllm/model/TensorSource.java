package org.robomiri.jllm.model;

/**
 * Where {@link ModelWeights} gets its tensors from, by the canonical names the {@link ModelConverter} writes
 * ({@code model.layers.N.self_attn.q_proj.weight}, {@code model.layers.N.mlp.gate_up}, ...): the {@link QuantFile}
 * cache converted from safetensors, or a GGUF file whose llama.cpp tensor names are translated to these
 * (see {@code org.robomiri.jllm.gguf.GgufWeights}).
 */
public interface TensorSource {

    boolean has(String name);

    /**
     * A quantized matrix.
     *
     * @throws IllegalArgumentException if there is no such tensor or it is not a quantized matrix
     */
    QTensor tensor(String name);

    /**
     * A vector of floats (norm gains, biases, the MoE router).
     *
     * @throws IllegalArgumentException if there is no such tensor or it cannot be read as floats
     */
    float[] floats(String name);
}
