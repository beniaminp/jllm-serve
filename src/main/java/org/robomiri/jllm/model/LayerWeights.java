package org.robomiri.jllm.model;

/** The tensors of one layer of a {@link ModelWeights}, mirroring the layer's spec in the {@link Architecture}. */
public sealed interface LayerWeights permits DecoderLayer, MambaLayer, Qwen3NextLayer {

    /** The widest {@code cols} of any quantized tensor of the layer (the matmul scratch holds four rows of it). */
    int maxCols();
}
