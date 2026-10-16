package org.robomiri.jllm.model;

/**
 * Block quantization formats. Each block of {@link #BLOCK} consecutive weights in a row shares one f32 scale.
 *
 * <ul>
 *   <li>{@code Q4_0}: 16 bytes per block; byte {@code j} holds weight {@code j} in the low nibble and weight
 *   {@code j+16} in the high nibble, value = (nibble - 8) * scale.</li>
 *   <li>{@code Q8_0}: 32 signed bytes per block, value = q * scale.</li>
 *   <li>{@code Q2_K}: llama.cpp's 2-bit K-quant, see {@link Q2K}.</li>
 *   <li>{@code Q4_0G}, {@code Q8_0G}, {@code Q4_K}, {@code Q5_K}: llama.cpp's own block layouts ({@link Ggml}),
 *   scales inside the block. They exist only as views of a GGUF file's tensors ({@link #gguf}); the engine
 *   never produces them, so they are not a valid {@code model.quant} and not stored in a cache.</li>
 * </ul>
 */
public enum QType {
    Q4_0(16),
    Q8_0(32),
    /** llama.cpp Q2_K: 256-weight super-blocks of 84 bytes carrying their own 4-bit scales and mins ({@link Q2K}). */
    Q2_K(Q2K.BLOCK_BYTES, Q2K.BLOCK, 0),
    /** ggml Q4_0: 18-byte blocks of 32 weights, f16 scale then 16 nibble bytes (same nibble order as {@link #Q4_0}). */
    Q4_0G(Ggml.Q4_0_BYTES, QType.BLOCK, 0, true),
    /** ggml Q8_0: 34-byte blocks of 32 weights, f16 scale then 32 signed bytes. */
    Q8_0G(Ggml.Q8_0_BYTES, QType.BLOCK, 0, true),
    /** ggml Q4_K: 144-byte super-blocks of 256 weights (4-bit quants, 6-bit scales and mins). */
    Q4_K(Ggml.Q4_K_BYTES, Ggml.QK_K, 0, true),
    /** ggml Q5_K: 176-byte super-blocks of 256 weights (5-bit quants, 6-bit scales and mins). */
    Q5_K(Ggml.Q5_K_BYTES, Ggml.QK_K, 0, true);

    public static final int BLOCK = 32;

    /** Bytes of quantized data per block (scales are stored separately). */
    public final int blockBytes;
    /** Weights per block: 32 for Q4_0/Q8_0, 256 (a super-block) for Q2_K. */
    public final int blockSize;
    /** Bytes of separately stored f32 scale per block; 0 when the scales live inside the block. */
    public final int scaleBytes;
    /**
     * True for the formats that exist only as in-place views of a GGUF file's tensors: rows are contiguous
     * blocks with no scale region and no page padding, and the engine cannot quantize to them.
     */
    public final boolean gguf;

    QType(int blockBytes) {
        this(blockBytes, BLOCK, Float.BYTES);
    }

    QType(int blockBytes, int blockSize, int scaleBytes) {
        this(blockBytes, blockSize, scaleBytes, false);
    }

    QType(int blockBytes, int blockSize, int scaleBytes, boolean gguf) {
        this.blockBytes = blockBytes;
        this.blockSize = blockSize;
        this.scaleBytes = scaleBytes;
        this.gguf = gguf;
    }

    public static QType parse(String s) {
        return switch (s.toLowerCase()) {
            case "q4", "q4_0" -> Q4_0;
            case "q8", "q8_0" -> Q8_0;
            case "q2k", "q2_k" -> Q2_K;
            default -> throw new IllegalArgumentException("unknown quantization '" + s + "' (use q4, q8 or q2k)");
        };
    }
}
