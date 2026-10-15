package org.robomiri.jllm.model;

/**
 * Block quantization formats. Each block of {@link #BLOCK} consecutive weights in a row shares one f32 scale.
 *
 * <ul>
 *   <li>{@code Q4_0}: 16 bytes per block; byte {@code j} holds weight {@code j} in the low nibble and weight
 *   {@code j+16} in the high nibble, value = (nibble - 8) * scale.</li>
 *   <li>{@code Q8_0}: 32 signed bytes per block, value = q * scale.</li>
 * </ul>
 */
public enum QType {
    Q4_0(16),
    Q8_0(32),
    /** llama.cpp Q2_K: 256-weight super-blocks of 84 bytes carrying their own 4-bit scales and mins ({@link Q2K}). */
    Q2_K(Q2K.BLOCK_BYTES, Q2K.BLOCK, 0);

    public static final int BLOCK = 32;

    /** Bytes of quantized data per block (scales are stored separately). */
    public final int blockBytes;
    /** Weights per block: 32 for Q4_0/Q8_0, 256 (a super-block) for Q2_K. */
    public final int blockSize;
    /** Bytes of separately stored f32 scale per block; 0 when the scales live inside the block. */
    public final int scaleBytes;

    QType(int blockBytes) {
        this(blockBytes, BLOCK, Float.BYTES);
    }

    QType(int blockBytes, int blockSize, int scaleBytes) {
        this.blockBytes = blockBytes;
        this.blockSize = blockSize;
        this.scaleBytes = scaleBytes;
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
