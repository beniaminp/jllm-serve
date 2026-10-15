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
    Q8_0(32);

    public static final int BLOCK = 32;

    /** Bytes of quantized data per block (scales are stored separately). */
    public final int blockBytes;

    QType(int blockBytes) {
        this.blockBytes = blockBytes;
    }

    public static QType parse(String s) {
        return switch (s.toLowerCase()) {
            case "q4", "q4_0" -> Q4_0;
            case "q8", "q8_0" -> Q8_0;
            default -> throw new IllegalArgumentException("unknown quantization '" + s + "' (use q4 or q8)");
        };
    }
}
