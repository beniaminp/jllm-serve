package org.robomiri.jllm.model;

/**
 * Block quantization formats. Each block of {@link #BLOCK} consecutive weights in a row shares one f32 scale.
 *
 * <ul>
 *   <li>{@code Q4_0}: 16 bytes per block; byte {@code j} holds weight {@code j} in the low nibble and weight
 *   {@code j+16} in the high nibble, value = (nibble - 8) * scale.</li>
 *   <li>{@code Q8_0}: 32 signed bytes per block, value = q * scale.</li>
 *   <li>{@code Q2_K}: llama.cpp's 2-bit K-quant, see {@link Q2K}.</li>
 *   <li>{@code Q4_0G}, {@code Q8_0G}, {@code Q4_K}, {@code Q5_K}, {@code Q4_1}, {@code Q5_0}, {@code Q5_1},
 *   {@code Q3_K}, {@code IQ4_NL}, {@code IQ4_XS}, {@code MXFP4}, {@code F16}, {@code BF16}: llama.cpp's own block
 *   layouts ({@link Ggml}), scales inside the block (F16 / BF16: plain halves, grouped in "blocks" of 32). They exist
 *   only as views of a GGUF file's tensors ({@link #gguf}); the engine never produces them, so they are not a valid
 *   {@code model.quant} and not stored in a cache.</li>
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
    Q5_K(Ggml.Q5_K_BYTES, Ggml.QK_K, 0, true),
    /** llama.cpp Q6_K: 256-weight super-blocks of 210 bytes carrying their own int8 sub-block scales ({@link Q6K}). */
    Q6_K(Q6K.BLOCK_BYTES, Q6K.BLOCK, 0),
    /** ggml Q4_1: 20-byte blocks of 32 weights, f16 scale d, f16 min m, 16 nibble bytes; w = q d + m. */
    Q4_1(Ggml.Q4_1_BYTES, QType.BLOCK, 0, true),
    /** ggml Q5_0: 22-byte blocks of 32 weights, f16 d, 32 high bits, 16 nibble bytes; w = (q5 - 16) d. */
    Q5_0(Ggml.Q5_0_BYTES, QType.BLOCK, 0, true),
    /** ggml Q5_1: 24-byte blocks of 32 weights, f16 d, f16 m, 32 high bits, 16 nibble bytes; w = q5 d + m. */
    Q5_1(Ggml.Q5_1_BYTES, QType.BLOCK, 0, true),
    /** ggml Q3_K: 110-byte super-blocks of 256 weights (2 bits + a high-bit mask, sixteen 6-bit scales). */
    Q3_K(Ggml.Q3_K_BYTES, Ggml.QK_K, 0, true),
    /** ggml IQ4_NL: 18-byte blocks of 32 weights, f16 d, 16 nibble bytes indexing a 16-entry table. */
    IQ4_NL(Ggml.IQ4_NL_BYTES, QType.BLOCK, 0, true),
    /** ggml IQ4_XS: 136-byte super-blocks of 256 weights, eight 32-weight sub-blocks with 6-bit scales, nibbles into the IQ4_NL table. */
    IQ4_XS(Ggml.IQ4_XS_BYTES, Ggml.QK_K, 0, true),
    /** ggml MXFP4: 17-byte blocks of 32 weights, an e8m0 exponent byte and 16 nibble bytes of e2m1 values. */
    MXFP4(Ggml.MXFP4_BYTES, QType.BLOCK, 0, true),
    /** f16 weights of a GGUF file, used as they are: "blocks" of 32 halves (64 bytes). */
    F16(64, QType.BLOCK, 0, true),
    /** bf16 weights of a GGUF file, used as they are: "blocks" of 32 bfloat16 (64 bytes). */
    BF16(64, QType.BLOCK, 0, true);

    public static final int BLOCK = 32;

    /** Bytes of quantized data per block (scales are stored separately). */
    public final int blockBytes;
    /** Weights per block: 32 for Q4_0/Q8_0, 256 (a super-block) for Q2_K / Q6_K. */
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

    /** The format of the embedding / LM head: {@code q8} (the default) or {@code q6k}. */
    public static QType parseHead(String s) {
        QType t;
        try {
            t = parse(s);
        } catch (IllegalArgumentException e) {
            t = null;
        }
        if (t != Q8_0 && t != Q6_K) {
            throw new IllegalArgumentException("unknown head quantization '" + s + "' (use q8 or q6k)");
        }
        return t;
    }

    public static QType parse(String s) {
        return switch (s.toLowerCase()) {
            case "q4", "q4_0" -> Q4_0;
            case "q8", "q8_0" -> Q8_0;
            case "q2k", "q2_k" -> Q2_K;
            case "q6k", "q6_k" -> Q6_K;
            default -> throw new IllegalArgumentException("unknown quantization '" + s + "' (use q4, q8, q2k or q6k)");
        };
    }
}
