package org.robomiri.jllm.gguf;

/**
 * The tensor element types of the ggml / GGUF format, with the ids a GGUF file stores and the block geometry
 * ({@code blockSize} elements in {@code typeSize} bytes) that determines a row's size. Not every type can be
 * computed with here (see {@link GgufWeights}); all of them are listed so a file's tensor sizes can be validated
 * and an unsupported type is reported by name instead of as a number.
 */
public enum GgmlType {
    F32(0, 1, 4),
    F16(1, 1, 2),
    Q4_0(2, 32, 18),
    Q4_1(3, 32, 20),
    Q5_0(6, 32, 22),
    Q5_1(7, 32, 24),
    Q8_0(8, 32, 34),
    Q8_1(9, 32, 36),
    Q2_K(10, 256, 84),
    Q3_K(11, 256, 110),
    Q4_K(12, 256, 144),
    Q5_K(13, 256, 176),
    Q6_K(14, 256, 210),
    Q8_K(15, 256, 292),
    IQ2_XXS(16, 256, 66),
    IQ2_XS(17, 256, 74),
    IQ3_XXS(18, 256, 98),
    IQ1_S(19, 256, 50),
    IQ4_NL(20, 32, 18),
    IQ3_S(21, 256, 110),
    IQ2_S(22, 256, 82),
    IQ4_XS(23, 256, 136),
    I8(24, 1, 1),
    I16(25, 1, 2),
    I32(26, 1, 4),
    I64(27, 1, 8),
    F64(28, 1, 8),
    IQ1_M(29, 256, 56),
    BF16(30, 1, 2),
    TQ1_0(34, 256, 54),
    TQ2_0(35, 256, 66),
    MXFP4(39, 32, 17);

    /** The id stored in the file. */
    public final int id;
    /** Elements per block (1 for the plain types). */
    public final int blockSize;
    /** Bytes per block. */
    public final int typeSize;

    GgmlType(int id, int blockSize, int typeSize) {
        this.id = id;
        this.blockSize = blockSize;
        this.typeSize = typeSize;
    }

    /** The type with the given file id, or null if this build does not know it. */
    public static GgmlType of(int id) {
        for (GgmlType t : values()) {
            if (t.id == id) {
                return t;
            }
        }
        return null;
    }

    /** Bytes of a row of {@code n} elements ({@code n} must be a multiple of {@link #blockSize}). */
    public long rowBytes(long n) {
        return n / blockSize * typeSize;
    }
}
