package org.robomiri.jllm.gguf;

import org.robomiri.jllm.model.Ggml;
import org.robomiri.jllm.model.Q2K;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * What the engine can do with each ggml tensor type: compute with it in place (the {@link #nativeType} view), or
 * only dequantize it row by row so the loader can requantize it to Q8_0. Everything else (the IQ and ternary
 * types, MXFP4, integers) is rejected with the type's name.
 */
final class GgmlCodec {

    private static final ValueLayout.OfFloat LE_F32 = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgmlCodec() {
    }

    /** The engine format that views this ggml type's bytes without converting them, or null. */
    static QType nativeType(GgmlType t) {
        return switch (t) {
            case Q4_0 -> QType.Q4_0G;
            case Q8_0 -> QType.Q8_0G;
            case Q4_K -> QType.Q4_K;
            case Q5_K -> QType.Q5_K;
            case Q2_K -> QType.Q2_K;
            default -> null;
        };
    }

    /** True if {@link #dequantizeRow} handles the type. */
    static boolean canDequantize(GgmlType t) {
        return switch (t) {
            case F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K -> true;
            default -> false;
        };
    }

    /** True for the floating-point types a matrix is quantized from at load (as opposed to a quantized format). */
    static boolean isFloat(GgmlType t) {
        return t == GgmlType.F32 || t == GgmlType.F16 || t == GgmlType.BF16;
    }

    /** Dequantizes the row of {@code cols} elements at byte {@code off} of {@code seg} into {@code dst[o ..)}. */
    static void dequantizeRow(GgmlType t, MemorySegment seg, long off, int cols, float[] dst, int o) {
        switch (t) {
            case F32 -> {
                for (int i = 0; i < cols; i++) {
                    dst[o + i] = seg.get(LE_F32, off + 4L * i);
                }
            }
            case F16 -> {
                for (int i = 0; i < cols; i++) {
                    dst[o + i] = Ggml.f16(seg, off + 2L * i);
                }
            }
            case BF16 -> {
                for (int i = 0; i < cols; i++) {
                    dst[o + i] = Ggml.bf16(seg, off + 2L * i);
                }
            }
            case Q4_0 -> Ggml.dequantQ4_0(seg, off, cols, dst, o);
            case Q4_1 -> Ggml.dequantQ4_1(seg, off, cols, dst, o);
            case Q5_0 -> Ggml.dequantQ5_0(seg, off, cols, dst, o);
            case Q5_1 -> Ggml.dequantQ5_1(seg, off, cols, dst, o);
            case Q8_0 -> Ggml.dequantQ8_0(seg, off, cols, dst, o);
            case Q2_K -> Q2K.dequantizeRow(seg, off, cols, dst, o);
            case Q3_K -> Ggml.dequantQ3_K(seg, off, cols, dst, o);
            case Q4_K -> Ggml.dequantQ4_K(seg, off, cols, dst, o);
            case Q5_K -> Ggml.dequantQ5_K(seg, off, cols, dst, o);
            case Q6_K -> Ggml.dequantQ6_K(seg, off, cols, dst, o);
            default -> throw new IllegalArgumentException("unsupported ggml tensor type " + t
                    + " (supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)");
        }
    }
}
