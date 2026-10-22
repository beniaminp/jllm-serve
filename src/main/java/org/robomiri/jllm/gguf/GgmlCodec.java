package org.robomiri.jllm.gguf;

import org.robomiri.jllm.model.Ggml;
import org.robomiri.jllm.model.Q2K;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * What the engine can do with each ggml tensor type: compute with it in place (the {@link #nativeType} view), or
 * only dequantize it row by row so the loader can requantize it to Q8_0. Everything else (the other IQ types, the
 * ternary types, integers) is rejected with the type's name.
 *
 * <p>The system property {@code jllm.gguf.requantize} (a comma-separated list of ggml type names, or {@code all})
 * switches the in-place view of those types off, so they are dequantized and requantized to Q8_0 at load as before
 * these types were supported natively (for A/B comparisons and as a fallback).
 */
final class GgmlCodec {

    private static final ValueLayout.OfFloat LE_F32 = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    static final String SUPPORTED = "F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS,"
            + " MXFP4";

    /** True when {@code -Djllm.gguf.requantize} is set (to anything, {@code none} included): it then overrides the backend defaults. */
    private static final boolean EXPLICIT = System.getProperty("jllm.gguf.requantize") != null;

    /**
     * What the CPU backend requantizes by default: its decode is faster on the Q8_0 kernels than on the in-place kernels of
     * these formats by more than 25% (Q5_1: 29%, Q3_K: 37%, F16 and BF16 read twice the bytes of Q8_0; measured in docs/16
     * 16.9). Metal reads everything in place.
     */
    private static final Set<GgmlType> CPU_DEFAULT = EnumSet.of(GgmlType.Q5_1, GgmlType.Q3_K, GgmlType.F16, GgmlType.BF16);

    private static volatile Set<GgmlType> requantize = parseRequantize(System.getProperty("jllm.gguf.requantize", ""));

    /**
     * Sets the default for the backend the next model is loaded for ({@code cpu}: {@link #CPU_DEFAULT}; otherwise nothing is
     * requantized) unless {@code -Djllm.gguf.requantize} was given. Returns the types now requantized by that default.
     */
    static Set<GgmlType> backendDefault(boolean cpu) {
        if (!EXPLICIT) {
            requantize = cpu ? EnumSet.copyOf(CPU_DEFAULT) : EnumSet.noneOf(GgmlType.class);
        }
        return EXPLICIT ? EnumSet.noneOf(GgmlType.class) : requantize;
    }

    private GgmlCodec() {
    }

    /** Parses the value of {@code jllm.gguf.requantize}: type names separated by commas, {@code all}, {@code none} or empty. */
    static Set<GgmlType> parseRequantize(String list) {
        Set<GgmlType> out = EnumSet.noneOf(GgmlType.class);
        for (String n : list.split(",")) {
            n = n.trim();
            if (n.isEmpty() || n.equalsIgnoreCase("none")) {
                continue;
            }
            if (n.equalsIgnoreCase("all")) {
                out.addAll(List.of(GgmlType.values()));
                continue;
            }
            GgmlType t = null;
            for (GgmlType c : GgmlType.values()) {
                if (c.name().equalsIgnoreCase(n)) {
                    t = c;
                }
            }
            if (t == null) {
                throw new IllegalArgumentException("jllm.gguf.requantize: unknown ggml type '" + n + "'");
            }
            out.add(t);
        }
        return out;
    }

    /** Sets the types that are requantized to Q8_0 instead of being used in place (tests; the property sets it at startup). */
    static void setRequantize(String list) {
        requantize = parseRequantize(list);
    }

    /** The types currently requantized instead of viewed (for the load report). */
    static Set<GgmlType> requantized() {
        return requantize;
    }

    /** The engine format that views this ggml type's bytes without converting them, or null. */
    static QType nativeType(GgmlType t) {
        if (requantize.contains(t)) {
            return null;
        }
        return switch (t) {
            case Q4_0 -> QType.Q4_0G;
            case Q8_0 -> QType.Q8_0G;
            case Q4_K -> QType.Q4_K;
            case Q5_K -> QType.Q5_K;
            case Q2_K -> QType.Q2_K;
            case Q6_K -> QType.Q6_K;
            case Q4_1 -> QType.Q4_1;
            case Q5_0 -> QType.Q5_0;
            case Q5_1 -> QType.Q5_1;
            case Q3_K -> QType.Q3_K;
            case IQ4_NL -> QType.IQ4_NL;
            case IQ4_XS -> QType.IQ4_XS;
            case MXFP4 -> QType.MXFP4;
            case F16 -> QType.F16;
            case BF16 -> QType.BF16;
            default -> null;
        };
    }

    /** True if {@link #dequantizeRow} handles the type. */
    static boolean canDequantize(GgmlType t) {
        return switch (t) {
            case F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4 -> true;
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
            case F16 -> Ggml.dequantF16(seg, off, cols, dst, o);
            case BF16 -> Ggml.dequantBF16(seg, off, cols, dst, o);
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
            case IQ4_NL -> Ggml.dequantIQ4_NL(seg, off, cols, dst, o);
            case IQ4_XS -> Ggml.dequantIQ4_XS(seg, off, cols, dst, o);
            case MXFP4 -> Ggml.dequantMXFP4(seg, off, cols, dst, o);
            default -> throw new IllegalArgumentException("unsupported ggml tensor type " + t + " (supported: " + SUPPORTED + ")");
        }
    }
}
