package org.robomiri.jllm.metal;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java mirror of the {@code #define} constants of {@code kernels.metal} that the dispatch code depends on (tile
 * sizes, thread counts, limits). {@code MetalParamsTest} parses the shader and checks every value in
 * {@link #mirrored()} against it.
 */
public final class MetalConstants {

    /** Tokens per simdgroup of the single-row matmuls. */
    public static final int TOK_TILE = 8;
    /** Rows per simdgroup of matmul_pf / matmul3_pf. */
    public static final int PF_ROWS = 2;
    /** Gate/up units per simdgroup of matmul_swiglu_pf. */
    public static final int PF_UNITS = 1;
    /** Tokens per simdgroup of the prefill matmuls. */
    public static final int PF_TOKS = 8;

    private MetalConstants() {
    }

    /** Shader constant name -> the Java value. */
    public static Map<String, Double> mirrored() {
        Map<String, Double> m = new LinkedHashMap<>();
        m.put("TOK_TILE", (double) TOK_TILE);
        m.put("PF_ROWS", (double) PF_ROWS);
        m.put("PF_UNITS", (double) PF_UNITS);
        m.put("PF_TOKS", (double) PF_TOKS);
        return m;
    }
}
