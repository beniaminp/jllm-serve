package org.robomiri.jllm.metal;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java mirror of the constants block of {@code kernels.metal} (tile sizes, thread counts, limits): the values the
 * dispatch code and the backends' checks depend on. {@code MetalParamsTest} parses the shader and checks every
 * value in {@link #mirrored()} against its {@code #define}.
 */
public final class MetalConstants {

    /** Weights per quantization block (Q4_0 / Q8_0). */
    public static final int QK = 32;
    /** Tokens per simdgroup of the single-row matmuls. */
    public static final int TOK_TILE = 8;
    /** Rows per simdgroup of matmul_pf / matmul3_pf. */
    public static final int PF_ROWS = 2;
    /** Gate/up units per simdgroup of matmul_swiglu_pf. */
    public static final int PF_UNITS = 1;
    /** Tokens per simdgroup of the prefill matmuls. */
    public static final int PF_TOKS = 8;
    /** Matmul threadgroup: 4 simdgroups = 4 rows. */
    public static final int MM_THREADS = 128;
    /** rmsnorm / router / attention_combine threadgroup (one row each). */
    public static final int ROW_THREADS = 256;
    /** Attention threadgroup (8 simdgroups). */
    public static final int ATT_THREADS = 256;
    /** Tokens per attention dispatch: the scores scratch holds one row of ctx floats per (tile token, head). */
    public static final int ATT_TILE = 32;
    /** mla_kv_store / mamba_conv / mamba_scan threadgroup. */
    public static final int CHANNEL_THREADS = 64;
    /** Experts the router kernels can hold in threadgroup memory. */
    public static final int MAX_EXPERTS = 1024;
    /** DeepSeek expert groups the router can hold. */
    public static final int MAX_GROUPS = 64;
    /** Partials attention_combine can merge. */
    public static final int FLASH_MAX_SPLITS = 64;
    /** SSM state width the Mamba scan holds in registers. */
    public static final int MAMBA_MAX_STATE = 64;
    /** Conv kernel width the Mamba conv holds in registers. */
    public static final int MAMBA_MAX_CONV = 8;
    /** |argument| bound of the kernels' fast-math tanh. */
    public static final float TANH_CLAMP = 20.0f;

    private MetalConstants() {
    }

    /** Shader constant name -> the Java value. */
    public static Map<String, Double> mirrored() {
        Map<String, Double> m = new LinkedHashMap<>();
        m.put("QK", (double) QK);
        m.put("TOK_TILE", (double) TOK_TILE);
        m.put("PF_ROWS", (double) PF_ROWS);
        m.put("PF_UNITS", (double) PF_UNITS);
        m.put("PF_TOKS", (double) PF_TOKS);
        m.put("MM_THREADS", (double) MM_THREADS);
        m.put("ROW_THREADS", (double) ROW_THREADS);
        m.put("ATT_THREADS", (double) ATT_THREADS);
        m.put("ATT_TILE", (double) ATT_TILE);
        m.put("CHANNEL_THREADS", (double) CHANNEL_THREADS);
        m.put("MAX_EXPERTS", (double) MAX_EXPERTS);
        m.put("MAX_GROUPS", (double) MAX_GROUPS);
        m.put("FLASH_MAX_SPLITS", (double) FLASH_MAX_SPLITS);
        m.put("MAMBA_MAX_STATE", (double) MAMBA_MAX_STATE);
        m.put("MAMBA_MAX_CONV", (double) MAMBA_MAX_CONV);
        m.put("TANH_CLAMP", (double) TANH_CLAMP);
        return m;
    }
}
