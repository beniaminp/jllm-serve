package org.robomiri.jllm.metal;

/**
 * The parameter structs of {@code kernels.metal}, one constant per {@code struct ... Params}, as the sequence of
 * their 4-byte fields ({@code i} = int, {@code f} = float). {@link MetalKernels} writes every struct through
 * {@link Params} against its layout, which checks the field kinds and count at the dispatch, and
 * {@code MetalParamsTest} checks the layouts against the struct definitions in the shader source, so a field
 * added on one side cannot silently shift the others.
 */
public enum KernelParams {
    MATMUL("MatmulParams", "iiiiffi"),
    MATMUL3("Matmul3Params", "iiiiifi"),
    SWIGLU("SwigluParams", "iiifi"),
    NORM("NormParams", "if"),
    ROPE("RopeParams", "iiiiifiii"),
    ROPE_Q8("RopeQ8Params", "iiiiifiiiii"),
    ATTN("AttnParams", "iiiiiifiiiiiif"),
    ATTN_SPLIT("AttnSplitParams", "iiiiifiiifii"),
    ATTN_COMBINE("AttnCombineParams", "iii"),
    FLASH("FlashParams", "iiiifiiiif"),
    KV_STORE("KvStoreParams", "iiii"),
    KV_PUT("KvPutParams", "ii"),
    ROUTER("RouterParams", "iiiii"),
    ROUTER_NORM("RouterNormParams", "iiiiif"),
    ROUTER_DS("RouterDsParams", "iiiiiiiiifi"),
    ROUTER_DS_NORM("RouterDsNormParams", "iiiiiiiiifif"),
    MOE("MoeParams", "iii"),
    MOE_SORT("MoeSortParams", "iii"),
    EMBED("EmbedParams", "iif"),
    ARGMAX("ArgmaxParams", "ii"),
    MLA("MlaParams", "iiiii"),
    MAMBA_CONV("MambaConvParams", "iii"),
    MAMBA_SCAN("MambaScanParams", "iiii"),
    GDN_CONV("GdnConvParams", "iiiiii"),
    GDN_RULE("GdnRuleParams", "iiiiiiiifii"),
    GDN_NORM("GdnNormParams", "iiiif"),
    GATE("GateParams", "i");

    /** The struct's name in kernels.metal. */
    public final String struct;
    /** One character per field in declaration order: {@code i} for int, {@code f} for float. */
    public final String kinds;

    KernelParams(String struct, String kinds) {
        this.struct = struct;
        this.kinds = kinds;
    }

    /** The struct's size: every field is 4 bytes and there is no padding (the Java side relies on it). */
    public int bytes() {
        return kinds.length() * 4;
    }
}
