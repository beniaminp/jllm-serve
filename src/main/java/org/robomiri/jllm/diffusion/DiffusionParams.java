package org.robomiri.jllm.diffusion;

/**
 * The parameter structs of {@code diffusion.metal}: one constant per {@code struct ... Params}, as the sequence of its
 * fields ({@code i} = 4-byte int, {@code f} = 4-byte float, {@code l} = 8-byte long, which is always the first field so
 * that no padding arises). {@link MetalOps} writes every struct against its layout, which checks the kinds and the count at
 * the dispatch, and {@code DiffusionParamsTest} checks the layouts against the struct definitions in the shader source.
 */
enum DiffusionParams {
    GEMM("DfGemmParams", "iiiiiiiiiiiiifiiii"),
    CONV("DfConvParams", "iiiiiiiiiiiiiiii"),
    NORM("DfNormParams", "iiifii"),
    GN("DfGnParams", "iiifiiii"),
    FLASH("DfFlashParams", "iiiiiiifi"),
    SOFTMAX("DfSoftmaxParams", "iiii"),
    EW("DfEwParams", "liiii"),
    MOD("DfModParams", "liiiiiii");

    final String struct;
    final String kinds;

    DiffusionParams(String struct, String kinds) {
        this.struct = struct;
        this.kinds = kinds;
    }

    int bytes() {
        int n = 0;
        for (char c : kinds.toCharArray()) {
            n += c == 'l' ? 8 : 4;
        }
        return n;
    }
}
