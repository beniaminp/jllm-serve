package org.robomiri.jllm.model;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Checkpoints block-quantized to FP8 (DeepSeek-V3 / R1 as published, and other {@code "quant_method": "fp8"}
 * releases): every quantized linear layer {@code X} is stored as {@code X.weight} ({@code F8_E4M3},
 * {@code [out, in]}) and {@code X.weight_scale_inv} (f32, {@code [ceil(out / bh), ceil(in / bw)]}), and
 * <pre>
 *     w[r][c] = fp8(X.weight[r][c]) * X.weight_scale_inv[r / bh][c / bw]
 * </pre>
 * with the block {@code [bh, bw]} from {@code quantization_config.weight_block_size} ({@code [128, 128]} by default;
 * despite its name the scale is multiplied, as in DeepSeek's {@code weight_dequant}). The converter reads such a
 * matrix through {@link #dequantized} like any float tensor and stores it in the engine's own format, exactly as
 * {@link MlxQuant} does for MLX checkpoints. Norms, embeddings, the LM head and the routers stay bf16 in these
 * checkpoints and are read as they are.
 */
public final class Fp8Quant {

    /** The default block, DeepSeek-V3's. */
    public static final int DEFAULT_BLOCK = 128;

    /** Rows of a scale block ({@code weight_block_size[0]}). */
    public final int blockRows;
    /** Columns of a scale block ({@code weight_block_size[1]}). */
    public final int blockCols;

    public Fp8Quant(int blockRows, int blockCols) {
        if (blockRows <= 0 || blockCols <= 0) {
            throw new IllegalArgumentException("invalid FP8 block " + blockRows + "x" + blockCols);
        }
        this.blockRows = blockRows;
        this.blockCols = blockCols;
    }

    /**
     * The FP8 block quantization of the checkpoint in {@code dir}: {@code quantization_config.weight_block_size} of
     * its {@code config.json}, {@code [128, 128]} when there is none. Only called for a checkpoint that has
     * {@code weight_scale_inv} tensors.
     */
    public static Fp8Quant read(Path dir) throws IOException {
        Path cfg = dir.resolve("config.json");
        Map<String, Object> q = Files.isRegularFile(cfg) ? Json.map(Json.parse(Files.readString(cfg)), "quantization_config") : null;
        String method = Json.str(q, "quant_method", "fp8");
        if (!method.equals("fp8")) {
            throw new IOException("checkpoint has weight_scale_inv tensors but quantization_config.quant_method is '"
                    + method + "' (only fp8 is supported)");
        }
        List<Object> block = Json.list(q, "weight_block_size");
        if (block == null) {
            return new Fp8Quant(DEFAULT_BLOCK, DEFAULT_BLOCK);
        }
        if (block.size() != 2 || !(block.get(0) instanceof Number r) || !(block.get(1) instanceof Number c)
                || r.intValue() <= 0 || c.intValue() <= 0) {
            throw new IOException("invalid quantization_config.weight_block_size " + block + " (expected [rows, cols])");
        }
        return new Fp8Quant(r.intValue(), c.intValue());
    }

    /**
     * The float view of an FP8 matrix: shape {@code [out, in]}, each element its FP8 value times the scale of its
     * block. A scale with a single element applies to the whole matrix (per-tensor quantization).
     *
     * @throws IOException if the weight is not a 2-D FP8 tensor or the scale grid does not cover it
     */
    public SafeTensors.Tensor dequantized(String name, SafeTensors.Tensor weight, SafeTensors.Tensor scaleInv)
            throws IOException {
        if (weight.shape().length != 2 || (weight.dtype() != SafeTensors.DType.F8_E4M3
                && weight.dtype() != SafeTensors.DType.F8_E5M2)) {
            throw new IOException("FP8 tensor " + name + " must be a 2-D F8_E4M3 / F8_E5M2 matrix, got " + weight);
        }
        int rows = weight.dim(0);
        int cols = weight.dim(1);
        long n = scaleInv.numel();
        int gridRows;
        int gridCols;
        int bh;
        int bw;
        if (n == 1) {
            gridRows = gridCols = 1;
            bh = rows;
            bw = cols;
        } else {
            gridRows = (rows + blockRows - 1) / blockRows;
            gridCols = (cols + blockCols - 1) / blockCols;
            bh = blockRows;
            bw = blockCols;
            if (scaleInv.shape().length != 2 || scaleInv.dim(0) != gridRows || scaleInv.dim(1) != gridCols) {
                throw new IOException("FP8 tensor " + name + ": scale " + scaleInv + " does not match the "
                        + blockRows + "x" + blockCols + " blocks of " + weight + " (expected [" + gridRows + ", "
                        + gridCols + "])");
            }
        }
        float[] scale = scaleInv.toFloatArray(); // one per block: 8064 floats for DeepSeek-V3's largest matrix
        int gc = gridCols;
        SafeTensors.Tensor.Reader reader = (start, dst, off, len) -> {
            weight.toFloat(start, dst, off, len);
            for (int i = 0; i < len; i++) {
                long e = start + i;
                int r = (int) (e / cols);
                int c = (int) (e - (long) r * cols);
                dst[off + i] *= scale[(r / bh) * gc + c / bw];
            }
        };
        return new SafeTensors.Tensor(name, weight.dtype(), weight.shape(), weight.data(), reader);
    }
}
