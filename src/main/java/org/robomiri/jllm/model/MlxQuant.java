package org.robomiri.jllm.model;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Checkpoints quantized by Apple's MLX ({@code mlx_lm.convert -q}, the {@code mlx-community/*-4bit} repositories):
 * every quantized linear layer {@code X} is stored as {@code X.weight} (unsigned {@code bits}-wide codes packed
 * little-endian into uint32 words, {@code [out, in * bits / 32]}), {@code X.scales} and {@code X.biases}
 * ({@code [out, in / groupSize]}, f16 or bf16), and
 * <pre>
 *     w[r][c] = scales[r][c / g] * q[r][c] + biases[r][c / g]        (affine: the bias is the group's minimum)
 * </pre>
 * {@code config.json} names the scheme ({@code "quantization": {"group_size": 64, "bits": 4}}; layers may differ:
 * the bit width of each tensor follows from its shapes). The converter reads such a matrix through
 * {@link #dequantized} like any float tensor and stores it in the engine's own format ({@code model.quant}), so every
 * family and both backends work unchanged; {@code --model.quant q8} (or {@code q6k}) keeps a 4-bit checkpoint's
 * values almost exactly, {@code q4} re-quantizes them once more.
 */
public final class MlxQuant {

    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    /** The default group size of the checkpoint ({@code quantization.group_size}). */
    public final int groupSize;
    /** The default bit width ({@code quantization.bits}); informational, each tensor's own follows from its shapes. */
    public final int bits;

    private MlxQuant(int groupSize, int bits) {
        this.groupSize = groupSize;
        this.bits = bits;
    }

    /** The MLX quantization of the checkpoint in {@code dir}, or null if it is not MLX-quantized. */
    public static MlxQuant read(Path dir) throws IOException {
        Path cfg = dir.resolve("config.json");
        if (!Files.isRegularFile(cfg)) {
            return null;
        }
        Object root = Json.parse(Files.readString(cfg));
        Map<String, Object> q = Json.map(root, "quantization");
        if (q == null) {
            q = Json.map(root, "quantization_config");
        }
        if (q == null || !(q.get("group_size") instanceof Number) || !(q.get("bits") instanceof Number)) {
            return null; // not MLX (or another scheme such as GPTQ / AWQ, which name their tensors differently)
        }
        String mode = Json.str(q, "mode", "affine");
        if (!mode.equals("affine")) {
            throw new IOException("MLX quantization mode '" + mode + "' is not supported (only affine)");
        }
        return new MlxQuant(((Number) q.get("group_size")).intValue(), ((Number) q.get("bits")).intValue());
    }

    /**
     * The float view of a packed MLX matrix: shape {@code [out, in]}, values dequantized on read. The group size is
     * the checkpoint's unless the shapes rule it out; the bit width follows from the shapes.
     *
     * @throws IOException if the shapes do not fit together or the bit width is not 2, 4 or 8
     */
    public SafeTensors.Tensor dequantized(String name, SafeTensors.Tensor packed, SafeTensors.Tensor scales,
                                          SafeTensors.Tensor biases) throws IOException {
        if (packed.shape().length != 2 || scales.shape().length != 2 || packed.dtype() != SafeTensors.DType.U32
                || !java.util.Arrays.equals(scales.shape(), biases.shape()) || scales.dim(0) != packed.dim(0)) {
            throw new IOException("malformed MLX quantized tensor " + name + ": " + packed + ", " + scales + ", " + biases);
        }
        int rows = packed.dim(0);
        int groups = scales.dim(1);
        long packedBits = (long) packed.dim(1) * 32;
        // in = groups * g and packedBits = in * bits: with the checkpoint's group size first (an 8-bit layer with
        // groups of 64 has the shapes of a 4-bit one with groups of 128), the bit width follows
        int bits = -1;
        int group = -1;
        for (int g : new int[]{groupSize, 64, 32, 128}) {
            long in = (long) groups * g;
            if (packedBits % in == 0) {
                long b = packedBits / in;
                if (b == 2 || b == 4 || b == 8) {
                    bits = (int) b;
                    group = g;
                    break;
                }
            }
        }
        if (bits < 0) {
            throw new IOException("MLX tensor " + name + ": unsupported bit width or group size (" + packed + ", "
                    + scales + "); 2, 4 and 8 bits with groups of 32, 64 or 128 are supported");
        }
        int cols = groups * group;
        int perWord = 32 / bits;
        int mask = (1 << bits) - 1;
        int b = bits;
        int g = group;
        SafeTensors.Tensor.Reader reader = (start, dst, off, n) -> {
            float[] s = new float[groups];
            float[] z = new float[groups];
            long row = -1;
            for (int i = 0; i < n; i++) {
                long e = start + i;
                long r = e / cols;
                int c = (int) (e % cols);
                if (r != row) {
                    row = r;
                    scales.toFloat(r * groups, s, 0, groups);
                    biases.toFloat(r * groups, z, 0, groups);
                }
                int word = packed.data().getAtIndex(INT_LE, r * packed.dim(1) + c / perWord);
                int q = (word >>> ((c % perWord) * b)) & mask;
                dst[off + i] = s[c / g] * q + z[c / g];
            }
        };
        return new SafeTensors.Tensor(name, packed.dtype(), new long[]{rows, cols}, packed.data(), reader);
    }
}
