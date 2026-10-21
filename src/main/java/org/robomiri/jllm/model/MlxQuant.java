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
 *
 * <p>The microscaling modes ({@code "mode": "mxfp4"} / {@code "mxfp8"}, OCP MX) pack the codes the same way but
 * store no biases: each group of 32 shares a power-of-two scale, a uint8 E8M0 exponent in {@code X.scales}, and
 * <pre>
 *     w[r][c] = fp(q[r][c]) * 2^(scales[r][c / 32] - 127)              (E8M0 255 is NaN)
 * </pre>
 * with {@code fp} the FP4 E2M1 value of a 4-bit code (mxfp4: sign bit 3, then {@code 0, 0.5, 1, 1.5, 2, 3, 4, 6})
 * or the FP8 E4M3 value of an 8-bit code (mxfp8). {@code nvfp4} (groups of 16 with E4M3 scales) is refused.
 */
public final class MlxQuant {

    /** The {@code quantization.mode} of the checkpoint. */
    public enum Mode {
        /** {@code scale * q + bias}, any of 2 / 4 / 8 bits. */
        AFFINE("affine", 0),
        /** FP4 E2M1 codes, E8M0 scale per 32. */
        MXFP4("mxfp4", 4),
        /** FP8 E4M3 codes, E8M0 scale per 32. */
        MXFP8("mxfp8", 8);

        /** The config.json spelling. */
        public final String key;
        /** Bits per code of a microscaling mode, 0 for affine (the shapes decide). */
        final int bits;

        Mode(String key, int bits) {
            this.key = key;
            this.bits = bits;
        }
    }

    /** The group size of the microscaling modes (the OCP MX block). */
    public static final int MX_GROUP = 32;

    /** The 16 values of an FP4 E2M1 code: bit 3 the sign, bits 2-1 the exponent (bias 1), bit 0 the mantissa. */
    private static final float[] E2M1 = {0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f};

    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    /** The default group size of the checkpoint ({@code quantization.group_size}). */
    public final int groupSize;
    /** The default bit width ({@code quantization.bits}); informational, each tensor's own follows from its shapes. */
    public final int bits;
    /** How codes become values. */
    public final Mode mode;

    private MlxQuant(int groupSize, int bits, Mode mode) {
        this.groupSize = groupSize;
        this.bits = bits;
        this.mode = mode;
    }

    /** The value of an FP4 E2M1 code (the low 4 bits of {@code code}). */
    public static float e2m1ToFloat(int code) {
        return E2M1[code & 0xF];
    }

    /** The value of an E8M0 scale byte: {@code 2^(e - 127)}, NaN for 255. */
    public static float e8m0ToFloat(int e) {
        e &= 0xFF;
        return e == 0xFF ? Float.NaN : Math.scalb(1f, e - 127);
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
        String key = Json.str(q, "mode", "affine");
        Mode mode = null;
        for (Mode m : Mode.values()) {
            if (m.key.equals(key)) {
                mode = m;
            }
        }
        if (mode == null) {
            throw new IOException("MLX quantization mode '" + key + "' is not supported (affine, mxfp4, mxfp8)");
        }
        int groupSize = ((Number) q.get("group_size")).intValue();
        int bits = ((Number) q.get("bits")).intValue();
        if (mode != Mode.AFFINE && (groupSize != MX_GROUP || bits != mode.bits)) {
            throw new IOException("MLX " + key + " needs group_size " + MX_GROUP + " and bits " + mode.bits + ", got "
                    + groupSize + " / " + bits);
        }
        return new MlxQuant(groupSize, bits, mode);
    }

    /**
     * The float view of a packed MLX matrix: shape {@code [out, in]}, values dequantized on read. The group size is
     * the checkpoint's unless the shapes rule it out; the bit width follows from the shapes.
     *
     * @param biases the affine offsets; ignored (and may be null) in the microscaling modes
     * @throws IOException if the shapes do not fit together or the bit width is not 2, 4 or 8
     */
    public SafeTensors.Tensor dequantized(String name, SafeTensors.Tensor packed, SafeTensors.Tensor scales,
                                          SafeTensors.Tensor biases) throws IOException {
        if (mode != Mode.AFFINE) {
            return microscaled(name, packed, scales);
        }
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

    /** mxfp4 / mxfp8: {@code bits}-wide codes packed like affine ones, one uint8 E8M0 exponent per 32 columns. */
    private SafeTensors.Tensor microscaled(String name, SafeTensors.Tensor packed, SafeTensors.Tensor scales)
            throws IOException {
        int b = mode.bits;
        if (packed.shape().length != 2 || scales.shape().length != 2 || packed.dtype() != SafeTensors.DType.U32
                || scales.dtype() != SafeTensors.DType.U8 || scales.dim(0) != packed.dim(0)
                || (long) packed.dim(1) * 32 != (long) scales.dim(1) * MX_GROUP * b) {
            throw new IOException("malformed MLX " + mode.key + " tensor " + name + ": " + packed + ", " + scales
                    + " (expected U32 [out, in * " + b + " / 32] and U8 [out, in / " + MX_GROUP + "])");
        }
        int rows = packed.dim(0);
        int groups = scales.dim(1);
        int cols = groups * MX_GROUP;
        int words = packed.dim(1);
        int perWord = 32 / b;
        int mask = (1 << b) - 1;
        boolean fp4 = mode == Mode.MXFP4;
        SafeTensors.Tensor.Reader reader = (start, dst, off, n) -> {
            float[] s = new float[groups];
            long row = -1;
            for (int i = 0; i < n; i++) {
                long e = start + i;
                long r = e / cols;
                int c = (int) (e % cols);
                if (r != row) {
                    row = r;
                    for (int g = 0; g < groups; g++) {
                        s[g] = e8m0ToFloat(scales.data().get(ValueLayout.JAVA_BYTE, r * groups + g));
                    }
                }
                int word = packed.data().getAtIndex(INT_LE, r * words + c / perWord);
                int q = (word >>> ((c % perWord) * b)) & mask;
                dst[off + i] = (fp4 ? E2M1[q] : SafeTensors.e4m3ToFloat(q)) * s[c / MX_GROUP];
            }
        };
        return new SafeTensors.Tensor(name, packed.dtype(), new long[]{rows, cols}, packed.data(), reader);
    }
}
