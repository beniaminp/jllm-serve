package org.robomiri.jllm.model;

/**
 * Element type of the KV cache ({@code backend.kv-dtype}): plain f32, f16 to halve its memory and traffic, or q8 to
 * quarter it: int8 values with one f32 scale per row of a kv head ({@code headDim} values of one position), symmetric,
 * {@code scale = max|x| / 127}, {@code q = rint(x / scale)}.
 */
public enum KvDtype {
    F32(4),
    F16(2),
    Q8(1);

    /** Bytes per stored value, without q8's per-row scales (see {@link #rowBytes}). */
    public final int bytes;

    KvDtype(int bytes) {
        this.bytes = bytes;
    }

    /**
     * Bytes of one position's row of {@code heads} heads, {@code width} values in all: the values, then for q8 one f32
     * scale per head. This is also the row stride of the Metal caches and of a saved cache.
     */
    public long rowBytes(int width, int heads) {
        return (long) width * bytes + (this == Q8 ? 4L * heads : 0);
    }

    /** {@link #rowBytes} in units of the stored value: the row stride the Metal attention kernels take. */
    public int rowStride(int width, int heads) {
        return (int) (rowBytes(width, heads) / bytes);
    }

    /** The q8 scale of a row whose largest magnitude is {@code max}: {@code max / 127} (0 for an all-zero row). */
    public static float q8Scale(float max) {
        return max / 127f;
    }

    /** {@code x} quantized with {@code scale}: {@code rint(x / scale)}, clamped to [-127, 127]; 0 when the scale is 0. */
    public static byte q8(float x, float scale) {
        if (scale == 0f) {
            return 0;
        }
        return (byte) Math.max(-127, Math.min(127, (int) Math.rint(x / scale)));
    }

    public static KvDtype parse(String s) {
        return switch (s.trim().toLowerCase()) {
            case "f32", "float32", "fp32" -> F32;
            case "f16", "float16", "fp16", "half" -> F16;
            case "q8", "int8" -> Q8;
            default -> throw new IllegalArgumentException("backend.kv-dtype must be f32, f16 or q8, got '" + s + "'");
        };
    }

    @Override
    public String toString() {
        return name().toLowerCase();
    }
}
