package org.robomiri.jllm.model;

/**
 * Element type of the KV cache ({@code backend.kv-dtype}): plain f32, f16 to halve its memory and traffic, q8 to
 * nearly quarter it: int8 values with one f32 scale per row of a kv head ({@code headDim} values of one position),
 * symmetric, {@code scale = max|x| / 127}, {@code q = rint(x / scale)}; or q4, about an eighth: Walsh–Hadamard rotated
 * keys, 4-bit codes and a half scale per group of 32 values ({@link KvQ4}), plus the latest
 * {@code backend.kv-q4-recent} positions kept as halves. q8 optionally rotates the keys like q4
 * ({@code backend.kv-q8-rotate}).
 */
public enum KvDtype {
    F32(4),
    F16(2),
    Q8(1),
    Q4(1);

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
        if (this == Q4) { // nibbles, then one half scale per group of 32 (KvQ4)
            return width / 2 + 2L * (width / KvQ4.GROUP);
        }
        return (long) width * bytes + (this == Q8 ? 4L * heads : 0);
    }

    /** Default of {@code backend.kv-q4-recent}. */
    public static final int Q4_RECENT_DEFAULT = 128;

    private static volatile int q4Recent = Q4_RECENT_DEFAULT;

    /**
     * Positions a q4 cache keeps exact besides their q4 rows ({@code backend.kv-q4-recent}): set once from the settings
     * by the model loader, read when a backend creates its cache.
     */
    public static int q4Recent() {
        return q4Recent;
    }

    public static void setQ4Recent(int recent) {
        if (recent < 0) {
            throw new IllegalArgumentException("backend.kv-q4-recent must be >= 0, got " + recent);
        }
        q4Recent = recent;
    }

    /**
     * What a backend built without the settings (library use, tests) does: the plain q8 format. The shipped
     * {@code jllm.properties} sets {@code backend.kv-q8-rotate=true} (docs/07 §7.13).
     */
    public static final boolean Q8_ROTATE_DEFAULT = false;

    private static volatile boolean q8Rotate = Q8_ROTATE_DEFAULT;

    /**
     * Whether a q8 cache stores Walsh–Hadamard rotated keys ({@code backend.kv-q8-rotate}, see {@link KvQ4#block}): set
     * once from the settings by the model loader, read when a backend creates its cache. It only applies to
     * grouped-query attention with a head size that is a multiple of {@link KvQ4#GROUP}; elsewhere (DeepSeek's latent
     * cache, other head sizes) the cache stays plain q8.
     */
    public static boolean q8Rotate() {
        return q8Rotate;
    }

    public static void setQ8Rotate(boolean rotate) {
        q8Rotate = rotate;
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
            case "q4", "int4" -> Q4;
            default -> throw new IllegalArgumentException("backend.kv-dtype must be f32, f16, q8 or q4, got '" + s + "'");
        };
    }

    @Override
    public String toString() {
        return name().toLowerCase();
    }
}
