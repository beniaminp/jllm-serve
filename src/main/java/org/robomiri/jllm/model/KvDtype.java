package org.robomiri.jllm.model;

/** Element type of the KV cache ({@code backend.kv-dtype}): plain f32, or f16 to halve its memory and traffic. */
public enum KvDtype {
    F32(4),
    F16(2);

    public final int bytes;

    KvDtype(int bytes) {
        this.bytes = bytes;
    }

    public static KvDtype parse(String s) {
        return switch (s.trim().toLowerCase()) {
            case "f32", "float32", "fp32" -> F32;
            case "f16", "float16", "fp16", "half" -> F16;
            default -> throw new IllegalArgumentException("backend.kv-dtype must be f32 or f16, got '" + s + "'");
        };
    }

    @Override
    public String toString() {
        return name().toLowerCase();
    }
}
