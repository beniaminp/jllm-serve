package org.robomiri.jllm;

import org.robomiri.jllm.gguf.GgmlType;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** A minimal GGUF v3 writer for tests: typed metadata, tensors with raw block bytes, a chosen alignment. */
final class GgufWriter {

    static final int U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, BOOL = 7, STRING = 8, ARRAY = 9,
            U64 = 10, I64 = 11, F64 = 12;

    private final List<Object[]> kv = new ArrayList<>();
    private final List<Object[]> tensors = new ArrayList<>();
    private long alignment = 32;
    private int version = 3;

    GgufWriter version(int v) {
        version = v;
        return this;
    }

    GgufWriter alignment(long a) {
        alignment = a;
        return this;
    }

    private static ByteBuffer buf(int n) {
        return ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN);
    }

    private static void putString(ByteArrayOutputStream o, String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        o.writeBytes(buf(8).putLong(b.length).array());
        o.writeBytes(b);
    }

    private void put(String key, int type, byte[] value) {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        o.writeBytes(buf(4).putInt(type).array());
        o.writeBytes(value);
        kv.add(new Object[]{key, o.toByteArray()});
    }

    GgufWriter u32(String key, long v) {
        put(key, U32, buf(4).putInt((int) v).array());
        return this;
    }

    GgufWriter i32(String key, int v) {
        put(key, I32, buf(4).putInt(v).array());
        return this;
    }

    GgufWriter u64(String key, long v) {
        put(key, U64, buf(8).putLong(v).array());
        return this;
    }

    GgufWriter f32(String key, float v) {
        put(key, F32, buf(4).putFloat(v).array());
        return this;
    }

    GgufWriter f64(String key, double v) {
        put(key, F64, buf(8).putDouble(v).array());
        return this;
    }

    GgufWriter bool(String key, boolean v) {
        put(key, BOOL, new byte[]{(byte) (v ? 1 : 0)});
        return this;
    }

    GgufWriter u8(String key, int v) {
        put(key, U8, new byte[]{(byte) v});
        return this;
    }

    GgufWriter i16(String key, int v) {
        put(key, I16, buf(2).putShort((short) v).array());
        return this;
    }

    GgufWriter str(String key, String v) {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        putString(o, v);
        put(key, STRING, o.toByteArray());
        return this;
    }

    GgufWriter strings(String key, String... v) {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        o.writeBytes(buf(12).putInt(STRING).putLong(v.length).array());
        for (String s : v) {
            putString(o, s);
        }
        put(key, ARRAY, o.toByteArray());
        return this;
    }

    GgufWriter i32s(String key, int... v) {
        ByteBuffer b = buf(12 + 4 * v.length).putInt(I32).putLong(v.length);
        for (int x : v) {
            b.putInt(x);
        }
        put(key, ARRAY, b.array());
        return this;
    }

    GgufWriter f32s(String key, float... v) {
        ByteBuffer b = buf(12 + 4 * v.length).putInt(F32).putLong(v.length);
        for (float x : v) {
            b.putFloat(x);
        }
        put(key, ARRAY, b.array());
        return this;
    }

    /** An array of arrays of i32 (nested arrays are legal GGUF). */
    GgufWriter nested(String key, int[]... v) {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        o.writeBytes(buf(12).putInt(ARRAY).putLong(v.length).array());
        for (int[] a : v) {
            ByteBuffer b = buf(12 + 4 * a.length).putInt(I32).putLong(a.length);
            for (int x : a) {
                b.putInt(x);
            }
            o.writeBytes(b.array());
        }
        put(key, ARRAY, o.toByteArray());
        return this;
    }

    /** @param dims ggml order (innermost first) */
    GgufWriter tensor(String name, GgmlType type, long[] dims, byte[] data) {
        tensors.add(new Object[]{name, type, dims, data});
        return this;
    }

    /** An F32 tensor, dims innermost first. */
    GgufWriter f32Tensor(String name, long[] dims, float[] v) {
        ByteBuffer b = buf(4 * v.length);
        for (float x : v) {
            b.putFloat(x);
        }
        return tensor(name, GgmlType.F32, dims, b.array());
    }

    byte[] toBytes() {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        o.writeBytes(buf(4).putInt(0x46554747).array());
        o.writeBytes(buf(4).putInt(version).array());
        o.writeBytes(buf(16).putLong(tensors.size()).putLong(kv.size()).array());
        for (Object[] e : kv) {
            putString(o, (String) e[0]);
            o.writeBytes((byte[]) e[1]);
        }
        long off = 0;
        List<Long> offsets = new ArrayList<>();
        for (Object[] t : tensors) {
            long[] dims = (long[]) t[2];
            byte[] data = (byte[]) t[3];
            putString(o, (String) t[0]);
            o.writeBytes(buf(4).putInt(dims.length).array());
            for (long d : dims) {
                o.writeBytes(buf(8).putLong(d).array());
            }
            o.writeBytes(buf(4).putInt(((GgmlType) t[1]).id).array());
            off = (off + alignment - 1) / alignment * alignment;
            o.writeBytes(buf(8).putLong(off).array());
            offsets.add(off);
            off += data.length;
        }
        while (o.size() % alignment != 0) {
            o.write(0);
        }
        long start = o.size();
        for (int i = 0; i < tensors.size(); i++) {
            while (o.size() < start + offsets.get(i)) {
                o.write(0);
            }
            o.writeBytes((byte[]) tensors.get(i)[3]);
        }
        return o.toByteArray();
    }

    void write(Path p) throws IOException {
        Files.write(p, toBytes());
    }
}
