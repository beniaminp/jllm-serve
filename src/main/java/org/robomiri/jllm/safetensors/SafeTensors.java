package org.robomiri.jllm.safetensors;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.stream.Stream;

/**
 * Reader for the <a href="https://github.com/huggingface/safetensors">safetensors</a> format.
 *
 * <p>File layout: {@code u64 little-endian header size N}, {@code N bytes of JSON header}, then raw tensor data.
 * Files are memory-mapped through the FFM API, so opening is instant and tensors are read lazily from the page cache.
 * Sharded checkpoints ({@code model.safetensors.index.json}) are supported.
 */
public final class SafeTensors implements AutoCloseable {

    public enum DType {
        F64(8), F32(4), F16(2), BF16(2), I64(8), I32(4), I16(2), I8(1), U64(8), U32(4), U16(2), U8(1), BOOL(1),
        /** OCP FP8 {@code float8_e4m3fn}: 1 sign, 4 exponent (bias 7), 3 mantissa bits, no infinities, NaN = S.1111.111. */
        F8_E4M3(1),
        /** OCP FP8 {@code float8_e5m2}: the upper byte of an IEEE half (bias 15, 2 mantissa bits, with inf and NaN). */
        F8_E5M2(1);

        public final int bytes;

        DType(int bytes) {
            this.bytes = bytes;
        }
    }

    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfDouble DOUBLE_LE = ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    /** The 256 values of {@link DType#F8_E4M3} and of {@link DType#F8_E5M2}, indexed by the unsigned byte. */
    private static final float[] E4M3 = new float[256];
    private static final float[] E5M2 = new float[256];

    static {
        for (int b = 0; b < 256; b++) {
            int exp = (b >>> 3) & 0xF;
            int man = b & 7;
            float v;
            if (exp == 0xF && man == 7) {
                v = Float.NaN;
            } else if (exp == 0) {
                v = Math.scalb((float) man, -9);             // subnormal: man/8 * 2^(1-7)
            } else {
                v = Math.scalb((float) (8 + man), exp - 10); // (1 + man/8) * 2^(exp-7)
            }
            E4M3[b] = (b & 0x80) != 0 ? -v : v;
            E5M2[b] = Float.float16ToFloat((short) (b << 8));
        }
    }

    /** The exact value of an FP8 E4M3 ({@code float8_e4m3fn}) byte. */
    public static float e4m3ToFloat(int b) {
        return E4M3[b & 0xFF];
    }

    /** The exact value of an FP8 E5M2 byte (the upper half of a float16). */
    public static float e5m2ToFloat(int b) {
        return E5M2[b & 0xFF];
    }

    /**
     * A tensor view over mapped file memory, or a computed tensor ({@code reader} != null: its floats are produced by
     * the reader, e.g. a dequantized MLX matrix; {@code data} is then the packed source and {@code dtype} its type).
     */
    public record Tensor(String name, DType dtype, long[] shape, MemorySegment data, Reader reader) {

        /** Produces {@code n} floats from flat element index {@code start}. */
        public interface Reader {
            void toFloat(long start, float[] dst, int dstOff, int n);
        }

        public Tensor(String name, DType dtype, long[] shape, MemorySegment data) {
            this(name, dtype, shape, data, null);
        }

        public long numel() {
            long n = 1;
            for (long d : shape) {
                n = Math.multiplyExact(n, d);
            }
            return n;
        }

        public int dim(int i) {
            return Math.toIntExact(shape[i]);
        }

        /**
         * Converts {@code n} elements starting at flat element index {@code start} to float.
         *
         * <p>Half-precision data is staged into a per-thread {@code short[]} with one bulk copy and converted in
         * a plain array loop, instead of one bounds-checked segment access per element.
         */
        public void toFloat(long start, float[] dst, int dstOff, int n) {
            if (reader != null) {
                reader.toFloat(start, dst, dstOff, n);
                return;
            }
            switch (dtype) {
                case F32 -> MemorySegment.copy(data, FLOAT_LE, start * 4, dst, dstOff, n);
                case BF16 -> {
                    short[] s = shortScratch(n);
                    MemorySegment.copy(data, SHORT_LE, start * 2, s, 0, n);
                    for (int i = 0; i < n; i++) {
                        dst[dstOff + i] = Float.intBitsToFloat((s[i] & 0xFFFF) << 16);
                    }
                }
                case F16 -> {
                    short[] s = shortScratch(n);
                    MemorySegment.copy(data, SHORT_LE, start * 2, s, 0, n);
                    for (int i = 0; i < n; i++) {
                        dst[dstOff + i] = Float.float16ToFloat(s[i]);
                    }
                }
                case F8_E4M3, F8_E5M2 -> {
                    byte[] b = byteScratch(n);
                    MemorySegment.copy(data, ValueLayout.JAVA_BYTE, start, b, 0, n);
                    float[] table = dtype == DType.F8_E4M3 ? E4M3 : E5M2;
                    for (int i = 0; i < n; i++) {
                        dst[dstOff + i] = table[b[i] & 0xFF];
                    }
                }
                case F64 -> {
                    double[] d = new double[n];
                    MemorySegment.copy(data, DOUBLE_LE, start * 8, d, 0, n);
                    for (int i = 0; i < n; i++) {
                        dst[dstOff + i] = (float) d[i];
                    }
                }
                default -> throw new UnsupportedOperationException("cannot convert " + dtype + " tensor " + name + " to float");
            }
        }

        private static final ThreadLocal<short[]> SHORT_SCRATCH = ThreadLocal.withInitial(() -> new short[0]);
        private static final ThreadLocal<byte[]> BYTE_SCRATCH = ThreadLocal.withInitial(() -> new byte[0]);

        private static byte[] byteScratch(int n) {
            byte[] b = BYTE_SCRATCH.get();
            if (b.length < n) {
                b = new byte[Math.max(n, 4096)];
                BYTE_SCRATCH.set(b);
            }
            return b;
        }

        private static short[] shortScratch(int n) {
            short[] s = SHORT_SCRATCH.get();
            if (s.length < n) {
                s = new short[Math.max(n, 4096)];
                SHORT_SCRATCH.set(s);
            }
            return s;
        }

        public float[] toFloatArray() {
            float[] out = new float[Math.toIntExact(numel())];
            toFloat(0, out, 0, out.length);
            return out;
        }

        @Override
        public String toString() {
            return name + " " + dtype + Arrays.toString(shape);
        }
    }

    private final Arena arena;
    private final Map<String, Tensor> tensors;
    private final Map<String, String> metadata;

    private SafeTensors(Arena arena, Map<String, Tensor> tensors, Map<String, String> metadata) {
        this.arena = arena;
        this.tensors = tensors;
        this.metadata = metadata;
    }

    /** True for a plain {@code .safetensors} file name: no directory components, no {@code ..}, not blank. */
    public static boolean isShardName(String s) {
        return s != null && !s.isBlank() && !s.contains("/") && !s.contains("\\") && !s.contains("..")
                && s.indexOf('\0') < 0 && s.endsWith(".safetensors");
    }

    /**
     * Reads the {@code weight_map} of a {@code model.safetensors.index.json}: tensor name -> shard file name.
     * The index is downloaded data, so every shard name must satisfy {@link #isShardName}.
     */
    public static Map<String, String> readWeightMap(Path index) throws IOException {
        Map<String, Object> weightMap;
        try {
            weightMap = Json.map(Json.parse(Files.readString(index)), "weight_map");
        } catch (RuntimeException e) {
            throw new IOException(index + ": " + e.getMessage(), e);
        }
        if (weightMap == null) {
            throw new IOException(index + " has no weight_map");
        }
        Map<String, String> m = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : weightMap.entrySet()) {
            if (!(e.getValue() instanceof String s) || !isShardName(s)) {
                throw new IOException(index + ": invalid shard file name " + e.getValue() + " for " + e.getKey());
            }
            m.put(e.getKey(), s);
        }
        return m;
    }

    /** Opens a single {@code .safetensors} file or a model directory (single file, sharded index or all files). */
    public static SafeTensors open(Path path) throws IOException {
        List<Path> files;
        if (Files.isDirectory(path)) {
            Path index = path.resolve("model.safetensors.index.json");
            if (Files.isRegularFile(index)) {
                files = new TreeSet<>(readWeightMap(index).values()).stream().map(path::resolve).toList();
            } else {
                try (Stream<Path> s = Files.list(path)) {
                    files = s.filter(p -> p.getFileName().toString().endsWith(".safetensors")).sorted().toList();
                }
            }
            if (files.isEmpty()) {
                throw new IOException("no .safetensors files in " + path);
            }
        } else {
            files = List.of(path);
        }
        Arena arena = Arena.ofShared();
        try {
            Map<String, Tensor> tensors = new LinkedHashMap<>();
            Map<String, Path> definedIn = new java.util.HashMap<>();
            Map<String, String> metadata = new LinkedHashMap<>();
            for (Path f : files) {
                readFile(f, arena, tensors, definedIn, metadata);
            }
            return new SafeTensors(arena, Collections.unmodifiableMap(tensors), Collections.unmodifiableMap(metadata));
        } catch (IOException | RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    /** A tensor's byte range in the data section, for the overlap check. */
    private record Range(String name, long begin, long end) {
    }

    /**
     * Parses one file's header into {@code tensors}. A name already defined (by this file or an earlier shard)
     * and tensors whose {@code data_offsets} overlap are rejected: both mean the file does not describe one
     * tensor per name, and loading it would silently read the wrong data.
     */
    private static void readFile(Path file, Arena arena, Map<String, Tensor> tensors, Map<String, Path> definedIn,
                                 Map<String, String> metadata) throws IOException {
        MemorySegment mapped;
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            mapped = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
        }
        if (mapped.byteSize() < 8) {
            throw new IOException(file + ": too small for safetensors");
        }
        long headerLen = mapped.get(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 0);
        if (headerLen <= 0 || 8 + headerLen > mapped.byteSize() || headerLen > 100_000_000) {
            throw new IOException(file + ": invalid safetensors header length " + headerLen);
        }
        byte[] headerBytes = mapped.asSlice(8, headerLen).toArray(ValueLayout.JAVA_BYTE);
        Map<String, Object> header;
        try {
            header = Json.parseObject(new String(headerBytes, StandardCharsets.UTF_8));
        } catch (RuntimeException ex) {
            throw new IOException(file + ": invalid safetensors header: " + ex.getMessage(), ex);
        }
        long dataStart = 8 + headerLen;
        long dataSize = mapped.byteSize() - dataStart;
        List<Range> ranges = new java.util.ArrayList<>(header.size());
        for (Map.Entry<String, Object> e : header.entrySet()) {
            String name = e.getKey();
            if (name.equals("__metadata__")) {
                if (e.getValue() instanceof Map<?, ?> m) {
                    m.forEach((k, v) -> metadata.put(String.valueOf(k), String.valueOf(v)));
                }
                continue;
            }
            Path other = definedIn.get(name);
            if (other != null) {
                throw new IOException(file + ": duplicate tensor " + name + (other.equals(file) ? "" : " (also in " + other + ")"));
            }
            Object info = e.getValue();
            if (!(info instanceof Map)) {
                throw new IOException(file + ": malformed entry for " + name);
            }
            String dtypeName = Json.str(info, "dtype", null);
            DType dtype = dtypeName == null ? null : parseDType(dtypeName);
            if (dtype == null) {
                throw new IOException(file + (dtypeName == null ? ": missing dtype for tensor " + name
                        : ": unsupported dtype '" + dtypeName + "' for tensor " + name));
            }
            List<Object> shapeList = Json.list(info, "shape");
            List<Object> offsets = Json.list(info, "data_offsets");
            if (shapeList == null || offsets == null || offsets.size() != 2
                    || !shapeList.stream().allMatch(o -> o instanceof Number)
                    || !offsets.stream().allMatch(o -> o instanceof Number)) {
                throw new IOException(file + ": malformed entry for " + name);
            }
            long[] shape = shapeList.stream().mapToLong(o -> ((Number) o).longValue()).toArray();
            long begin = ((Number) offsets.get(0)).longValue();
            long end = ((Number) offsets.get(1)).longValue();
            long bytes;
            try {
                long numel = 1;
                for (long d : shape) {
                    if (d < 0) {
                        throw new IOException(file + ": negative dimension in shape of " + name);
                    }
                    numel = Math.multiplyExact(numel, d);
                }
                bytes = Math.multiplyExact(numel, (long) dtype.bytes);
            } catch (ArithmeticException ex) {
                throw new IOException(file + ": shape of " + name + " overflows", ex);
            }
            if (begin < 0 || end < begin || end > dataSize || end - begin != bytes) {
                throw new IOException(file + ": bad data_offsets for " + name + " (" + begin + ".." + end
                        + ", expected " + bytes + " bytes, " + dataSize + " available)");
            }
            MemorySegment slice;
            try {
                slice = mapped.asSlice(dataStart + begin, end - begin);
            } catch (RuntimeException ex) {
                throw new IOException(file + ": cannot map tensor " + name, ex);
            }
            tensors.put(name, new Tensor(name, dtype, shape, slice));
            definedIn.put(name, file);
            if (end > begin) {
                ranges.add(new Range(name, begin, end));
            }
        }
        ranges.sort(java.util.Comparator.comparingLong(Range::begin));
        for (int i = 1; i < ranges.size(); i++) {
            Range prev = ranges.get(i - 1);
            Range cur = ranges.get(i);
            if (cur.begin() < prev.end()) {
                throw new IOException(file + ": data_offsets of " + prev.name() + " (" + prev.begin() + ".." + prev.end()
                        + ") and " + cur.name() + " (" + cur.begin() + ".." + cur.end() + ") overlap");
            }
        }
    }

    private static DType parseDType(String s) {
        for (DType t : DType.values()) {
            if (t.name().equals(s)) {
                return t;
            }
        }
        return null;
    }

    public Map<String, Tensor> tensors() {
        return tensors;
    }

    public Map<String, String> metadata() {
        return metadata;
    }

    public boolean has(String name) {
        return tensors.containsKey(name);
    }

    public Tensor get(String name) {
        Tensor t = tensors.get(name);
        if (t == null) {
            throw new IllegalArgumentException("tensor not found: " + name);
        }
        return t;
    }

    @Override
    public void close() {
        arena.close();
    }
}
