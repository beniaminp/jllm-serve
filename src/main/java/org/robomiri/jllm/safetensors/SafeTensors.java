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
        F64(8), F32(4), F16(2), BF16(2), I64(8), I32(4), I16(2), I8(1), U8(1), BOOL(1);

        public final int bytes;

        DType(int bytes) {
            this.bytes = bytes;
        }
    }

    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfDouble DOUBLE_LE = ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    /** A tensor view over mapped file memory. */
    public record Tensor(String name, DType dtype, long[] shape, MemorySegment data) {

        public long numel() {
            long n = 1;
            for (long d : shape) {
                n *= d;
            }
            return n;
        }

        public int dim(int i) {
            return Math.toIntExact(shape[i]);
        }

        /** Converts {@code n} elements starting at flat element index {@code start} to float. */
        public void toFloat(long start, float[] dst, int dstOff, int n) {
            switch (dtype) {
                case F32 -> MemorySegment.copy(data, FLOAT_LE, start * 4, dst, dstOff, n);
                case BF16 -> {
                    for (int i = 0; i < n; i++) {
                        dst[dstOff + i] = Float.intBitsToFloat((data.get(SHORT_LE, (start + i) * 2) & 0xFFFF) << 16);
                    }
                }
                case F16 -> {
                    for (int i = 0; i < n; i++) {
                        dst[dstOff + i] = Float.float16ToFloat(data.get(SHORT_LE, (start + i) * 2));
                    }
                }
                case F64 -> {
                    for (int i = 0; i < n; i++) {
                        dst[dstOff + i] = (float) data.get(DOUBLE_LE, (start + i) * 8);
                    }
                }
                default -> throw new UnsupportedOperationException("cannot convert " + dtype + " tensor " + name + " to float");
            }
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

    /** Opens a single {@code .safetensors} file or a model directory (single file, sharded index or all files). */
    public static SafeTensors open(Path path) throws IOException {
        List<Path> files;
        if (Files.isDirectory(path)) {
            Path index = path.resolve("model.safetensors.index.json");
            if (Files.isRegularFile(index)) {
                Map<String, Object> weightMap = Json.map(Json.parse(Files.readString(index)), "weight_map");
                if (weightMap == null) {
                    throw new IOException(index + " has no weight_map");
                }
                TreeSet<String> shards = new TreeSet<>();
                weightMap.values().forEach(v -> shards.add((String) v));
                files = shards.stream().map(path::resolve).toList();
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
            Map<String, String> metadata = new LinkedHashMap<>();
            for (Path f : files) {
                readFile(f, arena, tensors, metadata);
            }
            return new SafeTensors(arena, Collections.unmodifiableMap(tensors), Collections.unmodifiableMap(metadata));
        } catch (IOException | RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    private static void readFile(Path file, Arena arena, Map<String, Tensor> tensors, Map<String, String> metadata)
            throws IOException {
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
        Map<String, Object> header = Json.parseObject(new String(headerBytes, StandardCharsets.UTF_8));
        long dataStart = 8 + headerLen;
        long dataSize = mapped.byteSize() - dataStart;
        for (Map.Entry<String, Object> e : header.entrySet()) {
            String name = e.getKey();
            if (name.equals("__metadata__")) {
                if (e.getValue() instanceof Map<?, ?> m) {
                    m.forEach((k, v) -> metadata.put(String.valueOf(k), String.valueOf(v)));
                }
                continue;
            }
            Object info = e.getValue();
            DType dtype = DType.valueOf(Json.str(info, "dtype", "?"));
            List<Object> shapeList = Json.list(info, "shape");
            List<Object> offsets = Json.list(info, "data_offsets");
            if (shapeList == null || offsets == null || offsets.size() != 2) {
                throw new IOException(file + ": malformed entry for " + name);
            }
            long[] shape = shapeList.stream().mapToLong(o -> ((Number) o).longValue()).toArray();
            long begin = ((Number) offsets.get(0)).longValue();
            long end = ((Number) offsets.get(1)).longValue();
            long numel = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
            if (begin < 0 || end > dataSize || end - begin != numel * dtype.bytes) {
                throw new IOException(file + ": bad data_offsets for " + name);
            }
            tensors.put(name, new Tensor(name, dtype, shape, mapped.asSlice(dataStart + begin, end - begin)));
        }
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
