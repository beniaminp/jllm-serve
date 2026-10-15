package org.robomiri.jllm.model;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * jllm's own on-disk format for quantized weights, so a model is quantized once and memory-mapped afterwards.
 *
 * <p>Two files: {@code jllm-<quant>.bin} holds page-aligned blobs in exactly the {@link QTensor} memory layout
 * (so a mapped slice can be used directly, even zero-copy by Metal), and {@code jllm-<quant>.json} describes
 * them: {@code {"tensors": {name: {type, rows, cols, offset, size}}}}. The JSON is written last and acts as the
 * "conversion complete" marker.
 */
public final class QuantFile implements AutoCloseable {

    public static final int VERSION = 1;

    public record Entry(String type, int rows, int cols, long offset, long size) {
    }

    private final Arena arena;
    private final MemorySegment mapped;
    private final Map<String, Entry> entries;

    private QuantFile(Arena arena, MemorySegment mapped, Map<String, Entry> entries) {
        this.arena = arena;
        this.mapped = mapped;
        this.entries = entries;
    }

    public static boolean exists(Path dir, QType quant) {
        return Files.isRegularFile(jsonPath(dir, quant)) && Files.isRegularFile(binPath(dir, quant));
    }

    public static Path binPath(Path dir, QType quant) {
        return dir.resolve("jllm-" + quant.name().toLowerCase() + ".bin");
    }

    public static Path jsonPath(Path dir, QType quant) {
        return dir.resolve("jllm-" + quant.name().toLowerCase() + ".json");
    }

    public static QuantFile open(Path dir, QType quant) throws IOException {
        Path json = jsonPath(dir, quant);
        Map<String, Entry> entries = readIndex(json, Json.map(parseJson(json), "tensors"));
        Path bin = binPath(dir, quant);
        Arena arena = Arena.ofShared();
        try {
            MemorySegment mapped;
            try (FileChannel ch = FileChannel.open(bin, StandardOpenOption.READ)) {
                mapped = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
            }
            for (Map.Entry<String, Entry> e : entries.entrySet()) {
                checkEntry(e.getKey(), e.getValue(), mapped.byteSize(), bin);
            }
            return new QuantFile(arena, mapped, entries);
        } catch (IOException | RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    private static Map<String, Object> parseJson(Path json) throws IOException {
        Map<String, Object> header;
        try {
            header = Json.parseObject(Files.readString(json));
        } catch (RuntimeException e) {
            throw new IOException(json + " is not a valid jllm cache index: " + e.getMessage() + "; delete it to reconvert", e);
        }
        if (Json.num(header, "version", 0) != VERSION) {
            throw new IOException("unsupported jllm cache version; delete " + json);
        }
        return header;
    }

    /** Parses a {@code {name: {type, rows, cols, offset, size}}} map, rejecting anything malformed. */
    static Map<String, Entry> readIndex(Path json, Map<String, Object> tensors) throws IOException {
        if (tensors == null) {
            throw new IOException(json + " has no \"tensors\" map; delete it to reconvert");
        }
        Map<String, Entry> entries = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : tensors.entrySet()) {
            Object v = e.getValue();
            if (!(v instanceof Map)) {
                throw new IOException(json + ": malformed entry for " + e.getKey());
            }
            entries.put(e.getKey(), new Entry(Json.str(v, "type", "?"), (int) Json.num(v, "rows", -1),
                    (int) Json.num(v, "cols", -1), Json.num(v, "offset", -1), Json.num(v, "size", -1)));
        }
        return entries;
    }

    /** Expected byte size of an entry in the file layout, or -1 if its type is unknown. */
    static long expectedSize(Entry e) {
        if (e.type().equals("F32")) {
            return QTensor.align((long) e.rows() * e.cols() * Float.BYTES, QTensor.PAGE);
        }
        for (QType t : QType.values()) {
            if (t.name().equals(e.type())) {
                return e.cols() % QType.BLOCK == 0 ? QTensor.byteSize(t, e.rows(), e.cols()) : -1;
            }
        }
        return -1;
    }

    /** Verifies that an index entry describes a well-formed, in-bounds blob of {@code fileSize} bytes. */
    static void checkEntry(String name, Entry e, long fileSize, Path bin) throws IOException {
        String problem = null;
        if (e.rows() < 0 || e.cols() < 0 || e.offset() < 0 || e.size() < 0) {
            problem = "negative field";
        } else if (e.offset() % QTensor.PAGE != 0) {
            problem = "offset is not page aligned";
        } else if (expectedSize(e) != e.size()) {
            problem = "size " + e.size() + " does not match " + e.rows() + "x" + e.cols() + " " + e.type();
        } else {
            long end;
            try {
                end = Math.addExact(e.offset(), e.size());
            } catch (ArithmeticException ex) {
                end = Long.MAX_VALUE;
            }
            if (end > fileSize) {
                problem = "extends to " + end + " but " + bin + " has " + fileSize + " bytes";
            }
        }
        if (problem != null) {
            throw new IOException("corrupt jllm cache entry " + name + ": " + problem + "; delete " + bin
                    + " and its .json to reconvert");
        }
    }

    public boolean has(String name) {
        return entries.containsKey(name);
    }

    public QTensor tensor(String name) {
        Entry e = entry(name);
        return QTensor.view(QType.valueOf(e.type()), e.rows(), e.cols(), mapped.asSlice(e.offset(), e.size()));
    }

    public float[] floats(String name) {
        Entry e = entry(name);
        if (!e.type().equals("F32")) {
            throw new IllegalArgumentException(name + " is " + e.type() + ", not F32");
        }
        float[] out = new float[e.rows() * e.cols()];
        MemorySegment.copy(mapped, ValueLayout.JAVA_FLOAT, e.offset(), out, 0, out.length);
        return out;
    }

    private Entry entry(String name) {
        Entry e = entries.get(name);
        if (e == null) {
            throw new IllegalArgumentException("tensor not in cache: " + name);
        }
        return e;
    }

    public long byteSize() {
        return mapped.byteSize();
    }

    @Override
    public void close() {
        arena.close();
    }

    /** Streams tensors into a new cache; call {@link #finish()} to write the index. */
    public static final class Writer implements AutoCloseable {
        private final Path dir;
        private final QType quant;
        private final Path bin;
        private final FileChannel ch;
        private final Map<String, Object> index = new LinkedHashMap<>();
        private long offset;

        public Writer(Path dir, QType quant) throws IOException {
            this.dir = dir;
            this.quant = quant;
            this.bin = binPath(dir, quant);
            Files.deleteIfExists(jsonPath(dir, quant));
            this.ch = FileChannel.open(bin, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                    StandardOpenOption.TRUNCATE_EXISTING);
        }

        public void put(String name, QTensor t) throws IOException {
            write(t.data);
            add(name, t.type.name(), t.rows, t.cols, t.data.byteSize());
        }

        public void putFloats(String name, float[] v) throws IOException {
            long size = QTensor.align((long) v.length * Float.BYTES, QTensor.PAGE);
            try (Arena a = Arena.ofConfined()) {
                MemorySegment seg = a.allocate(size, QTensor.PAGE);
                MemorySegment.copy(v, 0, seg, ValueLayout.JAVA_FLOAT, 0, v.length);
                write(seg);
            }
            add(name, "F32", 1, v.length, size);
        }

        private void add(String name, String type, int rows, int cols, long size) {
            if (index.containsKey(name)) {
                throw new IllegalArgumentException("duplicate tensor " + name);
            }
            index.put(name, Json.obj("type", type, "rows", rows, "cols", cols, "offset", offset, "size", size));
            offset += size;
        }

        private void write(MemorySegment seg) throws IOException {
            long pos = 0;
            long len = seg.byteSize();
            while (pos < len) {
                long n = Math.min(len - pos, 1L << 30);
                ByteBuffer bb = seg.asSlice(pos, n).asByteBuffer();
                while (bb.hasRemaining()) {
                    ch.write(bb);
                }
                pos += n;
            }
        }

        public long bytesWritten() {
            return offset;
        }

        public void finish() throws IOException {
            ch.force(true);
            ch.close();
            Files.writeString(jsonPath(dir, quant), Json.write(Json.obj("version", VERSION, "quant", quant.name(), "tensors", index)));
        }

        @Override
        public void close() throws IOException {
            if (ch.isOpen()) { // finish() was not called: leave no half-written cache behind
                ch.close();
                Files.deleteIfExists(bin);
            }
        }
    }
}
