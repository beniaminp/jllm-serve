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

    private final Arena arena = Arena.ofShared();
    private final MemorySegment mapped;
    private final Map<String, Entry> entries;

    private QuantFile(Path bin, Map<String, Entry> entries) throws IOException {
        this.entries = entries;
        try (FileChannel ch = FileChannel.open(bin, StandardOpenOption.READ)) {
            mapped = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
        }
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
        Map<String, Object> header = Json.parseObject(Files.readString(jsonPath(dir, quant)));
        if (Json.num(header, "version", 0) != VERSION) {
            throw new IOException("unsupported jllm cache version; delete " + jsonPath(dir, quant));
        }
        Map<String, Entry> entries = new LinkedHashMap<>();
        Map<String, Object> tensors = Json.map(header, "tensors");
        for (Map.Entry<String, Object> e : tensors.entrySet()) {
            Object v = e.getValue();
            entries.put(e.getKey(), new Entry(Json.str(v, "type", "?"), (int) Json.num(v, "rows", 0),
                    (int) Json.num(v, "cols", 0), Json.num(v, "offset", 0), Json.num(v, "size", 0)));
        }
        return new QuantFile(binPath(dir, quant), entries);
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
