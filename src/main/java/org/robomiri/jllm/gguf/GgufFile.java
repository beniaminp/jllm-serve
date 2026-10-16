package org.robomiri.jllm.gguf;

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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A memory-mapped GGUF file (versions 2 and 3): the key/value metadata, the tensor directory and zero-copy
 * access to each tensor's bytes. See docs/16-gguf.md for the byte layout.
 *
 * <p>A model split over {@code name-00001-of-0000N.gguf} files is opened as one: any part may be named, all
 * {@code N} must exist, the metadata is that of part 1 (llama.cpp writes it only there) and the tensors of all
 * parts form one directory. Every file is validated like {@link org.robomiri.jllm.safetensors.SafeTensors}: the
 * header is parsed with bounds-checked reads, sizes are computed with overflow checks, and each tensor must lie
 * inside its file, be aligned to {@code general.alignment} and not overlap another, so a truncated or corrupt
 * file fails with an {@link IOException} naming the file and the field instead of reading outside the mapping.
 *
 * <p>Metadata values are {@link Long} (every integer type; a {@code u64} above 2^63 wraps), {@link Double},
 * {@link Boolean}, {@link String}, and for arrays {@code String[]}, {@code long[]}, {@code double[]},
 * {@code boolean[]} or {@code Object[]} (nested arrays).
 */
public final class GgufFile implements AutoCloseable {

    /** The bytes {@code G G U F} read as a little-endian int. */
    public static final int MAGIC = 0x46554747;
    public static final int DEFAULT_ALIGNMENT = 32;
    /** ggml's {@code GGML_MAX_DIMS}. */
    public static final int MAX_DIMS = 4;

    private static final ValueLayout.OfByte I8 = ValueLayout.JAVA_BYTE;
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;
    private static final ValueLayout.OfShort I16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(LE);
    private static final ValueLayout.OfInt I32 = ValueLayout.JAVA_INT_UNALIGNED.withOrder(LE);
    private static final ValueLayout.OfLong I64 = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(LE);
    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(LE);
    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(LE);

    private static final Pattern SPLIT = Pattern.compile("^(.*)-(\\d{5})-of-(\\d{5})\\.gguf$");

    // metadata value types
    private static final int T_U8 = 0, T_I8 = 1, T_U16 = 2, T_I16 = 3, T_U32 = 4, T_I32 = 5, T_F32 = 6, T_BOOL = 7,
            T_STRING = 8, T_ARRAY = 9, T_U64 = 10, T_I64 = 11, T_F64 = 12;

    /**
     * A tensor directory entry.
     *
     * @param dims   ggml's {@code ne}: innermost first, so {@code dims[0]} is the row length (the columns of a
     *               matrix) and {@code dims[1]} the number of rows
     * @param part   the file (0-based part of a split model) holding the data
     * @param offset byte offset of the data from the start of that file
     * @param size   byte size of the data
     */
    public record Tensor(String name, long[] dims, GgmlType type, int part, long offset, long size) {

        /** Row length {@code ne[0]}. */
        public long cols() {
            return dims[0];
        }

        /** Number of rows: the product of the remaining dimensions. */
        public long rows() {
            long r = 1;
            for (int i = 1; i < dims.length; i++) {
                r *= dims[i];
            }
            return r;
        }

        public long elements() {
            return cols() * rows();
        }

        /** Bytes of one row. */
        public long rowBytes() {
            return type.rowBytes(cols());
        }

        @Override
        public String toString() {
            return name + " " + type + " " + java.util.Arrays.toString(dims);
        }
    }

    private record Part(Path path, MemorySegment map, long fileSize) {
    }

    private final Arena arena;
    private final List<Part> parts;
    private final Map<String, Object> metadata;
    private final Map<String, Tensor> tensors;
    private final int version;
    private final long alignment;

    private GgufFile(Arena arena, List<Part> parts, Map<String, Object> metadata, Map<String, Tensor> tensors,
                     int version, long alignment) {
        this.arena = arena;
        this.parts = parts;
        this.metadata = Collections.unmodifiableMap(metadata);
        this.tensors = Collections.unmodifiableMap(tensors);
        this.version = version;
        this.alignment = alignment;
    }

    /**
     * Opens {@code file}, and every other part when its name is {@code <stem>-0000i-of-0000N.gguf}.
     *
     * @throws IOException if a file is missing, truncated or not a valid GGUF v2/v3 file
     */
    public static GgufFile open(Path file) throws IOException {
        List<Path> files = splitFiles(file);
        Arena arena = Arena.ofShared();
        try {
            List<Part> parts = new ArrayList<>(files.size());
            Map<String, Object> metadata = null;
            Map<String, Tensor> tensors = new LinkedHashMap<>();
            int version = 0;
            long alignment = DEFAULT_ALIGNMENT;
            for (int i = 0; i < files.size(); i++) {
                Path p = files.get(i);
                MemorySegment map;
                try (FileChannel ch = FileChannel.open(p, StandardOpenOption.READ)) {
                    map = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
                }
                parts.add(new Part(p, map, map.byteSize()));
                Header h = parseHeader(p, map, i, tensors);
                if (i == 0) {
                    metadata = h.metadata;
                    version = h.version;
                    alignment = h.alignment;
                } else {
                    checkSplitMetadata(p, h.metadata, i, files.size());
                }
            }
            if (files.size() > 1) {
                long count = num(metadata.get("split.count"), files.size());
                if (count != files.size()) {
                    throw new IOException(files.getFirst() + ": split.count is " + count + " but " + files.size()
                            + " files are named -of-" + String.format("%05d", files.size()));
                }
            }
            return new GgufFile(arena, parts, metadata, tensors, version, alignment);
        } catch (IOException | RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    /** The files of the model {@code file} belongs to: just {@code file}, or all parts of a split, in order. */
    static List<Path> splitFiles(Path file) throws IOException {
        if (!Files.isRegularFile(file)) {
            throw new IOException("no such GGUF file: " + file);
        }
        Matcher m = SPLIT.matcher(file.getFileName().toString());
        if (!m.matches()) {
            return List.of(file);
        }
        int total = Integer.parseInt(m.group(3));
        if (total < 1 || total > 99999) {
            throw new IOException(file + ": invalid split count in the file name");
        }
        List<Path> out = new ArrayList<>(total);
        for (int i = 1; i <= total; i++) {
            Path p = file.resolveSibling(String.format("%s-%05d-of-%05d.gguf", m.group(1), i, total));
            if (!Files.isRegularFile(p)) {
                throw new IOException("split GGUF is incomplete: " + p.getFileName() + " is missing (parts 1.."
                        + total + " are needed; opened via " + file.getFileName() + ")");
            }
            out.add(p);
        }
        return out;
    }

    /** The names of all parts of the split set {@code fileName} (one of {@code stem-0000i-of-0000N.gguf}) belongs to, or just it. */
    public static List<String> splitNames(String fileName) {
        Matcher m = SPLIT.matcher(fileName);
        if (!m.matches()) {
            return List.of(fileName);
        }
        int total = Integer.parseInt(m.group(3));
        List<String> out = new ArrayList<>(Math.max(total, 1));
        for (int i = 1; i <= total; i++) {
            out.add(String.format("%s-%05d-of-%05d.gguf", m.group(1), i, total));
        }
        return out;
    }

    private static void checkSplitMetadata(Path p, Map<String, Object> md, int index, int total) throws IOException {
        Object no = md.get("split.no");
        if (no != null && num(no, -1) != index) {
            throw new IOException(p + ": split.no is " + no + " but the file name says part " + (index + 1));
        }
    }

    private record Header(Map<String, Object> metadata, int version, long alignment) {
    }

    /** Bounds-checked little-endian reader over a mapped file. */
    private static final class Cursor {
        final Path file;
        final MemorySegment seg;
        long pos;

        Cursor(Path file, MemorySegment seg) {
            this.file = file;
            this.seg = seg;
        }

        long remaining() {
            return seg.byteSize() - pos;
        }

        void need(long n, String what) throws IOException {
            if (n < 0 || n > remaining()) {
                throw new IOException(file + ": truncated header: " + what + " needs " + n + " bytes at offset " + pos
                        + " but only " + remaining() + " remain");
            }
        }

        int u8(String what) throws IOException {
            need(1, what);
            return seg.get(I8, pos++) & 0xFF;
        }

        int i8(String what) throws IOException {
            need(1, what);
            return seg.get(I8, pos++);
        }

        int u16(String what) throws IOException {
            need(2, what);
            int v = seg.get(I16, pos) & 0xFFFF;
            pos += 2;
            return v;
        }

        int i16(String what) throws IOException {
            need(2, what);
            int v = seg.get(I16, pos);
            pos += 2;
            return v;
        }

        int i32(String what) throws IOException {
            need(4, what);
            int v = seg.get(I32, pos);
            pos += 4;
            return v;
        }

        long u32(String what) throws IOException {
            return i32(what) & 0xFFFFFFFFL;
        }

        long i64(String what) throws IOException {
            need(8, what);
            long v = seg.get(I64, pos);
            pos += 8;
            return v;
        }

        float f32(String what) throws IOException {
            need(4, what);
            float v = seg.get(F32, pos);
            pos += 4;
            return v;
        }

        double f64(String what) throws IOException {
            need(8, what);
            double v = seg.get(F64, pos);
            pos += 8;
            return v;
        }

        /** A u64 that must be a plausible non-negative count or length (below 2^62). */
        long count(String what) throws IOException {
            long v = i64(what);
            if (v < 0 || v > (1L << 62)) {
                throw new IOException(file + ": invalid " + what + " " + Long.toUnsignedString(v));
            }
            return v;
        }

        String string(String what) throws IOException {
            long len = count(what + " length");
            need(len, what);
            if (len > Integer.MAX_VALUE - 8) {
                throw new IOException(file + ": " + what + " of " + len + " bytes is too long");
            }
            byte[] b = new byte[(int) len];
            MemorySegment.copy(seg, I8, pos, b, 0, b.length);
            pos += len;
            return new String(b, StandardCharsets.UTF_8);
        }
    }

    private static Header parseHeader(Path file, MemorySegment map, int part, Map<String, Tensor> tensors)
            throws IOException {
        Cursor c = new Cursor(file, map);
        if (map.byteSize() < 24) {
            throw new IOException(file + ": too small for GGUF (" + map.byteSize() + " bytes)");
        }
        int magic = c.i32("magic");
        if (magic != MAGIC) {
            throw new IOException(file + ": not a GGUF file (magic 0x" + Integer.toHexString(magic) + ")");
        }
        int version = c.i32("version");
        if (version == 1 || version < 1 || version > 3) {
            throw new IOException(file + ": unsupported GGUF version " + version + " (versions 2 and 3 are read)");
        }
        long tensorCount = c.count("tensor count");
        long kvCount = c.count("metadata count");
        // every entry takes more than 8 bytes, so a count beyond the file size is a corrupt header
        if (tensorCount > map.byteSize() / 8 || kvCount > map.byteSize() / 8) {
            throw new IOException(file + ": implausible counts: " + tensorCount + " tensors, " + kvCount
                    + " metadata entries in " + map.byteSize() + " bytes");
        }
        Map<String, Object> md = new LinkedHashMap<>();
        for (long i = 0; i < kvCount; i++) {
            String key = c.string("metadata key");
            int type = c.i32("type of " + key);
            Object v = readValue(c, type, key, 0);
            if (md.put(key, v) != null) {
                throw new IOException(file + ": duplicate metadata key " + key);
            }
        }
        long alignment = DEFAULT_ALIGNMENT;
        if (md.containsKey("general.alignment")) {
            alignment = num(md.get("general.alignment"), -1);
            if (alignment <= 0 || alignment > (1 << 20) || Long.bitCount(alignment) != 1) {
                throw new IOException(file + ": invalid general.alignment " + md.get("general.alignment")
                        + " (a power of two is required)");
            }
        }
        record Info(String name, long[] dims, GgmlType type, long offset, long size) {
        }
        List<Info> infos = new ArrayList<>((int) Math.min(tensorCount, 1 << 20));
        for (long i = 0; i < tensorCount; i++) {
            String name = c.string("tensor name");
            long nd = c.u32("dimension count of " + name);
            if (nd < 1 || nd > MAX_DIMS) {
                throw new IOException(file + ": tensor " + name + " has " + nd + " dimensions (1.." + MAX_DIMS + " allowed)");
            }
            long[] dims = new long[(int) nd];
            long elements = 1;
            for (int d = 0; d < nd; d++) {
                dims[d] = c.i64("dimension of " + name);
                if (dims[d] <= 0) {
                    throw new IOException(file + ": tensor " + name + " has a non-positive dimension " + dims[d]);
                }
                try {
                    elements = Math.multiplyExact(elements, dims[d]);
                } catch (ArithmeticException e) {
                    throw new IOException(file + ": the shape of tensor " + name + " overflows", e);
                }
            }
            int typeId = c.i32("type of " + name);
            GgmlType type = GgmlType.of(typeId);
            if (type == null) {
                throw new IOException(file + ": tensor " + name + " has unknown ggml type id " + typeId);
            }
            if (dims[0] % type.blockSize != 0) {
                throw new IOException(file + ": tensor " + name + " has rows of " + dims[0] + " elements, not a multiple"
                        + " of the " + type + " block size " + type.blockSize);
            }
            long size;
            try {
                size = Math.multiplyExact(elements / type.blockSize, (long) type.typeSize);
            } catch (ArithmeticException e) {
                throw new IOException(file + ": the size of tensor " + name + " overflows", e);
            }
            long offset = c.i64("offset of " + name);
            infos.add(new Info(name, dims, type, offset, size));
        }
        long dataStart = (c.pos + alignment - 1) / alignment * alignment;
        if (dataStart < c.pos || dataStart > map.byteSize()) {
            throw new IOException(file + ": the tensor data would start at " + dataStart + ", beyond the end of the file ("
                    + map.byteSize() + " bytes); the file is truncated");
        }
        long dataSize = map.byteSize() - dataStart;
        List<Tensor> mine = new ArrayList<>(infos.size());
        java.util.Set<String> names = new java.util.HashSet<>();
        for (Info t : infos) {
            if (t.offset < 0 || t.offset % alignment != 0) {
                throw new IOException(file + ": tensor " + t.name + " has offset " + t.offset + ", not a multiple of the alignment "
                        + alignment);
            }
            if (t.offset > dataSize || t.size > dataSize - t.offset) {
                throw new IOException(file + ": tensor " + t.name + " (" + t.size + " bytes at data offset " + t.offset
                        + ") extends beyond the end of the file (" + dataSize + " bytes of data); the file is truncated");
            }
            if (tensors.containsKey(t.name) || !names.add(t.name)) {
                throw new IOException(file + ": duplicate tensor " + t.name);
            }
            mine.add(new Tensor(t.name, t.dims, t.type, part, dataStart + t.offset, t.size));
        }
        List<Tensor> sorted = new ArrayList<>(mine);
        sorted.sort(Comparator.comparingLong(Tensor::offset));
        for (int i = 1; i < sorted.size(); i++) {
            Tensor a = sorted.get(i - 1);
            Tensor b = sorted.get(i);
            if (b.offset() < a.offset() + a.size()) {
                throw new IOException(file + ": the data of " + a.name() + " and " + b.name() + " overlap");
            }
        }
        for (Tensor t : mine) {
            tensors.put(t.name(), t);
        }
        return new Header(md, version, alignment);
    }

    private static Object readValue(Cursor c, int type, String key, int depth) throws IOException {
        return switch (type) {
            case T_U8 -> (long) c.u8(key);
            case T_I8 -> (long) c.i8(key);
            case T_U16 -> (long) c.u16(key);
            case T_I16 -> (long) c.i16(key);
            case T_U32 -> c.u32(key);
            case T_I32 -> (long) c.i32(key);
            case T_F32 -> (double) c.f32(key);
            case T_BOOL -> {
                int b = c.u8(key);
                if (b > 1) {
                    throw new IOException(c.file + ": metadata " + key + " is a bool with value " + b);
                }
                yield b == 1;
            }
            case T_STRING -> c.string("value of " + key);
            case T_U64, T_I64 -> c.i64(key);
            case T_F64 -> c.f64(key);
            case T_ARRAY -> readArray(c, key, depth);
            default -> throw new IOException(c.file + ": metadata " + key + " has unknown value type " + type);
        };
    }

    private static Object readArray(Cursor c, String key, int depth) throws IOException {
        if (depth > 4) {
            throw new IOException(c.file + ": metadata " + key + " nests arrays too deeply");
        }
        int et = c.i32("element type of " + key);
        long n = c.count("length of " + key);
        // the smallest element is one byte (a string takes 8 for its length, a nested array 12)
        long minBytes = switch (et) {
            case T_U8, T_I8, T_BOOL -> 1;
            case T_U16, T_I16 -> 2;
            case T_U32, T_I32, T_F32 -> 4;
            case T_U64, T_I64, T_F64, T_STRING -> 8;
            case T_ARRAY -> 12;
            default -> throw new IOException(c.file + ": metadata " + key + " is an array of unknown type " + et);
        };
        if (n > c.remaining() / minBytes) {
            throw new IOException(c.file + ": array " + key + " claims " + n + " elements but only " + c.remaining()
                    + " bytes remain");
        }
        int len = (int) n;
        switch (et) {
            case T_STRING: {
                String[] a = new String[len];
                for (int i = 0; i < len; i++) {
                    a[i] = c.string("element of " + key);
                }
                return a;
            }
            case T_F32, T_F64: {
                double[] a = new double[len];
                for (int i = 0; i < len; i++) {
                    a[i] = et == T_F32 ? c.f32(key) : c.f64(key);
                }
                return a;
            }
            case T_BOOL: {
                boolean[] a = new boolean[len];
                for (int i = 0; i < len; i++) {
                    a[i] = c.u8(key) != 0;
                }
                return a;
            }
            case T_ARRAY: {
                Object[] a = new Object[len];
                for (int i = 0; i < len; i++) {
                    a[i] = readValue(c, T_ARRAY, key, depth + 1);
                }
                return a;
            }
            default: {
                long[] a = new long[len];
                for (int i = 0; i < len; i++) {
                    a[i] = (Long) readValue(c, et, key, depth + 1);
                }
                return a;
            }
        }
    }

    private static long num(Object o, long def) {
        return o instanceof Number n ? n.longValue() : def;
    }

    // ------------------------------------------------------------------ access

    public int version() {
        return version;
    }

    /** {@code general.alignment} (32 when the file does not set it). */
    public long alignment() {
        return alignment;
    }

    public Map<String, Object> metadata() {
        return metadata;
    }

    public boolean has(String key) {
        return metadata.containsKey(key);
    }

    public Object get(String key) {
        return metadata.get(key);
    }

    public String str(String key, String def) {
        return metadata.get(key) instanceof String s ? s : def;
    }

    public long num(String key, long def) {
        return num(metadata.get(key), def);
    }

    public double dbl(String key, double def) {
        return metadata.get(key) instanceof Number n ? n.doubleValue() : def;
    }

    public boolean bool(String key, boolean def) {
        return metadata.get(key) instanceof Boolean b ? b : def;
    }

    /** The string array under {@code key}, or null. */
    public String[] strings(String key) {
        return metadata.get(key) instanceof String[] a ? a : null;
    }

    /** The integer array under {@code key}, or null. */
    public long[] longs(String key) {
        return metadata.get(key) instanceof long[] a ? a : null;
    }

    /** The float array under {@code key}, or null. */
    public double[] doubles(String key) {
        return metadata.get(key) instanceof double[] a ? a : null;
    }

    /** All tensors, in file order (part by part). */
    public Map<String, Tensor> tensors() {
        return tensors;
    }

    public boolean hasTensor(String name) {
        return tensors.containsKey(name);
    }

    public Tensor tensor(String name) {
        Tensor t = tensors.get(name);
        if (t == null) {
            throw new IllegalArgumentException("tensor not found in " + parts.getFirst().path().getFileName() + ": " + name);
        }
        return t;
    }

    /** The mapped bytes of {@code t}, a slice of the file mapping (no copy). */
    public MemorySegment data(Tensor t) {
        return parts.get(t.part()).map().asSlice(t.offset(), t.size());
    }

    /** The whole mapping of part {@code i}, for callers that bind page ranges of it. */
    public MemorySegment mapping(int part) {
        return parts.get(part).map();
    }

    /** Total size of all parts in bytes. */
    public long byteSize() {
        long n = 0;
        for (Part p : parts) {
            n += p.fileSize();
        }
        return n;
    }

    public List<Path> paths() {
        return parts.stream().map(Part::path).toList();
    }

    @Override
    public void close() {
        arena.close();
    }
}
