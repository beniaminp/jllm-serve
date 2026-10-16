package org.robomiri.jllm.model;

import org.robomiri.jllm.json.Json;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * jllm's own on-disk format for quantized weights, so a model is quantized once and memory-mapped afterwards.
 *
 * <p>Two files: {@code jllm-<quant>.bin} holds page-aligned blobs in exactly the {@link QTensor} memory layout
 * (so a mapped slice can be used directly, even zero-copy by Metal), and {@code jllm-<quant>.json} describes
 * them: {@code {"version", "layout", "quant", "model", "tensors": {name: {type, rows, cols, offset, size}}}}.
 * The JSON is written last and acts as the "conversion complete" marker.
 *
 * <p>A cache that cannot be used — unparseable or malformed index, another {@link #VERSION} or
 * {@link #LAYOUT_VERSION}, a blob that does not fit the file — is reported as a {@link StaleCacheException},
 * which {@link ModelWeights#load} answers by deleting the files and converting again.
 */
public final class QuantFile implements AutoCloseable {

    /** Format of the two files themselves (blob layout, index schema). */
    public static final int VERSION = 1;

    /**
     * What the {@link ModelConverter} puts into a cache: tensor names, which tensors are fused, split or folded
     * and how. Bumped whenever that changes so caches written by an older converter are reconverted instead of
     * being loaded with wrong contents. 1 = caches written before the key existed (they carry no {@code layout}).
     */
    public static final int LAYOUT_VERSION = 2;

    /** The cache on disk cannot be used as it is; deleting and reconverting it is the remedy. */
    public static final class StaleCacheException extends IOException {
        public StaleCacheException(String message) {
            super(message);
        }

        public StaleCacheException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    public record Entry(String type, int rows, int cols, long offset, long size) {
    }

    /**
     * Where a cache came from: the repository and revision the model was requested as (recorded as given), and
     * the Hub commit that revision resolved to when it is known. Part of the cache identity: a cache converted
     * from another repo or revision is not reused.
     *
     * @param sha the Hub commit, null when unknown (local files)
     */
    public record Identity(String repo, String revision, String sha) {

        /** Parses {@code repo@revision} ({@code revision} is null without an {@code @}); null for null. */
        public static Identity parse(String name, String sha) {
            if (name == null) {
                return null;
            }
            int at = name.lastIndexOf('@');
            return at < 0 ? new Identity(name, null, sha) : new Identity(name.substring(0, at), name.substring(at + 1), sha);
        }

        /** {@code repo@revision}, the form {@link ModelWeights#load} compares. */
        public String name() {
            return revision == null ? repo : repo + "@" + revision;
        }

        /** True if {@code other} names the same repo and revision (the commit is informational). */
        public boolean sameAs(Identity other) {
            return other != null && name().equals(other.name());
        }

        static Identity read(Map<String, Object> header) {
            String repo = Json.str(header, "repo", null);
            return repo == null ? null : new Identity(repo, Json.str(header, "revision", null), Json.str(header, "sha", null));
        }

        /** Adds the non-null fields to a header map. */
        static void write(Map<String, Object> header, Identity id) {
            if (id == null) {
                return;
            }
            header.put("repo", id.repo());
            if (id.revision() != null) {
                header.put("revision", id.revision());
            }
            if (id.sha() != null) {
                header.put("sha", id.sha());
            }
        }
    }

    private final Arena arena;
    private final MemorySegment mapped;
    private final Map<String, Entry> entries;
    private final Identity identity;

    private QuantFile(Arena arena, MemorySegment mapped, Map<String, Entry> entries, Identity identity) {
        this.arena = arena;
        this.mapped = mapped;
        this.entries = entries;
        this.identity = identity;
    }

    /** The repo / revision / commit recorded at conversion, or null if the cache was converted from local files. */
    public Identity identity() {
        return identity;
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

    /** Removes the cache files of {@code quant} in {@code dir} (index, blobs, any resume sidecar and temp files). */
    public static void delete(Path dir, QType quant) throws IOException {
        for (Path p : new Path[]{jsonPath(dir, quant), binPath(dir, quant), partialPath(dir, quant)}) {
            Files.deleteIfExists(p);
            Files.deleteIfExists(tmpPath(p));
        }
    }

    private static Path tmpPath(Path target) {
        return target.resolveSibling(target.getFileName() + ".tmp");
    }

    /**
     * Writes {@code content} to {@code target} atomically: into a sibling {@code .tmp} file, forced to disk,
     * then renamed over the target, so a crash leaves either the old file or the complete new one.
     */
    static void writeAtomic(Path target, String content) throws IOException {
        Path tmp = tmpPath(target);
        try (FileChannel ch = FileChannel.open(tmp, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer bb = ByteBuffer.wrap(content.getBytes(StandardCharsets.UTF_8));
            while (bb.hasRemaining()) {
                ch.write(bb);
            }
            ch.force(true);
        }
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    /**
     * Maps the cache of {@code quant} in {@code dir}.
     *
     * @throws StaleCacheException if the index is unparseable or malformed, was written by another version or
     *                             converter layout, or describes blobs the {@code .bin} does not hold
     */
    public static QuantFile open(Path dir, QType quant) throws IOException {
        Path json = jsonPath(dir, quant);
        Map<String, Object> header = parseJson(json);
        Map<String, Entry> entries = readIndex(json, Json.map(header, "tensors"));
        Path bin = binPath(dir, quant);
        if (!Files.isRegularFile(bin)) {
            throw new StaleCacheException(bin + " is missing");
        }
        Arena arena = Arena.ofShared();
        try {
            MemorySegment mapped;
            try (FileChannel ch = FileChannel.open(bin, StandardOpenOption.READ)) {
                mapped = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
            }
            for (Map.Entry<String, Entry> e : entries.entrySet()) {
                checkEntry(e.getKey(), e.getValue(), mapped.byteSize(), bin);
            }
            return new QuantFile(arena, mapped, entries, Identity.read(header));
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
            throw new StaleCacheException(json + " is not a valid jllm cache index: " + e.getMessage(), e);
        }
        checkVersions(json, header);
        return header;
    }

    /** Rejects an index or sidecar written by another file version or converter layout. */
    static void checkVersions(Path file, Map<String, Object> header) throws StaleCacheException {
        long version = Json.num(header, "version", 0);
        if (version != VERSION) {
            throw new StaleCacheException(file + " has cache version " + version + ", this build reads " + VERSION);
        }
        long layout = Json.num(header, "layout", 1);
        if (layout != LAYOUT_VERSION) {
            throw new StaleCacheException(file + " was written by converter layout " + layout + ", this build needs "
                    + LAYOUT_VERSION);
        }
    }

    /** Parses a {@code {name: {type, rows, cols, offset, size}}} map, rejecting anything malformed. */
    static Map<String, Entry> readIndex(Path json, Map<String, Object> tensors) throws IOException {
        if (tensors == null) {
            throw new StaleCacheException(json + " has no \"tensors\" map");
        }
        Map<String, Entry> entries = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : tensors.entrySet()) {
            Object v = e.getValue();
            if (!(v instanceof Map)) {
                throw new StaleCacheException(json + ": malformed entry for " + e.getKey());
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
            throw new StaleCacheException("corrupt jllm cache entry " + name + ": " + problem + " (" + bin + ")");
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


    /** Sidecar that records the units already written while a conversion is in progress. */
    public static Path partialPath(Path dir, QType quant) {
        return dir.resolve("jllm-" + quant.name().toLowerCase() + ".partial.json");
    }

    /**
     * Streams tensors into a new cache; call {@link #finish()} to write the index.
     *
     * <p>A conversion is a sequence of units (one layer each). After every unit {@link #commitUnit} forces the
     * data to disk and rewrites the sidecar {@code jllm-<quant>.partial.json} ({@code version, layout, quant,
     * model, units, tensors}) atomically, so a crash loses at most the unit in progress. {@link #openOrResume}
     * picks a consistent sidecar up again: it must come from this file version and converter layout, the
     * recorded units must be a prefix of the planned ones, the recorded entries must form a contiguous,
     * well-formed prefix of the {@code .bin}, and the {@code .bin} must hold at least that many bytes (a longer
     * tail from an interrupted unit is truncated). On any doubt the conversion starts over.
     */
    public static final class Writer implements AutoCloseable {
        private final Path dir;
        private final QType quant;
        private final Path bin;
        private final Path partial;
        private final String model;
        private final Identity identity;
        private final FileChannel ch;
        private final Map<String, Object> index = new LinkedHashMap<>();
        private final List<String> units = new ArrayList<>();
        private long offset;
        /** Byte count covered by the last commit; everything beyond it is discarded on failure. */
        private long committed;

        /** Starts a fresh cache, discarding any previous partial conversion. */
        public Writer(Path dir, QType quant) throws IOException {
            this(dir, quant, "", null, false);
        }

        private Writer(Path dir, QType quant, String model, Identity identity, boolean resume) throws IOException {
            this.dir = dir;
            this.quant = quant;
            this.model = model;
            this.identity = identity;
            this.bin = binPath(dir, quant);
            this.partial = partialPath(dir, quant);
            Files.deleteIfExists(jsonPath(dir, quant));
            if (!resume) {
                Files.deleteIfExists(partial);
                this.ch = FileChannel.open(bin, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                        StandardOpenOption.TRUNCATE_EXISTING);
            } else {
                this.ch = FileChannel.open(bin, StandardOpenOption.WRITE);
            }
        }

        /**
         * Resumes the conversion recorded in the sidecar when it is consistent with {@code model},
         * {@code identity} and {@code plannedUnits}, otherwise starts fresh. {@link #completedUnits()} tells
         * which units to skip.
         *
         * @param identity where the checkpoint comes from, recorded in the cache; null for local files
         */
        public static Writer openOrResume(Path dir, QType quant, String model, Identity identity, List<String> plannedUnits)
                throws IOException {
            Resumable r = inspect(dir, quant, model, identity, plannedUnits);
            if (r == null) {
                return new Writer(dir, quant, model, identity, false);
            }
            Writer w = new Writer(dir, quant, model, identity, true);
            try {
                if (w.ch.size() != r.bytes()) {
                    w.ch.truncate(r.bytes());
                    w.ch.force(true);
                }
                w.ch.position(r.bytes());
            } catch (IOException e) {
                w.ch.close();
                throw e;
            }
            r.entries().forEach((name, e) -> w.index.put(name,
                    Json.obj("type", e.type(), "rows", e.rows(), "cols", e.cols(), "offset", e.offset(), "size", e.size())));
            w.units.addAll(r.units());
            w.offset = r.bytes();
            w.committed = r.bytes();
            return w;
        }

        private record Resumable(List<String> units, Map<String, Entry> entries, long bytes) {
        }

        /** Returns what a sidecar lets us reuse, or null (printing why) when the conversion must start over. */
        private static Resumable inspect(Path dir, QType quant, String model, Identity identity, List<String> plannedUnits) {
            Path partial = partialPath(dir, quant);
            Path bin = binPath(dir, quant);
            if (!Files.isRegularFile(partial)) {
                return null;
            }
            String reason;
            try {
                Map<String, Object> side = Json.parseObject(Files.readString(partial));
                checkVersions(partial, side);
                List<Object> unitObjs = Json.list(side, "units");
                Identity recorded = Identity.read(side);
                if (!quant.name().equals(Json.str(side, "quant", null))
                        || !model.equals(Json.str(side, "model", null)) || unitObjs == null) {
                    reason = "it was written for a different model or format";
                } else if (!(identity == null ? recorded == null : identity.sameAs(recorded))
                        || (identity != null && identity.sha() != null && recorded.sha() != null
                            && !identity.sha().equals(recorded.sha()))) {
                    reason = "it was written for " + (recorded == null ? "local files" : recorded.name()
                            + (recorded.sha() != null ? " (" + recorded.sha() + ")" : ""));
                } else if (!Files.isRegularFile(bin)) {
                    reason = bin.getFileName() + " is missing";
                } else {
                    List<String> units = new ArrayList<>();
                    for (Object o : unitObjs) {
                        units.add(String.valueOf(o));
                    }
                    Map<String, Entry> entries = readIndex(partial, Json.map(side, "tensors"));
                    long bytes = 0;
                    for (Map.Entry<String, Entry> e : entries.entrySet()) {
                        if (e.getValue().offset() != bytes) {
                            throw new IOException("entries are not contiguous at " + e.getKey());
                        }
                        checkEntry(e.getKey(), e.getValue(), Long.MAX_VALUE, bin);
                        bytes += e.getValue().size();
                    }
                    if (units.isEmpty() || units.size() > plannedUnits.size()
                            || !units.equals(plannedUnits.subList(0, units.size()))) {
                        reason = "its units do not match the conversion plan";
                    } else if (Files.size(bin) < bytes) {
                        reason = bin.getFileName() + " is shorter than the " + bytes + " bytes recorded";
                    } else {
                        return new Resumable(units, entries, bytes);
                    }
                }
            } catch (IOException | RuntimeException e) {
                reason = e.getMessage();
            }
            System.err.println("Ignoring partial conversion " + partial.getFileName() + ": " + reason);
            return null;
        }

        /** Units already written by a previous, interrupted conversion (a prefix of the plan). */
        public List<String> completedUnits() {
            return Collections.unmodifiableList(units);
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

        /** Marks everything written so far as unit {@code unitId}: data is forced to disk and the sidecar updated. */
        public void commitUnit(String unitId) throws IOException {
            ch.force(false);
            units.add(unitId);
            committed = offset;
            Map<String, Object> side = header();
            side.put("units", units);
            side.put("tensors", index);
            writeAtomic(partial, Json.write(side));
        }

        /** The common head of the index and the sidecar: versions, quant, model description and identity. */
        private Map<String, Object> header() {
            Map<String, Object> h = Json.obj("version", VERSION, "layout", LAYOUT_VERSION, "quant", quant.name(),
                    "model", model);
            Identity.write(h, identity);
            return h;
        }

        /** Forces the blobs to disk and publishes the index (atomically: it is the "conversion complete" marker). */
        public void finish() throws IOException {
            ch.force(true);
            ch.close();
            Map<String, Object> h = header();
            h.put("tensors", index);
            writeAtomic(jsonPath(dir, quant), Json.write(h));
            Files.deleteIfExists(partial);
        }

        /** Without {@link #finish()}: keeps the committed units for a later resume, or removes an empty cache. */
        @Override
        public void close() throws IOException {
            if (!ch.isOpen()) {
                return;
            }
            try {
                if (committed > 0 && Files.isRegularFile(partial)) {
                    ch.truncate(committed);
                    ch.force(true);
                }
            } finally {
                ch.close();
                if (committed == 0 || !Files.isRegularFile(partial)) {
                    Files.deleteIfExists(bin);
                    Files.deleteIfExists(partial);
                }
            }
        }
    }
}
