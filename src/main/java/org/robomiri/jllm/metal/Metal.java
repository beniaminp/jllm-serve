package org.robomiri.jllm.metal;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Panama FFM bindings for {@code libjllm_metal.dylib} (see {@code src/main/native/jllm_metal.m}).
 *
 * <p>Thin and generic: compile shader source, create/wrap buffers, record dispatches, commit and wait.
 */
public final class Metal implements AutoCloseable {

    /**
     * A Metal buffer handle plus a CPU view of its (shared) contents. {@code offset} is where the data of interest
     * starts inside the buffer: 0, except for a weight tensor that is a slice of a larger memory-mapped file
     * ({@link #wrapRange}); {@link MetalKernels} adds it to every binding offset.
     */
    public record Buffer(MemorySegment handle, MemorySegment contents, long size, long offset) {
        public Buffer(MemorySegment handle, MemorySegment contents, long size) {
            this(handle, contents, size, 0);
        }
    }

    /** Metal wraps whole pages; Apple Silicon's are 16 KiB. */
    private static final long PAGE = 16384;

    private static final int ERR_LEN = 4096;
    private static final Linker LINKER = Linker.nativeLinker();

    private final MethodHandle create;
    private final MethodHandle waitPending; // null with a library built before jm_wait_pending existed
    private final MethodHandle destroy;
    private final MethodHandle deviceName;
    private final MethodHandle workingSet;
    private final MethodHandle maxTgMem;
    private final MethodHandle pipeline;
    private final MethodHandle pipelineMaxThreads;
    private final MethodHandle bufferNew;
    private final MethodHandle bufferWrap;
    private final MethodHandle bufferContents;
    private final MethodHandle bufferRelease;
    private final MethodHandle begin;
    private final MethodHandle dispatch;
    private final MethodHandle abort;
    private final MethodHandle commit;
    private final MethodHandle commitWait;

    private final Arena arena = Arena.ofShared();
    private final MemorySegment err = arena.allocate(ERR_LEN);
    private final MemorySegment ctx;
    private final List<Buffer> buffers = new ArrayList<>();
    private final Map<String, Integer> pipelines = new HashMap<>();

    /** Identifies the compiled kernel source (MetalTuner caches its decisions per device and source). */
    final String sourceHash;

    private Metal(Path lib, String shaderSource) {
        this.sourceHash = Integer.toHexString(shaderSource.hashCode()) + (EMULATOR ? "-emu" : "");
        SymbolLookup l = SymbolLookup.libraryLookup(lib, arena);
        create = handle(l, "jm_create", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_INT));
        destroy = handle(l, "jm_destroy", FunctionDescriptor.ofVoid(ADDRESS));
        deviceName = handle(l, "jm_device_name", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT));
        workingSet = handle(l, "jm_recommended_working_set", FunctionDescriptor.of(JAVA_LONG, ADDRESS));
        maxTgMem = handle(l, "jm_device_max_tg_mem", FunctionDescriptor.of(JAVA_LONG, ADDRESS));
        pipeline =handle(l, "jm_pipeline", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT));
        pipelineMaxThreads = handle(l, "jm_pipeline_max_threads", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
        bufferNew = handle(l, "jm_buffer_new", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT));
        bufferWrap = handle(l, "jm_buffer_wrap", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_LONG));
        bufferContents = handle(l, "jm_buffer_contents", FunctionDescriptor.of(ADDRESS, ADDRESS));
        bufferRelease = handle(l, "jm_buffer_release", FunctionDescriptor.ofVoid(ADDRESS));
        begin = handle(l, "jm_begin", FunctionDescriptor.ofVoid(ADDRESS));
        dispatch = handle(l, "jm_dispatch", FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, ADDRESS, ADDRESS, JAVA_INT,
                ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
        abort = handle(l, "jm_abort", FunctionDescriptor.ofVoid(ADDRESS));
        commit = handle(l, "jm_commit", FunctionDescriptor.ofVoid(ADDRESS));
        commitWait =handle(l, "jm_commit_wait", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT));
        waitPending = l.find("jm_wait_pending").map(a -> LINKER.downcallHandle(a,
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT))).orElse(null);
        try {
            ctx = (MemorySegment) create.invokeExact(arena.allocateFrom(shaderSource), err, ERR_LEN);
        } catch (Throwable t) {
            throw rethrow(t);
        }
        if (ctx.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("Metal initialisation failed: " + error());
        }
    }

    private static MethodHandle handle(SymbolLookup l, String name, FunctionDescriptor fd) {
        return LINKER.downcallHandle(l.findOrThrow(name), fd);
    }

    /** Loads the native library and compiles the bundled kernels. */
    public static Metal create() {
        return create(null);
    }

    /** @param explicitLib path of libjllm_metal.dylib, or null to search the usual locations */
    public static Metal create(String explicitLib) {
        return create(explicitLib, "");
    }

    /**
     * @param defines preprocessor lines (e.g. {@code "#define X 1\n"}) prepended to the kernel source; they select
     *                optional kernel instantiations. Compiler line numbers still refer to kernels.metal.
     */
    public static Metal create(String explicitLib, String defines) {
        Path lib = findLibrary(explicitLib);
        String src;
        try (InputStream in = Metal.class.getResourceAsStream("kernels.metal")) {
            if (in == null) {
                throw new IllegalStateException("kernels.metal resource missing");
            }
            src = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        return new Metal(lib, defines.isEmpty() ? src : defines + "#line 1\n" + src);
    }

    /** True if this looks like a machine where the Metal backend can work. */
    public static boolean available() {
        return available(null);
    }

    public static boolean available(String explicitLib) {
        return (System.getProperty("os.name", "").toLowerCase().contains("mac") || EMULATOR)
                && locateLibrary(explicitLib) != null;
    }

    /**
     * {@code -Djllm.metal.emulator=true}: the library is the CPU emulator ({@code scripts/build-metal-emu.sh}), which
     * runs the same kernels on any OS, very slowly; used to test the Metal backend without an Apple GPU.
     */
    static final boolean EMULATOR = Boolean.getBoolean("jllm.metal.emulator");

    private static Path findLibrary(String explicitLib) {
        Path p = locateLibrary(explicitLib);
        if (p == null) {
            throw new IllegalStateException("libjllm_metal.dylib not found. Build it with ./scripts/build-native.sh "
                    + "or point -Djllm.metal.lib=/path/to/libjllm_metal.dylib at it");
        }
        return p;
    }

    private static Path locateLibrary(String explicitLib) {
        List<Path> candidates = new ArrayList<>();
        if (explicitLib != null && !explicitLib.isBlank()) {
            candidates.add(Path.of(explicitLib));
        }
        String prop = System.getProperty("jllm.metal.lib");
        if (prop != null) {
            candidates.add(Path.of(prop));
        }
        String env = System.getenv("JLLM_METAL_LIB");
        if (env != null) {
            candidates.add(Path.of(env));
        }
        candidates.add(Path.of("target", "native", "libjllm_metal.dylib"));
        candidates.add(Path.of("libjllm_metal.dylib"));
        try { // next to the jar / classes directory
            Path codeSource = Path.of(Metal.class.getProtectionDomain().getCodeSource().getLocation().toURI());
            Path base = Files.isDirectory(codeSource) ? codeSource.getParent() : codeSource.getParent();
            candidates.add(base.resolve("native").resolve("libjllm_metal.dylib"));
        } catch (Exception ignored) {
            // not resolvable
        }
        for (Path c : candidates) {
            if (Files.isRegularFile(c)) {
                return c.toAbsolutePath();
            }
        }
        return null;
    }

    // ------------------------------------------------------------------ API

    public String deviceName() {
        try {
            int ignored = (int) deviceName.invokeExact(ctx, err, ERR_LEN);
            return err.getString(0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public long recommendedWorkingSet() {
        try {
            return (long) workingSet.invokeExact(ctx);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Threadgroup memory limit in bytes (static arrays and the dynamic [[threadgroup(0)]] argument combined). */
    public long maxThreadgroupMemory() {
        try {
            return (long) maxTgMem.invokeExact(ctx);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The pipeline index of kernel {@code name}, created (compiled for the GPU) on first use and memoized, so
     * callers look kernels up by name at dispatch time and only the kernels a model actually runs get built.
     */
    public int pipeline(String name) {
        Integer cached = pipelines.get(name);
        if (cached != null) {
            return cached;
        }
        try {
            int idx = (int) pipeline.invokeExact(ctx, arena.allocateFrom(name), err, ERR_LEN);
            if (idx < 0) {
                throw new IllegalStateException("pipeline " + name + ": " + error());
            }
            pipelines.put(name, idx);
            return idx;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public int maxThreads(int pipelineIndex) {
        try {
            return (int) pipelineMaxThreads.invokeExact(ctx, pipelineIndex);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** New zero-filled shared buffer. */
    public Buffer newBuffer(long bytes) {
        return newBuffer(bytes, true);
    }

    /**
     * New shared buffer. With {@code zero == false} the contents are left uninitialised, which skips a memset
     * (and the page commit it implies) for scratch buffers that are always written before they are read.
     */
    public Buffer newBuffer(long bytes, boolean zero) {
        try {
            MemorySegment h = (MemorySegment) bufferNew.invokeExact(ctx, Math.max(16L, bytes), zero ? 1 : 0);
            if (h.equals(MemorySegment.NULL)) {
                throw new OutOfMemoryError("Metal buffer of " + bytes + " bytes");
            }
            return track(h, bytes);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Wraps Java-owned native memory zero-copy. The segment must be page aligned (address and size) and must
     * outlive this object; the shim prints the reason to stderr and this throws if Metal refuses.
     */
    public Buffer wrap(MemorySegment seg) {
        try {
            MemorySegment h = (MemorySegment) bufferWrap.invokeExact(ctx, seg, seg.byteSize());
            if (h.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("cannot wrap " + seg.byteSize() + " bytes at 0x"
                        + Long.toHexString(seg.address()) + " as a zero-copy Metal buffer (see stderr)");
            }
            return track(h, seg.byteSize());
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Zero-copy view of a slice of larger native memory (a tensor inside a memory-mapped GGUF file) that is not page
     * aligned: wraps the pages the slice touches and returns a buffer whose {@link Buffer#offset} is the slice's
     * position in the first of them. The extra bytes before and after the slice belong to the same mapping, so
     * they are valid memory; the buffer is never written. Page-aligned segments are wrapped as they are.
     */
    public Buffer wrapRange(MemorySegment seg) {
        long addr = seg.address();
        long base = addr & -PAGE;
        long end = (addr + seg.byteSize() + PAGE - 1) & -PAGE;
        if (base == addr && end == addr + seg.byteSize()) {
            return wrap(seg);
        }
        Buffer b = wrap(MemorySegment.ofAddress(base).reinterpret(end - base));
        return new Buffer(b.handle(), b.contents(), b.size(), addr - base);
    }

    private Buffer track(MemorySegment h, long size) throws Throwable {
        MemorySegment contents = ((MemorySegment) bufferContents.invokeExact(h)).reinterpret(size);
        Buffer b = new Buffer(h, contents, size);
        buffers.add(b);
        return b;
    }

    public void begin() {
        try {
            begin.invokeExact(ctx);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Records a dispatch. {@code bufs}/{@code offsets} are native arrays of {@code nbuf} pointers/u64 offsets;
     * {@code params} holds {@code plen} bytes bound after the buffers.
     */
    public void dispatch(int pipe, MemorySegment bufs, MemorySegment offsets, int nbuf, MemorySegment params, int plen,
                         int gx, int gy, int gz, int tx, int ty, int tz) {
        dispatch(pipe, bufs, offsets, nbuf, params, plen, 0, gx, gy, gz, tx, ty, tz);
    }

    /** As above, with {@code tgmem} bytes of dynamic threadgroup memory for the kernel's [[threadgroup(0)]]. */
    public void dispatch(int pipe, MemorySegment bufs, MemorySegment offsets, int nbuf, MemorySegment params, int plen,
                         int tgmem, int gx, int gy, int gz, int tx, int ty, int tz) {
        try {
            dispatch.invokeExact(ctx, pipe, bufs, offsets, nbuf, params, plen, tgmem, gx, gy, gz, tx, ty, tz);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Drops the command buffer being recorded without executing it (ends the encoder first, which Metal
     * insists on). Safe to call when nothing is being recorded.
     */
    public void abort() {
        try {
            abort.invokeExact(ctx);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Submits the dispatches recorded so far without waiting and keeps recording into a fresh command buffer, so
     * the GPU starts on them while the rest of the pass is being encoded. Execution order is preserved (one
     * in-order queue); {@link #commitAndWait()} waits for everything and reports a failure of any part.
     */
    public void commit() {
        try {
            commit.invokeExact(ctx);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Whether {@link #waitPending} is available (the native library is recent enough). */
    public boolean canWaitPending() {
        return waitPending != null;
    }

    /** Waits until at most {@code keep} buffers submitted with {@link #commit} are unfinished (oldest first). */
    public void waitPending(int keep) {
        try {
            int rc = (int) waitPending.invokeExact(ctx, keep, err, ERR_LEN);
            if (rc != 0) {
                throw new IllegalStateException("Metal command buffer failed: " + error());
            }
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public void commitAndWait() {
        try {
            int rc = (int) commitWait.invokeExact(ctx, err, ERR_LEN);
            if (rc != 0) {
                throw new IllegalStateException("Metal command buffer failed: " + error());
            }
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private String error() {
        return err.getString(0);
    }

    private boolean closed;

    /** Releases every buffer and the context; a second call is a no-op. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            for (Buffer b : buffers) {
                bufferRelease.invokeExact(b.handle());
            }
            buffers.clear();
            destroy.invokeExact(ctx);
        } catch (Throwable t) {
            throw rethrow(t);
        } finally {
            arena.close();
        }
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException re) {
            return re;
        }
        if (t instanceof Error e) {
            throw e;
        }
        return new RuntimeException(t);
    }
}
