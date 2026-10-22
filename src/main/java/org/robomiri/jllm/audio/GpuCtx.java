package org.robomiri.jllm.audio;

import org.robomiri.jllm.metal.Metal;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * The audio models' Metal context: the {@code audio.metal} library on its own {@link Metal} instance, and a dispatch
 * helper that binds buffers and writes a kernel's parameter struct (32-bit ints and floats, in declaration order).
 */
final class GpuCtx implements AutoCloseable {

    final Metal metal;
    private final Arena arena = Arena.ofShared();
    private final MemorySegment bufArgs = arena.allocate(ADDRESS, 16);
    private final MemorySegment offArgs = arena.allocate(JAVA_LONG, 16);
    private final MemorySegment params = arena.allocate(128);

    private GpuCtx(Metal metal) {
        this.metal = metal;
    }

    /** @return the context, or null when there is no Metal library / GPU on this machine */
    static GpuCtx create(String lib) throws IOException {
        if (!Metal.available(lib)) {
            return null;
        }
        String src;
        try (InputStream in = GpuCtx.class.getResourceAsStream("audio.metal")) {
            if (in == null) {
                throw new IOException("audio.metal is missing from the jar");
            }
            src = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
        return new GpuCtx(Metal.createFromSource(lib, src));
    }

    Metal.Buffer buffer(long bytes) {
        return metal.newBuffer(Math.max(16, bytes), false);
    }

    /** A buffer holding a copy of {@code data}. */
    Metal.Buffer floats(float[] data) {
        Metal.Buffer b = buffer(4L * data.length);
        MemorySegment.copy(data, 0, b.contents(), JAVA_FLOAT, b.offset(), data.length);
        return b;
    }

    /** Copies floats into a buffer at element {@code off}. */
    static void put(Metal.Buffer b, long off, float[] data, int n) {
        MemorySegment.copy(data, 0, b.contents(), JAVA_FLOAT, b.offset() + 4 * off, n);
    }

    /** Copies {@code n} floats starting at element {@code off} out of a buffer. */
    static void get(Metal.Buffer b, long off, float[] dst, int n) {
        MemorySegment.copy(b.contents(), JAVA_FLOAT, b.offset() + 4 * off, dst, 0, n);
    }

    /**
     * Records one dispatch of {@code kernel}: {@code bufs} bound at 0.., then the parameter struct built from
     * {@code ps} (Integer or Float values) at index {@code bufs.length}. {@code g*} count threadgroups, {@code t*} are
     * the threads of one.
     */
    void run(String kernel, int gx, int gy, int gz, int tx, int ty, int tz, Object[] ps, Metal.Buffer... bufs) {
        for (int i = 0; i < bufs.length; i++) {
            bufArgs.setAtIndex(ADDRESS, i, bufs[i].handle());
            offArgs.setAtIndex(JAVA_LONG, i, bufs[i].offset());
        }
        int n = 0;
        for (Object o : ps) {
            if (o instanceof Integer v) {
                params.setAtIndex(JAVA_INT, n++, v);
            } else if (o instanceof Float f) {
                params.setAtIndex(JAVA_FLOAT, n++, f);
            } else {
                throw new IllegalArgumentException("kernel parameters are ints and floats");
            }
        }
        metal.dispatch(metal.pipeline(kernel), bufArgs, offArgs, bufs.length, params, 4 * n, 0, gx, gy, gz, tx, ty, tz);
    }

    /** Same with a byte offset on the buffers. */
    void runAt(String kernel, int gx, int gy, int gz, int tx, int ty, int tz, Object[] ps, Metal.Buffer[] bufs, long[] byteOffsets) {
        for (int i = 0; i < bufs.length; i++) {
            bufArgs.setAtIndex(ADDRESS, i, bufs[i].handle());
            offArgs.setAtIndex(JAVA_LONG, i, bufs[i].offset() + byteOffsets[i]);
        }
        int n = 0;
        for (Object o : ps) {
            if (o instanceof Integer v) {
                params.setAtIndex(JAVA_INT, n++, v);
            } else {
                params.setAtIndex(JAVA_FLOAT, n++, (Float) o);
            }
        }
        metal.dispatch(metal.pipeline(kernel), bufArgs, offArgs, bufs.length, params, 4 * n, 0, gx, gy, gz, tx, ty, tz);
    }

    void begin() {
        metal.begin();
    }

    void commitAndWait() {
        metal.commitAndWait();
    }

    @Override
    public void close() {
        try {
            metal.close();
        } finally {
            arena.close();
        }
    }

    static int ceilDiv(int a, int b) {
        return (a + b - 1) / b;
    }

    static final ValueLayout.OfFloat F32 = JAVA_FLOAT;
}
