package org.robomiri.jllm.metal;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.stream.IntStream;

/**
 * The KV cache of a slot whose context does not fit the GPU's share of memory (docs/26 §26.2.2): the rows of every layer
 * live in host memory (memory of the process, or memory-mapped files for the layers past a host budget, which the
 * operating system pages between RAM and the SSD), and the GPU sees one layer at a time in two small <b>windows</b>
 * (a K and a V buffer of {@code ctx} rows each, used by even and odd layers alternately).
 *
 * <p>A forward pass over positions {@code [pos, pos + n)} does, for layer l: {@link #stage} copies rows {@code [0, pos)}
 * of layer l from the host into window {@code l & 1} (while the GPU is busy with layer l - 1, the other window); the GPU
 * stores the new rows of layer l into the window at {@code pos..} and attends over the window with the ordinary flat
 * kernels; {@link #writeBack} copies the new rows from the window to the host. The GPU therefore runs the very kernels
 * of the flat cache on the very same bytes, so the results are bit-identical to it; the cost is the copy of the whole
 * cache through the CPU for every pass (the price of a context that does not fit).
 */
final class StreamedKv implements AutoCloseable {

    private final int layers;
    private final int ctx;
    private final long rowBytes;
    private final Metal.Buffer[] winK;
    private final Metal.Buffer[] winV;
    private final MemorySegment[] hostK;
    private final MemorySegment[] hostV;
    private final boolean[] mapped;
    private final Arena arena = Arena.ofShared();
    private long stagedBytes;
    private long stagedNanos;
    private long writtenBytes;
    private long passes;
    private final int ramLayers;

    /**
     * @param winK      the two K windows (each {@code ctx * rowBytes})
     * @param winV      the two V windows
     * @param hostBytes host memory for the layers' rows; layers beyond it are memory-mapped files in {@code dir}
     *                  (0 = all in memory)
     */
    StreamedKv(int layers, int ctx, long rowBytes, Metal.Buffer[] winK, Metal.Buffer[] winV, long hostBytes, Path dir) {
        this.layers = layers;
        this.ctx = ctx;
        this.rowBytes = rowBytes;
        this.winK = winK;
        this.winV = winV;
        this.hostK = new MemorySegment[layers];
        this.hostV = new MemorySegment[layers];
        this.mapped = new boolean[layers];
        long perLayer = 2L * ctx * rowBytes;
        this.ramLayers = hostBytes <= 0 || dir == null ? layers : (int) Math.min(layers, Math.max(0, hostBytes / perLayer));
        try {
            for (int l = 0; l < layers; l++) {
                if (l < ramLayers) {
                    hostK[l] = arena.allocate((long) ctx * rowBytes, 64);
                    hostV[l] = arena.allocate((long) ctx * rowBytes, 64);
                } else {
                    mapped[l] = true;
                    hostK[l] = mapFile(dir);
                    hostV[l] = mapFile(dir);
                }
            }
        } catch (IOException e) {
            arena.close();
            throw new java.io.UncheckedIOException(e);
        } catch (RuntimeException | Error e) {
            arena.close();
            throw e;
        }
    }

    private MemorySegment mapFile(Path dir) throws IOException {
        Files.createDirectories(dir);
        Path f = Files.createTempFile(dir, "kvstream", ".bin");
        try (FileChannel ch = FileChannel.open(f, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            MemorySegment m = ch.map(FileChannel.MapMode.READ_WRITE, 0, (long) ctx * rowBytes, arena);
            Files.deleteIfExists(f); // unlinked while mapped: the space goes with the mapping
            return m;
        }
    }

    /** The host rows of layer {@code l}'s keys (row p at {@code p * rowBytes}). */
    MemorySegment keys(int l) {
        return hostK[l];
    }

    MemorySegment values(int l) {
        return hostV[l];
    }

    /** Layers whose rows are in memory-mapped files. */
    int mappedLayers() {
        return layers - ramLayers;
    }

    /** Copies rows {@code [0, pos)} of layer {@code l} into window {@code l & 1}. */
    void stage(int l, int pos) {
        if (pos <= 0) {
            return;
        }
        long t0 = System.nanoTime();
        long n = (long) pos * rowBytes;
        copy(hostK[l], winK[l & 1].contents(), 0, n);
        copy(hostV[l], winV[l & 1].contents(), 0, n);
        stagedBytes += 2 * n;
        stagedNanos += System.nanoTime() - t0;
    }

    /** Copies rows {@code [pos, pos + n)} of window {@code l & 1} (just computed by the GPU) back to layer {@code l}. */
    void writeBack(int l, int pos, int n) {
        long off = (long) pos * rowBytes;
        long len = (long) n * rowBytes;
        MemorySegment.copy(winK[l & 1].contents(), off, hostK[l], off, len);
        MemorySegment.copy(winV[l & 1].contents(), off, hostV[l], off, len);
        writtenBytes += 2 * len;
    }

    /** Asks the operating system to read the first {@code pos} rows of a mapped layer ahead of use. */
    void prefetch(int l, int pos) {
        if (l >= layers || !mapped[l] || pos <= 0) {
            return;
        }
        long n = (long) pos * rowBytes;
        Os.willNeed(hostK[l], n);
        Os.willNeed(hostV[l], n);
    }

    void endPass() {
        passes++;
    }

    long stagedBytes() {
        return stagedBytes;
    }

    long stagedNanos() {
        return stagedNanos;
    }

    long writtenBytes() {
        return writtenBytes;
    }

    long passes() {
        return passes;
    }

    /** Bytes of host memory the rows occupy (memory of the process and mapped files). */
    long hostBytes() {
        return 2L * layers * ctx * rowBytes;
    }

    private static final long CHUNK = 8L << 20;

    private static void copy(MemorySegment from, MemorySegment to, long off, long n) {
        if (n <= 2 * CHUNK) {
            MemorySegment.copy(from, off, to, off, n);
            return;
        }
        int parts = (int) ((n + CHUNK - 1) / CHUNK);
        IntStream.range(0, parts).parallel().forEach(i -> {
            long o = off + i * CHUNK;
            MemorySegment.copy(from, o, to, o, Math.min(CHUNK, off + n - o));
        });
    }

    @Override
    public void close() {
        arena.close();
    }

    /** madvise(WILLNEED), when the C library has it. */
    private static final class Os {
        private static final MethodHandle MADVISE;

        static {
            MethodHandle h = null;
            try {
                Linker l = Linker.nativeLinker();
                h = l.defaultLookup().find("madvise").map(s -> l.downcallHandle(s,
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT))).orElse(null);
            } catch (Throwable ignored) {
                // no madvise: the pages are read when the copy touches them
            }
            MADVISE = h;
        }

        private static final int MADV_WILLNEED = 3;

        static void willNeed(MemorySegment seg, long bytes) {
            if (MADVISE == null) {
                return;
            }
            try {
                long page = 16384;
                long a = seg.address() & -page;
                long len = ((seg.address() + bytes + page - 1) & -page) - a;
                int rc = (int) MADVISE.invokeExact(MemorySegment.ofAddress(a), len, MADV_WILLNEED);
                assert rc == 0 || rc == -1;
            } catch (Throwable ignored) {
                // advice only
            }
        }
    }
}
