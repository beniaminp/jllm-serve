package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Ggml;
import org.robomiri.jllm.model.Q6K;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * The int8 decode dot products in C ({@code libjllm_cpu}, {@code src/main/native/jllm_cpu.c}), called through the
 * Foreign Function &amp; Memory API: the same arithmetic as {@link Int8Kernels} and {@link KQuantKernels} (int8
 * activations per 32-block, exact integer sums per sub-block, float scales), but on the CPU's byte dot-product
 * instructions, which the Vector API cannot reach: AVX512-VNNI / AVX-VNNI {@code vpdpbusd}, AVX2
 * {@code vpmaddubsw}, ARM {@code sdot}. See docs/07 §7.16.
 *
 * <p>One downcall covers a range of rows (a thread's chunk of a matmul), so its cost (tens of nanoseconds) is spread
 * over thousands of blocks. The weights are passed as the tensors' own memory segments (memory-mapped files, no copy);
 * the quantized activation lives in an {@link Activation} buffer, and the output is the caller's {@code float[]},
 * reached directly through {@link Linker.Option#critical critical} downcalls with heap access.
 *
 * <p>Configuration ({@link #configure}): {@code backend.cpu.native=auto|true|false} and the library path
 * {@code backend.cpu.native-lib} / {@code -Djllm.cpu.nativeLib}. A library is used only after a self-check against the
 * Java kernels ({@link #selfCheck}); with {@code auto}, a missing or failing library is logged once and the Java
 * kernels are used.
 */
public final class NativeCpu {

    /** Type ids of the C interface. */
    static final int T_Q4_0 = 0, T_Q8_0 = 1, T_Q4_0G = 2, T_Q8_0G = 3, T_Q4_K = 4, T_Q5_K = 5, T_Q6_K = 6;

    private static final int ABI = 1;
    private static final Linker LINKER = Linker.nativeLinker();

    private final Path path;
    private final MethodHandle dotRows;
    private final MethodHandle quantize;
    private final MethodHandle setLevel;
    private final MethodHandle levelName;
    private final int detected;

    private NativeCpu(Path path) {
        this.path = path;
        SymbolLookup l = SymbolLookup.libraryLookup(path, Arena.global());
        MethodHandle abi = LINKER.downcallHandle(l.findOrThrow("jllm_cpu_abi"), FunctionDescriptor.of(JAVA_INT));
        int v;
        try {
            v = (int) abi.invokeExact();
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
        if (v != ABI) {
            throw new IllegalStateException("interface version " + v + ", expected " + ABI + " (rebuild it with"
                    + " scripts/build-native-cpu.sh)");
        }
        Linker.Option critical = Linker.Option.critical(true);
        dotRows = LINKER.downcallHandle(l.findOrThrow("jllm_cpu_dot_rows"), FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS,
                JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT), critical);
        quantize = LINKER.downcallHandle(l.findOrThrow("jllm_cpu_quantize"),
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS), critical);
        setLevel = LINKER.downcallHandle(l.findOrThrow("jllm_cpu_set_level"), FunctionDescriptor.of(JAVA_INT, JAVA_INT));
        levelName = LINKER.downcallHandle(l.findOrThrow("jllm_cpu_level_name"), FunctionDescriptor.of(ADDRESS, JAVA_INT));
        try {
            detected = (int) LINKER.downcallHandle(l.findOrThrow("jllm_cpu_detected_level"),
                    FunctionDescriptor.of(JAVA_INT)).invokeExact();
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    // ------------------------------------------------------------------ configuration (process wide)

    private static boolean configured;
    private static String configuredKey;
    private static NativeCpu active;
    /** Loaded and checked libraries by path: a library is loaded once per process. */
    private static final Map<Path, Object> LOADED = new HashMap<>();

    /**
     * Applies {@code backend.cpu.native} ({@code auto}, {@code true}, {@code false}) and {@code backend.cpu.native-lib}
     * (null or blank: search) for the CPU matmuls created from now on, and returns the library in use (null: the Java
     * kernels). {@code auto} logs why a library is not used; {@code true} throws instead.
     */
    public static synchronized NativeCpu configure(String mode, String explicitLib) {
        String m = mode == null ? "auto" : mode.trim().toLowerCase(Locale.ROOT);
        String key = m + "|" + (explicitLib == null ? "" : explicitLib.trim());
        if (configured && key.equals(configuredKey)) {
            return active;
        }
        NativeCpu chosen = switch (m) {
            case "false" -> null;
            case "auto", "true" -> {
                Path p = locate(explicitLib);
                if (p == null) {
                    String msg = "libjllm_cpu not found (build it with scripts/build-native-cpu.sh, or set"
                            + " backend.cpu.native-lib / -Djllm.cpu.nativeLib)";
                    if (m.equals("true")) {
                        throw new IllegalStateException("backend.cpu.native=true: " + msg);
                    }
                    System.err.println("native CPU kernels: " + msg + "; using the Java kernels");
                    yield null;
                }
                Object r = LOADED.computeIfAbsent(p, NativeCpu::loadChecked);
                if (r instanceof NativeCpu n) {
                    yield n;
                }
                if (m.equals("true")) {
                    throw new IllegalStateException("backend.cpu.native=true: " + r);
                }
                System.err.println("native CPU kernels: " + r + "; using the Java kernels");
                yield null;
            }
            default -> throw new IllegalArgumentException("backend.cpu.native must be auto, true or false, got '"
                    + mode + "'");
        };
        if (chosen != null && (active != chosen || !configured)) {
            System.err.println("native CPU kernels: " + chosen.level() + " (" + chosen.path + ")");
        }
        active = chosen;
        configured = true;
        configuredKey = key;
        return active;
    }

    /**
     * The library the CPU matmuls use (null: the Java kernels). Without an earlier {@link #configure} (tests, embedding
     * code), {@code -Djllm.cpu.native} (default {@code auto}) and the default search apply.
     */
    public static synchronized NativeCpu get() {
        if (!configured) {
            configure(System.getProperty("jllm.cpu.native", "auto"), null);
        }
        return active;
    }

    /** Loads and self-checks {@code lib}: the library, or a String saying why it cannot be used. */
    private static Object loadChecked(Path lib) {
        try {
            NativeCpu n = new NativeCpu(lib);
            String fail = n.selfCheck();
            return fail == null ? n : lib + " failed its self-check (" + fail + ")";
        } catch (Throwable t) {
            return lib + " cannot be loaded (" + t.getMessage() + ")";
        }
    }

    /** Loads {@code lib} without the self-check and without making it the active library (tests). */
    static NativeCpu load(Path lib) {
        return new NativeCpu(lib);
    }

    static final String FILE = System.mapLibraryName("jllm_cpu");

    /**
     * The library to load: {@code explicitLib}, {@code -Djllm.cpu.nativeLib}, {@code $JLLM_CPU_NATIVE_LIB},
     * {@code target/native/}, the working directory, {@code native/} next to the jar or classes directory; null when
     * none exists.
     */
    static Path locate(String explicitLib) {
        List<Path> candidates = new ArrayList<>();
        if (explicitLib != null && !explicitLib.isBlank()) {
            candidates.add(Path.of(explicitLib.trim()));
        }
        String prop = System.getProperty("jllm.cpu.nativeLib");
        if (prop != null && !prop.isBlank()) {
            candidates.add(Path.of(prop));
        }
        String env = System.getenv("JLLM_CPU_NATIVE_LIB");
        if (env != null && !env.isBlank()) {
            candidates.add(Path.of(env));
        }
        candidates.add(Path.of("target", "native", FILE));
        candidates.add(Path.of(FILE));
        try { // next to the jar / classes directory
            Path codeSource = Path.of(NativeCpu.class.getProtectionDomain().getCodeSource().getLocation().toURI());
            candidates.add(codeSource.getParent().resolve("native").resolve(FILE));
        } catch (Exception ignored) {
            // not a file location (e.g. nested in a fat jar)
        }
        for (Path c : candidates) {
            if (Files.isRegularFile(c)) {
                return c.toAbsolutePath();
            }
        }
        return null;
    }

    // ------------------------------------------------------------------ levels

    /** Name of the kernel level in use, e.g. {@code avx512-vnni}, {@code neon-dotprod}, {@code scalar}. */
    public String level() {
        return name(setLevel(-1));
    }

    /** The best level this CPU supports. */
    int detectedLevel() {
        return detected;
    }

    String name(int level) {
        try {
            MemorySegment s = (MemorySegment) levelName.invokeExact(level);
            return s.reinterpret(64).getString(0);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    /** Selects kernel level {@code level} (tests); a level this CPU lacks is refused. Returns the level in use. */
    int setLevel(int level) {
        try {
            return (int) setLevel.invokeExact(level);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    public Path path() {
        return path;
    }

    // ------------------------------------------------------------------ kernels

    /** The C type id of {@code t}, or -1. */
    static int typeId(QType t) {
        return switch (t) {
            case Q4_0 -> T_Q4_0;
            case Q8_0 -> T_Q8_0;
            case Q4_0G -> T_Q4_0G;
            case Q8_0G -> T_Q8_0G;
            case Q4_K -> T_Q4_K;
            case Q5_K -> T_Q5_K;
            case Q6_K -> T_Q6_K;
            default -> -1;
        };
    }

    /** Whether the library has a kernel for {@code w} (for a split tensor: both of its parts). */
    public boolean supports(QTensor w) {
        if (w.second != null) {
            return supports(w.partOf(0)) && supports(w.second);
        }
        int t = typeId(w.type);
        if (t < 0) {
            return false;
        }
        return t >= T_Q4_K ? w.cols % Ggml.QK_K == 0 : w.cols % QType.BLOCK == 0;
    }

    /**
     * A quantized activation in native memory: {@code n} int8 values, then a float scale and an int sum per 32-block.
     * From an automatic arena, so any thread can read it; written by one thread before the row loop is published.
     */
    public static final class Activation {
        private MemorySegment q = MemorySegment.NULL;
        private MemorySegment d = MemorySegment.NULL;
        private MemorySegment s = MemorySegment.NULL;
        private int capacity;

        void ensure(int n) {
            if (n > capacity) {
                Arena a = Arena.ofAuto();
                int nb = n / QType.BLOCK;
                MemorySegment all = a.allocate(n + 8L * nb, 64);
                q = all.asSlice(0, n);
                d = all.asSlice(n, 4L * nb);
                s = all.asSlice(n + 4L * nb, 4L * nb);
                capacity = n;
            }
        }

        /** The int8 values (natural order). */
        byte get(int i) {
            return q.get(ValueLayout.JAVA_BYTE, i);
        }

        float scale(int b) {
            return d.getAtIndex(ValueLayout.JAVA_FLOAT, b);
        }

        int sum(int b) {
            return s.getAtIndex(ValueLayout.JAVA_INT, b);
        }

        /** Sets the activation directly (tests): {@code q[i]}, scales {@code d}, block sums computed. */
        void set(byte[] values, float[] scales) {
            ensure(values.length);
            MemorySegment.copy(values, 0, q, ValueLayout.JAVA_BYTE, 0, values.length);
            for (int b = 0; b < values.length / QType.BLOCK; b++) {
                int sum = 0;
                for (int j = 0; j < QType.BLOCK; j++) {
                    sum += values[b * QType.BLOCK + j];
                }
                s.setAtIndex(ValueLayout.JAVA_INT, b, sum);
                d.setAtIndex(ValueLayout.JAVA_FLOAT, b, scales[b]);
            }
        }
    }

    /** Quantizes {@code x[off .. off + n)} (n a multiple of 32) into {@code a}, exactly as {@link Int8Kernels#quantize}. */
    public void quantize(float[] x, int off, int n, Activation a) {
        a.ensure(n);
        try {
            quantize.invokeExact(MemorySegment.ofArray(x).asSlice(4L * off, 4L * n), n, a.q, a.d, a.s);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    /**
     * {@code out[outOff + i] (+)= dot(W[r0 + i, :], x)} for {@code i < r1 - r0}, where x is the activation of {@code a}
     * from 32-block {@code xBlock} on (a MoE slot's offset). A split tensor is called part by part.
     */
    public void dotRows(QTensor w, int r0, int r1, Activation a, int xBlock, float[] out, int outOff, boolean acc) {
        if (w.second == null) {
            call(w, r0, r1, a, xBlock, out, outOff, acc);
            return;
        }
        // the part changes every `run` rows: at rows / 2 (stacked gate | up) or every expert's half (interleaved)
        int run = w.interleave == 0 ? w.rows / 2 : w.interleave;
        for (int r = r0; r < r1; ) {
            int end = Math.min(r1, (r / run + 1) * run);
            int pr = w.partRow(r);
            call(w.partOf(r), pr, pr + (end - r), a, xBlock, out, outOff + (r - r0), acc);
            r = end;
        }
    }

    private void call(QTensor w, int r0, int r1, Activation a, int xBlock, float[] out, int outOff, boolean acc) {
        if (r0 < 0 || r1 > w.rows || r0 > r1 || outOff < 0 || outOff + (r1 - r0) > out.length
                || (long) xBlock * QType.BLOCK + w.cols > a.capacity) {
            throw new IndexOutOfBoundsException("rows " + r0 + ".." + r1 + " of " + w.rows + ", out " + outOff
                    + " of " + out.length + ", activation block " + xBlock);
        }
        long xo = (long) xBlock * QType.BLOCK;
        try {
            dotRows.invokeExact(typeId(w.type), w.data, w.type.gguf || w.type == QType.Q6_K ? 0L : w.scalesOffset,
                    r0, r1, w.cols, a.q.asSlice(xo), a.d.asSlice(4L * xBlock), a.s.asSlice(4L * xBlock),
                    MemorySegment.ofArray(out).asSlice(4L * outOff), acc ? 1 : 0);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    // ------------------------------------------------------------------ self-check

    /**
     * Compares the library with the Java kernels on random rows of every type: the activation quantizer bit for bit;
     * with unit weight / activation scales (every intermediate an integer below 2^24) the dot products exactly; with
     * random scales within float rounding. Returns null when everything matches, else what did not.
     */
    String selfCheck() {
        try (Arena arena = Arena.ofConfined()) {
            Random rnd = new Random(42);
            String q = checkQuantize(rnd, 256 + 64);
            if (q != null) {
                return q;
            }
            for (QType t : new QType[]{QType.Q4_0, QType.Q8_0, QType.Q4_0G, QType.Q8_0G, QType.Q4_K, QType.Q5_K, QType.Q6_K}) {
                String f = checkType(arena, rnd, t, 3, 256, true);
                if (f == null) {
                    f = checkType(arena, rnd, t, 3, t.blockSize == Ggml.QK_K ? 512 : 96, false);
                }
                if (f != null) {
                    return f;
                }
            }
        }
        return null;
    }

    /** Native quantizer vs {@link Int8Kernels#quantize}: the same values, scales and block sums. Null when equal. */
    String checkQuantize(Random rnd, int n) {
        float[] x = new float[n + 3];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) (rnd.nextGaussian() * (i % 7 == 0 ? 8 : 1)) * (i % 97 == 0 ? 0 : 1);
        }
        if (n >= 96) {
            java.util.Arrays.fill(x, 3 + 64, 3 + 96, 0f); // an all-zero block: d = 0
        }
        short[] jq = new short[n];
        float[] js = new float[n / QType.BLOCK];
        Int8Kernels.quantize(x, 3, n, jq, js);
        Activation a = new Activation();
        quantize(x, 3, n, a);
        for (int b = 0; b < n / QType.BLOCK; b++) {
            if (Float.floatToRawIntBits(js[b]) != Float.floatToRawIntBits(a.scale(b))) {
                return "quantize: scale of block " + b + " " + a.scale(b) + " != " + js[b];
            }
            int sum = 0;
            for (int j = 0; j < QType.BLOCK; j++) {
                int v = jq[b * QType.BLOCK + Int8Kernels.permutedIndex(j)];
                sum += v;
                if (a.get(b * QType.BLOCK + j) != v) {
                    return "quantize: element " + (b * QType.BLOCK + j) + " " + a.get(b * QType.BLOCK + j) + " != " + v;
                }
            }
            if (a.sum(b) != sum) {
                return "quantize: sum of block " + b;
            }
        }
        return null;
    }

    /**
     * {@code rows} random rows of {@code type} and an activation of {@code cols}, native vs Java. {@code exact}: unit
     * scales (and small activations or unit sub-block scales), so both sides must agree bit for bit; otherwise random
     * scales and a tolerance of 3e-5 of {@code Σ|w_i·x_i|}. Null when they agree.
     */
    String checkType(Arena arena, Random rnd, QType type, int rows, int cols, boolean exact) {
        boolean small = exact && rnd.nextBoolean();
        QTensor w = randomTensor(arena, type, rows, cols, rnd, exact, small);
        byte[] xq = new byte[cols];
        float[] xs = new float[cols / QType.BLOCK];
        int lim = small ? 7 : 127;
        for (int i = 0; i < cols; i++) {
            xq[i] = (byte) (rnd.nextInt(2 * lim + 1) - lim);
        }
        for (int b = 0; b < xs.length; b++) {
            xs[b] = exact ? 1f : (float) (0.002 + 0.03 * rnd.nextDouble());
        }
        return compare(w, xq, xs, exact, rows);
    }

    /** Native vs Java dot products of every row of {@code w} with the activation ({@code xq}, {@code xs}); null when they agree. */
    String compare(QTensor w, byte[] xq, float[] xs, boolean exact, int rows) {
        int cols = w.cols;
        Activation a = new Activation();
        a.set(xq, xs);
        short[] perm = new short[cols];
        int[] sums = new int[cols / QType.BLOCK];
        for (int b = 0; b < cols / QType.BLOCK; b++) {
            for (int j = 0; j < QType.BLOCK; j++) {
                perm[b * QType.BLOCK + Int8Kernels.permutedIndex(j)] = xq[b * QType.BLOCK + j];
            }
            sums[b] = a.sum(b);
        }
        float[] nat = new float[rows + 2];
        dotRows(w, 0, rows, a, 0, nat, 1, false);
        float[] row = new float[cols];
        for (int r = 0; r < rows; r++) {
            float java = KQuantKernels.supports(w) ? KQuantKernels.dotRow(w, r, perm, xs, sums, 0)
                    : Int8Kernels.dotRow(w, r, perm, xs, 0);
            float n = nat[1 + r];
            if (exact) {
                if (Float.floatToRawIntBits(java) != Float.floatToRawIntBits(n)) {
                    return w.type + " " + w.rows + "x" + cols + " row " + r + " (unit scales): native " + n + " != java " + java;
                }
            } else {
                VectorKernels.dequantRow(w, r, row, 0);
                double mag = 0;
                for (int i = 0; i < cols; i++) {
                    mag += Math.abs(row[i] * xq[i] * xs[i / QType.BLOCK]);
                }
                if (!(Math.abs(n - java) <= 3e-5 * mag + 1e-30)) {
                    return w.type + " " + w.rows + "x" + cols + " row " + r + ": native " + n + " vs java " + java
                            + " (magnitude " + mag + ")";
                }
            }
        }
        if (nat[0] != 0 || nat[rows + 1] != 0) {
            return w.type + ": wrote outside its rows";
        }
        return null;
    }

    /**
     * A {@code rows × cols} tensor of {@code type} with random quants. {@code unit}: every weight scale 1 (f32 or f16);
     * for the K-quants also every sub-block scale and min 1 unless {@code randomSub} (random 6-bit / int8 sub-block
     * scales; then the activations must be small to keep the sums exact). Otherwise random positive f16 / f32 scales.
     */
    static QTensor randomTensor(Arena arena, QType type, int rows, int cols, Random rnd, boolean unit, boolean randomSub) {
        int nb = cols / type.blockSize;
        long quantBytes = (long) rows * nb * type.blockBytes;
        if (!type.gguf && type != QType.Q6_K) { // the engine's Q4_0 / Q8_0: blocks, then f32 scales
            MemorySegment seg = arena.allocate(QTensor.byteSize(type, rows, cols), 64);
            for (long i = 0; i < quantBytes; i++) {
                seg.set(ValueLayout.JAVA_BYTE, i, (byte) rnd.nextInt(256));
            }
            QTensor t = QTensor.view(type, rows, cols, seg);
            for (int r = 0; r < rows; r++) {
                for (int b = 0; b < nb; b++) {
                    seg.set(ValueLayout.JAVA_FLOAT_UNALIGNED, t.scaleOffset(r) + 4L * b,
                            unit ? 1f : (float) (0.001 + 0.05 * rnd.nextDouble()));
                }
            }
            return t;
        }
        MemorySegment seg = arena.allocate(quantBytes, 16);
        for (long i = 0; i < quantBytes; i++) {
            seg.set(ValueLayout.JAVA_BYTE, i, (byte) rnd.nextInt(256));
        }
        ValueLayout.OfShort f16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (long blk = 0; blk < quantBytes; blk += type.blockBytes) {
            short d = unit ? Float.floatToFloat16(1f) : Float.floatToFloat16((float) (0.001 + 0.05 * rnd.nextDouble()));
            short dmin = unit ? Float.floatToFloat16(1f) : Float.floatToFloat16((float) (0.001 + 0.02 * rnd.nextDouble()));
            switch (type) {
                case Q4_0G, Q8_0G -> seg.set(f16, blk, d);
                case Q4_K, Q5_K -> {
                    seg.set(f16, blk, d);
                    seg.set(f16, blk + Ggml.K_DMIN, dmin);
                    if (unit && !randomSub) { // sub-block scales and mins all 1
                        for (int j = 0; j < 8; j++) {
                            seg.set(ValueLayout.JAVA_BYTE, blk + Ggml.K_SCALES + j, (byte) 1);
                        }
                        for (int j = 8; j < 12; j++) {
                            seg.set(ValueLayout.JAVA_BYTE, blk + Ggml.K_SCALES + j, (byte) 0x11);
                        }
                    }
                }
                case Q6_K -> {
                    seg.set(f16, blk + Q6K.D_OFFSET, d);
                    if (unit && !randomSub) {
                        for (int j = 0; j < 16; j++) {
                            seg.set(ValueLayout.JAVA_BYTE, blk + Q6K.SCALES_OFFSET + j, (byte) 1);
                        }
                    }
                }
                default -> throw new IllegalArgumentException(type.name());
            }
        }
        return QTensor.mapped(type, rows, cols, seg);
    }
}
