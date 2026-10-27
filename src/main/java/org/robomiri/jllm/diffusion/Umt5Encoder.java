package org.robomiri.jllm.diffusion;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * The UMT5 encoder ({@code UMT5EncoderModel}: Wan's text encoder, {@code google/umt5-xxl}) on the CPU, with its linear layers optionally
 * quantized to int8 while the weights are read.
 *
 * <pre>
 *   x = embedding(ids)                                              [L, 4096]
 *   24 times:  x += Attn(RMS(x))    q, k, v, o without bias; NO 1/√d scaling; every layer adds its own learned relative-position bias
 *                                   {@code bias[h, i, j] = table[bucket(j − i), h]} (32 buckets, half per direction, logarithmic beyond 8, up to 128)
 *              x += W_o (gelu_tanh(W_0 RMS(x)) ⊙ W_1 RMS(x))        gated feed-forward, d_ff = 10240
 *   y = RMS(x)
 * </pre>
 *
 * Only the real tokens are encoded: diffusers pads the prompt to 512 tokens (the pipeline's max_sequence_length), masks the padding in attention and then zeroes the padded
 * rows, which gives the same rows as encoding the unpadded sequence (masked keys have weight exactly 0).
 *
 * <p><b>int8.</b> The 5.7 B parameters are 22.7 GB as shipped (float32) and 11 GB as bfloat16: far beyond the memory this machine can
 * give. With {@code q8} every linear weight is quantized per row in groups of 32 values to int8 with a float scale ({@code w ≈ s·q},
 * the Q8_0 layout of chapter 6): 4.6 GB for the 24 layers, read tensor by tensor so that the float32 file is never resident. Activations
 * stay float; the products are summed in float. The embedding table is read row by row from the memory-mapped file.
 *
 * <p><b>int8 cache.</b> Quantizing means reading the whole float32 file. With the cache on, the int8 rows and their scales are written
 * once to {@link #CACHE_FILE} in the encoder's folder (its index {@link #CACHE_INDEX}, written last, records the float files' names,
 * sizes and modification times); a later encoder whose float files still match maps that file instead (5.2 GB of int8 rows and scales
 * rather than 18.5 GB of float32 weights, and no quantization). The mapped weights are the same bytes, so the encodings are bit for bit those of a fresh quantization.
 */
public final class Umt5Encoder implements AutoCloseable {

    private static final VectorSpecies<Float> F128 = FloatVector.SPECIES_128;
    private static final VectorSpecies<Byte> B128 = ByteVector.SPECIES_128;
    private static final int GROUP = 32;

    /** The int8 copy of the linear layers, next to the float files: per layer its int8 rows, then its scales (little-endian floats). */
    public static final String CACHE_FILE = "jllm-q8.bin";
    /** The index of {@link #CACHE_FILE}, written last: {@code {version, group, sources: [{file, size, modified}], bytes, tensors: {name: {n, k, q, s}}}}. */
    public static final String CACHE_INDEX = "jllm-q8.json";
    /** Bumped when the layout of the cache or the quantization changes, so older caches are rewritten instead of read. */
    static final int CACHE_VERSION = 1;
    private static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final String[] LINEARS = {"0.SelfAttention.q", "0.SelfAttention.k", "0.SelfAttention.v", "0.SelfAttention.o",
            "1.DenseReluDense.wi_0", "1.DenseReluDense.wi_1", "1.DenseReluDense.wo"};

    /** A linear layer {@code y = x·Wᵀ} with {@code W [n, k]}: float, or int8 groups with scales. */
    private static final class Lin {
        final int n;
        final int k;
        final float[] w;           // float mode
        final MemorySegment q;     // int8 mode: [n][k]
        final float[] scales;      // int8 mode: [n][k / 32]

        Lin(int n, int k, float[] w, MemorySegment q, float[] scales) {
            this.n = n;
            this.k = k;
            this.w = w;
            this.q = q;
            this.scales = scales;
        }

        /** {@code out[t][n] = Σ_k x[t][k] · W[n][k]} for {@code rows} token rows. */
        float[] apply(float[] x, int rows) {
            float[] out = new float[rows * n];
            IntStream.range(0, n).parallel().forEach(o -> {
                for (int t = 0; t < rows; t++) {
                    out[t * n + o] = w != null ? dotF(o, x, t * k) : dotQ(o, x, t * k);
                }
            });
            return out;
        }

        private float dotF(int o, float[] x, int xo) {
            FloatVector acc = FloatVector.zero(F128);
            int i = 0;
            for (; i < F128.loopBound(k); i += F128.length()) {
                acc = FloatVector.fromArray(F128, w, o * k + i).fma(FloatVector.fromArray(F128, x, xo + i), acc);
            }
            float s = sum4(acc);
            for (; i < k; i++) {
                s += w[o * k + i] * x[xo + i];
            }
            return s;
        }

        private float dotQ(int o, float[] x, int xo) {
            float total = 0f;
            long base = (long) o * k;
            int groups = k / GROUP;
            for (int g = 0; g < groups; g++) {
                FloatVector p = FloatVector.zero(F128);
                for (int j = 0; j < GROUP; j += 16) {
                    ByteVector b = ByteVector.fromMemorySegment(B128, q, base + g * GROUP + j, ByteOrder.LITTLE_ENDIAN);
                    int xi = xo + g * GROUP + j;
                    for (int part = 0; part < 4; part++) {
                        FloatVector wf = (FloatVector) b.convertShape(VectorOperators.B2F, F128, part);
                        p = wf.fma(FloatVector.fromArray(F128, x, xi + part * 4), p);
                    }
                }
                total += scales[o * groups + g] * sum4(p);
            }
            return total;
        }
    }

    /**
     * The four lanes added in a fixed order, {@code (l0 + l1) + (l2 + l3)}: {@code reduceLanes(ADD)} leaves the order open and the
     * interpreter's fallback adds them one after the other, so an encoder's first (cold) prompt differed in the last bit from the next.
     */
    private static float sum4(FloatVector v) {
        return (v.lane(0) + v.lane(1)) + (v.lane(2) + v.lane(3));
    }

    private record Layer(float[] n1, Lin q, Lin k, Lin v, Lin o, float[] bias, float[] n2, Lin wi0, Lin wi1, Lin wo) {
    }

    public final int dModel;
    public final int heads;
    private final int dKv;
    private final int dFf;
    private final int buckets;
    private final int maxDistance;
    private final float eps;
    private final boolean q8;
    private final SafeTensors.Tensor embed;
    private final Layer[] layers;
    private final float[] finalNorm;
    private final Checkpoint ckpt;
    private final Arena arena = Arena.ofShared();
    private long residentBytes;
    /** The mapped {@link #CACHE_FILE} and its tensor index while the layers are read from it, else null. */
    private MemorySegment cacheSeg;
    private Map<String, Object> cacheTensors;
    private final boolean fromCache;
    private final Map<String, Lin> lins = new LinkedHashMap<>();

    /**
     * @param dir text_encoder folder of a diffusers pipeline ({@code config.json}, {@code model[-0000x-of-0000y].safetensors})
     * @param q8  quantize the linear layers to int8 (otherwise they are kept as float: only for small models)
     */
    public Umt5Encoder(Path dir, boolean q8) throws IOException {
        this(dir, q8, false);
    }

    /**
     * @param cache with {@code q8}: map {@link #CACHE_FILE} when it was written from the same float files, else quantize them and write
     *              it (best effort: when the folder is read-only or the disk is full, a warning says so and the next encoder quantizes
     *              again); without {@code q8} it has no effect
     */
    public Umt5Encoder(Path dir, boolean q8, boolean cache) throws IOException {
        this.q8 = q8;
        this.ckpt = Checkpoint.open(dir, "model");
        dModel = ckpt.getInt("d_model", 4096);
        heads = ckpt.getInt("num_heads", 64);
        dKv = ckpt.getInt("d_kv", 64);
        dFf = ckpt.getInt("d_ff", 10240);
        buckets = ckpt.getInt("relative_attention_num_buckets", 32);
        maxDistance = ckpt.getInt("relative_attention_max_distance", 128);
        eps = (float) ckpt.getDouble("layer_norm_epsilon", 1e-6);
        int nl = ckpt.getInt("num_layers", 24);
        if (!"gated-gelu".equals(ckpt.getString("feed_forward_proj", "gated-gelu")) || dModel % GROUP != 0 || dFf % GROUP != 0
                || (dKv * heads) % GROUP != 0) {
            throw new IOException("unsupported UMT5 configuration in " + dir);
        }
        if (q8 && cache) {
            openCache(dir, nl);
        }
        embed = ckpt.has("shared.weight") ? ckpt.get("shared.weight") : ckpt.get("encoder.embed_tokens.weight");
        layers = new Layer[nl];
        for (int i = 0; i < nl; i++) {
            String p = "encoder.block." + i + ".layer.";
            layers[i] = new Layer(vec(p + "0.layer_norm.weight"), lin(p + "0.SelfAttention.q.weight"), lin(p + "0.SelfAttention.k.weight"),
                    lin(p + "0.SelfAttention.v.weight"), lin(p + "0.SelfAttention.o.weight"), vec(p + "0.SelfAttention.relative_attention_bias.weight"),
                    vec(p + "1.layer_norm.weight"), lin(p + "1.DenseReluDense.wi_0.weight"), lin(p + "1.DenseReluDense.wi_1.weight"),
                    lin(p + "1.DenseReluDense.wo.weight"));
        }
        finalNorm = vec("encoder.final_layer_norm.weight");
        fromCache = cacheSeg != null;
        cacheSeg = null; // the layers hold their slices of the mapping, which lives as long as the encoder
        cacheTensors = null;
        if (q8 && cache && !fromCache) {
            writeCache(dir);
        }
    }

    /** True when the linear layers were mapped from {@link #CACHE_FILE} rather than quantized from the float files. */
    public boolean fromCache() {
        return fromCache;
    }

    /** The names of the linear layers, in the order the cache stores them. */
    private static List<String> linearNames(int layers) {
        List<String> names = new ArrayList<>();
        for (int i = 0; i < layers; i++) {
            for (String l : LINEARS) {
                names.add("encoder.block." + i + ".layer." + l + ".weight");
            }
        }
        return names;
    }

    /** {@code [{file, size, modified}]} of the float files the weights are read from: what a cache must have been written from. */
    private List<Object> sources() throws IOException {
        List<Object> out = new ArrayList<>();
        for (Path f : ckpt.files()) {
            out.add(Json.obj("file", f.getFileName().toString(), "size", Files.size(f), "modified", Files.getLastModifiedTime(f).toMillis()));
        }
        return out;
    }

    /**
     * Maps the cache when its index matches this encoder (version, group, the float files, the file size and every layer's shape and
     * byte range); otherwise leaves {@link #cacheSeg} null, saying why when a cache was there.
     */
    private void openCache(Path dir, int layers) throws IOException {
        Path idx = dir.resolve(CACHE_INDEX);
        Path bin = dir.resolve(CACHE_FILE);
        if (!Files.isRegularFile(idx) || !Files.isRegularFile(bin)) {
            return;
        }
        String why = null;
        Map<String, Object> index = null;
        try {
            index = Json.parseObject(Files.readString(idx, StandardCharsets.UTF_8));
        } catch (RuntimeException e) {
            why = "its index does not parse";
        }
        long size = Files.size(bin);
        if (why == null && (Json.num(index, "version", -1) != CACHE_VERSION || Json.num(index, "group", -1) != GROUP)) {
            why = "another version";
        } else if (why == null && !Json.write(index.get("sources")).equals(Json.write(sources()))) {
            why = "the float32 files changed";
        } else if (why == null && Json.num(index, "bytes", -1) != size) {
            why = "the file is " + size + " bytes, not " + Json.num(index, "bytes", -1);
        }
        Map<String, Object> tensors = why == null && index.get("tensors") instanceof Map<?, ?> ? Json.map(index, "tensors") : null;
        if (why == null && tensors == null) {
            why = "its index has no tensors";
        }
        for (String name : why == null ? linearNames(layers) : List.<String>of()) {
            Object e = tensors.get(name);
            SafeTensors.Tensor t = ckpt.has(name) ? ckpt.get(name) : null;
            if (!(e instanceof Map<?, ?>) || t == null) {
                why = "no entry for " + name;
                break;
            }
            long n = Json.num(e, "n", -1);
            long k = Json.num(e, "k", -1);
            long q = Json.num(e, "q", -1);
            long sc = Json.num(e, "s", -1);
            if (t.shape().length != 2 || n != t.shape()[0] || k != t.shape()[1] || q < 0 || sc < 0 || q + n * k > size
                    || sc + n * (k / GROUP) * 4 > size) {
                why = "the entry of " + name + " does not fit";
                break;
            }
        }
        if (why != null) {
            System.err.println("UMT5: not using the int8 copy in " + dir + " (" + why + "); quantizing the float32 weights again");
            return;
        }
        try (FileChannel ch = FileChannel.open(bin, StandardOpenOption.READ)) {
            cacheSeg = ch.map(FileChannel.MapMode.READ_ONLY, 0, size, arena);
        }
        cacheTensors = tensors;
    }

    /**
     * Writes {@link #CACHE_FILE} (as {@code .partial}, renamed when complete) and then {@link #CACHE_INDEX} from the quantized layers.
     * Failures only cost the next encoder the quantization again: they are reported and the partial files removed.
     */
    private void writeCache(Path dir) {
        Path bin = dir.resolve(CACHE_FILE);
        Path idx = dir.resolve(CACHE_INDEX);
        Path part = dir.resolve(CACHE_FILE + ".partial");
        Path idxPart = dir.resolve(CACHE_INDEX + ".partial");
        try {
            Files.deleteIfExists(idx); // without its index a cache file is never read
            long need = 0;
            for (Lin l : lins.values()) {
                need = align(align(need + (long) l.n * l.k) + l.scales.length * 4L);
            }
            long free = Files.getFileStore(dir).getUsableSpace();
            if (free < need + (1L << 30)) {
                System.err.printf("UMT5: not writing the int8 copy (%.1f GB) to %s: %.1f GB free%n", need / 1e9, dir, free / 1e9);
                return;
            }
            Map<String, Object> tensors = new LinkedHashMap<>();
            long bytes;
            try (FileChannel ch = FileChannel.open(part, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
                long off = 0;
                for (Map.Entry<String, Lin> e : lins.entrySet()) {
                    Lin l = e.getValue();
                    long q = off;
                    writeFully(ch, l.q.asByteBuffer(), q);
                    long sc = align(q + (long) l.n * l.k);
                    ByteBuffer sb = ByteBuffer.allocate(l.scales.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                    sb.asFloatBuffer().put(l.scales);
                    writeFully(ch, sb, sc);
                    off = align(sc + l.scales.length * 4L);
                    tensors.put(e.getKey(), Json.obj("n", l.n, "k", l.k, "q", q, "s", sc));
                }
                ch.force(true);
                bytes = ch.size();
            }
            Files.move(part, bin, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            Map<String, Object> index = Json.obj("version", CACHE_VERSION, "group", GROUP, "sources", sources(), "bytes", bytes, "tensors", tensors);
            Files.writeString(idxPart, Json.write(index), StandardCharsets.UTF_8);
            Files.move(idxPart, idx, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            System.err.printf("UMT5: wrote the int8 copy of the text encoder (%.1f GB) to %s; later runs map it%n", bytes / 1e9, bin);
        } catch (IOException | RuntimeException e) {
            System.err.println("warning: could not write the int8 copy of the UMT5 encoder to " + dir + ": " + e
                    + " (the next run quantizes the float32 weights again)");
            try {
                Files.deleteIfExists(part);
                Files.deleteIfExists(idxPart);
            } catch (IOException ignored) {
                // nothing more to clean up: a .partial file is never read
            }
        }
    }

    private static long align(long off) {
        return (off + 63) & ~63L;
    }

    private static void writeFully(FileChannel ch, ByteBuffer b, long pos) throws IOException {
        while (b.hasRemaining()) {
            pos += ch.write(b, pos);
        }
    }

    private float[] vec(String name) {
        float[] v = ckpt.get(name).toFloatArray();
        residentBytes += v.length * 4L;
        return v;
    }

    private Lin lin(String name) {
        Lin l = cacheSeg != null ? cachedLin(name) : quantizedLin(name);
        lins.put(name, l);
        return l;
    }

    /** A linear layer whose int8 rows are a slice of the mapped cache (the scales are copied to the heap). */
    private Lin cachedLin(String name) {
        Map<String, Object> e = Json.map(cacheTensors, name);
        int n = (int) Json.num(e, "n", -1);
        int k = (int) Json.num(e, "k", -1);
        MemorySegment q = cacheSeg.asSlice(Json.num(e, "q", -1), (long) n * k);
        float[] scales = new float[n * (k / GROUP)];
        MemorySegment.copy(cacheSeg, LE_FLOAT, Json.num(e, "s", -1), scales, 0, scales.length);
        residentBytes += (long) n * k + scales.length * 4L;
        return new Lin(n, k, null, q, scales);
    }

    private Lin quantizedLin(String name) {
        SafeTensors.Tensor t = ckpt.get(name);
        int n = (int) t.shape()[0];
        int k = (int) t.shape()[1];
        if (!q8) {
            float[] w = t.toFloatArray();
            residentBytes += w.length * 4L;
            return new Lin(n, k, w, null, null);
        }
        MemorySegment seg = arena.allocate((long) n * k, 64);
        float[] scales = new float[n * (k / GROUP)];
        int groups = k / GROUP;
        IntStream.range(0, (n + 15) / 16).parallel().forEach(blk -> {
            float[] row = new float[k];
            byte[] qrow = new byte[k];
            for (int o = blk * 16; o < Math.min(n, blk * 16 + 16); o++) {
                t.toFloat((long) o * k, row, 0, k);
                for (int g = 0; g < groups; g++) {
                    float amax = 0f;
                    for (int j = 0; j < GROUP; j++) {
                        amax = Math.max(amax, Math.abs(row[g * GROUP + j]));
                    }
                    float s = amax / 127f;
                    scales[o * groups + g] = s;
                    float inv = s == 0f ? 0f : 1f / s;
                    for (int j = 0; j < GROUP; j++) {
                        qrow[g * GROUP + j] = (byte) Math.max(-127, Math.min(127, Math.round(row[g * GROUP + j] * inv)));
                    }
                }
                MemorySegment.copy(qrow, 0, seg, ValueLayout.JAVA_BYTE, (long) o * k, k);
            }
        });
        residentBytes += (long) n * k + scales.length * 4L;
        return new Lin(n, k, null, seg, scales);
    }

    /** Bytes of weights held in memory (the embedding table stays in the memory-mapped file). */
    public long residentBytes() {
        return residentBytes;
    }

    private float[] rms(float[] x, int rows, float[] w) {
        float[] out = new float[x.length];
        int d = dModel;
        for (int r = 0; r < rows; r++) {
            double q = 0;
            for (int j = 0; j < d; j++) {
                q += (double) x[r * d + j] * x[r * d + j];
            }
            float rstd = (float) (1.0 / Math.sqrt(q / d + eps));
            for (int j = 0; j < d; j++) {
                out[r * d + j] = w[j] * (x[r * d + j] * rstd);
            }
        }
        return out;
    }

    /** T5's bidirectional bucket of the relative position {@code memory − query}. */
    int bucket(int rel) {
        int nb = buckets / 2;
        int b = rel > 0 ? nb : 0;
        int r = Math.abs(rel);
        int maxExact = nb / 2;
        if (r < maxExact) {
            return b + r;
        }
        double ratio = Math.log((double) r / maxExact) / Math.log((double) maxDistance / maxExact) * (nb - maxExact);
        return b + Math.min(maxExact + (int) ratio, nb - 1);
    }

    /** The encoder's last hidden states {@code [ids.length, dModel]} for the token ids (no padding). */
    public float[] encode(int[] ids) {
        int l = ids.length;
        int d = dModel;
        float[] x = new float[l * d];
        for (int i = 0; i < l; i++) {
            embed.toFloat((long) ids[i] * d, x, i * d, d);
        }
        int inner = heads * dKv;
        for (Layer ly : layers) {
            float[] n1 = rms(x, l, ly.n1);
            float[] q = ly.q.apply(n1, l);
            float[] k = ly.k.apply(n1, l);
            float[] v = ly.v.apply(n1, l);
            float[] ctx = new float[l * inner];
            IntStream.range(0, heads).parallel().forEach(h -> {
                float[] s = new float[l];
                for (int i = 0; i < l; i++) {
                    float mx = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < l; j++) {
                        float dot = 0f;
                        for (int c = 0; c < dKv; c++) {
                            dot += q[i * inner + h * dKv + c] * k[j * inner + h * dKv + c];
                        }
                        dot += ly.bias[bucket(j - i) * heads + h];
                        s[j] = dot;
                        mx = Math.max(mx, dot);
                    }
                    float sum = 0f;
                    for (int j = 0; j < l; j++) {
                        s[j] = (float) Math.exp(s[j] - mx);
                        sum += s[j];
                    }
                    for (int j = 0; j < l; j++) {
                        float p = s[j] / sum;
                        for (int c = 0; c < dKv; c++) {
                            ctx[i * inner + h * dKv + c] += p * v[j * inner + h * dKv + c];
                        }
                    }
                }
            });
            float[] o = ly.o.apply(ctx, l);
            for (int i = 0; i < x.length; i++) {
                x[i] += o[i];
            }
            float[] n2 = rms(x, l, ly.n2);
            float[] g = ly.wi0.apply(n2, l);
            float[] u = ly.wi1.apply(n2, l);
            for (int i = 0; i < g.length; i++) {
                float a = g[i];
                float gelu = 0.5f * a * (1f + (float) Math.tanh(0.7978845608028654 * (a + 0.044715f * a * a * a)));
                g[i] = gelu * u[i];
            }
            float[] f = ly.wo.apply(g, l);
            for (int i = 0; i < x.length; i++) {
                x[i] += f[i];
            }
        }
        return rms(x, l, finalNorm);
    }

    @Override
    public void close() {
        arena.close();
        ckpt.close();
    }
}
