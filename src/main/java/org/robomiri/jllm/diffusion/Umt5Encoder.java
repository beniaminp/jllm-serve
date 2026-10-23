package org.robomiri.jllm.diffusion;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
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
 */
public final class Umt5Encoder implements AutoCloseable {

    private static final VectorSpecies<Float> F128 = FloatVector.SPECIES_128;
    private static final VectorSpecies<Byte> B128 = ByteVector.SPECIES_128;
    private static final int GROUP = 32;
    /** The int8 copy of the linear layers, in {@code ~/.cache/jllm/umt5/} (see {@link #writeCache}). */
    private static final int CACHE_MAGIC = 0x554D5438; // "UMT8"
    private static final int CACHE_VERSION = 1;
    private static final int CACHE_CHUNK = 1 << 20;

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
            float s = acc.reduceLanes(VectorOperators.ADD);
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
                total += scales[o * groups + g] * p.reduceLanes(VectorOperators.ADD);
            }
            return total;
        }
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
    private final CacheTarget cache;
    private int cacheIndex;
    private long residentBytes;

    /**
     * @param dir text_encoder folder of a diffusers pipeline ({@code config.json}, {@code model[-0000x-of-0000y].safetensors})
     * @param q8  quantize the linear layers to int8 (otherwise they are kept as float: only for small models)
     */
    public Umt5Encoder(Path dir, boolean q8) throws IOException {
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
        embed = ckpt.has("shared.weight") ? ckpt.get("shared.weight") : ckpt.get("encoder.embed_tokens.weight");
        cache = q8 ? cacheTarget(dir) : null;
        List<Lin> cached = cache == null ? null : readCache(cache, arena, nl * 7);
        layers = new Layer[nl];
        for (int i = 0; i < nl; i++) {
            String p = "encoder.block." + i + ".layer.";
            layers[i] = new Layer(vec(p + "0.layer_norm.weight"), lin(cached, p + "0.SelfAttention.q.weight"),
                    lin(cached, p + "0.SelfAttention.k.weight"), lin(cached, p + "0.SelfAttention.v.weight"),
                    lin(cached, p + "0.SelfAttention.o.weight"), vec(p + "0.SelfAttention.relative_attention_bias.weight"),
                    vec(p + "1.layer_norm.weight"), lin(cached, p + "1.DenseReluDense.wi_0.weight"),
                    lin(cached, p + "1.DenseReluDense.wi_1.weight"), lin(cached, p + "1.DenseReluDense.wo.weight"));
        }
        finalNorm = vec("encoder.final_layer_norm.weight");
        if (cache != null && cached == null) {
            writeCache(cache);
        }
    }

    private float[] vec(String name) {
        float[] v = ckpt.get(name).toFloatArray();
        residentBytes += v.length * 4L;
        return v;
    }

    private Lin lin(List<Lin> cached, String name) throws IOException {
        SafeTensors.Tensor t = ckpt.get(name);
        int n = (int) t.shape()[0];
        int k = (int) t.shape()[1];
        if (!q8) {
            float[] w = t.toFloatArray();
            residentBytes += w.length * 4L;
            return new Lin(n, k, w, null, null);
        }
        if (cached != null) {
            if (cacheIndex >= cached.size()) {
                throw new IOException("int8 cache for " + ckpt.dir + " is incomplete: delete " + cache.file());
            }
            Lin l = cached.get(cacheIndex++);
            if (l.n != n || l.k != k) {
                throw new IOException("int8 cache for " + ckpt.dir + " does not match the checkpoint: delete " + cache.file());
            }
            residentBytes += (long) n * k + l.scales.length * 4L;
            return l;
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

    // ------------------------------------------------------------------ the int8 cache

    /** {@code ~/.cache/jllm/umt5/<key>.q8}: the quantized linear layers of one {@code text_encoder} folder. */
    private record CacheTarget(Path file, long key) {
    }

    /** All seven quantized linears of a layer, in the order {@link #writeCache} stores them. */
    private static Lin[] linears(Layer ly) {
        return new Lin[]{ly.q(), ly.k(), ly.v(), ly.o(), ly.wi0(), ly.wi1(), ly.wo()};
    }

    /**
     * Derived from the folder path and the size and modification time of {@code config.json} and the weight files, so that replacing the
     * model invalidates the cache. 0 when the folder cannot be listed.
     */
    private static long cacheKey(Path dir) {
        try {
            List<Path> files = new ArrayList<>();
            try (var listing = Files.list(dir)) {
                listing.filter(p -> {
                    String n = p.getFileName().toString();
                    return n.equals("config.json") || (n.startsWith("model")
                            && (n.endsWith(".safetensors") || n.endsWith(".safetensors.index.json")));
                }).sorted().forEach(files::add);
            }
            long h = 0xcbf29ce484222325L;
            for (char c : dir.toAbsolutePath().normalize().toString().toCharArray()) {
                h = (h ^ c) * 0x100000001b3L;
            }
            for (Path f : files) {
                for (char c : f.getFileName().toString().toCharArray()) {
                    h = (h ^ c) * 0x100000001b3L;
                }
                h = (h ^ Files.size(f)) * 0x100000001b3L;
                h = (h ^ Files.getLastModifiedTime(f).toMillis()) * 0x100000001b3L;
            }
            return h;
        } catch (IOException e) {
            return 0;
        }
    }

    private static CacheTarget cacheTarget(Path dir) {
        String home = System.getProperty("user.home");
        if (home == null || home.isBlank()) {
            return null;
        }
        long key = cacheKey(dir);
        if (key == 0) {
            return null;
        }
        return new CacheTarget(Path.of(home, ".cache", "jllm", "umt5").resolve(Long.toHexString(key) + ".q8"), key);
    }

    /** The cached layers, or null when there is no usable cache (the caller then quantizes and writes one). */
    private static List<Lin> readCache(CacheTarget target, Arena arena, int expected) {
        if (!Files.isRegularFile(target.file())) {
            return null;
        }
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(target.file()), CACHE_CHUNK))) {
            if (in.readInt() != CACHE_MAGIC || in.readInt() != CACHE_VERSION || in.readLong() != target.key()
                    || in.readInt() != expected) {
                return null;
            }
            int[][] dims = new int[expected][2];
            for (int i = 0; i < expected; i++) {
                dims[i][0] = in.readInt();
                dims[i][1] = in.readInt();
                if (dims[i][0] <= 0 || dims[i][1] <= 0 || dims[i][1] % GROUP != 0) {
                    return null;
                }
            }
            List<Lin> out = new ArrayList<>(expected);
            for (int[] d : dims) {
                MemorySegment q = arena.allocate((long) d[0] * d[1], 64);
                readBytes(in, q, (long) d[0] * d[1]);
                out.add(new Lin(d[0], d[1], null, q, readFloats(in, d[0] * (d[1] / GROUP))));
            }
            return out;
        } catch (IOException | RuntimeException e) {
            return null;
        }
    }

    /** Writes the int8 copy (best effort: a full disk or a read-only home only means the next request quantizes again). */
    private void writeCache(CacheTarget target) {
        Path tmp = null;
        try {
            Files.createDirectories(target.file().getParent());
            tmp = Files.createTempFile(target.file().getParent(), "umt5-", ".tmp");
            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(tmp), CACHE_CHUNK))) {
                out.writeInt(CACHE_MAGIC);
                out.writeInt(CACHE_VERSION);
                out.writeLong(target.key());
                out.writeInt(layers.length * 7);
                for (Layer ly : layers) {
                    for (Lin l : linears(ly)) {
                        out.writeInt(l.n);
                        out.writeInt(l.k);
                    }
                }
                for (Layer ly : layers) {
                    for (Lin l : linears(ly)) {
                        writeBytes(out, l.q, (long) l.n * l.k);
                        writeFloats(out, l.scales);
                    }
                }
            }
            Files.move(tmp, target.file(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            tmp = null;
        } catch (IOException | RuntimeException e) {
            // no cache
        } finally {
            if (tmp != null) {
                try {
                    Files.deleteIfExists(tmp);
                } catch (IOException ignored) {
                    // left behind
                }
            }
        }
    }

    private static void writeBytes(DataOutputStream out, MemorySegment src, long count) throws IOException {
        byte[] chunk = new byte[CACHE_CHUNK];
        for (long at = 0; at < count; at += chunk.length) {
            int n = (int) Math.min(chunk.length, count - at);
            MemorySegment.copy(src, ValueLayout.JAVA_BYTE, at, chunk, 0, n);
            out.write(chunk, 0, n);
        }
    }

    private static void readBytes(DataInputStream in, MemorySegment dst, long count) throws IOException {
        byte[] chunk = new byte[CACHE_CHUNK];
        for (long at = 0; at < count; at += chunk.length) {
            int n = (int) Math.min(chunk.length, count - at);
            in.readFully(chunk, 0, n);
            MemorySegment.copy(chunk, 0, dst, ValueLayout.JAVA_BYTE, at, n);
        }
    }

    private static void writeFloats(DataOutputStream out, float[] v) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(1 << 16).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < v.length; ) {
            bb.clear();
            int n = Math.min(bb.capacity() / 4, v.length - i);
            bb.asFloatBuffer().put(v, i, n);
            out.write(bb.array(), 0, n * 4);
            i += n;
        }
    }

    private static float[] readFloats(DataInputStream in, int count) throws IOException {
        float[] v = new float[count];
        byte[] chunk = new byte[1 << 16];
        for (int i = 0; i < count; ) {
            int n = Math.min(chunk.length / 4, count - i);
            in.readFully(chunk, 0, n * 4);
            ByteBuffer.wrap(chunk, 0, n * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(v, i, n);
            i += n;
        }
        return v;
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
