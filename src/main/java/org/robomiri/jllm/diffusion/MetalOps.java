package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * {@link Ops} on the GPU with the kernels of {@code diffusion.metal}: float16 (weights and activations stored as half,
 * arithmetic in float) or float32. Several {@code MetalOps} (a half one for the UNet and text encoders, a float one for the
 * VAE of SDXL) share one {@link Gpu}: one Metal library, one command stream, one buffer pool.
 *
 * <p>Dispatches are recorded into the Metal command buffer and submitted in slices of {@link #SLICE}; a download (or
 * {@link #sync}) submits the rest and waits. Because the command stream executes in order, a buffer returned to the pool
 * can be handed out again at once; only a CPU write ({@link #upload}) has to wait for the GPU first.
 */
final class MetalOps implements Ops {

    private static final int SLICE = 192;
    private static final boolean FLASH = !"false".equals(System.getProperty("jllm.diffusion.flash"));
    private static final boolean PROFILE = Boolean.getBoolean("jllm.diffusion.profile");
    /** {@code -Djllm.diffusion.simpleNorm=true}: group norm always in one threadgroup per group (the first, slower form). */
    private static final boolean SIMPLE_NORM = Boolean.getBoolean("jllm.diffusion.simpleNorm");

    /** The per-kernel times collected with {@code -Djllm.diffusion.profile=true}, as text (empty without it). */
    String profileReport() {
        StringBuilder sb = new StringBuilder();
        gpu.profile.entrySet().stream().sorted((a, b) -> Long.compare(b.getValue()[0], a.getValue()[0])).forEach(e ->
                sb.append(String.format("  %-22s %8.1f ms  %5d calls  %7.3f ms each%n", e.getKey(), e.getValue()[0] / 1e6,
                        e.getValue()[1], e.getValue()[0] / 1e6 / e.getValue()[1])));
        return sb.toString();
    }

    void resetProfile() {
        gpu.profile.clear();
    }

    /** The shared device state. */
    static final class Gpu implements AutoCloseable {
        final Metal metal;
        final Arena arena = Arena.ofShared();
        final MemorySegment bufs = arena.allocate(16 * 8L);
        final MemorySegment offs = arena.allocate(16 * 8L);
        final MemorySegment params = arena.allocate(256);
        final Map<Long, ArrayDeque<Metal.Buffer>> pool = new HashMap<>();
        /** kernel name -> {nanoseconds, calls}, with -Djllm.diffusion.profile=true */
        final Map<String, long[]> profile = new java.util.TreeMap<>();
        boolean recording;
        int pending;
        long poolBytes;
        long liveBytes;
        long peakBytes;

        Gpu() {
            // -Djllm.diffusion.defines="#define DF_PAD_BYTES 16\n": shader variants for experiments (see the macros at the top of diffusion.metal)
            this.metal = Metal.create(null, System.getProperty("jllm.diffusion.defines", "").replace("\\n", "\n"), "diffusion.metal");
        }

        @Override
        public void close() {
            metal.close();
            arena.close();
        }
    }

    /** A float16 / float32 buffer with a shape. */
    static final class MT extends Tn {
        final Metal.Buffer buf;
        final boolean half;
        final boolean weight;
        final long bytes;

        MT(Metal.Buffer buf, boolean half, boolean weight, long bytes, int... shape) {
            super(shape);
            this.buf = buf;
            this.half = half;
            this.weight = weight;
            this.bytes = bytes;
        }
    }

    static boolean available() {
        return Metal.available();
    }

    /** Opens the GPU for a pipeline: {@code precision} auto / f16 / f32 for the UNet and text encoders. */
    static Devices open(String precision, boolean xl) {
        String p = precision == null ? "auto" : precision.toLowerCase(java.util.Locale.ROOT);
        boolean half = !p.equals("f32");
        Gpu gpu = new Gpu();
        MetalOps main = new MetalOps(gpu, half);
        MetalOps vae = xl || !half ? new MetalOps(gpu, false) : main;
        String desc = "Metal " + gpu.metal.deviceName() + " " + (half ? "float16" : "float32")
                + (vae != main ? ", VAE float32" : "");
        return new Devices(main, vae, desc, () -> {
            if (vae != main) {
                vae.releaseAll();
            }
            main.releaseAll();
            gpu.close();
        });
    }

    private final Gpu gpu;
    private final boolean half;
    private final int es;
    /** Columns of the output tile of the matrix kernels: 64 for half, 32 for float (DF_BNT in diffusion.metal). */
    private final int tn;
    private final String sfx;
    private final java.util.List<MT> weights = new java.util.ArrayList<>();

    MetalOps(Gpu gpu, boolean half) {
        this.gpu = gpu;
        this.half = half;
        this.es = half ? 2 : 4;
        this.tn = half ? 64 : 32;
        this.sfx = half ? "_h" : "_f";
    }

    @Override
    public String name() {
        return "metal";
    }

    @Override
    public boolean half() {
        return half;
    }

    // ------------------------------------------------------------------ memory

    private Metal.Buffer take(long bytes) {
        long size = Math.max(16, bytes);
        ArrayDeque<Metal.Buffer> q = gpu.pool.get(size);
        if (q != null && !q.isEmpty()) {
            gpu.poolBytes -= size;
            return q.pop();
        }
        Metal.Buffer b;
        try {
            b = gpu.metal.newBuffer(size, false);
        } catch (OutOfMemoryError e) { // give the pooled scratch memory back and try once more
            trim(0);
            b = gpu.metal.newBuffer(size, false);
        }
        gpu.liveBytes += size;
        gpu.peakBytes = Math.max(gpu.peakBytes, gpu.liveBytes);
        return b;
    }

    @Override
    public Tn create(int... shape) {
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        long bytes = n * es;
        return new MT(take(bytes), half, false, Math.max(16, bytes), shape);
    }

    @Override
    public Tn upload(float[] data, int... shape) {
        sync(); // the buffer may still be read by recorded work
        MT t = (MT) create(shape);
        write(t.buf, data, half);
        return t;
    }

    private static void write(Metal.Buffer b, float[] data, boolean half) {
        MemorySegment c = b.contents();
        if (half) {
            short[] s = new short[data.length];
            for (int i = 0; i < s.length; i++) {
                s[i] = Float.floatToFloat16(data[i]);
            }
            MemorySegment.copy(s, 0, c, ValueLayout.JAVA_SHORT_UNALIGNED, 0, s.length);
        } else {
            MemorySegment.copy(data, 0, c, ValueLayout.JAVA_FLOAT_UNALIGNED, 0, data.length);
        }
    }

    @Override
    public float[] download(Tn t) {
        sync();
        MT m = (MT) t;
        int n = (int) m.numel();
        float[] out = new float[n];
        MemorySegment c = m.buf.contents();
        if (m.half) {
            short[] s = new short[n];
            MemorySegment.copy(c, ValueLayout.JAVA_SHORT_UNALIGNED, 0, s, 0, n);
            for (int i = 0; i < n; i++) {
                out[i] = Float.float16ToFloat(s[i]);
            }
        } else {
            MemorySegment.copy(c, ValueLayout.JAVA_FLOAT_UNALIGNED, 0, out, 0, n);
        }
        return out;
    }

    @Override
    public void free(Tn t) {
        MT m = (MT) t;
        if (m.weight) {
            weights.remove(m);
            gpu.metal.release(m.buf);
            gpu.liveBytes -= m.bytes;
        } else {
            gpu.pool.computeIfAbsent(m.bytes, k -> new ArrayDeque<>()).push(m.buf);
            gpu.poolBytes += m.bytes;
        }
    }

    @Override
    public void trim(long keepBytes) {
        if (gpu.poolBytes <= keepBytes) {
            return;
        }
        sync();
        for (ArrayDeque<Metal.Buffer> q : gpu.pool.values()) {
            for (Metal.Buffer b : q) {
                gpu.metal.release(b);
                gpu.liveBytes -= b.size();
            }
        }
        gpu.pool.clear();
        gpu.poolBytes = 0;
    }

    void releaseAll() {
        sync();
        for (MT w : new java.util.ArrayList<>(weights)) {
            free(w);
        }
    }

    /** Bytes in buffers created through this device (activations, pooled, and weights). */
    long liveBytes() {
        return gpu.liveBytes;
    }

    long peakBytes() {
        return gpu.peakBytes;
    }

    @Override
    public void sync() {
        if (gpu.recording) {
            gpu.metal.commitAndWait();
            gpu.recording = false;
            gpu.pending = 0;
        }
    }

    @Override
    public void close() {
        releaseAll();
    }

    @Override
    public Tn weight(SafeTensors.Tensor st, Kind kind) {
        int[] shape = new int[st.shape().length];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = (int) st.shape()[i];
        }
        boolean h = half && kind != Kind.VECTOR;
        int wes = h ? 2 : 4;
        long n = st.numel();
        long bytes = Math.max(16, n * wes);
        Metal.Buffer buf = gpu.metal.newBuffer(bytes, false);
        gpu.liveBytes += bytes;
        gpu.peakBytes = Math.max(gpu.peakBytes, gpu.liveBytes);
        MemorySegment c = buf.contents();
        int[] outShape = shape;
        if (kind == Kind.CONV) {
            int co = shape[0];
            int ci = shape[1];
            int kh = shape[2];
            int kw = shape[3];
            outShape = new int[]{co, kh, kw, ci};
            float[] src = st.toFloatArray();
            int k = ci * kh * kw;
            IntStream.range(0, co).parallel().forEach(o -> {
                if (h) {
                    short[] row = new short[k];
                    for (int cc = 0; cc < ci; cc++) {
                        for (int y = 0; y < kh; y++) {
                            for (int x = 0; x < kw; x++) {
                                row[(y * kw + x) * ci + cc] = Float.floatToFloat16(src[((o * ci + cc) * kh + y) * kw + x]);
                            }
                        }
                    }
                    MemorySegment.copy(row, 0, c, ValueLayout.JAVA_SHORT_UNALIGNED, (long) o * k * 2, k);
                } else {
                    float[] row = new float[k];
                    for (int cc = 0; cc < ci; cc++) {
                        for (int y = 0; y < kh; y++) {
                            for (int x = 0; x < kw; x++) {
                                row[(y * kw + x) * ci + cc] = src[((o * ci + cc) * kh + y) * kw + x];
                            }
                        }
                    }
                    MemorySegment.copy(row, 0, c, ValueLayout.JAVA_FLOAT_UNALIGNED, (long) o * k * 4, k);
                }
            });
        } else if (kind == Kind.LINEAR) {
            int rows = shape[0];
            int k = (int) (n / rows);
            outShape = new int[]{rows, k}; // a 1x1 convolution weight [O, I, 1, 1] is a linear weight
            if (h && st.dtype() == SafeTensors.DType.F16 && st.reader() == null) {
                MemorySegment.copy(st.data(), 0, c, 0, n * 2);
            } else {
                final int chunk = 1 << 22;
                float[] tmp = new float[(int) Math.min(n, chunk)];
                for (long off = 0; off < n; off += chunk) {
                    int len = (int) Math.min(chunk, n - off);
                    st.toFloat(off, tmp, 0, len);
                    if (h) {
                        short[] s = new short[len];
                        for (int i = 0; i < len; i++) {
                            s[i] = Float.floatToFloat16(tmp[i]);
                        }
                        MemorySegment.copy(s, 0, c, ValueLayout.JAVA_SHORT_UNALIGNED, off * 2, len);
                    } else {
                        MemorySegment.copy(tmp, 0, c, ValueLayout.JAVA_FLOAT_UNALIGNED, off * 4, len);
                    }
                }
            }
        } else {
            float[] src = st.toFloatArray();
            MemorySegment.copy(src, 0, c, ValueLayout.JAVA_FLOAT_UNALIGNED, 0, src.length);
        }
        MT t = new MT(buf, h, true, bytes, outShape);
        weights.add(t);
        return t;
    }

    // ------------------------------------------------------------------ dispatch

    /** Writes a parameter struct field by field against its {@link DiffusionParams} layout. */
    private static final class Pw {
        final MemorySegment seg;
        DiffusionParams layout;
        int field;
        int off;

        Pw(MemorySegment seg) {
            this.seg = seg;
        }

        Pw reset(DiffusionParams l) {
            layout = l;
            field = 0;
            off = 0;
            return this;
        }

        private void check(char kind) {
            if (field >= layout.kinds.length() || layout.kinds.charAt(field) != kind) {
                throw new IllegalStateException(layout.struct + ": field " + field + " written as " + kind + ", layout " + layout.kinds);
            }
            field++;
        }

        Pw i(int v) {
            check('i');
            seg.set(ValueLayout.JAVA_INT_UNALIGNED, off, v);
            off += 4;
            return this;
        }

        Pw f(float v) {
            check('f');
            seg.set(ValueLayout.JAVA_FLOAT_UNALIGNED, off, v);
            off += 4;
            return this;
        }

        Pw l(long v) {
            check('l');
            seg.set(ValueLayout.JAVA_LONG_UNALIGNED, off, v);
            off += 8;
            return this;
        }

        int done() {
            if (field != layout.kinds.length()) {
                throw new IllegalStateException(layout.struct + ": " + field + " of " + layout.kinds.length() + " fields written");
            }
            return layout.bytes();
        }
    }

    private Pw pw;

    private Pw p(DiffusionParams l) {
        if (pw == null) {
            pw = new Pw(gpu.params);
        }
        return pw.reset(l);
    }

    private void run(String kernel, int gx, int gy, int gz, int threads, int plen, Metal.Buffer... bufs) {
        if (!gpu.recording) {
            gpu.metal.begin();
            gpu.recording = true;
        }
        for (int i = 0; i < bufs.length; i++) {
            gpu.bufs.setAtIndex(ValueLayout.ADDRESS, i, bufs[i].handle());
            gpu.offs.setAtIndex(ValueLayout.JAVA_LONG, i, bufs[i].offset());
        }
        long t0 = PROFILE ? System.nanoTime() : 0;
        gpu.metal.dispatch(gpu.metal.pipeline(kernel), gpu.bufs, gpu.offs, bufs.length, gpu.params, plen, gx, gy, gz, threads, 1, 1);
        if (PROFILE) { // one command buffer per dispatch: the time of each kernel, summed per kernel name
            gpu.metal.commitAndWait();
            gpu.recording = false;
            gpu.pending = 0;
            long[] acc = gpu.profile.computeIfAbsent(kernel, k -> new long[2]);
            acc[0] += System.nanoTime() - t0;
            acc[1]++;
            return;
        }
        if (++gpu.pending >= SLICE) {
            gpu.metal.commit();
            gpu.pending = 0;
        }
    }

    private static int cdiv(long a, long b) {
        return (int) ((a + b - 1) / b);
    }

    private static Metal.Buffer buf(Tn t) {
        return ((MT) t).buf;
    }

    // ------------------------------------------------------------------ operations

    @Override
    public Tn linear(Tn x, Tn w, Tn bias, Tn residual, Act act) {
        int rows = x.rows();
        int k = x.last();
        int n = w.dim(0);
        if (w.dim(1) != k) {
            throw new IllegalArgumentException("linear: input " + x + " does not fit weight " + w);
        }
        int[] shape = x.shape.clone();
        shape[shape.length - 1] = n;
        Tn out = create(shape);
        gemm(x, w, out, bias, residual, act, rows, n, k);
        return out;
    }

    private void gemm(Tn x, Tn w, Tn out, Tn bias, Tn residual, Act act, int m, int n, int k) {
        int plen = p(DiffusionParams.GEMM).i(m).i(n).i(k).i(k).i(k).i(n).i(1).i(0).i(0).i(0).i(0).i(0).i(0).f(1f)
                .i(bias != null ? 1 : 0).i(residual != null ? 1 : 0).i(act.ordinal()).i(k % 8 == 0 && n % 8 == 0 ? 1 : 0).done();
        Metal.Buffer o = buf(out);
        run("df_gemm_nt" + sfx, cdiv(n, tn), cdiv(m, 64), 1, 128, plen, buf(x), buf(w), o,
                bias != null ? buf(bias) : o, residual != null ? buf(residual) : o);
    }

    @Override
    public Tn conv2d(Tn x, Tn w, Tn bias, int stride, int padT, int padL, int padB, int padR, Tn chanBias, Tn residual) {
        int b = x.dim(0);
        int h = x.dim(1);
        int wd = x.dim(2);
        int ci = x.dim(3);
        int co = w.dim(0);
        int kh = w.dim(1);
        int kw = w.dim(2);
        if (w.dim(3) != ci) {
            throw new IllegalArgumentException("conv2d: input " + x + " does not fit weight " + w);
        }
        int ho = (h + padT + padB - kh) / stride + 1;
        int wo = (wd + padL + padR - kw) / stride + 1;
        MT out = (MT) create(b, ho, wo, co);
        if (kh == 1 && kw == 1 && stride == 1 && padT == 0 && padL == 0 && padB == 0 && padR == 0 && chanBias == null
                && ci % 8 == 0) {
            gemm(x, w, out, bias, residual, Act.NONE, b * h * wd, co, ci);
            return out;
        }
        convKernel(x, w, out, bias, chanBias, residual, b, h, wd, ci, ho, wo, co, kh, kw, stride, padT, padL, 0);
        return out;
    }

    private void convKernel(Tn x, Tn w, MT out, Tn bias, Tn chanBias, Tn residual, int b, int h, int wd, int ci, int ho, int wo, int co,
                            int kh, int kw, int stride, int padT, int padL, int ups) {
        int plen = p(DiffusionParams.CONV).i(b).i(h).i(wd).i(ci).i(ho).i(wo).i(co).i(kh).i(kw).i(stride).i(padT).i(padL)
                .i(bias != null ? 1 : 0).i(chanBias != null ? 1 : 0).i(residual != null ? 1 : 0).i(ups).done();
        Metal.Buffer o = out.buf;
        Metal.Buffer bb = bias != null ? buf(bias) : o;
        Metal.Buffer cb = chanBias != null ? buf(chanBias) : o;
        Metal.Buffer rb = residual != null ? buf(residual) : o;
        if (ci % 8 == 0 && co >= 16) {
            run("df_conv" + sfx, cdiv(co, tn), cdiv((long) b * ho * wo, 64), 1, 128, plen, buf(x), buf(w), o, bb, cb, rb);
        } else {
            long total = (long) b * ho * wo * co;
            run("df_conv_direct" + sfx, cdiv(total, 256), 1, 1, 256, plen, buf(x), buf(w), o, bb, cb, rb);
        }
    }

    @Override
    public Tn upsampleConv(Tn x, Tn w, Tn bias) {
        int b = x.dim(0);
        int h = x.dim(1) * 2;
        int wd = x.dim(2) * 2;
        int ci = x.dim(3);
        int co = w.dim(0);
        MT out = (MT) create(b, h, wd, co);
        convKernel(x, w, out, bias, null, null, b, h, wd, ci, h, wd, co, 3, 3, 1, 1, 1, 1);
        return out;
    }

    @Override
    public Tn groupNorm(Tn x, Tn gamma, Tn beta, int groups, float eps, boolean silu) {
        int b = x.dim(0);
        int c = x.last();
        Tn out = create(x.shape);
        int hw = x.rows() / b;
        int gs = c / groups;
        if ((long) hw * c <= 65536 || c % 4 != 0 || SIMPLE_NORM) { // small: one threadgroup per group does it all
            int plen = p(DiffusionParams.NORM).i(c).i(hw).i(groups).f(eps).i(silu ? 1 : 0).i(0).done();
            run("df_groupnorm" + sfx, groups, b, 1, 256, plen, buf(x), buf(gamma), buf(beta), buf(out));
            return out;
        }
        // large: statistics per chunk of pixels (many threadgroups), merged, then applied elementwise
        int chunkPix = Math.max(Math.max(1, 16384 / gs), (hw + 127) / 128);
        int chunks = (hw + chunkPix - 1) / chunkPix;
        MT part = (MT) createFloats((long) b * groups * chunks * 2);
        MT stats = (MT) createFloats((long) b * groups * 2);
        int plen = p(DiffusionParams.GN).i(c).i(hw).i(groups).f(eps).i(silu ? 1 : 0).i(chunkPix).i(chunks).i(b).done();
        run("df_gn_stats" + sfx, chunks, groups, b, 256, plen, buf(x), part.buf);
        run("df_gn_combine", groups, b, 1, 1, plen, part.buf, stats.buf);
        run("df_gn_apply" + sfx, cdiv((long) b * hw * c / 4, 256), 1, 1, 256, plen, buf(x), stats.buf, buf(gamma), buf(beta), buf(out));
        free(part);
        free(stats);
        return out;
    }

    @Override
    public Tn layerNorm(Tn x, Tn gamma, Tn beta, float eps) {
        int c = x.last();
        Tn out = create(x.shape);
        int plen = p(DiffusionParams.NORM).i(c).i(x.rows()).i(1).f(eps).i(0).i(gamma != null ? 1 : 0).done();
        Metal.Buffer o = buf(out);
        run("df_layernorm" + sfx, cdiv(x.rows(), 8), 1, 1, 256, plen, buf(x), gamma != null ? buf(gamma) : o,
                beta != null ? buf(beta) : o, o);
        return out;
    }

    @Override
    public Tn attention(Tn q, Tn k, Tn v, int heads, boolean causal) {
        int batch = q.dim(0);
        int lq = q.rows() / batch;
        int lk = k.rows() / batch;
        int dm = q.last();
        int hd = dm / heads;
        Tn out = create(q.shape);
        float scale = (float) (1.0 / Math.sqrt(hd));
        if (FLASH && flashWidth(hd) && dm % 4 == 0) {
            int plen = p(DiffusionParams.FLASH).i(heads).i(lq).i(lk).i(dm).i(dm).i(dm).i(dm).f(scale).i(causal ? 1 : 0).done();
            run("df_flash_d" + hd + sfx, heads, cdiv(lq, 32), batch, 128, plen, buf(q), buf(k), buf(v), buf(out));
            return out;
        }
        naiveAttention(q, k, v, out, batch, heads, lq, lk, hd, scale, causal);
        return out;
    }

    private static boolean flashWidth(int d) {
        return switch (d) {
            case 8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160 -> true;
            default -> false;
        };
    }

    /** Scores, softmax and the product with the values as three kernels over a float scores buffer, a few heads at a time. */
    private void naiveAttention(Tn q, Tn k, Tn v, Tn out, int batch, int heads, int lq, int lk, int hd, float scale, boolean causal) {
        int dm = heads * hd;
        int ld = (lk + 7) / 8 * 8;
        long maxScores = 64L << 20; // floats per chunk: 256 MB
        int perHead = (int) Math.max(1, Math.min(heads, maxScores / ((long) lq * ld)));
        MT scores = (MT) createFloats((long) perHead * lq * ld);
        boolean vec = dm % 8 == 0 && hd % 8 == 0 && ld % 8 == 0;
        MT qm = (MT) q;
        MT km = (MT) k;
        MT vm = (MT) v;
        MT om = (MT) out;
        for (int bi = 0; bi < batch; bi++) {
            for (int h0 = 0; h0 < heads; h0 += perHead) {
                int nh = Math.min(perHead, heads - h0);
                long qOff = ((long) bi * lq * dm + (long) h0 * hd) * es;
                long kOff = ((long) bi * lk * dm + (long) h0 * hd) * es;
                int plen = p(DiffusionParams.GEMM).i(lq).i(lk).i(hd).i(dm).i(dm).i(ld).i(nh).i(0).i(hd).i(0).i(hd).i(0).i(lq * ld)
                        .f(scale).i(0).i(0).i(0).i(vec ? 1 : 0).done();
                // the scores product is df_scores_{f,h}; offsets into q / k / v / out are applied as buffer offsets
                run(half ? "df_scores_h" : "df_gemm_nt_f", cdiv(lk, tn), cdiv(lq, 64), nh, 128, plen, view(qm, qOff), view(km, kOff), scores.buf,
                        scores.buf, scores.buf);
                plen = p(DiffusionParams.SOFTMAX).i(lk).i(ld).i(lq).i(causal ? 1 : 0).done();
                run("df_softmax", nh * lq, 1, 1, 256, plen, scores.buf);
                plen = p(DiffusionParams.GEMM).i(lq).i(hd).i(lk).i(ld).i(dm).i(dm).i(nh).i(0).i(lq * ld).i(0).i(hd).i(0).i(hd)
                        .f(1f).i(0).i(0).i(0).i(vec ? 1 : 0).done();
                run("df_pv" + sfx, cdiv(hd, tn), cdiv(lq, 64), nh, 128, plen, scores.buf, view(vm, kOff), view(om, qOff),
                        om.buf, om.buf);
            }
        }
        free(scores);
    }

    private static Metal.Buffer view(MT t, long byteOffset) {
        Metal.Buffer b = t.buf;
        return new Metal.Buffer(b.handle(), b.contents(), b.size(), b.offset() + byteOffset);
    }

    /** A float32 scratch buffer regardless of this device's activation type. */
    private Tn createFloats(long n) {
        long bytes = n * 4;
        return new MT(take(bytes), false, false, Math.max(16, bytes), (int) Math.min(n, Integer.MAX_VALUE));
    }

    @Override
    public Tn geglu(Tn x) {
        int c2 = x.last();
        int n = c2 / 2;
        int[] shape = x.shape.clone();
        shape[shape.length - 1] = n;
        Tn out = create(shape);
        long total = out.numel();
        int plen = p(DiffusionParams.EW).l(total).i(n).i(0).i(0).i(0).done();
        run("df_geglu" + sfx, cdiv(total, 256), 1, 1, 256, plen, buf(x), buf(out));
        return out;
    }

    @Override
    public Tn add(Tn a, Tn b) {
        Tn out = create(a.shape);
        long total = out.numel();
        int plen = p(DiffusionParams.EW).l(total).i(0).i(0).i(0).i(0).done();
        run("df_add" + sfx, cdiv(total, 256), 1, 1, 256, plen, buf(a), buf(b), buf(out));
        return out;
    }

    @Override
    public Tn silu(Tn x) {
        Tn out = create(x.shape);
        long total = out.numel();
        int plen = p(DiffusionParams.EW).l(total).i(0).i(0).i(0).i(0).done();
        run("df_silu" + sfx, cdiv(total, 256), 1, 1, 256, plen, buf(x), buf(out));
        return out;
    }

    @Override
    public Tn upsample2x(Tn x) {
        int b = x.dim(0);
        int h = x.dim(1);
        int w = x.dim(2);
        int c = x.dim(3);
        Tn out = create(b, h * 2, w * 2, c);
        long total = out.numel();
        int plen = p(DiffusionParams.EW).l(total).i(h).i(w).i(c).i(0).done();
        run("df_upsample2x" + sfx, cdiv(total, 256), 1, 1, 256, plen, buf(x), buf(out));
        return out;
    }

    @Override
    public Tn concat(Tn a, Tn b) {
        int ca = a.last();
        int cb = b.last();
        int[] shape = a.shape.clone();
        shape[shape.length - 1] = ca + cb;
        Tn out = create(shape);
        long total = out.numel();
        int plen = p(DiffusionParams.EW).l(total).i(ca).i(cb).i(0).i(0).done();
        run("df_concat" + sfx, cdiv(total, 256), 1, 1, 256, plen, buf(a), buf(b), buf(out));
        return out;
    }
}
