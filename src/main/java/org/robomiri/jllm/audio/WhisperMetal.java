package org.robomiri.jllm.audio;

import org.robomiri.jllm.audio.Dense.Mat;
import org.robomiri.jllm.audio.WhisperWeights.Layer;
import org.robomiri.jllm.metal.Metal;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Whisper on the GPU: the encoder (two convolutions as products over patches, 12-32 transformer layers) and the
 * cross-attention keys and values of every decoder layer run as one command buffer of {@code audio.metal} kernels; the
 * decoder, a few tokens against a short cache, runs on the CPU ({@link WhisperCpu}) on the keys and values the GPU wrote
 * (unified memory: they are copied out of the shared buffers, 110 MB for whisper-small).
 *
 * <p>Linear layers use half-precision weight tiles with float activations and accumulators (the checkpoint's f16
 * weights are used in place; f32 ones are converted once), attention products use float tiles; {@code exact} keeps the
 * checkpoint's precision everywhere.
 */
public final class WhisperMetal implements WhisperNet {

    private static final float LN_EPS = 1e-5f;

    /** A weight matrix on the GPU: buffer, element type, shape and bias (null for none). */
    private record Gm(Metal.Buffer w, boolean half, int rows, int cols, Metal.Buffer bias) {
    }

    private record GLayer(Gm q, Gm k, Gm v, Gm out, Gm fc1, Gm fc2, Metal.Buffer ln1g, Metal.Buffer ln1b, Metal.Buffer ln2g,
                          Metal.Buffer ln2b) {
    }

    private final WhisperWeights w;
    private final WhisperConfig c;
    private final GpuCtx g;
    private final WhisperCpu cpu;
    private final boolean exact;
    private final ReentrantLock lock = new ReentrantLock();
    private final List<Metal.Buffer> owned = new ArrayList<>();
    private final String device;

    private final Gm conv1;
    private final Gm conv2;
    private final Metal.Buffer encPos;
    private final GLayer[] enc;
    private final Metal.Buffer encLnG;
    private final Metal.Buffer encLnB;
    private final Gm[] crossK;
    private final Gm[] crossV;

    // activations, sized for one window
    private final Metal.Buffer melBuf;
    private final Metal.Buffer col1;
    private final Metal.Buffer h1;
    private final Metal.Buffer col2;
    private final Metal.Buffer x;
    private final Metal.Buffer xn;
    private final Metal.Buffer qb;
    private final Metal.Buffer kb;
    private final Metal.Buffer vb;
    private final Metal.Buffer qh;
    private final Metal.Buffer kh;
    private final Metal.Buffer vt;
    private final Metal.Buffer scores;
    private final Metal.Buffer ctx;
    private final Metal.Buffer hid;
    private final Metal.Buffer encOut;
    private final Metal.Buffer[] crossKh;
    private final Metal.Buffer[] crossVt;

    WhisperMetal(WhisperWeights w, GpuCtx g, int threads, boolean exact) {
        this.w = w;
        this.c = w.cfg;
        this.g = g;
        this.exact = exact;
        this.cpu = new WhisperCpu(w, threads);
        this.device = "metal (" + g.metal.deviceName() + (exact ? ", exact" : "") + ")";
        int d = c.dModel();
        int frames = c.windowFrames();
        int t2 = c.sourcePos();
        int heads = c.encHeads();
        conv1 = up(w.conv1);
        conv2 = up(w.conv2);
        encPos = own(g.floats(w.encPos));
        enc = new GLayer[c.encLayers()];
        for (int i = 0; i < enc.length; i++) {
            Layer l = w.enc[i];
            enc[i] = new GLayer(up(l.self.q), up(l.self.k), up(l.self.v), up(l.self.out), up(l.fc1), up(l.fc2),
                    own(g.floats(l.self.lnGamma)), own(g.floats(l.self.lnBeta)), own(g.floats(l.ffnLnGamma)),
                    own(g.floats(l.ffnLnBeta)));
        }
        encLnG = own(g.floats(w.encLnGamma));
        encLnB = own(g.floats(w.encLnBeta));
        crossK = new Gm[c.decLayers()];
        crossV = new Gm[c.decLayers()];
        crossKh = new Metal.Buffer[c.decLayers()];
        crossVt = new Metal.Buffer[c.decLayers()];
        for (int i = 0; i < c.decLayers(); i++) {
            crossK[i] = up(w.dec[i].cross.k);
            crossV[i] = up(w.dec[i].cross.v);
            crossKh[i] = own(g.buffer(4L * t2 * d));
            crossVt[i] = own(g.buffer(4L * t2 * d));
        }
        melBuf = own(g.buffer(4L * c.nMels() * frames));
        col1 = own(g.buffer(4L * frames * c.nMels() * 3));
        h1 = own(g.buffer(4L * frames * d));
        col2 = own(g.buffer(4L * t2 * d * 3));
        x = own(g.buffer(4L * t2 * d));
        xn = own(g.buffer(4L * t2 * d));
        qb = own(g.buffer(4L * t2 * d));
        kb = own(g.buffer(4L * t2 * d));
        vb = own(g.buffer(4L * t2 * d));
        qh = own(g.buffer(4L * t2 * d));
        kh = own(g.buffer(4L * t2 * d));
        vt = own(g.buffer(4L * t2 * d));
        scores = own(g.buffer(4L * heads * t2 * t2));
        ctx = own(g.buffer(4L * t2 * d));
        hid = own(g.buffer(4L * t2 * Math.max(c.encFfn(), 1)));
        encOut = own(g.buffer(4L * t2 * d));
    }

    private Metal.Buffer own(Metal.Buffer b) {
        owned.add(b);
        return b;
    }

    /** Uploads (or maps in place) a weight matrix. */
    private Gm up(Mat m) {
        SafeTensors.Tensor t = m.tensor();
        Metal.Buffer bias = m.bias == null ? null : own(g.floats(m.bias));
        if (t == null) {
            float[] f = m.floats();
            return new Gm(toHalfBuffer(f), true, m.rows, m.cols, bias);
        }
        boolean f16 = t.dtype() == SafeTensors.DType.F16;
        boolean f32 = t.dtype() == SafeTensors.DType.F32;
        if (f16 && t.reader() == null) {
            return new Gm(own(g.metal.wrapRange(t.data())), true, m.rows, m.cols, bias);
        }
        if (f32 && exact && t.reader() == null) {
            return new Gm(own(g.metal.wrapRange(t.data())), false, m.rows, m.cols, bias);
        }
        long n = (long) m.rows * m.cols;
        boolean half = !exact || !f32;
        Metal.Buffer b = own(g.buffer(n * (half ? 2 : 4)));
        MemorySegment seg = b.contents().asSlice(b.offset(), n * (half ? 2 : 4));
        float[] chunk = new float[1 << 16];
        for (long i = 0; i < n; i += chunk.length) {
            int len = (int) Math.min(chunk.length, n - i);
            t.toFloat(i, chunk, 0, len);
            if (half) {
                for (int j = 0; j < len; j++) {
                    seg.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), 2 * (i + j),
                            Float.floatToFloat16(chunk[j]));
                }
            } else {
                MemorySegment.copy(chunk, 0, seg, ValueLayout.JAVA_FLOAT_UNALIGNED, 4 * i, len);
            }
        }
        return new Gm(b, half, m.rows, m.cols, bias);
    }

    private Metal.Buffer toHalfBuffer(float[] f) {
        Metal.Buffer b = own(g.buffer(2L * f.length));
        MemorySegment seg = b.contents().asSlice(b.offset(), 2L * f.length);
        for (int j = 0; j < f.length; j++) {
            seg.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), 2L * j, Float.floatToFloat16(f[j]));
        }
        return b;
    }

    @Override
    public WhisperConfig config() {
        return c;
    }

    @Override
    public String name() {
        return device;
    }

    // ------------------------------------------------------------------ kernels

    private static final int GELU = 2;
    private static final int BIAS = 1;
    private static final int ACC = 4;

    /** out[t][r] (+)= act(bias + x[t] . w[r]) for a weight matrix on the GPU. */
    private void linear(Gm m, Metal.Buffer in, Metal.Buffer out, int ntok, int flags) {
        String kernel = m.half() ? (exact ? "wh_gemm_hf" : "wh_gemm_hh") : (exact ? "wh_gemm_ff" : "wh_gemm_fh");
        int fl = flags | (m.bias() != null ? BIAS : 0);
        g.run(kernel, GpuCtx.ceilDiv(m.rows(), 64), GpuCtx.ceilDiv(ntok, 32), 1, 128, 1, 1,
                new Object[]{m.rows(), m.cols(), ntok, m.cols(), m.cols(), m.rows(), fl, 0, 0, 0},
                m.w(), in, out, m.bias() != null ? m.bias() : in);
    }

    private void layerNorm(Metal.Buffer in, Metal.Buffer out, Metal.Buffer gamma, Metal.Buffer beta, int n, int d) {
        g.run("wh_layernorm", n, 1, 1, 256, 1, 1, new Object[]{d, LN_EPS, 1}, in, out, gamma, beta);
    }

    /** The batched attention of {@code n} query rows over {@code len} keys: S = Q K^T, softmax, ctx = P V. */
    private void attention(int n, int len, int heads, int hd, int cap, boolean causal, int pos0) {
        int d = heads * hd;
        g.run("wh_gemm_ff", GpuCtx.ceilDiv(len, 64), GpuCtx.ceilDiv(n, 32), heads, 128, 1, 1,
                new Object[]{len, hd, n, hd, hd, len, 0, cap * hd, n * hd, n * len}, kh, qh, scores, qh);
        g.run("wh_softmax", n, heads, 1, 256, 1, 1, new Object[]{n, len, causal ? 1 : 0, pos0}, scores);
        g.run("wh_gemm_ff", GpuCtx.ceilDiv(hd, 64), GpuCtx.ceilDiv(n, 32), heads, 128, 1, 1,
                new Object[]{hd, len, n, cap, len, d, 0, hd * cap, n * len, hd}, vt, scores, ctx, scores);
    }

    // ------------------------------------------------------------------ encoder

    @Override
    public Session encode(float[][] mel) {
        lock.lock();
        try {
            return encodeLocked(mel);
        } finally {
            lock.unlock();
        }
    }

    private Session encodeLocked(float[][] mel) {
        int d = c.dModel();
        int nm = c.nMels();
        int frames = c.windowFrames();
        int t2 = c.sourcePos();
        int heads = c.encHeads();
        int hd = c.encHeadDim();
        if (mel.length != nm || mel[0].length != frames) {
            throw new IllegalArgumentException("mel must be " + nm + " x " + frames);
        }
        for (int i = 0; i < nm; i++) {
            GpuCtx.put(melBuf, (long) i * frames, mel[i], frames);
        }
        GpuCtx.put(x, 0, w.encPos, t2 * d);          // the positions: the second convolution's output is added to them
        g.begin();
        try {
            g.run("wh_im2col_mel", GpuCtx.ceilDiv(frames * nm * 3, 256), 1, 1, 256, 1, 1, new Object[]{frames, nm, frames},
                    melBuf, col1);
            linear(conv1, col1, h1, frames, GELU);
            g.run("wh_im2col_s2", GpuCtx.ceilDiv(t2 * d * 3, 256), 1, 1, 256, 1, 1, new Object[]{t2, d, frames}, h1, col2);
            linear(conv2, col2, x, t2, GELU | ACC);
            float scale = (float) (1.0 / Math.sqrt(hd));
            for (GLayer l : enc) {
                layerNorm(x, xn, l.ln1g(), l.ln1b(), t2, d);
                linear(l.q(), xn, qb, t2, 0);
                linear(l.k(), xn, kb, t2, 0);
                linear(l.v(), xn, vb, t2, 0);
                g.run("wh_split_heads", GpuCtx.ceilDiv(t2 * d, 256), 1, 1, 256, 1, 1,
                        new Object[]{t2, heads, hd, t2, 0, scale, 1, 1}, qb, kb, vb, qh, kh, vt);
                attention(t2, t2, heads, hd, t2, false, 0);
                linear(l.out(), ctx, x, t2, ACC);
                layerNorm(x, xn, l.ln2g(), l.ln2b(), t2, d);
                linear(l.fc1(), xn, hid, t2, GELU);
                linear(l.fc2(), hid, x, t2, ACC);
            }
            layerNorm(x, encOut, encLnG, encLnB, t2, d);
            int dh = c.decHeads();
            int dhd = c.decHeadDim();
            for (int i = 0; i < c.decLayers(); i++) {          // cross-attention keys and values of every decoder layer
                linear(crossK[i], encOut, kb, t2, 0);
                linear(crossV[i], encOut, vb, t2, 0);
                g.run("wh_split_heads", GpuCtx.ceilDiv(t2 * d, 256), 1, 1, 256, 1, 1,
                        new Object[]{t2, dh, dhd, t2, 0, 1.0f, 0, 1}, qb, kb, vb, qh, crossKh[i], crossVt[i]);
            }
        } catch (RuntimeException e) {
            g.metal.abort();
            throw e;
        }
        g.commitAndWait();
        float[] encArr = new float[t2 * d];
        GpuCtx.get(encOut, 0, encArr, t2 * d);
        float[][] ck = new float[c.decLayers()][t2 * d];
        float[][] cv = new float[c.decLayers()][t2 * d];
        for (int i = 0; i < c.decLayers(); i++) {
            GpuCtx.get(crossKh[i], 0, ck[i], t2 * d);
            GpuCtx.get(crossVt[i], 0, cv[i], t2 * d);
        }
        return cpu.sessionOf(encArr, ck, cv);
    }

    @Override
    public void close() {
        cpu.close();
        for (Metal.Buffer b : owned) {
            g.metal.release(b);
        }
        owned.clear();
        g.close();
    }
}
