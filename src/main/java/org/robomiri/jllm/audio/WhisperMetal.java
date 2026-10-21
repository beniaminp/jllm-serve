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
 * Whisper on the GPU. The encoder (two convolutions as products over patches, 4-32 transformer layers) and the
 * cross-attention keys and values of every decoder layer run as one command buffer of {@code audio.metal} kernels. The
 * decoder runs there too when the model is wide enough to pay ({@code audio.metal.decoder}: auto = d_model of 768 or more),
 * one command buffer per step: embedding lookup, then per layer the layer norms, projections, attention against the
 * session's KV caches (the batched query-key product, a softmax and the value product) and the feed-forward block, and
 * the tied output projection of the last token. Smaller models decode on the CPU ({@link WhisperCpu}) from the keys and
 * values the GPU wrote (copied out of the shared buffers).
 *
 * <p>Linear layers use half-precision weight tiles with float activations and accumulators (the checkpoint's f16 weights are
 * used in place; f32 ones are converted once), attention products use float tiles; {@code exact} keeps the checkpoint's
 * precision everywhere. All GPU work is serialized by one lock (the dispatch scratch is shared).
 */
public final class WhisperMetal implements WhisperNet {

    private static final float LN_EPS = 1e-5f;
    /** Tokens per decoder call (longer prompts are fed in pieces). */
    private static final int DEC_CHUNK = 64;
    /** The encoder is submitted every this many layers (without waiting), so no command buffer runs for seconds. */
    private static final int COMMIT_LAYERS = 3;
    private static final int GELU = 2;
    private static final int BIAS = 1;
    private static final int ACC = 4;

    /** A weight matrix on the GPU: buffer, element type, shape and bias (null for none). */
    private record Gm(Metal.Buffer w, boolean half, int rows, int cols, Metal.Buffer bias) {
    }

    private record GLayer(Gm q, Gm k, Gm v, Gm out, Gm fc1, Gm fc2, Metal.Buffer ln1g, Metal.Buffer ln1b, Metal.Buffer ln2g,
                          Metal.Buffer ln2b) {
    }

    /** A decoder layer: self-attention, cross-attention (query and output; its keys and values are made once per window), FFN. */
    private record DLayer(Gm sq, Gm sk, Gm sv, Gm so, Gm cq, Gm co, Gm fc1, Gm fc2, Metal.Buffer ln1g, Metal.Buffer ln1b,
                          Metal.Buffer ln2g, Metal.Buffer ln2b, Metal.Buffer ln3g, Metal.Buffer ln3b) {
    }

    private final WhisperWeights w;
    private final WhisperConfig c;
    private final GpuCtx g;
    private final WhisperCpu cpu;
    private final boolean exact;
    private final boolean gpuDecoder;
    private final ReentrantLock lock = new ReentrantLock();
    private final List<Metal.Buffer> owned = new ArrayList<>();
    private final String device;

    private final Gm conv1;
    private final Gm conv2;
    private final GLayer[] enc;
    private final Metal.Buffer encLnG;
    private final Metal.Buffer encLnB;
    private final Gm[] crossK;
    private final Gm[] crossV;

    // encoder activations, sized for one window
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
    // cross-attention keys / values written by the last encode, for the CPU decoder
    private final Metal.Buffer[] crossKh;
    private final Metal.Buffer[] crossVt;

    // GPU decoder
    private final DLayer[] dec;
    private final Gm embed;
    private final Metal.Buffer decPos;
    private final Metal.Buffer decLnG;
    private final Metal.Buffer decLnB;
    private final Metal.Buffer dx;
    private final Metal.Buffer dxn;
    private final Metal.Buffer dq;
    private final Metal.Buffer dk;
    private final Metal.Buffer dv;
    private final Metal.Buffer dqh;
    private final Metal.Buffer dctx;
    private final Metal.Buffer dhid;
    private final Metal.Buffer dscores;
    private final Metal.Buffer tokBuf;
    private final Metal.Buffer lastLn;
    private final Metal.Buffer logitsBuf;

    WhisperMetal(WhisperWeights w, GpuCtx g, int threads, boolean exact, boolean gpuDecoder) {
        this.w = w;
        this.c = w.cfg;
        this.g = g;
        this.exact = exact;
        this.gpuDecoder = gpuDecoder;
        this.cpu = new WhisperCpu(w, threads);
        this.device = "metal (" + g.metal.deviceName() + (exact ? ", exact" : "") + (gpuDecoder ? "" : ", decoder on the CPU") + ")";
        int d = c.dModel();
        int frames = c.windowFrames();
        int t2 = c.sourcePos();
        int heads = c.encHeads();
        conv1 = up(w.conv1);
        conv2 = up(w.conv2);
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
            if (!gpuDecoder) {
                crossKh[i] = own(g.buffer(4L * t2 * d));
                crossVt[i] = own(g.buffer(4L * t2 * d));
            }
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
        if (gpuDecoder) {
            dec = new DLayer[c.decLayers()];
            for (int i = 0; i < dec.length; i++) {
                Layer l = w.dec[i];
                dec[i] = new DLayer(up(l.self.q), up(l.self.k), up(l.self.v), up(l.self.out), up(l.cross.q), up(l.cross.out),
                        up(l.fc1), up(l.fc2), own(g.floats(l.self.lnGamma)), own(g.floats(l.self.lnBeta)),
                        own(g.floats(l.cross.lnGamma)), own(g.floats(l.cross.lnBeta)), own(g.floats(l.ffnLnGamma)),
                        own(g.floats(l.ffnLnBeta)));
            }
            embed = up(w.embed);
            decPos = own(g.floats(w.decPos));
            decLnG = own(g.floats(w.decLnGamma));
            decLnB = own(g.floats(w.decLnBeta));
            int dh = c.decHeads();
            dx = own(g.buffer(4L * DEC_CHUNK * d));
            dxn = own(g.buffer(4L * DEC_CHUNK * d));
            dq = own(g.buffer(4L * DEC_CHUNK * d));
            dk = own(g.buffer(4L * DEC_CHUNK * d));
            dv = own(g.buffer(4L * DEC_CHUNK * d));
            dqh = own(g.buffer(4L * DEC_CHUNK * d));
            dctx = own(g.buffer(4L * DEC_CHUNK * d));
            dhid = own(g.buffer(4L * DEC_CHUNK * c.decFfn()));
            dscores = own(g.buffer(4L * dh * DEC_CHUNK * Math.max(t2, c.targetPos())));
            tokBuf = own(g.buffer(4L * DEC_CHUNK));
            lastLn = own(g.buffer(4L * d));
            logitsBuf = own(g.buffer(4L * c.vocab()));
        } else {
            dec = null;
            embed = null;
            decPos = decLnG = decLnB = dx = dxn = dq = dk = dv = dqh = dctx = dhid = dscores = tokBuf = lastLn = logitsBuf = null;
        }
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
        ValueLayout.OfShort sh = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
        for (long i = 0; i < n; i += chunk.length) {
            int len = (int) Math.min(chunk.length, n - i);
            t.toFloat(i, chunk, 0, len);
            if (half) {
                for (int j = 0; j < len; j++) {
                    seg.set(sh, 2 * (i + j), Float.floatToFloat16(chunk[j]));
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
        ValueLayout.OfShort sh = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
        for (int j = 0; j < f.length; j++) {
            seg.set(sh, 2L * j, Float.floatToFloat16(f[j]));
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

    /**
     * The batched attention of {@code n} query rows (per-head layout {@code q[h][n][hd]}) over {@code len} keys
     * ({@code k[h][cap][hd]}, {@code v[h][hd][cap]}): S = Q K^T, a softmax (causal from position {@code pos0}), and
     * ctx[n][d] = P V.
     */
    private void attention(Metal.Buffer q, Metal.Buffer k, Metal.Buffer v, Metal.Buffer s, Metal.Buffer out, int n, int len,
                           int heads, int hd, int cap, boolean causal, int pos0) {
        int d = heads * hd;
        g.run("wh_gemm_ff", GpuCtx.ceilDiv(len, 64), GpuCtx.ceilDiv(n, 32), heads, 128, 1, 1,
                new Object[]{len, hd, n, hd, hd, len, 0, cap * hd, n * hd, n * len}, k, q, s, q);
        g.run("wh_softmax", n, heads, 1, 256, 1, 1, new Object[]{n, len, causal ? 1 : 0, pos0}, s);
        g.run("wh_gemm_ff", GpuCtx.ceilDiv(hd, 64), GpuCtx.ceilDiv(n, 32), heads, 128, 1, 1,
                new Object[]{hd, len, n, cap, len, d, 0, hd * cap, n * len, hd}, v, s, out, s);
    }

    // ------------------------------------------------------------------ encoder

    @Override
    public Session encode(float[][] mel) {
        lock.lock();
        try {
            for (int attempt = 0; ; attempt++) {
                try {
                    return encodeLocked(mel);
                } catch (IllegalStateException e) {
                    if (!transient_(e) || attempt >= 2) {
                        throw e;
                    }
                    pause(attempt);
                }
            }
        } finally {
            lock.unlock();
        }
    }

    /**
     * macOS kills a command buffer when the GPU is too busy to answer the display ("Impacting Interactivity") or has been
     * reset; the work is idempotent (every pass starts from uploaded inputs), so a pass is tried again a little later.
     */
    private static boolean transient_(IllegalStateException e) {
        String m = String.valueOf(e.getMessage());
        return m.contains("command buffer failed");
    }

    private static void pause(int attempt) {
        System.err.println("whisper: the GPU refused a command buffer, trying again");
        try {
            Thread.sleep(250L * (attempt + 1));
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
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
        // a GPU decoder session owns the cross-attention buffers (the next encode must not overwrite them)
        Metal.Buffer[] ck = gpuDecoder ? new Metal.Buffer[c.decLayers()] : crossKh;
        Metal.Buffer[] cv = gpuDecoder ? new Metal.Buffer[c.decLayers()] : crossVt;
        if (gpuDecoder) {
            for (int i = 0; i < ck.length; i++) {
                ck[i] = g.buffer(4L * t2 * d);
                cv[i] = g.buffer(4L * t2 * d);
            }
        }
        g.begin();
        try {
            g.run("wh_im2col_mel", GpuCtx.ceilDiv(frames * nm * 3, 256), 1, 1, 256, 1, 1, new Object[]{frames, nm, frames},
                    melBuf, col1);
            linear(conv1, col1, h1, frames, GELU);
            g.run("wh_im2col_s2", GpuCtx.ceilDiv(t2 * d * 3, 256), 1, 1, 256, 1, 1, new Object[]{t2, d, frames}, h1, col2);
            linear(conv2, col2, x, t2, GELU | ACC);
            float scale = (float) (1.0 / Math.sqrt(hd));
            int layerNo = 0;
            for (GLayer l : enc) {
                if (layerNo++ % COMMIT_LAYERS == COMMIT_LAYERS - 1) {
                    g.metal.commit();      // a command buffer that holds the GPU for seconds is killed by the system's watchdog
                }
                layerNorm(x, xn, l.ln1g(), l.ln1b(), t2, d);
                linear(l.q(), xn, qb, t2, 0);
                linear(l.k(), xn, kb, t2, 0);
                linear(l.v(), xn, vb, t2, 0);
                g.run("wh_split_heads", GpuCtx.ceilDiv(t2 * d, 256), 1, 1, 256, 1, 1,
                        new Object[]{t2, heads, hd, t2, 0, scale, 1, 1}, qb, kb, vb, qh, kh, vt);
                attention(qh, kh, vt, scores, ctx, t2, t2, heads, hd, t2, false, 0);
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
                        new Object[]{t2, dh, dhd, t2, 0, 1.0f, 0, 1}, qb, kb, vb, qh, ck[i], cv[i]);
            }
        } catch (RuntimeException e) {
            g.metal.abort();
            if (gpuDecoder) {
                releaseAll(ck, cv);
            }
            throw e;
        }
        g.commitAndWait();
        float[] encArr = new float[t2 * d];
        GpuCtx.get(encOut, 0, encArr, t2 * d);
        if (gpuDecoder) {
            return new GpuSession(encArr, ck, cv);
        }
        float[][] ckArr = new float[c.decLayers()][t2 * d];
        float[][] cvArr = new float[c.decLayers()][t2 * d];
        for (int i = 0; i < c.decLayers(); i++) {
            GpuCtx.get(crossKh[i], 0, ckArr[i], t2 * d);
            GpuCtx.get(crossVt[i], 0, cvArr[i], t2 * d);
        }
        return cpu.sessionOf(encArr, ckArr, cvArr);
    }

    private void releaseAll(Metal.Buffer[]... sets) {
        for (Metal.Buffer[] set : sets) {
            for (Metal.Buffer b : set) {
                if (b != null) {
                    g.metal.release(b);
                }
            }
        }
    }

    // ------------------------------------------------------------------ decoder on the GPU

    private final class GpuSession implements Session {
        private final float[] encArr;
        private final Metal.Buffer[] crossKs;
        private final Metal.Buffer[] crossVs;
        private final Metal.Buffer[] selfK;
        private final Metal.Buffer[] selfVt;
        private boolean closed;

        GpuSession(float[] encArr, Metal.Buffer[] crossKs, Metal.Buffer[] crossVs) {
            this.encArr = encArr;
            this.crossKs = crossKs;
            this.crossVs = crossVs;
            int n = c.decLayers();
            selfK = new Metal.Buffer[n];
            selfVt = new Metal.Buffer[n];
            for (int i = 0; i < n; i++) {
                selfK[i] = g.buffer(4L * c.targetPos() * c.dModel());
                selfVt[i] = g.buffer(4L * c.targetPos() * c.dModel());
            }
        }

        @Override
        public void decode(int[] tokens, int off, int n, int pos, float[] logits) {
            if (pos < 0 || pos + n > c.targetPos()) {
                throw new IllegalArgumentException("decoder positions " + pos + ".." + (pos + n) + " exceed " + c.targetPos());
            }
            lock.lock();
            try {
                for (int done = 0; done < n; done += DEC_CHUNK) {
                    int m = Math.min(DEC_CHUNK, n - done);
                    for (int attempt = 0; ; attempt++) {
                        try {
                            chunk(this, tokens, off + done, m, pos + done, done + m == n ? logits : null);
                            break;
                        } catch (IllegalStateException e) {
                            if (!transient_(e) || attempt >= 2) {
                                throw e;
                            }
                            pause(attempt);
                        }
                    }
                }
            } finally {
                lock.unlock();
            }
        }

        @Override
        public float[][] crossAttention(int[] tokens, int[][] heads) {
            int t2 = c.sourcePos();
            int d = c.dModel();
            float[][] ck = new float[c.decLayers()][t2 * d];
            float[][] cv = new float[c.decLayers()][t2 * d];
            lock.lock();
            try {
                for (int i = 0; i < ck.length; i++) {
                    GpuCtx.get(crossKs[i], 0, ck[i], t2 * d);
                    GpuCtx.get(crossVs[i], 0, cv[i], t2 * d);
                }
            } finally {
                lock.unlock();
            }
            try (Session s = cpu.sessionOf(encArr, ck, cv)) {
                return s.crossAttention(tokens, heads);
            }
        }

        @Override
        public float[] encoderOutput() {
            return encArr;
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            lock.lock();
            try {
                releaseAll(crossKs, crossVs, selfK, selfVt);
            } finally {
                lock.unlock();
            }
        }
    }

    /** One decoder step over {@code n <= DEC_CHUNK} tokens at positions {@code [pos, pos + n)}; logits of the last when asked. */
    private void chunk(GpuSession s, int[] tokens, int off, int n, int pos, float[] logits) {
        int d = c.dModel();
        int heads = c.decHeads();
        int hd = c.decHeadDim();
        int t2 = c.sourcePos();
        int cap = c.targetPos();
        float scale = (float) (1.0 / Math.sqrt(hd));
        MemorySegment tb = tokBuf.contents();
        for (int i = 0; i < n; i++) {
            tb.setAtIndex(ValueLayout.JAVA_INT, tokBuf.offset() / 4 + i, tokens[off + i]);
        }
        g.begin();
        try {
            g.run(embed.half() ? "wh_embed_h" : "wh_embed_f", GpuCtx.ceilDiv(n * d, 256), 1, 1, 256, 1, 1, new Object[]{n, d, pos},
                    embed.w(), decPos, tokBuf, dx);
            for (int li = 0; li < dec.length; li++) {
                DLayer l = dec[li];
                layerNorm(dx, dxn, l.ln1g(), l.ln1b(), n, d);
                linear(l.sq(), dxn, dq, n, 0);
                linear(l.sk(), dxn, dk, n, 0);
                linear(l.sv(), dxn, dv, n, 0);
                g.run("wh_split_heads", GpuCtx.ceilDiv(n * d, 256), 1, 1, 256, 1, 1,
                        new Object[]{n, heads, hd, cap, pos, scale, 1, 1}, dq, dk, dv, dqh, s.selfK[li], s.selfVt[li]);
                attention(dqh, s.selfK[li], s.selfVt[li], dscores, dctx, n, pos + n, heads, hd, cap, true, pos);
                linear(l.so(), dctx, dx, n, ACC);
                layerNorm(dx, dxn, l.ln2g(), l.ln2b(), n, d);
                linear(l.cq(), dxn, dq, n, 0);
                g.run("wh_split_heads", GpuCtx.ceilDiv(n * d, 256), 1, 1, 256, 1, 1,
                        new Object[]{n, heads, hd, cap, 0, scale, 1, 0}, dq, dk, dv, dqh, s.selfK[li], s.selfVt[li]);
                attention(dqh, s.crossKs[li], s.crossVs[li], dscores, dctx, n, t2, heads, hd, t2, false, 0);
                linear(l.co(), dctx, dx, n, ACC);
                layerNorm(dx, dxn, l.ln3g(), l.ln3b(), n, d);
                linear(l.fc1(), dxn, dhid, n, GELU);
                linear(l.fc2(), dhid, dx, n, ACC);
            }
            if (logits != null) {
                g.runAt("wh_layernorm", 1, 1, 1, 256, 1, 1, new Object[]{d, LN_EPS, 1},
                        new Metal.Buffer[]{dx, lastLn, decLnG, decLnB}, new long[]{4L * (n - 1) * d, 0, 0, 0});
                linear(embed, lastLn, logitsBuf, 1, 0);
            }
        } catch (RuntimeException e) {
            g.metal.abort();
            throw e;
        }
        g.commitAndWait();
        if (logits != null) {
            GpuCtx.get(logitsBuf, 0, logits, c.vocab());
        }
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
