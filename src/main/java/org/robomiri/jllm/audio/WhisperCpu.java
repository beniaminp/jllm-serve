package org.robomiri.jllm.audio;

import org.robomiri.jllm.audio.WhisperWeights.Attn;
import org.robomiri.jllm.audio.WhisperWeights.Layer;
import org.robomiri.jllm.cpu.ThreadPool;
import org.robomiri.jllm.cpu.VectorKernels;

import java.util.concurrent.locks.ReentrantLock;

/**
 * Whisper on the CPU: every dense layer is {@link Dense#mul} (4 weight rows x 4 inputs register tiles over the
 * Vector API, parallel over weight rows), attention is two such products per head and block of queries
 * ({@code S = Q K^T}, softmax, {@code O = P V} with V stored transposed so both are row-by-row dot products), the
 * encoder's convolutions are an im2col followed by the same product.
 *
 * <p>One request at a time: the thread pool is shared, so encoder and decoder calls take a lock.
 */
public final class WhisperCpu implements WhisperNet {

    private static final float LN_EPS = 1e-5f;

    private final WhisperWeights w;
    private final WhisperConfig c;
    private final ThreadPool pool;
    private final ReentrantLock lock = new ReentrantLock();

    public WhisperCpu(WhisperWeights w, int threads) {
        this.w = w;
        this.c = w.cfg;
        this.pool = new ThreadPool(threads);
    }

    @Override
    public WhisperConfig config() {
        return c;
    }

    @Override
    public String name() {
        return "cpu (" + pool.threads() + " threads, Vector API " + Dense.F.length() * 32 + "-bit)";
    }

    @Override
    public void close() {
        pool.close();
    }

    // ------------------------------------------------------------------ encoder

    @Override
    public Session encode(float[][] mel) {
        lock.lock();
        try {
            return new CpuSession(encoder(mel));
        } finally {
            lock.unlock();
        }
    }

    /** A decoder session on an encoder output computed elsewhere (the Metal net's word-timestamp pass). */
    public Session sessionOf(float[] encoderOutput) {
        lock.lock();
        try {
            return new CpuSession(encoderOutput);
        } finally {
            lock.unlock();
        }
    }

    private float[] encoder(float[][] mel) {
        int d = c.dModel();
        int nm = c.nMels();
        int frames = c.windowFrames();
        if (mel.length != nm || mel[0].length != frames) {
            throw new IllegalArgumentException("mel must be " + nm + " x " + frames + ", got " + mel.length + " x "
                    + mel[0].length);
        }
        int t2 = c.sourcePos();
        // conv1 (k=3, pad 1) as im2col + product: X[t][ch * 3 + k] = mel[ch][t + k - 1]
        float[] x1 = new float[frames * nm * 3];
        pool.forRange(frames, 64, (s, e) -> {
            for (int t = s; t < e; t++) {
                int b = t * nm * 3;
                for (int ch = 0; ch < nm; ch++) {
                    float[] row = mel[ch];
                    for (int k = 0; k < 3; k++) {
                        int i = t + k - 1;
                        x1[b + ch * 3 + k] = i >= 0 && i < frames ? row[i] : 0f;
                    }
                }
            }
        });
        float[] h1 = new float[frames * d];
        Dense.mul(pool, w.conv1, x1, 0, nm * 3, frames, h1, 0, d, false);
        gelu(h1);
        // conv2 (k=3, stride 2, pad 1): X[t][ch * 3 + k] = h1[2t + k - 1][ch]
        float[] x2 = new float[t2 * d * 3];
        pool.forRange(t2, 64, (s, e) -> {
            for (int t = s; t < e; t++) {
                int b = t * d * 3;
                for (int k = 0; k < 3; k++) {
                    int i = 2 * t + k - 1;
                    if (i < 0 || i >= frames) {
                        continue;
                    }
                    int src = i * d;
                    for (int ch = 0; ch < d; ch++) {
                        x2[b + ch * 3 + k] = h1[src + ch];
                    }
                }
            }
        });
        float[] x = new float[t2 * d];
        Dense.mul(pool, w.conv2, x2, 0, d * 3, t2, x, 0, d, false);
        gelu(x);
        Dense.add(x, 0, w.encPos, 0, t2 * d);

        int heads = c.encHeads();
        int hd = c.encHeadDim();
        int ffn = c.encFfn();
        float[] xn = new float[t2 * d];
        float[] q = new float[t2 * d];
        float[] k = new float[t2 * d];
        float[] v = new float[t2 * d];
        float[] kh = new float[t2 * d];
        float[] vt = new float[t2 * d];
        float[] ctx = new float[t2 * d];
        float[] hid = new float[t2 * ffn];
        float scale = (float) (1.0 / Math.sqrt(hd));
        for (Layer l : w.enc) {
            Dense.layerNorm(x, 0, t2, d, l.self.lnGamma, l.self.lnBeta, LN_EPS, xn, 0);
            Dense.mul(pool, l.self.q, xn, 0, d, t2, q, 0, d, false);
            Dense.mul(pool, l.self.k, xn, 0, d, t2, k, 0, d, false);
            Dense.mul(pool, l.self.v, xn, 0, d, t2, v, 0, d, false);
            scale(q, scale);
            keysByHead(k, t2, heads, hd, kh, 0, t2);
            valuesByHead(v, t2, heads, hd, vt, 0, t2, t2);
            attend(q, t2, d, kh, t2 * hd, vt, hd * t2, t2, heads, hd, -1, ctx);
            Dense.mul(pool, l.self.out, ctx, 0, d, t2, x, 0, d, true);
            Dense.layerNorm(x, 0, t2, d, l.ffnLnGamma, l.ffnLnBeta, LN_EPS, xn, 0);
            Dense.mul(pool, l.fc1, xn, 0, d, t2, hid, 0, ffn, false);
            gelu(hid);
            Dense.mul(pool, l.fc2, hid, 0, ffn, t2, x, 0, d, true);
        }
        Dense.layerNorm(x, 0, t2, d, w.encLnGamma, w.encLnBeta, LN_EPS, x, 0);
        return x;
    }

    private void gelu(float[] a) {
        pool.forRange(a.length, 1 << 14, (s, e) -> Dense.gelu(a, s, e - s));
    }

    private static void scale(float[] a, float s) {
        for (int i = 0; i < a.length; i++) {
            a[i] *= s;
        }
    }

    /** Token-major {@code k[t][h * hd + e]} into head-major {@code kh[(h * cap + pos0 + t) * hd + e]}. */
    private static void keysByHead(float[] k, int n, int heads, int hd, float[] kh, int pos0, int cap) {
        int d = heads * hd;
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < heads; h++) {
                System.arraycopy(k, t * d + h * hd, kh, (h * cap + pos0 + t) * hd, hd);
            }
        }
    }

    /** Token-major {@code v[t][h * hd + e]} into transposed head-major {@code vt[(h * hd + e) * cap + pos0 + t]}. */
    private static void valuesByHead(float[] v, int n, int heads, int hd, float[] vt, int pos0, int cap, int unused) {
        int d = heads * hd;
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < heads; h++) {
                int src = t * d + h * hd;
                int dst = h * hd * cap + pos0 + t;
                for (int e = 0; e < hd; e++) {
                    vt[dst + e * cap] = v[src + e];
                }
            }
        }
    }

    // ------------------------------------------------------------------ attention

    private static final int QBLOCK = 32;
    /** When set (by {@code crossAttention}), the next {@code attend} copies the probabilities of these heads: [head] -> [query][key]. */
    private float[][] capture;
    private static final ThreadLocal<float[]> SCORES = new ThreadLocal<>();

    /**
     * Scaled-dot-product attention of {@code n} query rows {@code q[t * d ..]} (already multiplied by 1/sqrt(hd))
     * over {@code len} keys. {@code kh} holds each head's keys as {@code [len][hd]} blocks {@code kStride} apart,
     * {@code vt} each head's values transposed as {@code [hd][len]} blocks {@code vStride} apart (so a head's cache can
     * have a capacity larger than {@code len}: the row length of {@code vt} is {@code vRow}). With {@code pos0 >= 0}
     * the attention is causal: query t sees keys {@code 0 .. pos0 + t}; with -1 it sees all {@code len}.
     */
    private void attend(float[] q, int n, int d, float[] kh, int kStride, float[] vt, int vStride, int len, int heads, int hd,
                        int pos0, float[] ctx) {
        attend(q, n, d, kh, kStride, vt, vStride, len, len, heads, hd, pos0, ctx);
    }

    private void attend(float[] q, int n, int d, float[] kh, int kStride, float[] vt, int vStride, int len, int vRow,
                        int heads, int hd, int pos0, float[] ctx) {
        int blocks = (n + QBLOCK - 1) / QBLOCK;
        int tasks = heads * blocks;
        pool.forRangeFine(tasks, 1, (s, e) -> {
            float[] sc = SCORES.get();
            if (sc == null || sc.length < QBLOCK * len) {
                sc = new float[QBLOCK * len];
                SCORES.set(sc);
            }
            for (int task = s; task < e; task++) {
                int h = task / blocks;
                int t0 = (task % blocks) * QBLOCK;
                int nq = Math.min(QBLOCK, n - t0);
                // S[t][j] = q_t . k_j
                Dense.block(kh, h * kStride, len, hd, q, t0 * d + h * hd, d, nq, sc, 0, len, false);
                for (int t = 0; t < nq; t++) {
                    int vis = pos0 >= 0 ? Math.min(len, pos0 + t0 + t + 1) : len;
                    int o = t * len;
                    float max = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < vis; j++) {
                        max = Math.max(max, sc[o + j]);
                    }
                    float sum = VectorKernels.softmaxExp(sc, o, vis, max);
                    float inv = 1f / sum;
                    for (int j = 0; j < vis; j++) {
                        sc[o + j] *= inv;
                    }
                    for (int j = vis; j < len; j++) {
                        sc[o + j] = 0f;
                    }
                    float[][] cap = capture;
                    if (cap != null && cap[h] != null) {
                        System.arraycopy(sc, o, cap[h], (t0 + t) * len, len);
                    }
                }
                // O[t][e] = sum_j P[t][j] V[j][e], with V transposed: rows e, length vRow, first len used
                outProduct(vt, h * vStride, hd, vRow, len, sc, nq, ctx, t0 * d + h * hd, d);
            }
        });
    }

    /** {@code out[oo + t * os + e] = dot(vt[vo + e * vRow .. +len), p[t * len ..])}. */
    private static void outProduct(float[] vt, int vo, int rows, int vRow, int len, float[] p, int nq, float[] out, int oo,
                                   int os) {
        if (vRow == len) {
            Dense.block(vt, vo, rows, len, p, 0, len, nq, out, oo, os, false);
            return;
        }
        for (int r = 0; r < rows; r++) {
            for (int t = 0; t < nq; t++) {
                out[oo + t * os + r] = Dense.dot(vt, vo + r * vRow, p, t * len, len);
            }
        }
    }

    // ------------------------------------------------------------------ decoder

    private final class CpuSession implements Session {
        private final float[] enc;
        private final float[][] crossK;        // [layer][head][pos][hd]
        private final float[][] crossVt;       // [layer][head][hd][pos]
        private final float[][] selfK;         // [layer][head][maxPos][hd]
        private final float[][] selfVt;        // [layer][head][hd][maxPos]
        private final int maxPos;

        CpuSession(float[] enc) {
            this.enc = enc;
            int d = c.dModel();
            int t2 = c.sourcePos();
            int heads = c.decHeads();
            int hd = c.decHeadDim();
            maxPos = c.targetPos();
            crossK = new float[c.decLayers()][];
            crossVt = new float[c.decLayers()][];
            selfK = new float[c.decLayers()][];
            selfVt = new float[c.decLayers()][];
            float[] k = new float[t2 * d];
            float[] v = new float[t2 * d];
            for (int i = 0; i < c.decLayers(); i++) {
                Attn a = w.dec[i].cross;
                Dense.mul(pool, a.k, enc, 0, d, t2, k, 0, d, false);
                Dense.mul(pool, a.v, enc, 0, d, t2, v, 0, d, false);
                crossK[i] = new float[t2 * d];
                crossVt[i] = new float[t2 * d];
                keysByHead(k, t2, heads, hd, crossK[i], 0, t2);
                valuesByHead(v, t2, heads, hd, crossVt[i], 0, t2, t2);
                selfK[i] = new float[maxPos * d];
                selfVt[i] = new float[maxPos * d];
            }
        }

        @Override
        public float[] encoderOutput() {
            return enc;
        }

        @Override
        public void decode(int[] tokens, int off, int n, int pos, float[] logits) {
            lock.lock();
            try {
                run(tokens, off, n, pos, logits, null);
            } finally {
                lock.unlock();
            }
        }

        @Override
        public float[][] crossAttention(int[] tokens, int[][] heads) {
            lock.lock();
            try {
                int n = tokens.length;
                int t2 = c.sourcePos();
                float[][] out = new float[heads.length][n * t2];
                float[][][] cap = new float[c.decLayers()][c.decHeads()][];
                for (int i = 0; i < heads.length; i++) {
                    cap[heads[i][0]][heads[i][1]] = out[i];
                }
                run(tokens, 0, n, 0, new float[c.vocab()], cap);
                return out;
            } finally {
                lock.unlock();
            }
        }

        private void run(int[] tokens, int off, int n, int pos, float[] logits, float[][][] cap) {
            if (pos < 0 || pos + n > maxPos) {
                throw new IllegalArgumentException("decoder positions " + pos + ".." + (pos + n) + " exceed " + maxPos);
            }
            int d = c.dModel();
            int heads = c.decHeads();
            int hd = c.decHeadDim();
            int ffn = c.decFfn();
            int t2 = c.sourcePos();
            float[] x = new float[n * d];
            for (int t = 0; t < n; t++) {
                w.embed.row(tokens[off + t], x, t * d);
                if (c.scaleEmbedding()) {
                    float s = (float) Math.sqrt(d);
                    for (int i = 0; i < d; i++) {
                        x[t * d + i] *= s;
                    }
                }
                Dense.add(x, t * d, w.decPos, (pos + t) * d, d);
            }
            float[] xn = new float[n * d];
            float[] q = new float[n * d];
            float[] k = new float[n * d];
            float[] v = new float[n * d];
            float[] ctx = new float[n * d];
            float[] hid = new float[n * ffn];
            float scale = (float) (1.0 / Math.sqrt(hd));
            for (int li = 0; li < c.decLayers(); li++) {
                Layer l = w.dec[li];
                // causal self-attention over the cache
                Dense.layerNorm(x, 0, n, d, l.self.lnGamma, l.self.lnBeta, LN_EPS, xn, 0);
                Dense.mul(pool, l.self.q, xn, 0, d, n, q, 0, d, false);
                Dense.mul(pool, l.self.k, xn, 0, d, n, k, 0, d, false);
                Dense.mul(pool, l.self.v, xn, 0, d, n, v, 0, d, false);
                scale(q, scale);
                keysByHead(k, n, heads, hd, selfK[li], pos, maxPos);
                valuesByHead(v, n, heads, hd, selfVt[li], pos, maxPos, maxPos);
                attend(q, n, d, selfK[li], maxPos * hd, selfVt[li], hd * maxPos, pos + n, maxPos, heads, hd, pos, ctx);
                Dense.mul(pool, l.self.out, ctx, 0, d, n, x, 0, d, true);
                // cross-attention over the encoder output
                Dense.layerNorm(x, 0, n, d, l.cross.lnGamma, l.cross.lnBeta, LN_EPS, xn, 0);
                Dense.mul(pool, l.cross.q, xn, 0, d, n, q, 0, d, false);
                scale(q, scale);
                capture = cap == null ? null : cap[li];
                attend(q, n, d, crossK[li], t2 * hd, crossVt[li], hd * t2, t2, heads, hd, -1, ctx);
                capture = null;
                Dense.mul(pool, l.cross.out, ctx, 0, d, n, x, 0, d, true);
                Dense.layerNorm(x, 0, n, d, l.ffnLnGamma, l.ffnLnBeta, LN_EPS, xn, 0);
                Dense.mul(pool, l.fc1, xn, 0, d, n, hid, 0, ffn, false);
                gelu(hid);
                Dense.mul(pool, l.fc2, hid, 0, ffn, n, x, 0, d, true);
            }
            float[] last = new float[d];
            Dense.layerNorm(x, (n - 1) * d, 1, d, w.decLnGamma, w.decLnBeta, LN_EPS, last, 0);
            Dense.mul(pool, w.embed, last, 0, d, 1, logits, 0, c.vocab(), false);
        }

        @Override
        public void close() {
        }
    }
}
