package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.Rope;

import java.util.Arrays;

/**
 * Decoder forward pass on the CPU (Llama, Mistral, Qwen2, Qwen3): quantized matmuls with the Vector API,
 * multithreaded over output rows (decode) or row groups (prefill), f32 or f16 KV cache on the Java heap ({@link KvCache}).
 */
public final class CpuBackend implements Backend {

    private final ModelWeights w;
    private final ModelConfig c;
    private final int ctx;
    private final int maxBatch;
    private final ThreadPool pool;
    private final Rope rope;

    private final KvCache kv;
    // k/v of the current chunk, [maxBatch, kvDim]: projected, normalized and rotated here, then stored
    private final float[] kStage;
    private final float[] vStage;

    // activations, [maxBatch, width] row-major
    private final float[] x;
    private final float[] xb;
    private final float[] q;
    private final float[] att;
    private final float[] gateUp;
    private final float[] hidden;

    // MoE scratch (null for dense models)
    private final float[] routerLogits;
    private final int[] assign;
    private final float[] routeWeight;
    private final float[] moeGateUp;
    private final float[] moeHidden;

    private final ThreadLocal<float[]> scratch;

    public CpuBackend(ModelWeights w, int ctx, int maxBatch, int threads) {
        this(w, ctx, maxBatch, threads, KvDtype.F32);
    }

    public CpuBackend(ModelWeights w, int ctx, int maxBatch, int threads, KvDtype kvDtype) {
        this.w = w;
        this.c = w.cfg;
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        this.pool = new ThreadPool(threads);
        this.rope = Rope.of(c, ctx);
        kv = KvCache.create(kvDtype, c, ctx);
        kStage = new float[maxBatch * c.kvDim()];
        vStage = new float[maxBatch * c.kvDim()];
        x = new float[maxBatch * c.dim()];
        xb = new float[maxBatch * c.dim()];
        q = new float[maxBatch * c.qDim()];
        att = new float[maxBatch * c.qDim()];
        gateUp = new float[maxBatch * 2 * c.hiddenDim()];
        hidden = new float[maxBatch * c.hiddenDim()];
        if (c.isMoe()) {
            routerLogits = new float[c.numExperts()];
            assign = new int[c.expertsPerTok()];
            routeWeight = new float[c.expertsPerTok()];
            moeGateUp = new float[c.expertsPerTok() * 2 * c.moeDim()];
            moeHidden = new float[c.expertsPerTok() * c.moeDim()];
        } else {
            routerLogits = null;
            assign = null;
            routeWeight = null;
            moeGateUp = null;
            moeHidden = null;
        }
        int scratchLen = Math.max(ctx, 4 * Math.max(c.hiddenDim(), Math.max(c.dim(), c.qDim())));
        scratch = ThreadLocal.withInitial(() -> new float[scratchLen]);
    }

    @Override
    public String name() {
        return "cpu (" + VectorKernels.describe() + ", " + pool.threads() + " threads)";
    }

    @Override
    public int contextLength() {
        return ctx;
    }

    @Override
    public int maxBatch() {
        return maxBatch;
    }

    @Override
    public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
        if (n < 1 || n > maxBatch || pos + n > ctx) {
            throw new IllegalArgumentException("bad forward(n=" + n + ", pos=" + pos + ")");
        }
        int dim = c.dim();
        int kvDim = c.kvDim();
        for (int t = 0; t < n; t++) {
            w.embed.dequantizeRow(tokens[off + t], x, t * dim);
        }
        for (int l = 0; l < c.layers(); l++) {
            ModelWeights.Layer L = w.layers[l];
            rmsnorm(x, xb, L.attnNorm, n, dim);
            matmul(L.wq, xb, q, 0, n, false);
            matmul(L.wk, xb, kStage, 0, n, false);
            matmul(L.wv, xb, vStage, 0, n, false);
            if (L.bq != null) {
                addBias(q, 0, L.bq, n);
                addBias(kStage, 0, L.bk, n);
                addBias(vStage, 0, L.bv, n);
            }
            normRope(q, 0, c.qDim(), c.heads(), L.qNorm, n, pos);
            normRope(kStage, 0, kvDim, c.kvHeads(), L.kNorm, n, pos);
            kv.store(l, pos, kStage, vStage, n);
            attention(l, n, pos);
            matmul(L.wo, att, x, 0, n, true);

            rmsnorm(x, xb, L.ffnNorm, n, dim);
            if (L.moe) {
                moe(L, n);
            } else {
                matmul(L.wGateUp, xb, gateUp, 0, n, false);
                siluMul(n);
                matmul(L.wDown, hidden, x, 0, n, true);
            }
        }
        if (logits != null) {
            rmsnormRow(x, (n - 1) * dim, xb, 0, w.finalNorm, dim);
            matmul(w.lmHead, xb, logits, 0, 1, false);
        }
    }

    // ------------------------------------------------------------------ layers

    private void rmsnorm(float[] in, float[] out, float[] weight, int n, int width) {
        for (int t = 0; t < n; t++) {
            rmsnormRow(in, t * width, out, t * width, weight, width);
        }
    }

    private void rmsnormRow(float[] in, int io, float[] out, int oo, float[] weight, int width) {
        float ss = VectorKernels.sumSquares(in, io, width);
        float scale = (float) (1.0 / Math.sqrt(ss / width + c.normEps()));
        for (int i = 0; i < width; i++) {
            out[oo + i] = in[io + i] * scale * weight[i];
        }
    }

    /**
     * out[t, r] (+)= dot(W[r, :], in[t, :]) for t in [0, n). Decode streams quantized rows straight into FMAs;
     * prefill dequantizes four rows at a time and reuses them for every token in the batch.
     */
    private void matmul(QTensor wt, float[] in, float[] out, int outOff, int n, boolean acc) {
        int rows = wt.rows;
        int cols = wt.cols;
        if (n == 1) {
            pool.forRange(rows, 16, (r0, r1) -> {
                for (int r = r0; r < r1; r++) {
                    float v = VectorKernels.dotRow(wt, r, in, 0);
                    out[outOff + r] = acc ? out[outOff + r] + v : v;
                }
            });
            return;
        }
        int groups = (rows + 3) / 4;
        pool.forRange(groups, 1, (g0, g1) -> {
            float[] tmp = scratch.get();
            for (int g = g0; g < g1; g++) {
                int r = g * 4;
                if (r + 4 <= rows) {
                    for (int k = 0; k < 4; k++) {
                        VectorKernels.dequantRow(wt, r + k, tmp, k * cols);
                    }
                    for (int t = 0; t < n; t++) {
                        VectorKernels.dot4(tmp, 0, cols, in, t * cols, out, outOff + t * rows + r, acc);
                    }
                } else {
                    for (int rr = r; rr < rows; rr++) {
                        for (int t = 0; t < n; t++) {
                            float v = VectorKernels.dotRow(wt, rr, in, t * cols);
                            int o = outOff + t * rows + rr;
                            out[o] = acc ? out[o] + v : v;
                        }
                    }
                }
            }
        });
    }

    /** buf[off + t*bias.length + i] += bias[i] for n rows (Qwen2 q/k/v bias). */
    private static void addBias(float[] buf, int off, float[] bias, int n) {
        for (int t = 0; t < n; t++) {
            int o = off + t * bias.length;
            for (int i = 0; i < bias.length; i++) {
                buf[o + i] += bias[i];
            }
        }
    }

    /** Per-head RMSNorm (if {@code norm != null}) followed by rotary embedding, in place. */
    private void normRope(float[] v, int base, int stride, int nHeads, float[] norm, int n, int pos) {
        int hd = c.headDim();
        int half = hd / 2;
        for (int t = 0; t < n; t++) {
            float[] cos = rope.cos;
            float[] sin = rope.sin;
            int ro = (pos + t) * half;
            for (int h = 0; h < nHeads; h++) {
                int o = base + t * stride + h * hd;
                if (norm != null) {
                    rmsnormRow(v, o, v, o, norm, hd);
                }
                for (int i = 0; i < half; i++) {
                    float a = v[o + i];
                    float b = v[o + i + half];
                    float cs = cos[ro + i];
                    float sn = sin[ro + i];
                    v[o + i] = a * cs - b * sn;
                    v[o + i + half] = b * cs + a * sn;
                }
            }
        }
    }

    /** Causal grouped-query attention for tokens at positions [pos, pos+n). */
    private void attention(int layer, int n, int pos) {
        if (kv instanceof KvCache.F16 half) {
            half.attention(pool, layer, n, pos, q, att);
            return;
        }
        float[] kc = ((KvCache.F32) kv).key[layer];
        float[] vc = ((KvCache.F32) kv).val[layer];
        int hd = c.headDim();
        int nh = c.heads();
        int kvDim = c.kvDim();
        int qDim = c.qDim();
        int group = nh / c.kvHeads();
        float scale = (float) (1.0 / Math.sqrt(hd));
        int window = c.windowFor(layer); // 0 = full causal attention
        pool.forRange(n * nh, 1, (i0, i1) -> {
            float[] sc = scratch.get();
            for (int i = i0; i < i1; i++) {
                int t = i / nh;
                int h = i % nh;
                int ko = (h / group) * hd;
                int qo = t * qDim + h * hd;
                int len = pos + t + 1;
                int first = window > 0 ? Math.max(0, len - window) : 0;
                float max = Float.NEGATIVE_INFINITY;
                for (int s = first; s < len; s++) {
                    float d = VectorKernels.dot(q, qo, kc, s * kvDim + ko, hd) * scale;
                    sc[s] = d;
                    max = Math.max(max, d);
                }
                float sum = 0;
                for (int s = first; s < len; s++) {
                    float e = (float) Math.exp(sc[s] - max);
                    sc[s] = e;
                    sum += e;
                }
                float inv = 1f / sum;
                Arrays.fill(att, qo, qo + hd, 0f);
                for (int s = first; s < len; s++) {
                    VectorKernels.axpy(sc[s] * inv, vc, s * kvDim + ko, att, qo, hd);
                }
            }
        });
    }

    /**
     * Mixture of experts: for each token, softmax over the router logits, pick the top-k experts, run each
     * expert's SwiGLU MLP on the token and add the results to x weighted by the (renormalized) router probabilities.
     */
    private void moe(ModelWeights.Layer L, int n) {
        int dim = c.dim();
        int e = c.numExperts();
        int topK = c.expertsPerTok();
        int m = c.moeDim();
        int gu = 2 * m;
        for (int t = 0; t < n; t++) {
            int xo = t * dim;
            route(L.router, xo, e, topK);
            // all selected experts' gate/up rows in one parallel loop
            pool.forRange(topK * gu, 32, (i0, i1) -> {
                for (int i = i0; i < i1; i++) {
                    int k = i / gu;
                    int r = i - k * gu;
                    moeGateUp[i] = VectorKernels.dotRow(L.expGateUp, assign[k] * gu + r, xb, xo);
                }
            });
            for (int k = 0; k < topK; k++) {
                int gi = k * gu;
                int hi = k * m;
                for (int i = 0; i < m; i++) {
                    float g = moeGateUp[gi + i];
                    moeHidden[hi + i] = g / (1f + (float) Math.exp(-g)) * moeGateUp[gi + m + i];
                }
            }
            pool.forRange(dim, 8, (r0, r1) -> {
                for (int r = r0; r < r1; r++) {
                    float sum = 0;
                    for (int k = 0; k < topK; k++) {
                        sum += routeWeight[k] * VectorKernels.dotRow(L.expDown, assign[k] * dim + r, moeHidden, k * m);
                    }
                    x[xo + r] += sum;
                }
            });
        }
    }

    /** Router for one token: fills assign[] / routeWeight[] (same rule as transformers' Qwen3MoeSparseMoeBlock). */
    private void route(float[] router, int xo, int e, int topK) {
        int dim = c.dim();
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < e; i++) {
            float v = VectorKernels.dot(router, i * dim, xb, xo, dim);
            routerLogits[i] = v;
            max = Math.max(max, v);
        }
        float sum = 0;
        for (int i = 0; i < e; i++) {
            float p = (float) Math.exp(routerLogits[i] - max);
            routerLogits[i] = p;
            sum += p;
        }
        float selected = 0;
        for (int k = 0; k < topK; k++) {
            int best = -1;
            float bv = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < e; i++) {
                if (routerLogits[i] > bv) {
                    bv = routerLogits[i];
                    best = i;
                }
            }
            assign[k] = best;
            routeWeight[k] = bv / sum;
            selected += bv / sum;
            routerLogits[best] = Float.NEGATIVE_INFINITY;
        }
        if (c.normTopK() && selected > 0) {
            for (int k = 0; k < topK; k++) {
                routeWeight[k] /= selected;
            }
        }
    }

    private void siluMul(int n) {
        int hdim = c.hiddenDim();
        for (int t = 0; t < n; t++) {
            int gi = t * 2 * hdim;
            int oi = t * hdim;
            for (int i = 0; i < hdim; i++) {
                float g = gateUp[gi + i];
                hidden[oi + i] = g / (1f + (float) Math.exp(-g)) * gateUp[gi + hdim + i];
            }
        }
    }

    @Override
    public void close() {
        pool.close();
    }
}
