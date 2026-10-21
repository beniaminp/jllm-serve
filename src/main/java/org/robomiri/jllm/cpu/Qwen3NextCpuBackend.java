package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.AttnWeights;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.MlpWeights;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.Qwen3NextArch;
import org.robomiri.jllm.model.Qwen3NextLayer;
import org.robomiri.jllm.model.Rope;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.util.Arrays;

/**
 * Qwen3-Next ({@code Qwen3NextForCausalLM}) forward pass on the CPU (docs/14.14).
 *
 * <p>Gated DeltaNet layers keep a fixed-size state: a conv state {@code [convDim, convKernel - 1]} (the last inputs of
 * the causal conv) and a recurrent state per value head, stored <em>transposed</em> as {@code S[h][j][i]} (value dim
 * {@code j} major, key dim {@code i} contiguous) so that every column of transformers' {@code [keyDim, valueDim]}
 * state is a contiguous vector. The full-attention layers keep a {@link KvCache} (only those layers). The recurrence
 * runs token by token for single tokens (decode), and in the chunked (UT / WY) form for longer chunks, see
 * {@link #chunked}; the two agree to float rounding ({@code -Djllm.gdn.chunked=false} forces the token loop,
 * {@code -Djllm.gdn.chunk} sets the chunk length, 64 by default as in transformers).
 *
 * <p>The state belongs to one sequence like Mamba's ({@link MambaCpuBackend}): {@code pos == 0} starts a new one, any
 * other {@code pos} must equal the number of tokens fed so far, and it cannot be rewound
 * ({@link #supportsPrefixReuse} is false). {@link #saveState} / {@link #loadState} / {@link #copyStateFrom} transfer
 * the KV rows of the full-attention layers together with the recurrent and conv states, exactly, for {@code n} equal
 * to the number of tokens fed.
 */
public final class Qwen3NextCpuBackend implements Backend {

    /** The token-by-token recurrence also for chunks ({@code -Djllm.gdn.chunked=false}); for comparisons. */
    static final boolean CHUNKED = !"false".equals(System.getProperty("jllm.gdn.chunked"));
    /** Tokens per chunk of the chunked delta rule. */
    static final int CHUNK = Math.max(1, Integer.getInteger("jllm.gdn.chunk", 64));
    /** {@code eps} of the q / k L2 normalization, as transformers' (and FLA's) {@code l2norm}. */
    static final float L2_EPS = 1e-6f;

    private final ModelWeights w;
    private final Qwen3NextArch arch;
    private final Architecture.Core c;
    private final Qwen3NextArch.DeltaNet dn;
    private final int ctx;
    private final int maxBatch;
    private final ThreadPool pool;
    private final CpuMatmul mm;
    private final Rope rope;
    private final KvCache kv;
    private final MoeExperts experts;

    private final float[][] convState;
    private final float[][] state;
    /** Tokens fed into the current state, -1 while a forward is in flight or after one failed. */
    private int statePos = 0;

    // activations, [maxBatch, width] row-major
    private final float[] x;
    private final float[] xb;
    // DeltaNet
    private final float[] proj;
    private final float[] qkv;
    private final float[] o;
    private final float[] decay;   // [n, valueHeads]: log-space decay g = A * softplus(a + dt_bias)
    private final float[] beta;    // [n, valueHeads]
    // full attention
    private final float[] q;
    private final float[] gate;
    private final float[] kStage;
    private final float[] vStage;
    private final float[] att;
    // MLP
    private final float[] gateUp;
    private final float[] hidden;
    private final float[] shared;
    private final int[] assign;
    private final float[] routeWeight;
    private ThreadLocal<ChunkBuf> chunkBuf;
    private boolean chunkedPrefill = CHUNKED;

    /** The exact float path (int8 activations off), as the CPU-vs-Metal checks need it. */
    public Qwen3NextCpuBackend(ModelWeights w, int ctx, int maxBatch, int threads, KvDtype kvDtype) {
        this(w, ctx, maxBatch, threads, kvDtype, false);
    }

    /**
     * @param int8Activations int8 integer dot products for the single-token projections, as in {@link CpuBackend}
     */
    public Qwen3NextCpuBackend(ModelWeights w, int ctx, int maxBatch, int threads, KvDtype kvDtype, boolean int8Activations) {
        this.w = w;
        this.arch = w.qwen3Next();
        this.c = arch.core();
        this.dn = arch.deltaNet();
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        this.pool = new ThreadPool(threads);
        Attention.Gqa g = arch.attention();
        rope = new Rope(arch.rope(), ctx);
        kv = arch.fullLayers() > 0 ? KvCache.create(kvDtype, arch.fullLayers(), g, ctx) : null;
        convState = new float[arch.linearLayers()][dn.convStateSize()];
        state = new float[arch.linearLayers()][dn.stateSize()];
        int dim = c.dim();
        x = new float[maxBatch * dim];
        xb = new float[maxBatch * dim];
        proj = new float[maxBatch * dn.projDim()];
        qkv = new float[maxBatch * dn.convDim()];
        o = new float[maxBatch * dn.valueWidth()];
        decay = new float[maxBatch * dn.valueHeads()];
        beta = new float[maxBatch * dn.valueHeads()];
        q = new float[maxBatch * g.qDim()];
        gate = new float[maxBatch * g.qDim()];
        kStage = new float[maxBatch * g.kvDim()];
        vStage = new float[maxBatch * g.kvDim()];
        att = new float[maxBatch * g.qDim()];
        gateUp = new float[maxBatch * 2 * arch.maxMlpDim()];
        hidden = new float[maxBatch * arch.maxMlpDim()];
        shared = arch.anyMoe() ? new float[maxBatch * dim] : null;
        int slots = maxBatch * arch.maxTopK();
        assign = arch.anyMoe() ? new int[slots] : null;
        routeWeight = arch.anyMoe() ? new float[slots] : null;
        int scratchLen = Math.max(Math.max(ctx, arch.maxExperts()), 4 * CpuMatmul.maxCols(w));
        mm = new CpuMatmul(pool, int8Activations, CpuBackend.DETERMINISTIC, scratchLen);
        experts = arch.anyMoe()
                ? new MoeExperts(pool, mm, dim, arch.maxExperts(), arch.maxTopK(), arch.maxMoeDim(), maxBatch) : null;
        setRecurrence(CHUNKED, CHUNK);
        if (ctx > c.maxPositions()) {
            System.err.println("warning: context " + ctx + " exceeds the model's max_position_embeddings ("
                    + c.maxPositions() + "); RoPE positions beyond the training length are extrapolated");
        }
    }

    /**
     * How chunks of several tokens run the delta rule: the chunked form with chunks of {@code chunk} tokens, or (with
     * {@code chunked} false) the token loop that decode uses. For comparisons and benchmarks; the default comes from
     * {@code -Djllm.gdn.chunked} / {@code -Djllm.gdn.chunk}.
     */
    public void setRecurrence(boolean chunked, int chunk) {
        int len = Math.max(1, Math.min(chunk, maxBatch));
        chunkedPrefill = chunked;
        chunkBuf = ThreadLocal.withInitial(() -> new ChunkBuf(len, dn.valueDim()));
    }

    @Override
    public String name() {
        return "cpu qwen3-next (" + VectorKernels.describe() + (mm.int8() ? ", " + Int8Kernels.describe() : "")
                + (CpuBackend.DETERMINISTIC ? ", deterministic" : "") + ", " + pool.threads() + " threads)";
    }

    @Override
    public int contextLength() {
        return ctx;
    }

    @Override
    public int maxBatch() {
        return maxBatch;
    }

    /** The recurrent state of the DeltaNet layers cannot be rewound to an earlier position. */
    @Override
    public boolean supportsPrefixReuse() {
        return false;
    }

    @Override
    public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
        run(tokens, off, n, pos, logits, 0);
    }

    @Override
    public void forwardAll(int[] tokens, int off, int n, int pos, float[] logitsOut, int stride) {
        run(tokens, off, n, pos, logitsOut, stride);
    }

    private void run(int[] tokens, int off, int n, int pos, float[] logits, int stride) {
        if (n < 1 || n > maxBatch || pos + n > ctx) {
            throw new IllegalArgumentException("bad forward(n=" + n + ", pos=" + pos + ")");
        }
        if (pos == 0) {
            for (float[] s : convState) {
                Arrays.fill(s, 0f);
            }
            for (float[] s : state) {
                Arrays.fill(s, 0f);
            }
        } else if (pos != statePos) {
            throw new IllegalStateException("Qwen3-Next state holds " + statePos + " tokens, cannot continue at " + pos);
        }
        statePos = -1;
        int dim = c.dim();
        for (int t = 0; t < n; t++) {
            w.embed.dequantizeRow(tokens[off + t], x, t * dim);
        }
        for (int l = 0; l < c.layers(); l++) {
            Qwen3NextLayer L = w.qwen3NextLayer(l);
            rmsnorm(x, xb, L.attnNorm(), n, dim);
            switch (L.mixer()) {
                case Qwen3NextLayer.DeltaNet d -> deltaNet(d, arch.kindIndex(l), n);
                case Qwen3NextLayer.GatedAttention a -> attention(a, arch.kindIndex(l), n, pos);
            }
            rmsnorm(x, xb, L.ffnNorm(), n, dim);
            switch (L.mlp()) {
                case MlpWeights.Dense d -> {
                    int h = ((Mlp.Dense) arch.mlps().get(l)).hidden();
                    mm.mul(d.gateUp(), xb, gateUp, 0, n, false);
                    siluMul(n, h);
                    mm.mul(d.down(), hidden, x, 0, n, true);
                }
                case MlpWeights.Moe m -> moe((Mlp.Moe) arch.mlps().get(l), m, L.sharedGate(), n);
            }
        }
        if (logits == null) {
            statePos = pos + n;
            return;
        }
        if (stride == 0) {
            VectorKernels.rmsnormRow(x, (n - 1) * dim, xb, 0, w.finalNorm, dim, c.normEps());
            mm.mul(w.lmHead, Math.min(w.lmHead.rows, logits.length), xb, logits, 0, 1, false);
        } else {
            rmsnorm(x, xb, w.finalNorm, n, dim);
            int rows = Math.min(w.lmHead.rows, stride);
            float[] tmp = w.lmHead.rows == stride ? logits : new float[n * w.lmHead.rows];
            mm.mul(w.lmHead, rows, xb, tmp, 0, n, false);
            if (tmp != logits) {
                for (int i = 0; i < n; i++) {
                    System.arraycopy(tmp, i * w.lmHead.rows, logits, i * stride, rows);
                }
            }
        }
        statePos = pos + n;
    }

    // ------------------------------------------------------------------ Gated DeltaNet

    /**
     * One Gated DeltaNet layer over the {@code n} normalized rows of {@link #xb}, added to {@link #x}:
     * <pre>
     *   [q k v | z | b a] = in_proj · xb
     *   [q k v]           = silu(causal_conv1d([q k v]))           conv state: the last convKernel-1 inputs
     *   q = l2norm(q) / sqrt(keyDim),  k = l2norm(k),  beta = sigmoid(b),  g = A · softplus(a + dt_bias)  (A &lt; 0)
     *   S = exp(g) S;  S += k ((v - S^T k) beta)^T;  o = S^T q    per value head (k and q of its key head)
     *   x += out_proj · (rmsnorm(o) * norm_weight * silu(z))       per value head
     * </pre>
     */
    private void deltaNet(Qwen3NextLayer.DeltaNet d, int li, int n) {
        int pd = dn.projDim();
        int cd = dn.convDim();
        int kc = dn.convKernel();
        int nv = dn.valueHeads();
        int nk = dn.keyHeads();
        int dk = dn.keyDim();
        mm.mul(d.inProj(), xb, proj, 0, n, false);
        float[] cs = convState[li];
        float[] cw = d.conv();
        pool.forRange(cd, 32, (c0, c1) -> {
            for (int ch = c0; ch < c1; ch++) {
                for (int t = 0; t < n; t++) {
                    qkv[t * cd + ch] = MambaCpuBackend.convStep(cw, ch * kc, 0f, cs, ch * (kc - 1), kc, proj[t * pd + ch]);
                }
            }
        });
        // q / k L2 normalization (q also scaled by keyDim^-0.5), beta and the decay of every token and head
        float qScale = (float) (1.0 / Math.sqrt(dk));
        int kw = dn.keyWidth();
        int bOff = cd + dn.valueWidth();
        pool.forRange(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                int r = t * cd;
                for (int h = 0; h < nk; h++) {
                    l2norm(qkv, r + h * dk, dk, qScale);
                    l2norm(qkv, r + kw + h * dk, dk, 1f);
                }
                for (int h = 0; h < nv; h++) {
                    float b = proj[t * pd + bOff + h];
                    float a = proj[t * pd + bOff + nv + h];
                    beta[t * nv + h] = sigmoid(b);
                    decay[t * nv + h] = d.a()[h] * MambaCpuBackend.softplus(a + d.dtBias()[h]);
                }
            }
        });
        if (n > 1 && chunkedPrefill && !CpuBackend.DETERMINISTIC) {
            pool.forRange(nv, 1, (h0, h1) -> {
                ChunkBuf buf = chunkBuf.get();
                for (int h = h0; h < h1; h++) {
                    for (int t0 = 0; t0 < n; t0 += buf.chunk) {
                        chunked(state[li], h, t0, Math.min(buf.chunk, n - t0), buf);
                    }
                }
            });
        } else {
            recurrent(state[li], n);
        }
        // gated RMSNorm per (token, value head): o = rmsnorm(o) * w * silu(z)
        int dv = dn.valueDim();
        int vw = dn.valueWidth();
        float[] nw = d.norm();
        pool.forRange(n * nv, 4, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int t = i / nv;
                int h = i - t * nv;
                int oo = t * vw + h * dv;
                int zo = t * pd + cd + h * dv;
                VectorKernels.rmsnormRow(o, oo, o, oo, nw, dv, c.normEps());
                for (int j = 0; j < dv; j++) {
                    o[oo + j] *= VectorKernels.silu(proj[zo + j]);
                }
            }
        });
        mm.mul(d.outProj(), o, x, 0, n, true);
    }

    /** {@code v[o .. o + n) *= scale / sqrt(Σ v² + eps)}: transformers' {@code l2norm} (eps inside the root), then a scale. */
    static void l2norm(float[] v, int off, int n, float scale) {
        float ss = VectorKernels.sumSquares(v, off, n);
        float inv = (float) (1.0 / Math.sqrt(ss + L2_EPS));
        for (int i = 0; i < n; i++) {
            v[off + i] = v[off + i] * inv * scale;
        }
    }

    static float sigmoid(float v) {
        return 1f / (1f + (float) Math.exp(-v));
    }

    /**
     * The delta rule token by token, parallel over (value head, value dim j): every column {@code S[h][j]} (keyDim
     * floats) is advanced independently: {@code s = exp(g) s; d = (v_j - s.k) beta; s += d k; o_j = s.q}.
     */
    private void recurrent(float[] s, int n) {
        int nv = dn.valueHeads();
        int dk = dn.keyDim();
        int dv = dn.valueDim();
        int cd = dn.convDim();
        int kw = dn.keyWidth();
        int group = dn.group();
        int vw = dn.valueWidth();
        pool.forRange(nv * dv, 8, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int h = i / dv;
                int j = i - h * dv;
                int so = i * dk; // = (h * dv + j) * dk
                int kh = h / group;
                for (int t = 0; t < n; t++) {
                    recurrentStep(s, so, dk, qkv, t * cd + kw + kh * dk, qkv, t * cd + kh * dk,
                            qkv[t * cd + 2 * kw + h * dv + j], (float) Math.exp(decay[t * nv + h]), beta[t * nv + h],
                            o, t * vw + h * dv + j);
                }
            }
        });
    }

    /**
     * One token of one state column {@code s[so .. so + dk)}: {@code s *= gamma; d = (v - s.k) beta; s += d k;
     * out[oo] = s.q}, with k at {@code kv[ko..]} and q at {@code qv[qo..]}.
     */
    public static void recurrentStep(float[] s, int so, int dk, float[] kv, int ko, float[] qv, int qo, float v, float gamma,
                              float beta, float[] out, int oo) {
        VectorKernels.scale(s, so, dk, gamma);
        float kvMem = VectorKernels.dot(s, so, kv, ko, dk);
        float delta = (v - kvMem) * beta;
        VectorKernels.axpy(delta, kv, ko, s, so, dk);
        out[oo] = VectorKernels.dot(s, so, qv, qo, dk);
    }

    /** Per-thread scratch of {@link #chunked}: {@code chunk} tokens of one value head. */
    private static final class ChunkBuf {
        final int chunk;
        final float[] cum;    // [L]       cumulative log decay inside the chunk
        final float[] ws;     // [L, dv]   S0^T k_t
        final float[] qs;     // [L, dv]   S0^T q_t
        final float[] kk;     // [L, L]    k_t . k_u
        final float[] qk;     // [L, L]    q_t . k_u
        final float[] delta;  // [L, dv]   the corrections written to the state ("new values" minus the old prediction)

        ChunkBuf(int chunk, int dv) {
            this.chunk = chunk;
            cum = new float[chunk];
            ws = new float[chunk * dv];
            qs = new float[chunk * dv];
            kk = new float[chunk * chunk];
            qk = new float[chunk * chunk];
            delta = new float[chunk * dv];
        }
    }

    /**
     * The delta rule over the {@code len} tokens {@code [t0, t0 + len)} of value head {@code h} in the chunked form
     * (docs/14.14): with {@code c_t} the cumulative log decay inside the chunk and {@code S0} the state before it,
     * <pre>
     *   d_t = beta_t (v_t - e^{c_t} S0^T k_t - Σ_{u&lt;t} e^{c_t - c_u} (k_t.k_u) d_u)     forward substitution (the UT / WY system)
     *   o_t = e^{c_t} S0^T q_t + Σ_{u&lt;=t} e^{c_t - c_u} (q_t.k_u) d_u
     *   S   = e^{c_L} S0 + Σ_u e^{c_L - c_u} k_u d_u^T
     * </pre>
     * which is the token recurrence rearranged, so the outputs and the final state agree with it to float rounding.
     */
    private void chunked(float[] s, int h, int t0, int len, ChunkBuf b) {
        int nv = dn.valueHeads();
        int dk = dn.keyDim();
        int dv = dn.valueDim();
        int cd = dn.convDim();
        int kw = dn.keyWidth();
        int vw = dn.valueWidth();
        int kh = h / dn.group();
        int sBase = h * dv * dk;
        float acc = 0;
        for (int u = 0; u < len; u++) {
            acc += decay[(t0 + u) * nv + h];
            b.cum[u] = acc;
        }
        for (int u = 0; u < len; u++) {
            int ko = (t0 + u) * cd + kw + kh * dk;
            int qo = (t0 + u) * cd + kh * dk;
            for (int j = 0; j < dv; j++) {
                int so = sBase + j * dk;
                b.ws[u * dv + j] = VectorKernels.dot(s, so, qkv, ko, dk);
                b.qs[u * dv + j] = VectorKernels.dot(s, so, qkv, qo, dk);
            }
            for (int v = 0; v <= u; v++) {
                int kv = (t0 + v) * cd + kw + kh * dk;
                b.qk[u * len + v] = VectorKernels.dot(qkv, qo, qkv, kv, dk);
                if (v < u) {
                    b.kk[u * len + v] = VectorKernels.dot(qkv, ko, qkv, kv, dk);
                }
            }
        }
        for (int u = 0; u < len; u++) {
            int t = t0 + u;
            float bt = beta[t * nv + h];
            float et = (float) Math.exp(b.cum[u]);
            int vo = t * cd + 2 * kw + h * dv;
            int dOff = u * dv;
            for (int j = 0; j < dv; j++) {
                b.delta[dOff + j] = qkv[vo + j] - et * b.ws[dOff + j];
            }
            for (int v = 0; v < u; v++) {
                float coef = (float) Math.exp(b.cum[u] - b.cum[v]) * b.kk[u * len + v];
                VectorKernels.axpy(-coef, b.delta, v * dv, b.delta, dOff, dv);
            }
            VectorKernels.scale(b.delta, dOff, dv, bt);
            // output: the decayed old state read with q, plus the chunk's own corrections up to and including t
            int oo = t * vw + h * dv;
            for (int j = 0; j < dv; j++) {
                o[oo + j] = et * b.qs[dOff + j];
            }
            for (int v = 0; v <= u; v++) {
                float coef = (float) Math.exp(b.cum[u] - b.cum[v]) * b.qk[u * len + v];
                VectorKernels.axpy(coef, b.delta, v * dv, o, oo, dv);
            }
        }
        float cl = b.cum[len - 1];
        float eL = (float) Math.exp(cl);
        for (int j = 0; j < dv; j++) {
            int so = sBase + j * dk;
            VectorKernels.scale(s, so, dk, eL);
            for (int u = 0; u < len; u++) {
                float coef = (float) Math.exp(cl - b.cum[u]) * b.delta[u * dv + j];
                VectorKernels.axpy(coef, qkv, (t0 + u) * cd + kw + kh * dk, s, so, dk);
            }
        }
    }

    // ------------------------------------------------------------------ gated full attention

    /**
     * q / gate / k / v projections, per-head RMSNorm of q and k, RoPE on the first rotaryDim dims of every head, the
     * KV cache of this full-attention layer ({@code fi}-th), attention, {@code att *= sigmoid(gate)}, {@code x += o_proj . att}.
     */
    private void attention(Qwen3NextLayer.GatedAttention a, int fi, int n, int pos) {
        Attention.Gqa g = arch.attention();
        AttnWeights.Gqa wg = a.gqa();
        mm.mul(wg.wq(), xb, q, 0, n, false);
        mm.mul(a.gate(), xb, gate, 0, n, false);
        mm.mul(wg.wk(), xb, kStage, 0, n, false);
        mm.mul(wg.wv(), xb, vStage, 0, n, false);
        int hd = g.headDim();
        int half = arch.rope().rotaryDim() / 2;
        int qh = g.heads();
        int kh = g.kvHeads();
        pool.forRange(n * (qh + kh), 4, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int t = i / (qh + kh);
                int hh = i - t * (qh + kh);
                boolean isQ = hh < qh;
                float[] v = isQ ? q : kStage;
                int off = isQ ? t * g.qDim() + hh * hd : t * g.kvDim() + (hh - qh) * hd;
                VectorKernels.rmsnormRow(v, off, v, off, isQ ? wg.qNorm() : wg.kNorm(), hd, c.normEps());
                VectorKernels.rope(v, off, half, rope.cos, rope.sin, (pos + t) * half);
            }
        });
        kv.store(fi, pos, kStage, vStage, n);
        kv.attention(pool, fi, n, pos, q, att, 0, g.scale(), 0f);
        int qd = g.qDim();
        pool.forRange(n * qd, 1024, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                att[i] *= sigmoid(gate[i]);
            }
        });
        mm.mul(a.wo(), att, x, 0, n, true);
    }

    // ------------------------------------------------------------------ MLP

    /**
     * Routed experts (softmax router, top-k, optionally renormalized, as transformers' Qwen3NextTopKRouter) plus the
     * shared expert scaled by {@code sigmoid(shared_expert_gate . xb)}.
     */
    private void moe(Mlp.Moe spec, MlpWeights.Moe m, float[] sharedGate, int n) {
        int dim = c.dim();
        int topK = spec.topK();
        if (n > 1 && !CpuBackend.DETERMINISTIC) {
            pool.forRange(n, 1, (t0, t1) -> {
                float[] logits = mm.scratch.get();
                for (int t = t0; t < t1; t++) {
                    route(spec, m.router(), t, logits);
                }
            });
            experts.batched(m, spec.experts(), topK, spec.moeDim(), n, xb, assign, routeWeight, x);
        } else {
            float[] logits = mm.scratch.get();
            for (int t = 0; t < n; t++) {
                route(spec, m.router(), t, logits);
                experts.token(m, topK, spec.moeDim(), t, xb, assign, routeWeight, x);
            }
        }
        int sh = spec.shared().hidden();
        mm.mul(m.shared().gateUp(), xb, gateUp, 0, n, false);
        siluMul(n, sh);
        mm.mul(m.shared().down(), hidden, shared, 0, n, false);
        pool.forRange(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                float s = sigmoid(VectorKernels.dot(sharedGate, 0, xb, t * dim, dim));
                VectorKernels.axpy(s, shared, t * dim, x, t * dim, dim);
            }
        });
    }

    /** Softmax over the router logits of token {@code t}, the top-k, renormalized when the config says so. */
    private void route(Mlp.Moe spec, float[] router, int t, float[] logits) {
        int e = spec.experts();
        int topK = spec.topK();
        int dim = c.dim();
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < e; i++) {
            logits[i] = VectorKernels.dot(router, i * dim, xb, t * dim, dim);
            max = Math.max(max, logits[i]);
        }
        float sum = 0;
        for (int i = 0; i < e; i++) {
            float p = (float) Math.exp(logits[i] - max);
            logits[i] = p;
            sum += p;
        }
        int ao = t * topK;
        float selected = 0;
        for (int k = 0; k < topK; k++) {
            int best = -1;
            float bv = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < e; i++) {
                if (logits[i] > bv) {
                    bv = logits[i];
                    best = i;
                }
            }
            assign[ao + k] = best;
            routeWeight[ao + k] = bv / sum;
            selected += bv / sum;
            logits[best] = Float.NEGATIVE_INFINITY;
        }
        if (spec.normTopK() && selected > 0) {
            for (int k = 0; k < topK; k++) {
                routeWeight[ao + k] /= selected;
            }
        }
    }

    /** {@code hidden[t, i] = silu(gate[t, i]) * up[t, i]} over the stacked gate|up rows of width {@code 2 * hdim}. */
    private void siluMul(int n, int hdim) {
        pool.forRange(n * hdim, 512, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int t = i / hdim;
                int j = i - t * hdim;
                int gi = t * 2 * hdim;
                hidden[t * hdim + j] = VectorKernels.silu(gateUp[gi + j]) * gateUp[gi + hdim + j];
            }
        });
    }

    private void rmsnorm(float[] in, float[] out, float[] weight, int n, int width) {
        pool.forRange(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                VectorKernels.rmsnormRow(in, t * width, out, t * width, weight, width, c.normEps());
            }
        });
    }

    // ------------------------------------------------------------------ saved state

    /**
     * The KV rows of the full-attention layers ({@link KvCache#format}), then per DeltaNet layer the conv and the
     * recurrent state as f32. The same bytes as {@code Qwen3NextMetalBackend} writes.
     */
    @Override
    public String stateFormat() {
        return stateFormat(arch, kv == null ? "none" : kv.format());
    }

    /** The format string both Qwen3-Next backends report: the KV layout plus the recurrent state's shape. */
    public static String stateFormat(Qwen3NextArch arch, String kvFormat) {
        Qwen3NextArch.DeltaNet d = arch.deltaNet();
        return "qwen3next kv[" + kvFormat + "] gdn layers=" + arch.linearLayers() + " conv=" + d.convDim() + "x"
                + (d.convKernel() - 1) + " state=" + d.valueHeads() + "x" + d.valueDim() + "x" + d.keyDim() + " f32";
    }

    private void checkPositions(int n) {
        if (n != statePos) {
            throw new IllegalStateException("the Qwen3-Next state holds " + statePos + " tokens, not " + n
                    + " (a recurrent state cannot be cut)");
        }
    }

    @Override
    public void saveState(int n, WritableByteChannel out) throws IOException {
        checkPositions(n);
        if (kv != null) {
            kv.write(n, out);
        }
        for (int l = 0; l < convState.length; l++) {
            writeFloats(convState[l], out);
            writeFloats(state[l], out);
        }
    }

    @Override
    public void loadState(int n, ReadableByteChannel in) throws IOException {
        statePos = -1;
        if (kv != null) {
            kv.read(n, in);
        }
        for (int l = 0; l < convState.length; l++) {
            readFloats(convState[l], in);
            readFloats(state[l], in);
        }
        statePos = n;
    }

    @Override
    public void copyStateFrom(Backend other, int n) {
        if (!(other instanceof Qwen3NextCpuBackend o) || !stateFormat().equals(o.stateFormat())) {
            Backend.super.copyStateFrom(other, n);
            return;
        }
        o.checkPositions(n);
        statePos = -1;
        if (kv != null) {
            kv.copyFrom(o.kv, n);
        }
        for (int l = 0; l < convState.length; l++) {
            System.arraycopy(o.convState[l], 0, convState[l], 0, convState[l].length);
            System.arraycopy(o.state[l], 0, state[l], 0, state[l].length);
        }
        statePos = n;
    }

    static void writeFloats(float[] a, WritableByteChannel out) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(Math.min(a.length, 1 << 16) * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < a.length; ) {
            b.clear();
            int m = Math.min(a.length - i, b.capacity() / 4);
            b.asFloatBuffer().put(a, i, m);
            b.limit(m * 4);
            while (b.hasRemaining()) {
                out.write(b);
            }
            i += m;
        }
    }

    static void readFloats(float[] a, ReadableByteChannel in) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(Math.min(a.length, 1 << 16) * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < a.length; ) {
            b.clear();
            int m = Math.min(a.length - i, b.capacity() / 4);
            b.limit(m * 4);
            while (b.hasRemaining()) {
                if (in.read(b) < 0) {
                    throw new java.io.EOFException("saved Qwen3-Next state is truncated");
                }
            }
            b.flip();
            b.asFloatBuffer().get(a, i, m);
            i += m;
        }
    }

    @Override
    public void close() {
        pool.close();
    }
}
