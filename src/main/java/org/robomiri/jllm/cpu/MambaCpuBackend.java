package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.MambaArch;
import org.robomiri.jllm.model.MambaLayer;
import org.robomiri.jllm.model.ModelWeights;

import java.util.Arrays;

/**
 * Mamba (v1, {@code MambaForCausalLM}) forward pass on the CPU. There is no attention and no KV cache: every layer
 * keeps a fixed-size recurrent state, a conv state {@code [d_inner, d_conv-1]} and an SSM state
 * {@code [d_inner, d_state]} (both f32), that is advanced token by token.
 *
 * <p>Per layer and chunk of {@code n} tokens (see docs/15-mamba.md): RMSNorm, {@code in_proj} (quantized matmul over
 * the whole chunk), causal conv + SiLU, {@code x_proj} (matmul), then the selective scan, and {@code out_proj} added
 * to the residual. The matmuls are {@link CpuMatmul}, the same code as the transformer backend's, including its
 * int8 single-token path. The conv and the scan are sequential over the tokens of the chunk but independent per channel,
 * so they are parallelized over {@code d_inner}; the dot products inside use the Vector API. The scan's
 * {@code exp} is scalar ({@code d_state} = 16 values per channel and token).
 *
 * <p>The state belongs to one sequence: {@code pos == 0} starts a new one, any other {@code pos} must equal the
 * number of tokens fed so far (the engine restarts at 0 whenever it cannot extend the previous sequence).
 */
public final class MambaCpuBackend implements Backend {

    private final ModelWeights w;
    private final Architecture.Core c;
    private final MambaArch m;
    private final int ctx;
    private final int maxBatch;
    private final ThreadPool pool;

    private final float[][] convState;
    private final float[][] ssmState;
    /** Tokens fed into the current state, -1 while a forward is in flight or after one failed. */
    private int statePos = 0;

    private final float[] x;
    private final float[] xb;
    private final float[] xz;
    private final float[] xc;
    private final float[] dbc;
    private final float[] y;
    private final CpuMatmul mm;

    /** The exact float path (int8 activations off), as the CPU-vs-Metal checks need it. */
    public MambaCpuBackend(ModelWeights w, int ctx, int maxBatch, int threads) {
        this(w, ctx, maxBatch, threads, false);
    }

    /**
     * @param int8Activations int8 integer dot products for the single-token ({@code n == 1}) projections, as in
     *                        {@link CpuBackend}; the chunked prefill keeps the float path
     */
    public MambaCpuBackend(ModelWeights w, int ctx, int maxBatch, int threads, boolean int8Activations) {
        this.w = w;
        this.m = w.mambaArch();
        this.c = m.core();
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        this.pool = new ThreadPool(threads);
        int di = m.dInner();
        convState = new float[c.layers()][di * (m.dConv() - 1)];
        ssmState = new float[c.layers()][di * m.dState()];
        x = new float[maxBatch * c.dim()];
        xb = new float[maxBatch * c.dim()];
        xz = new float[maxBatch * 2 * di];
        xc = new float[maxBatch * di];
        dbc = new float[maxBatch * (m.dtRank() + 2 * m.dState())];
        y = new float[maxBatch * di];
        mm = new CpuMatmul(pool, int8Activations, CpuBackend.DETERMINISTIC, 4 * CpuMatmul.maxCols(w));
    }

    @Override
    public String name() {
        return "cpu mamba (" + VectorKernels.describe() + (mm.int8() ? ", " + Int8Kernels.describe() : "")
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

    /** The recurrent state cannot be rewound to an earlier position. */
    @Override
    public boolean supportsPrefixReuse() {
        return false;
    }

    @Override
    public void forward(int[] tokens, int off, int n, int pos, float[] logits) {
        if (n < 1 || n > maxBatch || pos + n > ctx) {
            throw new IllegalArgumentException("bad forward(n=" + n + ", pos=" + pos + ")");
        }
        if (pos == 0) {
            for (int l = 0; l < c.layers(); l++) {
                Arrays.fill(convState[l], 0f);
                Arrays.fill(ssmState[l], 0f);
            }
        } else if (pos != statePos) {
            throw new IllegalStateException("Mamba state holds " + statePos + " tokens, cannot continue at " + pos);
        }
        statePos = -1;
        int dim = c.dim();
        for (int t = 0; t < n; t++) {
            w.embed.dequantizeRow(tokens[off + t], x, t * dim);
        }
        for (int l = 0; l < c.layers(); l++) {
            MambaLayer L = w.mambaLayer(l);
            pool.forRange(n, 1, (t0, t1) -> {
                for (int t = t0; t < t1; t++) {
                    rmsnormRow(x, t * dim, xb, t * dim, L.norm(), dim);
                }
            });
            mm.mul(L.inProj(), xb, xz, 0, n, false);
            conv(L, l, n);
            mm.mul(L.xProj(), xc, dbc, 0, n, false);
            scan(L, l, n);
            mm.mul(L.outProj(), y, x, 0, n, true);
        }
        if (logits != null) {
            rmsnormRow(x, (n - 1) * dim, xb, 0, w.finalNorm, dim);
            // the caller sizes logits by cfg.vocabSize(); the LM head may have fewer (or, padded, more) rows
            mm.mul(w.lmHead, Math.min(w.lmHead.rows, logits.length), xb, logits, 0, 1, false);
        }
        statePos = pos + n;
    }

    // ------------------------------------------------------------------ mixer

    /**
     * Depthwise causal conv with the state of the previous {@code d_conv-1} inputs, then SiLU. Reads the x half of
     * {@code xz}, writes {@code xc}; the state is advanced past all {@code n} tokens.
     */
    private void conv(MambaLayer L, int layer, int n) {
        int di = m.dInner();
        int dc = m.dConv();
        float[] st = convState[layer];
        pool.forRange(di, 32, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                for (int t = 0; t < n; t++) {
                    xc[t * di + i] = convStep(L.conv(), i * dc, L.convBias()[i], st, i * (dc - 1), dc, xz[t * 2 * di + i]);
                }
            }
        });
    }

    /**
     * One token of one channel of the causal conv: the window is the {@code dc-1} states (oldest first) followed by
     * {@code u}; returns {@code silu(Σ w[k]·window[k] + bias)} and shifts {@code u} into the state.
     */
    public static float convStep(float[] weight, int wo, float bias, float[] state, int so, int dc, float u) {
        float sum = bias;
        for (int k = 0; k < dc - 1; k++) {
            sum += weight[wo + k] * state[so + k];
        }
        sum += weight[wo + dc - 1] * u;
        for (int k = 0; k < dc - 2; k++) {
            state[so + k] = state[so + k + 1];
        }
        if (dc > 1) {
            state[so + dc - 2] = u;
        }
        return VectorKernels.silu(sum);
    }

    /**
     * Selective scan, one channel at a time: {@code dt = softplus(dt_proj[i]·dt_in + bias)};
     * {@code s[n] = exp(dt·A[i,n])·s[n] + dt·B[n]·u}; {@code y = (Σ s[n]·C[n] + D·u)·silu(z)}.
     */
    private void scan(MambaLayer L, int layer, int n) {
        int di = m.dInner();
        int ds = m.dState();
        int rank = m.dtRank();
        int stride = rank + 2 * ds;
        float[] s = ssmState[layer];
        pool.forRange(di, 16, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int so = i * ds;
                for (int t = 0; t < n; t++) {
                    int bo = t * stride + rank;
                    float raw = VectorKernels.dot(L.dtProj(), i * rank, dbc, t * stride, rank) + L.dtBias()[i];
                    float dt = softplus(raw);
                    float u = xc[t * di + i];
                    float acc = scanStep(dt, u, L.a(), so, s, so, dbc, bo, ds);
                    float z = xz[t * 2 * di + di + i];
                    y[t * di + i] = (acc + L.d()[i] * u) * VectorKernels.silu(z);
                }
            }
        });
    }

    /**
     * One token of one channel of the selective scan over {@code ds} states: {@code s[k] = exp(dt·A[k])·s[k] +
     * dt·B[k]·u}, returning {@code Σ s[k]·C[k]} with {@code B = bc[bo..bo+ds)} and {@code C = bc[bo+ds..bo+2ds)}.
     */
    public static float scanStep(float dt, float u, float[] a, int ao, float[] s, int so, float[] bc, int bo, int ds) {
        float du = dt * u;
        float acc = 0f;
        for (int k = 0; k < ds; k++) {
            float sk = (float) Math.exp(dt * a[ao + k]) * s[so + k] + du * bc[bo + k];
            s[so + k] = sk;
            acc += sk * bc[bo + ds + k];
        }
        return acc;
    }

    /** {@code log(1 + exp(x))}, linear above 20 like torch's {@code softplus}. */
    public static float softplus(float x) {
        return x > 20f ? x : (float) Math.log1p(Math.exp(x));
    }

    // ------------------------------------------------------------------ shared pieces

    private void rmsnormRow(float[] in, int io, float[] out, int oo, float[] weight, int width) {
        VectorKernels.rmsnormRow(in, io, out, oo, weight, width, c.normEps());
    }

    @Override
    public void close() {
        pool.close();
    }
}
