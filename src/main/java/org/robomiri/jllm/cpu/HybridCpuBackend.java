package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.Attention;
import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.HybridArch;
import org.robomiri.jllm.model.HybridLayer;
import org.robomiri.jllm.model.KvDtype;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.MlpWeights;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.Rope;
import org.robomiri.jllm.model.Router;
import org.robomiri.jllm.model.Scaling;

import java.util.Arrays;

/**
 * The attention / state-space hybrids on the CPU (docs/14.22): Granite 4, Nemotron-H and Jamba. Per layer and chunk of
 * {@code n} tokens: RMSNorm, the mixer, a residual add scaled by the layer's multiplier, then (if the layer has one) the
 * feed-forward block the same way. The projections are {@link CpuMatmul}, the same code as every other backend's; the
 * attention layers keep a {@link KvCache} (only they), the state-space layers a fixed-size recurrent state that is
 * advanced token by token and parallelized over heads / channels, as {@link MambaCpuBackend} does for Mamba-1.
 *
 * <p>Mamba-2 (the SSD recurrence, written as its token loop): {@code dt = clamp(softplus(dt_raw + bias))},
 * {@code S[h][p][n] = exp(dt A[h]) S[h][p][n] + dt x[h][p] B[g][n]}, {@code y[h][p] = Σ_n S[h][p][n] C[g][n] + D[h] x[h][p]}
 * with {@code g} the head's group, then {@code y' = rmsnorm(y silu(z)) w} over {@code dInner / normGroups} channels at a
 * time. The chunked (matrix) form of the SSD paper computes the same numbers; the loop is exact and as cheap for the
 * state sizes of these models.
 *
 * <p>The state belongs to one sequence like Mamba's: {@code pos == 0} starts a new one, any other {@code pos} must equal
 * the number of tokens fed so far; it cannot be rewound ({@link #supportsPrefixReuse} is false).
 */
public final class HybridCpuBackend implements Backend {

    private final ModelWeights w;
    private final HybridArch arch;
    private final Architecture.Core c;
    private final HybridArch.Mamba2 m2;
    private final HybridArch.Mamba1 m1;
    private final int ctx;
    private final int maxBatch;
    private final ThreadPool pool;
    private final CpuMatmul mm;
    private final KvCache kv;
    private final Rope rope;
    private final MoeExperts experts;

    private final float[][] convState;
    private final float[][] ssmState;
    /** Tokens fed into the current state, -1 while a forward is in flight or after one failed. */
    private int statePos = 0;

    // activations, [maxBatch, width] row-major
    private final float[] x;
    private final float[] xb;
    private final float[] mixOut;   // a mixer's output projection, before the residual multiplier
    private final float[] ffnOut;
    // Mamba-2 / Mamba-1 scratch
    private final float[] proj;     // in_proj output: Mamba-2 [z | x B C | dt]; Mamba-1 [x | z]
    private final float[] conv;     // the conv + SiLU output
    private final float[] dbc;      // Mamba-1: x_proj output
    private final float[] y;        // the mixer output before the output projection
    // attention scratch
    private final float[] q;
    private final float[] att;
    private final float[] kStage;
    private final float[] vStage;
    // feed-forward scratch
    private final float[] gateUp;
    private final float[] hidden;
    private final int[] assign;
    private final float[] routeWeight;

    /** The exact float path (int8 activations off), as the CPU-vs-Metal checks need it. */
    public HybridCpuBackend(ModelWeights w, int ctx, int maxBatch, int threads, KvDtype kvDtype) {
        this(w, ctx, maxBatch, threads, kvDtype, false);
    }

    public HybridCpuBackend(ModelWeights w, int ctx, int maxBatch, int threads, KvDtype kvDtype, boolean int8Activations) {
        this.w = w;
        this.arch = w.hybrid();
        this.c = arch.core();
        this.m2 = arch.mamba2();
        this.m1 = arch.mamba1();
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        this.pool = new ThreadPool(threads);
        int dim = c.dim();
        Attention.Gqa g = arch.attention();
        if (g != null) {
            kv = KvCache.create(kvDtype, arch.attentionLayers(), g, ctx);
            rope = arch.rope().scaling() instanceof Scaling.Identity ? null : new Rope(arch.rope(), ctx);
            q = new float[maxBatch * g.qDim()];
            att = new float[maxBatch * g.qDim()];
            kStage = new float[maxBatch * g.kvDim()];
            vStage = new float[maxBatch * g.kvDim()];
        } else {
            kv = null;
            rope = null;
            q = att = kStage = vStage = null;
        }
        int layers = c.layers();
        int[] convSizes = new int[layers];
        int[] ssmSizes = new int[layers];
        for (int l = 0; l < layers; l++) {
            switch (arch.layers().get(l).mixer()) {
                case MAMBA2 -> {
                    convSizes[l] = m2.convStateSize();
                    ssmSizes[l] = m2.ssmStateSize();
                }
                case MAMBA1 -> {
                    convSizes[l] = m1.convStateSize();
                    ssmSizes[l] = m1.ssmStateSize();
                }
                default -> {
                }
            }
        }
        convState = new float[layers][];
        ssmState = new float[layers][];
        for (int l = 0; l < layers; l++) {
            convState[l] = new float[convSizes[l]];
            ssmState[l] = new float[ssmSizes[l]];
        }
        x = new float[maxBatch * dim];
        xb = new float[maxBatch * dim];
        mixOut = new float[maxBatch * dim];
        ffnOut = new float[maxBatch * dim];
        int projWidth = Math.max(m2 != null ? m2.projDim() : 0, m1 != null ? 2 * m1.dInner() : 0);
        proj = new float[maxBatch * Math.max(1, projWidth)];
        conv = new float[maxBatch * Math.max(1, Math.max(m2 != null ? m2.convDim() : 0, m1 != null ? m1.dInner() : 0))];
        dbc = m1 != null ? new float[maxBatch * (m1.dtRank() + 2 * m1.state())] : null;
        y = new float[maxBatch * Math.max(1, arch.maxMixerWidth())];
        gateUp = new float[maxBatch * 2 * Math.max(1, arch.maxMlpDim())];
        hidden = new float[maxBatch * Math.max(1, arch.maxMlpDim())];
        int slots = maxBatch * Math.max(1, arch.maxTopK());
        assign = arch.anyMoe() ? new int[slots] : null;
        routeWeight = arch.anyMoe() ? new float[slots] : null;
        int scratchLen = Math.max(Math.max(ctx, arch.maxExperts()), 4 * CpuMatmul.maxCols(w));
        mm = new CpuMatmul(pool, int8Activations, CpuBackend.DETERMINISTIC, scratchLen);
        experts = arch.anyMoe()
                ? new MoeExperts(pool, mm, dim, arch.maxExperts(), arch.maxTopK(), arch.maxMoeDim(), maxBatch) : null;
    }

    @Override
    public String name() {
        return "cpu hybrid (" + VectorKernels.describe() + (mm.int8() ? ", " + mm.describeInt8() : "")
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

    /** The recurrent state of the state-space layers cannot be rewound to an earlier position. */
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
            for (int l = 0; l < c.layers(); l++) {
                Arrays.fill(convState[l], 0f);
                Arrays.fill(ssmState[l], 0f);
            }
        } else if (pos != statePos) {
            throw new IllegalStateException("the hybrid state holds " + statePos + " tokens, cannot continue at " + pos);
        }
        statePos = -1;
        int dim = c.dim();
        for (int t = 0; t < n; t++) {
            w.embed.dequantizeRow(tokens[off + t], x, t * dim);
        }
        if (c.embedScale() != 1f) {
            float es = c.embedScale();
            for (int i = 0; i < n * dim; i++) {
                x[i] *= es;
            }
        }
        for (int l = 0; l < c.layers(); l++) {
            HybridArch.Layer spec = arch.layers().get(l);
            HybridLayer L = w.hybridLayer(l);
            if (L.mixer() != null) {
                rmsnorm(x, xb, L.norm1(), n, dim);
                switch (L.mixer()) {
                    case HybridLayer.Attn a -> attention(a, arch.kindIndex(l), n, pos);
                    case HybridLayer.M2 m -> mamba2(m, l, n);
                    case HybridLayer.M1 m -> mamba1(m, l, n);
                }
                addScaled(spec.residual(), mixOut, n);
            }
            if (L.mlp() != null) {
                rmsnorm(x, xb, L.mixer() != null ? L.norm2() : L.norm1(), n, dim);
                Arrays.fill(ffnOut, 0, n * dim, 0f);
                ffn(spec.ffn(), L.mlp(), n);
                addScaled(spec.residual(), ffnOut, n);
            }
        }
        if (logits != null) {
            int rows = Math.min(w.lmHead.rows, stride == 0 ? logits.length : stride);
            if (stride == 0) {
                VectorKernels.rmsnormRow(x, (n - 1) * dim, xb, 0, w.finalNorm, dim, c.normEps());
                mm.mul(w.lmHead, rows, xb, logits, 0, 1, false);
                divide(logits, 0, rows);
            } else {
                rmsnorm(x, xb, w.finalNorm, n, dim);
                float[] tmp = w.lmHead.rows == stride ? logits : new float[n * w.lmHead.rows];
                mm.mul(w.lmHead, rows, xb, tmp, 0, n, false);
                if (tmp != logits) {
                    for (int i = 0; i < n; i++) {
                        System.arraycopy(tmp, i * w.lmHead.rows, logits, i * stride, rows);
                    }
                }
                for (int i = 0; i < n; i++) {
                    divide(logits, i * stride, rows);
                }
            }
        }
        statePos = pos + n;
    }

    /** Granite's {@code logits / logits_scaling}. */
    private void divide(float[] logits, int base, int rows) {
        float d = arch.logitsDivisor();
        if (d != 1f) {
            for (int i = base; i < base + rows; i++) {
                logits[i] /= d;
            }
        }
    }

    /** {@code x[0, n·dim) += r * src}. */
    private void addScaled(float r, float[] src, int n) {
        int dim = c.dim();
        pool.forRange(n, 1, (t0, t1) -> {
            for (int i = t0 * dim, e = t1 * dim; i < e; i++) {
                x[i] += r == 1f ? src[i] : r * src[i];
            }
        });
    }

    // ------------------------------------------------------------------ attention

    /** q / k / v projections, RoPE if the model has one, the KV cache of this attention layer, attention, o_proj. */
    private void attention(HybridLayer.Attn a, int ai, int n, int pos) {
        Attention.Gqa g = arch.attention();
        mm.mul(a.gqa().wq(), xb, q, 0, n, false);
        mm.mul(a.gqa().wk(), xb, kStage, 0, n, false);
        mm.mul(a.gqa().wv(), xb, vStage, 0, n, false);
        if (rope != null) {
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
                    int o = isQ ? t * g.qDim() + hh * hd : t * g.kvDim() + (hh - qh) * hd;
                    VectorKernels.rope(v, o, half, rope.cos, rope.sin, (pos + t) * half);
                }
            });
        }
        kv.store(ai, pos, kStage, vStage, n);
        kv.attention(pool, ai, n, pos, q, att, 0, g.scale(), 0f);
        mm.mul(a.wo(), att, mixOut, 0, n, false);
    }

    // ------------------------------------------------------------------ Mamba-2

    private void mamba2(HybridLayer.M2 m, int layer, int n) {
        int di = m2.dInner();
        int cd = m2.convDim();
        int pw = m2.projDim();
        int k = m2.convKernel();
        int heads = m2.heads();
        int hp = m2.headDim();
        int ns = m2.state();
        int groups = m2.groups();
        int perGroup = heads / groups;
        mm.mul(m.inProj(), xb, proj, 0, n, false);
        // depthwise causal conv + SiLU over [x | B | C]
        float[] st = convState[layer];
        pool.forRange(cd, 32, (i0, i1) -> {
            for (int ch = i0; ch < i1; ch++) {
                for (int t = 0; t < n; t++) {
                    conv[t * cd + ch] = MambaCpuBackend.convStep(m.conv(), ch * k, m.convBias()[ch], st, ch * (k - 1), k,
                            proj[t * pw + di + ch]);
                }
            }
        });
        // the recurrence, per head
        float[] ss = ssmState[layer];
        float dtMin = m2.dtMin();
        float dtMax = m2.dtMax();
        pool.forRange(heads, 1, (h0, h1) -> {
            for (int h = h0; h < h1; h++) {
                int gOff = h / perGroup;
                for (int t = 0; t < n; t++) {
                    float dt = MambaCpuBackend.softplus(proj[t * pw + di + cd + h] + m.dtBias()[h]);
                    dt = Math.min(Math.max(dt, dtMin), dtMax);
                    float decayExp = dt * m.a()[h];
                    float dA = (float) Math.exp(decayExp);
                    int bo = t * cd + di + gOff * ns;
                    int co = t * cd + di + groups * ns + gOff * ns;
                    for (int p = 0; p < hp; p++) {
                        float xv = conv[t * cd + h * hp + p];
                        float dtx = dt * xv;
                        int so = (h * hp + p) * ns;
                        float acc = 0f;
                        for (int s = 0; s < ns; s++) {
                            float sv = ss[so + s] * dA + dtx * conv[bo + s];
                            ss[so + s] = sv;
                            acc += sv * conv[co + s];
                        }
                        y[t * di + h * hp + p] = acc + m.d()[h] * xv;
                    }
                }
            }
        });
        // gated RMSNorm: y silu(z), normalized per group of dInner / normGroups channels, times the gain
        int gs = di / m2.normGroups();
        int ng = m2.normGroups();
        pool.forRange(n * ng, 1, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int t = i / ng;
                int gi = i % ng;
                int base = t * di + gi * gs;
                float ssq = 0f;
                for (int j = 0; j < gs; j++) {
                    float v = y[base + j] * VectorKernels.silu(proj[t * pw + gi * gs + j]);
                    y[base + j] = v;
                    ssq += v * v;
                }
                float inv = (float) (1.0 / Math.sqrt(ssq / gs + m2.normEps()));
                for (int j = 0; j < gs; j++) {
                    y[base + j] = m.norm()[gi * gs + j] * (y[base + j] * inv);
                }
            }
        });
        mm.mul(m.outProj(), y, mixOut, 0, n, false);
    }

    // ------------------------------------------------------------------ Mamba-1 (Jamba)

    private void mamba1(HybridLayer.M1 m, int layer, int n) {
        int di = m1.dInner();
        int dc = m1.convKernel();
        int ds = m1.state();
        int rank = m1.dtRank();
        int stride = rank + 2 * ds;
        mm.mul(m.inProj(), xb, proj, 0, n, false);
        float[] st = convState[layer];
        pool.forRange(di, 32, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                for (int t = 0; t < n; t++) {
                    conv[t * di + i] = MambaCpuBackend.convStep(m.conv(), i * dc, m.convBias()[i], st, i * (dc - 1), dc,
                            proj[t * 2 * di + i]);
                }
            }
        });
        mm.mul(m.xProj(), conv, dbc, 0, n, false);
        // Jamba's norms on dt, B and C
        float eps = m1.normEps();
        pool.forRange(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                VectorKernels.rmsnormRow(dbc, t * stride, dbc, t * stride, m.dtNorm(), rank, eps);
                VectorKernels.rmsnormRow(dbc, t * stride + rank, dbc, t * stride + rank, m.bNorm(), ds, eps);
                VectorKernels.rmsnormRow(dbc, t * stride + rank + ds, dbc, t * stride + rank + ds, m.cNorm(), ds, eps);
            }
        });
        float[] s = ssmState[layer];
        pool.forRange(di, 16, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int so = i * ds;
                for (int t = 0; t < n; t++) {
                    int bo = t * stride + rank;
                    float raw = VectorKernels.dot(m.dtProj(), i * rank, dbc, t * stride, rank) + m.dtBias()[i];
                    float dt = MambaCpuBackend.softplus(raw);
                    float u = conv[t * di + i];
                    float acc = MambaCpuBackend.scanStep(dt, u, m.a(), so, s, so, dbc, bo, ds);
                    float z = proj[t * 2 * di + di + i];
                    y[t * di + i] = (acc + m.d()[i] * u) * VectorKernels.silu(z);
                }
            }
        });
        mm.mul(m.outProj(), y, mixOut, 0, n, false);
    }

    // ------------------------------------------------------------------ feed-forward

    /** The feed-forward block on the normed rows {@link #xb}, added to {@link #ffnOut}. */
    private void ffn(Mlp spec, MlpWeights mlp, int n) {
        switch (mlp) {
            case MlpWeights.Dense d -> dense(d, (Mlp.Dense) spec, n);
            case MlpWeights.Moe m -> moe((Mlp.Moe) spec, m, n);
        }
    }

    private void dense(MlpWeights.Dense d, Mlp.Dense spec, int n) {
        int hdim = spec.hidden();
        mm.mul(d.gateUp(), xb, gateUp, 0, n, false);
        if (spec.act() == Mlp.Activation.RELU2) {
            pool.forRange(n * hdim, 512, (i0, i1) -> {
                for (int i = i0; i < i1; i++) {
                    float v = Math.max(gateUp[i], 0f);
                    hidden[i] = v * v;
                }
            });
        } else {
            siluMul(n, hdim);
        }
        mm.mul(d.down(), hidden, ffnOut, 0, n, true);
    }

    private void moe(Mlp.Moe spec, MlpWeights.Moe m, int n) {
        int topK = spec.topK();
        if (n > 1 && !CpuBackend.DETERMINISTIC) {
            pool.forRange(n, 1, (t0, t1) -> {
                float[] logits = mm.scratch.get();
                for (int t = t0; t < t1; t++) {
                    route(spec, m.router(), t, logits);
                }
            });
            experts.batched(m, spec.experts(), topK, spec.moeDim(), n, xb, assign, routeWeight, ffnOut);
        } else {
            float[] logits = mm.scratch.get();
            for (int t = 0; t < n; t++) {
                route(spec, m.router(), t, logits);
                experts.token(m, topK, spec.moeDim(), t, xb, assign, routeWeight, ffnOut);
            }
        }
        if (spec.shared() != null) {
            mm.mul(m.shared().gateUp(), xb, gateUp, 0, n, false);
            siluMul(n, spec.shared().hidden());
            mm.mul(m.shared().down(), hidden, ffnOut, 0, n, true);
        }
    }

    /**
     * The router of token {@code t}: Granite's top-k of the logits then a softmax over the selected ones, or Jamba's softmax
     * over all logits then the top-k (renormalized only if the spec says so).
     */
    private void route(Mlp.Moe spec, float[] router, int t, float[] logits) {
        int e = spec.experts();
        int topK = spec.topK();
        int dim = c.dim();
        int ao = t * topK;
        for (int i = 0; i < e; i++) {
            logits[i] = VectorKernels.dot(router, i * dim, xb, t * dim, dim);
        }
        if (spec.router() instanceof Router.TopLogits) {
            for (int k = 0; k < topK; k++) {
                int best = 0;
                float bv = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < e; i++) {
                    if (logits[i] > bv) {
                        bv = logits[i];
                        best = i;
                    }
                }
                assign[ao + k] = best;
                routeWeight[ao + k] = bv;
                logits[best] = Float.NEGATIVE_INFINITY;
            }
            float max = routeWeight[ao];
            float sum = 0;
            for (int k = 0; k < topK; k++) {
                float p = (float) Math.exp(routeWeight[ao + k] - max);
                routeWeight[ao + k] = p;
                sum += p;
            }
            for (int k = 0; k < topK; k++) {
                routeWeight[ao + k] /= sum;
            }
            return;
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < e; i++) {
            max = Math.max(max, logits[i]);
        }
        float sum = 0;
        for (int i = 0; i < e; i++) {
            float p = (float) Math.exp(logits[i] - max);
            logits[i] = p;
            sum += p;
        }
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

    @Override
    public void close() {
        pool.close();
    }
}
