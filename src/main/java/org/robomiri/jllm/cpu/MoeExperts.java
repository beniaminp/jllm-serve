package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.MlpWeights;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.util.Arrays;

/**
 * The routed experts of a mixture-of-experts layer on the CPU, once the router has chosen them:
 * <pre>
 *     x[t] += Σ_k routeWeight[t·topK + k] · down_e(silu(gate_e(xb[t])) · up_e(xb[t])),    e = assign[t·topK + k]
 * </pre>
 * {@link CpuBackend} runs its MoE layers through it, and the Metal backend runs the experts it leaves in CPU memory
 * ({@code backend.metal.cpu-moe}) through its own instance. A slot whose {@code assign} is -1 is skipped (its expert
 * is computed elsewhere: the GPU's share of a layer whose experts are split between the two). One instance is used by one thread at a time
 * (its scratch is shared by the calls); the work inside a call is spread over the pool.
 */
public final class MoeExperts {

    private final ThreadPool pool;
    private final CpuMatmul mm;
    private final ThreadLocal<float[]> scratch;
    private final int dim;
    // per-slot buffers, slot = t * topK + k
    private final float[] gateUp;
    private final float[] hidden;
    private final float[] down;
    private final int[] expertStart;
    private final int[] expertFill;
    private final int[] usedExperts;
    private final int[] slotOrder;

    MoeExperts(ThreadPool pool, CpuMatmul mm, Decoder arch, int maxBatch) {
        this(pool, mm, arch.core().dim(), arch.maxExperts(), arch.maxTopK(), arch.maxMoeDim(), maxBatch);
    }

    /** Sized for layers of up to {@code maxExperts} experts of width {@code maxMoeDim}, {@code maxTopK} per token. */
    MoeExperts(ThreadPool pool, CpuMatmul mm, int dim, int maxExperts, int maxTopK, int maxMoeDim, int maxBatch) {
        this.pool = pool;
        this.mm = mm;
        this.scratch = mm.scratch;
        this.dim = dim;
        int slots = maxBatch * maxTopK;
        gateUp = new float[slots * 2 * maxMoeDim];
        hidden = new float[slots * maxMoeDim];
        down = new float[slots * dim];
        expertStart = new int[maxExperts + 1];
        expertFill = new int[maxExperts];
        usedExperts = new int[maxExperts];
        slotOrder = new int[slots];
    }

    /**
     * Experts for the layers of {@code w} on {@code pool}, for chunks of up to {@code maxBatch} tokens.
     *
     * @param int8Activations integer dot products for single tokens, as {@code backend.cpu.int8-activations}
     */
    public static MoeExperts create(ThreadPool pool, ModelWeights w, int maxBatch, boolean int8Activations) {
        Decoder arch = w.decoder();
        int scratchLen = Math.max(arch.maxExperts(), 4 * CpuMatmul.maxCols(w));
        return new MoeExperts(pool, new CpuMatmul(pool, int8Activations, CpuBackend.DETERMINISTIC, scratchLen), arch, maxBatch);
    }

    /** Token {@code t}: {@code x[t·dim ..] += Σ_k w_k · expert_k(xb[t·dim ..])} with the choices at {@code t·topK}. */
    public void token(MlpWeights.Moe w, int topK, int m, int t, float[] xb, int[] assign, float[] routeWeight, float[] x) {
        token(w, Mlp.ExpertSpec.PLAIN, topK, m, t, xb, assign, routeWeight, x);
    }

    /**
     * Gate/up activation of one slot (expert {@code expert}, routing weight {@code rw}) for the experts of gpt-oss and
     * Llama 4: the gate/up biases, the routing weight on the input (Llama 4), the clamped SwiGLU (gpt-oss) or the plain
     * one. {@code gu[go ..]} holds the {@code m} gate values then the {@code m} up values; {@code out[oo ..]} gets the
     * {@code m} hidden values. A skipped slot ({@code expert < 0}) gets zeros.
     */
    static void activate(Mlp.ExpertSpec es, float[] bias, int expert, float rw, float[] gu, int go, float[] out, int oo, int m) {
        if (expert < 0) {
            Arrays.fill(out, oo, oo + m, 0f);
            return;
        }
        int bo = expert * 2 * m;
        float limit = es.clampLimit();
        float alpha = es.alpha();
        for (int j = 0; j < m; j++) {
            float g = gu[go + j];
            float u = gu[go + m + j];
            if (bias != null) {
                g += bias[bo + j];
                u += bias[bo + m + j];
            }
            if (es.weightInput()) {
                g *= rw;
                u *= rw;
            }
            if (es.clamped()) {
                g = Math.min(g, limit);
                u = Math.max(-limit, Math.min(limit, u));
                out[oo + j] = (u + 1f) * (g / (1f + (float) Math.exp(-alpha * g)));
            } else {
                out[oo + j] = g / (1f + (float) Math.exp(-g)) * u;
            }
        }
    }

    /** As {@link #token(MlpWeights.Moe, int, int, int, float[], int[], float[], float[])} for experts with the arithmetic of {@code es}. */
    public void token(MlpWeights.Moe w, Mlp.ExpertSpec es, int topK, int m, int t, float[] xb, int[] assign, float[] routeWeight,
                      float[] x) {
        final boolean wIn = es.weightInput();
        final float[] db = w.downBias();
        int gu = 2 * m;
        int xo = t * dim;
        int ao = t * topK;
        QTensor expGateUp = w.expGateUp();
        QTensor expDown = w.expDown();
        // all selected experts' gate/up rows in one parallel loop
        if (mm.nativeInt8(expGateUp)) { // one downcall per run of rows of the same slot within a chunk
            mm.quantizeNative(xb, xo, dim);
            pool.forRange(topK * gu, 32, (i0, i1) -> {
                for (int i = i0; i < i1; ) {
                    int k = i / gu;
                    int end = Math.min(i1, (k + 1) * gu);
                    int e = assign[ao + k];
                    if (e < 0) {
                        Arrays.fill(gateUp, i, end, 0f);
                    } else {
                        int r = e * gu + i - k * gu;
                        mm.nativeRows(expGateUp, r, r + end - i, 0, gateUp, i, false);
                    }
                    i = end;
                }
            });
        } else if (mm.useInt8(expGateUp)) {
            short[] xq = mm.quantizeInt8(xb, xo, dim);
            float[] xs = mm.xs();
            pool.forRange(topK * gu, 32, (i0, i1) -> {
                for (int i = i0; i < i1; i++) {
                    int k = i / gu;
                    int r = i - k * gu;
                    int e = assign[ao + k];
                    gateUp[i] = e < 0 ? 0f : mm.int8Dot(expGateUp, e * gu + r, xq, xs, 0);
                }
            });
        } else {
            pool.forRange(topK * gu, 32, (i0, i1) -> {
                for (int i = i0; i < i1; i++) {
                    int k = i / gu;
                    int r = i - k * gu;
                    int e = assign[ao + k];
                    gateUp[i] = e < 0 ? 0f : VectorKernels.dotRow(expGateUp, e * gu + r, xb, xo);
                }
            });
        }
        for (int k = 0; k < topK; k++) {
            if (es.special()) {
                activate(es, w.gateUpBias(), assign[ao + k], routeWeight[ao + k], gateUp, k * gu, hidden, k * m, m);
            } else {
                VectorKernels.siluMul(gateUp, k * gu, gateUp, k * gu + m, hidden, k * m, m);
            }
        }
        if (mm.nativeInt8(expDown)) {
            // per chunk of output rows: each slot's rows into down[k * dim + r], then the same sum in slot order
            mm.quantizeNative(hidden, 0, topK * m);   // slot k starts at block k * m / 32
            pool.forRange(dim, 8, (r0, r1) -> {
                for (int k = 0; k < topK; k++) {
                    int e = assign[ao + k];
                    if (e >= 0) {
                        mm.nativeRows(expDown, e * dim + r0, e * dim + r1, k * m / QType.BLOCK, down, k * dim + r0, false);
                    }
                }
                for (int r = r0; r < r1; r++) {
                    float sum = 0;
                    for (int k = 0; k < topK; k++) {
                        int e = assign[ao + k];
                        if (e >= 0) {
                            sum += (wIn ? 1f : routeWeight[ao + k]) * (down[k * dim + r] + (db == null ? 0f : db[e * dim + r]));
                        }
                    }
                    x[xo + r] += sum;
                }
            });
        } else if (mm.useInt8(expDown)) {
            short[] xq = mm.quantizeInt8(hidden, 0, topK * m);   // slot k starts at block offset k * m
            float[] xs = mm.xs();
            pool.forRange(dim, 8, (r0, r1) -> {
                for (int r = r0; r < r1; r++) {
                    float sum = 0;
                    for (int k = 0; k < topK; k++) {
                        int e = assign[ao + k];
                        if (e >= 0) {
                            sum += (wIn ? 1f : routeWeight[ao + k]) * (mm.int8Dot(expDown, e * dim + r, xq, xs, k * m)
                                    + (db == null ? 0f : db[e * dim + r]));
                        }
                    }
                    x[xo + r] += sum;
                }
            });
        } else {
            pool.forRange(dim, 8, (r0, r1) -> {
                for (int r = r0; r < r1; r++) {
                    float sum = 0;
                    for (int k = 0; k < topK; k++) {
                        int e = assign[ao + k];
                        if (e >= 0) {
                            sum += (wIn ? 1f : routeWeight[ao + k]) * (VectorKernels.dotRow(expDown, e * dim + r, hidden, k * m)
                                    + (db == null ? 0f : db[e * dim + r]));
                        }
                    }
                    x[xo + r] += sum;
                }
            });
        }
    }

    /**
     * A chunk of {@code n} routed tokens: group the {@code n * topK} slots by expert (counting sort), then run each
     * expert once over all tokens routed to it with the dense prefill kernel (four rows dequantized once, reused for
     * every token). Down-projection results are kept per slot and combined into x per token in slot order.
     */
    public void batched(MlpWeights.Moe w, int experts, int topK, int m, int n, float[] xb, int[] assign,
                        float[] routeWeight, float[] x) {
        batched(w, Mlp.ExpertSpec.PLAIN, experts, topK, m, n, xb, assign, routeWeight, x);
    }

    /** As {@link #batched(MlpWeights.Moe, int, int, int, int, float[], int[], float[], float[])} for experts with the arithmetic of {@code es}. */
    public void batched(MlpWeights.Moe w, Mlp.ExpertSpec es, int experts, int topK, int m, int n, float[] xb, int[] assign,
                        float[] routeWeight, float[] x) {
        int gu = 2 * m;
        int slots = n * topK;

        Arrays.fill(expertStart, 0, experts + 1, 0);
        for (int s = 0; s < slots; s++) {
            if (assign[s] >= 0) {
                expertStart[assign[s] + 1]++;
            } else {
                Arrays.fill(down, s * dim, (s + 1) * dim, 0f); // a skipped slot adds nothing in the combine
            }
        }
        int used = 0;
        for (int i = 0; i < experts; i++) {
            if (expertStart[i + 1] > 0) {
                usedExperts[used++] = i;
            }
            expertStart[i + 1] += expertStart[i];
            expertFill[i] = expertStart[i];
        }
        for (int s = 0; s < slots; s++) {
            if (assign[s] >= 0) {
                slotOrder[expertFill[assign[s]]++] = s;
            }
        }

        expertMatmul(w.expGateUp(), gu, used, topK, xb, dim, false, gateUp, gu);

        pool.forRange(slots, 1, (s0, s1) -> {
            for (int s = s0; s < s1; s++) {
                if (es.special()) {
                    activate(es, w.gateUpBias(), assign[s], routeWeight[s], gateUp, s * gu, hidden, s * m, m);
                } else {
                    VectorKernels.siluMul(gateUp, s * gu, gateUp, s * gu + m, hidden, s * m, m);
                }
            }
        });

        expertMatmul(w.expDown(), dim, used, topK, hidden, m, true, down, dim);

        if (es.weightInput() || w.downBias() != null) { // the weight on the input and / or the experts' down biases
            float[] db = w.downBias();
            boolean wIn = es.weightInput();
            pool.forRange(n, 1, (t0, t1) -> {
                for (int t = t0; t < t1; t++) {
                    int s0 = t * topK;
                    for (int r = 0; r < dim; r++) {
                        float sum = 0;
                        for (int k = 0; k < topK; k++) {
                            int e = assign[s0 + k];
                            if (e >= 0) {
                                sum += (wIn ? 1f : routeWeight[s0 + k]) * (down[(s0 + k) * dim + r] + (db == null ? 0f : db[e * dim + r]));
                            }
                        }
                        x[t * dim + r] += sum;
                    }
                }
            });
            return;
        }
        pool.forRange(n, 1, (t0, t1) -> {
            for (int t = t0; t < t1; t++) {
                int s0 = t * topK;
                VectorKernels.combine(routeWeight, s0, topK, down, s0 * dim, dim, x, t * dim, dim);
            }
        });
    }

    /**
     * {@code out[slot, r] = dot(W[expert(slot) * rowsPer + r, :], in[src(slot), :])} for every slot of every
     * used expert, where {@code src(slot)} is the slot itself ({@code inBySlot}) or its token. Parallel over
     * (expert, 4-row group) pairs; each group is dequantized once and reused for all of the expert's slots.
     */
    private void expertMatmul(QTensor wt, int rowsPer, int used, int topK, float[] in, int inStride, boolean inBySlot,
                              float[] out, int outStride) {
        int cols = wt.cols;
        int groups = (rowsPer + 3) / 4;
        pool.forRange(used * groups, 1, (g0, g1) -> {
            float[] tmp = scratch.get();
            for (int g = g0; g < g1; g++) {
                int ex = usedExperts[g / groups];
                int r = (g % groups) * 4;
                int rowBase = ex * rowsPer;
                int sBeg = expertStart[ex];
                int sEnd = expertStart[ex + 1];
                if (r + 4 <= rowsPer) {
                    for (int k = 0; k < 4; k++) {
                        VectorKernels.dequantRow(wt, rowBase + r + k, tmp, k * cols);
                    }
                    int s = sBeg;
                    for (; s + 4 <= sEnd; s += 4) {
                        int sa = slotOrder[s];
                        int sb = slotOrder[s + 1];
                        int sc = slotOrder[s + 2];
                        int sd = slotOrder[s + 3];
                        VectorKernels.dot4x4(tmp, 0, cols, in,
                                (inBySlot ? sa : sa / topK) * inStride, (inBySlot ? sb : sb / topK) * inStride,
                                (inBySlot ? sc : sc / topK) * inStride, (inBySlot ? sd : sd / topK) * inStride,
                                out, sa * outStride + r, sb * outStride + r, sc * outStride + r, sd * outStride + r,
                                false);
                    }
                    for (; s < sEnd; s++) {
                        int slot = slotOrder[s];
                        int io = (inBySlot ? slot : slot / topK) * inStride;
                        VectorKernels.dot4(tmp, 0, cols, in, io, out, slot * outStride + r, false);
                    }
                } else {
                    for (int rr = r; rr < rowsPer; rr++) {
                        for (int s = sBeg; s < sEnd; s++) {
                            int slot = slotOrder[s];
                            int io = (inBySlot ? slot : slot / topK) * inStride;
                            out[slot * outStride + rr] = VectorKernels.dotRow(wt, rowBase + rr, in, io);
                        }
                    }
                }
            }
        });
    }
}
