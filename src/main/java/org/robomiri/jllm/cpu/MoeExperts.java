package org.robomiri.jllm.cpu;

import org.robomiri.jllm.model.Decoder;
import org.robomiri.jllm.model.MlpWeights;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;

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
        int gu = 2 * m;
        int xo = t * dim;
        int ao = t * topK;
        QTensor expGateUp = w.expGateUp();
        QTensor expDown = w.expDown();
        // all selected experts' gate/up rows in one parallel loop
        if (mm.useInt8(expGateUp)) {
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
            int gi = k * gu;
            int hi = k * m;
            for (int i = 0; i < m; i++) {
                float g = gateUp[gi + i];
                hidden[hi + i] = VectorKernels.silu(g) * gateUp[gi + m + i];
            }
        }
        if (mm.useInt8(expDown)) {
            short[] xq = mm.quantizeInt8(hidden, 0, topK * m);   // slot k starts at block offset k * m
            float[] xs = mm.xs();
            pool.forRange(dim, 8, (r0, r1) -> {
                for (int r = r0; r < r1; r++) {
                    float sum = 0;
                    for (int k = 0; k < topK; k++) {
                        int e = assign[ao + k];
                        if (e >= 0) {
                            sum += routeWeight[ao + k] * mm.int8Dot(expDown, e * dim + r, xq, xs, k * m);
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
                            sum += routeWeight[ao + k] * VectorKernels.dotRow(expDown, e * dim + r, hidden, k * m);
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

        pool.forRange(slots * m, 256, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                int s = i / m;
                int j = i - s * m;
                int gi = s * gu;
                float g = gateUp[gi + j];
                hidden[s * m + j] = VectorKernels.silu(g) * gateUp[gi + m + j];
            }
        });

        expertMatmul(w.expDown(), dim, used, topK, hidden, m, true, down, dim);

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
