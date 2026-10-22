package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.MambaArch;
import org.robomiri.jllm.model.MambaLayer;
import org.robomiri.jllm.model.ModelWeights;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Mamba (v1) forward pass on the GPU. The whole chunk is recorded into one command buffer; per layer that is
 * RMSNorm, {@code in_proj}, {@code mamba_conv}, {@code x_proj}, {@code mamba_scan}, {@code out_proj} (the three
 * projections use the shared quantized matmul kernels, norm is the shared rmsnorm kernel). The recurrent states
 * live in shared buffers that persist across calls, see {@link org.robomiri.jllm.cpu.MambaCpuBackend} for the
 * state contract ({@code pos == 0} resets).
 *
 * <p>Design trade-off: {@code mamba_conv} and {@code mamba_scan} run one thread per channel and loop over the
 * chunk's tokens inside the thread, so a layer needs one dispatch per stage regardless of chunk size, at the cost of
 * only {@code d_inner} parallel threads. A parallel scan over tokens would add parallelism for long prefills but
 * needs a combine step and extra scratch for little benefit at decode (one token), where the layer is bound by
 * weight reads.
 */
public final class MambaMetalBackend implements Backend {

    private final ModelWeights w;
    private final Architecture.Core c;
    private final MambaArch m;
    private final int ctx;
    private final int maxBatch;
    private final MetalKernels k;

    private Metal.Buffer embedQ;
    private Metal.Buffer lmHeadQ;
    private LayerBufs[] layers;
    private Metal.Buffer finalNorm;
    private Metal.Buffer[] convState;
    private Metal.Buffer[] ssmState;
    private int statePos = 0;

    private Metal.Buffer x;
    private Metal.Buffer xb;
    private Metal.Buffer xz;
    private Metal.Buffer xc;
    private Metal.Buffer dbc;
    private Metal.Buffer y;
    private Metal.Buffer logitsBuf;

    private static final class LayerBufs {
        Metal.Buffer norm;
        Metal.Buffer inProj;
        Metal.Buffer conv;
        Metal.Buffer convBias;
        Metal.Buffer xProj;
        Metal.Buffer dtProj;
        Metal.Buffer dtBias;
        Metal.Buffer a;
        Metal.Buffer d;
        Metal.Buffer outProj;
    }

    /**
     * Selects the concurrent compute encoder (true: independent dispatches overlap, a barrier goes between dependent ones)
     * or the serial one (false) for the passes that follow; the results are bit-identical. The default is
     * {@code -Djllm.metal.concurrent} (docs/08 section 8.28).
     */
    public void concurrent(boolean on) {
        k.concurrent(on ? MetalKernels.concurrentOn() : 0);
    }

    /** Memory barriers the concurrent encoder has inserted so far. */
    public long barriers() {
        return k.metal.barriers();
    }

    public MambaMetalBackend(ModelWeights w, int ctx, int maxBatch, String metalLib) {
        this.w = w;
        this.m = w.mambaArch();
        this.c = m.core();
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        if (m.dState() > MetalConstants.MAMBA_MAX_STATE) {
            throw new IllegalArgumentException("Metal Mamba scan supports state_size <= "
                    + MetalConstants.MAMBA_MAX_STATE + ", got " + m.dState());
        }
        if (m.dConv() > MetalConstants.MAMBA_MAX_CONV) {
            throw new IllegalArgumentException("Metal Mamba conv supports conv_kernel <= "
                    + MetalConstants.MAMBA_MAX_CONV + ", got " + m.dConv());
        }
        this.k = new MetalKernels(Metal.create(metalLib), c.normEps());
        try { // owns the Metal context from here on: closed again if anything below throws
            init();
        } catch (RuntimeException | Error e) {
            try {
                k.close();
            } catch (RuntimeException | Error suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    private void init() {
        embedQ = k.wrap(w.embed.data);
        lmHeadQ = w.lmHead == w.embed ? embedQ : k.wrap(w.lmHead.data);
        int di = m.dInner();
        layers = new LayerBufs[c.layers()];
        convState = new Metal.Buffer[c.layers()];
        ssmState = new Metal.Buffer[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            MambaLayer L = w.mambaLayer(l);
            LayerBufs b = new LayerBufs();
            b.norm = k.upload(L.norm());
            b.inProj = k.wrap(L.inProj().data);
            b.conv = k.upload(L.conv());
            b.convBias = k.upload(L.convBias());
            b.xProj = k.wrap(L.xProj().data);
            b.dtProj = k.upload(L.dtProj());
            b.dtBias = k.upload(L.dtBias());
            b.a = k.upload(L.a());
            b.d = k.upload(L.d());
            b.outProj = k.wrap(L.outProj().data);
            layers[l] = b;
            convState[l] = k.metal.newBuffer((long) di * (m.dConv() - 1) * 4); // zeroed: the initial state
            ssmState[l] = k.metal.newBuffer((long) di * m.dState() * 4);
        }
        finalNorm = k.upload(w.finalNorm);
        MetalTuner.prefillMatmuls(k, w); // f32 or half-weight MMA prefill kernels, per projection shape
        x = k.scratch((long) maxBatch * c.dim() * 4);
        xb = k.scratch((long) maxBatch * c.dim() * 4);
        xz = k.scratch((long) maxBatch * 2 * di * 4);
        xc = k.scratch((long) maxBatch * di * 4);
        dbc = k.scratch((long) maxBatch * (m.dtRank() + 2 * m.dState()) * 4);
        y = k.scratch((long) maxBatch * di * 4);
        logitsBuf = k.scratch((long) w.lmHead.rows * 4);
        long others = DeviceLedger.others(ledger);
        DeviceLedger.add(ledger);
        long ws = k.metal.recommendedWorkingSet();
        long gpu = w.byteSize() + k.metal.allocatedBytes();
        if (gpu + others > ws) {
            System.err.printf("warning: the GPU holds %.1f GB (this model %.1f GB, other backends %.1f GB), more than "
                    + "Metal's recommended working set (%.1f GB); expect paging%n", (gpu + others) / 1e9, gpu / 1e9,
                    others / 1e9, ws / 1e9);
        }
    }

    /** This backend in the {@link DeviceLedger}, so the other models' GPU budgets count it. */
    private final DeviceLedger.Resident ledger = new DeviceLedger.Resident() {
        @Override
        public ModelWeights weights() {
            return w;
        }

        @Override
        public long gpuWeightBytes() {
            return w.byteSize();
        }

        @Override
        public long ownBytes() {
            return k.metal.allocatedBytes();
        }
    };

    @Override
    public String name() {
        return "metal mamba (" + k.metal.deviceName() + ")";
    }

    @Override
    public int thermalState() {
        return k.metal.thermalState();
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
                convState[l].contents().fill((byte) 0);
                ssmState[l].contents().fill((byte) 0);
            }
        } else if (pos != statePos) {
            throw new IllegalStateException("Mamba state holds " + statePos + " tokens, cannot continue at " + pos);
        }
        statePos = -1;
        int dim = c.dim();
        float[] row = new float[dim];
        for (int t = 0; t < n; t++) {
            w.embed.dequantizeRow(tokens[off + t], row, 0);
            MemorySegment.copy(row, 0, x.contents(), ValueLayout.JAVA_FLOAT, (long) t * dim * 4, dim);
        }

        k.begin();
        boolean committed = false;
        try {
            for (int l = 0; l < c.layers(); l++) {
                LayerBufs L = layers[l];
                MambaLayer W = w.mambaLayer(l);
                k.rmsnorm(x, 0, L.norm, xb, n, dim);
                k.matmul(W.inProj(), L.inProj, xb, 0, xz, 0, n, false, null, 0f);
                k.mambaConv(xz, L.conv, L.convBias, convState[l], xc, m.dInner(), m.dConv(), n);
                k.matmul(W.xProj(), L.xProj, xc, 0, dbc, 0, n, false, null, 0f);
                k.mambaScan(xc, xz, dbc, L.dtProj, L.dtBias, L.a, L.d, ssmState[l], y, m.dInner(), m.dState(), m.dtRank(), n);
                k.matmul(W.outProj(), L.outProj, y, 0, x, 0, n, true, null, 0f);
                if (MetalKernels.COMMIT_LAYERS > 0 && (l + 1) % MetalKernels.COMMIT_LAYERS == 0 && l + 1 < c.layers()) {
                    k.commit(); // the GPU starts on these layers while the next ones are encoded
                }
            }
            if (logits != null) {
                k.rmsnorm(x, (long) (n - 1) * dim * 4, finalNorm, xb, 1, dim);
                k.matmul(w.lmHead, lmHeadQ, xb, 0, logitsBuf, 0, 1, false, null, 0f);
            }
            k.commitAndWait();
            committed = true;
        } finally {
            if (!committed) {
                k.abort(); // an un-ended encoder would make Metal abort the process on release
            }
        }
        if (logits != null) {
            MemorySegment.copy(logitsBuf.contents(), ValueLayout.JAVA_FLOAT, 0, logits, 0,
                    Math.min(w.lmHead.rows, logits.length));
        }
        statePos = pos + n;
    }

    @Override
    public void close() {
        DeviceLedger.remove(ledger);
        k.close();
    }
}
