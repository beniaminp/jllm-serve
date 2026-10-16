package org.robomiri.jllm.metal;

import org.robomiri.jllm.model.Backend;
import org.robomiri.jllm.model.MambaLayer;
import org.robomiri.jllm.model.ModelConfig;
import org.robomiri.jllm.model.ModelWeights;
import org.robomiri.jllm.model.QTensor;
import org.robomiri.jllm.model.QType;

import java.lang.foreign.Arena;
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

    private static final int MAX_BUFS = 12;
    private static final int TOK_TILE = 8; // must match kernels.metal

    private final ModelWeights w;
    private final ModelConfig c;
    private final ModelConfig.MambaParams m;
    private final int ctx;
    private final int maxBatch;
    private final Metal metal;
    private final Arena arena = Arena.ofShared();

    private int pMatmulQ4;
    private int pMatmulQ8;
    private int pMatmulQ2K;
    private int pMatmulPfQ4;
    private int pMatmulPfQ8;
    private int pMatmulPfQ2K;
    private int pRmsnorm;
    private int pConv;
    private int pScan;

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

    private MemorySegment bufArgs;
    private MemorySegment offArgs;
    private MemorySegment params;

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

    public MambaMetalBackend(ModelWeights w, int ctx, int maxBatch, String metalLib) {
        if (!w.cfg.isMamba()) {
            throw new IllegalArgumentException("not a Mamba model: " + w.cfg);
        }
        this.w = w;
        this.c = w.cfg;
        this.m = c.mamba();
        this.ctx = ctx;
        this.maxBatch = maxBatch;
        if (m.dState() > 64) {
            throw new IllegalArgumentException("Metal Mamba scan supports state_size <= 64, got " + m.dState());
        }
        this.metal = Metal.create(metalLib);
        try { // owns metal from here on: closed again if anything below throws
            init();
        } catch (RuntimeException | Error e) {
            try {
                metal.close();
            } catch (RuntimeException | Error suppressed) {
                e.addSuppressed(suppressed);
            }
            arena.close();
            throw e;
        }
    }

    private void init() {
        pMatmulQ4 = metal.pipeline("matmul_q4");
        pMatmulQ8 = metal.pipeline("matmul_q8");
        pMatmulQ2K = metal.pipeline("matmul_q2k");
        pMatmulPfQ4 = metal.pipeline("matmul_pf_q4");
        pMatmulPfQ8 = metal.pipeline("matmul_pf_q8");
        pMatmulPfQ2K = metal.pipeline("matmul_pf_q2k");
        pRmsnorm = metal.pipeline("rmsnorm");
        pConv = metal.pipeline("mamba_conv");
        pScan = metal.pipeline("mamba_scan");

        embedQ = metal.wrap(w.embed.data);
        lmHeadQ = w.lmHead == w.embed ? embedQ : metal.wrap(w.lmHead.data);
        int di = m.dInner();
        layers = new LayerBufs[c.layers()];
        convState = new Metal.Buffer[c.layers()];
        ssmState = new Metal.Buffer[c.layers()];
        for (int l = 0; l < c.layers(); l++) {
            MambaLayer L = w.layers[l].mamba;
            LayerBufs b = new LayerBufs();
            b.norm = upload(L.norm);
            b.inProj = metal.wrap(L.inProj.data);
            b.conv = upload(L.conv);
            b.convBias = upload(L.convBias);
            b.xProj = metal.wrap(L.xProj.data);
            b.dtProj = upload(L.dtProj);
            b.dtBias = upload(L.dtBias);
            b.a = upload(L.a);
            b.d = upload(L.d);
            b.outProj = metal.wrap(L.outProj.data);
            layers[l] = b;
            convState[l] = metal.newBuffer((long) di * (m.dConv() - 1) * 4);
            ssmState[l] = metal.newBuffer((long) di * m.dState() * 4);
        }
        finalNorm = upload(w.finalNorm);
        x = metal.newBuffer((long) maxBatch * c.dim() * 4, false);
        xb = metal.newBuffer((long) maxBatch * c.dim() * 4, false);
        xz = metal.newBuffer((long) maxBatch * 2 * di * 4, false);
        xc = metal.newBuffer((long) maxBatch * di * 4, false);
        dbc = metal.newBuffer((long) maxBatch * (m.dtRank() + 2 * m.dState()) * 4, false);
        y = metal.newBuffer((long) maxBatch * di * 4, false);
        logitsBuf = metal.newBuffer((long) w.lmHead.rows * 4, false);
        long ws = metal.recommendedWorkingSet();
        if (w.byteSize() > ws) {
            System.err.printf("warning: weights (%.1f GB) exceed Metal's recommended working set (%.1f GB); expect paging%n",
                    w.byteSize() / 1e9, ws / 1e9);
        }
        bufArgs = arena.allocate(ValueLayout.ADDRESS, MAX_BUFS);
        offArgs = arena.allocate(ValueLayout.JAVA_LONG, MAX_BUFS);
        params = arena.allocate(64);
    }

    private Metal.Buffer upload(float[] data) {
        Metal.Buffer b = metal.newBuffer((long) data.length * 4, false); // fully overwritten below
        MemorySegment.copy(data, 0, b.contents(), ValueLayout.JAVA_FLOAT, 0, data.length);
        return b;
    }

    @Override
    public String name() {
        return "metal mamba (" + metal.deviceName() + ")";
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

        metal.begin();
        boolean committed = false;
        try {
            for (int l = 0; l < c.layers(); l++) {
                LayerBufs L = layers[l];
                MambaLayer W = w.layers[l].mamba;
                rmsnorm(x, 0, L.norm, xb, n, dim);
                matmul(W.inProj, L.inProj, xb, xz, n, false);
                conv(L, l, n);
                matmul(W.xProj, L.xProj, xc, dbc, n, false);
                scan(L, l, n);
                matmul(W.outProj, L.outProj, y, x, n, true);
                if (MetalBackend.COMMIT_LAYERS > 0 && (l + 1) % MetalBackend.COMMIT_LAYERS == 0 && l + 1 < c.layers()) {
                    metal.commit(); // the GPU starts on these layers while the next ones are encoded
                }
            }
            if (logits != null) {
                rmsnorm(x, (long) (n - 1) * dim * 4, finalNorm, xb, 1, dim);
                matmul(w.lmHead, lmHeadQ, xb, logitsBuf, 1, false);
            }
            metal.commitAndWait();
            committed = true;
        } finally {
            if (!committed) {
                metal.abort(); // an un-ended encoder would make Metal abort the process on release
            }
        }
        if (logits != null) {
            MemorySegment.copy(logitsBuf.contents(), ValueLayout.JAVA_FLOAT, 0, logits, 0,
                    Math.min(w.lmHead.rows, logits.length));
        }
        statePos = pos + n;
    }

    // ------------------------------------------------------------------ dispatch helpers

    private void bind(int i, Metal.Buffer b, long offset) {
        bufArgs.setAtIndex(ValueLayout.ADDRESS, i, b.handle());
        offArgs.setAtIndex(ValueLayout.JAVA_LONG, i, offset);
    }

    private void matmul(QTensor t, Metal.Buffer wbuf, Metal.Buffer in, Metal.Buffer out, int n, boolean acc) {
        bind(0, wbuf, 0);
        bind(1, wbuf, MetalBackend.scaleOff(t));
        bind(2, in, 0);
        bind(3, out, 0);
        bind(4, finalNorm, 0); // dummy binding: only the *_norm kernel variants read it
        params.setAtIndex(ValueLayout.JAVA_INT, 0, t.rows);
        params.setAtIndex(ValueLayout.JAVA_INT, 1, t.cols);
        params.setAtIndex(ValueLayout.JAVA_INT, 2, n);
        params.setAtIndex(ValueLayout.JAVA_INT, 3, acc ? 1 : 0);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 4, c.normEps());
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 5, 0f); // no soft-cap
        int sgPerTg = 4; // 128 threads = 4 rows per threadgroup
        if (n >= MetalBackend.PREFILL_MIN_TOKENS) {
            int pipe = MetalBackend.byType(t.type, pMatmulPfQ4, pMatmulPfQ8, pMatmulPfQ2K);
            metal.dispatch(pipe, bufArgs, offArgs, 5, params, 24, MetalBackend.pfGrid(t.rows), MetalBackend.pfTiles(n), 1,
                    32 * sgPerTg, 1, 1);
            return;
        }
        int pipe = MetalBackend.byType(t.type, pMatmulQ4, pMatmulQ8, pMatmulQ2K);
        int gx = (t.rows + sgPerTg - 1) / sgPerTg;
        int gy = (n + TOK_TILE - 1) / TOK_TILE;
        metal.dispatch(pipe, bufArgs, offArgs, 5, params, 24, MetalBackend.UNUSED_TG_BYTES, gx, gy, 1, 32 * sgPerTg, 1, 1);
    }

    private void rmsnorm(Metal.Buffer in, long inOff, Metal.Buffer weight, Metal.Buffer out, int n, int width) {
        bind(0, in, inOff);
        bind(1, weight, 0);
        bind(2, out, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, width);
        params.setAtIndex(ValueLayout.JAVA_FLOAT, 1, c.normEps());
        metal.dispatch(pRmsnorm, bufArgs, offArgs, 3, params, 8, n, 1, 1, 256, 1, 1);
    }

    private void conv(LayerBufs L, int layer, int n) {
        bind(0, xz, 0);
        bind(1, L.conv, 0);
        bind(2, L.convBias, 0);
        bind(3, convState[layer], 0);
        bind(4, xc, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, m.dInner());
        params.setAtIndex(ValueLayout.JAVA_INT, 1, m.dConv());
        params.setAtIndex(ValueLayout.JAVA_INT, 2, n);
        metal.dispatch(pConv, bufArgs, offArgs, 5, params, 12, (m.dInner() + 63) / 64, 1, 1, 64, 1, 1);
    }

    private void scan(LayerBufs L, int layer, int n) {
        bind(0, xc, 0);
        bind(1, xz, 0);
        bind(2, dbc, 0);
        bind(3, L.dtProj, 0);
        bind(4, L.dtBias, 0);
        bind(5, L.a, 0);
        bind(6, L.d, 0);
        bind(7, ssmState[layer], 0);
        bind(8, y, 0);
        params.setAtIndex(ValueLayout.JAVA_INT, 0, m.dInner());
        params.setAtIndex(ValueLayout.JAVA_INT, 1, m.dState());
        params.setAtIndex(ValueLayout.JAVA_INT, 2, m.dtRank());
        params.setAtIndex(ValueLayout.JAVA_INT, 3, n);
        metal.dispatch(pScan, bufArgs, offArgs, 9, params, 16, (m.dInner() + 63) / 64, 1, 1, 64, 1, 1);
    }

    @Override
    public void close() {
        metal.close();
        arena.close();
    }
}
