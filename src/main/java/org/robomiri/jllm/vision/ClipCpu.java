package org.robomiri.jllm.vision;

import org.robomiri.jllm.cpu.ThreadPool;
import org.robomiri.jllm.vision.ClipModel.Layer;
import org.robomiri.jllm.vision.Preprocess.Pixels;

import java.util.ArrayList;
import java.util.List;

/**
 * The vision encoder on the CPU: patch embedding, transformer blocks, projector, in float32 with the weights
 * dequantized on the fly from the projector file (F16 / Q8_0 / ...). It follows llama.cpp's graph for each projector
 * type ({@code tools/mtmd/clip.cpp}, {@code models/siglip.cpp}, {@code models/qwen2vl.cpp}); see docs/20.
 */
public final class ClipCpu implements VisionEncoder {

    private final ClipModel m;
    private final ThreadPool pool;
    private final boolean ownPool;
    /** {@code -Djllm.vision.dump=<prefix>}: writes the activations after the patch embedding and each block (f32) for comparisons. */
    private static final String DUMP = System.getProperty("jllm.vision.dump");

    private static void dump(String name, float[] a, int n) {
        if (DUMP == null) {
            return;
        }
        try {
            java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(n * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            bb.asFloatBuffer().put(a, 0, n);
            java.nio.file.Files.write(java.nio.file.Path.of(DUMP + "." + name + ".f32"), bb.array());
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    public ClipCpu(ClipModel m, int threads) {
        this(m, new ThreadPool(threads), true);
    }

    public ClipCpu(ClipModel m, ThreadPool pool, boolean ownPool) {
        this.m = m;
        this.pool = pool;
        this.ownPool = ownPool;
        if (m.nEmbd % VMath.PAD != 0) {
            throw new IllegalArgumentException("the encoder width " + m.nEmbd + " is not a multiple of " + VMath.PAD);
        }
    }

    @Override
    public String name() {
        return "cpu (" + pool.threads() + " threads)";
    }

    @Override
    public Encoded encode(Pixels px) {
        return switch (m.projector) {
            case IDEFICS3, GEMMA3 -> siglip(px);
            case QWEN2VL, QWEN25VL, QWEN3VL -> qwen(px);
        };
    }

    @Override
    public void close() {
        if (ownPool) {
            pool.close();
        }
    }

    // ------------------------------------------------------------------ building blocks

    private void lin(Mat w, float[] bias, float[] x, int t, int kPad, float[] y, int ldy) {
        VMath.gemm(pool, (r0, n, dst) -> w.rows(r0, n, dst, kPad), w.rows, kPad, bias, x, t, y, ldy);
    }

    private void norm(boolean rms, float[] x, int t, int n, float[] w, float[] b, float[] out) {
        if (rms) {
            VMath.rmsNorm(pool, x, t, n, w, m.eps, out);
        } else {
            VMath.layerNorm(pool, x, t, n, w, b, m.eps, out);
        }
    }

    private int actKind() {
        return switch (m.act) {
            case GELU -> 0;
            case SILU -> 1;
            case GELU_QUICK -> 2;
        };
    }

    /**
     * The transformer blocks over {@code x} ({@code t x nEmbd}), in place.
     *
     * @param rms      RMSNorm instead of LayerNorm
     * @param posA     with {@code posB}: per-row 2-D rotary positions (null = none)
     * @param winStart row ranges of the windows used by the layers that are not full-attention (null = all full)
     */
    private void blocks(float[] x, int t, boolean rms, int[] posA, int[] posB, float theta, int[] winStart, int nWin,
                        LayerHook hook) {
        int e = m.nEmbd;
        int heads = m.nHead;
        int dh = e / heads;
        int ldF = VMath.pad(m.nFf);
        float scale = 1.0f / (float) Math.sqrt(dh);
        float[] h = new float[t * e];
        float[] q = new float[t * e];
        float[] k = new float[t * e];
        float[] v = new float[t * e];
        float[] att = new float[t * e];
        float[] tmp = new float[t * e];
        float[] f1 = new float[t * ldF];
        float[] f2 = null;
        int[] full = {0, t};
        for (int il = 0; il < m.nLayer; il++) {
            Layer l = m.layers.get(il);
            norm(rms, x, t, e, l.ln1W, l.ln1B, h);
            if (il == 0) {
                dump("ln1-0", h, t * e);
            }
            if (l.qkv != null) {
                float[] qkv = new float[t * 3 * e];
                lin(l.qkv, l.qkvB, h, t, e, qkv, 3 * e);
                for (int r = 0; r < t; r++) {
                    System.arraycopy(qkv, r * 3 * e, q, r * e, e);
                    System.arraycopy(qkv, r * 3 * e + e, k, r * e, e);
                    System.arraycopy(qkv, r * 3 * e + 2 * e, v, r * e, e);
                }
            } else {
                lin(l.q, l.qB, h, t, e, q, e);
                lin(l.k, l.kB, h, t, e, k, e);
                lin(l.v, l.vB, h, t, e, v, e);
            }
            if (posA != null) {
                VMath.rope2d(pool, q, t, heads, dh, posA, posB, theta);
                VMath.rope2d(pool, k, t, heads, dh, posA, posB, theta);
            }
            boolean fullAttn = winStart == null || m.nWaPattern <= 0 || (il + 1) % m.nWaPattern == 0;
            VMath.attention(pool, q, k, v, t, heads, dh, scale, fullAttn ? full : winStart, fullAttn ? 1 : nWin, att);
            lin(l.o, l.oB, att, t, e, tmp, e);
            VMath.addTo(pool, x, tmp, t * e);
            norm(rms, x, t, e, l.ln2W, l.ln2B, h);
            lin(l.up, l.upB, h, t, e, f1, ldF);
            if (l.gate != null) {
                if (f2 == null) {
                    f2 = new float[t * ldF];
                }
                lin(l.gate, l.gateB, h, t, e, f2, ldF);
                VMath.gated(pool, f2, f1, t * ldF, actKind());
                lin(l.down, l.downB, f2, t, ldF, tmp, e);
            } else {
                VMath.activate(pool, f1, t * ldF, actKind());
                lin(l.down, l.downB, f1, t, ldF, tmp, e);
            }
            VMath.addTo(pool, x, tmp, t * e);
            dump("layer_out-" + il, x, t * e);
            if (hook != null) {
                hook.after(il, l, x);
            }
        }
    }

    /** Called with the hidden state after each block (Qwen3-VL's deepstack taps some of them). */
    private interface LayerHook {
        void after(int il, Layer l, float[] x);
    }

    /** The patch embedding weights as a float matrix {@code [nEmbd][kPad]}: the sum of the file's conv tensors. */
    private float[] patchWeights(int kPad, String... names) {
        float[] w = null;
        for (String name : names) {
            if (!m.has(name)) {
                continue;
            }
            Mat c = m.convMat(name);
            float[] part = new float[c.rows * kPad];
            c.rows(0, c.rows, part, kPad);
            if (w == null) {
                w = part;
            } else {
                for (int i = 0; i < w.length; i++) {
                    w[i] += part[i];
                }
            }
        }
        if (w == null) {
            throw new IllegalArgumentException(m.path.getFileName() + ": no patch embedding tensor");
        }
        return w;
    }

    /**
     * The flattened patches of the image, one row per patch ({@code 3 * p * p} values in the conv weights' order:
     * channel, row, column; zero padded to {@code kPad}) for the patches {@code order[i] = (py, px)} in the given order.
     */
    private float[] unfold(Pixels px, int[] orderY, int[] orderX, int kPad) {
        int p = m.patchSize;
        int w = px.width();
        float[] src = px.hwc();
        int t = orderY.length;
        float[] out = new float[t * kPad];
        pool.forRange(t, 8, (a, b) -> {
            for (int i = a; i < b; i++) {
                int o = i * kPad;
                int y0 = orderY[i] * p;
                int x0 = orderX[i] * p;
                for (int c = 0; c < 3; c++) {
                    for (int ky = 0; ky < p; ky++) {
                        int s = 3 * ((y0 + ky) * w + x0) + c;
                        int d = o + c * p * p + ky * p;
                        for (int kx = 0; kx < p; kx++) {
                            out[d + kx] = src[s + 3 * kx];
                        }
                    }
                }
            }
        });
        return out;
    }

    private float[] patchEmbed(float[] patches, int t, int kPad, float[] w, float[] bias) {
        int e = m.nEmbd;
        float[] x = new float[t * e];
        VMath.gemm(pool, (r0, n, dst) -> System.arraycopy(w, r0 * kPad, dst, 0, n * kPad), e, kPad, bias, patches, t, x, e);
        return x;
    }

    // ------------------------------------------------------------------ SigLIP (SmolVLM, Gemma 3)

    private Encoded siglip(Pixels px) {
        int p = m.patchSize;
        if (px.width() % p != 0 || px.height() % p != 0) {
            throw new IllegalArgumentException("image " + px.width() + "x" + px.height() + " is not a multiple of the patch size " + p);
        }
        int gx = px.width() / p;
        int gy = px.height() / p;
        int t = gx * gy;
        int e = m.nEmbd;
        int kPad = VMath.pad(3 * p * p);
        int[] oy = new int[t];
        int[] ox = new int[t];
        for (int i = 0; i < t; i++) {
            oy[i] = i / gx;
            ox[i] = i % gx;
        }
        float[] x = patchEmbed(unfold(px, oy, ox, kPad), t, kPad, patchWeights(kPad, "v.patch_embd.weight"),
                m.vecOpt("v.patch_embd.bias"));
        Mat pos = m.mat("v.position_embd.weight");
        if (pos.rows != t || pos.cols != e) {
            throw new IllegalArgumentException("position embeddings " + pos + " do not fit " + gx + "x" + gy + " patches of width " + e);
        }
        float[] pe = pos.toFloats();
        VMath.addTo(pool, x, pe, t * e);
        float[] preW = m.vecOpt("v.pre_ln.weight");
        if (preW != null) {
            norm(false, x, t, e, preW, m.vecOpt("v.pre_ln.bias"), x);
        }
        blocks(x, t, false, null, null, 0, null, 0, null);
        float[] pw = m.vecOpt("v.post_ln.weight");
        if (pw != null) {
            norm(false, x, t, e, pw, m.vecOpt("v.post_ln.bias"), x);
        }
        return switch (m.projector) {
            case IDEFICS3 -> idefics3Projector(x, gx, gy);
            case GEMMA3 -> gemma3Projector(x, gx, gy);
            default -> throw new IllegalStateException();
        };
    }

    /** Pixel shuffle: each {@code s x s} block of patches becomes one token of {@code s*s*E} features, then a linear map. */
    private Encoded idefics3Projector(float[] x, int gx, int gy) {
        int s = m.nMerge;
        int e = m.nEmbd;
        int ox = gx / s;
        int oy = gy / s;
        int tokens = ox * oy;
        int feat = e * s * s;
        float[] shuffled = new float[tokens * feat];
        for (int by = 0; by < oy; by++) {
            for (int bx = 0; bx < ox; bx++) {
                int dst = (by * ox + bx) * feat;
                for (int ys = 0; ys < s; ys++) {
                    for (int xs = 0; xs < s; xs++) {
                        System.arraycopy(x, ((by * s + ys) * gx + bx * s + xs) * e, shuffled, dst + (ys * s + xs) * e, e);
                    }
                }
            }
        }
        Mat fc = m.mat("mm.model.fc.weight");
        float[] out = new float[tokens * fc.rows];
        lin(fc, null, shuffled, tokens, feat, out, fc.rows);
        return new Encoded(out, tokens, ox, oy);
    }

    /** Average pooling over {@code s x s} blocks of patches, RMS norm with the projector's weight, linear map. */
    private Encoded gemma3Projector(float[] x, int gx, int gy) {
        int s = m.nMerge;
        int e = m.nEmbd;
        int ox = gx / s;
        int oy = gy / s;
        int tokens = ox * oy;
        float[] pooled = new float[tokens * e];
        float inv = 1f / (s * s);
        for (int by = 0; by < oy; by++) {
            for (int bx = 0; bx < ox; bx++) {
                int dst = (by * ox + bx) * e;
                for (int ys = 0; ys < s; ys++) {
                    for (int xs = 0; xs < s; xs++) {
                        int src = ((by * s + ys) * gx + bx * s + xs) * e;
                        for (int i = 0; i < e; i++) {
                            pooled[dst + i] += x[src + i];
                        }
                    }
                }
                for (int i = 0; i < e; i++) {
                    pooled[dst + i] *= inv;
                }
            }
        }
        float[] nw = m.vec("mm.soft_emb_norm.weight");
        float[] normed = new float[tokens * e];
        VMath.rmsNorm(pool, pooled, tokens, e, nw, m.eps, normed);
        // mm.input_projection.weight is [in = e][out = text width] in the file's rows: out = normed . W
        Mat pj = m.mat("mm.input_projection.weight");
        float[] w = pj.toFloats(); // [e][nOut]
        int nOut = pj.cols;
        float[] wt = new float[nOut * e];
        for (int i = 0; i < e; i++) {
            for (int o = 0; o < nOut; o++) {
                wt[o * e + i] = w[i * nOut + o];
            }
        }
        float[] out = new float[tokens * nOut];
        VMath.gemm(pool, (r0, n, dst) -> System.arraycopy(wt, r0 * e, dst, 0, n * e), nOut, e, null, normed, tokens, out, nOut);
        return new Encoded(out, tokens, ox, oy);
    }

    // ------------------------------------------------------------------ Qwen-VL

    private Encoded qwen(Pixels px) {
        int p = m.patchSize;
        int mg = m.nMerge;
        if (px.width() % (p * mg) != 0 || px.height() % (p * mg) != 0) {
            throw new IllegalArgumentException("image " + px.width() + "x" + px.height() + " is not a multiple of " + p * mg);
        }
        int gx = px.width() / p;
        int gy = px.height() / p;
        int t = gx * gy;
        int e = m.nEmbd;
        boolean rms = m.projector == ClipModel.Projector.QWEN25VL;
        // patch order: merge blocks of mg x mg patches, block rows first (the order the merger concatenates)
        int[] oy = new int[t];
        int[] ox = new int[t];
        int n = 0;
        for (int y = 0; y < gy; y += mg) {
            for (int x = 0; x < gx; x += mg) {
                for (int dy = 0; dy < mg; dy++) {
                    for (int dx = 0; dx < mg; dx++) {
                        oy[n] = y + dy;
                        ox[n] = x + dx;
                        n++;
                    }
                }
            }
        }
        int mm = mg * mg;
        int bw = gx / mg;
        int bh = gy / mg;
        // window attention (Qwen2.5-VL): blocks are regrouped so that each window is a contiguous range
        int[] blockOrder = new int[bw * bh]; // blockOrder[dst] = source block
        int[] winStart = null;
        int nWin = 0;
        if (m.nWaPattern > 0) {
            int ws = m.windowSize > 0 ? m.windowSize : 112;
            int gw = Math.max(1, ws / p / mg);
            List<Integer> starts = new ArrayList<>();
            int dst = 0;
            for (int y = 0; y < bh; y += gw) {
                for (int x = 0; x < bw; x += gw) {
                    int wh = Math.min(gw, bh - y);
                    int wwid = Math.min(gw, bw - x);
                    starts.add(dst * mm);
                    for (int dy = 0; dy < wh; dy++) {
                        for (int dx = 0; dx < wwid; dx++) {
                            blockOrder[dst++] = (y + dy) * bw + (x + dx);
                        }
                    }
                }
            }
            starts.add(dst * mm);
            nWin = starts.size() - 1;
            winStart = starts.stream().mapToInt(Integer::intValue).toArray();
            int[] ny = new int[t];
            int[] nx = new int[t];
            for (int d = 0; d < bw * bh; d++) {
                System.arraycopy(oy, blockOrder[d] * mm, ny, d * mm, mm);
                System.arraycopy(ox, blockOrder[d] * mm, nx, d * mm, mm);
            }
            oy = ny;
            ox = nx;
        } else {
            for (int i = 0; i < blockOrder.length; i++) {
                blockOrder[i] = i;
            }
        }
        int kPad = VMath.pad(3 * p * p);
        float[] x = patchEmbed(unfold(px, oy, ox, kPad), t, kPad,
                patchWeights(kPad, "v.patch_embd.weight", "v.patch_embd.weight.1"), m.vecOpt("v.patch_embd.bias"));
        if (m.projector == ClipModel.Projector.QWEN3VL) { // learned positions, interpolated to the grid
            VMath.addTo(pool, x, interpolatedPositions(m.mat("v.position_embd.weight"), gx, gy, oy, ox), t * e);
        }
        float[] preW = m.vecOpt("v.pre_ln.weight");
        if (preW != null) {
            norm(rms, x, t, e, preW, m.vecOpt("v.pre_ln.bias"), x);
        }
        dump("inp", x, t * e);
        List<float[]> deep = new ArrayList<>();
        int mmTok = mm;
        LayerHook hook = m.projector != ClipModel.Projector.QWEN3VL ? null : (il, l, hidden) -> {
            if (l.hasDeepstack()) {
                deep.add(deepstackFeature(l, hidden, t / mmTok, e * mmTok));
            }
        };
        blocks(x, t, rms, oy, ox, 10000f, winStart, nWin, hook);
        float[] pw = m.vecOpt("v.post_ln.weight");
        if (pw != null) {
            norm(rms, x, t, e, pw, m.vecOpt("v.post_ln.bias"), x);
        }
        // merger: mg*mg consecutive tokens form one row of mm*E features
        int tokens = t / mm;
        int feat = e * mm;
        Mat w0 = m.mat("mm.0.weight");
        Mat w2 = m.mat("mm.2.weight");
        float[] hid = new float[tokens * VMath.pad(w0.rows)];
        lin(w0, m.vecOpt("mm.0.bias"), x, tokens, feat, hid, VMath.pad(w0.rows));
        VMath.activate(pool, hid, hid.length, 0);
        float[] merged = new float[tokens * w2.rows];
        lin(w2, m.vecOpt("mm.2.bias"), hid, tokens, VMath.pad(w0.rows), merged, w2.rows);
        float[] out = merged;
        if (m.nWaPattern > 0) { // back to raster order of the merged grid
            out = new float[merged.length];
            int dim = w2.rows;
            for (int d = 0; d < tokens; d++) {
                System.arraycopy(merged, d * dim, out, blockOrder[d] * dim, dim);
            }
        }
        return new Encoded(out, tokens, bw, bh, deep.isEmpty() ? null : deep.toArray(new float[0][]));
    }

    /**
     * Qwen3-VL deepstack: the hidden state of a tapped block, 2 x 2 tokens per row, LayerNorm over the merged row, then
     * fc1, GELU, fc2 (the same shape as the final merger): {@code tokens x projection_dim}.
     */
    private float[] deepstackFeature(Layer l, float[] hidden, int tokens, int feat) {
        float[] normed = new float[tokens * feat];
        VMath.layerNorm(pool, hidden, tokens, feat, l.dsNormW, l.dsNormB, m.eps, normed);
        float[] hid = new float[tokens * VMath.pad(l.dsFc1.rows)];
        lin(l.dsFc1, l.dsFc1B, normed, tokens, feat, hid, VMath.pad(l.dsFc1.rows));
        VMath.activate(pool, hid, hid.length, 0);
        float[] out = new float[tokens * l.dsFc2.rows];
        lin(l.dsFc2, l.dsFc2B, hid, tokens, VMath.pad(l.dsFc1.rows), out, l.dsFc2.rows);
        return out;
    }

    /**
     * Qwen3-VL's learned position embeddings (a square grid of {@code n x n}), resized to the image's patch grid with
     * bilinear interpolation, corners aligned (ggml's {@code BILINEAR | ALIGN_CORNERS}), and listed in the order of the
     * tokens ({@code oy[i], ox[i]} = row and column of token i).
     */
    static float[] interpolatedPositions(Mat pos, int gx, int gy, int[] oy, int[] ox) {
        int n = (int) Math.round(Math.sqrt(pos.rows));
        if (n * n != pos.rows) {
            throw new IllegalArgumentException("position embeddings " + pos + " are not a square grid");
        }
        int e = pos.cols;
        float[] tab = pos.toFloats();
        float[] out = new float[oy.length * e];
        float sfy = gy > 1 && n > 1 ? (float) (gy - 1) / (n - 1) : (float) gy / n;
        float sfx = gx > 1 && n > 1 ? (float) (gx - 1) / (n - 1) : (float) gx / n;
        for (int i = 0; i < oy.length; i++) {
            float y = oy[i] / sfy;
            float x = ox[i] / sfx;
            int y0 = Math.max(0, (int) Math.floor(y));
            int x0 = Math.max(0, (int) Math.floor(x));
            int y1 = Math.min(y0 + 1, n - 1);
            int x1 = Math.min(x0 + 1, n - 1);
            y0 = Math.min(y0, n - 1);
            x0 = Math.min(x0, n - 1);
            float dy = Math.min(Math.max(y - y0, 0f), 1f);
            float dx = Math.min(Math.max(x - x0, 0f), 1f);
            for (int c = 0; c < e; c++) {
                float v00 = tab[(y0 * n + x0) * e + c];
                float v01 = tab[(y0 * n + x1) * e + c];
                float v10 = tab[(y1 * n + x0) * e + c];
                float v11 = tab[(y1 * n + x1) * e + c];
                out[i * e + c] = (v00 * (1 - dx) + v01 * dx) * (1 - dy) + (v10 * (1 - dx) + v11 * dx) * dy;
            }
        }
        return out;
    }
}
