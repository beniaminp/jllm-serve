package org.robomiri.jllm.diffusion;

import java.io.IOException;
import java.nio.file.Path;

/**
 * {@code WanTransformer3DModel} of Wan 2.1 (text-to-video), read from the diffusers checkpoint: a <em>diffusion transformer</em> over
 * the tokens of a video latent. It predicts the velocity {@code v = ε − x₀} of a flow-matching model.
 *
 * <pre>
 *   tokens  = patchify(latent [F, h, w, 16])           2 × 2 patches of one frame → [F·h/2·w/2, 64] → linear → [N, dim]
 *   temb    = MLP(sinusoidal(t));  tproj = Linear(silu(temb)) → 6 chunks of dim (shift, scale, gate for attention; for the feed-forward)
 *   text    = Linear(gelu_tanh(Linear(umt5 embeddings [L, 4096])))                                    → [L, dim]
 *   block:  x += gate₁ · SelfAttn(modulate(LN(x)), 3-D rotary positions, RMS-normed q and k over all heads)
 *           x +=         CrossAttn(LN_affine(x), text)                                                  (RMS-normed q and k)
 *           x += gate₂ · FFN(modulate(LN(x)))                                                          (linear, GELU tanh, linear)
 *   out:    proj_out(modulate(LN(x))) → unpatchify → [F, h, w, 16]
 * </pre>
 *
 * Every parameter {@code scale_shift_table [6, dim]} of a block is added to the time projection before it is used as shift / scale /
 * gate. The 3-D rotary embedding splits each head's 128 channels into 44 for the frame axis and 42 + 42 for height and width.
 */
public final class WanTransformer implements AutoCloseable {

    private record Attn(Tn q, Tn qb, Tn k, Tn kb, Tn v, Tn vb, Tn o, Tn ob, Tn nq, Tn nk) {
    }

    private record Block(Tn table, Attn a1, Attn a2, Tn n2g, Tn n2b, Tn f0w, Tn f0b, Tn f2w, Tn f2b) {
    }

    private final Ops ops;
    private final Checkpoint ckpt;
    private final Loader ld;
    public final int dim;
    public final int heads;
    public final int headDim;
    public final int layers;
    public final int inChannels;
    public final int outChannels;
    public final int textDim;
    private final int freqDim;
    private final int[] patch;
    private final float eps;
    private final int ropeMax;
    private final Tn patchW, patchB;
    private final Tn timeW1, timeB1, timeW2, timeB2, projW, projB;
    private final Tn textW1, textB1, textW2, textB2;
    private final Block[] blocks;
    private final Tn outTable, outW, outB;

    public WanTransformer(Ops ops, Path dir) throws IOException {
        this.ops = ops;
        this.ckpt = Checkpoint.open(dir, "diffusion_pytorch_model");
        this.ld = new Loader(ops, ckpt);
        heads = ckpt.getInt("num_attention_heads", 40);
        headDim = ckpt.getInt("attention_head_dim", 128);
        dim = heads * headDim;
        layers = ckpt.getInt("num_layers", 40);
        inChannels = ckpt.getInt("in_channels", 16);
        outChannels = ckpt.getInt("out_channels", inChannels);
        textDim = ckpt.getInt("text_dim", 4096);
        freqDim = ckpt.getInt("freq_dim", 256);
        eps = (float) ckpt.getDouble("eps", 1e-6);
        ropeMax = ckpt.getInt("rope_max_seq_len", 1024);
        patch = ckpt.getIntArray("patch_size", 1, 2, 2);
        if (ckpt.config.get("image_dim") != null || ckpt.config.get("added_kv_proj_dim") != null || !ckpt.getBool("cross_attn_norm", true)
                || !"rms_norm_across_heads".equals(ckpt.getString("qk_norm", "rms_norm_across_heads")) || patch.length != 3 || patch[0] != 1) {
            throw new IOException("unsupported Wan configuration in " + dir + " (text-to-video with qk RMS norm across heads, patch (1, p, p))");
        }
        // the patch embedding is a convolution with kernel = stride = patch: a linear layer over the (kh, kw, channel) patch
        float[] pw = ckpt.get("patch_embedding.weight").toFloatArray(); // [dim, C, 1, ph, pw]
        int pp = patch[1] * patch[2];
        float[] w2 = new float[dim * pp * inChannels];
        for (int o = 0; o < dim; o++) {
            for (int c = 0; c < inChannels; c++) {
                for (int k = 0; k < pp; k++) {
                    w2[o * pp * inChannels + k * inChannels + c] = pw[(o * inChannels + c) * pp + k];
                }
            }
        }
        patchW = ld.adopt(ops.matrix(w2, Ops.Kind.LINEAR, dim, pp * inChannels));
        patchB = ld.vec("patch_embedding.bias");
        timeW1 = ld.lin("condition_embedder.time_embedder.linear_1.weight");
        timeB1 = ld.vec("condition_embedder.time_embedder.linear_1.bias");
        timeW2 = ld.lin("condition_embedder.time_embedder.linear_2.weight");
        timeB2 = ld.vec("condition_embedder.time_embedder.linear_2.bias");
        projW = ld.lin("condition_embedder.time_proj.weight");
        projB = ld.vec("condition_embedder.time_proj.bias");
        textW1 = ld.lin("condition_embedder.text_embedder.linear_1.weight");
        textB1 = ld.vec("condition_embedder.text_embedder.linear_1.bias");
        textW2 = ld.lin("condition_embedder.text_embedder.linear_2.weight");
        textB2 = ld.vec("condition_embedder.text_embedder.linear_2.bias");
        blocks = new Block[layers];
        for (int i = 0; i < layers; i++) {
            String p = "blocks." + i + ".";
            blocks[i] = new Block(ld.vec(p + "scale_shift_table"), attn(p + "attn1."), attn(p + "attn2."), ld.vec(p + "norm2.weight"), ld.vec(p + "norm2.bias"),
                    ld.lin(p + "ffn.net.0.proj.weight"), ld.vec(p + "ffn.net.0.proj.bias"), ld.lin(p + "ffn.net.2.weight"), ld.vec(p + "ffn.net.2.bias"));
        }
        outTable = ld.vec("scale_shift_table");
        outW = ld.lin("proj_out.weight");
        outB = ld.vec("proj_out.bias");
    }

    private Attn attn(String p) {
        return new Attn(ld.lin(p + "to_q.weight"), ld.vec(p + "to_q.bias"), ld.lin(p + "to_k.weight"), ld.vec(p + "to_k.bias"),
                ld.lin(p + "to_v.weight"), ld.vec(p + "to_v.bias"), ld.lin(p + "to_out.0.weight"), ld.vec(p + "to_out.0.bias"),
                ld.vec(p + "norm_q.weight"), ld.vec(p + "norm_k.weight"));
    }

    /** The text embeddings {@code [L, textDim]} (host floats) projected to the model width: {@code [1, L, dim]}; the caller frees it. */
    public Tn encodeText(float[] text, int length) {
        Tn in = ops.upload(text, 1, length, textDim);
        Tn a = ops.linear(in, textW1, textB1, null, Ops.Act.GELU_TANH);
        ops.free(in);
        Tn b = ops.linear(a, textW2, textB2, null, Ops.Act.NONE);
        ops.free(a);
        return b;
    }

    /** One tensor of rotary angles per token: {@code [N, hd/2]} cosines and sines (frame, height, width axes: 22 + 21 + 21 for 128). */
    private float[][] rotary(int f, int ph, int pw) {
        int hw = 2 * (headDim / 6);
        int tDim = headDim - 2 * hw;
        int half = headDim / 2;
        int n = f * ph * pw;
        float[] cos = new float[n * half];
        float[] sin = new float[n * half];
        int[] dims = {tDim, hw, hw};
        for (int fi = 0; fi < f; fi++) {
            for (int hi = 0; hi < ph; hi++) {
                for (int wi = 0; wi < pw; wi++) {
                    int tok = (fi * ph + hi) * pw + wi;
                    int[] pos = {fi, hi, wi};
                    int o = 0;
                    for (int ax = 0; ax < 3; ax++) {
                        for (int i = 0; i < dims[ax] / 2; i++) {
                            double freq = 1.0 / Math.pow(10000.0, (double) (2 * i) / dims[ax]);
                            double ang = pos[ax] * freq;
                            cos[tok * half + o] = (float) Math.cos(ang);
                            sin[tok * half + o] = (float) Math.sin(ang);
                            o++;
                        }
                    }
                }
            }
        }
        return new float[][]{cos, sin};
    }

    /**
     * One evaluation of the network.
     *
     * @param x    latent {@code [F, h, w, C]} (frames-major, channels last)
     * @param text the projected text embeddings from {@link #encodeText} {@code [L, dim]}
     * @return the velocity prediction in the layout of {@code x}
     */
    public float[] forward(float[] x, int f, int h, int w, float timestep, Tn text) {
        if (f % patch[0] != 0 || h % patch[1] != 0 || w % patch[2] != 0) {
            throw new IllegalArgumentException("latent " + f + "x" + h + "x" + w + " does not fit the patch " + java.util.Arrays.toString(patch));
        }
        int c = inChannels;
        int fp = f / patch[0];
        int hp = h / patch[1];
        int wp = w / patch[2];
        int n = fp * hp * wp;
        int ph = patch[1];
        int pw = patch[2];
        // patchify: token (f, i, j), feature (kh, kw, channel)
        float[] tok = new float[n * ph * pw * c];
        for (int fi = 0; fi < f; fi++) {
            for (int i = 0; i < hp; i++) {
                for (int j = 0; j < wp; j++) {
                    int t = (fi * hp + i) * wp + j;
                    for (int kh = 0; kh < ph; kh++) {
                        for (int kw = 0; kw < pw; kw++) {
                            System.arraycopy(x, ((fi * h + i * ph + kh) * w + j * pw + kw) * c, tok, (t * ph * pw + kh * pw + kw) * c, c);
                        }
                    }
                }
            }
        }
        float[][] rot = rotary(fp, hp, wp);
        Tn cosT = ops.vector(rot[0], n, headDim / 2);
        Tn sinT = ops.vector(rot[1], n, headDim / 2);
        // time conditioning
        Tn tin = ops.upload(UNet.timestepEmbedding(timestep, freqDim, true, 0), 1, freqDim);
        Tn t1 = ops.linear(tin, timeW1, timeB1, null, Ops.Act.SILU);
        ops.free(tin);
        Tn temb = ops.linear(t1, timeW2, timeB2, null, Ops.Act.NONE);
        ops.free(t1);
        Tn ts = ops.silu(temb);
        Tn tproj = ops.linear(ts, projW, projB, null, Ops.Act.NONE); // [1, 6·dim]
        ops.free(ts);
        Tn tokens = ops.upload(tok, 1, n, ph * pw * c);
        Tn hs = ops.linear(tokens, patchW, patchB, null, Ops.Act.NONE);
        ops.free(tokens);
        int d = dim;
        for (Block b : blocks) {
            // 1. self-attention with shift / scale / gate from the modulation
            Tn n1 = ops.layerNorm(hs, null, null, eps);
            Tn m1 = ops.modulate(n1, tproj, b.table, new int[]{6 * d, d, 0, d, 0}, 1);
            ops.free(n1);
            Tn a1 = attention(b.a1, m1, m1, cosT, sinT, n, null);
            ops.free(m1);
            Tn x1 = ops.gatedAdd(hs, a1, tproj, b.table, 6 * d, 2 * d, 2 * d, 1);
            ops.free(a1);
            ops.free(hs);
            // 2. cross-attention to the text
            Tn n2 = ops.layerNorm(x1, b.n2g, b.n2b, eps);
            Tn a2 = attention(b.a2, n2, text, null, null, n, x1);
            ops.free(n2);
            ops.free(x1);
            // 3. feed-forward
            Tn n3 = ops.layerNorm(a2, null, null, eps);
            Tn m3 = ops.modulate(n3, tproj, b.table, new int[]{6 * d, 4 * d, 3 * d, 4 * d, 3 * d}, 1);
            ops.free(n3);
            Tn g = ops.linear(m3, b.f0w, b.f0b, null, Ops.Act.GELU_TANH);
            ops.free(m3);
            Tn ff = ops.linear(g, b.f2w, b.f2b, null, Ops.Act.NONE);
            ops.free(g);
            hs = ops.gatedAdd(a2, ff, tproj, b.table, 6 * d, 5 * d, 5 * d, 1);
            ops.free(ff);
            ops.free(a2);
        }
        ops.free(tproj);
        ops.free(cosT);
        ops.free(sinT);
        Tn no = ops.layerNorm(hs, null, null, eps);
        ops.free(hs);
        Tn mo = ops.modulate(no, temb, outTable, new int[]{d, 0, 0, d, 0}, 1); // scale = table[1] + temb, shift = table[0] + temb
        ops.free(no);
        ops.free(temb);
        Tn y = ops.linear(mo, outW, outB, null, Ops.Act.NONE);
        ops.free(mo);
        float[] yo = ops.download(y);
        ops.free(y);
        // unpatchify: feature (kh, kw, channel) of token (f, i, j) is pixel (i·ph + kh, j·pw + kw)
        int oc = outChannels;
        float[] out = new float[f * h * w * oc];
        for (int fi = 0; fi < f; fi++) {
            for (int i = 0; i < hp; i++) {
                for (int j = 0; j < wp; j++) {
                    int t = (fi * hp + i) * wp + j;
                    for (int kh = 0; kh < ph; kh++) {
                        for (int kw = 0; kw < pw; kw++) {
                            System.arraycopy(yo, (t * ph * pw + kh * pw + kw) * oc, out, ((fi * h + i * ph + kh) * w + j * pw + kw) * oc, oc);
                        }
                    }
                }
            }
        }
        return out;
    }

    /** Attention with RMS-normed q / k over all heads (and rotary positions in self-attention); {@code residual + to_out(attention)}. */
    private Tn attention(Attn a, Tn x, Tn ctx, Tn cos, Tn sin, int period, Tn residual) {
        Tn q = ops.linear(x, a.q, a.qb, null, Ops.Act.NONE);
        Tn k = ops.linear(ctx, a.k, a.kb, null, Ops.Act.NONE);
        Tn v = ops.linear(ctx, a.v, a.vb, null, Ops.Act.NONE);
        Tn qn = ops.rmsNorm(q, a.nq, eps);
        Tn kn = ops.rmsNorm(k, a.nk, eps);
        ops.free(q);
        ops.free(k);
        if (cos != null) {
            Tn qr = ops.rope(qn, cos, sin, heads, period);
            Tn kr = ops.rope(kn, cos, sin, heads, period);
            ops.free(qn);
            ops.free(kn);
            qn = qr;
            kn = kr;
        }
        Tn o = ops.attention(qn, kn, v, heads, false);
        ops.free(qn);
        ops.free(kn);
        ops.free(v);
        Tn out = ops.linear(o, a.o, a.ob, residual, Ops.Act.NONE);
        ops.free(o);
        return out;
    }

    public long weightBytes() {
        return ckpt.bytes();
    }

    @Override
    public void close() {
        ld.free();
        ckpt.close();
    }
}
