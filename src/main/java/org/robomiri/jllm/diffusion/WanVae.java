package org.robomiri.jllm.diffusion;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * The decoder of {@code AutoencoderKLWan} (Wan 2.1's causal 3-D VAE): 16-channel latents of {@code 8 × 8} pixels and 4 frames each become
 * RGB video frames, the first latent frame giving one picture and every further one four.
 *
 * <pre>
 *   z → post_quant 1×1×1 → causal 3×3×3 conv (16 → 384) → ResNet · spatial attention · ResNet
 *     → 4 up blocks [3 ResNets, then a resampler: 2× in space (nearest + 3×3 conv, channels halve), the first two also 2× in time]
 *     → RMS norm, SiLU, causal 3×3×3 conv (96 → 3), clamp to [−1, 1]
 * </pre>
 *
 * <b>Causal convolutions.</b> A 3 × 3 × 3 convolution looks at the current frame and the two before it (zeros before the first), so
 * {@code out[t] = Σ_k W_k ⋆ x[t − 2 + k]}: three 2-D convolutions of shifted copies of the clip, accumulated through the convolution's
 * residual input ({@link Ops#frameShift}). diffusers decodes frame by frame with a feature cache that holds exactly those two previous
 * frames; the whole-clip form here computes the same numbers.
 * <b>Temporal upsampling</b> ({@code upsample3d}): the first frame passes through unchanged; the others go through a causal
 * {@code (3, 1, 1)} convolution to twice the channels (zero history before the second frame, as the cache starts with "Rep") whose
 * two halves become two consecutive frames.
 */
public final class WanVae implements AutoCloseable {

    private record Conv3(Tn[] w, Tn b) {
    }

    private record Res(Tn n1, Conv3 c1, Tn n2, Conv3 c2, Tn shortW, Tn shortB) {
    }

    private record Attn(Tn norm, Tn q, Tn qb, Tn k, Tn kb, Tn v, Tn vb, Tn proj, Tn projB) {
    }

    private record Up(Res[] res, boolean up, boolean temporal, Tn convW, Tn convB, Tn[] timeA, Tn[] timeB, Tn timeBiasA, Tn timeBiasB) {
    }

    private static final float NORM_EPS = 1e-24f; // F.normalize clamps the norm at 1e-12

    private final Ops ops;
    private final Checkpoint ckpt;
    private final Loader ld;
    public final int zDim;
    public final float[] latentsMean;
    public final float[] latentsStd;
    private final Tn postW, postB;
    private final Conv3 convIn, convOut;
    private final Res[] mid = new Res[2];
    private final Attn midAttn;
    private final Up[] ups;
    private final Tn normOut;

    public WanVae(Ops ops, Path dir) throws IOException {
        this.ops = ops;
        this.ckpt = Checkpoint.open(dir, "diffusion_pytorch_model");
        this.ld = new Loader(ops, ckpt);
        zDim = ckpt.getInt("z_dim", 16);
        int[] mult = ckpt.getIntArray("dim_mult", 1, 2, 4, 4);
        int blocks = ckpt.getInt("num_res_blocks", 2);
        boolean[] down = new boolean[mult.length - 1];
        List<?> td = (List<?>) ckpt.config.get("temperal_downsample");
        for (int i = 0; i < down.length; i++) {
            down[i] = td != null && i < td.size() && Boolean.TRUE.equals(td.get(i));
        }
        if (ckpt.getBool("is_residual", false) || ckpt.config.get("patch_size") != null) {
            throw new IOException("unsupported Wan VAE variant in " + dir + " (the 2.1 decoder without residual up blocks / patching)");
        }
        latentsMean = floats("latents_mean", zDim, 0f);
        latentsStd = floats("latents_std", zDim, 1f);
        postW = ld.lin("post_quant_conv.weight");
        postB = ld.vec("post_quant_conv.bias");
        convIn = conv3("decoder.conv_in");
        mid[0] = res("decoder.mid_block.resnets.0.");
        midAttn = attn("decoder.mid_block.attentions.0.");
        mid[1] = res("decoder.mid_block.resnets.1.");
        ups = new Up[mult.length];
        for (int i = 0; i < mult.length; i++) {
            String p = "decoder.up_blocks." + i + ".";
            Res[] rs = new Res[blocks + 1];
            for (int j = 0; j < rs.length; j++) {
                rs[j] = res(p + "resnets." + j + ".");
            }
            boolean up = i != mult.length - 1;
            boolean temporal = up && down[mult.length - 2 - i]; // temperal_upsample = reversed temperal_downsample
            if (!up) {
                ups[i] = new Up(rs, false, false, null, null, null, null, null, null);
                continue;
            }
            String u = p + "upsamplers.0.";
            Tn convW = ld.conv(u + "resample.1.weight");
            Tn convB = ld.vec(u + "resample.1.bias");
            Tn[] ta = null;
            Tn[] tb = null;
            Tn ba = null;
            Tn bb = null;
            if (temporal) { // time_conv [2C, C, 3, 1, 1]: output channels 0..C-1 are the first new frame, C..2C-1 the second
                float[] w = ckpt.get(u + "time_conv.weight").toFloatArray();
                float[] b = ckpt.get(u + "time_conv.bias").toFloatArray();
                int c2 = b.length;
                int c = c2 / 2;
                ta = new Tn[3];
                tb = new Tn[3];
                for (int kt = 0; kt < 3; kt++) {
                    float[] a = new float[c * c];
                    float[] bw = new float[c * c];
                    for (int o = 0; o < c; o++) {
                        for (int ci = 0; ci < c; ci++) {
                            a[o * c + ci] = w[(o * c + ci) * 3 + kt];
                            bw[o * c + ci] = w[((o + c) * c + ci) * 3 + kt];
                        }
                    }
                    ta[kt] = ld.adopt(ops.matrix(a, Ops.Kind.LINEAR, c, c));
                    tb[kt] = ld.adopt(ops.matrix(bw, Ops.Kind.LINEAR, c, c));
                }
                ba = ld.adopt(ops.vector(java.util.Arrays.copyOfRange(b, 0, c), c));
                bb = ld.adopt(ops.vector(java.util.Arrays.copyOfRange(b, c, c2), c));
            }
            ups[i] = new Up(rs, true, temporal, convW, convB, ta, tb, ba, bb);
        }
        normOut = ld.vec("decoder.norm_out.gamma");
        convOut = conv3("decoder.conv_out");
    }

    private float[] floats(String key, int n, float def) {
        float[] out = new float[n];
        java.util.Arrays.fill(out, def);
        if (ckpt.config.get(key) instanceof List<?> l) {
            for (int i = 0; i < n && i < l.size(); i++) {
                out[i] = ((Number) l.get(i)).floatValue();
            }
        }
        return out;
    }

    /** A causal 3 × 3 × 3 convolution {@code [O, I, 3, 3, 3]} as three 2-D weights (one per temporal tap). */
    private Conv3 conv3(String p) {
        SafeTensorsShape s = new SafeTensorsShape(ckpt.get(p + ".weight").shape());
        float[] w = ckpt.get(p + ".weight").toFloatArray();
        int o = s.d[0];
        int in = s.d[1];
        Tn[] parts = new Tn[3];
        for (int kt = 0; kt < 3; kt++) {
            float[] part = new float[o * 9 * in];
            for (int oo = 0; oo < o; oo++) {
                for (int c = 0; c < in; c++) {
                    for (int k = 0; k < 9; k++) {
                        part[oo * 9 * in + k * in + c] = w[((oo * in + c) * 3 + kt) * 9 + k];
                    }
                }
            }
            parts[kt] = ld.adopt(ops.matrix(part, Ops.Kind.CONV, o, 3, 3, in));
        }
        return new Conv3(parts, ld.vec(p + ".bias"));
    }

    private record SafeTensorsShape(int[] d) {
        SafeTensorsShape(long[] s) {
            this(java.util.Arrays.stream(s).mapToInt(v -> (int) v).toArray());
        }
    }

    private Res res(String p) {
        boolean sc = ld.has(p + "conv_shortcut.weight");
        return new Res(ld.vec(p + "norm1.gamma"), conv3(p + "conv1"), ld.vec(p + "norm2.gamma"), conv3(p + "conv2"),
                sc ? ld.lin(p + "conv_shortcut.weight") : null, sc ? ld.vec(p + "conv_shortcut.bias") : null);
    }

    private Attn attn(String p) {
        float[] w = ckpt.get(p + "to_qkv.weight").toFloatArray();
        float[] b = ckpt.get(p + "to_qkv.bias").toFloatArray();
        int c = b.length / 3;
        Tn[] ws = new Tn[3];
        Tn[] bs = new Tn[3];
        for (int i = 0; i < 3; i++) {
            ws[i] = ld.adopt(ops.matrix(java.util.Arrays.copyOfRange(w, i * c * c, (i + 1) * c * c), Ops.Kind.LINEAR, c, c));
            bs[i] = ld.adopt(ops.vector(java.util.Arrays.copyOfRange(b, i * c, (i + 1) * c), c));
        }
        return new Attn(ld.vec(p + "norm.gamma"), ws[0], bs[0], ws[1], bs[1], ws[2], bs[2], ld.lin(p + "proj.weight"), ld.vec(p + "proj.bias"));
    }

    // ------------------------------------------------------------------ layers (all on [T, H, W, C] with the frames as the batch)

    /** Causal convolution; {@code residual} (same shape as the output) is added. Frees nothing. */
    private Tn causalConv(Tn x, Conv3 c, Tn residual) {
        Tn s2 = ops.frameShift(x, 2);
        Tn a = ops.conv2d(s2, c.w[0], c.b, 1, 1, 1, 1, 1, null, residual);
        ops.free(s2);
        Tn s1 = ops.frameShift(x, 1);
        Tn b = ops.conv2d(s1, c.w[1], null, 1, 1, 1, 1, 1, null, a);
        ops.free(s1);
        ops.free(a);
        Tn d = ops.conv2d(x, c.w[2], null, 1, 1, 1, 1, 1, null, b);
        ops.free(b);
        return d;
    }

    private Tn resBlock(Res r, Tn x) {
        Tn h = x;
        if (r.shortW != null) {
            h = ops.linear(x, r.shortW, r.shortB, null, Ops.Act.NONE);
        }
        Tn n1 = ops.rmsNorm(x, r.n1, NORM_EPS);
        Tn a1 = ops.silu(n1);
        ops.free(n1);
        Tn c1 = causalConv(a1, r.c1, null);
        ops.free(a1);
        Tn n2 = ops.rmsNorm(c1, r.n2, NORM_EPS);
        ops.free(c1);
        Tn a2 = ops.silu(n2);
        ops.free(n2);
        Tn out = causalConv(a2, r.c2, h);
        ops.free(a2);
        if (h != x) {
            ops.free(h);
        }
        ops.free(x);
        return out;
    }

    private Tn attention(Attn a, Tn x) {
        Tn n = ops.rmsNorm(x, a.norm, NORM_EPS);
        Tn q = ops.linear(n, a.q, a.qb, null, Ops.Act.NONE);
        Tn k = ops.linear(n, a.k, a.kb, null, Ops.Act.NONE);
        Tn v = ops.linear(n, a.v, a.vb, null, Ops.Act.NONE);
        ops.free(n);
        Tn o = ops.attention(q, k, v, 1, false); // one head over the H·W positions of each frame
        ops.free(q);
        ops.free(k);
        ops.free(v);
        Tn out = ops.linear(o, a.proj, a.projB, x, Ops.Act.NONE);
        ops.free(o);
        ops.free(x);
        return out;
    }

    private Tn upsample(Up u, Tn x) {
        int t = x.dim(0);
        int h = x.dim(1);
        int w = x.dim(2);
        Tn y = x;
        if (u.temporal && t > 1) {
            Tn first = ops.rows(x, 0, 1);
            Tn tail = ops.rows(x, 1, t - 1);
            Tn[] halves = new Tn[2];
            for (int half = 0; half < 2; half++) {
                Tn[] wk = half == 0 ? u.timeA : u.timeB;
                Tn bias = half == 0 ? u.timeBiasA : u.timeBiasB;
                Tn s2 = ops.frameShift(tail, 2);
                Tn a = ops.linear(s2, wk[0], bias, null, Ops.Act.NONE);
                ops.free(s2);
                Tn s1 = ops.frameShift(tail, 1);
                Tn b = ops.linear(s1, wk[1], null, a, Ops.Act.NONE);
                ops.free(s1);
                ops.free(a);
                halves[half] = ops.linear(tail, wk[2], null, b, Ops.Act.NONE);
                ops.free(b);
            }
            ops.free(tail);
            Tn ab = ops.stack(halves[0], halves[1]); // [2(T-1), H, W, C]: first the "a" frames, then the "b" frames
            ops.free(halves[0]);
            ops.free(halves[1]);
            int p = h * w;
            int c = ab.last();
            Tn pix = ops.swapAxes(ab, 1, 2, (t - 1) * p, t - 1, p, 2, c);          // [T-1, P, 2, C]: pixel-major
            ops.free(ab);
            Tn frames = ops.swapAxes(pix, t - 1, p, 2, 2 * (t - 1), h, w, c);     // [T-1, 2, P, C]: frame a then frame b of each step
            ops.free(pix);
            y = ops.stack(first, frames);
            ops.free(first);
            ops.free(frames);
            ops.free(x);
        }
        Tn out = ops.upsampleConv(y, u.convW, u.convB);
        ops.free(y);
        return out;
    }

    /**
     * Decodes latents.
     *
     * @param z {@code [F, h, w, z_dim]} (already un-normalised: {@code z / std + mean})
     * @return pixels {@code [F', H, W, 3]} in {@code [−1, 1]}, {@code F' = 1 + 4 (F − 1)}, {@code H = 8 h}, {@code W = 8 w}
     */
    public float[] decode(float[] z, int f, int h, int w) {
        Tn x = ops.upload(z, f, h, w, zDim);
        Tn p = ops.linear(x, postW, postB, null, Ops.Act.NONE);
        ops.free(x);
        x = causalConv(p, convIn, null);
        ops.free(p);
        x = resBlock(mid[0], x);
        x = attention(midAttn, x);
        x = resBlock(mid[1], x);
        for (Up u : ups) {
            for (Res r : u.res) {
                x = resBlock(r, x);
            }
            if (u.up) {
                x = upsample(u, x);
            }
        }
        Tn n = ops.rmsNorm(x, normOut, NORM_EPS);
        ops.free(x);
        Tn a = ops.silu(n);
        ops.free(n);
        Tn o = causalConv(a, convOut, null);
        ops.free(a);
        float[] px = ops.download(o);
        ops.free(o);
        for (int i = 0; i < px.length; i++) {
            px[i] = Math.max(-1f, Math.min(1f, px[i]));
        }
        return px;
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
