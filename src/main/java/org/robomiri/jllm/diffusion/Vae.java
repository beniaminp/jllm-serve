package org.robomiri.jllm.diffusion;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * {@code AutoencoderKL}: the variational auto-encoder between pixels {@code [B, H, W, 3]} (in -1..1) and latents
 * {@code [B, H/f, W/f, C]} (f = 2^(blocks-1) = 8 for Stable Diffusion, C = 4).
 *
 * <pre>
 *   encoder: conv_in → [ResNet × n, strided conv] per block → ResNet, attention, ResNet → norm, silu, conv_out → quant_conv
 *            → (mean, log-variance); a latent is mean + exp(½ logvar) · ε, scaled by scaling_factor
 *   decoder: post_quant_conv → conv_in → ResNet, attention, ResNet → [ResNet × (n+1), 2x upsample + conv] per block
 *            → norm, silu, conv_out
 * </pre>
 *
 * Large images are processed in overlapping tiles blended linearly, so the memory of the attention in the middle block
 * (one token per latent pixel) stays bounded.
 */
public final class Vae implements AutoCloseable {

    private record Res(Tn n1g, Tn n1b, Tn c1w, Tn c1b, Tn n2g, Tn n2b, Tn c2w, Tn c2b, Tn sw, Tn sb) {
    }

    private record Conv(Tn w, Tn b) {
    }

    private record Mid(Res r0, Tn ng, Tn nb, Tn q, Tn qb, Tn k, Tn kb, Tn v, Tn vb, Tn o, Tn ob, Res r1) {
    }

    private record Stage(Res[] res, Conv sample) {
    }

    private final Ops ops;
    private final Checkpoint ckpt;
    private final Loader ld;
    public final int latentChannels;
    public final int scale; // pixels per latent pixel
    public final float scalingFactor;
    public final boolean forceUpcast;
    private final int groups;
    private final float eps = 1e-6f;
    private final int blocks;

    private final Conv postQuant;
    private final Conv quant;
    private final Conv encIn;
    private final Stage[] enc;
    private final Mid encMid;
    private final Tn encNormG, encNormB;
    private final Conv encOut;
    private final Conv decIn;
    private final Mid decMid;
    private final Stage[] dec;
    private final Tn decNormG, decNormB;
    private final Conv decOut;
    /** Largest latent edge decoded in one piece; bigger images are tiled. */
    public int tileLatents = 64;
    /** Overlap of neighbouring tiles, in latent pixels. */
    public int tileOverlap = 8;

    public Vae(Ops ops, Path dir) throws IOException {
        this.ops = ops;
        this.ckpt = Checkpoint.open(dir, "diffusion_pytorch_model");
        this.ld = new Loader(ops, ckpt);
        int[] blockOut = ckpt.getIntArray("block_out_channels", 128, 256, 512, 512);
        blocks = blockOut.length;
        scale = 1 << (blocks - 1);
        latentChannels = ckpt.getInt("latent_channels", 4);
        scalingFactor = (float) ckpt.getDouble("scaling_factor", 0.18215);
        forceUpcast = ckpt.getBool("force_upcast", true);
        groups = ckpt.getInt("norm_num_groups", 32);
        int layers = ckpt.getInt("layers_per_block", 2);
        postQuant = ld.has("post_quant_conv.weight") ? new Conv(ld.conv("post_quant_conv.weight"), ld.vec("post_quant_conv.bias")) : null;
        quant = ld.has("quant_conv.weight") ? new Conv(ld.conv("quant_conv.weight"), ld.vec("quant_conv.bias")) : null;

        encIn = new Conv(ld.conv("encoder.conv_in.weight"), ld.vec("encoder.conv_in.bias"));
        enc = new Stage[blocks];
        for (int i = 0; i < blocks; i++) {
            String p = "encoder.down_blocks." + i + ".";
            Res[] rs = new Res[layers];
            for (int j = 0; j < layers; j++) {
                rs[j] = res(p + "resnets." + j + ".");
            }
            Conv dn = ld.has(p + "downsamplers.0.conv.weight")
                    ? new Conv(ld.conv(p + "downsamplers.0.conv.weight"), ld.vec(p + "downsamplers.0.conv.bias")) : null;
            enc[i] = new Stage(rs, dn);
        }
        encMid = mid("encoder.mid_block.");
        encNormG = ld.vec("encoder.conv_norm_out.weight");
        encNormB = ld.vec("encoder.conv_norm_out.bias");
        encOut = new Conv(ld.conv("encoder.conv_out.weight"), ld.vec("encoder.conv_out.bias"));

        decIn = new Conv(ld.conv("decoder.conv_in.weight"), ld.vec("decoder.conv_in.bias"));
        decMid = mid("decoder.mid_block.");
        dec = new Stage[blocks];
        for (int i = 0; i < blocks; i++) {
            String p = "decoder.up_blocks." + i + ".";
            Res[] rs = new Res[layers + 1];
            for (int j = 0; j <= layers; j++) {
                rs[j] = res(p + "resnets." + j + ".");
            }
            Conv up = ld.has(p + "upsamplers.0.conv.weight")
                    ? new Conv(ld.conv(p + "upsamplers.0.conv.weight"), ld.vec(p + "upsamplers.0.conv.bias")) : null;
            dec[i] = new Stage(rs, up);
        }
        decNormG = ld.vec("decoder.conv_norm_out.weight");
        decNormB = ld.vec("decoder.conv_norm_out.bias");
        decOut = new Conv(ld.conv("decoder.conv_out.weight"), ld.vec("decoder.conv_out.bias"));
    }

    private Res res(String p) {
        return new Res(ld.vec(p + "norm1.weight"), ld.vec(p + "norm1.bias"), ld.conv(p + "conv1.weight"), ld.vec(p + "conv1.bias"),
                ld.vec(p + "norm2.weight"), ld.vec(p + "norm2.bias"), ld.conv(p + "conv2.weight"), ld.vec(p + "conv2.bias"),
                ld.has(p + "conv_shortcut.weight") ? ld.conv(p + "conv_shortcut.weight") : null,
                ld.has(p + "conv_shortcut.bias") ? ld.vec(p + "conv_shortcut.bias") : null);
    }

    private Mid mid(String p) {
        String a = p + "attentions.0.";
        // old checkpoints name the attention projections query/key/value/proj_attn and the norm "group_norm"
        boolean legacy = !ld.has(a + "to_q.weight");
        String[] n = legacy ? new String[]{"query", "key", "value", "proj_attn"} : new String[]{"to_q", "to_k", "to_v", "to_out.0"};
        return new Mid(res(p + "resnets.0."), ld.vec(a + "group_norm.weight"), ld.vec(a + "group_norm.bias"),
                ld.lin(a + n[0] + ".weight"), ld.vec(a + n[0] + ".bias"), ld.lin(a + n[1] + ".weight"), ld.vec(a + n[1] + ".bias"),
                ld.lin(a + n[2] + ".weight"), ld.vec(a + n[2] + ".bias"), ld.lin(a + n[3] + ".weight"), ld.vec(a + n[3] + ".bias"),
                res(p + "resnets.1."));
    }

    // ------------------------------------------------------------------ blocks

    private Tn resnet(Res r, Tn x) {
        Tn h = ops.groupNorm(x, r.n1g, r.n1b, groups, eps, true);
        Tn c1 = ops.conv2d(h, r.c1w, r.c1b, 1, 1, 1, 1, 1, null, null);
        ops.free(h);
        Tn h2 = ops.groupNorm(c1, r.n2g, r.n2b, groups, eps, true);
        ops.free(c1);
        Tn shortcut = x;
        if (r.sw != null) {
            shortcut = ops.conv2d(x, r.sw, r.sb, 1, 0, 0, 0, 0, null, null);
        }
        Tn out = ops.conv2d(h2, r.c2w, r.c2b, 1, 1, 1, 1, 1, null, shortcut);
        ops.free(h2);
        if (shortcut != x) {
            ops.free(shortcut);
        }
        ops.free(x);
        return out;
    }

    private Tn middle(Mid m, Tn x) {
        x = resnet(m.r0, x);
        Tn n = ops.groupNorm(x, m.ng, m.nb, groups, eps, false);
        Tn q = ops.linear(n, m.q, m.qb, null, Ops.Act.NONE);
        Tn k = ops.linear(n, m.k, m.kb, null, Ops.Act.NONE);
        Tn v = ops.linear(n, m.v, m.vb, null, Ops.Act.NONE);
        ops.free(n);
        Tn a = ops.attention(q, k, v, 1, false);
        ops.free(q);
        ops.free(k);
        ops.free(v);
        Tn o = ops.linear(a, m.o, m.ob, x, Ops.Act.NONE);
        ops.free(a);
        ops.free(x);
        return resnet(m.r1, o);
    }

    // ------------------------------------------------------------------ decode

    /**
     * Decodes latents (already divided by the scaling factor) to pixels.
     *
     * @param z {@code [B, h, w, C]} host floats
     * @return {@code [B, h·f, w·f, 3]} host floats, roughly in -1..1
     */
    public float[] decode(float[] z, int b, int h, int w) {
        if (h <= tileLatents && w <= tileLatents) {
            return decodeTile(z, b, h, w);
        }
        return decodeTiled(z, b, h, w, this::decodeTile);
    }

    /** Decodes one tile of latents {@code [b, th, tw, C]} to pixels {@code [b, th·f, tw·f, 3]}. */
    interface TileDecoder {
        float[] decode(float[] tile, int b, int th, int tw);
    }

    /** The tiling plan and the blending of {@link #decode}, for any tile decoder (tests use a decoder that is a pure function). */
    float[] decodeTiled(float[] z, int b, int h, int w, TileDecoder tileDecoder) {
        int c = latentChannels;
        int hs = h * scale;
        int ws = w * scale;
        int step = tileLatents - tileOverlap;
        int blend = tileOverlap * scale;
        float[] out = new float[b * hs * ws * 3];
        float[] wsum = new float[hs * ws];
        for (int y0 = 0; y0 < h; y0 += step) {
            int th = Math.min(tileLatents, h - y0);
            for (int x0 = 0; x0 < w; x0 += step) {
                int tw = Math.min(tileLatents, w - x0);
                float[] tile = new float[b * th * tw * c];
                for (int bi = 0; bi < b; bi++) {
                    for (int y = 0; y < th; y++) {
                        System.arraycopy(z, ((bi * h + y0 + y) * w + x0) * c, tile, ((bi * th + y) * tw) * c, tw * c);
                    }
                }
                float[] img = tileDecoder.decode(tile, b, th, tw);
                blendInto(out, wsum, img, b, y0 * scale, x0 * scale, th * scale, tw * scale, hs, ws, blend,
                        y0 > 0, x0 > 0, y0 + th < h, x0 + tw < w, 3);
                if (x0 + tw >= w) {
                    break;
                }
            }
            if (y0 + th >= h) {
                break;
            }
        }
        for (int i = 0; i < hs * ws; i++) {
            float inv = 1f / wsum[i];
            for (int bi = 0; bi < b; bi++) {
                for (int k = 0; k < 3; k++) {
                    out[(bi * hs * ws + i) * 3 + k] *= inv;
                }
            }
        }
        return out;
    }

    /** Adds a tile into the canvas with a linear ramp of {@code blend} pixels at the edges that touch a neighbour. */
    private static void blendInto(float[] out, float[] wsum, float[] tile, int b, int y0, int x0, int th, int tw, int hs, int ws,
                                  int blend, boolean top, boolean left, boolean bottom, boolean right, int ch) {
        for (int y = 0; y < th; y++) {
            float wy = 1f;
            if (top && y < blend) {
                wy = (y + 0.5f) / blend;
            }
            if (bottom && th - 1 - y < blend) {
                wy = Math.min(wy, (th - 1 - y + 0.5f) / blend);
            }
            for (int x = 0; x < tw; x++) {
                float wx = 1f;
                if (left && x < blend) {
                    wx = (x + 0.5f) / blend;
                }
                if (right && tw - 1 - x < blend) {
                    wx = Math.min(wx, (tw - 1 - x + 0.5f) / blend);
                }
                float wt = wy * wx;
                int pix = (y0 + y) * ws + x0 + x;
                wsum[pix] += wt;
                for (int bi = 0; bi < b; bi++) {
                    for (int k = 0; k < ch; k++) {
                        out[(bi * hs * ws + pix) * ch + k] += wt * tile[((bi * th + y) * tw + x) * ch + k];
                    }
                }
            }
        }
    }

    private float[] decodeTile(float[] z, int b, int h, int w) {
        Tn x = ops.upload(z, b, h, w, latentChannels);
        if (postQuant != null) {
            Tn n = ops.conv2d(x, postQuant.w, postQuant.b, 1, 0, 0, 0, 0, null, null);
            ops.free(x);
            x = n;
        }
        Tn t = ops.conv2d(x, decIn.w, decIn.b, 1, 1, 1, 1, 1, null, null);
        ops.free(x);
        x = middle(decMid, t);
        for (Stage s : dec) {
            for (Res r : s.res) {
                x = resnet(r, x);
            }
            if (s.sample != null) {
                Tn big = ops.upsample2x(x);
                ops.free(x);
                x = ops.conv2d(big, s.sample.w, s.sample.b, 1, 1, 1, 1, 1, null, null);
                ops.free(big);
            }
        }
        Tn nrm = ops.groupNorm(x, decNormG, decNormB, groups, eps, true);
        ops.free(x);
        Tn out = ops.conv2d(nrm, decOut.w, decOut.b, 1, 1, 1, 1, 1, null, null);
        ops.free(nrm);
        float[] img = ops.download(out);
        ops.free(out);
        return img;
    }

    // ------------------------------------------------------------------ encode

    /** The posterior of an image: per-latent-pixel mean and standard deviation (both {@code [B, h, w, C]}, unscaled). */
    public record Posterior(float[] mean, float[] std, int h, int w) {
    }

    /**
     * @param img {@code [B, H, W, 3]} host floats in -1..1; H and W multiples of the scale factor
     */
    public Posterior encode(float[] img, int b, int hp, int wp) {
        int tilePx = tileLatents * scale;
        if (hp <= tilePx && wp <= tilePx) {
            return encodeTile(img, b, hp, wp);
        }
        int c = latentChannels;
        int h = hp / scale;
        int w = wp / scale;
        int stepPx = (tileLatents - tileOverlap) * scale;
        int blendLat = tileOverlap;
        float[] mean = new float[b * h * w * c];
        float[] std = new float[b * h * w * c];
        float[] wsum = new float[h * w];
        for (int y0 = 0; y0 < hp; y0 += stepPx) {
            int th = Math.min(tilePx, hp - y0);
            for (int x0 = 0; x0 < wp; x0 += stepPx) {
                int tw = Math.min(tilePx, wp - x0);
                float[] tile = new float[b * th * tw * 3];
                for (int bi = 0; bi < b; bi++) {
                    for (int y = 0; y < th; y++) {
                        System.arraycopy(img, ((bi * hp + y0 + y) * wp + x0) * 3, tile, ((bi * th + y) * tw) * 3, tw * 3);
                    }
                }
                Posterior p = encodeTile(tile, b, th, tw);
                blendInto(mean, wsum, p.mean, b, y0 / scale, x0 / scale, p.h, p.w, h, w, blendLat,
                        y0 > 0, x0 > 0, y0 + th < hp, x0 + tw < wp, c);
                float[] ws2 = new float[h * w]; // the std uses the same weights; blend into a scratch weight map
                blendInto(std, ws2, p.std, b, y0 / scale, x0 / scale, p.h, p.w, h, w, blendLat,
                        y0 > 0, x0 > 0, y0 + th < hp, x0 + tw < wp, c);
                if (x0 + tw >= wp) {
                    break;
                }
            }
            if (y0 + th >= hp) {
                break;
            }
        }
        for (int i = 0; i < h * w; i++) {
            float inv = 1f / wsum[i];
            for (int bi = 0; bi < b; bi++) {
                for (int k = 0; k < c; k++) {
                    mean[(bi * h * w + i) * c + k] *= inv;
                    std[(bi * h * w + i) * c + k] *= inv;
                }
            }
        }
        return new Posterior(mean, std, h, w);
    }

    private Posterior encodeTile(float[] img, int b, int hp, int wp) {
        Tn x = ops.upload(img, b, hp, wp, 3);
        Tn t = ops.conv2d(x, encIn.w, encIn.b, 1, 1, 1, 1, 1, null, null);
        ops.free(x);
        x = t;
        for (Stage s : enc) {
            for (Res r : s.res) {
                x = resnet(r, x);
            }
            if (s.sample != null) {
                Tn n = ops.conv2d(x, s.sample.w, s.sample.b, 2, 0, 0, 1, 1, null, null); // asymmetric (0,1,0,1) padding
                ops.free(x);
                x = n;
            }
        }
        x = middle(encMid, x);
        Tn nrm = ops.groupNorm(x, encNormG, encNormB, groups, eps, true);
        ops.free(x);
        Tn moments = ops.conv2d(nrm, encOut.w, encOut.b, 1, 1, 1, 1, 1, null, null);
        ops.free(nrm);
        if (quant != null) {
            Tn q = ops.conv2d(moments, quant.w, quant.b, 1, 0, 0, 0, 0, null, null);
            ops.free(moments);
            moments = q;
        }
        float[] m = ops.download(moments);
        ops.free(moments);
        int h = hp / scale;
        int w = wp / scale;
        int c = latentChannels;
        float[] mean = new float[b * h * w * c];
        float[] std = new float[b * h * w * c];
        for (int i = 0; i < b * h * w; i++) {
            for (int k = 0; k < c; k++) {
                mean[i * c + k] = m[i * 2 * c + k];
                float logvar = Math.max(-30f, Math.min(20f, m[i * 2 * c + c + k]));
                std[i * c + k] = (float) Math.exp(0.5 * logvar);
            }
        }
        return new Posterior(mean, std, h, w);
    }

    /** Bytes of the checkpoint this network was read from. */
    public long weightBytes() {
        return ckpt.bytes();
    }

    @Override
    public void close() {
        ld.free();
        ckpt.close();
    }

    public List<String> info() {
        return List.of("blocks=" + blocks, "scale=" + scale, "latent=" + latentChannels);
    }
}
