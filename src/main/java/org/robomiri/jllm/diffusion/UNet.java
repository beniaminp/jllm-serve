package org.robomiri.jllm.diffusion;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/**
 * {@code UNet2DConditionModel} of Stable Diffusion 1.x / 2.x (including SD-Turbo) and SDXL (SDXL-Turbo), read from the
 * diffusers checkpoint. It predicts the noise (or, with other prediction types, the velocity) in a latent image
 * {@code [B, h, w, C]} at one timestep, given the text embeddings:
 *
 * <pre>
 *   emb  = MLP(sinusoidal(t))  (+ MLP([pooled text, sinusoidal(time ids)]) for SDXL)
 *   x    = conv_in(sample)
 *   down: per block, per layer   x = ResNet(x, emb); x = Transformer(x, text)   (if the block has attention); push x
 *         then a strided 3x3 convolution (push)
 *   mid:  ResNet, Transformer, ResNet
 *   up:   per block, per layer   x = ResNet(concat(x, pop), emb); x = Transformer(x, text); then 2x upsample + 3x3 conv
 *   out:  conv_out(silu(group_norm(x)))
 * </pre>
 *
 * The configuration is read from {@code config.json}; the checkpoint's tensor names are diffusers'. Everything runs on
 * the given {@link Ops}.
 */
public final class UNet implements AutoCloseable {

    /** What the network is conditioned on besides the timestep. */
    public record Cond(Tn encoder, float[] pooled, float[] timeIds) {
    }

    private record Res(Tn n1g, Tn n1b, Tn c1w, Tn c1b, Tn tw, Tn tb, Tn n2g, Tn n2b, Tn c2w, Tn c2b, Tn sw, Tn sb) {
    }

    private record Attn(Tn q, Tn k, Tn v, Tn ow, Tn ob, int heads) {
    }

    private record Block(Tn n1g, Tn n1b, Attn a1, Tn n2g, Tn n2b, Attn a2, Tn n3g, Tn n3b, Tn f0w, Tn f0b, Tn f2w, Tn f2b) {
    }

    private record Trans(Tn ng, Tn nb, Tn pinw, Tn pinb, Block[] blocks, Tn poutw, Tn poutb) {
    }

    private record Conv(Tn w, Tn b) {
    }

    private record Down(Res[] res, Trans[] att, Conv down) {
    }

    private record Up(Res[] res, Trans[] att, Conv up) {
    }

    private final Ops ops;
    private final Checkpoint ckpt;
    private final Loader ld;
    public final int inChannels;
    public final int outChannels;
    public final int blockCount;
    public final int crossDim;
    public final boolean textTime;
    public final int sampleSize;
    private final int groups;
    private final float eps;
    private final boolean flipSinToCos;
    private final float freqShift;
    private final int baseChannels;
    private final int addTimeDim;

    private final Tn timeW1, timeB1, timeW2, timeB2;
    private final Tn addW1, addB1, addW2, addB2;
    private final Conv convIn;
    private final Down[] downs;
    private final Res[] midRes = new Res[2];
    private final Trans midAtt;
    private final Up[] ups;
    private final Tn normOutG, normOutB;
    private final Conv convOut;

    public UNet(Ops ops, Path dir) throws IOException {
        this.ops = ops;
        this.ckpt = Checkpoint.open(dir, "diffusion_pytorch_model");
        this.ld = new Loader(ops, ckpt);
        int[] blockOut = ckpt.getIntArray("block_out_channels", 320, 640, 1280, 1280);
        blockCount = blockOut.length;
        baseChannels = blockOut[0];
        inChannels = ckpt.getInt("in_channels", 4);
        outChannels = ckpt.getInt("out_channels", 4);
        crossDim = ckpt.getInt("cross_attention_dim", 768);
        sampleSize = ckpt.getInt("sample_size", 64);
        groups = ckpt.getInt("norm_num_groups", 32);
        eps = (float) ckpt.getDouble("norm_eps", 1e-5);
        flipSinToCos = ckpt.getBool("flip_sin_to_cos", true);
        freqShift = (float) ckpt.getDouble("freq_shift", 0);
        int layersPerBlock = ckpt.getInt("layers_per_block", 2);
        List<String> downTypes = ckpt.getStringList("down_block_types");
        List<String> upTypes = ckpt.getStringList("up_block_types");
        // diffusers' naming quirk: "attention_head_dim" holds the NUMBER of heads of each block
        int[] heads = ckpt.config.get("num_attention_heads") != null
                ? ckpt.getIntList("num_attention_heads", blockCount, 8) : ckpt.getIntList("attention_head_dim", blockCount, 8);
        int[] tLayers = ckpt.getIntList("transformer_layers_per_block", blockCount, 1);
        boolean linearProj = ckpt.getBool("use_linear_projection", false);
        String addType = ckpt.getString("addition_embed_type", null);
        textTime = "text_time".equals(addType);
        addTimeDim = ckpt.getInt("addition_time_embed_dim", 256);
        if (ckpt.config.get("time_cond_proj_dim") != null || ckpt.config.get("class_embed_type") != null
                || ckpt.config.get("num_class_embeds") != null || ckpt.config.get("encoder_hid_dim_type") != null
                || (addType != null && !textTime)) {
            throw new IOException("unsupported UNet conditioning in " + dir + " (class/addition embeddings other than text_time)");
        }
        if (!"positional".equals(ckpt.getString("time_embedding_type", "positional"))) {
            throw new IOException("unsupported time_embedding_type in " + dir);
        }
        if (!"default".equals(ckpt.getString("resnet_time_scale_shift", "default"))) {
            throw new IOException("unsupported resnet_time_scale_shift in " + dir);
        }

        timeW1 = ld.lin("time_embedding.linear_1.weight");
        timeB1 = ld.vec("time_embedding.linear_1.bias");
        timeW2 = ld.lin("time_embedding.linear_2.weight");
        timeB2 = ld.vec("time_embedding.linear_2.bias");
        if (textTime) {
            addW1 = ld.lin("add_embedding.linear_1.weight");
            addB1 = ld.vec("add_embedding.linear_1.bias");
            addW2 = ld.lin("add_embedding.linear_2.weight");
            addB2 = ld.vec("add_embedding.linear_2.bias");
        } else {
            addW1 = addB1 = addW2 = addB2 = null;
        }
        convIn = new Conv(ld.conv("conv_in.weight"), ld.vec("conv_in.bias"));

        downs = new Down[blockCount];
        for (int i = 0; i < blockCount; i++) {
            boolean cross = downTypes.get(i).startsWith("CrossAttn");
            String p = "down_blocks." + i + ".";
            Res[] rs = new Res[layersPerBlock];
            Trans[] at = cross ? new Trans[layersPerBlock] : null;
            for (int j = 0; j < layersPerBlock; j++) {
                rs[j] = res(p + "resnets." + j + ".");
                if (cross) {
                    at[j] = trans(p + "attentions." + j + ".", heads[i], tLayers[i], linearProj);
                }
            }
            Conv dc = ld.has(p + "downsamplers.0.conv.weight")
                    ? new Conv(ld.conv(p + "downsamplers.0.conv.weight"), ld.vec(p + "downsamplers.0.conv.bias")) : null;
            downs[i] = new Down(rs, at, dc);
        }
        midRes[0] = res("mid_block.resnets.0.");
        midAtt = trans("mid_block.attentions.0.", heads[blockCount - 1], tLayers[blockCount - 1], linearProj);
        midRes[1] = res("mid_block.resnets.1.");
        ups = new Up[blockCount];
        for (int i = 0; i < blockCount; i++) {
            boolean cross = upTypes.get(i).startsWith("CrossAttn");
            String p = "up_blocks." + i + ".";
            int lvl = blockCount - 1 - i; // the level (resolution) this up block works at
            Res[] rs = new Res[layersPerBlock + 1];
            Trans[] at = cross ? new Trans[layersPerBlock + 1] : null;
            for (int j = 0; j <= layersPerBlock; j++) {
                rs[j] = res(p + "resnets." + j + ".");
                if (cross) {
                    at[j] = trans(p + "attentions." + j + ".", heads[lvl], tLayers[lvl], linearProj);
                }
            }
            Conv uc = ld.has(p + "upsamplers.0.conv.weight")
                    ? new Conv(ld.conv(p + "upsamplers.0.conv.weight"), ld.vec(p + "upsamplers.0.conv.bias")) : null;
            ups[i] = new Up(rs, at, uc);
        }
        normOutG = ld.vec("conv_norm_out.weight");
        normOutB = ld.vec("conv_norm_out.bias");
        convOut = new Conv(ld.conv("conv_out.weight"), ld.vec("conv_out.bias"));
    }

    private Res res(String p) {
        return new Res(ld.vec(p + "norm1.weight"), ld.vec(p + "norm1.bias"), ld.conv(p + "conv1.weight"), ld.vec(p + "conv1.bias"),
                ld.lin(p + "time_emb_proj.weight"), ld.vec(p + "time_emb_proj.bias"),
                ld.vec(p + "norm2.weight"), ld.vec(p + "norm2.bias"), ld.conv(p + "conv2.weight"), ld.vec(p + "conv2.bias"),
                ld.has(p + "conv_shortcut.weight") ? ld.conv(p + "conv_shortcut.weight") : null,
                ld.has(p + "conv_shortcut.bias") ? ld.vec(p + "conv_shortcut.bias") : null);
    }

    private Attn attn(String p, int heads) {
        return new Attn(ld.lin(p + "to_q.weight"), ld.lin(p + "to_k.weight"), ld.lin(p + "to_v.weight"),
                ld.lin(p + "to_out.0.weight"), ld.vec(p + "to_out.0.bias"), heads);
    }

    private Trans trans(String p, int heads, int depth, boolean linearProj) {
        Block[] bs = new Block[depth];
        for (int i = 0; i < depth; i++) {
            String q = p + "transformer_blocks." + i + ".";
            bs[i] = new Block(ld.vec(q + "norm1.weight"), ld.vec(q + "norm1.bias"), attn(q + "attn1.", heads),
                    ld.vec(q + "norm2.weight"), ld.vec(q + "norm2.bias"), attn(q + "attn2.", heads),
                    ld.vec(q + "norm3.weight"), ld.vec(q + "norm3.bias"),
                    ld.lin(q + "ff.net.0.proj.weight"), ld.vec(q + "ff.net.0.proj.bias"),
                    ld.lin(q + "ff.net.2.weight"), ld.vec(q + "ff.net.2.bias"));
        }
        return new Trans(ld.vec(p + "norm.weight"), ld.vec(p + "norm.bias"), ld.lin(p + "proj_in.weight"),
                ld.vec(p + "proj_in.bias"), bs, ld.lin(p + "proj_out.weight"), ld.vec(p + "proj_out.bias"));
    }

    // ------------------------------------------------------------------ forward

    /**
     * Sinusoidal embedding of a timestep (diffusers {@code get_timestep_embedding}): {@code [cos | sin]} of
     * {@code t · 10000^(-i / (dim/2 - shift))}, or {@code [sin | cos]} without {@code flipSinToCos}.
     */
    static float[] timestepEmbedding(float t, int dim, boolean flipSinToCos, float freqShift) {
        int half = dim / 2;
        float[] out = new float[dim];
        float lg = (float) -Math.log(10000.0);
        for (int i = 0; i < half; i++) {
            float exponent = lg * i / (half - freqShift);
            float freq = (float) Math.exp(exponent);
            float arg = t * freq;
            float s = (float) Math.sin(arg);
            float c = (float) Math.cos(arg);
            if (flipSinToCos) {
                out[i] = c;
                out[half + i] = s;
            } else {
                out[i] = s;
                out[half + i] = c;
            }
        }
        return out;
    }

    private final Set<Tn> held = java.util.Collections.newSetFromMap(new IdentityHashMap<>());

    private void release(Tn t) {
        if (!held.contains(t)) {
            ops.free(t);
        }
    }

    /**
     * One evaluation of the network.
     *
     * @param sample {@code [B, h, w, inChannels]}
     * @return the prediction {@code [B, h, w, outChannels]} (the caller frees it)
     */
    public Tn forward(Tn sample, float timestep, Cond cond) {
        int b = sample.dim(0);
        float[] te = timestepEmbedding(timestep, baseChannels, flipSinToCos, freqShift);
        float[] teb = new float[b * baseChannels];
        for (int i = 0; i < b; i++) {
            System.arraycopy(te, 0, teb, i * baseChannels, baseChannels);
        }
        Tn tin = ops.upload(teb, b, baseChannels);
        Tn l1 = ops.linear(tin, timeW1, timeB1, null, Ops.Act.SILU);
        ops.free(tin);
        Tn emb = ops.linear(l1, timeW2, timeB2, null, Ops.Act.NONE);
        ops.free(l1);
        if (textTime) {
            if (cond.pooled() == null || cond.timeIds() == null) {
                throw new IllegalArgumentException("this UNet needs pooled text embeddings and time ids (SDXL)");
            }
            int pd = cond.pooled().length / b;
            int nIds = cond.timeIds().length / b;
            float[] a = new float[b * (pd + nIds * addTimeDim)];
            for (int i = 0; i < b; i++) {
                int o = i * (pd + nIds * addTimeDim);
                System.arraycopy(cond.pooled(), i * pd, a, o, pd);
                for (int j = 0; j < nIds; j++) {
                    float[] e = timestepEmbedding(cond.timeIds()[i * nIds + j], addTimeDim, flipSinToCos, freqShift);
                    System.arraycopy(e, 0, a, o + pd + j * addTimeDim, addTimeDim);
                }
            }
            Tn ain = ops.upload(a, b, a.length / b);
            Tn a1 = ops.linear(ain, addW1, addB1, null, Ops.Act.SILU);
            ops.free(ain);
            Tn aug = ops.linear(a1, addW2, addB2, null, Ops.Act.NONE);
            ops.free(a1);
            Tn sum = ops.add(emb, aug);
            ops.free(emb);
            ops.free(aug);
            emb = sum;
        }
        Tn embAct = ops.silu(emb);
        ops.free(emb);

        List<Tn> skips = new ArrayList<>();
        Tn x = ops.conv2d(sample, convIn.w, convIn.b, 1, 1, 1, 1, 1, null, null);
        skips.add(x);
        held.add(x);
        for (Down d : downs) {
            for (int j = 0; j < d.res.length; j++) {
                x = resnet(d.res[j], x, embAct);
                if (d.att != null) {
                    x = transformer(d.att[j], x, cond.encoder());
                }
                skips.add(x);
                held.add(x);
            }
            if (d.down != null) {
                Tn n = ops.conv2d(x, d.down.w, d.down.b, 2, 1, 1, 1, 1, null, null);
                release(x);
                x = n;
                skips.add(x);
                held.add(x);
            }
        }
        x = resnet(midRes[0], x, embAct);
        x = transformer(midAtt, x, cond.encoder());
        x = resnet(midRes[1], x, embAct);
        for (Up u : ups) {
            for (int j = 0; j < u.res.length; j++) {
                Tn skip = skips.removeLast();
                held.remove(skip);
                Tn cat = ops.concat(x, skip);
                release(x);
                ops.free(skip);
                x = resnet(u.res[j], cat, embAct);
                if (u.att != null) {
                    x = transformer(u.att[j], x, cond.encoder());
                }
            }
            if (u.up != null) {
                Tn big = ops.upsample2x(x);
                ops.free(x);
                x = ops.conv2d(big, u.up.w, u.up.b, 1, 1, 1, 1, 1, null, null);
                ops.free(big);
            }
        }
        ops.free(embAct);
        Tn nrm = ops.groupNorm(x, normOutG, normOutB, groups, eps, true);
        ops.free(x);
        Tn out = ops.conv2d(nrm, convOut.w, convOut.b, 1, 1, 1, 1, 1, null, null);
        ops.free(nrm);
        held.clear();
        return out;
    }

    /** ResNet block; frees {@code x} unless it is kept as a skip connection. */
    private Tn resnet(Res r, Tn x, Tn embAct) {
        Tn h = ops.groupNorm(x, r.n1g, r.n1b, groups, eps, true);
        Tn t = ops.linear(embAct, r.tw, r.tb, null, Ops.Act.NONE);
        Tn c1 = ops.conv2d(h, r.c1w, r.c1b, 1, 1, 1, 1, 1, t, null);
        ops.free(h);
        ops.free(t);
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
        release(x);
        return out;
    }

    /** Spatial transformer: self-attention, cross-attention to the text, feed-forward, all with residuals. */
    private Tn transformer(Trans t, Tn x, Tn enc) {
        Tn n = ops.groupNorm(x, t.ng, t.nb, groups, 1e-6f, false);
        Tn h = ops.linear(n, t.pinw, t.pinb, null, Ops.Act.NONE);
        ops.free(n);
        for (Block bl : t.blocks) {
            h = block(bl, h, enc);
        }
        Tn out = ops.linear(h, t.poutw, t.poutb, x, Ops.Act.NONE);
        ops.free(h);
        release(x);
        return out;
    }

    private Tn block(Block b, Tn h, Tn enc) {
        Tn n1 = ops.layerNorm(h, b.n1g, b.n1b, 1e-5f);
        Tn a1 = attention(b.a1, n1, n1, h);
        ops.free(n1);
        ops.free(h);
        Tn n2 = ops.layerNorm(a1, b.n2g, b.n2b, 1e-5f);
        Tn a2 = attention(b.a2, n2, enc, a1);
        ops.free(n2);
        ops.free(a1);
        Tn n3 = ops.layerNorm(a2, b.n3g, b.n3b, 1e-5f);
        Tn g = ops.linear(n3, b.f0w, b.f0b, null, Ops.Act.NONE);
        ops.free(n3);
        Tn gg = ops.geglu(g);
        ops.free(g);
        Tn out = ops.linear(gg, b.f2w, b.f2b, a2, Ops.Act.NONE);
        ops.free(gg);
        ops.free(a2);
        return out;
    }

    /** {@code residual + out_proj(attention(q(x), k(ctx), v(ctx)))}. */
    private Tn attention(Attn a, Tn x, Tn ctx, Tn residual) {
        Tn q = ops.linear(x, a.q, null, null, Ops.Act.NONE);
        Tn k = ops.linear(ctx, a.k, null, null, Ops.Act.NONE);
        Tn v = ops.linear(ctx, a.v, null, null, Ops.Act.NONE);
        Tn o = ops.attention(q, k, v, a.heads, false);
        ops.free(q);
        ops.free(k);
        ops.free(v);
        Tn out = ops.linear(o, a.ow, a.ob, residual, Ops.Act.NONE);
        ops.free(o);
        return out;
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
}
