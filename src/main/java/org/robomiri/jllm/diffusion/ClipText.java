package org.robomiri.jllm.diffusion;

import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.nio.file.Path;

/**
 * The CLIP text transformer of Stable Diffusion ({@code CLIPTextModel}, {@code CLIPTextModelWithProjection}): token +
 * position embeddings, pre-norm transformer layers with <em>causal</em> self-attention, a final layer norm. Weights are
 * read from the checkpoint with the same names as Hugging Face transformers; the layer math runs on {@link Ops}.
 *
 * <p>Which output a pipeline wants differs: SD 1.x / 2.x use the final-normed last layer, SDXL the (un-normed) input of
 * the last layer ({@code hidden_states[-2]}) of both encoders plus the projected end-of-text state of the second one.
 */
public final class ClipText implements AutoCloseable {

    /** @param hidden {@code [B, L, hidden]} on the ops' device (the caller frees it); @param pooled projected eos state {@code [B, projection]}, or null */
    public record Output(Tn hidden, float[] pooled) {
    }

    private record Layer(Tn ln1g, Tn ln1b, Tn wq, Tn bq, Tn wk, Tn bk, Tn wv, Tn bv, Tn wo, Tn bo,
                         Tn ln2g, Tn ln2b, Tn w1, Tn b1, Tn w2, Tn b2) {
    }

    private final Ops ops;
    private final Checkpoint ckpt;
    public final int hidden;
    public final int heads;
    public final int layerCount;
    public final int maxPositions;
    public final int projection; // 0 for CLIPTextModel
    private final float eps;
    private final Ops.Act act;
    private final SafeTensors.Tensor tokenEmb;
    private final float[] posEmb;
    private final Layer[] layers;
    private final Tn finalG;
    private final Tn finalB;
    private final Tn textProj;

    public ClipText(Ops ops, Path dir) throws IOException {
        this.ops = ops;
        this.ckpt = Checkpoint.open(dir, "model");
        hidden = ckpt.getInt("hidden_size", 768);
        heads = ckpt.getInt("num_attention_heads", 12);
        layerCount = ckpt.getInt("num_hidden_layers", 12);
        maxPositions = ckpt.getInt("max_position_embeddings", 77);
        eps = (float) ckpt.getDouble("layer_norm_eps", 1e-5);
        String a = ckpt.getString("hidden_act", "quick_gelu");
        act = a.equals("quick_gelu") ? Ops.Act.QUICK_GELU : Ops.Act.GELU;
        boolean withProj = ckpt.has("text_projection.weight");
        projection = withProj ? ckpt.getInt("projection_dim", hidden) : 0;
        String p = ckpt.has("text_model.embeddings.token_embedding.weight") ? "text_model." : "";
        tokenEmb = ckpt.get(p + "embeddings.token_embedding.weight");
        posEmb = ckpt.get(p + "embeddings.position_embedding.weight").toFloatArray();
        layers = new Layer[layerCount];
        for (int i = 0; i < layerCount; i++) {
            String q = p + "encoder.layers." + i + ".";
            layers[i] = new Layer(vec(q + "layer_norm1.weight"), vec(q + "layer_norm1.bias"),
                    lin(q + "self_attn.q_proj.weight"), vec(q + "self_attn.q_proj.bias"),
                    lin(q + "self_attn.k_proj.weight"), vec(q + "self_attn.k_proj.bias"),
                    lin(q + "self_attn.v_proj.weight"), vec(q + "self_attn.v_proj.bias"),
                    lin(q + "self_attn.out_proj.weight"), vec(q + "self_attn.out_proj.bias"),
                    vec(q + "layer_norm2.weight"), vec(q + "layer_norm2.bias"),
                    lin(q + "mlp.fc1.weight"), vec(q + "mlp.fc1.bias"),
                    lin(q + "mlp.fc2.weight"), vec(q + "mlp.fc2.bias"));
        }
        finalG = vec(p + "final_layer_norm.weight");
        finalB = vec(p + "final_layer_norm.bias");
        textProj = withProj ? lin("text_projection.weight") : null;
    }

    private Tn vec(String name) {
        return ops.weight(ckpt.get(name), Ops.Kind.VECTOR);
    }

    private Tn lin(String name) {
        return ops.weight(ckpt.get(name), Ops.Kind.LINEAR);
    }

    /**
     * @param ids        token ids, {@code [B][L]} with {@code L ≤ maxPositions}
     * @param skipLayers layers left out at the end (0: all layers; 1: SDXL's {@code hidden_states[-2]}; CLIP-skip n)
     * @param finalNorm  whether the final layer norm is applied to the returned hidden states
     * @param pooled     also compute the projected end-of-text state (needs a model with a text projection)
     */
    public Output encode(int[][] ids, int skipLayers, boolean finalNorm, boolean pooled) {
        int b = ids.length;
        int len = ids[0].length;
        float[] x = new float[b * len * hidden];
        float[] row = new float[hidden];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < len; j++) {
                tokenEmb.toFloat((long) ids[i][j] * hidden, row, 0, hidden);
                int o = (i * len + j) * hidden;
                for (int c = 0; c < hidden; c++) {
                    x[o + c] = row[c] + posEmb[j * hidden + c];
                }
            }
        }
        Tn h = ops.upload(x, b, len, hidden);
        Tn kept = null;
        int stop = layerCount - skipLayers;
        for (int i = 0; i < layerCount; i++) {
            if (i == stop && skipLayers > 0) {
                kept = h; // the input of layer `stop`
            }
            Tn next = layer(layers[i], h, b, len);
            if (h != kept) {
                ops.free(h);
            }
            h = next;
        }
        // h: output of the last layer; kept: the output of layer stop-1 when layers are skipped
        Tn selected;
        if (skipLayers == 0) {
            selected = h;
            h = null;
        } else {
            selected = kept;
        }
        float[] pool = null;
        if (pooled) {
            if (textProj == null) {
                throw new IllegalStateException("this text encoder has no text_projection");
            }
            Tn fin = h != null ? ops.layerNorm(h, finalG, finalB, eps) : ops.layerNorm(selected, finalG, finalB, eps);
            float[] fh = ops.download(fin);
            ops.free(fin);
            float[] eos = new float[b * hidden];
            for (int i = 0; i < b; i++) {
                int best = 0;
                for (int j = 1; j < len; j++) {
                    if (ids[i][j] > ids[i][best]) {
                        best = j; // the end-of-text token has the largest id
                    }
                }
                System.arraycopy(fh, (i * len + best) * hidden, eos, i * hidden, hidden);
            }
            Tn e = ops.upload(eos, b, hidden);
            Tn proj = ops.linear(e, textProj, null, null, Ops.Act.NONE);
            pool = ops.download(proj);
            ops.free(e);
            ops.free(proj);
        }
        if (h != null) {
            ops.free(h);
        }
        if (finalNorm) {
            Tn n = ops.layerNorm(selected, finalG, finalB, eps);
            ops.free(selected);
            selected = n;
        }
        return new Output(selected, pool);
    }

    private Tn layer(Layer l, Tn h, int b, int len) {
        Tn n1 = ops.layerNorm(h, l.ln1g, l.ln1b, eps);
        Tn q = ops.linear(n1, l.wq, l.bq, null, Ops.Act.NONE);
        Tn k = ops.linear(n1, l.wk, l.bk, null, Ops.Act.NONE);
        Tn v = ops.linear(n1, l.wv, l.bv, null, Ops.Act.NONE);
        ops.free(n1);
        Tn a = ops.attention(q, k, v, heads, true);
        ops.free(q);
        ops.free(k);
        ops.free(v);
        Tn h2 = ops.linear(a, l.wo, l.bo, h, Ops.Act.NONE);
        ops.free(a);
        Tn n2 = ops.layerNorm(h2, l.ln2g, l.ln2b, eps);
        Tn f = ops.linear(n2, l.w1, l.b1, null, act);
        ops.free(n2);
        Tn h3 = ops.linear(f, l.w2, l.b2, h2, Ops.Act.NONE);
        ops.free(f);
        ops.free(h2);
        return h3;
    }

    @Override
    public void close() {
        ckpt.close();
    }
}
