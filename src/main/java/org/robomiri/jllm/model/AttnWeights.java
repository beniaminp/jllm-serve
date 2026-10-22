package org.robomiri.jllm.model;

/** The attention projections of one decoder layer ({@code o_proj} is the layer's, see {@link DecoderLayer}). */
public sealed interface AttnWeights permits AttnWeights.Gqa, AttnWeights.Mla {

    int maxCols();

    /** True if every tensor has the shape {@code spec} implies for a model of width {@code dim}. */
    boolean matches(Attention spec, int dim);

    /** The tensors the spec asks for, under {@code a} ({@code model.layers.N.self_attn.}). */
    static AttnWeights load(TensorSource f, String a, Attention spec) {
        return switch (spec) {
            case Attention.Gqa g -> Gqa.load(f, a, g);
            case Attention.Mla m -> Mla.load(f, a, m);
        };
    }

    /**
     * Grouped-query attention: {@code q_proj [qDim, dim]}, {@code k_proj} / {@code v_proj [kvDim, dim]}, the
     * q/k/v biases when the spec has them (all three or none), the per-head {@code q_norm} / {@code k_norm}
     * gains {@code [headDim]} when it has QK-norm (both or none), and the per-head {@code sinks [heads]} logits when it
     * has attention sinks.
     */
    record Gqa(QTensor wq, QTensor wk, QTensor wv, float[] bq, float[] bk, float[] bv, float[] qNorm, float[] kNorm,
               float[] sinks) implements AttnWeights {

        /** A layer without attention sinks. */
        public Gqa(QTensor wq, QTensor wk, QTensor wv, float[] bq, float[] bk, float[] bv, float[] qNorm, float[] kNorm) {
            this(wq, wk, wv, bq, bk, bv, qNorm, kNorm, null);
        }

        static Gqa load(TensorSource f, String a, Attention.Gqa g) {
            return new Gqa(f.tensor(a + "q_proj.weight"), f.tensor(a + "k_proj.weight"), f.tensor(a + "v_proj.weight"),
                    g.bias() ? f.floats(a + "q_proj.bias") : null,
                    g.bias() ? f.floats(a + "k_proj.bias") : null,
                    g.bias() ? f.floats(a + "v_proj.bias") : null,
                    g.qkNorm() ? f.floats(a + "q_norm.weight") : g.l2Norm() ? ones(g.headDim()) : null,
                    g.qkNorm() ? f.floats(a + "k_norm.weight") : g.l2Norm() ? ones(g.headDim()) : null,
                    g.sinks() ? f.floats(a + "sinks") : null);
        }

        /** Llama 4's L2 norm of q and k is an RMS norm with all gains 1 (a rotation does not change the norm, so it runs before the RoPE). */
        private static float[] ones(int n) {
            float[] v = new float[n];
            java.util.Arrays.fill(v, 1f);
            return v;
        }

        @Override
        public boolean matches(Attention spec, int dim) {
            if (!(spec instanceof Attention.Gqa g)) {
                return false;
            }
            return wq.rows == g.qDim() && wq.cols == dim
                    && wk.rows == g.kvDim() && wk.cols == dim
                    && wv.rows == g.kvDim() && wv.cols == dim
                    && g.bias() == (bq != null) && g.bias() == (bk != null) && g.bias() == (bv != null)
                    && (bq == null || (bq.length == g.qDim() && bk.length == g.kvDim() && bv.length == g.kvDim()))
                    && (g.qkNorm() || g.l2Norm()) == (qNorm != null) && (g.qkNorm() || g.l2Norm()) == (kNorm != null)
                    && (qNorm == null || (qNorm.length == g.headDim() && kNorm.length == g.headDim()))
                    && g.sinks() == (sinks != null) && (sinks == null || sinks.length == g.heads());
        }

        @Override
        public int maxCols() {
            return Math.max(Math.max(wq.cols, wk.cols), wv.cols);
        }
    }

    /**
     * DeepSeek Multi-head Latent Attention. {@code kv_a_proj_with_mqa [kvLora + rope, dim]} is split by the
     * converter into {@link #kvA} (the first {@code kvLora} rows) and {@link #kPe} (the last {@code rope} rows),
     * so that both outputs are contiguous per token and the latent can be normalized in place.
     *
     * @param query  {@code q_proj}, or the low-rank {@code q_a_proj} / {@code q_a_layernorm} / {@code q_b_proj}
     * @param kvA    latent rows of {@code kv_a_proj_with_mqa}, {@code [kvLora, dim]}
     * @param kPe    rotary key rows of {@code kv_a_proj_with_mqa}, {@code [rope, dim]}; shared by all heads
     * @param kvB    {@code kv_b_proj [heads*(nope+v), kvLora]}: per head, {@code nope} key rows then {@code v}
     *               value rows. The backends never multiply by it directly: its two halves are absorbed into
     *               the query and applied after the softmax ({@link #kvBKeys}, {@link #kvBValues}), which is
     *               what lets the cache hold the latent
     * @param kB     null for a checkpoint; a GGUF written since llama.cpp's MLA absorption stores {@code kv_b_proj} already
     *               split (and the key half transposed) as {@code attn_k_b}: {@code [heads * kvLora, nope]}, row
     *               {@code h * kvLora + j} holding column {@code j} of head {@code h}'s key matrix. Then {@code kvB} is null
     * @param vB     the value half, {@code attn_v_b}: {@code [heads * v, kvLora]}, row {@code h * v + r} = row {@code r} of head {@code h}
     */
    record Mla(Query query, QTensor kvA, QTensor kPe, float[] kvALayernorm, QTensor kvB, QTensor kB, QTensor vB)
            implements AttnWeights {

        /** The query projection: one matrix, or the low-rank pair with a norm in between ({@code q_lora_rank > 0}). */
        public sealed interface Query permits Proj, Lora {
            int maxCols();
        }

        /** {@code q_proj [heads*qk, dim]}. */
        public record Proj(QTensor w) implements Query {
            @Override
            public int maxCols() {
                return w.cols;
            }
        }

        /** {@code q_a_proj [qLora, dim]}, {@code q_a_layernorm [qLora]}, {@code q_b_proj [heads*qk, qLora]}. */
        public record Lora(QTensor a, float[] norm, QTensor b) implements Query {
            @Override
            public int maxCols() {
                return Math.max(a.cols, b.cols);
            }
        }

        static Mla load(TensorSource f, String a, Attention.Mla m) {
            Query q = m.qLoraRank() > 0
                    ? new Lora(f.tensor(a + "q_a_proj.weight"), f.floats(a + "q_a_layernorm.weight"), f.tensor(a + "q_b_proj.weight"))
                    : new Proj(f.tensor(a + "q_proj.weight"));
            boolean split = !f.has(a + "kv_b_proj.weight") && f.has(a + "k_b_proj.weight");
            return new Mla(q, f.tensor(a + "kv_a_latent.weight"), f.tensor(a + "k_pe_proj.weight"),
                    f.floats(a + "kv_a_layernorm.weight"), split ? null : f.tensor(a + "kv_b_proj.weight"),
                    split ? f.tensor(a + "k_b_proj.weight") : null, split ? f.tensor(a + "v_b_proj.weight") : null);
        }

        @Override
        public boolean matches(Attention spec, int dim) {
            if (!(spec instanceof Attention.Mla m)) {
                return false;
            }
            int h = m.heads();
            boolean q = switch (query) {
                case Proj p -> m.qLoraRank() == 0 && p.w().rows == h * m.qkDim() && p.w().cols == dim;
                case Lora l -> m.qLoraRank() > 0 && l.a().rows == m.qLoraRank() && l.a().cols == dim
                        && l.norm().length == m.qLoraRank() && l.b().rows == h * m.qkDim() && l.b().cols == m.qLoraRank();
            };
            return q && kvA.rows == m.kvLoraRank() && kvA.cols == dim
                    && kPe.rows == m.rope() && kPe.cols == dim
                    && kvALayernorm.length == m.kvLoraRank()
                    && (kvB != null ? kvB.rows == h * (m.nope() + m.vDim()) && kvB.cols == m.kvLoraRank()
                    : kB != null && vB != null && kB.rows == h * m.kvLoraRank() && kB.cols == m.nope()
                    && vB.rows == h * m.vDim() && vB.cols == m.kvLoraRank());
        }

        @Override
        public int maxCols() {
            int b = kvB != null ? kvB.cols : Math.max(kB.cols, vB.cols);
            return Math.max(Math.max(query.maxCols(), kvA.cols), Math.max(kPe.cols, b));
        }

        /**
         * The key half of {@link #kvB} dequantized to f32, {@code W_kb_k[h] [nope, kvLora]} for every head: laid
         * out {@code [heads][nope][kvLora]} as stored, or {@code [heads][kvLora][nope]} when {@code transposed}.
         * Absorbed into the query: {@code q_abs[h] = W_kb_k[h]^T q_nope[h]} is {@code kvLora} wide and is dotted
         * with the cached latent directly. Small ({@code heads * nope * kvLora} floats per layer), computed once
         * per backend at load.
         */
        public float[] kvBKeys(Attention.Mla m, boolean transposed) {
            return half(m, 0, m.nope(), transposed);
        }

        /**
         * The value half of {@link #kvB}, {@code W_kb_v[h] [v, kvLora]} per head: {@code [heads][v][kvLora]} as
         * stored or {@code [heads][kvLora][v]} when {@code transposed}. Applied after the softmax to the
         * attention-weighted latent: {@code out[h] = W_kb_v[h] (sum_s w_s c_kv[s])}.
         */
        public float[] kvBValues(Attention.Mla m, boolean transposed) {
            return half(m, m.nope(), m.vDim(), transposed);
        }

        /**
         * The key ({@code keys}) or value half from the split tensors of a GGUF. {@code kB} is {@code [heads][kvl][nope]}, which
         * is the {@code transposed} key layout; {@code vB} is {@code [heads][v][kvl]}, the plain value layout; the other
         * layout of each is a transposition of every head's block.
         */
        private float[] splitHalf(Attention.Mla m, boolean keys, boolean transposed) {
            int kvl = m.kvLoraRank();
            int heads = m.heads();
            QTensor t = keys ? kB : vB;
            int width = keys ? m.nope() : m.vDim();    // columns of t
            int rowsPerHead = keys ? kvl : width;       // rows of t per head
            boolean asStored = keys == transposed;      // k: transposed layout is the stored one; v: the plain one
            float[] out = new float[heads * width * kvl];
            float[] row = new float[t.cols];
            for (int h = 0; h < heads; h++) {
                int base = h * width * kvl;
                for (int r = 0; r < rowsPerHead; r++) {
                    t.dequantizeRow(h * rowsPerHead + r, row, 0);
                    if (asStored) {
                        System.arraycopy(row, 0, out, base + r * t.cols, t.cols);
                    } else {
                        for (int j = 0; j < t.cols; j++) {
                            out[base + j * rowsPerHead + r] = row[j];
                        }
                    }
                }
            }
            return out;
        }

        /** Rows {@code [first, first + count)} of every head's {@code nope + v} block of kvB, optionally transposed per head. */
        private float[] half(Attention.Mla m, int first, int count, boolean transposed) {
            if (kvB == null) {
                return splitHalf(m, first == 0, transposed);
            }
            int kvl = m.kvLoraRank();
            int block = m.nope() + m.vDim();
            int heads = m.heads();
            float[] out = new float[heads * count * kvl];
            float[] row = new float[kvl];
            for (int h = 0; h < heads; h++) {
                int base = h * count * kvl;
                for (int r = 0; r < count; r++) {
                    kvB.dequantizeRow(h * block + first + r, row, 0);
                    if (transposed) {
                        for (int j = 0; j < kvl; j++) {
                            out[base + j * count + r] = row[j];
                        }
                    } else {
                        System.arraycopy(row, 0, out, base + r * kvl, kvl);
                    }
                }
            }
            return out;
        }
    }
}
