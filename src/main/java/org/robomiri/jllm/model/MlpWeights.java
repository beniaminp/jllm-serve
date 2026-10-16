package org.robomiri.jllm.model;

/** The feed-forward tensors of one decoder layer. */
public sealed interface MlpWeights permits MlpWeights.Dense, MlpWeights.Moe {

    int maxCols();

    /** True if every tensor has the shape {@code spec} implies for a model of width {@code dim}. */
    boolean matches(Mlp spec, int dim);

    /** The tensors the spec asks for, under {@code p} ({@code model.layers.N.}). */
    static MlpWeights load(QuantFile f, String p, Mlp spec) {
        return switch (spec) {
            case Mlp.Dense d -> new Dense(f.tensor(p + "mlp.gate_up"), f.tensor(p + "mlp.down_proj.weight"));
            case Mlp.Moe m -> Moe.load(f, p, m);
        };
    }

    /**
     * A dense gated MLP: {@code gateUp} rows {@code [0, hidden)} are gate_proj, rows {@code [hidden, 2*hidden)}
     * are up_proj; {@code down [dim, hidden]}.
     */
    record Dense(QTensor gateUp, QTensor down) implements MlpWeights {

        @Override
        public boolean matches(Mlp spec, int dim) {
            return spec instanceof Mlp.Dense d && matches(d, dim);
        }

        boolean matches(Mlp.Dense d, int dim) {
            return gateUp.rows == 2 * d.hidden() && gateUp.cols == dim && down.rows == dim && down.cols == d.hidden();
        }

        @Override
        public int maxCols() {
            return Math.max(gateUp.cols, down.cols);
        }
    }

    /**
     * A mixture of experts, all experts stacked: expert {@code e}'s gate/up is rows {@code [e*2*moeDim,
     * (e+1)*2*moeDim)} of {@code expGateUp} (gate first, then up), its down rows {@code [e*dim, (e+1)*dim)} of
     * {@code expDown} (cols = moeDim).
     *
     * @param router     {@code [experts × dim]}, f32
     * @param routerBias DeepSeek-V3 {@code e_score_correction_bias [experts]}, null for the other routers
     * @param shared     DeepSeek's shared experts as one dense SwiGLU, null when the layer has none
     */
    record Moe(float[] router, float[] routerBias, QTensor expGateUp, QTensor expDown, Dense shared) implements MlpWeights {

        static Moe load(QuantFile f, String p, Mlp.Moe m) {
            boolean bias = m.router() instanceof Router.DeepSeek r && r.hasBias();
            Dense shared = m.shared() == null ? null
                    : new Dense(f.tensor(p + "mlp.shared.gate_up"), f.tensor(p + "mlp.shared.down"));
            return new Moe(f.floats(p + "mlp.router"), bias ? f.floats(p + "mlp.router_bias") : null,
                    f.tensor(p + "mlp.experts.gate_up"), f.tensor(p + "mlp.experts.down"), shared);
        }

        @Override
        public boolean matches(Mlp spec, int dim) {
            if (!(spec instanceof Mlp.Moe m)) {
                return false;
            }
            boolean bias = m.router() instanceof Router.DeepSeek r && r.hasBias();
            return router.length == m.experts() * dim
                    && expGateUp.rows == m.experts() * 2 * m.moeDim() && expGateUp.cols == dim
                    && expDown.rows == m.experts() * dim && expDown.cols == m.moeDim()
                    && (m.shared() == null ? shared == null : shared != null && shared.matches(m.shared(), dim))
                    && (bias ? routerBias != null && routerBias.length == m.experts() : routerBias == null);
        }

        @Override
        public int maxCols() {
            int m = Math.max(expGateUp.cols, expDown.cols);
            return shared == null ? m : Math.max(m, shared.maxCols());
        }
    }
}
