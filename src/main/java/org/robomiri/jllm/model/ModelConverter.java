package org.robomiri.jllm.model;

import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * Converts a Hugging Face checkpoint into a {@link QuantFile}, one unit (embedding, layer, final norm + head)
 * at a time.
 *
 * <p>Every cache tensor is described by one {@link Rule}: the checkpoint tensors it is computed from and the
 * transform that computes it (quantize, split rows, stack, flatten a 3-D tensor, keep as f32, fold the Gemma
 * {@code 1 + w}, {@code -exp}). A unit is an ordered list of rules; the checkpoint tensors it needs are derived
 * from those rules, so the shard bookkeeping below can never disagree with what the conversion reads.
 *
 * <p>Shards are opened only when a tensor in them is needed and closed (and {@link ShardProvider#release
 * released}) as soon as every tensor the converter will ever read from them has been consumed, so at most two
 * shards are open at any time. Tensors the converter never reads (e.g. {@code lm_head.weight} of a model with
 * tied embeddings, or {@code rotary_emb.inv_freq} buffers) do not keep a shard alive.
 * MoE experts are stacked into two tensors per layer (gate/up for all experts, down for all experts).
 */
public final class ModelConverter {

    private static final String EMBED = "model.embed_tokens.weight";
    private static final String FINAL_NORM = "model.norm.weight";
    private static final String LM_HEAD = "lm_head.weight";

    /** Computes one cache tensor, a {@link QTensor} allocated in the arena or a {@code float[]}, from its sources. */
    @FunctionalInterface
    private interface Transform {
        Object apply(Arena a, List<SafeTensors.Tensor> sources) throws IOException;
    }

    /**
     * One cache tensor: {@code sources} are the checkpoint tensors it is computed from, in the order the
     * transform expects them; {@code transform} only ever sees those.
     */
    private record Rule(String cacheName, List<String> sources, Transform transform) {
    }

    /** One resumable step of the conversion: its rules, in cache order. */
    private record Unit(String id, List<Rule> rules) {
        /** Every checkpoint tensor the unit reads, derived from its rules. */
        List<String> inputs() {
            LinkedHashSet<String> in = new LinkedHashSet<>();
            for (Rule r : rules) {
                in.addAll(r.sources());
            }
            return List.copyOf(in);
        }
    }

    private final Path dir;
    private final Architecture arch;
    private final QType quant;
    /** Format of the embedding and the LM head (Q8_0 unless the head-quant setting says otherwise). */
    private final QType head;
    private final ShardProvider provider;
    private final Map<String, String> shardOf;
    /** Canonical tensor name -> name in the checkpoint, only where they differ (multimodal Gemma prefixes). */
    private final Map<String, String> checkpointName = new HashMap<>();
    private final List<Unit> units;
    /** Per shard: how many planned tensor reads are still outstanding. */
    private final Map<String, Integer> remaining = new HashMap<>();
    private final Map<String, SafeTensors> open = new LinkedHashMap<>();
    /** MLX-quantized checkpoint (packed weight + scales + biases per linear layer), or null. */
    private final MlxQuant mlx;
    /** FP8 block-quantized checkpoint ({@code weight} + {@code weight_scale_inv} per linear layer), or null. */
    private final Fp8Quant fp8;

    private ModelConverter(Path dir, Architecture arch, QType quant, QType head, ShardProvider provider)
            throws IOException {
        this.dir = dir;
        this.arch = arch;
        this.quant = quant;
        this.head = head;
        this.provider = provider;
        this.shardOf = shardMap();
        this.fp8 = shardOf.keySet().stream().anyMatch(n -> n.endsWith(".weight_scale_inv")) ? Fp8Quant.read(dir) : null;
        if (fp8 != null) {
            System.err.printf("FP8 checkpoint (%dx%d blocks): dequantized while converting to %s%n", fp8.blockRows,
                    fp8.blockCols, quant);
        }
        this.mlx = MlxQuant.read(dir);
        if (mlx != null) {
            System.err.printf("MLX checkpoint (%s, %d-bit, groups of %d): dequantized while converting to %s%s%n",
                    mlx.mode.key, mlx.bits, mlx.groupSize, quant, quant == QType.Q4_0 && mlx.bits <= 4
                            ? " (re-quantized once more; --model.quant q8 keeps the MLX values almost exactly)" : "");
        }
        this.units = plan();
        for (Unit u : units) {
            for (String name : u.inputs()) {
                String shard = shardOf.get(name);
                if (shard != null) {
                    remaining.merge(shard, 1, Integer::sum);
                }
            }
        }
    }

    /**
     * Converts the checkpoint in {@code dir} into {@code jllm-<quant>.bin/.json}. A conversion interrupted by a
     * crash is resumed from the units recorded in {@code jllm-<quant>.partial.json} (see {@link QuantFile.Writer}).
     */
    public static void convert(Path dir, Architecture arch, QType quant, ShardProvider provider) throws IOException {
        convert(dir, arch, quant, provider, null);
    }

    /**
     * @param identity {@code repo@revision} the checkpoint was requested as, recorded in the cache together with
     *                 the provider's {@link ShardProvider#commitSha() commit}; null for local files
     */
    public static void convert(Path dir, Architecture arch, QType quant, ShardProvider provider, String identity)
            throws IOException {
        convert(dir, arch, quant, QuantFile.DEFAULT_HEAD, provider, identity);
    }

    /** As above, storing the embedding and LM head as {@code head} ({@code jllm-<quant>-h6k.bin} for Q6_K). */
    public static void convert(Path dir, Architecture arch, QType quant, QType head, ShardProvider provider,
                               String identity) throws IOException {
        long t0 = System.nanoTime();
        ModelConverter c = new ModelConverter(dir, arch, quant, head, provider);
        QuantFile.Identity id = QuantFile.Identity.parse(identity, provider.commitSha());
        try (QuantFile.Writer w = QuantFile.Writer.openOrResume(dir, quant, head, arch.toString(), id, c.unitIds())) {
            c.run(w);
            w.finish();
            System.err.printf("\rConverted %s to %s in %.0fs (%.0f MB) -> %s%n", arch.core().architecture(), quant,
                    (System.nanoTime() - t0) / 1e9, w.bytesWritten() / 1e6, QuantFile.binPath(dir, quant, head));
        } finally {
            c.open.values().forEach(SafeTensors::close);
            c.open.clear();
        }
    }

    private List<String> unitIds() {
        return units.stream().map(Unit::id).toList();
    }

    private void run(QuantFile.Writer w) throws IOException {
        int skip = w.completedUnits().size(); // a prefix of the plan, verified by the writer
        if (skip > 0) {
            System.err.printf("Resuming conversion: %d/%d units already converted (%.0f MB)%n", skip, units.size(),
                    w.bytesWritten() / 1e6);
        }
        for (int i = 0; i < units.size(); i++) {
            Unit u = units.get(i);
            if (i >= skip) {
                for (Rule r : u.rules()) {
                    // One arena per cache tensor, closed as soon as it is written: peak off-heap memory is the
                    // largest tensor (the stacked experts of a MoE layer), not the whole unit.
                    try (Arena a = Arena.ofShared()) {
                        apply(w, a, r);
                    }
                }
                w.commitUnit(u.id());
            }
            // Also for skipped units: their shards are consumed, so a low-disk provider may delete them.
            u.inputs().forEach(this::done);
            System.err.printf("\rConverting to %s: %d/%d units (%.0f MB written)", quant, i + 1, units.size(),
                    w.bytesWritten() / 1e6);
        }
        // Everything has been read: nothing may stay open, whatever the bookkeeping says.
        for (String shard : new ArrayList<>(open.keySet())) {
            release(shard);
        }
    }

    /** Reads the rule's sources, computes the cache tensor and writes it. */
    private void apply(QuantFile.Writer w, Arena a, Rule r) throws IOException {
        List<SafeTensors.Tensor> src = new ArrayList<>(r.sources().size());
        for (String name : r.sources()) {
            src.add(tensor(name));
        }
        Object v = r.transform().apply(a, src);
        if (v instanceof QTensor t) {
            w.put(r.cacheName(), t);
        } else {
            w.putFloats(r.cacheName(), (float[]) v);
        }
    }

    // ------------------------------------------------------------------ rules

    /** Quantizes a 2-D checkpoint tensor to {@code type} under the same name. */
    private static Rule quantize(String name, QType type) {
        return quantize(name, name, type, 0, 0);
    }

    /** Quantizes the 2-D tensor {@code source} to {@code type}; {@code rows} / {@code cols} are checked unless 0. */
    private static Rule quantize(String cacheName, String source, QType type, int rows, int cols) {
        return new Rule(cacheName, List.of(source), (a, s) -> {
            SafeTensors.Tensor t = matrix(s.getFirst());
            if ((rows > 0 && t.dim(0) != rows) || (cols > 0 && t.dim(1) != cols)) {
                throw new IOException(t + " does not match [" + (rows > 0 ? rows : "*") + ", " + (cols > 0 ? cols : "*") + "]");
            }
            return quantize(a, type, t, 0, t.dim(0));
        });
    }

    /**
     * Quantizes rows {@code [start, start + rows)} of {@code source}, which must be {@code [totalRows, cols]}.
     * A row slice quantizes exactly like the whole matrix because quantization blocks run along a row.
     */
    private static Rule splitRows(String cacheName, String source, QType type, int start, int rows, int totalRows, int cols) {
        return new Rule(cacheName, List.of(source), (a, s) -> {
            SafeTensors.Tensor t = matrix(s.getFirst());
            if (t.dim(0) != totalRows || t.dim(1) != cols) {
                throw new IOException(t + " does not match [" + totalRows + ", " + cols + "]");
            }
            return quantize(a, type, t, start, rows);
        });
    }

    /**
     * Quantizes several 2-D {@code [rows_i, cols]} tensors as one {@code [Σ rows_i, cols]} tensor, in order.
     * Every part must have {@code rowsEach} rows and {@code cols} columns where those are not 0.
     */
    private static Rule stack(String cacheName, List<String> sources, QType type, int rowsEach, int cols) {
        return new Rule(cacheName, List.copyOf(sources), (a, s) -> {
            int width = matrix(s.getFirst()).dim(1);
            int[] starts = new int[s.size() + 1];
            for (int i = 0; i < s.size(); i++) {
                SafeTensors.Tensor t = matrix(s.get(i));
                if (t.dim(1) != width || (cols > 0 && t.dim(1) != cols) || (rowsEach > 0 && t.dim(0) != rowsEach)) {
                    throw new IOException("cannot stack " + t + ": expected [" + (rowsEach > 0 ? rowsEach : "*") + ", "
                            + (cols > 0 ? cols : width) + "]");
                }
                starts[i + 1] = starts[i] + t.dim(0);
            }
            SafeTensors.Tensor[] arr = s.toArray(new SafeTensors.Tensor[0]);
            return QTensor.quantize(a, type, starts[s.size()], width, (r, dst) -> {
                int i = Arrays.binarySearch(starts, r);
                if (i < 0) {
                    i = -i - 2;
                }
                arr[i].toFloat((long) (r - starts[i]) * width, dst, 0, width);
            });
        });
    }

    /** Quantizes a 3-D {@code [a, b, c]} tensor of exactly {@code shape} as {@code [a*b, c]} (row-major, no data moves). */
    private static Rule flatten3d(String cacheName, String source, QType type, long[] shape) {
        return new Rule(cacheName, List.of(source), (a, s) -> {
            SafeTensors.Tensor t = s.getFirst();
            if (!Arrays.equals(t.shape(), shape)) {
                throw new IOException(t + " does not match " + Arrays.toString(shape));
            }
            if (t.reader() != null) {
                throw new IOException("stacked expert tensor " + t.name()
                        + " is MLX- or FP8-quantized: not supported, use the bf16 checkpoint");
            }
            SafeTensors.Tensor flat = new SafeTensors.Tensor(t.name(), t.dtype(), new long[]{shape[0] * shape[1], shape[2]}, t.data());
            return quantize(a, type, flat, 0, flat.dim(0));
        });
    }

    /** Stores a tensor of any rank as a flat f32 vector under {@code cacheName}. */
    private static Rule f32(String cacheName, String source) {
        return new Rule(cacheName, List.of(source), (a, s) -> s.getFirst().toFloatArray());
    }

    /** Gemma norm gains: the checkpoint stores {@code w}, the model applies {@code 1 + w} (exact in f32, as in transformers). */
    private static Rule f32PlusOne(String cacheName, String source) {
        return new Rule(cacheName, List.of(source), (a, s) -> GemmaMath.foldNormGain(s.getFirst().toFloatArray()));
    }

    /** Mamba's {@code A = -exp(A_log)}: the scan only ever uses that form. */
    private static Rule negExp(String cacheName, String source) {
        return new Rule(cacheName, List.of(source), (a, s) -> {
            float[] v = s.getFirst().toFloatArray();
            for (int i = 0; i < v.length; i++) {
                v[i] = (float) -Math.exp(v[i]);
            }
            return v;
        });
    }

    /** A norm weight under its own name, with the Gemma {@code 1 + w} folded in where the family needs it. */
    private Rule norm(String name) {
        return arch.core().normPlusOne() ? f32PlusOne(name, name) : f32(name, name);
    }

    private static SafeTensors.Tensor matrix(SafeTensors.Tensor t) throws IOException {
        if (t.shape().length != 2) {
            throw new IOException("expected a 2-D tensor, got " + t);
        }
        return t;
    }

    private static QTensor quantize(Arena a, QType type, SafeTensors.Tensor t, int start, int rows) {
        int cols = t.dim(1);
        return QTensor.quantize(a, type, rows, cols, (r, dst) -> t.toFloat((long) (start + r) * cols, dst, 0, cols));
    }

    // ------------------------------------------------------------------ plan

    /** The ordered conversion units; each lists its cache tensors as rules. */
    private List<Unit> plan() {
        return switch (arch) {
            case MambaArch m -> planMamba(m);
            case Decoder d -> planDecoder(d);
            case Qwen3NextArch q -> planQwen3Next(q);
        };
    }

    /**
     * Embedding, one unit per layer, final norm + head. A layer's rules follow its {@link LayerSpec}: the norms
     * the checkpoint has, the attention variant (grouped-query with optional biases, or MLA) and the MLP variant
     * (dense, or the stacked experts). How a checkpoint spells a tensor (Phi-3's fused {@code qkv_proj}, GLM's
     * fused {@code gate_up_proj}, 3-D expert tensors, multimodal prefixes) is detected from the shard map, not
     * from the spec: it is a layout of the files, not of the model.
     */
    private List<Unit> planDecoder(Decoder d) {
        int dim = d.core().dim();
        List<Unit> units = new ArrayList<>();
        // Embedding first: it is the largest single tensor and lives in the first shard.
        units.add(new Unit("embed", List.of(quantize(EMBED, head))));
        for (int l = 0; l < d.core().layers(); l++) {
            LayerSpec spec = d.layers().get(l);
            String p = "model.layers." + l + ".";
            List<Rule> rules = new ArrayList<>();
            for (String n : layerNorms(p)) {
                rules.add(norm(n));
            }
            switch (spec.attn()) {
                case Attention.Gqa g -> {
                    if (g.bias()) {
                        // q/k/v projection biases (Qwen2, GLM), stored as f32 under their checkpoint names
                        for (String n : new String[]{"q_proj", "k_proj", "v_proj"}) {
                            rules.add(f32(p + "self_attn." + n + ".bias", p + "self_attn." + n + ".bias"));
                        }
                    }
                    if (fusedQkv(p)) {
                        // Phi-3: [q rows | k rows | v rows] -> the three cache tensors, so the backends see a plain Llama layer
                        String qkv = p + "self_attn.qkv_proj.weight";
                        int q = g.qDim();
                        int kv = g.kvDim();
                        rules.add(splitRows(p + "self_attn.q_proj.weight", qkv, quant, 0, q, q + 2 * kv, dim));
                        rules.add(splitRows(p + "self_attn.k_proj.weight", qkv, quant, q, kv, q + 2 * kv, dim));
                        rules.add(splitRows(p + "self_attn.v_proj.weight", qkv, quant, q + kv, kv, q + 2 * kv, dim));
                        rules.add(quantize(p + "self_attn.o_proj.weight", quant));
                    } else {
                        for (String n : new String[]{"q_proj", "k_proj", "v_proj", "o_proj"}) {
                            rules.add(quantize(p + "self_attn." + n + ".weight", quant));
                        }
                    }
                    if (g.outBias()) {
                        rules.add(f32(p + "self_attn.o_proj.bias", p + "self_attn.o_proj.bias"));
                    }
                }
                case Attention.Mla m -> rules.addAll(mlaRules(p, m, dim));
            }
            switch (spec.mlp()) {
                case Mlp.Moe m -> rules.addAll(moeRules(p, m, dim));
                case Mlp.Dense dense -> {
                    if (fusedGateUp(p)) {
                        // GLM: gate_up_proj is already [gate; up] stacked, the layout of mlp.gate_up
                        rules.add(quantize(p + "mlp.gate_up", p + "mlp.gate_up_proj.weight", quant, 0, 0));
                    } else {
                        rules.add(stack(p + "mlp.gate_up", List.of(p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"), quant, 0, 0));
                    }
                    rules.add(quantize(p + "mlp.down_proj.weight", quant));
                }
            }
            units.add(new Unit("layer." + l, List.copyOf(rules)));
        }
        units.add(new Unit("final", finalRules(norm(FINAL_NORM))));
        return units;
    }

    /** Final norm, plus the LM head (in the head format) when the checkpoint has an untied one. */
    private List<Rule> finalRules(Rule finalNorm) {
        List<Rule> rules = new ArrayList<>(List.of(finalNorm));
        if (hasLmHead()) {
            rules.add(quantize(LM_HEAD, head));
        }
        return List.copyOf(rules);
    }

    /**
     * DeepSeek attention. {@code kv_a_proj_with_mqa [kvLora + rope, dim]} is stored as two tensors,
     * {@code kv_a_latent.weight} (first kvLora rows) and {@code k_pe_proj.weight} (last rope rows).
     */
    private List<Rule> mlaRules(String p, Attention.Mla d, int dim) {
        String at = p + "self_attn.";
        List<Rule> rules = new ArrayList<>();
        if (d.qLoraRank() > 0) {
            rules.add(f32(at + "q_a_layernorm.weight", at + "q_a_layernorm.weight"));
            rules.add(quantize(at + "q_a_proj.weight", quant));
            rules.add(quantize(at + "q_b_proj.weight", quant));
        } else {
            rules.add(quantize(at + "q_proj.weight", quant));
        }
        rules.add(f32(at + "kv_a_layernorm.weight", at + "kv_a_layernorm.weight"));
        String kva = at + "kv_a_proj_with_mqa.weight";
        int total = d.latentDim();
        rules.add(splitRows(at + "kv_a_latent.weight", kva, quant, 0, d.kvLoraRank(), total, dim));
        rules.add(splitRows(at + "k_pe_proj.weight", kva, quant, d.kvLoraRank(), d.rope(), total, dim));
        rules.add(quantize(at + "kv_b_proj.weight", quant));
        rules.add(quantize(at + "o_proj.weight", quant));
        return rules;
    }

    /**
     * A MoE layer: the f32 router, DeepSeek's shared experts (stacked gate/up, down) and V3 correction bias,
     * then the experts. Two checkpoint layouts exist: per-expert 2-D tensors ({@code mlp.experts.N.gate_proj.weight},
     * router {@code mlp.gate.weight}; transformers <= 4.53) and fused 3-D tensors ({@code mlp.experts.gate_up_proj}
     * {@code [E, 2m, dim]}, {@code mlp.experts.down_proj [E, dim, m]}, router {@code mlp.router.weight}).
     * Both end up as the same stacked {@code [E*2m, dim]} and {@code [E*dim, m]} matrices.
     */
    private List<Rule> moeRules(String p, Mlp.Moe moe, int dim) {
        int e = moe.experts();
        int m = moe.moeDim();
        List<Rule> rules = new ArrayList<>();
        rules.add(f32(p + "mlp.router", routerName(p)));
        if (moe.shared() != null) {
            String sp = p + "mlp.shared_experts.";
            int sh = moe.shared().hidden();
            rules.add(stack(p + "mlp.shared.gate_up", List.of(sp + "gate_proj.weight", sp + "up_proj.weight"), quant, sh, 0));
            rules.add(quantize(p + "mlp.shared.down", sp + "down_proj.weight", quant, 0, sh));
        }
        if (moe.router() instanceof Router.DeepSeek r && r.hasBias()) {
            rules.add(f32(p + "mlp.router_bias", p + "mlp.gate.e_score_correction_bias"));
        }
        if (fusedExperts(p)) {
            rules.add(flatten3d(p + "mlp.experts.gate_up", p + "mlp.experts.gate_up_proj", quant, new long[]{e, 2L * m, dim}));
            rules.add(flatten3d(p + "mlp.experts.down", p + "mlp.experts.down_proj", quant, new long[]{e, dim, m}));
            return rules;
        }
        List<String> gateUp = new ArrayList<>(2 * e);
        List<String> downs = new ArrayList<>(e);
        for (int i = 0; i < e; i++) {
            String ep = p + "mlp.experts." + i + ".";
            gateUp.add(ep + "gate_proj.weight");   // rows [i*2m, i*2m+m) = gate_i
            gateUp.add(ep + "up_proj.weight");     // rows [i*2m+m, (i+1)*2m) = up_i
            downs.add(ep + "down_proj.weight");
        }
        rules.add(stack(p + "mlp.experts.gate_up", gateUp, quant, m, 0));
        rules.add(stack(p + "mlp.experts.down", downs, quant, 0, m));
        return rules;
    }

    private List<String> layerNorms(String p) {
        List<String> out = new ArrayList<>(4);
        for (String n : new String[]{"input_layernorm.weight", "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                "post_self_attn_layernorm.weight", "post_mlp_layernorm.weight"}) {
            if (shardOf.containsKey(p + n)) {
                out.add(p + n);
            }
        }
        return out;
    }

    private String routerName(String p) {
        return shardOf.containsKey(p + "mlp.gate.weight") ? p + "mlp.gate.weight" : p + "mlp.router.weight";
    }

    /** Phi-3 checkpoints store q, k and v as one {@code self_attn.qkv_proj} (rows in that order). */
    private boolean fusedQkv(String p) {
        return shardOf.containsKey(p + "self_attn.qkv_proj.weight");
    }

    /** GLM checkpoints store the dense MLP's gate and up projections as one {@code mlp.gate_up_proj}. */
    private boolean fusedGateUp(String p) {
        return shardOf.containsKey(p + "mlp.gate_up_proj.weight");
    }

    private boolean fusedExperts(String p) {
        return shardOf.containsKey(p + "mlp.experts.gate_up_proj");
    }

    private boolean hasLmHead() {
        return !arch.core().tiedEmbeddings() && shardOf.containsKey(LM_HEAD);
    }

    // ------------------------------------------------------------------ Mamba

    private static final String M_EMBED = "backbone.embeddings.weight";
    private static final String M_FINAL_NORM = "backbone.norm_f.weight";

    /** embedding, one unit per Mamba block, final norm (+ lm_head if the checkpoint has an untied one). */
    private List<Unit> planMamba(MambaArch m) {
        List<Unit> units = new ArrayList<>();
        units.add(new Unit("embed", List.of(quantize(M_EMBED, head))));
        for (int l = 0; l < m.core().layers(); l++) {
            String p = "backbone.layers." + l + ".";
            String q = p + "mixer.";
            List<Rule> rules = new ArrayList<>();
            rules.add(f32(p + "norm.weight", p + "norm.weight"));
            rules.add(quantize(q + "in_proj.weight", quant));
            rules.add(f32(q + "conv1d.weight", q + "conv1d.weight"));
            if (m.convBias()) {
                rules.add(f32(q + "conv1d.bias", q + "conv1d.bias"));
            }
            rules.add(quantize(q + "x_proj.weight", quant));
            // dt_proj has dt_rank columns (48 for mamba-130m: not a multiple of the block size): stays f32
            rules.add(f32(q + "dt_proj.weight", q + "dt_proj.weight"));
            rules.add(f32(q + "dt_proj.bias", q + "dt_proj.bias"));
            rules.add(negExp(q + "A", q + "A_log"));
            rules.add(f32(q + "D", q + "D"));
            rules.add(quantize(q + "out_proj.weight", quant));
            units.add(new Unit("layer." + l, List.copyOf(rules)));
        }
        units.add(new Unit("final", finalRules(f32(M_FINAL_NORM, M_FINAL_NORM))));
        return units;
    }

    // ------------------------------------------------------------------ Qwen3-Next

    /**
     * Quantizes rows gathered from several 2-D {@code [rowsOf[i], cols]} sources as one tensor: output row {@code r} is
     * row {@code map[r][1]} of source {@code map[r][0]}. A pure row permutation, so it quantizes exactly like the rows in
     * place (quantization blocks run along a row).
     */
    private static Rule gather(String cacheName, List<String> sources, QType type, int[][] map, int[] rowsOf, int cols) {
        return new Rule(cacheName, List.copyOf(sources), (a, s) -> {
            for (int i = 0; i < s.size(); i++) {
                SafeTensors.Tensor t = matrix(s.get(i));
                if (t.dim(0) != rowsOf[i] || t.dim(1) != cols) {
                    throw new IOException(t + " does not match [" + rowsOf[i] + ", " + cols + "]");
                }
            }
            SafeTensors.Tensor[] arr = s.toArray(new SafeTensors.Tensor[0]);
            return QTensor.quantize(a, type, map.length, cols,
                    (r, dst) -> arr[map[r][0]].toFloat((long) map[r][1] * cols, dst, 0, cols));
        });
    }

    /**
     * Embedding, one unit per layer, final norm + head. The zero-centred norms (input / post-attention / q / k / final)
     * get their {@code 1 + w} folded in; the DeltaNet's gated norm is stored as is. {@code in_proj_qkvz} and
     * {@code in_proj_ba} (rows grouped per key head: [q k v z] and [b a] of each group) become one
     * {@code linear_attn.in_proj} with the rows [q | k | v | z | b | a], each part in head order; {@code q_proj} (per
     * head: headDim query rows then headDim gate rows) becomes {@code q_proj} and {@code q_gate}. The shared expert is
     * stored like DeepSeek's ({@code mlp.shared.gate_up} / {@code mlp.shared.down}) plus its gate vector. The
     * multi-token-prediction tensors ({@code mtp.*}) are never read.
     */
    private List<Unit> planQwen3Next(Qwen3NextArch q) {
        int dim = q.core().dim();
        List<Unit> units = new ArrayList<>();
        units.add(new Unit("embed", List.of(quantize(EMBED, head))));
        for (int l = 0; l < q.core().layers(); l++) {
            String p = "model.layers." + l + ".";
            List<Rule> rules = new ArrayList<>();
            rules.add(f32PlusOne(p + "input_layernorm.weight", p + "input_layernorm.weight"));
            rules.add(f32PlusOne(p + "post_attention_layernorm.weight", p + "post_attention_layernorm.weight"));
            if (q.isFull(l)) {
                String a = p + "self_attn.";
                Attention.Gqa g = q.attention();
                int hd = g.headDim();
                int[][] qRows = new int[g.qDim()][];
                int[][] gateRows = new int[g.qDim()][];
                for (int h = 0; h < g.heads(); h++) {
                    for (int i = 0; i < hd; i++) {
                        qRows[h * hd + i] = new int[]{0, h * 2 * hd + i};
                        gateRows[h * hd + i] = new int[]{0, h * 2 * hd + hd + i};
                    }
                }
                int[] qShape = {2 * g.qDim()};
                rules.add(gather(a + "q_proj.weight", List.of(a + "q_proj.weight"), quant, qRows, qShape, dim));
                rules.add(gather(a + "q_gate", List.of(a + "q_proj.weight"), quant, gateRows, qShape, dim));
                rules.add(quantize(a + "k_proj.weight", quant));
                rules.add(quantize(a + "v_proj.weight", quant));
                rules.add(quantize(a + "o_proj.weight", quant));
                rules.add(f32PlusOne(a + "q_norm.weight", a + "q_norm.weight"));
                rules.add(f32PlusOne(a + "k_norm.weight", a + "k_norm.weight"));
            } else {
                String a = p + "linear_attn.";
                Qwen3NextArch.DeltaNet d = q.deltaNet();
                int nk = d.keyHeads();
                int dk = d.keyDim();
                int rv = d.group() * d.valueDim(); // v (and z) rows per key-head group
                int r = d.group();
                int group = 2 * dk + 2 * rv;
                int[][] rows = new int[d.projDim()][];
                int o = 0;
                for (int part = 0; part < 4; part++) { // q, k, v, z
                    int start = part == 0 ? 0 : part == 1 ? dk : part == 2 ? 2 * dk : 2 * dk + rv;
                    int len = part < 2 ? dk : rv;
                    for (int g = 0; g < nk; g++) {
                        for (int i = 0; i < len; i++) {
                            rows[o++] = new int[]{0, g * group + start + i};
                        }
                    }
                }
                for (int part = 0; part < 2; part++) { // b, a
                    for (int g = 0; g < nk; g++) {
                        for (int j = 0; j < r; j++) {
                            rows[o++] = new int[]{1, g * 2 * r + part * r + j};
                        }
                    }
                }
                rules.add(gather(a + "in_proj", List.of(a + "in_proj_qkvz.weight", a + "in_proj_ba.weight"), quant, rows,
                        new int[]{nk * group, nk * 2 * r}, dim));
                rules.add(f32(a + "conv1d.weight", a + "conv1d.weight"));
                rules.add(negExp(a + "A", a + "A_log"));
                rules.add(f32(a + "dt_bias", a + "dt_bias"));
                rules.add(f32(a + "norm.weight", a + "norm.weight"));
                rules.add(quantize(a + "out_proj.weight", quant));
            }
            switch (q.mlps().get(l)) {
                case Mlp.Moe m -> {
                    rules.addAll(moeRules(p, new Mlp.Moe(m.experts(), m.topK(), m.moeDim(), m.normTopK(), m.router(), null), dim));
                    String sp = p + "mlp.shared_expert.";
                    int sh = m.shared().hidden();
                    rules.add(stack(p + "mlp.shared.gate_up", List.of(sp + "gate_proj.weight", sp + "up_proj.weight"), quant, sh, 0));
                    rules.add(quantize(p + "mlp.shared.down", sp + "down_proj.weight", quant, 0, sh));
                    rules.add(f32(p + "mlp.shared_gate", p + "mlp.shared_expert_gate.weight"));
                }
                case Mlp.Dense dense -> {
                    rules.add(stack(p + "mlp.gate_up", List.of(p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"), quant, 0, 0));
                    rules.add(quantize(p + "mlp.down_proj.weight", quant));
                }
            }
            units.add(new Unit("layer." + l, List.copyOf(rules)));
        }
        units.add(new Unit("final", finalRules(f32PlusOne(FINAL_NORM, FINAL_NORM))));
        return units;
    }

    // ------------------------------------------------------------------ shards

    private SafeTensors.Tensor tensor(String name) throws IOException {
        String shard = shardOf.get(name);
        if (shard == null) {
            throw new IOException("tensor not found in checkpoint: " + name);
        }
        SafeTensors st = open.get(shard);
        if (st == null) {
            st = SafeTensors.open(provider.ensure(ShardProvider.checkName(dir, shard)));
            open.put(shard, st);
        }
        SafeTensors.Tensor t = st.get(checkpointName.getOrDefault(name, name));
        if (fp8 != null && (t.dtype() == SafeTensors.DType.F8_E4M3 || t.dtype() == SafeTensors.DType.F8_E5M2)
                && shardOf.containsKey(name + "_scale_inv")) {
            return fp8.dequantized(name, t, tensor(name + "_scale_inv"));
        }
        if (mlx != null && t != null && t.dtype() == SafeTensors.DType.U32 && name.endsWith(".weight")) {
            String base = name.substring(0, name.length() - ".weight".length());
            if (shardOf.containsKey(base + ".scales") && mlx.mode != MlxQuant.Mode.AFFINE) {
                return mlx.dequantized(name, t, tensor(base + ".scales"), null);
            }
            if (shardOf.containsKey(base + ".scales") && shardOf.containsKey(base + ".biases")) {
                return mlx.dequantized(name, t, tensor(base + ".scales"), tensor(base + ".biases"));
            }
        }
        return t;
    }

    /**
     * Maps the prefixes of multimodal checkpoints onto the plain names: {@code language_model.model.X}
     * (Gemma 3 4B+ as released) and {@code model.language_model.X} (transformers >= 4.52) become {@code model.X}.
     * Vision / projector tensors keep their names and are never read.
     */
    private static String canonicalName(String n) {
        if (n.startsWith("language_model.model.")) {
            return "model." + n.substring("language_model.model.".length());
        }
        if (n.startsWith("model.language_model.")) {
            return "model." + n.substring("model.language_model.".length());
        }
        if (n.startsWith("language_model.lm_head.")) {
            return "lm_head." + n.substring("language_model.lm_head.".length());
        }
        return n;
    }

    private Map<String, String> canonical(Map<String, String> raw) {
        Map<String, String> m = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : raw.entrySet()) {
            String c = canonicalName(e.getKey());
            if (!c.equals(e.getKey())) {
                checkpointName.put(c, e.getKey());
            }
            m.put(c, e.getValue());
        }
        return m;
    }

    /** Marks a planned tensor read consumed; closes and releases its shard once nothing else in it is needed. */
    private void done(String name) {
        String shard = shardOf.get(name);
        if (shard == null) {
            return;
        }
        Integer left = remaining.computeIfPresent(shard, (k, v) -> v - 1);
        if (left != null && left <= 0) {
            release(shard);
        }
    }

    /** Closes {@code shard} if it is open and hands it back to the provider (which may delete it). */
    private void release(String shard) {
        remaining.remove(shard);
        SafeTensors st = open.remove(shard);
        if (st != null) {
            st.close();
        }
        try {
            provider.release(ShardProvider.checkName(dir, shard));
        } catch (IOException e) {
            System.err.println("warning: could not release " + shard + ": " + e.getMessage());
        }
    }

    /**
     * tensor name -> shard file name: the {@code weight_map} of {@code model.safetensors.index.json} when there
     * is one, otherwise the contents of {@code model.safetensors} (the same set {@code HuggingFaceHub.download}
     * fetches; other {@code .safetensors} files in the directory are not part of the checkpoint).
     */
    private Map<String, String> shardMap() throws IOException {
        Path index = dir.resolve("model.safetensors.index.json");
        if (Files.isRegularFile(index)) {
            // The index is downloaded data: every shard name is validated before it reaches the provider,
            // whose low-disk implementation deletes dir/name.
            Map<String, String> m = new LinkedHashMap<>();
            for (Map.Entry<String, String> e : SafeTensors.readWeightMap(index).entrySet()) {
                m.put(e.getKey(), ShardProvider.checkName(dir, e.getValue()));
            }
            return canonical(m);
        }
        Path single = provider.ensure("model.safetensors");
        Map<String, String> m = new LinkedHashMap<>();
        try (SafeTensors st = SafeTensors.open(single)) {
            for (String name : st.tensors().keySet()) {
                m.put(name, single.getFileName().toString());
            }
        }
        return canonical(m);
    }
}
