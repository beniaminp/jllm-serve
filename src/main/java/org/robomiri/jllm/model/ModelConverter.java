package org.robomiri.jllm.model;

import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Converts a Hugging Face checkpoint into a {@link QuantFile}, one unit (embedding, layer, final norm + head)
 * at a time.
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

    /** One conversion step: {@code inputs} are the checkpoint tensors it reads, {@code step} writes its outputs. */
    private record Unit(String id, List<String> inputs, Step step) {
    }

    @FunctionalInterface
    private interface Step {
        void run(QuantFile.Writer w, Arena a) throws IOException;
    }

    private final Path dir;
    private final ModelConfig cfg;
    private final QType quant;
    private final ShardProvider provider;
    private final Map<String, String> shardOf;
    private final List<Unit> units;
    /** Per shard: how many planned tensor reads are still outstanding. */
    private final Map<String, Integer> remaining = new HashMap<>();
    private final Map<String, SafeTensors> open = new LinkedHashMap<>();

    private ModelConverter(Path dir, ModelConfig cfg, QType quant, ShardProvider provider) throws IOException {
        this.dir = dir;
        this.cfg = cfg;
        this.quant = quant;
        this.provider = provider;
        this.shardOf = shardMap();
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
    public static void convert(Path dir, ModelConfig cfg, QType quant, ShardProvider provider) throws IOException {
        long t0 = System.nanoTime();
        ModelConverter c = new ModelConverter(dir, cfg, quant, provider);
        try (QuantFile.Writer w = QuantFile.Writer.openOrResume(dir, quant, cfg.toString(), c.unitIds())) {
            c.run(w);
            w.finish();
            System.err.printf("\rConverted %s to %s in %.0fs (%.0f MB) -> %s%n", cfg.architecture(), quant,
                    (System.nanoTime() - t0) / 1e9, w.bytesWritten() / 1e6, QuantFile.binPath(dir, quant));
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
                try (Arena a = Arena.ofShared()) {
                    u.step().run(w, a);
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

    // ------------------------------------------------------------------ plan

    /** The ordered conversion units together with every checkpoint tensor each of them reads. */
    private List<Unit> plan() {
        List<Unit> units = new ArrayList<>();
        // Embedding first: it is the largest single tensor and lives in the first shard.
        units.add(new Unit("embed", List.of(EMBED), (w, a) -> w.put(EMBED, quantize(a, QType.Q8_0, tensor(EMBED)))));
        for (int l = 0; l < cfg.layers(); l++) {
            int layer = l;
            String p = "model.layers." + l + ".";
            List<String> in = new ArrayList<>(layerNorms(p));
            if (fusedQkv(p)) {
                in.add(p + "self_attn.qkv_proj.weight");
                in.add(p + "self_attn.o_proj.weight");
            } else {
                for (String n : new String[]{"q_proj", "k_proj", "v_proj", "o_proj"}) {
                    in.add(p + "self_attn." + n + ".weight");
                }
            }
            in.addAll(biasNames(p));
            if (cfg.isMoeLayer(l)) {
                in.add(routerName(p));
                if (fusedExperts(p)) {
                    in.add(p + "mlp.experts.gate_up_proj");
                    in.add(p + "mlp.experts.down_proj");
                } else {
                    for (int i = 0; i < cfg.numExperts(); i++) {
                        String ep = p + "mlp.experts." + i + ".";
                        in.add(ep + "gate_proj.weight");
                        in.add(ep + "up_proj.weight");
                        in.add(ep + "down_proj.weight");
                    }
                }
            } else if (fusedGateUp(p)) {
                in.add(p + "mlp.gate_up_proj.weight");
                in.add(p + "mlp.down_proj.weight");
            } else {
                in.add(p + "mlp.gate_proj.weight");
                in.add(p + "mlp.up_proj.weight");
                in.add(p + "mlp.down_proj.weight");
            }
            units.add(new Unit("layer." + l, List.copyOf(in), (w, a) -> convertLayer(w, a, layer, p)));
        }
        List<String> fin = new ArrayList<>(List.of(FINAL_NORM));
        if (hasLmHead()) {
            fin.add(LM_HEAD);
        }
        units.add(new Unit("final", List.copyOf(fin), (w, a) -> {
            w.putFloats(FINAL_NORM, tensor(FINAL_NORM).toFloatArray());
            if (hasLmHead()) {
                w.put(LM_HEAD, quantize(a, QType.Q8_0, tensor(LM_HEAD)));
            }
        }));
        return units;
    }

    /** q/k/v projection biases (Qwen2), stored as f32 under their checkpoint names. */
    private List<String> biasNames(String p) {
        if (!cfg.attentionBias()) {
            return List.of();
        }
        return List.of(p + "self_attn.q_proj.bias", p + "self_attn.k_proj.bias", p + "self_attn.v_proj.bias");
    }

    private List<String> layerNorms(String p) {
        List<String> out = new ArrayList<>(4);
        for (String n : new String[]{"input_layernorm.weight", "post_attention_layernorm.weight",
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
        return !cfg.tiedEmbeddings() && shardOf.containsKey(LM_HEAD);
    }

    // ------------------------------------------------------------------ steps

    private void convertLayer(QuantFile.Writer w, Arena a, int l, String p) throws IOException {
        for (String n : layerNorms(p)) {
            w.putFloats(n, tensor(n).toFloatArray());
        }
        for (String n : biasNames(p)) {
            w.putFloats(n, tensor(n).toFloatArray());
        }
        if (fusedQkv(p)) {
            // Phi-3: [q rows | k rows | v rows] -> the three cache tensors, so the backends see a plain Llama layer
            String name = p + "self_attn.qkv_proj.weight";
            SafeTensors.Tensor qkv = tensor(name);
            int q = cfg.qDim();
            int kv = cfg.kvDim();
            if (qkv.shape().length != 2 || qkv.dim(0) != q + 2 * kv || qkv.dim(1) != cfg.dim()) {
                throw new IOException("layer " + l + ": " + qkv + " does not match [" + (q + 2 * kv) + ", " + cfg.dim() + "]");
            }
            w.put(p + "self_attn.q_proj.weight", quantizeRows(a, quant, qkv, 0, q));
            w.put(p + "self_attn.k_proj.weight", quantizeRows(a, quant, qkv, q, kv));
            w.put(p + "self_attn.v_proj.weight", quantizeRows(a, quant, qkv, q + kv, kv));
            String o = p + "self_attn.o_proj.weight";
            w.put(o, quantize(a, quant, tensor(o)));
        } else {
            for (String n : new String[]{"q_proj", "k_proj", "v_proj", "o_proj"}) {
                String name = p + "self_attn." + n + ".weight";
                w.put(name, quantize(a, quant, tensor(name)));
            }
        }
        if (cfg.isMoeLayer(l)) {
            convertMoe(w, a, l, p);
        } else if (fusedGateUp(p)) {
            // GLM: gate_up_proj is already [gate; up] stacked, the layout of mlp.gate_up
            w.put(p + "mlp.gate_up", quantize(a, quant, tensor(p + "mlp.gate_up_proj.weight")));
            String d = p + "mlp.down_proj.weight";
            w.put(d, quantize(a, quant, tensor(d)));
        } else {
            String g = p + "mlp.gate_proj.weight";
            String u = p + "mlp.up_proj.weight";
            String d = p + "mlp.down_proj.weight";
            w.put(p + "mlp.gate_up", stack(a, quant, List.of(tensor(g), tensor(u))));
            w.put(d, quantize(a, quant, tensor(d)));
        }
    }

    /**
     * Two checkpoint layouts exist: per-expert 2-D tensors ({@code mlp.experts.N.gate_proj.weight}, router
     * {@code mlp.gate.weight}; transformers <= 4.53) and fused 3-D tensors ({@code mlp.experts.gate_up_proj}
     * {@code [E, 2m, dim]}, {@code mlp.experts.down_proj [E, dim, m]}, router {@code mlp.router.weight}).
     * Both end up as the same stacked {@code [E*2m, dim]} and {@code [E*dim, m]} matrices.
     */
    private void convertMoe(QuantFile.Writer w, Arena a, int l, String p) throws IOException {
        int e = cfg.numExperts();
        int m = cfg.moeDim();
        w.putFloats(p + "mlp.router", tensor(routerName(p)).toFloatArray());

        if (fusedExperts(p)) {
            SafeTensors.Tensor gu = tensor(p + "mlp.experts.gate_up_proj");
            SafeTensors.Tensor dn = tensor(p + "mlp.experts.down_proj");
            if (gu.shape().length != 3 || gu.dim(0) != e || gu.dim(1) != 2 * m || gu.dim(2) != cfg.dim()
                    || dn.shape().length != 3 || dn.dim(0) != e || dn.dim(1) != cfg.dim() || dn.dim(2) != m) {
                throw new IOException("layer " + l + ": unexpected fused expert shapes " + gu + " / " + dn);
            }
            w.put(p + "mlp.experts.gate_up", quantize(a, quant, flatten(gu)));
            w.put(p + "mlp.experts.down", quantize(a, quant, flatten(dn)));
            return;
        }

        List<SafeTensors.Tensor> gu = new ArrayList<>(2 * e);
        List<SafeTensors.Tensor> downs = new ArrayList<>(e);
        for (int i = 0; i < e; i++) {
            String ep = p + "mlp.experts." + i + ".";
            SafeTensors.Tensor gate = tensor(ep + "gate_proj.weight");
            SafeTensors.Tensor up = tensor(ep + "up_proj.weight");
            SafeTensors.Tensor down = tensor(ep + "down_proj.weight");
            if (gate.dim(0) != m || up.dim(0) != m || down.dim(1) != m) {
                throw new IOException("expert " + i + " of layer " + l + " does not match moe_intermediate_size " + m);
            }
            gu.add(gate);   // rows [i*2m, i*2m+m) = gate_i
            gu.add(up);     // rows [i*2m+m, (i+1)*2m) = up_i
            downs.add(down);
        }
        w.put(p + "mlp.experts.gate_up", stack(a, quant, gu));
        w.put(p + "mlp.experts.down", stack(a, quant, downs));
    }

    /** Views a {@code [a, b, c]} tensor as {@code [a*b, c]} (row-major, so no data moves). */
    private static SafeTensors.Tensor flatten(SafeTensors.Tensor t) {
        long[] s = t.shape();
        return new SafeTensors.Tensor(t.name(), t.dtype(), new long[]{s[0] * s[1], s[2]}, t.data());
    }

    // ------------------------------------------------------------------ helpers

    private static QTensor quantize(Arena a, QType type, SafeTensors.Tensor t) {
        int rows = t.dim(0);
        int cols = t.dim(1);
        return QTensor.quantize(a, type, rows, cols, (r, dst) -> t.toFloat((long) r * cols, dst, 0, cols));
    }

    /** Quantizes rows {@code [start, start + rows)} of a 2-D tensor. */
    private static QTensor quantizeRows(Arena a, QType type, SafeTensors.Tensor t, int start, int rows) {
        int cols = t.dim(1);
        return QTensor.quantize(a, type, rows, cols, (r, dst) -> t.toFloat((long) (start + r) * cols, dst, 0, cols));
    }

    /** Quantizes several {@code [rows_i, cols]} matrices as one {@code [Σ rows_i, cols]} tensor. */
    private static QTensor stack(Arena a, QType type, List<SafeTensors.Tensor> parts) {
        int cols = parts.getFirst().dim(1);
        int[] starts = new int[parts.size() + 1];
        for (int i = 0; i < parts.size(); i++) {
            if (parts.get(i).dim(1) != cols) {
                throw new IllegalArgumentException("cannot stack " + parts.get(i) + ": cols differ");
            }
            starts[i + 1] = starts[i] + parts.get(i).dim(0);
        }
        SafeTensors.Tensor[] arr = parts.toArray(new SafeTensors.Tensor[0]);
        return QTensor.quantize(a, type, starts[parts.size()], cols, (r, dst) -> {
            int i = java.util.Arrays.binarySearch(starts, r);
            if (i < 0) {
                i = -i - 2;
            }
            arr[i].toFloat((long) (r - starts[i]) * cols, dst, 0, cols);
        });
    }

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
        return st.get(name);
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

    /** tensor name -> shard file name, from the index file or by opening the single safetensors file. */
    private Map<String, String> shardMap() throws IOException {
        Path index = dir.resolve("model.safetensors.index.json");
        if (Files.isRegularFile(index)) {
            Map<String, Object> wm = org.robomiri.jllm.json.Json.map(
                    org.robomiri.jllm.json.Json.parse(Files.readString(index)), "weight_map");
            if (wm == null) {
                throw new IOException(index + " has no weight_map");
            }
            // The index is downloaded data: every shard name is validated before it reaches the provider,
            // whose low-disk implementation deletes dir/name.
            Map<String, String> m = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : wm.entrySet()) {
                if (!(e.getValue() instanceof String shard)) {
                    throw new IOException(index + ": weight_map entry for " + e.getKey() + " is not a file name");
                }
                m.put(e.getKey(), ShardProvider.checkName(dir, shard));
            }
            return m;
        }
        List<Path> files;
        try (Stream<Path> s = Files.list(dir)) {
            files = s.filter(f -> f.getFileName().toString().endsWith(".safetensors")).sorted().toList();
        }
        if (files.isEmpty()) {
            files = List.of(provider.ensure("model.safetensors"));
        }
        Map<String, String> m = new LinkedHashMap<>();
        for (Path f : files) {
            try (SafeTensors st = SafeTensors.open(f)) {
                for (String name : st.tensors().keySet()) {
                    m.put(name, f.getFileName().toString());
                }
            }
        }
        return m;
    }
}
