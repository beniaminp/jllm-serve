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
 * Converts a Hugging Face checkpoint into a {@link QuantFile}, one layer at a time.
 *
 * <p>Shards are opened only when a tensor in them is needed and closed (and {@link ShardProvider#release
 * released}) as soon as every tensor they hold has been consumed, so at most two shards are open at any time.
 * MoE experts are stacked into two tensors per layer (gate/up for all experts, down for all experts).
 */
public final class ModelConverter {

    private final Path dir;
    private final ModelConfig cfg;
    private final QType quant;
    private final ShardProvider provider;
    private final Map<String, String> shardOf;
    private final Map<String, Integer> remaining = new HashMap<>();
    private final Map<String, SafeTensors> open = new LinkedHashMap<>();

    private ModelConverter(Path dir, ModelConfig cfg, QType quant, ShardProvider provider) throws IOException {
        this.dir = dir;
        this.cfg = cfg;
        this.quant = quant;
        this.provider = provider;
        this.shardOf = shardMap();
        for (String shard : shardOf.values()) {
            remaining.merge(shard, 1, Integer::sum);
        }
    }

    public static void convert(Path dir, ModelConfig cfg, QType quant, ShardProvider provider) throws IOException {
        long t0 = System.nanoTime();
        ModelConverter c = new ModelConverter(dir, cfg, quant, provider);
        try (QuantFile.Writer w = new QuantFile.Writer(dir, quant)) {
            c.run(w);
            w.finish();
            System.err.printf("\rConverted %s to %s in %.0fs (%.0f MB) -> %s%n", cfg.architecture(), quant,
                    (System.nanoTime() - t0) / 1e9, w.bytesWritten() / 1e6, QuantFile.binPath(dir, quant));
        } finally {
            c.open.values().forEach(SafeTensors::close);
        }
    }

    private void run(QuantFile.Writer w) throws IOException {
        // Embedding first: it is the largest single tensor and lives in the first shard.
        try (Arena a = Arena.ofShared()) {
            w.put("model.embed_tokens.weight", quantize(a, QType.Q8_0, tensor("model.embed_tokens.weight")));
        }
        done("model.embed_tokens.weight");

        for (int l = 0; l < cfg.layers(); l++) {
            String p = "model.layers." + l + ".";
            List<String> used = new ArrayList<>();
            try (Arena a = Arena.ofShared()) {
                for (String n : new String[]{"input_layernorm.weight", "post_attention_layernorm.weight",
                        "self_attn.q_norm.weight", "self_attn.k_norm.weight"}) {
                    if (shardOf.containsKey(p + n)) {
                        w.putFloats(p + n, tensor(p + n).toFloatArray());
                        used.add(p + n);
                    }
                }
                for (String n : new String[]{"q_proj", "k_proj", "v_proj", "o_proj"}) {
                    String name = p + "self_attn." + n + ".weight";
                    w.put(name, quantize(a, quant, tensor(name)));
                    used.add(name);
                }
                if (cfg.isMoeLayer(l)) {
                    convertMoe(w, a, l, p, used);
                } else {
                    String g = p + "mlp.gate_proj.weight";
                    String u = p + "mlp.up_proj.weight";
                    String d = p + "mlp.down_proj.weight";
                    w.put(p + "mlp.gate_up", stack(a, quant, List.of(tensor(g), tensor(u))));
                    w.put(d, quantize(a, quant, tensor(d)));
                    used.add(g);
                    used.add(u);
                    used.add(d);
                }
            }
            used.forEach(this::done);
            System.err.printf("\rConverting to %s: layer %d/%d (%.0f MB written)", quant, l + 1, cfg.layers(),
                    w.bytesWritten() / 1e6);
        }

        try (Arena a = Arena.ofShared()) {
            w.putFloats("model.norm.weight", tensor("model.norm.weight").toFloatArray());
            done("model.norm.weight");
            if (!cfg.tiedEmbeddings() && shardOf.containsKey("lm_head.weight")) {
                w.put("lm_head.weight", quantize(a, QType.Q8_0, tensor("lm_head.weight")));
                done("lm_head.weight");
            }
        }
    }

    /**
     * Two checkpoint layouts exist: per-expert 2-D tensors ({@code mlp.experts.N.gate_proj.weight}, router
     * {@code mlp.gate.weight}; transformers <= 4.53) and fused 3-D tensors ({@code mlp.experts.gate_up_proj}
     * {@code [E, 2m, dim]}, {@code mlp.experts.down_proj [E, dim, m]}, router {@code mlp.router.weight}).
     * Both end up as the same stacked {@code [E*2m, dim]} and {@code [E*dim, m]} matrices.
     */
    private void convertMoe(QuantFile.Writer w, Arena a, int l, String p, List<String> used) throws IOException {
        int e = cfg.numExperts();
        int m = cfg.moeDim();
        String router = shardOf.containsKey(p + "mlp.gate.weight") ? p + "mlp.gate.weight" : p + "mlp.router.weight";
        w.putFloats(p + "mlp.router", tensor(router).toFloatArray());
        used.add(router);

        if (shardOf.containsKey(p + "mlp.experts.gate_up_proj")) {
            SafeTensors.Tensor gu = tensor(p + "mlp.experts.gate_up_proj");
            SafeTensors.Tensor dn = tensor(p + "mlp.experts.down_proj");
            if (gu.shape().length != 3 || gu.dim(0) != e || gu.dim(1) != 2 * m || gu.dim(2) != cfg.dim()
                    || dn.shape().length != 3 || dn.dim(0) != e || dn.dim(1) != cfg.dim() || dn.dim(2) != m) {
                throw new IOException("layer " + l + ": unexpected fused expert shapes " + gu + " / " + dn);
            }
            w.put(p + "mlp.experts.gate_up", quantize(a, quant, flatten(gu)));
            w.put(p + "mlp.experts.down", quantize(a, quant, flatten(dn)));
            used.add(gu.name());
            used.add(dn.name());
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
            used.add(gate.name());
            used.add(up.name());
            used.add(down.name());
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
            st = SafeTensors.open(provider.ensure(shard));
            open.put(shard, st);
        }
        return st.get(name);
    }

    /** Marks a tensor consumed; closes and releases its shard once nothing else in it is needed. */
    private void done(String name) {
        String shard = shardOf.get(name);
        int left = remaining.merge(shard, -1, Integer::sum);
        if (left <= 0) {
            SafeTensors st = open.remove(shard);
            if (st != null) {
                st.close();
            }
            try {
                provider.release(shard);
            } catch (IOException e) {
                System.err.println("warning: could not release " + shard + ": " + e.getMessage());
            }
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
            Map<String, String> m = new LinkedHashMap<>();
            wm.forEach((k, v) -> m.put(k, (String) v));
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
