package org.robomiri.jllm.gguf;

import org.robomiri.jllm.model.Architecture;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Resolves the {@link Architecture} of a GGUF file from its metadata. The keys llama.cpp writes
 * ({@code <arch>.block_count}, {@code embedding_length}, {@code attention.head_count}, {@code rope.freq_base}, ...)
 * are translated into the Hugging Face {@code config.json} keys the same model would have, and handed to the
 * parsers of {@code Families}, so a family's defaults, validation and canonical {@code Core.description} are
 * exactly those of a safetensors load of the same model.
 *
 * <p>Supported {@code general.architecture} values: {@code qwen3}, {@code qwen3moe}, {@code qwen2} and
 * {@code llama} (which is also what llama.cpp calls Mistral / Mixtral-less Llama-style models).
 */
final class GgufArchitecture {

    /** The GGUF architectures this build can run, with the tensor naming they share ({@link GgufWeights}). */
    enum Family {
        LLAMA("llama", "llama"),
        QWEN2("qwen2", "qwen2"),
        QWEN3("qwen3", "qwen3"),
        QWEN3MOE("qwen3moe", "qwen3_moe");

        final String ggufName;
        final String modelType;

        Family(String ggufName, String modelType) {
            this.ggufName = ggufName;
            this.modelType = modelType;
        }

        static Family of(String ggufName) {
            for (Family f : values()) {
                if (f.ggufName.equals(ggufName)) {
                    return f;
                }
            }
            return null;
        }
    }

    record Result(Architecture arch, Family family, Map<String, Object> config) {
    }

    private GgufArchitecture() {
    }

    static Result parse(GgufFile g) throws IOException {
        String name = g.str("general.architecture", null);
        if (name == null) {
            throw new IOException("not a model GGUF: no general.architecture");
        }
        Family family = Family.of(name);
        if (family == null) {
            List<String> ok = new ArrayList<>();
            for (Family f : Family.values()) {
                ok.add(f.ggufName);
            }
            throw new IOException("unsupported GGUF architecture '" + name + "' (supported: " + String.join(", ", ok) + ")");
        }
        Map<String, Object> c = config(g, family);
        try {
            return new Result(Architecture.parse(c), family, c);
        } catch (IllegalArgumentException e) {
            throw new IOException("unsupported GGUF model (" + name + "): " + e.getMessage(), e);
        }
    }

    private static long need(GgufFile g, String key) throws IOException {
        if (!(g.get(key) instanceof Number n)) {
            throw new IOException("GGUF metadata has no " + key);
        }
        if (n.longValue() <= 0) {
            throw new IOException("GGUF metadata " + key + " must be positive, got " + n);
        }
        return n.longValue();
    }

    /** The Hugging Face {@code config.json} the file's metadata describes. */
    static Map<String, Object> config(GgufFile g, Family family) throws IOException {
        String p = family.ggufName + ".";
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("model_type", family.modelType);
        long dim = need(g, p + "embedding_length");
        long heads = scalar(g, p + "attention.head_count");
        c.put("hidden_size", dim);
        c.put("num_hidden_layers", need(g, p + "block_count"));
        c.put("num_attention_heads", heads);
        c.put("num_key_value_heads", g.get(p + "attention.head_count_kv") == null ? heads : scalar(g, p + "attention.head_count_kv"));
        if (g.get(p + "attention.key_length") instanceof Number kl) {
            c.put("head_dim", kl.longValue());
        }
        if (g.get(p + "feed_forward_length") instanceof Number ff) {
            c.put("intermediate_size", ff.longValue());
        }
        GgufFile.Tensor embed = g.tensors().get("token_embd.weight");
        if (embed == null) {
            throw new IOException("GGUF has no token_embd.weight");
        }
        c.put("vocab_size", embed.rows());
        c.put("rope_theta", g.dbl(p + "rope.freq_base", 10000.0));
        c.put("rms_norm_eps", g.dbl(p + "attention.layer_norm_rms_epsilon", 1e-5));
        c.put("max_position_embeddings", g.num(p + "context_length", 4096));
        c.put("tie_word_embeddings", !g.hasTensor("output.weight"));
        if (g.get("tokenizer.ggml.bos_token_id") instanceof Number b) {
            c.put("bos_token_id", b.longValue());
        }
        List<Object> eos = new ArrayList<>();
        for (String k : new String[]{"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"}) {
            if (g.get(k) instanceof Number e && !eos.contains(e.longValue())) {
                eos.add(e.longValue());
            }
        }
        c.put("eos_token_id", eos);
        c.put("hidden_act", "silu");

        long headDim = c.get("head_dim") instanceof Long h ? h : dim / heads;
        if (g.get(p + "rope.dimension_count") instanceof Number rd && rd.longValue() > 0 && rd.longValue() < headDim) {
            c.put("partial_rotary_factor", (double) rd.longValue() / headDim);
        }

        if (family == Family.LLAMA) {
            // llama.cpp's converter permutes the q and k rows of the llama architecture (and only of it) to the
            // interleaved RoPE layout that llama.cpp applies; qwen2 / qwen3 weights keep the Hugging Face layout
            c.put(Architecture.GGUF_INTERLEAVED_ROPE, true);
            long window = g.num(p + "attention.sliding_window", 0);
            if (window > 0) {
                c.put("model_type", "mistral");
                c.put("sliding_window", window);
            } else if (g.str("general.name", "").toLowerCase().contains("mistral")) {
                c.put("model_type", "mistral");
            }
        }
        if (family == Family.QWEN3MOE) {
            c.put("num_experts", need(g, p + "expert_count"));
            c.put("num_experts_per_tok", need(g, p + "expert_used_count"));
            c.put("moe_intermediate_size", need(g, p + "expert_feed_forward_length"));
            c.put("norm_topk_prob", true); // llama.cpp always normalizes the selected expert weights for qwen3moe
        }
        Map<String, Object> scaling = ropeScaling(g, p, family, headDim);
        if (scaling != null) {
            c.put("rope_scaling", scaling);
        }
        return c;
    }

    /** An attention head count must be one number for the whole model (some architectures store one per layer). */
    private static long scalar(GgufFile g, String key) throws IOException {
        if (g.get(key) instanceof long[]) {
            throw new IOException("GGUF metadata " + key + " is a per-layer array, which is not supported");
        }
        return need(g, key);
    }

    private static Map<String, Object> ropeScaling(GgufFile g, String p, Family family, long headDim) throws IOException {
        String type = g.str(p + "rope.scaling.type", "none");
        Map<String, Object> rs = new LinkedHashMap<>();
        switch (type) {
            case "", "none" -> {
            }
            case "linear" -> {
                rs.put("rope_type", "linear");
                rs.put("factor", g.dbl(p + "rope.scaling.factor", 1.0));
            }
            default -> throw new IOException("unsupported rope scaling type '" + type + "' (" + p + "rope.scaling.type);"
                    + " supported: none, linear, and llama3 via rope_freqs.weight");
        }
        if (g.hasTensor("rope_freqs.weight")) {
            // llama.cpp stores Llama 3.x's scaled frequencies as per-pair divisors
            GgufFile.Tensor t = g.tensor("rope_freqs.weight");
            if (t.type() != GgmlType.F32 || t.dims().length != 1) {
                throw new IOException("rope_freqs.weight must be an F32 vector, got " + t);
            }
            double[] f = new double[(int) t.cols()];
            MemorySegment seg = g.data(t);
            float[] row = new float[f.length];
            GgmlCodec.dequantizeRow(GgmlType.F32, seg, 0, f.length, row, 0);
            double max = 1;
            for (int i = 0; i < f.length; i++) {
                f[i] = row[i];
                max = Math.max(max, f[i]);
            }
            long ctx = g.num(p + "context_length", 8192);
            Map<String, Object> out = new LinkedHashMap<>();
            if (matchesLlama3(f, max, g.dbl(p + "rope.freq_base", 10000.0), headDim)) {
                out.put("rope_type", "llama3");
                out.put("factor", max);
                out.put("low_freq_factor", 1.0);
                out.put("high_freq_factor", 4.0);
                out.put("original_max_position_embeddings", 8192L);
            } else {
                // any other table of divisors: the longrope form with one table for every length
                List<Object> table = new ArrayList<>(f.length);
                for (double v : f) {
                    table.add(v);
                }
                out.put("rope_type", "longrope");
                out.put("short_factor", table);
                out.put("long_factor", table);
                out.put("factor", 1.0);
                out.put("attention_factor", 1.0);
                out.put("original_max_position_embeddings", Math.max(ctx, 2));
            }
            return out;
        }
        return rs.isEmpty() ? null : rs;
    }

    /** True if {@code f} is the Llama 3.x table (low = 1, high = 4, original context 8192) for the largest divisor {@code factor}. */
    private static boolean matchesLlama3(double[] f, double factor, double base, long headDim) {
        if (factor <= 1 || f.length != headDim / 2) {
            return false;
        }
        double low = 8192.0 / 1.0;
        double high = 8192.0 / 4.0;
        for (int i = 0; i < f.length; i++) {
            double freq = 1.0 / Math.pow(base, (2.0 * i) / headDim);
            double wavelen = 2 * Math.PI / freq;
            double want;
            if (wavelen < high) {
                want = 1;
            } else if (wavelen > low) {
                want = factor;
            } else {
                double smooth = (8192.0 / wavelen - 1.0) / (4.0 - 1.0);
                want = 1 / ((1 - smooth) / factor + smooth);
            }
            if (Math.abs(f[i] - want) > 1e-4 * want) {
                return false;
            }
        }
        return true;
    }
}
