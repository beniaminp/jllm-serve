package org.robomiri.jllm;

import org.robomiri.jllm.gguf.GgmlType;
import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.safetensors.SafeTensors;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A re-implementation, for tests, of what llama.cpp's {@code convert_hf_to_gguf.py} (tag b11669: {@code conversion/*.py},
 * {@code gguf-py/gguf/constants.py}, {@code tensor_mapping.py}) writes for the model families jllm runs, applied to the
 * tiny random Hugging Face checkpoints of {@code scripts/make-tiny-model.py}: the same metadata keys, tensor names,
 * dimension order, q/k permutation (llama), {@code +1} on Gemma's norms, {@code -exp(A_log)} for Mamba, the split of
 * DeepSeek's {@code kv_b_proj} into {@code attn_k_b} / {@code attn_v_b}, merged experts, {@code rope_freqs} /
 * {@code rope_factors_*} tensors and which tensors are forced to F32. Where llama.cpp's converter cannot express a
 * model (a GLM {@code head_dim} different from {@code hidden / heads}, Qwen3-MoE with dense layers, DeepSeek with
 * {@code mscale != mscale_all_dim}) this class throws instead of writing something the real converter would not.
 *
 * <p>The file is a plain GGUF: a synthetic byte-level BPE vocabulary of the model's vocabulary size (256 byte tokens,
 * a few merges, the special tokens of the config) lets llama.cpp load it. {@link Outtype} chooses the tensors' format:
 * F32; Q8_0 (gguf-py's {@code quants.Q8_0}: matrices whose row length is not a multiple of 32 fall back to F16, as the
 * real converter does); or {@code DEQ_Q8_0}, F32 tensors holding exactly the values the Q8_0 file means, which jllm
 * requantizes without change and llama.cpp runs in float, so the two engines see the same numbers.
 */
final class HfToGguf {

    enum Outtype {
        F32, Q8_0, DEQ_Q8_0
    }

    /** Options of {@link #convert}: {@code legacyMla} writes DeepSeek's single {@code attn_kv_b} (GGUFs before March 2025). */
    record Options(Outtype type, boolean legacyMla, boolean fuseQkv, boolean fuseGateUpExps) {
        static Options of(Outtype t) {
            return new Options(t, false, false, false);
        }
    }

    private record T(float[] data, long[] shape) {
    }

    /** Test hook: write the Llama-style conversion under another GGUF architecture name (e.g. mistral3). */
    static String archOverride;

    private final Map<String, Object> hp;
    private final Map<String, Object> raw;
    private final Map<String, float[]> data = new HashMap<>();
    private final Map<String, long[]> shapes = new HashMap<>();
    private final Options opt;
    private final GgufWriter w = new GgufWriter();
    private final TreeMap<String, T> out = new TreeMap<>();
    private String arch;
    private String p;

    private HfToGguf(Path dir, Options opt) throws IOException {
        this.opt = opt;
        raw = Json.parseObject(Files.readString(dir.resolve("config.json")));
        Map<String, Object> merged = new LinkedHashMap<>(raw);
        if (raw.get("text_config") instanceof Map<?, ?> tc) {
            @SuppressWarnings("unchecked")
            Map<String, Object> t = (Map<String, Object>) tc;
            merged.putAll(t);
        }
        hp = merged;
        try (var files = Files.list(dir)) {
            for (Path f : (Iterable<Path>) files.filter(x -> x.toString().endsWith(".safetensors")).sorted()::iterator) {
                try (SafeTensors st = SafeTensors.open(f)) {
                    for (SafeTensors.Tensor t : st.tensors().values()) {
                        data.put(t.name(), t.toFloatArray());
                        shapes.put(t.name(), t.shape());
                    }
                }
            }
        }
    }

    /** Converts the checkpoint in {@code hfDir} to {@code out}; returns the GGUF architecture written. */
    static String convert(Path hfDir, Path out, Options opt) throws IOException {
        HfToGguf c = new HfToGguf(hfDir, opt);
        c.run();
        c.w.write(out);
        return c.arch;
    }

    static String convert(Path hfDir, Path out, Outtype t) throws IOException {
        return convert(hfDir, out, Options.of(t));
    }

    // ------------------------------------------------------------------ small helpers

    private static double num(Object o, String key, double def) {
        return o instanceof Map<?, ?> m && m.get(key) instanceof Number n ? n.doubleValue() : def;
    }

    private double hpNum(String key, double def) {
        return num(hp, key, def);
    }

    private boolean has(String key) {
        return hp.get(key) != null;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object o, String key) {
        return o instanceof Map<?, ?> m && m.get(key) instanceof Map<?, ?> r ? (Map<String, Object>) r : null;
    }

    private int layers() {
        return (int) hpNum("num_hidden_layers", 0);
    }

    private String hfArch() {
        return raw.get("architectures") instanceof List<?> l && !l.isEmpty() ? String.valueOf(l.getFirst()) : "";
    }

    private void need(boolean ok, String msg) {
        if (!ok) {
            throw new IllegalArgumentException("llama.cpp's converter cannot express this model: " + msg);
        }
    }

    // ------------------------------------------------------------------ driver

    private void run() throws IOException {
        String a = hfArch();
        switch (a) {
            case "LlamaForCausalLM", "MistralForCausalLM" -> llama();
            case "Qwen2ForCausalLM" -> qwen("qwen2", false);
            case "Qwen3ForCausalLM" -> qwen("qwen3", false);
            case "Qwen3MoeForCausalLM" -> qwen("qwen3moe", true);
            case "Gemma2ForCausalLM" -> gemma(false);
            case "Gemma3ForCausalLM", "Gemma3ForConditionalGeneration" -> gemma(true);
            case "Phi3ForCausalLM" -> phi3();
            case "GlmForCausalLM" -> chatglm();
            case "Glm4ForCausalLM" -> glm4();
            case "MambaForCausalLM" -> mamba();
            case "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM" -> deepseek2();
            default -> throw new IllegalArgumentException("no converter for " + a);
        }
        emit();
    }

    private void begin(String archName) {
        arch = archName;
        p = archName + ".";
        w.str("general.architecture", archName).str("general.name", "tiny-random"); // not "phi3": llama.cpp keys vocab tweaks on the name
    }

    // ------------------------------------------------------------------ metadata shared by TextModel.set_gguf_parameters

    /** rope_parameters as TextModel.__init__ builds them from a legacy (transformers 4) config. */
    private Map<String, Object> ropeParams() {
        Map<String, Object> rp = new LinkedHashMap<>();
        Object src = hp.get("rope_parameters") != null ? hp.get("rope_parameters") : hp.get("rope_scaling");
        if (src instanceof Map<?, ?> m) {
            m.forEach((k, v) -> rp.put(String.valueOf(k), v));
        }
        if (!rp.containsKey("full_attention") && !rp.containsKey("sliding_attention")) {
            if (hp.get("rope_local_base_freq") != null) {
                Map<String, Object> s = new LinkedHashMap<>();
                s.put("rope_theta", hp.get("rope_local_base_freq"));
                rp.put("sliding_attention", s);
            }
            if (!rp.containsKey("rope_theta") && hp.get("rope_theta") != null) {
                rp.put("rope_theta", hp.get("rope_theta"));
            }
            if (!rp.containsKey("rope_type") && rp.get("type") != null) {
                rp.put("rope_type", rp.get("type"));
            }
            if (!rp.containsKey("partial_rotary_factor") && hp.get("partial_rotary_factor") != null) {
                rp.put("partial_rotary_factor", hp.get("partial_rotary_factor"));
            }
            if (!rp.containsKey("original_max_position_embeddings") && hp.get("original_max_position_embeddings") != null) {
                rp.put("original_max_position_embeddings", hp.get("original_max_position_embeddings"));
            }
        }
        return rp;
    }

    private Map<String, Object> fullRope() {
        Map<String, Object> rp = ropeParams();
        return rp.get("full_attention") instanceof Map<?, ?> f ? castMap(f) : rp;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> castMap(Map<?, ?> m) {
        return (Map<String, Object>) m;
    }

    /** TextModel.set_gguf_parameters, without the vocabulary keys. */
    private void textParams(boolean attention) {
        w.u32(p + "block_count", layers());
        if (has("max_position_embeddings")) {
            w.u32(p + "context_length", (long) hpNum("max_position_embeddings", 0));
        }
        w.u32(p + "embedding_length", (long) hpNum("hidden_size", 0));
        if (has("intermediate_size")) {
            w.u32(p + "feed_forward_length", (long) hpNum("intermediate_size", 0));
        }
        if (attention) {
            w.u32(p + "attention.head_count", (long) hpNum("num_attention_heads", 0));
            if (has("num_key_value_heads")) {
                w.u32(p + "attention.head_count_kv", (long) hpNum("num_key_value_heads", 0));
            }
        }
        Map<String, Object> rp = fullRope();
        Object type = rp.get("rope_type");
        if (type instanceof String t && rp.get("factor") != null) {
            double factor = num(rp, "factor", 1);
            if (t.equals("linear")) {
                w.str(p + "rope.scaling.type", "linear").f32(p + "rope.scaling.factor", (float) factor);
            } else if (t.equals("yarn")) {
                w.str(p + "rope.scaling.type", "yarn").f32(p + "rope.scaling.factor", (float) factor)
                        .u32(p + "rope.scaling.original_context_length", (long) num(rp, "original_max_position_embeddings", 0));
                if (rp.get("truncate") instanceof Boolean b) {
                    w.bool(p + "rope.scaling.yarn_truncate", b);
                }
                Object af = rp.get("attention_factor") != null ? rp.get("attention_factor") : rp.get("attn_factor");
                if (af instanceof Number n) {
                    w.f32(p + "rope.scaling.yarn_attn_factor", n.floatValue());
                }
                if (rp.get("beta_fast") instanceof Number n) {
                    w.f32(p + "rope.scaling.yarn_beta_fast", n.floatValue());
                }
                if (rp.get("beta_slow") instanceof Number n) {
                    w.f32(p + "rope.scaling.yarn_beta_slow", n.floatValue());
                }
            } else if (t.equals("su") || t.equals("longrope")) {
                w.str(p + "rope.scaling.type", "longrope");
            }
        }
        if (rp.get("rope_theta") instanceof Number n) {
            w.f32(p + "rope.freq_base", n.floatValue());
        }
        if (map(ropeParams(), "sliding_attention") != null && map(ropeParams(), "sliding_attention").get("rope_theta") instanceof Number n) {
            w.f32(p + "rope.freq_base_swa", n.floatValue());
        }
        if (has("rms_norm_eps")) {
            w.f32(p + "attention.layer_norm_rms_epsilon", (float) hpNum("rms_norm_eps", 0));
        }
        if (has("num_experts") || has("num_local_experts") || has("n_routed_experts")) {
            w.u32(p + "expert_count", (long) Math.max(hpNum("num_experts", 0), Math.max(hpNum("num_local_experts", 0), hpNum("n_routed_experts", 0))));
        }
        if (has("num_experts_per_tok")) {
            w.u32(p + "expert_used_count", (long) hpNum("num_experts_per_tok", 0));
        }
        if (has("n_group")) {
            w.u32(p + "expert_group_count", (long) hpNum("n_group", 0));
        }
        if (has("topk_group")) {
            w.u32(p + "expert_group_used_count", (long) hpNum("topk_group", 0));
        }
        if (hp.get("scoring_func") instanceof String sf) {
            w.u32(p + "expert_gating_func", sf.equals("sigmoid") ? 2 : 1);
        }
        if (has("head_dim")) {
            w.u32(p + "attention.key_length", (long) hpNum("head_dim", 0));
            w.u32(p + "attention.value_length", (long) hpNum("head_dim", 0));
        }
        w.u32("general.file_type", 1);
    }

    private int headDim() {
        return has("head_dim") ? (int) hpNum("head_dim", 0) : (int) hpNum("hidden_size", 0) / (int) hpNum("num_attention_heads", 1);
    }

    // ------------------------------------------------------------------ families

    private static final Pattern LAYER = Pattern.compile("^model\\.layers\\.(\\d+)\\.(.+)$");

    private void llama() {
        begin(archOverride != null ? archOverride : "llama");
        textParams(true);
        w.u32(p + "vocab_size", (long) hpNum("vocab_size", 0));
        w.u32(p + "rope.dimension_count", headDim());
        int heads = (int) hpNum("num_attention_heads", 0);
        int kv = (int) hpNum("num_key_value_heads", heads);
        // Llama 3.x: the converter writes the scaled frequencies as an extra tensor (generate_extra_tensors)
        Map<String, Object> rp = fullRope();
        if ("llama3".equalsIgnoreCase(String.valueOf(rp.get("rope_type")))) {
            double base = num(rp, "rope_theta", 10000);
            int dim = headDim();
            double factor = num(rp, "factor", 8), low = num(rp, "low_freq_factor", 1), high = num(rp, "high_freq_factor", 4);
            double old = num(rp, "original_max_position_embeddings", 8192);
            float[] f = new float[dim / 2];
            for (int i = 0; i < f.length; i++) {
                double freq = 1.0 / Math.pow(base, (2.0 * i) / dim);
                double wavelen = 2 * Math.PI / freq;
                if (wavelen < old / high) {
                    f[i] = 1;
                } else if (wavelen > old / low) {
                    f[i] = (float) factor;
                } else {
                    double smooth = (old / wavelen - low) / (high - low);
                    f[i] = (float) (1 / ((1 - smooth) / factor + smooth));
                }
            }
            put("rope_freqs.weight", f, f.length);
        }
        for (String name : sortedNames()) {
            Matcher m = LAYER.matcher(name);
            if (name.equals("model.embed_tokens.weight")) {
                put("token_embd.weight", name);
            } else if (name.equals("model.norm.weight")) {
                put("output_norm.weight", name);
            } else if (name.equals("lm_head.weight")) {
                if (!bool("tie_word_embeddings")) {
                    put("output.weight", name);
                }
            } else if (m.matches()) {
                String b = "blk." + m.group(1) + ".";
                String s = m.group(2);
                float[] d = data.get(name);
                long[] sh = shapes.get(name);
                if (s.equals("self_attn.q_proj.weight") || s.equals("self_attn.q_proj.bias")) {
                    d = GgufModelTest.permute(d, (int) sh[0], sh.length > 1 ? (int) sh[1] : 1, heads);
                } else if (s.equals("self_attn.k_proj.weight") || s.equals("self_attn.k_proj.bias")) {
                    d = GgufModelTest.permute(d, (int) sh[0], sh.length > 1 ? (int) sh[1] : 1, kv);
                }
                put(b + attnOrMlp(s), d, sh);
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        vocab();
    }

    /** The llama.cpp name (after {@code blk.N.}) of a layer tensor shared by the Llama-style families. */
    private String attnOrMlp(String s) {
        return switch (s) {
            case "input_layernorm.weight" -> "attn_norm.weight";
            case "post_attention_layernorm.weight" -> "ffn_norm.weight";
            case "self_attn.q_norm.weight" -> "attn_q_norm.weight";
            case "self_attn.k_norm.weight" -> "attn_k_norm.weight";
            case "self_attn.q_proj.weight" -> "attn_q.weight";
            case "self_attn.k_proj.weight" -> "attn_k.weight";
            case "self_attn.v_proj.weight" -> "attn_v.weight";
            case "self_attn.q_proj.bias" -> "attn_q.bias";
            case "self_attn.k_proj.bias" -> "attn_k.bias";
            case "self_attn.v_proj.bias" -> "attn_v.bias";
            case "self_attn.o_proj.weight" -> "attn_output.weight";
            case "self_attn.o_proj.bias" -> "attn_output.bias";
            case "self_attn.qkv_proj.weight" -> "attn_qkv.weight";
            case "mlp.gate_proj.weight" -> "ffn_gate.weight";
            case "mlp.up_proj.weight" -> "ffn_up.weight";
            case "mlp.gate_up_proj.weight" -> "ffn_up.weight";
            case "mlp.down_proj.weight" -> "ffn_down.weight";
            case "mlp.gate.weight" -> "ffn_gate_inp.weight";
            default -> throw new IllegalArgumentException("unmapped layer tensor " + s);
        };
    }

    private boolean bool(String key) {
        return hp.get(key) instanceof Boolean b && b;
    }

    /** Qwen2 / Qwen3 / Qwen3-MoE (conversion/qwen.py): HF layout, no permutation. */
    private void qwen(String archName, boolean moe) {
        begin(archName);
        textParams(true);
        if (moe) {
            need(hpNum("decoder_sparse_step", 1) == 1 && !(hp.get("mlp_only_layers") instanceof List<?> l && !l.isEmpty()),
                    "llama.cpp's qwen3moe has an expert block in every layer (decoder_sparse_step 1, no mlp_only_layers)");
            w.u32(p + "expert_feed_forward_length", (long) hpNum("moe_intermediate_size", 0));
        }
        for (String name : sortedNames()) {
            Matcher m = LAYER.matcher(name);
            if (name.equals("model.embed_tokens.weight")) {
                put("token_embd.weight", name);
            } else if (name.equals("model.norm.weight")) {
                put("output_norm.weight", name);
            } else if (name.equals("lm_head.weight")) {
                if (!bool("tie_word_embeddings")) {
                    put("output.weight", name);
                }
            } else if (m.matches()) {
                String s = m.group(2);
                if (s.startsWith("mlp.experts.")) {
                    continue; // merged below
                }
                put("blk." + m.group(1) + "." + attnOrMlp(s), name);
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        if (moe) {
            mergeExperts(layers());
        }
        vocab();
    }

    /** {@code mlp.experts.N.{gate,up,down}_proj.weight} to the stacked {@code ffn_*_exps.weight} (Qwen2MoeModel.modify_tensors). */
    private void mergeExperts(int blocks) {
        int e = (int) Math.max(hpNum("num_experts", 0), Math.max(hpNum("num_local_experts", 0), hpNum("n_routed_experts", 0)));
        for (int l = 0; l < blocks; l++) {
            if (!data.containsKey("model.layers." + l + ".mlp.experts.0.gate_proj.weight")) {
                continue;
            }
            for (String[] part : new String[][]{{"down_proj", "ffn_down_exps"}, {"gate_proj", "ffn_gate_exps"}, {"up_proj", "ffn_up_exps"}}) {
                long[] sh = shapes.get("model.layers." + l + ".mlp.experts.0." + part[0] + ".weight");
                float[] all = new float[e * (int) (sh[0] * sh[1])];
                for (int x = 0; x < e; x++) {
                    float[] one = data.get("model.layers." + l + ".mlp.experts." + x + "." + part[0] + ".weight");
                    System.arraycopy(one, 0, all, x * one.length, one.length);
                }
                put("blk." + l + "." + part[1] + ".weight", all, e, sh[0], sh[1]);
            }
        }
        if (opt.fuseGateUpExps()) {
            for (int l = 0; l < blocks; l++) {
                T g = out.remove("blk." + l + ".ffn_gate_exps.weight");
                T u = out.remove("blk." + l + ".ffn_up_exps.weight");
                if (g == null) {
                    continue;
                }
                int per = g.data.length / (int) g.shape[0];
                float[] fused = new float[2 * g.data.length];
                for (int x = 0; x < g.shape[0]; x++) {
                    System.arraycopy(g.data, x * per, fused, 2 * x * per, per);
                    System.arraycopy(u.data, x * per, fused, (2 * x + 1) * per, per);
                }
                out.put("blk." + l + ".ffn_gate_up_exps.weight", new T(fused, new long[]{g.shape[0], 2 * g.shape[1], g.shape[2]}));
            }
        }
    }

    /** Gemma 2 / Gemma 3 (conversion/gemma.py): {@code w + 1} on every norm, no output tensor, embedding rows cut to the vocabulary. */
    private void gemma(boolean g3) {
        begin(g3 ? "gemma3" : "gemma2");
        w.u32(p + "context_length", (long) hpNum("max_position_embeddings", g3 ? 131072 : 0));
        w.u32(p + "embedding_length", (long) hpNum("hidden_size", 0));
        w.u32(p + "block_count", layers());
        w.u32(p + "feed_forward_length", (long) hpNum("intermediate_size", 0));
        w.u32(p + "attention.head_count", (long) hpNum("num_attention_heads", g3 ? 8 : 0));
        w.u32(p + "attention.head_count_kv", (long) hpNum("num_key_value_heads", g3 ? 4 : hpNum("num_attention_heads", 0)));
        w.f32(p + "attention.layer_norm_rms_epsilon", (float) hpNum("rms_norm_eps", 1e-6));
        w.u32(p + "attention.key_length", (long) hpNum("head_dim", g3 ? 256 : 0));
        w.u32(p + "attention.value_length", (long) hpNum("head_dim", g3 ? 256 : 0));
        w.u32("general.file_type", 1);
        if (g3) {
            Map<String, Object> rp = ropeParams();
            Map<String, Object> full = fullRope();
            w.f32(p + "rope.freq_base", (float) num(full, "rope_theta", 1_000_000.0));
            if (hp.get("attn_logit_softcapping") != null) {
                throw new IllegalArgumentException("attn_logit_softcapping is removed in Gemma 3");
            }
            if (hp.get("final_logit_softcapping") instanceof Number n && n.doubleValue() != 0) {
                w.f32(p + "final_logit_softcapping", n.floatValue());
            }
            if (hpNum("sliding_window_pattern", -1) != 1) {
                w.u32(p + "attention.sliding_window", (long) hpNum("sliding_window", 0));
            }
            // TextModel.set_gguf_parameters: linear / yarn scaling of the full-attention rope, the local base
            if (full.get("rope_type") instanceof String t && t.equals("linear") && full.get("factor") != null) {
                w.str(p + "rope.scaling.type", "linear").f32(p + "rope.scaling.factor", (float) num(full, "factor", 1));
            }
            if (map(rp, "sliding_attention") != null && map(rp, "sliding_attention").get("rope_theta") instanceof Number n) {
                w.f32(p + "rope.freq_base_swa", n.floatValue());
            }
        } else {
            w.f32(p + "attn_logit_softcapping", (float) hpNum("attn_logit_softcapping", 0));
            w.f32(p + "final_logit_softcapping", (float) hpNum("final_logit_softcapping", 0));
            w.u32(p + "attention.sliding_window", (long) hpNum("sliding_window", 0));
        }
        String prefix = hp.containsKey("text_config") || raw.containsKey("text_config") ? "language_model." : "";
        for (String name0 : sortedNames()) {
            String name = name0;
            if (name.startsWith("language_model.")) {
                name = name.substring("language_model.".length());
            } else if (name.startsWith("vision_tower.") || name.startsWith("multi_modal_projector.")) {
                continue;
            }
            if (name.equals("lm_head.weight")) {
                continue; // tied: llama.cpp skips it
            }
            Matcher m = LAYER.matcher(name);
            float[] d = data.get(name0);
            long[] sh = shapes.get(name0);
            if (name.endsWith("norm.weight")) {
                d = d.clone();
                for (int i = 0; i < d.length; i++) {
                    d[i] += 1f;
                }
            }
            if (name.equals("model.embed_tokens.weight")) {
                put("token_embd.weight", d, sh);
            } else if (name.equals("model.norm.weight")) {
                put("output_norm.weight", d, sh);
            } else if (m.matches()) {
                String s = m.group(2);
                String g = switch (s) {
                    case "post_attention_layernorm.weight" -> "post_attention_norm.weight";
                    case "pre_feedforward_layernorm.weight" -> "ffn_norm.weight";
                    case "post_feedforward_layernorm.weight" -> "post_ffw_norm.weight";
                    default -> attnOrMlp(s);
                };
                put("blk." + m.group(1) + "." + g, d, sh);
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name0 + prefix);
            }
        }
        vocab();
    }

    /** Phi-3 (conversion/phi.py): fused attn_qkv and ffn_up (gate|up), LongRoPE factors as extra tensors. */
    private void phi3() {
        begin("phi3");
        int embd = (int) hpNum("hidden_size", 0);
        int heads = (int) hpNum("num_attention_heads", 0);
        Map<String, Object> rp = fullRope();
        int orig = (int) num(rp, "original_max_position_embeddings", 0);
        int maxPos = (int) hpNum("max_position_embeddings", 0);
        double rot = num(rp, "partial_rotary_factor", 1.0);
        int ropeDims = (int) (rot * embd) / heads;
        w.u32(p + "context_length", maxPos);
        w.u32(p + "rope.scaling.original_context_length", orig);
        w.u32(p + "embedding_length", embd);
        w.u32(p + "feed_forward_length", (long) hpNum("intermediate_size", 0));
        w.u32(p + "block_count", layers());
        w.u32(p + "attention.head_count", heads);
        w.u32(p + "attention.head_count_kv", (long) hpNum("num_key_value_heads", heads));
        w.f32(p + "attention.layer_norm_rms_epsilon", (float) hpNum("rms_norm_eps", 1e-5));
        w.u32(p + "rope.dimension_count", ropeDims);
        w.f32(p + "rope.freq_base", (float) num(rp, "rope_theta", 10000));
        w.u32("general.file_type", 1);
        w.u32(p + "attention.sliding_window", has("sliding_window") ? (long) hpNum("sliding_window", 0) : 0);
        if (rp.get("long_factor") instanceof List<?> lf && !lf.isEmpty()) {
            double scale = (double) maxPos / orig;
            String type = String.valueOf(rp.get("rope_type")).toLowerCase();
            double attn = scale > 1.0 ? (type.equals("yarn") ? 0.1 * Math.log(scale) + 1.0
                    : Math.sqrt(1 + Math.log(scale) / Math.log(orig))) : 1.0;
            w.f32(p + "rope.scaling.attn_factor", (float) attn);
            need(lf.size() == ropeDims / 2, "rope factor count");
            put("rope_factors_long.weight", floats(lf), lf.size());
            put("rope_factors_short.weight", floats((List<?>) rp.get("short_factor")), lf.size());
        }
        for (String name : sortedNames()) {
            Matcher m = LAYER.matcher(name);
            if (name.equals("model.embed_tokens.weight")) {
                put("token_embd.weight", name);
            } else if (name.equals("model.norm.weight")) {
                put("output_norm.weight", name);
            } else if (name.equals("lm_head.weight")) {
                if (!bool("tie_word_embeddings")) {
                    put("output.weight", name);
                }
            } else if (m.matches()) {
                put("blk." + m.group(1) + "." + attnOrMlp(m.group(2)), name);
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        vocab();
    }

    private static float[] floats(List<?> l) {
        float[] f = new float[l.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = ((Number) l.get(i)).floatValue();
        }
        return f;
    }

    /** ChatGLM, the GGUF architecture of {@code GlmForCausalLM} (GLM-4-9b-chat-hf, GLM-Edge): conversion/chatglm.py. */
    private void chatglm() {
        begin("chatglm");
        int embd = (int) hpNum("hidden_size", 0);
        int heads = (int) hpNum("num_attention_heads", 0);
        need(!has("head_dim") || (int) hpNum("head_dim", 0) == embd / heads,
                "chatglm has no key_length: head_dim must be hidden_size / num_attention_heads");
        w.u32(p + "context_length", has("seq_length") ? (long) hpNum("seq_length", 0) : embd);
        w.u32(p + "embedding_length", embd);
        w.u32(p + "feed_forward_length", (long) hpNum("intermediate_size", 4L * embd));
        w.u32(p + "block_count", layers());
        w.u32(p + "attention.head_count", heads);
        w.u32(p + "attention.head_count_kv", (long) hpNum("num_key_value_heads", heads));
        w.f32(p + "attention.layer_norm_rms_epsilon", (float) hpNum("layernorm_epsilon", 1e-5));
        w.u32("general.file_type", 1);
        double prf = num(ropeParams(), "partial_rotary_factor", 0.5);
        w.u32(p + "rope.dimension_count", (int) ((embd / heads) * prf));
        w.bool("tokenizer.ggml.add_bos_token", false);
        w.f32(p + "rope.freq_base", (float) (10000.0 * (has("rope_ratio") ? hpNum("rope_ratio", 1) : 1)));
        for (String name : sortedNames()) {
            Matcher m = LAYER.matcher(name);
            if (name.equals("model.embed_tokens.weight")) {
                put("token_embd.weight", name);
            } else if (name.equals("model.norm.weight")) {
                put("output_norm.weight", name);
            } else if (name.equals("lm_head.weight")) {
                if (!bool("tie_word_embeddings")) {
                    put("output.weight", name);
                }
            } else if (m.matches()) {
                put("blk." + m.group(1) + "." + attnOrMlp(m.group(2)), name);
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        vocab();
    }

    /** GLM-4-0414 ({@code Glm4ForCausalLM}): conversion/glm.py, the sandwich norms of {@code post_self_attn_layernorm} / {@code post_mlp_layernorm}. */
    private void glm4() {
        begin("glm4");
        textParams(true);
        w.u32(p + "rope.dimension_count", (int) (headDim() * num(ropeParams(), "partial_rotary_factor", 0.5)));
        for (String name : sortedNames()) {
            Matcher m = LAYER.matcher(name);
            if (name.equals("model.embed_tokens.weight")) {
                put("token_embd.weight", name);
            } else if (name.equals("model.norm.weight")) {
                put("output_norm.weight", name);
            } else if (name.equals("lm_head.weight")) {
                if (!bool("tie_word_embeddings")) {
                    put("output.weight", name);
                }
            } else if (m.matches()) {
                String s = m.group(2);
                String g = switch (s) {
                    case "post_self_attn_layernorm.weight" -> "post_attention_norm.weight";
                    case "post_mlp_layernorm.weight" -> "post_ffw_norm.weight";
                    default -> attnOrMlp(s);
                };
                put("blk." + m.group(1) + "." + g, name);
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        vocab();
    }

    /** Mamba v1 (conversion/mamba.py): ssm_* tensors, {@code A = -exp(A_log)}, the conv kernel squeezed. */
    private void mamba() {
        begin("mamba");
        int dModel = (int) hpNum("hidden_size", 0);
        int dInner = has("intermediate_size") ? (int) hpNum("intermediate_size", 0) : 2 * dModel;
        need(dInner == 2 * dModel, "llama.cpp supports an expansion factor of 2 only");
        int dConv = has("conv_kernel") ? (int) hpNum("conv_kernel", 4) : 4;
        int dState = has("state_size") ? (int) hpNum("state_size", 16) : 16;
        int dtRank = hp.get("time_step_rank") instanceof Number n ? n.intValue() : -(dModel / -16);
        w.u32(p + "context_length", 1 << 20);
        w.u32(p + "embedding_length", dModel);
        w.u32(p + "feed_forward_length", 0);
        w.u32(p + "attention.head_count", 0);
        w.u32(p + "block_count", layers());
        w.u32(p + "ssm.conv_kernel", dConv);
        w.u32(p + "ssm.inner_size", dInner);
        w.u32(p + "ssm.state_size", dState);
        w.u32(p + "ssm.time_step_rank", dtRank);
        w.f32(p + "attention.layer_norm_rms_epsilon", (float) hpNum("layer_norm_epsilon", 1e-5));
        w.bool(p + "ssm.dt_b_c_rms", false);
        w.u32("general.file_type", 1);
        float[] embd = data.get("backbone.embeddings.weight");
        for (String name : sortedNames()) {
            float[] d = data.get(name);
            long[] sh = shapes.get(name);
            if (name.equals("backbone.embeddings.weight")) {
                put("token_embd.weight", d, sh);
            } else if (name.equals("backbone.norm_f.weight")) {
                put("output_norm.weight", d, sh);
            } else if (name.equals("lm_head.weight")) {
                if (!java.util.Arrays.equals(d, embd)) {
                    put("output.weight", d, sh);
                }
            } else {
                Matcher m = Pattern.compile("^backbone\\.layers\\.(\\d+)\\.(.+)$").matcher(name);
                if (!m.matches()) {
                    throw new IllegalArgumentException("unmapped tensor " + name);
                }
                String b = "blk." + m.group(1) + ".";
                switch (m.group(2)) {
                    case "norm.weight" -> put(b + "attn_norm.weight", d, sh);
                    case "mixer.in_proj.weight" -> put(b + "ssm_in.weight", d, sh);
                    case "mixer.conv1d.weight" -> put(b + "ssm_conv1d.weight", d, sh[0], sh[2]); // [d_inner, 1, d_conv] squeezed
                    case "mixer.conv1d.bias" -> put(b + "ssm_conv1d.bias", d, sh);
                    case "mixer.x_proj.weight" -> put(b + "ssm_x.weight", d, sh);
                    case "mixer.dt_proj.weight" -> put(b + "ssm_dt.weight", d, sh);
                    case "mixer.dt_proj.bias" -> put(b + "ssm_dt.bias", d, sh);
                    case "mixer.A_log" -> {
                        float[] a = new float[d.length];
                        for (int i = 0; i < a.length; i++) {
                            a[i] = -(float) Math.exp(d[i]);
                        }
                        put(b + "ssm_a", a, sh);
                    }
                    case "mixer.D" -> put(b + "ssm_d", d, sh);
                    case "mixer.out_proj.weight" -> put(b + "ssm_out.weight", d, sh);
                    default -> throw new IllegalArgumentException("unmapped tensor " + name);
                }
            }
        }
        vocab();
    }

    /** DeepSeek-V2 / V3 (conversion/deepseek.py DeepseekV2Model): MLA as MQA with a split kv_b, shared experts, MTP layers dropped. */
    private void deepseek2() {
        begin("deepseek2");
        int kvl = (int) hpNum("kv_lora_rank", 512);
        int rope = (int) hpNum("qk_rope_head_dim", 0);
        int nope = (int) hpNum("qk_nope_head_dim", 0);
        int vDim = (int) hpNum("v_head_dim", 0);
        int heads = (int) hpNum("num_attention_heads", 0);
        Map<String, Object> rs = fullRope();
        if (rs.get("rope_type") != null && "yarn".equals(rs.get("rope_type"))) {
            need(!(rs.get("mscale") instanceof Number ms) || !(rs.get("mscale_all_dim") instanceof Number ma)
                    || ms.doubleValue() == ma.doubleValue(), "the GGUF keeps only mscale_all_dim, so mscale must equal it");
        }
        Map<String, Object> saved = new LinkedHashMap<>(hp);
        hp.put("num_key_value_heads", opt.legacyMla() ? (Object) (long) heads : (Object) 1L);
        hp.putIfAbsent("rms_norm_eps", 1e-6);
        // set_gguf_parameters of DeepseekV2Model, after TextModel's
        textParams(true);
        hp.clear();
        hp.putAll(saved);
        boolean hasMoe = hp.get("n_routed_experts") != null;
        w.u32(p + "leading_dense_block_count", hp.get("first_k_dense_replace") instanceof Number n ? n.longValue() : (hasMoe ? 0 : layers()));
        w.u32(p + "vocab_size", (long) hpNum("vocab_size", 0));
        if (hp.get("q_lora_rank") instanceof Number q) {
            w.u32(p + "attention.q_lora_rank", q.longValue());
        }
        w.u32(p + "attention.kv_lora_rank", kvl);
        if (opt.legacyMla()) {
            // before llama.cpp's MLA absorption: K and V widths of the decompressed heads, an unsplit attn_kv_b
            w.u32(p + "attention.key_length", nope + rope);
            w.u32(p + "attention.value_length", vDim);
        } else {
            w.u32(p + "attention.key_length", kvl + rope);
            w.u32(p + "attention.value_length", kvl);
            w.u32(p + "attention.key_length_mla", nope + rope);
            w.u32(p + "attention.value_length_mla", vDim);
        }
        w.u32(p + "expert_feed_forward_length", (long) (has("moe_intermediate_size") ? hpNum("moe_intermediate_size", 0) : hpNum("intermediate_size", 0)));
        w.u32(p + "expert_shared_count", (long) hpNum("n_shared_experts", 0));
        if (hp.get("routed_scaling_factor") instanceof Number n) {
            w.f32(p + "expert_weights_scale", n.floatValue());
        }
        if (bool("norm_topk_prob")) {
            w.bool(p + "expert_weights_norm", true);
        }
        w.u32(p + "rope.dimension_count", rope);
        if (rs.get("mscale_all_dim") instanceof Number ma) {
            w.f32(p + "rope.scaling.yarn_log_multiplier", (float) (0.1 * ma.doubleValue()));
        }
        int blocks = layers();
        Pattern mtp = Pattern.compile("^model\\.layers\\.(\\d+)\\..*$");
        for (String name : sortedNames()) {
            Matcher mm = mtp.matcher(name);
            if (mm.matches() && Integer.parseInt(mm.group(1)) >= blocks) {
                continue; // multi-token prediction layer
            }
            Matcher m = LAYER.matcher(name);
            float[] d = data.get(name);
            long[] sh = shapes.get(name);
            if (name.equals("model.embed_tokens.weight")) {
                put("token_embd.weight", d, sh);
            } else if (name.equals("model.norm.weight")) {
                put("output_norm.weight", d, sh);
            } else if (name.equals("lm_head.weight")) {
                if (!bool("tie_word_embeddings")) {
                    put("output.weight", d, sh);
                }
            } else if (m.matches()) {
                String b = "blk." + m.group(1) + ".";
                String s = m.group(2);
                if (s.startsWith("mlp.experts.")) {
                    continue;
                }
                if (s.equals("self_attn.kv_b_proj.weight")) {
                    if (opt.legacyMla()) {
                        put(b + "attn_kv_b.weight", d, sh);
                    } else {
                        int kvlen = (int) sh[1];
                        int block = vDim + nope;
                        float[] kb = new float[heads * kvlen * nope]; // (heads, kvl, nope): k_b transposed
                        float[] vb = new float[heads * vDim * kvlen];  // (heads, v, kvl)
                        for (int h = 0; h < heads; h++) {
                            for (int r = 0; r < nope; r++) {
                                for (int j = 0; j < kvlen; j++) {
                                    kb[(h * kvlen + j) * nope + r] = d[(h * block + r) * kvlen + j];
                                }
                            }
                            for (int r = 0; r < vDim; r++) {
                                System.arraycopy(d, (h * block + nope + r) * kvlen, vb, (h * vDim + r) * kvlen, kvlen);
                            }
                        }
                        put(b + "attn_k_b.weight", kb, heads, kvlen, nope);
                        put(b + "attn_v_b.weight", vb, heads, vDim, kvlen);
                    }
                    continue;
                }
                String g = switch (s) {
                    case "self_attn.q_a_proj.weight" -> "attn_q_a.weight";
                    case "self_attn.q_a_layernorm.weight" -> "attn_q_a_norm.weight";
                    case "self_attn.q_b_proj.weight" -> "attn_q_b.weight";
                    case "self_attn.kv_a_proj_with_mqa.weight" -> "attn_kv_a_mqa.weight";
                    case "self_attn.kv_a_layernorm.weight" -> "attn_kv_a_norm.weight";
                    case "mlp.gate.e_score_correction_bias" -> "exp_probs_b.bias";
                    case "mlp.shared_experts.gate_proj.weight" -> "ffn_gate_shexp.weight";
                    case "mlp.shared_experts.up_proj.weight" -> "ffn_up_shexp.weight";
                    case "mlp.shared_experts.down_proj.weight" -> "ffn_down_shexp.weight";
                    default -> attnOrMlp(s);
                };
                put(b + g, d, sh);
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        mergeExperts(blocks);
        vocab();
    }

    // ------------------------------------------------------------------ tensors

    private List<String> sortedNames() {
        List<String> n = new ArrayList<>(data.keySet());
        java.util.Collections.sort(n);
        return n;
    }

    private void put(String ggufName, String hfName) {
        put(ggufName, data.get(hfName), shapes.get(hfName));
    }

    private void put(String ggufName, float[] d, long... numpyShape) {
        if (out.put(ggufName, new T(d, numpyShape)) != null) {
            throw new IllegalArgumentException("duplicate tensor " + ggufName);
        }
    }

    /** gguf-py {@code quants.Q8_0.quantize_blocks}: d = amax / 127 (float32), q = round-half-away(x / d), d stored as f16. */
    static byte[] q8Blocks(float[] x) {
        int nb = x.length / 32;
        ByteBuffer b = ByteBuffer.allocate(nb * 34).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < nb; i++) {
            float amax = 0;
            for (int j = 0; j < 32; j++) {
                amax = Math.max(amax, Math.abs(x[i * 32 + j]));
            }
            float d = amax / 127f;
            float id = d == 0 ? 0 : 1f / d;
            b.putShort(Float.floatToFloat16(d));
            for (int j = 0; j < 32; j++) {
                float v = x[i * 32 + j] * id;
                float a = Math.abs(v);
                float fl = (float) Math.floor(a);
                float r = fl + (float) Math.floor(2 * (a - fl));
                b.put((byte) (v < 0 ? -r : r));
            }
        }
        return b.array();
    }

    static float[] dequantQ8(byte[] blocks) {
        ByteBuffer b = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
        float[] outv = new float[blocks.length / 34 * 32];
        for (int i = 0; i < blocks.length / 34; i++) {
            float d = Float.float16ToFloat(b.getShort(i * 34));
            for (int j = 0; j < 32; j++) {
                outv[i * 32 + j] = b.get(i * 34 + 2 + j) * d;
            }
        }
        return outv;
    }

    private static byte[] f16Bytes(float[] x) {
        ByteBuffer b = ByteBuffer.allocate(x.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : x) {
            b.putShort(Float.floatToFloat16(v));
        }
        return b.array();
    }

    private static float[] f16Round(float[] x) {
        float[] r = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            r[i] = Float.float16ToFloat(Float.floatToFloat16(x[i]));
        }
        return r;
    }

    /** llama.cpp's {@code --fuse-qkv}: attn_q, attn_k and attn_v (weights, and biases when all three have one) become {@code attn_qkv}. */
    private void fuseQkv() {
        for (int l = 0; l < layers(); l++) {
            for (String suffix : new String[]{".weight", ".bias"}) {
                T q = out.get("blk." + l + ".attn_q" + suffix);
                T k = out.get("blk." + l + ".attn_k" + suffix);
                T v = out.get("blk." + l + ".attn_v" + suffix);
                if (q == null || k == null || v == null) {
                    continue;
                }
                float[] f = new float[q.data.length + k.data.length + v.data.length];
                System.arraycopy(q.data, 0, f, 0, q.data.length);
                System.arraycopy(k.data, 0, f, q.data.length, k.data.length);
                System.arraycopy(v.data, 0, f, q.data.length + k.data.length, v.data.length);
                long[] shape = q.shape.clone();
                shape[0] = q.shape[0] + k.shape[0] + v.shape[0];
                out.remove("blk." + l + ".attn_q" + suffix);
                out.remove("blk." + l + ".attn_k" + suffix);
                out.remove("blk." + l + ".attn_v" + suffix);
                out.put("blk." + l + ".attn_qkv" + suffix, new T(f, shape));
            }
        }
    }

    /** Writes the collected tensors sorted by name, in the format {@link #opt} asks and llama.cpp's rules for F32 tensors allow. */
    private void emit() {
        if (opt.fuseQkv()) {
            fuseQkv();
        }
        for (Map.Entry<String, T> e : out.entrySet()) {
            String name = e.getKey();
            T t = e.getValue();
            long[] dims = new long[t.shape.length];
            for (int i = 0; i < dims.length; i++) {
                dims[i] = t.shape[t.shape.length - 1 - i];
            }
            boolean f32Only = t.shape.length <= 1 || name.endsWith("_norm.weight") || name.endsWith("ffn_gate_inp.weight")
                    || name.endsWith("ssm_conv1d.weight") || !name.endsWith(".weight");
            long cols = t.shape[t.shape.length - 1];
            if (f32Only || opt.type() == Outtype.F32) {
                w.f32Tensor(name, dims, t.data);
            } else if (cols % 32 != 0) {
                // gguf.quants.QuantError: the converter falls back to F16
                if (opt.type() == Outtype.Q8_0) {
                    w.tensor(name, GgmlType.F16, dims, f16Bytes(t.data));
                } else {
                    w.f32Tensor(name, dims, f16Round(t.data));
                }
            } else {
                byte[] blocks = q8Blocks(t.data);
                if (opt.type() == Outtype.Q8_0) {
                    w.tensor(name, GgmlType.Q8_0, dims, blocks);
                } else {
                    w.f32Tensor(name, dims, dequantQ8(blocks));
                }
            }
        }
    }

    // ------------------------------------------------------------------ vocabulary

    private static final String[][] MERGES = {{"Ġ", "t"}, {"h", "e"}, {"Ġt", "he"}, {"i", "n"}, {"Ġ", "a"},
            {"e", "r"}, {"o", "n"}, {"Ġ", "s"}, {"a", "n"}, {"Ġ", "o"}, {"r", "e"}, {"Ġt", "o"}, {"e", "s"},
            {"Ġ", "w"}, {"i", "t"}};

    /** GPT-2's bytes_to_unicode. */
    static String[] byteChars() {
        String[] c = new String[256];
        List<Integer> bs = new ArrayList<>();
        for (int i = '!'; i <= '~'; i++) {
            bs.add(i);
        }
        for (int i = 0xA1; i <= 0xAC; i++) {
            bs.add(i);
        }
        for (int i = 0xAE; i <= 0xFF; i++) {
            bs.add(i);
        }
        int n = 0;
        for (int b = 0; b < 256; b++) {
            c[b] = String.valueOf((char) (bs.contains(b) ? b : 256 + n++));
        }
        return c;
    }

    /**
     * A synthetic byte-level BPE vocabulary of the model's size: id {@code b} is the GPT-2 character of byte {@code b}
     * (so ASCII text tokenizes to its bytes), a few merges follow, the config's special tokens replace their ids
     * (control tokens), the rest is unused filler. Enough for llama.cpp to load the model and tokenize a prompt.
     */
    private void vocab() {
        int embd = out.get("token_embd.weight").shape[0] > 0 ? (int) out.get("token_embd.weight").shape[0] : 0;
        String[] tokens = new String[embd];
        int[] types = new int[embd];
        String[] bc = byteChars();
        for (int i = 0; i < embd; i++) {
            if (i < 256) {
                tokens[i] = bc[i];
                types[i] = 1;
            } else {
                tokens[i] = "<unused_" + i + ">";
                types[i] = 5;
            }
        }
        List<String> merges = new ArrayList<>();
        int id = 256;
        for (String[] m : MERGES) {
            tokens[id] = m[0] + m[1];
            types[id] = 1;
            merges.add(m[0] + " " + m[1]);
            id++;
        }
        List<Integer> specials = new ArrayList<>();
        for (String key : new String[]{"bos_token_id", "eos_token_id", "pad_token_id"}) {
            Object v = raw.get(key) != null ? raw.get(key) : hp.get(key);
            if (v instanceof Number n) {
                specials.add(n.intValue());
            } else if (v instanceof List<?> l) {
                for (Object o : l) {
                    specials.add(((Number) o).intValue());
                }
            }
        }
        for (int s : specials) {
            if (s >= 0 && s < embd) {
                tokens[s] = "<|special_" + s + "|>";
                types[s] = 3;
            }
        }
        w.str("tokenizer.ggml.model", "gpt2").str("tokenizer.ggml.pre", "default").strings("tokenizer.ggml.tokens", tokens)
                .i32s("tokenizer.ggml.token_type", types).strings("tokenizer.ggml.merges", merges.toArray(new String[0]));
        Object bos = raw.get("bos_token_id") != null ? raw.get("bos_token_id") : hp.get("bos_token_id");
        Object eos = raw.get("eos_token_id") != null ? raw.get("eos_token_id") : hp.get("eos_token_id");
        if (bos instanceof Number n) {
            w.u32("tokenizer.ggml.bos_token_id", n.longValue());
        }
        if (eos instanceof Number n) {
            w.u32("tokenizer.ggml.eos_token_id", n.longValue());
        } else if (eos instanceof List<?> l && !l.isEmpty()) {
            w.u32("tokenizer.ggml.eos_token_id", ((Number) l.getFirst()).longValue());
        }
        if (!w.has("tokenizer.ggml.add_bos_token")) {
            w.bool("tokenizer.ggml.add_bos_token", false);
        }
        w.bool("tokenizer.ggml.add_eos_token", false);
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("usage: HfToGguf <hf-dir> <out.gguf> f32|q8_0|deq-q8_0 [--legacy-mla] [--fuse-qkv] [--fuse-gate-up-exps]");
            System.exit(2);
        }
        Outtype t = switch (args[2].toLowerCase()) {
            case "f32" -> Outtype.F32;
            case "q8_0" -> Outtype.Q8_0;
            case "deq-q8_0" -> Outtype.DEQ_Q8_0;
            default -> throw new IllegalArgumentException(args[2]);
        };
        List<String> flags = List.of(args).subList(3, args.length);
        String a = convert(Path.of(args[0]), Path.of(args[1]), new Options(t, flags.contains("--legacy-mla"),
                flags.contains("--fuse-qkv"), flags.contains("--fuse-gate-up-exps")));
        System.out.println(args[1] + ": " + a);
    }
}
