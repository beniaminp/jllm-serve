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
    /** U8 tensors (gpt-oss's MXFP4 blocks and scales), kept as bytes; their shapes are in {@link #shapes}. */
    private final Map<String, byte[]> bytes = new HashMap<>();
    /** Tensors written as they are (MXFP4 repacked from the checkpoint), whatever {@link Outtype} says. */
    private final TreeMap<String, Object[]> rawTensors = new TreeMap<>();
    private final Options opt;
    private final GgufWriter w = new GgufWriter().overwriteDuplicates();
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
                        shapes.put(t.name(), t.shape());
                        if (t.dtype() == SafeTensors.DType.U8) {
                            byte[] b = new byte[Math.toIntExact(t.numel())];
                            java.lang.foreign.MemorySegment.copy(t.data(), java.lang.foreign.ValueLayout.JAVA_BYTE, 0, b, 0, b.length);
                            bytes.put(t.name(), b);
                        } else {
                            data.put(t.name(), t.toFloatArray());
                        }
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
            case "GptOssForCausalLM" -> gptOss();
            case "Llama4ForCausalLM", "Llama4ForConditionalGeneration" -> llama4();
            case "GraniteMoeHybridForCausalLM" -> graniteHybrid();
            case "NemotronHForCausalLM" -> nemotronH();
            case "JambaForCausalLM" -> jamba();
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

    // ------------------------------------------------------------------ gpt-oss, Llama 4

    /** The layer types of a config (layer_types), or null. */
    private List<?> layerTypes() {
        return hp.get("layer_types") instanceof List<?> l ? l : null;
    }

    /** {@code [E][a][b]} as {@code [E][b][a]}. */
    private static float[] transposeEach(float[] d, int e, int a, int b) {
        float[] t = new float[d.length];
        for (int x = 0; x < e; x++) {
            for (int i = 0; i < a; i++) {
                for (int j = 0; j < b; j++) {
                    t[(x * b + j) * a + i] = d[(x * a + i) * b + j];
                }
            }
        }
        return t;
    }

    /** Rows {@code start, start + step, ...} ({@code n} of them) of each of {@code e} blocks of {@code rows x cols}. */
    private static float[] pickRows(float[] d, int e, int rows, int cols, int start, int step, int n) {
        float[] o = new float[e * n * cols];
        for (int x = 0; x < e; x++) {
            for (int r = 0; r < n; r++) {
                System.arraycopy(d, (x * rows + start + r * step) * cols, o, (x * n + r) * cols, cols);
            }
        }
        return o;
    }

    private static float[] negExp(float[] d) {
        float[] a = new float[d.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = -(float) Math.exp(d[i]);
        }
        return a;
    }

    /**
     * gpt-oss's MXFP4 experts (GptOssModel.repack_mxfp4): {@code blocks [E, R, nb, 16]} / {@code scales [E, R, nb]}, rows
     * {@code start, start + step, ...} of each expert, as ggml MXFP4 blocks (the scale byte, then byte j holding element j
     * in its low and element j + 16 in its high nibble, where the checkpoint packs elements 2i / 2i + 1 into byte i).
     */
    private void mxfp4(String gguf, String hf, int start, int step) {
        byte[] blocks = bytes.get(hf + "_blocks");
        byte[] scales = bytes.get(hf + "_scales");
        long[] sh = shapes.get(hf + "_blocks");
        int e = (int) sh[0];
        int rows = (int) sh[1];
        int nb = (int) sh[2];
        int n = rows / step;
        byte[] o = new byte[e * n * nb * 17];
        int k = 0;
        for (int x = 0; x < e; x++) {
            for (int r = 0; r < n; r++) {
                int row = start + r * step;
                for (int b = 0; b < nb; b++) {
                    int idx = (x * rows + row) * nb + b;
                    o[k++] = scales[idx];
                    for (int j = 0; j < 16; j++) {
                        int lo = nib(blocks, idx * 16, j);
                        int hi = nib(blocks, idx * 16, j + 16);
                        o[k++] = (byte) (lo | (hi << 4));
                    }
                }
            }
        }
        rawTensors.put(gguf, new Object[]{GgmlType.MXFP4, new long[]{32L * nb, n, e}, o});
    }

    private static int nib(byte[] b, int off, int k) {
        int v = b[off + (k >> 1)] & 0xFF;
        return (k & 1) == 0 ? v & 0xF : v >>> 4;
    }

    /**
     * gpt-oss (conversion/gpt_oss.py): TextModel's keys plus {@code attention.sliding_window} and the expert width; the
     * attention sinks as {@code attn_sinks.weight}, the MLP norm as {@code post_attention_norm} (the arch has no ffn_norm),
     * the router and its bias, the experts' gate and up taken from the alternating columns of {@code gate_up_proj} (float
     * checkpoints, transposed to rows) or from its alternating rows of MXFP4 blocks (repacked raw), with their biases.
     * llama.cpp derives the windowed layers from a pattern of 2 and hard-codes the SwiGLU clamp (7.0, alpha 1.702).
     */
    private void gptOss() {
        begin("gpt-oss");
        need(hpNum("swiglu_limit", 7.0) == 7.0 && hpNum("swiglu_alpha", 1.702) == 1.702,
                "llama.cpp's graph hard-codes swiglu_limit 7.0 and alpha 1.702");
        List<?> types = layerTypes();
        for (int l = 0; types != null && l < layers(); l++) {
            need(String.valueOf(types.get(l)).equals(l % 2 == 0 ? "sliding_attention" : "full_attention"),
                    "llama.cpp windows the even layers (load_swa_pattern 2)");
        }
        textParams(true);
        w.u32(p + "attention.sliding_window", (long) hpNum("sliding_window", 0));
        w.u32(p + "expert_feed_forward_length", (long) hpNum("intermediate_size", 0));
        int e = (int) hpNum("num_local_experts", 0);
        int m = (int) hpNum("intermediate_size", 0);
        int d = (int) hpNum("hidden_size", 0);
        for (String name : sortedNames()) {
            Matcher mt = LAYER.matcher(name);
            float[] v = data.get(name);
            long[] sh = shapes.get(name);
            if (name.equals("model.embed_tokens.weight")) {
                put("token_embd.weight", name);
            } else if (name.equals("model.norm.weight")) {
                put("output_norm.weight", name);
            } else if (name.equals("lm_head.weight")) {
                put("output.weight", name);
            } else if (mt.matches()) {
                String b = "blk." + mt.group(1) + ".";
                switch (mt.group(2)) {
                    case "post_attention_layernorm.weight" -> put(b + "post_attention_norm.weight", v, sh);
                    case "self_attn.sinks" -> put(b + "attn_sinks.weight", v, sh);
                    case "mlp.router.weight" -> put(b + "ffn_gate_inp.weight", v, sh);
                    case "mlp.router.bias" -> put(b + "ffn_gate_inp.bias", v, sh);
                    case "mlp.experts.gate_up_proj_bias" -> {
                        put(b + "ffn_gate_exps.bias", pickRows(v, e, 2 * m, 1, 0, 2, m), e, m);
                        put(b + "ffn_up_exps.bias", pickRows(v, e, 2 * m, 1, 1, 2, m), e, m);
                    }
                    case "mlp.experts.down_proj_bias" -> put(b + "ffn_down_exps.bias", v, sh);
                    case "mlp.experts.gate_up_proj" -> {
                        float[] t = transposeEach(v, e, d, 2 * m); // [E, 2m, dim]
                        put(b + "ffn_gate_exps.weight", pickRows(t, e, 2 * m, d, 0, 2, m), e, m, d);
                        put(b + "ffn_up_exps.weight", pickRows(t, e, 2 * m, d, 1, 2, m), e, m, d);
                    }
                    case "mlp.experts.down_proj" -> put(b + "ffn_down_exps.weight", transposeEach(v, e, m, d), e, d, m);
                    default -> put(b + attnOrMlp(mt.group(2)), v, sh);
                }
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        for (String name : new java.util.TreeSet<>(bytes.keySet())) {
            Matcher mt = LAYER.matcher(name);
            if (!mt.matches() || !name.endsWith("_blocks")) {
                continue;
            }
            String b = "blk." + mt.group(1) + ".";
            String hf = name.substring(0, name.length() - "_blocks".length());
            if (hf.endsWith("mlp.experts.gate_up_proj")) {
                mxfp4(b + "ffn_gate_exps.weight", hf, 0, 2);
                mxfp4(b + "ffn_up_exps.weight", hf, 1, 2);
            } else if (hf.endsWith("mlp.experts.down_proj")) {
                mxfp4(b + "ffn_down_exps.weight", hf, 0, 1);
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        vocab();
    }

    /** LlamaModel.generate_extra_tensors: a llama3 rope_scaling as per-pair divisors ({@code rope_freqs.weight}). */
    private void llama3Freqs() {
        Map<String, Object> rp = fullRope();
        if (!"llama3".equalsIgnoreCase(String.valueOf(rp.get("rope_type")))) {
            return;
        }
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

    /**
     * Llama 4 text (conversion/llama.py Llama4Model): no q / k permutation; {@code intermediate_size_mlp} becomes the
     * feed_forward_length and {@code intermediate_size} the expert width; {@code interleave_moe_layer_step}; the llama3
     * frequencies as {@code rope_freqs}; the experts' {@code gate_up_proj [E, dim, 2m]} transposed and split in halves.
     * llama.cpp's loader hard-codes the chunk (8192), the temperature (floor 8192, scale 0.1), the NoPE layers (every
     * 4th) and the q / k norm (unless 128 experts), so a config must agree with them to be expressible.
     */
    private void llama4() {
        begin("llama4");
        int n = layers();
        int e = (int) hpNum("num_local_experts", 0);
        need(hpNum("attention_chunk_size", 8192) == 8192 && hpNum("floor_scale", 8192) == 8192
                && hpNum("attn_scale", 0.1) == 0.1 && !(hp.get("attn_temperature_tuning") instanceof Boolean b && !b),
                "llama.cpp hard-codes attention_chunk_size 8192, floor_scale 8192, attn_scale 0.1 and the temperature tuning");
        need(!(hp.get("use_qk_norm") instanceof Boolean q) || q == (e != 128), "llama.cpp uses the q / k norm unless the model has 128 experts");
        need(hp.get("moe_layers") == null, "llama.cpp places the MoE layers by interleave_moe_layer_step");
        if (hp.get("no_rope_layers") instanceof List<?> nr) {
            for (int l = 0; l < n; l++) {
                need(((Number) nr.get(l)).intValue() == ((l + 1) % 4 != 0 ? 1 : 0), "llama.cpp's NoPE layers are every 4th");
            }
        }
        List<?> types = layerTypes();
        for (int l = 0; types != null && l < n; l++) {
            need(String.valueOf(types.get(l)).equals(l % 4 < 3 ? "chunked_attention" : "full_attention"),
                    "llama.cpp chunks three layers of four (load_swa_pattern 4)");
        }
        Map<String, Object> saved = new LinkedHashMap<>(hp);
        hp.put("intermediate_size_moe", hp.get("intermediate_size"));
        hp.put("intermediate_size", hp.get("intermediate_size_mlp"));
        textParams(true);
        w.u32(p + "vocab_size", (long) hpNum("vocab_size", 0));
        w.u32(p + "rope.dimension_count", headDim());
        w.u32(p + "interleave_moe_layer_step", (long) hpNum("interleave_moe_layer_step", 1));
        w.u32(p + "expert_feed_forward_length", (long) hpNum("intermediate_size_moe", 0));
        hp.clear();
        hp.putAll(saved);
        llama3Freqs();
        int m = (int) hpNum("intermediate_size", 0);
        int d = (int) hpNum("hidden_size", 0);
        for (String name0 : sortedNames()) {
            // Llama4ForConditionalGeneration: TextModel.filter_tensors drops the vision tower and the projector,
            // ModelBase.filter_tensors strips "language_model."
            if (name0.contains("vision_") || name0.contains("multi_modal_projector") || name0.contains("patch_embedding")) {
                continue;
            }
            String name = name0.startsWith("language_model.") ? name0.substring("language_model.".length()) : name0;
            Matcher mt = LAYER.matcher(name);
            float[] v = data.get(name0);
            long[] sh = shapes.get(name0);
            if (name.equals("model.embed_tokens.weight")) {
                put("token_embd.weight", v, sh);
            } else if (name.equals("model.norm.weight")) {
                put("output_norm.weight", v, sh);
            } else if (name.equals("lm_head.weight")) {
                if (!bool("tie_word_embeddings")) {
                    put("output.weight", v, sh);
                }
            } else if (mt.matches()) {
                String b = "blk." + mt.group(1) + ".";
                String s0 = mt.group(2);
                switch (s0) {
                    case "feed_forward.router.weight" -> put(b + "ffn_gate_inp.weight", v, sh);
                    case "feed_forward.experts.gate_up_proj" -> {
                        float[] t = transposeEach(v, e, d, 2 * m); // [E, 2m, dim]: gate rows, then up rows
                        put(b + "ffn_gate_exps.weight", pickRows(t, e, 2 * m, d, 0, 1, m), e, m, d);
                        put(b + "ffn_up_exps.weight", pickRows(t, e, 2 * m, d, m, 1, m), e, m, d);
                    }
                    case "feed_forward.experts.down_proj" -> put(b + "ffn_down_exps.weight", transposeEach(v, e, m, d), e, d, m);
                    case "feed_forward.shared_expert.gate_proj.weight" -> put(b + "ffn_gate_shexp.weight", v, sh);
                    case "feed_forward.shared_expert.up_proj.weight" -> put(b + "ffn_up_shexp.weight", v, sh);
                    case "feed_forward.shared_expert.down_proj.weight" -> put(b + "ffn_down_shexp.weight", v, sh);
                    default -> put(b + attnOrMlp(s0.replace("feed_forward.", "mlp.")), v, sh);
                }
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        vocab();
    }

    // ------------------------------------------------------------------ attention / state-space hybrids

    /**
     * A Mamba-2 mixer (Mamba2Model.modify_tensors): conv1d squeezed to {@code [conv, k]}, {@code dt_bias} as
     * {@code ssm_dt.bias}, {@code A = -exp(A_log)} and {@code D} as {@code [heads, 1]}, the gated norm as
     * {@code [groups, d_inner / groups]}. Returns false for a name that is not one of its tensors.
     */
    private boolean mamba2Tensor(String b, String s, float[] v, long[] sh, int groups) {
        switch (s) {
            case "in_proj.weight" -> put(b + "ssm_in.weight", v, sh);
            case "conv1d.weight" -> put(b + "ssm_conv1d.weight", v, sh[0], sh[2]);
            case "conv1d.bias" -> put(b + "ssm_conv1d.bias", v, sh);
            case "dt_bias" -> put(b + "ssm_dt.bias", v, sh);
            case "A_log" -> put(b + "ssm_a", negExp(v), sh[0], 1);
            case "D" -> put(b + "ssm_d", v, sh[0], 1);
            case "norm.weight" -> put(b + "ssm_norm.weight", v, groups, sh[0] / groups);
            case "out_proj.weight" -> put(b + "ssm_out.weight", v, sh);
            default -> {
                return false;
            }
        }
        return true;
    }

    /** {@code q}, {@code k} (permuted as LlamaModel does: the hybrids keep undo_permute), {@code v}, {@code o}. */
    private void permutedAttention(String b, String s, float[] v, long[] sh, String prefix) {
        int heads = (int) hpNum("num_attention_heads", 0);
        int kv = (int) hpNum("num_key_value_heads", heads);
        String n = s.substring(prefix.length());
        float[] d = switch (n) {
            case "q_proj.weight" -> GgufModelTest.permute(v, (int) sh[0], (int) sh[1], heads);
            case "k_proj.weight" -> GgufModelTest.permute(v, (int) sh[0], (int) sh[1], kv);
            default -> v;
        };
        put(b + attnOrMlp("self_attn." + n), d, sh);
    }

    /** {@code [2h, dim]} as its two halves of {@code h} rows. */
    private static float[][] halves(float[] v, int rowsEach, int cols) {
        return new float[][]{java.util.Arrays.copyOfRange(v, 0, rowsEach * cols), java.util.Arrays.copyOfRange(v, rowsEach * cols, 2 * rowsEach * cols)};
    }

    /**
     * Granite 4 (conversion/granite.py GraniteHybridModel over GraniteMoeModel / Mamba2Model): TextModel's keys (head_dim
     * dropped as GraniteModel does), the muP multipliers as {@code attention.scale} / {@code embedding_scale} /
     * {@code residual_scale} / {@code logit_scale}, the shared expert width, the Mamba-2 sizes ({@code time_step_rank} is
     * the head count), a per-layer {@code attention.head_count_kv} that is 0 on the Mamba layers, and
     * {@code rope.scaling.finetuned} false (no RoPE when there are Mamba layers, whatever {@code position_embedding_type} says).
     */
    private void graniteHybrid() {
        begin("granitehybrid");
        int n = layers();
        int dim = (int) hpNum("hidden_size", 0);
        int heads = (int) hpNum("num_attention_heads", 0);
        List<?> types = layerTypes();
        need(types != null, "layer_types");
        boolean[] attn = new boolean[n];
        boolean anySsm = false;
        for (int l = 0; l < n; l++) {
            attn[l] = "attention".equals(String.valueOf(types.get(l)));
            anySsm |= !attn[l];
        }
        need(!anySsm || !"rope".equals(hp.get("position_embedding_type")), "llama.cpp's granitehybrid has no RoPE next to Mamba layers");
        need(!has("head_dim") || (int) hpNum("head_dim", 0) == dim / heads, "GraniteModel drops head_dim: it must be hidden_size / heads");
        int groups = (int) hpNum("mamba_n_groups", 1);
        need(groups == 1, "llama.cpp normalizes the Mamba-2 output per group, transformers' Granite over all of d_inner");
        int inner = (int) (hpNum("mamba_expand", 2) * dim);
        need(inner == 2 * dim && inner == (int) (hpNum("mamba_n_heads", 0) * hpNum("mamba_d_head", 0)),
                "llama.cpp supports an expansion factor of 2 (mamba_n_heads x mamba_d_head = 2 x hidden_size)");
        Map<String, Object> saved = new LinkedHashMap<>(hp);
        hp.remove("head_dim");
        textParams(true);
        hp.clear();
        hp.putAll(saved);
        w.u32(p + "vocab_size", (long) hpNum("vocab_size", 0));
        w.u32(p + "rope.dimension_count", dim / heads);
        for (String[] k : new String[][]{{"attention_multiplier", "attention.scale"}, {"embedding_multiplier", "embedding_scale"},
                {"residual_multiplier", "residual_scale"}, {"logits_scaling", "logit_scale"}}) {
            if (hpNum(k[0], 0) != 0) {
                w.f32(p + k[1], (float) hpNum(k[0], 0));
            }
        }
        if (hpNum("shared_intermediate_size", 0) != 0) {
            w.u32(p + "expert_shared_feed_forward_length", (long) hpNum("shared_intermediate_size", 0));
        }
        w.u32(p + "ssm.conv_kernel", (long) hpNum("mamba_d_conv", 4));
        w.u32(p + "ssm.state_size", (long) hpNum("mamba_d_state", 256));
        w.u32(p + "ssm.group_count", groups);
        w.u32(p + "ssm.inner_size", inner);
        w.u32(p + "ssm.time_step_rank", (long) hpNum("mamba_n_heads", 0));
        int kvHeads = (int) hpNum("num_key_value_heads", heads);
        int[] kv = new int[n];
        for (int l = 0; l < n; l++) {
            kv[l] = attn[l] ? kvHeads : 0;
        }
        w.i32s(p + "attention.head_count_kv", kv);
        w.bool(p + "rope.scaling.finetuned", !anySsm);
        if (anySsm) {
            w.u32(p + "context_length", 1 << 20);
        }
        boolean experts = hpNum("num_local_experts", 0) > 0;
        int e = (int) hpNum("num_local_experts", 0);
        int m = (int) hpNum("intermediate_size", 0);
        int sm = (int) hpNum("shared_intermediate_size", m);
        for (String name : sortedNames()) {
            Matcher mt = LAYER.matcher(name);
            float[] v = data.get(name);
            long[] sh = shapes.get(name);
            if (name.equals("model.embed_tokens.weight")) {
                put("token_embd.weight", name);
            } else if (name.equals("model.norm.weight")) {
                put("output_norm.weight", name);
            } else if (name.equals("lm_head.weight")) {
                if (!bool("tie_word_embeddings")) {
                    put("output.weight", name);
                }
            } else if (mt.matches()) {
                String b = "blk." + mt.group(1) + ".";
                String s0 = mt.group(2);
                if (s0.startsWith("mamba.")) {
                    need(mamba2Tensor(b, s0.substring("mamba.".length()), v, sh, groups), "unmapped tensor " + name);
                } else if (s0.startsWith("self_attn.")) {
                    permutedAttention(b, s0, v, sh, "self_attn.");
                } else {
                    switch (s0) {
                        case "input_layernorm.weight" -> put(b + "attn_norm.weight", v, sh);
                        case "post_attention_layernorm.weight" -> put(b + "ffn_norm.weight", v, sh);
                        case "block_sparse_moe.router.layer.weight" -> put(b + "ffn_gate_inp.weight", v, sh);
                        case "block_sparse_moe.input_linear.weight" -> {
                            put(b + "ffn_gate_exps.weight", pickRows(v, e, 2 * m, dim, 0, 1, m), e, m, dim);
                            put(b + "ffn_up_exps.weight", pickRows(v, e, 2 * m, dim, m, 1, m), e, m, dim);
                        }
                        case "block_sparse_moe.output_linear.weight" -> put(b + "ffn_down_exps.weight", v, sh);
                        case "shared_mlp.input_linear.weight" -> {
                            float[][] gu = halves(v, sm, dim);
                            put(b + (experts ? "ffn_gate_shexp.weight" : "ffn_gate.weight"), gu[0], sm, dim);
                            put(b + (experts ? "ffn_up_shexp.weight" : "ffn_up.weight"), gu[1], sm, dim);
                        }
                        case "shared_mlp.output_linear.weight" -> put(b + (experts ? "ffn_down_shexp.weight" : "ffn_down.weight"), v, sh);
                        default -> throw new IllegalArgumentException("unmapped tensor " + name);
                    }
                }
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        vocab();
    }

    /**
     * Nemotron-H (conversion/nemotron.py NemotronHModel over GraniteHybridModel): per layer of
     * {@code hybrid_override_pattern} a Mamba-2 mixer, an attention mixer or a ReLU-squared MLP, told apart by the per-layer
     * {@code attention.head_count_kv} and {@code feed_forward_length}; {@code layer_norm_epsilon} as
     * {@code attention.layer_norm_epsilon}; {@code key_length} = head_dim. llama.cpp's scan has no {@code time_step_min}
     * clamp, so a config with one cannot be expressed.
     */
    private void nemotronH() {
        begin("nemotron_h");
        String pattern = String.valueOf(hp.get("hybrid_override_pattern"));
        int n = pattern.length();
        need(n == layers(), "hybrid_override_pattern has one character per layer");
        need(pattern.indexOf('E') < 0 && hp.get("num_experts_per_tok") == null, "Nemotron-H MoE is the nemotron_h_moe architecture");
        need(hpNum("time_step_min", 0.001) == 0, "llama.cpp does not clamp the step size at time_step_min");
        int dim = (int) hpNum("hidden_size", 0);
        int heads = (int) hpNum("num_attention_heads", 0);
        int headDim = (int) hpNum("head_dim", dim / Math.max(1, heads));
        int groups = (int) hpNum("n_groups", 8);
        int mh = (int) hpNum("mamba_num_heads", 0);
        int inner = mh * (int) hpNum("mamba_head_dim", 0);
        w.u32(p + "block_count", n);
        w.u32(p + "context_length", 1 << 20);
        w.u32(p + "embedding_length", dim);
        int ff = (int) hpNum("intermediate_size", 0);
        int[] ffs = new int[n];
        int[] kv = new int[n];
        for (int l = 0; l < n; l++) {
            ffs[l] = pattern.charAt(l) == '-' ? ff : 0;
            kv[l] = pattern.charAt(l) == '*' ? (int) hpNum("num_key_value_heads", heads) : 0;
        }
        w.i32s(p + "feed_forward_length", ffs);
        w.u32(p + "attention.head_count", heads);
        w.i32s(p + "attention.head_count_kv", kv);
        w.f32(p + "attention.layer_norm_epsilon", (float) hpNum("layer_norm_epsilon", 1e-5));
        w.u32(p + "vocab_size", (long) hpNum("vocab_size", 0));
        w.u32(p + "rope.dimension_count", dim / heads);
        w.u32(p + "ssm.conv_kernel", (long) hpNum("conv_kernel", 4));
        w.u32(p + "ssm.state_size", (long) hpNum("ssm_state_size", 128));
        w.u32(p + "ssm.group_count", groups);
        w.u32(p + "ssm.inner_size", inner);
        w.u32(p + "ssm.time_step_rank", mh);
        w.bool(p + "rope.scaling.finetuned", false);
        w.u32(p + "attention.key_length", headDim);
        w.u32(p + "attention.value_length", headDim);
        w.u32("general.file_type", 1);
        Pattern layer = Pattern.compile("^backbone\\.layers\\.(\\d+)\\.(.+)$");
        for (String name : sortedNames()) {
            Matcher mt = layer.matcher(name);
            float[] v = data.get(name);
            long[] sh = shapes.get(name);
            if (name.equals("backbone.embeddings.weight")) {
                put("token_embd.weight", name);
            } else if (name.equals("backbone.norm_f.weight")) {
                put("output_norm.weight", name);
            } else if (name.equals("lm_head.weight")) {
                if (!bool("tie_word_embeddings")) {
                    put("output.weight", name);
                }
            } else if (mt.matches()) {
                int l = Integer.parseInt(mt.group(1));
                String b = "blk." + l + ".";
                String s0 = mt.group(2);
                char kind = pattern.charAt(l);
                if (s0.equals("norm.weight")) {
                    put(b + "attn_norm.weight", v, sh);
                } else if (kind == 'M') {
                    need(mamba2Tensor(b, s0.substring("mixer.".length()), v, sh, groups), "unmapped tensor " + name);
                } else if (kind == '*') {
                    permutedAttention(b, s0, v, sh, "mixer.");
                } else {
                    switch (s0) {
                        case "mixer.up_proj.weight" -> put(b + "ffn_up.weight", v, sh);
                        case "mixer.down_proj.weight" -> put(b + "ffn_down.weight", v, sh);
                        default -> throw new IllegalArgumentException("unmapped tensor " + name);
                    }
                }
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        vocab();
    }

    /**
     * Jamba (conversion/jamba.py): its own key set (a per-layer {@code attention.head_count_kv} from the attention period
     * and offset, the Mamba-1 sizes, {@code expert_count}); the Mamba-1 tensors with the dt / B / C norms, conv squeezed,
     * {@code A = -exp(A_log)}; the experts merged into {@code ffn_*_exps}.
     */
    private void jamba() {
        begin("jamba");
        int n = layers();
        int dim = (int) hpNum("hidden_size", 0);
        int inner = (int) (hpNum("mamba_expand", 2) * dim);
        need(inner == 2 * dim, "llama.cpp supports an expansion factor of 2");
        int offset = (int) hpNum("attn_layer_offset", 4);
        int period = (int) hpNum("attn_layer_period", 8);
        int[] kv = new int[n];
        for (int l = offset; l < n; l++) {
            kv[l] = (l - offset) % period == 0 ? (int) hpNum("num_key_value_heads", 0) : 0;
        }
        w.u32(p + "block_count", n);
        w.u32(p + "context_length", (long) hpNum("max_position_embeddings", 0));
        w.u32(p + "embedding_length", dim);
        w.u32(p + "feed_forward_length", (long) hpNum("intermediate_size", 0));
        w.u32(p + "attention.head_count", (long) hpNum("num_attention_heads", 0));
        w.i32s(p + "attention.head_count_kv", kv);
        w.u32(p + "ssm.conv_kernel", (long) hpNum("mamba_d_conv", 4));
        w.u32(p + "ssm.inner_size", inner);
        w.u32(p + "ssm.state_size", (long) hpNum("mamba_d_state", 16));
        w.u32(p + "ssm.time_step_rank", has("mamba_dt_rank") ? (long) hpNum("mamba_dt_rank", 0) : -(dim / -16));
        w.f32(p + "attention.layer_norm_rms_epsilon", (float) (has("layer_norm_epsilon") ? hpNum("layer_norm_epsilon", 1e-6) : hpNum("rms_norm_eps", 1e-6)));
        w.u32(p + "expert_count", (long) Math.max(hpNum("num_local_experts", 0), hpNum("num_experts", 0)));
        w.u32(p + "expert_used_count", (long) hpNum("num_experts_per_tok", 0));
        w.u32("general.file_type", 1);
        int e = (int) Math.max(hpNum("num_local_experts", 0), hpNum("num_experts", 0));
        for (String name : sortedNames()) {
            Matcher mt = LAYER.matcher(name);
            float[] v = data.get(name);
            long[] sh = shapes.get(name);
            if (name.equals("model.embed_tokens.weight")) {
                put("token_embd.weight", name);
            } else if (name.equals("model.final_layernorm.weight")) {
                put("output_norm.weight", name);
            } else if (name.equals("lm_head.weight")) {
                if (!bool("tie_word_embeddings")) {
                    put("output.weight", name);
                }
            } else if (mt.matches()) {
                String b = "blk." + mt.group(1) + ".";
                String s0 = mt.group(2);
                if (s0.contains(".experts.")) {
                    continue; // merged below
                }
                String g = switch (s0) {
                    case "input_layernorm.weight" -> "attn_norm.weight";
                    case "pre_ff_layernorm.weight" -> "ffn_norm.weight";
                    case "feed_forward.router.weight" -> "ffn_gate_inp.weight";
                    case "feed_forward.gate_proj.weight" -> "ffn_gate.weight";
                    case "feed_forward.up_proj.weight" -> "ffn_up.weight";
                    case "feed_forward.down_proj.weight" -> "ffn_down.weight";
                    case "mamba.in_proj.weight" -> "ssm_in.weight";
                    case "mamba.conv1d.bias" -> "ssm_conv1d.bias";
                    case "mamba.x_proj.weight" -> "ssm_x.weight";
                    case "mamba.dt_proj.weight" -> "ssm_dt.weight";
                    case "mamba.dt_proj.bias" -> "ssm_dt.bias";
                    case "mamba.D" -> "ssm_d";
                    case "mamba.out_proj.weight" -> "ssm_out.weight";
                    case "mamba.dt_layernorm.weight" -> "ssm_dt_norm.weight";
                    case "mamba.b_layernorm.weight" -> "ssm_b_norm.weight";
                    case "mamba.c_layernorm.weight" -> "ssm_c_norm.weight";
                    case "mamba.conv1d.weight", "mamba.A_log" -> null;
                    default -> attnOrMlp(s0);
                };
                if (s0.equals("mamba.conv1d.weight")) {
                    put(b + "ssm_conv1d.weight", v, sh[0], sh[2]);
                } else if (s0.equals("mamba.A_log")) {
                    put(b + "ssm_a", negExp(v), sh);
                } else {
                    put(b + g, v, sh);
                }
            } else {
                throw new IllegalArgumentException("unmapped tensor " + name);
            }
        }
        for (int l = 0; l < n; l++) {
            String ep = "model.layers." + l + ".feed_forward.experts.";
            if (!data.containsKey(ep + "0.gate_proj.weight")) {
                continue;
            }
            for (String[] part : new String[][]{{"down_proj", "ffn_down_exps"}, {"gate_proj", "ffn_gate_exps"}, {"up_proj", "ffn_up_exps"}}) {
                long[] sh = shapes.get(ep + "0." + part[0] + ".weight");
                float[] all = new float[e * (int) (sh[0] * sh[1])];
                for (int x = 0; x < e; x++) {
                    float[] one = data.get(ep + x + "." + part[0] + ".weight");
                    System.arraycopy(one, 0, all, x * one.length, one.length);
                }
                put("blk." + l + "." + part[1] + ".weight", all, e, sh[0], sh[1]);
            }
        }
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
        for (Map.Entry<String, Object[]> e : rawTensors.entrySet()) {
            w.tensor(e.getKey(), (GgmlType) e.getValue()[0], (long[]) e.getValue()[1], (byte[]) e.getValue()[2]);
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
