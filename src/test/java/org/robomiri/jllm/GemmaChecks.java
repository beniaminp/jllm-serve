package org.robomiri.jllm;

import org.robomiri.jllm.model.Architecture;
import org.robomiri.jllm.model.GemmaMath;
import org.robomiri.jllm.model.LayerSpec;
import org.robomiri.jllm.model.Mlp;
import org.robomiri.jllm.model.Rope;
import org.robomiri.jllm.model.Scaling;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.robomiri.jllm.SmokeTest.check;
import static org.robomiri.jllm.SmokeTest.decoder;
import static org.robomiri.jllm.SmokeTest.gqa;
import static org.robomiri.jllm.SmokeTest.rope;

/**
 * Gemma 2 / 3 / 4 checks that need no model: the scalar math against hand-computed values and the config parsing
 * against the real Hub configs (unsloth/gemma-2-2b-it, unsloth/gemma-3-1b-it, unsloth/gemma-3-4b-it,
 * google/gemma-4-12b-it).
 */
final class GemmaChecks {

    private GemmaChecks() {
    }

    static void run() throws Exception {
        math();
        configs();
    }

    static void math() {
        // gelu_pytorch_tanh: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). By hand for x = 1:
        // u = 0.7978845608 * 1.044715 = 0.8335696, tanh(u) = 0.6823837, so 0.5 * 1.6823837 = 0.8411919.
        check(Math.abs(GemmaMath.geluTanh(1f) - 0.8411919f) < 1e-5f, "gelu_tanh(1) = " + GemmaMath.geluTanh(1f));
        check(Math.abs(GemmaMath.geluTanh(-1f) + 0.1588081f) < 1e-5f, "gelu_tanh(-1) = " + GemmaMath.geluTanh(-1f));
        check(Math.abs(GemmaMath.geluTanh(3f) - 2.9963627f) < 1e-5f, "gelu_tanh(3) = " + GemmaMath.geluTanh(3f));
        check(GemmaMath.geluTanh(0f) == 0f && Math.abs(GemmaMath.geluTanh(10f) - 10f) < 1e-6f, "gelu_tanh limits");

        // soft-capping: 50 * tanh(100 / 50) = 50 * 0.96402758 = 48.201379; small inputs pass through; cap 0 = off
        check(Math.abs(GemmaMath.softcap(100f, 50f) - 48.201379f) < 1e-4f, "softcap(100, 50)");
        check(Math.abs(GemmaMath.softcap(0.5f, 50f) - 0.49998333f) < 1e-6f && GemmaMath.softcap(7f, 0f) == 7f,
                "softcap small / off");

        // RMSNorm with gain (1 + w): x = [1 2 3 4], w = [0 0.5 -0.5 1], eps = 0.
        // mean(x^2) = 30/4 = 7.5, rsqrt = 0.36514837, gain = [1 1.5 0.5 2]
        float[] x = {1, 2, 3, 4};
        float[] w = {0, 0.5f, -0.5f, 1};
        float[] y = GemmaMath.rmsnorm(x, GemmaMath.foldNormGain(w), 0f);
        float[] expect = {0.36514837f, 1.0954451f, 0.5477226f, 2.9211870f};
        boolean ok = true;
        for (int i = 0; i < 4; i++) {
            ok &= Math.abs(y[i] - expect[i]) < 1e-6f;
        }
        check(ok, "gemma rmsnorm (1 + w) against hand values");
        check(GemmaMath.foldNormGain(new float[]{0, -1, 0.25f})[1] == 0f, "folded gain 1 + w");
    }

    static void configs() throws Exception {
        Path tmp = Files.createTempDirectory("jllm-gemma-cfg");
        try {
            // unsloth/gemma-2-2b-it
            Architecture g2 = cfg(tmp, """
                    {"architectures":["Gemma2ForCausalLM"],"attention_bias":false,"attn_logit_softcapping":50.0,
                     "bos_token_id":2,"eos_token_id":[1,107],"final_logit_softcapping":30.0,"head_dim":256,
                     "hidden_act":"gelu_pytorch_tanh","hidden_activation":"gelu_pytorch_tanh","hidden_size":2304,
                     "intermediate_size":9216,"max_position_embeddings":8192,"model_type":"gemma2",
                     "num_attention_heads":8,"num_hidden_layers":26,"num_key_value_heads":4,
                     "query_pre_attn_scalar":256,"rms_norm_eps":1e-06,"rope_theta":10000.0,"sliding_window":4096,
                     "torch_dtype":"bfloat16","vocab_size":256000}""");
            LayerSpec l0 = decoder(g2).layers().getFirst();
            check(g2.core().modelType().equals("gemma2") && !gqa(g2, 0).qkNorm() && l0.norms() == LayerSpec.Norms.SANDWICH_GEMMA
                    && l0.mlp() instanceof Mlp.Dense d0 && d0.act() == Mlp.Activation.GELU_TANH && g2.core().normPlusOne(),
                    "gemma2 basics");
            check(g2.core().tiedEmbeddings(), "gemma2 ties embeddings although the config omits the key");
            check(gqa(g2, 0).softcap() == 50f && g2.core().finalSoftcap() == 30f && gqa(g2, 0).scale() == 0.0625f,
                    "gemma2 caps and scale 1/16");
            check(gqa(g2, 0).window() == 4096 && gqa(g2, 1).window() == 0 && gqa(g2, 24).window() == 4096 && gqa(g2, 25).window() == 0,
                    "gemma2 even layers are local");
            check(decoder(g2).ropeTables().size() == 1 && gqa(g2, 0).ropeTable() == 0 && g2.core().embedScale() == 48f,
                    "gemma2 embed scale sqrt(2304) = 48, bos 2, one rope table");
            check(g2.core().eosTokenIds().equals(List.of(1, 107)) && gqa(g2, 0).headDim() == 256 && g2.core().vocabSize() == 256000,
                    "gemma2 ids and shapes");

            // unsloth/gemma-3-1b-it (transformers 4.51: no layer_types, sliding_window_pattern 6)
            Architecture g3 = cfg(tmp, """
                    {"architectures":["Gemma3ForCausalLM"],"attention_bias":false,"attn_logit_softcapping":null,
                     "bos_token_id":2,"eos_token_id":106,"final_logit_softcapping":null,"head_dim":256,
                     "hidden_activation":"gelu_pytorch_tanh","hidden_size":1152,"intermediate_size":6912,
                     "max_position_embeddings":32768,"model_type":"gemma3_text","num_attention_heads":4,
                     "num_hidden_layers":26,"num_key_value_heads":1,"query_pre_attn_scalar":256,"rms_norm_eps":1e-06,
                     "rope_local_base_freq":10000,"rope_scaling":null,"rope_theta":1000000,"sliding_window":512,
                     "sliding_window_pattern":6,"torch_dtype":"bfloat16","vocab_size":262144}""");
            check(gqa(g3, 0).qkNorm() && gqa(g3, 0).softcap() == 0f && g3.core().finalSoftcap() == 0f
                    && rope(g3).scaling() instanceof Scaling.None && rope(g3).theta() == 1e6 && g3.core().vocabSize() == 262144
                    && decoder(g3).ropeTables().size() == 2 && decoder(g3).ropeTables().get(1).theta() == 1e4, "gemma3 1b basics");
            boolean pat = true;
            for (int l = 0; l < 26; l++) {
                boolean global = l % 6 == 5;
                pat &= (gqa(g3, l).window() == (global ? 0 : 512)) && ((gqa(g3, l).ropeTable() == 1) == !global);
            }
            check(pat, "gemma3 5 local : 1 global (layers 5, 11, 17, 23 global)");
            // sqrt(1152) = 33.941 rounds to 34.0 in bfloat16 (8 significant bits), as transformers' normalizer does
            check(g3.core().embedScale() == 34.0f, "gemma3 embed scale bf16(sqrt(1152)) = 34");

            // unsloth/gemma-3-4b-it: multimodal wrapper, decoder settings in text_config
            Architecture g4 = cfg(tmp, """
                    {"architectures":["Gemma3ForConditionalGeneration"],"boi_token_index":255999,"bos_token_id":2,
                     "eoi_token_index":256000,"eos_token_id":106,"image_token_index":262144,"mm_tokens_per_image":256,
                     "model_type":"gemma3","pad_token_id":0,
                     "text_config":{"attention_bias":false,"attn_logit_softcapping":null,"head_dim":256,
                       "hidden_activation":"gelu_pytorch_tanh","hidden_size":2560,"intermediate_size":10240,
                       "max_position_embeddings":131072,"model_type":"gemma3_text","num_attention_heads":8,
                       "num_hidden_layers":34,"num_key_value_heads":4,"query_pre_attn_scalar":256,"rms_norm_eps":1e-06,
                       "rope_local_base_freq":10000.0,"rope_scaling":{"factor":8.0,"rope_type":"linear"},
                       "rope_theta":1000000.0,"sliding_window":1024,"sliding_window_pattern":6,
                       "torch_dtype":"bfloat16","vocab_size":262208},
                     "torch_dtype":"bfloat16","vision_config":{"hidden_size":1152,"model_type":"siglip_vision_model"}}""");
            check(g4.core().modelType().equals("gemma3_text") && g4.core().dim() == 2560 && g4.core().layers() == 34
                    && g4.core().vocabSize() == 262208
                    && gqa(g4, 0).window() == 1024 && gqa(g4, 5).window() == 0 && gqa(g4, 33).window() == 1024
                    && g4.core().bosTokenId() == 2 && g4.core().eosTokenIds().equals(List.of(106)), "gemma3 4b text_config");
            check(rope(g4).scaling() instanceof Scaling.Linear lin && lin.factor() == 8.0
                    && gqa(g4, 0).ropeTable() == 1 && gqa(g4, 5).ropeTable() == 0
                    && decoder(g4).ropeTables().get(1).scaling() instanceof Scaling.None, "gemma3 4b linear rope x8 global, plain local");

            // transformers >= 5 spelling: layer_types and rope_parameters keyed by layer type
            Architecture g5 = cfg(tmp, """
                    {"architectures":["Gemma3ForCausalLM"],"model_type":"gemma3_text","hidden_size":64,
                     "intermediate_size":128,"num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,
                     "head_dim":32,"vocab_size":100,"sliding_window":8,"bos_token_id":2,"eos_token_id":[1,106],
                     "layer_types":["sliding_attention","full_attention","full_attention","sliding_attention"],
                     "rope_parameters":{"full_attention":{"rope_type":"linear","factor":4.0,"rope_theta":1000000.0},
                                        "sliding_attention":{"rope_type":"default","rope_theta":10000.0}}}""");
            check(gqa(g5, 0).window() == 8 && gqa(g5, 1).window() == 0 && gqa(g5, 2).window() == 0 && gqa(g5, 3).window() == 8
                    && rope(g5).theta() == 1e6 && rope(g5).scaling() instanceof Scaling.Linear l5 && l5.factor() == 4.0
                    && gqa(g5, 3).ropeTable() == 1, "gemma3 layer_types / rope_parameters");

            // google/gemma-4-12b-it: the encoder-free multimodal wrapper; the decoder is Gemma4UnifiedTextConfig
            Architecture g6 = cfg(tmp, """
                    {"architectures":["Gemma4UnifiedForConditionalGeneration"],"bos_token_id":2,"eos_token_id":[1,106],
                     "model_type":"gemma4_unified","tie_word_embeddings":true,"dtype":"bfloat16",
                     "text_config":{"attention_bias":false,"attention_k_eq_v":true,"bos_token_id":2,"eos_token_id":1,
                       "final_logit_softcapping":30.0,"global_head_dim":512,"head_dim":256,
                       "hidden_activation":"gelu_pytorch_tanh","hidden_size":3840,"hidden_size_per_layer_input":0,
                       "intermediate_size":15360,"max_position_embeddings":262144,
                       "model_type":"gemma4_unified_text","num_attention_heads":16,
                       "num_global_key_value_heads":1,"num_hidden_layers":48,"num_key_value_heads":8,
                       "num_kv_shared_layers":0,"rms_norm_eps":1e-6,
                       "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention",
                         "sliding_attention","full_attention","sliding_attention","sliding_attention",
                         "sliding_attention","sliding_attention","sliding_attention","full_attention",
                         "sliding_attention","sliding_attention","sliding_attention","sliding_attention",
                         "sliding_attention","full_attention","sliding_attention","sliding_attention",
                         "sliding_attention","sliding_attention","sliding_attention","full_attention",
                         "sliding_attention","sliding_attention","sliding_attention","sliding_attention",
                         "sliding_attention","full_attention","sliding_attention","sliding_attention",
                         "sliding_attention","sliding_attention","sliding_attention","full_attention",
                         "sliding_attention","sliding_attention","sliding_attention","sliding_attention",
                         "sliding_attention","full_attention","sliding_attention","sliding_attention",
                         "sliding_attention","sliding_attention","sliding_attention","full_attention"],
                       "rope_parameters":{"full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,
                                                            "rope_type":"proportional"},
                                          "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}},
                       "sliding_window":1024,"tie_word_embeddings":true,"vocab_size":262144,
                       "vocab_size_per_layer_input":262144}}""");
            check(g6.core().modelType().equals("gemma4_unified_text") && g6.core().dim() == 3840 && g6.core().layers() == 48
                    && g6.core().vocabSize() == 262144 && !g6.core().normPlusOne() && g6.core().tiedEmbeddings()
                    && g6.core().finalSoftcap() == 30f && g6.core().embedScale() == 62f,
                    "gemma4 12b wrapper, tied head, bf16(sqrt(3840)) = 62, plain (not 1+w) norms");
            check(g6.core().eosTokenIds().equals(List.of(1, 106)), "gemma4 wrapper eos list is kept");
            // 5 local : 1 full, ending full
            boolean pat4 = true;
            for (int l = 0; l < 48; l++) {
                boolean global = l % 6 == 5;
                pat4 &= (gqa(g6, l).window() == (global ? 0 : 1024)) && ((gqa(g6, l).ropeTable() == 1) == !global);
            }
            check(pat4 && gqa(g6, 47).window() == 0, "gemma4 5 local : 1 full, last layer full");
            check(gqa(g6, 0).headDim() == 256 && gqa(g6, 0).kvHeads() == 8 && gqa(g6, 0).qkNorm()
                    && !gqa(g6, 0).kEqV() && gqa(g6, 0).vNorm() && gqa(g6, 0).scale() == 1f
                    && gqa(g6, 0).softcap() == 0f,
                    "gemma4 local layers: 256-wide head, 8 KV heads, q/k norm, v_proj + V norm, unscaled attention");
            check(gqa(g6, 5).headDim() == 512 && gqa(g6, 5).kvHeads() == 1 && gqa(g6, 5).kEqV()
                    && gqa(g6, 5).vNorm(), "gemma4 full layers: 512-wide head, one KV head, K = V, V norm");
            check(decoder(g6).ropeTables().size() == 2
                    && decoder(g6).ropeTables().get(0).rotaryDim() == 512
                    && decoder(g6).ropeTables().get(0).activePairs() == 64
                    && decoder(g6).ropeTables().get(0).theta() == 1e6
                    && decoder(g6).ropeTables().get(1).rotaryDim() == 256
                    && decoder(g6).ropeTables().get(1).activePairs() == 128
                    && decoder(g6).ropeTables().get(1).theta() == 1e4,
                    "gemma4 proportional RoPE: 64 of 256 pairs on the full head, plain local table");
            check(!decoder(g6).uniformShape(), "gemma4 layers have different attention head shapes");
            double[] inv = Rope.invFreq(decoder(g6).ropeTables().get(0), 16);
            check(inv.length == 256 && inv[0] == 1.0 && inv[63] > 0 && inv[64] == 0.0 && inv[255] == 0.0,
                    "gemma4 proportional RoPE frequencies: 64 live pairs, then the unrotated tail");
        } finally {
            Files.deleteIfExists(tmp.resolve("config.json"));
            Files.deleteIfExists(tmp);
        }
    }

    private static Architecture cfg(Path dir, String json) throws java.io.IOException {
        Files.writeString(dir.resolve("config.json"), json);
        return Architecture.load(dir);
    }
}
